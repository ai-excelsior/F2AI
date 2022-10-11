import pandas as pd
from typing import Dict, List, Tuple
from typing import TYPE_CHECKING
from aie_feast.views import FeatureViews, LabelViews
from common.utils import read_file
from common.psl_utils import psy_conn
import os
from copy import deepcopy

if TYPE_CHECKING:
    from aie_feast.featurestore import FeatureStore

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"
CREATE_COL = "created_timestamp"


class IterableDataset:
    def __init__(
        self, materialize_pd: pd.DataFrame, entity_index: pd.DataFrame, all_features: list, all_labels: list
    ):
        self.materialize_pd = materialize_pd
        self.entity_index = entity_index
        self.all_features = all_features
        self.all_labels = all_labels  # self.dataset.fs._get_available_labels(self.service)

    def __iter__(self):
        return iter([self.get_context(self.entity_index.iloc[[i]]) for i in range(len(self.entity_index))])

    def get_context(self, entity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_views_pd = deepcopy(entity)
        label_views_pd = deepcopy(entity)

        feature_views_pd = entity.merge(
            self.materialize_pd[self.all_features], on=list(entity.columns[:-1]) + [TIME_COL], how="inner"
        )
        label_views_pd = entity.merge(
            self.materialize_pd[self.all_labels], on=list(entity.columns[:-1]) + [TIME_COL], how="inner"
        )

        return feature_views_pd, label_views_pd

    def get_period(self, fea_collect: list):
        period_dict = {}
        for fea in fea_collect:
            for k, v in fea.items():
                v = v if v else 0
                if v in period_dict:
                    period_dict[v].append(k)
                else:
                    period_dict[v] = [k] if k != "__all__" else None
        return period_dict


class Dataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service_name: str,
        sampler: callable = None,
    ):
        self.fs = fs
        self.service_name = service_name
        self.entity_index = sampler()

    def to_pytorch(self) -> IterableDataset:
        """convert to iterablt pytorch dataset"""

        if self.fs.service[self.service_name].materialize_type == "file":  # file-record
            materialize_pd = read_file(
                os.path.join(
                    self.fs.project_folder, f"{self.fs.service[self.service_name].materialize_path}"
                ),
                type=self.fs.service[self.service_name].materialize_path.split(".")[-1],
                time_col=[TIME_COL, MATERIALIZE_TIME],
            )
        elif self.fs.service[self.service_name].materialize_type == "pgsql":
            conn = psy_conn(**self.fs.connection.__dict__)

        return IterableDataset(
            materialize_pd,
            self.entity_index,
            self.fs._get_available_features(self.fs.service[self.service_name]),
            self.fs._get_available_labels(self.fs.service[self.service_name]),
        )
