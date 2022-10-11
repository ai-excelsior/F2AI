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
    def __init__(self, dataset: "Dataset", materialize_pd: pd.DataFrame):
        self.dataset = dataset
        self.materialize_pd = materialize_pd
        self.service = self.dataset.fs.service[self.dataset.service_name]
        self.all_features = self.dataset.fs._get_avaliable_features(self.service)
        self.all_labels = self.dataset.fs._get_avaliable_features(self.service)

    def __iter__(self):
        return iter(
            [
                self.get_context(self.dataset.entity_index[i : i + 1])
                for i in range(self.dataset.entity_index.last_valid_index() + 1)
            ]
        )

    def get_context(self, entity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fs = self.dataset.fs
        feature_views_pd = deepcopy(entity)
        label_views_pd = deepcopy(entity)
        for view_fea, fea_name in self.service.features.items():
            feature_view: FeatureViews = fs.features[view_fea]
            entities = [e for e in fs._get_avaliable_entity(feature_view) if e in list(entity.columns[:-1])]

            cols = [c for c in fs._get_avaliable_features(feature_view) if c in self.all_features]
            period_dict = self.get_period(fea_name)  # features in the same view may not have the same period
            for period, features in period_dict.items():
                features = features if features else cols
                feature_views_pd = feature_views_pd.merge(
                    fs.get_period_features(feature_view, entity, period, features).drop(columns=[CREATE_COL])
                    if period
                    else fs.get_features(feature_view, entity, features).drop(columns=[CREATE_COL]),
                    on=entities + [TIME_COL],
                    how="left",
                )

        for view_label, label_name in self.service.labels.items():
            label_view: LabelViews = fs.labels[view_label]
            cols = [c for c in fs._get_available_labels(feature_view) if c in self.all_labels]
            period = self.get_period(label_name)
            label_views_pd.merge(
                fs.get_period_labels(label_view, entity, period, cols)
                if period
                else fs.get_labels(label_view, entity, cols)
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

        return IterableDataset(self, materialize_pd)
