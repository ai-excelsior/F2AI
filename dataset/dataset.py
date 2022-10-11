import pandas as pd
from typing import Dict, List, Tuple
from typing import TYPE_CHECKING
from aie_feast.views import FeatureViews, LabelViews
from common.utils import read_file
from common.psl_utils import psy_conn
import os

if TYPE_CHECKING:
    from aie_feast.featurestore import FeatureStore

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"


class IterableDataset:
    def __init__(self, dataset: "Dataset", materialize_pd: pd.DataFrame):
        self.dataset = dataset
        self.materialize_pd = materialize_pd

    def __iter__(self):
        return iter(
            [
                self.get_context(self.dataset.entity_index[i : i + 1])
                for i in range(self.dataset.entity_index.last_valid_index() + 1)
            ]
        )

    def get_context(self, entity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fs = self.dataset.fs
        feature_views_pd, label_views_pd = pd.DataFrame(), pd.DataFrame()
        for feature_view_key, col_name_and_period in self.dataset.service.features.items():
            feature_view: FeatureViews = fs.features[feature_view_key]
            cols = fs._get_avaliable_features(feature_view)
            period = self.get_period(col_name_and_period)
            feature_views_pd.merge(
                fs.get_period_features(feature_view, entity, period, cols)
                if period
                else fs.get_features(feature_view, entity, cols)
            )

        for label_view_key, col_name_and_period in self.dataset.service.labels.items():
            label_view: LabelViews = fs.labels[label_view_key]
            cols = fs._get_available_labels(label_view)
            period = self.get_period(col_name_and_period)
            label_views_pd.merge(
                fs.get_period_labels(label_view, entity, period, cols)
                if period
                else fs.get_labels(label_view, entity, cols)
            )

        return feature_views_pd, label_views_pd

    def get_period(self, col_name_and_period: Dict):
        for _, period in col_name_and_period.items():
            if period is not None:
                return period


class Dataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service_name: str,
        start: str = None,
        end: str = None,
        time_bucket: str = "10 days",
        stride: int = 1,
        sampler: callable = None,
    ):
        self.fs = fs
        self.service_name = service_name
        self.start = start
        self.end = end
        self.bucket = time_bucket
        self.stride = stride
        self.entity_index = sampler(start=start, end=end, time_bucket=time_bucket, stride=stride)()

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
