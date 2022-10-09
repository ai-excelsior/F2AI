import pandas as pd
from typing import Dict, List, Tuple
from typing import TYPE_CHECKING
from aie_feast.service import Service
from aie_feast.views import FeatureViews, LabelViews
from common.utils import read_file


if TYPE_CHECKING:
    from aie_feast.featurestore import FeatureStore

class IterableDataset:

    def __init__(self, dataset: "Dataset", materialize_pd: pd.DataFrame):
        self.dataset = dataset
        self.materialize_pd = materialize_pd

    def __iter__(self):
        entity = self.dataset.entity
        for i in range(entity.last_valid_index() + 1):
            yield self.get_context(entity[i: i + 1])

    def get_context(self, entity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fs = self.dataset.fs
        feature_views_pd, label_views_pd = pd.DataFrame(), pd.DataFrame()
        for feature_view_key, col_name_and_period in self.dataset.service.features.items():
            feature_view: FeatureViews = fs.features[feature_view_key]
            cols = fs._get_avaliable_features(feature_view)
            period = self.get_period(col_name_and_period)
            feature_views_pd.merge(
                fs.get_period_features(feature_view, entity, period, cols, include=self.dataset.include)
                if period else
                fs.get_features(feature_view, entity, cols, include=self.dataset.include)
            )

        for label_view_key, col_name_and_period in self.dataset.service.labels.items():
            label_view: LabelViews = fs.labels[label_view_key]
            cols = fs._get_available_labels(label_view)
            period = self.get_period(col_name_and_period)
            label_views_pd.merge(
                fs.get_period_labels(label_view, entity, period, cols, include=self.dataset.include)
                if period else
                fs.get_labels(label_view, entity, cols, include=self.dataset.include)
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
        service: Service,
        start: str,
        end: str,
        sampler: callable,
        bucket: int,
        stride: int,
        include: str
    ):
        self.fs = fs
        self.service = service
        self.start = start
        self.end = end
        self.sampler = sampler
        self.bucket = bucket
        self.stride = stride
        self.include = include
        # TODO
        self.entity: pd.DataFrame = self.sampler()

    def to_pytorch(self) -> IterableDataset:
        """convert to iterablt pytorch dataset"""
        if self.service.materialize_path.endswith('.parquet'):
            materialize_pd = read_file(self.service.materialize_path, type='parquet', time_col='event_timestamp')

        return IterableDataset(self, materialize_pd)
