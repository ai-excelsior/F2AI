from collections import defaultdict
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from torch.utils.data import IterableDataset


if TYPE_CHECKING:
    from aie_feast.offline_stores.offline_store import OfflineStore
    from aie_feast.service import Service

TIME_COL = "event_timestamp"


class IterableDataset(IterableDataset):
    def __init__(
        self,
        fs: "OfflineStore",
        service: "Service",
        entity_index: pd.DataFrame,
        entity_cols: list = [],
        batch: int = None,
        feature_views=None,
        label_views=None,
        join_keys=None,
    ):
        self.fs = fs
        self.service = service
        self.entity_index = entity_index
        self.entity_cols = entity_cols
        self.feature_views = feature_views
        self.label_views = label_views
        self.batch = batch if batch else len(self.entity_index) // 10
        self.join_keys = join_keys

        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)

    def __iter__(self):
        for i in range(len(self.entity_index)):
            if i % self.batch == 0:  # batch merge
                self.get_context(i // self.batch)
            to_return = (
                self.data_sample[0][
                    (
                        self.data_sample[0][[TIME_COL, *self.join_keys]]
                        == self.entity_index.iloc[i % self.batch][[TIME_COL, *self.join_keys]]
                    ).all(axis=1)
                ].drop(columns=[TIME_COL]),
                self.data_sample[1][
                    (
                        self.data_sample[1][[TIME_COL, *self.join_keys]]
                        == self.entity_index.iloc[i % self.batch][[TIME_COL, *self.join_keys]]
                    ).all(axis=1)
                ].drop(columns=[TIME_COL]),
            )
            if not to_return[0].isnull().all().all() and not to_return[1].isnull().all().all():
                yield to_return

    def get_context(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        source = self.fs.get_offline_source(self.service)
        self.data_sample = self.fs.get_context(
            source=source,
            entity=self.entity_index.iloc[i * self.batch : (i + 1) * self.batch],
            all_features=self.all_features,
            all_labels=self.all_labels,
            feature_list=list(self.service.get_feature_names(self.feature_views)),
            label_list=list(self.service.get_label_names(self.label_views)),
            ttl=self.service.ttl,
            join_keys=self.join_keys,
            entity_cols=self.entity_cols,
        )

    def get_feature_period(self, service: "Service", with_labels=False) -> dict:
        """_summary_

        Args:
            service (Service): materialized service to construct
            with_labels (bool, optional): get labels or not

        Returns:
            Dict: {period1:[fea1,fea2],period2[fea5],0:[fea3,fea4]}, 0 means no period
        """
        period_dict = defaultdict(list)

        if with_labels:
            for label in service.get_label_objects(self.label_views):
                period = label.period.strip('"') if label.period else 0
                period_dict[period].append(label)  # TODO:period

        else:
            for feature in service.get_feature_objects(self.feature_views):
                period = feature.period.strip('"') if feature.period else 0
                period_dict[period].append(feature)

        return period_dict


class Dataset:
    def __init__(
        self,
        fs: "OfflineStore",
        service: "Service",
        sampler: callable,
        feature_views,
        label_views,
    ):
        self.fs = fs
        self.service = service
        self.sampler = sampler
        self.feature_views = feature_views
        self.label_views = label_views

    def to_pytorch(self, batch: int = None, entity_cols: list = []) -> IterableDataset:
        """convert to iterablt pytorch dataset really hold data"""
        entity_index = self.sampler()
        join_keys = list(entity_index.columns[:-1])

        return IterableDataset(
            fs=self.fs,
            service=self.service,
            entity_index=entity_index,
            entity_cols=entity_cols,
            batch=batch,
            feature_views=self.feature_views,
            label_views=self.label_views,
            join_keys=join_keys,
        )
