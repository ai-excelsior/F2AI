from collections import defaultdict
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from torch.utils.data import IterableDataset
from copy import deepcopy


if TYPE_CHECKING:
    from aie_feast.offline_stores.offline_store import OfflineStore
    from aie_feast.service import Service
    from aie_feast.featurestore import FeatureStore

TIME_COL = "event_timestamp"
QUERY_COL = "query_timestamp"


class IterableDataset(IterableDataset):
    def __init__(
        self,
        fs: "OfflineStore",
        service: "Service",
        entity_index: pd.DataFrame,
        batch: int = None,
    ):
        self.fs = fs
        self.service = service
        self.entity_index = entity_index
        self.batch = batch if batch else len(self.entity_index) // 10

        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)

    def __iter__(self):
        for i in range(len(self.entity_index)):
            if i % self.batch == 0:  # batch merge
                self.get_context(i // self.batch)
            to_return = (
                self.data_sample[0][
                    (
                        self.data_sample[0][[*self.entity_index.columns]]
                        == self.entity_index.iloc[i % self.batch][[*self.entity_index.columns]]
                    ).all(axis=1)
                ].drop(columns=self.entity_index.columns),
                self.data_sample[1][
                    (
                        self.data_sample[1][[*self.entity_index.columns]]
                        == self.entity_index.iloc[i % self.batch][[*self.entity_index.columns]]
                    ).all(axis=1)
                ].drop(columns=self.entity_index.columns),
            )
            if not to_return[0].isnull().all().all() and not to_return[1].isnull().all().all():
                yield to_return

    def get_context(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_views_pd = deepcopy(self.entity_index.iloc[i * self.batch : (i + 1) * self.batch])
        label_views_pd = deepcopy(self.entity_index.iloc[i * self.batch : (i + 1) * self.batch])
        entity_cols = self.fs._get_keys_to_join(self.service)

        for period, features in self.all_features.items():
            if period:
                feature_views_pd = self.fs.get_period_features(
                    feature_view=self.service,
                    entity_df=feature_views_pd,
                    period=period,
                    features=features,
                    include=True,
                    how="right",
                    entity_cols=entity_cols,
                )
                feature_views_pd.drop(columns=[TIME_COL], inplace=True)
                # always merge on TIME_COL
                feature_views_pd.rename(columns={QUERY_COL: TIME_COL}, inplace=True)
            else:
                feature_views_pd = self.fs.get_features(
                    feature_view=self.service,
                    entity_df=feature_views_pd,
                    features=features,
                    include=True,
                    how="right",
                    entity_cols=entity_cols,
                )
            feature_views_pd[TIME_COL] = pd.to_datetime(feature_views_pd[TIME_COL], utc=True)

        for period, features in self.all_labels.items():
            if period:
                label_views_pd = self.fs.get_period_labels(
                    label_view=self.service,
                    entity_df=label_views_pd,
                    period=period,
                    include=False,
                    how="right",
                    entity_cols=entity_cols,
                )
                label_views_pd.drop(columns=[TIME_COL], inplace=True)
                label_views_pd.rename(columns={QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
            else:
                label_views_pd = self.fs.get_labels(
                    label_view=self.service,
                    entity_df=label_views_pd,
                    include=True,
                    how="right",
                    entity_cols=entity_cols,
                )
            label_views_pd[TIME_COL] = pd.to_datetime(label_views_pd[TIME_COL], utc=True)

        self.data_sample = (
            feature_views_pd[
                entity_cols + [TIME_COL] + list(self.service.get_feature_names(self.fs.feature_views))
            ],
            label_views_pd[
                entity_cols + [TIME_COL] + list(self.service.get_label_names(self.fs.label_views))
            ],
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
            for label in service.get_label_objects(self.fs.label_views):
                period = label.period.strip('"') if label.period else 0
                period_dict[period].append(label.name)  # TODO:period

        else:
            for feature in service.get_feature_objects(self.fs.feature_views):
                period = feature.period.strip('"') if feature.period else 0
                period_dict[period].append(feature.name)

        return period_dict


class Dataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service: "Service",
        sampler: callable,
    ):
        self.fs = fs
        self.service = service
        self.sampler = sampler

    def to_pytorch(self, batch: int = None) -> IterableDataset:
        """convert to iterablt pytorch dataset really hold data"""
        entity_index = self.sampler()

        return IterableDataset(
            fs=self.fs,
            service=self.service,
            entity_index=entity_index,
            batch=batch,
        )
