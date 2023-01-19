from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING, Optional
from torch.utils.data import IterableDataset

from ..common.utils import batched
from ..common.time_field import DEFAULT_EVENT_TIMESTAMP_FIELD, QUERY_COL
from ..definitions import Period
from .entities_sampler import EntitiesSampler


if TYPE_CHECKING:
    from ..definitions.services import Service
    from ..featurestore import FeatureStore


class TorchIterableDataset(IterableDataset):
    """A pytorch portaled dataset."""

    def __init__(
        self,
        feature_store: "FeatureStore",
        service: "Service",
        sampler: "EntitiesSampler",
        chunk_size: int = 64,
    ):
        assert sampler.iterable, "TorchIterableDataset only support iterable sampler"

        self._feature_store = feature_store
        self._service = service
        self._sampler = sampler
        self._chunk_size = chunk_size

    def get_feature_period(self) -> Optional[Period]:
        features = self._service.get_feature_objects(self._feature_store.feature_views)
        periods = [Period.from_str(x.period) for x in features if x.period is not None]
        return max(periods) if len(periods) > 0 else None

    def get_label_period(self) -> Optional[Period]:
        labels = self._service.get_label_objects(self._feature_store.label_views)
        periods = [Period.from_str(x.period) for x in labels if x.period is not None]
        return max(periods) if len(periods) > 0 else None

    def __iter__(self):
        feature_period = self.get_feature_period()
        label_period = self.get_label_period()
        join_keys = self._service.get_join_keys(
            self._feature_store.feature_views,
            self._feature_store.label_views,
            self._feature_store.entities,
        )

        for x in batched(self._sampler, batch_size=self._chunk_size):
            entity_df = pd.DataFrame(x)
            labels_df = None

            if feature_period is None:
                features_df = self._feature_store.get_features(self._service, entity_df)
                labels = self._service.get_label_names(self._feature_store.label_views)
                label_columns = join_keys + labels + [DEFAULT_EVENT_TIMESTAMP_FIELD]
                labels_df = features_df[label_columns]
                features_df = features_df.drop(columns=labels)
            else:
                features_df = self._feature_store.get_period_features(
                    self._service, entity_df, feature_period
                )

            # get corresponding labels if not present in features_df
            if labels_df is None:
                labels_df = self._feature_store.get_period_labels(self._service, entity_df, label_period)

            if feature_period:
                group_columns = join_keys + [QUERY_COL]
            else:
                group_columns = join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]

            labels_group = labels_df.groupby(group_columns)
            for name, x_features in features_df.groupby(group_columns):
                y_labels = labels_group.get_group(name)
                # TODO: test this with tabular dataset.
                # TODO: y_labels seems not correct.
                yield (x_features, y_labels)
