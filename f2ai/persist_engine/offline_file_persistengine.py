from __future__ import annotations
from operator import imod
from typing import List
import pandas as pd
import datetime

from ..offline_stores.offline_file_store import OfflineFileStore
from ..definitions import (
    FileSource,
    OfflinePersistEngine,
    OfflinePersistEngineType,
    BackOffTime,
    PersistFeatureView,
    PersistLabelView,
)
from ..common.utils import write_df_to_dataset
from ..common.time_field import TIME_COL, MATERIALIZE_TIME


class OfflineFilePersistEngine(OfflinePersistEngine):
    type: OfflinePersistEngineType = OfflinePersistEngineType.FILE
    offline_store: OfflineFileStore

    def materialize(
        self,
        feature_views: List[PersistFeatureView],
        label_view: PersistLabelView,
        destination: FileSource,
        back_off_time: BackOffTime,
        service_name: str,
    ):
        # retrieve entity_df
        # TODO:
        # 1. 这里是否需要进行更合理的抽象，而不是使用一个私有函数
        # 2. 在读取数据之前，框定时间可以可以提高效率
        entity_df = self.offline_store._read_file(
            source=label_view.source, features=label_view.labels, join_keys=label_view.join_keys
        )

        entity_df.drop(columns=["created_timestamp"], errors="ignore")
        entity_df = entity_df[
            (entity_df[TIME_COL] >= back_off_time.start) & (entity_df[TIME_COL] < back_off_time.end)
        ]

        # join features recursively
        # TODO: this should be reimplemented to directly consume multi feature_views and do a performance test.
        joined_frame = entity_df
        for feature_view in feature_views:
            joined_frame = self.offline_store.get_features(
                entity_df=joined_frame,
                features=feature_view.features,
                source=feature_view.source,
                join_keys=feature_view.join_keys,
                ttl=feature_view.ttl,
                include=True,
                how="right",
            )
        tz = joined_frame[TIME_COL][0].tz if not joined_frame.empty else None
        joined_frame[MATERIALIZE_TIME] = pd.Timestamp(datetime.datetime.now(), tz=tz)
        write_df_to_dataset(
            joined_frame, destination.path, time_col=destination.timestamp_field, period=back_off_time.step
        )
        return service_name
