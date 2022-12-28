from __future__ import annotations
from typing import Dict
import pandas as pd
import datetime

from ..definitions import OfflinePersistEngine, OfflinePersistEngineType
from ..offline_stores.offline_file_store import OfflineFileStore
from ..definitions import FileSource
from ..common.utils import to_file

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"


class OfflineFilePersistEngine(OfflinePersistEngine):
    type: OfflinePersistEngineType = OfflinePersistEngineType.FILE
    store: OfflineFileStore

    def materialize(
        self,
        save_path: FileSource,
        all_views: Dict,
        start: pd.Timestamp = None,
        end: pd.Timestamp = None,
        **kwargs,
    ):
        feature_views = all_views["features"]
        label_view = all_views["label"]

        source = label_view["source"]
        # TODO:
        # 1. 这里是否需要进行更合理的抽象，而不是使用一个私有函数
        # 2. 在读取数据之前，框定时间可以可以提高效率
        joined_frame = self.store._read_file(
            source=source, features=label_view["labels"], join_keys=label_view["join_keys"]
        )
        # TODO: 这里是否应该丢弃created_timestamp？
        joined_frame.drop(columns=["created_timestamp"], errors="ignore")
        joined_frame = joined_frame[(joined_frame[TIME_COL] >= start) & (joined_frame[TIME_COL] < end)]

        # join features dataframe
        for feature_view in feature_views:
            features = feature_view["features"]
            feature_name = [
                f.name for f in features if f.name not in [label.name for label in label_view["labels"]]
            ]
            if feature_name:  # this view has new features other than those in joined_frame
                features = [n for n in features if n.name in feature_name]
                join_keys = feature_view["join_keys"]
                source = feature_view["source"]
                joined_frame = self.store.get_features(
                    entity_df=joined_frame,
                    features=features,
                    source=source,
                    join_keys=join_keys,
                    ttl=feature_view["ttl"],
                    include=True,
                    how="right",
                )

        joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.datetime.now(), utc=True)
        to_file(joined_frame[sorted(joined_frame.columns)], save_path.path, "csv", mode="a", header=0)
        kwargs["signal"].send(1)
