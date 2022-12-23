from executing import Source
from definitions import offline_store
from f2ai.definitions import OnlinePersistEngine, OnlinePersistEngineType, OnlineStore
from typing import List, Dict
import pandas as pd
from f2ai.definitions import (
    OfflineStore,
    OnlineStore,
    Period,
    FeatureView,
)

DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"


class OnlineLocalPersistEngine(OnlinePersistEngine):
    type: OnlinePersistEngineType = OnlinePersistEngineType.LOCAL
    store: OnlineStore

    def materialize(
        self,
        save_path: str,
        feature_views: Dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        off_store: OfflineStore,
    ):

        date_df = pd.DataFrame(data=[end], columns=[DEFAULT_EVENT_TIMESTAMP_FIELD])
        period = -Period.from_str(str(end - start))
        entities_in_range = off_store.get_latest_entities(
            source=feature_views["source"],
            group_keys=feature_views["join_keys"],
            entity_df=date_df,
            start=start,
        ).drop(columns=DEFAULT_EVENT_TIMESTAMP_FIELD)
        data_to_write = off_store.get_period_features(
            entity_df=pd.merge(entities_in_range, date_df, how="cross"),
            features=feature_views["features"],
            source=feature_views["source"],
            period=period,
            join_keys=feature_views["join_keys"],
            ttl=feature_views["ttl"],
        )
        self.store.write_batch(feature_views["name"], save_path, data_to_write)
