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
        service: FeatureView,
        source: Source,
        start: pd.Timestamp,
        end: pd.Timestamp,
        join_keys: List[str],
        off_store: OfflineStore,
    ):
        save_path = self.store.get_online_source()
        date_df = pd.DataFrame(data=[end], columns=[DEFAULT_EVENT_TIMESTAMP_FIELD])
        period = -Period.from_str(str(end - start))
        entities_in_range = off_store.get_latest_entities(
            source=source,
            group_keys=join_keys,
            entity_df=date_df,
            start=start,
        ).drop(columns=DEFAULT_EVENT_TIMESTAMP_FIELD)
        data_to_write = off_store.get_period_features(
            entity_df=pd.merge(entities_in_range, date_df, how="cross"),
            features=service.get_feature_objects(),
            source=source,
            period=period,
            join_keys=join_keys,
            ttl=service.ttl,
        )
        self.store.write_batch(service, save_path, data_to_write)
