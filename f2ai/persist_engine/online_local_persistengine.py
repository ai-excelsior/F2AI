import pandas as pd

from ..definitions import (
    Period,
    BackOffTime,
    OnlinePersistEngine,
    OnlinePersistEngineType,
    PersistFeatureView,
)

from ..common.time_field import DEFAULT_EVENT_TIMESTAMP_FIELD


class OnlineLocalPersistEngine(OnlinePersistEngine):
    type: OnlinePersistEngineType = OnlinePersistEngineType.LOCAL

    def materialize(self, prefix: str, feature_view: PersistFeatureView, back_off_time: BackOffTime):
        date_df = pd.DataFrame(data=[back_off_time.end], columns=[DEFAULT_EVENT_TIMESTAMP_FIELD])
        period = -Period.from_str(str(back_off_time.end - back_off_time.start))
        entities_in_range = self.offline_store.get_latest_entities(
            source=feature_view.source,
            group_keys=feature_view.join_keys,
            entity_df=date_df,
            start=back_off_time.start,
        ).drop(columns=DEFAULT_EVENT_TIMESTAMP_FIELD)

        data_to_write = self.offline_store.get_period_features(
            entity_df=pd.merge(entities_in_range, date_df, how="cross"),
            features=feature_view.features,
            source=feature_view.source,
            period=period,
            join_keys=feature_view.join_keys,
            ttl=feature_view.ttl,
        )
        self.online_store.write_batch(
            feature_view.name,
            prefix,
            data_to_write,
            feature_view.ttl,
            join_keys=feature_view.join_keys,
        )
