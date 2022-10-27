import pandas as pd
from typing import List, Optional
from dateutil.relativedelta import relativedelta
from .offline_store import OfflineStore, OfflineStoreType
from aie_feast.common.utils import parse_date


# DEFAULT_TIMESTAMP_FIELD = "event_timestamp"
# DEFAULT_CREATED_TIMESTAMP_FIELD = "created_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
SOURCE_CREATED_TIMESTAMP_FIELD = "_created_timestamp_"


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE

    @classmethod
    def point_in_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        timestamp_field: Optional[str] = None,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[str] = None,
        join_keys: List[str] = [],
        include: bool = True,
    ):
        # renames to keep things simple
        if timestamp_field:
            entity_df = entity_df.rename(columns={timestamp_field: ENTITY_EVENT_TIMESTAMP_FIELD})
            source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        # TODO: pre filter source_df by ttl
        # TODO: pre filter source_df by entities
        # TODO: pre filter source_df by created_timestamp_field

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        if timestamp_field:
            df = cls.point_in_time_filter(df, include=include, ttl=ttl)
            df = cls.point_in_time_latest(df, join_keys)

        return df.drop(columns=[SOURCE_EVENT_TIMESTAMP_FIELD]).rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: timestamp_field}
        )

    @classmethod
    def point_in_time_filter(
        cls,
        df: pd.DataFrame,
        include: bool = True,
        ttl: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        if ttl:
            timedelta = relativedelta(**parse_date(ttl))
            earliest_timestamp = df[entity_timestamp_field].map(lambda x: x - timedelta)

        if include:
            candidates = df[entity_timestamp_field] >= df[source_timestamp_field]
            if ttl:
                candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
        else:
            candidates = df[entity_timestamp_field] > df[source_timestamp_field]
            if ttl:
                candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
        return df[candidates]

    @classmethod
    def point_in_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        return (
            df.groupby(group_keys + [entity_timestamp_field])
            .apply(lambda x: x.sort_values(source_timestamp_field, ascending=False).head(1))
            .reset_index(drop=True)
        )
