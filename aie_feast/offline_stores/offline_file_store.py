from ctypes import Union
from datetime import datetime
import pandas as pd
from typing import List, Optional, Set
from dateutil.relativedelta import relativedelta
from aie_feast.common.utils import parse_date, read_file
from aie_feast.definitions import Feature
from aie_feast.common.source import FileSource
from aie_feast.common.utils import get_stats_result
from .offline_store import OfflineStore, OfflineStoreType


TIME_COL = "event_timestamp"
# DEFAULT_CREATED_TIMESTAMP_FIELD = "created_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
SOURCE_CREATED_TIMESTAMP_FIELD = "_created_timestamp_"
QUERY_COL = "query_timestamp"


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE

    def read(self, source: FileSource, features: Set[Feature] = {}, join_keys: List[str] = []):
        time_columns = [source.timestamp_field]
        if source.created_timestamp_field:
            time_columns.append(source.created_timestamp_field)

        feature_columns = [feature.name for feature in features]
        all_columns = list(set(time_columns + join_keys + feature_columns))

        return read_file(
            source.path, file_format=source.file_format, time_cols=time_columns, entity_cols=join_keys
        )[all_columns]

    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        join_keys: List[str] = [],
        ttl: Optional[str] = None,
        include: bool = True,
    ):
        source_df = self.read(source=source, features=features, join_keys=join_keys)

        return self.point_in_time_join(
            entity_df=entity_df,
            source_df=source_df,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
        )

    def get_period_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        period: str,
        join_keys: List[str] = [],
        ttl: Optional[str] = None,
        include: bool = True,
        is_label: bool = False,
    ):
        source_df = self.read(source=source, features=features, join_keys=join_keys)

        return self.point_on_time_join(
            entity_df=entity_df,
            source_df=source_df,
            period=period,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            is_label=is_label,
        )

    def stats(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        start,
        fn: str = "mean",
        join_keys: list = [],
        include: str = "both",
        keys_only: bool = False,
    ):
        source_df = self.read(source=source, features=features, join_keys=join_keys)
        if keys_only:
            feature_columns = join_keys
        else:
            feature_columns = [feature.name for feature in features]

        entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={source.timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if len(entity_df.columns) > 1:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        if join_keys:
            result = df.groupby(join_keys).apply(
                get_stats_result,
                fn,
                use_cols=feature_columns,
                include=include,
                start=start,
            )
        else:
            result = get_stats_result(
                df,
                fn,
                use_cols=feature_columns,
                include=include,
                start=start,
            )
        return result

    def get_latest_entities(self, source: FileSource, join_keys: list):
        source_df = self.read(source=source, features=[], join_keys=join_keys)
        df = source_df.sort_values(by=source.timestamp_field, ascending=False, ignore_index=True)
        return df.drop_duplicates(subset=join_keys, keep="first")

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
            entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
            source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field and created_timestamp_field in entity_df.columns:
            entity_df = entity_df.drop(columns=[created_timestamp_field])

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - relativedelta(
                **parse_date(ttl)
            )
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        # pre filter source_df by entities
        if len(join_keys) > 0:
            unique_entity_df = entity_df[join_keys].groupby(join_keys).size().reset_index().drop(columns=[0])
            source_df = unique_entity_df.merge(source_df, on=join_keys, how="inner")

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        if timestamp_field:
            df = cls.point_in_time_filter(df, include=include, ttl=ttl)
            df = cls.point_in_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(columns=[SOURCE_EVENT_TIMESTAMP_FIELD]).rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: TIME_COL}
        )

    @classmethod
    def point_on_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        period: str,
        timestamp_field: Optional[str],
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[str] = None,
        join_keys: List[str] = [],
        include: bool = True,
        is_label: bool = False,
    ):
        # renames to keep things simple
        entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field and created_timestamp_field in entity_df.columns:
            entity_df = entity_df.drop(columns=[created_timestamp_field])

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - relativedelta(
                **parse_date(ttl)
            )
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        # pre filter source_df by entities
        if len(join_keys) > 0:
            unique_entity_df = entity_df[join_keys].groupby(join_keys).size().reset_index().drop(columns=[0])
            source_df = unique_entity_df.merge(source_df, on=join_keys, how="inner")

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        df = cls.point_on_time_filter(df, period, include=include, ttl=ttl, is_label=is_label)
        df = cls.point_on_time_latest(df, join_keys, created_timestamp_field)

        return df.rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: QUERY_COL, SOURCE_EVENT_TIMESTAMP_FIELD: TIME_COL}
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
    def point_on_time_filter(
        cls,
        df: pd.DataFrame,
        period: str,
        include: bool = True,
        ttl: Optional[str] = None,
        is_label=False,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        period = relativedelta(**parse_date(period))
        if ttl:
            timedelta = relativedelta(**parse_date(ttl))
            earliest_timestamp = df[entity_timestamp_field].map(lambda x: x - timedelta)

        if is_label:  # get forward data
            if include:
                candidates = (df[entity_timestamp_field] <= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] > df[source_timestamp_field].map(lambda x: x - period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] < df[source_timestamp_field]) & (
                    df[entity_timestamp_field] >= df[source_timestamp_field].map(lambda x: x - period)
                )

                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
            return df[candidates]
        else:  # get backward data
            if include:
                candidates = (df[entity_timestamp_field] >= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] < df[source_timestamp_field].map(lambda x: x + period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] > df[source_timestamp_field]) & (
                    df[entity_timestamp_field] <= df[source_timestamp_field].map(lambda x: x + period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
        return df[candidates]

    @classmethod
    def point_in_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        sort_by = [source_timestamp_field]
        if created_timestamp_field:
            sort_by.append(created_timestamp_field)

        df.sort_values(by=sort_by, ascending=False, ignore_index=True, inplace=True)
        df.drop_duplicates(
            subset=group_keys + [entity_timestamp_field],
            keep="first",
            inplace=True,
        )
        return df

    @classmethod
    def point_on_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        if created_timestamp_field:
            df.sort_values(by=[created_timestamp_field], ascending=False, ignore_index=True, inplace=True)
            df.drop_duplicates(
                subset=group_keys + [entity_timestamp_field, source_timestamp_field],
                keep="first",
                inplace=True,
            )
        return df
