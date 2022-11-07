from pydantic import Field
import pandas as pd
from pypika import Query, Parameter, functions as fn, JoinType, Table
from typing import List, Optional, Set
from aie_feast.definitions import Feature
from aie_feast.common.source import SqlSource
from aie_feast.common.utils import build_agg_query, build_filter_time_query
from aie_feast.period import Period
from .offline_store import OfflineStore, OfflineStoreType

TIME_COL = "event_timestamp"
# DEFAULT_CREATED_TIMESTAMP_FIELD = "created_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
SOURCE_CREATED_TIMESTAMP_FIELD = "_created_timestamp_"
QUERY_COL = "query_timestamp"


class OfflinePostgresStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.PGSQL

    host: str
    port: str = "5432"
    database: str = "postgres"
    db_schema: str = Field(alias="schema", default="public")
    user: str
    password: str

    def read(
        self,
        source: SqlSource,
        features: Set[Feature] = {},
        join_keys: List[str] = [],
        alias: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        time_columns = [f"{source.timestamp_field} as {alias}"]
        if source.created_timestamp_field:
            time_columns.append(source.created_timestamp_field)

        feature_columns = [feature.name for feature in features]
        all_columns = list(set(time_columns + join_keys + feature_columns))
        source_df = Query.from_(source.name).select(Parameter(",".join(all_columns)))
        return source_df

    def stats(
        self,
        entity_df: SqlSource,
        features: Set[Feature],
        source: SqlSource,
        start,
        fn: str = "mean",
        group_keys: list = [],
        include: str = "both",
        keys_only: bool = False,
        join_keys: bool = False,
    ):
        source_df = self.read(source=source, features=features, join_keys=group_keys)
        entity_df = self.read(
            source=entity_df,
            features=[],
            join_keys=group_keys if join_keys else [],
            alias=ENTITY_EVENT_TIMESTAMP_FIELD,
        )

        if join_keys:
            sql_join = Query.from_(source_df).inner_join(entity_df).using(*group_keys)
        else:
            sql_join = Query.from_(source_df).cross_join(entity_df).cross()

        sql_filter = build_filter_time_query(sql_join, start, include)
        sql_agg = build_agg_query(sql_filter, features, group_keys, fn, keys_only)

        return sql_agg

    def get_latest_entities(self, source: SqlSource, group_keys: list = [], entities: str = None):
        source_df = self.read(source=source, features=[], join_keys=group_keys)
        q = Query.from_(source_df).groupby(*group_keys)

        if entities:
            q = q.inner_join(Table(entities.name)).using(*group_keys)

        return q.select(*group_keys, fn.Max(Parameter(SOURCE_EVENT_TIMESTAMP_FIELD)))

    def get_features(
        self,
        query: SqlSource,
        features: Set[Feature],
        source: SqlSource,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ):

        source_df = self.read(source=source, features=features, join_keys=join_keys)
        entity_df = self.read(
            source=query, features=[], join_keys=join_keys, alias=ENTITY_EVENT_TIMESTAMP_FIELD
        )

        sql_query = self.point_in_time_join(
            entity_df=entity_df,
            source_df=source_df,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )
        return sql_query.select(
            Parameter(
                f"{','.join(join_keys + [ENTITY_EVENT_TIMESTAMP_FIELD]  +[feature.name for feature in features])}"
            )
        )

    @classmethod
    def point_in_time_join(
        cls,
        entity_df: Query,
        source_df: Query,
        timestamp_field: Optional[str] = None,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how: str = "inner",
    ):

        if ttl:
            min_entity_timestamp = Query.from_(entity_df).select(
                fn.Min(Parameter(ENTITY_EVENT_TIMESTAMP_FIELD)) + ttl.to_pgsql_interval()
            )
            if include:
                pre_fil = Parameter(SOURCE_EVENT_TIMESTAMP_FIELD) > min_entity_timestamp
            else:
                pre_fil = Parameter(SOURCE_EVENT_TIMESTAMP_FIELD) >= min_entity_timestamp
            source_df = source_df.where(pre_fil)

        if len(join_keys) > 0:
            sql_join = Query.from_(source_df).join(entity_df, JoinType.__getattr__(how)).using(*join_keys)
        else:
            sql_join = Query.from_(source_df).cross_join(entity_df).cross()

        if timestamp_field:
            sql_query = cls.point_in_time_filter(sql_join, include=include, ttl=ttl)
            sql_query = cls.point_in_time_latest(sql_join, join_keys, created_timestamp_field)
        return sql_query

    @classmethod
    def point_on_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        period: str,
        timestamp_field: Optional[str],
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        is_label: bool = False,
        how="inner",
    ):
        # renames to keep things simple
        entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field:
            entity_df = entity_df.drop(columns=[created_timestamp_field], erros="ignore")

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - ttl.to_pandas_dateoffset()
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        df = cls.point_on_time_filter(df, period, include=include, ttl=ttl, is_label=is_label)
        df = cls.point_on_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(columns=[created_timestamp_field], erros="ignore").rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: QUERY_COL, SOURCE_EVENT_TIMESTAMP_FIELD: TIME_COL}
        )

    @classmethod
    def point_in_time_filter(
        cls,
        df: Query,
        include: bool = True,
        ttl: Optional[Period] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        if include:
            candidates = Parameter(entity_timestamp_field) >= Parameter(source_timestamp_field)
            if ttl:
                candidates = candidates & (
                    Parameter(source_timestamp_field)
                    > Parameter(entity_timestamp_field) + ttl.to_pgsql_interval()
                )
        else:
            candidates = Parameter(entity_timestamp_field) >= Parameter(source_timestamp_field)
            if ttl:
                candidates = candidates & (
                    Parameter(source_timestamp_field)
                    >= Parameter(entity_timestamp_field) + ttl.to_pgsql_interval()
                )
        return df.where(candidates)

    @classmethod
    def point_on_time_filter(
        cls,
        df: pd.DataFrame,
        period: str,
        include: bool = True,
        ttl: Optional[Period] = None,
        is_label=False,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        if ttl:
            earliest_timestamp = df[entity_timestamp_field] - ttl.to_pandas_dateoffset()

        if is_label:  # get forward data
            if include:
                candidates = (df[entity_timestamp_field] <= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] > df[source_timestamp_field] - ttl.to_pandas_dateoffset()
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] < df[source_timestamp_field]) & (
                    df[entity_timestamp_field] >= df[source_timestamp_field] - ttl.to_pandas_dateoffset()
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
        df: Query,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        sort_by = [source_timestamp_field]
        if created_timestamp_field:
            sort_by.append(created_timestamp_field)

        latest_time = df.groupby(Parameter(",".join(group_keys + [entity_timestamp_field]))).select(
            *[fn.Max(Parameter(item), item) for item in sort_by],
            Parameter(",".join(group_keys + [entity_timestamp_field])),
        )

        return df.inner_join(latest_time).using(*(sort_by + group_keys + [entity_timestamp_field]))

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
