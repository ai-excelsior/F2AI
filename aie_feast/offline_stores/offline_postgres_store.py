from __future__ import annotations
import uuid
import pandas as pd
from io import StringIO
from typing import List, Optional, Set, TYPE_CHECKING, Union
from pydantic import Field, PrivateAttr
from pypika import Query, Parameter, functions as fn, JoinType, Table
from aie_feast.definitions import Feature
from aie_feast.common.source import SqlSource
from aie_feast.common.utils import TIME_COL, build_agg_query, build_filter_time_query
from aie_feast.period import Period
from aie_feast.common.utils import convert_dtype_to_sqlalchemy_type
from .offline_store import OfflineStore, OfflineStoreType
from aie_feast.views import LabelView, FeatureView
from aie_feast.definitions import Entity


DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
DEFAULT_CREATED_TIMESTAMP_FIELD = "created_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
SOURCE_CREATED_TIMESTAMP_FIELD = "_created_timestamp_"
QUERY_COL = "query_timestamp"

if TYPE_CHECKING:
    from psycopg2 import connection
    from aie_feast.service import Service


class OfflinePostgresStore(OfflineStore):

    type: OfflineStoreType = OfflineStoreType.PGSQL

    host: str
    port: str = "5432"
    database: str = "postgres"
    db_schema: str = Field(alias="schema", default="public")
    user: str
    password: str

    _conn: Optional[connection] = PrivateAttr(default=None)

    @property
    def connect_url(self):
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def psy_conn(self):
        from psycopg2 import connect

        if self._conn is None:
            self._conn = connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.database,
            )

        return self._conn

    def get_offline_source(self, service: Service) -> SqlSource:
        return SqlSource(
            name=service.name,
            query=service.materialize_path,
            timestamp_field="event_timestamp",
            created_timestamp_field="materialize_time",
        )

    def get_sqlalchemy_engine(self):
        from sqlalchemy import create_engine

        return create_engine(self.connect_url)

    def _get_entity(
        self,
        entity_df,
        source,
        join_keys,
        timestamp_field: str = DEFAULT_EVENT_TIMESTAMP_FIELD,
        alias: str = ENTITY_EVENT_TIMESTAMP_FIELD,
    ):
        table_name = None
        if isinstance(entity_df, pd.DataFrame):
            table_name = self.upload_df(
                entity_df=entity_df,
                source=source,
                join_keys=join_keys,
            )
            entity_df = SqlSource(name=table_name, timestamp_field=timestamp_field)

        if isinstance(entity_df, SqlSource):
            entity_df = self.read(source=entity_df, features=[], join_keys=join_keys, alias=alias)

        return entity_df, table_name

    def _create_sqlalchemy_table(self, df: pd.DataFrame, table_name: str, index_columns: List[str]):
        from sqlalchemy import Column, Table, MetaData

        column_names_and_types = [
            (str(df.columns[i]), convert_dtype_to_sqlalchemy_type(df.iloc[:, i]))
            for i in range(len(df.columns))
        ]
        columns = [Column(name, typ, index=name in index_columns) for name, typ in column_names_and_types]
        return Table(table_name, MetaData(), *columns)

    def upload_df(
        self, entity_df: pd.DataFrame, source: SqlSource, join_keys: List[str] = [], table_name: str = None
    ):
        time_cols = [source.timestamp_field]
        if source.created_timestamp_field:
            time_cols.append(source.created_timestamp_field)

        if table_name is None:
            table_name = f"f2ai_tmp_{uuid.uuid4().hex[:8]}"

        engine = self.get_sqlalchemy_engine()
        table = self._create_sqlalchemy_table(
            entity_df, table_name=table_name, index_columns=time_cols + join_keys
        )
        table.create(engine, checkfirst=True)

        with self.psy_conn.cursor() as cursor:
            buffer = StringIO()
            entity_df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)

            cursor.copy_from(buffer, table=table_name, sep=",", columns=entity_df.columns)
            self.psy_conn.commit()

        return table_name

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

    def materialize(
        self,
        service: Service,
        feature_views: List[FeatureView],
        label_views: List[LabelView],
        sources: List[SqlSource],
        entities: List[Entity],
        incremental_begin,
    ):

        label_view = service.get_label_view(label_views)

        max_timestamp = Query.from_(service.materialize_path).select(
            fn.Max(Parameter(sources[label_view.batch_source].timestamp_field))
        )
        max_timestamp_label = Query.from_(label_view.batch_source).select(
            fn.Max(Parameter(sources[label_view.batch_source].timestamp_field))
        )

        return max_timestamp, max_timestamp_label

    def stats(
        self,
        entity_df: Union[SqlSource, Query],
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
        entity_df, table_name = self._get_entity(entity_df, source, group_keys if join_keys else [])

        if join_keys:
            sql_join = Query.from_(source_df).inner_join(entity_df).using(*group_keys)
        else:
            sql_join = Query.from_(source_df).cross_join(entity_df).cross()

        sql_filter = build_filter_time_query(sql_join, start, include)
        sql_agg = build_agg_query(sql_filter, features, group_keys, fn, keys_only)

        df = self._get_dataframe(group_keys, [], [f"{c.name}_{fn}" for c in features], sql_agg)
        self._drop_table(table_name)

        return df

    def get_latest_entities(self, source: SqlSource, group_keys: list = [], entity_df: pd.DataFrame = None):
        source_df = self.read(source=source, features=[], join_keys=group_keys)
        q = Query.from_(source_df).groupby(*group_keys)

        if entity_df is not None:
            entity_df, table_name = self._get_entity(entity_df, source, group_keys)
            q = q.inner_join(entity_df).using(*group_keys)

        sql_result = q.select(*group_keys, fn.Max(Parameter(SOURCE_EVENT_TIMESTAMP_FIELD)))
        df = self._get_dataframe(group_keys, [TIME_COL], [], sql_result)
        if entity_df is not None:
            self._drop_table(table_name)
        return df

    def get_features(
        self,
        entity_df: Union[SqlSource, Query],
        features: Set[Feature],
        source: SqlSource,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ):

        source_df = self.read(source=source, features=features, join_keys=join_keys)
        entity_df, table_name = self._get_entity(entity_df, source, join_keys)
        feature_names = [feature.name for feature in features]

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
        sql_result = sql_query.select(
            Parameter(
                f"{','.join(join_keys + [ENTITY_EVENT_TIMESTAMP_FIELD+' as '+DEFAULT_EVENT_TIMESTAMP_FIELD ] + feature_names)}"
            )
        )
        # execute sql
        df = self._get_dataframe(join_keys, [DEFAULT_EVENT_TIMESTAMP_FIELD], feature_names, sql_result)
        self._drop_table(table_name)
        return df

    def _drop_table(self, table_name):
        with self.psy_conn.cursor() as cursor:
            cursor.execute(f'drop table if exists "{table_name}"')
            self.psy_conn.commit()

    def _get_dataframe(self, join_keys, timecol, feature_names, sql_result):
        with self.psy_conn.cursor() as cursor:
            cursor.execute(sql_result.get_sql())
            data = cursor.fetchall()
            self.psy_conn.commit()

            return pd.DataFrame(data, columns=join_keys + timecol + feature_names)

    def get_period_features(
        self,
        entity_df: Union[pd.DataFrame, Query, SqlSource],
        features: Set[Feature],
        source: SqlSource,
        period: Period,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ):
        feature_names = [feature.name for feature in features]
        source_df = self.read(source=source, features=features, join_keys=join_keys)
        entity_df, table_name = self._get_entity(entity_df, source, join_keys)

        sql_result = self.point_on_time_join(
            entity_df=entity_df,
            source_df=source_df,
            period=period,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )

        # query source
        time_cols = [f"{source.timestamp_field} as {SOURCE_EVENT_TIMESTAMP_FIELD}"]
        if source.created_timestamp_field:
            time_cols.append(f"{source.created_timestamp_field} as {DEFAULT_CREATED_TIMESTAMP_FIELD}")
        df = (
            Query.from_(source.query)
            .select(Parameter(",".join(join_keys + time_cols + feature_names)))
            .as_("df")
        )

        # join features
        if len(join_keys) > 0:
            sql_join = (
                Query.from_(
                    Query.from_(table_name).select(
                        Parameter(",".join(join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]))
                    )
                )
                .inner_join(df)
                .using(*join_keys)
                .select(Parameter(f"df.*, {DEFAULT_EVENT_TIMESTAMP_FIELD}"))
                .as_("sql_join")
            )
        else:
            sql_join = (
                Query.from_(table_name)
                .cross_join(df)
                .cross()
                .select(Parameter(f"df.*, {DEFAULT_EVENT_TIMESTAMP_FIELD}"))
                .as_("sql_join")
            )

        # get period features, filter feature by period
        if period.is_neg:
            sql_filter_with_period = (
                Query.from_(sql_join)
                .select(sql_join.star)
                .where(
                    Parameter(
                        f" ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz <= {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz) and ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz > {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                    if include
                    else Parameter(
                        f" ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz < {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz) and ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz >= {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                )
                .as_("sql_filter_with_period")
            )
        else:
            sql_filter_with_period = (
                Query.from_(sql_join)
                .select(sql_join.star)
                .where(
                    Parameter(
                        f" ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz >= {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz) and ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz < {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                    if include
                    else Parameter(
                        f" ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz > {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz) and ({SOURCE_EVENT_TIMESTAMP_FIELD}::timestamptz <= {DEFAULT_EVENT_TIMESTAMP_FIELD}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                )
                .as_("sql_filter_with_period")
            )

        # get point in time feature
        if source.created_timestamp_field:
            sql_result = (
                Query.from_(
                    Query.from_(sql_filter_with_period).select(
                        sql_filter_with_period.star,
                        Parameter(
                            f"row_number() over (partition by ({','.join(join_keys+[DEFAULT_EVENT_TIMESTAMP_FIELD, SOURCE_EVENT_TIMESTAMP_FIELD])}) order by {source.created_timestamp_field} DESC)"
                        ),
                    )
                )
                .select(
                    Parameter(
                        ",".join(
                            join_keys
                            + [
                                f"{DEFAULT_EVENT_TIMESTAMP_FIELD} as {QUERY_COL}",
                                f"{SOURCE_EVENT_TIMESTAMP_FIELD} as {DEFAULT_EVENT_TIMESTAMP_FIELD}",
                            ]
                            + feature_names
                        )
                    )
                )
                .where(Parameter("row_number=1"))
            )
        else:
            sql_result = Query.from_(sql_filter_with_period).select(
                Parameter(
                    ",".join(
                        join_keys
                        + [
                            f"{DEFAULT_EVENT_TIMESTAMP_FIELD} as {QUERY_COL}",
                            f"{SOURCE_EVENT_TIMESTAMP_FIELD} as {DEFAULT_EVENT_TIMESTAMP_FIELD}",
                        ]
                        + feature_names
                    )
                ),
            )

        # execute sql
        df = self._get_dataframe(
            join_keys, [QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD], feature_names, sql_result
        )
        self._drop_table(table_name)
        return df

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
        entity_df = entity_df.rename(columns={DEFAULT_EVENT_TIMESTAMP_FIELD: ENTITY_EVENT_TIMESTAMP_FIELD})
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
            df = source_df.merge(entity_df, on=join_keys, how=how)
        else:
            df = source_df.merge(entity_df, how="cross")

        df = cls.point_on_time_filter(df, period, include=include, ttl=ttl, is_label=is_label)
        df = cls.point_on_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(columns=[created_timestamp_field], erros="ignore").rename(
            columns={
                ENTITY_EVENT_TIMESTAMP_FIELD: QUERY_COL,
                SOURCE_EVENT_TIMESTAMP_FIELD: DEFAULT_EVENT_TIMESTAMP_FIELD,
            }
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
