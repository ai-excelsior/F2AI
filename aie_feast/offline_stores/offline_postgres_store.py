from __future__ import annotations
import uuid
import pandas as pd
from io import StringIO
from typing import List, Optional, Set, TYPE_CHECKING, Union
from pydantic import Field, PrivateAttr
from pypika import Query, Parameter, functions as fn, JoinType
from aie_feast.definitions import Feature, Entity, Period, LabelView, FeatureView
from aie_feast.common.source import SqlSource
from aie_feast.common.utils import build_agg_query, build_filter_time_query
from aie_feast.common.utils import convert_dtype_to_sqlalchemy_type
from .offline_store import OfflineStore, OfflineStoreType
from aie_feast.definitions import Entity
from datetime import datetime


DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
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
            timestamp_field=DEFAULT_EVENT_TIMESTAMP_FIELD,
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
        **kwargs,
    ):
        table_name = None
        #  features = []
        if isinstance(entity_df, pd.DataFrame):
            table_name = self._upload_entity_df(
                entity_df=entity_df,
                source=source,
                join_keys=join_keys,
            )
            entity_df = SqlSource(name=table_name, timestamp_field=timestamp_field)

        if isinstance(entity_df, SqlSource):
            entity_df = self.read(
                source=entity_df, features=kwargs.get("residue", []), join_keys=join_keys, alias=alias
            )

        return entity_df, table_name

    def _create_sqlalchemy_table(self, df: pd.DataFrame, table_name: str, index_columns: List[str]):
        from sqlalchemy import Column, Table, MetaData

        column_names_and_types = [
            (str(df.columns[i]), convert_dtype_to_sqlalchemy_type(df.iloc[:, i]))
            for i in range(len(df.columns))
        ]
        columns = [Column(name, typ, index=name in index_columns) for name, typ in column_names_and_types]
        return Table(table_name, MetaData(), *columns)

    def _upload_entity_df(
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
        time_columns = []
        if source.timestamp_field:
            time_columns.append(f"{source.timestamp_field} as {alias}")
        if source.created_timestamp_field:
            time_columns.append(source.created_timestamp_field)

        feature_columns = [feature if isinstance(feature, str) else feature.name for feature in features]
        all_columns = list(set(time_columns + join_keys + feature_columns))
        source_df = Query.from_(source.query).select(Parameter(",".join(all_columns)))
        return source_df

    def materialize_dbt(
        self,
        service: Service,
        label_views: List[LabelView],
        sources: List[SqlSource],
    ):

        label_view = service.get_label_view(label_views)

        max_timestamp = Query.from_(service.materialize_path).select(
            fn.Max(Parameter(sources[label_view.batch_source].timestamp_field))
        )
        max_timestamp_label = Query.from_(label_view.batch_source).select(
            fn.Max(Parameter(sources[label_view.batch_source].timestamp_field))
        )

        return max_timestamp, max_timestamp_label

    def materialize(
        self,
        service: Service,
        feature_views: List[FeatureView],
        label_views: List[LabelView],
        sources: List[SqlSource],
        entities: List[Entity],
        start: str,
        end: str,
        fromnow: str,
    ):

        label_view = service.get_label_view(label_views)
        feature_views = service.get_feature_views(feature_views)
        labels = label_view.get_label_objects()
        all_entity_cols = [entities[entity].join_keys[0] for entity in label_view.entities]
        all_feature_names = set([label.name for label in labels])
        entity_dataframe = self.read(
            source=sources[label_view.batch_source],
            features=labels,
            join_keys=all_entity_cols,
            alias=ENTITY_EVENT_TIMESTAMP_FIELD,
        )

        if fromnow:
            entity_dataframe = entity_dataframe.where(Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) >= fromnow)
        elif start and end:
            entity_dataframe = entity_dataframe.where(
                Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) > start
                and Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) < end
            )
        else:
            raise ValueError("either (start,end) or fromnow should be given")

        time_columns = [Parameter(ENTITY_EVENT_TIMESTAMP_FIELD)]
        result_sql = entity_dataframe

        for featureview in feature_views:
            entity_cols = [entities[entity].join_keys[0] for entity in featureview.entities]
            features = featureview.get_feature_objects()
            source = sources[featureview.batch_source]
            all_join_cols = time_columns + entity_cols

            source_df = self.read(source=source, features=features, join_keys=entity_cols)

            sql_query = self._point_in_time_join(
                entity_df=entity_dataframe.as_(f"{featureview.name}_entity_df"),
                source_df=source_df.as_(f"{featureview.name}_source_df"),
                timestamp_field=source.timestamp_field,
                created_timestamp_field=source.created_timestamp_field,
                ttl=featureview.ttl,
                join_keys=entity_cols,
                include=True,
            )
            feature_names = [
                feature.name
                for feature in features
                if feature.name not in [feature.name for feature in labels]
            ]
            all_feature_names = all_feature_names | set(feature_names)
            result_sql = (
                Query.from_(result_sql)
                .left_join(
                    sql_query.select(
                        Parameter(
                            f"{','.join(entity_cols + [ENTITY_EVENT_TIMESTAMP_FIELD ] + feature_names)}"
                        )
                    ).as_(featureview.name)
                )
                .using(*all_join_cols)
                .select("*")
                .as_(f"{featureview.name}_join")
            )
        df = self._get_dataframe(
            join_keys=all_entity_cols,
            timecol=[ENTITY_EVENT_TIMESTAMP_FIELD],
            feature_names=list(all_feature_names),
            sql_result=result_sql,
        )
        df["materialize_time"] = pd.to_datetime(datetime.now(), utc=True)

        return df
        # return df

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
        df = self._get_dataframe(group_keys, [DEFAULT_EVENT_TIMESTAMP_FIELD], [], sql_result)
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

        entity_cols = kwargs.pop("entity_cols", [])
        source_df = self.read(source=source, features=features, join_keys=join_keys + entity_cols)

        residue = [c for c in entity_df.columns if c not in join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]]
        entity_df, table_name = self._get_entity(entity_df, source, join_keys, residue=residue)

        sql_query = self._point_in_time_join(
            entity_df=entity_df,
            source_df=source_df,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )
        # execute sql
        feature_names = [feature.name for feature in features]
        sql_result = sql_query.select(
            Parameter(
                f"{','.join(list(set(join_keys + entity_cols))+ [ENTITY_EVENT_TIMESTAMP_FIELD ] + list(set(residue + feature_names)))}"
            )
        )
        df = self._get_dataframe(
            list(set(join_keys + entity_cols)),
            [DEFAULT_EVENT_TIMESTAMP_FIELD],
            list(set(residue + feature_names)),
            sql_result,
        )
        self._drop_table(table_name)
        return df.sort_values(
            by=list(set(join_keys + entity_cols)) + [DEFAULT_EVENT_TIMESTAMP_FIELD],
            ascending=True,
            ignore_index=True,
        )

    def _drop_table(self, table_name):
        with self.psy_conn.cursor() as cursor:
            cursor.execute(f'drop table if exists "{table_name}"')
            self.psy_conn.commit()

    def _get_dataframe(self, join_keys: list, timecol: list, feature_names: list, sql_result):
        """_summary_

        Args:
            join_keys (list): entities
            timecol (list): time cols
            feature_names (list): feature cols
            sql_result (_type_): sql query

        Returns:
            pd.DataFrame:
        """
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
        entity_cols = kwargs.pop("entity_cols", [])
        source_df = self.read(source=source, features=features, join_keys=join_keys + entity_cols)

        residue = [c for c in entity_df.columns if c not in join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]]
        entity_df, table_name = self._get_entity(entity_df, source, join_keys, residue=residue)

        sql_query = self._point_on_time_join(
            entity_df=entity_df,
            source_df=source_df,
            period=period,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )

        feature_names = [feature.name for feature in features]
        sql_result = sql_query.select(
            Parameter(
                f"{','.join(list(set(join_keys + entity_cols))+ [ENTITY_EVENT_TIMESTAMP_FIELD,SOURCE_EVENT_TIMESTAMP_FIELD] + list(set(feature_names+residue)))}"
            )
        )
        df = self._get_dataframe(
            list(set(join_keys + entity_cols)),
            [QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD],
            list(set(feature_names + residue)),
            sql_result,
        )
        self._drop_table(table_name)
        return df.sort_values(
            by=[QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD], ascending=True, ignore_index=True
        )

    def query(self, query: str, return_df: bool = True, *args, **kwargs):
        with self.psy_conn.cursor() as cursor:
            cursor.execute(query)
            if return_df:
                return pd.DataFrame(cursor.fetchall())
            else:
                return cursor.fetchall()

    @classmethod
    def _point_in_time_join(
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
            sql_query = cls._point_in_time_filter(sql_join, include=include, ttl=ttl)
            sql_query = cls._point_in_time_latest(sql_query, join_keys, created_timestamp_field)
            return sql_query
        return sql_join

    @classmethod
    def _point_on_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        period: str,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how="inner",
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
        # query source

        if len(join_keys) > 0:
            sql_join = Query.from_(source_df).join(entity_df, JoinType.__getattr__(how)).using(*join_keys)
        else:
            sql_join = Query.from_(source_df).cross_join(entity_df).cross()

        sql_query = cls._point_on_time_filter(sql_join, period=period, include=include, ttl=ttl)
        sql_query = cls._point_on_time_latest(sql_query, join_keys, created_timestamp_field)

        return sql_query

    @classmethod
    def _point_in_time_filter(
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
                    > Parameter(entity_timestamp_field) - ttl.to_pgsql_interval()
                )
        else:
            candidates = Parameter(entity_timestamp_field) >= Parameter(source_timestamp_field)
            if ttl:
                candidates = candidates & (
                    Parameter(source_timestamp_field)
                    >= Parameter(entity_timestamp_field) - ttl.to_pgsql_interval()
                )
        return df.where(candidates)

    @classmethod
    def _point_on_time_filter(
        cls,
        df: Query,
        period: str,
        include: bool = True,
        ttl: Optional[Period] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        if ttl:
            earliest_timestamp = Parameter(entity_timestamp_field) - ttl.to_pgsql_interval()

        if period.is_neg:
            if include:
                candidates = (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    >= Parameter(f"{source_timestamp_field}::timestamptz")
                ) & (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    < Parameter(f"{source_timestamp_field}::timestamptz - {period.to_pgsql_interval()}")
                )
                if ttl:
                    candidates = candidates & (Parameter(source_timestamp_field) > earliest_timestamp)
            else:
                candidates = (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    > Parameter(f"{source_timestamp_field}::timestamptz")
                ) & (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    <= Parameter(f"{source_timestamp_field}::timestamptz - {period.to_pgsql_interval()}")
                )
                if ttl:
                    candidates = candidates & (Parameter(source_timestamp_field) >= earliest_timestamp)
        else:
            if include:
                candidates = (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    <= Parameter(f"{source_timestamp_field}::timestamptz")
                ) & (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    > Parameter(f"{source_timestamp_field}::timestamptz - {period.to_pgsql_interval()}")
                )
                if ttl:
                    candidates = candidates & (Parameter(source_timestamp_field) > earliest_timestamp)
            else:
                candidates = (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    < Parameter(f"{source_timestamp_field}::timestamptz")
                ) & (
                    Parameter(f"{entity_timestamp_field}::timestamptz")
                    >= Parameter(f"{source_timestamp_field}::timestamptz - {period.to_pgsql_interval()}")
                )
                if ttl:
                    candidates = candidates & (Parameter(source_timestamp_field) >= earliest_timestamp)
        return df.where(candidates)

    @classmethod
    def _point_in_time_latest(
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
    def _point_on_time_latest(
        cls,
        df: Query,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        if created_timestamp_field:
            latest_time = df.groupby(
                Parameter(",".join(group_keys + [entity_timestamp_field, source_timestamp_field]))
            ).select(
                fn.Max(Parameter(created_timestamp_field), created_timestamp_field),
                Parameter(",".join(group_keys + [entity_timestamp_field, source_timestamp_field])),
            )
            df = df.inner_join(latest_time).using(
                *(group_keys + [entity_timestamp_field, created_timestamp_field])
            )

        return df
