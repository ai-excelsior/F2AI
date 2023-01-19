from __future__ import annotations
import uuid
import pandas as pd
import datetime
from io import StringIO
from typing import List, Optional, Set, TYPE_CHECKING, Union, Tuple
from pydantic import Field, PrivateAttr
from pypika import Query, Parameter, functions as fn, JoinType, Field as PikaField
from pypika.queries import QueryBuilder

from ..definitions import (
    Feature,
    Period,
    Service,
    SqlSource,
    OfflineStoreType,
    OfflineStore,
    StatsFunctions,
)
from ..common.utils import convert_dtype_to_sqlalchemy_type
from ..common.time_field import (
    DEFAULT_EVENT_TIMESTAMP_FIELD,
    ENTITY_EVENT_TIMESTAMP_FIELD,
    SOURCE_EVENT_TIMESTAMP_FIELD,
    QUERY_COL,
    MATERIALIZE_TIME,
)


if TYPE_CHECKING:
    from psycopg2 import connection


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

        if self._conn is None or self._conn.closed:
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
            query=f"{self.materialize_path}.{service.name}",
            timestamp_field=DEFAULT_EVENT_TIMESTAMP_FIELD,
            created_timestamp_field=MATERIALIZE_TIME,
        )

    def get_sqlalchemy_engine(self):
        from sqlalchemy import create_engine

        return create_engine(self.connect_url)

    def _get_entity(
        self,
        entity_df: pd.DataFrame,
        source: SqlSource,
        join_keys: list,
        timestamp_field: str = DEFAULT_EVENT_TIMESTAMP_FIELD,
        alias: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        **kwargs,
    ) -> Tuple(Query, str):

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
    ) -> QueryBuilder:
        """_summary_

        Args:
            source (SqlSource): data source to build query from
            features (Set[Feature], optional): features cols to select. Defaults to {}.
            join_keys (List[str], optional): entity cols to select. Defaults to [].
            alias (str, optional): alias of time col to avoid duplicates of merge. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            Query: _description_
        """
        time_columns = []
        if source.timestamp_field:
            time_columns.append(f"{source.timestamp_field} as {alias}")
        if source.created_timestamp_field and alias != ENTITY_EVENT_TIMESTAMP_FIELD:
            time_columns.append(source.created_timestamp_field)

        feature_columns = [feature if isinstance(feature, str) else feature.name for feature in features]
        all_columns = list(set(time_columns + join_keys + feature_columns))
        source_sql = Query.from_(source.query).select(Parameter(",".join(all_columns)))
        return source_sql

    def stats(
        self,
        source: SqlSource,
        fn: StatsFunctions,
        features: Set[Feature] = {},
        group_keys: List[str] = [],
        start: datetime.datetime = None,
        end: datetime.datetime = None,
    ) -> pd.DataFrame:
        source_sql = Query.from_(source.query)
        if fn == StatsFunctions.UNIQUE:
            features = []

        if start is not None:
            source_sql = source_sql.where(PikaField(source.timestamp_field) >= start)
        if end is not None:
            source_sql = source_sql.where(PikaField(source.timestamp_field) <= end)
        stats_sql = build_stats_query(source_sql, features=features, stats_fn=fn, group_keys=group_keys)

        return self._get_dataframe(stats_sql, group_keys + [feature.name for feature in features])

    def get_latest_entities(
        self,
        source: SqlSource,
        join_keys: List[str] = [],
        group_keys: list = None,
        entity_df: pd.DataFrame = None,
        start: datetime = None,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            source (SqlSource): data source of featureview
            join_keys List[str]:list to join ,default to [],
            group_keys (list, optional): dimension of stats. Defaults to [].
            entity_df (pd.DataFrame, optional): query condition specified by users. Defaults to None.
            start: start time
        Returns:
            pd.DataFrame: _description_
        """
        source_df = self.read(source=source, features=[], join_keys=group_keys)
        source_df = source_df.where(Parameter(source.timestamp_field) > Parameter(f"'{start}'::timestamptz"))
        entity_df, table_name = self._get_entity(entity_df, source, join_keys)

        if join_keys:
            sql_join = Query.from_(source_df).join(entity_df, JoinType.__getattr__("inner")).using(*join_keys)
        else:
            sql_join = Query.from_(source_df).cross_join(entity_df).cross()

        sql_query = self._point_in_time_filter(sql_join)
        sql_result = sql_query.groupby(*group_keys).select(
            *group_keys, fn.Max(Parameter(SOURCE_EVENT_TIMESTAMP_FIELD))
        )
        df = self._get_dataframe(sql_result, group_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD])
        if entity_df is not None:
            self._drop_table(table_name)
        return df

    def get_features(
        self,
        entity_df: Union[SqlSource, Query, pd.DataFrame],
        features: Set[Feature],
        source: SqlSource,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            entity_df (pd.DataFrame): query condition specified by users
            features:(Set(Feature)): features to be queried
            source (Sqlsource): data source of featureview
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
        Returns:
            Query: _description_
        """

        entity_cols = kwargs.pop("entity_cols", [])
        source_df = self.read(source=source, features=features, join_keys=join_keys + entity_cols)
        if isinstance(entity_df, pd.DataFrame):
            residue = [c for c in entity_df.columns if c not in join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]]
        else:
            residue = kwargs.get("residue", [])

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
            sql_result,
            list(set(join_keys + entity_cols))
            + [DEFAULT_EVENT_TIMESTAMP_FIELD]
            + list(set(residue + feature_names)),
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

    def _get_dataframe(
        self,
        sql_result: Query,
        columns: List[str],
    ) -> pd.DataFrame:
        """_summary_

        Args:
            join_keys (list): entities
            columns (List[str]): column names

        Returns:
            pd.DataFrame:
        """
        with self.psy_conn.cursor() as cursor:
            cursor.execute(sql_result if isinstance(sql_result, str) else sql_result.get_sql())
            data = cursor.fetchall()
            self.psy_conn.commit()

            return pd.DataFrame(data, columns=columns)

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
    ) -> pd.DataFrame:
        """_summary_

        Args:
            entity_df (pd.DataFrame): query condition specified by users
            features:(Set(Feature)): features to be queried
            source (Sqlsource): data source of featureview
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
             ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
        Returns:
            Query: _description_
        """
        entity_cols = kwargs.pop("entity_cols", [])
        source_df = self.read(source=source, features=features, join_keys=join_keys + entity_cols)

        if isinstance(entity_df, pd.DataFrame):
            residue = [c for c in entity_df.columns if c not in join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]]
        else:
            residue = kwargs.get("residue", [])

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
            sql_result,
            list(set(join_keys + entity_cols))
            + [QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD]
            + list(set(feature_names + residue)),
        )
        self._drop_table(table_name)
        return df.sort_values(
            by=[QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD],
            ascending=True,
            ignore_index=True,
        )

    def query(self, query: str, return_df: bool = True, *args, **kwargs) -> pd.DataFrame:
        """_summary_

        Args:
            query (str): user specified query to execute
            return_df (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
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
    ) -> Query:
        """_summary_

        Args:
            entity_df (Query): query condition specified by users
            source_df (Query): query taken from data
            created_timestamp_field (Optional[str], optional): timestamp of upload of datas. Defaults to None.
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
            how (str, optional): join method. Defaults to "inner".

        Returns:
            Query: _description_
        """

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
        entity_df: Query,
        source_df: Query,
        period: Period,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how="inner",
    ) -> Query:
        """_summary_

        Args:
            entity_df (Query): query condition specified by users
            source_df (Query): query taken from data
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            created_timestamp_field (Optional[str], optional): timestamp of upload of datas. Defaults to None.
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
            how (str, optional): join method. Defaults to "inner".

        Returns:
            Query: _description_
        """

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
    ) -> Query:
        """_summary_

        Args:
            df (Query): Query to filter
            include (bool, optional): whether take < entity_timestamp_field or <= entity_timestamp_field. Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp speicified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            Query: _description_
        """

        if include:
            candidates = Parameter(entity_timestamp_field) >= Parameter(source_timestamp_field)
            if ttl:
                candidates = candidates & (
                    Parameter(source_timestamp_field)
                    > Parameter(entity_timestamp_field) - Parameter(ttl.to_pgsql_interval())
                )
        else:
            candidates = Parameter(entity_timestamp_field) >= Parameter(source_timestamp_field)
            if ttl:
                candidates = candidates & (
                    Parameter(source_timestamp_field)
                    >= Parameter(entity_timestamp_field) - Parameter(ttl.to_pgsql_interval())
                )
        return df.where(candidates)

    @classmethod
    def _point_on_time_filter(
        cls,
        df: Query,
        period: Period,
        include: bool = True,
        ttl: Optional[Period] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ) -> Query:
        """_summary_

        Args:
            df (Query): Query to filter
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            include (bool, optional): whether take < entity_timestamp_field or <= entity_timestamp_field. Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp speicified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            Query: _description_
        """

        earliest_timestamp = None
        if ttl:
            earliest_timestamp = Parameter(entity_timestamp_field) - Parameter(ttl.to_pgsql_interval())

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
    ) -> Query:
        """_summary_

        Args:
            df (Query): Query to be filter
            group_keys (List[str], optional): entities cols. Defaults to []
            created_timestamp_field (Optional[str], optional): data upload time col. Defaults to None
            entity_timestamp_field (str, optional):  query time col, Defaults to `ENTITY_EVENT_TIMESTAMP_FIELD`
            source_timestamp_field (str, optional): event taken time col, Defaults to `SOURCE_EVENT_TIMESTAMP_FIELD`

        Returns:
            Query: _description_
        """
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
    ) -> Query:
        """_summary_

        Args:
            df (Query): Query to be filter
            group_keys (List[str], optional): entities cols. Defaults to []
            created_timestamp_field (Optional[str], optional): data upload time col. Defaults to None
            entity_timestamp_field (str, optional):  query time col, Defaults to `ENTITY_EVENT_TIMESTAMP_FIELD`
            source_timestamp_field (str, optional): event taken time col, Defaults to `SOURCE_EVENT_TIMESTAMP_FIELD`

        Returns:
            Query: _description_
        """
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


SQL_AGG_FUNCTIONS = {
    StatsFunctions.AVG: fn.Avg,
    StatsFunctions.MIN: fn.Min,
    StatsFunctions.MAX: fn.Max,
    StatsFunctions.STD: fn.Std,
}

SQL_GROUP_AGG_FUNCTIONS = {
    StatsFunctions.MODE: "MODE()",
    StatsFunctions.MEDIAN: "PERCENTILE_CONT(0.5)",
}


def build_stats_query(
    q: QueryBuilder,
    stats_fn: StatsFunctions,
    features: List[Feature] = [],
    group_keys: List[str] = [],
) -> QueryBuilder:
    if stats_fn == StatsFunctions.UNIQUE:
        selects = [Parameter(name) for name in group_keys]
        return q.select(*selects).distinct()

    if stats_fn in SQL_AGG_FUNCTIONS:
        selects = group_keys + [
            SQL_AGG_FUNCTIONS[stats_fn](Parameter(feature.name), feature.name) for feature in features
        ]
    elif stats_fn in SQL_GROUP_AGG_FUNCTIONS:
        selects = group_keys + [
            Parameter(
                f"{SQL_GROUP_AGG_FUNCTIONS[stats_fn]} WITHIN GROUP (ORDER BY {feature.name}) as {feature.name}"
            )
            for feature in features
        ]

    if len(group_keys) > 0:
        return q.groupby(*group_keys).select(*selects)

    return q.select(*selects)
