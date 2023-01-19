from __future__ import annotations
from typing import List
from pypika import Query, Parameter, Table, PostgreSQLQuery

from ..definitions import (
    SqlSource,
    OfflinePersistEngine,
    OfflinePersistEngineType,
    BackOffTime,
    PersistFeatureView,
    PersistLabelView,
)
from ..offline_stores.offline_postgres_store import OfflinePostgresStore
from ..common.time_field import *


class OfflinePgsqlPersistEngine(OfflinePersistEngine):

    type: OfflinePersistEngineType = OfflinePersistEngineType.PGSQL

    store: OfflinePostgresStore

    def materialize(
        self,
        feature_views: List[PersistFeatureView],
        label_view: PersistLabelView,
        destination: SqlSource,
        back_off_time: BackOffTime,
    ):
        join_sql = self.store.read(
            source=label_view.source,
            features=label_view.labels,
            join_keys=label_view.join_keys,
            alias=ENTITY_EVENT_TIMESTAMP_FIELD,
        ).where(
            (Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) <= back_off_time.end)
            & (Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) >= back_off_time.start)
        )

        feature_names = []
        label_names = [label.name for label in label_view.labels]
        for feature_view in feature_views:
            source_sql = self.store.read(
                source=feature_view.source,
                features=feature_view.features,
                join_keys=feature_view.join_keys,
            )
            feature_names += [feature.name for feature in feature_view.features]

            keep_columns = label_view.join_keys + feature_names + label_names + [ENTITY_EVENT_TIMESTAMP_FIELD]
            join_sql = self.store._point_in_time_join(
                entity_df=join_sql,
                source_df=source_sql,
                timestamp_field=feature_view.source.timestamp_field,
                created_timestamp_field=feature_view.source.created_timestamp_field,
                ttl=feature_view.ttl,
                join_keys=feature_view.join_keys,
                include=True,
                how="right",
            ).select(Parameter(f"{', '.join(keep_columns)}"))

        data_columns = label_view.join_keys + feature_names + label_names
        unique_columns = label_view.join_keys + [DEFAULT_EVENT_TIMESTAMP_FIELD]
        join_sql = Query.from_(join_sql).select(
            Parameter(f"{', '.join(data_columns)}"),
            Parameter(f"{ENTITY_EVENT_TIMESTAMP_FIELD} as {DEFAULT_EVENT_TIMESTAMP_FIELD}"),
            Parameter(f"current_timestamp as {MATERIALIZE_TIME}"),
        )

        with self.store.psy_conn as con:
            with con.cursor() as cursor:
                cursor.execute(f"select to_regclass('{destination.query}')")
                (table_name,) = cursor.fetchone()
                is_table_exists = table_name in destination.query

                if not is_table_exists:
                    # create table from select.
                    cursor.execute(
                        Query.create_table(destination.query).as_select(join_sql).get_sql(quote_char="")
                    )

                    # add unique constraint
                    # TODO: vs unique index.
                    cursor.execute(
                        f"alter table {destination.query} add constraint unique_key_{destination.query.split('.')[-1]} unique ({Parameter(', '.join(unique_columns))})"
                    )
                else:
                    table = Table(destination.query)
                    all_columns = data_columns + [DEFAULT_EVENT_TIMESTAMP_FIELD] + [MATERIALIZE_TIME]

                    insert_sql = (
                        PostgreSQLQuery.into(table)
                        .columns(*all_columns)
                        .from_(join_sql)
                        .select(Parameter(f"{','.join(all_columns)}"))
                        .on_conflict(*unique_columns)
                    )
                    for c in all_columns:
                        insert_sql = insert_sql.do_update(table.field(c), Parameter(f"excluded.{c}"))

                    cursor.execute(insert_sql.get_sql(quote_char=""))
