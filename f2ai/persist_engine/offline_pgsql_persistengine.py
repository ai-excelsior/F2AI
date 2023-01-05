from __future__ import annotations
import uuid
from typing import Dict
from pypika import Query, Parameter, Table, PostgreSQLQuery

from ..definitions import OfflinePersistEngine, OfflinePersistEngineType
from ..definitions import SqlSource
from ..offline_stores.offline_postgres_store import OfflinePostgresStore

DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
QUERY_COL = "query_timestamp"
MATERIALIZE_TIME = "materialize_time"


class OfflinePgsqlPersistEngine(OfflinePersistEngine):

    type: OfflinePersistEngineType = OfflinePersistEngineType.PGSQL

    store: OfflinePostgresStore

    def materialize(
        self,
        save_path: SqlSource,
        all_views: Dict,
        start: str = None,
        end: str = None,
        **kwargs,
    ):

        feature_views = all_views["features"]
        label_view = all_views["label"]

        source = label_view["source"]
        joined_frame = self.store.read(
            source=source,
            features=label_view["labels"],
            join_keys=label_view["join_keys"],
            alias=ENTITY_EVENT_TIMESTAMP_FIELD,
        )
        label_names = [label if isinstance(label, str) else label.name for label in label_view["labels"]]

        condition = (Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) <= end) & (
            Parameter(DEFAULT_EVENT_TIMESTAMP_FIELD) >= start
        )

        joined_frame = joined_frame.where(condition)
        feature_names = []
        feature_views.sort(key=lambda x: x["source"].name)
        for featureview in feature_views[::-1]:
            entity_cols = featureview["join_keys"]
            features = featureview["features"]
            feature_names = feature_names + [
                f.name for f in features if f.name not in [label.name for label in label_view["labels"]]
            ]
            if [f.name for f in features if f.name not in [label.name for label in label_view["labels"]]]:
                source = featureview["source"]
                source_df = self.store.read(source=source, features=features, join_keys=entity_cols)
                sql_query = self.store._point_in_time_join(
                    entity_df=joined_frame,
                    source_df=source_df,
                    timestamp_field=source.timestamp_field,
                    created_timestamp_field=source.created_timestamp_field,
                    ttl=featureview["ttl"],
                    join_keys=entity_cols,
                    include=True,
                    how="right",
                )
                joined_frame = sql_query.select(
                    Parameter(
                        f"{','.join(label_view['join_keys']+[ENTITY_EVENT_TIMESTAMP_FIELD]+feature_names+label_names)}"
                    )
                )
        cols_except_time = (
            [
                f.name
                for featureview in feature_views
                for f in featureview["features"]
                if f.name not in [label.name for label in label_view["labels"]]
            ]
            + [label.name for label in label_view["labels"]]
            + label_view["join_keys"]
        )

        unique_keys = [DEFAULT_EVENT_TIMESTAMP_FIELD] + label_view["join_keys"]
        join_query = Query.from_(joined_frame).select(
            Parameter(f"{','.join(cols_except_time)}"),
            Parameter(f"{ENTITY_EVENT_TIMESTAMP_FIELD} as {DEFAULT_EVENT_TIMESTAMP_FIELD}"),
            Parameter(f"current_timestamp as {MATERIALIZE_TIME}"),
        )

        materialize_table = Query.create_table(save_path.query).as_select(join_query)

        try:
            with self.store.psy_conn.cursor() as cursor:
                cursor.execute(
                    materialize_table if isinstance(materialize_table, str) else materialize_table.get_sql(quote_char="")
                )
                cursor.execute(
                    f"alter table {save_path.query} add constraint unique_key_{uuid.uuid4().hex[:8]} unique ({Parameter(','.join(unique_keys))})"
                )
                self.store.psy_conn.commit()
                kwargs["signal"].send(1)
        except:
            self.store.psy_conn.commit()
            with self.store.psy_conn.cursor() as cursor:
                materialize_table = Table(save_path.query)
                all_columns = cols_except_time + [DEFAULT_EVENT_TIMESTAMP_FIELD] + [MATERIALIZE_TIME]

                insert_fns = (
                    PostgreSQLQuery.into(materialize_table)
                    .columns(*all_columns)
                    .from_(join_query)
                    .select(Parameter(f"{','.join(all_columns)}"))
                    .on_conflict(*unique_keys)
                )
                for c in all_columns:
                    insert_fns = insert_fns.do_update(materialize_table.field(c), Parameter(f"excluded.{c}"))
                cursor.execute(
                    f"alter table {save_path.query} add constraint unique_key_{uuid.uuid4().hex[:8]} unique ({Parameter(','.join(unique_keys))})"
                )
                cursor.execute(insert_fns if isinstance(insert_fns, str) else insert_fns.get_sql(quote_char=""))
                self.store.psy_conn.commit()
                kwargs["signal"].send(1)
