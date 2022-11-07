import psycopg2
import pandas as pd
import datetime
from sqlalchemy import create_engine

from aie_feast.offline_stores.offline_postgres_store import OfflinePostgresStore


def psy_conn(store: OfflinePostgresStore):
    return psycopg2.connect(
        user=store.user,
        password=store.password,
        host=store.host,
        port=store.port,
        dbname=store.database,
    )


def close_conn(conn, tables: list = []):

    [remove_table(tab, conn) for tab in tables]
    conn.close()


def execute_sql(sql, conn):
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor


def to_pgsql(df: pd.DataFrame, tbl_name, store: OfflinePostgresStore):
    engine = create_engine(
        f"postgresql+psycopg2://{store.user}:{store.password}@{store.host}:{store.port}/{store.database}"
    )
    suffix = str(datetime.datetime.now())
    df.to_sql(f"{tbl_name}_{suffix}", engine, schema=store.db_schema, if_exists="replace")
    return suffix


def sql_df(sql, conn):
    df = list(execute_sql(sql, conn).fetchall())
    conn.commit()
    return df


def remove_table(tbl_name, conn):
    sql = f'drop table if exists "{tbl_name}" '
    execute_sql(sql, conn)
    conn.commit()
