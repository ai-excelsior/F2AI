import psycopg2
from sqlalchemy import create_engine


def psy_conn(user, passwd, host, port, database, **kwargs):
    conn = psycopg2.connect(
        user=user,
        password=passwd,
        host=host,
        port=port,
        dbname=database,
    )
    return conn


def close_conn(conn, tables: list = None):
    for tab in tables:
        remove_table(tab, conn)
    conn.close()


def execute_sql(sql, conn):
    cursor = conn.cursor()
    cursor.execute(sql)
    return cursor


def to_pgsql(df, tbl_name, **kwagrs):
    engine = create_engine(
        f"postgresql+psycopg2://{kwagrs.get('user')}:{kwagrs.get('passwd')}@{kwagrs.get('host')}:{kwagrs.get('port')}/{kwagrs.get('database')}"
    )
    df.to_sql(f"{tbl_name}", engine, schema=kwagrs.get("schema"), if_exists="replace")


def sql_df(sql, conn):
    df = list(execute_sql(sql, conn).fetchall())
    conn.commit()
    return df


def remove_table(tbl_name, conn):
    sql = f"drop table if exists {tbl_name}"
    execute_sql(sql, conn)
