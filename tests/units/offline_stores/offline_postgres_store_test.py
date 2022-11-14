import os
import pytest
import datetime
import pandas as pd
from pypika import Query
from unittest.mock import MagicMock

from f2ai.offline_stores.offline_postgres_store import build_stats_query, OfflinePostgresStore
from f2ai.definitions import Feature, FeatureDTypes, StatsFunctions, SqlSource

feature_city = Feature(name="city", dtype=FeatureDTypes.STRING, view_name="zipcode_table")
feature_population = Feature(name="population", dtype=FeatureDTypes.INT, view_name="zipcode_table")

FILE_DIR = os.path.dirname(__file__)


def read_sql_str(*args) -> str:
    with open(os.path.join(FILE_DIR, "postgres_sqls", f"{'_'.join(args)}.sql"), "r") as f:
        return f.read()


@pytest.mark.parametrize("fn", [(fn) for fn in StatsFunctions if fn != StatsFunctions.UNIQUE])
def test_build_stats_query_with_group_by_numeric(fn: StatsFunctions):
    q = Query.from_("zipcode_table")
    sql = build_stats_query(
        q,
        features=[feature_population],
        stats_fn=fn,
        group_keys=["zipcode"],
    )
    expected_sql = read_sql_str("stats_query", fn.value)
    assert sql.get_sql() == expected_sql


@pytest.mark.parametrize("fn", [StatsFunctions.UNIQUE])
def test_build_stats_query_with_group_by_categorical(fn: StatsFunctions):
    q = Query.from_("zipcode_table")
    sql = build_stats_query(
        q,
        stats_fn=fn,
        group_keys=["zipcode"],
    )
    expected_sql = read_sql_str("stats_query", fn.value)
    assert sql.get_sql() == expected_sql


def test_stats_numeric():
    source = SqlSource(name="foo", query="zipcode_table", timestamp_field="event_timestamp")
    store = OfflinePostgresStore(
        host="localhost",
        user="foo",
        password="bar",
    )

    mock = MagicMock()
    store._get_dataframe = mock

    store.stats(
        source=source,
        features=[feature_population],
        fn=StatsFunctions.AVG,
        group_keys=["zipcode"],
        start=datetime.datetime(year=2017, month=1, day=1),
        end=datetime.datetime(year=2018, month=1, day=1),
    )
    sql, columns = mock.call_args[0]  # the first call
    assert ",".join(columns) == "zipcode,population"
    assert sql.get_sql() == read_sql_str("store_stats_query", "numeric")


def test_stats_unique():
    source = SqlSource(name="foo", query="zipcode_table", timestamp_field="event_timestamp")
    store = OfflinePostgresStore(host="localhost", user="foo", password="bar")

    mock = MagicMock(return_value=pd.DataFrame({"zipcode": ["A"]}))
    store._get_dataframe = mock

    store.stats(
        source=source,
        features=[feature_city],
        fn=StatsFunctions.UNIQUE,
        group_keys=["zipcode"],
        start=datetime.datetime(year=2017, month=1, day=1),
        end=datetime.datetime(year=2018, month=1, day=1),
    )
    sql, columns = mock.call_args[0]  # the first call

    assert ",".join(columns) == "zipcode"
    assert sql.get_sql() == read_sql_str("store_stats_query", "categorical")
