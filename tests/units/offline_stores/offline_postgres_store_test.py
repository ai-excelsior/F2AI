import os
import pytest
from pypika import Query

from f2ai.offline_stores.offline_postgres_store import build_stats_query
from f2ai.definitions import Feature, FeatureDTypes, StatsFunctions

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
    ).get_sql()
    expected_sql = read_sql_str("stats_query", fn.value)
    assert sql == expected_sql


@pytest.mark.parametrize("fn", [(fn) for fn in StatsFunctions if fn == StatsFunctions.UNIQUE])
def test_build_stats_query_with_group_by_categorical(fn: StatsFunctions):
    q = Query.from_("zipcode_table")
    sql = build_stats_query(
        q,
        features=[feature_population, feature_city],
        stats_fn=fn,
        group_keys=["zipcode"],
    ).get_sql()
    expected_sql = read_sql_str("stats_query", fn.value)
    assert sql == expected_sql
