from f2ai.definitions import Period
from pandas import DateOffset


def test_period_to_pandas_dateoffset():
    offset = Period(n=1, unit="day").to_pandas_dateoffset()
    assert offset == DateOffset(days=1)


def test_period_to_sql_interval():
    interval = Period(n=1, unit="day").to_pgsql_interval()
    assert interval == "interval '1 days'"


def test_period_from_str():
    ten_years = Period.from_str("10 years").to_pandas_dateoffset()
    one_day = Period.from_str("1day").to_pandas_dateoffset()

    assert ten_years == DateOffset(years=10)
    assert one_day == DateOffset(days=1)


def test_period_negative():
    ten_years = Period.from_str("10 years")
    neg_ten_years = -ten_years
    assert neg_ten_years.n == -10

    neg_ten_years = Period.from_str("-10 years")
    assert neg_ten_years.n == -10
