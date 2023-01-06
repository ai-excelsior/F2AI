import re
import pandas as pd
from pydantic import BaseModel
from enum import Enum
from typing import Any, List
from datetime import timedelta, datetime
from functools import reduce


class AvailablePeriods(Enum):
    """Available Period definitions which supported by F2AI."""

    YEARS = "years"
    MONTHS = "months"
    WEEKS = "weeks"
    DAYS = "days"
    HOURS = "hours"
    MINUTES = "minutes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"


PANDAS_TIME_COMPONENTS_MAP = {
    AvailablePeriods.YEARS: "year",
    AvailablePeriods.MONTHS: "month",
    AvailablePeriods.WEEKS: "day",
    AvailablePeriods.DAYS: "day",
    AvailablePeriods.HOURS: "hour",
    AvailablePeriods.MINUTES: "minute",
    AvailablePeriods.SECONDS: "second",
    AvailablePeriods.MILLISECONDS: "microsecond",
    AvailablePeriods.MICROSECONDS: "microsecond",
    AvailablePeriods.NANOSECONDS: "nanosecond",
}

PANDAS_FREQ_STR_MAP = {
    AvailablePeriods.YEARS: "YS",
    AvailablePeriods.MONTHS: "MS",
    AvailablePeriods.WEEKS: "W",
    AvailablePeriods.DAYS: "D",
    AvailablePeriods.HOURS: "H",
    AvailablePeriods.MINUTES: "T",
    AvailablePeriods.SECONDS: "S",
    AvailablePeriods.MILLISECONDS: "L",
    AvailablePeriods.MICROSECONDS: "U",
    AvailablePeriods.NANOSECONDS: "N",
}


class Period(BaseModel):
    """A wrapper of different representations of a time range. Useful to convert to underline utils like pandas DateOffset, Postgres interval strings."""

    n: int = 1
    unit: AvailablePeriods = AvailablePeriods.DAYS

    def __init__(__pydantic_self__, **data: Any) -> None:
        if not data.get("unit", "s").endswith("s"):
            data["unit"] = data.get("unit", "") + "s"
        super().__init__(**data)

    def __str__(self) -> str:
        return f"{self.n} {self.unit.value}"

    def __neg__(self) -> "Period":
        return Period(n=-self.n, unit=self.unit.value)

    @property
    def is_neg(self):
        return self.n < 0

    def to_pandas_dateoffset(self, normalize=False):
        from pandas import DateOffset

        return DateOffset(**{self.unit.value: self.n}, normalize=normalize)

    def to_pgsql_interval(self):
        return f"interval '{self.n} {self.unit.value}'"

    def to_py_timedelta(self):
        if self.unit == AvailablePeriods.YEARS:
            return timedelta(days=365 * self.n)
        if self.unit == AvailablePeriods.MONTHS:
            return timedelta(days=30 * self.n)
        return timedelta(**{self.unit.value: self.n})

    def to_pandas_freq_str(self):
        return f'{self.n}{PANDAS_FREQ_STR_MAP[self.unit]}'

    @classmethod
    def from_str(cls, s: str):
        """Construct a period from str, egg: 10 years, 1day, -1 month.

        Args:
            s (str): string representation of a period
        """
        n, unit = re.search("(-?\d+)\s?(\w+)", s).groups()
        return cls(n=int(n), unit=unit)

    def get_pandas_datetime_components(self) -> List[str]:
        index_of_period = list(PANDAS_TIME_COMPONENTS_MAP.keys()).index(self.unit)
        components = list(PANDAS_TIME_COMPONENTS_MAP.values())[: index_of_period + 1]

        return reduce(lambda xs, x: xs + [x] if x not in xs else xs, components, [])

    def normalize(self, dt: pd.Timestamp, norm_type: str):
        if self.unit in {
            AvailablePeriods.YEARS,
            AvailablePeriods.MONTHS,
            AvailablePeriods.WEEKS,
            AvailablePeriods.DAYS,
        }:
            if norm_type == "floor":
                return dt.normalize() + self.to_pandas_dateoffset()
            else:
                return dt.normalize()

        freq = self.to_pandas_freq_str()
        if norm_type == "floor":
            return dt.floor(freq)
        elif norm_type == "ceil":
            return dt.ceil(freq)
        else:
            return dt.round(freq)
