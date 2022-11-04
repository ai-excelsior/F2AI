from pydantic import BaseModel
from enum import Enum
from typing import Any
import re
from datetime import timedelta


class AvaliablePeriods(Enum):
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


class Period(BaseModel):
    n: int = 1
    unit: AvaliablePeriods = AvaliablePeriods.DAYS

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

    def to_pandas_dateoffset(self):
        from pandas import DateOffset

        return DateOffset(**{self.unit.value: self.n})

    def to_pgsql_interval(self):
        return f"interval '{self.n} {self.unit.value}'"

    def to_py_timedelta(self):
        if self.unit == AvaliablePeriods.YEARS:
            return timedelta(days=365 * self.n)
        if self.unit == AvaliablePeriods.MONTHS:
            return timedelta(days=30 * self.n)
        return timedelta(**{self.unit.value: self.n})

    @classmethod
    def from_str(cls, s: str):
        """Construct a period from str, egg: 10 years, 1day

        Args:
            s (str): string representation of a period
        """
        n, unit = re.search("(-?\d+)\s?(\w+)", s).groups()
        return cls(n=int(n), unit=unit)
