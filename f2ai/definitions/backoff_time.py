import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Iterator, Union

from f2ai.definitions.period import Period


@dataclass
class BackOffTime:
    """
    Useful to define how to split data by time.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    step: Period

    def __init__(
        self,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        step: Union[str, Period],
    ) -> None:
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        if start.tz != "UTC":
            start = pd.to_datetime(start, utc=True)
        if end.tz != "UTC":
            end = pd.to_datetime(end, utc=True)

        if isinstance(step, str):
            step = Period.from_str(step)

        self.start = start
        self.end = end
        self.step = step

    def to_units(self) -> Iterator[Period]:
        pd_offset = self.step.to_pandas_dateoffset()
        start = self.step.normalize(self.start, 'ceil')
        end = self.step.normalize(self.end, 'floor')

        bins = pd.date_range(
            start=start,
            end=end,
            freq=pd_offset,
        )
        for (start, end) in zip(bins[:-1], bins[1:]):
            yield BackOffTime(
                start=start,
                end=end,
                step=self.step,
            )


def cfg_to_date(fromnow, start, end, step):

    if fromnow:
        end = pd.to_datetime(datetime.now(), utc=True)
        start = end - Period.from_str(fromnow).to_py_timedelta()
    else:
        start = pd.to_datetime(start, utc=True) if start else pd.to_datetime(0, utc=True)
        end = pd.to_datetime(end, utc=True) if end else pd.to_datetime(datetime.now(), utc=True)

    return BackOffTime(start=start, end=end, step=Period.from_str(step))


def backoff_to_split(backoff: BackOffTime):
    bins = pd.date_range(
        start=backoff.start,
        end=backoff.end + backoff.step.to_py_timedelta(),
        freq=backoff.step.to_py_timedelta(),
        inclusive="both",
    )
    res = [x for x in zip(bins[:-1], bins[1:])]
    return res
