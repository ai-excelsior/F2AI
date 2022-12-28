import pandas as pd
from datetime import datetime
from dataclasses import dataclass

from f2ai.definitions.period import Period


@dataclass
class BackoffTime:
    start: pd.Timestamp
    end: pd.Timestamp
    step: Period


def cfg_to_date(fromnow, start, end, step):

    if fromnow:
        end = pd.to_datetime(datetime.now(), utc=True)
        start = end - Period.from_str(fromnow).to_py_timedelta()
    else:
        start = pd.to_datetime(start, utc=True) if start else pd.to_datetime(0, utc=True)
        end = pd.to_datetime(end, utc=True) if end else pd.to_datetime(datetime.now(), utc=True)

    return BackoffTime(start=start, end=end, step=Period.from_str(step))


def backoff_to_split(backoff: BackoffTime):
    bins = pd.date_range(
        start=backoff.start,
        end=backoff.end + backoff.step.to_py_timedelta(),
        freq=backoff.step.to_py_timedelta(),
        inclusive="both",
    )
    res = [x for x in zip(bins[:-1], bins[1:])]
    return res
