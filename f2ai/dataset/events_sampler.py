import numpy as np
import pandas as pd
import abc
from typing import Union, Any, Dict

from ..definitions import Period


class EventsSampler:
    """
    This sampler is used to sample event_timestamp column. Customization could be done by inheriting this class.
    """

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> pd.DatetimeIndex:
        """
        Get all sample results

        Returns:
            pd.DatetimeIndex: An datetime index
        """
        pass

    @abc.abstractmethod
    def __iter__(self, *args: Any, **kwds: Any) -> pd.Timestamp:
        """
        Iterable way to get sample result.

        Returns:
            pd.Timestamp
        """
        pass


class EvenEventsSampler(EventsSampler):
    """
    A sampler which using time to generate query entity dataframe. This sampler generally useful when you don't have entity keys.
    """

    def __init__(self, start: str, end: str, period: Union[str, Period], **kwargs):
        """
        evenly sample from a range of time, with given period.

        Args:
            start (str): start datetime
            end (str): end datetime
            period (str): a period string, egg: '1 day'.
            **kwargs (Any): additional arguments passed to pd.date_range.
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        self._start = start
        self._end = end
        self._period = Period.from_str(period)
        self._kwargs = kwargs

    def __call__(self) -> pd.DatetimeIndex:
        return self._get_date_range()

    def __iter__(self) -> Dict:
        datetime_indexes = self._get_date_range()
        for i in datetime_indexes:
            yield i

    def _get_date_range(self) -> pd.DatetimeIndex:
        return pd.date_range(self._start, self._end, freq=self._period.to_pandas_freq_str(), **self._kwargs)


class RandomNEventsSampler(EventsSampler):
    """
    Randomly sample a fixed number of event timestamp in a given time range.
    """

    def __init__(
        self,
        start: str,
        end: str,
        period: Union[str, Period],
        n: int = 1,
        random_state: int = None,
    ):
        """
        randomly sample fixed number of event timestamp.

        Args:
            start (str): start datetime
            end (str): end datetime
            period (str): a period string, egg: '1 day'.
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        self._start = start
        self._end = end
        self._period = Period.from_str(period)

        self._n = n
        self._rng = np.random.default_rng(random_state)

    def __call__(self) -> pd.DatetimeIndex:
        return self._get_date_range()

    def __iter__(self) -> Dict:
        datetime_indexes = self._get_date_range()
        for i in datetime_indexes:
            yield i

    def _get_date_range(self) -> pd.DatetimeIndex:
        datetimes = pd.date_range(self._start, self._end, freq=self._period.to_pandas_freq_str())
        indices = sorted(self._rng.choice(range(len(datetimes)), size=self._n, replace=False))
        return pd.DatetimeIndex([datetimes[i] for i in indices])
