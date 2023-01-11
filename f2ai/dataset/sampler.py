import numpy as np
import pandas as pd
import warnings
import abc
from typing import Union, Any, Dict

from ..definitions import Period

TIME_COL = "event_timestamp"


class AbstractSampler:
    @property
    def iterable(self):
        # TODO
        pass

    @property
    def iterable_only(self):
        # TODO
        pass

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> pd.DataFrame:
        """
        Get all sample results, which usually is an entity_df

        Returns:
            pd.DataFrame: An entity_dataframe used to query features or labels
        """
        pass

    @abc.abstractmethod
    def __iter__(self, *args: Any, **kwds: Any) -> Dict:
        """
        Iterable way to get sample result.

        Returns:
            pd.DataFrame: _description_
        """
        pass


class EventTimestampSampler(AbstractSampler):
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


class EvenEventTimestampSampler(AbstractSampler):
    """
    A sampler which using time to generate query entity dataframe. This sampler generally useful when you don't have entity keys.
    """

    def __init__(self, start: str, end: str, period: Union[str, Period]):
        """
        evenly sample from a range of time, with given period.

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

    def __call__(self) -> pd.DatetimeIndex:
        return self._get_date_range()

    def __iter__(self) -> Dict:
        datetime_indexes = self._get_date_range()
        for i in datetime_indexes:
            yield i

    def _get_date_range(self) -> pd.DatetimeIndex:
        return pd.date_range(self._start, self._end, freq=self._period.to_pandas_freq_str())


class RandomEventTimestampSampler(AbstractSampler):
    # TODO
    pass


# TODO: remove this later
class GroupFixedNumberSampler(AbstractSampler):
    def __init__(
        self,
        time_bucket: str,
        stride: int,
        start: str = None,
        end: str = None,
        group_ids: Union[tuple, list] = None,
        group_names: list = None,
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._group_names = group_names

    def random_bucket(self):
        bucket_num = self.time_bucket_num()
        bucket_mask = np.ones(bucket_num)
        return list(bucket_mask)

    def bucket_random_sample(self, all_date: pd.DataFrame):
        all_date.reset_index(drop=True, inplace=True)
        if len(all_date) > 0:
            return all_date.loc[range(0, len(all_date), self._stride), :]
        else:
            warnings.warn(
                "please reset input parameters to ensure you have at least one bucket to be sampled!",
                UserWarning,
            )
            return pd.DataFrame()

    def sample(self, bucket_mask):
        bucket_num = self.time_bucket_num()
        bucket_size, time_bucket_unit = self._time_bucket.split(" ")
        bucket_size = int(bucket_size)
        freq_dict = {
            "months": "MS",
            "weeks": "W",
            "days": "D",
            "hours": "H",
            "minutes": "min",
            "seconds": "S",
            "milliseconds": "ms",
        }

        all_date = pd.DataFrame(
            pd.date_range(start=self._start, end=self._end, freq=freq_dict[time_bucket_unit]),
            columns=[TIME_COL],
        )

        all_date["bucket_nbr"] = [x for n in range(bucket_num) for x in [n] * bucket_size][: len(all_date)]

        if self._group_ids is not None:
            group_keys = pd.DataFrame(self._group_ids, columns=self._group_names)
            result = group_keys.groupby(self._group_names).apply(
                lambda x: x.merge(
                    all_date[all_date["bucket_nbr"].isin([g for g, i in enumerate(bucket_mask) if i])]
                    .groupby(["bucket_nbr"])
                    .apply(lambda x: self.bucket_random_sample(x))
                    .droplevel(level="bucket_nbr"),
                    how="cross",
                )
            )
            if len(result) > 0:
                result = result.drop(columns="bucket_nbr").droplevel(level=self._group_names)
        else:
            all_date = all_date[all_date["bucket_nbr"].isin([g for g, i in enumerate(bucket_mask) if i])]
            result = all_date.groupby(["bucket_nbr"]).apply(lambda x: self.bucket_random_sample(x))
            if len(result) > 0:
                result = result.drop(columns="bucket_nbr").droplevel(level="bucket_nbr")
        # result.reset_index(inplace=True, drop=True)

        return result.drop_duplicates().reset_index(drop=True)

    def __call__(self):
        bucket_mask = self.random_bucket()
        return self.sample(bucket_mask)


class GroupSampler(AbstractSampler):
    """
    Sample every group with given event_timestamp_sampler.
    """

    def __init__(
        self, event_timestamp_sampler: EventTimestampSampler, group_df: pd.DataFrame, random_state: int = None
    ) -> None:
        super().__init__()

        self._event_timestamp_sampler = event_timestamp_sampler
        self._rng = np.random.default_rng(random_state)

        self._group_df = group_df

    def __call__(self) -> pd.DataFrame:
        return pd.merge(
            pd.DataFrame({"event_timestamp": self._event_timestamp_sampler()}), self._group_df, how="cross"
        )[list(self._group_df.columns) + ["event_timestamp"]]

    def __iter__(self) -> Dict:
        for event_timestamp in iter(self._event_timestamp_sampler):
            for _, entity_row in self._group_df.iterrows():
                d = entity_row.to_dict()
                d["event_timestamp"] = event_timestamp
                yield d


class GroupNInstanceSampler(AbstractSampler):
    """
    Sample N instance from each group.
    """

    def __init__(
        self, event_timestamp_sampler: EventTimestampSampler, group_df: pd.DataFrame, n=1, random_state=None
    ) -> None:
        super().__init__()

        self._event_timestamp_sampler = event_timestamp_sampler
        self._rng = np.random.default_rng(seed=random_state)

        self._event_timestamps = self._event_timestamp_sampler()
        self._group_df = group_df
        self._n = n

    def _sample_n(self, row: pd.DataFrame):
        event_timestamps = self._rng.choice(self._event_timestamps, size=self._n, replace=False)
        return pd.merge(row, pd.DataFrame({"event_timestamp": event_timestamps}), how="cross")

    def __call__(self) -> pd.DataFrame:
        return (
            self._group_df.groupby(list(self._group_df.columns), group_keys=False, sort=False)
            .apply(self._sample_n)
            .reset_index(drop=True)
        )

    def __iter__(self) -> Dict:
        for i, row in self._group_df.iterrows():
            sampled_df = self._sample_n(pd.DataFrame([row]))
            for j, row in sampled_df.iterrows():
                yield row.to_dict()


class GroupProbSampler(AbstractSampler):
    # TODO
    pass
