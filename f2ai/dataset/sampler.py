import numpy as np
import pandas as pd
import warnings
import abc
from typing import Union, List, Any, Dict

from ..definitions import Period

TIME_COL = "event_timestamp"


class AbstractSampler:
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


class EvenTimeSampler(AbstractSampler):
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

    def __call__(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"event_timestamp": pd.date_range(self._start, self._end, freq=self._period.to_pandas_freq_str())}
        )

    def __iter__(self) -> Dict:
        datetime_indexes = pd.date_range(self._start, self._end, freq=self._period.to_pandas_freq_str())
        for i in datetime_indexes:
            yield dict(event_timestamp=i)


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


class GroupRandomSampler(GroupFixedNumberSampler):
    def __init__(
        self,
        time_bucket: str,
        stride: int,
        ratio: float,
        start: str = None,
        end: str = None,
        group_ids: List[str] = None,
        group_names: list = None,
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._ratio = ratio
        self._group_names = group_names

    def random_bucket(self):
        bucket_num = self.time_bucket_num()
        bucket_mask = np.zeros(bucket_num)
        bucket_mask[np.where(np.random.random_sample(bucket_num) < self._ratio)[0]] = 1
        return list(bucket_mask)

    def __call__(self):
        bucket_mask = self.random_bucket()
        return self.sample(bucket_mask)


class UniformNPerGroupSampler(GroupFixedNumberSampler):
    def __init__(
        self,
        time_bucket: str,
        stride: int,
        n_groups: int,
        avg_nbr: int,
        start: str = None,
        end: str = None,
        group_ids: List[str] = None,
        group_names: list = None,
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._n_groups = n_groups
        self._avg_nbr = avg_nbr
        self._group_names = group_names

    def random_bucket(self):
        bucket_num = self.time_bucket_num()
        bucket_mask = np.zeros(bucket_num)
        avg_length = bucket_num // self._n_groups
        assert avg_length > 0, "time_bucket should be smaller to ensure every group have at least one bucket."
        p = self._avg_nbr / avg_length
        assert p < 1, "p is too large!"
        bucket_mask[np.where(np.random.random_sample(self._n_groups) < p)[0]] = 1
        return list(bucket_mask)

    def __call__(self):
        bucket_mask = self.random_bucket()
        return self.sample(bucket_mask)
