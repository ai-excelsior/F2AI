# from operator import le
import numpy as np
import math
import pandas as pd
import warnings
from typing import Union, List

TIME_COL = "event_timestamp"


class AbstractSampler:
    def __init__(self, time_bucket: str, stride: int, start: str = None, end: str = None):
        """
        args
        time_bucket:size of time_bucket like "2 days" or "10 minutes".
        stride: stride like 4 or 5, int type.
        start: start of dataset for sample
        end:end of dataset for sample

        """
        self._time_bucket = time_bucket
        self._start = pd.to_datetime(start, utc=True) if start else pd.to_datetime(0, utc=True)
        self._end = pd.to_datetime(end, utc=True) if start else pd.to_datetime(datetime.now(), utc=True)
        self._stride = stride

        assert self._end > self._start, "end should be greater than start!"
        assert self._stride < int(
            self._time_bucket.split(" ", 1)[0]
        ), "time_bucket should be grater than stride!"

    def time_bucket_num(self):

        delta = self._end - self._start
        delta_days = delta.components.days
        delta_months = delta_days // 30
        delta_weeks = delta_days // 7
        delta_hours = delta.components.hours + delta_days * 24
        delta_minutes = delta.components.minutes + delta_hours * 60
        delta_seconds = delta.components.seconds + delta_minutes * 60
        delta_milliseconds = delta.components.milliseconds + delta_seconds * 1000

        time_freq = self._time_bucket.split(" ", 1)[1]

        bucket_num = math.ceil(locals()[f"delta_{time_freq}"] / int(self._time_bucket.split(" ", 1)[0]))
        return bucket_num

    def __call__(self):
        raise ValueError("error!")


class GroupFixednbrSampler(AbstractSampler):
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
        if len(all_date) > 0:
            return all_date.loc[range(all_date.index[0], all_date.index[-1], self._stride), :]
        else:
            warnings.warn(
                "please reset input parameters to ensure you have at least one bucket to be sampled!",
                UserWarning,
            )
            return pd.DataFrame()

    def sample(self, bucket_mask):
        bucket_num = self.time_bucket_num()
        bucket_size = int(self._time_bucket.split(" ", 1)[0])
        time_bucket_unit = self._time_bucket.split(" ", 1)[1].rstrip("s")
        freq_dict = {
            "month": "MS",
            "week": "W",
            "day": "D",
            "hour": "H",
            "minute": "min",
            "second": "S",
            "millisecond": "ms",
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

        return result.drop_duplicates()

    def __call__(self):
        bucket_mask = self.random_bucket()
        return self.sample(bucket_mask)


class GroupRandomSampler(GroupFixednbrSampler):
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


class UniformNPerGroupSampler(GroupFixednbrSampler):
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
