import numpy as np
import math
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import List


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
        self._start = start
        self._end = end
        self._stride = stride

        assert self._end > self._start, "end should be grater than start!"
        assert self._stride < int(
            self._time_bucket.split(" ", 1)[0]
        ), "stride should be grater than time_bucket!"

    def time_bucket_num(self, start: str, end: str, is_stride: bool = False):
        if start and end:  # TODO from和to如果是None要如何处理
            delta = pd.to_datetime(end, utc=True) - pd.to_datetime(start, utc=True)
            delta_days = delta.components.days
            delta_month = delta_days // 30
            delta_week = delta_days // 7
            delta_hours = delta.components.hours + delta_days * 24
            delta_minutes = delta.components.minutes + delta_hours * 60
            delta_seconds = delta.components.seconds + delta_minutes * 60
            delta_milliseconds = delta.components.milliseconds + delta_seconds * 1000

            time_freq = self._time_bucket.split(" ", 1)[1]

            if is_stride:
                bucket_num = math.ceil(locals()[f"delta_{time_freq}"] / self._stride)
            else:
                bucket_num = math.ceil(
                    locals()[f"delta_{time_freq}"] / int(self._time_bucket.split(" ", 1)[0])
                )
        else:
            raise ValueError("data start and end can not be None!")

        return bucket_num

    def __call__(self):
        raise ValueError("error!")


class GroupFixednbrSampler(AbstractSampler):
    def __init__(
        self, time_bucket: str, stride: int, start: str = None, end: str = None, group_ids: list[str] = []
    ):
        super.__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end, is_stride=False)
        bucket_mask = np.ones(bucket_num)
        return list(bucket_mask)

    def bucket_random_sample(self):
        bucket_mask = self.random_bucket()

        bucket_num = len(bucket_mask)
        bucket_size = int(self._time_bucket.split(" ", 1)[0])
        bucket_freq = locals()[self._time_bucket.split(" ", 1)[1]]  # TODO 参数传法不对

        for i in range(bucket_num):
            if bucket_mask[i] == 1:
                sample_list = []
                time_bucket_start = pd.to_datetime(self._start, utc=True) + relativedelta(
                    bucket_freq=i * bucket_size
                )
                time_bucket_end = time_bucket_start + relativedelta(bucket_freq=(i + 1) * bucket_size)
                bucket_stride_num = self.time_bucket_num(
                    start=time_bucket_start, end=time_bucket_end, is_stride=True
                )
                stride_random = np.random.randint(self._stride, size=bucket_stride_num)

                sample_list = [
                    time_bucket_start
                    + relativedelta(bucket_freq=j * self._stride)
                    + relativedelta(bucket_freq=stride_random[j])
                    for j in range(bucket_stride_num)
                ]

        return sample_list

    def __call__(self):
        if self._group_ids is not None:
            sample = []
            for group_key in self._group_ids:
                group_sample = self.bucket_random_sample()
                group_sample = pd.DataFrame(group_sample)
                group_sample["group_key"] = group_key
                sample.append(group_sample, column="event_timestamp")
            result = pd.concat(sample)
            result.sort_values(by=["group_key", "event_timestamp"], inplace=True)
        else:
            random_sample = self.bucket_random_sample()
            result = pd.DataFrame(random_sample, columns=["event_timestamp"])
            result.sort_values(by=["event_timestamp"], inplace=True)

        return result.drop_duplicates()


class GroupRandomSampler(GroupFixednbrSampler):
    def __init__(
        self,
        time_bucket: str,
        stride: int,
        ratio: float,
        start: str = None,
        end: str = None,
        group_ids: list[str] = None,
    ):
        super.__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._ratio = ratio

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end, is_stride=False)
        bucket_mask = np.zeros(bucket_num)
        bucket_mask[np.where(np.random.random_sample(bucket_num) < self._ratio)[0]] = 1
        return list(bucket_mask)


class UniformNPerGroupSampler(GroupFixednbrSampler):
    def __init__(
        self,
        time_bucket: str,
        stride: int,
        n_groups: int,
        avg_nbr: int,
        start: str = None,
        end: str = None,
        group_ids: list[str] = None,
    ):
        super.__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._n_groups = n_groups
        self._avg_nbr = avg_nbr

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end, is_stride=False)
        bucket_mask = np.zeros(bucket_num)
        avg_length = bucket_num // self._n_groups
        assert avg_length > 0, "time_bucket should be smaller to ensure every group have at least one bucket."
        p = self._avg_nbr / avg_length
        assert p < 1, "p is too large!"
        bucket_mask[np.where(np.random.random_sample(self._n_groups) < p)[0]] = 1
        return list(bucket_mask)


if __name__ == "__main__":
    # dataset = pd.read_csv("./common/sample.csv")
    # dataset = dataset["event_timestamp"]
    time_bucket = " 2 days"
    stride = 1
    start = "2010-01-01 00:00:00"
    end = "2010-01-01 00:00:00"
    pd.to_datetime(end) - pd.to_datetime(start)
    # sample1 = GroupFixednbrSampler(time_bucket, stride, start, end, group_ids=None)
