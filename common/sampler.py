import numpy as np
import math
import pandas as pd


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

        assert self._end > self._start, "end should be greater than start!"
        assert self._stride < int(
            self._time_bucket.split(" ", 1)[0]
        ), "time_bucket should be grater than stride!"

    def time_bucket_num(self, start: str, end: str):
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

            bucket_num = math.ceil(locals()[f"delta_{time_freq}"] / int(self._time_bucket.split(" ", 1)[0]))
        else:
            raise ValueError("data start and end can not be None!")

        return bucket_num

    def __call__(self):
        raise ValueError("error!")


class GroupFixednbrSampler(AbstractSampler):
    def __init__(
        self, time_bucket: str, stride: int, start: str = None, end: str = None, group_ids: list[str] = []
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end)
        bucket_mask = np.ones(bucket_num)
        return list(bucket_mask)

    def bucket_random_sample(self, all_date: pd.DataFrame):

        basis_index = list(range(0, len(all_date) + 1, self._stride))
        random_index = np.random.randint(self._stride, size=len(basis_index))
        sample_index = basis_index + random_index
        sample_index[-1] = min(sample_index[-1], len(all_date) - 1)  # TODO 这里要判断一下？

        return all_date.iloc[sample_index, :]

    def sample(self, bucket_mask):
        selected_bucket = list(np.where(np.array(bucket_mask) == 1)[0])

        bucket_num = len(bucket_mask)
        bucket_size = int(self._time_bucket.split(" ", 1)[0])
        time_bucket_unit = self._time_bucket.split(" ", 1)[1]
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
            pd.date_range(start=self._start, end=self._end, freq=freq_dict[time_bucket_unit]).values,
            columns=["timeIndex"],
        )

        all_date["bucket_nbr"] = pd.merge(
            pd.DataFrame(range(bucket_num)), pd.DataFrame(range(bucket_size)), how="cross"
        ).loc[0 : len(all_date), "0_x"]

        all_date = all_date[all_date["bucket_nbr"].isin(selected_bucket)]

        if self._group_ids is not None:
            group_keys = pd.DataFrame(self._group_ids, columns=["group_ids"])
            all_date = pd.merge(group_keys, all_date, how="cross")
            result = all_date.groupby(["group_ids", "bucket_nbr"]).apply(
                lambda x: self.bucket_random_sample(x)
            )
            result = result[["group_ids", "timeIndex"]].droplevel(level=["group_ids", "bucket_nbr"])
            result.sort_values(by=["group_ids", "timeIndex"], inplace=True)

        else:
            result = all_date.groupby(["bucket_nbr"]).apply(lambda x: self.bucket_random_sample(x))
            result = result["timeIndex"].droplevel(level="bucket_nbr")
            result.sort_values(inplace=True)
        result.reset_index(inplace=True, drop=True)

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
        group_ids: list[str] = None,
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._ratio = ratio

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end)
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
        group_ids: list[str] = None,
    ):
        super().__init__(time_bucket, stride, start, end)
        self._group_ids = group_ids
        self._n_groups = n_groups
        self._avg_nbr = avg_nbr

    def random_bucket(self):
        bucket_num = self.time_bucket_num(start=self._start, end=self._end)
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


if __name__ == "__main__":
    time_bucket = "4 days"
    stride = 3
    start = "2010-01-01 00:00:00"
    end = "2010-01-30 00:00:00"
    group_ids = ["A", "B"]
    ratio = 0.7
    n_groups = 2
    avg_nbr = 2

    sample1 = GroupFixednbrSampler(time_bucket, stride, start, end, group_ids=group_ids)()
    sample2 = GroupRandomSampler(time_bucket, stride, ratio, start, end, group_ids=group_ids)()
    sample3 = UniformNPerGroupSampler(time_bucket, stride, n_groups, avg_nbr, start, end, group_ids)()
    print(sample1)
    print(sample2)
    print(sample3)
