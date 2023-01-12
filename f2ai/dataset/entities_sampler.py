import numpy as np
import pandas as pd
import abc
from typing import Any, Dict

from .events_sampler import EventsSampler

TIME_COL = "event_timestamp"


class EntitiesSampler:
    @property
    def iterable(self):
        is_abstract = getattr(self.__iter__, "__isabstractmethod__", False)
        return not is_abstract

    @property
    def iterable_only(self):
        is_call_abstract = getattr(self.__call__, "__isabstractmethod__", False)
        if self.iterable:
            return is_call_abstract
        return False

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


# TODO: test this
class NoEntitiesSampler(EntitiesSampler):
    """
    This class will directly convert an EventsSampler to EntitiesSampler. Useful when no entity keys are exists.
    """

    def __init__(self, events_sampler: EventsSampler) -> None:
        super().__init__()

        self._events_sampler = events_sampler

    def __call__(self) -> pd.DataFrame:
        return pd.DataFrame({"event_timestamp": self._events_sampler()})

    def __iter__(self) -> Dict:
        for event_timestamp in self._events_sampler:
            yield {"event_timestamp": event_timestamp}


class EvenEntitiesSampler(EntitiesSampler):
    """
    Sample every group with given events_sampler.
    """

    def __init__(
        self, events_sampler: EventsSampler, group_df: pd.DataFrame, random_state: int = None
    ) -> None:
        super().__init__()

        self._events_sampler = events_sampler
        self._rng = np.random.default_rng(random_state)

        self._group_df = group_df

    def __call__(self) -> pd.DataFrame:
        return pd.merge(
            pd.DataFrame({"event_timestamp": self._events_sampler()}), self._group_df, how="cross"
        )[list(self._group_df.columns) + ["event_timestamp"]]

    def __iter__(self) -> Dict:
        for event_timestamp in iter(self._events_sampler):
            for _, entity_row in self._group_df.iterrows():
                d = entity_row.to_dict()
                d["event_timestamp"] = event_timestamp
                yield d


class FixedNEntitiesSampler(EntitiesSampler):
    """
    Sample N instance from each group with given probability.
    """

    def __init__(self, events_sampler: EventsSampler, group_df: pd.DataFrame, n=1, random_state=None) -> None:
        """
        Args:
            events_sampler (EventTimestampSampler): an EventTimestampSampler instance.
            group_df (pd.DataFrame): a group_df is a dataframe with contains some entity columns. Optionally, it may contains a columns named `p`, which indicates the probability of this this group.
            n (int, optional): how many instance per group. Defaults to 1.
            random_state (_type_, optional): Defaults to None.
        """

        super().__init__()

        self._events_sampler = events_sampler
        self._rng = np.random.default_rng(seed=random_state)

        self._event_timestamps = self._events_sampler()

        if "p" in group_df.columns:
            assert group_df["p"].sum() == 1, "sum all weights should be 1"

            take_n = (group_df["p"] * len(group_df) * n).round().astype(int).rename("_f2ai_take_n_")
            self._group_df = pd.concat([group_df, take_n], axis=1)
        else:
            self._group_df = group_df.assign(_f2ai_take_n_=n)

    def _sample_n(self, row: pd.DataFrame):
        event_timestamps = self._rng.choice(self._event_timestamps, size=row["_f2ai_take_n_"].iloc[0])
        return pd.merge(
            row.drop(columns=["p", "_f2ai_take_n_"], errors="ignore"),
            pd.DataFrame({"event_timestamp": event_timestamps}),
            how="cross",
        )

    def __call__(self) -> pd.DataFrame:
        group_keys = [column for column in self._group_df.columns if column not in {"p", "_f2ai_take_n_"}]
        return (
            self._group_df.groupby(group_keys, group_keys=False, sort=False)
            .apply(self._sample_n)
            .reset_index(drop=True)
        )

    def __iter__(self) -> Dict:
        for i, row in self._group_df.iterrows():
            sampled_df = self._sample_n(pd.DataFrame([row]))
            for j, row in sampled_df.iterrows():
                yield row.to_dict()
