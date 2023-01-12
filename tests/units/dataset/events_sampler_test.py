import pandas as pd
from f2ai.dataset import (
    EvenEventsSampler,
    RandomNEventsSampler,
)


def test_even_events_sampler():
    sampler = EvenEventsSampler(start="2022-10-02", end="2022-12-02", period="1 day")

    assert len(sampler()) == 62
    assert next(iter(sampler)) == pd.Timestamp("2022-10-02 00:00:00")


def test_random_n_events_sampler():
    sampler = RandomNEventsSampler(start="2022-10-02", end="2022-12-02", period="1 day", n=2, random_state=666)

    assert len(sampler()) == 2
    assert next(iter(sampler)) == pd.Timestamp("2022-10-05 00:00:00")
