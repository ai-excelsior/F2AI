import pandas as pd
from f2ai.dataset import EvenTimeSampler


def test_even_time_sampler():
    sampler = EvenTimeSampler(start="2022-10-02", end="2022-12-02", period="1 day")

    assert len(sampler()) == 62
    assert next(iter(sampler)) == dict(event_timestamp=pd.Timestamp("2022-10-02 00:00:00", freq="D"))
