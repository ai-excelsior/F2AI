import pandas as pd
from f2ai.dataset import (
    EvenEventTimestampSampler,
    RandomNTimestampSampler,
    GroupNInstanceSampler,
    GroupSampler,
)


def test_even_event_timestamp_sampler():
    sampler = EvenEventTimestampSampler(start="2022-10-02", end="2022-12-02", period="1 day")

    assert len(sampler()) == 62
    assert next(iter(sampler)) == pd.Timestamp("2022-10-02 00:00:00")


def test_group_sampler():
    event_time_sampler = EvenEventTimestampSampler(start="2022-10-02", end="2022-10-04", period="1 day")
    group_df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "banana"],
            "sauce": ["tomato", "chili", "tomato"],
        }
    )
    sampler = GroupSampler(event_time_sampler, group_df)

    sampled_df = sampler()
    assert len(sampled_df) == 9
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_group_n_instance_sampler():
    event_time_sampler = EvenEventTimestampSampler(start="2022-10-02", end="2022-10-04", period="1 day")
    group_df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "banana"],
            "sauce": ["tomato", "chili", "tomato"],
        }
    )
    sampler = GroupNInstanceSampler(event_time_sampler, group_df, n=2)

    sampled_df = sampler()
    assert len(sampled_df) == 6
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_group_probability_sampler():
    event_time_sampler = EvenEventTimestampSampler(start="2022-10-02", end="2022-10-06", period="1 day")
    group_df = pd.DataFrame(
        {"fruit": ["apple", "banana", "banana"], "sauce": ["tomato", "chili", "tomato"], "p": [0.2, 0.6, 0.2]}
    )
    sampler = GroupNInstanceSampler(event_time_sampler, group_df, n=3)

    sampled_df = sampler()
    assert len(sampled_df) == 9
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_random_n_event_timestamp_sampler():
    sampler = RandomNTimestampSampler(start="2022-10-02", end="2022-12-02", period="1 day", n=2, random_state=666)

    assert len(sampler()) == 2
    assert next(iter(sampler)) == pd.Timestamp("2022-10-05 00:00:00")
