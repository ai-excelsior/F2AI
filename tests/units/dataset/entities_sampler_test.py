import pandas as pd
from f2ai.dataset import (
    NoEntitiesSampler,
    EvenEventsSampler,
    FixedNEntitiesSampler,
    EvenEntitiesSampler,
)


def test_even_entities_sampler():
    event_time_sampler = EvenEventsSampler(start="2022-10-02", end="2022-10-04", period="1 day")
    group_df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "banana"],
            "sauce": ["tomato", "chili", "tomato"],
        }
    )
    sampler = EvenEntitiesSampler(event_time_sampler, group_df)

    sampled_df = sampler()
    assert len(sampled_df) == 9
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_fixed_n_entities_sampler():
    event_time_sampler = EvenEventsSampler(start="2022-10-02", end="2022-10-04", period="1 day")
    group_df = pd.DataFrame(
        {
            "fruit": ["apple", "banana", "banana"],
            "sauce": ["tomato", "chili", "tomato"],
        }
    )
    sampler = FixedNEntitiesSampler(event_time_sampler, group_df, n=2)

    sampled_df = sampler()
    assert len(sampled_df) == 6
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_fixed_n_prob_entities_sampler():
    event_time_sampler = EvenEventsSampler(start="2022-10-02", end="2022-10-06", period="1 day")
    group_df = pd.DataFrame(
        {"fruit": ["apple", "banana", "banana"], "sauce": ["tomato", "chili", "tomato"], "p": [0.2, 0.6, 0.2]}
    )
    sampler = FixedNEntitiesSampler(event_time_sampler, group_df, n=3)

    sampled_df = sampler()
    assert len(sampled_df) == 9
    assert list(sampled_df.columns) == ["fruit", "sauce", "event_timestamp"]

    next_sampled_item = next(iter(sampler))
    assert all([key in {"fruit", "sauce", "event_timestamp"} for key in next_sampled_item.keys()])


def test_sampler_properties():
    event_time_sampler = EvenEventsSampler(start="2022-10-02", end="2022-10-06", period="1 day")
    sampler = NoEntitiesSampler(event_time_sampler)
    assert sampler.iterable
    assert not sampler.iterable_only
