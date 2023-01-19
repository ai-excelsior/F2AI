import pandas as pd
from f2ai.offline_stores.offline_file_store import OfflineFileStore
from f2ai.definitions import Period, FileSource, Feature, FeatureDTypes, StatsFunctions

import pytest
from unittest.mock import MagicMock
from f2ai.common.time_field import TimeField

mock_point_in_time_filter_df = pd.DataFrame(
    {
        "join_key": ["A", "A", "A", "A"],
        TimeField.ENTITY_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
        TimeField.SOURCE_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:21"),
        ],
    },
)


def test_point_in_time_filter_simple():
    result_df = OfflineFileStore._point_in_time_filter(mock_point_in_time_filter_df)
    assert len(result_df) == 3
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")


def test_point_in_time_filter_not_include():
    result_df = OfflineFileStore._point_in_time_filter(mock_point_in_time_filter_df, include=False)
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:19")


def test_point_in_time_filter_with_ttl():
    result_df = OfflineFileStore._point_in_time_filter(
        mock_point_in_time_filter_df, ttl=Period.from_str("2 seconds")
    )
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:19")


def test_point_on_time_filter_simple():
    result_df = OfflineFileStore._point_on_time_filter(
        mock_point_in_time_filter_df, -Period.from_str("2 seconds"), include=True
    )
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:19")


def test_point_on_time_filter_simple_label():
    result_df = OfflineFileStore._point_on_time_filter(
        mock_point_in_time_filter_df, Period.from_str("2 seconds"), include=True
    )
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:21")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:20")


def test_point_on_time_filter_not_include():
    result_df = OfflineFileStore._point_on_time_filter(
        mock_point_in_time_filter_df, period=-Period.from_str("2 seconds"), include=False
    )
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:19")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:18")


def test_point_on_time_filter_not_include_label():
    result_df = OfflineFileStore._point_on_time_filter(
        mock_point_in_time_filter_df, Period.from_str("2 seconds"), include=False
    )
    assert len(result_df) == 1
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:21")


def test_point_on_time_filter_with_ttl():
    result_df = OfflineFileStore._point_on_time_filter(
        mock_point_in_time_filter_df,
        period=-Period.from_str("3 seconds"),
        ttl=Period.from_str("2 seconds"),
        include=True,
    )
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:19")


mock_point_in_time_latest_df = pd.DataFrame(
    {
        "join_key": ["A", "A", "B", "B"],
        TimeField.ENTITY_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
        TimeField.SOURCE_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:11"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
    },
)


def test_point_in_time_latest_with_group_keys():
    result_df = OfflineFileStore._point_in_time_latest(mock_point_in_time_latest_df, ["join_key"])

    df_a = result_df[result_df["join_key"] == "A"].iloc[0]
    df_b = result_df[result_df["join_key"] == "B"].iloc[0]

    assert df_a["_source_event_timestamp_"] == pd.Timestamp("2021-08-25 20:16:19")
    assert df_b["_source_event_timestamp_"] == pd.Timestamp("2021-08-25 20:16:20")


def test_point_in_time_latest_without_group_keys():
    result_df = OfflineFileStore._point_in_time_latest(mock_point_in_time_latest_df)
    assert result_df.loc[0]["_source_event_timestamp_"] == pd.Timestamp("2021-08-25 20:16:20")


mock_source_df = pd.DataFrame(
    {
        "join_key": ["A", "A", "B", "B"],
        "event_timestamp": [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:11"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
        "feature": [1, 2, 3, 4],
    },
)
mock_entity_df = pd.DataFrame(
    {
        "join_key": ["A", "B", "A"],
        "event_timestamp": [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:19"),
        ],
        "request_feature": [6, 5, 4],
    }
)


def test_point_in_time_join_with_join_keys():
    result_df = OfflineFileStore._point_in_time_join(
        mock_entity_df, mock_source_df, timestamp_field="event_timestamp", join_keys=["join_key"]
    )
    assert len(result_df) == 3


def test_point_in_time_join_with_ttl():
    result_df = OfflineFileStore._point_in_time_join(
        mock_entity_df,
        mock_source_df,
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
        ttl=Period.from_str("2 seconds"),
    )
    assert len(result_df) == 2


def test_point_in_time_join_with_extra_entities_in_source():
    result_df = OfflineFileStore._point_in_time_join(
        pd.DataFrame(
            {
                "join_key": ["A"],
                "event_timestamp": [pd.Timestamp("2021-08-25 20:16:18")],
                "request_feature": [6],
            }
        ),
        mock_source_df,
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
    )
    assert len(result_df) == 1


def test_point_in_time_join_with_created_timestamp():
    result_df = OfflineFileStore._point_in_time_join(
        mock_entity_df,
        pd.DataFrame(
            {
                "join_key": ["A", "A"],
                "event_timestamp": [
                    pd.Timestamp("2021-08-25 20:16:18"),
                    pd.Timestamp("2021-08-25 20:16:18"),
                ],
                "created_timestamp": [
                    pd.Timestamp("2021-08-25 20:16:21"),
                    pd.Timestamp("2021-08-25 20:16:20"),
                ],
                "feature": [5, 6],
            },
        ),
        timestamp_field="event_timestamp",
        created_timestamp_field="created_timestamp",
        join_keys=["join_key"],
        ttl=Period.from_str("2 seconds"),
    )
    assert all(result_df["feature"] == [5, 5])


def test_point_on_time_join_with_join_keys():
    result_df = OfflineFileStore._point_on_time_join(
        mock_entity_df,
        mock_source_df,
        period=-Period.from_str("2 seconds"),
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
        include=False,
    )
    assert len(result_df) == 1
    assert result_df["join_key"].values == "A"


def test_point_on_time_join_with_join_keys_label():
    result_df = OfflineFileStore._point_on_time_join(
        mock_entity_df,
        mock_source_df,
        period=Period.from_str("2 seconds"),
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
    )
    assert len(result_df) == 4


def test_point_on_time_join_with_ttl():
    result_df = OfflineFileStore._point_on_time_join(
        mock_entity_df,
        mock_source_df,
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
        ttl=Period.from_str("2 seconds"),
        period=-Period.from_str("10 seconds"),
    )
    assert len(result_df) == 3
    assert "B" not in result_df["join_key"].values


def test_point_on_time_join_with_extra_entities_in_source():
    result_df = OfflineFileStore._point_on_time_join(
        pd.DataFrame(
            {
                "join_key": ["A"],
                "event_timestamp": [pd.Timestamp("2021-08-25 20:16:22")],
                "request_feature": [6],
            }
        ),
        mock_source_df,
        period=Period.from_str("3 seconds"),
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
    )
    assert len(result_df) == 0


def test_point_on_time_join_with_created_timestamp():
    result_df = OfflineFileStore._point_on_time_join(
        mock_entity_df,
        pd.DataFrame(
            {
                "join_key": ["A", "A", "A", "A"],
                "event_timestamp": [
                    pd.Timestamp("2021-08-25 20:16:16"),
                    pd.Timestamp("2021-08-25 20:16:17"),
                    pd.Timestamp("2021-08-25 20:16:18"),
                    pd.Timestamp("2021-08-25 20:16:18"),
                ],
                "materialize_time": [
                    pd.Timestamp("2021-08-25 20:16:17"),
                    pd.Timestamp("2021-08-25 20:16:19"),
                    pd.Timestamp("2021-08-25 20:16:21"),
                    pd.Timestamp("2021-08-25 20:16:20"),
                ],
                "feature": [3, 4, 5, 6],
            },
        ),
        timestamp_field="event_timestamp",
        created_timestamp_field="materialize_time",
        join_keys=["join_key"],
        period=-Period.from_str("2 seconds"),
    )
    assert all(result_df["feature"] == [4, 5, 5])
    assert result_df[result_df["event_timestamp"] == pd.Timestamp("2021-08-25 20:16:18")].shape == (2, 5)
    assert pd.Timestamp("2021-08-25 20:16:20") not in result_df["event_timestamp"]


mocked_stats_input_df = pd.DataFrame(
    {
        "join_key": ["A", "B", "A", "B"],
        "event_timestamp": [
            pd.Timestamp("2021-08-25 20:16:16"),
            pd.Timestamp("2021-08-25 20:16:17"),
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:18"),
        ],
        "F1": [1, 2, 3, 4],
    }
)


@pytest.mark.parametrize("fn", [(fn) for fn in StatsFunctions])
def test_stats(fn):
    store = OfflineFileStore()
    file_source = FileSource(name="mock", path="mock")
    features = [Feature(name="F1", dtype=FeatureDTypes.FLOAT, view_name="hello")]

    store._read_file = MagicMock(return_value=mocked_stats_input_df)

    result_df = store.stats(
        file_source,
        features,
        fn,
        group_keys=["join_key"],
    )
    if fn == StatsFunctions.UNIQUE:
        assert ",".join(result_df.columns) == "join_key"
    else:
        assert ",".join(result_df.index.names) == "join_key"
        assert isinstance(result_df, pd.DataFrame)
