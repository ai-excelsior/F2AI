import pandas as pd
from aie_feast.offline_stores.offline_file_store import (
    OfflineFileStore,
    ENTITY_EVENT_TIMESTAMP_FIELD,
    SOURCE_EVENT_TIMESTAMP_FIELD,
)

mock_point_in_time_filter_df = pd.DataFrame(
    {
        "join_key": ["A", "A", "A", "A"],
        ENTITY_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
        SOURCE_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:21"),
        ],
    },
)


def test_point_in_time_filter_simple():
    result_df = OfflineFileStore.point_in_time_filter(mock_point_in_time_filter_df)
    assert len(result_df) == 3
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")


def test_point_in_time_filter_not_include():
    result_df = OfflineFileStore.point_in_time_filter(mock_point_in_time_filter_df, include=False)
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:19")


def test_point_in_time_filter_with_ttl():
    result_df = OfflineFileStore.point_in_time_filter(mock_point_in_time_filter_df, ttl="2 seconds")
    assert len(result_df) == 2
    assert result_df["_source_event_timestamp_"].max() == pd.Timestamp("2021-08-25 20:16:20")
    assert result_df["_source_event_timestamp_"].min() == pd.Timestamp("2021-08-25 20:16:19")


mock_point_in_time_latest_df = pd.DataFrame(
    {
        "join_key": ["A", "A", "B", "B"],
        ENTITY_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
        SOURCE_EVENT_TIMESTAMP_FIELD: [
            pd.Timestamp("2021-08-25 20:16:18"),
            pd.Timestamp("2021-08-25 20:16:19"),
            pd.Timestamp("2021-08-25 20:16:11"),
            pd.Timestamp("2021-08-25 20:16:20"),
        ],
    },
)


def test_point_in_time_latest_with_group_keys():
    result_df = OfflineFileStore.point_in_time_latest(mock_point_in_time_latest_df, ["join_key"])
    assert result_df.loc[0]["_source_event_timestamp_"] == pd.Timestamp("2021-08-25 20:16:19")
    assert result_df.loc[1]["_source_event_timestamp_"] == pd.Timestamp("2021-08-25 20:16:20")


def test_point_in_time_latest_without_group_keys():
    result_df = OfflineFileStore.point_in_time_latest(mock_point_in_time_latest_df)
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
    result_df = OfflineFileStore.point_in_time_join(
        mock_entity_df, mock_source_df, timestamp_field="event_timestamp", join_keys=["join_key"]
    )
    assert len(result_df) == 3


def test_point_in_time_join_with_ttl():
    result_df = OfflineFileStore.point_in_time_join(
        mock_entity_df,
        mock_source_df,
        timestamp_field="event_timestamp",
        join_keys=["join_key"],
        ttl="2 seconds",
    )
    assert len(result_df) == 2


def test_point_in_time_join_with_extra_entities_in_source():
    result_df = OfflineFileStore.point_in_time_join(
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
    result_df = OfflineFileStore.point_in_time_join(
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
        ttl="2 seconds",
    )
    assert all(result_df["feature"] == [5, 5])
