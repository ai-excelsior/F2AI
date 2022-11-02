import pandas as pd
import timeit
from os import path
from aie_feast import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler

LINE_LIMIT = 1000


def get_credit_score_entities(project_folder: str):
    query_entities = pd.read_parquet(
        path.join(project_folder, "row_data/loan_table.parquet"),
        columns=["loan_id", "dob_ssn", "zipcode", "created_timestamp", "event_timestamp"],
    ).iloc[:LINE_LIMIT]
    return query_entities.astype(
        {
            "loan_id": "string",
            "dob_ssn": "string",
            "zipcode": "string",
        }
    )


def get_guizhou_traffic_entities(project_folder: str):
    query_entities = pd.read_csv(
        path.join(project_folder, "raw_data/gy_link_travel_time.csv"),
        usecols=["link_id", "event_timestamp"],
        nrows=5,
    )
    return query_entities.astype({"link_id": "string"})


def test_get_features_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    entity_df = get_credit_score_entities(project_folder)
    store = FeatureStore(project_folder)
    store.get_features("zipcode_features", entity_df)

    measured_time = timeit.timeit(lambda: store.get_features("zipcode_features", entity_df), number=10)
    print(f"get_features performance: {measured_time}s")
    print(store.get_features("zipcode_features", entity_df))


def test_get_labels_from_label_views(make_credit_score):
    project_folder = make_credit_score("file")
    entity_df = get_credit_score_entities(project_folder)
    store = FeatureStore(project_folder)
    store.get_labels("loan_label_view", entity_df)


def test_materialize(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)

    measured_time = timeit.timeit(lambda: store.materialize("credit_scoring_v1"), number=1)
    print(f"materialize performance: {measured_time}s")


def test_get_period_features_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("file")
    entity_df = get_guizhou_traffic_entities(project_folder)
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_period_features("gy_link_travel_time_features", entity_df, period="20 minutes"),
        number=10,
    )
    print(f"get_period_features performance: {measured_time}s")


def test_stats_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    entity_df = get_credit_score_entities(project_folder)
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(lambda: store.stats("loan_features", entity_df, fn="mean"), number=10)
    print(f"stats performance: {measured_time}s")


def test_get_latest_entity_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(lambda: store.get_latest_entities("loan_features"), number=10)
    print(f"get_latest_entities performance: {measured_time}s")
    print(store.get_latest_entities("loan_features"))


def test_materialize(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(lambda: store.materialize(service_name="credit_scoring_v1"), number=1)
    print(f"stats performance: {measured_time}s")


def test_sampler_with_groups(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    groups = store.stats("loan_features", group_key=["zipcode", "dob_ssn"], keys_only=True, fn="unique")
    groups = list(zip(groups["zipcode"], groups["dob_ssn"]))
    measured_time = timeit.timeit(
        lambda: GroupFixednbrSampler(
            time_bucket="10 days",
            stride=1,
            group_ids=groups,
            group_names=["zipcode", "dob_ssn"],
            start="2020-08-20",
            end="2021-08-30",
        )(),
        number=5,
    )
    print(f"sampler with groups performance: {measured_time}s")


def test_dataset_to_pytorch(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    store.materialize(service_name="credit_scoring_v1")
    ds = store.get_dataset(
        service_name="credit_scoring_v1",
        sampler=GroupFixednbrSampler(
            time_bucket="10 days",
            stride=1,
            group_ids=None,
            group_names=None,
            start="2020-08-20",
            end="2021-08-30",
        ),
    )
    measured_time = timeit.timeit(lambda: list(ds.to_pytorch()), number=1)
    print(f"dataset.to_pytorch performance: {measured_time}s")
