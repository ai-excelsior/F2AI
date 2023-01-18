import pandas as pd
import timeit
from os import path
from f2ai import FeatureStore
from f2ai.dataset import EvenEventsSampler, FixedNEntitiesSampler, NoEntitiesSampler
from f2ai.definitions import BackOffTime, Period

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


def test_get_labels_from_label_views(make_credit_score):
    project_folder = make_credit_score("file")
    entity_df = get_credit_score_entities(project_folder)
    store = FeatureStore(project_folder)
    store.get_labels("loan_label_view", entity_df)


def test_materialize(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    back_off_time = BackOffTime(start="2020-08-25", end="2021-08-26", step="1 month")
    measured_time = timeit.timeit(
        lambda: store.materialize(
            service_or_views="credit_scoring_v1", back_off_time=back_off_time, online=False
        ),
        number=1,
    )
    print(f"materialize performance: {measured_time}s")


def test_get_features_from_service(make_credit_score):
    """this test should run after materialize"""
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    entity_df = get_credit_score_entities(project_folder)
    store.get_features("credit_scoring_v1", entity_df)


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
    store = FeatureStore(project_folder)

    measured_time = timeit.timeit(lambda: store.stats("loan_features", fn="avg"), number=10)
    print(f"stats performance: {measured_time}s")


def test_get_latest_entities_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_latest_entities("loan_features", pd.DataFrame({"dob_ssn": ["19960703_3449"]})),
        number=10,
    )
    print(f"get_latest_entities performance: {measured_time}s")


def test_get_latest_entity_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(lambda: store.get_latest_entities("loan_features"), number=10)
    print(f"get_latest_entities performance: {measured_time}s")


def test_sampler_with_groups(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    group_df = store.stats("loan_features", group_keys=["zipcode", "dob_ssn"], fn="unique")
    events_sampler = EvenEventsSampler(start="2020-08-20", end="2021-08-30", period="10 days")
    entities_sampler = FixedNEntitiesSampler(
        events_sampler,
        group_df=group_df,
    )
    measured_time = timeit.timeit(
        lambda: entities_sampler(),
        number=1,
    )
    print(f"sampler with groups performance: {measured_time}s")


def test_dataset_to_pytorch(make_credit_score):
    project_folder = make_credit_score("file")
    backoff = BackOffTime(
        start=pd.Timestamp("2020-05-01"), end=pd.Timestamp("2020-07-01"), step=Period.from_str("1 month")
    )
    store = FeatureStore(project_folder)
    store.materialize(service_or_views="credit_scoring_v1", back_off_time=backoff)

    events_sampler = EvenEventsSampler(start="2020-08-20", end="2021-08-30", period="10 days")
    ds = store.get_dataset(
        service="credit_scoring_v1",
        sampler=NoEntitiesSampler(events_sampler),
    )
    measured_time = timeit.timeit(lambda: list(ds.to_pytorch()), number=1)
    print(f"dataset.to_pytorch performance: {measured_time}s")


def test_online_materialize(make_credit_score):
    project_folder = make_credit_score("file")
    back_off_time = BackOffTime(
        start=pd.Timestamp("2020-05-01"), end=pd.Timestamp("2020-07-01"), step=Period.from_str("1 month")
    )
    store = FeatureStore(project_folder)
    store.materialize(service_or_views="credit_scoring_v1", back_off_time=back_off_time, online=True)
