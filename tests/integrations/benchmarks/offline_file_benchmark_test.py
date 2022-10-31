import pandas as pd
import timeit
from os import path
from aie_feast import FeatureStore

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
    entity_df.rename(columns={"link_id": "link"}, inplace=True)
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.stats("loan_features", entity_df=entity_df, fn="unique"),
        number=10,
    )
    print(f"stats performance: {measured_time}s")


def test_get_latest_entity_from_feature_view(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(lambda: store.get_latest_entities("loan_features"), number=10)
    print(f"get_latest_entities performance: {measured_time}s")
    print(store.get_latest_entities("loan_features"))
