import pandas as pd
import timeit
from os import path
from aie_feast import FeatureStore

LINE_LIMIT = 10


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


def test_get_features(make_credit_score):
    project_folder = make_credit_score("file")
    entity_df = get_credit_score_entities(project_folder)
    store = FeatureStore(project_folder)
    store.get_features(store.feature_views["zipcode_features"], entity_df)

    measured_time = timeit.timeit(
        lambda: store.get_features(store.feature_views["zipcode_features"], entity_df), number=10
    )
    print(f"get_features performance: {measured_time}s")


def test_materialize(make_credit_score):
    project_folder = make_credit_score("file")
    store = FeatureStore(project_folder)

    measured_time = timeit.timeit(lambda: store.materialize("credit_scoring_v1"), number=1)
    print(f"materialize performance: {measured_time}s")
