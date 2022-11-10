from pyexpat import features
import pandas as pd
import timeit
from f2ai import FeatureStore
from f2ai.common.sampler import GroupFixednbrSampler


def get_guizhou_traffic_entities(store):
    query_entities = pd.DataFrame(
        store.offline_store._get_dataframe(
            sql_result=f"select link_id,event_timestamp from {store.services['traval_time_prediction_embedding_v1'].materialize_path} limit 100",
            join_keys=["link_id"],
            timecol=["event_timestamp"],
            feature_names=[],
        ),
        columns=["link_id", "event_timestamp"],
    )
    return query_entities.astype({"link_id": "string"})


def test_get_features_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    entity_df = get_guizhou_traffic_entities(store)

    store.get_features("gy_link_travel_time_features", entity_df)

    measured_time = timeit.timeit(
        lambda: store.get_features("gy_link_travel_time_features", entity_df), number=10
    )
    print(f"get_features performance pgsql: {measured_time}s")


def test_stats_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    entity_df = get_guizhou_traffic_entities(store)

    measured_time = timeit.timeit(
        lambda: store.stats("gy_link_travel_time_features", fn="mean", entity_df=entity_df), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_unique_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.stats("gy_link_travel_time_features", group_key=["link_id"]), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_get_latest_entities_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_latest_entities(
            "gy_link_travel_time_features",
            pd.DataFrame({"link_id": ["3377906281518510514", "4377906284141600514"]}),
        ),
        number=10,
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_get_latest_entity_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_latest_entities("gy_link_travel_time_features"), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_get_period_features_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    entity_df = get_guizhou_traffic_entities(store)
    measured_time = timeit.timeit(
        lambda: store.get_period_features("gy_link_travel_time_features", entity_df, period="10 minutes"),
        number=10,
    )
    print(f"get_features performance pgsql: {measured_time}s")


def test_dataset_to_pytorch_pgsql(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    ds = store.get_dataset(
        service_name="traval_time_prediction_embedding_v1",
        sampler=GroupFixednbrSampler(
            time_bucket="1 hours",
            stride=1,
            group_ids=None,
            group_names=None,
            start="2016-03-05 00:00:00",
            end="2016-03-06 00:00:00",
        ),
    )
    measured_time = timeit.timeit(lambda: list(ds.to_pytorch(64)), number=1)
    print(f"dataset.to_pytorch pgsql performance: {measured_time}s")


def test_materialize(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.materialize("traval_time_prediction_embedding_v1", incremental_begin="4 minutes"),
        number=2,
    )
    print(f"materialize performance pgsql: {measured_time}s")
