import pandas as pd
import timeit
from f2ai import FeatureStore
from f2ai.definitions import StatsFunctions
from f2ai.dataset import EvenEventsSampler, NoEntitiesSampler
from f2ai.definitions import BackOffTime


def get_guizhou_traffic_entities(store: FeatureStore):
    columns = ["link_id", "event_timestamp"]
    query_entities = store.offline_store._get_dataframe(
        f"select {', '.join(columns)} from gy_link_travel_time order by event_timestamp limit 1000",
        columns=columns,
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

    measured_time = timeit.timeit(
        lambda: store.stats("gy_link_travel_time_features", fn=StatsFunctions.AVG), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_unique_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)

    measured_time = timeit.timeit(
        lambda: store.stats("gy_link_travel_time_features", group_keys=["link_id"]), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


def test_get_latest_entities_from_feature_view_with_entity_df(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_latest_entities(
            "gy_link_travel_time_features",
            pd.DataFrame({"link_id": ["3377906281518510514", "4377906284141600514"]}),
        ),
        number=10,
    )
    print(f"get_latest_entities with entity_df performance pgsql: {measured_time}s")


def test_get_latest_entity_from_feature_view(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    measured_time = timeit.timeit(
        lambda: store.get_latest_entities("gy_link_travel_time_features"), number=10
    )
    print(f"get_latest_entities performance pgsql: {measured_time}s")


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
    events_sampler = EvenEventsSampler(start="2016-03-05 00:00:00", end="2016-03-06 00:00:00", period='1 hours')

    ds = store.get_dataset(
        service="traval_time_prediction_embedding_v1",
        sampler=NoEntitiesSampler(events_sampler),
    )
    measured_time = timeit.timeit(lambda: list(ds.to_pytorch(64)), number=1)
    print(f"dataset.to_pytorch pgsql performance: {measured_time}s")


def test_materialize(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    backoff_time = BackOffTime(
        start="2016-03-01 08:02:00+08", end="2016-03-01 08:06:00+08", step="4 minutes"
    )
    measured_time = timeit.timeit(
        lambda: store.materialize(
            service="traval_time_prediction_embedding_v1",
            backoff=backoff_time,
        ),
        number=1,
    )

    print(f"materialize performance pgsql: {measured_time}s")
