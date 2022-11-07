import pandas as pd
import timeit
from os import path
from aie_feast import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler
from aie_feast.common.psl_utils import sql_df, psy_conn


def get_guizhou_traffic_entities(store):

    query_entities = pd.DataFrame(
        sql_df(
            sql=f"select link_id,event_timestamp from {store.services['traval_time_prediction_embedding_v1'].materialize_path} limit 100",
            conn=psy_conn(store.offline_store),
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
        lambda: store.get_latest_entities("gy_link_travel_time_features"), number=10
    )
    print(f"stats performance pgsql: {measured_time}s")


# def test_dataset_to_pytorch_pgsql(make_guizhou_traffic):
#     project_folder = make_guizhou_traffic("pgsql")
#     store = FeatureStore(project_folder)
#     ds = store.get_dataset(
#         service_name="traval_time_prediction_embedding_v1",
#         sampler=GroupFixednbrSampler(
#             time_bucket="12 hours",
#             stride=1,
#             group_ids=None,
#             group_names=None,
#             start="2016-03-01 08:02:00",
#             end="2016-07-01 00:00:00",
#         ),
#     )
#     measured_time = timeit.timeit(lambda: list(ds.to_pytorch()), number=1)
#     print(f"dataset.to_pytorch pgsql performance: {measured_time}s")
