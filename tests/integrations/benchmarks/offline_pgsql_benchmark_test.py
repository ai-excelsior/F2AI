import pandas as pd
import timeit
from os import path
from aie_feast import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler


def test_dataset_to_pytorch_pgsql(make_guizhou_traffic):
    project_folder = make_guizhou_traffic("pgsql")
    store = FeatureStore(project_folder)
    store.materialize(service_name="credit_scoring_v1")
    ds = store.get_dataset(
        service_name="traval_time_prediction_embedding_v1",
        sampler=GroupFixednbrSampler(
            time_bucket="10 days",
            stride=1,
            group_ids=None,
            group_names=None,
            start="2016-03-01 08:02:00",
            end="2016-07-01 00:00:00",
        ),
    )
    measured_time = timeit.timeit(lambda: list(ds.to_pytorch()), number=1)
    print(f"dataset.to_pytorch performance: {measured_time}s")
