import os
import zipfile
import tempfile
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pmdarima.model_selection import train_test_split
from f2ai.common.sampler import GroupFixednbrSampler
from f2ai.common.utils import get_bucket_from_oss_url
from f2ai.featurestore import FeatureStore


if __name__ == "__main__":
    download_from = "oss://aiexcelsior-shanghai-test/xyz_test_data/guizhou_traffic.zip"
    save_path = "/tmp/"
    save_dir = tempfile.mkdtemp(prefix=save_path)
    bucket, key = get_bucket_from_oss_url(download_from)
    print(bucket)
    print(key)
    dest_zip_filepath = os.path.join(save_dir, key)
    os.makedirs(os.path.dirname(dest_zip_filepath), exist_ok=True)
    bucket.get_object_to_file(key, dest_zip_filepath)
    zipfile.ZipFile(dest_zip_filepath).extractall(dest_zip_filepath.rsplit("/", 1)[0])
    os.remove(dest_zip_filepath)
    print(f"Project downloaded and saved in {dest_zip_filepath.rsplit('/',1)[0]}")

    TIME_COL = "event_timestamp"
    fs = FeatureStore(f"file://{save_dir}/{key.rstrip('.zip')}")

    entities = [
        fs.entities[entity_name]
        for entity_name in fs.entities.keys()
        if entity_name in fs._get_feature_to_use(fs.services["traval_time_prediction_embedding_v1"])
    ]

    # print(
    #     f'Earliest timestamp: {fs.get_latest_entities("traval_time_prediction_embedding_v1")[TIME_COL].min()}'
    # )
    # print(
    #     f'Latest timestamp: {fs.get_latest_entities("traval_time_prediction_embedding_v1")[TIME_COL].max()}'
    # )

    unique_entity = fs.stats(
        "traval_time_prediction_embedding_v1",
        fn="unique",
        group_key=[],
        start="2016-03-01",
        end="2016-03-31",
        features=["link_id"],
    )

    sample = fs.get_dataset(
        service_name="traval_time_prediction_embedding_v1",
        sampler=GroupFixednbrSampler(
            time_bucket="20 minutes",
            stride=4,
            group_ids=list(unique_entity["link_id"].map(lambda x: str(x))),
            group_names=["link_id"],
            start="2016-03-01",
            end="2016-03-10",
        ),
    )
    ids = sample.to_pytorch()
    entity_df = ids.entity_index
    data = fs.get_labels("traval_time_prediction_embedding_v1", entity_df)

    data["month"] = data["event_timestamp"].map(lambda x: pd.to_datetime(x).month)
    data["day"] = data["event_timestamp"].map(lambda x: pd.to_datetime(x).day)
    data["hour"] = data["event_timestamp"].map(lambda x: pd.to_datetime(x).hour)
    label_mean = (
        data.groupby(["month", "day", "hour"])["travel_time"].mean().droplevel(level=["month", "day"])
    )
    data = np.array(label_mean)

    train, test = train_test_split(data, train_size=200)
    model = pm.auto_arima(train, max_p=10, max_q=10, max_d=4)
    forecasts = model.predict(test.shape[0])

    x = np.arange(data.shape[0])
    plt.figure()
    plt.plot(x, data, c="blue")
    plt.plot(x[200:], forecasts, c="red")

    MSE = mean_squared_error(test, forecasts)
