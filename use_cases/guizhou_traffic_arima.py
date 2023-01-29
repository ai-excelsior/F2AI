import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import pipeline
from pmdarima.preprocessing import DateFeaturizer, FourierFeaturizer

from f2ai.dataset import TorchIterableDataset
from f2ai.featurestore import FeatureStore


def fill_missing(data: pd.DataFrame, time_col="event_timestamp", query_col="query_timestamp"):
    min_of_time = data[time_col].min()
    max_of_time = data[time_col].max()
    query_time = data.iloc[0][query_col]

    data = data.set_index(time_col, drop=True)
    if query_time < min_of_time:
        date_index = pd.date_range(query_time, max_of_time, freq="2T", name=time_col)[1:]
    else:
        date_index = pd.date_range(min_of_time, query_time, freq="2T", name=time_col)

    df = data.reindex(date_index, method="nearest").resample("12T").sum()

    return df.reset_index()


#! f2ai materialize travel_time_prediction_arima_v1 --start=2016-03-01 --end=2016-04-01 --step='1 day'
if __name__ == "__main__":
    ex_vars = ["event_timestamp"]
    fs = FeatureStore("/Users/liyu/.f2ai/f2ai-guizhou_traffic_file")

    # 选择4条路的4个时间段分别进行预测
    entity_df = pd.DataFrame(
        {
            "link_id": [
                "4377906286525800514",
                "4377906285681600514",
                "3377906281774510514",
                "4377906280784800514",
            ],
            "event_timestamp": [
                pd.Timestamp("2016-03-31 07:00:00"),
                pd.Timestamp("2016-03-31 09:00:00"),
                pd.Timestamp("2016-03-31 12:00:00"),
                pd.Timestamp("2016-03-31 18:00:00"),
            ],
        }
    )
    dataset = TorchIterableDataset(fs, "travel_time_prediction_arima_v1", entity_df)

    fig = plt.figure(figsize=(16, 16))
    axes = fig.subplots(2, 2)
    for i, (look_back, look_forward) in enumerate(dataset):
        look_back = fill_missing(look_back)
        look_forward = fill_missing(look_forward)

        pipe = pipeline.Pipeline(
            [
                ("date", DateFeaturizer(column_name="event_timestamp", with_day_of_month=False)),
                ("fourier", FourierFeaturizer(m=24 * 5, k=4)),
                ("arima", pm.arima.ARIMA(order=(6, 0, 1))),
            ]
        )
        pipe.fit(look_back["travel_time"], X=look_back[ex_vars])
        look_forward["y_pred"] = pipe.predict(len(look_forward), X=look_forward[ex_vars])
        melted_df = pd.melt(look_forward, id_vars=["event_timestamp"], value_vars=["travel_time", "y_pred"])
        sns.lineplot(melted_df, x="event_timestamp", y="value", hue="variable", ax=axes[i // 2, i % 2])

    fig.savefig("f2ai_guizhou_traffic_arima", bbox_inches="tight")
