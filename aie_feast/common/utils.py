from typing import List, Union, Tuple
import pandas as pd
from pypika import functions as fn, Parameter
from pypika.queries import QueryBuilder
from pandas._libs.tslibs.timestamps import Timestamp
import oss2
import os

FSKEY = "__FS__"
TIME_COL = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"


def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]


def get_default_value():
    return None


def schema_to_dict(schema):
    return {item["name"]: item.get("dtype", "string") for item in schema}


# TODO: refactor this to speedup column oriented file format
def read_file(path, file_format=None, time_cols=None, entity_cols=None):
    time_cols = [i for i in time_cols if i]
    path = remove_prefix(path, "file://")


    dtype_str = {en: str for en in entity_cols}

    if file_format is None:
        file_format = path.split(".")[-1]

    if file_format.startswith("parq"):
        df = pd.read_parquet(path)
    elif file_format.startswith("tsv"):
        df = pd.read_csv(path, sep="\t", parse_dates=time_cols if time_cols else [], dtype=dtype_str)
    elif file_format.startswith("txt"):
        df = pd.read_csv(path, sep=" ", parse_dates=time_cols if time_cols else [], dtype=dtype_str)
    else:
        df = pd.read_csv(path, parse_dates=time_cols if time_cols else [], dtype=dtype_str)
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], utc=True)

    return df


def to_file(file, path, type):
    path = remove_prefix(path, "file://")
    if type.startswith("parq"):
        file.to_parquet(path)
    elif type.startswith("tsv"):
        file.to_csv(path, sep="\t")
    elif type.startswith("txt"):
        file.to_csv(path, sep=" ")
    else:
        file.to_csv(path)


def build_filter_time_query(
    q: QueryBuilder, start: Union[Timestamp, str], end: Union[Timestamp, str], include: str, timecol: str
) -> "QueryBuilder":
    if include == "neither":
        return (
            q.where(
                Parameter(
                    f" ({timecol}::timestamp > '{start}'::timestamp) and ({timecol}::timestamp < '{end}'::timestamp) "
                )
            )
            if end
            else q.where(
                Parameter(
                    f" ({timecol}::timestamp > '{start}'::timestamp) and ({timecol}::timestamp < {TIME_COL}_tmp::timestamp) "
                )
            )
        )
    elif include == "left":
        return (
            q.where(
                Parameter(
                    f" ({timecol}::timestamp >= '{start}'::timestamp) and ({timecol}::timestamp < '{end}'::timestamp) "
                )
            )
            if end
            else q.where(
                Parameter(
                    f" ({timecol}::timestamp >= '{start}'::timestamp) and ({timecol}::timestamp < {TIME_COL}_tmp::timestamp) "
                )
            )
        )
    elif include == "right":
        return (
            q.where(
                Parameter(
                    f" ({timecol}::timestamp > '{start}'::timestamp) and ({timecol}::timestamp <= '{end}'::timestamp) "
                )
            )
            if end
            else q.where(
                Parameter(
                    f" ({timecol}::timestamp > '{start}'::timestamp) and ({timecol}::timestamp <= {TIME_COL}_tmp::timestamp) "
                )
            )
        )
    else:
        return (
            q.where(
                Parameter(
                    f" ({timecol}::timestamp >= '{start}'::timestamp) and ({timecol}::timestamp <= '{end}'::timestamp) "
                )
            )
            if end
            else q.where(
                Parameter(
                    f" ({timecol}::timestamp >= '{start}'::timestamp) and ({timecol}::timestamp <= {TIME_COL}_tmp::timestamp) "
                )
            )
        )


def build_agg_query(
    q: QueryBuilder,
    features: List[str],
    entity_cols: List[str],
    agg_type: str,
    start: Union[Timestamp, str],
    end: Union[Timestamp, str],
    include: str,
    timecol: str = TIME_COL,
) -> "QueryBuilder":

    q = build_filter_time_query(q, start, end, include, timecol)

    if agg_type == "mean":
        return (
            q.groupby(*entity_cols).select(*([fn.Avg(Parameter(fea), fea) for fea in features] + entity_cols))
            if entity_cols
            else q.select(*([fn.Avg(Parameter(fea), fea) for fea in features] + entity_cols))
        )
    elif agg_type == "sum":
        return (
            q.groupby(*entity_cols).select(*([fn.Sum(Parameter(fea), fea) for fea in features] + entity_cols))
            if entity_cols
            else q.select(*([fn.Sum(Parameter(fea), fea) for fea in features] + entity_cols))
        )
    elif agg_type == "max":
        return (
            q.groupby(*entity_cols).select(*([fn.Max(Parameter(fea), fea) for fea in features] + entity_cols))
            if entity_cols
            else q.select(*([fn.Max(Parameter(fea), fea) for fea in features] + entity_cols))
        )
    elif agg_type == "min":
        return (
            q.groupby(*entity_cols).select(*([fn.Min(Parameter(fea), fea) for fea in features] + entity_cols))
            if entity_cols
            else q.select(*([fn.Min(Parameter(fea), fea) for fea in features] + entity_cols))
        )
    elif agg_type == "std":
        return (
            q.groupby(*entity_cols).select(*([fn.Std(Parameter(fea), fea) for fea in features] + entity_cols))
            if entity_cols
            else q.select(*([fn.Std(Parameter(fea), fea) for fea in features] + entity_cols))
        )
    elif agg_type == "mode":
        return (
            q.groupby(*entity_cols).select(
                *(
                    [Parameter(f"MODE() WITHIN GROUP (ORDER BY {fea}) as {fea}") for fea in features]
                    + entity_cols
                )
            )
            if entity_cols
            else q.select(
                *(
                    [Parameter(f"MODE() WITHIN GROUP (ORDER BY {fea}) as {fea}") for fea in features]
                    + entity_cols
                )
            )
        )
    elif agg_type == "median":
        return (
            q.groupby(*entity_cols).select(
                *(
                    [
                        Parameter(f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {fea}) as {fea}")
                        for fea in features
                    ]
                    + entity_cols
                )
            )
            if entity_cols
            else q.select(
                *(
                    [
                        Parameter(f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {fea}) as {fea}")
                        for fea in features
                    ]
                    + entity_cols
                )
            )
        )
    elif agg_type == "unique":
        return (
            q.groupby(*entity_cols).select(
                *([Parameter(f"distinct({fea} as {fea})") for fea in features] + entity_cols)
            )
            if features
            else q.select(Parameter(f"distinct({','.join(entity_cols)})"))
        )


# TODO: refactor freq later, to keep consistant with transform_pgsql_period
def transform_freq(dt):
    value, freq = dt.split(" ")
    if freq == "quarters":
        freq = "months"
        value = int(value) * 3
    elif freq == "milliseconds":
        freq = "microseconds"
        value = int(value) * 10e3
    elif freq == "second":
        freq = "seconds"
        value = int(value)

    return {freq: int(value)}


def transform_pgsql_period(period, is_label: bool = False):
    value, freq = period.split(" ")
    if freq == "minutes" or freq == "minute":
        freq = "mins"
    elif freq == "quarters" or freq == "quater":
        freq = "months"
        value = int(value) * 3
    elif freq in ["millisecs", "milliseconds", "milli", "millisecond"]:
        freq = "seconds"
        value = int(value) / 10e3
    elif freq in ["microsecs", "microseconds"]:
        freq = "seconds"
        value = int(value) / 10e6
    return str(value) + " " + freq if is_label else "-" + str(value) + " " + freq


def parse_date(dt):
    return {**transform_freq(dt)} if dt else 0


# TODO: this is confused with get_latest_record
def get_newest_record(df, time_col, join_keys, create_time):
    if len(df) == 0:
        return get_latest_record(df, time_col, create_time)
    return (
        df.groupby(join_keys + [time_col + "_y"])
        .apply(get_latest_record, time_col, create_time)
        .reset_index(drop=True)
    )


def get_latest_record(df, time_col, create_time):
    df = df[df[time_col + "_x"] == df[time_col + "_x"].max()]
    if create_time in df.columns:  # only used when have duplicate event_timestamp values
        df = df[df[create_time] == df[create_time].max()]
        df.drop(columns=create_time, inplace=True)
    df.rename(columns={time_col + "_x": create_time}, inplace=True)  # rename action timestamp
    df.rename(columns={time_col + "_y": time_col}, inplace=True)  # time defined in `entity_df`
    return df


def get_consistent_format(views):
    return views if isinstance(views, dict) else {FSKEY: views}


def get_stats_result(data: pd.DataFrame, fn: str, use_cols: list, include, start):
    if include == "neither":
        return data[
            (data[SOURCE_EVENT_TIMESTAMP_FIELD] < data[ENTITY_EVENT_TIMESTAMP_FIELD])
            & (data[SOURCE_EVENT_TIMESTAMP_FIELD] > start)
        ][use_cols].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    elif include == "left":
        return data[
            (data[SOURCE_EVENT_TIMESTAMP_FIELD] < data[ENTITY_EVENT_TIMESTAMP_FIELD])
            & (data[SOURCE_EVENT_TIMESTAMP_FIELD] >= start)
        ][use_cols].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    elif include == "right":
        return data[
            (data[SOURCE_EVENT_TIMESTAMP_FIELD] <= data[ENTITY_EVENT_TIMESTAMP_FIELD])
            & (data[SOURCE_EVENT_TIMESTAMP_FIELD] > start)
        ][use_cols].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    else:
        return data[
            (data[SOURCE_EVENT_TIMESTAMP_FIELD] <= data[ENTITY_EVENT_TIMESTAMP_FIELD])
            & (data[SOURCE_EVENT_TIMESTAMP_FIELD] >= start)
        ][use_cols].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)


def get_bucket(bucket, endpoint=None):
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = endpoint or os.environ.get("OSS_ENDPOINT")

    return oss2.Bucket(oss2.Auth(key_id, key_secret), endpoint, bucket)


def parse_oss_url(url: str) -> Tuple[str, str, str]:
    """
    url format:  oss://{bucket}/{key}
    """
    url = remove_prefix(url, "oss://")
    components = url.split("/")
    return components[0], "/".join(components[1:])


def get_bucket_from_oss_url(url: str):
    bucket_name, key = parse_oss_url(url)
    return get_bucket(bucket_name), key
