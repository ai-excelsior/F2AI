from typing import List, Union, Tuple
import pandas as pd
from pypika import functions as fn, Parameter, Query
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


def read_file(
    path,
    parse_dates: List[str] = [],
    str_cols: List[str] = [],
    keep_cols: List[str] = [],
    file_format=None,
):
    path = remove_prefix(path, "file://")
    dtypes = {en: str for en in str_cols}
    usecols = list(set(keep_cols + parse_dates + str_cols))

    if file_format is None:
        file_format = path.split(".")[-1]

    if file_format.startswith("parq"):
        df = pd.read_parquet(path, columns=usecols).astype(dtypes)
    elif file_format.startswith("tsv"):
        df = pd.read_csv(path, sep="\t", parse_dates=parse_dates, dtype=dtypes, usecols=usecols)
    elif file_format.startswith("txt"):
        df = pd.read_csv(path, sep=" ", parse_dates=parse_dates, dtype=dtypes, usecols=usecols)
    else:
        df = pd.read_csv(path, parse_dates=parse_dates, dtype=dtypes, usecols=usecols)

    for col in parse_dates:
        df[col] = pd.to_datetime(df[col], utc=True)

    return df


def to_file(file, path, type):
    path = remove_prefix(path, "file://")
    if type.startswith("parq"):
        file.to_parquet(path, index=False)
    elif type.startswith("tsv"):
        file.to_csv(path, sep="\t", index=False)
    elif type.startswith("txt"):
        file.to_csv(path, sep=" ", index=False)
    else:
        file.to_csv(path, index=False)


def build_filter_time_query(
    q: QueryBuilder,
    start: Union[Timestamp, str],
    include: str,
    entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
    source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
) -> "QueryBuilder":
    if include == "neither":
        return q.where(
            Parameter(
                f" ({source_timestamp_field}::timestamptz > '{start}'::timestamptz) and ({source_timestamp_field}::timestamptz < {entity_timestamp_field}::timestamptz) "
            )
        )
    elif include == "left":
        return q.where(
            Parameter(
                f" ({source_timestamp_field}::timestamptz >= '{start}'::timestamptz) and ({source_timestamp_field}::timestamptz < {entity_timestamp_field}::timestamptz) "
            )
        )
    elif include == "right":
        return q.where(
            Parameter(
                f" ({source_timestamp_field}::timestamptz > '{start}'::timestamptz) and ({source_timestamp_field}::timestamptz <= {entity_timestamp_field}::timestamptz) "
            )
        )
    else:
        return q.where(
            Parameter(
                f" ({source_timestamp_field}::timestamptz >= '{start}'::timestamptz) and ({source_timestamp_field}::timestamptz <= {entity_timestamp_field}::timestamptz) "
            )
        )


def build_agg_query(
    q: QueryBuilder, features: List[str], entity_cols: List[str], agg_type: str, keys_only
) -> "QueryBuilder":
    if keys_only:
        return q.groupby(*entity_cols).select(*entity_cols)
    if agg_type == "mean":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [fn.Avg(Parameter(fea.name), fea.name) for fea in features])
            )
            if entity_cols
            else q.select(*(entity_cols + [fn.Avg(Parameter(fea.name), fea.name) for fea in features]))
        )
    elif agg_type == "sum":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [fn.Sum(Parameter(fea.name), fea.name) for fea in features])
            )
            if entity_cols
            else q.select(*(entity_cols + [fn.Sum(Parameter(fea.name), fea.name) for fea in features]))
        )
    elif agg_type == "max":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [fn.Max(Parameter(fea.name), fea.name) for fea in features])
            )
            if entity_cols
            else q.select(*(entity_cols + [fn.Max(Parameter(fea.name), fea.name) for fea in features]))
        )
    elif agg_type == "min":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [fn.Min(Parameter(fea.name), fea) for fea in features])
            )
            if entity_cols
            else q.select(*(entity_cols + [fn.Min(Parameter(fea.name), fea.name) for fea in features]))
        )
    elif agg_type == "std":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [fn.Std(Parameter(fea.name), fea.name) for fea in features])
            )
            if entity_cols
            else q.select(*(entity_cols + [fn.Std(Parameter(fea.name), fea.name) for fea in features]))
        )
    elif agg_type == "mode":
        return (
            q.groupby(*entity_cols).select(
                *(
                    entity_cols
                    + [
                        Parameter(f"MODE() WITHIN GROUP (ORDER BY {fea.name}) as {fea.name}")
                        for fea in features
                    ]
                )
            )
            if entity_cols
            else q.select(
                *(
                    entity_cols
                    + [
                        Parameter(f"MODE() WITHIN GROUP (ORDER BY {fea.name}) as {fea.name}")
                        for fea in features
                    ]
                )
            )
        )
    elif agg_type == "median":
        return (
            q.groupby(*entity_cols).select(
                *(
                    entity_cols
                    + [
                        Parameter(f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {fea.name}) as {fea.name}")
                        for fea in features
                    ]
                )
            )
            if entity_cols
            else q.select(
                *(
                    entity_cols
                    + [
                        Parameter(f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {fea.name}) as {fea.name}")
                        for fea in features
                    ]
                )
            )
        )
    elif agg_type == "unique":
        return (
            q.groupby(*entity_cols).select(
                *(entity_cols + [Parameter(f"distinct({fea.name} as {fea.name})") for fea in features])
            )
            if entity_cols
            else q.select(
                *(entity_cols + [Parameter(f"distinct({fea.name} as {fea.name})") for fea in features])
            )
        )


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


# the code below copied from https://github.com/pandas-dev/pandas/blob/91111fd99898d9dcaa6bf6bedb662db4108da6e6/pandas/io/sql.py#L1155
def convert_dtype_to_sqlalchemy_type(col):
    from sqlalchemy.types import (
        TIMESTAMP,
        BigInteger,
        Boolean,
        Date,
        DateTime,
        Float,
        Integer,
        SmallInteger,
        Text,
        Time,
    )
    from pandas._libs.lib import infer_dtype

    col_type = infer_dtype(col, skipna=True)

    if col_type == "datetime64" or col_type == "datetime":
        try:
            if col.dt.tz is not None:
                return TIMESTAMP(timezone=True)
        except AttributeError:
            if getattr(col, "tz", None) is not None:
                return TIMESTAMP(timezone=True)
        return DateTime

    if col_type == "timedelta64":
        return BigInteger
    elif col_type == "floating":
        if col.dtype == "float32":
            return Float(precision=23)
        else:
            return Float(precision=53)
    elif col_type == "integer":
        if col.dtype.name.lower() in ("int8", "uint8", "int16"):
            return SmallInteger
        elif col.dtype.name.lower() in ("uint16", "int32"):
            return Integer
        elif col.dtype.name.lower() == "uint64":
            raise ValueError("Unsigned 64 bit integer datatype is not supported")
        else:
            return BigInteger
    elif col_type == "boolean":
        return Boolean
    elif col_type == "date":
        return Date
    elif col_type == "time":
        return Time
    elif col_type == "complex":
        raise ValueError("Complex datatypes not supported")

    return Text
