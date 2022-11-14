import oss2
import os
import pandas as pd
from typing import List, Tuple


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
