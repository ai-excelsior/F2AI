import pandas as pd
from pypika import Query, Parameter, Table
from pypika.queries import QueryBuilder
from common.connect import ConnectConfig
from common.psl_utils import psy_conn, sql_df, execute_sql


FSKEY = "__FS__"
TIME_COL = "event_timestamp"


def remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]


def get_default_value():
    return None


def schema_to_dict(schema):
    return {item["name"]: item.get("dtype", "string") for item in schema}


def service_to_dict(schema):
    item_dict = {}
    for item in schema:
        split_item = item.split(":")
        if len(split_item) < 2:
            raise ValueError("Please indicate features in table:feature format")
        elif len(split_item) > 3:
            raise ValueError("Please make sure colon not in name of table or features")
        else:
            split_item[1] = split_item[1] if split_item[1] != "*" else "__all__"
            if split_item[0] not in item_dict:
                item_dict.update(
                    {split_item[0]: [{split_item[1]: None if len(split_item) == 2 else split_item[2]}]}
                )
            else:
                item_dict[split_item[0]].append(
                    {split_item[1]: None if len(split_item) == 2 else split_item[2]}
                )

    return item_dict


def read_file(path, type, time_col=None, entity_cols=None):
    time_col = [i for i in time_col if i]
    path = remove_prefix(path, "file://")
    if type.startswith("parq"):
        df = pd.read_parquet(path)
    elif type.startswith("tsv"):
        df = pd.read_csv(path, sep="\t", parse_dates=time_col if time_col else [])
    elif type.startswith("txt"):
        df = pd.read_csv(path, sep=" ", parse_dates=time_col if time_col else [])
    else:
        df = pd.read_csv(path, parse_dates=time_col if time_col else [])
    for col in time_col:
        df[col] = pd.to_datetime(df[col], utc=True)
    if entity_cols:
        df[entity_cols] = df[entity_cols].astype("str")
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


def read_db(table_name: str, connection: ConnectConfig, time_col=None, entity_cols=None) -> pd.DataFrame:
    time_col = [i for i in time_col if i]
    if connection.type == "pgsql":
        conn = psy_conn(**connection.__dict__)
        q: QueryBuilder = Query.from_(table_name).select("*")
        cursor = execute_sql(q.get_sql(), conn)
        df = pd.DataFrame(
            cursor.fetchall(), columns=[c.name for c in cursor.description]
        )
        conn.close()
    for col in time_col:
        df[col] = pd.to_datetime(df[col], utc=True)
    if entity_cols:
        df[entity_cols] = df[entity_cols].astype("str")
    return df


def transform_freq(dt):
    value, freq = dt.split(" ")
    if freq == "quarters":
        freq = "months"
        value = int(value) * 3
    elif freq == "milliseconds":
        freq = "microseconds"
        value = int(value) * 10e3

    return {freq: int(value)}


def transform_pgsql_period(period, is_label: bool = False):
    value, freq = period.split(" ")
    if freq == "minutes":
        freq = "mins"
    elif freq == "quarters":
        freq = "months"
        value = int(value) * 3
    elif freq in ["millisecs", "milliseconds"]:
        freq = "secs"
        value = int(value) / 10e3
    elif freq in ["microsecs", "microseconds"]:
        freq = "secs"
        value = int(value) / 10e6
    return value + " " + freq if is_label else "-" + value + " " + freq


def parse_date(dt):
    return {**transform_freq(dt)} if dt else 0


def get_newest_record(df, time_col, entity_id, create_time):
    if len(df) == 0:
        return get_latest_record(df, time_col, create_time)
    return (
        df.groupby(entity_id + [time_col + "_y"])
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


def get_stats_result(data, fn, primary_keys, include, start):
    if include == "neither":
        return data[(data[TIME_COL + "_x"] < data[TIME_COL + "_y"]) & (data[TIME_COL + "_x"] > start)][
            [fea for fea in data.columns if fea not in primary_keys]
        ].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    elif include == "left":
        return data[(data[TIME_COL + "_x"] < data[TIME_COL + "_y"]) & (data[TIME_COL + "_x"] >= start)][
            [fea for fea in data.columns if fea not in primary_keys]
        ].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    elif include == "right":
        return data[(data[TIME_COL + "_x"] <= data[TIME_COL + "_y"]) & (data[TIME_COL + "_x"] > start)][
            [fea for fea in data.columns if fea not in primary_keys]
        ].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
    else:
        return data[(data[TIME_COL + "_x"] <= data[TIME_COL + "_y"]) & (data[TIME_COL + "_x"] >= start)][
            [fea for fea in data.columns if fea not in primary_keys]
        ].apply(lambda x: getattr(pd.Series, fn)(x), axis=0)
