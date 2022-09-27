import pandas as pd


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


def read_file(path, type, time_col=None):
    path = remove_prefix(path, "file://")
    if type.startswith("parq"):
        df = pd.read_parquet(path)
    elif type.startswith("tsv"):
        df = pd.read_csv(path, sep="\t", parse_dates=[time_col] if time_col else [])
    elif type.startswith("txt"):
        df = pd.read_csv(path, sep=" ", parse_dates=[time_col] if time_col else [])
    else:
        df = pd.read_csv(path, parse_dates=[time_col] if time_col else [])
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    return df


def transform_freq(dt):
    value, freq = dt.split(" ")
    if freq == "quarters":
        freq = "months"
        value = int(value) * 3

    return {freq: int(value)}


def parse_date(dt):
    return {**transform_freq(dt)} if dt else None


def get_grouped_record(df, time_col, entity_id):
    return df.groupby(entity_id).apply(get_latest_record, time_col).reset_index(drop=True)


def get_latest_record(df, time_col):
    df = df[df[time_col + "_x"] == df[time_col + "_x"].max()]
    df.drop(columns=time_col + "_x", inplace=True)  # drop action timestamp
    df.rename(columns={time_col + "_y": time_col}, inplace=True)  # rename
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
