from datetime import datetime
from typing import Dict, List, Union
import pandas as pd
import os
from pypika import Query, Tables, Parameter
from aie_feast.views import FeatureViews, LabelViews
from aie_feast.service import Service
from common.source import SourceConfig
from dataset.dataset import Dataset
from common.get_config import (
    get_conn_cfg,
    get_service_cfg,
    get_entity_cfg,
    get_label_views,
    get_feature_views,
    get_source_cfg,
)
from common.utils import (
    read_db,
    read_file,
    to_file,
    parse_date,
    get_newest_record,
    get_stats_result,
    transform_pgsql_period,
)
from common.psl_utils import execute_sql, psy_conn, to_pgsql, remove_table, close_conn, sql_df
from dateutil.relativedelta import relativedelta

pd.DataFrame.to_csv

TIME_COL = "event_timestamp"  # timestamp of action taken in original tables or period-query result, or query time in single-query result table
CREATE_COL = "created_timestamp"  # timestamp of record of the action taken in original tables, or timestamp of action taken in single-query result table
QUERY_COL = "query_timestamp"  # only use in period query, query time in period-query result table
TMP_TBL = "entity_df"  # temp table upload in database
MATERIALIZE_TIME = "materialize_time"  # timestamp to done materialize, only used in materialized result


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            self.connection = get_conn_cfg(os.path.join(project_folder, "feature_store.yml"))
        elif url and token and projectID:
            pass  # TODO: realize in future
        else:
            raise ValueError("one of config file or meta server project should be provided")
        # init each object using .yml in corresponding folders
        self.project_folder = project_folder
        self.sources = get_source_cfg(os.path.join(project_folder, "sources"))
        self.entity = get_entity_cfg(os.path.join(project_folder, "entities"))
        self.features = get_feature_views(os.path.join(project_folder, "feature_views"))
        self.labels = get_label_views(os.path.join(project_folder, "label_views"))
        self.service = get_service_cfg(os.path.join(project_folder, "services"))

    def __check_format(self, entity_df):
        assert (
            len(entity_df.columns) >= 1 and entity_df.columns[-1] == TIME_COL
        ), "Check entity_df make sure it has at least 1 columns and event_timestamp at the last column"

    def __check_fns(self, fn):
        assert fn in [
            "mean",
            "sum",
            "std",
            "mode",
            "median",
            "min",
            "max",
            "unique",
        ], f"{fn}is not a available function, you can use fs.query() to customize your function"

    def _get_available_features(self, view, check_type: bool = False):
        if isinstance(view, FeatureViews):
            features = (
                [k for k, v in view.features.items() if v not in ["string", "bool"]]
                if check_type
                else list(view.features.keys())
            )
        elif isinstance(view, Service):  # Services
            features = []
            for table, cols in view.features.items():
                if len(cols) == 1 and "__all__" in cols[0].keys():
                    features += (
                        [k for k, v in self.features[table].features.items() if v not in ["string", "bool"]]
                        if check_type
                        else list(self.features[table].features.keys())
                    )
                else:
                    for col in cols:
                        features += (
                            [
                                k
                                for k, _ in col.items()
                                if self.features[table].features[k] not in ["string", "bool"]
                            ]
                            if check_type
                            else list(col.keys())
                        )

        else:
            raise TypeError("must be FeatureViews or Service")
        return features

    def _get_available_labels(self, view, check_type: bool = False):
        if isinstance(view, LabelViews):
            labels = (
                [k for k, v in view.labels.items() if v not in ["string", "bool"]]
                if check_type
                else list(view.labels.keys())
            )
        elif isinstance(view, Service):  # Services
            labels = []
            for table, cols in view.labels.items():
                if len(cols) == 1 and "__all__" in cols[0].keys():
                    labels += (
                        [k for k, v in self.labels[table].labels.items() if v not in ["string", "bool"]]
                        if check_type
                        else list(self.labels[table].labels.keys())
                    )
                else:
                    for col in cols:
                        labels += (
                            [
                                k
                                for k, _ in col.items()
                                if self.labels[table].labels[k] not in ["string", "bool"]
                            ]
                            if check_type
                            else list(col.keys())
                        )
        else:
            raise TypeError("must be LabelViews or Service")
        return labels

    def _get_available_entity(self, view):
        entity = []
        if isinstance(view, (FeatureViews, LabelViews)):
            entity = list(view.entity)
        elif isinstance(view, Service):  # Services
            for table, _ in view.features.items():
                entity += self.features[table].entity
            for table, _ in view.labels.items():
                entity += self.labels[table].entity
            entity = list(set(entity))
        else:
            raise TypeError("must be FeatureViews, LabelViews or Service")
        return entity

    def get_features(
        self, feature_view, entity_df: pd.DataFrame, features: list = None, include: bool = True
    ):
        """non-series prediction use: get `features` of `entity_df` from `feature_views`

        Args:
            feature_view : Single FeatureViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition. Defaults to None.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional):  include timestamp defined in `entity_df` or not. Defaults to True.
        """
        self.__check_format(entity_df)
        if not features:
            features = self._get_available_features(feature_view)

        if self.connection.type == "file":
            return self._get_point_record(feature_view, entity_df, features, include)
        elif self.connection.type == "pgsql":
            return self._get_point_pgsql(feature_view, entity_df, features, include)

    def get_period_features(
        self,
        feature_view,
        entity_df: pd.DataFrame,
        period: str,
        features: List = None,
        include: bool = True,
    ):
        """time_series prediction use: get past `period` length `features` of `entity_df` from `feature_views`

        Args:
            feature_views:Single FeatureViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            period (str): length of look_back
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        self.__check_format(entity_df)
        if not features:
            features = self._get_available_features(feature_view)

        if self.connection.type == "file":
            return self._get_period_record(feature_view, entity_df, period, features, include, is_label=False)
        elif self.connection.type == "pgsql":
            return self._get_period_pgsql(feature_view, entity_df, period, features, include, is_label=False)

    def get_labels(self, label_view, entity_df: pd.DataFrame, include: bool = False):
        """non-time series prediction use: get labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        labels = self._get_available_labels(label_view)

        if self.connection.type == "file":
            return self._get_point_record(label_view, entity_df, labels, include)
        elif self.connection.type == "pgsql":
            return self._get_point_pgsql(label_view, entity_df, labels, include, is_label=True)

    def get_period_labels(
        self,
        label_view,
        entity_df: pd.DataFrame,
        period: str,
        include: bool = False,
    ):
        """time series prediction use: get from `start` to `end` length labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            period (str): length of look_forward
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        labels = self._get_available_labels(label_view)

        if self.connection.type == "file":
            return self._get_period_record(label_view, entity_df, period, labels, include, is_label=True)
        elif self.connection.type == "pgsql":
            return self._get_period_pgsql(label_view, entity_df, period, labels, include, is_label=False)

    def _get_point_record(self, views, entity_df: pd.DataFrame, features: list, include: bool = True):
        """non time-series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews/Service to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            features(list):columns to select besides times and entities
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        entity = self._get_available_entity(views)
        all_entity_col = {self.entity[en].entity: en for en in entity if en in list(entity_df.columns[:-1])}
        entity_name = list(all_entity_col.values())  # entity column name in table

        if isinstance(views, (FeatureViews, LabelViews)):  # read from single view
            df = self._read_local_file(views, features, all_entity_col)
            # rename entity columns
            if all_entity_col:
                df = df.merge(entity_df[entity_name + [TIME_COL]], on=entity_name, how="inner")
            else:
                df = df.merge(entity_df, how="cross")
            if self.sources[views.batch_source].event_time:  #  time-relavent features
                # match time_limit
                df = self._fil_timelimit(include, views.ttl, df)
                # newest record
                df = get_newest_record(df, TIME_COL, entity_name, CREATE_COL)
        else:  # `Service`, read from materialized table
            df = read_file(
                os.path.join(self.project_folder, views.materialize_path),
                "parquet",
                [TIME_COL, MATERIALIZE_TIME],
                list(all_entity_col.values()),
            )
            df = df[[col for col in list(all_entity_col.values()) + features + [TIME_COL, MATERIALIZE_TIME]]]
            if all_entity_col:
                df = df.merge(entity_df, on=entity_name, how="inner")
            else:
                df = df.merge(entity_df, how="cross")
            # match time_limit
            df = self._fil_timelimit(include, None, df)
            # newest record
            df = get_newest_record(df, TIME_COL, entity_name, CREATE_COL)
        return df

    def _get_point_pgsql(
        self, views, entity_df: pd.DataFrame, features: list, include: bool = True, is_label: bool = False
    ):
        entity = self._get_available_entity(views)
        entity_name = [en for en in entity if en in list(entity_df.columns[:-1])]
        all_entity_col = [self.entity[en].entity + " as " + en for en in entity_name]
        # connect to pgsql db
        conn = psy_conn(**self.connection.__dict__)
        to_pgsql(entity_df, TMP_TBL, **self.connection.__dict__)

        if isinstance(views, (FeatureViews, LabelViews)):
            assert self.sources[views.batch_source].event_time, "View is not time-relevant, no period to get"
            entity_df, df = Tables(f"{TMP_TBL}", f"{views.batch_source}")
            all_time_col = (
                [
                    f"{self.sources[views.batch_source].event_time} as {TIME_COL}_tmp",
                    f"{self.sources[views.batch_source].create_time} as {CREATE_COL}",
                ]
                if self.sources[views.batch_source].create_time
                else [f"{self.sources[views.batch_source].event_time} as  {TIME_COL}_tmp"]
            )
            create_time = CREATE_COL
        else:
            entity_df, df = Tables(f"{TMP_TBL}", f"{views.materialize_path}")
            all_time_col = [f"{TIME_COL} as {TIME_COL}_tmp", MATERIALIZE_TIME]
            create_time = MATERIALIZE_TIME
        df = Query.from_(df).select(Parameter(",".join(all_entity_col + all_time_col + features))).as_("df")
        if all_entity_col:
            sql_join = (
                Query.from_(entity_df)
                .inner_join(df)
                .using(",".join(entity_name))
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        else:
            sql_join = (
                Query.from_(entity_df)
                .cross_join(df)
                .cross()
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        sql_query = sql_join
        if self.sources[views.batch_source].event_time and views.ttl:
            sql_query = self._get_window_pgsql(sql_join, views.ttl, include, is_label)
        sql_result = (
            Query.from_(sql_query).select(
                Parameter(
                    ",".join(
                        entity_name
                        + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                        + features
                    )
                ),
            )
            if len(all_time_col) == 1
            else (
                Query.from_(
                    Query.from_(sql_query).select(
                        sql_query.star,
                        Parameter(
                            f"row_number() over (partition by ({','.join(entity_name)},{TIME_COL},{TIME_COL}_tmp) order by {create_time} DESC)"
                        ),
                    )
                )
                .select(
                    Parameter(
                        ",".join(
                            entity_name
                            + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                            + features
                        )
                    )
                )
                .where(Parameter("row_number=1"))
            )
        )
        result = pd.DataFrame(
            sql_df(sql_result.get_sql(), conn), columns=entity_name + [QUERY_COL, TIME_COL] + features
        )
        # remove entity_df and close connection
        remove_table(TMP_TBL, conn)
        close_conn(conn)
        return result

    def _get_period_pgsql(
        self,
        views,
        entity_df: pd.DataFrame,
        period: str,
        features: list,
        include: bool = True,
        is_label: bool = False,
    ):
        entity = self._get_available_entity(views)
        entity_name = [en for en in entity if en in list(entity_df.columns[:-1])]
        all_entity_col = [self.entity[en].entity + " as " + en for en in entity_name]
        # connect to pgsql db
        conn = psy_conn(**self.connection.__dict__)
        to_pgsql(entity_df, TMP_TBL, **self.connection.__dict__)
        period = transform_pgsql_period(period, is_label)

        if isinstance(views, (FeatureViews, LabelViews)):
            assert self.sources[views.batch_source].event_time, "View is not time-relevant, no period to get"
            entity_df, df = Tables(f"{TMP_TBL}", f"{views.batch_source}")
            all_time_col = (
                [
                    f"{self.sources[views.batch_source].event_time} as {TIME_COL}_tmp",
                    f"{self.sources[views.batch_source].create_time} as {CREATE_COL}",
                ]
                if self.sources[views.batch_source].create_time
                else [f"{self.sources[views.batch_source].event_time} as  {TIME_COL}_tmp"]
            )
            create_time = CREATE_COL
        else:
            entity_df, df = Tables(f"{TMP_TBL}", f"{views.materialize_path}")
            all_time_col = [f"{TIME_COL} as {TIME_COL}_tmp", MATERIALIZE_TIME]
            create_time = MATERIALIZE_TIME
        df = Query.from_(df).select(Parameter(",".join(all_entity_col + all_time_col + features))).as_("df")
        if all_entity_col:
            sql_join = (
                Query.from_(Query.from_(entity_df).select(",".join(entity_name), TIME_COL))
                .inner_join(df)
                .using(",".join(entity_name))
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        else:
            sql_join = (
                Query.from_(entity_df)
                .cross_join(df)
                .cross()
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        sql_query = self._get_window_pgsql(sql_join, period, include, is_label)
        sql_result = (
            Query.from_(sql_query).select(
                Parameter(
                    ",".join(
                        entity_name
                        + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                        + features
                    )
                ),
            )
            if len(all_time_col) == 1
            else (
                Query.from_(
                    Query.from_(sql_query).select(
                        sql_query.star,
                        Parameter(
                            f"row_number() over (partition by ({','.join(entity_name)},{TIME_COL},{TIME_COL}_tmp) order by {create_time} DESC)"
                        ),
                    )
                )
                .select(
                    Parameter(
                        ",".join(
                            entity_name
                            + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                            + features
                        )
                    )
                )
                .where(Parameter("row_number=1"))
            )
        )
        result = pd.DataFrame(
            sql_df(sql_result.get_sql(), conn), columns=entity_name + [QUERY_COL, TIME_COL] + features
        )
        # remove entity_df and close connection
        remove_table(TMP_TBL, conn)
        close_conn(conn)
        return result

    def _get_period_record(
        self,
        views,
        entity_df: pd.DataFrame,
        period: str,
        features: list,
        include: bool = True,
        is_label: bool = False,
    ):

        """series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            period: period.
            features:features/labels to return
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
            is_label (bool, optional): LabelViews of not. Defaults to False.
        """

        entity = self._get_available_entity(views)
        all_entity_col = {self.entity[en].entity: en for en in entity if en in list(entity_df.columns[:-1])}
        entity_name = list(all_entity_col.values())  # entity column name in table
        if isinstance(views, (FeatureViews, LabelViews)):
            assert self.sources[views.batch_source].event_time, "View is not time-relevant, no period to get"
            df_period = self._read_local_file(views, features, all_entity_col)
            # merge according to `entity`
            if all_entity_col:
                df_period = df_period.merge(entity_df[entity_name + [TIME_COL]], on=entity_name, how="inner")
            else:
                df_period = df_period.merge(entity_df, how="cross")
            # # match time_limit
            df_period = self._get_window_record(df_period, period, is_label, include)
            if CREATE_COL in df_period.columns:  # use`create_timestamp` to remove duplicates
                df_period.sort_values(by=[CREATE_COL], ascending=False, inplace=True, ignore_index=True)
                df_period.drop_duplicates(subset=entity_name + [QUERY_COL, TIME_COL], keep="first")
            # df_period.sort_values(by=entity_name + [QUERY_COL, TIME_COL], inplace=True, ignore_index=True)
        else:
            df_period = read_file(
                os.path.join(self.project_folder, views.materialize_path),
                "parquet",
                [TIME_COL, MATERIALIZE_TIME],
                list(all_entity_col.values()),
            )
            df_period = df_period[
                [col for col in list(all_entity_col.values()) + features + [TIME_COL, MATERIALIZE_TIME]]
            ]
            if all_entity_col:
                df_period = df_period.merge(entity_df, on=entity_name, how="inner")
            else:
                df_period = df_period.merge(entity_df, how="cross")
            df_period = self._get_window_record(df_period, period, is_label, include)
            df_period.sort_values(by=[MATERIALIZE_TIME], ascending=False, inplace=True, ignore_index=True)
            df_period.drop_duplicates(subset=entity_name + [QUERY_COL, TIME_COL], keep="first")
        # df_period.sort_values(by=entity_name + [QUERY_COL, TIME_COL], inplace=True, ignore_index=True)
        return df_period

    def _get_window_pgsql(
        self,
        join,
        period: str,
        include: bool = True,
        is_label: bool = False,
    ):
        if is_label:  # forward
            sql_query = (
                Query.from_(join)
                .select(join.star)
                .where(
                    Parameter(
                        f" ({TIME_COL}_tmp::timestamp >= {TIME_COL}::timestamp) and ({TIME_COL}_tmp::timestamp < {TIME_COL}::timestamp + '{period}')  "
                    )
                    if include
                    else Parameter(
                        f" ({TIME_COL}_tmp::timestamp > {TIME_COL}::timestamp) and ({TIME_COL}_tmp::timestamp <= {TIME_COL}::timestamp + '{period}')  "
                    )
                )
                .as_("sql_query")
            )
        else:  # backward
            sql_query = (
                Query.from_(join)
                .select(join.star)
                .where(
                    Parameter(
                        f" ({TIME_COL}_tmp::timestamp <= {TIME_COL}::timestamp) and ({TIME_COL}_tmp::timestamp > {TIME_COL}::timestamp + '{period}')  "
                    )
                    if include
                    else Parameter(
                        f" ({TIME_COL}_tmp::timestamp < {TIME_COL}::timestamp) and ({TIME_COL}_tmp::timestamp >= {TIME_COL}::timestamp + '{period}')  "
                    )
                )
                .as_("sql_query")
            )

        return sql_query

    def _get_window_record(self, df_period, period, is_label, include):
        if is_label:
            if include:
                df_period = df_period[  # latest time
                    (df_period[TIME_COL + "_y"] <= df_period[TIME_COL + "_x"])
                    & (  # earliest time
                        df_period[TIME_COL + "_x"]
                        < df_period[TIME_COL + "_y"].map(lambda x: x + relativedelta(**parse_date(period)))
                    )
                ]
            else:
                df_period = df_period[  # latest time
                    (df_period[TIME_COL + "_y"] < df_period[TIME_COL + "_x"])
                    & (  # earliest time
                        df_period[TIME_COL + "_x"]
                        <= df_period[TIME_COL + "_y"].map(lambda x: x + relativedelta(**parse_date(period)))
                    )
                ]
        else:
            df_period = self._fil_timelimit(include, period, df_period)
        # rename action timestamp based on labelviews
        df_period.rename(columns={TIME_COL + "_x": TIME_COL}, inplace=True)
        # time defined in `entity_df`
        df_period.rename(columns={TIME_COL + "_y": QUERY_COL}, inplace=True)
        return df_period

    def materialize(self, service_name: str, incremental_begin: str = None):
        """incrementally join `views` to generate tables

        Args:
            service_name (str): name of service to materialize
            incremental_begin (str): begin of materialization
                `None`: all-data for type=file, otherwise last materialzation time
                date-like str: corresponding date, e.g.: `2020-01-03 00:09:08`
                int-like + fre str:  latest int freq, e.g.: `30 days`

        """

        if self.connection.type == "file":
            self._offline_record_materialize(self.service[service_name], incremental_begin)
        elif self.connection.type == "pgsql":
            self._offline_pgsql_materialize(self.service[service_name], incremental_begin)

    def _offline_pgsql_materialize(self, service, incremental_begin):
        try:
            incremental_begin = pd.to_datetime(incremental_begin, utc=True) if incremental_begin else None
        except Exception:
            incremental_begin = parse_date(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        # dir to store dbt project
        dbt_path = os.path.join(self.project_folder.lstrip("file://"), f"{service.dbt_path}")
        os.system(
            f"cd {dbt_path} && dbt run --profiles-dir {dbt_path} --vars {{begin_point:{incremental_begin} }}"
        )

    def _offline_record_materialize(self, service: Service, incremental_begin):
        """materialize offline file

        Args:
            service (Service): service entity
        """
        try:
            incremental_begin = pd.to_datetime(incremental_begin if incremental_begin else 0, utc=True)
        except Exception:
            incremental_begin = parse_date(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        all_features_use = self._get_available_features(service)
        for label_key in service.labels.keys():
            labels = self._get_available_labels(service)
            all_entities = self._get_available_entity(service)
            all_entity_col = {self.entity[en].entity: en for en in all_entities}
            joined_frame = self._read_local_file(self.labels[label_key], labels, all_entity_col)
            joined_frame.drop(columns=[CREATE_COL], inplace=True)  # create timestamp makes no sense to labels
            if isinstance(incremental_begin, dict):
                incremental_begin = joined_frame[TIME_COL].max() - relativedelta(**incremental_begin)
                joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]
            else:
                joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]
        # join features dataframe
        for feature_key in service.features.keys():
            feature_view: FeatureViews = self.features[feature_key]
            feature_cols = [
                item for item in all_features_use if item in self._get_available_features(feature_view)
            ]
            fea_entities = self._get_available_entity(feature_view)
            entity_col = {self.entity[en].entity: en for en in fea_entities}
            tmp_fea = self._read_local_file(feature_view, feature_cols, entity_col)
            joined_frame = tmp_fea.merge(joined_frame, how="right", on=fea_entities)
            if self.sources[feature_view.batch_source].event_time:  # time relevant features
                # filter feature timestamp <= label timestamp
                joined_frame = self._fil_timelimit(include=True, ttl=feature_view.ttl, df=joined_frame)
                # get the latest record for each label time after filter feature timestamp <= label timestamp
                joined_frame = get_newest_record(joined_frame, TIME_COL, fea_entities, CREATE_COL)
                # feature timestamp makes no use to result
                joined_frame.drop(columns=[CREATE_COL], inplace=True)

        joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.now(), utc=True)
        to_file(
            joined_frame,
            os.path.join(self.project_folder, f"{service.materialize_path}"),
            f"{service.materialize_path}".split(".")[-1],
        )

    def _read_local_file(self, views, features, all_entity_col):
        df = df = read_file(
            os.path.join(self.project_folder, self.sources[views.batch_source].file_path),
            self.sources[views.batch_source].file_format,
            [
                self.sources[views.batch_source].event_time,
                self.sources[views.batch_source].create_time,
            ],
            list(all_entity_col.keys()),
        )
        df.rename(
            columns={
                self.sources[views.batch_source].event_time: TIME_COL,
                self.sources[views.batch_source].create_time: CREATE_COL,
            },
            inplace=True,
        )
        df.rename(columns=all_entity_col, inplace=True)
        df = df[
            [
                col
                for col in list(all_entity_col.values()) + features + [TIME_COL, CREATE_COL]
                if col in df.columns
            ]
        ]
        return df

    def _read_local_db_data(
        self, views: Union[FeatureViews, LabelViews], features: List, all_entity_col: Dict[str, str]
    ) -> pd.DataFrame:
        source: SourceConfig = self.sources[views.batch_source]
        df = read_db(
            source.name,
            self.connection,
            [
                source.event_time,
                source.create_time,
            ],
            list(all_entity_col.keys()),
        )
        df.rename(
            columns={
                source.event_time: TIME_COL,
                source.create_time: CREATE_COL,
            },
            inplace=True,
        )
        df.rename(columns=all_entity_col, inplace=True)
        df = df[
            [
                col
                for col in list(all_entity_col.values()) + features + [TIME_COL, CREATE_COL]
                if col in df.columns
            ]
        ]
        return df

    def _fil_timelimit(self, include, ttl, df):
        if include:
            return (
                df[  # latest time
                    (df[TIME_COL + "_y"] >= df[TIME_COL + "_x"])
                    & (  # earliest time
                        df[TIME_COL + "_x"]
                        > df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(ttl)))
                    )
                ]
                if ttl
                else df[df[TIME_COL + "_y"] >= df[TIME_COL + "_x"]]
            )  # latest time

        else:
            return (
                df[  # latest time
                    (df[TIME_COL + "_y"] > df[TIME_COL + "_x"])
                    & (  # earliest time
                        df[TIME_COL + "_x"]
                        >= df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(ttl)))
                    )
                ]
                if ttl
                else df[df[TIME_COL + "_y"] > df[TIME_COL + "_x"]]
            )  # latest time

    def stats(
        self,
        views,
        entity_df: pd.DataFrame = None,
        features: List[str] = None,
        group_key: List[str] = None,
        fn: str = "mean",
        start: str = None,
        end: str = None,
        include: str = "both",
        keys_only: bool = False,
    ):
        """get from `start` to `end` statistical `fn` results of `entity_df` from `views`, only work for numeric features varied with time

        Args:
            views (List): _description_
            entity_df (pd.DataFrame,optional), if given, ignore `start` and `end`. Defaults to None, has the supreme priority.
            group_key (list): joined-columns to do stats,  only works when `entity_df` is None, if None, means do stats on joined-entities.
            fn (str, optional): statistical method, min, max, std, avg, mode, median. Defaults to "mean".
            start (str, optional): start_time. Defaults to None, works and only works when `entity_df` is None.
            end (str, optional): end_time. Defaults to None, works and only works when `entity_df` is None.
            include(str,optional): whether to include `start` or `end` timestamp
        """
        self.__check_fns(fn)
        check_type = True if fn != "unique" else False
        if not features:
            features = (
                self._get_available_features(views, check_type)
                if isinstance(views, FeatureViews)
                else self._get_available_labels(views, check_type)
                if isinstance(views, LabelViews)
                else self._get_available_features(views, check_type)
                + self._get_available_labels(views, check_type)
            )

        if entity_df is not None:
            self.__check_format(entity_df)
            entities = entity_df.columns[:-1]
            start = pd.to_datetime(0, utc=True)
        else:
            entities = group_key if group_key else self._get_available_entity(views)
            end = end if end else pd.to_datetime(datetime.now(), utc=True)
            start = start if start else pd.to_datetime(0, utc=True)
        all_entity_col = {self.entity[en].entity: en for en in entities}
        if self.connection.type == "file":
            if isinstance(views, (FeatureViews, LabelViews)):
                df = self._read_local_file(views, features, all_entity_col)
            else:
                df = read_file(
                    os.path.join(self.project_folder, views.materialize_path + ".parquet"),
                    "parquet",
                    [TIME_COL],
                    list(all_entity_col.values()),
                )
                df = df[[col for col in features + list(all_entity_col.values()) + [TIME_COL]]]
        elif self.connection.type == "pgsql":
            if isinstance(views, (FeatureViews, LabelViews)):
                df = self._read_local_db_data(views, features, all_entity_col)
            else:
                df = read_db(
                    views.materialize_path,
                    self.connection,
                    [TIME_COL],
                    list(all_entity_col.values()),
                )
                df = df[[col for col in features + list(all_entity_col.values()) + [TIME_COL]]]

        if entity_df is not None and entities:
            df = df.merge(entity_df, how="right", on=entities)
        elif entity_df is not None:
            df = df.merge(entity_df, how="cross")
        else:
            df.rename(columns={TIME_COL: TIME_COL + "_x"}, inplace=True)
            # end_time limit
            df = df.assign(**{TIME_COL + "_y": end})
        if keys_only:
            assert fn == "unique", "keys_only=True can only be applied when fn==unique"
            result = list(
                df[(df[TIME_COL + "_x"] < df[TIME_COL + "_y"]) & (df[TIME_COL + "_x"] >= start)]
                .groupby(entities)
                .groups.keys()
            )
        else:
            result = df.groupby(entities).apply(
                get_stats_result,
                fn,
                primary_keys=entities + [TIME_COL + "_x", TIME_COL + "_y"],
                include=include,
                start=start,
            )
        return result

    def get_latest_entities(self, view, entity: List[str] = []):
        """get latest entity and its timestamp from a single FeatureViews/LabelViews or a materialzed Service

        Args:
            views (List): _description_
        """
        if not entity:
            entity = self._get_available_entity(view)

        all_entity_col = {self.entity[en].entity: en for en in entity}
        if self.connection.type == "file":
            if isinstance(view, (FeatureViews, LabelViews)):
                df = self._read_local_file(view, [], all_entity_col)
            else:
                df = read_file(
                    os.path.join(self.project_folder, view.materialize_path + ".parquet"),
                    "parquet",
                    [TIME_COL],
                    list(all_entity_col.values()),
                )

        elif self.connection.type == "pgsql":
            if isinstance(view, (FeatureViews, LabelViews)):
                df = self._read_local_db_data(view, [], all_entity_col)
            else:
                df = read_db(
                    view.materialize_path,
                    self.connection,
                    [TIME_COL],
                    list(all_entity_col.values()),
                )
        # sort by event_time, decending
        df.sort_values(by=TIME_COL, ascending=False, inplace=True, ignore_index=True)
        # due to `ascending=False`, keep the `first` record means the latest one
        df = df.drop_duplicates(subset=entity, keep="first")
        return df

    def get_dataset(
        self,
        service_name: str,
        sampler: callable = None,
    ) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            service_name(str): name of `SERVICE` to use
            sampler (callable, optional): sampler
        """
        return Dataset(
            fs=self,
            service_name=service_name,
            sampler=sampler,
        )

    def query(self, query: str = None):
        """customized query, for example, distinct

        Args:
            query (str, optional): _description_. Defaults to None.
        """
        conn = psy_conn(**self.connection.__dict__)
        return sql_df(query, conn)
