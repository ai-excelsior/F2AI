from datetime import datetime
from typing import List, Dict, cast
import pandas as pd
import os
from functools import reduce
from aie_feast.views import FeatureViews, LabelViews
from aie_feast.service import Service
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
    read_file,
    parse_date,
    get_newest_record,
    get_stats_result,
)
from common.psl_utils import execute_sql, psy_conn, to_pgsql, remove_table, close_conn, sql_df
from dateutil.relativedelta import relativedelta


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
        self.dataset = {}

    def __check_format(self, entity_df):
        if len(entity_df.columns) != 2 or entity_df.columns[1] != TIME_COL:
            raise ValueError(
                "Check entity_df make sure it has 2 columns and event_timestamp at the second column"
            )

    def __check_fns(self, fn):
        assert fn in [
            "mean",
            "sum",
            "std",
            "mode",
            "median",
            "min",
            "max",
        ], f"{fn}is not a available function, you can use fs.query() to customize your function"

    def _get_avaliable_features(self, view, check_type: bool = False):
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
                        [k for k, v in self.features[table].features.items() if v != "string"]
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
                        [k for k, v in self.labels[table].labels.items() if v != "string"]
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

    def _get_avaliable_entity(self, view, check_type: bool = False):
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
            raise TypeError("must be FeatureViews,LabelViews or Service")
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
            features = self._get_avaliable_features(feature_view)

        if self.connection.type == "file":
            return self._get_point_record(feature_view, entity_df, features, include)
        elif self.connection.type == "pgsql":
            conn = psy_conn(**self.connection.__dict__)

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
            features = self._get_avaliable_features(feature_view)

        if self.connection.type == "file":
            return self._get_period_record(feature_view, entity_df, period, features, include, is_label=False)

    def get_labels(self, label_views, entity_df: pd.DataFrame, include: bool = False):
        """non-time series prediction use: get labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        labels = self._get_available_labels(label_views)

        if self.connection.type == "file":
            return self._get_point_record(label_views, entity_df, labels, include)

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
        if not features:
            features = (
                self._get_avaliable_features(views, True)
                if isinstance(views, FeatureViews)
                else self._get_available_labels(views, True)
                if isinstance(views, LabelViews)
                else self._get_avaliable_features(views, True) + self._get_available_labels(views, True)
            )

        if entity_df is not None:
            self.__check_format(entity_df)
            entities = [entity_df.columns[0]]
            start = pd.to_datetime(0, utc=True)
        else:
            entities = group_key if group_key else self._get_avaliable_entity(views)
            end = end if end else pd.to_datetime(datetime.now(), utc=True)
            start = start if start else pd.to_datetime(0, utc=True)

        if self.connection.type == "file":
            all_entity_col = {self.entity[en].entity: en for en in entities}
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
            if entity_df is not None:
                df = df.merge(entity_df, how="right", on=entities)
            else:
                df.rename(columns={TIME_COL: TIME_COL + "_x"}, inplace=True)
                # end_time limit
                df = df.assign(**{TIME_COL + "_y": end})

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
            entity = self._get_avaliable_entity(view)

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
            # sort by event_time, decending
            df.sort_values(by=TIME_COL, ascending=False, inplace=True, ignore_index=True)
            # due to `ascending=False`, keep the `first` record means the latest one
            df = df.drop_duplicates(subset=entity, keep="first")
        return df

    def query(self, query: str = None):
        """customized query, for example, distinct

        Args:
            query (str, optional): _description_. Defaults to None.
        """
        pass

    def get_dataset(
        self,
        service,
        start: str = None,
        end: str = None,
        sampler: callable = None,
        bucket: int = None,
        stride: int = 1,
        include: str = "both",
    ) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            service: `SERVICE` to use
            start (str, optional): _description_. Defaults to None.
            end (str, optional): _description_. Defaults to None.
            sampler (callable, optional): _description_. Defaults to None.
            bucket (int, optional): time_bucket, Defaults to None means all in one bucket
            stride (int, optional): stride to sample, Defaults to 1 means no stride
            include(str,optional): whether to include `start` or `end` timestamp
        """

        service_entity: Service = service
        return Dataset(
            self, service_entity.features, service_entity.labels, start, end, sampler, bucket, stride, include
        )

    def _get_point_record(self, views, entity_df: pd.DataFrame, features: list = None, include: bool = True):
        """non time-series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews/Service to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            features(list):columns to select besides times and entities
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        entity = self._get_avaliable_entity(views)
        all_entity_col = {self.entity[en].entity: en for en in entity}
        entity_name = entity_df.columns[0]  # entity column name in table
        if isinstance(views, (FeatureViews, LabelViews)):  # read from single view
            df = self._read_local_file(views, features, all_entity_col)
            # rename entity columns
            df = df.merge(entity_df, on=entity_name, how="inner")
            if self.sources[views.batch_source].event_time:  #  time-relavent features
                # match time_limit
                df = self._fil_timelimit(include, views.ttl, df)
                # newest record
                df = get_newest_record(df, TIME_COL, [entity_name], CREATE_COL)
        else:  # `Service`, read from materialized table
            df = read_file(
                os.path.join(self.project_folder, views.materialize_path + ".parquet"),
                "parquet",
                [TIME_COL, MATERIALIZE_TIME],
                list(all_entity_col.values()),
            )
            df = df[[col for col in list(all_entity_col.values()) + features + [TIME_COL, MATERIALIZE_TIME]]]
            df = df.merge(entity_df, on=entity_name, how="inner")
            # match time_limit
            df = self._fil_timelimit(include, views.ttl, df)
            # newest record
            df = get_newest_record(df, TIME_COL, [entity_name], CREATE_COL)
        return df

    def _get_period_record(
        self,
        views,
        entity_df: pd.DataFrame,
        period: str,
        features: list = None,
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
        entity_name = entity_df.columns[0]  # entity column name in table
        entity = self._get_avaliable_entity(views)
        all_entity_col = {self.entity[en].entity: en for en in entity}

        if isinstance(views, (FeatureViews, LabelViews)):
            if self.sources[views.batch_source].event_time:  # period features make sense
                df_period = self._read_local_file(views, features, all_entity_col)
                # merge according to `entity`
                df_period = df_period.merge(entity_df, on=entity_name, how="inner")
                # # match time_limit
                df_period = self._get_time_window_record(df_period, period, is_label, include)
                if CREATE_COL in df_period.columns:  # use`create_timestamp` to remove duplicates
                    df_period.sort_values(by=[CREATE_COL], ascending=False, inplace=True, ignore_index=True)
                    df_period.drop_duplicates(subset=[entity_name, QUERY_COL, TIME_COL], keep="first")
                df_period.sort_values(by=[entity_name, QUERY_COL, TIME_COL], inplace=True, ignore_index=True)
            else:
                raise TypeError("View you give is not time-relevant, no period features/labels to get")
        else:
            df_period = read_file(
                os.path.join(self.project_folder, views.materialize_path + ".parquet"),
                "parquet",
                [TIME_COL, MATERIALIZE_TIME],
                list(all_entity_col.values()),
            )
            df_period = df_period[
                [col for col in list(all_entity_col.values()) + features + [TIME_COL, MATERIALIZE_TIME]]
            ]
            df_period = df_period.merge(entity_df, on=entity_name, how="inner")
            df_period = self._get_time_window_record(df_period, period, is_label, include)
            df_period.sort_values(by=[MATERIALIZE_TIME], ascending=False, inplace=True, ignore_index=True)
            df_period.drop_duplicates(subset=[entity_name, QUERY_COL, TIME_COL], keep="first")
            df_period.sort_values(by=[entity_name, QUERY_COL, TIME_COL], inplace=True, ignore_index=True)

    def _get_time_window_record(self, df_period, period, is_label, include):
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

    def materialize(self, service: Service):
        """incrementally join `views` to generate tables

        Args:
            views (List): _description_
        """
        if self.connection.type == "file":
            self._offline_file_materialize(service)

    def _offline_file_materialize(self, service: Service):
        """materialize offline file

        Args:
            service (Service): service entity
        """
        all_features_use = self._get_avaliable_features(service)
        for label_key in service.labels.keys():
            labels = self._get_available_labels(service)
            all_entities = self._get_avaliable_entity(service)
            all_entity_col = {self.entity[en].entity: en for en in all_entities}
            joined_frame = self._read_local_file(self.labels[label_key], labels, all_entity_col)
            joined_frame.drop(columns=[CREATE_COL], inplace=True)  # create timestamp makes no sense to labels
        # join features dataframe
        for feature_key in service.features.keys():
            feature_view: FeatureViews = self.features[feature_key]
            feature_cols = [
                item for item in all_features_use if item in self._get_avaliable_features(feature_view)
            ]
            fea_entities = self._get_avaliable_entity(feature_view)
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
        joined_frame.to_parquet(os.path.join(self.project_folder, f"{service.materialize_path}.parquet"))

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
