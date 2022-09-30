from datetime import datetime
from tokenize import group
from typing import List, Dict, cast
import pandas as pd
import os
from functools import reduce
from aie_feast.entity import Entity
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
    read_file,
    parse_date,
    get_newest_record,
    get_consistent_format,
    get_stats_result,
    get_period_grouped_record,
)
from common.psl_utils import execute_sql, psy_conn, to_pgsql, remove_table, close_conn, sql_df
from dateutil.relativedelta import relativedelta


TIME_COL = "event_timestamp"
CREATE_COL = "created_timestamp"  # use and only used when have multiple identical event_timestamp value
QUERY_COL = "query_timestamp"  # use in period query
TMP_TBL = "entity_df"  # temp table upload in database
MATERIALIZE_TIME = "materialize_time"  # time to done materialize


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

    def get_features(self, feature_view, entity_df: pd.DataFrame, features: list = [], include: bool = True):
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
            features = (
                features
                if features
                else reduce(
                    lambda a, b: a + b,
                    [
                        list(view.features.keys())
                        if isinstance(view, FeatureViews)
                        else list(view.labels.keys())
                        for view in feature_view.values()
                    ],
                )
            )
            # upload `entity_df` to database
            to_pgsql(entity_df, TMP_TBL, **self.connection.__dict__)
            entity_name = entity_df.columns[0]  # entity column name in table
            views_to_use = {name: view for name, view in feature_view.items() if entity_name in view.entity}
            sqls = []
            for view_name, cfg in views_to_use.items():
                if entity_name in cfg.entity:
                    # time column name in table
                    ent_select = [self.entity[en].entity + " as " + en for en in cfg.entity]
                    fea_select = list(cfg.features.keys())
                    if (
                        not self.sources[cfg.batch_source].event_time
                        and not self.sources[cfg.batch_source].create_time
                    ):  # non time relevant features and is unique for entity
                        sql = f"(SELECT a.{TIME_COL} ,b.* FROM {TMP_TBL} a LEFT JOIN (SELECT {','.join(fea_select + ent_select)} FROM {cfg.batch_source}) b using({entity_name})) as {view_name}"
                    elif not self.sources[cfg.batch_source].create_time:
                        # time relevant features and has no redundency
                        sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] + fea_select )} from \
                                (SELECT *,row_number() over (partition by c.{entity_name} order by c.{TIME_COL}_b  DESC ) as row_id from \
                                    (SELECT a.{TIME_COL} ,b.* FROM \
                                        {TMP_TBL} a LEFT JOIN \
                                        (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].event_time} as {TIME_COL}_b \
                                        FROM {cfg.batch_source}) b using ({entity_name}) \
                                    where cast(a.{TIME_COL} as date) >= cast(b.{TIME_COL}_b as date)) c \
                                )tmp where tmp.row_id=1) as {view_name}"
                    elif not self.sources[cfg.batch_source].event_time:
                        # non time relevant features but may have redundency:
                        sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] + fea_select)} from \
                                (SELECT *,row_number() over (partition by c.{entity_name} order by c.{CREATE_COL}  DESC ) as row_id from \
                                    (SELECT a.{TIME_COL},b.* FROM \
                                        {TMP_TBL} a LEFT JOIN \
                                        (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].create_time} as {CREATE_COL} FROM {cfg.batch_source}) b on using ({entity_name}) ) c \
                                ) tmp where tmp.row_id=1) as {view_name}"

                    else:
                        sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] +fea_select)} from \
                                (SELECT *,row_number() over (partition by c.{entity_name} order by c.{CREATE_COL} DESC, c.{TIME_COL}_b DESC ) as row_id from \
                                    (SELECT a.{TIME_COL} ,b.* FROM \
                                        {TMP_TBL} a LEFT JOIN \
                                        (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].event_time} as {TIME_COL}_b,{self.sources[cfg.batch_source].create_time} as {CREATE_COL} \
                                        FROM {cfg.batch_source}) b on a.{entity_name}=b.{entity_name} \
                                    where cast(a.{TIME_COL} as date) >= cast(b.{TIME_COL}_b as date)) c \
                                )tmp where tmp.row_id=1) as {view_name}"
                    sqls.append(sql)
            final_sql = "SELECT * FROM " + reduce(
                lambda a, b: f"{a} join {b} using ({entity_name},{TIME_COL})", sqls
            )
            result = pd.DataFrame(sql_df(final_sql, conn))
            result

    def _get_avaliable_features(self, view, check_type: bool = False):
        if isinstance(view, FeatureViews):
            features = [k for k, v in view.features.items() if v not in ["string", "bool"]]
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
            labels = [k for k, v in view.labels.items() if v not in ["string", "bool"]]
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

    def get_period_features(
        self,
        feature_views,
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
        feature_views = get_consistent_format(feature_views)
        if self.connection.type == "file":
            return (
                self._get_period_record(feature_views, entity_df, period, include, is_label=False)
                if not features
                else self._get_period_record(feature_views, entity_df, period, include, is_label=False)[
                    [TIME_COL, entity_df.columns[0]] + features
                ]
            )

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
        label_views,
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
        label_views = get_consistent_format(label_views)
        if self.connection.type == "file":
            return self._get_period_record(label_views, entity_df, period, include, is_label=True)

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
                df = read_file(
                    os.path.join(self.project_folder, self.sources[views.batch_source].file_path),
                    self.sources[views.batch_source].file_format,
                    [self.sources[views.batch_source].event_time],
                    list(all_entity_col.keys()),
                )
                df.rename(columns={self.sources[views.batch_source].event_time: TIME_COL}, inplace=True)
                df.rename(columns=all_entity_col, inplace=True)
                # filter columns
            else:
                df = read_file(
                    os.path.join(self.project_folder, views.materialize_path + ".parquet"),
                    "parquet",
                    [TIME_COL],
                    list(all_entity_col.values()),
                )

            df = df[[col for col in [features] + list(all_entity_col.values()) + [TIME_COL]]]
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
                df = read_file(
                    os.path.join(self.project_folder, self.sources[view.batch_source].file_path),
                    self.sources[view.batch_source].file_format,
                    [self.sources[view.batch_source].event_time],
                    list(all_entity_col.keys()),
                )
                df.rename(columns={self.sources[view.batch_source].event_time: TIME_COL}, inplace=True)
                df.rename(columns=all_entity_col, inplace=True)
            else:
                df = read_file(
                    os.path.join(self.project_folder, view.materialize_path + ".parquet"),
                    "parquet",
                    [TIME_COL],
                    list(all_entity_col.values()),
                )
            df = df[list(all_entity_col.values()) + [TIME_COL]]
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

        service_entity: Service = self.service["service"]
        return Dataset(
            self, service_entity.features, service_entity.labels, start, end, sampler, bucket, stride, include
        )

    def materialize(self, service: Service):
        """incrementally join `views` to generate tables

        Args:
            views (List): _description_
        """
        if self.connection.type == "file":
            self.offline_file_materialize(service)

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
            df = read_file(
                os.path.join(self.project_folder, self.sources[views.batch_source].file_path),
                self.sources[views.batch_source].file_format,
                [
                    self.sources[views.batch_source].event_time,
                    self.sources[views.batch_source].create_time,
                ],
                list(all_entity_col.keys()),
            )
            # rename time columns
            df.rename(
                columns={
                    self.sources[views.batch_source].event_time: TIME_COL,
                    self.sources[views.batch_source].create_time: CREATE_COL,
                },
                inplace=True,
            )
            df.rename(columns=all_entity_col, inplace=True)
            df = df[[col for col in list(all_entity_col.values()) + features + [TIME_COL, CREATE_COL]]]
            # rename entity columns
            df = df.merge(entity_df, on=entity_name, how="inner")
            if self.sources[views.batch_source].event_time:  #  time-relavent features
                # match time_limit
                df = self._fil_timelimit(include, views, df)
                # newest record
                df = get_newest_record(df, TIME_COL, entity_name, CREATE_COL)
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
            df = self._fil_timelimit(include, views, df)
            # newest record
            df = get_newest_record(df, TIME_COL, entity_name, CREATE_COL)
        return df

    def _fil_timelimit(self, include, cfg, df):
        if include:
            return (
                df[  # latest time
                    (df[TIME_COL + "_y"] >= df[TIME_COL + "_x"])
                    & (  # earliest time
                        df[TIME_COL + "_x"]
                        > df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                    )
                ]
                if cfg.ttl
                else df[df[TIME_COL + "_y"] >= df[TIME_COL + "_x"]]
            )  # latest time

        else:
            return (
                df[  # latest time
                    (df[TIME_COL + "_y"] > df[TIME_COL + "_x"])
                    & (  # earliest time
                        df[TIME_COL + "_x"]
                        >= df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                    )
                ]
                if cfg.ttl
                else df[df[TIME_COL + "_y"] > df[TIME_COL + "_x"]]
            )  # latest time

    def _get_period_record(
        self, views, entity_df: pd.DataFrame, period: str, include: bool = True, is_label: bool = False
    ):

        """series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            period: period.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
            is_label (bool, optional): LabelViews of not. Defaults to False.
        """
        entity_name = entity_df.columns[0]  # entity column name in table

        result = {}
        for name, cfg in views.items():
            if entity_name in cfg.entity and self.sources[cfg.batch_source].event_time:
                df = self._read_and_filter(cfg, is_label)
                df_for_period = df
                # merge according to `entity`
                df = df.merge(entity_df, on=entity_name, how="inner")
                # filter time condition
                fil = self._fil_timelimit(include, cfg, df)
                newest_record = get_period_grouped_record(fil, TIME_COL, entity_name, CREATE_COL)
                df_for_period = pd.merge(
                    df_for_period, newest_record[entity_name].drop_duplicates(), on=entity_name, how="inner"
                )
                df = self._get_time_window_record(
                    entity_name, TIME_COL, df_for_period, newest_record, period, is_label
                )
                df.sort_values(by=[entity_name, QUERY_COL, TIME_COL], inplace=True)
                df.drop_duplicates()
                df.reset_index(inplace=True, drop=True)
                result.update({name: df})
        return result

    def _get_time_window_record(self, entity_name, TIME_COL, df_for_period, newest_record, period, is_label):
        period_df = []
        for info in newest_record[[entity_name, TIME_COL]].values:
            if is_label:
                df = df_for_period[
                    (df_for_period[entity_name] == info[0])
                    & (df_for_period[TIME_COL] < (info[1] + relativedelta(**parse_date(period))))
                    & (df_for_period[TIME_COL] >= info[1])
                ]
            else:
                df = df_for_period[
                    (df_for_period[entity_name] == info[0])
                    & (df_for_period[TIME_COL] > (info[1] - relativedelta(**parse_date(period))))
                    & (df_for_period[TIME_COL] <= info[1])
                ]

            df[QUERY_COL] = info[1]
            period_df.append(df)
        return pd.concat(period_df)

    def _read_and_filter(self, cfg, is_label):
        all_entity_col = {self.entity[en].entity: en for en in cfg.entity}
        df = read_file(
            os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
            self.sources[cfg.batch_source].file_format,
            [self.sources[cfg.batch_source].event_time, self.sources[cfg.batch_source].create_time],
            list(all_entity_col.keys()),
        )
        df.rename(
            columns={
                self.sources[cfg.batch_source].event_time: TIME_COL,
                self.sources[cfg.batch_source].create_time: CREATE_COL,
            },
            inplace=True,
        )

        if is_label:
            df = df[
                [
                    col
                    for col in list(all_entity_col.keys()) + list(cfg.labels.keys()) + [TIME_COL, CREATE_COL]
                    if col in df.columns
                ]
            ]
        else:
            df = df[
                [
                    col
                    for col in list(all_entity_col.keys())
                    + list(cfg.features.keys())
                    + [TIME_COL, CREATE_COL]
                    if col in df.columns
                ]
            ]
        df.rename(columns=all_entity_col, inplace=True)
        return df

    def offline_file_materialize(self, service: Service):
        """materialize offline file

        Args:
            service (Service): service entity
        """
        # get feature views
        feature_views = service.features
        # get label views
        label_views = service.labels
        # get label dataframe
        joined_frame = pd.DataFrame()
        for label_key, _ in label_views.items():
            label_view: LabelViews = self.labels[label_key]
            batch_source: SourceConfig = self.sources[label_view.batch_source]
            file_path = batch_source.file_path
            dataframe = pd.read_parquet(os.path.join(self.project_folder, file_path))
            dataframe.rename(columns={
                cast(Entity, self.entity[entity_key]).entity: entity_key for entity_key in label_view.entity
            }, inplace=True)
            joined_frame = pd.concat(
                [
                    joined_frame,
                    dataframe[
                        label_view.entity
                        + list(label_view.labels.keys())
                        + [batch_source.event_time, batch_source.create_time]
                    ],
                ]
            )

        # join features dataframe
        for feature_key, feature_col_maps in feature_views.items():
            feature_view: FeatureViews = self.features[feature_key]
            cols = list(map(lambda i: next(iter(i)), feature_col_maps))
            feature_cols = list(feature_view.features.keys()) if "__all__" in cols else cols
            batch_source: SourceConfig = self.sources[feature_view.batch_source]
            event_time_field: str = batch_source.event_time
            create_time_field: str = batch_source.create_time
            file_path = batch_source.file_path
            dataframe = pd.read_parquet(os.path.join(self.project_folder, file_path))
            dataframe.rename(columns={
                cast(Entity, self.entity[entity_key]).entity: entity_key for entity_key in feature_view.entity
            }, inplace=True)
            dataframe = dataframe[feature_view.entity + feature_cols + [event_time_field, create_time_field]]
            joined_frame = pd.merge(
                joined_frame,
                dataframe,
                how="left",
                on=feature_view.entity,
                suffixes=(None, f"_{feature_key}"),
            )
            # clean duplicate rows by created_timestamp
            duplicated_rows = joined_frame[
                joined_frame[[event_time_field, f"{event_time_field}_{feature_key}"]].duplicated(keep=False)
            ]
            duplicated_max_create_time = duplicated_rows.groupby(
                [event_time_field, f"{event_time_field}_{feature_key}"]
            )[f"{create_time_field}_{feature_key}"].max()
            should_drop_index = duplicated_rows[
                ~(
                    duplicated_rows[f"{create_time_field}_{feature_key}"].isin(
                        duplicated_max_create_time.to_list()
                    )
                )
            ].index
            joined_frame = joined_frame.drop(should_drop_index)

        # filter
        columns: List[str] = joined_frame.columns.to_list()
        if "event_timestamp" in columns:
            event_timestamp_temp_cols = [
                column for column in columns if column.startswith("event_timestamp_")
            ]
            joined_frame = joined_frame.drop(
                joined_frame[
                    reduce(
                        lambda x, y: x | y,
                        [
                            (
                                joined_frame["event_timestamp"].astype("datetime64[ns, UTC]")
                                < joined_frame[col].astype("datetime64[ns, UTC]")
                            )
                            for col in event_timestamp_temp_cols
                        ],
                    )
                ].index
            )

        # save
        joined_frame.to_parquet(os.path.join(self.project_folder, f"{service.materialize_path}" if service.materialize_path else f"{service}_offline.parquet"))
