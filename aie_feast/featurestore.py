from datetime import datetime
from typing import List, Dict
import pandas as pd
import os
from functools import reduce

from aie_feast.views import FeatureViews

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
    FSKEY,
    read_file,
    parse_date,
    get_grouped_record,
    get_consistent_format,
    get_stats_result,
    get_period_grouped_record,
)
from common.psl_utils import psy_conn, to_pgsql, remove_table, close_conn
from dateutil.relativedelta import relativedelta


TIME_COL = "event_timestamp"
CREATE_COL = "created_timestamp"
<<<<<<< HEAD
QUERY_COL = "query_timestamp"
=======
TMP_TBL = "entity_df"
>>>>>>> 01f38eb (add a little pgsql relevent)


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
        # self.service = get_service_cfg(os.path.join(project_folder, "services"))

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

    def get_features(
        self, feature_views, entity_df: pd.DataFrame, features: List = None, include: bool = True
    ):
        """non-series prediction use: get `features` of `entity_df` from `feature_views`

        Args:
            feature_views : FeatureViews to lookup. Defaults to [].
            entity_df (pd.DataFrame): condition. Defaults to None.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional):  include timestamp defined in `entity_df` or not. Defaults to True.
        """
        self.__check_format(entity_df)
        feature_views = get_consistent_format(feature_views)
        if self.connection.type == "file":
            return (
                self._get_point_record(feature_views, entity_df, include, is_label=False)
                if not features
                else self._get_point_record(feature_views, entity_df, include, is_label=False)[
                    [TIME_COL, entity_df.columns[0]] + features
                ]
            )
        elif self.connection.type == "pgsql":
            conn = psy_conn(**self.connection.__dict__)
            # upload `entity_df` to database
            to_pgsql(entity_df, TMP_TBL, **self.connection.__dict__)

            # remove table entity_df
            remove_table(TMP_TBL, conn)
            conn.close()

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
            feature_views:
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

    def get_labels(
        self,
        label_views,
        entity_df: pd.DataFrame,
        include: bool = False,
    ):
        """non-time series prediction use: get labels of `entity_df` from `label_views`

        Args:
            label_views:
            entity_df (pd.DataFrame): condition
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_views = get_consistent_format(label_views)
        if self.connection.type == "file":
            return self._get_point_record(label_views, entity_df, include, is_label=True)

    def get_period_labels(
        self,
        label_views,
        entity_df: pd.DataFrame,
        period: str,
        include: bool = False,
    ):
        """time series prediction use: get from `start` to `end` length labels of `entity_df` from `label_views`

        Args:
            label_views:
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
        group: bool = True,
        fn: str = "mean",
        start: str = None,
        end: str = None,
        include: str = "both",
    ):
        """get from `start` to `end` statistical `fn` results of `entity_df` from `views`, only work for numeric features varied with time

        Args:
            views (List): _description_
            entity_df (pd.DataFrame,optional), if given, ignore `start` and `end`. Defaults to None.
            group (bool, optional): whether to group according to `entity_df`. Defaults to True.
            fn (str, optional): statistical method, min, max, std, avg, mode,median. Defaults to "mean".
            start (str, optional): start_time. Defaults to None.
            end (str, optional): end_time. Defaults to None.
            include(str,optional): whether to include `start` or `end` timestamp
        """
        views = get_consistent_format(views)
        self.__check_fns(fn)
        features = (
            features
            if features
            else reduce(
                lambda a, b: a + b,
                [
                    list(view.features.keys()) if isinstance(view, FeatureViews) else list(view.labels.keys())
                    for view in views.values()
                ],
            )
        )
        dict_data = {}

        if entity_df is not None:
            self.__check_format(entity_df)
            entities = [entity_df.columns[0]]
            start = pd.to_datetime(0, utc=True)
        else:
            entities = self.entity.keys()
            end = end if end else pd.to_datetime(datetime.now(), utc=True)
            start = start if start else pd.to_datetime(0, utc=True)

        if self.connection.type == "file":
            for entity_name in entities:
                dfs = []
                for _, cfg in views.items():
                    if entity_name in cfg.entity and self.sources[cfg.batch_source].event_time:
                        all_entity_col = {self.entity[en].entity: en for en in cfg.entity}
                        df = read_file(
                            os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
                            self.sources[cfg.batch_source].file_format,
                            self.sources[cfg.batch_source].event_time,
                            list(all_entity_col.keys()),
                        )
                        df.rename(columns={self.sources[cfg.batch_source].event_time: TIME_COL}, inplace=True)
                        # filter columns
                        df = df[
                            [
                                col
                                for col in [
                                    fea
                                    for fea, type in cfg.features.items()
                                    if fea in features and type != "string"
                                ]
                                + list(all_entity_col.keys())
                                + [TIME_COL]
                                if col in df.columns
                            ]
                        ]
                        df.rename(columns=all_entity_col, inplace=True)
                        if entity_df is not None:
                            df = df.merge(entity_df, how="right", on=entity_name)
                        else:
                            df.rename(
                                columns={TIME_COL: TIME_COL + "_x"},
                                inplace=True,
                            )
                            # end_time limit
                            df = df.assign(**{TIME_COL + "_y": end})
                        dfs.append(df)
                    if dfs:
                        dict_data.update({entity_name: dfs})
            result = {}
            for fea, datas in dict_data.items():
                dfs = []
                for data in datas:
                    if not group:  # create a virtual group
                        data[fea] = fea
                    dfs.append(
                        data.groupby(fea).apply(
                            get_stats_result,
                            fn,
                            primary_keys=[fea, TIME_COL + "_x", TIME_COL + "_y"],
                            include=include,
                            start=start,
                        )
                    )
                if dfs:
                    result.update({fea: reduce(lambda l, r: pd.merge(l, r, on=[fea], how="outer"), dfs)})
            return result

    def get_latest_entities(self, views, entity: List[str] = None):
        """get latest entity and its timestamp from `views`

        Args:
            views (List): _description_
        """
        result = {}
        views = get_consistent_format(views)
        entities = {en: self.entity[en] for en in entity} if entity else self.entity
        for name, entity in entities.items():
            if self.connection.type == "file":
                dfs = []
                for view in views.values():
                    if name in view.entity and self.sources[view.batch_source].event_time:
                        all_entity_col = {self.entity[en].entity: en for en in view.entity}
                        df = read_file(
                            os.path.join(self.project_folder, self.sources[view.batch_source].file_path),
                            self.sources[view.batch_source].file_format,
                            self.sources[view.batch_source].event_time,
                            list(all_entity_col.keys()),
                        )
                        df = df[[entity.entity, self.sources[view.batch_source].event_time]]
                        # sort by event_time, decending
                        df.sort_values(
                            by=self.sources[view.batch_source].event_time,
                            ascending=False,
                            inplace=True,
                            ignore_index=True,
                        )
                        # due to `ascending=False`, keep the `first` record means the latest one
                        en = df.drop_duplicates(subset=entity.entity, keep="first")
                        en.columns = [name, TIME_COL]
                        dfs.append(en)
                if dfs:  # views have this entity
                    dfs = reduce(lambda l, r: pd.merge(l, r, on=name, how="outer"), dfs)
                    # .astype() to avoid nan caused by outer-merge
                    dfs[TIME_COL] = dfs[[col for col in dfs.columns if col != name]].apply(
                        lambda row: row.astype(en[TIME_COL].dtype).max(), axis=1
                    )
                    result.update({name: dfs[[name, TIME_COL]]})
        return result

    def query(self, query: str = None):
        """customized query

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
        pass

    def materialize(self, views):
        """incrementally join `views` to generate tables

        Args:
            views (List): _description_
        """

    def _get_point_record(self, views, entity_df: pd.DataFrame, include: bool = True, is_label: bool = False):
        """non-series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
            is_label (bool, optional): LabelViews of not. Defaults to False.
        """
        entity_name = entity_df.columns[0]  # entity column name in table
        dfs = []
        for _, cfg in views.items():
            if entity_name in cfg.entity:
                # time column name in table
                all_entity_col = {self.entity[en].entity: en for en in cfg.entity}
                df = read_file(
                    os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
                    self.sources[cfg.batch_source].file_format,
                    [self.sources[cfg.batch_source].event_time, self.sources[cfg.batch_source].create_time],
                    list(all_entity_col.keys()),
                )
                # ensure the time col of result df
                df.rename(
                    columns={
                        self.sources[cfg.batch_source].event_time: TIME_COL,
                        self.sources[cfg.batch_source].create_time: CREATE_COL,
                    },
                    inplace=True,
                )
                # filter feature/label columns
                if is_label:
                    df = df[
                        [
                            col
                            for col in list(all_entity_col.keys())
                            + list(cfg.labels.keys())
                            + [TIME_COL, CREATE_COL]
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
                # merge according to `entity`
                df = df.merge(entity_df, on=entity_name, how="inner")
                # filter time condition
                if self.sources[cfg.batch_source].event_time:  #  time-relavent features
                    fil = self._fil_timelimit(include, cfg, df)
                    # newest record
                    dfs.append(get_grouped_record(fil, TIME_COL, entity_name, CREATE_COL))
                else:  #  not time-relavent features
                    dfs.append(df)
        return reduce(lambda l, r: pd.merge(l, r, on=[TIME_COL, entity_name], how="inner"), dfs)

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
                all_entity_col = {self.entity[en].entity: en for en in cfg.entity}
                dfs = []
                # time column name in table
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
                            for col in list(all_entity_col.keys())
                            + list(cfg.labels.keys())
                            + [TIME_COL, CREATE_COL]
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

                df_for_period = df
                # merge according to `entity`
                df = df.merge(entity_df, on=entity_name, how="inner")
                # filter time condition

                fil = self._fil_timelimit(include, cfg, df)

                newest_record = get_period_grouped_record(fil, TIME_COL, entity_name, CREATE_COL)

                df_for_period = pd.merge(
                    df_for_period, newest_record[entity_name].drop_duplicates(), on=entity_name, how="inner"
                )

                df = self.get_period_record(
                    entity_name, TIME_COL, df_for_period, newest_record, period, is_label
                )

                df.sort_values(by=[entity_name, QUERY_COL, TIME_COL], inplace=True)
                df.drop_duplicates()
                df.reset_index(inplace=True, drop=True)
                result.update({name: df})
        return result

    def get_period_record(self, entity_name, TIME_COL, df_for_period, newest_record, period, is_label):
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
