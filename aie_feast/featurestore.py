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
)
from dateutil.relativedelta import relativedelta


TIME_COL = "event_timestamp"


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
        """get from `start` to `end` statistical `fn` results of `entity_df` from `views`, only work for features varied with time

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
                        df = read_file(
                            os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
                            self.sources[cfg.batch_source].file_format,
                            self.sources[cfg.batch_source].event_time,
                        )
                        # filter columns
                        df = df[
                            [
                                fea
                                for fea, type in cfg.features.items()
                                if fea in features and type != "string"
                            ]
                            + [
                                self.sources[cfg.batch_source].event_time,
                                self.entity[entity_name].entity,
                            ]
                        ]
                        df.rename(
                            columns={
                                self.sources[cfg.batch_source].event_time: TIME_COL,
                                self.entity[entity_name].entity: entity_name,
                            },
                            inplace=True,
                        )
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

    def get_latest_entities(self, views):
        """get latest entity and its timestamp from `views`

        Args:
            views (List): _description_
        """
        result = {}
        views = get_consistent_format(views)
        for name, entity in self.entity.items():
            if self.connection.type == "file":
                dfs = []
                for view in views.values():
                    if name in view.entity and self.sources[view.batch_source].event_time:
                        df = read_file(
                            os.path.join(self.project_folder, self.sources[view.batch_source].file_path),
                            self.sources[view.batch_source].file_format,
                            self.sources[view.batch_source].event_time,
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
                df = read_file(
                    os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
                    self.sources[cfg.batch_source].file_format,
                    self.sources[cfg.batch_source].event_time,
                )
                # ensure the time col of result df
                df.rename(
                    columns={
                        self.sources[cfg.batch_source].event_time: TIME_COL,
                        self.entity[entity_df.columns[0]].entity: entity_name,
                    },
                    inplace=True,
                )
                # filter feature/label columns
                if is_label:
                    df = df[
                        [
                            col
                            for col in cfg.entity + list(cfg.labels.keys()) + [TIME_COL]
                            if col in df.columns
                        ]
                    ]
                else:
                    df = df[
                        [
                            col
                            for col in cfg.entity + list(cfg.features.keys()) + [TIME_COL]
                            if col in df.columns
                        ]
                    ]
                # merge according to `entity`
                df = df.merge(entity_df, on=entity_name, how="inner")
                # filter time condition
                if include:
                    fil = (
                        df[  # latest time
                            (df[TIME_COL + "_y"] >= df[TIME_COL + "_x"])
                            & (  # earliest time
                                df[TIME_COL + "_x"]
                                > df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                            )
                        ]
                        if cfg.ttl
                        else df[df[TIME_COL + "_y"] >= df[TIME_COL + "_x"]]  # latest time
                    )
                else:
                    fil = (
                        df[  # latest time
                            (df[TIME_COL + "_y"] > df[TIME_COL + "_x"])
                            & (  # earliest time
                                df[TIME_COL + "_x"]
                                >= df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                            )
                        ]
                        if cfg.ttl
                        else df[df[TIME_COL + "_y"] > df[TIME_COL + "_x"]]  # latest time
                    )
                    # newest record
                dfs.append(get_grouped_record(fil, TIME_COL, entity_name))
        return reduce(lambda l, r: pd.merge(l, r, on=[TIME_COL, entity_name], how="inner"), dfs)

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
        dfs = []
        for _, cfg in views.items():
            if entity_name in cfg.entity and self.sources[cfg.batch_source].event_time:
                # time column name in table
                df = read_file(
                    os.path.join(self.project_folder, self.sources[cfg.batch_source].file_path),
                    self.sources[cfg.batch_source].file_format,
                    self.sources[cfg.batch_source].event_time,
                )
                # ensure the time col of result df
                df.rename(
                    columns={
                        self.sources[cfg.batch_source].event_time: TIME_COL,
                        self.entity[entity_df.columns[0]].entity: entity_name,
                    },
                    inplace=True,
                )
                # filter feature/label columns
                if is_label:
                    df = df[[col for col in list(entity_df.columns) + list(cfg.labels.keys())]]
                else:
                    df = df[[col for col in list(entity_df.columns) + list(cfg.features.keys())]]

                df_for_period = df
                # merge according to `entity`
                df = df.merge(entity_df, on=entity_name, how="inner")
                # filter time condition
                if include:
                    fil = (
                        df[  # latest time
                            (df[TIME_COL + "_y"] >= df[TIME_COL + "_x"])
                            & (  # earliest time
                                df[TIME_COL + "_x"]
                                > df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                            )
                        ]
                        if cfg.ttl
                        else df[df[TIME_COL + "_y"] >= df[TIME_COL + "_x"]]  # latest time
                    )
                else:
                    fil = (
                        df[  # latest time
                            (df[TIME_COL + "_y"] > df[TIME_COL + "_x"])
                            & (  # earliest time
                                df[TIME_COL + "_x"]
                                >= df[TIME_COL + "_y"].map(lambda x: x - relativedelta(**parse_date(cfg.ttl)))
                            )
                        ]
                        if cfg.ttl
                        else df[df[TIME_COL + "_y"] > df[TIME_COL + "_x"]]  # latest time
                    )
                    # newest record
                newest_record = get_grouped_record(fil, TIME_COL, entity_name)
                # TODO 这个时间不对
                # get time window data
                df_for_period = df_for_period[
                    df_for_period[entity_name].apply(lambda x: x in newest_record[entity_name].values)
                ]

                df = self.get_period_record(
                    entity_name, TIME_COL, df_for_period, newest_record, period, is_label
                )

                df.sort_values(by=["query_timestamp", TIME_COL])

                dfs.append(df)

                # TODO 多个view返回的场景要修改一下
        return pd.concat(dfs) if len(dfs) > 0 else None

    def get_period_record(self, entity_name, TIME_COL, df_for_period, newest_record, period, is_label):
        period_df = []
        for entity_value in newest_record[entity_name]:
            if is_label:
                df = df_for_period[
                    (df_for_period[entity_name] == entity_value)
                    & (
                        df_for_period[TIME_COL]
                        < (
                            pd.to_datetime(
                                newest_record[newest_record[entity_name] == entity_value][TIME_COL].values[0]
                            )
                            + relativedelta(**parse_date(period))
                        ).tz_localize("utc")
                    )
                    & (
                        df_for_period[TIME_COL]
                        >= (
                            pd.to_datetime(
                                newest_record[newest_record[entity_name] == entity_value][TIME_COL].values[0]
                            )
                        ).tz_localize("utc")
                    )
                ].drop_duplicates()

            else:
                df = df_for_period[
                    (df_for_period[entity_name] == entity_value)
                    & (
                        df_for_period[TIME_COL]
                        > (
                            pd.to_datetime(
                                newest_record[newest_record[entity_name] == entity_value][TIME_COL].values[0]
                            )
                            - relativedelta(**parse_date(period))
                        ).tz_localize("utc")
                    )
                    & (
                        df_for_period[TIME_COL]
                        <= (
                            pd.to_datetime(
                                newest_record[newest_record[entity_name] == entity_value][TIME_COL].values[0]
                            )
                        ).tz_localize("utc")
                    )
                ].drop_duplicates()

            df["query_timestamp"] = pd.to_datetime(
                newest_record[newest_record[entity_name] == entity_value][TIME_COL].values[0]
            )

            period_df.append(df)

        return pd.concat(period_df)
