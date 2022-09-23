from typing import List
import pandas as pd
import os
from functools import reduce
from dataset.dataset import Dataset
from common.get_config import (
    get_conn_cfg,
    get_service_cfg,
    get_entity_cfg,
    get_label_views,
    get_feature_views,
    get_source_cfg,
)
from common.utils import read_file, parse_date, get_grouped_record, get_consistent_format
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
        self.service = get_service_cfg(os.path.join(project_folder, "services"))

    def __check_format(self, entity_df):
        if len(entity_df.columns) != 2 or entity_df.columns[1] != TIME_COL:
            raise ValueError(
                "Check entity_df make sure it has 2 columns and event_timestamp at the second column"
            )

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
            feature_views (List): _description_
            entity_df (pd.DataFrame): _description_
            period (str): length of look_back
            features (List, optional): _description_. Defaults to None.
            include (bool, optional): _description_. Defaults to True, means include timestamp defined in `entity_df`
        """
        self.__check_format(entity_df)
        pass

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
            label_views (List): _description_
            entity_df (pd.DataFrame): _description_
            period (str): length of look_forward
            include (bool, optional): _description_. Defaults to False, means not include timestamp defined in `entity_df`
        """
        self.__check_format(entity_df)
        pass

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
        """get from `start` to `end` statistical `fn` results of `entity_df` from `views`

        Args:
            views (List): _description_
            entity_df (pd.DataFrame,optional), if given, ignore `start` and `end`. Defaults to None.
            group (bool, optional): whether to group according to `entity_df`. Defaults to True.
            fn (str, optional): statistical method, min, max, std, avg, mode,median. Defaults to "mean".
            start (str, optional): start_time. Defaults to None.
            end (str, optional): end_time. Defaults to None.
            include(str,optional): whether to include `start` or `end` timestamp
        """
        pass

    def get_latest_entities(self, views):
        """get latest entity and its timestamp

        Args:
            views (List): _description_
        """
        pass

    def query(self, query: str = None):
        """customized query

        Args:
            query (str, optional): _description_. Defaults to None.
        """
        pass

    def get_dataset(
        self,
        views,
        start: str = None,
        end: str = None,
        sampler: callable = None,
        stride: int = 1,
        include: str = "both",
    ) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            views (List): _description_
            start (str, optional): _description_. Defaults to None.
            end (str, optional): _description_. Defaults to None.
            sampler (callable, optional): _description_. Defaults to None.
            stride (int, optional): stride to sample, Defaults to 1
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
                    df = df[[col for col in list(entity_df.columns) + list(cfg.labels.keys())]]
                else:
                    df = df[[col for col in list(entity_df.columns) + list(cfg.features.keys())]]
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
