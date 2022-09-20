from typing import Dict, Any, List, Union
import pandas as pd
import os
from dataset.dataset import Dataset
from common.utils import get_connection, get_forecast_cfg, get_entity_cfg
from .views import FeatureViews, LabelViews, Service


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            connect_cfg = project_folder + r"/feature_store.yml"
            get_connection(connect_cfg)
        elif url and token and projectID:
            pass  # TODO: realize in future
        else:
            raise ValueError("one of config file or meta server project should be provided")
        # init each object using .yml in corresponding folders
        self.features = {
            cfg: FeatureViews(cfg)
            for _, _, cfg in os.walk(project_folder + r"/feature_views")
            if cfg.endswith(".yml")
        }

        self.labels = {
            cfg: LabelViews(cfg)
            for _, _, cfg in os.walk(project_folder + r"/label_views")
            if cfg.endswith(".yml")
        }

        self.forecast_cfg = {
            cfg: get_forecast_cfg(cfg)
            for _, _, cfg in os.walk(project_folder + r"/services")
            if cfg.endswith(".yml")
        }
        self.entity = {
            cfg: get_forecast_cfg(cfg)
            for _, _, cfg in os.walk(project_folder + r"/entity")
            if cfg.endswith(".yml")
        }

    def get_features(self, feature_views, entity_df: pd.DataFrame, features: List = None):
        """non-series prediction use: get `features` of `entity_df` from `feature_views`

        Args:
            feature_views (List, optional): _description_. Defaults to [].
            entity_df (pd.DataFrame, optional): _description_. Defaults to None.
            features (List, optional): _description_. Defaults to None.
        """
        pass

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
        pass

    def get_labels(
        self,
        label_views,
        entity_df: pd.DataFrame,
        include: bool = False,
    ):
        """non-time series prediction use: get from `start` to `end` length labels of `entity_df` from `label_views`

        Args:
            label_views (List): _description_
            entity_df (pd.DataFrame): _description_
            start (str, optional): _description_. Defaults to None.
            end (str, optional): _description_. Defaults to None.
            include (bool, optional): _description_. Defaults to False, means not include timestamp defined in `entity_df`
        """
        pass

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
        pass

    def stats(
        self,
        views,
        entity_df: pd.DataFrame = None,
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

    def get_dataset(self, views, start: str = None, end: str = None, sampler: callable = None) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            views (List): _description_
            start (str, optional): _description_. Defaults to None.
            end (str, optional): _description_. Defaults to None.
            sampler (callable, optional): _description_. Defaults to None.
        """
        pass

    def materialize(self, views):
        """incrementally join `views` to generate tables

        Args:
            views (List): _description_
        """
