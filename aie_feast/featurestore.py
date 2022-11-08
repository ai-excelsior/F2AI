from hologram import T
import pandas as pd
import os
import json
import docker
from datetime import datetime
from aie_feast.common.jinja import jinja_env
from typing import Dict, List, Union
from pypika import Query, Parameter
from aie_feast.common.source import FileSource, SqlSource
from aie_feast.offline_stores.offline_file_store import OfflineFileStore
from aie_feast.offline_stores.offline_postgres_store import OfflinePostgresStore
from aie_feast.views import FeatureView, LabelView
from aie_feast.service import Service
from aie_feast.dataset.dataset import Dataset
from aie_feast.common.get_config import (
    get_offline_store_from_cfg,
    get_service_cfg,
    get_entity_cfg,
    get_label_views,
    get_feature_views,
    get_source_cfg,
)
from aie_feast.common.utils import to_file, remove_prefix
from aie_feast.common.psl_utils import execute_sql, psy_conn, to_pgsql, close_conn, sql_df
from aie_feast.period import Period
from aie_feast.definitions import Feature


TIME_COL = "event_timestamp"  # timestamp of action taken in original tables or period-query result, or query time in single-query result table
CREATE_COL = "created_timestamp"  # timestamp of record of the action taken in original tables, or timestamp of action taken in single-query result table
QUERY_COL = "query_timestamp"  # only use in period query, query time in period-query result table
TMP_TBL = "entity_df"  # temp table upload in database
MATERIALIZE_TIME = "materialize_time"  # timestamp to done materialize, only used in materialized result
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            self.offline_store = get_offline_store_from_cfg(os.path.join(project_folder, "feature_store.yml"))
        elif url and token and projectID:
            pass  # TODO: realize in future
        else:
            raise ValueError("one of config file or meta server project should be provided")

        # init each object using .yml in corresponding folders
        self.project_folder = project_folder
        self.sources = get_source_cfg(os.path.join(project_folder, "sources"), self.offline_store.type)
        self.entities = get_entity_cfg(os.path.join(project_folder, "entities"))
        self.feature_views = get_feature_views(os.path.join(project_folder, "feature_views"))
        self.label_views = get_label_views(os.path.join(project_folder, "label_views"))
        self.services = get_service_cfg(os.path.join(project_folder, "services"))

        # for file source, modify the path if it is not a absolute path
        for _, source in self.sources.items():
            if isinstance(source, FileSource) and not os.path.isabs(source.path):
                source.path = os.path.join(self.project_folder, source.path)

        # fmodity the services' materialize path if it is not a absolute path
        for _, service in self.services.items():
            if service.materialize_type == "file" and not os.path.isabs(service.materialize_path):
                service.materialize_path = os.path.join(self.project_folder, service.materialize_path)

    def __check_format(self, entity_df):
        if isinstance(entity_df, pd.DataFrame):
            # TODO: Remove this constraint in future
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

    def _get_feature_to_use(self, views, features: list = [], is_numeric: bool = False) -> List[Feature]:
        """_summary_

        Args:
            views (_type_): FeatureView, LabelView or Service
            features (list, optional): feature(label) to use
            is_numeric (bool, optional): whether return numeric feature(label) only

        Returns:
            _type_: corresponding feature(label)
        """
        if isinstance(views, FeatureView):
            buildin_features = views.get_feature_objects(is_numeric)
        elif isinstance(views, LabelView):
            buildin_features = views.get_label_objects(is_numeric)
        else:  # Service
            buildin_features = views.get_feature_objects(
                self.feature_views, is_numeric
            ) | views.get_label_objects(self.label_views, is_numeric)

        if features:
            features = set(features)
            features = [feature for feature in buildin_features if feature.name in features]
        else:
            features = buildin_features
        return features

    def _get_keys_to_join(self, view, entity_columns=[]) -> list:
        """_summary_

        Args:
            view (_type_):  FeatureView, LabelView or Service
            entity_columns (list, optional): filter

        Returns:
            list: col name to join
        """

        if isinstance(view, FeatureView):
            avaliable_entity_names = list(view.entities)
        elif isinstance(view, LabelView):
            avaliable_entity_names = list(view.entities)
        elif isinstance(view, Service):
            avaliable_entity_names = list(view.get_entities(self.feature_views, self.label_views))
        else:
            raise TypeError("must be FeatureViews, LabelViews or Service")

        if entity_columns:  # need filter
            entity_names = list(
                {
                    join_key
                    for entity_name in avaliable_entity_names
                    for join_key in self.entities[entity_name].join_keys
                    if join_key in entity_columns
                }
            )
        else:
            entity_names = list(
                {
                    join_key
                    for entity_name in avaliable_entity_names
                    for join_key in self.entities[entity_name].join_keys
                }
            )

        return entity_names

    def _get_views(self, view_name) -> Union[FeatureView, LabelView, Service]:
        """_summary_

        Args:
            view_name (_type_): name of FeatureView, LabelView or Service

        Returns:
            _type_: FeatureView, LabelView or Service
        """

        if view_name in self.feature_views.keys():
            return self.feature_views[view_name]
        elif view_name in self.label_views.keys():
            return self.label_views[view_name]
        elif view_name in self.services.keys():
            return self.services[view_name]
        else:
            raise ValueError("Can't find any views/services")

    def get_features(
        self,
        feature_view: str,
        entity_df: Union[pd.DataFrame, str],
        features: list = None,
        include: bool = True,
        **kwargs,
    ):
        """non-series prediction use: get `features` of `entity_df` from `feature_views`

        Args:
            feature_view : Single FeatureViews or Service(after materialzed) name to lookup.
            entity_df (pd.DataFrame): condition.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional):  include timestamp defined in `entity_df` or not. Defaults to True.
        """

        self.__check_format(entity_df)
        feature_view = self._get_views(feature_view)
        assert isinstance(
            feature_view, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        feature_objects = self._get_feature_to_use(feature_view, features)
        join_keys = self._get_keys_to_join(feature_view, entity_df.columns)

        if isinstance(feature_view, (FeatureView, LabelView)):
            source = self.sources[feature_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(feature_view)

        return self.offline_store.get_features(
            entity_df=entity_df,
            features=feature_objects,
            source=source,
            join_keys=join_keys,
            ttl=feature_view.ttl,
            include=include,
            **kwargs,
        )

    def get_period_features(
        self,
        feature_view: str,
        entity_df: pd.DataFrame,
        period: str,
        features: List[str] = None,
        include: bool = True,
        **kwargs,
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
        feature_view = self._get_views(feature_view)
        assert isinstance(
            feature_view, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        period = -Period.from_str(period)
        feature_objects = self._get_feature_to_use(feature_view, features)
        join_keys = self._get_keys_to_join(feature_view, entity_df.columns)

        if isinstance(feature_view, (FeatureView, LabelView)):
            source = self.sources[feature_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(feature_view)

        return self.offline_store.get_period_features(
            entity_df=entity_df,
            features=feature_objects,
            source=source,
            period=period,
            join_keys=join_keys,
            ttl=feature_view.ttl,
            include=include,
            **kwargs,
        )

    def get_labels(self, label_view, entity_df: pd.DataFrame, include: bool = True, **kwargs):
        """non-time series prediction use: get labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) name to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        assert isinstance(label_view, (LabelView, Service)), "only allowed LabelView and Service"
        feature_objects = self._get_feature_to_use(label_view)
        join_keys = self._get_keys_to_join(label_view, entity_df.columns)

        if isinstance(label_view, (FeatureView, LabelView)):
            source = self.sources[label_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(label_view)

        return self.offline_store.get_features(
            entity_df=entity_df,
            features=feature_objects,
            source=source,
            join_keys=join_keys,
            ttl=label_view.ttl,
            include=include,
            **kwargs,
        )

    def get_period_labels(
        self, label_view: str, entity_df: pd.DataFrame, period: str, include: bool = False, **kwargs
    ):
        """time series prediction use: get from `start` to `end` length labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) name to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            period (str): length of look_forward, can be negative, egg, -1 days
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        period = Period.from_str(period)
        label_objects = self._get_feature_to_use(label_view)
        join_keys = self._get_keys_to_join(label_view, entity_df.columns)

        if isinstance(label_view, (FeatureView, LabelView)):
            source = self.sources[label_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(label_view)

        return self.offline_store.get_period_features(
            entity_df=entity_df,
            features=label_objects,
            source=source,
            period=period,
            join_keys=join_keys,
            ttl=label_view.ttl,
            include=include,
            **kwargs,
        )

    def materialize(self, service_name: str, incremental_begin: str = None):
        """incrementally join `views` to generate tables

        Args:
            service_name (str): name of service to materialize
            incremental_begin (str): begin of materialization
                `None`: all-data for type=file, otherwise last materialzation time
                date-like str: corresponding date, e.g.: `2020-01-03 00:09:08`
                int-like + fre str:  latest int freq, e.g.: `30 days`

        """

        if self.offline_store.type == "file":
            self._offline_record_materialize(self.services[service_name], incremental_begin)
        elif self.offline_store.type == "pgsql":
            self._offline_pgsql_materialize(self.services[service_name], incremental_begin)

    def _offline_pgsql_materialize(self, service: Service, incremental_begin):
        try:
            incremental_begin = pd.to_datetime(incremental_begin, utc=True) if incremental_begin else None
        except Exception:
            incremental_begin = Period.from_str(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        # dir to store dbt project
        label_view: LabelView = service.get_label_view(self.label_views)
        label_view_dict = label_view.dict()
        label_view_dict.update(
            {
                "labels": [label.name for label in label_view.get_label_objects()],
                "event_time": self.sources[label_view.batch_source].timestamp_field,
                "create_time": self.sources[label_view.batch_source].created_timestamp_field,
            }
        )

        all_features_use = [feature.name for feature in service.get_feature_objects(self.feature_views)]

        feature_views = []
        for feature_view in service.get_feature_views(self.feature_views):
            feature_view_dict = feature_view.dict()
            feature_view_dict.update(
                {
                    "features": [
                        feature_name
                        for feature_name in feature_view.get_feature_names()
                        if feature_name in all_features_use and feature_name not in label_view_dict["labels"]
                    ],
                    "event_time": self.sources[feature_view.batch_source].timestamp_field,
                    "create_time": self.sources[feature_view.batch_source].created_timestamp_field,
                }
            )
            feature_views.append(feature_view_dict)

        entity_names = list(service.get_entities(self.feature_views, self.label_views))
        entities_dict = {entity_name: self.entities[entity_name].join_keys[0] for entity_name in entity_names}

        conn = psy_conn(self.offline_store)

        max_timestamp, max_timestamp_label = self.offline_store.materialize(
            service=service,
            feature_views=self.feature_views,
            label_views=self.label_views,
            sources=self.sources,
            entities=self.entities,
            incremental_begin=incremental_begin,
        )

        label_result = pd.to_datetime(sql_df(max_timestamp_label.get_sql(), conn)[0][0])
        try:
            result = pd.to_datetime(sql_df(max_timestamp.get_sql(), conn)[0][0])
        except:
            result = pd.to_datetime("1970-01-01 00:00:00", utc=True)

        conn.close()

        if incremental_begin is None:
            incremental_begin = result.tz_localize(None)
        elif isinstance(incremental_begin, Period):
            incremental_begin = label_result - incremental_begin.to_py_timedelta()
        else:
            incremental_begin = incremental_begin

        dbt_path = os.path.join(
            remove_prefix(self.project_folder, "file://"),
            f"{service.dbt_path}",
            f"{service.materialize_path}",
        )
        pd.to_datetime
        dict_var = {
            "labelviews": label_view_dict,
            "featureviews": feature_views,
            "entities": entities_dict,
            "increment_begin": str(incremental_begin),
        }
        json_var = json.dumps(dict_var)
        self.schedule_local_dbt_container(service.materialize_path, json_var, dbt_path)
        # os.system(f"cd {dbt_path} && dbt run --profiles-dir {dbt_path} --vars '{json_var}' ")

    def schedule_local_dbt_container(self, profile_name: str, vars: Dict, dbt_path: str):
        docker_client = docker.from_env()
        dbt_profiles = jinja_env.get_template("profiles.yaml").render(
            profile=profile_name,
            host=self.offline_store.host,
            port=self.offline_store.port,
            user=self.offline_store.user,
            password=self.offline_store.password,
            database=self.offline_store.database,
            db_schema=self.offline_store.db_schema,
        )
        profile_path = os.path.join(dbt_path, "profiles.yml")
        if not os.path.exists(profile_path):
            with open(profile_path, "w") as f:
                f.write(dbt_profiles)
        docker_client.containers.run(
            "ghcr.io/dbt-labs/dbt-postgres:1.3.latest",
            command=f"run --vars '{vars}' ",
            volumes={
                dbt_path: {"bind": "/usr/app", "mode": "rw"},
                profile_path: {"bind": "/root/.dbt/profiles.yml", "mode": "rw"},
            },
            network="host",
            remove=True,
        )

    def _offline_record_materialize(self, service: Service, incremental_begin):
        """materialize offline file

        Args:
            service (Service): service entity
            incremental_begin: time to begin materialize
        """
        try:
            incremental_begin = pd.to_datetime(incremental_begin if incremental_begin else 0, utc=True)
        except Exception:
            incremental_begin = Period.from_str(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        joined_frame = self.offline_store.materialize(
            service=service,
            feature_views=self.feature_views,
            label_views=self.label_views,
            sources=self.sources,
            entities=self.entities,
            incremental_begin=incremental_begin,
        )
        joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.now(), utc=True)
        to_file(
            joined_frame,
            os.path.join(self.project_folder, f"{service.materialize_path}"),
            f"{service.materialize_path}".split(".")[-1],
        )
        print(
            f"materialize done, file saved at {os.path.join(self.project_folder, service.materialize_path)}"
        )

    def stats(
        self,
        view: str,
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
            views (List): name of view to look up
            entity_df (pd.DataFrame,optional), if given, ignore `start` and `end`. Defaults to None, has the supreme priority.
            group_key (list): joined-columns to do stats,  only works when `entity_df` is None. if None, means do stats on joined-entities, also accept `[]` means no grouping.
            fn (str, optional): statistical method, min, max, std, avg, mode, median. Defaults to "mean".
            start (str, optional): start_time. Defaults to None, works and only works when `entity_df` is None.
            end (str, optional): end_time. Defaults to None, works and only works when `entity_df` is None.
            include (str, optional): whether to include `start` or `end` timestamp
            keys_only (bool, optional): whether to take action on keys, only available when fn=unique, return a list
        """
        self.__check_fns(fn)
        view = self._get_views(view)

        if entity_df is not None:
            self.__check_format(entity_df)
            entities = entity_df.columns
            entity_df[TIME_COL] = pd.to_datetime(entity_df[TIME_COL], utc=True)
            start = pd.to_datetime(0, utc=True)
        else:
            # group_key can be empty list, which is different as None
            entities = group_key
            entity_df = pd.DataFrame(columns=[TIME_COL])
            entity_df[TIME_COL] = [
                pd.to_datetime(end, utc=True) if end else pd.to_datetime(datetime.now(), utc=True)
            ]
            start = pd.to_datetime(start, utc=True) if start else pd.to_datetime(0, utc=True)

        join_keys = self._get_keys_to_join(view, entities)
        features = self._get_feature_to_use(view, features, fn != "unique")

        if keys_only:
            assert fn == "unique", "keys_only=True can only be applied when fn=unique"
            assert join_keys, "no key available for keys_only=True"
            features = []

        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
            assert source.timestamp_field, "stats can only apply on time relative data"
        else:
            source = self.offline_store.get_offline_source(view)

        return self.offline_store.stats(
            entity_df=entity_df,
            features=features,
            source=source,
            fn=fn,
            start=start,
            group_keys=join_keys,
            include=include,
            keys_only=keys_only,
            join_keys=len(entity_df.columns[:-1]),
        )

    def get_latest_entities(self, view: str, entity: pd.DataFrame = None):
        """get latest entity and its timestamp from a single FeatureViews/LabelViews or a materialzed Service
        entity can either be None(all joined-entities in view), entity names or entity value(specific entities)

        Args:
            views (List): view to look up
        """
        view = self._get_views(view)
        if isinstance(entity, pd.DataFrame):
            entities = list(entity.columns)
            entity[TIME_COL] = 0
        else:
            entities = entity
            entity = None
        join_keys = self._get_keys_to_join(view, entities)

        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
            assert source.timestamp_field, "stats can only apply on time relative data"
        else:
            source = self.offline_store.get_offline_source(view)

        return self.offline_store.get_latest_entities(source=source, group_keys=join_keys, entity_df=entity)

    def get_dataset(self, service_name: str, sampler: callable = None) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            service_name(str): name of `SERVICE` to use
            sampler (callable, optional): sampler
        """
        return Dataset(
            fs=self.offline_store,
            service=self.services[service_name],
            sampler=sampler,
            project_folder=self.project_folder,
            feature_views=self.feature_views,
            label_views=self.label_views,
        )

    def query(self, query: str = None, return_df: bool = True):
        """customized query, only works when connection.type != 'file'

        Args:
            query (str, optional): full sql query to execute in db
        """
        assert (
            self.offline_store.type != "file"
        ), "query doesnt work for file type project, you can manualy read local files in pandas"
        if self.offline_store.type == "pgsql":
            conn = psy_conn(self.offline_store)
            if return_df:
                result = pd.DataFrame(sql_df(query, conn))
            else:
                execute_sql(query, conn)
                result = None
                conn.commit()
            close_conn(conn)
            return result
