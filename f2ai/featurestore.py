import json
import os
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pipe, Pool
from typing import Dict, List, Optional, Union

import docker
import pandas as pd
from tqdm import tqdm

from f2ai.common.get_config import (
    get_entity_cfg,
    get_feature_views,
    get_label_views,
    get_service_cfg,
    get_source_cfg,
)
from f2ai.common.jinja import jinja_env
from f2ai.common.read_file import read_yml
from f2ai.common.utils import remove_prefix
from f2ai.dataset.dataset import Dataset
from f2ai.definitions import (
    BackoffTime,
    Feature,
    FeatureView,
    FileSource,
    LabelView,
    Period,
    Service,
    StatsFunctions,
    backoff_to_split,
    init_offline_store_from_cfg,
    init_online_store_from_cfg,
    init_persist_engine_from_cfg,
    OnlineStore,
)
from f2ai.online_stores.online_redis_store import OnlineRedisStore

TIME_COL = "event_timestamp"  # timestamp of action taken in original tables or period-query result, or query time in single-query result table
MATERIALIZE_TIME = "materialize_time"  # timestamp to done materialize, only used in materialized result


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            cfg = read_yml(os.path.join(project_folder, "feature_store.yml"))
            self.offline_store = init_offline_store_from_cfg(cfg["offline_store"])
            self.online_store = init_online_store_from_cfg(cfg["online_store"])
            self.persist_engine = init_persist_engine_from_cfg(self.offline_store, self.online_store)
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
                TIME_COL in entity_df.columns
            ), "Check entity_df make sure it has at least 1 columns and event_timestamp in it"

    def _get_features_to_use(
        self, views, features: list = [], is_numeric: bool = False, choose: str = "both"
    ) -> List[Feature]:
        """_summary_

        Args:
            views (_type_): FeatureView, LabelView or Service
            features (list, optional): feature(label) to use
            is_numeric (bool, optional): whether return numeric feature(label) only
            choose: whether to return features or labels or both

        Returns:
            _type_: corresponding feature(label)
        """
        if isinstance(views, FeatureView):
            buildin_features = views.get_feature_objects(is_numeric)
        elif isinstance(views, LabelView):
            buildin_features = views.get_label_objects(is_numeric)
        else:  # Service
            if choose == "labels":
                buildin_features = views.get_label_objects(self.label_views, is_numeric)
            elif choose == "features":
                buildin_features = views.get_feature_objects(self.feature_views, is_numeric)
            else:
                buildin_features = views.get_feature_objects(
                    self.feature_views, is_numeric
                ) | views.get_label_objects(self.label_views, is_numeric)

        if features:
            features = set(features)
            features = list([feature for feature in buildin_features if feature.name in features])
        else:
            features = list(set(buildin_features))
        return features

    def _get_keys_to_join(self, view, in_columns: List[str] = None) -> List[str]:
        """_summary_

        Args:
            view (_type_):  FeatureView, LabelView or Service
            entity_columns (list, optional): filter

        Returns:
            list: col name to join
        """

        if isinstance(view, FeatureView):
            available_entity_names = list(view.entities)
        elif isinstance(view, LabelView):
            available_entity_names = list(view.entities)
        elif isinstance(view, Service):
            available_entity_names = list(view.get_entities(self.feature_views, self.label_views))
        else:
            raise TypeError("must be FeatureViews, LabelViews or Service")

        # need filter
        if in_columns is not None:
            return list(
                {
                    join_key
                    for entity_name in available_entity_names
                    for join_key in self.entities[entity_name].join_keys
                    if join_key in in_columns
                }
            )

        return list(
            {
                join_key
                for entity_name in available_entity_names
                for join_key in self.entities[entity_name].join_keys
            }
        )

    def _get_views(
        self, view_name: Union[str, LabelView, Service, FeatureView]
    ) -> Union[FeatureView, LabelView, Service]:
        """_summary_

        Args:
            view_name (_type_): (name) of FeatureView, LabelView or Service

        Returns:
            _type_: FeatureView, LabelView or Service
        """
        if isinstance(view_name, (FeatureView, LabelView, Service)):
            return view_name

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
        feature_view: Union[str, FeatureView, Service],
        entity_df: Union[pd.DataFrame, str],
        features: list = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Get features in a certain view from offline store. If you want features from online store, try to call `get_online_features`.

        Args:
            feature_view (Union[str, FeatureView, Service]): A name which we can retrieve certain view from F2AI. Or, an instance of FeatureView or Service. If you want retrieve features from a service, materialize is require before to retrieve features.
            entity_df (pd.DataFrame): A query DataFrame which must contains entity columns and event_timestamp field.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional):  include timestamp defined in `entity_df` or not. Defaults to True.

        Returns:
            pd.DataFrame: a pandas' DataFrame, where you can found your features in this.
        """

        self.__check_format(entity_df)
        feature_view = self._get_views(feature_view)

        assert isinstance(
            feature_view, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        feature_objects = self._get_features_to_use(feature_view, features, choose="features")
        join_keys = self._get_keys_to_join(feature_view, list(entity_df.columns))

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

    def get_online_features(
        self,
        feature_view_name: str,
        entity_df: Union[pd.DataFrame, str],
        **kwargs,
    ) -> pd.DataFrame:
        """Get features in a certain view from online store.

        Args:
            feature_view (Union[str, FeatureView, Service]): A name which we can retrieve certain view from F2AI. Or, an instance of FeatureView or Service. If you want retrieve features from a service, materialize is require before to retrieve features.
            entity_df (pd.DataFrame): A query DataFrame which must contains entity columns.

        Returns:
            pd.DataFrame: a pandas' DataFrame, where you can found your features in this.
        """

        self.__check_format(entity_df)
        feature_view = self._get_views(feature_view_name)

        assert isinstance(feature_view, (FeatureView, Service)), "only allowed FeatureView and Service"

        join_keys = self._get_keys_to_join(feature_view, list(entity_df.columns))

        path = remove_prefix(self.project_folder, "file://")
        hkey = path.split("/")[-1] + ":" + feature_view_name

        return self.online_store.read_batch(
            entity_df=entity_df[join_keys],
            hkey=hkey,
            ttl=feature_view.ttl,
            **kwargs,
        )

    def get_period_features(
        self,
        feature_view: Union[str, FeatureView, Service],
        entity_df: pd.DataFrame,
        period: str,
        features: List[str] = None,
        include: Optional[bool] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get a period of features with given standing time point in entity_df. If period is positive, return features between (event_timestamp - period, event_timestamp]. Otherwise, between (event_timestamp, event_timestamp - period]

        Args:
            feature_view (Union[str, FeatureView, Service]): A name which we can retrieve certain feature view or service from F2AI. Or, an instance of FeatureView or Service. If you want retrieve features from a service, materialize is required.
            entity_df (pd.DataFrame): A pandas DataFrame which must contains entity keys and event_timestamp.
            period (str): A period str. Egg, 2 hours.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to None.
        """
        self.__check_format(entity_df)
        feature_view = self._get_views(feature_view)
        assert isinstance(feature_view, (FeatureView, Service)), "only allowed FeatureView and Service"

        period: Period = -Period.from_str(period)
        feature_objects = self._get_features_to_use(feature_view, features, choose="features")
        join_keys = self._get_keys_to_join(feature_view, list(entity_df.columns))

        if include is None:
            include = period.is_neg

        if isinstance(feature_view, (FeatureView, LabelView)):
            source = self.sources[feature_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(feature_view)

        assert source.timestamp_field, "no period can be applied on non-relevant data"

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

    def get_labels(
        self,
        label_view: Union[str, LabelView, Service],
        entity_df: pd.DataFrame,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Get labels for a certain LabelView or Service. Label usually means observed, key indicator of the business goal in Machine Learning context.

        Args:
            label_view (Union[str, LabelView, Service]): A name which we can retrieve certain label view or service from F2AI. Or, an instance of LabelView or Service. If you want retrieve labels from a service, materialize is required.
            entity_df (pd.DataFrame): A DataFrame used to query labels which must contains entity and event_timestamp columns.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        assert isinstance(label_view, (LabelView, Service)), "only allowed LabelView and Service"

        feature_objects = self._get_features_to_use(label_view, choose="labels")
        join_keys = self._get_keys_to_join(label_view, list(entity_df.columns))

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
        self,
        label_view: Union[str, LabelView, Service],
        entity_df: pd.DataFrame,
        period: str,
        include: Optional[bool] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Using a point of time in `entity_df` to query a period of labels. If you are giving a positive period, you will get labels between (event_timestamp, event_timestamp + period]. If you are giving a negative period, you will get labels between (event_timestamp + period, event_timestamp].

        Args:
            label_view (Union[str, LabelView, Service]): A name which we can retrieve certain label view or service from F2AI. Or, an instance of LabelView or Service. If you want retrieve labels from a service, materialize is require before to retrieve labels.
            entity_df (pd.DataFrame): A DataFrame used to query labels which must contains entity and event_timestamp columns.
            period (str): When looking backward, using negative period, egg, -2 days. Otherwise, using positive period, egg, 1 day.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to None, which means automatically decide include or not.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        period: Period = Period.from_str(period)
        label_objects = self._get_features_to_use(label_view, choose="labels")
        join_keys = self._get_keys_to_join(label_view, list(entity_df.columns))

        if isinstance(label_view, (FeatureView, LabelView)):
            source = self.sources[label_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(label_view)

        if include is None:
            include = period.is_neg

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

    def materialize(self, service: Union[str, Service, FeatureView], backoff: BackoffTime, online: bool):
        """Offline materialize, join features which specify by service name.

        Args:
            service (Union[str, Service]): name of service to materialize or an instance.
            backoff: start, end and step to materialize
        """
        if online:
            views_to_search = deepcopy(self.services)
            views_to_search.update(self.feature_views)
        else:
            views_to_search = self.services

        if isinstance(service, str):
            service_name = service
            service = views_to_search.get(service, None)
            assert (
                service is not None
            ), f"Service: {service_name} is not found in feature store in current materialize type."

        join_keys = list(
            {
                join_key
                for entity_name in service.get_label_entities(self.label_views)
                for join_key in self.entities[entity_name].join_keys
            }
        )
        label_view = service.get_label_views(self.label_views)[0]
        label_view_dict = {
            "source": self.sources[label_view.batch_source],
            "labels": label_view.get_label_objects(),
            "join_keys": join_keys,
        }

        all_feature_views = [
            {
                "join_keys": [self.entities[entity].join_keys[0] for entity in feature_view.entities],
                "features": feature_view.get_feature_objects(),
                "source": self.sources[feature_view.batch_source],
                "ttl": feature_view.ttl,
            }
            for feature_view in service.get_feature_views(self.feature_views)
        ]

        dest_path = self.offline_store.get_offline_source(service)

        batch_params = [
            [dest_path, all_feature_views, label_view_dict, t[0], t[1]] for t in backoff_to_split(backoff)
        ]
        cpu_ava = max(os.cpu_count() // 2, 1)
        with tqdm(total=len(batch_params), desc="materializing") as pbar:
            with Pool(processes=cpu_ava) as pool:
                r, w = Pipe(duplex=False)
                for args in batch_params:
                    pool.apply_async(
                        func=self.persist_engine.materialize, args=args, kwds={"signal": w, "online": online}
                    )
                    pbar.update(r.recv())
        # if online:
        #     redis_store = OnlineRedisStore(self.online_store)
        #     redis_store.write_batch()

        # print(f"materialize done, saved at '{dest_path.path if dest_path.path else dest_path.query}'")

    def _offline_pgsql_materialize_dbt(self, service: Service, incremental_begin):
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

        conn = self.offline_store.psy_conn(self.offline_store)

        max_timestamp, max_timestamp_label = self.offline_store.materialize_dbt(
            service=service,
            label_views=self.label_views,
            sources=self.sources,
        )

        label_result = pd.to_datetime(self.offline_store._get_dataframe(sql_result=max_timestamp_label)[0][0])
        try:
            result = pd.to_datetime(self.offline_store._get_dataframe(sql_result=max_timestamp)[0][0])
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
        self._schedule_local_dbt_container(service.materialize_path, json_var, dbt_path)
        # os.system(f"cd {dbt_path} && dbt run --profiles-dir {dbt_path} --vars '{json_var}' ")

    def _schedule_local_dbt_container(self, profile_name: str, vars: Dict, dbt_path: str):
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

    # def _offline_record_materialize(self, service: Service, incremental_begin):
    #     """materialize offline file

    #     Args:
    #         service (Service): service entity
    #         incremental_begin: time to begin materialize
    #     """
    #     try:
    #         incremental_begin = pd.to_datetime(incremental_begin if incremental_begin else 0, utc=True)
    #     except Exception:
    #         incremental_begin = Period.from_str(incremental_begin)
    #     except:
    #         raise TypeError("please check your `incremental_begin` type")

    #     joined_frame = self.offline_store.materialize(
    #         service=service,
    #         feature_views=self.feature_views,
    #         label_views=self.label_views,
    #         sources=self.sources,
    #         entities=self.entities,
    #         incremental_begin=incremental_begin,
    #     )
    #     joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.now(), utc=True)
    #     to_file(
    #         joined_frame,
    #         os.path.join(self.project_folder, f"{service.materialize_path}"),
    #         f"{service.materialize_path}".split(".")[-1],
    #     )
    #     print(
    #         f"materialize done, file saved at {os.path.join(self.project_folder, service.materialize_path)}"
    #     )

    def stats(
        self,
        view: Union[str, LabelView, Service, FeatureView],
        fn: Union[str, StatsFunctions] = StatsFunctions.AVG,
        features: List[str] = None,
        group_keys: List[str] = None,
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """Get statistical information from a FeatureView | LabelView | Service. You can get min, max, std, avg, mode and median from numeric features. Or, you can retrieve all combinations of categorical values with unique operation. Note, hen the fn is unique, group_keys is required and features is ignored.

        Args:
            views (List): name of view to look up.
            fn (str, optional): statistical method, min, max, std, avg, mode, median. Defaults to "avg".
            features (List[str]): A subset of features of this view. Default to None.
            group_keys (list): joined-columns to do stats. if None, means do stats on joined-entities, also accept `[]` means no grouping.
            start (str, optional): start_time. Defaults to None.
            end (str, optional): end_time. Defaults to None.
        """
        if isinstance(fn, str):
            fn = StatsFunctions(fn)
        view = self._get_views(view)

        if group_keys is None:
            group_keys = self._get_keys_to_join(view)

        is_numeric = fn != StatsFunctions.UNIQUE
        if fn == StatsFunctions.UNIQUE:
            features = []
            assert len(group_keys) > 0, "for unique fn, group_keys is required"
        else:
            features = self._get_features_to_use(view, features, is_numeric=is_numeric)

        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
        else:
            source = self.offline_store.get_offline_source(view)

        return self.offline_store.stats(
            source=source,
            features=features,
            fn=fn,
            group_keys=group_keys,
            start=start,
            end=end,
        )

    def get_latest_entities(
        self,
        view: Union[str, LabelView, Service, FeatureView],
        entity: Union[pd.DataFrame, List[str]] = None,
        start: Union[str, datetime] = 0,
        end: Union[str, datetime] = datetime.now(),
    ) -> pd.DataFrame:
        """_summary_

        Args:
            view (Union[str, LabelView, Service, FeatureView]): FeatureViews/LabelViews/Service to look up
            entity (Union[pd.DataFrame, List[str]], optional): entity to look up in views, either pd.DataFrame specify the entity or List specify the entity join-keys.
                e.g. entity = pd.DataFrame('link_id'=['1234567','AAVVDDD']) or entity = ['link_id'].Defaults to None.
            start (Union[str, pd.Timestamp], optional): start time to look up. Defaults to 0.
            end (Union[str, pd.Timestamp], optional): end time to look up. Defaults to None.

        Returns:
            pd.DataFrame: _description_

        """
        view = self._get_views(view)
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        if isinstance(entity, pd.DataFrame):
            join_keys = list(entity.columns)
            entity[TIME_COL] = end
            group_keys = self._get_keys_to_join(view, join_keys)
        else:
            join_keys = []
            group_keys = self._get_keys_to_join(view, entity)
            entity = pd.DataFrame({TIME_COL: [end]})

        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
            assert source.timestamp_field, "stats can only apply on time relative data"
        else:
            source = self.offline_store.get_offline_source(view)

        return self.offline_store.get_latest_entities(
            source=source, join_keys=join_keys, group_keys=group_keys, entity_df=entity, start=start
        )

    def get_dataset(self, service: Union[str, Service], sampler: callable = None) -> Dataset:
        """Get an abstraction of F2AI customized Dataset with a specific service.

        Args:
            service (Union[str, Service]): name of `SERVICE` to use, or a Service instance.
            sampler (callable, optional): sampler
        """
        if isinstance(service, str):
            service_name = service
            service = self.services[service_name]
            assert (
                service is not None
            ), f"service: {service_name} is not found, please double check your service name"

        return Dataset(fs=self, service=service, sampler=sampler)

    def query(self, *args, **kwargs) -> pd.DataFrame:
        """Run a query though different types of offline store.
        The usecase of this method is highly depending on different types of offline store.
        """
        return self.offline_store.query(*args, **kwargs)
