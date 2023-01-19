import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Union

from .definitions.offline_store import OfflineStore
from .definitions.online_store import OnlineStore
from .definitions.persist_engine import RealPersistEngine
from .common.get_config import (
    get_entity_cfg,
    get_feature_views,
    get_label_views,
    get_service_cfg,
    get_source_cfg,
)
from .common.read_file import read_yml
from .dataset.dataset import Dataset
from .definitions import (
    BackOffTime,
    Feature,
    FeatureView,
    FileSource,
    LabelView,
    Period,
    Service,
    StatsFunctions,
    init_offline_store_from_cfg,
    init_online_store_from_cfg,
    init_persist_engine_from_cfg,
)

from .common.time_field import *


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            cfg = read_yml(os.path.join(project_folder, "feature_store.yml"))
            self.name = cfg["project"]
            self.offline_store: OfflineStore = init_offline_store_from_cfg(
                cfg["offline_store"], cfg["project"]
            )
            self.online_store: OnlineStore = init_online_store_from_cfg(cfg["online_store"], cfg["project"])
            self.persist_engine: RealPersistEngine = init_persist_engine_from_cfg(
                self.offline_store, self.online_store
            )
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

    def __check_format(self, entity_df):
        if isinstance(entity_df, pd.DataFrame):
            # TODO: Remove this constraint in future
            assert (
                TIME_COL in entity_df.columns
            ), "Check entity_df make sure it has at least 1 columns and event_timestamp in it"

    def _get_features_to_use(
        self, views, features: List[str] = [], is_numeric: bool = False, choose: str = "both"
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
        view_features: List[Feature] = []
        if isinstance(views, FeatureView):
            view_features += views.get_feature_objects(is_numeric)

        if isinstance(views, LabelView):
            view_features += views.get_label_objects(is_numeric)

        if isinstance(views, Service):
            if choose == "features" or choose == "both":
                view_features += views.get_feature_objects(self.feature_views, is_numeric)

            if choose == "labels" or choose == "both":
                view_features += views.get_label_objects(self.label_views, is_numeric)

        if features:
            include_features = set(features)
            return list([feature for feature in view_features if feature.name in include_features])

        return view_features

    def _get_keys_to_join(self, view, in_columns: List[str] = None) -> List[str]:
        """_summary_

        Args:
            view (_type_):  FeatureView, LabelView or Service
            entity_columns (list, optional): filter

        Returns:
            list: col name to join
        """

        if isinstance(view, FeatureView):
            available_entity_names = view.entities
        elif isinstance(view, LabelView):
            available_entity_names = view.entities
        elif isinstance(view, Service):
            available_entity_names = view.get_entities(self.feature_views, self.label_views)
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

        if view_name in self.feature_views:
            return self.feature_views[view_name]
        elif view_name in self.label_views:
            return self.label_views[view_name]
        elif view_name in self.services:
            return self.services[view_name]
        else:
            raise ValueError(f"Can't find any views/services in feature store with name: {view_name}")

    def get_features(
        self,
        feature_view: Union[str, FeatureView, Service],
        entity_df: Union[pd.DataFrame, str],
        features: List[str] = None,
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

        if isinstance(feature_view, Service):
            feature_objects = self._get_features_to_use(feature_view, features, choose="both")
        else:
            feature_objects = self._get_features_to_use(feature_view, features, choose="features")

        if isinstance(feature_view, (FeatureView, LabelView)):
            source = self.sources[feature_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(feature_view)

        # check feature columns should not appear in entity_df columns
        assert not any(
            [feature.name in entity_df.columns for feature in feature_objects]
        ), "Naming conflict: entity_df should not contain any columns same as features"

        join_keys = self._get_keys_to_join(feature_view, list(entity_df.columns))
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
            pd.DataFrame: a pandas DataFrame, where you can found your features in this.
        """
        assert (
            isinstance(entity_df, pd.DataFrame) and len(entity_df.columns) >= 1
        ), "entity_df should be a pd.Dataframe and have at least one column of entities"

        feature_view = self._get_views(feature_view_name)
        join_keys = self._get_keys_to_join(feature_view)
        assert not [
            col for col in join_keys if col not in entity_df.columns
        ], f"make sure columns in entity_df mush involve that in {feature_view_name},{','.join([col for col in join_keys if col not in entity_df.columns])} not in entity_df's columns"

        return self.online_store.read_batch(
            entity_df=entity_df,
            project_name=self.online_store.name,
            view=feature_view,
            feature_views=self.feature_views,
            entities=self.entities,
            join_keys=join_keys.sort(),
            **kwargs,
        )

    def get_period_features(
        self,
        feature_view: Union[str, FeatureView, Service],
        entity_df: pd.DataFrame,
        period: Union[str, Period],
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

        if isinstance(period, str):
            period: Period = -Period.from_str(period)

        if include is None:
            include = period.is_neg

        if isinstance(feature_view, Service):
            feature_objects = self._get_features_to_use(feature_view, features, choose="both")
        else:
            feature_objects = self._get_features_to_use(feature_view, features, choose="features")

        if isinstance(feature_view, (FeatureView, LabelView)):
            source = self.sources[feature_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(feature_view)

        assert source.timestamp_field, "no period can be applied on non-relevant data"

        join_keys = self._get_keys_to_join(feature_view, list(entity_df.columns))
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

        if isinstance(label_view, Service):
            feature_objects = self._get_features_to_use(label_view, choose="both")
        else:
            feature_objects = self._get_features_to_use(label_view, choose="labels")

        if isinstance(label_view, (FeatureView, LabelView)):
            source = self.sources[label_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(label_view)

        join_keys = self._get_keys_to_join(label_view, list(entity_df.columns))
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
        period: Union[str, Period],
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

        if isinstance(period, str):
            period: Period = Period.from_str(period)

        if include is None:
            include = period.is_neg

        if isinstance(label_view, (FeatureView, LabelView)):
            source = self.sources[label_view.batch_source]
        else:
            source = self.offline_store.get_offline_source(label_view)

        label_objects = self._get_features_to_use(label_view, choose="labels")
        join_keys = self._get_keys_to_join(label_view, list(entity_df.columns))
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

    def materialize(
        self,
        service_or_views: Union[str, Service, FeatureView],
        back_off_time: BackOffTime,
        online: bool = False,
    ):
        """
        Materialize data into another place. An offline materialize means to join features and save it in a predefined path. An online materialize means sync features to online store.

        Args:
            service_or_views (Union[str, Service]): name of services or views to materialize.
            back_off_time: A time range and given step to materialize.
        """
        if not isinstance(service_or_views, list):
            service_or_views = [service_or_views]

        if online:
            # filter with views to materialize.
            feature_view_names = []
            for x in service_or_views:
                if isinstance(x, str):
                    if x in self.feature_views:
                        feature_view_names.append(x)
                    elif x in self.services:
                        feature_view_names += self.services[x].get_feature_view_names(self.feature_views)
                elif isinstance(x, Service):
                    feature_view_names += x.get_feature_view_names(self.feature_views)
                elif isinstance(x, FeatureView):
                    feature_view_names.append(x.name)

            # unique feature views
            feature_view_names = list(dict.fromkeys(feature_view_names))
            feature_views = [self.feature_views[x] for x in feature_view_names if x in self.feature_views]
            self.persist_engine.materialize_online(
                self.name,
                feature_views,
                self.entities,
                self.sources,
                back_off_time,
            )
        else:
            service_names = []
            for x in service_or_views:
                if isinstance(x, Service):
                    service_names.append(x.name)
                elif isinstance(x, str):
                    service_names.append(x)

            service_names = list(dict.fromkeys(service_names))
            services = [self.services[x] for x in service_names if x in self.services]
            assert (
                len(services) > 0
            ), f"The service {service_or_views} is not found in store, please double check it."

            self.persist_engine.materialize_offline(
                services, self.label_views, self.feature_views, self.entities, self.sources, back_off_time
            )

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
        """Get latest entity event timestamp. Useful when you want to know how many combinations of entity do you have, and their event timestamp.

        Args:
            view (Union[str, LabelView, Service, FeatureView]): FeatureViews/LabelViews/Service to look up
            entity (Union[pd.DataFrame, List[str]], optional): entity to look up in views, either pd.DataFrame specify the entity or List specify the entity join-keys.
                e.g. entity = pd.DataFrame('link_id'=['1234567','AAVVDDD']) or entity = ['link_id'].Defaults to None.
            start (Union[str, pd.Timestamp], optional): start time to look up. Defaults to 0.
            end (Union[str, pd.Timestamp], optional): end time to look up. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with entity keys and latest event timestamp.

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
