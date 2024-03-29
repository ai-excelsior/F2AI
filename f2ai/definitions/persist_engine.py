from __future__ import annotations
import abc
import os
from typing import Dict, List, Optional
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm
from multiprocessing import Pool
from f2ai.definitions import (
    OfflineStoreType,
    OfflineStore,
    OnlineStore,
    Entity,
    Service,
    FeatureView,
    LabelView,
    BackOffTime,
    Source,
    Feature,
    Period,
)


class PersistFeatureView(BaseModel):
    """
    Another feature view that usually used when finalizing the results.
    """

    name: str
    source: Source
    features: List[Feature] = []
    join_keys: List[str] = []
    ttl: Optional[Period]


class PersistLabelView(BaseModel):
    """
    Another label view that usually used when finalizing the results.
    """

    source: Source
    labels: List[Feature] = []
    join_keys: List[str] = []


class PersistEngineType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize PersistEngine from configuration."""

    OFFLINE = "offline"
    ONLINE = "online"


class OfflinePersistEngineType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize PersistEngine from configuration."""

    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class OnlinePersistEngineType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize PersistEngine from configuration."""

    LOCAL = "local"
    DISTRIBUTE = "distribute"


class PersistEngine(BaseModel):
    type: PersistEngineType

    class Config:
        extra = "allow"


class OfflinePersistEngine(PersistEngine):
    type: OfflinePersistEngineType
    offline_store: OfflineStore

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def materialize(
        self,
        feature_views: List[PersistFeatureView],
        label_view: PersistLabelView,
        destination: Source,
        back_off_time: BackOffTime,
    ):
        pass


class OnlinePersistEngine(PersistEngine):
    type: OnlinePersistEngineType
    online_store: OnlineStore
    offline_store: OfflineStore

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def materialize(self, prefix: str, feature_view: PersistFeatureView, back_off_time: BackOffTime):
        pass


class RealPersistEngine(BaseModel):
    offline_engine: OfflinePersistEngine
    online_engine: OnlinePersistEngine

    def materialize_offline(
        self,
        services: List[Service],
        label_views: Dict[str, LabelView],
        feature_views: Dict[str, FeatureView],
        entities: Dict[str, Entity],
        sources: Dict[str, Source],
        back_off_time: BackOffTime,
    ):
        cpu_ava = max(os.cpu_count() // 2, 1)

        # with Pool(processes=cpu_ava) as pool:
        service_to_list_of_args = []
        back_off_segments = list(back_off_time.to_units())
        for service in services:
            destination = self.offline_engine.offline_store.get_offline_source(service)
            label_view = service.get_label_views(label_views)[0]
            label_view = PersistLabelView(
                source=sources[label_view.batch_source],
                labels=label_view.get_label_objects(),
                join_keys=[
                    join_key
                    for entity_name in label_view.entities
                    for join_key in entities[entity_name].join_keys
                ],
            )
            label_names = set([label.name for label in label_view.labels])

            feature_views = [
                PersistFeatureView(
                    name=feature_view.name,
                    join_keys=[
                        join_key
                        for entity_name in feature_view.entities
                        for join_key in entities[entity_name].join_keys
                    ],
                    features=[
                        feature
                        for feature in feature_view.get_feature_objects()
                        if feature.name not in label_names
                    ],
                    source=sources[feature_view.batch_source],
                    ttl=feature_view.ttl,
                )
                for feature_view in service.get_feature_views(feature_views)
            ]
            feature_views = [feature_view for feature_view in feature_views if len(feature_view.features) > 0]
            service_to_list_of_args += [
                (feature_views, label_view, destination, per_backoff, service.name)
                for per_backoff in back_off_segments
            ]

        bars = {
            service.name: tqdm(
                total=len(back_off_segments), desc=f"materializing {service.name}", position=pos
            )
            for pos, service in enumerate(services)
        }

        pool = Pool(processes=cpu_ava)

        for param in service_to_list_of_args:
            pool.apply_async(
                self.offline_engine.materialize,
                param,
                callback=lambda x: bars[x].update(),
                error_callback=lambda x: print(x.__cause__),
            )

        pool.close()
        pool.join()
        [bars[bar].close() for bar in bars]

    def materialize_online(
        self,
        prefix: str,
        feature_views: List[FeatureView],
        entities: Dict[str, Entity],
        sources: Dict[str, Source],
        back_off_time: BackOffTime,
    ):
        cpu_ava = max(os.cpu_count() // 2, 1)

        feature_backoffs = []
        back_off_segments = list(back_off_time.to_units())
        for feature_view in feature_views:
            join_keys = list(
                {
                    join_key
                    for entity_name in feature_view.entities
                    for join_key in entities[entity_name].join_keys
                }
            )
            feature_view = PersistFeatureView(
                name=feature_view.name,
                join_keys=join_keys,
                features=feature_view.get_feature_objects(),
                source=sources[feature_view.batch_source],
                ttl=feature_view.ttl,
            )
            feature_backoffs += [
                (prefix, feature_view, per_backoff, feature_view.name) for per_backoff in back_off_segments
            ]

        bars = {
            feature_view.name: tqdm(
                total=len(back_off_segments), desc=f"materializing {feature_view.name}", position=pos
            )
            for pos, feature_view in enumerate(feature_views)
        }

        pool = Pool(processes=cpu_ava)

        for param in feature_backoffs:
            pool.apply_async(
                self.online_engine.materialize,
                param,
                callback=lambda x: bars[x].update(),
                error_callback=lambda x: print(x.__cause__),
            )

        pool.close()
        pool.join()
        [bars[bar].close() for bar in bars]


def init_persist_engine_from_cfg(offline_store: OfflineStore, online_store: OnlineStore):
    """Initialize an implementation of PersistEngine from yaml config.

    Args:
        cfg (Dict[Any]): a parsed config object.

    Returns:
        RealPersistEngine: Contains Offline and Online PersistEngine
    """

    if offline_store.type == OfflineStoreType.FILE:
        from ..persist_engine.offline_file_persistengine import OfflineFilePersistEngine
        from ..persist_engine.online_local_persistengine import OnlineLocalPersistEngine

        offline_persist_engine_cls = OfflineFilePersistEngine
        online_persist_engine_cls = OnlineLocalPersistEngine

    elif offline_store.type == OfflineStoreType.PGSQL:
        from ..persist_engine.offline_pgsql_persistengine import OfflinePgsqlPersistEngine
        from ..persist_engine.online_local_persistengine import OnlineLocalPersistEngine

        offline_persist_engine_cls = OfflinePgsqlPersistEngine
        online_persist_engine_cls = OnlineLocalPersistEngine

    elif offline_store.type == OfflineStoreType.SPARK:
        from ..persist_engine.offline_spark_persistengine import OfflineSparkPersistEngine
        from ..persist_engine.online_spark_persistengine import OnlineSparkPersistEngine

        offline_persist_engine_cls = OfflineSparkPersistEngine
        online_persist_engine_cls = OnlineSparkPersistEngine

    offline_engine = offline_persist_engine_cls(offline_store=offline_store)
    online_engine = online_persist_engine_cls(offline_store=offline_store, online_store=online_store)
    return RealPersistEngine(offline_engine=offline_engine, online_engine=online_engine)
