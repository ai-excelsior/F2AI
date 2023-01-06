from __future__ import annotations
import abc
import os
from typing import Dict, Union
from pydantic import BaseModel
from enum import Enum
from multiprocessing import Pipe, Pool
from tqdm import tqdm

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
)

DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
QUERY_COL = "query_timestamp"
MATERIALIZE_TIME = "materialize_time"


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
    store: OfflineStore

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def materialize(self, **kwargs):
        pass


class OnlinePersistEngine(PersistEngine):
    type: OnlinePersistEngineType
    store: OnlineStore

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def materialize(self, **kwargs):
        pass


class RealPersistEngine(BaseModel):
    offline_engine: OfflinePersistEngine
    online_engine: OnlinePersistEngine

    def materialize(
        self,
        service: Union[Service, FeatureView],
        label_views: Dict[str, LabelView],
        feature_views: Dict[str, FeatureView],
        entities: Dict[Entity],
        sources: Dict[str, Source],
        back_off_time: BackOffTime,
        online: bool = False,
    ):

        cpu_ava = max(os.cpu_count() // 2, 1)
        if online:
            service: Dict[str, FeatureView] = (
                feature_views if isinstance(service, Service) else {service.name: service}
            )
            destination = self.online_engine.store.get_online_source()
            for name, feature_view in service.items():
                join_keys = list(
                    {
                        join_key
                        for entity_name in feature_view.entities
                        for join_key in entities[entity_name].join_keys
                    }
                )
                all_views = {
                    "join_keys": join_keys,
                    "features": feature_view.get_feature_objects(),
                    "source": sources[feature_view.batch_source],
                    "ttl": feature_view.ttl,
                    "name": name,
                }
                batch_params = [
                    [
                        destination,
                        all_views,
                        segment,
                        self.offline_engine.store,
                    ]
                    for segment in back_off_time.to_units()
                ]
                with tqdm(total=len(batch_params), desc=f"materializing {name}") as pbar:
                    with Pool(processes=cpu_ava) as pool:
                        r, w = Pipe(duplex=False)
                        for args in batch_params:
                            pool.apply(func=self.online_engine.materialize, args=args, kwds={"signal": w})
                            pbar.update(r.recv())
        else:
            assert isinstance(service, Service), "offline materialize can only be applied on Service"
            destination = self.offline_engine.store.get_offline_source(service)
            join_keys = list(
                {
                    join_key
                    for entity_name in service.get_label_entities(label_views)
                    for join_key in entities[entity_name].join_keys
                }
            )
            label_view = service.get_label_views(label_views)[0]
            label_view_dict = {
                "source": sources[label_view.batch_source],
                "labels": label_view.get_label_objects(),
                "join_keys": join_keys,
            }
            all_feature_views = [
                {
                    "join_keys": [entities[entity].join_keys[0] for entity in feature_view.entities],
                    "features": feature_view.get_feature_objects(),
                    "source": sources[feature_view.batch_source],
                    "ttl": feature_view.ttl,
                }
                for feature_view in service.get_feature_views(feature_views)
            ]
            all_views = {"label": label_view_dict, "features": all_feature_views}
            batch_params = [
                [
                    destination,
                    all_views,
                    segment
                ]
                for segment in back_off_time.to_units()
            ]
            with tqdm(total=len(batch_params), desc=f"materializing {service.name}") as pbar:
                with Pool(processes=cpu_ava) as pool:
                    r, w = Pipe(duplex=False)
                    for args in batch_params:
                        pool.apply_async(func=self.offline_engine.materialize, args=args, kwds={"signal": w})
                        pbar.update(r.recv())


def init_persist_engine_from_cfg(cfg1, cfg2):
    """Initialize an implementation of PersistEngine from yaml config.

    Args:
        cfg (Dict[Any]): a parsed config object.

    Returns:
        RealPersistEngine: Contains Offline and Online PersistEngine
    """

    if cfg1.type == OfflineStoreType.FILE:
        from ..persist_engine.offline_file_persistengine import OfflineFilePersistEngine
        from ..persist_engine.online_local_persistengine import OnlineLocalPersistEngine

        off_persist = OfflineFilePersistEngine(**{"type": "file", "store": cfg1})
        on_persis = OnlineLocalPersistEngine(**{"type": "local", "store": cfg2})

    if cfg1.type == OfflineStoreType.PGSQL:
        from ..persist_engine.offline_pgsql_persistengine import OfflinePgsqlPersistEngine
        from ..persist_engine.online_local_persistengine import OnlineLocalPersistEngine

        off_persist = OfflinePgsqlPersistEngine(**{"type": "pgsql", "store": cfg1})
        on_persis = OnlineLocalPersistEngine(**{"type": "local", "store": cfg2})

    if cfg1.type == OfflineStoreType.SPARK:
        from ..persist_engine.offline_spark_persistengine import OfflineSparkPersistEngine
        from ..persist_engine.online_spark_persistengine import OnlineSparkPersistEngine

        off_persist = OfflineSparkPersistEngine(**{"type": "spark", "store": cfg1})
        on_persis = OnlineSparkPersistEngine(**{"type": "distribute", "store": cfg2})

    return RealPersistEngine(offline_engine=off_persist, online_engine=on_persis)
