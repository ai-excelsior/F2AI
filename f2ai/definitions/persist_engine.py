from __future__ import annotations
import abc
from datetime import datetime
from executing import Source
import pandas as pd
from typing import List, Dict, Union
from pydantic import BaseModel
from enum import Enum
from f2ai.definitions import (
    OfflineStoreType,
    OfflineStore,
    OnlineStore,
    Entity,
    Service,
    FeatureView,
    LabelView,
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
    off_line: OfflinePersistEngine
    on_line: OnlinePersistEngine

    def materialize(
        self,
        service: Union[Service, FeatureView],
        label_views: Dict[str, LabelView],
        feature_views: Dict[str, FeatureView],
        entities: Dict[Entity],
        sources: Dict[str, Source],
        start: pd.Timestamp,
        end: pd.Timestamp,
        online: bool = False,
        **kwargs,
    ):

        if online:
            service = feature_views if isinstance(service, Service) else {service.name: service}
            save_path = self.on_line.store.get_online_source()
            for name, feature_view in service.items():
                join_keys = list(
                    {
                        join_key
                        for entity_name in feature_view.entities
                        for join_key in entities[entity_name].join_keys
                    }
                )
                if feature_view.ttl is not None:
                    start = max(
                        start, pd.to_datetime(datetime.now(), utc=True) - feature_view.ttl.to_py_timedelta()
                    )
                all_views = {
                    "join_keys": join_keys,
                    "features": feature_view.get_feature_objects(),
                    "source": sources[feature_view.batch_source],
                    "ttl": feature_view.ttl,
                    "name": name,
                }
            self.on_line.materialize(save_path, all_views, start, end, self.off_line.store, **kwargs)
        else:
            assert isinstance(service, Service), "offline materialize can only be applied on Service"
            save_path = self.off_line.store.get_offline_source(service)
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
            self.off_line.materialize(save_path, all_views, start, end, **kwargs)


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

    return RealPersistEngine(off_line=off_persist, on_line=on_persis)
