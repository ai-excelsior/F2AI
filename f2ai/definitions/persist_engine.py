from __future__ import annotations
import uuid
import pandas as pd
import datetime
from io import StringIO
from typing import List, Optional, Set, TYPE_CHECKING, Union, Tuple, Dict
from pydantic import Field, PrivateAttr
from pypika import Query, Parameter, functions as fn, JoinType, Field as PikaField, Table, PostgreSQLQuery
from pypika.queries import QueryBuilder
from f2ai.definitions import (
    Feature,
    Period,
    LabelView,
    FileSource,
    SqlSource,
    OfflineStoreType,
    OfflineStore,
    OnlineStore,
    StatsFunctions,
    Source,
)
from f2ai.common.utils import convert_dtype_to_sqlalchemy_type


DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
QUERY_COL = "query_timestamp"
MATERIALIZE_TIME = "materialize_time"
from pydantic import BaseModel
from enum import Enum
from .offline_store import OfflineStoreType
from .online_store import OnlineStoreType


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


class OnlinePersistEngine(PersistEngine):
    type: OnlinePersistEngineType
    store: OnlineStore

    class Config:
        extra = "allow"


class RealPersistEngine(BaseModel):
    off_line: OfflinePersistEngine
    on_line: OnlinePersistEngine

    def materialize(
        self,
        save_path: Source,
        feature_views: List[Dict],
        label_view: Dict,
        start: str = None,
        end: str = None,
        online: bool = False,
        **kwargs,
    ):

        if online:
            self.on_line.materialize(save_path, feature_views, label_view, start, end, **kwargs)
        else:
            self.off_line.materialize(save_path, feature_views, label_view, start, end, **kwargs)


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