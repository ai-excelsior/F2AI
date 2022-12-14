from __future__ import annotations
import abc
import pandas as pd
from enum import Enum
from typing import Dict, Any, TYPE_CHECKING, Set, List, Optional
from pydantic import BaseModel

if TYPE_CHECKING:
    from .services import Service
    from .sources import Source
    from .features import Feature
    from .period import Period


class OnlineStoreType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize OnlineStore from configuration. If you want to add a new type of online store, you definitely want to modify this."""

    REDIS = "redis"


class OnlineStore(BaseModel):
    """An abstraction of what functionalities a OnlineStore should implements. If you want to be one of the online store contributor. This is the core."""

    type: OnlineStoreType

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def write_batch(self, service: Service) -> Source:
        """materialize data on redis

        Args:
            service (Service): an instance of Service

        Returns:
            Source
        """
        pass

    @abc.abstractmethod
    def read_batch(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: Source,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """get data from current online store.

        Args:
            entity_df (pd.DataFrame): A query DataFrame which include entities and event_timestamp column.
            features (Set[Feature]): A set of Features you want to retrieve.
            source (Source): A specific implementation of Source. For example, OnlinePostgresStore will receive a SqlSource which point to table with time semantic.
            join_keys (List[str], optional): Which columns to join the entity_df with source. Defaults to [].
            ttl (Optional[Period], optional): Time to Live, if feature's event_timestamp exceeds the ttl, it will be dropped. Defaults to None.
            include (bool, optional): If include (<=) the event_timestamp in entity_df, else (<). Defaults to True.

        Returns:
            pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def read_period_batch(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: Source,
        period: Period,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """get a period of features from online store between [event_timestamp, event_timestamp + period] if period > 0 else [event_timestamp + period, event_timestamp].

        Args:
            entity_df (pd.DataFrame): A query DataFrame which include entities and event_timestamp column.
            features (Set[Feature]): A set of Features you want to retrieve.
            source (Source): A specific implementation of Source. For example, OnlinePostgresStore will receive a SqlSource which point to table with time semantic.
            period (Period): A Period instance, which wrapped by F2AI.
            join_keys (List[str], optional): Which columns to join the entity_df with source. Defaults to [].. Defaults to [].
            ttl (Optional[Period], optional): Time to Live, if feature's event_timestamp exceeds the ttl, it will be dropped. Defaults to None.
            include (bool, optional): If include (<=) the event_timestamp in entity_df, else (<). Defaults to True.

        Returns:
            pd.DataFrame
        """
        pass


def init_online_store_from_cfg(cfg: Dict[Any]) -> OnlineStore:
    """Initialize an implementation of OnlineStore from yaml config.

    Args:
        cfg (Dict[Any]): a parsed config object.

    Returns:
        OnlineStore: Different types of OnlineStore.
    """
    offline_store_type = OnlineStore(cfg["type"])

    if offline_store_type == OnlineStoreType.REDIS:
        from ..online_stores.OnlineRedisStore import OnlineRedisStore

        redis_conf = cfg.pop("redis_conf", {})
        return OnlineRedisStore(**cfg, **redis_conf)

    raise TypeError(f"offline store type must be one of [{','.join(e.value for e in OnlineStore)}]")
