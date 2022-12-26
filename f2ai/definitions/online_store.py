from __future__ import annotations

import abc
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from .feature_view import FeatureView
    from .period import Period
    from .sources import Source


class OnlineStoreType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize OnlineStore from configuration. If you want to add a new type of online store, you definitely want to modify this."""

    REDIS = "redis"


class OnlineStore(BaseModel):
    """An abstraction of what functionalities a OnlineStore should implements. If you want to be one of the online store contributor. This is the core."""

    type: OnlineStoreType
    name: str

    class Config:
        extra = "allow"

    @abc.abstractmethod
    def write_batch(
        self, featrue_view: FeatureView, project_name: str, dt: pd.DataFrame, ttl: Optional[Period]
    ) -> Source:
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
        hkey: str,
        ttl: Optional[Period] = None,
        period: Optional[Period] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """get data from current online store.

        Args:
            entity_df (pd.DataFrame): A query DataFrame which include entities and event_timestamp column.
            hkey: hash key.
            ttl (Optional[Period], optional): Time to Live, if feature's event_timestamp exceeds the ttl, it will be dropped. Defaults to None.

        Returns:
            pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def get_online_source(self, **kwargs):
        pass


def init_online_store_from_cfg(cfg: Dict[Any], name: str) -> OnlineStore:
    """Initialize an implementation of OnlineStore from yaml config.

    Args:
        cfg (Dict[Any]): a parsed config object.

    Returns:
        OnlineStore: Different types of OnlineStore.
    """
    online_store_type = OnlineStoreType(cfg["type"])

    if online_store_type == OnlineStoreType.REDIS:
        from ..online_stores.online_redis_store import OnlineRedisStore

        redis_conf = cfg.pop("redis_conf", {})
        redis_conf.update({"name": name})
        return OnlineRedisStore(**cfg, **redis_conf)

    raise TypeError(f"offline store type must be one of [{','.join(e.value for e in OnlineStore)}]")
