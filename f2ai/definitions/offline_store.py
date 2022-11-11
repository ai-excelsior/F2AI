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


class OfflineStoreType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize OfflineStore from configuration. If you want to add a new type of offline store, you definitely want to modify this."""

    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class OfflineStore(BaseModel):
    """An abstraction of what functionalities a OfflineStore should implements. If you want to be a one of the offline store contributor. This is the core."""

    type: OfflineStoreType

    @abc.abstractmethod
    def get_offline_source(self, service: Service) -> Source:
        """get offline materialized source with a specific service

        Args:
            service (Service): an instance of Service

        Returns:
            Source
        """
        pass

    @abc.abstractmethod
    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: Source,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """get features from current offline store.

        Args:
            entity_df (pd.DataFrame): A query DataFrame which include entities and event_timestamp column.
            features (Set[Feature]): A set of Features you want to retrieve.
            source (Source): A specific implementation of Source. For example, OfflinePostgresStore will receive a SqlSource which point to table with time semantic.
            join_keys (List[str], optional): Which columns to join the entity_df with source. Defaults to [].
            ttl (Optional[Period], optional): Time to Live, if feature's event_timestamp exceeds the ttl, it will be dropped. Defaults to None.
            include (bool, optional): If include (<=) the event_timestamp in entity_df, else (<). Defaults to True.

        Returns:
            pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def get_period_features(
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
        """get a period of features from offline store between [event_timestamp, event_timestamp + period] if period > 0 else [event_timestamp + period, event_timestamp].

        Args:
            entity_df (pd.DataFrame): A query DataFrame which include entities and event_timestamp column.
            features (Set[Feature]): A set of Features you want to retrieve.
            source (Source): A specific implementation of Source. For example, OfflinePostgresStore will receive a SqlSource which point to table with time semantic.
            period (Period): A Period instance, which wrapped by F2AI.
            join_keys (List[str], optional): Which columns to join the entity_df with source. Defaults to [].. Defaults to [].
            ttl (Optional[Period], optional): Time to Live, if feature's event_timestamp exceeds the ttl, it will be dropped. Defaults to None.
            include (bool, optional): If include (<=) the event_timestamp in entity_df, else (<). Defaults to True.

        Returns:
            pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def get_latest_entities(
        self, source: Source, join_keys: List[str] = [], entity_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """get latest unique entities from a source. Which is useful when you want to know how many entities you have, or what is the latest features appear in your data source.

        Args:
            source (Source): A specific implementation of Source. For example, OfflinePostgresStore will receive a SqlSource which point to table with time semantic.
            join_keys (List[str], optional): Which columns to join the entity_df with source. Defaults to [].
            entity_df (pd.DataFrame, optional): A query DataFrame which include entities and event_timestamp column. Defaults to None.

        Returns:
            pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def stats():
        pass

    @abc.abstractmethod
    def query(self, query: str, *args, **kwargs) -> Any:
        """
        Run a query with specific offline store. egg:
            if you are using pgsql, this will run a query via psycopg2
            if you are using spark, this will run a query via sparksql
        """
        pass


def init_offline_store_from_cfg(cfg: Dict[Any]) -> OfflineStore:
    """Initialize an implementation of OfflineStore from yaml config.

    Args:
        cfg (Dict[Any]): a parsed config object.

    Returns:
        OfflineStore: Different types of OfflineStore.
    """
    offline_store_type = OfflineStoreType(cfg["type"])

    if offline_store_type == OfflineStoreType.FILE:
        from ..offline_stores.offline_file_store import OfflineFileStore

        return OfflineFileStore()

    if offline_store_type == OfflineStoreType.PGSQL:
        from ..offline_stores.offline_postgres_store import OfflinePostgresStore

        pgsql_conf = cfg.pop("pgsql_conf", {})
        return OfflinePostgresStore(**cfg, **pgsql_conf)

    if offline_store_type == OfflineStoreType.SPARK:
        from ..offline_stores.offline_spark_store import OfflineSparkStore

        return OfflineSparkStore(**cfg)

    raise TypeError(f"offline store type must be one of [{','.join(e.value for e in OfflineStoreType)}]")
