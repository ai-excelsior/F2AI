from __future__ import annotations
import abc
from enum import Enum
from typing import Dict, Any, TYPE_CHECKING
from pydantic import BaseModel
from .services import Service

if TYPE_CHECKING:
    from .sources import Source


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
    def query(self, query: str, *args, **kwargs) -> Any:
        """
        Run a query with specific offline store. egg:
            if you are using pgsql, this will run a query via psycopg2
            if you are using spark, this will run a query via sparksql
        """
        pass


def init_offline_store_from_cfg(cfg: Dict[Any]) -> OfflineStore:
    offline_store_type = OfflineStoreType(cfg["type"])

    if offline_store_type == OfflineStoreType.FILE:
        from ..offline_stores.offline_file_store import OfflineFileStore

        return OfflineFileStore()

    if offline_store_type == OfflineStoreType.PGSQL:
        from ..offline_stores.offline_postgres_store import OfflinePostgresStore

        return OfflinePostgresStore(**cfg)

    if offline_store_type == OfflineStoreType.SPARK:
        from ..offline_stores.offline_spark_store import OfflineSparkStore

        return OfflineSparkStore(**cfg)

    raise TypeError(f"offline store type must be one of [{','.join(e.value for e in OfflineStoreType)}]")
