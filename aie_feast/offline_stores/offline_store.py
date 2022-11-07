from __future__ import annotations
from pydantic import BaseModel
from enum import Enum
from typing import TYPE_CHECKING
import abc

if TYPE_CHECKING:
    from aie_feast.service import Service
    from aie_feast.common.source import Source


class OfflineStoreType(str, Enum):
    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class OfflineStore(BaseModel):
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
