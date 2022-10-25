from pydantic import BaseModel
from enum import Enum


class OfflineStoreType(str, Enum):
    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class OfflineStore(BaseModel):
    type: OfflineStoreType
