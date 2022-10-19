from pydantic import BaseModel
from enum import Enum


class OfflineStoreType(str, Enum):
    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class FeatureDataTypes(str, Enum):
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class Feature(BaseModel):
    name: str
    dtype: FeatureDataTypes
