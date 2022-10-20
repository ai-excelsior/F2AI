from pydantic import BaseModel
from enum import Enum
from typing import List, Any, Optional


class OfflineStoreType(str, Enum):
    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"


class FeatureDataTypes(str, Enum):
    INT = "int"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT = "float"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOLEAN = "bool"


NUMERIC_FEATURE_DATA_TYPES = {
    FeatureDataTypes.INT,
    FeatureDataTypes.INT32,
    FeatureDataTypes.INT64,
    FeatureDataTypes.FLOAT,
    FeatureDataTypes.FLOAT32,
    FeatureDataTypes.FLOAT64,
}


class Feature(BaseModel):
    name: str
    dtype: FeatureDataTypes

    def is_numeric(self):
        if self.dtype in NUMERIC_FEATURE_DATA_TYPES:
            return True
        return False


class Entity(BaseModel):
    name: str
    description: Optional[str]
    join_keys: List[str] = []

    def __init__(__pydantic_self__, **data: Any) -> None:

        join_keys = data.pop("join_keys", [])
        if len(join_keys) == 0:
            join_keys = [data.get("name")]

        super().__init__(**data, join_keys=join_keys)
