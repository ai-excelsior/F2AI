from __future__ import annotations
import dill
from pydantic import BaseModel
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Any, Optional

if TYPE_CHECKING:
    from aie_feast.views import BaseView


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


class SchemaType(str, Enum):
    FEATURE = 0
    LABEL = 1


NUMERIC_FEATURE_DATA_TYPES = {
    FeatureDataTypes.INT,
    FeatureDataTypes.INT32,
    FeatureDataTypes.INT64,
    FeatureDataTypes.FLOAT,
    FeatureDataTypes.FLOAT32,
    FeatureDataTypes.FLOAT64,
}


class FeatureSchema(BaseModel):
    """A defination of a single feature"""

    name: str
    description: Optional[str]
    dtype: FeatureDataTypes

    def is_numeric(self):
        if self.dtype in NUMERIC_FEATURE_DATA_TYPES:
            return True
        return False


class Feature(BaseModel):
    """A feature or label that connected with a specific view"""

    name: str
    dtype: FeatureDataTypes
    period: Optional[str]
    schema_type: SchemaType
    view_name: str

    def __hash__(self) -> int:
        return hash(dill.dumps(self.dict()))


class Entity(BaseModel):
    name: str
    description: Optional[str]
    join_keys: List[str] = []

    def __init__(__pydantic_self__, **data: Any) -> None:

        join_keys = data.pop("join_keys", [])
        if len(join_keys) == 0:
            join_keys = [data.get("name")]

        super().__init__(**data, join_keys=join_keys)

    def __hash__(self) -> int:
        return hash(dill.dumps(self.dict()))


class SchemaAnchor(BaseModel):
    """
    Feature anchor is a link that refer to a specific feature.
    """

    view_name: str
    schema_name: str
    period: Optional[str]

    @classmethod
    def from_yamls(cls, cfgs: List[str]) -> "List[SchemaAnchor]":
        return [cls.from_yaml(cfg) for cfg in cfgs]

    @classmethod
    def from_yaml(cls, cfg: str) -> "SchemaAnchor":
        components = cfg.split(":")

        if len(components) < 2:
            raise ValueError("Please indicate features in table:feature format")
        elif len(components) > 3:
            raise ValueError("Please make sure colon not in name of table or features")
        elif len(components) == 2:
            view_name, schema_name = components
            return cls(view_name=view_name, schema_name=schema_name)
        elif len(components) == 3:
            view_name, schema_name, period = components
            return cls(view_name=view_name, schema_name=schema_name, period=period)

    def get_features_from_views(self, views: Dict[str, BaseView], is_numeric=False) -> List[Feature]:
        from aie_feast.views import FeatureView

        view: BaseView = views[self.view_name]
        schema_type = SchemaType.FEATURE if isinstance(view, FeatureView) else SchemaType.LABEL

        if self.schema_name == "*":
            return [
                Feature(
                    name=feature_schema.name,
                    dtype=feature_schema.dtype,
                    period=self.period,
                    view_name=view.name,
                    schema_type=schema_type,
                )
                for feature_schema in view.schemas
                if (feature_schema.is_numeric() if is_numeric else True)
            ]

        feature_schema = next((schema for schema in view.schemas if schema.name == self.schema_name), None)
        if feature_schema and (feature_schema.is_numeric() if is_numeric else True):
            return [
                Feature(
                    name=feature_schema.name,
                    dtype=feature_schema.dtype,
                    period=self.period,
                    view_name=view.name,
                    schema_type=schema_type,
                )
            ]

        return []
