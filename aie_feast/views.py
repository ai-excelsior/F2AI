from typing import List, Optional, Dict, Set

from aie_feast.definitions import FeatureSchema, Feature, SchemaType
from pydantic import BaseModel, Field


class BaseView(BaseModel):
    name: str
    description: Optional[str]
    entities: List[str] = []
    schemas: List[FeatureSchema] = Field(alias="schema", default=[])
    batch_source: Optional[str]
    ttl: Optional[str]
    tags: Dict[str, str] = {}


class FeatureView(BaseView):
    def get_feature_names(self):
        return [feature.name for feature in self.schemas]

    def get_feature_objects(self, is_numeric=False) -> Set[Feature]:
        return {
            Feature(
                name=schema.name,
                dtype=schema.dtype,
                schema_type=SchemaType.FEATURE,
                view_name=self.name,
            )
            for schema in self.schemas
            if (schema.is_numeric() if is_numeric else True)
        }


class LabelView(BaseView):
    request_source: Optional[str]

    def get_label_names(self):
        return [label.name for label in self.schemas]

    def get_label_objects(self, is_numeric=False) -> Set[Feature]:
        return {
            Feature(
                name=schema.name,
                dtype=schema.dtype,
                schema_type=SchemaType.LABEL,
                view_name=self.name,
            )
            for schema in self.schemas
            if (schema.is_numeric() if is_numeric else True)
        }
