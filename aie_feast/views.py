from typing import List, Optional, Dict, Set, Any

from aie_feast.definitions import FeatureSchema, Feature
from aie_feast.period import Period
from pydantic import BaseModel, Field


class BaseView(BaseModel):
    name: str
    description: Optional[str]
    entities: List[str] = []
    schemas: List[FeatureSchema] = Field(alias="schema", default=[])
    batch_source: Optional[str]
    ttl: Optional[Period]
    tags: Dict[str, str] = {}

    def __init__(__pydantic_self__, **data: Any) -> None:
        if isinstance(data.get("ttl", None), str):
            data["ttl"] = Period.from_str(data.get("ttl", None))
        super().__init__(**data)


class FeatureView(BaseView):
    def get_feature_names(self):
        return [feature.name for feature in self.schemas]

    def get_feature_objects(self, is_numeric=False) -> Set[Feature]:
        return {
            Feature.create_feature_from_schema(schema, self.name)
            for schema in self.schemas
            if (schema.is_numeric() if is_numeric else True)
        }


class LabelView(BaseView):
    request_source: Optional[str]

    def get_label_names(self):
        return [label.name for label in self.schemas]

    def get_label_objects(self, is_numeric=False) -> Set[Feature]:
        return {
            Feature.create_label_from_schema(schema, self.name)
            for schema in self.schemas
            if (schema.is_numeric() if is_numeric else True)
        }
