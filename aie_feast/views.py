from typing import List, Optional, Dict
from aie_feast.definitions import Feature
from pydantic import BaseModel, Field


class FeatureView(BaseModel):
    name: str
    description: Optional[str]
    entities: List[str] = []
    schemas: List[Feature] = Field(alias="schema", default=[])
    batch_source: Optional[str]
    ttl: Optional[str]
    tags: Dict[str, str] = {}

    def get_feature_names(self):
        return [feature.name for feature in self.schemas]


class LabelView(BaseModel):
    name: str
    description: Optional[str]
    entities: List[str] = []
    schemas: List[Feature] = Field(alias="schema", default=[])
    batch_source: Optional[str]
    request_source: Optional[str]
    ttl: Optional[str]


# @dataclass
# class LabelView:
#     """realize object using one .yml"""

#     entity: List[str]
#     labels: List[str]
#     batch_source: str
#     ttl: str = field(default_factory=get_default_value)
#     request_source: str = field(default_factory=get_default_value)
