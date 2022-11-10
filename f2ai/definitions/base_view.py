from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .features import FeatureSchema
from .period import Period


class BaseView(BaseModel):
    """Abstraction of common part of FeatureView and LabelView."""

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
