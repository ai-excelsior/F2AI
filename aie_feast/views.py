from typing import List
from dataclasses import dataclass, field
from aie_feast.common.utils import get_default_value


@dataclass
class FeatureViews:
    """realize object using one .yml"""

    entity: List[str]
    features: List[str]
    batch_source: str
    ttl: str = field(default_factory=get_default_value)
    exogenous: bool = field(default_factory=get_default_value)
    request_source: str = field(default_factory=get_default_value)


@dataclass
class LabelViews:
    """realize object using one .yml"""

    entity: List[str]
    labels: List[str]
    batch_source: str
    ttl: str = field(default_factory=get_default_value)
    request_source: str = field(default_factory=get_default_value)
