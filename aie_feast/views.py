from typing import List
from dataclasses import dataclass, field


@dataclass
class FeatureViews:
    """realize object using one .yml"""

    entity: List[str]
    features: List[str]
    batch_source: str
    ttl: str = field(default_factory=str)
    exogenous: bool = field(default_factory=bool)
    request_source: str = field(default_factory=str)


@dataclass
class LabelViews:
    """realize object using one .yml"""

    entity: List[str]
    labels: List[str]
    batch_source: str
    ttl: str = field(default_factory=str)
    request_source: str = field(default_factory=str)
