from typing import Dict
from dataclasses import dataclass, field
from common.utils import get_default_value


@dataclass
class Service:
    """compose of FeatureViews and LabelViews used in service for non-time-series task"""

    features: Dict
    labels: Dict
    materialize_path: str = field(default_factory=get_default_value)
    dbt_path: str = field(default_factory=get_default_value)


@dataclass
class ForecastService:
    """compose of FeatureViews and LabelViews used in service for time-series task"""

    features: Dict
    labels: Dict
    look_back: int
    look_forward: int
