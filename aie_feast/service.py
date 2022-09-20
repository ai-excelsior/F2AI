from typing import Dict, Any, List, Union
from .views import LabelViews, FeatureViews
from dataclasses import dataclass, field


@dataclass
class Service:
    """compose of FeatureViews and LabelViews used in service for non-time-series task"""

    features: Dict
    labels: Dict


@dataclass
class ForecastService:
    """compose of FeatureViews and LabelViews used in service for time-series task"""

    features: Dict
    labels: Dict
    look_back: int
    look_forward: int
