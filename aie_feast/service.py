from typing import Dict, Any, List, Union
from .views import LabelViews, FeatureViews


class Service:
    """compose of FeatureViews and LabelViews used in service"""

    def __init__(self, features: List[FeatureViews], labels: List[LabelViews]):
        pass


class ForecastService(Service):
    """compose of FeatureViews and LabelViews used in forecast_service"""

    def __init__(self, features: List[FeatureViews], labels: List[LabelViews]):
        pass
