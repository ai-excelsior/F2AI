from typing import Dict, Any, List, Union


class FeatureViews:
    """realize object using one .yml"""

    pass


class LabelViews:
    """realize object using one .yml"""

    pass


class Service:
    """compose of FeatureViews and LabelViews used in service"""

    def __init__(self, features: List[FeatureViews], labels: List[LabelViews]):
        pass


class ForcastService(Service):
    """compose of FeatureViews and LabelViews used in forecast_service"""

    def __init__(self, features: List[FeatureViews], labels: List[LabelViews]):
        pass
