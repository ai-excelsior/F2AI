from typing import Dict, Any, List, Union


class ForecastConfig:
    context_period: str
    predict_period: str
    features: List[str]
    labels: List[str]


class EntityConfig:
    join_keys: str
