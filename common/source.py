from dataclasses import dataclass, field
from typing import Dict, List
from .utils import get_default_value


@dataclass
class SourceConfig:
    """realize the data sources(not database) relation"""

    name: str
    type: str = field(default_factory=get_default_value)
    file_format: int = field(default_factory=get_default_value)
    path: str = field(default_factory=get_default_value)
    request_features: Dict = field(default_factory=get_default_value)
    description: str = ''
    schema: List[Dict[str, str]] = field(default_factory=list)
