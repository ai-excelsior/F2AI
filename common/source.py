from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SourceConfig:
    """realize the data sources(not database) relation"""

    name: str
    type: str = field(default_factory=str)
    file_format: int = field(default_factory=int)
    file_path: str = field(default_factory=str)
    request_features: Dict = field(default_factory=dict)
