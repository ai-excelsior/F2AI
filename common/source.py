from dataclasses import dataclass, field
from typing import Dict, List
from .utils import get_default_value


@dataclass
class SourceConfig:
    """realize the data sources(not database) relation"""

    name: str
    event_time: str
    create_time: str = field(default_factory=get_default_value)
    type: str = field(default_factory=get_default_value)
    file_format: int = field(default_factory=get_default_value)
    file_path: str = field(default_factory=get_default_value)
    request_features: List[Dict[str, str]] = field(default_factory=list)
    description: str = ""
    tags: List[dict] = field(default_factory=get_default_value)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.type == "file_source":
            assert self.file_format, "file_format is required"
            assert self.file_path, "file path is required"
        if self.type == "request_source":
            assert self.request_features, "request_features is required"
