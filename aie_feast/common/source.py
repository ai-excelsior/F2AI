from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from aie_feast.definations import Feature, OfflineStoreType
from .utils import get_default_value


class Source(BaseModel):
    name: str
    description: Optional[str]
    timestamp_field: Optional[str]
    created_timestamp_field: Optional[str] = Field(alias="created_timestamp_column")
    tags: Dict[str, str] = {}


class FileFormatEnum(str, Enum):
    PARQUET = "parquet"
    TSV = "tsv"
    CSV = "csv"
    TEXT = "text"


class FileSource(Source):
    file_format: FileFormatEnum
    path: str


class RequestSource(Source):
    features: List[Feature] = Field(alias="schema")


class SqlSource(Source):
    query: str


def parse_source_yaml(o: Dict, offline_store_type: OfflineStoreType) -> Source:
    if offline_store_type == OfflineStoreType.FILE:
        return FileSource(**o)
    elif offline_store_type == OfflineStoreType.PGSQL:
        return SqlSource(**o)
    elif offline_store_type == OfflineStoreType.SPARK:
        raise Exception("spark is not supported yet!")
    else:
        return Source(**o)


@dataclass
class SourceConfig:
    """realize the data sources(not database) relation"""

    name: str
    event_time: str = field(default_factory=get_default_value)
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
