from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from aie_feast.definitions import FeatureSchema, OfflineStoreType


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
    schemas: List[FeatureSchema] = Field(alias="schema")


class SqlSource(Source):
    query: str


def parse_source_yaml(o: Dict, offline_store_type: OfflineStoreType) -> Source:
    if o.get("type", None) == "request_source":
        return RequestSource(**o)

    if offline_store_type == OfflineStoreType.FILE:
        return FileSource(**o)
    elif offline_store_type == OfflineStoreType.PGSQL:
        return SqlSource(**o)
    elif offline_store_type == OfflineStoreType.SPARK:
        raise Exception("spark is not supported yet!")
    else:
        return Source(**o)
