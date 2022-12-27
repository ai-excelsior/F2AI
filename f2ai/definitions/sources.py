from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

from ..common.utils import read_file
from .offline_store import OfflineStoreType
from .features import FeatureSchema


class Source(BaseModel):
    """An abstract class which describe the common part of a Source. A source usually defines where to access data and what the time semantic it has. In F2AI, we have 2 kinds of time semantic:

    1. timestamp_field: the event timestamp which represent when the record happened, which is the main part of point-in-time join.
    2. created_timestamp_field: the created timestamp which represent when the record created, which usually happened in multi cycles of feature generation scenario.
    """

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
    file_format: FileFormatEnum = FileFormatEnum.CSV
    path: str

    def read_file(self, str_cols: List[str] = [], keep_cols: List[str] = []):
        time_columns = [self.timestamp_field]
        if self.created_timestamp_field:
            time_columns.append(self.created_timestamp_field)

        return read_file(
            self.path,
            file_format=self.file_format,
            parse_dates=time_columns,
            str_cols=str_cols,
            keep_cols=keep_cols,
        )


class RequestSource(Source):
    schemas: List[FeatureSchema] = Field(alias="schema")


class SqlSource(Source):
    query: str

    def __init__(__pydantic_self__, **data: Any) -> None:

        query = data.pop("query", "")
        if query == "":
            query = data.get("name")

        super().__init__(**data, query=query)


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
