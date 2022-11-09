from enum import Enum


class OfflineStoreType(str, Enum):
    """A constant numerate choices which is used to indicate how to initialize OfflineStore from configuration. If you want to add a new type of offline store, you definitely want to modify this."""

    FILE = "file"
    PGSQL = "pgsql"
    SPARK = "spark"
