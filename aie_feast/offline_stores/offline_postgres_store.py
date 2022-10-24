from pydantic import Field
from .offline_store import OfflineStore, OfflineStoreType


class OfflinePostgresStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.PGSQL

    host: str
    port: str = "5432"
    database: str = "postgres"
    db_schema: str = Field(alias="schema", default="public")
    user: str
    password: str
