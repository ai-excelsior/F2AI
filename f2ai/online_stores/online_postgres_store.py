from ..definitions.online_store import OnlineStore, OnlineStoreType


class OnlinePostgresStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.PGSQL
    pass
