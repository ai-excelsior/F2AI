from ..definitions.online_store import OnlineStore, OnlineStoreType


class OnlineRedisStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.REDIS
    pass
