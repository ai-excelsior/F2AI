from ..definitions.online_store import OnlineStore, OnlineStoreType


class OnlineSparkStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.SPARK
    pass
