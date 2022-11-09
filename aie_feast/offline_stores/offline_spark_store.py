from ..definitions import OfflineStore, OfflineStoreType


class OfflineSparkStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.SPARK
    pass
