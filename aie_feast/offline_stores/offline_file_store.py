from .offline_store import OfflineStore, OfflineStoreType


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE
    pass
