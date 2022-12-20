from f2ai.definitions import OfflinePersistEngine, OfflinePersistEngineType


class OfflineSparkPersistEngine(OfflinePersistEngine):
    type: OfflinePersistEngine = OfflinePersistEngineType.SPARK
