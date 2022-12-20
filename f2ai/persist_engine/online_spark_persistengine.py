from f2ai.definitions import OnlinePersistEngine, OnlinePersistEngineType


class OnlineSparkPersistEngine(OnlinePersistEngine):
    type: OnlinePersistEngine = OnlinePersistEngineType.DISTRIBUTE
