from f2ai.definitions import OnlinePersistEngine, OnlinePersistEngineType, OnlineStore


class OnlineLocalPersistEngine(OnlinePersistEngine):
    type: OnlinePersistEngineType = OnlinePersistEngineType.LOCAL

    store: None  # need modify
