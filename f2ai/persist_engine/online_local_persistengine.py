from f2ai.definitions import OnlinePersistEngine, OnlinePersistEngineType, OnlineStore
from typing import List, Dict
import pandas as pd


class OnlineLocalPersistEngine(OnlinePersistEngine):
    type: OnlinePersistEngineType = OnlinePersistEngineType.LOCAL
    store: None  # need modify

    def materialize(
        self,
        save_path,
        feature_views: List[Dict],
        label_view: Dict,
        start: pd.Timestamp = None,
        end: pd.Timestamp = None,
        **kwargs,
    ):
        pass
