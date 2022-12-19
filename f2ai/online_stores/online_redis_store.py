import uuid
from typing import Any, List, Optional, Set

import pandas as pd
from redis import Redis

from ..definitions.online_store import Feature, FeatureView, OnlineStore, OnlineStoreType, Period, Source


class OnlineRedisStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.REDIS
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""

    def __init__(self, host: str, port: int, db: int, password: str):
        self.client = Redis(host=host, port=port, db=db, password=password)

    def write_batch(self, featrue_view: FeatureView, project_name: str, dt: pd.DataFrame) -> Source:
        set_key = uuid.uuid4().hex[:8]
        self.client.hset(project_name, featrue_view.name, set_key)
        return super().write_batch(featrue_view, project_name, dt)

    def read_batch(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: Source,
        join_keys: List[str] = ...,
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        return super().read_batch(entity_df, features, source, join_keys, ttl, include, **kwargs)

    def read_period_batch(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: Source,
        period: Period,
        join_keys: List[str] = ...,
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        return super().read_period_batch(
            entity_df, features, source, period, join_keys, ttl, include, **kwargs
        )
