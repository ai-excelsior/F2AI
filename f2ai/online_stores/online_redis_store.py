import uuid
from typing import Optional

import pandas as pd
from redis import Redis

from ..definitions.online_store import FeatureView, OnlineStore, OnlineStoreType, Period, Source


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
        hkey: str,
        ttl: Optional[Period] = None,
        **kwargs,
    ) -> pd.DataFrame:

        if ttl:
            min_entity_timestamp = entity_df["event_timestamp"].min() - ttl.to_pandas_dateoffset()

        zset_key = self.client.hget(hkey)
        data = self.client.zrange(zset_key, start=0, end=-1, withscores=True)
        data = data[data["event_timestamp"] >= min_entity_timestamp]

        return data
