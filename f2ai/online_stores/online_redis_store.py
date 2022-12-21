import uuid
import pickle
import pandas as pd
from redis import Redis
from typing import Optional
from pydantic import PrivateAttr
from datetime import datetime
from f2ai.definitions import FeatureView, OnlineStore, OnlineStoreType, Period, Source, online_store


class OnlineRedisStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.REDIS
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""

    _cilent: Optional[Redis] = PrivateAttr(default=None)

    @property
    def client(self):

        if self._cilent is None:
            self._cilent = Redis(
                password=self.password,
                host=self.host,
                port=self.port,
                db=self.db,
            )

        return self._cilent

    def write_batch(self, featrue_view: FeatureView, project_name: str, dt: pd.DataFrame) -> Source:
        set_key = uuid.uuid4().hex[:8]
        self.client.hset(project_name, featrue_view.name, set_key)
        return super().write_batch(featrue_view, project_name, dt)

    def read_batch(
        self,
        entity_df: pd.DataFrame,
        hkey: str,
        ttl: Optional[Period] = None,
        period: Optional[Period] = None,
        **kwargs,
    ) -> pd.DataFrame:

        if ttl:
            min_ttl_timestamp = pd.to_datetime(datetime.now(), utc=True) - ttl.to_pandas_dateoffset()

        zset_key = self.client.hget(hkey.split(":")[0], hkey.split(":")[1])
        data = self.client.zrangebyscore(
            zset_key, min=min_ttl_timestamp, max=pd.to_datetime(datetime.now(), utc=True), withscores=False
        )
        columns = list(pickle.loads(data[0]).keys())
        batch_data_list = []
        {batch_data_list.append([pickle.loads(data[i])[key] for key in columns]) for i in range(len(data))}
        data = pd.DataFrame(batch_data_list, columns=columns)
        if period:
            min_period_timestamp = data["event_timestamp"].max() - period.to_pandas_dateoffset()
            data = data[data["event_timestamp"] >= min_period_timestamp]
        batch_data = pd.merge(entity_df, data, on=entity_df.columns, how="inner")

        return batch_data

    def set_up():
        pass


if __name__ == "__main__":
    online = OnlineRedisStore(host="localhost", port=6379, db=0, password="")
    # online.client.hset("guizhou_traffic", "a", "1")
    pass
