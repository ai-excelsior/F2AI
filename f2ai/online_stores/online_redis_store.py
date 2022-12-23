import json
import pickle
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from f2ai.common.utils import DateEncoder
from f2ai.definitions import FeatureView, OnlineStore, OnlineStoreType, Period, Source, online_store
from pydantic import PrivateAttr
from redis import Redis


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

    def write_batch(self, featrue_view: FeatureView, project_name: str, dt: pd.DataFrame):
        if self.client.hget(project_name, featrue_view.name) == None:
            zset_key = uuid.uuid4().hex[:8]
            self.client.hset(project_name, featrue_view.name, zset_key)
        else:
            zset_key = self.client.hget(project_name, featrue_view.name)
        zset_dict = {}
        for row in dt.to_dict("records"):
            if row.get("event_timestamp") == None:
                event_timestamp = datetime.now().timestamp()
            else:
                event_timestamp = row.get("event_timestamp").timestamp()
            zset_dict.setdefault(json.dumps(row, cls=DateEncoder), event_timestamp)
        self.client.zadd(name=zset_key, mapping=zset_dict)

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
        else:
            min_ttl_timestamp = pd.to_datetime("1970-01-01", utc=True)

        zset_key = self.client.hget(hkey.split(":")[0], hkey.split(":")[1])
        if zset_key:
            data = self.client.zrangebyscore(
                zset_key,
                min=min_ttl_timestamp.timestamp(),
                max=datetime.now().timestamp(),
                withscores=False,
            )
            columns = list(json.loads(data[0]).keys())
            batch_data_list = []
            {batch_data_list.append([json.loads(data[i])[key] for key in columns]) for i in range(len(data))}
            data = pd.DataFrame(batch_data_list, columns=columns)
            if period:
                min_period_timestamp = data["event_timestamp"].max() - period.to_pandas_dateoffset()
                data = data[data["event_timestamp"] >= min_period_timestamp]
            batch_data = pd.merge(entity_df, data, on=list(entity_df.columns), how="inner")
        else:
            batch_data = None
        return batch_data

    def set_up():
        pass
