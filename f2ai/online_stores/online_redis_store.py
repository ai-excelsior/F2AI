import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from f2ai.common.utils import DateEncoder
from f2ai.definitions import OnlineStore, OnlineStoreType, Period
from pydantic import PrivateAttr
from redis import Redis

DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"


class OnlineRedisStore(OnlineStore):
    type: OnlineStoreType = OnlineStoreType.REDIS
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    name: str

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

    def write_batch(self, name: str, project_name: str, dt: pd.DataFrame, ttl: Optional[Period] = None):
        if self.client.hget(project_name, name) is None:
            zset_key = uuid.uuid4().hex[:8]
            self.client.hset(name=project_name, key=name, value=zset_key)
        else:
            zset_key = self.client.hget(name=project_name, key=name)
        zset_dict = {}
        for row in dt.to_dict(orient="records"):
            event_timestamp = pd.to_datetime(row.get(DEFAULT_EVENT_TIMESTAMP_FIELD, datetime.now()), utc=True)
            zset_dict.setdefault(json.dumps(row, cls=DateEncoder), event_timestamp.timestamp())
        self.client.zadd(name=zset_key, mapping=zset_dict)
        if ttl is not None:
            expir_time = event_timestamp + ttl.to_py_timedelta()
            self.client.expireat(zset_key, expir_time)

    def read_batch(
        self, hkey: str, ttl: Optional[Period] = None, period: Optional[Period] = None
    ) -> pd.DataFrame:

        min_timestamp = max(
            (
                pd.to_datetime(datetime.now(), utc=True) - period.to_pandas_dateoffset()
                if period
                else pd.to_datetime(0, utc=True)
            ),
            (
                pd.to_datetime(datetime.now(), utc=True) - ttl.to_pandas_dateoffset()
                if ttl
                else pd.to_datetime(0, utc=True)
            ),
        )
        zset_key = self.client.hget(hkey.split(":")[0], hkey.split(":")[1])
        if zset_key:
            data = self.client.zrangebyscore(
                name=zset_key,
                min=min_timestamp.timestamp(),
                max=datetime.now().timestamp(),
                withscores=False,
            )
            columns = list(json.loads(data[0]).keys())
            batch_data_list = [[json.loads(data[i])[key] for key in columns] for i in range(len(data))]
            data = pd.DataFrame(data=batch_data_list, columns=columns)
        else:
            data = None
        return data

    def get_online_source(self):
        return self.name
