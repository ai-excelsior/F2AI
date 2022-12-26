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

    def write_batch(self, name: str, project_name: str, dt: pd.DataFrame, ttl: Period = None):
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
        self,
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
        else:
            data = None
        return data

    def get_online_source(self):
        return self.name
