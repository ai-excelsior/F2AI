import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from f2ai.common.utils import DateEncoder
from f2ai.definitions import OnlineStore, OnlineStoreType, Period
from pydantic import PrivateAttr
from redis import Redis, ConnectionPool

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
            pool = ConnectionPool(
                host=self.host, port=self.port, db=self.db, password=self.password, decode_responses=True
            )
            self._cilent = Redis(connection_pool=pool)

        return self._cilent

    def write_batch(
        self, name: str, project_name: str, dt: pd.DataFrame, ttl: Optional[Period] = None, **kwargs
    ):
        pipe = self.client.pipeline()
        if self.client.hget(project_name, name) is None:
            zset_key = uuid.uuid4().hex[:8]
            pipe.hset(name=project_name, key=name, value=zset_key)
        else:
            zset_key = self.client.hget(name=project_name, key=name)
            # remove data that has expired in `zset`` according to `score`
            if ttl is not None:
                pipe.zremrangebyscore(
                    name=zset_key, min=0, max=(datetime.now() - ttl.to_py_timedelta()).timestamp()
                )
        if not dt.empty:
            zset_dict = {
                json.dumps(sorted(row.items(), key=lambda x: x[0]), cls=DateEncoder): pd.to_datetime(
                    row.get(DEFAULT_EVENT_TIMESTAMP_FIELD, datetime.now()), utc=True
                ).timestamp()
                for row in dt.to_dict(orient="records")
            }
            pipe.zadd(name=zset_key, mapping=zset_dict)
            if ttl is not None:  # add a general expire constrains on hash-key
                expir_time = dt[DEFAULT_EVENT_TIMESTAMP_FIELD].max() + ttl.to_py_timedelta()
                pipe.expireat(zset_key, expir_time)
        pipe.execute()
        kwargs["signal"].send(1)

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
