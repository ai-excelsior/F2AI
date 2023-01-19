import json
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Union
import functools
import pandas as pd
from pydantic import PrivateAttr
from redis import Redis, ConnectionPool

from ..common.utils import DateEncoder
from ..definitions import OnlineStore, OnlineStoreType, Period, FeatureView, Service, Entity
from ..common.time_field import *


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
        self,
        name: str,
        project_name: str,
        dt: pd.DataFrame,
        ttl: Optional[Period] = None,
        join_keys: List[str] = None,
    ):
        pipe = self.client.pipeline()
        if not dt.empty:
            for group_data in dt.groupby(join_keys):
                all_entities = functools.reduce(
                    lambda x, y: f"{x},{y}",
                    list(
                        map(
                            lambda x, y: x + ":" + y,
                            join_keys,
                            [group_data[0]] if isinstance(group_data[0], str) else list(group_data[0]),
                        )
                    ),
                )
                if self.client.hget(f"{project_name}:{name}", all_entities) is None:
                    zset_key = uuid.uuid4().hex
                    pipe.hset(name=f"{project_name}:{name}", key=all_entities, value=zset_key)
                else:
                    zset_key = self.client.hget(name=f"{project_name}:{name}", key=all_entities)
                    # remove data that has expired in `zset`` according to `score`
                if ttl is not None:
                    pipe.zremrangebyscore(
                        name=zset_key, min=0, max=(datetime.now() - ttl.to_py_timedelta()).timestamp()
                    )
                zset_dict = {
                    json.dumps(row, cls=DateEncoder): pd.to_datetime(
                        row.get(DEFAULT_EVENT_TIMESTAMP_FIELD, datetime.now()), utc=True
                    ).timestamp()
                    for row in group_data[1]
                    .drop(columns=[QUERY_COL], errors="ignore")
                    .to_dict(orient="records")
                }
                pipe.zadd(name=zset_key, mapping=zset_dict)
                if ttl is not None:  # add a general expire constrains on hash-key
                    expir_time = group_data[1][DEFAULT_EVENT_TIMESTAMP_FIELD].max() + ttl.to_py_timedelta()
                    pipe.expireat(zset_key, expir_time)
            pipe.execute()

    def read_batch(
        self,
        entity_df: pd.DataFrame,
        project_name: str,
        view: Union[Service, FeatureView],
        feature_views: Dict[str, FeatureView],
        entities: Dict[str, Entity],
        join_keys: List[str],
        **kwargs,
    ):
        if isinstance(view, FeatureView):
            data = self._read_batch(
                hkey=f"{project_name}:{view.name}",
                ttl=view.ttl,
                period=None,
                entity_df=entity_df[join_keys],
                **kwargs,
            )
            entity_df = pd.merge(entity_df, data, on=join_keys, how="inner") if not data.empty else None
        elif isinstance(view, Service):
            for featureview in view.features:
                fea_entities = functools.reduce(
                    lambda x, y: x + y,
                    [entities[entity].join_keys for entity in feature_views[featureview.view_name].entities],
                )
                fea_join_keys = [join_key for join_key in join_keys if join_key in fea_entities]
                feature_view_batch = self._read_batch(
                    hkey=f"{project_name}:{featureview.view_name}",
                    ttl=feature_views[featureview.view_name].ttl,
                    entity_df=entity_df[fea_join_keys],
                    period=featureview.period,
                    **kwargs,
                )
                if not feature_view_batch.empty:
                    entity_df = feature_view_batch.merge(entity_df, on=fea_join_keys, how="inner")
        else:
            raise TypeError("online read only allow FeatureView and Service")

        return entity_df

    def _read_batch(
        self,
        hkey: str,
        ttl: Optional[Period] = None,
        period: Optional[Period] = None,
        entity_df: pd.DataFrame = None,
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
        dt_group = entity_df.groupby(list(entity_df.columns))
        all_entities = [
            functools.reduce(
                lambda x, y: f"{x},{y}",
                list(
                    map(
                        lambda x, y: x + ":" + y,
                        list(entity_df.columns),
                        [group_data[0]] if isinstance(group_data[0], str) else list(group_data[0]),
                    )
                ),
            )
            for group_data in dt_group
        ]
        all_zset_key = self.client.hmget(hkey, all_entities)
        if all_zset_key:
            data = [  # newest record
                self.client.zrevrangebyscore(
                    name=zset_key,
                    min=min_timestamp.timestamp(),
                    max=datetime.now().timestamp(),
                    withscores=False,
                    start=0,
                    num=1,
                )
                for zset_key in all_zset_key
                if zset_key
            ]
            columns = list(json.loads(data[0][0]).keys())
            batch_data_list = [[json.loads(data[i][0])[key] for key in columns] for i in range(len(data))]
            data = pd.DataFrame(data=batch_data_list, columns=columns)
        else:
            data = pd.DataFrame()
        return data
