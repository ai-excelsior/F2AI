from collections import defaultdict
import pandas as pd
import os
from typing import TYPE_CHECKING, Tuple
from copy import deepcopy
from pypika import Query, Parameter
from aie_feast.common.psl_utils import to_pgsql
from aie_feast.common.psl_utils import sql_df, psy_conn
from torch.utils.data import IterableDataset
from aie_feast.common.source import FileSource, SqlSource
from aie_feast.views import FeatureView
from aie_feast.period import Period

if TYPE_CHECKING:
    from aie_feast.offline_stores.offline_store import OfflineStore
    from aie_feast.service import Service

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"
CREATE_COL = "created_timestamp"
QUERY_COL = "query_timestamp"
SAM_TBL = "sampler_df"
ROW = "row_nbr"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"


class IterableDataset(IterableDataset):
    def __init__(
        self,
        fs: "OfflineStore",
        service: "Service",
        entity_index: pd.DataFrame,
        table_suffix: str = None,
        project_folder: str = None,
        batch: int = None,
        feature_views=None,
        label_views=None,
        join_keys=None,
    ):
        self.fs = fs
        self.service = service
        self.entity_index = entity_index
        self.table_suffix = table_suffix
        self.project_folder = project_folder
        self.feature_views = feature_views
        self.label_views = label_views
        self.batch = batch if batch else len(self.entity_index) // 10
        self.join_keys = join_keys

        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)

    def __iter__(self):
        for i in range(len(self.entity_index)):
            if i % self.batch == 0:  # batch merge
                self.get_context(i // self.batch)
            to_return = (
                self.data_sample[0].iloc[[i % self.batch]],
                self.data_sample[1].iloc[[i % self.batch]],
            )
            if not to_return[0].isnull().all().all() and not to_return[1].isnull().all().all():
                yield to_return

    def get_context(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_list = []
        label_list = []
        if self.fs.type == "file":
            entity = self.entity_index.iloc[i * self.batch : (i + 1) * self.batch]
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            to_drop = entity.columns
            source = FileSource(
                name=f"{self.service.name}_source",
                path=os.path.join(self.project_folder, self.service.materialize_path),
                timestamp_field=TIME_COL,
                created_timestamp_field=MATERIALIZE_TIME,
            )
            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs.get_period_features(
                        source=source,
                        entity_df=entity,
                        period=-Period.from_str(period),
                        features=features,
                        include=True,
                        ttl=self.service.ttl,
                        how="right",
                        join_keys=self.join_keys,
                    )
                    tmp_result.drop(columns=[TIME_COL], inplace=True)
                    tmp_result.rename(columns={QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_features(
                        source=source,
                        entity_df=entity,
                        features=features,
                        include=True,
                        ttl=self.service.ttl,
                        how="right",
                        join_keys=self.join_keys,
                    )
                feature_views_pd = feature_views_pd.merge(tmp_result, how="left", on=list(entity.columns))

            for period, features in self.all_labels.items():
                if period:
                    tmp_result = self.fs.get_period_features(
                        source=source,
                        entity_df=entity,
                        period=Period.from_str(period),
                        features=features,
                        include=True,
                        ttl=self.service.ttl,
                        how="right",
                        join_keys=self.join_keys,
                    )
                    tmp_result.drop(columns=[TIME_COL], inplace=True)
                    tmp_result.rename(columns={QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_features(
                        source=source,
                        entity_df=entity,
                        features=features,
                        include=True,
                        ttl=self.service.ttl,
                        how="right",
                        join_keys=self.join_keys,
                    )
                label_views_pd = label_views_pd.merge(tmp_result, how="left", on=list(entity.columns))
        elif self.fs.type == "pgsql":
            source = SqlSource(
                name=f"{self.service.name}",
                timestamp_field=TIME_COL,
                created_timestamp_field=MATERIALIZE_TIME,
            )
            conn = psy_conn(self.fs)
            entity = Query.from_(f"{SAM_TBL}_{self.table_suffix}").where(
                Parameter(f"row_nbr >= {i*self.batch} and row_nbr < {(i+1)*self.batch}")
            )
            feature_views_pd = entity.select(*self.join_keys, TIME_COL)
            label_views_pd = entity.select(*self.join_keys, TIME_COL)
            to_drop = self.join_keys
            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs.get_period_features(
                        source=source,
                        entity_df=entity.select(
                            *self.join_keys, Parameter(f"{TIME_COL} as {ENTITY_EVENT_TIMESTAMP_FIELD}")
                        ),
                        period=-Period.from_str(period),
                        features=features,
                        include=True,
                        join_keys=self.join_keys,
                        ttl=self.service.ttl,
                        return_df=False,
                    )
                    # tmp_result = Query.from_(tmp_result[0]).select( #TODO
                    #     *tmp_result[1], Parameter(f"{QUERY_COL} as {TIME_COL}"), *features
                    # )  # always merge on TIME_COL, remove cols not in feature schema
                else:
                    tmp_result = self.fs.get_features(
                        source=source,
                        entity_df=entity.select(
                            *self.join_keys, Parameter(f"{TIME_COL} as {ENTITY_EVENT_TIMESTAMP_FIELD}")
                        ),
                        features=features,
                        include=True,
                        join_keys=self.join_keys,
                        ttl=self.service.ttl,
                        return_df=False,
                    )
                feature_views_pd = (
                    Query.from_(feature_views_pd)
                    .left_join(tmp_result)
                    .using(Parameter(",".join(self.join_keys + [TIME_COL])))
                    .select(feature_views_pd.star, Parameter(",".join([f.name for f in features])))
                )

            for period, features in self.all_labels.items():
                if period:
                    tmp_result = self.fs.get_period_features(
                        source=source,
                        entity_df=entity.select(
                            *self.join_keys, Parameter(f"{TIME_COL} as {ENTITY_EVENT_TIMESTAMP_FIELD}")
                        ),
                        period=Period.from_str(period),
                        features=features,
                        include=True,
                        join_keys=self.join_keys,
                        ttl=self.service.ttl,
                        return_df=False,
                    )
                    # TODO
                    # tmp_result = Query.from_(tmp_result).select(
                    #     Parameter(
                    #         f"{','.join(self.entity_name + [ENTITY_EVENT_TIMESTAMP_FIELD]  +[feature.name for feature in features])}"
                    #     )
                    # )

                    #     *tmp_result[1], Parameter(f"{QUERY_COL} as {TIME_COL}"), *features
                    # )  # always merge on TIME_COL, remove cols not in feature schema
                else:
                    tmp_result = self.fs.get_features(
                        source=source,
                        entity_df=entity.select(
                            *self.join_keys, Parameter(f"{TIME_COL} as {ENTITY_EVENT_TIMESTAMP_FIELD}")
                        ),
                        features=features,
                        include=True,
                        join_keys=self.join_keys,
                        ttl=self.service.ttl,
                        return_df=False,
                    )
                label_views_pd = (
                    Query.from_(label_views_pd)
                    .left_join(tmp_result)
                    .using(*(self.join_keys + [TIME_COL]))
                    .select(label_views_pd.star, Parameter(",".join([f.name for f in features])))
                )  # tmp_result[1] + TIME_COL
            #     label_list += features
            # label_views_pd = Query.from_(label_views_pd).select(*self.entity_name, *label_list)

            feature_views_pd = pd.DataFrame(
                sql_df(feature_views_pd.get_sql(), conn),
                columns=self.join_keys
                + [TIME_COL]
                + [f.name for item in self.all_features.values() for f in item],
            )
            label_views_pd = pd.DataFrame(
                sql_df(label_views_pd.get_sql(), conn),
                columns=self.join_keys
                + [TIME_COL]
                + [f.name for item in self.all_labels.values() for f in item],
            )
        self.data_sample = (feature_views_pd.drop(columns=to_drop), label_views_pd.drop(columns=to_drop))
        return feature_views_pd.drop(columns=to_drop), label_views_pd.drop(columns=to_drop)

    def get_feature_period(self, service: "Service", with_labels=False) -> dict:
        """_summary_

        Args:
            service (Service): materialized service to construct
            with_labels (bool, optional): get labels or not

        Returns:
            Dict: {period1:[fea1,fea2],period2[fea5],0:[fea3,fea4]}, 0 means no period
        """
        period_dict = defaultdict(list)

        if with_labels:
            for label in service.get_label_objects(self.label_views):
                period = label.period.strip('"') if label.period else 0
                period_dict[period].append(label)  # TODO:period

        else:
            for feature in service.get_feature_objects(self.feature_views):
                period = feature.period.strip('"') if feature.period else 0
                period_dict[period].append(feature)

        return period_dict


class Dataset:
    def __init__(
        self,
        fs: "OfflineStore",
        service: "Service",
        sampler: callable,
        project_folder: str,
        feature_views,
        label_views,
    ):
        self.fs = fs
        self.service = service
        self.sampler = sampler
        self.project_folder = project_folder
        self.feature_views = feature_views
        self.label_views = label_views

    def to_pytorch(self, batch: int = None) -> IterableDataset:
        """convert to iterablt pytorch dataset really hold data"""
        entity_index = self.sampler()
        join_keys = list(entity_index.columns[:-1])
        table_suffix = None
        if self.fs.type == "pgsql":
            entity_index[ROW] = range(len(entity_index))
            table_suffix = to_pgsql(entity_index, SAM_TBL, self.fs)
        return IterableDataset(
            fs=self.fs,
            service=self.service,
            entity_index=entity_index,
            table_suffix=table_suffix,
            project_folder=self.project_folder,
            batch=batch,
            feature_views=self.feature_views,
            label_views=self.label_views,
            join_keys=join_keys,
        )
