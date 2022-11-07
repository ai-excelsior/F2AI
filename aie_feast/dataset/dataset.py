from collections import defaultdict
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from copy import deepcopy
from pypika import Query, Parameter
from aie_feast.common.psl_utils import to_pgsql
from aie_feast.common.psl_utils import sql_df, psy_conn
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from aie_feast.featurestore import FeatureStore
    from aie_feast.service import Service

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"
CREATE_COL = "created_timestamp"
QUERY_COL = "query_timestamp"
SAM_TBL = "sampler_df"
ROW = "row_nbr"


class IterableDataset(IterableDataset):
    def __init__(
        self,
        fs: "FeatureStore",
        service_name: str,
        entity_index: pd.DataFrame,
        table_suffix: str = None,
        batch: int = None,
    ):
        self.fs = fs
        self.service_name = service_name
        self.entity_index = entity_index
        self.entity_name = list(self.entity_index.columns[:-1])
        self.service = self.fs.services[self.service_name]
        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)
        self.table_suffix = table_suffix
        self.batch = batch if batch else len(self.entity_index) // 10

    def __iter__(self):
        if self.fs.offline_store.type == "file":
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
        if self.fs.offline_store.type == "file":
            entity = self.entity_index.iloc[i * self.batch : (i + 1) * self.batch]
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            to_drop = entity.columns

            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs.get_period_features(
                        self.service_name, entity, period, features, True, how="right"
                    )
                    tmp_result.drop(columns=[TIME_COL], inplace=True)
                    tmp_result.rename(columns={QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_features(self.service_name, entity, features, True, how="right")
                feature_views_pd = feature_views_pd.merge(tmp_result, how="left", on=list(entity.columns))

            for period, features in self.all_labels.items():
                if period:
                    tmp_result = self.fs.get_period_labels(
                        self.service_name, entity, period, False, how="right"
                    )
                    tmp_result.drop(columns=[TIME_COL], inplace=True)
                    tmp_result.rename(columns={QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_labels(self.service_name, entity, True, how="right")
                label_views_pd = label_views_pd.merge(tmp_result, how="left", on=list(entity.columns))

        elif self.fs.offline_store.type == "pgsql":
            conn = psy_conn(self.fs.offline_store)
            entity = (
                Query.from_(f"{SAM_TBL}_{self.table_suffix}")
                .select(*self.entity_name)
                .where(Parameter(f"row_nbr >= {i*self.batch} and row_number < {(i+1)*self.batch}"))
            )
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            to_drop = self.entity_name
            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs._get_period_pgsql(
                        self.service, entity, -period, features, True, self.entity_name
                    )
                    tmp_result[0] = Query.from_(tmp_result[0]).select(
                        *tmp_result[1], Parameter(f"{QUERY_COL} as {TIME_COL}"), *features
                    )  # always merge on TIME_COL, remove cols not in feature schema
                else:
                    tmp_result = self.fs._get_point_pgsql(
                        self.service, entity, features, True, self.entity_name
                    )
                feature_views_pd = (
                    Query.from_(feature_views_pd)
                    .inner_join(tmp_result[0])
                    .using(*self.entity_name)  # tmp_result[1] + TIME_COL
                    .select("*")
                )
                feature_list += features
            feature_views_pd = Query.from_(feature_views_pd).select(*self.entity_name, *feature_list)

            for period, features in self.all_labels.items():
                if period:
                    tmp_result = self.fs._get_period_pgsql(
                        self.service, entity, period, features, False, self.entity_name
                    )
                    tmp_result[0] = Query.from_(tmp_result[0]).select(
                        *tmp_result[1], Parameter(f"{QUERY_COL} as {TIME_COL}"), *features
                    )  # always merge on TIME_COL, remove cols not in feature schema
                else:
                    tmp_result = self.fs._get_point_pgsql(
                        self.service, entity, features, True, self.entity_name
                    )
                label_views_pd = (
                    Query.from_(label_views_pd).inner_join(tmp_result[0]).using(*self.entity_name).select("*")
                )  # tmp_result[1] + TIME_COL
                label_list += features
            label_views_pd = Query.from_(label_views_pd).select(*self.entity_name, *label_list)

            feature_views_pd = pd.DataFrame(
                sql_df(feature_views_pd.get_sql(), conn), columns=self.entity_name + feature_list
            )
            label_views_pd = pd.DataFrame(
                sql_df(label_views_pd.get_sql(), conn), columns=self.entity_name + label_list
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
            for label in service.get_label_objects(self.fs.label_views):
                period = label.period.strip('"') if label.period else 0
                period_dict[period].append(label.name)

        else:
            for feature in service.get_feature_objects(self.fs.feature_views):
                period = feature.period.strip('"') if feature.period else 0
                period_dict[period].append(feature.name)

        return period_dict


class Dataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service_name: str,
        sampler: callable = None,
    ):
        self.fs = fs
        self.service_name = service_name
        self.sampler = sampler

    def to_pytorch(self, batch: int = None) -> IterableDataset:
        """convert to iterablt pytorch dataset really hold data"""
        entity_index = self.sampler()
        table_suffix = None
        if self.fs.offline_store.type == "pgsql":
            entity_index[ROW] = range(len(entity_index))
            table_suffix = to_pgsql(entity_index, SAM_TBL, self.fs.offline_store)
        return IterableDataset(self.fs, self.service_name, entity_index, table_suffix, batch)
