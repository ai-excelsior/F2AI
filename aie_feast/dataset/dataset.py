from collections import defaultdict
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from copy import deepcopy
from pypika import Query, Parameter
from aie_feast.common.psl_utils import to_pgsql
from aie_feast.common.psl_utils import sql_df, psy_conn

if TYPE_CHECKING:
    from aie_feast.featurestore import FeatureStore
    from aie_feast.service import Service

TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"
CREATE_COL = "created_timestamp"
QUERY_COL = "query_timestamp"
SAM_TBL = "sampler_df"
ROW = "row_nbr"


class IterableDataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service_name: str,
        entity_index: pd.DataFrame,
        table_suffix: str = None,
    ):
        self.fs = fs
        self.service_name = service_name
        self.entity_index = entity_index
        self.entity_name = list(self.entity_index.columns[:-1])
        self.service = self.fs.services[self.service_name]
        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)
        self.table_suffix = table_suffix

    def __iter__(self):
        for i in range(len(self.entity_index)):
            data_sample = self.get_context(i)
            if not data_sample[0].empty and not data_sample[1].empty:
                yield data_sample

    def get_context(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_list = []
        label_list = []
        if self.fs.connection.type == "file":
            entity = self.entity_index.iloc[[i]]
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            to_drop = entity.columns

            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs.get_period_features(self.service, entity, period, features, True)
                    tmp_result.rename({QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_features(self.service, entity, features, True)
                feature_views_pd = feature_views_pd.merge(tmp_result, how="inner", on=list(entity.columns))
                feature_list += features
            feature_views_pd = feature_views_pd[list(entity.columns) + feature_list]

            for period, features in self.all_labels.items():
                if period:
                    tmp_result = self.fs.get_period_labels(self.service, entity, period, False)
                    tmp_result.rename({QUERY_COL: TIME_COL}, inplace=True)  # always merge on TIME_COL
                else:
                    tmp_result = self.fs.get_labels(self.service, entity, True)
                label_views_pd = label_views_pd.merge(tmp_result, how="inner", on=list(entity.columns))
                label_list += features
            label_views_pd = label_views_pd[list(entity.columns) + label_list]

        elif self.fs.connection.type == "pgsql":
            conn = psy_conn(**self.fs.connection.__dict__)
            entity = (
                Query.from_(f"{SAM_TBL}_{self.table_suffix}")
                .select(*self.entity_name)
                .where(Parameter(f"row_nbr={i}"))
            )
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            to_drop = self.entity_name
            for period, features in self.all_features.items():
                if period:
                    tmp_result = self.fs._get_period_pgsql(
                        self.service, entity, period, features, True, False, self.entity_name
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
                        self.service, entity, period, features, False, True, self.entity_name
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

        return feature_views_pd.drop(columns=to_drop).dropna(how="all"), label_views_pd.drop(
            columns=to_drop
        ).dropna(how="all")

    def get_feature_period(self, service: "Service", is_label=False) -> dict:
        """_summary_

        Args:
            service (Service): materialized service to construct
            is_label (bool, optional): get labels or not

        Returns:
            Dict: {period1:[fea1,fea2],period2[fea5],0:[fea3,fea4]}, 0 means no period
        """
        period_dict = defaultdict(list)

        if is_label:
            for label in service.get_labels(self.fs.label_views):
                period = label.period if label.period else 0
                period_dict[period].append(label.name)

        else:
            for feature in service.get_features(self.fs.feature_views):
                period = feature.period if feature.period else 0
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

    def to_pytorch(self) -> IterableDataset:
        """convert to iterablt pytorch dataset really hold data"""
        entity_index = self.sampler()
        table_suffix = None
        if self.fs.connection.type == "pgsql":
            entity_index[ROW] = range(len(entity_index))
            table_suffix = to_pgsql(entity_index, SAM_TBL, **self.fs.connection.__dict__)
        return IterableDataset(
            self.fs,
            self.service_name,
            entity_index,
            table_suffix,
        )
