from re import S
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from copy import deepcopy
from pypika import Query, Tables, Parameter
from common.psl_utils import to_pgsql

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
    ):
        self.fs = fs
        self.service_name = service_name
        self.entity_col = list(entity_index.columns[:-2])

        if self.fs.connection.type == "file":
            self.entity_index = entity_index
        elif self.fs.connection.type == "pgsql":
            self.entity_index = Tables(f"{SAM_TBL}")

        self.service = self.fs.service[self.service_name]
        self.all_features = self.get_feature_period(self.service)
        self.all_labels = self.get_feature_period(self.service, True)

    def __iter__(self):
        for i in range(len(self.entity_index)):
            data_sample = self.get_context(i)
            if not data_sample[0].empty and not data_sample[1].empty:
                yield data_sample

    def get_context(self, i: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.fs.connection.type == "file":
            entity = self.entity_index.iloc[[i]]
            feature_views_pd = deepcopy(entity)
            label_views_pd = deepcopy(entity)
            for period, features in self.all_features.items():
                if period:
                    feature_views_pd.rename({TIME_COL: QUERY_COL}, inplace=True)
                feature_views_pd = feature_views_pd.merge(
                    self.fs.get_period_features(self.service, entity, period, features, True)
                    if period
                    else self.fs.get_features(self.service, entity, features, True),
                    how="inner",
                    on=list(entity.columns),
                )
            for period, features in self.all_labels.items():
                if period:
                    label_views_pd.rename({TIME_COL: QUERY_COL}, inplace=True)
                label_views_pd = label_views_pd.merge(
                    self.fs.get_period_labels(self.service, entity, period, True)
                    if period
                    else self.fs.get_labels(self.service, entity, True),
                    how="inner",
                    on=list(entity.columns),
                )

        elif self.fs.connection.type == "pgsql":
            entity = [
                (
                    Query.from_(SAM_TBL)
                    .select(Parameter("*"))
                    .where(Parameter(f"{ROW} = {i}"))
                    .as_("entity_df")
                ),
                self.entity_col,
            ]
            for period, features in self.all_features.items():
                if period:
                    features_result = Query.from_(entity).inner_join()

                feature_views_pd = (
                    self.fs.get_period_features(self.service, entity, period, features, True)
                    if period
                    else self.fs.get_features(self.service, entity, features, True),
                )

            for period, features in self.all_labels.items():
                if period:
                    label_views_pd.rename({TIME_COL: QUERY_COL}, inplace=True)
                label_views_pd = label_views_pd.merge(
                    self.fs.get_period_labels(self.service, entity, period, True)
                    if period
                    else self.fs.get_labels(self.service, entity, True),
                    how="inner",
                    on=list(entity.columns),
                )
            feature_views_pd, feature_views_pd = pd.DataFrame()
        return feature_views_pd.drop(columns=entity.columns).dropna(how="all"), label_views_pd.drop(
            columns=entity.columns
        ).dropna(how="all")

    def get_feature_period(self, service: "Service", is_label=False) -> dict:
        """_summary_

        Args:
            service (Service): materialized service to construct
            is_label (bool, optional): get labels or not

        Returns:
            Dict: {period1:[fea1,fea2],period2[fea5],0:[fea3,fea4]}, 0 means no period
        """
        period_dict = {}
        if is_label:
            for table, cols in service.labels.items():
                for fea in cols:
                    for k, v in fea.items():
                        v = v if v else 0
                        k = [k] if k != "__all__" else list(self.fs.features[table].labels.keys())
                        if v in period_dict:
                            period_dict[v] = period_dict[v] + k
                        else:
                            period_dict[v] = k
        else:
            for table, cols in service.features.items():
                for fea in cols:
                    for k, v in fea.items():
                        v = v if v else 0
                        k = [k] if k != "__all__" else list(self.fs.features[table].features.keys())
                        if v in period_dict:
                            period_dict[v] = period_dict[v] + k
                        else:
                            period_dict[v] = k
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
        if self.fs.connection.type == "pgsql":
            entity_index[ROW] = range(len(entity_index))
            to_pgsql(entity_index, SAM_TBL, **self.fs.connection.__dict__)
        return IterableDataset(
            self.fs,
            self.service_name,
            entity_index,
        )
