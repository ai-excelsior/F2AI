import pandas as pd
import os
import json
import docker
from datetime import datetime
from aie_feast.common.jinja import jinja_env
from typing import Dict, List, Union
from pypika import Query, Parameter, functions
from aie_feast.common.source import FileSource, SqlSource
from aie_feast.offline_stores.offline_file_store import OfflineFileStore
from aie_feast.offline_stores.offline_postgres_store import OfflinePostgresStore
from aie_feast.views import FeatureView, LabelView
from aie_feast.service import Service
from aie_feast.dataset.dataset import Dataset
from aie_feast.common.get_config import (
    get_offline_store_from_cfg,
    get_service_cfg,
    get_entity_cfg,
    get_label_views,
    get_feature_views,
    get_source_cfg,
)
from aie_feast.common.utils import to_file, build_agg_query, remove_prefix
from aie_feast.common.psl_utils import execute_sql, psy_conn, to_pgsql, close_conn, sql_df
from aie_feast.period import Period


TIME_COL = "event_timestamp"  # timestamp of action taken in original tables or period-query result, or query time in single-query result table
CREATE_COL = "created_timestamp"  # timestamp of record of the action taken in original tables, or timestamp of action taken in single-query result table
QUERY_COL = "query_timestamp"  # only use in period query, query time in period-query result table
TMP_TBL = "entity_df"  # temp table upload in database
MATERIALIZE_TIME = "materialize_time"  # timestamp to done materialize, only used in materialized result
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"


class FeatureStore:
    def __init__(self, project_folder=None, url=None, token=None, projectID=None):
        if project_folder:
            self.offline_store = get_offline_store_from_cfg(os.path.join(project_folder, "feature_store.yml"))
        elif url and token and projectID:
            pass  # TODO: realize in future
        else:
            raise ValueError("one of config file or meta server project should be provided")

        # init each object using .yml in corresponding folders
        self.project_folder = project_folder
        self.sources = get_source_cfg(os.path.join(project_folder, "sources"), self.offline_store.type)
        self.entities = get_entity_cfg(os.path.join(project_folder, "entities"))
        self.feature_views = get_feature_views(os.path.join(project_folder, "feature_views"))
        self.label_views = get_label_views(os.path.join(project_folder, "label_views"))
        self.services = get_service_cfg(os.path.join(project_folder, "services"))

        # for file source, modify the path if it is not a absolute path
        for _, source in self.sources.items():
            if isinstance(source, FileSource) and not os.path.isabs(source.path):
                source.path = os.path.join(self.project_folder, source.path)

    def __check_format(self, entity_df):
        if isinstance(entity_df, pd.DataFrame):
            # TODO: Remove this constraint in future
            assert (
                len(entity_df.columns) >= 1 and entity_df.columns[-1] == TIME_COL
            ), "Check entity_df make sure it has at least 1 columns and event_timestamp at the last column"

    def __check_fns(self, fn):
        assert fn in [
            "mean",
            "sum",
            "std",
            "mode",
            "median",
            "min",
            "max",
            "unique",
        ], f"{fn}is not a available function, you can use fs.query() to customize your function"

    def _get_available_features(
        self, view: Union[FeatureView, Service], is_numeric: bool = False
    ) -> List[str]:
        if isinstance(view, FeatureView):
            features = view.get_feature_objects(is_numeric)
        elif isinstance(view, Service):
            features = view.get_feature_objects(self.feature_views, is_numeric)
        else:
            raise TypeError("must be FeatureViews or Service")

        return [feature.name for feature in features]

    def _get_available_labels(self, view: Union[LabelView, Service], is_numeric: bool = False):
        if isinstance(view, LabelView):
            labels = view.get_label_objects(is_numeric)
        elif isinstance(view, Service):
            labels = view.get_label_objects(self.label_views, is_numeric)
        else:
            raise TypeError("must be LabelViews or Service")

        return [label.name for label in labels]

    def _get_available_entity_names(self, view) -> List[str]:
        entities = []
        if isinstance(view, FeatureView):
            entities = list(view.entities)
        elif isinstance(view, LabelView):
            entities = list(view.entities)
        elif isinstance(view, Service):
            entities = list(view.get_entities(self.feature_views, self.label_views))
        else:
            raise TypeError("must be FeatureViews, LabelViews or Service")
        return entities

    def _get_views(self, view_name):
        if view_name in self.feature_views.keys():
            return self.feature_views[view_name]
        elif view_name in self.label_views.keys():
            return self.label_views[view_name]
        elif view_name in self.services.keys():
            return self.services[view_name]
        else:
            raise ValueError("Can't find any views/services")

    def get_features(
        self,
        feature_view: str,
        entity_df: Union[pd.DataFrame, str],
        features: list = None,
        include: bool = True,
        **kwargs,
    ):
        """non-series prediction use: get `features` of `entity_df` from `feature_views`

        Args:
            feature_view : Single FeatureViews or Service(after materialzed) name to lookup.
            entity_df (pd.DataFrame): condition.
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional):  include timestamp defined in `entity_df` or not. Defaults to True.
        """

        self.__check_format(entity_df)
        feature_view = self._get_views(feature_view)

        if not features:
            features = self._get_available_features(feature_view)

        if self.offline_store.type == "file":
            return self._get_point_record(feature_view, entity_df, features, include, **kwargs)
        elif self.offline_store.type == "pgsql":
            table_suffix = to_pgsql(entity_df, TMP_TBL, self.offline_store)
            # connect to pgsql db
            conn = psy_conn(self.offline_store)
            sql_result, entity_name, features = self._get_point_pgsql(
                feature_view, f"{TMP_TBL}_{table_suffix}", features, include, entity_df.columns
            )
            result = pd.DataFrame(
                sql_df(sql_result.get_sql(), conn), columns=entity_name + [TIME_COL] + features
            )
            # remove entity_df and close connection
            close_conn(conn, tables=[f"{TMP_TBL}_{table_suffix}"])
            return result

    def get_period_features(
        self,
        feature_view: str,
        entity_df: pd.DataFrame,
        period: str,
        features: List = None,
        include: bool = True,
        **kwargs,
    ):
        """time_series prediction use: get past `period` length `features` of `entity_df` from `feature_views`

        Args:
            feature_views:Single FeatureViews or Service(after materialzed) to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            period (str): length of look_back
            features (List, optional): features to return. Defaults to None means all features.
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        feature_view = self._get_views(feature_view)
        self.__check_format(entity_df)
        period = Period.from_str(period)

        if not features:
            features = self._get_available_features(feature_view)

        if self.offline_store.type == "file":
            return self._get_period_record(feature_view, entity_df, period, features, include, **kwargs)
        elif self.offline_store.type == "pgsql":
            table_suffix = to_pgsql(entity_df, TMP_TBL, self.offline_store)
            # connect to pgsql db
            conn = psy_conn(self.offline_store)
            sql_result, entity_cols = self._get_period_pgsql(
                feature_view,
                f"{TMP_TBL}_{table_suffix}",
                period,
                features,
                include,
                list(entity_df.columns[:-1]),
            )
            result = pd.DataFrame(
                sql_df(sql_result.get_sql(), conn), columns=entity_cols + [QUERY_COL, TIME_COL] + features
            )
            # remove entity_df and close connection
            close_conn(conn, tables=[f"{TMP_TBL}_{table_suffix}"])
            return result

    def get_labels(self, label_view, entity_df: pd.DataFrame, include: bool = True, **kwargs):
        """non-time series prediction use: get labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) name to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        labels = self._get_available_labels(label_view)

        if self.offline_store.type == "file":
            return self._get_point_record(label_view, entity_df, labels, include, **kwargs)
        elif self.offline_store.type == "pgsql":
            table_suffix = to_pgsql(entity_df, TMP_TBL, self.offline_store)
            # connect to pgsql db
            conn = psy_conn(self.offline_store)
            sql_result, entity_name, features = self._get_point_pgsql(
                label_view, f"{TMP_TBL}_{table_suffix}", labels, include, entity_df.columns
            )
            result = pd.DataFrame(
                sql_df(sql_result.get_sql(), conn),
                columns=entity_name + [TIME_COL] + features,
            )
            # remove entity_df and close connection
            close_conn(conn, tables=[f"{TMP_TBL}_{table_suffix}"])
            return result

    def get_period_labels(
        self, label_view, entity_df: pd.DataFrame, period: str, include: bool = False, **kwargs
    ):
        """time series prediction use: get from `start` to `end` length labels of `entity_df` from `label_views`

        Args:
            label_views:Single LabelViews or Service(after materialzed) name to lookup. Defaults to None.
            entity_df (pd.DataFrame): condition
            period (str): length of look_forward, can be negative, egg, -1 days
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to False.
        """
        self.__check_format(entity_df)
        label_view = self._get_views(label_view)
        labels = self._get_available_labels(label_view)

        if self.offline_store.type == "file":
            return self._get_period_record(label_view, entity_df, period, labels, include, **kwargs)
        elif self.offline_store.type == "pgsql":
            table_suffix = to_pgsql(entity_df, TMP_TBL, self.offline_store)
            # connect to pgsql db
            conn = psy_conn(self.offline_store)
            sql_result, entity_cols = self._get_period_pgsql(
                label_view,
                f"{TMP_TBL}_{table_suffix}",
                period,
                labels,
                include,
                list(entity_df.columns[:-1]),
            )
            result = pd.DataFrame(
                sql_df(sql_result.get_sql(), conn), columns=entity_cols + [QUERY_COL, TIME_COL] + labels
            )
            # remove entity_df and close connection
            close_conn(conn, tables=[f"{TMP_TBL}_{table_suffix}"])
            return result

    def _get_point_record(
        self, view, entity_df: pd.DataFrame, features: list, include: bool = True, **kwargs
    ):
        """non time-series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews/Service to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            features(list):columns to select besides times and entities
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        assert isinstance(
            view, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        assert isinstance(self.offline_store, OfflineFileStore), "only OfflineFileStore supportted "

        avaliable_entity_names = self._get_available_entity_names(view)
        join_keys = list(
            {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
                if join_key in entity_df.columns
            }
        )

        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
            assert isinstance(source, FileSource), "only work for file source in _get_point_record"
        else:
            source = FileSource(
                name=f"{view.name}_source",
                path=os.path.join(self.project_folder, view.materialize_path),
                timestamp_field=TIME_COL,
                created_timestamp_field=MATERIALIZE_TIME,
            )

        if isinstance(view, FeatureView):
            buildin_features = view.get_feature_objects()
        elif isinstance(view, LabelView):
            buildin_features = view.get_label_objects()
        else:  # Service
            buildin_features = view.get_feature_objects(self.feature_views) | view.get_label_objects(
                self.label_views
            )

        if features:
            features = set(features)
            features = [feature for feature in buildin_features if feature.name in features]

        return self.offline_store.get_features(
            entity_df=entity_df,
            features=features,
            source=source,
            join_keys=join_keys,
            ttl=view.ttl,
            include=include,
            **kwargs,
        )

    def _get_point_pgsql(
        self,
        views,
        table_name: str,
        features: list,
        include: bool = True,
        entity_columns: list = None,
        **kwargs,
    ):

        assert isinstance(
            views, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        assert isinstance(self.offline_store, OfflinePostgresStore), "only OfflinePostgresStore supportted "

        avaliable_entity_names = self._get_available_entity_names(views)
        entity_names = list(
            {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
                if join_key in entity_columns
            }
        )
        table_name = SqlSource(name=table_name, timestamp_field=TIME_COL)
        if isinstance(views, (FeatureView, LabelView)):
            join_keys = entity_names
            source = self.sources[views.batch_source]
            assert isinstance(source, SqlSource), "only work for sql source in _get_point_pgsql"
        else:
            source = SqlSource(
                name=f"{views.name}",
                timestamp_field=TIME_COL,
                created_timestamp_field=MATERIALIZE_TIME,
            )
        ###
        if isinstance(views, FeatureView):
            buildin_features = views.get_feature_objects()
        elif isinstance(views, LabelView):
            buildin_features = views.get_label_objects()
        else:  # Service
            buildin_features = views.get_feature_objects(self.feature_views) | views.get_label_objects(
                self.label_views
            )
        if features:
            features = set(features)
            features = [feature for feature in buildin_features if feature.name in features]

        return (
            self.offline_store.get_features(
                query=table_name,
                features=features,
                source=source,
                join_keys=join_keys,
                ttl=views.ttl,
                include=include,
                **kwargs,
            ),
            join_keys,
            [f.name for f in features],
        )

    def _get_period_pgsql(
        self,
        view: Union[FeatureView, LabelView, Service],
        table_name: str,
        period: Period,
        features: list,
        include: bool = True,
        entity_df_columns: list = None,
    ):
        # avaliable_entity_names = self._get_available_entity_names(view)
        # entity_names = [
        #     entity_name for entity_name in avaliable_entity_names if entity_name in entity_df_columns
        # ]

        if isinstance(view, (FeatureView, LabelView)):
            assert self.sources[
                view.batch_source
            ].timestamp_field, "View is not time-relevant, no period to get"

            entity_cols = [
                join_key
                for entity_name in view.entities
                for join_key in self.entities[entity_name].join_keys
                if join_key in entity_df_columns
            ]
            time_cols = (
                [
                    f"{self.sources[view.batch_source].timestamp_field} as {TIME_COL}_tmp",
                    f"{self.sources[view.batch_source].created_timestamp_field} as {CREATE_COL}",
                ]
                if self.sources[view.batch_source].created_timestamp_field
                else [f"{self.sources[view.batch_source].timestamp_field} as  {TIME_COL}_tmp"]
            )
            create_time = CREATE_COL
            df = (
                Query.from_(view.batch_source)
                .select(Parameter(",".join(entity_cols + time_cols + features)))
                .as_("df")
            )
        else:
            entities = view.get_entities(self.feature_views, self.label_views)
            entity_cols = [
                join_key
                for entity in entities
                for join_key in entity.join_keys
                if join_key in entity_df_columns
            ]
            time_cols = [f"{TIME_COL} as {TIME_COL}_tmp", MATERIALIZE_TIME]

            create_time = MATERIALIZE_TIME
            df = (
                Query.from_(view.materialize_path)
                .select(Parameter(",".join(entity_cols + time_cols + features)))
                .as_("df")
            )

        if entity_cols:
            sql_join = (
                Query.from_(Query.from_(table_name).select(Parameter(",".join(entity_cols + [TIME_COL]))))
                .inner_join(df)
                .using(*entity_cols)
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        else:
            sql_join = (
                Query.from_(table_name)
                .cross_join(df)
                .cross()
                .select(Parameter(f"df.*, {TIME_COL}"))
                .as_("sql_join")
            )
        sql_query = self._get_window_pgsql(sql_join, period, include)
        sql_result = (
            Query.from_(sql_query).select(
                Parameter(
                    ",".join(
                        entity_cols
                        + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                        + features
                    )
                ),
            )
            if len(time_cols) == 1
            else (
                Query.from_(
                    Query.from_(sql_query).select(
                        sql_query.star,
                        Parameter(
                            f"row_number() over (partition by ({','.join(entity_cols+[TIME_COL,TIME_COL+'_tmp'])}) order by {create_time} DESC)"
                        ),
                    )
                )
                .select(
                    Parameter(
                        ",".join(
                            entity_cols
                            + [f"{TIME_COL} as {QUERY_COL}", f"{TIME_COL}_tmp as {TIME_COL}"]
                            + features
                        )
                    )
                )
                .where(Parameter("row_number=1"))
            )
        )
        return [sql_result, entity_cols]

    def _get_period_record(
        self,
        view,
        entity_df: pd.DataFrame,
        period: Period,
        features: list,
        include: bool = True,
        **kwargs,
    ):

        """series prediction use

        Args:
            views (List, optional): FeatureViews/LabelViews to lookup.
            entity_df (pd.DataFrame): condition. Defaults to None.
            period: period to get data.
            features:features/labels to return
            include (bool, optional): include timestamp defined in `entity_df` or not. Defaults to True.
        """
        assert isinstance(
            view, (FeatureView, LabelView, Service)
        ), "only allowed FeatureView, LabelView and Service"
        assert isinstance(self.offline_store, OfflineFileStore), "only OfflineFileStore supportted "

        avaliable_entity_names = self._get_available_entity_names(view)

        join_keys = list(
            {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
                if join_key in entity_df.columns
            }
        )
        # TODO: support multi join keys in future
        if isinstance(view, (FeatureView, LabelView)):
            source = self.sources[view.batch_source]
            assert isinstance(source, FileSource), "only work for file source in _get_point_record"
        else:
            source = FileSource(
                name=f"{view.name}_source",
                path=os.path.join(self.project_folder, view.materialize_path),
                timestamp_field=TIME_COL,
                created_timestamp_field=MATERIALIZE_TIME,
            )

        if isinstance(view, FeatureView):
            buildin_features = view.get_feature_objects()
        elif isinstance(view, LabelView):
            buildin_features = view.get_label_objects()
        else:  # Service
            buildin_features = view.get_feature_objects(self.feature_views)
        if features:
            features = set(features)
            features = [feature for feature in buildin_features if feature.name in features]

        return self.offline_store.get_period_features(
            entity_df=entity_df,
            features=features,
            source=source,
            period=period,
            join_keys=join_keys,
            ttl=view.ttl,
            include=include,
            **kwargs,
        )

    def _get_window_pgsql(
        self,
        join,
        period: Period,
        include: bool = True,
    ):
        # look back
        if period.is_neg:
            sql_query = (
                Query.from_(join)
                .select(join.star)
                .where(
                    Parameter(
                        f" ({TIME_COL}_tmp::timestamptz <= {TIME_COL}::timestamptz) and ({TIME_COL}_tmp::timestamptz > {TIME_COL}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                    if include
                    else Parameter(
                        f" ({TIME_COL}_tmp::timestamptz < {TIME_COL}::timestamptz) and ({TIME_COL}_tmp::timestamptz >= {TIME_COL}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                )
                .as_("sql_query")
            )
        else:
            sql_query = (
                Query.from_(join)
                .select(join.star)
                .where(
                    Parameter(
                        f" ({TIME_COL}_tmp::timestamptz >= {TIME_COL}::timestamptz) and ({TIME_COL}_tmp::timestamptz < {TIME_COL}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                    if include
                    else Parameter(
                        f" ({TIME_COL}_tmp::timestamptz > {TIME_COL}::timestamptz) and ({TIME_COL}_tmp::timestamptz <= {TIME_COL}::timestamptz + {period.to_pgsql_interval()})  "
                    )
                )
                .as_("sql_query")
            )

        return sql_query

    def materialize(self, service_name: str, incremental_begin: str = None):
        """incrementally join `views` to generate tables

        Args:
            service_name (str): name of service to materialize
            incremental_begin (str): begin of materialization
                `None`: all-data for type=file, otherwise last materialzation time
                date-like str: corresponding date, e.g.: `2020-01-03 00:09:08`
                int-like + fre str:  latest int freq, e.g.: `30 days`

        """

        if self.offline_store.type == "file":
            self._offline_record_materialize(self.services[service_name], incremental_begin)
        elif self.offline_store.type == "pgsql":
            self._offline_pgsql_materialize(self.services[service_name], incremental_begin)

    def _offline_pgsql_materialize(self, service: Service, incremental_begin):
        try:
            incremental_begin = pd.to_datetime(incremental_begin, utc=True) if incremental_begin else None
        except Exception:
            incremental_begin = Period.from_str(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        # dir to store dbt project
        label_view: LabelView = service.get_label_view(self.label_views)
        label_view_dict = label_view.dict()
        label_view_dict.update(
            {
                "labels": [label.name for label in label_view.get_label_objects()],
                "event_time": self.sources[label_view.batch_source].timestamp_field,
                "create_time": self.sources[label_view.batch_source].created_timestamp_field,
            }
        )

        all_features_use = [feature.name for feature in service.get_feature_objects(self.feature_views)]

        feature_views = []
        for feature_view in service.get_feature_views(self.feature_views):
            feature_view_dict = feature_view.dict()
            feature_view_dict.update(
                {
                    "features": [
                        feature_name
                        for feature_name in feature_view.get_feature_names()
                        if feature_name in all_features_use and feature_name not in label_view_dict["labels"]
                    ],
                    "event_time": self.sources[feature_view.batch_source].timestamp_field,
                    "create_time": self.sources[feature_view.batch_source].created_timestamp_field,
                }
            )
            feature_views.append(feature_view_dict)

        entity_names = self._get_available_entity_names(service)
        entities_dict = {entity_name: self.entities[entity_name].join_keys[0] for entity_name in entity_names}

        conn = psy_conn(self.offline_store)
        max_timestamp = Query.from_(service.materialize_path).select(
            functions.Max(Parameter(label_view_dict["event_time"]))
        )
        max_timestamp_label = Query.from_(label_view.batch_source).select(
            functions.Max(Parameter(label_view_dict["event_time"]))
        )

        label_result = pd.to_datetime(sql_df(max_timestamp_label.get_sql(), conn)[0][0])
        try:
            result = pd.to_datetime(sql_df(max_timestamp.get_sql(), conn)[0][0])
        except:
            result = pd.to_datetime("1970-01-01 00:00:00", utc=True)

        conn.close()

        if incremental_begin is None:
            incremental_begin = result.tz_localize(None)
        elif isinstance(incremental_begin, Period):
            incremental_begin = label_result - incremental_begin.to_py_timedelta()
        else:
            incremental_begin = incremental_begin

        dbt_path = os.path.join(
            remove_prefix(self.project_folder, "file://"),
            f"{service.dbt_path}",
            f"{service.materialize_path}",
        )
        pd.to_datetime
        dict_var = {
            "labelviews": label_view_dict,
            "featureviews": feature_views,
            "entities": entities_dict,
            "increment_begin": str(incremental_begin),
        }
        json_var = json.dumps(dict_var)
        self.schedule_local_dbt_container(service.materialize_path, json_var, dbt_path)
        # os.system(f"cd {dbt_path} && dbt run --profiles-dir {dbt_path} --vars '{json_var}' ")

    def schedule_local_dbt_container(self, profile_name: str, vars: Dict, dbt_path: str):
        docker_client = docker.from_env()
        dbt_profiles = jinja_env.get_template("profiles.yaml").render(
            profile=profile_name,
            host=self.offline_store.host,
            port=self.offline_store.port,
            user=self.offline_store.user,
            password=self.offline_store.password,
            database=self.offline_store.database,
            db_schema=self.offline_store.db_schema,
        )
        profile_path = os.path.join(dbt_path, "profiles.yml")
        if not os.path.exists(profile_path):
            with open(profile_path, "w") as f:
                f.write(dbt_profiles)
        docker_client.containers.run(
            "ghcr.io/dbt-labs/dbt-postgres:1.3.latest",
            command=f"run --vars '{vars}' ",
            volumes={
                dbt_path: {"bind": "/usr/app", "mode": "rw"},
                profile_path: {"bind": "/root/.dbt/profiles.yml", "mode": "rw"},
            },
            network="host",
            remove=True,
        )

    def _offline_record_materialize(self, service: Service, incremental_begin):
        """materialize offline file

        Args:
            service (Service): service entity
            incremental_begin: time to begin materialize
        """
        try:
            incremental_begin = pd.to_datetime(incremental_begin if incremental_begin else 0, utc=True)
        except Exception:
            incremental_begin = Period.from_str(incremental_begin)
        except:
            raise TypeError("please check your `incremental_begin` type")

        joined_frame = self.offline_store.materialize(
            service=service,
            feature_views=self.feature_views,
            label_views=self.label_views,
            sources=self.sources,
            entities=self.entities,
            incremental_begin=incremental_begin,
        )
        joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.now(), utc=True)
        to_file(
            joined_frame,
            os.path.join(self.project_folder, f"{service.materialize_path}"),
            f"{service.materialize_path}".split(".")[-1],
        )
        print(
            f"materialize done, file saved at {os.path.join(self.project_folder, service.materialize_path)}"
        )

    def stats(
        self,
        view: str,
        entity_df: pd.DataFrame = None,
        features: List[str] = None,
        group_key: List[str] = None,
        fn: str = "mean",
        start: str = None,
        end: str = None,
        include: str = "both",
        keys_only: bool = False,
    ):
        """get from `start` to `end` statistical `fn` results of `entity_df` from `views`, only work for numeric features varied with time

        Args:
            views (List): name of view to look up
            entity_df (pd.DataFrame,optional), if given, ignore `start` and `end`. Defaults to None, has the supreme priority.
            group_key (list): joined-columns to do stats,  only works when `entity_df` is None. if None, means do stats on joined-entities, also accept `[]` means no grouping.
            fn (str, optional): statistical method, min, max, std, avg, mode, median. Defaults to "mean".
            start (str, optional): start_time. Defaults to None, works and only works when `entity_df` is None.
            end (str, optional): end_time. Defaults to None, works and only works when `entity_df` is None.
            include (str, optional): whether to include `start` or `end` timestamp
            keys_only (bool, optional): whether to take action on keys, only available when fn=unique, return a list
        """
        self.__check_fns(fn)
        view = self._get_views(view)
        avaliable_entity_names = self._get_available_entity_names(view)

        if entity_df is not None:
            self.__check_format(entity_df)
            entities = (
                list(entity_df.columns[:-1])
                if len(entity_df.columns[:-1])
                else [
                    item
                    for entity_name in avaliable_entity_names
                    for item in self.entities[entity_name].join_keys
                ]
            )
            entity_df[TIME_COL] = pd.to_datetime(entity_df[TIME_COL], utc=True)
            start = pd.to_datetime(0, utc=True)
        else:
            entities = (
                group_key
                if group_key is not None  # group_key can be empty list
                else [
                    item
                    for entity_name in avaliable_entity_names
                    for item in self.entities[entity_name].join_keys
                ]
            )
            entity_df = pd.DataFrame(columns=[TIME_COL])
            entity_df[TIME_COL] = [
                pd.to_datetime(end, utc=True) if end else pd.to_datetime(datetime.now(), utc=True)
            ]
            start = pd.to_datetime(start, utc=True) if start else pd.to_datetime(0, utc=True)

        join_keys = list(
            {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
                if join_key in entities
            }
        )
        if isinstance(view, FeatureView):
            buildin_features = view.get_feature_objects(fn != "unique")
        elif isinstance(view, LabelView):
            buildin_features = view.get_label_objects(fn != "unique")
        else:  # Service
            buildin_features = view.get_features(self.feature_views, fn != "unique")

        if features:
            features = set(features)
            features = [feature for feature in buildin_features if feature.name in features]
        else:
            features = buildin_features

        if keys_only:
            assert fn == "unique", "keys_only=True can only be applied when fn=unique"
            assert join_keys, "no key available for keys_only=True"
            features = []

        if self.offline_store.type == "file":
            if isinstance(view, (FeatureView, LabelView)):
                source = self.sources[view.batch_source]
                assert isinstance(source, FileSource), "only work for file source in _get_point_record"
                assert source.timestamp_field, "stats can only apply on time relative data"
            else:
                source = FileSource(
                    name=f"{view.name}_source",
                    path=os.path.join(self.project_folder, view.materialize_path),
                    timestamp_field=TIME_COL,
                    created_timestamp_field=MATERIALIZE_TIME,
                )
            return self.offline_store.stats(
                entity_df=entity_df,
                features=features,
                source=source,
                fn=fn,
                start=start,
                group_keys=join_keys,
                include=include,
                keys_only=keys_only,
                join_keys=len(entity_df.columns[:-1]),
            )
        elif self.offline_store.type == "pgsql":
            conn = psy_conn(self.offline_store)
            table_suffix = to_pgsql(entity_df, TMP_TBL, self.offline_store)
            table_name = SqlSource(name=f"{TMP_TBL}_{table_suffix}", timestamp_field=TIME_COL)
            if isinstance(view, (FeatureView, LabelView)):
                source = self.sources[view.batch_source]
                assert isinstance(source, SqlSource), "only work for Sql source"
                assert source.timestamp_field, "stats can only apply on time relative data"
            else:
                source = SqlSource(
                    name=f"{view.name}",
                    timestamp_field=TIME_COL,
                    created_timestamp_field=MATERIALIZE_TIME,
                )
            result_sql = self.offline_store.stats(
                entity_df=table_name,
                features=features,
                source=source,
                fn=fn,
                start=start,
                group_keys=join_keys,
                include=include,
                keys_only=keys_only,
                join_keys=len(entity_df.columns[:-1]),
            )
            result = pd.DataFrame(
                sql_df(result_sql.get_sql(), conn), columns=[f"{c.name}_{fn}" for c in features] + join_keys
            )

            close_conn(conn, [f"{TMP_TBL}_{table_suffix}"])
            return result

    def get_latest_entities(self, view: str, entity: Union[List[str], pd.DataFrame] = None):
        """get latest entity and its timestamp from a single FeatureViews/LabelViews or a materialzed Service
        entity can either be None(all joined-entities in view), entity names or entity value(specific entities)

        Args:
            views (List): view to look up
        """
        view = self._get_views(view)
        avaliable_entity_names = self._get_available_entity_names(view)
        if isinstance(entity, pd.DataFrame):
            entities = entity.columns
        elif entity:
            entities = entity
        else:
            entities = {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
            }
        join_keys = list(
            {
                join_key
                for entity_name in avaliable_entity_names
                for join_key in self.entities[entity_name].join_keys
                if join_key in entities
            }
        )

        if self.offline_store.type == "file":
            if isinstance(view, (FeatureView, LabelView)):
                source = self.sources[view.batch_source]
                assert isinstance(source, FileSource), "only work for File source"
                assert source.timestamp_field, "get_latest_entities can only apply on time relative data"
            else:
                source = FileSource(
                    name=f"{view.name}_source",
                    path=os.path.join(self.project_folder, view.materialize_path),
                    timestamp_field=TIME_COL,
                    created_timestamp_field=MATERIALIZE_TIME,
                )
            return self.offline_store.get_latest_entities(
                source=source,
                group_keys=join_keys,
                entities=entity if isinstance(entity, pd.DataFrame) else None,
            )

        elif self.offline_store.type == "pgsql":
            if isinstance(entity, pd.DataFrame):
                table_suffix = to_pgsql(entity, TMP_TBL, self.offline_store)
                table_name = SqlSource(name=f"{TMP_TBL}_{table_suffix}")
            else:
                table_name = None
            conn = psy_conn(self.offline_store)
            if isinstance(view, (FeatureView, LabelView)):
                source = self.sources[view.batch_source]
                assert isinstance(source, SqlSource), "only work for Sql source"
                assert source.timestamp_field, "get_latest_entities can only apply on time relative data"
            else:
                source = SqlSource(
                    name=f"{view.name}", timestamp_field=TIME_COL, created_timestamp_field=MATERIALIZE_TIME
                )
            result_sql = self.offline_store.get_latest_entities(
                source=source, group_keys=join_keys, entities=table_name
            )
            result = pd.DataFrame(sql_df(result_sql.get_sql(), conn), columns=join_keys + [TIME_COL])
            close_conn(conn, [f"{TMP_TBL}_{table_suffix}"])
            return result

    def get_dataset(self, service_name: str, sampler: callable = None) -> Dataset:
        """get from `start` to `end` length data for training from `views`

        Args:
            service_name(str): name of `SERVICE` to use
            sampler (callable, optional): sampler
        """
        return Dataset(
            fs=self,
            service_name=service_name,
            sampler=sampler,
        )

    def query(self, query: str = None, return_df: bool = True):
        """customized query, only works when connection.type != 'file'

        Args:
            query (str, optional): full sql query to execute in db
        """
        assert (
            self.offline_store.type != "file"
        ), "query doesnt work for file type project, you can manualy read local files in pandas"
        if self.offline_store.type == "pgsql":
            conn = psy_conn(self.offline_store)
            if return_df:
                result = pd.DataFrame(sql_df(query, conn))
            else:
                execute_sql(query, conn)
                result = None
                conn.commit()
            close_conn(conn)
            return result
