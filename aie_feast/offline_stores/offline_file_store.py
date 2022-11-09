import pandas as pd
from typing import List, Optional, Set, Dict
from aie_feast.common.utils import get_stats_result
from aie_feast.definitions import (
    Feature,
    Entity,
    Period,
    FeatureView,
    LabelView,
    Service,
    OfflineStoreType,
    OfflineStore,
    FileSource,
)


DEFAULT_EVENT_TIMESTAMP_FIELD = "event_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
QUERY_COL = "query_timestamp"
TIME_COL = "event_timestamp"
MATERIALIZE_TIME = "materialize_time"


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE

    def get_offline_source(self, service: Service) -> FileSource:
        return FileSource(
            name=f"{service.name}_source",
            path=service.materialize_path,
            timestamp_field="event_timestamp",
            created_timestamp_field="materialize_time",
        )

    def materialize(
        self,
        service: Service,
        feature_views: Dict[str, FeatureView],
        label_views: Dict[str, LabelView],
        sources: Dict[str, FileSource],
        entities: Dict[str, Entity],
        start: str = None,
        end: str = None,
        fromnow: str = None,
    ):

        all_cols_name = service.get_feature_names(feature_views) | service.get_label_names(label_views)
        label_view = service.get_label_views(label_views)[0]
        labels = service.get_label_objects(label_views)
        join_keys = list(
            {
                join_key
                for entity_name in service.get_label_entities(label_views)
                for join_key in entities[entity_name].join_keys
            }
        )
        source = sources[label_view.batch_source]
        joined_frame = self._read_file(source=source, features=labels, join_keys=list(join_keys))
        # create timestamp makes no sense to labels
        joined_frame.drop(columns=source.created_timestamp_field, inplace=True, errors="ignore")
        if isinstance(incremental_begin, Period):
            incremental_begin = (
                joined_frame[DEFAULT_EVENT_TIMESTAMP_FIELD].max() - incremental_begin.to_pandas_dateoffset()
            )
            joined_frame = joined_frame[joined_frame[DEFAULT_EVENT_TIMESTAMP_FIELD] >= incremental_begin]
        else:
            joined_frame = joined_frame[joined_frame[DEFAULT_EVENT_TIMESTAMP_FIELD] >= incremental_begin]
        # if isinstance(incremental_begin, Period):
        #     incremental_begin = joined_frame[TIME_COL].max() - incremental_begin.to_pandas_dateoffset()
        #     joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]
        # else:
        #     joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]
        if start and end:
            joined_frame = joined_frame[(joined_frame[TIME_COL] >= start) & (joined_frame[TIME_COL] <= end)]
        elif fromnow:
            joined_frame = joined_frame[joined_frame[TIME_COL] >= fromnow]

        # join features dataframe
        for feature_view in service.get_feature_views(feature_views):
            feature_name = [
                n
                for n in feature_view.get_feature_names()
                if n in all_cols_name and n not in joined_frame.columns
            ]
            if feature_name:  # this view has new features other than those in joined_frame
                features = [n for n in feature_view.get_feature_objects() if n.name in feature_name]
                join_keys = list(
                    {
                        join_key
                        for entity_name in service.get_feature_entities(feature_view)
                        for join_key in entities[entity_name].join_keys
                    }
                )
                source = sources[feature_view.batch_source]
                joined_frame = self.get_features(
                    entity_df=joined_frame,
                    features=features,
                    source=source,
                    join_keys=join_keys,
                    ttl=feature_view.ttl,
                    include=True,
                    how="right",
                )

        # joined_frame[MATERIALIZE_TIME] = pd.to_datetime(datetime.now(), utc=True)
        # to_file(
        #     joined_frame,
        #     os.path.join(self.project_folder, f"{service.materialize_path}"),
        #     f"{service.materialize_path}".split(".")[-1],
        # )
        # print(
        #     f"materialize done, file saved at {os.path.join(self.project_folder, service.materialize_path)}"
        # )
        return joined_frame

    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        source_df = self._read_file(
            source=source, features=features, join_keys=join_keys + kwargs.pop("entity_cols", [])
        )

        return self._point_in_time_join(
            entity_df=entity_df,
            source_df=source_df,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )

    def get_period_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        period: Period,
        join_keys: List[str] = [],
        ttl: Optional[Period] = None,
        include: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        source_df = self._read_file(
            source=source, features=features, join_keys=join_keys + kwargs.pop("entity_cols", [])
        )

        return self._point_on_time_join(
            entity_df=entity_df,
            source_df=source_df,
            period=period,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs,
        )

    def stats(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        start,
        fn: str = "mean",
        group_keys: list = [],
        include: str = "both",
        keys_only: bool = False,
        join_keys: bool = False,
    ) -> pd.DataFrame:
        source_df = self._read_file(source=source, features=features, join_keys=group_keys)

        feature_columns = [feature.name for feature in features]

        entity_df = entity_df.rename(columns={DEFAULT_EVENT_TIMESTAMP_FIELD: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={source.timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if join_keys:
            df = source_df.merge(entity_df, on=group_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        if keys_only:
            return df[group_keys].groupby(group_keys).size().reset_index().drop(columns=[0])

        if group_keys:
            result = df.groupby(group_keys).apply(
                get_stats_result,
                fn,
                use_cols=feature_columns,
                include=include,
                start=start,
            )
        else:
            result = get_stats_result(
                df,
                fn,
                use_cols=feature_columns,
                include=include,
                start=start,
            )
        return result

    def get_latest_entities(
        self, source: FileSource, group_keys: list, entity_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """_summary_

        Args:
            source (FileSource): data source of featureview
            group_keys (list, optional): dimension of stats. Defaults to [].
            entity_df (pd.DataFrame, optional): query condition specified by users. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        source_df = self._read_file(source=source, features=[], join_keys=group_keys)
        if entity_df is not None:
            source_df = source_df.merge(entity_df, on=group_keys, how="inner")

        source_df = source_df[group_keys + [source.timestamp_field]]
        df = source_df.sort_values(by=source.timestamp_field, ascending=False, ignore_index=True)
        return df.drop_duplicates(subset=group_keys, keep="first")

    def query(self, *args, **kwargs) -> None:
        assert False, "query is not implemented for OfflineFileStore"

    def _read_file(self, source: FileSource, features: Set[Feature] = {}, join_keys: List[str] = []):
        feature_columns = [feature.name for feature in features]

        return source.read_file(
            str_cols=join_keys,
            keep_cols=feature_columns,
        )

    @classmethod
    def _point_in_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        timestamp_field: Optional[str] = None,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how: str = "inner",
    ) -> pd.DataFrame:
        """_summary_

        Args:
            entity_df (pd.DataFrame): dataframe condition specified by users
            source_df (pd.DataFrame): dataframe taken from data
            created_timestamp_field (Optional[str], optional): timestamp of upload of datas. Defaults to None.
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
            how (str, optional): merge method. Defaults to "inner".

        Returns:
            pd.DataFrame: _description_
        """
        # renames to keep things simple
        if timestamp_field:
            entity_df = entity_df.rename(
                columns={DEFAULT_EVENT_TIMESTAMP_FIELD: ENTITY_EVENT_TIMESTAMP_FIELD}
            )
            source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field:
            entity_df = entity_df.drop(columns=[created_timestamp_field], errors="ignore")

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - ttl.to_pandas_dateoffset()
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how=how)
        else:
            df = source_df.merge(entity_df, how="cross")

        if timestamp_field:
            df = cls._point_in_time_filter(df, include=include, ttl=ttl)
            df = cls._point_in_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(
            columns=[SOURCE_EVENT_TIMESTAMP_FIELD, created_timestamp_field], errors="ignore"
        ).rename(columns={ENTITY_EVENT_TIMESTAMP_FIELD: DEFAULT_EVENT_TIMESTAMP_FIELD})

    @classmethod
    def _point_on_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        period: Period,
        timestamp_field: Optional[str],
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[Period] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how: str = "inner",
    ) -> pd.DataFrame:
        """_summary_

        Args:
            entity_df (pd.DataFrame): dataframe condition specified by users
            source_df (pd.DataFrame): dataframe taken from data
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            created_timestamp_field (Optional[str], optional): timestamp of upload of datas. Defaults to None.
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            join_keys (List[str], optional): intersection of entities defined in yaml and entity_df.columns. Defaults to [].
            include (bool, optional): whether take user specified time in consideration. Defaults to True.
            how (str, optional): merge method. Defaults to "inner".

        Returns:
            pd.Dataframe: _description_
        """
        # renames to keep things simple
        entity_df = entity_df.rename(columns={DEFAULT_EVENT_TIMESTAMP_FIELD: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field:
            entity_df = entity_df.drop(columns=[created_timestamp_field], errors="ignore")

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - ttl.to_pandas_dateoffset()
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how=how)
        else:
            df = source_df.merge(entity_df, how="cross")

        df = cls._point_on_time_filter(df, period, include=include, ttl=ttl)
        df = cls._point_on_time_latest(df, join_keys, created_timestamp_field)

        df = df.drop(columns=[created_timestamp_field], errors="ignore").rename(
            columns={
                ENTITY_EVENT_TIMESTAMP_FIELD: QUERY_COL,
                SOURCE_EVENT_TIMESTAMP_FIELD: DEFAULT_EVENT_TIMESTAMP_FIELD,
            }
        )
        return df.sort_values(
            by=[QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD], ascending=True, ignore_index=True
        )

    @classmethod
    def _point_in_time_filter(
        cls,
        df: pd.DataFrame,
        include: bool = True,
        ttl: Optional[Period] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): DataFrame to filter
            include (bool, optional): whether take < entity_timestamp_field or <= entity_timestamp_field. Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp speicified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            pd.DataFrame: _description_
        """

        earliest_timestamp = None
        if ttl:
            earliest_timestamp = df[entity_timestamp_field] - ttl.to_pandas_dateoffset()

        if include:
            candidates = df[entity_timestamp_field] >= df[source_timestamp_field]
            if ttl:
                candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
        else:
            candidates = df[entity_timestamp_field] > df[source_timestamp_field]
            if ttl:
                candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
        return df[candidates]

    @classmethod
    def _point_on_time_filter(
        cls,
        df: pd.DataFrame,
        period: Period,
        include: bool = True,
        ttl: Optional[Period] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): DataFrame to filter
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            include (bool, optional): whether take < entity_timestamp_field or <= entity_timestamp_field. Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp speicified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            pd.DataFrame: _description_
        """

        earliest_timestamp = None
        offset = period.to_pandas_dateoffset()
        if ttl:
            earliest_timestamp = df[entity_timestamp_field] - ttl.to_pandas_dateoffset()

        # get backward data
        if period.is_neg:
            if include:
                candidates = (df[entity_timestamp_field] >= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] < df[source_timestamp_field] - offset
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] > df[source_timestamp_field]) & (
                    df[entity_timestamp_field] <= df[source_timestamp_field] - offset
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
        # get forward data
        else:
            if include:
                candidates = (df[entity_timestamp_field] <= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] > df[source_timestamp_field] - offset
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] < df[source_timestamp_field]) & (
                    df[entity_timestamp_field] >= df[source_timestamp_field] - offset
                )

                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
            return df[candidates]

        return df[candidates]

    @classmethod
    def _point_in_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame,): DataFrame, to be filter
            group_keys (List[str], optional): entities cols. Defaults to []
            created_timestamp_field (Optional[str], optional): data upload time col. Defaults to None
            entity_timestamp_field (str, optional):  query time col, Defaults to `ENTITY_EVENT_TIMESTAMP_FIELD`
            source_timestamp_field (str, optional): event taken time col, Defaults to `SOURCE_EVENT_TIMESTAMP_FIELD`

        Returns:
            pd.DataFrame,: _description_
        """
        sort_by = [source_timestamp_field]
        if created_timestamp_field:
            sort_by.append(created_timestamp_field)

        df.sort_values(by=sort_by, ascending=False, ignore_index=True, inplace=True)
        df.drop_duplicates(
            subset=group_keys + [entity_timestamp_field],
            keep="first",
            inplace=True,
        )
        return df

    @classmethod
    def _point_on_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): DataFrame to be filter
            group_keys (List[str], optional): entities cols. Defaults to []
            created_timestamp_field (Optional[str], optional): data upload time col. Defaults to None
            entity_timestamp_field (str, optional):  query time col, Defaults to `ENTITY_EVENT_TIMESTAMP_FIELD`
            source_timestamp_field (str, optional): event taken time col, Defaults to `SOURCE_EVENT_TIMESTAMP_FIELD`

        Returns:
            pd.DataFrame: _description_
        """
        if created_timestamp_field:
            df.sort_values(
                by=[created_timestamp_field],
                ascending=False,
                ignore_index=True,
                inplace=True,
            )
            df.drop_duplicates(
                subset=group_keys + [entity_timestamp_field, source_timestamp_field],
                keep="first",
                inplace=True,
            )
        return df
