import pandas as pd
import datetime
import os
from typing import List, Optional, Set

from ..definitions import (
    Feature,
    Period,
    Service,
    OfflineStoreType,
    OfflineStore,
    FileSource,
    StatsFunctions,
)

from ..common.time_field import (
    DEFAULT_EVENT_TIMESTAMP_FIELD,
    ENTITY_EVENT_TIMESTAMP_FIELD,
    SOURCE_EVENT_TIMESTAMP_FIELD,
    QUERY_COL,
    MATERIALIZE_TIME,
)


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE

    def get_offline_source(self, service: Service) -> FileSource:
        return FileSource(
            name=f"{service.name}_source",
            path=os.path.join(self.materialize_path, service.name),
            timestamp_field=DEFAULT_EVENT_TIMESTAMP_FIELD,
            created_timestamp_field=MATERIALIZE_TIME,
        )

    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: List[Feature],
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
        features: List[Feature],
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
        source: FileSource,
        features: Set[Feature],
        fn: StatsFunctions,
        group_keys: List[str] = [],
        start: datetime.datetime = None,
        end: datetime.datetime = None,
    ) -> pd.DataFrame:
        source_df = self._read_file(source=source, features=features, join_keys=group_keys)
        if fn == StatsFunctions.UNIQUE:
            features = []

        # filter source_df by start and end
        if end is not None:
            source_df: pd.DataFrame = source_df[source_df[source.timestamp_field] <= end]
        if start is not None:
            source_df: pd.DataFrame = source_df[source_df[source.timestamp_field] >= start]

        # only keep entity join keys and features
        source_df = source_df[group_keys + [feature.name for feature in features]]

        if fn == StatsFunctions.UNIQUE:
            return source_df.drop_duplicates(subset=group_keys)

        if fn == StatsFunctions.MODE:
            return source_df.groupby(group_keys).agg(lambda x: x.value_counts().index[0])

        return getattr(source_df.groupby(group_keys), fn.value if fn != StatsFunctions.AVG else "mean")()

    def get_latest_entities(
        self,
        source: FileSource,
        join_keys: List[str] = None,
        group_keys: list = None,
        entity_df: pd.DataFrame = None,
        start: datetime = None,
    ) -> pd.DataFrame:
        """
        Get latest entity keys and it's event timestamp.

        Args:
            source (FileSource): Where to find data
            group_keys (list, optional): dimension of stats. Defaults to [].
            entity_df (pd.DataFrame, optional): query condition specified by users. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe
        """
        source_df = self._read_file(source=source, features=[], join_keys=group_keys)
        source_df = source_df.rename(columns={source.timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})
        source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= start]
        entity_df = entity_df.rename(columns={DEFAULT_EVENT_TIMESTAMP_FIELD: ENTITY_EVENT_TIMESTAMP_FIELD})

        if join_keys:
            source_df = source_df.merge(entity_df, on=group_keys, how="inner")
        else:
            source_df = source_df.merge(entity_df, how="cross")

        source_df = source_df[group_keys + [ENTITY_EVENT_TIMESTAMP_FIELD, SOURCE_EVENT_TIMESTAMP_FIELD]]
        source_df = (
            self._point_in_time_filter(source_df)
            .drop(columns=[ENTITY_EVENT_TIMESTAMP_FIELD])
            .rename(columns={SOURCE_EVENT_TIMESTAMP_FIELD: source.timestamp_field})
        )

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
            pd.DataFrame: A point in time joined dataframe.
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

        df = df.drop(columns=[SOURCE_EVENT_TIMESTAMP_FIELD, created_timestamp_field], errors="ignore").rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: DEFAULT_EVENT_TIMESTAMP_FIELD}
        )

        # move join_keys ahead
        desired_column_order = sorted(df.columns, key=lambda x: x in join_keys, reverse=True)
        return df[desired_column_order]

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
        # TODO: this should be removed in future.
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
            pd.Dataframe: A point on time joined dataframe
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
            by=[QUERY_COL, DEFAULT_EVENT_TIMESTAMP_FIELD],
            ascending=True,
            ignore_index=True,
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
            include (bool, optional): Whether include event timestamp. If true, the query result will located in (entity_timestamp - ttl, entity_timestamp]. Otherwise, it will located in [entity_timestamp - ttl, entity_timestamp) Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of feature_view . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp specified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            pd.DataFrame: filtered DataFrame using different time meaning.
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
        """
        filter dataframe with given time meaning.

        Args:
            df (pd.DataFrame): DataFrame to filter
            period (Period): period to take, negative means look back from entity_timestamp_field, positive means look_forward from entity_timestamp_field
            include (bool, optional): whether take < entity_timestamp_field or <= entity_timestamp_field. Defaults to True
            ttl (Optional[Period], optional): requirement of timeliness of featureview . Defaults to None means no requirement
            entity_timestamp_field (str, optional): timestamp speicified by users. Defaults to ENTITY_EVENT_TIMESTAMP_FIELD.
            source_timestamp_field (str, optional): timestamp recorded in data. Defaults to SOURCE_EVENT_TIMESTAMP_FIELD.

        Returns:
            pd.DataFrame: filtered DataFrame.
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
        """
        get latest entity row if there are duplicate exist.

        Args:
            df (pd.DataFrame,): DataFrame, to be filter
            group_keys (List[str], optional): entities cols. Defaults to []
            created_timestamp_field (Optional[str], optional): data upload time col. Defaults to None
            entity_timestamp_field (str, optional):  query time col, Defaults to `ENTITY_EVENT_TIMESTAMP_FIELD`
            source_timestamp_field (str, optional): event taken time col, Defaults to `SOURCE_EVENT_TIMESTAMP_FIELD`

        Returns:
            pd.DataFrame: filtered DataFrame
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
