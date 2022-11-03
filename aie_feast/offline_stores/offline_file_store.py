from feast import Entity, FeatureView
import pandas as pd
from typing import List, Optional, Set
from dateutil.relativedelta import relativedelta
from aie_feast.common.utils import parse_date
from aie_feast.definitions import Feature
from aie_feast.common.source import FileSource
from aie_feast.common.utils import get_stats_result
from aie_feast.service import Service
from aie_feast.views import LabelView
from .offline_store import OfflineStore, OfflineStoreType


TIME_COL = "event_timestamp"
# DEFAULT_CREATED_TIMESTAMP_FIELD = "created_timestamp"
ENTITY_EVENT_TIMESTAMP_FIELD = "_entity_event_timestamp_"
SOURCE_EVENT_TIMESTAMP_FIELD = "_source_event_timestamp_"
SOURCE_CREATED_TIMESTAMP_FIELD = "_created_timestamp_"
QUERY_COL = "query_timestamp"


class OfflineFileStore(OfflineStore):
    type: OfflineStoreType = OfflineStoreType.FILE

    def _read_file(self, source: FileSource, features: Set[Feature] = {}, join_keys: List[str] = []):
        feature_columns = [feature.name for feature in features]

        return source.read_file(
            str_cols=join_keys,
            keep_cols=feature_columns,
        )

    def materialize(
        self,
        service: Service,
        feature_views: List[FeatureView],
        label_views: List[LabelView],
        sources: List[FileSource],
        entities: List[Entity],
        incremental_begin,
    ):
        all_cols_name = service.get_feature_names(feature_views) | service.get_label_names(label_views)
        label_view = service.get_label_view(label_views)
        labels = label_view.get_labels()
        join_keys = list(
            {
                join_key
                for entity_name in service.get_label_entities(label_view)
                for join_key in entities[entity_name].join_keys
            }
        )
        source = sources[label_view.batch_source]
        joined_frame = self._read_file(source=source, features=labels, join_keys=list(join_keys))
        # create timestamp makes no sense to labels
        joined_frame.drop(columns=source.created_timestamp_field, inplace=True, errors="ignore")
        if isinstance(incremental_begin, dict):
            incremental_begin = joined_frame[TIME_COL].max() - relativedelta(**incremental_begin)
            joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]
        else:
            joined_frame = joined_frame[joined_frame[TIME_COL] >= incremental_begin]

        # join features dataframe
        for feature_view in service.get_feature_views(feature_views):
            feature_name = [
                n
                for n in feature_view.get_feature_names()
                if n in all_cols_name and n not in joined_frame.columns
            ]
            if feature_name:  # this view has new features other than those in joined_frame
                features = [n for n in feature_view.get_features() if n.name in feature_name]
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
        return joined_frame

    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        join_keys: List[str] = [],
        ttl: Optional[str] = None,
        include: bool = True,
        **kwargs
    ):
        source_df = self._read_file(source=source, features=features, join_keys=join_keys)

        return self.point_in_time_join(
            entity_df=entity_df,
            source_df=source_df,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            **kwargs
        )

    def get_period_features(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        period: str,
        join_keys: List[str] = [],
        ttl: Optional[str] = None,
        include: bool = True,
        is_label: bool = False,
        **kwargs
    ):
        source_df = self._read_file(source=source, features=features, join_keys=join_keys)

        return self.point_on_time_join(
            entity_df=entity_df,
            source_df=source_df,
            period=period,
            timestamp_field=source.timestamp_field,
            created_timestamp_field=source.created_timestamp_field,
            ttl=ttl,
            join_keys=join_keys,
            include=include,
            is_label=is_label,
            **kwargs
        )

    def stats(
        self,
        entity_df: pd.DataFrame,
        features: Set[Feature],
        source: FileSource,
        start,
        fn: str = "mean",
        join_keys: list = [],
        include: str = "both",
        keys_only: bool = False,
    ):
        source_df = self._read_file(source=source, features=features, join_keys=join_keys)

        feature_columns = [feature.name for feature in features]

        entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={source.timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if len(entity_df.columns) > 1:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        if keys_only:
            return df[join_keys].groupby(join_keys).size().reset_index().drop(columns=[0])

        if join_keys:
            result = df.groupby(join_keys).apply(
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

    def get_latest_entities(self, source: FileSource, join_keys: list):
        source_df = self._read_file(source=source, features=[], join_keys=join_keys)
        df = source_df.sort_values(by=source.timestamp_field, ascending=False, ignore_index=True)
        return df.drop_duplicates(subset=join_keys, keep="first")

    @classmethod
    def point_in_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        timestamp_field: Optional[str] = None,
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[str] = None,
        join_keys: List[str] = [],
        include: bool = True,
        how: str = "inner",
    ):
        # renames to keep things simple
        if timestamp_field:
            entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
            source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field:
            entity_df = entity_df.drop(columns=[created_timestamp_field], errors="ignore")

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - relativedelta(
                **parse_date(ttl)
            )
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        # pre filter source_df by entities
        # if len(join_keys) > 0:
        #     unique_entity_df = entity_df[join_keys].groupby(join_keys).size().reset_index().drop(columns=[0])
        #     source_df = unique_entity_df.merge(source_df, on=join_keys, how="inner")

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how=how)
        else:
            df = source_df.merge(entity_df, how="cross")

        if timestamp_field:
            df = cls.point_in_time_filter(df, include=include, ttl=ttl)
            df = cls.point_in_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(
            columns=[SOURCE_EVENT_TIMESTAMP_FIELD, created_timestamp_field], errors="ignore"
        ).rename(columns={ENTITY_EVENT_TIMESTAMP_FIELD: TIME_COL})

    @classmethod
    def point_on_time_join(
        cls,
        entity_df: pd.DataFrame,
        source_df: pd.DataFrame,
        period: str,
        timestamp_field: Optional[str],
        created_timestamp_field: Optional[str] = None,
        ttl: Optional[str] = None,
        join_keys: List[str] = [],
        include: bool = True,
        is_label: bool = False,
        how="inner",
    ):
        # renames to keep things simple
        entity_df = entity_df.rename(columns={TIME_COL: ENTITY_EVENT_TIMESTAMP_FIELD})
        source_df = source_df.rename(columns={timestamp_field: SOURCE_EVENT_TIMESTAMP_FIELD})

        if created_timestamp_field:
            entity_df = entity_df.drop(columns=[created_timestamp_field], errors="ignore")

        # pre filter source_df by ttl
        if ttl:
            min_entity_timestamp = entity_df[ENTITY_EVENT_TIMESTAMP_FIELD].min() - relativedelta(
                **parse_date(ttl)
            )
            if include:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] > min_entity_timestamp]
            else:
                source_df = source_df[source_df[SOURCE_EVENT_TIMESTAMP_FIELD] >= min_entity_timestamp]

        # pre filter source_df by entities
        # if len(join_keys) > 0:
        #     unique_entity_df = entity_df[join_keys].groupby(join_keys).size().reset_index().drop(columns=[0])
        #     source_df = unique_entity_df.merge(source_df, on=join_keys, how="inner")

        if len(join_keys) > 0:
            df = source_df.merge(entity_df, on=join_keys, how="inner")
        else:
            df = source_df.merge(entity_df, how="cross")

        df = cls.point_on_time_filter(df, period, include=include, ttl=ttl, is_label=is_label)
        df = cls.point_on_time_latest(df, join_keys, created_timestamp_field)

        return df.drop(columns=[created_timestamp_field], errors="ignore").rename(
            columns={ENTITY_EVENT_TIMESTAMP_FIELD: QUERY_COL, SOURCE_EVENT_TIMESTAMP_FIELD: TIME_COL}
        )

    @classmethod
    def point_in_time_filter(
        cls,
        df: pd.DataFrame,
        include: bool = True,
        ttl: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        if ttl:
            timedelta = relativedelta(**parse_date(ttl))
            earliest_timestamp = df[entity_timestamp_field].map(lambda x: x - timedelta)

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
    def point_on_time_filter(
        cls,
        df: pd.DataFrame,
        period: str,
        include: bool = True,
        ttl: Optional[str] = None,
        is_label=False,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        """filter the joined results within [entity_timestamp - ttl, entity_timestamp]"""

        earliest_timestamp = None
        period = relativedelta(**parse_date(period))
        if ttl:
            timedelta = relativedelta(**parse_date(ttl))
            earliest_timestamp = df[entity_timestamp_field].map(lambda x: x - timedelta)

        if is_label:  # get forward data
            if include:
                candidates = (df[entity_timestamp_field] <= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] > df[source_timestamp_field].map(lambda x: x - period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] < df[source_timestamp_field]) & (
                    df[entity_timestamp_field] >= df[source_timestamp_field].map(lambda x: x - period)
                )

                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
            return df[candidates]
        else:  # get backward data
            if include:
                candidates = (df[entity_timestamp_field] >= df[source_timestamp_field]) & (
                    df[entity_timestamp_field] < df[source_timestamp_field].map(lambda x: x + period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] > earliest_timestamp)
            else:
                candidates = (df[entity_timestamp_field] > df[source_timestamp_field]) & (
                    df[entity_timestamp_field] <= df[source_timestamp_field].map(lambda x: x + period)
                )
                if ttl:
                    candidates = candidates & (df[source_timestamp_field] >= earliest_timestamp)
        return df[candidates]

    @classmethod
    def point_in_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        sort_by = group_keys + [source_timestamp_field]
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
    def point_on_time_latest(
        cls,
        df: pd.DataFrame,
        group_keys: List[str] = [],
        created_timestamp_field: Optional[str] = None,
        entity_timestamp_field: str = ENTITY_EVENT_TIMESTAMP_FIELD,
        source_timestamp_field: str = SOURCE_EVENT_TIMESTAMP_FIELD,
    ):
        if created_timestamp_field:
            df.sort_values(
                by=group_keys + [created_timestamp_field],
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
