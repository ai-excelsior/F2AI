from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field

from .definitions import Entity, SchemaAnchor, Feature, FeatureView, LabelView


class Service(BaseModel):
    name: str
    description: Optional[str]
    features: List[SchemaAnchor] = []
    labels: List[SchemaAnchor] = []
    ttl: Optional[str] = None

    # TODO: remove below configs in future
    materialize_path: Optional[str] = Field(alias="materialize", default="materialize_table")
    materialize_type: Optional[str] = Field(alias="type", default="file")
    dbt_path: Optional[str] = Field(alias="dbt", default="dbt_path")

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "Service":
        cfg["features"] = SchemaAnchor.from_strs(cfg.pop("features", []))
        cfg["labels"] = SchemaAnchor.from_strs(cfg.pop("labels", []))
        return cls(**cfg)

    def get_feature_names(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        return set([feature.name for feature in self.get_feature_objects(feature_views, is_numeric)])

    def get_label_names(self, label_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        return set([label.name for label in self.get_label_objects(label_views, is_numeric)])

    def get_feature_objects(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        """
        get features based on features' schema anchor
        """
        return {
            feature
            for schema_anchor in self.features
            for feature in schema_anchor.get_features_from_views(feature_views, is_numeric)
        }

    def get_label_objects(self, label_views: Dict[str, LabelView], is_numeric=False) -> Set[Feature]:
        """
        get labels based on labels' schema anchor
        """
        return {
            label
            for schema_anchor in self.labels
            for label in schema_anchor.get_features_from_views(label_views, is_numeric)
        }

    def get_feature_views(self, feature_views: Dict[str, FeatureView]) -> List[FeatureView]:
        if isinstance(feature_views, dict):
            feature_view_names = {anchor.view_name for anchor in self.features}
            return [feature_views[feature_view_name] for feature_view_name in feature_view_names]
        elif isinstance(feature_views, FeatureView):
            return [feature_views]

    def get_feature_entities(self, feature_views: Dict[str, FeatureView]) -> Set[Entity]:
        return {
            entity
            for feature_view in self.get_feature_views(feature_views)
            for entity in feature_view.entities
        }

    def get_label_views(self, label_views: Dict[str, LabelView]) -> List[LabelView]:
        if isinstance(label_views, dict):
            label_view_names = {anchor.view_name for anchor in self.labels}
            return [label_views[label_view_name] for label_view_name in label_view_names]
        elif isinstance(label_views, LabelView):
            return [label_views]

    def get_label_entities(self, label_views: Dict[str, LabelView]) -> Set[Entity]:
        return {entity for label_view in self.get_label_views(label_views) for entity in label_view.entities}

    def get_entities(
        self, feature_views: Dict[str, FeatureView], label_views: Dict[str, LabelView]
    ) -> Set[Entity]:
        return self.get_feature_entities(feature_views).union(self.get_label_entities(label_views))

    def get_label_view(self, label_views: Dict[str, LabelView]) -> LabelView:
        return self.get_label_views(label_views)[0]
