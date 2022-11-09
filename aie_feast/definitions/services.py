from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field

from .entities import Entity
from .features import SchemaAnchor, Feature
from .feature_view import FeatureView
from .label_view import LabelView


class Service(BaseModel):
    """A Service is a combination of a group of feature views and label views, which usually directly related to a certain AI model. Tbe best practice which F2AI suggested is, treating services are immutable. Egg: if you want to train different combinations of features for a specific AI model, you may want to create multiple Services, like: linear_reg_v1, linear_reg_v2."""

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
        """Construct a Service from parsed yaml config file."""

        cfg["features"] = SchemaAnchor.from_strs(cfg.pop("features", []))
        cfg["labels"] = SchemaAnchor.from_strs(cfg.pop("labels", []))

        return cls(**cfg)

    def get_feature_names(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        return set([feature.name for feature in self.get_feature_objects(feature_views, is_numeric)])

    def get_label_names(self, label_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        return set([label.name for label in self.get_label_objects(label_views, is_numeric)])

    def get_feature_objects(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> Set[Feature]:
        """get all the feature objects which included in this service based on features' schema anchor.

        Args:
            feature_views (Dict[str, FeatureView]): A group of FeatureViews.
            is_numeric (bool, optional): If only include numeric features. Defaults to False.

        Returns:
            Set[Feature]
        """
        return {
            feature
            for schema_anchor in self.features
            for feature in schema_anchor.get_features_from_views(feature_views, is_numeric)
        }

    def get_label_objects(self, label_views: Dict[str, LabelView], is_numeric=False) -> Set[Feature]:
        """get all the label objects which included in this service based on labels' schema anchor.

        Args:
            feature_views (Dict[str, LabelView]): A group of LabelViews.
            is_numeric (bool, optional): If only include numeric labels. Defaults to False.

        Returns:
            Set[Feature]
        """
        return {
            label
            for schema_anchor in self.labels
            for label in schema_anchor.get_features_from_views(label_views, is_numeric)
        }

    def get_feature_views(self, feature_views: Dict[str, FeatureView]) -> List[FeatureView]:
        """Get FeatureViews of this service. This will automatically filter out the feature view not given by parameters.

        Args:
            feature_views (Dict[str, FeatureView])

        Returns:
            List[FeatureView]
        """
        feature_view_names = {anchor.view_name for anchor in self.features}
        return [feature_views[feature_view_name] for feature_view_name in feature_view_names]

    def get_label_views(self, label_views: Dict[str, LabelView]) -> List[LabelView]:
        """Get LabelViews of this service. This will automatically filter out the label view not given by parameters.

        Args:
            label_views (Dict[str, LabelView])

        Returns:
            List[LabelView]
        """
        label_view_names = {anchor.view_name for anchor in self.labels}
        return [label_views[label_view_name] for label_view_name in label_view_names]

    def get_feature_entities(self, feature_views: Dict[str, FeatureView]) -> Set[Entity]:
        """Get all entities which appeared in related feature views to this service and without duplicate entity.

        Args:
            feature_views (Dict[str, FeatureView])

        Returns:
            Set[Entity]
        """
        return {
            entity
            for feature_view in self.get_feature_views(feature_views)
            for entity in feature_view.entities
        }

    def get_label_entities(self, label_views: Dict[str, LabelView]) -> Set[str]:
        """Get all entities which appeared in related label views to this service and without duplicate entity.

        Args:
            label_views (Dict[str, LabelView])

        Returns:
            Set[str]
        """
        return {entity for label_view in self.get_label_views(label_views) for entity in label_view.entities}

    def get_entities(
        self, feature_views: Dict[str, FeatureView], label_views: Dict[str, LabelView]
    ) -> Set[str]:
        """Get all entities which appeared in this service and without duplicate entity.

        Args:
            feature_views (Dict[str, FeatureView])
            label_views (Dict[str, LabelView])

        Returns:
            Set[str]
        """
        return self.get_feature_entities(feature_views).union(self.get_label_entities(label_views))

    # TODO: remove in future
    def get_label_view(self, label_views: Dict[str, LabelView]) -> LabelView:
        return self.get_label_views(label_views)[0]
