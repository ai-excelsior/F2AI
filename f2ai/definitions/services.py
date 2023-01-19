from typing import List, Optional, Dict
from pydantic import BaseModel

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

    @classmethod
    def from_yaml(cls, cfg: Dict) -> "Service":
        """Construct a Service from parsed yaml config file."""

        cfg["features"] = SchemaAnchor.from_strs(cfg.pop("features", []))
        cfg["labels"] = SchemaAnchor.from_strs(cfg.pop("labels", []))

        return cls(**cfg)

    def get_feature_names(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> List[str]:
        return [feature.name for feature in self.get_feature_objects(feature_views, is_numeric)]

    def get_label_names(self, label_views: Dict[str, FeatureView], is_numeric=False) -> List[str]:
        return [label.name for label in self.get_label_objects(label_views, is_numeric)]

    def get_feature_objects(self, feature_views: Dict[str, FeatureView], is_numeric=False) -> List[Feature]:
        """get all the feature objects which included in this service based on features' schema anchor.

        Args:
            feature_views (Dict[str, FeatureView]): A group of FeatureViews.
            is_numeric (bool, optional): If only include numeric features. Defaults to False.

        Returns:
            List[Feature]
        """
        return list(
            dict.fromkeys(
                feature
                for schema_anchor in self.features
                for feature in schema_anchor.get_features_from_views(feature_views, is_numeric)
            )
        )

    def get_label_objects(self, label_views: Dict[str, LabelView], is_numeric=False) -> List[Feature]:
        """get all the label objects which included in this service based on labels' schema anchor.

        Args:
            feature_views (Dict[str, LabelView]): A group of LabelViews.
            is_numeric (bool, optional): If only include numeric labels. Defaults to False.

        Returns:
            List[Feature]
        """
        return list(
            dict.fromkeys(
                label
                for schema_anchor in self.labels
                for label in schema_anchor.get_features_from_views(label_views, is_numeric)
            )
        )

    def get_feature_view_names(self, feature_views: Dict[str, FeatureView]) -> List[str]:
        """
        Get the name of feature view names related to this service.

        Args:
            feature_views (Dict[str, FeatureView]): list of FeatureViews to filter.

        Returns:
            List[str]: names of FeatureView
        """
        feature_view_names = list(dict.fromkeys([anchor.view_name for anchor in self.features]))
        return [x for x in feature_view_names if x in feature_views]

    def get_feature_views(self, feature_views: Dict[str, FeatureView]) -> List[FeatureView]:
        """Get FeatureViews of this service. This will automatically filter out the feature view not given by parameters.

        Args:
            feature_views (Dict[str, FeatureView])

        Returns:
            List[FeatureView]
        """
        feature_view_names = self.get_feature_view_names(feature_views)
        return [feature_views[feature_view_name] for feature_view_name in feature_view_names]

    def get_label_views(self, label_views: Dict[str, LabelView]) -> List[LabelView]:
        """Get LabelViews of this service. This will automatically filter out the label view not given by parameters.

        Args:
            label_views (Dict[str, LabelView])

        Returns:
            List[LabelView]
        """
        label_view_names = list(dict.fromkeys([anchor.view_name for anchor in self.labels]))
        return [label_views[label_view_name] for label_view_name in label_view_names]

    def get_feature_entities(self, feature_views: Dict[str, FeatureView]) -> List[Entity]:
        """Get all entities which appeared in related feature views to this service and without duplicate entity.

        Args:
            feature_views (Dict[str, FeatureView])

        Returns:
            List[Entity]
        """
        return list(
            dict.fromkeys(
                entity
                for feature_view in self.get_feature_views(feature_views)
                for entity in feature_view.entities
            )
        )

    def get_label_entities(self, label_views: Dict[str, LabelView]) -> List[str]:
        """Get all entities which appeared in related label views to this service and without duplicate entity.

        Args:
            label_views (Dict[str, LabelView])

        Returns:
            List[str]
        """
        return list(
            dict.fromkeys(
                entity for label_view in self.get_label_views(label_views) for entity in label_view.entities
            )
        )

    def get_entities(
        self, feature_views: Dict[str, FeatureView], label_views: Dict[str, LabelView]
    ) -> List[str]:
        """Get all entities which appeared in this service and without duplicate entity.

        Args:
            feature_views (Dict[str, FeatureView])
            label_views (Dict[str, LabelView])

        Returns:
            List[str]
        """
        return list(
            dict.fromkeys(self.get_feature_entities(feature_views) + self.get_label_entities(label_views))
        )

    def get_join_keys(
        self,
        feature_views: Dict[str, FeatureView],
        label_views: Dict[str, FeatureView],
        entities: Dict[str, Entity],
    ) -> List[str]:
        return list(
            dict.fromkeys(
                [
                    join_key
                    for x in self.get_entities(feature_views, label_views)
                    for join_key in entities[x].join_keys
                ]
            )
        )
