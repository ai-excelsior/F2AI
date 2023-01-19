from __future__ import annotations
from enum import Enum
from typing import Optional, TYPE_CHECKING, List, Dict
from pydantic import BaseModel

from .dtypes import FeatureDTypes, NUMERIC_FEATURE_D_TYPES

if TYPE_CHECKING:
    from .base_view import BaseView


class SchemaType(str, Enum):
    """Schema used to describe a data column in a table. We only have 2 options in such context, feature or label. Label is the observation data which usually appear in supervised machine learning. In F2AI, label is treated as a special feature."""

    FEATURE = 0
    LABEL = 1


class FeatureSchema(BaseModel):
    """A FeatureSchema is used to describe a data column but no table information included."""

    name: str
    description: Optional[str]
    dtype: FeatureDTypes

    def is_numeric(self):
        if self.dtype in NUMERIC_FEATURE_D_TYPES:
            return True
        return False


class SchemaAnchor(BaseModel):
    """
    SchemaAnchor links a view to a group of FeatureSchemas with period information included if it has.
    """

    view_name: str
    schema_name: str
    period: Optional[str]

    @classmethod
    def from_strs(cls, cfgs: List[str]) -> "List[SchemaAnchor]":
        """Construct from a list of strings.

        Args:
            cfgs (List[str])

        Returns:
            List[SchemaAnchor]
        """
        return [cls.from_str(cfg) for cfg in cfgs]

    @classmethod
    def from_str(cls, cfg: str) -> "SchemaAnchor":
        """Construct from a string.

        Args:
            cfg (str): a string with specific format, egg: {feature_view_name}:{feature_name}:{period}

        Returns:
            SchemaAnchor
        """
        components = cfg.split(":")

        if len(components) < 2:
            raise ValueError("Please indicate features in table:feature format")
        elif len(components) > 3:
            raise ValueError("Please make sure colon not in name of table or features")
        elif len(components) == 2:
            view_name, schema_name = components
            return cls(view_name=view_name, schema_name=schema_name)
        elif len(components) == 3:
            view_name, schema_name, period = components
            return cls(view_name=view_name, schema_name=schema_name, period=period)

    def get_features_from_views(self, views: Dict[str, BaseView], is_numeric=False) -> List[Feature]:
        """With given views, construct a series of features based on this SchemaAnchor.

        Args:
            views (Dict[str, BaseView])
            is_numeric (bool, optional): If only return numeric features. Defaults to False.

        Returns:
            List[Feature]
        """
        from .feature_view import FeatureView

        view: BaseView = views[self.view_name]
        schema_type = SchemaType.FEATURE if isinstance(view, FeatureView) else SchemaType.LABEL

        if self.schema_name == "*":
            return [
                Feature.create_from_schema(feature_schema, view.name, schema_type, self.period)
                for feature_schema in view.schemas
                if (feature_schema.is_numeric() if is_numeric else True)
            ]

        feature_schema = next((schema for schema in view.schemas if schema.name == self.schema_name), None)
        if feature_schema and (feature_schema.is_numeric() if is_numeric else True):
            return [Feature.create_from_schema(feature_schema, view.name, schema_type, self.period)]

        return []


class Feature(BaseModel):
    """A Feature which include all necessary information which F2AI should know."""

    name: str
    dtype: FeatureDTypes
    period: Optional[str]
    schema_type: SchemaType = SchemaType.FEATURE
    view_name: str

    @classmethod
    def create_feature_from_schema(
        cls, schema: FeatureSchema, view_name: str, period: str = None
    ) -> "Feature":
        return cls.create_from_schema(schema, view_name, SchemaType.FEATURE, period)

    @classmethod
    def create_label_from_schema(cls, schema: FeatureSchema, view_name: str, period: str = None) -> "Feature":
        return cls.create_from_schema(schema, view_name, SchemaType.LABEL, period)

    @classmethod
    def create_from_schema(
        cls, schema: FeatureSchema, view_name: str, schema_type: SchemaType, period: str
    ) -> "Feature":
        return Feature(
            name=schema.name, dtype=schema.dtype, schema_type=schema_type, view_name=view_name, period=period
        )

    def __hash__(self) -> int:
        return hash(f"{self.view_name}:{self.name}:{self.period}, {self.schema_type}")
