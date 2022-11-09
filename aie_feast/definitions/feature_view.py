from typing import Set, List


from .base_view import BaseView
from .features import Feature


class FeatureView(BaseView):
    def get_feature_names(self) -> List[str]:
        return [feature.name for feature in self.schemas]

    def get_feature_objects(self, is_numeric=False) -> Set[Feature]:
        return {
            Feature.create_feature_from_schema(schema, self.name)
            for schema in self.schemas
            if (schema.is_numeric() if is_numeric else True)
        }
