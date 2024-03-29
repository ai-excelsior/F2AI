from typing import Optional, List

from .base_view import BaseView
from .features import Feature


class LabelView(BaseView):
    request_source: Optional[str]

    def get_label_names(self):
        return [label.name for label in self.schemas]

    def get_label_objects(self, is_numeric=False) -> List[Feature]:
        return list(
            dict.fromkeys(
                [
                    Feature.create_label_from_schema(schema, self.name)
                    for schema in self.schemas
                    if (schema.is_numeric() if is_numeric else True)
                ]
            )
        )
