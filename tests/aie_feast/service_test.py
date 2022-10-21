from aie_feast.service import Service
from aie_feast.views import FeatureView
from aie_feast.definitions import SchemaAnchor, FeatureSchema


def test_get_features_from_service():
    service = Service(
        name="foo",
        features=[
            SchemaAnchor(view_name="fv", schema_name="*"),
            SchemaAnchor(view_name="fv", schema_name="foo"),
        ],
    )
    feature_views = [FeatureView(name="fv", schema=[FeatureSchema(name="f1", dtype="string")])]

    features = service.get_features(feature_views)

    assert len(features) == 1
    assert features[0].name == "f1"