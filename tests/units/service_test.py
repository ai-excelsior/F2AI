from aie_feast.service import Service
from aie_feast.definitions import SchemaAnchor, FeatureSchema, FeatureView


def test_get_features_from_service():
    service = Service(
        name="foo",
        features=[
            SchemaAnchor(view_name="fv", schema_name="*"),
            SchemaAnchor(view_name="fv", schema_name="foo"),
        ],
    )
    feature_views = {"fv": FeatureView(name="fv", schema=[FeatureSchema(name="f1", dtype="string")])}

    features = service.get_feature_objects(feature_views)

    assert len(features) == 1
    assert list(features)[0].name == "f1"
