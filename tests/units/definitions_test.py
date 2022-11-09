from aie_feast.definitions import Entity, SchemaAnchor


def test_entity_auto_join_keys():
    entity = Entity(name="foo")
    assert entity.join_keys == ["foo"]


def test_parse_cfg_to_feature_anchor():
    feature_anchor = SchemaAnchor.from_str("fv1:f1")
    assert feature_anchor.view_name == "fv1"
    assert feature_anchor.schema_name == "f1"
    assert feature_anchor.period is None

    feature_anchor = SchemaAnchor.from_str("fv1:f1:1 day")
    assert feature_anchor.period == "1 day"
