from aie_feast.definitions import SchemaAnchor


def test_parse_cfg_to_feature_anchor():
    feature_anchor = SchemaAnchor.from_str("fv1:f1")
    assert feature_anchor.view_name == "fv1"
    assert feature_anchor.schema_name == "f1"
    assert feature_anchor.period is None

    feature_anchor = SchemaAnchor.from_str("fv1:f1:1 day")
    assert feature_anchor.period == "1 day"
