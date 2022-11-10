from f2ai.definitions import Entity


def test_entity_auto_join_keys():
    entity = Entity(name="foo")
    assert entity.join_keys == ["foo"]
