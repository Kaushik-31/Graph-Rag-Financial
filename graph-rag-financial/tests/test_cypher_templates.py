"""Smoke tests for Cypher template generation."""
from src.graph.cypher_templates import (
    NEIGHBORS_1HOP,
    NEIGHBORS_2HOP,
    entities_of_type_related_cypher,
    path_between_cypher,
)


def test_neighbors_templates_have_required_params():
    assert "$entity" in NEIGHBORS_1HOP
    assert "$limit" in NEIGHBORS_1HOP
    assert "$entity" in NEIGHBORS_2HOP


def test_path_between_substitutes_max_hops():
    cypher = path_between_cypher(3)
    assert "*..3" in cypher
    assert "$entity_a" in cypher and "$entity_b" in cypher


def test_entities_of_type_substitutes_max_hops():
    cypher = entities_of_type_related_cypher(2)
    assert "*1..2" in cypher
    assert "$target" in cypher
    assert "$entity_type" in cypher
