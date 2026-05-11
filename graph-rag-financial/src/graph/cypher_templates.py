"""Reusable Cypher patterns for common multi-hop financial queries.

These are used by the retriever when the LLM router decides a structured
graph query is more reliable than free-form Cypher generation.
"""

# 1-hop: direct neighbors of an entity, optionally filtered by relation
NEIGHBORS_1HOP = """
MATCH (a:Entity {name: $entity})-[r]-(b:Entity)
WHERE $rel_type IS NULL OR type(r) = $rel_type
RETURN a.name AS source, type(r) AS relation, b.name AS target,
       b.type AS target_type, r.evidence AS evidence
LIMIT $limit
"""

# 2-hop: traverse two hops with a relation filter on the first hop
NEIGHBORS_2HOP = """
MATCH (a:Entity {name: $entity})-[r1]-(b:Entity)-[r2]-(c:Entity)
WHERE c <> a
RETURN a.name AS start, type(r1) AS rel1, b.name AS mid,
       type(r2) AS rel2, c.name AS end, c.type AS end_type,
       r1.evidence AS evidence1, r2.evidence AS evidence2
LIMIT $limit
"""

# Path between two named entities (up to N hops)
PATH_BETWEEN = """
MATCH path = shortestPath(
  (a:Entity {name: $entity_a})-[*..%d]-(b:Entity {name: $entity_b})
)
RETURN [n IN nodes(path) | n.name] AS node_names,
       [r IN relationships(path) | type(r)] AS relations,
       length(path) AS hops
"""

# Find all entities matching a type that connect to a target via any path
ENTITIES_OF_TYPE_RELATED_TO = """
MATCH (a:Entity)-[*1..%d]-(b:Entity {name: $target})
WHERE a.type = $entity_type AND a <> b
RETURN DISTINCT a.name AS name, a.type AS type
LIMIT $limit
"""


def path_between_cypher(max_hops: int) -> str:
    return PATH_BETWEEN % max_hops


def entities_of_type_related_cypher(max_hops: int) -> str:
    return ENTITIES_OF_TYPE_RELATED_TO % max_hops
