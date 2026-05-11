"""Neo4j connection and write helpers.

Schema: each entity becomes a node labeled with its type plus a generic :Entity
label so we can run polymorphic queries. Relations are typed edges with an
`evidence` property pointing back to the source sentence.
"""
from contextlib import contextmanager
from typing import Iterable, List

from neo4j import GraphDatabase

from src.ingestion.extractor import Entity, Relation
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Neo4jWriter:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self):
        self.driver.close()

    @contextmanager
    def session(self):
        with self.driver.session(database=settings.neo4j_database) as sess:
            yield sess

    def setup_constraints(self) -> None:
        """Ensure unique entity names so MERGE is well-behaved."""
        with self.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def upsert_entities(self, entities: Iterable[Entity]) -> None:
        rows = [
            {"name": e.name, "type": e.type, "aliases": e.aliases}
            for e in entities
        ]
        if not rows:
            return
        cypher = (
            "UNWIND $rows AS row "
            "MERGE (e:Entity {name: row.name}) "
            "SET e.type = row.type, e.aliases = row.aliases "
            "WITH e, row "
            "CALL apoc.create.addLabels(e, [row.type]) YIELD node "
            "RETURN count(*)"
        )
        # Fallback if APOC is unavailable
        fallback = (
            "UNWIND $rows AS row "
            "MERGE (e:Entity {name: row.name}) "
            "SET e.type = row.type, e.aliases = row.aliases"
        )
        with self.session() as s:
            try:
                s.run(cypher, rows=rows).consume()
            except Exception:
                s.run(fallback, rows=rows).consume()

    def upsert_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return
        # Group by relation type since Cypher needs a literal type for the edge.
        by_type: dict[str, list] = {}
        for r in relations:
            by_type.setdefault(r.type, []).append(
                {"source": r.source, "target": r.target, "evidence": r.evidence}
            )
        with self.session() as s:
            for rel_type, rows in by_type.items():
                cypher = (
                    "UNWIND $rows AS row "
                    "MATCH (a:Entity {name: row.source}) "
                    "MATCH (b:Entity {name: row.target}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.evidence = row.evidence"
                )
                s.run(cypher, rows=rows).consume()
        logger.info("Upserted %d relations across %d types", sum(len(v) for v in by_type.values()), len(by_type))
