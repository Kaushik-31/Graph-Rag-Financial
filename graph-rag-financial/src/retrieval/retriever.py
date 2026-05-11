"""Hybrid retriever that combines Pinecone vector search with Neo4j traversal."""
from dataclasses import dataclass, field
from typing import List

from src.graph.cypher_templates import (
    NEIGHBORS_1HOP,
    NEIGHBORS_2HOP,
    path_between_cypher,
)
from src.graph.neo4j_writer import Neo4jWriter
from src.graph.vector_store import VectorStore
from src.retrieval.router import RouterDecision, route_query
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    question: str
    decision: RouterDecision
    vector_hits: List[dict] = field(default_factory=list)
    graph_hits: List[dict] = field(default_factory=list)

    def to_context(self) -> str:
        """Linearize all retrieved evidence into a single prompt-ready string."""
        parts = []
        if self.graph_hits:
            parts.append("## Knowledge Graph Evidence")
            for i, hit in enumerate(self.graph_hits, 1):
                parts.append(f"[G{i}] {self._format_graph_hit(hit)}")
        if self.vector_hits:
            parts.append("\n## Document Evidence")
            for i, hit in enumerate(self.vector_hits, 1):
                src = hit.get("metadata", {}).get("source", "unknown")
                parts.append(f"[V{i}] (source: {src})\n{hit['text']}")
        return "\n".join(parts)

    @staticmethod
    def _format_graph_hit(hit: dict) -> str:
        if "node_names" in hit:
            chain = " -> ".join(hit["node_names"])
            rels = " | ".join(hit.get("relations", []))
            return f"Path: {chain} (relations: {rels}, hops: {hit.get('hops')})"
        if "mid" in hit:
            return (
                f"{hit['start']} --[{hit['rel1']}]-- {hit['mid']} "
                f"--[{hit['rel2']}]-- {hit['end']} "
                f"(evidence: {hit.get('evidence1', '')} / {hit.get('evidence2', '')})"
            )
        return (
            f"{hit['source']} --[{hit['relation']}]--> {hit['target']} "
            f"(evidence: {hit.get('evidence', '')})"
        )


class HybridRetriever:
    def __init__(self):
        self.vectors = VectorStore()
        self.graph = Neo4jWriter()

    def close(self):
        self.graph.close()

    def retrieve(self, question: str) -> RetrievalResult:
        decision = route_query(question)
        logger.info("Router decision: %s | entities=%s", decision.query_type, decision.entities)
        result = RetrievalResult(question=question, decision=decision)

        if decision.query_type == "factual":
            result.vector_hits = self.vectors.query(question)
            return result

        if decision.query_type == "relational":
            result.graph_hits = self._graph_relational(decision.entities)
            result.vector_hits = self.vectors.query(question, top_k=4)
            return result

        # multi_hop
        result.graph_hits = self._graph_multi_hop(decision.entities)
        result.vector_hits = self.vectors.query(question, top_k=4)
        return result

    def _graph_relational(self, entities: List[str]) -> List[dict]:
        if not entities:
            return []
        with self.graph.session() as s:
            if len(entities) >= 2:
                cypher = path_between_cypher(settings.graph_max_hops)
                rec = s.run(
                    cypher,
                    entity_a=entities[0],
                    entity_b=entities[1],
                ).data()
                if rec:
                    return rec
            # Fallback: 1-hop neighborhood of the first entity
            return s.run(
                NEIGHBORS_1HOP,
                entity=entities[0],
                rel_type=None,
                limit=20,
            ).data()

    def _graph_multi_hop(self, entities: List[str]) -> List[dict]:
        if not entities:
            return []
        with self.graph.session() as s:
            return s.run(
                NEIGHBORS_2HOP,
                entity=entities[0],
                limit=30,
            ).data()
