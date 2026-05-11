"""Entity and relation extraction using a structured GPT-4 prompt.

The schema is intentionally narrow for the financial domain. Expanding the
schema (e.g., adding LITIGATES or LICENSES) means updating both the prompt
and the Cypher templates in src/graph/cypher_templates.py.
"""
import json
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


ENTITY_TYPES = [
    "Company", "Person", "Product", "Sector",
    "Geography", "FinancialMetric", "Event",
]

RELATION_TYPES = [
    "SUBSIDIARY_OF", "SUPPLIES", "COMPETES_WITH", "PARTNERS_WITH",
    "ACQUIRED", "INVESTS_IN", "EMPLOYS", "OPERATES_IN",
    "REPORTS_METRIC", "INVOLVED_IN",
]


class Entity(BaseModel):
    name: str
    type: str = Field(..., description=f"One of: {ENTITY_TYPES}")
    aliases: List[str] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    target: str
    type: str = Field(..., description=f"One of: {RELATION_TYPES}")
    evidence: str = Field(..., description="Verbatim sentence supporting the relation")


class ExtractionResult(BaseModel):
    entities: List[Entity]
    relations: List[Relation]


EXTRACTION_PROMPT = """You are a financial knowledge graph extractor. From the text below, extract:
1. Entities (companies, people, products, sectors, geographies, financial metrics, events)
2. Relations between entities

Use ONLY these entity types: {entity_types}
Use ONLY these relation types: {relation_types}

For each relation, include the verbatim sentence from the text that supports it.
Skip relations you are not confident about.

Return STRICT JSON matching this schema:
{{
  "entities": [{{"name": "...", "type": "...", "aliases": []}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "evidence": "..."}}]
}}

Text:
{text}
"""


def extract_entities_and_relations(text: str) -> ExtractionResult:
    """Run GPT-4 extraction over a chunk of text."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    prompt = EXTRACTION_PROMPT.format(
        entity_types=ENTITY_TYPES,
        relation_types=RELATION_TYPES,
        text=text,
    )
    response = llm.invoke(prompt)
    try:
        payload = json.loads(response.content)
        return ExtractionResult(**payload)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Extraction failed for chunk: %s", exc)
        return ExtractionResult(entities=[], relations=[])


def merge_extractions(results: List[ExtractionResult]) -> Tuple[List[Entity], List[Relation]]:
    """Deduplicate entities by lowercase name; keep all relations."""
    seen: dict[str, Entity] = {}
    for r in results:
        for ent in r.entities:
            key = ent.name.lower().strip()
            if key not in seen:
                seen[key] = ent
            else:
                # Merge aliases
                existing = seen[key]
                merged = list(set(existing.aliases) | set(ent.aliases))
                seen[key] = Entity(name=existing.name, type=existing.type, aliases=merged)
    relations = [rel for r in results for rel in r.relations]
    return list(seen.values()), relations
