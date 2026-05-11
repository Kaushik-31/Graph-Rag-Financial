"""LLM-based query router.

Decides which retrieval strategy to use:
- factual: vector only
- relational: graph + vector
- multi_hop: graph-first multi-hop, then vector grounding
"""
import json
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.utils.config import settings


QueryType = Literal["factual", "relational", "multi_hop"]


class RouterDecision(BaseModel):
    query_type: QueryType
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities mentioned in the query",
    )
    rationale: str


ROUTER_PROMPT = """Classify the following financial question into one of three retrieval strategies:

- "factual": question is about a single entity or a single fact (e.g., "What was Apple's 2023 revenue?")
- "relational": question asks about a direct relationship between two named entities (e.g., "Does TSMC supply Nvidia?")
- "multi_hop": question requires traversing 2+ relationships (e.g., "Which suppliers of Tesla have partnerships with Nvidia?")

Also extract any named entities (companies, people, products) mentioned.

Return STRICT JSON:
{{
  "query_type": "factual" | "relational" | "multi_hop",
  "entities": ["..."],
  "rationale": "brief reason"
}}

Question: {question}
"""


def route_query(question: str) -> RouterDecision:
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    response = llm.invoke(ROUTER_PROMPT.format(question=question))
    payload = json.loads(response.content)
    return RouterDecision(**payload)
