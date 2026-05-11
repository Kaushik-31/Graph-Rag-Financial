"""Answer synthesizer: takes hybrid retrieval context and produces final answer."""
from langchain_openai import ChatOpenAI

from src.retrieval.retriever import RetrievalResult
from src.utils.config import settings


SYNTHESIS_PROMPT = """You are a financial research assistant. Answer the user's question using ONLY the evidence provided below. Cite sources inline as [G1], [V1], etc., matching the evidence labels.

If the evidence is insufficient, say so explicitly. Do not fabricate companies, numbers, or relationships.

Question: {question}

Evidence:
{context}

Answer:
"""


def synthesize(result: RetrievalResult) -> str:
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    )
    prompt = SYNTHESIS_PROMPT.format(
        question=result.question,
        context=result.to_context() or "(no evidence retrieved)",
    )
    response = llm.invoke(prompt)
    return response.content
