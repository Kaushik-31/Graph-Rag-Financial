"""Tests for RetrievalResult context formatting."""
from src.retrieval.retriever import RetrievalResult
from src.retrieval.router import RouterDecision


def make_decision():
    return RouterDecision(query_type="relational", entities=["Tesla", "Nvidia"], rationale="test")


def test_to_context_with_graph_and_vector_hits():
    result = RetrievalResult(
        question="test?",
        decision=make_decision(),
        vector_hits=[{"text": "Apple revenue grew 5%", "metadata": {"source": "10K.txt"}}],
        graph_hits=[
            {"source": "Tesla", "relation": "SUPPLIES", "target": "Nvidia", "evidence": "Tesla supplies Nvidia."}
        ],
    )
    context = result.to_context()
    assert "Knowledge Graph Evidence" in context
    assert "Document Evidence" in context
    assert "Tesla" in context
    assert "10K.txt" in context


def test_to_context_with_path_hit():
    result = RetrievalResult(
        question="test?",
        decision=make_decision(),
        graph_hits=[{"node_names": ["A", "B", "C"], "relations": ["R1", "R2"], "hops": 2}],
    )
    context = result.to_context()
    assert "A -> B -> C" in context
    assert "hops: 2" in context


def test_to_context_empty():
    result = RetrievalResult(question="test?", decision=make_decision())
    assert result.to_context() == ""
