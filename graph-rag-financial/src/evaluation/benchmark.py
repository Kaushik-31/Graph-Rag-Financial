"""Benchmark harness comparing Graph RAG against vector-only baseline.

Loads a JSONL benchmark file with question/expected_keywords pairs, runs both
systems, and computes recall over expected keywords as a proxy for accuracy.

Usage:
    python -m src.evaluation.benchmark --suite benchmark_v1
"""
import argparse
import json
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from src.graph.vector_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.retrieval.synthesizer import synthesize
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


BASELINE_PROMPT = """Answer the question using only the context below. Cite sources as [V1], [V2], etc.

Question: {question}

Context:
{context}

Answer:
"""


def baseline_rag(question: str, vector_store: VectorStore) -> str:
    """Vanilla vector-only RAG for comparison."""
    hits = vector_store.query(question, top_k=8)
    context = "\n".join(f"[V{i+1}] {h['text']}" for i, h in enumerate(hits))
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        api_key=settings.openai_api_key,
    )
    return llm.invoke(BASELINE_PROMPT.format(question=question, context=context)).content


def keyword_recall(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in answer_lower)
    return hits / len(expected)


def run_benchmark(suite_path: Path) -> None:
    questions = [json.loads(line) for line in suite_path.read_text().splitlines() if line.strip()]
    vector_store = VectorStore()
    hybrid = HybridRetriever()

    results = []
    try:
        for q in questions:
            question = q["question"]
            expected = q.get("expected_keywords", [])
            category = q.get("category", "unknown")

            # Baseline
            t0 = time.time()
            baseline_answer = baseline_rag(question, vector_store)
            baseline_latency = time.time() - t0

            # Graph RAG
            t0 = time.time()
            retrieval = hybrid.retrieve(question)
            graph_answer = synthesize(retrieval)
            graph_latency = time.time() - t0

            results.append({
                "question": question,
                "category": category,
                "baseline_recall": keyword_recall(baseline_answer, expected),
                "graph_recall": keyword_recall(graph_answer, expected),
                "baseline_latency": baseline_latency,
                "graph_latency": graph_latency,
            })
    finally:
        hybrid.close()

    # Aggregate
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"\n{'Category':<25} {'Baseline':>10} {'Graph RAG':>12} {'Delta':>10}")
    print("-" * 60)
    for cat, rows in by_cat.items():
        b = sum(r["baseline_recall"] for r in rows) / len(rows)
        g = sum(r["graph_recall"] for r in rows) / len(rows)
        print(f"{cat:<25} {b:>10.1%} {g:>12.1%} {g-b:>+10.1%}")
    overall_b = sum(r["baseline_recall"] for r in results) / len(results)
    overall_g = sum(r["graph_recall"] for r in results) / len(results)
    print("-" * 60)
    print(f"{'OVERALL':<25} {overall_b:>10.1%} {overall_g:>12.1%} {overall_g-overall_b:>+10.1%}")

    out_path = suite_path.parent / f"{suite_path.stem}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote detailed results to %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="benchmark_v1", help="Benchmark suite name")
    parser.add_argument("--dir", default="data/processed", help="Directory containing the suite jsonl")
    args = parser.parse_args()
    suite_path = Path(args.dir) / f"{args.suite}.jsonl"
    run_benchmark(suite_path)


if __name__ == "__main__":
    main()
