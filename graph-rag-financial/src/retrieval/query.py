"""Command-line interface for querying the Graph RAG system.

Usage:
    python -m src.retrieval.query --question "Which suppliers of Tesla partner with Nvidia?"
"""
import argparse

from src.retrieval.retriever import HybridRetriever
from src.retrieval.synthesizer import synthesize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Query the Graph RAG system")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved evidence")
    args = parser.parse_args()

    retriever = HybridRetriever()
    try:
        result = retriever.retrieve(args.question)
        if args.show_context:
            print("=" * 80)
            print("RETRIEVED CONTEXT")
            print("=" * 80)
            print(result.to_context())
            print("=" * 80)
        answer = synthesize(result)
        print("\nANSWER:\n")
        print(answer)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
