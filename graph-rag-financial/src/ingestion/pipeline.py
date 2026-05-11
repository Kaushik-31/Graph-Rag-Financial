"""End-to-end ingestion pipeline.

Reads documents from an input directory, chunks them, embeds and upserts to
Pinecone, extracts entities and relations with GPT-4, and writes them to Neo4j.

Usage:
    python -m src.ingestion.pipeline --input data/raw/sec_filings/
"""
import argparse
from pathlib import Path

from tqdm import tqdm

from src.graph.neo4j_writer import Neo4jWriter
from src.graph.vector_store import VectorStore
from src.ingestion.chunker import chunk_document
from src.ingestion.extractor import (
    extract_entities_and_relations,
    merge_extractions,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def read_documents(input_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, text) tuples for supported file types."""
    docs = []
    for path in sorted(input_dir.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def run_pipeline(input_dir: Path) -> None:
    logger.info("Reading documents from %s", input_dir)
    docs = read_documents(input_dir)
    if not docs:
        logger.warning("No documents found.")
        return

    vector_store = VectorStore()
    graph = Neo4jWriter()
    graph.setup_constraints()

    try:
        for filename, text in tqdm(docs, desc="Documents"):
            chunks = chunk_document(text, source=filename)
            vector_store.upsert_chunks(chunks)

            extractions = []
            for chunk in chunks:
                extractions.append(extract_entities_and_relations(chunk.text))
            entities, relations = merge_extractions(extractions)

            graph.upsert_entities(entities)
            graph.upsert_relations(relations)
            logger.info(
                "Processed %s: %d chunks, %d entities, %d relations",
                filename, len(chunks), len(entities), len(relations),
            )
    finally:
        graph.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Graph RAG")
    parser.add_argument("--input", required=True, help="Input directory with .txt/.md files")
    args = parser.parse_args()
    run_pipeline(Path(args.input))


if __name__ == "__main__":
    main()
