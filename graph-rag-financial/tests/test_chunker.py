"""Unit tests for the chunker."""
from src.ingestion.chunker import chunk_document


def test_chunk_document_returns_chunks():
    text = "Section A. " * 200
    chunks = chunk_document(text, source="test.txt")
    assert len(chunks) > 0
    assert all(c.source == "test.txt" for c in chunks)
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


def test_chunk_document_preserves_indexing():
    text = "Item 1. " * 500
    chunks = chunk_document(text, source="filing.txt")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_document_with_extra_metadata():
    chunks = chunk_document(
        "some text",
        source="x.txt",
        extra_metadata={"company": "Apple", "year": 2023},
    )
    assert chunks[0].metadata["company"] == "Apple"
    assert chunks[0].metadata["year"] == 2023
