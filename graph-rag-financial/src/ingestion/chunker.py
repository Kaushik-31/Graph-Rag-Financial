"""Chunking strategy for financial documents.

We use a recursive character splitter with separators tuned for SEC filings
and earnings transcripts (where section headers and Q/A boundaries matter).
"""
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict


def chunk_document(text: str, source: str, extra_metadata: dict | None = None) -> List[Chunk]:
    """Split a document into overlapping chunks ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\nItem ", "\nQ:", "\nA:", "\n", ". ", " "],
        length_function=len,
    )
    pieces = splitter.split_text(text)
    extra = extra_metadata or {}
    return [
        Chunk(
            text=piece,
            source=source,
            chunk_index=i,
            metadata={"source": source, "chunk_index": i, **extra},
        )
        for i, piece in enumerate(pieces)
    ]
