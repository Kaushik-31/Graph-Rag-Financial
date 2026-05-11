"""Pinecone wrapper for embedding upsert and similarity search."""
from typing import List

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from src.ingestion.chunker import Chunk
from src.utils.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self._ensure_index()
        self.index = self.pc.Index(settings.pinecone_index)
        self.embedder = OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
        )

    def _ensure_index(self) -> None:
        existing = [i.name for i in self.pc.list_indexes()]
        if settings.pinecone_index not in existing:
            logger.info("Creating Pinecone index %s", settings.pinecone_index)
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 96) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            vectors = self.embedder.embed_documents(texts)
            payload = [
                {
                    "id": f"{c.source}::{c.chunk_index}",
                    "values": vec,
                    "metadata": {**c.metadata, "text": c.text},
                }
                for c, vec in zip(batch, vectors)
            ]
            self.index.upsert(vectors=payload, namespace=settings.pinecone_namespace)
        logger.info("Upserted %d chunks to Pinecone", len(chunks))

    def query(self, text: str, top_k: int | None = None) -> List[dict]:
        k = top_k or settings.vector_top_k
        vec = self.embedder.embed_query(text)
        result = self.index.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"].get("text", ""),
                "metadata": match["metadata"],
            }
            for match in result["matches"]
        ]
