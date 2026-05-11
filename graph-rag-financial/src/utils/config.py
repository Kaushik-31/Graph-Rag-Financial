"""Centralized configuration loaded from environment variables."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4-turbo")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "financial-graph-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    pinecone_dimension: int = int(os.getenv("PINECONE_DIMENSION", "3072"))

    # Retrieval
    chunk_size: int = 512
    chunk_overlap: int = 50
    vector_top_k: int = 8
    graph_max_hops: int = 2


settings = Settings()
