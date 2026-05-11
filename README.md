# Graph RAG Knowledge System for Financial Datasets

A hybrid retrieval system that combines **Neo4j knowledge graphs** with **Pinecone vector search** to enable multi-hop entity reasoning over complex financial documents. Outperforms standard RAG by **28% on relationship-heavy queries**.

## Overview

Standard RAG systems retrieve semantically similar text chunks but struggle with queries that require traversing relationships across entities (e.g., "Which subsidiaries of companies that supply Apple have reported declining revenue?"). This system addresses that limitation by combining:

1. **Vector retrieval** (Pinecone) for semantic similarity
2. **Graph traversal** (Neo4j) for multi-hop entity reasoning
3. **GPT-4 orchestration** for query planning and answer synthesis

## Architecture

```
                    User Query
                         |
                         v
              +----------------------+
              |  Query Router (LLM)  |
              +----------------------+
                /                  \
               v                    v
    +-------------------+   +--------------------+
    | Pinecone Vector   |   | Neo4j Knowledge    |
    | Search (semantic) |   | Graph (relational) |
    +-------------------+   +--------------------+
               \                    /
                v                  v
              +----------------------+
              |  Context Aggregator  |
              +----------------------+
                         |
                         v
              +----------------------+
              | GPT-4 Answer Synth.  |
              +----------------------+
                         |
                         v
                    Final Answer
```

## Key Features

- **Hybrid retrieval** combining dense vector search with graph traversal
- **Entity extraction pipeline** using GPT-4 to build knowledge graphs from SEC filings, earnings reports, and news
- **Multi-hop Cypher generation** for complex relationship queries
- **Configurable retrieval depth** (1-hop, 2-hop, 3-hop neighborhoods)
- **Evaluation harness** with comparison against baseline RAG

## Results

Evaluated on a benchmark of 200 financial queries split into three categories:

| Query Type            | Standard RAG | Graph RAG    | Improvement |
|-----------------------|--------------|--------------|-------------|
| Single-entity factual | 84.2%        | 85.1%        | +0.9%       |
| Multi-entity relation | 51.3%        | 79.6%        | **+28.3%**  |
| Multi-hop reasoning   | 38.7%        | 71.2%        | **+32.5%**  |
| **Overall**           | **58.1%**    | **78.6%**    | **+20.5%**  |

Latency: median 1.4s (Graph RAG) vs 0.9s (Standard RAG). The added graph traversal cost is justified for relationship queries.

## Tech Stack

- **Neo4j 5.x** - Knowledge graph storage and Cypher query execution
- **Pinecone** - Managed vector database for dense retrieval
- **LangChain** - Orchestration framework for retrieval chains
- **OpenAI GPT-4** - Entity extraction, query routing, answer synthesis
- **OpenAI text-embedding-3-large** - Dense embeddings (3072 dim)
- **Python 3.10+**

## Quick Start

### Prerequisites

```bash
# Required services
- Neo4j 5.x (local or Aura)
- Pinecone account with index created
- OpenAI API key
```

### Installation

```bash
git clone https://github.com/Kaushik-31/graph-rag-financial.git
cd graph-rag-financial
pip install -r requirements.txt
cp .env.example .env  # fill in credentials
```

### Run the pipeline

```bash
# 1. Ingest documents and build graph + vector index
python -m src.ingestion.pipeline --input data/raw/sec_filings/

# 2. Query the system
python -m src.retrieval.query --question "Which suppliers of Tesla have R&D partnerships with Nvidia?"

# 3. Run evaluation
python -m src.evaluation.benchmark --suite benchmark_v1
```

## Project Structure

```
graph-rag-financial/
├── src/
│   ├── ingestion/        # Document parsing, entity extraction
│   ├── graph/            # Neo4j schema, Cypher templates
│   ├── retrieval/        # Hybrid retriever, query router
│   ├── evaluation/       # Benchmark harness
│   └── utils/            # Shared helpers
├── notebooks/            # Exploratory analysis
├── data/                 # Sample financial documents
├── tests/                # Unit tests
└── configs/              # YAML configs for retrieval modes
```

## How It Works

### 1. Ingestion

For each input document:
- Chunk text (recursive splitter, 512 tokens, 50 overlap)
- Embed chunks and upsert to Pinecone with metadata
- Extract entities and relations using GPT-4 with a structured schema
- Write nodes and edges to Neo4j

### 2. Query Routing

The router LLM classifies the incoming question into:
- **Factual** (single entity): vector search only
- **Relational** (2+ entities): graph traversal + vector grounding
- **Multi-hop**: graph-first with depth-2 expansion, then vector retrieval on each hop

### 3. Answer Synthesis

Retrieved subgraphs are linearized into natural language descriptions, concatenated with vector chunks, and passed to GPT-4 with a synthesis prompt that requires citations.

## Author

**Kaushik Mantha**
[GitHub](https://github.com/Kaushik-31) | [LinkedIn](https://linkedin.com/in/kaushikmantha/) | mnkaushik31@gmail.com

## License

MIT
