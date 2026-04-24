# RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) pipeline that turns a folder of documents into a queryable knowledge base with grounded, source-cited answers.

Built as a freelance client project. The client had hundreds of internal policy documents and procedures — lookups that took 15–20 minutes were reduced to under 10 seconds.

## Architecture

```
Documents → Chunking → FAISS (dense) + BM25 (sparse) → Cross-encoder reranking → LLM generation → FastAPI
```

- **Dense retrieval**: FAISS vector store with sentence-transformer embeddings
- **Sparse retrieval**: BM25 keyword index for recall on exact terms
- **Reranking**: Cross-encoder model re-scores top candidates before generation
- **Generation**: OpenAI/Anthropic API with grounded prompts and source citations
- **Service layer**: FastAPI REST API with streaming support
- **Containerised**: Docker + docker-compose for easy deployment
- **CI**: GitHub Actions runs tests on every commit

## Stack

- Python 3.11
- `langchain`, `faiss-cpu`, `rank-bm25`, `sentence-transformers`
- `fastapi`, `uvicorn`
- `openai` / `anthropic`
- Docker, GitHub Actions

## Quickstart

```bash
git clone https://github.com/oommaarrr/rag-knowledge-assistant
cd rag-knowledge-assistant
cp .env.example .env  # add your API key
docker-compose up
```

Then POST to `http://localhost:8000/query`:

```json
{
  "question": "What is the process for approving a new vendor?",
  "top_k": 5
}
```

## Evaluation

Each pipeline stage is independently evaluated:

| Stage | Metric | Result |
|---|---|---|
| Retrieval (dense) | Recall@5 | 0.87 |
| Retrieval (hybrid) | Recall@5 | 0.93 |
| After reranking | Precision@3 | 0.89 |
| Generation | Faithfulness | 0.91 |

## Project Structure

```
rag-knowledge-assistant/
├── ingest/
│   ├── loader.py          # Document loading and chunking
│   ├── embedder.py        # Embedding generation
│   └── indexer.py         # FAISS + BM25 index building
├── retrieval/
│   ├── dense.py           # FAISS retriever
│   ├── sparse.py          # BM25 retriever
│   ├── hybrid.py          # Score fusion
│   └── reranker.py        # Cross-encoder reranking
├── generation/
│   ├── prompt.py          # Prompt templates
│   └── generator.py       # LLM call + source citation
├── api/
│   └── main.py            # FastAPI app
├── eval/
│   └── evaluate.py        # Pipeline evaluation scripts
├── tests/
│   └── test_pipeline.py   # CI test suite
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt
