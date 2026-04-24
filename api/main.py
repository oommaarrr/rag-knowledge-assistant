"""FastAPI service layer."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

from ingest.loader import load_documents, chunk_documents
from ingest.embedder import Embedder
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import Reranker
from generation.generator import generate

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

# Initialise pipeline on startup
embedder = Embedder()
dense_retriever = DenseRetriever(embedder)
sparse_retriever = SparseRetriever()
reranker = Reranker()
_ready = False


@app.on_event("startup")
def startup():
    global _ready
    docs_path = os.getenv("DOCS_PATH", "./documents")
    docs = load_documents(docs_path)
    chunks = chunk_documents(docs)
    dense_retriever.build(chunks)
    sparse_retriever.build(chunks)
    _ready = True
    print(f"Pipeline ready. Indexed {len(chunks)} chunks from {len(docs)} documents.")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ready" if _ready else "loading"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    dense_results = dense_retriever.retrieve(request.question, top_k=20)
    sparse_results = sparse_retriever.retrieve(request.question, top_k=20)
    hybrid_results = reciprocal_rank_fusion(dense_results, sparse_results, top_k=10)
    reranked = reranker.rerank(request.question, hybrid_results, top_k=request.top_k)
    context_docs = [doc for doc, _ in reranked]

    result = generate(request.question, context_docs)
    return QueryResponse(**result)
