"""CI test suite."""
import pytest
from langchain_core.documents import Document
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import reciprocal_rank_fusion


def test_sparse_retriever():
    docs = [
        Document(page_content="vendor approval process requires three signatures"),
        Document(page_content="holiday policy allows 25 days per year"),
        Document(page_content="expense claims must be submitted within 30 days"),
    ]
    retriever = SparseRetriever()
    retriever.build(docs)
    results = retriever.retrieve("vendor approval", top_k=2)
    assert len(results) > 0
    assert "vendor" in results[0][0].page_content.lower()


def test_rrf_fusion():
    doc_a = Document(page_content="vendor approval")
    doc_b = Document(page_content="holiday policy")
    dense = [(doc_a, 0.9), (doc_b, 0.7)]
    sparse = [(doc_a, 5.0), (doc_b, 2.0)]
    fused = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(fused) == 2
    assert fused[0][0].page_content == "vendor approval"


def test_health_endpoint():
    from fastapi.testclient import TestClient
    import os
    os.environ["DOCS_PATH"] = "./tests/fixtures"
    # just check the app imports cleanly
    from api.main import app
    assert app is not None
