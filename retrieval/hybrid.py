"""Hybrid retrieval: fuse dense + sparse scores with RRF."""
from typing import List, Tuple
from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    """Combine dense and sparse results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(sparse_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[k], s) for k, s in ranked[:top_k]]
