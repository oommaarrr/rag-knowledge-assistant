"""Cross-encoder reranker."""
from sentence_transformers import CrossEncoder
from typing import List, Tuple
from langchain_core.documents import Document

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        pairs = [[query, doc.page_content] for doc, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]
