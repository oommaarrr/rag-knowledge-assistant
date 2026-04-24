"""BM25 sparse retriever."""
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from langchain.schema import Document


class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]):
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]
