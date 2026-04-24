"""FAISS dense retriever."""
import faiss
import numpy as np
from typing import List, Tuple
from langchain.schema import Document


class DenseRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.documents: List[Document] = []

    def build(self, documents: List[Document]):
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed(texts).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vecs
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_vec = self.embedder.embed_query(query).astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        return [(self.documents[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
