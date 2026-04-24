"""Document loading and chunking."""
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(docs_path: str) -> List[Document]:
    """Load all documents from a directory."""
    path = Path(docs_path)
    documents = []

    for file in path.rglob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix in (".txt", ".md"):
            loader = TextLoader(str(file))
        else:
            continue
        documents.extend(loader.load())

    return documents


def chunk_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(documents)
