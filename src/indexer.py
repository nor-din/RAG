import json
from typing import Any

from src.bm25 import BM25
from src.semantic import Semantic


def load_chunk(path: str) -> Any:
    with open(path, "r") as file:
        chunks = json.load(file)
    return chunks


def indexer(chunks_path: str, index_path: str, embeddings_path: str) -> None:
    chunks = load_chunk(chunks_path)
    corpus = [chunk["text"] for chunk in chunks]
    retriever_bm25 = BM25()
    retriever_bm25.index(corpus)
    retriever_bm25.save(index_path)
    retriever_semantic = Semantic()
    retriever_semantic.index(corpus)
    retriever_semantic.save(embeddings_path)
