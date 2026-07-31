import json
from typing import Any

from src.bm25 import BM25


def load_chunk(path: str) -> Any:
    with open(path, "r") as file:
        chunks = json.load(file)
    return chunks


def indexer(chunks_path: str, index_path: str) -> None:
    chunks = load_chunk(chunks_path)
    corpus = [chunk["text"] for chunk in chunks]
    retriever = BM25()
    retriever.index(corpus)
    retriever.save(index_path)
