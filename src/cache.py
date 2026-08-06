import json
from typing import Any, Dict, Tuple

from src.bm25 import BM25
from src.retriever import search
from src.semantic import Semantic


class Cache:
    """Local cache for index objects and query results."""

    def __init__(self) -> None:
        self.query_cached: Dict[str, Any] = {}
        self.bm25: BM25 | None = None
        self.semantic: Semantic | None = None

    def get_index(
        self, index_path: str, embeddings_path: str
    ) -> Tuple[BM25, Semantic]:
        """Load and cache index objects from disk."""
        if self.bm25 is None:
            self.bm25 = BM25()
            self.bm25.load(index_path)
        if self.semantic is None:
            self.semantic = Semantic()
            self.semantic.load(embeddings_path)
        assert self.bm25 is not None
        assert self.semantic is not None
        return self.bm25, self.semantic

    def cache_query(
        self,
        chunks_path: str,
        index_path: str,
        embeddings_path: str,
        query: str,
        k: int,
    ) -> Any:
        """Search with caching for repeated query requests."""
        key = f"{query}-{k}"
        if key in self.query_cached:
            print("query founded in cache")
            return self.query_cached[key]

        print("Cache miss. Searching...")
        bm25, semantic = self.get_index(index_path, embeddings_path)
        result = search(chunks_path, bm25, semantic, query, k)
        self.query_cached[key] = result
        return self.query_cached[key]

    def save(self, path: str) -> None:
        """Serialize the cached query map to disk."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.query_cached, file, indent=2)

    def load(self, path: str) -> None:
        """Load a saved cache map from disk."""
        with open(path, "r", encoding="utf-8") as file:
            self.query_cached = json.load(file)
