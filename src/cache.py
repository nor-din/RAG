import json
from src.retriever import search
from src.bm25 import BM25
from src.semantic import Semantic


class Cache:
    def __init__(self) -> None:
       self.query_cached: dict = {}
       self.bm25 = None
       self.semantic = None
    def get_index(self, index_path, embeddings_path):
        if self.bm25 is None:
            self.bm25 = BM25()
            self.bm25.load(index_path)
        if self.semantic is None:
            self.semantic = Semantic()
            self.semantic.load(embeddings_path)
        return self.bm25, self.semantic
    def cache_query(self,chunks_path, index_path, embeddings_path, query, k):
        key = f"{query}-{k}"
        if key in self.query_cached:
            print("query founded in cache")
            return self.query_cached[key]
        print("Cache miss. Searching...")
        bm25, semantic = self.get_index(index_path, embeddings_path)
        result = search(chunks_path, bm25, semantic, query, k)
        self.query_cached[key] = result
        return self.query_cached[key]
    def save(self, path):
        with open(path, 'w') as file:
            json.dump(self.query_cached, file, indent=2)
    def load(self, path):
        with open(path, 'r') as file:
            self.query_cached = json.load(file)