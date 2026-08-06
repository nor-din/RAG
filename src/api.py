import os
import sys
from fastapi import FastAPI, HTTPException
from src.__main__ import CLI
from src.generator import load_model, genrate_answer, load_chunk
from src.retriever import search as ft_search
from src.cache import Cache

CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/bm25_index.json"
EMBEDDINGS_PATH = "data/processed/embeddings.pt"
CACHE_PATH = "data/processed/query_cache.json"

cache = Cache()
if os.path.exists(CACHE_PATH):
    cache.load(CACHE_PATH)
app = FastAPI()

@app.get("/")
def root():
    return """/search  → find relevant chunks
            /answer  → generate answer"""

@app.get("/search")
def search(query: str, k: int):
    try:
        if not os.path.exists(CHUNKS_PATH):
            return f"Error: Repository not found at {CHUNKS_PATH}"
        if not os.path.exists(INDEX_PATH):
            return f"Error: Repository not found at {INDEX_PATH}"
        if not os.path.exists(EMBEDDINGS_PATH):
            return f"Error: Repository not found at {EMBEDDINGS_PATH}"
        if k <= 0:
            return "Error: k must be > 0"
        if not query or not query.strip():
            return "Error: query cannot empty"
        result = cache.cache_query(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH, query, k)
        cache.save(CACHE_PATH)
        return {"result": [dict(r) for r in result]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/answer")
def answer(query: str, k: int):
    try:
        if not os.path.exists(CHUNKS_PATH):
            return f"Error: Repository not found at {CHUNKS_PATH}"
        if not os.path.exists(INDEX_PATH):
            return f"Error: Repository not found at {INDEX_PATH}"
        if not os.path.exists(EMBEDDINGS_PATH):
            return f"Error: Repository not found at {EMBEDDINGS_PATH}"
        if k <= 0:
            return "Error: k must be > 0"
        if not query or not query.strip():
            return "Error: query cannot empty"
        pipe = load_model()
        chunks_retriever = cache.cache_query(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH, query, k)
        all_chunks = load_chunk(CHUNKS_PATH)
        answer = genrate_answer(pipe, query, all_chunks, chunks_retriever)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
