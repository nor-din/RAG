import os
import fire
from src.chunker import chunker, save_chunks
from src.indexer import indexer
from src.retriever import search as ft_search
from src.retriever import search_dataset as ft_search_dataset

REPO_PATH    = "data/raw/vllm-0.10.1"
CHUNKS_PATH  = "data/processed/chunks.json"
INDEX_PATH   = "data/processed/bm25_index.json"


class CLI:

    def index(self, max_chunk_size=2000):
        chunks = chunker(REPO_PATH, max_chunk_size)
        save_chunks(chunks, CHUNKS_PATH)
        indexer(CHUNKS_PATH, INDEX_PATH)

    def search(self, query, k=10):
        result = ft_search(CHUNKS_PATH, INDEX_PATH, query, k)
        if not query or not query.strip():
            print("Error: query cannot empty")
            return
        for r in result:
            print(r)
    
    def search_dataset(self, dataset_path, save_directory="data/output/search_results", k=10):
        os.makedirs(save_directory, exist_ok=True)
        file_name = os.path.basename(dataset_path)
        output_path = os.path.join(save_directory, file_name)
        ft_search_dataset(CHUNKS_PATH, INDEX_PATH, dataset_path, output_path, k)

if __name__ == "__main__":
    fire.Fire(CLI)
