import os
import sys
import fire

from src.chunker import chunker, save_chunks
from src.evaluation import evaluation
from src.generator import (genrate_answer, genrate_dataset, load_chunk,
                           load_model)
from src.incremental import incremental
from src.retriever import search as ft_search
from src.retriever import search_dataset as ft_search_dataset

REPO_PATH = "data/raw/vllm-0.10.1"
CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/bm25_index.json"
EMBEDDINGS_PATH = "data/processed/embeddings.pt"
TIME_PATH = "data/processed/time_files.json"


class CLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        try:
            if max_chunk_size <= 0:
                print("Error: max_chunk_size must be > 0")
                return
            if not os.path.exists(REPO_PATH):
                print(f"Error: Repository not found at {REPO_PATH}")
                return
            chunks = chunker(REPO_PATH, max_chunk_size)
            save_chunks(chunks, CHUNKS_PATH)
            incremental(REPO_PATH,TIME_PATH,CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH, max_chunk_size)
        except Exception as e:
            print(f"Error indexing: {e}")
            sys.exit(1)

    def search(self, query: str, k: int = 10) -> None:
        try:
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            if not os.path.exists(INDEX_PATH):
                print(f"Error: Repository not found at {INDEX_PATH}")
                return
            if not os.path.exists(EMBEDDINGS_PATH):
                print(f"Error: Repository not found at {EMBEDDINGS_PATH}")
                return
            if k <= 0:
                print("Error: k must be > 0")
                return
            if not query or not query.strip():
                print("Error: query cannot empty")
                return
            result = ft_search(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH,query, k)
            for r in result:
                print(r)
        except Exception as e:
            print(f"Error searching: {e}")
            sys.exit(1)

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str = "data/output/search_results",
        k: int = 10,
    ) -> None:

        try:
            os.makedirs(save_directory, exist_ok=True)
            file_name = os.path.basename(dataset_path)
            output_path = os.path.join(save_directory, file_name)
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            if not os.path.exists(INDEX_PATH):
                print(f"Error: Repository not found at {INDEX_PATH}")
                return
            if not os.path.exists(EMBEDDINGS_PATH):
                print(f"Error: Repository not found at {EMBEDDINGS_PATH}")
                return
            if k <= 0:
                print("Error: k must be > 0")
                return
            ft_search_dataset(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH,
                              dataset_path, output_path, k)
        except Exception as e:
            print(f"Error searching dataset: {e}")
            sys.exit(1)

    def answer(self, query: str, k: int = 10) -> None:
        try:
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            if not os.path.exists(INDEX_PATH):
                print(f"Error: Repository not found at {INDEX_PATH}")
                return
            if not os.path.exists(EMBEDDINGS_PATH):
                print(f"Error: Repository not found at {EMBEDDINGS_PATH}")
                return
            if k <= 0:
                print("Error: k must be > 0")
                return
            if not query or not query.strip():
                print("Error: query cannot empty")
                return
            pipe = load_model()
            chunks_retriever = ft_search(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH, query, k)
            all_chunks = load_chunk(CHUNKS_PATH)
            answer = genrate_answer(pipe, query, all_chunks, chunks_retriever)
            print("----------------------------------------------------------")
            print(answer)
        except Exception as e:
            print(f"Error answer: {e}")
            sys.exit(1)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        k: int = 10,
    ) -> None:
        try:
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            if k <= 0:
                print("Error: k must be > 0")
                return
            os.makedirs(save_directory, exist_ok=True)
            file_name = os.path.basename(student_search_results_path)
            output_path = os.path.join(save_directory, file_name)
            pipe = load_model()
            all_chunks = load_chunk(CHUNKS_PATH)
            genrate_dataset(
                pipe, all_chunks, student_search_results_path, output_path, k
            )
            print(f"Saved student_search_results to {output_path}")
        except Exception as e:
            print(f"Error answer dataset: {e}")
            sys.exit(1)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str, k: int = 10
    ) -> None:
        try:
            if k <= 0:
                print("Error: k must be > 0")
                return
            evaluation(student_search_results_path, dataset_path, k)
        except Exception as e:
            print(f"Error answer dataset: {e}")
            sys.exit(1)


if __name__ == "__main__":
    fire.Fire(CLI)
