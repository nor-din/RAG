"""Command-line interface entrypoint for the RAG project.

This module exposes the core CLI commands used to index the corpus,
search both individual queries and datasets, generate answers, evaluate
recall metrics, and launch the HTTP API.
"""

import os
import sys

from src.cache import Cache
from src.chunker import chunker, save_chunks
from src.evaluation import evaluation
from src.generator import (genrate_answer, genrate_dataset, load_chunk,
                           load_model)
from src.indexer import indexer
from src.incremental import incremental, save_times
from src.retriever import search as ft_search
from src.retriever import search_dataset as ft_search_dataset
try:
    import fire
    import uvicorn
except (ModuleNotFoundError, ImportError):
    print("Module not installed")
    sys.exit(1)

REPO_PATH = "data/raw"
CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "data/processed/tfidf_index.json"
EMBEDDINGS_PATH = "data/processed/embeddings.pt"
TIME_PATH = "data/processed/time_files.json"
CACHE_PATH = "data/processed/query_cache.json"


class CLI:
    """Command-line interface for indexing, retrieval, and generation."""

    def __init__(self) -> None:
        """Initialize the CLI with a cached query store."""
        self.cache = Cache()
        if os.path.exists(CACHE_PATH):
            self.cache.load(CACHE_PATH)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Create document chunks and build the search index.

        Args:
            max_chunk_size: Maximum size in characters for each chunk.
        """
        try:
            if max_chunk_size <= 0:
                print("Error: max_chunk_size must be > 0")
                return
            if not os.path.exists(REPO_PATH):
                print(f"Error: Repository not found at {REPO_PATH}")
                return
            chunks = chunker(REPO_PATH, max_chunk_size)
            save_chunks(chunks, CHUNKS_PATH)
            indexer(CHUNKS_PATH, INDEX_PATH, EMBEDDINGS_PATH)
            save_times(REPO_PATH, TIME_PATH)
            print("Full indexing complete!")
        except Exception as e:
            print(f"Error indexing: {e}")
            sys.exit(1)

    def incremental_index(self, max_chunk_size: int = 2000) -> None:
        """Run incremental re-indexing for files changed since last index.

        Args:
            max_chunk_size: Maximum size in characters for each generated chunk.
        """
        try:
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            if not os.path.exists(TIME_PATH):
                print("No timestamps found. Running full index.")
                self.index(max_chunk_size)
                return
            incremental(
                REPO_PATH,
                TIME_PATH,
                CHUNKS_PATH,
                INDEX_PATH,
                EMBEDDINGS_PATH,
                max_chunk_size,
            )
            print("Incremental indexing complete!")
        except Exception as e:
            print(f"Error incremental: {e}")
            sys.exit(1)

    def search(self, query: str, k: int) -> None:
        """Search the index for a single query and print the results.

        Args:
            query: Text query to search for.
            k: Number of top results to return.
        """
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
                print("Error: query cannot be empty")
                return
            result = self.cache.cache_query(
                CHUNKS_PATH,
                INDEX_PATH,
                EMBEDDINGS_PATH,
                query,
                k,
            )
            self.cache.save(CACHE_PATH)
            for r in result:
                print(r)
        except Exception as e:
            print(f"Error searching: {e}")
            sys.exit(1)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Run search for a dataset and write StudentSearchResults output.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of top results to return for each question.
            save_directory: Output directory for result JSON files.
        """
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
            ft_search_dataset(
                CHUNKS_PATH,
                INDEX_PATH,
                EMBEDDINGS_PATH,
                dataset_path,
                output_path,
                k,
            )
        except Exception as e:
            print(f"Error searching dataset: {e}")
            sys.exit(1)

    def answer(self, query: str, k: int) -> None:
        """Generate and print an answer for a single query.

        Args:
            query: The user question to answer.
            k: Number of top chunks to retrieve from the index.
        """
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
                print("Error: query cannot be empty")
                return
            pipe = load_model()
            tfidf, semantic = self.cache.get_index(
                INDEX_PATH,
                EMBEDDINGS_PATH,
            )
            chunks_retriever = ft_search(
                CHUNKS_PATH,
                tfidf,
                semantic,
                query,
                k,
            )
            all_chunks = load_chunk(CHUNKS_PATH)
            answer = genrate_answer(
                pipe,
                query,
                all_chunks,
                chunks_retriever,
            )
            print(
                "----------------------------------------------------------"
            )
            print(answer)
        except Exception as e:
            print(f"Error answer: {e}")
            sys.exit(1)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset and save JSON output.

        Args:
            student_search_results_path: Path to the student search results JSON.
            save_directory: Directory where answer results are written.
        """
        try:
            if not os.path.exists(CHUNKS_PATH):
                print(f"Error: Repository not found at {CHUNKS_PATH}")
                return
            os.makedirs(save_directory, exist_ok=True)
            file_name = os.path.basename(student_search_results_path)
            output_path = os.path.join(save_directory, file_name)
            pipe = load_model()
            all_chunks = load_chunk(CHUNKS_PATH)
            genrate_dataset(
                pipe,
                all_chunks,
                student_search_results_path,
                output_path,
            )
            print(
                f"Saved student_search_results to {output_path}"
            )
        except Exception as e:
            print(f"Error answer dataset: {e}")
            sys.exit(1)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Evaluate recall metrics for a search results file against ground truth.

        Args:
            student_search_results_path: Path to the student's search results JSON.
            dataset_path: Path to the ground-truth dataset JSON.
        """
        try:
            evaluation(student_search_results_path, dataset_path)
        except Exception as e:
            print(f"Error evaluating dataset: {e}")
            sys.exit(1)

    def http_api(self) -> None:
        """Launch the HTTP API for search and answer endpoints.

        This command starts the FastAPI server defined in src/api.py.
        """
        try:
            uvicorn.run("src.api:app")
        except Exception as e:
            print(f"Error HTTP API: {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("exit by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
