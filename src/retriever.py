import json
import sys
from typing import Any, Dict, List

from src.bm25 import BM25
from src.semantic import Semantic
from src.hybrid_retrieval import rrf

try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    print("Module not installed")
    sys.exit(1)
from src.models import (MinimalSearchResults, MinimalSource, RagDataset,
                        StudentSearchResults)


def load_json(path: str) -> Any:
    with open(path, "r") as file:
        content = json.load(file)
    return content


def search(
    chunks_path: str,
    bm25: BM25,
    semantic: Semantic,
    query: str,
    k: int,
) -> List[Dict[str, Any]]:
    chunks = load_json(chunks_path)
    idx_bm25 = [idx for _, idx in bm25.searcher(query, k=k)]
    idx_semantic = semantic.search(query, k=k)
    results = rrf(idx_bm25, idx_semantic, k)
    found: list = []
    for idx in results:
        source = MinimalSource(
            file_path=chunks[idx]["file_path"],
            first_character_index=chunks[idx]["start"],
            last_character_index=chunks[idx]["end"],
        )
        found.append(dict(source))
    return found


def search_dataset(
    chunks_path: str,
    index_path: str,
    embeddings_path: str,
    dataset_path: str,
    output_path: str,
    k: int,
) -> None:
    all_results = []
    dataset = RagDataset(**load_json(dataset_path))
    questions_dataset = dataset.rag_questions
    questions = [question for question in questions_dataset]
    bm25 = BM25()
    semantic = Semantic()
    bm25.load(index_path)
    semantic.load(embeddings_path)
    chunks = load_json(chunks_path)
    for query in tqdm(questions, desc="Searching"):
        idx_bm25 = [idx for _, idx in bm25.searcher(query.question, k=k)]
        idx_semantic = semantic.search(query.question, k=k)
        results = rrf(idx_bm25, idx_semantic, k)
        found: list = []
        for idx in results:
            source = MinimalSource(
                file_path=chunks[idx]["file_path"],
                first_character_index=chunks[idx]["start"],
                last_character_index=chunks[idx]["end"],
            )
            found.append(source)
        search_result = MinimalSearchResults(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=found,
        )
        all_results.append(search_result)

    output = StudentSearchResults(search_results=all_results, k=k)

    with open(output_path, "w") as file:
        json.dump(output.model_dump(), file, indent=2)
    print(f"Saved {len(all_results)} results to {output_path}")
