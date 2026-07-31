import json
import sys
from typing import Any, Dict, List

from src.bm25 import BM25

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
    chunks_path: str, index_path: str, query: str, k: int = 5
) -> List[Dict[str, Any]]:
    chunks = load_json(chunks_path)
    retrieve = BM25()
    retrieve.load(index_path)
    results = retrieve.searcher(query, k=k)
    found: list = []
    for _, idx in results:
        source = MinimalSource(
            file_path=chunks[idx]["file_path"],
            first_character_index=chunks[idx]["start"],
            last_character_index=chunks[idx]["end"],
        )
        found.append(source)
    return found


def search_dataset(
    chunks_path: str,
    index_path: str,
    dataset_path: str,
    output_path: str,
    k: int = 5
) -> None:
    all_results = []
    dataset = RagDataset(**load_json(dataset_path))
    questions_dataset = dataset.rag_questions
    questions = [question for question in questions_dataset]
    for query in tqdm(questions, desc="Searching"):
        results = search(chunks_path, index_path, query.question, k)
        sources = [MinimalSource(**r) for r in results]
        search_result = MinimalSearchResults(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=sources,
        )
        all_results.append(search_result)

    output = StudentSearchResults(search_results=all_results, k=k)

    with open(output_path, "w") as file:
        json.dump(output.model_dump(), file, indent=2)
    print(f"Saved {len(all_results)} results to {output_path}")
