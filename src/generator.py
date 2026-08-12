"""Model loading and answer generation support for the RAG system."""

import json
import sys
from typing import Any, Dict, List

try:
    import torch
    from tqdm import tqdm
    from transformers import pipeline
except (ModuleNotFoundError, ImportError) as e:
    print(f"Module not installed: {e}")
    sys.exit(1)
from src.models import (MinimalAnswer, StudentSearchResults,
                        StudentSearchResultsAndAnswer)


def load_chunk(path: str) -> Any:
    """Load chunks from a JSON file.

    Args:
        path: Path to the chunk metadata JSON.

    Returns:
        Parsed chunk data loaded from disk.
    """
    try:
        with open(path, "r") as file:
            chunks = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    return chunks


def load_model() -> Any:
    """Load the text generation model.

    Returns:
        A text generation pipeline instantiated with Qwen/Qwen3-0.6B.
    """
    pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B",
                    dtype=torch.float16)
    return pipe


def genrate_answer(
    pipe: Any,
    question: str,
    all_chunks: List[Dict[str, Any]],
    chunks_retriever: List[Dict[str, Any]],
) -> str:
    """Generate an answer using retrieved chunks.

    Args:
        pipe: The text generation pipeline.
        question: The user question to answer.
        all_chunks: All indexed chunks from the repository.
        chunks_retriever: Retrieved chunk metadata to use as context.

    Returns:
        A generated answer string.
    """
    context = []

    for results in chunks_retriever:
        for chunk in all_chunks:
            if (
                results["file_path"] == chunk["file_path"]
                and results["first_character_index"] == chunk["start"]
            ):
                context.append(chunk["text"])
                break
    perfect_chunks = "\n\n".join(context)
    user_prompt = f"/no_think\nContext: {perfect_chunks}\nQuestion: {question}"
    prompt = [
        {"role": "system", "content": "Answer using only the context."},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    output = pipe(prompt, max_new_tokens=300)
    assisant = output[0]["generated_text"][2]["content"]
    result1: str = assisant.split("</think>")[-1].strip()
    result2: str = assisant.strip()
    if "</think>" in assisant:
        return result1
    return result2


def genrate_dataset(
    pipe: Any,
    all_chunks: List[Dict[str, Any]],
    dataset_path: str,
    output_path: str,
) -> None:
    """Generate answers for a dataset and save results.

    Args:
        pipe: The text generation pipeline.
        all_chunks: All indexed chunks from the repository.
        dataset_path: Path to the student search results JSON.
        output_path: Path to save the generated answers JSON.
    """
    all_results = []
    dataset = StudentSearchResults(**load_chunk(dataset_path))
    questions = dataset.search_results
    for query in tqdm(questions, desc="Generating answers"):
        answer = genrate_answer(
            pipe, query.question, all_chunks, query.retrieved_sources
        )
        results = MinimalAnswer(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=query.retrieved_sources,
            answer=answer,
        )
        all_results.append(results)
    output = StudentSearchResultsAndAnswer(
        search_results=all_results,
        k=dataset.k,
    )
    with open(output_path, "w") as file:
        json.dump(output.model_dump(), file, indent=2)
