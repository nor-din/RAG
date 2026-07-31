import sys
import json
try:
    import torch
    from transformers import pipeline
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    print("Module not installed")
    sys.exit(1)
from src.models import StudentSearchResultsAndAnswer, MinimalAnswer, StudentSearchResults


def load_chunk(path):
    with open(path, "r") as file:
        chunks = json.load(file)
    return chunks

def load_model():
    pipe =pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B",
        dtype=torch.float16
    )
    return pipe

def genrate_answer(pipe, question, all_chunks ,chunks_retriever):
    context = []

    for results in chunks_retriever:
        for chunk in all_chunks:
            if results['file_path'] == chunk['file_path'] and results['first_character_index'] == chunk['start']:
                context.append(chunk['text'])
                break
    perfect_chunks = '\n\n'.join(context)
    prompt = [
        {"role": "system", "content": "Answer using only the context."},
        {"role": "user", "content": f"/no_think\nContext: {perfect_chunks}\nQuestion: {question}"}
    ]
    output = pipe(prompt, max_new_tokens=300)
    assisant = output[0]['generated_text'][2]['content']
    if "</think>" in assisant:
        return assisant.split("</think>")[-1].strip()
    return assisant.strip()

def genrate_dataset(pipe ,all_chunks, dataset_path, output_path, k=5):
    all_results = []
    dataset = StudentSearchResults(**load_chunk(dataset_path))
    questions = dataset.search_results
    for query in tqdm(questions, desc="Generating answers"):
        answer = genrate_answer(pipe, query.question, all_chunks, query.retrieved_sources)
        results = MinimalAnswer(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=query.retrieved_sources,
            answer=answer
        )
        all_results.append(results.model_dump())
    output = StudentSearchResultsAndAnswer(
        search_results=all_results,
        k=k
    )
    with open(output_path, 'w') as file:
        json.dump(output.model_dump(), file, indent=2)
