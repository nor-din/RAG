import json
import torch
from transformers import pipeline


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

def genrate_answer(pipe, question, chunks_retriever, chunks_path):
    context = []
    all_chunks = load_chunk(chunks_path)

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
    return assisant.split("</think>")[1].strip()