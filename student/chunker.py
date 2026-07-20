import json
from pathlib import Path


def find_files(path):
    all_files = []
    folder = Path(path)
    for file in folder.rglob("*"):
        if file.suffix == ".md" or file.suffix == ".py":
            all_files.append(file)
    return all_files

def read_file(path_file):
    with open(path_file, "r", errors="ignore", encoding="utf-8") as file:
        content = file.read()
    return content

def chunker(path , chunk=2000):
    all_chunks = []
    all_files = find_files(path)
    for file in all_files:
        text = read_file(file)
        for i in range(0, len(text), chunk):
            piece = text[i: i + chunk]
            all_chunks.append({
                "file_path":str(file),
                "text":piece,
                "start":i,
                "end": i+len(piece)
            })
    return all_chunks
def save_chunks(chunks, output_path="chunks.json"):
    with open(output_path, "w") as file:
        json.dump(chunks, file, indent=2)
chunks = chunker("data/raw/vllm-0.10.1")
print(f"Total chunks: {len(chunks)}")
save_chunks(chunks)