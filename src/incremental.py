import os
import sys
import json
from pathlib import Path
from src.chunker import chunk_python_code, chunk_markedown
from src.indexer import indexer

def save_times(repo_path, output_path):
    folder = Path(repo_path)
    save_files = {}
    for file in folder.rglob("*"):
        if file.suffix == ".py" or file.suffix == ".md":
            save_files[str(file)] = os.path.getmtime(file)
    with open(output_path, "w") as file:
        json.dump(save_files, file, indent=2)
    
def get_changed(repo_path, times_path):
    with open(times_path, "r") as file:
        times_files = json.load(file)
    folder = Path(repo_path)
    save_files = []
    for file in folder.rglob("*"):
        if file.suffix == ".py" or file.suffix == ".md":
            saved_time = times_files.get(str(file), 0)
            if saved_time != os.path.getmtime(file):
                save_files.append(str(file))
    return save_files

def incremental(repo_path, times_path, chunk_path, index_path ,embeddings_path,max_chunk_size: int = 2000):
    with open(chunk_path, "r") as file:
        old_chunks = json.load(file)
    saved_files = get_changed(repo_path, times_path)
    clean_chunks = [
        chunk for chunk in old_chunks
        if chunk['file_path'] in saved_files
    ]
    if not saved_files:
        print("No files changed. Index is up to date!")
        sys.exit(1)
    for file in saved_files:
        if file.endswith(".py"):
            chunk = chunk_python_code(file, max_chunk_size)
            clean_chunks += chunk
        elif file.endswith(".md"):
            chunk = chunk_markedown(file, max_chunk_size)
            clean_chunks += chunk
    with open(chunk_path, "w") as f:
        json.dump(clean_chunks, f, indent=2)
    indexer(chunk_path, index_path, embeddings_path)
    save_times(repo_path, times_path)