import json
from pathlib import Path




def read_file(path_file):
    with open(path_file, "r", errors="ignore", encoding="utf-8") as file:
        content = file.read()
    return content

def chunk_python_code(path):
    chunks = []
    folder = Path(path)
    for file in folder.rglob("*"):
        text = read_file(file)
        if file.suffix == ".py":
            
    
def save_chunks(chunks, output_path):
    with open(output_path, "w") as file:
        json.dump(chunks, file, indent=2)
