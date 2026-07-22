import json
from pathlib import Path




def read_file(path_file):
    with open(path_file, "r", errors="ignore", encoding="utf-8") as file:
        content = file.read()
    return content

def chunk_python_code(path, max_chunk_size=2000):
    chunks = []
    folder = Path(path)
    for file in folder.rglob("*.py"):
        text = read_file(file)
        current = ""
        for line in text.split("\n"):
            if line.startswith("def") or line.startswith("class"):
                if current:
                    if len(current) > max_chunk_size:
                        for i in range(0, len(current), max_chunk_size):
                            piece = current[i:i+max_chunk_size]
                            chunks.append({
                                "file_path": str(file),
                                "text": piece,
                                "start": text.find(piece),
                                "end": text.find(piece) + len(piece)
                            })
                    else:
                        chunks.append({
                            "file_path": str(file),
                            "text": current,
                            "start": text.find(current),
                            "end": text.find(current) + len(current)
                        })
                current = line
            else:
                current += '\n' + line
        if current:
            chunks.append({
                "file_path": str(file),
                "text": current,
                "start": text.find(current),
                "end": text.find(current) + len(current)
                })
    return chunks

def chunk_markedown(path, max_chunk_size=2000):
    chunks = []
    folder = Path(path)
    for file in folder.rglob("*.md"):
        current = ""
        text = read_file(file)
        for line in text.split("\n"):
            if line.startswith("#"):
                if current:
                    if len(current) > max_chunk_size:
                        for i in range(0, len(current), max_chunk_size):
                            piece = current[i:i+max_chunk_size]
                            chunks.append({
                                "file_path": str(file),
                                "text": piece,
                                "start": text.find(piece),
                                "end": len(piece) + text.find(piece)
                            })
                    else:
                        chunks.append({
                                "file_path": str(file),
                                "text": current,
                                "start": text.find(current),
                                "end": len(current) + text.find(current)
                            })
                current = line
            else:
                current += '\n' + line
        if current:
            chunks.append({
                "file_path": str(file),
                "text": current,
                "start": text.find(current),
                "end": len(current) + text.find(current)
            })
    return chunks

def chunker(path, max_chunk_size=2000):
    chunks = []
    chunks += chunk_python_code(path, max_chunk_size)
    chunks += chunk_markedown(path, max_chunk_size)
    return chunks

def save_chunks(chunks, output_path):
    with open(output_path, "w") as file:
        json.dump(chunks, file, indent=2)
