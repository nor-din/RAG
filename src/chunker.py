import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except (ModuleNotFoundError, ImportError):
    print("Module not installed")
    sys.exit(1)
from typing import Any, Dict, List


def read_file(path_file: Path) -> str:
    with open(path_file, "r", errors="ignore", encoding="utf-8") as file:
        content = file.read()
    return content


def chunk_python_code(
        path: str,
        max_chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    chunks = []
    folder = Path(path)
    py_files = folder.rglob("*.py")
    for file in tqdm(py_files, desc="Chunking py files"):
        text = read_file(file)
        current = ""
        for line in text.split("\n"):
            if line.startswith("def") or line.startswith("class"):
                current = current.strip()
                if current:
                    if len(current) > max_chunk_size:
                        for i in range(0, len(current), max_chunk_size):
                            piece = current[i: i + max_chunk_size]
                            chunks.append(
                                {
                                    "file_path": str(file),
                                    "text": piece,
                                    "start": text.find(piece),
                                    "end": text.find(piece) + len(piece),
                                }
                            )
                    else:
                        chunks.append(
                            {
                                "file_path": str(file),
                                "text": current,
                                "start": text.find(current),
                                "end": text.find(current) + len(current),
                            }
                        )
                current = line
            else:
                current += "\n" + line
        current = current.strip()
        if current:
            if len(current) > max_chunk_size:
                for i in range(0, len(current), max_chunk_size):
                    piece = current[i: i + max_chunk_size]
                    chunks.append(
                        {
                            "file_path": str(file),
                            "text": piece,
                            "start": text.find(piece),
                            "end": text.find(piece) + len(piece),
                        }
                    )
            else:
                chunks.append(
                    {
                        "file_path": str(file),
                        "text": current,
                        "start": text.find(current),
                        "end": text.find(current) + len(current),
                    }
                )
    return chunks


def chunk_markedown(
        path: str,
        max_chunk_size: int = 2000
) -> List[Dict[str, Any]]:
    chunks = []
    folder = Path(path)
    md_files = folder.rglob("*.md")
    for file in tqdm(md_files, desc="Chunking md files"):
        text = read_file(file)
        current = ""
        for line in text.split("\n"):
            if line.startswith("#"):
                current = current.strip()
                if current:
                    if len(current) > max_chunk_size:
                        for i in range(0, len(current), max_chunk_size):
                            piece = current[i: i + max_chunk_size]
                            chunks.append(
                                {
                                    "file_path": str(file),
                                    "text": piece,
                                    "start": text.find(piece),
                                    "end": text.find(piece) + len(piece),
                                }
                            )
                    else:
                        chunks.append(
                            {
                                "file_path": str(file),
                                "text": current,
                                "start": text.find(current),
                                "end": text.find(current) + len(current),
                            }
                        )
                current = line
            else:
                current += "\n" + line
        current = current.strip()
        if current:
            if len(current) > max_chunk_size:
                for i in range(0, len(current), max_chunk_size):
                    piece = current[i: i + max_chunk_size]
                    chunks.append(
                        {
                            "file_path": str(file),
                            "text": piece,
                            "start": text.find(piece),
                            "end": text.find(piece) + len(piece),
                        }
                    )
            else:
                chunks.append(
                    {
                        "file_path": str(file),
                        "text": current,
                        "start": text.find(current),
                        "end": text.find(current) + len(current),
                    }
                )
    return chunks


def chunker(path: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    chunks = []
    chunks += chunk_python_code(path, max_chunk_size)
    chunks += chunk_markedown(path, max_chunk_size)
    return chunks


def save_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(chunks, file, indent=2)
