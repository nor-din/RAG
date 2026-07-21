import json
import bm25s

def load_chunk(path):
    with open(path, "r") as file:
        chunks = json.load(file)
    return chunks
def indexer():
    chunks = load_chunk("chunks.json")
    corpus = [chunk['text'] for chunk in chunks]
    retriever = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(corpus)
    retriever.index(corpus_tokens)
    retriever.save("bm25_index")
indexer()