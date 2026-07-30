import json
from src.bm25 import BM25
from tqdm import tqdm


def load_json(path):
    with open(path, 'r') as file:
        content = json.load(file)
    return content

def search(chunks_path, index_path, query, k=5):
    chunks = load_json(chunks_path)
    retrieve = BM25()
    retrieve.load(index_path)
    results= retrieve.searcher(query, k=k)
    found = []
    for _, idx in results:
            found.append({
                "file_path":chunks[idx]['file_path'],
                "first_character_index":chunks[idx]['start'],
                "last_character_index":chunks[idx]['end']
            })
    return found

def search_dataset(chunks_path, index_path, dataset_path, output_path, k=5):
    all_results = []
    dataset = load_json(dataset_path)
    questions_dataset = dataset['rag_questions']
    questions = [question for question in questions_dataset]
    for query in tqdm(questions, desc="Searching"):
        results = search(chunks_path, index_path, query['question'], k)
        all_results.append({
            "question_id": query['question_id'],
            "question": query['question'],
            "retrieved_sources": results
        })
    output = {
        "search_results":all_results,
        "k":k
    }
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=2)
    print(f"Saved {len(all_results)} results to {output_path}")
