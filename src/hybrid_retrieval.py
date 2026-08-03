def rrf(bm25_result, semantic_result, k):
    scores = {}
    for score, idx in enumerate(bm25_result):
        scores[idx] = scores.get(idx, 0) + 1/(score + 60)
    for score, idx in enumerate(semantic_result):
        scores[idx] = scores.get(idx, 0) + 1/(score + 60)
    final_score = sorted(scores, key=scores.get , reverse=True)
    return final_score[:k]
