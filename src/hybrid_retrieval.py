def rrf(
    bm25_result: list[int],
    semantic_result: list[int],
    k: int,
) -> list[int]:
    scores: dict[int, float] = {}
    for score, idx in enumerate(bm25_result):
        scores[idx] = scores.get(idx, 0.0) + 1 / (score + 60)
    for score, idx in enumerate(semantic_result):
        scores[idx] = scores.get(idx, 0.0) + 1 / (score + 60)
    final_score = sorted(scores, key=lambda idx: scores[idx], reverse=True)
    return final_score[:k]
