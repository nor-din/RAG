"""Hybrid retrieval utilities for ranking and combining multiple results.

This module contains methods to merge lexical and semantic rankings into a
single result list.
"""


def rrf(
    ibm25_result: list[int],
    semantic_result: list[int],
    k: int,
) -> list[int]:
    """Fuse BM25 and semantic rankings using Reciprocal Rank Fusion (RRF).

    Args:
        ibm25_result: Ranked document IDs from BM25 retrieval.
        semantic_result: Ranked document IDs from semantic retrieval.
        k: Number of top documents to return.

    Returns:
        A list of the top `k` document IDs ranked by their combined RRF scores.
    """
    scores: dict[int, float] = {}
    for score, idx in enumerate(ibm25_result):
        scores[idx] = scores.get(idx, 0.0) + 1.5 / (score + 60)
    for score, idx in enumerate(semantic_result):
        scores[idx] = scores.get(idx, 0.0) + 0.5 / (score + 60)
    final_score = sorted(scores, key=lambda idx: scores[idx], reverse=True)
    return final_score[:k]
