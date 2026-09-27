"""Reciprocal Rank Fusion: merge the ranked results of several queries.

Each passage scores the sum of 1 / (k + rank) over the lists it appears in.
Only ranks are used, never distances, which are not comparable from one
query to another. A passage found by several queries rises above one that a
single query ranked first: agreement between readings of the question is
the signal.
"""

from collections.abc import Sequence

from prestie.knowledge.store import SearchHit

# Standard RRF constant: dampens the gap between the first ranks, so agreement
# between lists matters more than a single list's top position.
RRF_K = 60


def reciprocal_rank_fusion(
    hit_lists: Sequence[Sequence[SearchHit]], n_results: int, k: int = RRF_K
) -> list[SearchHit]:
    """Merge ranked lists: each passage scores the sum of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (k + rank)
            first_seen.setdefault(hit.id, hit)
    # sorted() is stable: ties keep their first-appearance order.
    ranked = sorted(first_seen, key=lambda chunk_id: -scores[chunk_id])
    return [first_seen[chunk_id] for chunk_id in ranked[:n_results]]
