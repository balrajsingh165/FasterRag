"""Retrieval quality metrics.

Four measures, each answering a different question, which is why the harness reports all
four rather than picking one:

* **recall@k** — did the relevant material make it into the window at all? This is the
  ceiling on everything downstream: a chunk the retriever never returned cannot be reranked
  into place, cited, or answered from.
* **precision@k** — how much of the window is relevant? Recall cannot see a window padded
  with noise, and that noise is what crowds the context budget and the model's attention.
* **MRR** — how far down did the user have to look? Sensitive only to the *first* hit, so it
  measures whether the top result is usable.
* **nDCG@k** — how well ordered is the whole window, discounted by position? The measure
  that notices a retriever which finds everything but ranks it badly.

Relevance is binary here: a chunk either is ground truth or is not. Graded relevance would
need a judgement scale the golden-set schema does not carry, and inventing one would make
the numbers unreproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["dcg", "ndcg_at_k", "precision_at_k", "recall_at_k", "reciprocal_rank"]

_LOG_BASE = 2.0


def recall_at_k(hits: Sequence[bool], total_relevant: int, k: int | None = None) -> float:
    """Return the share of relevant items that appear in the first ``k`` results."""
    if total_relevant <= 0:
        return 0.0

    window = hits if k is None else hits[:k]
    return min(sum(window) / total_relevant, 1.0)


def precision_at_k(hits: Sequence[bool], k: int) -> float:
    """Return the share of the first ``k`` positions that hold a relevant result.

    The denominator is ``k``, not the number returned: an empty slot is not a relevant one.
    """
    if k <= 0:
        return 0.0
    return sum(hits[:k]) / k


def reciprocal_rank(hits: Sequence[bool]) -> float:
    """Return the reciprocal of the first relevant result's position, or zero if none is."""
    for position, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / position
    return 0.0


def dcg(hits: Sequence[bool], k: int | None = None) -> float:
    """Return the discounted cumulative gain of a ranking under binary relevance."""
    window = hits if k is None else hits[:k]
    return sum(
        1.0 / math.log(position + 1, _LOG_BASE)
        for position, hit in enumerate(window, start=1)
        if hit
    )


def ndcg_at_k(hits: Sequence[bool], total_relevant: int, k: int | None = None) -> float:
    """Return the normalized discounted cumulative gain."""
    if total_relevant <= 0:
        return 0.0

    window = len(hits) if k is None else k
    ideal_hits = [True] * min(total_relevant, window)
    ideal = dcg(ideal_hits)
    if ideal == 0:
        return 0.0

    return min(dcg(hits, k) / ideal, 1.0)
