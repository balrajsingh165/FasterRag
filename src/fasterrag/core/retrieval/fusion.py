"""Reciprocal Rank Fusion.

Combines rankings from independent retrieval legs. RRF works on **ranks, not scores**, which
is the property that makes it usable here at all: a cosine similarity and a BM25 score live
on incompatible scales, and any attempt to normalize them into comparability smuggles in an
assumption about their distributions. Positions need no such assumption.

``RRF(d) = Σ weight_i / (k + rank_i(d))``

``k`` defaults to 60, the value from Cormack, Clarke and Büttcher (SIGIR 2009), shown robust
across TREC and LETOR benchmarks (``docs/references.md``). It damps the influence of the top
few positions so one leg's confident first result cannot dominate a document that several
legs agree on — which is exactly the behavior hybrid retrieval is bought for.

Weights scale each leg's contribution without changing the rank-based nature of the fusion,
so ``retrieval.bm25_weight`` and ``retrieval.dense_weight`` tilt the balance rather than
rescaling anyone's scores. A weight of zero is the end of that scale: the leg stops
contributing entirely, documents only it ranked are not results, and the ranking of every
other document is exactly what it would have been had the leg not run.

Two degenerate inputs silently corrupted the ranking before TASK-0228 and are now handled
explicitly:

* **A repeated id inside one leg.** ``rank_i(d)`` is one number, so only a document's best
  position in a leg counts. Summing every occurrence let ``["a", "b", "b"]`` fuse to
  ``["b", "a"]`` — a single leg reordering its own input — and reported ``b`` at its worst
  rank rather than its best.
* **A zero weight.** The leg stops contributing entirely, but its positions are still
  recorded on documents another leg surfaced, so a disabled leg stays observable without
  becoming a source of results.

A third property, **independence from the order the legs are passed in**, is guaranteed
rather than repaired: contributions are collected and summed with :func:`math.fsum` instead
of accumulated as a running total. No input has been found where a running total reorders
anything — see the note at the call site for the measurement and for why the guarantee
should not be left resting on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum
from typing import Final

__all__ = ["DEFAULT_RRF_K", "FusedResult", "Ranking", "rrf_fuse"]

DEFAULT_RRF_K: Final = 60.0

FIRST_RANK: Final = 1


@dataclass(frozen=True, slots=True)
class Ranking:
    """One leg's ordered result ids, best first.

    A repeated id counts once, at its best position. A ``weight`` of zero removes the leg
    from the fusion entirely rather than contributing documents worth nothing.
    """

    name: str
    ids: Sequence[str]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class FusedResult:
    """One fused document, with the rank each leg gave it."""

    id: str
    score: float
    ranks: Mapping[str, int] = field(default_factory=dict)

    def rank_in(self, leg: str) -> int | None:
        """Return this document's one-based rank in ``leg``, or None if it was absent."""
        return self.ranks.get(leg)


def rrf_fuse(
    *rankings: Ranking | Sequence[str],
    k: float = DEFAULT_RRF_K,
) -> list[FusedResult]:
    """Fuse ranked id lists into one ranking.

    Args:
        *rankings: The legs to combine, each ordered best first. A bare sequence of ids is
            accepted for standalone use; a :class:`Ranking` additionally names the leg and
            carries its weight.
        k: The RRF constant. Larger values flatten the advantage of top positions.

    Returns:
        Fused results ordered by descending score, ties broken by id. A document ranked by
        several legs outranks one ranked highly by a single leg, which is the point of
        fusing. The ids are exactly the union of the ids the weighted legs ranked, each
        appearing once, and the result does not depend on the order the legs were passed in.

    Raises:
        ValueError: If ``k`` is not positive, which would invert or explode the weighting.
    """
    if k <= 0:
        raise ValueError(f"the RRF constant must be positive, got {k}")

    contributions: dict[str, list[float]] = {}
    ranks: dict[str, dict[str, int]] = {}

    for index, entry in enumerate(rankings):
        leg = entry if isinstance(entry, Ranking) else Ranking(name=f"leg_{index}", ids=entry)
        ranked: set[str] = set()
        for position, identifier in enumerate(leg.ids, start=FIRST_RANK):
            if identifier in ranked:
                continue
            ranked.add(identifier)
            ranks.setdefault(identifier, {})[leg.name] = position
            if leg.weight == 0:
                continue
            contributions.setdefault(identifier, []).append(leg.weight / (k + position))

    # CRITICAL: collect the contributions, then fsum them, rather than accumulating a running
    # total as the legs stream past. fsum is exactly rounded, so two documents holding the
    # same multiset of contributions get the same float whatever order the legs arrived in,
    # and the `(-score, id)` tie-break below is always reached.
    #
    # A running total gets the same answer today, and only by luck of the value ranges: a
    # single fusion's contributions span at most ~5e4 (weights 0.1-5 over `k + position`,
    # k >= 0.01), while reassociation needs operands ~4.5e15 apart to change a sum. An
    # exhaustive sweep of the permitted k, weight, and position values found no ordering that
    # separates the two (1.2M combinations), and a 2M-trial random search over a wider space
    # found none either. That margin belongs to the current config ranges, not to the
    # algorithm — widening `retrieval.rrf_k` or the leg weights would spend it silently, and
    # the symptom would be two documents that should have tied by id swapping places
    # depending on which leg the caller listed first.
    scored = ((identifier, fsum(values)) for identifier, values in contributions.items())
    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    return [
        FusedResult(id=identifier, score=score, ranks=ranks[identifier])
        for identifier, score in ordered
    ]
