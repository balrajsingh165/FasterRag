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
rescaling anyone's scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = ["DEFAULT_RRF_K", "FusedResult", "Ranking", "rrf_fuse"]

DEFAULT_RRF_K: Final = 60.0

FIRST_RANK: Final = 1


@dataclass(frozen=True, slots=True)
class Ranking:
    """One leg's ordered result ids, best first."""

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
        Fused results ordered by descending score. A document ranked by several legs
        outranks one ranked highly by a single leg, which is the point of fusing.

    Raises:
        ValueError: If ``k`` is not positive, which would invert or explode the weighting.
    """
    if k <= 0:
        raise ValueError(f"the RRF constant must be positive, got {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for index, entry in enumerate(rankings):
        leg = entry if isinstance(entry, Ranking) else Ranking(name=f"leg_{index}", ids=entry)
        for position, identifier in enumerate(leg.ids, start=FIRST_RANK):
            scores[identifier] = scores.get(identifier, 0.0) + leg.weight / (k + position)
            ranks.setdefault(identifier, {})[leg.name] = position

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        FusedResult(id=identifier, score=score, ranks=ranks[identifier])
        for identifier, score in ordered
    ]
