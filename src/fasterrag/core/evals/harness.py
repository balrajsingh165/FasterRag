"""The retrieval eval harness.

Runs a golden set through a retriever and reports recall@k, MRR, and nDCG@k. It is what
turns "retrieval seems better" into a number, and it is the only thing that can populate the
benchmark ledger for retrieval quality (``docs/benchmarks.md``).

Two reporting rules keep the numbers honest:

* **Adversarial records are excluded from the averages**, not scored as zero. A query the
  corpus cannot answer has no relevant chunks, so recall against it is undefined; averaging a
  zero in would make a system that correctly refuses look worse than one that guesses.
* **Per-query results are kept**, not just the aggregate. An average that moved is only
  actionable once you can see which queries moved it.

Faithfulness is deliberately absent: it scores a generated answer against its context and
needs the LLM call site that arrives with the generation slice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from fasterrag.core.evals.golden import GoldenRecord
from fasterrag.core.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.observability.logging import get_logger

__all__ = ["EvalReport", "QueryScore", "Retriever", "evaluate"]

_logger = get_logger(__name__)


class Retriever(Protocol):
    """What the harness needs from anything it evaluates."""

    async def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Any = None,
    ) -> list[ScoredChunk]:
        """Return the chunks a query retrieves."""
        ...


@dataclass(frozen=True, slots=True)
class QueryScore:
    """How one golden query scored."""

    id: str
    retrieved: tuple[str, ...]
    hits: tuple[bool, ...]
    total_relevant: int
    recall: float
    reciprocal_rank: float
    ndcg: float
    adversarial: bool = False

    @property
    def found_nothing_relevant(self) -> bool:
        """Return whether the retriever missed this query entirely."""
        return not self.adversarial and not any(self.hits)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate retrieval quality over a golden set."""

    k: int
    scored: int
    adversarial: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: list[QueryScore] = field(default_factory=list)

    @property
    def misses(self) -> list[QueryScore]:
        """Return the queries where nothing relevant was retrieved at all."""
        return [score for score in self.per_query if score.found_nothing_relevant]

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report, the shape CI and the ledger consume."""
        return {
            "k": self.k,
            "scored": self.scored,
            "adversarial": self.adversarial,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "misses": [score.id for score in self.misses],
        }


def _mean(values: Sequence[float]) -> float:
    """Return the mean, or zero for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


async def evaluate(
    golden: Sequence[GoldenRecord],
    retriever: Retriever,
    *,
    k: int = 10,
    collection: str | None = None,
) -> EvalReport:
    """Score a retriever against a golden set.

    Args:
        golden: The ground-truth records.
        retriever: The retriever under test.
        k: Window size for recall@k and nDCG@k.
        collection: Collection to search, when the retriever supports one.

    Returns:
        The report, with per-query detail alongside the aggregates.
    """
    scores: list[QueryScore] = []

    for record in golden:
        results = await retriever.retrieve(
            record.query,
            collection=collection,
            top_k=k,
            filters=record.metadata or None,
        )
        hits = tuple(record.is_relevant(result.chunk_id, result.document_id) for result in results)
        total = len(set(record.relevant_chunk_ids) | set(record.relevant_document_ids))

        scores.append(
            QueryScore(
                id=record.id,
                retrieved=tuple(result.chunk_id for result in results),
                hits=hits,
                total_relevant=total,
                recall=recall_at_k(hits, total, k),
                reciprocal_rank=reciprocal_rank(hits),
                ndcg=ndcg_at_k(hits, total, k),
                adversarial=record.adversarial,
            )
        )

    measurable = [score for score in scores if not score.adversarial]
    report = EvalReport(
        k=k,
        scored=len(measurable),
        adversarial=len(scores) - len(measurable),
        recall_at_k=_mean([score.recall for score in measurable]),
        mrr=_mean([score.reciprocal_rank for score in measurable]),
        ndcg_at_k=_mean([score.ndcg for score in measurable]),
        per_query=scores,
    )

    _logger.info("evaluated a golden set", extra=report.as_dict())
    return report
