"""Running the eval harness against a live collection.

One place that turns "a golden set plus a collection" into a scored report, so the reindex
gate (D2), ``fasterrag benchmark --suite eval``, and CI all measure the same thing the same
way. Three callers scoring a collection three slightly different ways would make their
numbers incomparable, which is the failure a shared golden-set schema was meant to prevent
one layer up.

The dataset is a *directory*, not a file: a golden set alone is not enough to reproduce a
score. Knowing which corpus it was written against is what makes the number mean anything,
and keeping them together makes the pairing hard to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, GoldenRecord, evaluate, load_golden_set
from fasterrag.core.identity import document_id
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger
from fasterrag.services.querying import RetrievalService
from fasterrag.services.regression import Baseline, GateResult, check_regression

__all__ = [
    "BASELINE_NAME",
    "CORPUS_DIR",
    "GOLDEN_NAME",
    "EvalDataset",
    "load_dataset",
    "run_eval",
    "score_collection",
]

GOLDEN_NAME: Final = "golden.jsonl"
BASELINE_NAME: Final = "baseline.json"
CORPUS_DIR: Final = "corpus"

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A golden set, the corpus it was written against, and its recorded baseline."""

    root: Path
    records: list[GoldenRecord]
    k: int
    baseline: Baseline | None = None

    @property
    def corpus(self) -> Path:
        """Return the corpus directory."""
        return self.root / CORPUS_DIR

    def resolved(self) -> list[GoldenRecord]:
        r"""Return the records with filename ground truth rewritten to document ids.

        # CRITICAL: the corpus path is used exactly as given, never resolved. A document id
        # hashes the source string it was ingested under, and on Windows `d:\\...` and
        # `D:\\...` hash differently even though they name one file — building the id from a
        # differently-normalised path yields zero matches and a score of 0.0 that reads as a
        # retrieval failure rather than a path mismatch (TASK-0141).
        """
        rewritten: list[GoldenRecord] = []
        for record in self.records:
            names = record.relevant_document_ids
            identifiers = tuple(
                name if name.startswith("d_") else document_id(str(self.corpus / name))
                for name in names
            )
            rewritten.append(
                GoldenRecord(
                    id=record.id,
                    query=record.query,
                    source=record.source,
                    created_at=record.created_at,
                    relevant_chunk_ids=record.relevant_chunk_ids,
                    relevant_document_ids=identifiers,
                    answer_reference=record.answer_reference,
                    metadata=record.metadata,
                )
            )
        return rewritten


def load_dataset(root: Path, *, default_k: int = 5) -> EvalDataset:
    """Load a dataset directory: golden set, corpus, and any committed baseline.

    Raises:
        FasterRagError: With ``NOT_FOUND`` when the directory holds no golden set. An eval
            run against a dataset that does not exist should say so, not score zero.
    """
    golden_path = root / GOLDEN_NAME
    if not golden_path.is_file():
        raise FasterRagError(
            f"{root} holds no {GOLDEN_NAME}; an eval dataset is a directory containing a "
            f"golden set and the {CORPUS_DIR}/ it was written against",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    baseline_path = root / BASELINE_NAME
    baseline = None
    k = default_k
    if baseline_path.is_file():
        from fasterrag.services.regression import load_baseline

        baseline = load_baseline(baseline_path)
        if baseline is not None:
            k = baseline.k

    return EvalDataset(root=root, records=load_golden_set(golden_path), k=k, baseline=baseline)


async def score_collection(
    dataset: EvalDataset,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    *,
    collection: str,
) -> EvalReport:
    """Score one collection against a dataset's golden set."""
    retrieval = RetrievalService(settings, adapter, router)
    report = await evaluate(dataset.resolved(), retrieval, k=dataset.k, collection=collection)

    metrics.RETRIEVAL_QUALITY.set(report.recall_at_k, metric="recall_at_k")
    metrics.RETRIEVAL_QUALITY.set(report.mrr, metric="mrr")
    metrics.RETRIEVAL_QUALITY.set(report.ndcg_at_k, metric="ndcg_at_k")

    _logger.info(
        "scored a collection against a golden set",
        extra={
            "collection": collection,
            "dataset": str(dataset.root),
            "k": report.k,
            "recall_at_k": round(report.recall_at_k, 4),
            "mrr": round(report.mrr, 4),
        },
    )
    return report


async def run_eval(
    dataset_root: Path,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    *,
    collection: str,
) -> tuple[EvalReport, GateResult]:
    """Score a collection and judge it against the dataset's committed baseline.

    Returns:
        The report and the gate's verdict. Both are returned because they answer different
        questions: the report says how the collection scored, and the verdict says whether
        that is acceptable — and a caller that only saw the verdict could not say by how
        much a run missed.
    """
    dataset = load_dataset(dataset_root)
    report = await score_collection(dataset, settings, adapter, router, collection=collection)
    return report, check_regression(report, dataset.baseline, settings)


def summarize(report: EvalReport) -> dict[str, Any]:
    """Return the compact form the CLI and the ledger print."""
    return {
        "k": report.k,
        "scored": report.scored,
        "adversarial": report.adversarial,
        "recall_at_k": round(report.recall_at_k, 4),
        "mrr": round(report.mrr, 4),
        "ndcg_at_k": round(report.ndcg_at_k, 4),
    }
