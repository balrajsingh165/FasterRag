"""Running the eval harness against a live collection.

One place that turns "a golden set plus a collection" into a scored report, so the reindex
gate (D2), ``fasterrag benchmark --suite eval``, and CI all measure the same thing the same
way. Three callers scoring a collection three slightly different ways would make their
numbers incomparable, which is the failure a shared golden-set schema was meant to prevent
one layer up.

The dataset is a *directory*, not a file: a golden set alone is not enough to reproduce a
score. Knowing which corpus it was written against is what makes the number mean anything,
and keeping them together makes the pairing hard to get wrong.

Generation is scored here too, only when asked (``--faithfulness``): every answerable golden
query is answered and graded with P3, and the mean faithfulness and citation rate are judged
against the dataset's ``generation-baseline.json`` in the retrieval gate's three-valued
vocabulary (TASK-0481).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, GoldenRecord, evaluate, load_golden_set
from fasterrag.core.faithfulness import P3_TEMPLATE_VERSION
from fasterrag.core.generation import P1_TEMPLATE_VERSION
from fasterrag.core.identity import document_id
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger
from fasterrag.services.composition import build_generation
from fasterrag.services.querying import RetrievalService
from fasterrag.services.regression import (
    Baseline,
    GateResult,
    Verdict,
    check_regression,
    comparability_fingerprint,
)

__all__ = [
    "BASELINE_NAME",
    "CORPUS_DIR",
    "GENERATION_BASELINE_NAME",
    "GENERATION_TOLERANCE",
    "GOLDEN_NAME",
    "EvalDataset",
    "GenerationBaseline",
    "GenerationGate",
    "check_generation",
    "generation_fingerprint",
    "load_dataset",
    "load_generation_baseline",
    "run_eval",
    "score_collection",
    "score_faithfulness",
    "write_generation_baseline",
]

GOLDEN_NAME: Final = "golden.jsonl"
BASELINE_NAME: Final = "baseline.json"
CORPUS_DIR: Final = "corpus"
GENERATION_BASELINE_NAME: Final = "generation-baseline.json"
GENERATION_TOLERANCE: Final = 0.05

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
        """Return the records with filename ground truth rewritten to document ids."""
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
    """Load a dataset directory: golden set, corpus, and any committed baseline."""
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
    metrics.RETRIEVAL_QUALITY.set(report.precision_at_k, metric="precision_at_k")
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
    """Score a collection and judge it against the dataset's committed baseline."""
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
        "precision_at_k": round(report.precision_at_k, 4),
        "mrr": round(report.mrr, 4),
        "ndcg_at_k": round(report.ndcg_at_k, 4),
    }


def _grading(settings: Settings) -> Settings:
    """Return ``settings`` with P3 grading on, so every answer is scored, withheld or not."""
    generation = settings.generation.model_copy(
        update={"grounded_or_refuse": True, "citations": True}
    )
    return settings.model_copy(update={"generation": generation})


async def score_faithfulness(
    dataset: EvalDataset,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    *,
    collection: str,
) -> dict[str, Any]:
    """Answer every answerable golden query and grade each answer with P3.

    Grading is on for the run whatever ``generation.grounded_or_refuse`` says, and a withheld
    answer's score counts: it was graded, and leaving it out would hide the worst answers.
    """
    service = build_generation(_grading(settings), adapter, router)
    answerable = [record for record in dataset.resolved() if not record.adversarial]

    scores: list[float] = []
    refused = 0
    ungraded = 0
    cited = 0
    try:
        for record in answerable:
            answer = await service.answer(record.query, collection=collection)
            if answer.insufficient_evidence:
                refused += 1
            if answer.citations:
                cited += 1
            if answer.faithfulness is None:
                ungraded += 1
            else:
                scores.append(answer.faithfulness)
                metrics.FAITHFULNESS.observe(answer.faithfulness)
    finally:
        await service.close()

    mean = sum(scores) / len(scores) if scores else 0.0
    _logger.info(
        "scored faithfulness over a golden set",
        extra={"graded": len(scores), "refused": refused, "ungraded": ungraded, "mean": mean},
    )
    return {
        "mean_faithfulness": round(mean, 4),
        "citation_rate": round(cited / len(answerable), 4) if answerable else 0.0,
        "graded": len(scores),
        "refused": refused,
        "ungraded": ungraded,
        "cited": cited,
        "answerable": len(answerable),
    }


def generation_fingerprint(settings: Settings) -> dict[str, str]:
    """Return what a generation score is only comparable across: retrieval, and the model.

    Prompt template versions are recorded but not compared: a prompt change is what the
    generation gate exists to judge, so it must be measured against the baseline, not excused.
    """
    fingerprint = comparability_fingerprint(settings)
    fingerprint["llm.provider"] = json.dumps(settings.llm.provider)
    fingerprint["llm.model"] = json.dumps(settings.llm.model)
    fingerprint["generation.context_budget_tokens"] = json.dumps(
        settings.generation.context_budget_tokens
    )
    return fingerprint


@dataclass(frozen=True, slots=True)
class GenerationBaseline:
    """A recorded generation score, the setup that produced it, and its tolerance."""

    recorded_at: str
    mean_faithfulness: float
    citation_rate: float
    graded: int
    answerable: int
    tolerance: float
    fingerprint: dict[str, str]
    p1_template_version: str
    p3_template_version: str

    def as_dict(self) -> dict[str, Any]:
        """Return the committed form."""
        return {
            "recorded_at": self.recorded_at,
            "mean_faithfulness": self.mean_faithfulness,
            "citation_rate": self.citation_rate,
            "graded": self.graded,
            "answerable": self.answerable,
            "tolerance": self.tolerance,
            "fingerprint": dict(self.fingerprint),
            "p1_template_version": self.p1_template_version,
            "p3_template_version": self.p3_template_version,
        }


def load_generation_baseline(root: Path) -> GenerationBaseline | None:
    """Read a dataset's committed generation baseline, or None when it has none."""
    source = root / GENERATION_BASELINE_NAME
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("a generation baseline is a JSON object")
        baseline = GenerationBaseline(**payload)
        if not all(isinstance(value, str) for value in baseline.fingerprint.values()):
            raise ValueError("'fingerprint' must map field names to recorded values")
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise FasterRagError(
            f"{source} exists but is not a valid generation baseline: {exc}. Re-record it "
            "with --record-generation-baseline rather than editing it by hand",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        ) from exc
    return baseline


def write_generation_baseline(
    root: Path, scores: dict[str, Any], settings: Settings
) -> GenerationBaseline:
    """Record a run as the dataset's generation baseline; an explicit act, never automatic."""
    baseline = GenerationBaseline(
        recorded_at=datetime.now(tz=UTC).isoformat(),
        mean_faithfulness=float(scores["mean_faithfulness"]),
        citation_rate=float(scores["citation_rate"]),
        graded=int(scores["graded"]),
        answerable=int(scores["answerable"]),
        tolerance=GENERATION_TOLERANCE,
        fingerprint=generation_fingerprint(settings),
        p1_template_version=P1_TEMPLATE_VERSION,
        p3_template_version=P3_TEMPLATE_VERSION,
    )
    target = root / GENERATION_BASELINE_NAME
    try:
        target.write_text(
            json.dumps(baseline.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise FasterRagError(
            f"the generation baseline could not be written to {target}: {exc}",
            code=ErrorCode.INTERNAL,
            retryable=False,
        ) from exc
    return baseline


@dataclass(frozen=True, slots=True)
class GenerationGate:
    """The generation gate's verdict, in the retrieval gate's three-valued vocabulary."""

    failures: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """Return the outcome; a run compared against nothing is never within tolerance."""
        if self.failures:
            return Verdict.REGRESSED
        if self.obstacles:
            return Verdict.NOT_COMPARABLE
        return Verdict.WITHIN_TOLERANCE

    @property
    def passed(self) -> bool:
        """Return whether the run may proceed."""
        return self.verdict is Verdict.WITHIN_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable result CI consumes."""
        return {
            "passed": self.passed,
            "verdict": self.verdict.value,
            "failures": self.failures,
            "obstacles": self.obstacles,
        }

    def summary(self) -> str:
        """Return a one-line verdict for a terminal."""
        if self.verdict is Verdict.REGRESSED:
            return "generation gate REGRESSED: " + "; ".join(self.failures)
        if self.verdict is Verdict.NOT_COMPARABLE:
            return "generation gate NOT COMPARABLE, no regression ruled out: " + "; ".join(
                self.obstacles
            )
        return "generation gate passed"


def check_generation(
    scores: dict[str, Any], baseline: GenerationBaseline | None, settings: Settings
) -> GenerationGate:
    """Judge a faithfulness run against the committed generation baseline."""
    if baseline is None:
        return GenerationGate(
            obstacles=[
                f"no {GENERATION_BASELINE_NAME} is committed for this dataset, so a regression "
                "cannot be detected; record one with --record-generation-baseline"
            ]
        )

    current = generation_fingerprint(settings)
    moved = sorted(
        name
        for name in set(current) | set(baseline.fingerprint)
        if current.get(name) != baseline.fingerprint.get(name)
    )
    obstacles = [
        f"{name} was {baseline.fingerprint.get(name)} when the baseline was recorded and is "
        f"{current.get(name)} now"
        for name in moved
    ]
    for name in ("answerable", "graded"):
        if scores.get(name) != getattr(baseline, name):
            obstacles.append(
                f"{scores.get(name)} answers were {name} against {getattr(baseline, name)} in "
                "the baseline, so the two means are not over the same answers"
            )
    if obstacles:
        return GenerationGate(obstacles=obstacles)

    failures = [
        f"{name} fell from {recorded:.4f} to {float(scores[name]):.4f}, beyond the recorded "
        f"tolerance of {baseline.tolerance:.4f}"
        for name, recorded in (
            ("mean_faithfulness", baseline.mean_faithfulness),
            ("citation_rate", baseline.citation_rate),
        )
        if recorded - float(scores[name]) > baseline.tolerance
    ]
    return GenerationGate(failures=failures)
