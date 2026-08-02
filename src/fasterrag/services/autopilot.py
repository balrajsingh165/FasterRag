"""Eval-driven auto-tuning that only ever suggests (D6).

Every framework hands you a dozen retrieval knobs and a shrug; tuning them is folklore
carried between projects by whoever tuned last. Autopilot replaces the folklore with a
measurement: it evaluates candidate configurations against a golden set built from *your*
corpus and reports what each one actually scored.

**It never writes ``config.yaml``.** The output is a suggested diff and the measured deltas
behind it; a human reads them and decides. That is not timidity — an auto-tuner that edits
production configuration turns a bad measurement into an outage, and the measurement is only
as good as the golden set behind it.

**Only query-time parameters are searched.** The fusion weights, ``rrf_k``, and reranking all
change how an existing index is queried, so a candidate costs one evaluation. ``top_k`` is
excluded even though it is query-time: measuring recall@k means retrieving exactly k, so the
harness fixes the depth and a top_k candidate is a no-op by construction.
Chunk size and overlap change how the index is *built*, so every candidate would cost a full
re-chunk and re-embed of the corpus — hours on a real one. That search is a different
operation with a different cost profile and is deliberately not folded in here (TASK-0145).

**The search is time-boxed and reports what it did not reach.** A tuner that silently
evaluated a third of its grid and announced a winner would be presenting the best of an
arbitrary subset as though it were the best available.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import ValidationError

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.evaluation import EvalDataset, score_collection

__all__ = [
    "SUGGESTION_FILE",
    "Candidate",
    "Suggestion",
    "TrialResult",
    "candidate_grid",
    "render_suggestion",
    "tune",
]

SUGGESTION_FILE: Final = "autopilot-suggestion.yaml"

# CRITICAL: query-time only. Every key here changes how an existing index is searched, so a
# trial costs one evaluation. Adding an index-time key (chunk_size, overlap, the embedding
# model) would make each trial a full corpus rebuild, which is a different operation with a
# different budget and must not be smuggled into this grid.
_SEARCHED_KEYS: Final[tuple[str, ...]] = (
    "retrieval.bm25_weight",
    "retrieval.dense_weight",
    "retrieval.rrf_k",
    "retrieval.rerank",
)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One configuration under test, as an override of the current settings."""

    overrides: dict[str, Any]

    @property
    def label(self) -> str:
        """Return a short, stable description of what this candidate changes."""
        if not self.overrides:
            return "baseline (current configuration)"
        return ", ".join(f"{key}={value}" for key, value in sorted(self.overrides.items()))

    def apply(self, settings: Settings) -> Settings:
        """Return ``settings`` with this candidate's overrides applied.

        A deep copy is validated rather than mutated: a candidate that quietly altered the
        live settings object would leak into every later trial and into the running process.
        """
        payload = settings.model_dump(mode="json")
        for key, value in self.overrides.items():
            section, _, field_name = key.partition(".")
            payload.setdefault(section, {})[field_name] = value
        return Settings.model_validate(payload)


@dataclass(frozen=True, slots=True)
class TrialResult:
    """What one candidate scored."""

    candidate: Candidate
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized trial."""
        return {
            "overrides": self.candidate.overrides,
            "label": self.candidate.label,
            "k": self.k,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class Suggestion:
    """What Autopilot found, and what it did not get to."""

    baseline: TrialResult
    best: TrialResult
    trials: list[TrialResult] = field(default_factory=list)
    evaluated: int = 0
    skipped: int = 0
    seconds: float = 0.0
    created_at: str = ""

    @property
    def improves(self) -> bool:
        """Return whether the winner actually beats the current configuration.

        Ties are not improvements. Suggesting a change that measured identically would ask a
        human to review a diff for no reason, and would train them to stop reading them.
        """
        return self.best.ndcg_at_k > self.baseline.ndcg_at_k

    @property
    def deltas(self) -> dict[str, float]:
        """Return the winner's measured change against the current configuration."""
        return {
            "recall_at_k": round(self.best.recall_at_k - self.baseline.recall_at_k, 4),
            "mrr": round(self.best.mrr - self.baseline.mrr, 4),
            "ndcg_at_k": round(self.best.ndcg_at_k - self.baseline.ndcg_at_k, 4),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized suggestion."""
        return {
            "created_at": self.created_at,
            "improves": self.improves,
            "suggested_overrides": self.best.candidate.overrides if self.improves else {},
            "deltas": self.deltas,
            "baseline": self.baseline.as_dict(),
            "best": self.best.as_dict(),
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 2),
            "trials": [trial.as_dict() for trial in self.trials],
            "applied": False,
            "note": (
                "Autopilot never writes config.yaml. Review these measurements and apply the "
                "overrides yourself if you agree with them."
            ),
        }


def candidate_grid(settings: Settings) -> list[Candidate]:
    """Return the query-time candidates to try, cheapest and most promising first.

    Ordered rather than arbitrary, because the search is time-boxed: whatever the budget
    cuts off should be the least promising end of the list, not a random tail.
    """
    current = settings.retrieval
    candidates: list[Candidate] = [Candidate({})]

    # CRITICAL: `retrieval.top_k` is deliberately absent. Measuring recall@k requires
    # retrieving exactly k, so the harness fixes the retrieval depth to the dataset's k and a
    # top_k candidate cannot change any score. Searching it would burn budget on trials whose
    # results are identical by construction, and report the tie as a genuine finding.

    # CRITICAL: within the schema's 0.0-1.0 bounds. A candidate outside them is rejected by
    # validation, and a grid that generated invalid entries would spend its budget on trials
    # that can never run.
    for dense, sparse in ((1.0, 0.5), (0.5, 1.0), (1.0, 0.25), (0.25, 1.0)):
        if (dense, sparse) != (current.dense_weight, current.bm25_weight):
            candidates.append(
                Candidate({"retrieval.dense_weight": dense, "retrieval.bm25_weight": sparse})
            )

    for rrf_k in (10.0, 30.0, 60.0):
        if rrf_k != current.rrf_k:
            candidates.append(Candidate({"retrieval.rrf_k": rrf_k}))

    if current.rerank:
        candidates.append(Candidate({"retrieval.rerank": False}))

    return candidates


async def tune(
    dataset: EvalDataset,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    *,
    collection: str,
    budget_seconds: float = 300.0,
    candidates: Sequence[Candidate] | None = None,
) -> Suggestion:
    """Search query-time configurations against a golden set and suggest the best.

    Args:
        dataset: The golden set and corpus to measure against.
        settings: The current configuration, which is also the first trial.
        adapter: The vector database holding ``collection``.
        router: The embedding router; shared across trials so the model loads once.
        collection: The collection every trial searches. Never modified.
        budget_seconds: Wall-clock ceiling. The search stops cleanly when exceeded and
            reports how many candidates it did not reach.
        candidates: Overrides the default grid, for tests and for a narrowed search.

    Returns:
        The suggestion, including every trial that ran.

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` when ``autopilot.enabled`` is false.
            Tuning is a risky feature behind a flag defaulting to false, and running it
            because someone typed the command would defeat the flag.
    """
    if not settings.autopilot.enabled:
        raise FasterRagError(
            "autopilot.enabled is false; set it to true to allow eval-driven tuning",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    grid = list(candidates if candidates is not None else candidate_grid(settings))
    started = time.perf_counter()
    trials: list[TrialResult] = []
    invalid = 0

    for index, candidate in enumerate(grid):
        elapsed = time.perf_counter() - started
        if index > 0 and elapsed >= budget_seconds:
            _logger.info(
                "autopilot stopped at its budget",
                extra={"evaluated": len(trials), "skipped": len(grid) - len(trials)},
            )
            break

        try:
            candidate_settings = candidate.apply(settings)
        except ValidationError:
            # A candidate outside the schema's bounds is skipped, never fatal: one bad grid
            # entry must not discard the trials that already ran.
            _logger.warning(
                "autopilot skipped an invalid candidate",
                extra={"overrides": candidate.overrides},
            )
            invalid += 1
            continue

        trial_started = time.perf_counter()
        report = await score_collection(
            dataset,
            candidate_settings,
            adapter,
            router,
            collection=collection,
        )
        trials.append(
            TrialResult(
                candidate=candidate,
                recall_at_k=report.recall_at_k,
                mrr=report.mrr,
                ndcg_at_k=report.ndcg_at_k,
                k=report.k,
                seconds=time.perf_counter() - trial_started,
            )
        )

    if not trials:
        raise FasterRagError(
            "autopilot evaluated nothing; the budget was too small for even one trial",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    baseline = trials[0]
    # CRITICAL: ties resolve to the baseline. `max` returns the first maximum, and the
    # baseline is first, so a candidate that merely matches the current configuration never
    # displaces it — a suggestion has to earn the human's attention.
    best = max(trials, key=lambda trial: (trial.ndcg_at_k, trial.recall_at_k, trial.mrr))

    suggestion = Suggestion(
        baseline=baseline,
        best=best,
        trials=trials,
        evaluated=len(trials),
        skipped=len(grid) - len(trials) - invalid,
        seconds=time.perf_counter() - started,
        created_at=datetime.now(tz=UTC).isoformat(),
    )

    _logger.info(
        "autopilot finished",
        extra={
            "evaluated": suggestion.evaluated,
            "skipped": suggestion.skipped,
            "improves": suggestion.improves,
            "deltas": suggestion.deltas,
        },
    )
    return suggestion


def render_suggestion(suggestion: Suggestion) -> str:
    """Render the suggestion as the YAML fragment an operator would paste.

    A fragment rather than a whole file: emitting a complete ``config.yaml`` would invite
    someone to overwrite theirs with it, losing every key Autopilot never looked at.
    """
    lines = [
        "# Autopilot suggestion — NOT APPLIED.",
        "# Autopilot never writes config.yaml. Review the measurements below and paste the",
        "# overrides into your own config if you agree with them.",
        f"# generated: {suggestion.created_at}",
        f"# evaluated: {suggestion.evaluated} candidates"
        + (f", {suggestion.skipped} skipped at the budget" if suggestion.skipped else ""),
        "#",
        f"# baseline : ndcg@{suggestion.baseline.k}={suggestion.baseline.ndcg_at_k:.4f} "
        f"recall={suggestion.baseline.recall_at_k:.4f} mrr={suggestion.baseline.mrr:.4f}",
    ]

    if not suggestion.improves:
        lines.extend(
            [
                "#",
                "# No candidate beat the current configuration. Nothing to change.",
                "{}",
            ]
        )
        return "\n".join(lines) + "\n"

    deltas = suggestion.deltas
    lines.extend(
        [
            f"# suggested: ndcg@{suggestion.best.k}={suggestion.best.ndcg_at_k:.4f} "
            f"recall={suggestion.best.recall_at_k:.4f} mrr={suggestion.best.mrr:.4f}",
            f"# delta    : ndcg {deltas['ndcg_at_k']:+.4f}, recall {deltas['recall_at_k']:+.4f}, "
            f"mrr {deltas['mrr']:+.4f}",
            "",
        ]
    )

    sections: dict[str, dict[str, Any]] = {}
    for key, value in sorted(suggestion.best.candidate.overrides.items()):
        section, _, field_name = key.partition(".")
        sections.setdefault(section, {})[field_name] = value

    for section, values in sections.items():
        lines.append(f"{section}:")
        for field_name, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            lines.append(f"  {field_name}: {rendered}")

    return "\n".join(lines) + "\n"
