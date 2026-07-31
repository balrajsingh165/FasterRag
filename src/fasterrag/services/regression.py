"""The retrieval regression gate (D7).

Makes a retrieval-quality drop fail a change the way a type error does. Config and index
changes are otherwise unguarded: nothing else in a normal CI run notices that recall fell,
so the regression ships and is discovered weeks later as "search got worse".

Three rules decide whether the gate can be trusted:

* **A baseline is only comparable to a run that could have produced it.** It records the
  embedding model and the retrieval-affecting config hash behind its numbers. Comparing
  across a model change measures the model, not the change under test, so a mismatched
  baseline is refused rather than silently compared.
* **A missing baseline blocks rather than passes.** A gate that waves through anything it
  cannot check is worse than no gate: it looks like protection while providing none.
* **Improvements never block.** Only drops beyond ``eval.recall_tolerance`` and
  ``eval.ndcg_tolerance`` fail, so tuning upward is never obstructed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.evals.harness import EvalReport
from fasterrag.core.identity import retrieval_config_hash
from fasterrag.observability.logging import get_logger

__all__ = [
    "Baseline",
    "GateResult",
    "check_regression",
    "load_baseline",
    "write_baseline",
]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Baseline:
    """A recorded measurement, and the setup that produced it."""

    recorded_at: str
    k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    scored: int
    embedding_model: str
    config_hash: str

    def comparable_to(self, settings: Settings) -> bool:
        """Return whether a run under ``settings`` measures the same thing this did."""
        return (
            self.embedding_model == settings.embeddings.model
            and self.config_hash == retrieval_config_hash(settings)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the committed JSON form."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Whether a run may proceed, and why not when it may not."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    report: EvalReport | None = None
    baseline: Baseline | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable result CI consumes."""
        return {
            "passed": self.passed,
            "failures": self.failures,
            "report": self.report.as_dict() if self.report else None,
            "baseline": self.baseline.as_dict() if self.baseline else None,
        }

    def summary(self) -> str:
        """Return a one-line verdict for a terminal."""
        if self.passed:
            return "retrieval regression gate passed"
        return "retrieval regression gate BLOCKED: " + "; ".join(self.failures)


def load_baseline(path: str | Path) -> Baseline | None:
    """Read a committed baseline, or return None when none exists yet."""
    source = Path(path)
    if not source.is_file():
        return None

    payload = json.loads(source.read_text(encoding="utf-8"))
    return Baseline(**payload)


def write_baseline(path: str | Path, report: EvalReport, settings: Settings) -> Baseline:
    """Record a run as the new baseline.

    Writing a baseline is an explicit act, never automatic: a gate that re-baselined itself
    on every run would ratchet quality downward one tolerated drop at a time.
    """
    baseline = Baseline(
        recorded_at=datetime.now(tz=UTC).isoformat(),
        k=report.k,
        recall_at_k=round(report.recall_at_k, 6),
        mrr=round(report.mrr, 6),
        ndcg_at_k=round(report.ndcg_at_k, 6),
        scored=report.scored,
        embedding_model=settings.embeddings.model,
        config_hash=retrieval_config_hash(settings),
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(baseline.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _logger.info("recorded a retrieval baseline", extra=baseline.as_dict())
    return baseline


def check_regression(
    report: EvalReport,
    baseline: Baseline | None,
    settings: Settings,
) -> GateResult:
    """Compare a run against its baseline and decide whether it may proceed.

    Args:
        report: The run being judged.
        baseline: The recorded reference, or None when none is committed.
        settings: Configuration supplying the tolerances and identifying the setup.

    Returns:
        The verdict, naming every metric that regressed.
    """
    if not settings.eval.regression_gate:
        return GateResult(passed=True, report=report, baseline=baseline)

    if baseline is None:
        return GateResult(
            passed=False,
            failures=[
                "no baseline is committed, so a regression cannot be detected; record one "
                "with a measured run before enabling the gate"
            ],
            report=report,
        )

    if not baseline.comparable_to(settings):
        return GateResult(
            passed=False,
            failures=[
                "the baseline was recorded with a different embedding model or retrieval "
                f"configuration ({baseline.embedding_model} at {baseline.config_hash[:12]}); "
                "comparing across that measures the setup change, not this one — re-record "
                "the baseline deliberately"
            ],
            report=report,
            baseline=baseline,
        )

    if report.k != baseline.k:
        return GateResult(
            passed=False,
            failures=[
                f"the baseline measured recall@{baseline.k} but this run measured "
                f"recall@{report.k}; the two are not comparable"
            ],
            report=report,
            baseline=baseline,
        )

    failures = [
        message
        for message in (
            _regression(
                "recall@k", baseline.recall_at_k, report.recall_at_k, settings.eval.recall_tolerance
            ),
            _regression(
                "nDCG@k", baseline.ndcg_at_k, report.ndcg_at_k, settings.eval.ndcg_tolerance
            ),
        )
        if message
    ]

    result = GateResult(passed=not failures, failures=failures, report=report, baseline=baseline)
    _logger.info("retrieval regression gate evaluated", extra=result.as_dict())
    return result


def _regression(name: str, before: float, after: float, tolerance: float) -> str | None:
    """Return a failure message when a metric dropped beyond its tolerance."""
    drop = before - after
    if drop <= tolerance:
        return None
    return (
        f"{name} fell from {before:.4f} to {after:.4f}, a drop of {drop:.4f} "
        f"beyond the {tolerance:.4f} tolerance"
    )
