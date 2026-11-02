"""The retrieval regression gate (D7).

Makes a retrieval-quality drop fail a change the way a type error does. Config and index
changes are otherwise unguarded: nothing else in a normal CI run notices that recall fell,
so the regression ships and is discovered weeks later as "search got worse".

Four rules decide whether the gate can be trusted:

* **A baseline is only comparable to a run that could have produced it.** It records the
  embedding model and a *comparability fingerprint* of the settings that decide what comes
  back. Comparing across a retrieval-behaviour change measures that change, not the one
  under test, so a mismatched baseline is refused rather than silently compared.
* **A missing baseline blocks rather than passes.** A gate that waves through anything it
  cannot check is worse than no gate: it looks like protection while providing none.
* **Nothing compared is never a pass.** The verdict is three-valued, in the vocabulary
  ``scripts/compare_nightly.py`` uses for the nightly benchmark, so "could not compare" is
  reported as itself instead of collapsing into a green tick.
* **Improvements never block.** Only drops beyond ``eval.recall_tolerance`` and
  ``eval.ndcg_tolerance`` fail, so tuning upward is never obstructed.

**Why the comparability fingerprint is not the lockfile hash.** One hash was serving two
different questions, and they want opposite answers:

* ``core.identity.retrieval_config_hash`` answers *"has the index drifted from the config
  that built it?"* for ``index.lock``. **Any** difference over the whole chunking,
  embeddings, and retrieval models matters there, because the index really was built under
  the old one. Being broad is the point, and it stays exactly as broad as it was.
* This module answers *"is this eval result comparable to the recorded baseline?"* Only
  settings that change what retrieval returns matter. Under the lockfile hash, adding a
  knob whose default preserves behaviour retired every recorded baseline — which is how
  ``chunking.token_counter``, ``chars_per_token``, ``semantic_percentile``,
  ``retrieval.bm25_k1`` and ``bm25_b`` retired the 2026-08-02 handbook baseline (TASK-0244),
  along with the identity-version-3 change to the digest itself.

So comparability is an **explicitly enumerated list of fields**, recorded as values rather
than as a digest, and a field a baseline does not mention is read as that field's current
default — which is what lets a behaviour-preserving addition leave a baseline alive.

The two errors run in opposite directions and are not equally bad. Too broad retires good
baselines: annoying, and visible the moment it happens. Too narrow silently compares runs
that are not comparable: dangerous, and invisible. So the enumeration errs broad — a field
that only matters under some other setting's value is still listed — and
``BEHAVIOUR_NEUTRAL_FIELDS`` names every field deliberately left out, so that
``tests/unit/services/test_comparability.py`` can fail the build when a new setting appears
in neither list. A knob cannot be added to the schema and silently omitted from here.

``COMPARABILITY_VERSION`` covers the one case the "absent means default" rule cannot: a new
field whose default does **not** preserve behaviour. Bumping it retires every baseline
recorded before the change, which is the broad, visible error rather than the quiet one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.evals.harness import EvalReport
from fasterrag.core.identity import retrieval_config_hash
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "BEHAVIOUR_NEUTRAL_FIELDS",
    "COMPARABILITY_FIELDS",
    "COMPARABILITY_SECTIONS",
    "COMPARABILITY_VERSION",
    "DEFAULT_FINGERPRINT",
    "Baseline",
    "GateResult",
    "Verdict",
    "check_regression",
    "comparability_fingerprint",
    "load_baseline",
    "write_baseline",
]

_logger = get_logger(__name__)

# CRITICAL: bump this when a setting is added whose default does NOT reproduce the previous
# behaviour, or when an existing entry below changes meaning. Every baseline recorded under
# an older version is then refused until it is re-measured. The "absent means default" rule
# in `Baseline.incomparabilities` is sound only while that promise holds.
#
#   1  2026-11-02: the fingerprint replaced `retrieval_config_hash` for comparability
#      (TASK-0244). Baselines recorded before it carry version 0 and are refused.
COMPARABILITY_VERSION: Final = 1

COMPARABILITY_SECTIONS: Final[tuple[str, ...]] = ("chunking", "embeddings", "retrieval")

# CRITICAL: every field here is part of what a recorded score means. Entries that only bite
# under another setting's value — `semantic_percentile` outside the semantic strategy,
# `chars_per_token` outside the estimating counter — are listed anyway: a conditional
# exemption is a second rule that can be wrong, and being wrong here is the invisible
# direction. Anything added to one of COMPARABILITY_SECTIONS belongs in this tuple or in
# BEHAVIOUR_NEUTRAL_FIELDS below, and the exhaustiveness test fails until it is in one.
COMPARABILITY_FIELDS: Final[tuple[str, ...]] = (
    "chunking.strategy",
    "chunking.chunk_size",
    "chunking.overlap",
    "chunking.token_counter",
    "chunking.chars_per_token",
    "chunking.semantic_percentile",
    "chunking.contextual_enrichment",
    "chunking.context_tokens",
    "embeddings.provider",
    "embeddings.model",
    "embeddings.dimensions",
    "embeddings.tiering",
    "retrieval.top_k",
    "retrieval.hybrid",
    "retrieval.bm25_weight",
    "retrieval.dense_weight",
    "retrieval.rrf_k",
    "retrieval.bm25_k1",
    "retrieval.bm25_b",
    "retrieval.rerank",
    "retrieval.reranker_model",
    "retrieval.rerank_top_n",
)

# CRITICAL: left out because changing one cannot change which chunks a query returns, only
# how fast or from where they are fetched. `retrieval_config_hash` excludes the same three
# from the lockfile for the same reason, so the two agree about them.
BEHAVIOUR_NEUTRAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "embeddings.api_key_env",
        "embeddings.batch_size",
        "embeddings.cache",
    }
)


class Verdict(StrEnum):
    """What the gate established, keeping "nothing was compared" out of "passed"."""

    WITHIN_TOLERANCE = "within-tolerance"
    REGRESSED = "regressed"
    NOT_COMPARABLE = "not-comparable"


def _render(value: Any) -> str:
    """Serialize one recorded setting so equal values always compare equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def comparability_fingerprint(settings: Settings) -> dict[str, str]:
    """Return the enumerated settings a recorded score is only comparable across.

    Values rather than a digest, so a refusal can name the field that moved.
    """
    sections: dict[str, dict[str, Any]] = {
        name: getattr(settings, name).model_dump(mode="json") for name in COMPARABILITY_SECTIONS
    }
    fingerprint: dict[str, str] = {}
    for dotted in COMPARABILITY_FIELDS:
        section, _, name = dotted.partition(".")
        fingerprint[dotted] = _render(sections[section][name])
    return fingerprint


DEFAULT_FINGERPRINT: Final[dict[str, str]] = comparability_fingerprint(Settings.model_validate({}))


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
    fingerprint_version: int = 0
    fingerprint: dict[str, str] = field(default_factory=dict)

    def incomparabilities(self, settings: Settings) -> list[str]:
        """Return every reason a run under ``settings`` would not measure what this did.

        A field this baseline does not mention is read as its current default, which is
        what keeps a behaviour-preserving addition from retiring a good baseline.
        """
        if self.fingerprint_version != COMPARABILITY_VERSION:
            return [
                f"the baseline carries comparability fingerprint version "
                f"{self.fingerprint_version} and this build reads version "
                f"{COMPARABILITY_VERSION}, so what it recorded cannot be lined up with "
                "what this run configured; re-record it with a measured run"
            ]

        reasons: list[str] = []
        if self.embedding_model != settings.embeddings.model:
            reasons.append(
                f"the baseline was recorded with embedding model {self.embedding_model} and "
                f"this run uses {settings.embeddings.model}"
            )

        current = comparability_fingerprint(settings)
        for dotted in COMPARABILITY_FIELDS:
            recorded = self.fingerprint.get(dotted, DEFAULT_FINGERPRINT[dotted])
            if recorded != current[dotted]:
                reasons.append(f"{dotted} was {recorded} and is {current[dotted]}")
        return reasons

    def comparable_to(self, settings: Settings) -> bool:
        """Return whether a run under ``settings`` measures the same thing this did."""
        return not self.incomparabilities(settings)

    def as_dict(self) -> dict[str, Any]:
        """Return the committed JSON form."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Whether a run may proceed, what was established, and what stopped it."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    report: EvalReport | None = None
    baseline: Baseline | None = None
    obstacles: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """Return the three-valued outcome; an unchecked run is never within tolerance."""
        if self.failures:
            return Verdict.REGRESSED
        if self.obstacles:
            return Verdict.NOT_COMPARABLE
        return Verdict.WITHIN_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable result CI consumes."""
        return {
            "passed": self.passed,
            "verdict": self.verdict.value,
            "failures": self.failures,
            "obstacles": self.obstacles,
            "report": self.report.as_dict() if self.report else None,
            "baseline": self.baseline.as_dict() if self.baseline else None,
        }

    def summary(self) -> str:
        """Return a one-line verdict for a terminal."""
        if self.verdict is Verdict.REGRESSED:
            return "retrieval regression gate REGRESSED: " + "; ".join(self.failures)
        if self.verdict is Verdict.NOT_COMPARABLE:
            return (
                "retrieval regression gate NOT COMPARABLE, no regression ruled out: "
                + "; ".join(self.obstacles)
            )
        return "retrieval regression gate passed"


def _validate(payload: Any) -> dict[str, Any]:
    """Return a loaded baseline document, refusing a fingerprint of the wrong shape."""
    if not isinstance(payload, dict):
        raise ValueError(f"a baseline is a JSON object, not a {type(payload).__name__}")

    recorded = payload.get("fingerprint", {})
    if not isinstance(recorded, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in recorded.items()
    ):
        raise ValueError("'fingerprint' must be a flat mapping of field name to recorded value")

    document: dict[str, Any] = payload
    return document


def load_baseline(path: str | Path) -> Baseline | None:
    """Read a committed baseline, or return None when none exists yet."""
    source = Path(path)
    if not source.is_file():
        return None

    try:
        payload = _validate(json.loads(source.read_text(encoding="utf-8")))
        return Baseline(**payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FasterRagError(
            f"{source} exists but is not a valid baseline: {exc}. Re-record it with a "
            "measured run rather than editing it by hand",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        ) from exc


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
        fingerprint_version=COMPARABILITY_VERSION,
        fingerprint=comparability_fingerprint(settings),
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

    A disabled gate passes and still reports ``not-comparable``: the operator opted out, so
    nothing may block, but nothing was established either and no caller may read it as green.
    """
    if not settings.eval.regression_gate:
        return GateResult(
            passed=True,
            report=report,
            baseline=baseline,
            obstacles=[
                "eval.regression_gate is false, so this run was scored and not judged "
                "against anything"
            ],
        )

    if baseline is None:
        return GateResult(
            passed=False,
            report=report,
            obstacles=[
                "no baseline is committed, so a regression cannot be detected; record one "
                "with a measured run before enabling the gate"
            ],
        )

    incomparable = baseline.incomparabilities(settings)
    if incomparable:
        return GateResult(
            passed=False,
            report=report,
            baseline=baseline,
            obstacles=[
                "the baseline was recorded under a retrieval setup this run does not "
                "reproduce; comparing across it measures the setup change rather than this "
                "one, so re-record the baseline deliberately",
                *incomparable,
            ],
        )

    if report.k != baseline.k:
        return GateResult(
            passed=False,
            report=report,
            baseline=baseline,
            obstacles=[
                f"the baseline measured recall@{baseline.k} but this run measured "
                f"recall@{report.k}; the two are not comparable"
            ],
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
