"""Eval-driven auto-tuning that only ever suggests (D6).

Every framework hands you a dozen retrieval knobs and a shrug; tuning them is folklore
carried between projects by whoever tuned last. Autopilot replaces the folklore with a
measurement: it evaluates candidate configurations against a golden set built from *your*
corpus and reports what each one actually scored.

**It never writes ``config.yaml``.** The output is a suggested diff and the measured deltas
behind it; a human reads them and decides. That is not timidity — an auto-tuner that edits
production configuration turns a bad measurement into an outage, and the measurement is only
as good as the golden set behind it.

**Two searches, not one, because they cost different things.**

:func:`tune` searches *query-time* parameters. The fusion weights, ``rrf_k``, and reranking
all change how an existing index is queried, so one index serves every candidate and a
candidate costs one evaluation. ``top_k`` is excluded even though it is query-time: measuring
recall@k means retrieving exactly k, so the harness fixes the depth and a top_k candidate is
a no-op by construction.

:func:`sweep_index_time` searches *index-time* parameters — chunk size and overlap. These
change how the index is built, so no index can serve two candidates: each one is a different
index, and scoring it means re-chunking and re-embedding the corpus first. Three rules make
that affordable to reason about rather than merely possible to run:

* **The cost is priced before it is paid.** :func:`plan_index_sweep` runs D9's estimator
  (``services/estimation.py``) once per candidate and reports chunks, embedding tokens, and
  list-price cost. A sweep refuses to start unless the caller passes ``approved_rebuilds``
  covering what the plan says it would build, so the expensive default is "do nothing".
* **Every rebuild is scratch and is dropped.** Candidates are built into collections named
  ``autopilot-sweep-*``, never behind the live alias, and the drop runs in a ``finally`` over
  every name recorded *before* its build started — a rebuild that dies halfway still gets
  cleaned up. A drop that itself fails is reported as a leak rather than swallowed.
* **The corpus swept is the dataset's own.** A golden set only means anything against the
  corpus it was written against, and that corpus is small enough to rebuild a handful of
  times; the production corpus is neither.

**Neither search writes ``config.yaml``.** A sweep does create and delete scratch collections
— that is real work on the vector database — but the configuration it recommends is only ever
a suggestion, and an index-time suggestion additionally needs a reindex before it means
anything, which the rendered fragment says out loud.

**Both searches are time-boxed and report what they did not reach.** A tuner that silently
evaluated a third of its grid and announced a winner would be presenting the best of an
arbitrary subset as though it were the best available.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from pydantic import ValidationError

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.estimation import PRICES_DATED, estimate_sources, require_estimator
from fasterrag.services.evaluation import EvalDataset, score_collection

__all__ = [
    "SCRATCH_PREFIX",
    "SUGGESTION_FILE",
    "Candidate",
    "IndexSweepPlan",
    "RebuildEstimate",
    "RebuildFn",
    "Suggestion",
    "SweepOutcome",
    "TrialResult",
    "candidate_grid",
    "index_candidate_grid",
    "plan_index_sweep",
    "render_plan",
    "render_suggestion",
    "sweep_index_time",
    "tune",
]

SUGGESTION_FILE: Final = "autopilot-suggestion.yaml"

# CRITICAL: a sweep collection must never be mistaken for a live one, so the name is
# deliberately unlike `reindex.green_name`'s `<alias>-v<stamp>`: nothing resolves an alias to
# it, no rollback can restore it, and an operator grepping for leftovers after a crash has one
# fixed prefix to look for. Changing this orphans collections created under the old one.
SCRATCH_PREFIX: Final = "autopilot-sweep"

RebuildFn = Callable[[str, Settings], Awaitable[None]]
"""Build one scratch collection from a candidate's settings.

Injected rather than imported so this module stays independent of the ingestion service, and
so the sweep's cleanup and budget behaviour are testable without a vector database. The
implementation owns creating the collection at the candidate's dimensions.
"""

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

# CRITICAL: index-time only, and searched by `sweep_index_time`, never by `tune`. Every key
# here costs one full re-chunk and re-embed of the corpus per candidate, which is why the two
# searches have separate entry points, separate budgets and separate opt-ins. Moving a key
# from this tuple into `_SEARCHED_KEYS` would make `tune` score candidates against an index
# that was not built for them — silently wrong numbers rather than a visible failure.
_INDEX_SEARCHED_KEYS: Final[tuple[str, ...]] = (
    "chunking.chunk_size",
    "chunking.overlap",
)

_INDEX_SECTIONS: Final[frozenset[str]] = frozenset({"chunking", "embeddings"})

_logger = get_logger(__name__)


def _is_index_time(key: str) -> bool:
    """Return whether a key changes how the index is built rather than how it is queried."""
    return key.partition(".")[0] in _INDEX_SECTIONS


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


@dataclass(frozen=True, slots=True)
class RebuildEstimate:
    """What one index-time candidate would cost to build before it can be scored."""

    candidate: Candidate
    collection: str
    documents: int
    chunks: int
    tokens: int
    cost_usd: float | None
    basis: str

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized estimate."""
        return {
            "overrides": self.candidate.overrides,
            "label": self.candidate.label,
            "collection": self.collection,
            "documents": self.documents,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class IndexSweepPlan:
    """What an index-time sweep would build, priced before any of it is built."""

    collection: str
    corpus: str
    estimates: list[RebuildEstimate] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    prices_dated: str = PRICES_DATED

    @property
    def rebuilds(self) -> int:
        """Return how many scratch indexes the sweep would build."""
        return len(self.estimates)

    @property
    def tokens(self) -> int:
        """Return the embedding tokens every rebuild would consume between them."""
        return sum(estimate.tokens for estimate in self.estimates)

    @property
    def cost_usd(self) -> float | None:
        """Return the total list price, or ``None`` when any rebuild is unpriced.

        One unpriced rebuild makes the total unknown, not smaller. Summing the rest would
        report a confident figure that is wrong by an amount nobody can see.
        """
        if any(estimate.cost_usd is None for estimate in self.estimates):
            return None
        return sum(estimate.cost_usd or 0.0 for estimate in self.estimates)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized plan."""
        return {
            "collection": self.collection,
            "corpus": self.corpus,
            "rebuilds": self.rebuilds,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "prices_dated": self.prices_dated,
            "scratch_prefix": SCRATCH_PREFIX,
            "estimates": [estimate.as_dict() for estimate in self.estimates],
            "rejected": list(self.rejected),
        }


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    """What an index-time sweep measured, and what it did to the database to measure it."""

    plan: IndexSweepPlan
    suggestion: Suggestion
    rebuilt: int = 0
    scratch: list[str] = field(default_factory=list)
    leaked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized outcome."""
        return {
            **self.suggestion.as_dict(),
            "plan": self.plan.as_dict(),
            "rebuilt": self.rebuilt,
            "scratch_collections": list(self.scratch),
            "leaked_collections": list(self.leaked),
            "note": (
                "Autopilot never writes config.yaml. This sweep did create and drop "
                f"{self.rebuilt} scratch collection(s) named {SCRATCH_PREFIX}-*; any listed "
                "under leaked_collections survived and must be dropped by hand. Applying an "
                "index-time suggestion also needs a reindex: changing config.yaml alone "
                "leaves the live index as the old settings built it."
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


def index_candidate_grid(settings: Settings) -> list[Candidate]:
    """Return the index-time candidates to try, each one a whole rebuild's worth of work.

    A sparse ladder, not a grid: every rung is a re-chunk and re-embed of the corpus, so the
    default search is five rebuilds at most rather than a dense product of two dimensions.
    """
    current = settings.chunking
    candidates: list[Candidate] = []

    # CRITICAL: one key per candidate, never a pair, and a ladder rather than a product.
    # A candidate that moved chunk size and overlap together would cost the same rebuild as
    # a single-key one while reporting a score neither half can be credited with. The rungs
    # are anchored on the shipped default and spaced by halving and doubling, because a
    # golden set of a few dozen queries cannot separate adjacent values; overlap 0 is in
    # deliberately, since "does overlap earn its duplicated tokens" is the one overlap
    # question whose answer moves the bill as well as the score. Widening either tuple is
    # not free: every entry added is another full re-chunk and re-embed of the corpus.
    for chunk_size in (384, 768, 1536):
        if chunk_size != current.chunk_size:
            candidates.append(Candidate({"chunking.chunk_size": chunk_size}))

    for overlap in (0, 128):
        if overlap != current.overlap:
            candidates.append(Candidate({"chunking.overlap": overlap}))

    return candidates


def _corpus_sources(dataset: EvalDataset) -> list[str]:
    """Return the corpus files every scratch index in a sweep is rebuilt from."""
    corpus = dataset.corpus
    if not corpus.is_dir():
        raise FasterRagError(
            f"{corpus} is missing; an index-time sweep rebuilds the dataset's own corpus, so "
            "the dataset directory must carry the corpus/ its golden set was written against",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    files = sorted(
        str(path)
        for path in corpus.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(corpus).parts)
    )
    if not files:
        raise FasterRagError(
            f"{corpus} holds no files; there is nothing for a sweep to rebuild",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )
    return files


def plan_index_sweep(
    dataset: EvalDataset,
    settings: Settings,
    *,
    collection: str,
    candidates: Sequence[Candidate] | None = None,
    stamp: str | None = None,
) -> IndexSweepPlan:
    """Price every rebuild an index-time sweep would perform, before performing any.

    Reuses D9's estimator rather than inventing a second notion of cost, so the figure here
    is the same one ``fasterrag estimate`` prints and carries the same provenance.
    """
    require_estimator(settings)

    grid = list(candidates if candidates is not None else index_candidate_grid(settings))
    sources = _corpus_sources(dataset)
    moment = stamp or datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")

    estimates: list[RebuildEstimate] = []
    rejected: list[str] = []

    for candidate in grid:
        # CRITICAL: the baseline is skipped here because it must never be rebuilt. The live
        # collection already holds the current configuration's index, which is precisely
        # what a baseline measurement wants; building a copy of it would pay for a rebuild
        # to learn nothing and would compare the candidates against a second index rather
        # than against the one production is actually serving.
        if not candidate.overrides:
            continue

        query_time = [key for key in candidate.overrides if not _is_index_time(key)]
        if query_time:
            rejected.append(f"{candidate.label} (query-time; search it with tune() instead)")
            continue

        try:
            candidate_settings = candidate.apply(settings)
        except ValidationError:
            rejected.append(f"{candidate.label} (rejected by the configuration schema)")
            continue

        estimate = estimate_sources(sources, candidate_settings)
        priced = estimate.providers[0] if estimate.providers else None
        estimates.append(
            RebuildEstimate(
                candidate=candidate,
                collection=f"{SCRATCH_PREFIX}-{moment}-{len(estimates):02d}",
                documents=estimate.documents,
                chunks=estimate.chunks,
                tokens=estimate.tokens,
                cost_usd=priced.cost_usd if priced is not None else None,
                basis=priced.basis if priced is not None else "no embedding provider priced",
            )
        )

    plan = IndexSweepPlan(
        collection=collection,
        corpus=str(dataset.corpus),
        estimates=estimates,
        rejected=rejected,
    )
    _logger.info(
        "planned an index-time sweep",
        extra={
            "rebuilds": plan.rebuilds,
            "tokens": plan.tokens,
            "cost_usd": plan.cost_usd,
            "rejected": len(rejected),
        },
    )
    return plan


async def _drop_scratch(adapter: VectorDBAdapter, names: Sequence[str], leaked: list[str]) -> None:
    """Drop every scratch collection a sweep created, recording any that survived."""
    for name in names:
        try:
            await adapter.drop_collection(name)
        # CRITICAL: the net is wide here on purpose, and one failure never stops the loop.
        # This runs in the cleanup path of an operation that creates indexes, so an
        # unexpected error from one drop must not leave the remaining scratch collections
        # on a production database. Nothing is swallowed: each failure is logged with the
        # collection name and reported to the caller as a leak.
        except Exception as exc:
            leaked.append(name)
            _logger.error(
                "autopilot could not drop a scratch collection; drop it by hand",
                extra={"collection": name, "error": str(exc)},
            )


async def _score_candidate(
    candidate: Candidate,
    dataset: EvalDataset,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    collection: str,
) -> TrialResult:
    """Score one candidate against a collection and time how long that took."""
    started = time.perf_counter()
    report = await score_collection(dataset, settings, adapter, router, collection=collection)
    return TrialResult(
        candidate=candidate,
        recall_at_k=report.recall_at_k,
        mrr=report.mrr,
        ndcg_at_k=report.ndcg_at_k,
        k=report.k,
        seconds=time.perf_counter() - started,
    )


def _require_router_factory(
    plan: IndexSweepPlan, router_factory: Callable[[Settings], TieringRouter] | None
) -> None:
    """Refuse an embedding-model sweep that would score candidates with the wrong router."""
    if router_factory is not None:
        return

    unsupported = [
        estimate.candidate.label
        for estimate in plan.estimates
        if any(key.startswith("embeddings.") for key in estimate.candidate.overrides)
    ]
    if unsupported:
        raise FasterRagError(
            "scoring an embedding candidate needs a router built from that candidate's own "
            "settings — the supplied router embeds queries with the current model, which "
            f"would search {unsupported} with vectors from a different model and report the "
            "mismatch as a quality result. Pass router_factory, or drop those candidates",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )


async def sweep_index_time(
    plan: IndexSweepPlan,
    dataset: EvalDataset,
    settings: Settings,
    adapter: VectorDBAdapter,
    router: TieringRouter,
    *,
    rebuild: RebuildFn,
    approved_rebuilds: int = 0,
    budget_seconds: float = 3600.0,
    router_factory: Callable[[Settings], TieringRouter] | None = None,
) -> SweepOutcome:
    """Rebuild one scratch index per planned candidate, score it, and suggest the best.

    ``approved_rebuilds`` defaults to zero, so the expensive behaviour is opt-in at the call
    site: a caller that has not read the plan's cost cannot accidentally pay it.
    """
    if not settings.autopilot.enabled:
        raise FasterRagError(
            "autopilot.enabled is false; set it to true to allow eval-driven tuning",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    if approved_rebuilds < plan.rebuilds:
        raise FasterRagError(
            f"this sweep would rebuild the corpus {plan.rebuilds} time(s) at an estimated "
            f"{plan.tokens} embedding tokens, and {approved_rebuilds} rebuild(s) are "
            "approved; read the plan and raise approved_rebuilds once you accept that cost",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    _require_router_factory(plan, router_factory)

    started = time.perf_counter()
    created: list[str] = []
    leaked: list[str] = []
    trials: list[TrialResult] = []

    try:
        trials.append(
            await _score_candidate(
                Candidate({}), dataset, settings, adapter, router, plan.collection
            )
        )

        for estimate in plan.estimates:
            if time.perf_counter() - started >= budget_seconds:
                _logger.info(
                    "index sweep stopped at its budget",
                    extra={"rebuilt": len(created), "skipped": plan.rebuilds - len(created)},
                )
                break

            candidate_settings = estimate.candidate.apply(settings)
            # CRITICAL: recorded before the build starts, never after it succeeds. A rebuild
            # that dies halfway has already created the collection, and a name recorded only
            # on success would leave exactly that wreckage behind on a production database.
            created.append(estimate.collection)
            await rebuild(estimate.collection, candidate_settings)

            trial_router = router_factory(candidate_settings) if router_factory else router
            try:
                trials.append(
                    await _score_candidate(
                        estimate.candidate,
                        dataset,
                        candidate_settings,
                        adapter,
                        trial_router,
                        estimate.collection,
                    )
                )
            finally:
                if trial_router is not router:
                    await trial_router.close()
    finally:
        await _drop_scratch(adapter, created, leaked)

    best = max(trials, key=lambda trial: (trial.ndcg_at_k, trial.recall_at_k, trial.mrr))
    suggestion = Suggestion(
        baseline=trials[0],
        best=best,
        trials=trials,
        evaluated=len(trials),
        skipped=plan.rebuilds - (len(trials) - 1),
        seconds=time.perf_counter() - started,
        created_at=datetime.now(tz=UTC).isoformat(),
    )

    _logger.info(
        "index sweep finished",
        extra={
            "rebuilt": len(created),
            "evaluated": suggestion.evaluated,
            "improves": suggestion.improves,
            "leaked": len(leaked),
        },
    )
    return SweepOutcome(
        plan=plan,
        suggestion=suggestion,
        rebuilt=len(created),
        scratch=created,
        leaked=leaked,
    )


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
    """Search query-time configurations against a golden set and suggest the best."""
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

    if any(_is_index_time(key) for key in suggestion.best.candidate.overrides):
        lines.extend(
            [
                "# These are index-time settings. Pasting them in is not enough: the live",
                "# index was built by the old ones and keeps serving what they produced",
                "# until `fasterrag index reembed` rebuilds it.",
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


def render_plan(plan: IndexSweepPlan) -> str:
    """Render what an index-time sweep would do, for a human to read before approving it.

    The whole point of the plan is that it is read, so this leads with the number of
    rebuilds and the cost rather than burying them under a table of candidates.
    """
    cost = (
        f"${plan.cost_usd:.4f} at list prices (checked {plan.prices_dated})"
        if plan.cost_usd is not None
        else "unknown — at least one rebuild uses an unpriced embedding model"
    )
    lines = [
        "Index-time sweep plan — NOTHING HAS BEEN BUILT YET.",
        f"  baseline       {plan.collection} (already built; scored as-is, no rebuild)",
        f"  corpus         {plan.corpus}",
        f"  rebuilds       {plan.rebuilds}",
        f"  tokens         {plan.tokens} to embed across every rebuild",
        f"  cost           {cost}",
        "",
        "candidates (chunks / tokens / cost):",
    ]

    for estimate in plan.estimates:
        priced = f"${estimate.cost_usd:.4f}" if estimate.cost_usd is not None else "unpriced"
        lines.append(
            f"  {estimate.chunks:>7} / {estimate.tokens:>9} / {priced:>10}   "
            f"{estimate.candidate.label}  -> {estimate.collection}"
        )

    for entry in plan.rejected:
        lines.append(f"  skipped: {entry}")

    lines.extend(
        [
            "",
            f"Each candidate is built into its own {SCRATCH_PREFIX}-* collection, scored, and",
            "dropped. Nothing is written to config.yaml and no alias is moved, so the live",
            "index keeps serving throughout. Approve the rebuild count to run it.",
        ]
    )
    return "\n".join(lines) + "\n"
