from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, GoldenRecord
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services import autopilot as module
from fasterrag.services.autopilot import (
    SCRATCH_PREFIX,
    Candidate,
    IndexSweepPlan,
    RebuildEstimate,
    Suggestion,
    TrialResult,
    candidate_grid,
    index_candidate_grid,
    plan_index_sweep,
    render_plan,
    render_suggestion,
    sweep_index_time,
    tune,
)
from fasterrag.services.evaluation import EvalDataset


def enabled(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {"autopilot": {"enabled": True}}
    payload.update(overrides)
    return Settings.model_validate(payload)


def dataset(tmp_path: Path) -> EvalDataset:
    return EvalDataset(
        root=tmp_path,
        records=[
            GoldenRecord(id="q_1", query="q", source="human", created_at="2026-08-02"),
        ],
        k=5,
    )


def trial(label: str, ndcg: float, recall: float = 0.0, mrr: float = 0.0) -> TrialResult:
    return TrialResult(
        candidate=Candidate({"retrieval.top_k": 7} if label else {}),
        recall_at_k=recall,
        mrr=mrr,
        ndcg_at_k=ndcg,
        k=5,
    )


def scores(*values: float) -> Any:
    """Return a scorer yielding the given ndcg values in order."""
    remaining = list(values)

    async def score(
        ds: Any, settings: Any, adapter: Any, router: Any, *, collection: str
    ) -> EvalReport:
        value = remaining.pop(0) if remaining else 0.0
        return EvalReport(
            k=5, scored=1, adversarial=0, recall_at_k=value, mrr=value, ndcg_at_k=value
        )

    return score


def test_a_candidate_applies_its_overrides_without_mutating_the_original() -> None:
    settings = enabled()
    candidate = Candidate({"retrieval.top_k": 42})

    applied = candidate.apply(settings)

    assert applied.retrieval.top_k == 42
    assert settings.retrieval.top_k != 42


def test_a_candidate_override_is_validated() -> None:
    """An out-of-bounds candidate is rejected by the schema, never silently searched."""
    with pytest.raises(ValidationError):
        Candidate({"retrieval.top_k": 9999}).apply(enabled())


def test_the_baseline_candidate_changes_nothing() -> None:
    settings = enabled()

    assert Candidate({}).apply(settings).retrieval.top_k == settings.retrieval.top_k
    assert Candidate({}).label.startswith("baseline")


def test_the_grid_starts_from_the_current_configuration() -> None:
    grid = candidate_grid(enabled())

    assert grid[0].overrides == {}


def test_the_grid_never_retries_the_current_value() -> None:
    grid = candidate_grid(enabled(retrieval={"rrf_k": 30}))

    assert {"retrieval.rrf_k": 30.0} not in [candidate.overrides for candidate in grid]


def test_the_grid_never_searches_top_k() -> None:
    """Measuring recall@k fixes the retrieval depth, so a top_k candidate cannot differ."""
    for candidate in candidate_grid(enabled()):
        assert "retrieval.top_k" not in candidate.overrides


def test_the_grid_searches_only_query_time_keys() -> None:
    """An index-time key would make each trial a full corpus rebuild."""
    for candidate in candidate_grid(enabled()):
        for key in candidate.overrides:
            assert key.startswith("retrieval."), f"{key} is not a query-time parameter"


def test_the_grid_offers_to_turn_reranking_off_when_it_is_on() -> None:
    grid = candidate_grid(enabled(retrieval={"rerank": True}))

    assert {"retrieval.rerank": False} in [candidate.overrides for candidate in grid]


async def test_tuning_is_refused_when_the_flag_is_off(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError) as failure:
        await tune(dataset(tmp_path), Settings(), None, None, collection="c")  # type: ignore[arg-type]

    assert failure.value.code is ErrorCode.VALIDATION_FAILED


async def test_a_better_candidate_is_suggested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", scores(0.5, 0.9))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        candidates=[Candidate({}), Candidate({"retrieval.top_k": 20})],
    )

    assert suggestion.improves is True
    assert suggestion.best.candidate.overrides == {"retrieval.top_k": 20}
    assert suggestion.deltas["ndcg_at_k"] == pytest.approx(0.4)


async def test_a_worse_candidate_is_not_suggested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", scores(0.9, 0.4))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        candidates=[Candidate({}), Candidate({"retrieval.top_k": 20})],
    )

    assert suggestion.improves is False
    assert suggestion.best.candidate.overrides == {}


async def test_a_tie_never_displaces_the_current_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suggesting a change that measured identically trains people to ignore diffs."""
    monkeypatch.setattr(module, "score_collection", scores(0.8, 0.8))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        candidates=[Candidate({}), Candidate({"retrieval.top_k": 20})],
    )

    assert suggestion.improves is False
    assert suggestion.best.candidate.overrides == {}


async def test_every_trial_is_reported_not_just_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", scores(0.5, 0.6, 0.7))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        candidates=[
            Candidate({}),
            Candidate({"retrieval.top_k": 5}),
            Candidate({"retrieval.top_k": 20}),
        ],
    )

    assert suggestion.evaluated == 3
    assert len(suggestion.as_dict()["trials"]) == 3


async def test_the_budget_stops_the_search_and_reports_what_it_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", scores(0.5, 0.6, 0.7, 0.8))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        budget_seconds=0.0,
        candidates=[
            Candidate({}),
            Candidate({"retrieval.top_k": 5}),
            Candidate({"retrieval.top_k": 20}),
        ],
    )

    assert suggestion.evaluated == 1
    assert suggestion.skipped == 2


async def test_the_baseline_always_runs_even_at_a_zero_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a baseline there is nothing to measure an improvement against."""
    monkeypatch.setattr(module, "score_collection", scores(0.5))

    suggestion = await tune(
        dataset(tmp_path),
        enabled(),
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        collection="c",
        budget_seconds=0.0,
        candidates=[Candidate({}), Candidate({"retrieval.top_k": 5})],
    )

    assert suggestion.evaluated == 1
    assert suggestion.baseline.candidate.overrides == {}


def test_the_serialized_suggestion_states_it_was_not_applied() -> None:
    suggestion = Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)

    payload = suggestion.as_dict()

    assert payload["applied"] is False
    assert "never writes config.yaml" in payload["note"]


def test_the_serialized_suggestion_omits_overrides_when_nothing_improved() -> None:
    suggestion = Suggestion(baseline=trial("", 0.9), best=trial("", 0.9), evaluated=2)

    assert suggestion.as_dict()["suggested_overrides"] == {}


def test_the_rendered_suggestion_says_it_is_not_applied() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)
    )

    assert "NOT APPLIED" in rendered
    assert "never writes config.yaml" in rendered


def test_the_rendered_suggestion_carries_the_overrides_as_yaml() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)
    )

    assert "retrieval:" in rendered
    assert "top_k: 7" in rendered


def test_the_rendered_suggestion_reports_the_measured_delta() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)
    )

    assert "delta" in rendered
    assert "+0.4000" in rendered


def test_a_run_that_found_nothing_renders_an_empty_change() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.9), best=trial("", 0.9), evaluated=2)
    )

    assert "No candidate beat the current configuration" in rendered
    assert "retrieval:" not in rendered


def test_the_rendered_suggestion_is_a_fragment_not_a_whole_config() -> None:
    """A whole file would invite overwriting a config full of keys Autopilot never saw."""
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)
    )

    assert "vector_db:" not in rendered
    assert "embeddings:" not in rendered


def test_the_rendered_suggestion_reports_what_the_budget_skipped() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2, skipped=7)
    )

    assert "7 skipped" in rendered


class FakeAdapter:
    """A vector database that only records the scratch collections it was asked to drop."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.drops: list[str] = []
        self.fail_on = fail_on

    async def drop_collection(self, name: str) -> bool:
        self.drops.append(name)
        if name == self.fail_on:
            raise RuntimeError("the backend refused the drop")
        return True


class FakeRouter:
    """An embedding router that records whether the sweep closed it."""

    def __init__(self, label: str = "base") -> None:
        self.label = label
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def fake_adapter(fail_on: str | None = None) -> Any:
    """Return the stub adapter typed loosely, as every call site passes it to the service."""
    return FakeAdapter(fail_on)


def fake_router(label: str = "base") -> Any:
    """Return the stub router typed loosely, as every call site passes it to the service."""
    return FakeRouter(label)


class FakeRebuild:
    """Stands in for the ingestion path that builds one scratch index."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_on = fail_on

    async def __call__(self, collection: str, settings: Settings) -> None:
        self.calls.append((collection, settings.chunking.chunk_size))
        if collection == self.fail_on:
            raise FasterRagError(
                "the rebuild died halfway",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )


class FakeScorer:
    """Returns the given ndcg values in order, recording what it was asked to score."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.collections: list[str] = []
        self.routers: list[Any] = []
        self.chunk_sizes: list[int] = []

    async def __call__(
        self,
        dataset: Any,
        settings: Settings,
        adapter: Any,
        router: Any,
        *,
        collection: str,
    ) -> EvalReport:
        self.collections.append(collection)
        self.routers.append(router)
        self.chunk_sizes.append(settings.chunking.chunk_size)
        value = self.values.pop(0) if self.values else 0.0
        return EvalReport(
            k=5, scored=1, adversarial=0, recall_at_k=value, mrr=value, ndcg_at_k=value
        )


def sweepable(**overrides: Any) -> Settings:
    """Return settings with autopilot on and the tokenizer-free counter, as CI uses."""
    payload: dict[str, Any] = {
        "autopilot": {"enabled": True},
        "chunking": {"token_counter": "estimate"},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def sweep_dataset(tmp_path: Path) -> EvalDataset:
    """Return a dataset whose corpus holds one real file for the estimator to chunk."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "handbook.md").write_text("Parental leave policy. " * 400, encoding="utf-8")
    return EvalDataset(
        root=tmp_path,
        records=[GoldenRecord(id="q_1", query="q", source="human", created_at="2026-11-04")],
        k=5,
    )


def planned(tmp_path: Path, settings: Settings, **kwargs: Any) -> IndexSweepPlan:
    """Return a plan over the fixture dataset, stamped so the names are predictable."""
    return plan_index_sweep(
        sweep_dataset(tmp_path),
        settings,
        collection="live",
        stamp="20261104000000",
        **kwargs,
    )


def test_the_index_grid_searches_only_index_time_keys() -> None:
    """A query-time key in this grid would buy a whole rebuild to learn nothing new."""
    for candidate in index_candidate_grid(sweepable()):
        for key in candidate.overrides:
            assert key.startswith("chunking."), f"{key} does not change how the index is built"


def test_the_index_grid_changes_one_key_per_candidate() -> None:
    for candidate in index_candidate_grid(sweepable()):
        assert len(candidate.overrides) == 1


def test_the_index_grid_never_retries_the_current_value() -> None:
    grid = index_candidate_grid(sweepable(chunking={"chunk_size": 768, "overlap": 0}))

    overrides = [candidate.overrides for candidate in grid]
    assert {"chunking.chunk_size": 768} not in overrides
    assert {"chunking.overlap": 0} not in overrides


def test_the_index_grid_offers_no_baseline_because_it_needs_no_rebuild() -> None:
    assert all(candidate.overrides for candidate in index_candidate_grid(sweepable()))


def test_the_plan_prices_every_rebuild_before_anything_is_built(tmp_path: Path) -> None:
    plan = planned(tmp_path, sweepable())

    assert plan.rebuilds == len(index_candidate_grid(sweepable()))
    assert plan.tokens > 0
    assert all(estimate.chunks > 0 for estimate in plan.estimates)
    assert plan.cost_usd == 0.0


def test_the_plan_names_a_distinct_scratch_collection_per_rebuild(tmp_path: Path) -> None:
    """A shared name would have each candidate overwrite the one scored before it."""
    plan = planned(tmp_path, sweepable())

    names = [estimate.collection for estimate in plan.estimates]
    assert len(set(names)) == len(names)
    assert all(name.startswith(SCRATCH_PREFIX) for name in names)


def test_the_plan_never_names_the_live_collection(tmp_path: Path) -> None:
    plan = planned(tmp_path, sweepable())

    assert all(estimate.collection != plan.collection for estimate in plan.estimates)


def test_a_chunk_size_candidate_changes_what_the_rebuild_would_cost(tmp_path: Path) -> None:
    """Pricing every candidate identically would make the estimator decorative."""
    plan = planned(tmp_path, sweepable())

    chunks = {
        estimate.candidate.overrides.get("chunking.chunk_size"): estimate.chunks
        for estimate in plan.estimates
        if "chunking.chunk_size" in estimate.candidate.overrides
    }
    assert chunks[384] > chunks[1536]


def test_an_unpriced_rebuild_makes_the_total_unknown_not_smaller() -> None:
    priced = RebuildEstimate(
        candidate=Candidate({"chunking.chunk_size": 384}),
        collection="s0",
        documents=1,
        chunks=10,
        tokens=100,
        cost_usd=0.5,
        basis="list price",
    )
    unpriced = RebuildEstimate(
        candidate=Candidate({"chunking.overlap": 0}),
        collection="s1",
        documents=1,
        chunks=10,
        tokens=100,
        cost_usd=None,
        basis="no published price",
    )

    assert IndexSweepPlan("live", "c", [priced]).cost_usd == 0.5
    assert IndexSweepPlan("live", "c", [priced, unpriced]).cost_usd is None


def test_the_plan_refuses_a_query_time_candidate(tmp_path: Path) -> None:
    plan = planned(tmp_path, sweepable(), candidates=[Candidate({"retrieval.rrf_k": 10.0})])

    assert plan.rebuilds == 0
    assert "tune()" in plan.rejected[0]


def test_the_plan_drops_a_candidate_the_schema_refuses(tmp_path: Path) -> None:
    """An overlap wider than the chunk cannot be built, so it must not be budgeted for."""
    plan = planned(
        tmp_path,
        sweepable(chunking={"token_counter": "estimate", "chunk_size": 100}),
        candidates=[Candidate({"chunking.overlap": 128})],
    )

    assert plan.rebuilds == 0
    assert "schema" in plan.rejected[0]


def test_the_plan_is_refused_when_the_estimator_is_off(tmp_path: Path) -> None:
    """Without the estimator there is no cost to show, and the cost is the whole point."""
    settings = sweepable(cost={"estimator": False})

    with pytest.raises(FasterRagError) as failure:
        planned(tmp_path, settings)

    assert failure.value.code is ErrorCode.VALIDATION_FAILED


def test_the_plan_refuses_a_dataset_with_no_corpus(tmp_path: Path) -> None:
    """The refusal has to name the contract, not just the emptiness it ran into."""
    with pytest.raises(FasterRagError) as failure:
        plan_index_sweep(dataset(tmp_path), sweepable(), collection="live")

    assert failure.value.code is ErrorCode.NOT_FOUND
    assert "corpus/" in failure.value.detail


async def test_a_sweep_is_refused_when_the_flag_is_off(tmp_path: Path) -> None:
    plan = planned(tmp_path, sweepable())
    settings = Settings.model_validate({"chunking": {"token_counter": "estimate"}})

    with pytest.raises(FasterRagError) as failure:
        await sweep_index_time(
            plan,
            sweep_dataset(tmp_path),
            settings,
            fake_adapter(),
            fake_router(),
            rebuild=FakeRebuild(),
            approved_rebuilds=99,
        )

    assert failure.value.code is ErrorCode.VALIDATION_FAILED


async def test_a_sweep_rebuilds_nothing_unless_its_rebuilds_are_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expensive behaviour is opt-in: the default approval is zero rebuilds."""
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5))
    plan = planned(tmp_path, sweepable())
    rebuild = FakeRebuild()
    adapter = fake_adapter()

    with pytest.raises(FasterRagError) as failure:
        await sweep_index_time(
            plan,
            sweep_dataset(tmp_path),
            sweepable(),
            adapter,
            fake_router(),
            rebuild=rebuild,
        )

    assert failure.value.code is ErrorCode.VALIDATION_FAILED
    assert rebuild.calls == []
    assert adapter.drops == []


async def test_a_partial_approval_is_refused_rather_than_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running two of five would quietly redefine approved as started."""
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5))
    plan = planned(tmp_path, sweepable())
    rebuild = FakeRebuild()

    with pytest.raises(FasterRagError):
        await sweep_index_time(
            plan,
            sweep_dataset(tmp_path),
            sweepable(),
            fake_adapter(),
            fake_router(),
            rebuild=rebuild,
            approved_rebuilds=plan.rebuilds - 1,
        )

    assert rebuild.calls == []


async def test_an_approved_sweep_scores_the_baseline_on_the_live_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current configuration already has an index; rebuilding it would be waste."""
    scorer = FakeScorer(0.5, 0.6)
    monkeypatch.setattr(module, "score_collection", scorer)
    plan = planned(tmp_path, sweepable(), candidates=[Candidate({"chunking.chunk_size": 384})])
    rebuild = FakeRebuild()

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        fake_adapter(),
        fake_router(),
        rebuild=rebuild,
        approved_rebuilds=1,
    )

    assert scorer.collections[0] == "live"
    assert rebuild.calls == [(plan.estimates[0].collection, 384)]
    assert outcome.rebuilt == 1


async def test_a_candidate_is_scored_against_its_own_scratch_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(0.5, 0.9)
    monkeypatch.setattr(module, "score_collection", scorer)
    plan = planned(tmp_path, sweepable(), candidates=[Candidate({"chunking.chunk_size": 384})])

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        fake_adapter(),
        fake_router(),
        rebuild=FakeRebuild(),
        approved_rebuilds=1,
    )

    assert scorer.collections[1] == plan.estimates[0].collection
    assert scorer.chunk_sizes[1] == 384
    assert outcome.suggestion.best.candidate.overrides == {"chunking.chunk_size": 384}


async def test_a_sweep_drops_every_scratch_collection_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.6, 0.7))
    plan = planned(
        tmp_path,
        sweepable(),
        candidates=[
            Candidate({"chunking.chunk_size": 384}),
            Candidate({"chunking.overlap": 0}),
        ],
    )
    adapter = fake_adapter()

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        adapter,
        fake_router(),
        rebuild=FakeRebuild(),
        approved_rebuilds=2,
    )

    assert adapter.drops == [estimate.collection for estimate in plan.estimates]
    assert outcome.leaked == []


async def test_a_failed_rebuild_still_drops_the_collection_it_half_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is recorded before the build, so wreckage is cleaned up, not abandoned."""
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.6, 0.7))
    plan = planned(
        tmp_path,
        sweepable(),
        candidates=[
            Candidate({"chunking.chunk_size": 384}),
            Candidate({"chunking.overlap": 0}),
        ],
    )
    adapter = fake_adapter()
    doomed = plan.estimates[1].collection

    with pytest.raises(FasterRagError):
        await sweep_index_time(
            plan,
            sweep_dataset(tmp_path),
            sweepable(),
            adapter,
            fake_router(),
            rebuild=FakeRebuild(fail_on=doomed),
            approved_rebuilds=2,
        )

    assert adapter.drops == [plan.estimates[0].collection, doomed]


async def test_a_drop_that_fails_is_reported_as_a_leak_and_never_stops_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.6, 0.7))
    plan = planned(
        tmp_path,
        sweepable(),
        candidates=[
            Candidate({"chunking.chunk_size": 384}),
            Candidate({"chunking.overlap": 0}),
        ],
    )
    stuck = plan.estimates[0].collection
    adapter = fake_adapter(stuck)

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        adapter,
        fake_router(),
        rebuild=FakeRebuild(),
        approved_rebuilds=2,
    )

    assert outcome.leaked == [stuck]
    assert adapter.drops == [estimate.collection for estimate in plan.estimates]
    assert stuck in outcome.as_dict()["leaked_collections"]


async def test_an_embedding_candidate_is_refused_without_a_router_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Querying a new model with the old router would measure the mismatch, not quality."""
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.6))
    plan = planned(
        tmp_path,
        sweepable(),
        candidates=[Candidate({"embeddings.model": "text-embedding-3-small"})],
    )
    rebuild = FakeRebuild()

    with pytest.raises(FasterRagError) as failure:
        await sweep_index_time(
            plan,
            sweep_dataset(tmp_path),
            sweepable(),
            fake_adapter(),
            fake_router(),
            rebuild=rebuild,
            approved_rebuilds=1,
        )

    assert failure.value.code is ErrorCode.VALIDATION_FAILED
    assert rebuild.calls == []


async def test_a_candidate_router_is_used_for_its_trial_and_then_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = FakeScorer(0.5, 0.6)
    monkeypatch.setattr(module, "score_collection", scorer)
    plan = planned(
        tmp_path,
        sweepable(),
        candidates=[Candidate({"embeddings.model": "text-embedding-3-small"})],
    )
    base = fake_router("base")
    built: list[Any] = []

    def factory(candidate_settings: Settings) -> Any:
        router = fake_router(candidate_settings.embeddings.model)
        built.append(router)
        return router

    await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        fake_adapter(),
        base,
        rebuild=FakeRebuild(),
        approved_rebuilds=1,
        router_factory=factory,
    )

    assert scorer.routers[0] is base
    assert scorer.routers[1] is built[0]
    assert built[0].closed is True
    assert base.closed is False


async def test_the_budget_stops_the_sweep_before_a_rebuild_and_reports_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget that stopped only after paying for a rebuild would not be a budget."""
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.6, 0.7))
    plan = planned(tmp_path, sweepable())
    rebuild = FakeRebuild()

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        fake_adapter(),
        fake_router(),
        rebuild=rebuild,
        approved_rebuilds=plan.rebuilds,
        budget_seconds=0.0,
    )

    assert rebuild.calls == []
    assert outcome.rebuilt == 0
    assert outcome.suggestion.evaluated == 1
    assert outcome.suggestion.skipped == plan.rebuilds


async def test_the_outcome_states_what_it_built_and_that_nothing_was_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "score_collection", FakeScorer(0.5, 0.9))
    plan = planned(tmp_path, sweepable(), candidates=[Candidate({"chunking.chunk_size": 384})])

    outcome = await sweep_index_time(
        plan,
        sweep_dataset(tmp_path),
        sweepable(),
        fake_adapter(),
        fake_router(),
        rebuild=FakeRebuild(),
        approved_rebuilds=1,
    )
    payload = outcome.as_dict()

    assert payload["applied"] is False
    assert "never writes config.yaml" in payload["note"]
    assert "needs a reindex" in payload["note"]
    assert payload["scratch_collections"] == [plan.estimates[0].collection]
    assert payload["plan"]["rebuilds"] == 1


def test_an_index_time_suggestion_says_a_reindex_is_still_needed() -> None:
    """Pasting a chunk size into config.yaml does not re-chunk anything by itself."""
    best = TrialResult(
        candidate=Candidate({"chunking.chunk_size": 384}),
        recall_at_k=0.9,
        mrr=0.9,
        ndcg_at_k=0.9,
        k=5,
    )

    rendered = render_suggestion(Suggestion(baseline=trial("", 0.5), best=best, evaluated=2))

    assert "index reembed" in rendered
    assert "chunk_size: 384" in rendered


def test_a_query_time_suggestion_does_not_tell_anyone_to_reindex() -> None:
    rendered = render_suggestion(
        Suggestion(baseline=trial("", 0.5), best=trial("x", 0.9), evaluated=2)
    )

    assert "index reembed" not in rendered


def test_the_rendered_plan_leads_with_the_rebuilds_and_the_cost(tmp_path: Path) -> None:
    rendered = render_plan(planned(tmp_path, sweepable()))

    assert "NOTHING HAS BEEN BUILT YET" in rendered
    assert "rebuilds" in rendered
    assert "cost" in rendered


def test_the_rendered_plan_names_every_scratch_collection_it_would_create(
    tmp_path: Path,
) -> None:
    """An operator who has to find leftovers after a crash needs the names in advance."""
    plan = planned(tmp_path, sweepable())

    rendered = render_plan(plan)

    for estimate in plan.estimates:
        assert estimate.collection in rendered
