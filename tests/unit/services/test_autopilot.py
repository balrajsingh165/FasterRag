from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, GoldenRecord
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services import autopilot as module
from fasterrag.services.autopilot import (
    Candidate,
    Suggestion,
    TrialResult,
    candidate_grid,
    render_suggestion,
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
