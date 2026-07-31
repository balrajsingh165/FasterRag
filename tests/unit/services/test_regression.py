from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.evals.harness import EvalReport
from fasterrag.core.identity import retrieval_config_hash
from fasterrag.services.regression import (
    Baseline,
    check_regression,
    load_baseline,
    write_baseline,
)


def settings(*, gate: bool = True, **overrides: Any) -> Settings:
    payload: dict[str, Any] = {"eval": {"regression_gate": gate}}
    payload.update(overrides)
    return Settings.model_validate(payload)


def report(recall: float = 0.9, ndcg: float = 0.85, k: int = 10) -> EvalReport:
    return EvalReport(k=k, scored=20, adversarial=0, recall_at_k=recall, mrr=0.8, ndcg_at_k=ndcg)


def baseline_for(
    configured: Settings, *, recall: float = 0.9, ndcg: float = 0.85, k: int = 10
) -> Baseline:
    return Baseline(
        recorded_at="2026-07-30T00:00:00+00:00",
        k=k,
        recall_at_k=recall,
        mrr=0.8,
        ndcg_at_k=ndcg,
        scored=20,
        embedding_model=configured.embeddings.model,
        config_hash=retrieval_config_hash(configured),
    )


def test_a_disabled_gate_never_blocks() -> None:
    configured = settings(gate=False)

    result = check_regression(report(recall=0.1), None, configured)

    assert result.passed is True


def test_identical_results_pass() -> None:
    configured = settings()

    result = check_regression(report(), baseline_for(configured), configured)

    assert result.passed is True
    assert result.failures == []
    assert "passed" in result.summary()


def test_an_improvement_never_blocks() -> None:
    configured = settings()
    base = baseline_for(configured, recall=0.7, ndcg=0.6)

    result = check_regression(report(recall=0.95, ndcg=0.92), base, configured)

    assert result.passed is True


def test_a_drop_inside_tolerance_passes() -> None:
    configured = settings()
    base = baseline_for(configured, recall=0.90)

    result = check_regression(report(recall=0.885), base, configured)

    assert result.passed is True


def test_a_recall_drop_beyond_tolerance_blocks() -> None:
    configured = settings()
    base = baseline_for(configured, recall=0.90)

    result = check_regression(report(recall=0.80), base, configured)

    assert result.passed is False
    assert any("recall@k fell from 0.9000 to 0.8000" in message for message in result.failures)
    assert "BLOCKED" in result.summary()


def test_an_ndcg_drop_beyond_tolerance_blocks() -> None:
    configured = settings()
    base = baseline_for(configured, ndcg=0.85)

    result = check_regression(report(ndcg=0.70), base, configured)

    assert result.passed is False
    assert any("nDCG@k" in message for message in result.failures)


def test_both_metrics_are_reported_when_both_regress() -> None:
    configured = settings()
    base = baseline_for(configured, recall=0.9, ndcg=0.9)

    result = check_regression(report(recall=0.5, ndcg=0.4), base, configured)

    assert len(result.failures) == 2


def test_a_configured_tolerance_is_honored() -> None:
    lenient = settings(eval={"regression_gate": True, "recall_tolerance": 0.25})
    base = baseline_for(lenient, recall=0.90)

    assert check_regression(report(recall=0.70), base, lenient).passed is True


def test_a_missing_baseline_blocks_rather_than_passing_vacuously() -> None:
    result = check_regression(report(), None, settings())

    assert result.passed is False
    assert any("no baseline is committed" in message for message in result.failures)


def test_a_baseline_from_another_embedding_model_is_refused() -> None:
    configured = settings()
    stale = Baseline(
        recorded_at="2026-07-30T00:00:00+00:00",
        k=10,
        recall_at_k=0.9,
        mrr=0.8,
        ndcg_at_k=0.85,
        scored=20,
        embedding_model="some-other-model",
        config_hash=retrieval_config_hash(configured),
    )

    result = check_regression(report(), stale, configured)

    assert result.passed is False
    assert any("different embedding model" in message for message in result.failures)


def test_a_baseline_from_another_retrieval_config_is_refused() -> None:
    configured = settings()
    changed = settings(chunking={"chunk_size": 256})

    result = check_regression(report(), baseline_for(changed), configured)

    assert result.passed is False
    assert any("retrieval\nconfiguration" in m or "configuration" in m for m in result.failures)


def test_an_unrelated_config_change_does_not_invalidate_a_baseline() -> None:
    configured = settings()
    unrelated = settings(app={"port": 9000})

    assert baseline_for(unrelated).comparable_to(configured) is True


def test_a_baseline_measured_at_another_k_is_refused() -> None:
    configured = settings()

    result = check_regression(report(k=5), baseline_for(configured, k=10), configured)

    assert result.passed is False
    assert any("not comparable" in message for message in result.failures)


def test_a_baseline_round_trips_through_its_file(tmp_path: Path) -> None:
    configured = settings()
    path = tmp_path / "baseline.json"

    written = write_baseline(path, report(), configured)
    loaded = load_baseline(path)

    assert loaded == written
    assert loaded is not None
    assert loaded.comparable_to(configured) is True


def test_a_baseline_records_what_produced_it(tmp_path: Path) -> None:
    configured = settings()

    written = write_baseline(tmp_path / "baseline.json", report(), configured)

    assert written.embedding_model == configured.embeddings.model
    assert written.config_hash == retrieval_config_hash(configured)
    assert written.recorded_at.endswith("+00:00")


def test_no_baseline_file_loads_as_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "absent.json") is None


def test_the_result_serializes_for_ci() -> None:
    configured = settings()
    payload = check_regression(report(recall=0.5), baseline_for(configured), configured).as_dict()

    assert payload["passed"] is False
    assert payload["failures"]
    assert payload["report"]["recall_at_k"] == 0.5
    assert payload["baseline"]["embedding_model"] == configured.embeddings.model


def test_a_freshly_written_baseline_passes_its_own_run(tmp_path: Path) -> None:
    configured = settings()
    measured = report()

    written = write_baseline(tmp_path / "baseline.json", measured, configured)

    assert check_regression(measured, written, configured).passed is True


@pytest.mark.parametrize("drop", [0.021, 0.5, 0.9])
def test_any_drop_beyond_tolerance_blocks(drop: float) -> None:
    configured = settings()
    base = baseline_for(configured, recall=0.95)

    result = check_regression(report(recall=0.95 - drop), base, configured)

    assert result.passed is False
