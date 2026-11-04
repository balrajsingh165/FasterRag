import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.evals.harness import EvalReport
from fasterrag.core.identity import retrieval_config_hash
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.regression import (
    COMPARABILITY_VERSION,
    Baseline,
    Verdict,
    check_regression,
    comparability_fingerprint,
    load_baseline,
    write_baseline,
)


def settings(*, gate: bool = True, **overrides: Any) -> Settings:
    payload: dict[str, Any] = {"eval": {"regression_gate": gate}}
    payload.update(overrides)
    return Settings.model_validate(payload)


def report(recall: float = 0.9, ndcg: float = 0.85, k: int = 10, mrr: float = 0.8) -> EvalReport:
    return EvalReport(k=k, scored=20, adversarial=0, recall_at_k=recall, mrr=mrr, ndcg_at_k=ndcg)


BOUNDARY_TOLERANCE: float = 0.25
BOUNDARY_DROPS: list[float] = [0.21875, 0.25, 0.28125]
BOUNDARY_IDS: list[str] = ["just-inside", "exactly-on", "just-beyond"]


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
        fingerprint_version=COMPARABILITY_VERSION,
        fingerprint=comparability_fingerprint(configured),
    )


def test_a_disabled_gate_never_blocks() -> None:
    configured = settings(gate=False)

    result = check_regression(report(recall=0.1), None, configured)

    assert result.passed is True


def test_a_disabled_gate_reports_that_it_established_nothing() -> None:
    """Passing and having compared something are different claims, and CI reads the second."""
    result = check_regression(report(), None, settings(gate=False))

    assert result.verdict is Verdict.NOT_COMPARABLE
    assert any("eval.regression_gate is false" in reason for reason in result.obstacles)


def test_identical_results_pass() -> None:
    configured = settings()

    result = check_regression(report(), baseline_for(configured), configured)

    assert result.passed is True
    assert result.verdict is Verdict.WITHIN_TOLERANCE
    assert result.failures == []
    assert result.obstacles == []
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
    assert result.verdict is Verdict.REGRESSED
    assert any("recall@k fell from 0.9000 to 0.8000" in message for message in result.failures)
    assert "REGRESSED" in result.summary()


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


@pytest.mark.parametrize("drop", BOUNDARY_DROPS, ids=BOUNDARY_IDS)
def test_the_tolerance_boundary_cases_are_exact_in_binary(drop: float) -> None:
    """A case only lands on the boundary when the subtraction behind it is exact."""
    assert 1.0 - (1.0 - drop) == drop


@pytest.mark.parametrize(
    ("drop", "blocks"),
    list(zip(BOUNDARY_DROPS, [False, False, True], strict=True)),
    ids=BOUNDARY_IDS,
)
def test_the_recall_tolerance_is_the_largest_drop_that_still_passes(
    drop: float, blocks: bool
) -> None:
    """``recall_tolerance`` is the maximum allowed drop, so a drop equal to it still passes."""
    configured = settings(eval={"regression_gate": True, "recall_tolerance": BOUNDARY_TOLERANCE})
    base = baseline_for(configured, recall=1.0)

    result = check_regression(report(recall=1.0 - drop), base, configured)

    assert result.passed is (not blocks)
    assert (result.verdict is Verdict.REGRESSED) is blocks


@pytest.mark.parametrize(
    ("drop", "blocks"),
    list(zip(BOUNDARY_DROPS, [False, False, True], strict=True)),
    ids=BOUNDARY_IDS,
)
def test_the_ndcg_tolerance_is_the_largest_drop_that_still_passes(
    drop: float, blocks: bool
) -> None:
    """The second metric reads its own config key, so its boundary is proven separately."""
    configured = settings(eval={"regression_gate": True, "ndcg_tolerance": BOUNDARY_TOLERANCE})
    base = baseline_for(configured, ndcg=1.0)

    result = check_regression(report(ndcg=1.0 - drop), base, configured)

    assert result.passed is (not blocks)
    assert (result.verdict is Verdict.REGRESSED) is blocks


def test_a_missing_baseline_blocks_rather_than_passing_vacuously() -> None:
    result = check_regression(report(), None, settings())

    assert result.passed is False
    assert result.verdict is Verdict.NOT_COMPARABLE
    assert any("no baseline is committed" in message for message in result.obstacles)


def test_a_missing_baseline_is_not_reported_as_a_regression() -> None:
    """The distinction CI acts on: nothing was measured against, nothing regressed."""
    result = check_regression(report(), None, settings())

    assert result.failures == []
    assert "NOT COMPARABLE" in result.summary()


def test_a_baseline_from_another_embedding_model_is_refused() -> None:
    configured = settings()
    stale = baseline_for(settings(embeddings={"model": "some-other-model"}))

    result = check_regression(report(), stale, configured)

    assert result.passed is False
    assert result.verdict is Verdict.NOT_COMPARABLE
    assert any("some-other-model" in message for message in result.obstacles)


def test_a_baseline_from_another_retrieval_config_is_refused() -> None:
    configured = settings()
    changed = settings(chunking={"chunk_size": 256})

    result = check_regression(report(), baseline_for(changed), configured)

    assert result.passed is False
    assert result.verdict is Verdict.NOT_COMPARABLE
    assert any("chunking.chunk_size was 256 and is 768" in m for m in result.obstacles)


def test_an_unrelated_config_change_does_not_invalidate_a_baseline() -> None:
    configured = settings()
    unrelated = settings(app={"port": 9000})

    assert baseline_for(unrelated).comparable_to(configured) is True


def test_a_baseline_measured_at_another_k_is_refused() -> None:
    configured = settings()

    result = check_regression(report(k=5), baseline_for(configured, k=10), configured)

    assert result.passed is False
    assert result.verdict is Verdict.NOT_COMPARABLE
    assert any("not comparable" in message for message in result.obstacles)


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
    assert written.fingerprint_version == COMPARABILITY_VERSION
    assert written.fingerprint == comparability_fingerprint(configured)
    assert written.recorded_at.endswith("+00:00")


def test_no_baseline_file_loads_as_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "absent.json") is None


def test_a_fingerprint_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    """A hand-edited fingerprint that is not field-to-value would compare nothing silently."""
    broken = tmp_path / "baseline.json"
    broken.write_text(
        '{"recorded_at": "x", "k": 5, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, '
        '"scored": 1, "embedding_model": "m", "config_hash": "h", "fingerprint": ["a"]}',
        encoding="utf-8",
    )

    with pytest.raises(FasterRagError):
        load_baseline(broken)


def test_the_result_serializes_for_ci() -> None:
    configured = settings()
    payload = check_regression(report(recall=0.5), baseline_for(configured), configured).as_dict()

    assert payload["passed"] is False
    assert payload["verdict"] == "regressed"
    assert payload["failures"]
    assert payload["obstacles"] == []
    assert payload["report"]["recall_at_k"] == 0.5
    assert payload["baseline"]["embedding_model"] == configured.embeddings.model


def test_an_uncomparable_result_serializes_as_its_own_verdict() -> None:
    """CI must never read "could not compare" off the same field as "nothing regressed"."""
    payload = check_regression(report(), None, settings()).as_dict()

    assert payload["verdict"] == "not-comparable"
    assert payload["failures"] == []
    assert payload["obstacles"]


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


@pytest.mark.parametrize("attribute", ["recall_at_k", "embedding_model", "fingerprint_version"])
def test_a_baseline_cannot_be_edited_after_it_is_recorded(attribute: str) -> None:
    """A record a caller can rewrite in place is no longer a record of what was measured."""
    base = baseline_for(settings())

    with pytest.raises(FrozenInstanceError):
        setattr(base, attribute, "rewritten")


@pytest.mark.parametrize("attribute", ["passed", "failures", "obstacles"])
def test_a_gate_result_cannot_be_edited_after_the_gate_decided(attribute: str) -> None:
    """A verdict a caller can flip to passing after the fact is not a gate."""
    configured = settings()
    result = check_regression(report(recall=0.5), baseline_for(configured), configured)

    with pytest.raises(FrozenInstanceError):
        setattr(result, attribute, True)


def test_a_recorded_baseline_is_rounded_to_six_decimals(tmp_path: Path) -> None:
    """The file is committed and read as a diff, so a float's full tail is noise not signal."""
    measured = report(recall=0.123456789, ndcg=0.555555555, mrr=0.987654321)

    written = write_baseline(tmp_path / "baseline.json", measured, settings())

    assert written.recall_at_k == 0.123457
    assert written.mrr == 0.987654
    assert written.ndcg_at_k == 0.555556


def test_a_written_baseline_is_in_the_committed_canonical_form(tmp_path: Path) -> None:
    """Sorted keys and a two-space indent keep a re-recording from rewriting every line."""
    path = tmp_path / "baseline.json"

    write_baseline(path, report(), settings())
    text = path.read_text(encoding="utf-8")
    document = json.loads(text)

    assert list(document) == sorted(document)
    assert list(document["fingerprint"]) == sorted(document["fingerprint"])
    assert text.startswith('{\n  "' + sorted(document)[0] + '"')
    assert text.endswith("}\n")


def test_a_baseline_is_written_into_a_directory_that_does_not_exist_yet(tmp_path: Path) -> None:
    """A gate records into a dataset directory CI may be creating for the first time."""
    path = tmp_path / "datasets" / "handbook" / "baseline.json"

    written = write_baseline(path, report(), settings())

    assert load_baseline(path) == written


def test_a_fingerprint_recording_a_bare_number_is_refused(tmp_path: Path) -> None:
    """A number never equals the text a field renders as, so it would mismatch forever."""
    broken = tmp_path / "baseline.json"
    broken.write_text(
        '{"recorded_at": "x", "k": 5, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, '
        '"scored": 1, "embedding_model": "m", "config_hash": "h", '
        '"fingerprint": {"retrieval.top_k": 10}}',
        encoding="utf-8",
    )

    with pytest.raises(FasterRagError):
        load_baseline(broken)


def test_a_corrupt_baseline_is_not_retryable(tmp_path: Path) -> None:
    """Reading the same unparseable file again cannot succeed, so no caller may retry it."""
    broken = tmp_path / "baseline.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(FasterRagError) as caught:
        load_baseline(broken)

    assert caught.value.retryable is False
    assert caught.value.code is ErrorCode.VALIDATION_FAILED
