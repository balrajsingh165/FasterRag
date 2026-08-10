"""The ``benchmark`` wrapper: which suite, which exit code, and what it refuses to measure."""

import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import benchmark as benchmark_command
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.core.evals.harness import EvalReport
from fasterrag.errors import FasterRagError
from fasterrag.services.benchmark import Fingerprint
from fasterrag.services.estimation import estimate_sources
from fasterrag.services.regression import GateResult
from tests.unit.cli.conftest import Closeable, corpus

MACHINE = Fingerprint(
    cpu="Test CPU",
    cores=8,
    ram_gb=32.0,
    gpu="none",
    storage_gb=500.0,
    os="test-os",
    python="3.14.0",
    fasterrag="0.1.0",
)


class FakeService(Closeable):
    """A generation service the query suite calls and must first disarm."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: object | None = object()
        self.questions: list[str] = []
        self.error: Exception | None = None

    async def answer(self, question: str, **kwargs: Any) -> object:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return object()


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the hardware fingerprint off the critical path of every test."""
    monkeypatch.setattr(benchmark_command, "fingerprint", lambda: MACHINE)


@pytest.fixture
def query_service(monkeypatch: pytest.MonkeyPatch) -> FakeService:
    service = FakeService()
    monkeypatch.setattr(benchmark_command, "create_vector_db_adapter", lambda s: Closeable())
    monkeypatch.setattr(benchmark_command, "_build_generation", lambda s, adapter: service)
    return service


class Gate:
    """Controls the verdict the eval suite gets back."""

    def __init__(self) -> None:
        self.passed = True
        self.error: Exception | None = None
        self.collections: list[str] = []


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Gate:
    controller = Gate()
    report = EvalReport(k=5, scored=10, adversarial=2, recall_at_k=0.9, mrr=0.8, ndcg_at_k=0.85)

    async def fake_run_eval(
        root: Path, settings: object, adapter: object, router: object, *, collection: str
    ) -> tuple[EvalReport, GateResult]:
        controller.collections.append(collection)
        if controller.error is not None:
            raise controller.error
        return report, GateResult(passed=controller.passed, failures=["recall fell 0.05"])

    monkeypatch.setattr(benchmark_command, "create_vector_db_adapter", lambda s: Closeable())
    monkeypatch.setattr(benchmark_command, "create_embedding_router", lambda s: Closeable())
    monkeypatch.setattr(benchmark_command, "run_eval", fake_run_eval)
    return controller


def test_the_ingest_suite_needs_a_corpus(config: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--config", config, "--suite", "ingest"])

    assert code == ExitCode.USAGE
    assert "--sources" in capsys.readouterr().err


def test_the_ingest_suite_measures_the_files_inside_a_directory(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory reached the estimator raw, so the suite timed one failed read."""
    source = corpus(tmp_path / "corpus")

    code = main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(source),
            "--iterations",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == ExitCode.SUCCESS
    assert payload["suites"][0]["dataset"].startswith("2 documents")


def test_the_timed_work_is_the_expanded_corpus_and_not_the_directory(
    config: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timed loop must parse the same files the report counts.

    Expanding only for the report would leave the throughput measured over one failed
    directory read while the entry beside it claimed two documents — and the throughput is
    the number a ``--ledger`` entry publishes.
    """
    source = corpus(tmp_path / "corpus")
    seen: list[list[str]] = []

    def spy(sources: Any, settings: Any, **kwargs: Any) -> Any:
        seen.append([str(item) for item in sources])
        return estimate_sources(sources, settings, **kwargs)

    monkeypatch.setattr(benchmark_command, "estimate_sources", spy)

    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(source),
            "--iterations",
            "1",
        ]
    )

    assert seen, "the estimator was never called, so this asserts nothing"
    assert all(sorted(Path(name).name for name in call) == ["a.txt", "b.txt"] for call in seen)


def test_a_corpus_that_expands_to_nothing_is_refused_rather_than_measured(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ledger entry for zero documents is a published number backed by nothing."""
    empty = tmp_path / "empty"
    empty.mkdir()

    code = main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(empty),
            "--ledger",
        ]
    )

    out = capsys.readouterr()
    assert code == ExitCode.USAGE
    assert "no readable files" in out.err
    assert "ledger entries" not in out.out


def test_the_ledger_flag_emits_a_pasteable_entry(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(corpus(tmp_path / "corpus")),
            "--iterations",
            "1",
            "--ledger",
        ]
    )

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "BENCH-0001" in out
    assert "Test CPU" in out


def test_without_the_ledger_flag_no_entry_is_printed(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(corpus(tmp_path / "corpus")),
            "--iterations",
            "1",
        ]
    )

    assert "BENCH-0001" not in capsys.readouterr().out


def test_the_dataset_flag_names_the_run_in_the_report(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "ingest",
            "--sources",
            str(corpus(tmp_path / "corpus")),
            "--iterations",
            "1",
            "--dataset",
            "handbook-v2",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["suites"][0]["dataset"] == "handbook-v2"


def test_the_query_suite_disables_the_semantic_cache(
    config: str, query_service: FakeService
) -> None:
    """The suite repeats one question, so a live cache would report hit latency as query latency."""
    main(["benchmark", "--config", config, "--suite", "query", "--iterations", "1"])

    assert query_service.cache is None


def test_the_query_flag_is_the_question_that_gets_asked(
    config: str, query_service: FakeService
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "query",
            "--iterations",
            "1",
            "--query",
            "what is the notice period?",
        ]
    )

    assert set(query_service.questions) == {"what is the notice period?"}


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_query_suite_maps_the_error_to_its_documented_code(
    config: str, query_service: FakeService, retryable: bool, expected: ExitCode
) -> None:
    query_service.error = FasterRagError("qdrant is not answering", retryable=retryable)

    code = main(["benchmark", "--config", config, "--suite", "query", "--iterations", "1"])

    assert code == expected


def test_all_runs_both_latency_suites(
    config: str, tmp_path: Path, query_service: FakeService, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--sources",
            str(corpus(tmp_path / "corpus")),
            "--iterations",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert [suite["suite"] for suite in payload["suites"]] == ["query", "ingest"]


def test_a_failing_suite_stops_the_run_rather_than_reporting_a_partial_set(
    config: str, query_service: FakeService, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["benchmark", "--config", config, "--iterations", "1"])

    assert code == ExitCode.USAGE
    assert "suites" not in capsys.readouterr().out


def test_the_eval_suite_needs_a_dataset(config: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "--config", config, "--suite", "eval"])

    assert code == ExitCode.USAGE
    assert "--dataset" in capsys.readouterr().err


def test_a_passing_gate_exits_zero(
    config: str, tmp_path: Path, gate: Gate, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["benchmark", "--config", config, "--suite", "eval", "--dataset", str(tmp_path)])

    assert code == ExitCode.SUCCESS
    assert "recall@5        0.9000" in capsys.readouterr().out


def test_a_blocked_gate_exits_five(config: str, tmp_path: Path, gate: Gate) -> None:
    """Exit 5 is what lets CI tell a blocked gate from a crashed run."""
    gate.passed = False

    code = main(["benchmark", "--config", config, "--suite", "eval", "--dataset", str(tmp_path)])

    assert code == ExitCode.REGRESSION


def test_the_eval_suite_scores_the_named_collection(
    config: str, tmp_path: Path, gate: Gate
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "eval",
            "--dataset",
            str(tmp_path),
            "--collection",
            "legal",
        ]
    )

    assert gate.collections == ["legal"]


def test_the_eval_suite_falls_back_to_the_configured_collection(
    config: str, tmp_path: Path, gate: Gate
) -> None:
    main(["benchmark", "--config", config, "--suite", "eval", "--dataset", str(tmp_path)])

    assert gate.collections == ["default"]


def test_the_eval_suite_as_json_is_one_document(
    config: str, tmp_path: Path, gate: Gate, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "benchmark",
            "--config",
            config,
            "--suite",
            "eval",
            "--dataset",
            str(tmp_path),
            "--json",
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["eval"]["ndcg_at_k"] == 0.85
    assert payload["gate"]["passed"] is True
    assert "recall@5" not in out


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_eval_suite_maps_the_error_to_its_documented_code(
    config: str, tmp_path: Path, gate: Gate, retryable: bool, expected: ExitCode
) -> None:
    gate.error = FasterRagError("qdrant is not answering", retryable=retryable)

    code = main(["benchmark", "--config", config, "--suite", "eval", "--dataset", str(tmp_path)])

    assert code == expected


def test_the_eval_suite_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    code = main(
        ["benchmark", "--config", bad_config, "--suite", "eval", "--dataset", str(tmp_path)]
    )

    assert code == ExitCode.USAGE


def test_the_ingest_suite_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    code = main(
        ["benchmark", "--config", bad_config, "--suite", "ingest", "--sources", str(tmp_path)]
    )

    assert code == ExitCode.USAGE
