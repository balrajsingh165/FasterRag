"""The ``traces`` wrapper: which exit code, and one JSON document."""

import json
from pathlib import Path

import pytest

from fasterrag.cli.commands import traces
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.core.tracing import Span, Trace
from fasterrag.services.traces import TraceStore
from tests.unit.cli.conftest import write_config


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TraceStore:
    """Point the command at a store rooted in the test's own directory."""
    built = TraceStore(tmp_path / "traces")
    monkeypatch.setattr(traces, "create_trace_store", lambda settings: built)
    return built


def a_trace(trace_id: str = "a" * 32) -> Trace:
    return Trace(
        trace_id=trace_id,
        query="what is the notice period?",
        collection="docs",
        created_at="2026-08-09T00:00:00Z",
        retrieved=[{"chunk_id": "c1"}],
        spans=[Span(name="retrieval", start_ms=0.0, end_ms=12.5, attributes={"k": 5})],
    )


def test_an_empty_store_is_reported_rather_than_treated_as_a_failure(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["traces", "--config", config, "list"])

    assert code == ExitCode.SUCCESS
    assert "no stored traces" in capsys.readouterr().out


def test_an_empty_store_names_the_setting_when_storage_is_off(
    tmp_path: Path,
    store: TraceStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Otherwise "no stored traces" reads as "none yet" on a deployment storing none, ever."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(tmp_path, "traces:\n  store: false\n")

    main(["traces", "--config", config, "list"])

    assert "traces.store is false" in capsys.readouterr().out


def test_listing_prints_each_id(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.store(a_trace())

    code = main(["traces", "--config", config, "list"])

    assert code == ExitCode.SUCCESS
    assert "a" * 32 in capsys.readouterr().out


def test_the_limit_flag_reaches_the_store(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    for index in range(3):
        store.store(a_trace(f"{index:032d}"))

    main(["traces", "--config", config, "list", "--limit", "1", "--json"])

    assert len(json.loads(capsys.readouterr().out)["traces"]) == 1


def test_listing_as_json_is_one_document_with_no_prose(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["traces", "--config", config, "list", "--json"])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert json.loads(out) == {"traces": []}
    assert "no stored traces" not in out


def test_showing_a_stored_trace_prints_its_spans(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.store(a_trace())

    code = main(["traces", "--config", config, "show", "a" * 32])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "what is the notice period?" in out
    assert "retrieval" in out


def test_showing_a_stored_trace_as_json_is_one_document(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.store(a_trace())

    main(["traces", "--config", config, "show", "a" * 32, "--json"])

    assert json.loads(capsys.readouterr().out)["trace_id"] == "a" * 32


def test_an_absent_trace_exits_one_and_names_the_retention_window(
    config: str, store: TraceStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing trace is a failure, not a usage error: the command was written correctly."""
    code = main(["traces", "--config", config, "show", "b" * 32])

    err = capsys.readouterr().err
    assert code == ExitCode.FAILURE
    assert "retention_days" in err


def test_an_invalid_config_is_a_usage_error(bad_config: str) -> None:
    assert main(["traces", "--config", bad_config, "list"]) == ExitCode.USAGE
