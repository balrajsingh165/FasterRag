"""The ``replay`` wrapper (D8): refusals, rendering choices, and cleanup."""

import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import replay as replay_command
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.core.tracing import Trace
from fasterrag.errors import FasterRagError
from fasterrag.services.replay import ReplayResult, RetrievalDiff
from tests.unit.cli.conftest import Closeable, write_config

TRACE_ID = "c" * 32


class FakeRouter(Closeable):
    """A tiering router the wrapper only builds, hands on, and closes."""

    default = object()


class FakeStore:
    """A trace store returning whatever the test put in it."""

    def __init__(self, trace: Trace | None) -> None:
        self.trace = trace
        self.asked: list[str] = []

    def load(self, trace_id: str) -> Trace | None:
        self.asked.append(trace_id)
        return self.trace


class Harness:
    """Holds the doubles a replay run needs, plus what came back."""

    def __init__(self) -> None:
        self.adapter = Closeable()
        self.router = FakeRouter()
        self.service = Closeable()
        self.result = ReplayResult(trace_id=TRACE_ID, query="what is the notice period?")
        self.error: Exception | None = None
        self.store = FakeStore(Trace(trace_id=TRACE_ID, query="what is the notice period?"))


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    built = Harness()

    async def fake_replay(trace: Trace, candidate: object, service: object) -> ReplayResult:
        if built.error is not None:
            raise built.error
        return built.result

    monkeypatch.setattr(replay_command, "create_trace_store", lambda settings: built.store)
    monkeypatch.setattr(replay_command, "create_vector_db_adapter", lambda settings: built.adapter)
    monkeypatch.setattr(replay_command, "create_embedding_router", lambda settings: built.router)
    monkeypatch.setattr(replay_command, "create_llm_adapter", lambda settings: object())
    monkeypatch.setattr(replay_command, "RetrievalService", lambda *args, **kwargs: object())
    monkeypatch.setattr(replay_command, "GenerationService", lambda *args, **kwargs: built.service)
    monkeypatch.setattr(replay_command, "replay_trace", fake_replay)
    return built


def test_replay_is_refused_when_the_feature_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(tmp_path, "traces:\n  replay: false\n")

    code = main(["replay", "--config", config, "--trace", TRACE_ID])

    assert code == ExitCode.USAGE
    assert "traces.replay" in capsys.readouterr().err


def test_an_invalid_candidate_config_is_refused_before_anything_runs(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config that turns out to be invalid mid-run leaves the diff unattributable."""
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")

    code = main(["replay", "--config", config, "--trace", TRACE_ID, "--candidate", str(candidate)])

    assert code == ExitCode.USAGE
    assert harness.store.asked == []


def test_an_absent_trace_exits_one(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.store.trace = None

    code = main(["replay", "--config", config, "--trace", TRACE_ID])

    assert code == ExitCode.FAILURE
    assert "no stored trace" in capsys.readouterr().err


def test_an_unchanged_replay_says_so_rather_than_listing_every_chunk(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["replay", "--config", config, "--trace", TRACE_ID])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "config          unchanged" in out
    assert "retrieval       identical" in out
    assert "answer          unchanged" in out


def test_a_changed_retrieval_set_is_printed_as_a_diff(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.result = ReplayResult(
        trace_id=TRACE_ID,
        query="q",
        config_changes=[{"key": "retrieval.top_k", "was": 5, "now": 8}],
        retrieval=RetrievalDiff(
            added=["new-chunk"],
            removed=["old-chunk"],
            reordered=[{"chunk_id": "moved", "was": 1, "now": 3}],
        ),
    )

    main(["replay", "--config", config, "--trace", TRACE_ID])

    out = capsys.readouterr().out
    assert "retrieval.top_k: 5 -> 8" in out
    assert "+ new-chunk" in out
    assert "- old-chunk" in out
    assert "~ moved: rank 1 -> 3" in out


def test_a_changed_answer_is_printed_in_full_by_default(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.result = ReplayResult(
        trace_id=TRACE_ID,
        query="q",
        original_answer="thirty days",
        replayed_answer="forty-five days",
        original_citations=["c1"],
        replayed_citations=["c2"],
    )

    main(["replay", "--config", config, "--trace", TRACE_ID])

    out = capsys.readouterr().out
    assert "was: thirty days" in out
    assert "now: forty-five days" in out


def test_diff_only_keeps_the_citations_and_drops_the_answer_text(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.result = ReplayResult(
        trace_id=TRACE_ID,
        query="q",
        original_answer="thirty days",
        replayed_answer="forty-five days",
        original_citations=["c1"],
        replayed_citations=["c2"],
    )

    main(["replay", "--config", config, "--trace", TRACE_ID, "--diff-only"])

    out = capsys.readouterr().out
    assert "thirty days" not in out
    assert "citations was: c1" in out


def test_replay_as_json_is_one_document_with_no_prose(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["replay", "--config", config, "--trace", TRACE_ID, "--json"])

    out = capsys.readouterr().out
    assert json.loads(out)["trace_id"] == TRACE_ID
    assert "retrieval       identical" not in out


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_replay_maps_the_error_to_its_documented_code(
    config: str, harness: Harness, retryable: bool, expected: ExitCode
) -> None:
    harness.error = FasterRagError("the backend is not answering", retryable=retryable)

    assert main(["replay", "--config", config, "--trace", TRACE_ID]) == expected


def test_a_failed_replay_still_closes_everything_it_built(config: str, harness: Harness) -> None:
    harness.error = FasterRagError("boom", retryable=False)

    main(["replay", "--config", config, "--trace", TRACE_ID])

    assert (harness.service.closed, harness.router.closed, harness.adapter.closed) == (1, 1, 1)


def test_a_valid_candidate_config_replaces_the_running_one(
    config: str, tmp_path: Path, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate is what gets built against; the running config only supplies the trace."""
    seen: list[Any] = []

    def record(settings: Any) -> Closeable:
        seen.append(settings)
        return harness.adapter

    monkeypatch.setattr(replay_command, "create_vector_db_adapter", record)
    candidate = write_config(tmp_path / "candidate", "retrieval:\n  top_k: 9\n")

    code = main(["replay", "--config", config, "--trace", TRACE_ID, "--candidate", candidate])

    assert code == ExitCode.SUCCESS
    assert seen[0].retrieval.top_k == 9


def test_an_invalid_running_config_is_a_usage_error(bad_config: str) -> None:
    assert main(["replay", "--config", bad_config, "--trace", TRACE_ID]) == ExitCode.USAGE
