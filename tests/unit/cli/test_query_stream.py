"""``fasterrag query`` prints what the streamed events say (TASK-0508).

The CLI read token counts from the top of the ``usage`` event, where the service nests them
under ``usage``, read timings from ``meta``, which never carries them, dropped ``truncated``
and ``evidence_dropped``, and ignored an ``error`` event, so a stream the provider broke off
exited 0 with half an answer.
"""

import argparse
from collections.abc import AsyncIterator
from typing import Any

import pytest

from fasterrag.cli.commands import pipeline
from fasterrag.cli.main import main
from fasterrag.cli.output import Console, ExitCode
from fasterrag.services.generation import QueryEvent
from tests.unit.cli.conftest import Closeable

CITATION = {"chunk_id": "c1", "source": "handbook.md", "page": 2, "span": None, "score": 0.9}


def usage(**overrides: Any) -> QueryEvent:
    """A ``usage`` event shaped exactly as ``services.generation._events`` builds it."""
    data: dict[str, Any] = {
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "estimated_cost_usd": 0.0004},
        "timings_ms": {"retrieve": 12, "generate": 340},
        "faithfulness": 0.91,
        "template_version": "p1",
        "mode": "full",
        "degraded": False,
        "truncated": False,
        "evidence_dropped": 0,
    }
    data.update(overrides)
    return QueryEvent(type="usage", data=data)


def meta(mode: str = "full") -> QueryEvent:
    return QueryEvent(
        type="meta",
        data={
            "trace_id": "tr_1",
            "mode": mode,
            "degraded": mode != "full",
            "cache": {"hit": False},
        },
    )


HEALTHY = [
    meta(),
    QueryEvent(type="token", data={"text": "Sourdough rises "}),
    QueryEvent(type="token", data={"text": "because of yeast."}),
    QueryEvent(type="citations", data={"citations": [CITATION]}),
    usage(),
    QueryEvent(type="done", data={}),
]

DEGRADED = [
    meta(),
    QueryEvent(type="meta", data={"trace_id": "tr_1", "mode": "extractive", "degraded": True}),
    QueryEvent(type="token", data={"text": "the retrieved passages"}),
    QueryEvent(type="citations", data={"citations": [CITATION]}),
    usage(mode="extractive", degraded=True, truncated=True, evidence_dropped=3),
    QueryEvent(type="done", data={}),
]

BROKEN_OFF = [
    meta(),
    QueryEvent(type="token", data={"text": "Sourdough ri"}),
    QueryEvent(
        type="error",
        data={
            "code": "GENERATION_FAILED",
            "detail": "the provider closed the stream",
            "trace_id": "tr_1",
            "retryable": False,
            "status": 502,
        },
    ),
]


class ScriptedService(Closeable):
    """A generation service whose stream replays a fixed list of events."""

    def __init__(self, events: list[QueryEvent]) -> None:
        super().__init__()
        self.events = events

    async def stream(self, question: str, **kwargs: Any) -> AsyncIterator[QueryEvent]:
        for event in self.events:
            yield event


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> list[ScriptedService]:
    """Make ``fasterrag query`` stream from whatever service the test appends."""
    built: list[ScriptedService] = []
    monkeypatch.setattr(pipeline, "create_vector_db_adapter", lambda settings: Closeable())
    monkeypatch.setattr(pipeline, "_build_generation", lambda settings, adapter: built[-1])
    return built


def query(config: str, *flags: str) -> int:
    return main(["query", "why does bread rise", "--config", config, *flags])


async def test_the_streamed_answer_carries_the_usage_events_counts_and_timings() -> None:
    """Counts sit under usage.usage and timings in usage; both were read as zero or empty."""
    args = argparse.Namespace(question="q", collection=None, top_k=None)

    answer = await pipeline._streamed(ScriptedService(HEALTHY), args, None, Console(quiet=True))

    assert (answer.prompt_tokens, answer.completion_tokens) == (120, 30)
    assert answer.estimated_cost_usd == 0.0004
    assert answer.timings_ms == {"retrieve": 12, "generate": 340}
    assert answer.faithfulness == 0.91
    assert answer.cache == {"hit": False}
    assert answer.trace_id == "tr_1"
    assert [citation.chunk_id for citation in answer.citations] == ["c1"]


def test_show_timings_prints_the_timings_the_usage_event_reports(
    config: str, scripted: list[ScriptedService], capsys: pytest.CaptureFixture[str]
) -> None:
    scripted.append(ScriptedService(HEALTHY))

    code = query(config, "--show-timings")
    out = capsys.readouterr().out

    assert code == ExitCode.SUCCESS
    assert "Sourdough rises because of yeast." in out
    assert "retrieve=12ms" in out
    assert "generate=340ms" in out
    assert "[^c1] handbook.md, page 2" in out


def test_a_degraded_stream_reports_its_mode_truncation_and_dropped_evidence(
    config: str, scripted: list[ScriptedService], capsys: pytest.CaptureFixture[str]
) -> None:
    """A cut-off answer and an answer grounded in part of its evidence looked complete."""
    scripted.append(ScriptedService(DEGRADED))

    code = query(config)
    out = capsys.readouterr().out

    assert code == ExitCode.SUCCESS
    assert "degraded: mode=extractive" in out
    assert "truncated:" in out
    assert "evidence dropped: 3" in out


def test_a_healthy_stream_reports_neither_truncation_nor_dropped_evidence(
    config: str, scripted: list[ScriptedService], capsys: pytest.CaptureFixture[str]
) -> None:
    scripted.append(ScriptedService(HEALTHY))

    query(config)
    out = capsys.readouterr().out

    assert "truncated" not in out
    assert "evidence dropped" not in out
    assert "degraded" not in out


def test_a_stream_ending_in_error_exits_non_zero_and_says_why(
    config: str, scripted: list[ScriptedService], capsys: pytest.CaptureFixture[str]
) -> None:
    """The error event was ignored, so half an answer exited 0 as though it were whole."""
    service = ScriptedService(BROKEN_OFF)
    scripted.append(service)

    code = query(config)
    captured = capsys.readouterr()

    assert code == ExitCode.FAILURE
    assert "[GENERATION_FAILED] the provider closed the stream" in captured.err
    assert service.closed == 1


def test_a_retryable_error_event_exits_as_unreachable(
    config: str, scripted: list[ScriptedService], capsys: pytest.CaptureFixture[str]
) -> None:
    """The event's own retryable flag picks the exit code, as a raised error's does."""
    events = [*BROKEN_OFF[:-1], QueryEvent(type="error", data={**BROKEN_OFF[-1].data})]
    events[-1].data.update(code="GENERATION_TIMEOUT", retryable=True, detail="timed out")
    scripted.append(ScriptedService(events))

    code = query(config)

    assert code == ExitCode.UNREACHABLE
    assert "[GENERATION_TIMEOUT] timed out" in capsys.readouterr().err
