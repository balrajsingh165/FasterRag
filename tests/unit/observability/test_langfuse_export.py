import base64
from datetime import UTC, datetime

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Span, Trace
from fasterrag.observability.langfuse_export import LangfuseExporter, build_batch
from fasterrag.services.traces import (
    LANGFUSE_PUBLIC_KEY_VAR,
    LANGFUSE_SECRET_KEY_VAR,
    TraceStore,
    create_langfuse_exporter,
)

STARTED = "2026-08-03T10:00:00+00:00"


def trace() -> Trace:
    return Trace(
        trace_id="a" * 32,
        query="what is the meal allowance",
        collection="policies",
        config_snapshot={"llm": {"model": "gpt-4o-mini"}},
        retrieved=[{"chunk_id": "c_1"}],
        prompt="context...\n\nquestion",
        response="The allowance is 41.",
        result={
            "answer": "The allowance is 41.",
            "mode": "full",
            "degraded": False,
            "usage": {"prompt_tokens": 386, "completion_tokens": 51},
        },
        spans=[
            Span(name="retrieve", start_ms=0.0, end_ms=120.0),
            Span(name="generation", start_ms=120.0, end_ms=1500.0),
        ],
        created_at=STARTED,
    )


def events_by_type(batch: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    return [event for event in batch if event["type"] == kind]


def test_the_trace_becomes_one_trace_create_event() -> None:
    batch = build_batch(trace())

    created = events_by_type(batch, "trace-create")
    assert len(created) == 1
    body = created[0]["body"]
    assert isinstance(body, dict)
    assert body["id"] == "a" * 32
    assert body["input"] == "what is the meal allowance"
    assert body["output"] == "The allowance is 41."


def test_every_span_becomes_an_observation() -> None:
    batch = build_batch(trace())

    observations = events_by_type(batch, "span-create") + events_by_type(batch, "generation-create")
    assert len(observations) == 2


def test_the_generation_stage_is_a_generation_not_a_span() -> None:
    """Langfuse derives model usage and cost only from `generation` observations.

    Sending it as a plain span leaves Langfuse's own cost view empty while the data sits
    right there in the payload.
    """
    batch = build_batch(trace())

    generations = events_by_type(batch, "generation-create")
    assert len(generations) == 1
    body = generations[0]["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-4o-mini"
    assert body["usage"] == {"input": 386, "output": 51, "unit": "TOKENS"}


def test_span_offsets_become_absolute_timestamps() -> None:
    """Our spans are millisecond offsets; Langfuse orders observations by wall clock.

    Sending raw offsets would place every trace at the epoch, stacked on top of each other.
    """
    batch = build_batch(trace())

    body = events_by_type(batch, "span-create")[0]["body"]
    assert isinstance(body, dict)
    start = datetime.fromisoformat(str(body["startTime"]))
    end = datetime.fromisoformat(str(body["endTime"]))

    assert start == datetime(2026, 8, 3, 10, 0, 0, tzinfo=UTC)
    assert (end - start).total_seconds() == pytest.approx(0.12)


def test_every_event_carries_a_distinct_id() -> None:
    """Langfuse deduplicates on event id; a collision silently drops an observation."""
    batch = build_batch(trace())

    identifiers = [event["id"] for event in batch]
    assert len(identifiers) == len(set(identifiers))


def test_a_malformed_created_at_does_not_lose_the_trace() -> None:
    """A misplaced observation is recoverable; a dropped trace is not."""
    broken = trace()
    object.__setattr__(broken, "created_at", "not-a-timestamp")

    batch = build_batch(broken)

    assert events_by_type(batch, "trace-create")


def test_the_credentials_are_sent_as_basic_auth_and_not_stored_in_the_url() -> None:
    exporter = LangfuseExporter("http://localhost:3000/", "pk-lf-x", "sk-lf-y")

    expected = base64.b64encode(b"pk-lf-x:sk-lf-y").decode("ascii")
    assert exporter._headers["Authorization"] == f"Basic {expected}"
    assert "pk-lf-x" not in exporter.host


def test_no_exporter_is_built_while_the_toggle_is_off() -> None:
    assert create_langfuse_exporter(Settings.model_validate({})) is None


def test_no_exporter_is_built_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusing to serve because a dashboard has no credentials inverts the dependency."""
    monkeypatch.delenv(LANGFUSE_PUBLIC_KEY_VAR, raising=False)
    monkeypatch.delenv(LANGFUSE_SECRET_KEY_VAR, raising=False)
    settings = Settings.model_validate({"observability": {"langfuse": True}})

    assert create_langfuse_exporter(settings) is None


def test_an_exporter_is_built_when_the_toggle_and_keys_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LANGFUSE_PUBLIC_KEY_VAR, "pk-lf-x")
    monkeypatch.setenv(LANGFUSE_SECRET_KEY_VAR, "sk-lf-y")
    settings = Settings.model_validate({"observability": {"langfuse": True}})

    assert create_langfuse_exporter(settings) is not None


def test_storing_without_a_running_loop_does_not_raise(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A CLI one-shot has no loop; the local write must still happen."""
    store = TraceStore(tmp_path, exporter=LangfuseExporter("http://x", "pk", "sk"))

    store.store(trace())

    assert store.load("a" * 32) is not None


async def test_an_export_task_is_retained_until_it_finishes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Asyncio holds only a weak reference: an unretained task can vanish mid-flight."""
    store = TraceStore(tmp_path, exporter=LangfuseExporter("http://127.0.0.1:1", "pk", "sk"))

    store.store(trace())

    assert store._exports
    await store.drain()
    assert not store._exports


async def test_the_full_http_request_reaches_an_ingestion_endpoint() -> None:
    """Exercises the real client, URL, headers, and body — not just the batch builder.

    A local server rather than a mock transport: the point is to confirm what actually goes
    over a socket, including the path the exporter appends and the auth header it sets.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            received["path"] = self.path
            received["auth"] = self.headers.get("Authorization")
            received["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_response(207)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exporter = LangfuseExporter(f"http://127.0.0.1:{server.server_port}", "pk-lf-x", "sk-lf-y")
        accepted = await exporter.export(trace())
        await exporter.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert accepted is True
    assert received["path"] == "/api/public/ingestion"
    assert str(received["auth"]).startswith("Basic ")
    body = received["body"]
    assert isinstance(body, dict)
    assert len(body["batch"]) == 3


async def test_an_unreachable_langfuse_is_reported_not_raised() -> None:
    """The query was answered long before this runs; a dashboard outage is not an outage."""
    exporter = LangfuseExporter("http://127.0.0.1:1", "pk", "sk")

    assert await exporter.export(trace()) is False
    await exporter.close()


async def test_a_rejected_batch_is_reported_not_raised() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Rejecting(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Rejecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exporter = LangfuseExporter(f"http://127.0.0.1:{server.server_port}", "pk", "sk")
        accepted = await exporter.export(trace())
        await exporter.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert accepted is False
