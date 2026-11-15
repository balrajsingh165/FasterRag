"""A client that goes away mid-answer stops the provider and leaves a trace (TASK-0467).

``POST /v1/query`` with ``stream: true`` relied on the server cancelling the response's
generator, and nothing proved that cancelling it reached the provider stream, or that the
abandoned query left anything behind for an operator to look up. These cases drive the
real application at the ASGI boundary — the only place a disconnect can be expressed — with
a provider that sends one piece of text and then waits, and disconnect after that text.

Both ways a server can report a disconnect are covered: ASGI HTTP 2.3 (uvicorn's) delivers
``http.disconnect`` to ``receive`` and Starlette cancels the response; 2.4 makes the next
``send`` raise ``OSError`` — and a provider that stalls means there is no next ``send``,
so the response has to listen for the disconnect itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect
from starlette.types import Message

from fasterrag.adapters.llm.base import Completion
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.api import dependencies as deps
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal
from fasterrag.services.traces import TraceStore
from tests.unit.api.test_query_tenancy import ACME_TEXT, MemoryRouter, MemoryVectorDB

FIRST_TEXT: Final = "The notice period "
BOUND_SECONDS: Final = 10.0
"""Far past what an abandoned query needs; reaching it means the provider was never stopped."""


class StallingLLM:
    """Streams one piece of text, then waits until it is closed, recording how it ended."""

    provider = "stalling"
    model = "stalling"

    def __init__(self) -> None:
        """Start with a stream neither opened nor closed."""
        self.opened = asyncio.Event()
        self.closed = asyncio.Event()

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Answer at once; only the stream stalls."""
        return Completion(text=FIRST_TEXT, model=self.model)

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield the first text, then wait for a second that never comes."""
        self.opened.set()
        try:
            yield FIRST_TEXT
            await asyncio.Event().wait()
            yield "is thirty days."
        finally:
            self.closed.set()

    async def health(self) -> HealthStatus:
        """Report healthy."""
        return HealthStatus(healthy=True, detail=None, latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing; the stream's own closing is what is observed."""


@pytest.fixture
def llm() -> StallingLLM:
    """Return the provider the application lends every query."""
    return StallingLLM()


@pytest.fixture
def traces(tmp_path: Path) -> TraceStore:
    """Return the trace store the application writes to."""
    return TraceStore(tmp_path / "traces")


@pytest.fixture
def application(tmp_path: Path, llm: StallingLLM, traces: TraceStore) -> FastAPI:
    """Return the application over an in-memory backend holding one chunk."""
    store = MemoryVectorDB()
    store.seed("default", "c_acme", ACME_TEXT, "acme")
    settings = Settings.model_validate(
        {"retrieval": {"rerank": False}, "cache": {"semantic": False}}
    )
    app = create_app(settings)
    app.state.vector_db = store
    app.state.journal = Journal(tmp_path / "journal")
    app.state.traces = traces
    app.dependency_overrides[deps.shared_embedding_router] = MemoryRouter
    app.dependency_overrides[deps.shared_llm] = lambda: llm
    return app


async def _stream_then_leave(app: FastAPI, spec_version: str) -> list[bytes]:
    """Ask a streamed question and disconnect once the first ``token`` event has arrived."""
    body = json.dumps({"query": "thirty days notice", "stream": True}).encode()
    asked = False
    gone = asyncio.Event()
    received: list[bytes] = []

    async def receive() -> Message:
        nonlocal asked
        if not asked:
            asked = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if gone.is_set():
            raise ConnectionResetError("the client closed the connection")
        if message["type"] == "http.response.body":
            received.append(message.get("body", b""))
            if b"event: token" in message.get("body", b""):
                gone.set()

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/query",
        "raw_path": b"/v1/query",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "state": {},
    }
    with contextlib.suppress(ClientDisconnect, ConnectionError):
        await asyncio.wait_for(app(scope, receive, send), BOUND_SECONDS)
    return received


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_a_client_that_disconnects_mid_stream_closes_the_provider_stream(
    application: FastAPI, llm: StallingLLM, traces: TraceStore, spec_version: str
) -> None:
    """Nothing proved a disconnect reached the provider, or that the query left a trace."""
    received = await _stream_then_leave(application, spec_version)

    assert any(b"event: token" in chunk for chunk in received)
    assert llm.opened.is_set()
    await asyncio.wait_for(llm.closed.wait(), BOUND_SECONDS)

    stored = [traces.load(trace_id) for trace_id in traces.recent()]
    assert len(stored) == 1
    trace = stored[0]
    assert trace is not None
    assert trace.result["aborted"] is True
    assert trace.response == FIRST_TEXT
    assert trace.result["answer"] == FIRST_TEXT
