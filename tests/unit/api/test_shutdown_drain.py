"""Shutdown drains ingestion before it closes what ingestion is using (TASK-0436).

The lifespan used to close the embedding router, the cache, and the vector database while
jobs were still running against them, and nothing stopped a new job being accepted on the
way down. These tests drive the lifespan by hand on one event loop, because the moment they
assert on — a request arriving *while* shutdown is waiting — cannot be reached through a
client whose context manager blocks until shutdown has finished.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from fasterrag.api import ingest as ingest_router
from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode
from fasterrag.services.jobs import INTERRUPTED
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.traces import TraceStore
from tests.unit.api.conftest import StubVectorDB

SOURCES = {"sources": [{"type": "path", "value": "corpus/"}]}
WAIT_SECONDS = 10


class RecordingResource:
    """Stands in for a process-scoped resource the lifespan closes on the way down."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.statuses_at_close: list[str] | None = None

    async def close(self) -> None:
        self.statuses_at_close = [
            (await self.journal.load_job(path.stem)).status
            for path in sorted((self.journal.root / "jobs").glob("*.json"))
        ]


class GatedIngestion:
    """Accepts jobs, and runs each one only as far as a gate the test controls."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()

    async def accept(self, sources: Any, **kwargs: Any) -> JobRecord:
        return await self.journal.create_job(
            kwargs.get("collection") or "default",
            [dict(source) for source in sources],
            idempotency_key=kwargs.get("idempotency_key"),
            tenant=kwargs.get("tenant"),
        )

    async def run(self, record: JobRecord, **kwargs: Any) -> JobRecord:
        await self.journal.save_job(replace(record, status="running"))
        self.waiting.set()
        await self.gate.wait()
        settled = replace(record, status="completed", counts={"total": 1, "indexed": 1})
        await self.journal.save_job(settled)
        return settled


class _StubRouter:
    async def close(self) -> None:
        return None


class Running:
    """An application whose startup has run, with the pieces a test needs to reach."""

    def __init__(
        self,
        app: FastAPI,
        client: httpx.AsyncClient,
        ingestion: GatedIngestion,
        resource: RecordingResource,
        shutdown: Callable[[], Awaitable[object]],
    ) -> None:
        self.app = app
        self.client = client
        self.ingestion = ingestion
        self.resource = resource
        self.shutdown = shutdown

    async def start_job(self) -> str:
        """Accept one job and wait until it is running and parked at the gate."""
        accepted = await self.client.post("/v1/ingest", json=SOURCES)
        assert accepted.status_code == 202, accepted.text
        await asyncio.wait_for(self.ingestion.waiting.wait(), WAIT_SECONDS)
        return str(accepted.json()["job_id"])

    async def status(self, job_id: str) -> JobRecord:
        journal: Journal = self.app.state.journal
        return await journal.load_job(job_id)


async def running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, grace_seconds: int
) -> Running:
    """Start an application whose shutdown waits ``grace_seconds`` for its jobs."""
    settings = Settings.model_validate({"reliability": {"shutdown_grace_seconds": grace_seconds}})
    app = create_app(settings)
    journal = Journal(tmp_path / "journal")
    ingestion = GatedIngestion(journal)
    resource = RecordingResource(journal)
    app.state.vector_db = StubVectorDB()
    app.state.journal = journal
    app.state.traces = TraceStore(tmp_path / "traces")
    app.state.embeddings = resource
    monkeypatch.setattr(
        ingest_router,
        "build_ingestion",
        lambda settings, adapter, journal, cache, **kwargs: ingestion,
    )
    monkeypatch.setattr(ingest_router, "build_embedding_router", lambda settings: _StubRouter())

    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    return Running(app, client, ingestion, resource, lambda: lifespan.__aexit__(None, None, None))


async def test_shutdown_waits_for_a_job_that_finishes_inside_the_grace_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown waited for nothing: it closed the adapters under a job still writing."""
    api = await running(tmp_path, monkeypatch, grace_seconds=WAIT_SECONDS)
    job_id = await api.start_job()

    stopping = asyncio.ensure_future(api.shutdown())
    await asyncio.sleep(0.05)
    still_draining = not stopping.done()
    api.ingestion.gate.set()
    await asyncio.wait_for(stopping, WAIT_SECONDS)

    assert still_draining
    assert (await api.status(job_id)).status == "completed"
    assert api.resource.statuses_at_close == ["completed"]


async def test_a_job_that_outlives_the_grace_period_is_cancelled_and_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job still running at exit kept its ``running`` record for good."""
    api = await running(tmp_path, monkeypatch, grace_seconds=1)
    job_id = await api.start_job()

    began = time.monotonic()
    await asyncio.wait_for(api.shutdown(), WAIT_SECONDS)
    waited = time.monotonic() - began

    settled = await api.status(job_id)
    assert (settled.status, settled.reason_code) == ("failed", INTERRUPTED)
    assert 0.9 <= waited < WAIT_SECONDS
    assert api.app.state.jobs.active == frozenset()


async def test_the_adapters_are_closed_only_after_the_jobs_have_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing first is what turned an interrupted job into errors against a closed client."""
    api = await running(tmp_path, monkeypatch, grace_seconds=0)
    await api.start_job()

    await asyncio.wait_for(api.shutdown(), WAIT_SECONDS)

    assert api.resource.statuses_at_close == ["failed"]


async def test_a_job_submitted_during_the_drain_is_refused_as_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job accepted on the way down was recorded and then never run."""
    api = await running(tmp_path, monkeypatch, grace_seconds=WAIT_SECONDS)
    await api.start_job()

    stopping = asyncio.ensure_future(api.shutdown())
    await asyncio.sleep(0.05)
    refused = await api.client.post("/v1/ingest", json=SOURCES)
    api.ingestion.gate.set()
    await asyncio.wait_for(stopping, WAIT_SECONDS)

    problem = refused.json()
    journal: Journal = api.app.state.journal
    assert refused.status_code == 503
    assert refused.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert problem["code"] == ErrorCode.NOT_READY.value
    assert problem["retryable"] is True
    assert len(list((journal.root / "jobs").glob("*.json"))) == 1


async def test_a_dead_letter_retry_during_the_drain_is_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry route starts a job as well, through the same door."""
    api = await running(tmp_path, monkeypatch, grace_seconds=WAIT_SECONDS)
    job_id = await api.start_job()

    stopping = asyncio.ensure_future(api.shutdown())
    await asyncio.sleep(0.05)
    refused = await api.client.post(f"/v1/ingest/{job_id}/retry-dlq")
    api.ingestion.gate.set()
    await asyncio.wait_for(stopping, WAIT_SECONDS)

    assert refused.status_code == 503
    assert refused.json()["code"] == ErrorCode.NOT_READY.value


async def test_reads_are_still_answered_during_the_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only starting work is refused; a client polling its job must still get an answer."""
    api = await running(tmp_path, monkeypatch, grace_seconds=WAIT_SECONDS)
    job_id = await api.start_job()

    stopping = asyncio.ensure_future(api.shutdown())
    await asyncio.sleep(0.05)
    polled = await api.client.get(f"/v1/ingest/{job_id}")
    api.ingestion.gate.set()
    await asyncio.wait_for(stopping, WAIT_SECONDS)

    assert polled.status_code == 200
    assert polled.json()["status"] == "running"


async def test_an_application_started_again_accepts_jobs_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain that latched for good would refuse every job after one restart of the app."""
    api = await running(tmp_path, monkeypatch, grace_seconds=0)
    await api.shutdown()

    lifespan = api.app.router.lifespan_context(api.app)
    await lifespan.__aenter__()
    api.ingestion.gate.set()
    accepted = await api.client.post("/v1/ingest", json=SOURCES)
    await lifespan.__aexit__(None, None, None)

    assert accepted.status_code == 202


def test_the_grace_period_defaults_to_twenty_seconds() -> None:
    assert Settings().reliability.shutdown_grace_seconds == 20
