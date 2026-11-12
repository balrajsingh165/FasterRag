"""What happens to an ingestion job between the ``202`` and the process exiting (TASK-0436).

A job used to run as a background task nothing held a reference to. These tests use a plain
``TestClient`` rather than the settling one in ``conftest.py``, because the moments they
assert on are exactly the ones in which a job is still in flight.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from fasterrag.api import ingest as ingest_router
from fasterrag.errors import ErrorCode
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.jobs import INTERRUPTED
from fasterrag.services.journal import Checkpoint, JobRecord, Journal

SOURCES = {"sources": [{"type": "path", "value": "corpus/"}]}


class GatedIngestion:
    """Accepts jobs, and runs each one only as far as a gate the test controls."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()
        self.started: list[str] = []

    async def accept(self, sources: Any, **kwargs: Any) -> JobRecord:
        return await self.journal.create_job(
            kwargs.get("collection") or "default",
            [
                {"type": "path", "value": source} if isinstance(source, str) else dict(source)
                for source in sources
            ],
            idempotency_key=kwargs.get("idempotency_key"),
            tenant=kwargs.get("tenant"),
        )

    async def run(self, record: JobRecord, **kwargs: Any) -> JobRecord:
        self.started.append(record.job_id)
        await self.journal.save_job(replace(record, status="running"))
        self.waiting.set()
        await self.gate.wait()
        settled = replace(record, status="completed", counts={"total": 1, "indexed": 1})
        await self.journal.save_job(settled)
        return settled

    async def release(self) -> None:
        self.gate.set()

    async def reached_the_gate(self) -> None:
        """Return once a job has recorded itself as running and is waiting to be released."""
        await self.waiting.wait()


class StubRouter:
    async def close(self) -> None:
        return None


@pytest.fixture
def ingestion(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> GatedIngestion:
    service = GatedIngestion(app.state.journal)
    monkeypatch.setattr(
        ingest_router,
        "build_ingestion",
        lambda settings, adapter, journal, cache, **kwargs: service,
    )
    monkeypatch.setattr(ingest_router, "create_embedding_router", lambda settings: StubRouter())
    return service


def settle(client: TestClient, app: FastAPI) -> None:
    """Wait, on the application's own loop, for every job in flight to finish."""
    portal = client.portal
    assert portal is not None
    portal.call(app.state.jobs.idle)


def test_an_accepted_job_is_held_by_the_registry_while_it_runs(
    app: FastAPI, ingestion: GatedIngestion
) -> None:
    """Nothing held a handle on a running job, so nothing could wait for it or stop it."""
    with TestClient(app) as client:
        accepted = client.post("/v1/ingest", json=SOURCES)
        job_id = accepted.json()["job_id"]
        in_flight = app.state.jobs.active
        assert client.portal is not None
        client.portal.call(ingestion.reached_the_gate)
        status_while_running = client.get(f"/v1/ingest/{job_id}").json()["status"]

        client.portal.call(ingestion.release)
        settle(client, app)

        assert accepted.status_code == 202
        assert in_flight == {job_id}
        assert status_while_running == "running"
        assert client.get(f"/v1/ingest/{job_id}").json()["status"] == "completed"
        assert app.state.jobs.active == frozenset()


def test_a_dead_letter_retry_is_held_by_the_registry_too(
    app: FastAPI, ingestion: GatedIngestion
) -> None:
    """The retry route started work through the same untracked background task."""
    journal: Journal = app.state.journal
    original = journal.create_job_blocking("default", [{"type": "path", "value": "corpus/"}])
    journal.dead_letter_blocking(
        original.job_id,
        document="doc_1",
        source="corpus/a.txt",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=1,
    )

    with TestClient(app) as client:
        retry_id = client.post(f"/v1/ingest/{original.job_id}/retry-dlq").json()["job_id"]
        in_flight = app.state.jobs.active

        assert client.portal is not None
        client.portal.call(ingestion.release)
        settle(client, app)

    assert in_flight == {retry_id}


async def test_a_job_cancelled_before_the_service_took_it_is_still_settled(
    app: FastAPI, ingestion: GatedIngestion
) -> None:
    """Cancellation is a ``BaseException``; the runner's handlers caught only ``Exception``."""
    journal: Journal = app.state.journal
    record = await journal.create_job("default", [{"type": "path", "value": "corpus/"}])
    request = cast(Request, SimpleNamespace(app=app))

    running = asyncio.ensure_future(ingest_router._run_job(request, record, None))
    while not ingestion.started:
        await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    settled = await journal.load_job(record.job_id)
    assert settled.status == "failed"
    assert settled.reason_code == INTERRUPTED


def test_the_status_body_says_why_a_job_stopped(client: TestClient, app: FastAPI) -> None:
    """A client could not tell an interrupted job from one whose documents all failed."""
    journal: Journal = app.state.journal
    record = journal.create_job_blocking("default", [{"type": "path", "value": "corpus/"}])
    journal.save_job_blocking(replace(record, status="failed", reason_code=INTERRUPTED))

    body = client.get(f"/v1/ingest/{record.job_id}").json()

    assert (body["status"], body["reason_code"]) == ("failed", INTERRUPTED)


def test_a_job_that_was_not_interrupted_reports_no_reason(client: TestClient, app: FastAPI) -> None:
    journal: Journal = app.state.journal
    record = journal.create_job_blocking("default", [{"type": "path", "value": "corpus/"}])

    assert client.get(f"/v1/ingest/{record.job_id}").json()["reason_code"] is None


class RecordingService(IngestionService):
    """The real service's acceptance, with a run that settles without a pipeline."""

    ran: ClassVar[list[str]] = []
    metadata: ClassVar[list[dict[str, Any] | None]] = []

    async def run(self, record: JobRecord, *, metadata: dict[str, Any] | None = None) -> JobRecord:
        RecordingService.ran.append(record.job_id)
        RecordingService.metadata.append(metadata)
        settled = replace(record, status="completed", counts={"total": 1, "indexed": 1})
        await self.journal.save_job(settled)
        return settled


@pytest.fixture
def real_acceptance(monkeypatch: pytest.MonkeyPatch) -> type[RecordingService]:
    """Build ingestion with the real ``accept``, which is where a replayed key is decided."""
    RecordingService.ran = []
    RecordingService.metadata = []

    def build(settings: Any, adapter: Any, journal: Journal, cache: Any, **kwargs: Any) -> Any:
        return RecordingService(
            settings, journal=journal, adapter=adapter, router=cast(Any, StubRouter())
        )

    monkeypatch.setattr(ingest_router, "build_ingestion", build)
    monkeypatch.setattr(ingest_router, "create_embedding_router", lambda settings: StubRouter())
    return RecordingService


def interrupt(app: FastAPI, job_id: str) -> None:
    """Leave a job as a cancelled run would: failed, INTERRUPTED, with a checkpoint."""
    journal: Journal = app.state.journal
    record = journal.load_job_blocking(job_id)
    journal.save_job_blocking(
        replace(
            record,
            status="failed",
            reason_code=INTERRUPTED,
            checkpoint=Checkpoint(last_document_index=4, written_at="2026-11-12T00:00:00+00:00"),
        )
    )


def test_replaying_the_body_key_of_an_interrupted_job_runs_it_again(
    client: TestClient, app: FastAPI, real_acceptance: type[RecordingService]
) -> None:
    """The replay returned the stuck record and started nothing, so the job could never finish."""
    body = {**SOURCES, "idempotency_key": "nightly-1", "metadata": {"team": "legal"}}
    job_id = client.post("/v1/ingest", json=body).json()["job_id"]
    interrupt(app, job_id)

    replayed = client.post("/v1/ingest", json=body)

    settled = client.get(f"/v1/ingest/{job_id}").json()
    assert replayed.status_code == 202
    assert replayed.json() == {"job_id": job_id, "status": "queued"}
    assert real_acceptance.ran == [job_id, job_id]
    assert real_acceptance.metadata[-1] == {"team": "legal"}
    assert (settled["status"], settled["reason_code"]) == ("completed", None)
    assert app.state.journal.load_job_blocking(job_id).checkpoint.last_document_index == 4


def test_replaying_the_body_key_of_a_finished_job_runs_nothing(
    client: TestClient, real_acceptance: type[RecordingService]
) -> None:
    """Resuming is for interruptions only: a replay must still never re-ingest a corpus."""
    body = {**SOURCES, "idempotency_key": "nightly-1"}
    job_id = client.post("/v1/ingest", json=body).json()["job_id"]

    replayed = client.post("/v1/ingest", json=body)

    assert replayed.json() == {"job_id": job_id, "status": "completed"}
    assert real_acceptance.ran == [job_id]


def test_replaying_the_header_key_of_an_interrupted_job_does_not_run_it_again(
    client: TestClient, app: FastAPI, real_acceptance: type[RecordingService]
) -> None:
    """The documented limit: the header is answered from the stored 202 before the route.

    Changing that is a decision about the idempotency middleware, not about ingestion, so
    this pins what the reference says rather than what would be convenient.
    """
    headers = {"Idempotency-Key": "8e0f3c1a-0000-4000-8000-000000000001"}
    first = client.post("/v1/ingest", json=SOURCES, headers=headers)
    job_id = first.json()["job_id"]
    interrupt(app, job_id)

    replayed = client.post("/v1/ingest", json=SOURCES, headers=headers)

    assert replayed.headers["Idempotent-Replayed"] == "true"
    assert replayed.json() == first.json()
    assert real_acceptance.ran == [job_id]
    assert client.get(f"/v1/ingest/{job_id}").json()["reason_code"] == INTERRUPTED
