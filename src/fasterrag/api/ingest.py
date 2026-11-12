"""Ingestion endpoints.

``POST /v1/ingest`` accepts and returns; it never blocks on parsing, embedding, or indexing.
A corpus takes minutes to hours, and a request that waited for it would time out at every
proxy between the caller and the server. The job is journalled first and *then* started, so
the id in the ``202`` response is queryable the instant the caller receives it — starting the
work first would create a window where the job is running but its id 404s.

The work runs as a task in this process, held by the application's
:class:`~fasterrag.services.jobs.JobRegistry` so shutdown can find it (TASK-0436). That is
a deployment choice, not a contract: the documented surface is "``202`` with a job id, poll
``GET /v1/ingest/{job_id}``", which a distributed queue satisfies identically. Moving to one
later changes no response, and the registry is where it plugs in (TASK-0130).

Per-document status (D3) is served from the journal, so a dead-lettered document stays
answerable long after the job that produced it finished.

Accepting is also the last moment at which a configuration mistake can be reported *to the
caller*. Everything after the ``202`` happens in a background task whose failures reach a
log, not the client, so the keyword-leg preflight (TASK-0119) runs here: ingesting hybrid
chunks into a dense-only collection is refused as a 409 before the job exists, rather than
accepted and then dead-lettering every document against an error about one point id.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response, status

from fasterrag.api.dependencies import (
    AcceptingJobs,
    CurrentSettings,
    CurrentTenant,
    CurrentVectorDB,
    JournalDep,
)
from fasterrag.api.schemas import IngestRequest
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import current_trace_id, get_logger, use_trace_id
from fasterrag.services.composition import build_ingestion, create_embedding_router
from fasterrag.services.jobs import settle_interrupted
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.reindex import require_keyword_leg

__all__ = ["router"]

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])

_logger = get_logger(__name__)


DEPRECATED_KEY_FIELD_DOC = "https://fasterrag.dev/docs/api-reference#idempotency"


def _mark_deprecated(response: Response) -> None:
    """Flag a response whose request used the superseded ``idempotency_key`` body field.

    Announced in the response rather than only in the docs, so a client that never reads
    them still learns before 1.0 removes the field (TASK-0318).
    """
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = f'<{DEPRECATED_KEY_FIELD_DOC}>; rel="deprecation"'
    _logger.warning(
        "the idempotency_key body field is deprecated; send the Idempotency-Key header",
        extra={"endpoint": "POST /v1/ingest"},
    )


async def _job_body(record: JobRecord, journal: Journal) -> dict[str, Any]:
    """Return the documented job-status body."""
    counts = dict(record.counts) or await journal.counts(record.job_id)
    return {
        "job_id": record.job_id,
        "status": record.status,
        "reason_code": record.reason_code,
        "counts": counts,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


async def _ensure_settled(journal: Journal, record: JobRecord, code: ErrorCode) -> None:
    """Mark a job ``failed`` with ``code`` unless the service already settled it.

    The current record is loaded rather than the one this request accepted, so a
    checkpoint the run reached is kept instead of overwritten with a stale copy.
    """
    try:
        current = await journal.load_job(record.job_id)
    except FasterRagError:
        current = record

    if current.status in {"queued", "running"}:
        await journal.save_job(replace(current, status="failed", reason_code=code.value))


async def _run_job(request: Request, record: JobRecord, metadata: dict[str, Any] | None) -> None:
    """Execute an accepted job, recording its outcome whatever happens."""
    settings = request.app.state.settings
    adapter = request.app.state.vector_db
    journal = request.app.state.journal
    embedding_router = create_embedding_router(settings)
    service = build_ingestion(
        settings,
        adapter,
        journal,
        getattr(request.app.state, "cache", None),
        router=embedding_router,
    )

    with use_trace_id():
        try:
            await service.run(record, metadata=metadata)
        except asyncio.CancelledError:
            # CRITICAL: cancellation is how shutdown stops a job, and it is a BaseException
            # the two handlers below never see. The service settles a job it was running;
            # this covers the one cancelled before the service took it, which would
            # otherwise stay `queued` with nothing left to run it.
            await settle_interrupted(journal, record)
            raise
        except FasterRagError as exc:
            _logger.error(
                "ingest job failed in the background",
                extra={
                    "job_id": record.job_id,
                    "code": exc.code.value,
                    "detail": exc.detail,
                    "trace_id": exc.trace_id,
                },
            )
            await _ensure_settled(journal, record, exc.code)
        except Exception as exc:
            # CRITICAL: a background task has no caller to raise to. Without this branch an
            # untyped failure is logged by the server and the job stays `running` forever,
            # which a client polling the job cannot tell apart from slow progress.
            _logger.error(
                "ingest job stopped on an unexpected error",
                extra={
                    "job_id": record.job_id,
                    "code": ErrorCode.INTERNAL.value,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "trace_id": current_trace_id(),
                },
            )
            await _ensure_settled(journal, record, ErrorCode.INTERNAL)
        finally:
            await embedding_router.close()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_ingest(
    body: IngestRequest,
    request: Request,
    response: Response,
    jobs: AcceptingJobs,
    settings: CurrentSettings,
    journal: JournalDep,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Accept an ingestion job and return its id immediately.

    Refuses first if the target index cannot serve the configured keyword leg; replaying an
    ``idempotency_key`` returns the original job rather than ingesting the corpus twice.
    """
    if body.idempotency_key is not None:
        _mark_deprecated(response)

    await require_keyword_leg(
        body.collection or settings.vector_db.collection.default_name, settings, adapter
    )

    service = build_ingestion(settings, adapter, journal, None)
    # CRITICAL: the tenant is recorded at creation. Without it every job is written
    # untenanted, and scoping the read path would then refuse a tenant access to the job it
    # just started — the isolation would present as a total outage rather than as isolation.
    record = await service.accept(
        [{"type": source.type, "value": source.value} for source in body.sources],
        collection=body.collection,
        idempotency_key=body.idempotency_key,
        tenant=tenant,
    )

    if record.status in {"queued"}:
        metadata = dict(body.metadata or {})
        if body.priority_class:
            metadata["priority_class"] = body.priority_class
        await jobs.submit(record.job_id, _run_job(request, record, metadata or None))

    return {"job_id": record.job_id, "status": record.status}


@router.get("/{job_id}")
async def job_status(job_id: str, journal: JournalDep, tenant: CurrentTenant) -> dict[str, Any]:
    """Return one job's status and counts."""
    return await _job_body(await journal.load_job(job_id, tenant=tenant), journal)


@router.get("/{job_id}/documents")
async def job_documents(
    job_id: str,
    journal: JournalDep,
    tenant: CurrentTenant,
    document_status: Annotated[str | None, Query(alias="status")] = None,
) -> dict[str, Any]:
    """Return per-document outcomes, optionally filtered by status (D3).

    Loading the job first means an unknown id 404s here too, rather than returning an empty
    list that reads as "this job ingested nothing".
    """
    await journal.load_job(job_id, tenant=tenant)
    documents = [
        {
            "document_id": record.document_id,
            "source": record.source,
            "status": record.status,
            "reason_code": record.reason_code,
            "attempts": record.attempts,
            "content_hash": record.content_hash,
        }
        for record in await journal.documents(job_id, status=document_status)
    ]
    return {"job_id": job_id, "documents": documents, "count": len(documents)}


@router.post("/{job_id}/retry-dlq", status_code=status.HTTP_202_ACCEPTED)
async def retry_dead_letters(
    job_id: str,
    request: Request,
    jobs: AcceptingJobs,
    settings: CurrentSettings,
    journal: JournalDep,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Re-ingest a job's dead-lettered documents as a new job, behind the same preflight.

    A new job rather than a mutation of the old one: the original is an audit record of what
    happened, and rewriting it would destroy the evidence of the failure being retried.
    """
    original = await journal.load_job(job_id, tenant=tenant)
    failed = await journal.dead_lettered(job_id)
    if not failed:
        return {"job_id": job_id, "retried": 0, "status": original.status}

    await require_keyword_leg(original.collection, settings, adapter)

    service = build_ingestion(settings, adapter, journal, None)
    retry = await service.accept(
        [record.source for record in failed],
        collection=original.collection,
        tenant=original.tenant,
    )
    await jobs.submit(retry.job_id, _run_job(request, retry, None))

    return {"job_id": retry.job_id, "retried": len(failed), "status": retry.status, "of": job_id}
