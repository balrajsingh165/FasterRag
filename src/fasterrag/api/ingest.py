"""Ingestion endpoints.

``POST /v1/ingest`` accepts and returns; it never blocks on parsing, embedding, or indexing.
A corpus takes minutes to hours, and a request that waited for it would time out at every
proxy between the caller and the server. The job is journalled first and *then* started, so
the id in the ``202`` response is queryable the instant the caller receives it — starting the
work first would create a window where the job is running but its id 404s.

The work runs as a background task in this process. That is a deployment choice, not a
contract: the documented surface is "``202`` with a job id, poll ``GET /v1/ingest/{job_id}``",
which a distributed queue satisfies identically. Moving to one later changes no response.

Per-document status (D3) is served from the journal, so a dead-lettered document stays
answerable long after the job that produced it finished.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query, Request, status

from fasterrag.api.dependencies import (
    CurrentSettings,
    CurrentVectorDB,
    JournalDep,
    build_embedding_router,
    build_ingestion,
)
from fasterrag.api.schemas import IngestRequest
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError
from fasterrag.observability.logging import get_logger, use_trace_id
from fasterrag.services.journal import JobRecord, Journal

__all__ = ["router"]

router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])

_logger = get_logger(__name__)

_SUPPORTED_SOURCE_TYPES = frozenset({"path"})


def _job_body(record: JobRecord, journal: Journal) -> dict[str, Any]:
    """Return the documented job-status body.

    Counts come from the journal rather than the record while a job is running: the record's
    counts are only written at checkpoints, so a caller polling between two checkpoints would
    otherwise see stale numbers and conclude the job had stalled.
    """
    counts = dict(record.counts) or journal.counts(record.job_id)
    return {
        "job_id": record.job_id,
        "status": record.status,
        "counts": counts,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


async def _run_job(request: Request, record: JobRecord, metadata: dict[str, Any] | None) -> None:
    """Execute an accepted job, recording its outcome whatever happens.

    A background task has no caller to raise to, so a failure that escaped here would vanish
    with only a stack trace in the log and a job stuck at ``queued`` forever. Every outcome
    therefore lands in the journal.
    """
    settings = request.app.state.settings
    adapter = request.app.state.vector_db
    journal = request.app.state.journal
    embedding_router = build_embedding_router(settings)
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
        except FasterRagError as exc:
            _logger.error(
                "ingest job failed",
                extra={"job_id": record.job_id, "code": exc.code.value, "detail": exc.detail},
            )
            record.status = "failed"
            journal.save_job(record)
        finally:
            await embedding_router.close()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_ingest(
    body: IngestRequest,
    request: Request,
    background: BackgroundTasks,
    settings: CurrentSettings,
    journal: JournalDep,
    adapter: CurrentVectorDB,
) -> dict[str, Any]:
    """Accept an ingestion job and return its id immediately.

    Replaying an ``idempotency_key`` returns the original job rather than starting a second
    ingest of the same corpus.
    """
    unsupported = {source.type for source in body.sources} - _SUPPORTED_SOURCE_TYPES
    if unsupported:
        # TODO: url and inline sources ship with the loader that fetches them; only local
        # paths are readable today, and accepting a source type nothing can read would
        # produce a job that dead-letters every document for no stated reason.
        raise IngestionError(
            f"source type(s) {', '.join(sorted(unsupported))} are not supported yet; "
            f"supported: {', '.join(sorted(_SUPPORTED_SOURCE_TYPES))}",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    service = build_ingestion(settings, adapter, journal, None)
    record = await service.accept(
        [source.value for source in body.sources],
        collection=body.collection,
        idempotency_key=body.idempotency_key,
    )

    if record.status in {"queued"}:
        background.add_task(_run_job, request, record, body.metadata)

    return {"job_id": record.job_id, "status": record.status}


@router.get("/{job_id}")
async def job_status(job_id: str, journal: JournalDep) -> dict[str, Any]:
    """Return one job's status and counts."""
    return _job_body(journal.load_job(job_id), journal)


@router.get("/{job_id}/documents")
async def job_documents(
    job_id: str,
    journal: JournalDep,
    document_status: Annotated[str | None, Query(alias="status")] = None,
) -> dict[str, Any]:
    """Return per-document outcomes, optionally filtered by status (D3).

    Loading the job first means an unknown id 404s here too, rather than returning an empty
    list that reads as "this job ingested nothing".
    """
    journal.load_job(job_id)
    documents = [
        {
            "document_id": record.document_id,
            "source": record.source,
            "status": record.status,
            "reason_code": record.reason_code,
            "attempts": record.attempts,
            "content_hash": record.content_hash,
        }
        for record in journal.documents(job_id, status=document_status)
    ]
    return {"job_id": job_id, "documents": documents, "count": len(documents)}


@router.post("/{job_id}/retry-dlq", status_code=status.HTTP_202_ACCEPTED)
async def retry_dead_letters(
    job_id: str,
    request: Request,
    background: BackgroundTasks,
    settings: CurrentSettings,
    journal: JournalDep,
    adapter: CurrentVectorDB,
) -> dict[str, Any]:
    """Re-ingest a job's dead-lettered documents as a new job.

    A new job rather than a mutation of the old one: the original is an audit record of what
    happened, and rewriting it would destroy the evidence of the failure being retried.
    """
    original = journal.load_job(job_id)
    failed = journal.dead_lettered(job_id)
    if not failed:
        return {"job_id": job_id, "retried": 0, "status": original.status}

    service = build_ingestion(settings, adapter, journal, None)
    retry = await service.accept(
        [record.source for record in failed],
        collection=original.collection,
        tenant=original.tenant,
    )
    background.add_task(_run_job, request, retry, None)

    return {"job_id": retry.job_id, "retried": len(failed), "status": retry.status, "of": job_id}
