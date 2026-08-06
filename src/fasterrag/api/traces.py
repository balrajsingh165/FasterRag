"""Trace inspection and replay endpoints (D8).

``GET /v1/traces/{trace_id}`` returns exactly what a past query did. ``POST /v1/replay``
re-executes it under a candidate configuration and returns a structured diff.

Replay builds a *second* set of services from the candidate configuration rather than
mutating the running one. Applying a candidate config to the live application to answer one
question would change every other in-flight query, which is the opposite of an isolated
experiment.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from fasterrag.adapters.embeddings.tiering import create_embedding_router
from fasterrag.api.dependencies import (
    CurrentSettings,
    CurrentTenant,
    CurrentVectorDB,
    build_generation,
)
from fasterrag.api.schemas import ReplayRequest
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.replay import replay_trace
from fasterrag.services.traces import TraceStore, create_trace_store

__all__ = ["get_trace_store", "router"]

router = APIRouter(prefix="/v1", tags=["traces"])


def get_trace_store(request: Request) -> TraceStore:
    """Return the application's trace store, built once per process."""
    store: TraceStore | None = getattr(request.app.state, "traces", None)
    if store is None:
        store = create_trace_store(request.app.state.settings)
        request.app.state.traces = store
    return store


TraceStoreDep = Annotated[TraceStore, Query()]


def _require_replay(settings: Settings) -> None:
    """Reject a replay when configuration has disabled it.

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` when ``traces.replay`` is false.
    """
    if not settings.traces.replay:
        raise FasterRagError(
            "replay is disabled; set traces.replay to true to enable it",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request, tenant: CurrentTenant) -> dict[str, Any]:
    """Return one stored trace in full.

    Raises:
        FasterRagError: With ``NOT_FOUND`` when the trace is unknown, expired past
            ``traces.retention_days``, or never stored because ``traces.store`` is false.
    """
    trace = get_trace_store(request).load(trace_id, tenant=tenant)
    if trace is None:
        raise FasterRagError(
            f"no stored trace {trace_id!r}; it may have expired or tracing may be disabled",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )
    return trace.as_dict()


@router.get("/traces")
async def list_traces(
    request: Request,
    tenant: CurrentTenant,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    """Return the most recently stored trace ids, newest first, scoped to the tenant."""
    return {"traces": get_trace_store(request).recent(limit, tenant=tenant)}


@router.post("/replay")
async def replay(
    body: ReplayRequest,
    request: Request,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Re-execute a past query under a candidate configuration and diff the outcome."""
    _require_replay(settings)

    # CRITICAL: tenant-scoped, exactly as GET /v1/traces/{id} is. Replay re-executes the
    # stored query and returns a diff, so an unscoped lookup here would let any tenant read
    # another's question and retrieved chunks — the same disclosure the trace endpoints
    # refuse, reached through a different door. An absent trace and another tenant's trace
    # are deliberately indistinguishable, for the reason `TraceStore.load` documents.
    trace = get_trace_store(request).load(body.trace_id, tenant=tenant)
    if trace is None:
        raise FasterRagError(
            f"no stored trace {body.trace_id!r}",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    candidate = settings.model_copy(deep=True)
    if body.config_overrides:
        candidate = Settings.model_validate(
            {**settings.model_dump(mode="json"), **body.config_overrides}
        )

    embedding_router = create_embedding_router(candidate)
    # CRITICAL: no cache and no trace store. A replay that populated the cache would make
    # the next real query answer from an experiment, and one that stored its own trace would
    # add to the record it was called to investigate.
    service = build_generation(candidate, adapter, embedding_router, cache=None)

    try:
        result = await replay_trace(trace, candidate, service)
    finally:
        await service.close()
        await embedding_router.close()

    return result.as_dict()
