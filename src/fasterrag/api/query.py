"""Query endpoints.

``POST /v1/query`` serves both shapes from one path: a JSON body, or a ``text/event-stream``
when ``stream`` is true. The service already produces typed events in the documented order,
so this module only serializes them — the event contract is enforced and tested where it is
produced, not here.

Streaming needs one thing HTTP does not give for free: a stream that fails after its headers
have gone out cannot change its status code. A 200 is already committed by the time a
provider dies, which is exactly why the contract makes the *absence* of ``done`` meaningful
and sends errors as an ``error`` event rather than as a status.

Every stream closes its services. An SSE generator that returns early — a client
disconnecting mid-answer is routine — would otherwise leak a connection pool per abandoned
request, and abandoned requests are the common case for a slow answer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from fasterrag.api.dependencies import (
    CurrentCache,
    CurrentEmbeddings,
    CurrentSettings,
    CurrentVectorDB,
    build_generation,
    build_retrieval,
)
from fasterrag.api.schemas import QueryRequest
from fasterrag.api.traces import get_trace_store
from fasterrag.errors import FasterRagError, problem_spec
from fasterrag.observability.logging import current_trace_id, get_logger
from fasterrag.services.generation import GenerationService, QueryEvent

__all__ = ["router"]

router = APIRouter(prefix="/v1", tags=["query"])

_logger = get_logger(__name__)

SSE_MEDIA_TYPE = "text/event-stream"


def encode_event(event: QueryEvent) -> str:
    """Return one SSE frame.

    Two constraints from the wire format, not from taste: the data field must be a single
    line, so the JSON is compact rather than indented, and a frame ends with a blank line,
    without which a client buffers the event indefinitely waiting for more of it.
    """
    payload = json.dumps(event.data, separators=(",", ":"), default=str)
    return f"event: {event.type}\ndata: {payload}\n\n"


async def _stream(service: GenerationService, body: QueryRequest) -> AsyncIterator[str]:
    """Yield the answer as SSE frames, releasing the services whatever happens."""
    try:
        async for event in service.stream(
            body.query,
            collection=body.collection,
            top_k=body.top_k,
            filters=body.filters,
        ):
            yield encode_event(event)
    except FasterRagError as exc:
        # CRITICAL: the status line is already sent, so this cannot become a problem
        # response. It goes out as an `error` event and the stream ends without `done`,
        # which is precisely the signal the contract gives clients for an incomplete answer.
        _logger.warning(
            "query stream failed after the response began",
            extra={"code": exc.code.value, "detail": exc.detail, "trace_id": exc.trace_id},
        )
        yield encode_event(
            QueryEvent(
                type="error",
                data={
                    "code": exc.code.value,
                    "detail": exc.detail,
                    "trace_id": exc.trace_id,
                    "retryable": exc.retryable,
                    "status": problem_spec(exc.code).status,
                },
            )
        )
    finally:
        await service.close()


@router.post("/query")
async def run_query(
    body: QueryRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    embeddings: CurrentEmbeddings,
    cache: CurrentCache,
    request: Request,
) -> Any:
    """Answer a question, as one JSON body or as a stream of SSE events."""
    service = build_generation(settings, adapter, embeddings, cache, get_trace_store(request))

    streaming = settings.llm.streaming if body.stream is None else body.stream
    if streaming:
        return StreamingResponse(
            _stream(service, body),
            media_type=SSE_MEDIA_TYPE,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        answer = await service.answer(
            body.query,
            collection=body.collection,
            top_k=body.top_k,
            filters=body.filters,
        )
    finally:
        await service.close()

    return answer.as_dict()


@router.post("/retrieve")
async def run_retrieve(
    body: QueryRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    embeddings: CurrentEmbeddings,
) -> dict[str, Any]:
    """Return the retrieved chunks without generating an answer.

    The retrieval-only path of ``docs/python-api.md``, for callers that bring their own LLM
    step, and the fastest way to see why a chunk ranked where it did — every leg's rank and
    score survives into the response rather than being collapsed into the fused number.
    """
    retrieval = build_retrieval(settings, adapter, embeddings)
    result = await retrieval.search(
        body.query,
        collection=body.collection,
        top_k=body.top_k,
        filters=body.filters,
    )

    return {
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text if body.include_chunks else None,
                "source": chunk.source,
                "document_id": chunk.document_id,
                "dense_rank": chunk.dense_rank,
                "dense_score": chunk.dense_score,
                "bm25_rank": chunk.bm25_rank,
                "bm25_score": chunk.bm25_score,
                "rrf_score": round(chunk.rrf_score, 6),
                "rerank_score": chunk.rerank_score,
                "final_rank": chunk.final_rank,
            }
            for chunk in result.chunks
        ],
        "mode": result.mode,
        "degraded": result.mode != "full",
        "trace_id": current_trace_id(),
    }
