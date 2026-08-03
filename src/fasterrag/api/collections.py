"""Collection management endpoints.

The same operations ``fasterrag index`` exposes, over the same adapter contract. Deletion
requires an explicit ``force`` for the same reason the CLI requires ``--force``: dropping a
collection destroys every vector in it and the only way back is a full re-ingest.

``POST /v1/collections`` sizes the collection from the configured embedding model rather than
from a request field. A caller-supplied dimension that disagreed with the model would produce
a collection nothing can write to, and the error would surface at first ingest rather than at
creation.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.api.dependencies import (
    CurrentSettings,
    CurrentTenant,
    CurrentVectorDB,
    build_embedding_router,
)
from fasterrag.api.schemas import CollectionRequest
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.tenancy import scoped_name, unscoped_name, visible_to

__all__ = ["router"]

router = APIRouter(prefix="/v1/collections", tags=["collections"])


async def _dimensions_of(embedder: EmbeddingAdapter) -> int:
    """Return the model's vector size, embedding a probe if it is not known yet.

    Raises:
        FasterRagError: If the model reports no usable size, since a collection created at
            the wrong width cannot be widened and forces a full re-embed to correct.
    """
    known = embedder.dimensions
    if known is not None:
        return known

    probed = len(await embedder.embed_query("dimension probe"))
    if not probed:
        raise FasterRagError(
            "the configured embedding model reported no vector size; ingest a document "
            "instead, which creates the collection automatically",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )
    return probed


@router.get("")
async def list_collections(adapter: CurrentVectorDB, tenant: CurrentTenant) -> dict[str, Any]:
    """List the caller's collections with their size and vector configuration.

    A tenant sees only its own, under the names it chose — the backend prefix never reaches
    a response, because a tenant should not learn that prefixing exists.
    """
    return {
        "collections": [
            {**info.as_dict(), "name": unscoped_name(info.name, tenant)}
            for info in await adapter.list_collections()
            if visible_to(info.name, tenant)
        ]
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CollectionRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Create a collection sized from the configured embedding model."""
    configured = settings.vector_db.collection
    embedding_router = build_embedding_router(settings)
    try:
        dimensions = await _dimensions_of(embedding_router.default)
    finally:
        await embedding_router.close()

    await adapter.create_collection(
        CollectionSpec(
            name=scoped_name(body.name, tenant),
            dimensions=dimensions,
            distance=body.distance or configured.distance,
            shard_number=body.shard_number or configured.shard_number,
            replication_factor=body.replication_factor or configured.replication_factor,
            sparse=settings.retrieval.hybrid,
        )
    )

    return {
        "name": body.name,
        "dimensions": dimensions,
        "distance": body.distance or configured.distance,
        "sparse": settings.retrieval.hybrid,
    }


@router.get("/{name}")
async def get_collection(
    name: str, adapter: CurrentVectorDB, tenant: CurrentTenant
) -> dict[str, Any]:
    """Return one collection's detail.

    Another tenant's collection is reported as *absent* rather than forbidden: a distinct
    error would confirm the name exists, which is enough to enumerate a competitor's
    collections one guess at a time.
    """
    target = scoped_name(name, tenant)
    for info in await adapter.list_collections():
        if info.name == target:
            return {**info.as_dict(), "name": name}

    raise FasterRagError(f"no collection named {name!r}", code=ErrorCode.NOT_FOUND, retryable=False)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    name: str,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
    force: Annotated[bool, Query()] = False,
) -> Response:
    """Drop a collection and everything in it.

    Refuses without ``?force=true``. The vectors do not come back, and a DELETE that a proxy
    or a retry could replay is not something to make easy.
    """
    if not force:
        raise FasterRagError(
            f"deleting {name!r} destroys its vectors; repeat the request with ?force=true",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    if not await adapter.drop_collection(scoped_name(name, tenant)):
        raise FasterRagError(
            f"no collection named {name!r}", code=ErrorCode.NOT_FOUND, retryable=False
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
