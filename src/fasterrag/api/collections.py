"""Collection management endpoints.

The same operations ``fasterrag index`` exposes, over the same adapter contract. Deletion
requires an explicit ``force`` for the same reason the CLI requires ``--force``: dropping a
collection destroys every vector in it and the only way back is a full re-ingest.

``POST /v1/collections`` sizes the collection from the configured embedding model rather than
from a request field. A caller-supplied dimension that disagreed with the model would produce
a collection nothing can write to, and the error would surface at first ingest rather than at
creation.

**The creation response reports the live collection, not the request** (TASK-0320). It used
to echo ``"sparse": settings.retrieval.hybrid`` whatever the backend did, so a backend that
accepted the create over an existing dense-only collection — which Qdrant does, verifying
compatibility and returning without adding a sparse vector — was reported as hybrid when it
had no keyword leg at all. That is an affirmative untruth in a 201, not a deferred failure:
the client is told the thing exists. The create is therefore followed by
:func:`~fasterrag.services.reindex.require_keyword_leg`, which reads the collection back from
the server, refuses the mismatch with the same 409 ``CONFLICT`` ``POST /v1/ingest`` returns,
and otherwise supplies the ``sparse`` the response reports. ``fasterrag index create``
re-checks after creating for the same reason.

``DELETE /v1/collections/{name}/documents/{document_id}`` lives here rather than under
``/v1/ingest`` because a document is addressed inside a collection, and so that it resolves
the collection exactly as its siblings do — through :func:`scoped_name` — instead of
introducing a third way to spell one (TASK-0443). The chunk filter additionally requires the
caller's tenant tag, so the route is correct under either tenancy model TASK-0339 may choose.
Until TASK-0381 makes ingest apply the same scoping, a tenant's documents ingested through
``POST /v1/ingest`` live under the unprefixed name, which this route does not address.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Response, status

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.api.dependencies import (
    CurrentCache,
    CurrentSettings,
    CurrentTenant,
    CurrentVectorDB,
    JournalDep,
    build_documents,
    build_embedding_router,
)
from fasterrag.api.schemas import CollectionRequest
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.reindex import require_keyword_leg
from fasterrag.services.tenancy import scoped_name, unscoped_name, visible_to

__all__ = ["router"]

router = APIRouter(prefix="/v1/collections", tags=["collections"])


async def _dimensions_of(embedder: EmbeddingAdapter) -> int:
    """Return the model's vector size, embedding a probe if it is not known yet."""
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
    """Create a collection sized from the configured embedding model.

    Reads the collection back afterwards, so the reported keyword leg is the one the
    backend actually added and a backend that added none is a 409 rather than a 201.
    """
    configured = settings.vector_db.collection
    embedding_router = build_embedding_router(settings)
    try:
        dimensions = await _dimensions_of(embedding_router.default)
    finally:
        await embedding_router.close()

    target = scoped_name(body.name, tenant)
    await adapter.create_collection(
        CollectionSpec(
            name=target,
            dimensions=dimensions,
            distance=body.distance or configured.distance,
            shard_number=body.shard_number or configured.shard_number,
            replication_factor=body.replication_factor or configured.replication_factor,
            sparse=settings.retrieval.hybrid,
        )
    )
    audit = await require_keyword_leg(target, settings, adapter, display=body.name)

    return {
        "name": body.name,
        "dimensions": dimensions,
        "distance": body.distance or configured.distance,
        "sparse": audit.present,
    }


@router.get("/{name}")
async def get_collection(
    name: str, adapter: CurrentVectorDB, tenant: CurrentTenant
) -> dict[str, Any]:
    """Return one collection's detail."""
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


@router.delete("/{name}/documents/{document_id}")
async def delete_document(
    name: str,
    document_id: str,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    journal: JournalDep,
    cache: CurrentCache,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Delete one document's chunks from a collection, and every record it was indexed.

    Idempotent: a document nothing holds is a 200 with ``known: false``, because the
    backend cannot say whether a filter matched anything. A missing collection is a 404.
    """
    service = build_documents(settings, adapter, journal, cache)
    deletion = await service.delete(
        scoped_name(name, tenant), document_id, tenant=tenant, display=name
    )
    return deletion.as_dict()
