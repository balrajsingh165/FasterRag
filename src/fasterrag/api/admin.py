"""Administrative endpoints: doctor, provisioning, and preflight estimation.

The REST half of what ``fasterrag doctor``, ``fasterrag provision``, and ``fasterrag
estimate`` do, calling the same services so the two control planes report the same state.

``GET /v1/admin/doctor`` returns ``200`` even when checks fail. The failing checks *are* the
payload; a ``503`` would make the report unreadable to exactly the client that most needs it,
and the endpoint's own availability is not what it is reporting on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from fasterrag.api.dependencies import (
    CurrentSettings,
    CurrentTenant,
    CurrentVectorDB,
    build_embedding_router,
)
from fasterrag.api.schemas import EstimateRequest, ExportRequest, ImportRequest
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.archive import export_archive
from fasterrag.services.archive_import import import_archive, open_archive
from fasterrag.services.doctor import run_doctor
from fasterrag.services.estimation import estimate_sources, require_estimator
from fasterrag.services.lockfile import create_lock_store
from fasterrag.services.provisioning import provision_qdrant, qdrant_status
from fasterrag.services.tenancy import scoped_name, unscoped_name

__all__ = ["router"]

router = APIRouter(prefix="/v1", tags=["admin"])

_PROVISIONABLE = frozenset({"qdrant"})


@router.get("/admin/doctor")
async def doctor_report(settings: CurrentSettings) -> dict[str, Any]:
    """Return the machine-readable preflight report (D10)."""
    return (await run_doctor(settings)).as_dict()


def _require_provisionable(tool: str) -> None:
    """Reject a tool nothing can provision yet.

    Raises:
        FasterRagError: With ``NOT_FOUND`` for an unknown or unimplemented tool.
    """
    if tool not in _PROVISIONABLE:
        # TODO: langfuse and grafana provisioning ship with TASK-0043 and TASK-0044.
        raise FasterRagError(
            f"{tool!r} cannot be provisioned yet; supported: {', '.join(sorted(_PROVISIONABLE))}",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )


@router.post("/admin/provision/{tool}")
async def provision(tool: str, settings: CurrentSettings) -> dict[str, Any]:
    """Provision a managed dependency. Idempotent and doctor-gated by the service."""
    _require_provisionable(tool)
    result = await provision_qdrant(settings)
    return {"tool": result.tool, "status": result.status, "url": result.url}


@router.get("/admin/provision/{tool}/status")
async def provision_status(tool: str, settings: CurrentSettings) -> dict[str, Any]:
    """Report a managed dependency's provisioning state."""
    _require_provisionable(tool)
    result = await qdrant_status(settings)
    return {
        "tool": result.tool,
        "status": result.status,
        "url": result.url,
        "detail": result.detail,
    }


@router.post("/estimate")
async def estimate(body: EstimateRequest, settings: CurrentSettings) -> dict[str, Any]:
    """Report what ingesting sources would cost, before embedding any of them (D9).

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` when ``cost.estimator`` is false, so the
            setting reaches the REST surface it names and not only the CLI one.
    """
    require_estimator(settings)
    return estimate_sources(body.sources, settings, all_providers=body.all_providers).as_dict()


@router.post("/admin/export")
async def export_collection(
    body: ExportRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Write a collection to a portable archive (D11).

    Synchronous rather than a job id, unlike ingestion: an export reads what is already
    indexed and writes a file, so there is no queue it could be waiting behind and no
    partial state a caller would need to poll.
    """
    collection = scoped_name(body.collection or settings.vector_db.collection.default_name, tenant)
    store = create_lock_store(settings)
    lock = store.read(collection) if store is not None and store.enabled else None

    counts = await export_archive(
        settings,
        adapter,
        collection=collection,
        destination=Path(body.out),
        include_vectors=body.include_vectors,
        lock=lock,
        tenant=tenant,
    )
    return {"collection": unscoped_name(collection, tenant), **counts.as_dict()}


@router.post("/admin/import")
async def import_collection(
    body: ImportRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Import a previously exported archive (D11).

    Verification runs before anything is written, so a refused archive leaves the target
    untouched rather than half-populated.
    """
    reader = open_archive(Path(body.archive))
    target = scoped_name(body.target_collection or reader.collection, tenant)

    router_ = build_embedding_router(settings) if body.reembed else None
    try:
        counts = await import_archive(
            settings,
            adapter,
            reader,
            collection=target,
            reembed=body.reembed,
            router=router_,
        )
    finally:
        if router_ is not None:
            await router_.close()

    return {
        "collection": unscoped_name(target, tenant),
        "reembed": body.reembed,
        **counts.as_dict(),
    }
