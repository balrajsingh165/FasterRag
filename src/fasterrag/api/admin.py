"""Administrative endpoints: doctor, provisioning, and preflight estimation.

The REST half of what ``fasterrag doctor``, ``fasterrag provision``, and ``fasterrag
estimate`` do, calling the same services so the two control planes report the same state.

``GET /v1/admin/doctor`` is read-only and ``POST /v1/admin/doctor/fix`` is the repairing
verb beside it, which is the same split the CLI draws between ``doctor`` and ``doctor
--fix``. Both return ``200`` even when checks fail. The failing checks *are* the
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
from fasterrag.services.archive import export_archive
from fasterrag.services.archive_import import import_archive, open_archive
from fasterrag.services.doctor import apply_fixes, run_doctor
from fasterrag.services.estimation import estimate_sources, require_estimator
from fasterrag.services.lockfile import create_lock_store
from fasterrag.services.provision_registry import provision_tool, stop_tool, tool_status
from fasterrag.services.provisioning import ProvisionResult
from fasterrag.services.tenancy import scoped_name, unscoped_name

__all__ = ["router"]

router = APIRouter(prefix="/v1", tags=["admin"])


@router.get("/admin/doctor")
async def doctor_report(settings: CurrentSettings) -> dict[str, Any]:
    """Return the machine-readable preflight report (D10)."""
    return (await run_doctor(settings)).as_dict()


@router.post("/admin/doctor/fix")
async def doctor_fix(settings: CurrentSettings) -> dict[str, Any]:
    """Repair what is safe to repair, then report the machine as it stands (TASK-0258)."""
    return (await apply_fixes(settings)).as_dict()


def _reported(result: ProvisionResult) -> dict[str, Any]:
    """Render a provisioning outcome for the wire."""
    return {
        "tool": result.tool,
        "status": result.status,
        "url": result.url,
        "detail": result.detail,
    }


@router.post("/admin/provision/{tool}")
async def provision(tool: str, settings: CurrentSettings) -> dict[str, Any]:
    """Provision a managed dependency. Idempotent and doctor-gated by the service.

    Which tools exist is decided by ``services/provision_registry``, the same table the CLI
    dispatches through, so this surface cannot offer less than ``fasterrag provision`` again.
    """
    return _reported(await provision_tool(tool, settings))


@router.get("/admin/provision/{tool}/status")
async def provision_status(tool: str, settings: CurrentSettings) -> dict[str, Any]:
    """Report a managed dependency's provisioning state."""
    return _reported(await tool_status(tool, settings))


@router.delete("/admin/provision/{tool}")
async def provision_down(tool: str, settings: CurrentSettings) -> dict[str, Any]:
    """Stop a managed dependency's containers, the REST half of ``provision <tool> --down``."""
    return _reported(await stop_tool(tool, settings))


@router.post("/estimate")
async def estimate(body: EstimateRequest, settings: CurrentSettings) -> dict[str, Any]:
    """Report what ingesting sources would cost, before embedding any of them (D9)."""
    require_estimator(settings)
    return estimate_sources(body.sources, settings, all_providers=body.all_providers).as_dict()


@router.post("/admin/export")
async def export_collection(
    body: ExportRequest,
    settings: CurrentSettings,
    adapter: CurrentVectorDB,
    tenant: CurrentTenant,
) -> dict[str, Any]:
    """Write a collection to a portable archive (D11)."""
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
