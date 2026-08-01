"""Administrative endpoints: doctor, provisioning, and preflight estimation.

The REST half of what ``fasterrag doctor``, ``fasterrag provision``, and ``fasterrag
estimate`` do, calling the same services so the two control planes report the same state.

``GET /v1/admin/doctor`` returns ``200`` even when checks fail. The failing checks *are* the
payload; a ``503`` would make the report unreadable to exactly the client that most needs it,
and the endpoint's own availability is not what it is reporting on.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from fasterrag.api.dependencies import CurrentSettings
from fasterrag.api.schemas import EstimateRequest
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.doctor import run_doctor
from fasterrag.services.estimation import estimate_sources
from fasterrag.services.provisioning import provision_qdrant, qdrant_status

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
    """Report what ingesting sources would cost, before embedding any of them (D9)."""
    return estimate_sources(body.sources, settings, all_providers=body.all_providers).as_dict()
