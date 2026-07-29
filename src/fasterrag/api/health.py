"""Liveness and readiness endpoints.

``/healthz`` answers whether the process is alive and checks nothing else.
``/readyz`` actually exercises dependencies and returns a problem document listing the
failing ones, so a sick backend is visible before queries start failing
(``docs/api-reference.md``, ``docs/reliability.md`` §7).

Checks are registered rather than hard-coded: each slice adds the dependency it
introduces (vector DB health, queue backend, circuit-breaker state), and ``/readyz``
reports exactly what is actually verifiable at that point.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from fasterrag.api.problems import problem_response
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["DependencyStatus", "ReadinessCheck", "ReadinessRegistry", "router"]


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Outcome of one readiness check."""

    name: str
    ready: bool
    detail: str | None = None


ReadinessCheck = Callable[[], Awaitable[DependencyStatus]]


class ReadinessRegistry:
    """The dependency checks ``/readyz`` consults, in registration order."""

    def __init__(self) -> None:
        """Start with no registered checks."""
        self._checks: list[ReadinessCheck] = []

    def register(self, check: ReadinessCheck) -> None:
        """Add a dependency check."""
        self._checks.append(check)

    async def run(self) -> list[DependencyStatus]:
        """Run every registered check and return their statuses."""
        return [await check() for check in self._checks]


router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Report liveness: the process is up and able to answer. No dependencies checked."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    """Report readiness after actually checking every registered dependency.

    Returns:
        ``200`` with the per-dependency report when all checks pass, otherwise ``503``
        with a problem document naming the failing dependencies.
    """
    registry = cast(ReadinessRegistry, request.app.state.readiness)
    statuses = await registry.run()
    report = [
        {"name": status.name, "ready": status.ready, "detail": status.detail} for status in statuses
    ]

    failing = [status.name for status in statuses if not status.ready]
    if failing:
        error = FasterRagError(
            f"dependencies are not ready: {', '.join(failing)}",
            code=ErrorCode.NOT_READY,
        )
        return problem_response(
            error,
            instance=request.url.path,
            extensions={"dependencies": report},
        )

    return JSONResponse(content={"status": "ready", "dependencies": report})
