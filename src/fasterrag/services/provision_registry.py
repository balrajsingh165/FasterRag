"""The one table of what fasterRag can provision, and the three verbs each tool answers.

Both control planes dispatch through here. Before this module existed each kept its own
list: the CLI's ``if args.tool == "langfuse"`` chain grew Langfuse and Grafana when those
provisioners shipped, and the admin router's ``_PROVISIONABLE = {"qdrant"}`` did not — so
``POST /v1/admin/provision/langfuse`` answered ``NOT_FOUND`` from 2026-08-02, when both
provisioners landed, until 2026-08-11, while ``docs/api-reference.md`` documented all three
as supported the whole time (TASK-0251).

Neither an endpoint test nor the OpenAPI drift gate could see it: the route existed and was
served, and the disagreement lived in a value inside it. What stops the next one is that
there is now nothing to keep in sync — the CLI's ``choices=``, the REST surface, and the
error message that lists what is supported all read this table.

Provisioning is *not* tenant-scoped and deliberately takes no tenant: a container is one
process-wide resource, so scoping the request would imply a per-tenant stack that does not
exist. Authorisation is the ``admin`` scope, applied by the middleware to the whole
``/v1/admin`` prefix (``docs/security.md`` §2).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.grafana import grafana_status, provision_grafana, stop_grafana
from fasterrag.services.langfuse import langfuse_status, provision_langfuse, stop_langfuse
from fasterrag.services.provisioning import (
    ProvisionResult,
    provision_qdrant,
    qdrant_status,
    stop_qdrant,
)

__all__ = [
    "PROVISIONABLE_TOOLS",
    "ToolCommands",
    "provision_tool",
    "stop_tool",
    "tool_status",
]

ToolCommand = Callable[[Settings], Awaitable[ProvisionResult]]


@dataclass(frozen=True, slots=True)
class ToolCommands:
    """The three things one provisionable tool can be asked to do."""

    provision: ToolCommand
    status: ToolCommand
    stop: ToolCommand


async def _langfuse_status(_settings: Settings) -> ProvisionResult:
    """Report the Langfuse stack's state; where it lives is fixed, so nothing is read."""
    return await langfuse_status()


async def _stop_langfuse(_settings: Settings) -> ProvisionResult:
    """Stop the Langfuse stack, preserving its volumes and its secrets."""
    return await stop_langfuse()


async def _grafana_status(_settings: Settings) -> ProvisionResult:
    """Report the Grafana and Prometheus containers' state."""
    return await grafana_status()


async def _stop_grafana(_settings: Settings) -> ProvisionResult:
    """Stop the Grafana and Prometheus containers, leaving their manifests in place."""
    return await stop_grafana()


_TOOLS: Final[Mapping[str, ToolCommands]] = {
    "qdrant": ToolCommands(provision_qdrant, qdrant_status, stop_qdrant),
    "langfuse": ToolCommands(provision_langfuse, _langfuse_status, _stop_langfuse),
    "grafana": ToolCommands(provision_grafana, _grafana_status, _stop_grafana),
}

PROVISIONABLE_TOOLS: Final[tuple[str, ...]] = tuple(_TOOLS)


def _commands_for(tool: str) -> ToolCommands:
    """Return the verbs a tool answers.

    Raises:
        FasterRagError: With ``NOT_FOUND`` for an unknown tool, naming what is supported.
            The list comes from the table itself, so a refusal can never advertise a
            different set from the one that would be dispatched.
    """
    try:
        return _TOOLS[tool]
    except KeyError as exc:
        raise FasterRagError(
            f"{tool!r} cannot be provisioned; supported: {', '.join(PROVISIONABLE_TOOLS)}",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        ) from exc


async def provision_tool(tool: str, settings: Settings) -> ProvisionResult:
    """Bring a managed dependency to its configured state. Idempotent and doctor-gated."""
    return await _commands_for(tool).provision(settings)


async def tool_status(tool: str, settings: Settings) -> ProvisionResult:
    """Report a managed dependency's state without changing anything."""
    return await _commands_for(tool).status(settings)


async def stop_tool(tool: str, settings: Settings) -> ProvisionResult:
    """Stop a managed dependency's containers, preserving their data."""
    return await _commands_for(tool).stop(settings)
