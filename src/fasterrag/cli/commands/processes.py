"""The command that runs a long-lived process: ``serve``.

It blocks until interrupted, so it returns no meaningful exit code in normal use — the
code it does return distinguishes "refused to start" from "ran and was stopped", which is
what a supervisor restarts on.

Ingestion runs inside this process: there is no separate worker process, and the ``worker``
command that only checked the vector database and waited was removed (TASK-0130 tracks a
queue-backed one).

``BUILTIN_PROCESSES`` maps every process surface the documents may name to the callable that
implements it, and ``PLANNED_PROCESSES`` names the ones that are not built. The CLI dispatches
``serve`` through the first, and ``scripts/check_unbuilt_claims.py`` reads both with ``ast``,
the way it reads ``_BUILTIN_RUNGS``: a document that tells an operator to run a process the
registry does not hold fails the gate (TASK-0440).

Configuration is validated *before* the process starts. Failing at startup on a bad
config is the fail-fast contract of ``docs/config-reference.md``: a server that boots with
half-valid configuration fails later, under load, in a way that looks like a runtime bug.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
from collections.abc import Awaitable, Callable
from typing import Any, Final

from fasterrag.api.main import CONFIG_PATH_VAR, create_app
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.settings import settings_from
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError

__all__ = ["BUILTIN_PROCESSES", "PLANNED_PROCESSES", "exposed_without_auth", "run_serve"]


def exposed_without_auth(host: str, settings: Settings) -> bool:
    """Return whether this bind accepts unauthenticated requests from other machines."""
    if settings.security.auth:
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() != "localhost"


async def run_serve(args: argparse.Namespace, console: Console) -> ExitCode:
    """Run the API server until interrupted."""
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    import uvicorn

    host = args.host or settings.app.host
    port = args.port or settings.app.port
    console.emit(f"serving on http://{host}:{port} (config: {args.config})")
    if exposed_without_auth(host, settings):
        console.error(
            f"WARNING: binding {host} with security.auth off. Anyone who can reach this "
            "port can ingest files readable by this process and query them back. Set "
            "security.auth: true and ingestion.allowed_source_roots, or bind 127.0.0.1"
        )

    if args.reload:
        # CRITICAL: reload runs the application in a child process, which cannot receive an
        # in-memory Settings object. The path is handed over through the environment instead,
        # because the alternative — an import string with no configuration attached — would
        # silently serve ./config.yaml no matter what --config said.
        os.environ[CONFIG_PATH_VAR] = str(args.config)
        target: Any = "fasterrag.api.main:create_app"
    else:
        target = create_app(settings)

    config = uvicorn.Config(
        target,
        factory=args.reload,
        host=host,
        port=port,
        reload=args.reload,
        log_level=settings.app.log_level,
    )

    servers = [uvicorn.Server(config)]
    if settings.observability.dashboard:
        from fasterrag.observability.dashboard import create_dashboard

        dashboard_host = settings.observability.dashboard_host
        dashboard_port = settings.observability.dashboard_port
        console.emit(f"dashboard on http://{dashboard_host}:{dashboard_port} (read-only)")
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    create_dashboard(settings),
                    host=dashboard_host,
                    port=dashboard_port,
                    log_level=settings.app.log_level,
                )
            )
        )

    # CRITICAL: the dashboard runs beside the API rather than mounted into it. Sharing a
    # port would take away the operator's ability to bind the two differently, and the
    # dashboard displays prompts, responses, and corpus text — content that usually belongs
    # on an internal interface while the API faces callers.
    await asyncio.gather(*(server.serve() for server in servers))
    return ExitCode.SUCCESS


BUILTIN_PROCESSES: Final[
    dict[str, Callable[[argparse.Namespace, Console], Awaitable[ExitCode]]]
] = {
    "serve": run_serve,
}

# TODO: TASK-0130 builds a queue-backed worker. Move "worker" into BUILTIN_PROCESSES with its
# implementation when it ships, and out of REMOVED_COMMANDS in cli/parser.py, rather than
# deleting it here: this set is what makes every "the worker was removed" sentence decidable.
PLANNED_PROCESSES: Final[frozenset[str]] = frozenset({"worker"})
