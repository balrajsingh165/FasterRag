"""The commands that run a long-lived process: ``serve`` and ``worker``.

Both block until interrupted, so neither returns a meaningful exit code in normal use — the
code they do return distinguishes "refused to start" from "ran and was stopped", which is
what a supervisor restarts on.

Configuration is validated *before* either process starts. Failing at startup on a bad
config is the fail-fast contract of ``docs/config-reference.md``: a server that boots with
half-valid configuration fails later, under load, in a way that looks like a runtime bug.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from fasterrag.api.main import CONFIG_PATH_VAR, create_app
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.settings import settings_from
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import create_journal

__all__ = ["run_serve", "run_worker"]


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

        dashboard_port = settings.observability.dashboard_port
        console.emit(f"dashboard on http://{host}:{dashboard_port} (read-only)")
        servers.append(
            uvicorn.Server(
                uvicorn.Config(
                    create_dashboard(settings),
                    host=host,
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


async def run_worker(args: argparse.Namespace, console: Console) -> ExitCode:
    """Run the pipeline worker pools until interrupted.

    The pools are owned by the ingestion service rather than started here, so a job runs
    identically whether it arrived through the API, the CLI, or the library. This command
    validates the environment, reports the pool sizes it would run, and waits.
    """
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    pools = [pool.strip() for pool in args.pools.split(",") if pool.strip()]
    unknown = set(pools) - {"cpu", "embed", "index"}
    if unknown:
        console.error(f"unknown pool(s): {', '.join(sorted(unknown))}")
        return ExitCode.USAGE

    service = IngestionService(settings, journal=create_journal(settings))
    try:
        health = await service.adapter.health()
        if not health.healthy:
            console.problem(
                "VECTOR_DB_UNAVAILABLE",
                f"the vector database is not answering: {health.detail}",
                "start it with 'fasterrag provision qdrant', or check vector_db.host",
            )
            return ExitCode.UNREACHABLE

        cpu = args.cpu_workers or settings.workers.cpu_pool_size
        embed = args.embed_workers or settings.workers.embedding_pool_size
        console.emit(f"worker ready: pools={','.join(pools)} cpu={cpu} embed={embed}")
        console.document({"pools": pools, "cpu": cpu, "embed": embed, "status": "ready"})

        await asyncio.Event().wait()
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()

    return ExitCode.SUCCESS
