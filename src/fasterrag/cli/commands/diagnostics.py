"""The commands that inspect an installation: ``config validate``, ``doctor``, ``status``.

These are the three commands that must work when nothing else does. Each reports a broken
environment as a *result* rather than an exception, because a diagnostic that crashes on the
problem it exists to find is useless precisely when it matters.

They also own the distinct exit codes an operator scripts against: ``2`` for invalid
configuration, ``4`` for a failed preflight, ``3`` for a dependency that cannot be reached.
"""

from __future__ import annotations

import argparse
from typing import Any

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.output import Console, ExitCode
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.doctor import diagnose, format_report
from fasterrag.services.provisioning import container_state, docker_available

__all__ = ["run_config_validate", "run_doctor_command", "run_status"]


async def run_config_validate(args: argparse.Namespace, console: Console) -> ExitCode:
    """Validate configuration without starting anything.

    Exits ``2`` on invalid configuration, matching the usage/validation code: a bad config
    file is an operator mistake, not a runtime failure, and CI branches on the difference.
    """
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        console.document({"valid": False, "config": args.config, "detail": exc.detail})
        return ExitCode.USAGE

    console.emit(f"{args.config} is valid")
    console.document(
        {
            "valid": True,
            "config": args.config,
            "vector_db": settings.vector_db.provider,
            "embeddings": settings.embeddings.provider,
            "llm": settings.llm.provider,
            "collection": settings.vector_db.collection.default_name,
        }
    )
    return ExitCode.SUCCESS


async def run_doctor_command(args: argparse.Namespace, console: Console) -> ExitCode:
    """Run preflight diagnostics.

    Exits ``4`` when any check fails. Doctor gates provisioning, so this code is what stops
    an automated setup from proceeding into an environment that cannot host it.
    """
    report = await diagnose(args.config)

    console.lines(format_report(report))
    console.document(report.as_dict())

    if report.passed:
        console.emit("all checks passed")
        return ExitCode.SUCCESS

    console.error(f"{len(report.failures)} check(s) failed")
    return ExitCode.PREFLIGHT


async def _vector_db_status(settings: Settings) -> dict[str, Any]:
    """Report the vector database's reachability without raising.

    A status command that raises when a dependency is down cannot report that the dependency
    is down, which is the one thing it was run to find out.
    """
    adapter = create_vector_db_adapter(settings)
    try:
        health = await adapter.health()
        return {
            "provider": settings.vector_db.provider,
            "healthy": health.healthy,
            "detail": health.detail,
        }
    except FasterRagError as exc:
        return {
            "provider": settings.vector_db.provider,
            "healthy": False,
            "detail": exc.detail,
            "code": exc.code.value,
        }
    finally:
        await adapter.close()


async def run_status(args: argparse.Namespace, console: Console) -> ExitCode:
    """Report one screen of system state.

    Exits ``3`` when a dependency is unreachable, so a script can distinguish "fasterRag is
    misconfigured" from "fasterRag is fine but Qdrant is down".
    """
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    vector_db = await _vector_db_status(settings)
    docker = await docker_available()
    container = await container_state() if docker else None

    payload: dict[str, Any] = {
        "collection": args.collection or settings.vector_db.collection.default_name,
        "vector_db": vector_db,
        "docker": {
            "available": docker,
            "container_exists": bool(container and container.exists),
            "container_running": bool(container and container.running),
        },
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model,
            "cache": settings.embeddings.cache.backend
            if settings.embeddings.cache.enabled
            else "off",
        },
        "llm": {"provider": settings.llm.provider, "model": settings.llm.model},
        "retrieval": {
            "hybrid": settings.retrieval.hybrid,
            "top_k": settings.retrieval.top_k,
            "rerank": settings.retrieval.rerank,
        },
        "cache": {
            "semantic": settings.cache.semantic,
            "backend": settings.cache.backend,
            "threshold": settings.cache.similarity_threshold,
        },
        "workers": {
            "cpu_pool_size": settings.workers.cpu_pool_size,
            "embedding_pool_size": settings.workers.embedding_pool_size,
            "queue_depth": settings.workers.queue_depth,
        },
    }

    console.emit(f"collection      {payload['collection']}")
    console.emit(
        f"vector db       {vector_db['provider']}: "
        f"{'healthy' if vector_db['healthy'] else 'UNREACHABLE'} — {vector_db['detail']}"
    )
    console.emit(
        f"docker          {'available' if docker else 'not available'}"
        + (
            f" (container {'running' if container.running else 'stopped'})"
            if container and container.exists
            else ""
        )
    )
    console.emit(f"embeddings      {settings.embeddings.provider}/{settings.embeddings.model}")
    console.emit(f"llm             {settings.llm.provider}/{settings.llm.model}")
    console.emit(
        f"retrieval       {'hybrid' if settings.retrieval.hybrid else 'dense'}, "
        f"top_k={settings.retrieval.top_k}, "
        f"rerank={'on' if settings.retrieval.rerank else 'off'}"
    )
    console.emit(
        f"semantic cache  {'on' if settings.cache.semantic else 'off'} ({settings.cache.backend})"
    )
    console.emit(
        f"workers         cpu={settings.workers.cpu_pool_size} "
        f"embed={settings.workers.embedding_pool_size} "
        f"queue={settings.workers.queue_depth}"
    )
    console.document(payload)

    return ExitCode.SUCCESS if vector_db["healthy"] else ExitCode.UNREACHABLE
