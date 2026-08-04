"""The commands that inspect an installation: ``config validate``, ``doctor``, ``status``.

These are the three commands that must work when nothing else does. Each reports a broken
environment as a *result* rather than an exception, because a diagnostic that crashes on the
problem it exists to find is useless precisely when it matters.

They also own the distinct exit codes an operator scripts against: ``2`` for invalid
configuration, ``4`` for a failed preflight, ``3`` for a dependency that cannot be reached.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.output import Console, ExitCode
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.config.template import canonical_config_text, env_template_text
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.doctor import diagnose, format_report
from fasterrag.services.provisioning import container_state, docker_available

__all__ = [
    "run_config_init",
    "run_config_show",
    "run_config_validate",
    "run_doctor_command",
    "run_status",
]


async def run_config_init(args: argparse.Namespace, console: Console) -> ExitCode:
    """Write the canonical ``config.yaml`` into the current directory.

    The first command a ``pip install`` user has any reason to run. Without it every other
    command fails on a missing file whose only documented source is a repository they do not
    have.

    An existing file is never overwritten without ``--force``: a config.yaml is hand-edited
    the moment it exists, and silently replacing one would discard exactly the work the
    operator most wants to keep.
    """
    destination = Path(args.path)

    if destination.exists() and not args.force:
        console.error(
            f"{destination} already exists; pass --force to overwrite it, or --path to write "
            "somewhere else"
        )
        console.document({"written": False, "path": str(destination), "reason": "exists"})
        return ExitCode.USAGE

    try:
        body = canonical_config_text()
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    console.emit(f"wrote {destination}")

    # CRITICAL: the secrets template is written as `.env.example`, never as `.env`. The
    # loader reads `.env`, so writing there could overwrite an operator's real credentials
    # with placeholders — and a file full of `change-me` values that the system treats as
    # configured is worse than no file.
    example = destination.parent / ".env.example"
    wrote_example = not example.exists()
    if wrote_example:
        example.write_text(env_template_text(), encoding="utf-8")
        console.emit(f"wrote {example}")

    console.emit(f"next: copy {example.name} to .env, fill in what config.yaml references,")
    console.emit("      then run 'fasterrag doctor'")
    console.document(
        {
            "written": True,
            "path": str(destination),
            "env_example": str(example) if wrote_example else None,
        }
    )
    return ExitCode.SUCCESS


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


def _flatten(section: BaseModel, prefix: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(dotted_name, value, default)`` for every leaf setting under ``section``.

    Walks nested sections rather than listing fields by hand, so a setting added to the
    schema shows up here without anyone remembering to register it.
    """
    for name, field in type(section).model_fields.items():
        value = getattr(section, name)
        dotted = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            yield from _flatten(value, f"{dotted}.")
            continue

        default = field.get_default(call_default_factory=True)
        yield dotted, value, default


def _redacted(name: str, value: Any) -> Any:
    """Return the value, or a placeholder when the name suggests it carries a secret.

    # CRITICAL: config.yaml holds env-var *names*, never secret values, so nothing here
    # should be sensitive. This is a second line of defence for an operator who put a key
    # in the file anyway — printing it would then leak it into terminal scrollback,
    # screenshots, and pasted bug reports.
    """
    lowered = name.lower()
    if value and any(word in lowered for word in ("password", "secret", "token_value")):
        return "<redacted>"
    return value


async def run_config_show(args: argparse.Namespace, console: Console) -> ExitCode:
    """Print every setting with its effective value and its default.

    Answers "what can I change, and what is it right now?" without reading the schema.
    ``--changed`` narrows the listing to settings that differ from their default, which is
    the fastest way to see what a deployment actually customised.

    Missing environment variables do not stop the listing. This command is most useful on
    the half-configured installation that ``config validate`` refuses, and an operator
    reaching for it wants to see the settings, not be told again that a key is unset.
    """
    try:
        settings = load_settings(args.config, require_env=False)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    rows = [
        {"setting": name, "value": _redacted(name, value), "default": _redacted(name, default)}
        for name, value, default in _flatten(settings)
        if not args.changed or value != default
    ]

    if not rows:
        console.emit(f"{args.config} matches every default")
    for row in rows:
        marker = " " if row["value"] == row["default"] else "*"
        console.emit(
            f"{marker} {row['setting']:<48} {row['value']!r:<24} default={row['default']!r}"
        )

    console.document({"config": args.config, "settings": rows})
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
