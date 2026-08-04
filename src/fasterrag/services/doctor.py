"""Preflight diagnostics (D10).

``fasterrag doctor`` turns environment problems into named checks with concrete fixes
instead of stack traces, and it is what makes one-toggle provisioning survivable on an
arbitrary machine (``docs/differentiators.md`` D10). Every failing check carries a
``fix`` string; a check without one is a bug.

Checks marked ``blocks_provisioning`` are the environment preconditions that
provisioning refuses to run without. Backend reachability is deliberately not one of
them: before the backend is provisioned it is *supposed* to be unreachable, and gating
on it would make a cold start impossible.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import psutil

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.loader import DEFAULT_CONFIG_PATH, DEFAULT_ENV_FILE, load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.provisioning import (
    QDRANT_CONTAINER,
    container_state,
    docker_available,
    port_is_free,
    port_is_reachable,
)

__all__ = ["DoctorCheck", "DoctorReport", "diagnose", "run_doctor"]

MINIMUM_PYTHON: Final = (3, 12)
MINIMUM_FREE_DISK_GB: Final = 5.0
MINIMUM_AVAILABLE_MEMORY_GB: Final = 4.0

_BYTES_PER_GB: Final = 1024**3
_GPU_PROBE_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One preflight check and, when it fails, how to fix it."""

    name: str
    passed: bool
    detail: str
    fix: str = ""
    blocks_provisioning: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form used by ``--json`` and the admin endpoint."""
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "fix": self.fix,
            "blocks_provisioning": self.blocks_provisioning,
        }


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The full preflight result."""

    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[DoctorCheck]:
        """Return the failing checks."""
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable report."""
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }


def _check_python() -> DoctorCheck:
    """Verify the interpreter meets the documented minimum."""
    current = sys.version_info[:2]
    required = ".".join(str(part) for part in MINIMUM_PYTHON)
    return DoctorCheck(
        name="python",
        passed=current >= MINIMUM_PYTHON,
        detail=f"running Python {current[0]}.{current[1]}, requires {required}+",
        fix=f"Install Python {required} or newer and recreate the virtual environment.",
    )


def _check_disk(path: Path) -> DoctorCheck:
    """Verify there is room for the index."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / _BYTES_PER_GB
    return DoctorCheck(
        name="disk",
        passed=free_gb >= MINIMUM_FREE_DISK_GB,
        detail=f"{free_gb:.1f} GB free at {path}, requires {MINIMUM_FREE_DISK_GB:.0f} GB",
        fix=(
            f"Free at least {MINIMUM_FREE_DISK_GB:.0f} GB, or move the index and Docker "
            "volumes to a larger disk."
        ),
        blocks_provisioning=True,
    )


def _check_memory() -> DoctorCheck:
    """Report available RAM against the documented starting point."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / _BYTES_PER_GB
    total_gb = memory.total / _BYTES_PER_GB
    return DoctorCheck(
        name="memory",
        passed=available_gb >= MINIMUM_AVAILABLE_MEMORY_GB,
        detail=(
            f"{available_gb:.1f} GB available of {total_gb:.1f} GB total, "
            f"recommends {MINIMUM_AVAILABLE_MEMORY_GB:.0f} GB"
        ),
        fix=(
            "Close other workloads, or lower workers.embedding_pool_size and "
            "embeddings.batch_size to shrink the pipeline's footprint."
        ),
    )


async def _check_gpu() -> DoctorCheck:
    """Report GPU availability. Never fails — CPU embedding is a supported path."""
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        return DoctorCheck(
            name="gpu",
            passed=True,
            detail="no NVIDIA GPU detected; embedding will run on CPU",
        )

    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=_GPU_PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return DoctorCheck(
            name="gpu", passed=True, detail="nvidia-smi did not respond; assuming CPU embedding"
        )

    names = [line for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    if not names:
        return DoctorCheck(
            name="gpu", passed=True, detail="no NVIDIA GPU detected; embedding will run on CPU"
        )
    return DoctorCheck(name="gpu", passed=True, detail=f"GPU available: {', '.join(names)}")


async def _check_docker(settings: Settings) -> DoctorCheck:
    """Verify Docker is running whenever configuration depends on it."""
    needs_docker = settings.vector_db.mode == "docker" or any(
        (
            settings.observability.langfuse,
            settings.observability.grafana,
        )
    )
    if not needs_docker:
        return DoctorCheck(
            name="docker",
            passed=True,
            detail="not required: no configuration option depends on Docker",
        )

    available = await docker_available()
    return DoctorCheck(
        name="docker",
        passed=available,
        detail="docker daemon is responding" if available else "docker daemon is not responding",
        fix="Start Docker (Docker Desktop or the docker service) and run doctor again.",
        blocks_provisioning=True,
    )


async def _check_vector_db_ports(settings: Settings) -> DoctorCheck:
    """Check the REST and gRPC ports separately.

    Exposing only 6333 while a client attempts gRPC is a documented failure
    (``docs/failure-modes.md`` row 15), so each port is reported by name.
    """
    vector_db = settings.vector_db
    ports = {"REST": vector_db.port, "gRPC": vector_db.grpc_port}

    if vector_db.mode == "external":
        unreachable = [
            f"{label} {port}"
            for label, port in ports.items()
            if not port_is_reachable(vector_db.host, port)
        ]
        return DoctorCheck(
            name="vector_db_ports",
            passed=not unreachable,
            detail=(
                f"{vector_db.host}: all ports reachable"
                if not unreachable
                else f"{vector_db.host}: unreachable {', '.join(unreachable)}"
            ),
            fix=(
                "Start Qdrant and publish BOTH 6333 (REST) and 6334 (gRPC). Clients that "
                "attempt gRPC fail when only 6333 is exposed."
            ),
        )

    state = await container_state()
    if state.running:
        return DoctorCheck(
            name="vector_db_ports",
            passed=True,
            detail=f"ports held by the managed container {QDRANT_CONTAINER!r}",
            blocks_provisioning=True,
        )

    taken = [
        f"{label} {port}" for label, port in ports.items() if not port_is_free(vector_db.host, port)
    ]
    return DoctorCheck(
        name="vector_db_ports",
        passed=not taken,
        detail=("both ports are free" if not taken else f"already in use: {', '.join(taken)}"),
        fix=(
            "Stop whatever holds the port, or change vector_db.port / vector_db.grpc_port. "
            f"A stale container can be removed with 'docker rm -f {QDRANT_CONTAINER}'."
        ),
        blocks_provisioning=True,
    )


def _check_api_ports(settings: Settings) -> list[DoctorCheck]:
    """Check the ports fasterRag's own processes bind."""
    checks = [
        DoctorCheck(
            name="api_port",
            passed=port_is_free(settings.app.host, settings.app.port),
            detail=f"api port {settings.app.port} on {settings.app.host}",
            fix=(
                f"Free port {settings.app.port} or change app.port. It reads as in use "
                "when the API is already running."
            ),
        )
    ]
    if settings.observability.dashboard:
        port = settings.observability.dashboard_port
        checks.append(
            DoctorCheck(
                name="dashboard_port",
                passed=port_is_free(settings.app.host, port),
                detail=f"dashboard port {port}",
                fix=f"Free port {port} or change observability.dashboard_port.",
            )
        )
    return checks


def _check_secrets(settings: Settings) -> DoctorCheck:
    """Verify every environment variable the configuration references is populated."""
    missing = [
        f"{name} ({config_key})"
        for name, config_key in sorted(settings.referenced_env_vars().items())
        if not (os.environ.get(name) or "").strip()
    ]
    return DoctorCheck(
        name="secrets",
        passed=not missing,
        detail=(
            "all referenced variables are set" if not missing else f"missing: {', '.join(missing)}"
        ),
        fix="Set the named variables in .env. Never put the values in config.yaml.",
    )


async def _check_vector_db_reachable(settings: Settings) -> DoctorCheck:
    """Ask the adapter whether the backend answers, which also validates the API key."""
    try:
        adapter = create_vector_db_adapter(settings)
    except FasterRagError as exc:
        return DoctorCheck(
            name="vector_db",
            passed=False,
            detail=exc.detail,
            fix="Set vector_db.provider to a built-in provider, or install its plugin.",
        )

    try:
        status = await adapter.health()
    finally:
        await adapter.close()

    return DoctorCheck(
        name="vector_db",
        passed=status.healthy,
        detail=(
            f"reachable in {status.latency_ms} ms"
            if status.healthy
            else status.detail or "not reachable"
        ),
        fix=(
            "Run 'fasterrag provision qdrant' for a managed container, or start the "
            "external instance and check vector_db.host, the ports, and the API key."
        ),
    )


async def run_doctor(settings: Settings) -> DoctorReport:
    """Run every preflight check against validated settings.

    Args:
        settings: The configuration to diagnose.

    Returns:
        The report. Running doctor never raises for a failing environment — the failure
        is the result.
    """
    checks: list[DoctorCheck] = [
        DoctorCheck(name="config", passed=True, detail="configuration is valid"),
        _check_python(),
        _check_disk(Path.cwd()),
        _check_memory(),
        await _check_gpu(),
        await _check_docker(settings),
        _check_secrets(settings),
        await _check_vector_db_ports(settings),
        await _check_vector_db_reachable(settings),
    ]
    checks.extend(_check_api_ports(settings))
    return DoctorReport(checks=checks)


async def diagnose(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    overrides: Sequence[str] | None = None,
) -> DoctorReport:
    """Diagnose an installation starting from its configuration file.

    Invalid configuration is reported as a failed check rather than an exception, so
    ``fasterrag doctor`` stays useful precisely when configuration is the problem.

    Args:
        config_path: The configuration file to diagnose.
        env_file: Optional ``.env`` loaded before the presence check.
        overrides: ``--set`` overrides, so doctor checks the configuration the operator is
            about to run rather than the one on disk.
    """
    try:
        settings = load_settings(config_path, env_file=env_file, overrides=overrides)
    except ConfigError as exc:
        return DoctorReport(
            checks=[
                DoctorCheck(
                    name="config",
                    passed=False,
                    detail=exc.detail,
                    fix="Fix the named keys, then run 'fasterrag config validate'.",
                    blocks_provisioning=True,
                )
            ]
        )

    return await run_doctor(settings)


def format_report(report: DoctorReport) -> Sequence[str]:
    """Render the report as human-readable lines, fixes included."""
    lines: list[str] = []
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(f"[{marker}] {check.name}: {check.detail}")
        if not check.passed and check.fix:
            lines.append(f"       fix: {check.fix}")
    return lines
