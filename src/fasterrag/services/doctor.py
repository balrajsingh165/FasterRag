"""Preflight diagnostics (D10).

``fasterrag doctor`` turns environment problems into named checks with concrete fixes
instead of stack traces, and it is what makes one-toggle provisioning survivable on an
arbitrary machine (``docs/differentiators.md`` D10). Every failing check carries a
``fix`` string; a check without one is a bug.

Checks marked ``blocks_provisioning`` are the environment preconditions that
provisioning refuses to run without. Backend reachability is deliberately not one of
them: before the backend is provisioned it is *supposed* to be unreachable, and gating
on it would make a cold start impossible.

``--fix`` repairs the failures that can be repaired safely and then runs every check
again, so what the operator reads is what the machine now reports rather than what was
attempted. A repair qualifies only if it is idempotent, cannot destroy state, and cannot
surprise: creating the named Docker volume and starting fasterRag's *own* stopped
container qualify, and nothing else does. Freeing a port means killing somebody's
process, freeing disk means deleting files fasterRag did not write, and configuration and
secrets are the operator's to author — each of those keeps its concrete fix-it string and
is reported as needing a human.

A check that could not run is never reported as passing. That is the same defect as a
``--fix`` claiming success without re-checking, so the volume check is omitted outright
when Docker is down rather than passed on a probe that never happened.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import psutil

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.config.loader import DEFAULT_CONFIG_PATH, DEFAULT_ENV_FILE, load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, FasterRagError, ProvisioningError
from fasterrag.observability.logging import get_logger
from fasterrag.services.provisioning import (
    CONTAINER_LABEL,
    QDRANT_CONTAINER,
    await_backend_ready,
    container_state,
    docker_available,
    ensure_volume,
    port_is_free,
    port_is_reachable,
    start_container,
    volume_exists,
)

__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "FixAttempt",
    "FixOutcome",
    "FixStatus",
    "apply_fixes",
    "diagnose",
    "diagnose_and_fix",
    "format_fix_outcome",
    "format_report",
    "run_doctor",
]

MINIMUM_PYTHON: Final = (3, 12)
MINIMUM_FREE_DISK_GB: Final = 5.0
MINIMUM_AVAILABLE_MEMORY_GB: Final = 4.0

_BYTES_PER_GB: Final = 1024**3
_GPU_PROBE_TIMEOUT_SECONDS: Final = 10.0

_logger = get_logger(__name__)


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


async def _check_vector_db_volume(settings: Settings, *, docker_ready: bool) -> DoctorCheck | None:
    """Report whether the named volume the managed container stores its index on exists.

    Returns ``None`` when the probe could not run — with the daemon down Docker cannot be
    asked, and answering "the volume is fine" from a probe that never happened would be a
    check passing because it could not run. The docker check already names that cause, and
    reporting one root cause twice sends an operator after two problems that are one.

    A missing volume earns a check of its own because it is otherwise invisible: nothing
    looks wrong until the container is replaced, and then the index is silently gone. It is
    also the failure ``--fix`` can repair outright, and a repair needs a check to hang off.

    It deliberately does not block provisioning: ``fasterrag provision qdrant`` creates the
    volume itself, so gating provisioning on it would refuse the command that fixes it.
    """
    vector_db = settings.vector_db
    if vector_db.mode != "docker":
        return DoctorCheck(
            name="vector_db_volume",
            passed=True,
            detail=f"not required: vector_db.mode is {vector_db.mode!r}",
        )

    if not docker_ready:
        return None

    volume = vector_db.docker.volume
    create_it = (
        f"Run 'fasterrag doctor --fix', or create it with 'docker volume create {volume}'. "
        "'fasterrag provision qdrant' creates it too."
    )
    try:
        exists = await volume_exists(volume)
    except ProvisioningError as exc:
        _logger.warning(
            "could not inspect the configured docker volume",
            extra={"volume": volume, "code": exc.code.value, "trace_id": exc.trace_id},
        )
        return DoctorCheck(
            name="vector_db_volume",
            passed=False,
            detail=f"could not inspect {volume!r}: {exc.detail}",
            fix=exc.fix or create_it,
        )

    return DoctorCheck(
        name="vector_db_volume",
        passed=exists,
        detail=(
            f"storage volume {volume!r} exists"
            if exists
            else f"storage volume {volume!r} does not exist yet"
        ),
        fix=create_it,
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
    docker = await _check_docker(settings)
    checks: list[DoctorCheck] = [
        DoctorCheck(name="config", passed=True, detail="configuration is valid"),
        _check_python(),
        _check_disk(Path.cwd()),
        _check_memory(),
        await _check_gpu(),
        docker,
        _check_secrets(settings),
        await _check_vector_db_ports(settings),
    ]
    volume = await _check_vector_db_volume(settings, docker_ready=docker.passed)
    if volume is not None:
        checks.append(volume)
    checks.append(await _check_vector_db_reachable(settings))
    checks.extend(_check_api_ports(settings))
    return DoctorReport(checks=checks)


FixStatus = Literal["fixed", "failed", "manual"]

_STATUS_LABEL: Final[dict[str, str]] = {
    "fixed": "fixed",
    "failed": "not fixed",
    "manual": "needs human",
}

_NO_SAFE_FIX: Final[dict[str, str]] = {
    "config": "rewriting config.yaml would discard the edits it was run to explain",
    "python": "the interpreter running this command cannot replace itself",
    "disk": "freeing space means deleting files fasterRag did not write",
    "memory": "freeing memory means killing another process",
    "gpu": "hardware cannot be installed by a diagnostic",
    "docker": "starting the Docker daemon is a machine-level action outside fasterRag",
    "secrets": "fasterRag never writes .env, and a credential cannot be invented",
    "vector_db_ports": "freeing a port means killing whatever holds it",
    "api_port": "freeing a port means killing whatever holds it",
    "dashboard_port": "freeing a port means killing whatever holds it",
}

_UNKNOWN_CHECK: Final = "no automatic repair is registered for this check"


@dataclass(frozen=True, slots=True)
class _Repair:
    """What a fixer did to one check, or why it refused to touch it."""

    outcome: Literal["attempted", "declined", "errored"]
    detail: str

    @property
    def ran(self) -> bool:
        """Return whether Docker was asked to change anything, successfully or not."""
        return self.outcome != "declined"


@dataclass(frozen=True, slots=True)
class FixAttempt:
    """What ``--fix`` did about one failing check, including deciding not to touch it.

    ``status`` is decided *after* the checks are run again, never from the repair
    returning without raising: a repair that appeared to work and left the check failing
    is reported as ``failed``, because the only evidence that counts is the re-check.
    """

    check: str
    status: FixStatus
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """Return the machine-readable form used by ``--json``."""
        return {"check": self.check, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FixOutcome:
    """A report, the repairs attempted against it, and the report taken again after them."""

    before: DoctorReport
    after: DoctorReport
    attempts: list[FixAttempt] = field(default_factory=list)
    rechecked: bool = False

    @property
    def passed(self) -> bool:
        """Return whether every check passes now."""
        return self.after.passed

    @property
    def repaired(self) -> list[str]:
        """Return the checks that were failing before the repairs and pass after them."""
        healthy = {check.name for check in self.after.checks if check.passed}
        return [check.name for check in self.before.failures if check.name in healthy]

    @property
    def still_failing(self) -> list[DoctorCheck]:
        """Return the checks that fail even after the repairs."""
        return self.after.failures

    def as_dict(self) -> dict[str, Any]:
        """Return the post-fix report extended with what fixing did.

        Deliberately a superset of ``DoctorReport.as_dict``: a consumer of
        ``fasterrag doctor --json`` finds ``passed`` and ``checks`` in the same place
        whether or not ``--fix`` was passed, and those describe the machine as it stands
        after the repairs rather than as it was found.
        """
        return {
            **self.after.as_dict(),
            "fixes": {
                "rechecked": self.rechecked,
                "attempts": [attempt.as_dict() for attempt in self.attempts],
                "repaired": self.repaired,
                "still_failing": [check.name for check in self.still_failing],
            },
        }


async def _fix_vector_db_volume(settings: Settings) -> _Repair:
    """Create the named Docker volume the managed container mounts its storage on."""
    volume = settings.vector_db.docker.volume
    await ensure_volume(volume)
    return _Repair("attempted", f"created the named Docker volume {volume!r}")


async def _fix_vector_db(settings: Settings) -> _Repair:
    """Start fasterRag's own stopped container, and nothing else.

    Every branch that is not "our container, stopped" declines with the command a human
    should run. Creating a container is deliberately out of scope even though it would
    make more reports go green: it pulls an image and publishes ports, which is
    ``fasterrag provision qdrant``'s job and far more than a diagnostic should do unasked.
    """
    if settings.vector_db.mode != "docker":
        return _Repair(
            "declined",
            f"vector_db.mode is {settings.vector_db.mode!r}, so this backend is not "
            "fasterRag's to start",
        )

    state = await container_state()
    if not state.exists:
        return _Repair(
            "declined",
            "no managed container exists yet, and creating one pulls an image and "
            "publishes ports — run 'fasterrag provision qdrant'",
        )
    if not state.managed:
        return _Repair(
            "declined",
            f"a container named {QDRANT_CONTAINER!r} exists but does not carry the "
            f"{CONTAINER_LABEL} label, so fasterRag did not create it and will not start it",
        )
    if state.running:
        return _Repair(
            "declined",
            "the managed container is already running, so the backend is unreachable for "
            "another reason — check the API key and whether vector_db.https matches the server",
        )

    await start_container(QDRANT_CONTAINER)
    await await_backend_ready(settings)
    return _Repair("attempted", f"started the stopped managed container {QDRANT_CONTAINER!r}")


_FIXERS: Final[dict[str, Callable[[Settings], Awaitable[_Repair]]]] = {
    "vector_db_volume": _fix_vector_db_volume,
    "vector_db": _fix_vector_db,
}


async def _repair(check: DoctorCheck, settings: Settings) -> _Repair:
    """Attempt the repair registered for one failing check, or decline with the reason."""
    fixer = _FIXERS.get(check.name)
    if fixer is None:
        return _Repair("declined", _NO_SAFE_FIX.get(check.name, _UNKNOWN_CHECK))

    try:
        return await fixer(settings)
    except FasterRagError as exc:
        _logger.warning(
            "a doctor repair failed",
            extra={"check": check.name, "code": exc.code.value, "trace_id": exc.trace_id},
        )
        return _Repair("errored", f"the repair failed: {exc.detail}")


async def apply_fixes(settings: Settings) -> FixOutcome:
    """Diagnose, repair what is safe to repair, then diagnose again.

    The second diagnosis is the point of the command. Without it ``--fix`` could only
    report what it tried, and "created the volume" is not evidence that the check now
    passes — a gate that reports success it did not verify is the defect this exists to
    avoid.

    Args:
        settings: The configuration to diagnose and repair.

    Returns:
        Both reports and one attempt per failing check, each classified from the re-check.
    """
    before = await run_doctor(settings)
    repairs = [(check.name, await _repair(check, settings)) for check in before.failures]

    if not any(repair.ran for _, repair in repairs):
        return FixOutcome(
            before=before,
            after=before,
            attempts=[
                FixAttempt(check=name, status="manual", detail=repair.detail)
                for name, repair in repairs
            ],
            rechecked=False,
        )

    after = await run_doctor(settings)
    healthy = {check.name for check in after.checks if check.passed}

    attempts: list[FixAttempt] = []
    for name, repair in repairs:
        if repair.outcome == "declined":
            attempts.append(FixAttempt(check=name, status="manual", detail=repair.detail))
        elif name in healthy:
            attempts.append(FixAttempt(check=name, status="fixed", detail=repair.detail))
        elif repair.outcome == "errored":
            attempts.append(FixAttempt(check=name, status="failed", detail=repair.detail))
        else:
            attempts.append(
                FixAttempt(
                    check=name,
                    status="failed",
                    detail=f"{repair.detail}, but the check still fails",
                )
            )

    return FixOutcome(before=before, after=after, attempts=attempts, rechecked=True)


def _config_failure(exc: ConfigError) -> DoctorReport:
    """Return the one-check report for configuration that would not load."""
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
        return _config_failure(exc)

    return await run_doctor(settings)


async def diagnose_and_fix(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    overrides: Sequence[str] | None = None,
) -> FixOutcome:
    """Diagnose an installation, repair what is safe to repair, and check it again.

    Configuration that will not load is answered as needing a human rather than rewritten.
    ``config.yaml`` is hand-authored, and a diagnostic that edited it back into shape would
    destroy the intent it was run to explain.

    Args:
        config_path: The configuration file to diagnose.
        env_file: Optional ``.env`` loaded before the presence check.
        overrides: ``--set`` overrides, so the repairs apply to the configuration the
            operator is about to run rather than the one on disk.
    """
    try:
        settings = load_settings(config_path, env_file=env_file, overrides=overrides)
    except ConfigError as exc:
        report = _config_failure(exc)
        return FixOutcome(
            before=report,
            after=report,
            attempts=[FixAttempt(check="config", status="manual", detail=_NO_SAFE_FIX["config"])],
            rechecked=False,
        )

    return await apply_fixes(settings)


def format_report(report: DoctorReport) -> Sequence[str]:
    """Render the report as human-readable lines, fixes included."""
    lines: list[str] = []
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(f"[{marker}] {check.name}: {check.detail}")
        if not check.passed and check.fix:
            lines.append(f"       fix: {check.fix}")
    return lines


def format_fix_outcome(outcome: FixOutcome) -> Sequence[str]:
    """Render what ``--fix`` did: what was wrong, what was attempted, and what is true now.

    The declined attempts are printed even when nothing was repairable, because "there was
    nothing safe to do here, and this is why" is the answer somebody who passed ``--fix``
    ran the command for. Their concrete fix-it commands are already above, in the report.
    """
    lines: list[str] = list(format_report(outcome.before))

    if not outcome.attempts:
        lines.append("nothing to fix: every check already passes")
        return lines

    lines.append("")
    lines.append("fixes:")
    lines.extend(
        f"  [{_STATUS_LABEL[attempt.status]:<11}] {attempt.check}: {attempt.detail}"
        for attempt in outcome.attempts
    )

    if not outcome.rechecked:
        lines.append("")
        lines.append("nothing was changed, so the report above still stands")
        return lines

    lines.append("")
    lines.append(f"repaired: {', '.join(outcome.repaired) if outcome.repaired else 'nothing'}")
    lines.append("re-checked after fixing:")
    lines.extend(format_report(outcome.after))
    return lines
