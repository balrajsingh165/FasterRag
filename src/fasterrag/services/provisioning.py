"""Config-driven provisioning of system-managed containers.

Flipping a configuration value is the entire trigger: ``vector_db.mode: docker`` means
fasterRag launches and manages Qdrant itself, and no application code changes at toggle
time (``docs/architecture.md`` §10). Provisioning converges to the desired state, so
running it twice is a no-op, and it is gated by ``fasterrag doctor`` — the environment
is proven survivable before anything is mutated (D10).

Secrets never reach a command line. The Qdrant server key is exported into the
subprocess environment and passed through with ``-e NAME``, so the value appears in
neither ``argv`` nor any log line.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ProvisioningError
from fasterrag.observability.logging import get_logger

__all__ = [
    "CONTAINER_LABEL",
    "QDRANT_CONTAINER",
    "ContainerState",
    "DockerResult",
    "ProvisionResult",
    "container_state",
    "docker_available",
    "port_is_free",
    "provision_qdrant",
    "qdrant_status",
    "run_docker",
    "stop_qdrant",
]

QDRANT_CONTAINER: Final = "fasterrag-qdrant"
CONTAINER_LABEL: Final = "fasterrag.managed=true"

QDRANT_SERVER_KEY_VAR: Final = "QDRANT__SERVICE__API_KEY"
QDRANT_GRPC_PORT_VAR: Final = "QDRANT__SERVICE__GRPC_PORT"
_QDRANT_REST_PORT: Final = 6333
_QDRANT_GRPC_PORT: Final = 6334
_QDRANT_STORAGE: Final = "/qdrant/storage"

_COMMAND_TIMEOUT_SECONDS: Final = 60.0
_READY_TIMEOUT_SECONDS: Final = 90.0
_READY_POLL_SECONDS: Final = 0.5
_PORT_PROBE_TIMEOUT_SECONDS: Final = 1.0

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DockerResult:
    """The outcome of one ``docker`` invocation."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Return whether the command succeeded."""
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class ContainerState:
    """What a managed container is currently doing."""

    name: str
    exists: bool
    running: bool = False
    image: str | None = None


@dataclass(frozen=True, slots=True)
class ProvisionResult:
    """What provisioning left behind, and where to reach it."""

    tool: str
    status: str
    url: str | None = None
    detail: str | None = None


async def run_docker(
    args: Sequence[str],
    *,
    timeout: float = _COMMAND_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> DockerResult:
    """Run a ``docker`` command and capture its output.

    Args:
        args: Arguments after the ``docker`` executable.
        timeout: Seconds to wait before killing the process. Every external call is
            bounded; no unbounded await exists in the codebase.
        env: Extra environment for the subprocess, used to hand secrets to ``-e NAME``
            pass-through without putting them in ``argv``.

    Returns:
        The command's exit status and captured streams.

    Raises:
        ProvisioningError: If the docker executable is absent or the command times out.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
        )
    except FileNotFoundError as exc:
        raise ProvisioningError(
            "the docker executable was not found",
            fix="Install Docker and make sure the 'docker' command is on PATH.",
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ProvisioningError(
            f"docker {args[0]} did not finish within {timeout:.0f}s",
            fix="Check that the Docker daemon is responsive, then retry.",
        ) from exc

    return DockerResult(
        returncode=process.returncode if process.returncode is not None else 1,
        stdout=stdout.decode(errors="replace").strip(),
        stderr=stderr.decode(errors="replace").strip(),
    )


async def docker_available() -> bool:
    """Return whether the Docker daemon is reachable."""
    result = await run_docker(["info", "--format", "{{.ServerVersion}}"], timeout=20.0)
    return result.ok


async def container_state(name: str = QDRANT_CONTAINER) -> ContainerState:
    """Return the state of a managed container."""
    result = await run_docker(
        ["inspect", name, "--format", "{{.State.Running}}\t{{.Config.Image}}"],
        timeout=20.0,
    )
    if not result.ok:
        return ContainerState(name=name, exists=False)

    running, _, image = result.stdout.partition("\t")
    return ContainerState(
        name=name,
        exists=True,
        running=running.strip().lower() == "true",
        image=image.strip() or None,
    )


def port_is_free(host: str, port: int) -> bool:
    """Return whether a TCP port can still be bound on ``host``."""
    bind_host = "" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, port))
        except OSError:
            return False
    return True


def port_is_reachable(host: str, port: int, timeout: float = _PORT_PROBE_TIMEOUT_SECONDS) -> bool:
    """Return whether something is listening on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _wait_until_ready(host: str, ports: Sequence[int]) -> None:
    """Block until every port answers, or fail with an actionable error."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _READY_TIMEOUT_SECONDS

    pending = list(ports)
    while pending and loop.time() < deadline:
        pending = [port for port in pending if not port_is_reachable(host, port)]
        if pending:
            await asyncio.sleep(_READY_POLL_SECONDS)

    if pending:
        raise ProvisioningError(
            f"qdrant started but ports {pending} never answered within "
            f"{_READY_TIMEOUT_SECONDS:.0f}s",
            fix=(
                f"Inspect the container logs with 'docker logs {QDRANT_CONTAINER}'. "
                "Both the REST and gRPC ports must be published."
            ),
        )


def _run_arguments(settings: Settings) -> list[str]:
    """Build the ``docker run`` arguments for the configured Qdrant container."""
    vector_db = settings.vector_db
    args = [
        "run",
        "--detach",
        "--name",
        QDRANT_CONTAINER,
        "--label",
        CONTAINER_LABEL,
        "--restart",
        "unless-stopped",
        "--publish",
        f"{vector_db.port}:{_QDRANT_REST_PORT}",
        "--publish",
        f"{vector_db.grpc_port}:{_QDRANT_GRPC_PORT}",
        "--volume",
        f"{vector_db.docker.volume}:{_QDRANT_STORAGE}",
        # CRITICAL: Qdrant leaves its gRPC interface disabled unless this is set. Publishing
        # the port is not enough — without it only 6333 answers, which is exactly the
        # failure recorded in docs/failure-modes.md row 15.
        "--env",
        f"{QDRANT_GRPC_PORT_VAR}={_QDRANT_GRPC_PORT}",
    ]
    if vector_db.api_key_env:
        args += ["--env", QDRANT_SERVER_KEY_VAR]
    args.append(vector_db.docker.image)
    return args


def _server_key_environment(settings: Settings) -> dict[str, str]:
    """Return the subprocess environment carrying the Qdrant server key, if configured."""
    name = settings.vector_db.api_key_env
    if not name:
        return {}

    value = os.environ.get(name)
    if not value:
        raise ProvisioningError(
            f"vector_db.api_key_env names {name}, but that variable is not set",
            fix=f"Set {name} in .env, or set vector_db.api_key_env to null for local development.",
        )
    return {QDRANT_SERVER_KEY_VAR: value}


async def _require_provisioning_gate(settings: Settings) -> None:
    """Refuse to provision until the environment checks pass.

    Only the checks that describe the environment gate provisioning. Backend
    reachability deliberately does not: it is expected to fail before the backend
    exists, and gating on it would make a cold start impossible.
    """
    from fasterrag.services.doctor import run_doctor

    report = await run_doctor(settings)
    blocking = [check for check in report.checks if not check.passed and check.blocks_provisioning]
    if blocking:
        listed = "; ".join(f"{check.name}: {check.detail}" for check in blocking)
        fixes = " ".join(check.fix for check in blocking if check.fix)
        raise ProvisioningError(
            f"preflight checks failed, so nothing was provisioned — {listed}",
            fix=fixes or "Run 'fasterrag doctor' for the full report.",
        )


async def provision_qdrant(settings: Settings) -> ProvisionResult:
    """Bring the system-managed Qdrant container to its configured state.

    Converges rather than reinstalls: an already-correct container is left running, a
    stopped one is started, and one built from a different image is replaced. Data
    survives replacement because storage lives on a named volume.

    Args:
        settings: Validated configuration.

    Returns:
        The running container's status and URL.

    Raises:
        ProvisioningError: If preflight checks fail, Docker refuses a step, or the
            container never becomes reachable. Each carries a concrete fix.
    """
    if settings.vector_db.mode != "docker":
        raise ProvisioningError(
            f"vector_db.mode is {settings.vector_db.mode!r}, so Qdrant is not managed by fasterRag",
            fix="Set vector_db.mode to 'docker' to have fasterRag manage the container.",
        )

    await _require_provisioning_gate(settings)

    environment = _server_key_environment(settings)
    await _ensure_volume(settings.vector_db.docker.volume)

    state = await container_state()
    desired_image = settings.vector_db.docker.image

    if state.exists and state.image != desired_image:
        _logger.info(
            "replacing the managed qdrant container after an image change",
            extra={"from_image": state.image, "to_image": desired_image},
        )
        await _remove_container()
        state = ContainerState(name=QDRANT_CONTAINER, exists=False)

    if not state.exists:
        result = await run_docker(_run_arguments(settings), env=environment)
        if not result.ok:
            raise ProvisioningError(
                f"could not start the qdrant container: {result.stderr}",
                fix=(
                    f"Free ports {settings.vector_db.port} and {settings.vector_db.grpc_port}, "
                    f"or remove a stale container with 'docker rm -f {QDRANT_CONTAINER}'."
                ),
            )
    elif not state.running:
        result = await run_docker(["start", QDRANT_CONTAINER])
        if not result.ok:
            raise ProvisioningError(
                f"could not start the existing qdrant container: {result.stderr}",
                fix=f"Inspect it with 'docker logs {QDRANT_CONTAINER}'.",
            )

    await _wait_until_ready(
        settings.vector_db.host,
        [settings.vector_db.port, settings.vector_db.grpc_port],
    )

    url = f"http://{settings.vector_db.host}:{settings.vector_db.port}"
    _logger.info("qdrant is provisioned and reachable", extra={"url": url})
    return ProvisionResult(tool="qdrant", status="running", url=url)


async def _ensure_volume(volume: str) -> None:
    """Create the named storage volume if it does not exist yet."""
    result = await run_docker(["volume", "create", volume], timeout=20.0)
    if not result.ok:
        raise ProvisioningError(
            f"could not create the docker volume {volume!r}: {result.stderr}",
            fix="Check Docker permissions, or choose another vector_db.docker.volume name.",
        )


async def _remove_container() -> None:
    """Remove the managed container, leaving its data volume untouched."""
    result = await run_docker(["rm", "--force", QDRANT_CONTAINER])
    if not result.ok:
        raise ProvisioningError(
            f"could not remove the existing qdrant container: {result.stderr}",
            fix=f"Remove it manually with 'docker rm -f {QDRANT_CONTAINER}'.",
        )


async def stop_qdrant(settings: Settings) -> ProvisionResult:
    """Stop the managed container, preserving its data volume."""
    state = await container_state()
    if not state.exists:
        return ProvisionResult(tool="qdrant", status="absent", detail="no managed container exists")

    if state.running:
        result = await run_docker(["stop", QDRANT_CONTAINER])
        if not result.ok:
            raise ProvisioningError(
                f"could not stop the qdrant container: {result.stderr}",
                fix=f"Stop it manually with 'docker stop {QDRANT_CONTAINER}'.",
            )

    return ProvisionResult(
        tool="qdrant",
        status="stopped",
        detail=f"data preserved on volume {settings.vector_db.docker.volume!r}",
    )


async def qdrant_status(settings: Settings) -> ProvisionResult:
    """Report the managed container's state without changing anything."""
    if settings.vector_db.mode != "docker":
        return ProvisionResult(
            tool="qdrant",
            status="external",
            url=f"http://{settings.vector_db.host}:{settings.vector_db.port}",
            detail="vector_db.mode is 'external', so this instance is not managed by fasterRag",
        )

    state = await container_state()
    if not state.exists:
        return ProvisionResult(tool="qdrant", status="absent")
    if not state.running:
        return ProvisionResult(tool="qdrant", status="stopped", detail=f"image {state.image}")

    return ProvisionResult(
        tool="qdrant",
        status="running",
        url=f"http://{settings.vector_db.host}:{settings.vector_db.port}",
        detail=f"image {state.image}",
    )
