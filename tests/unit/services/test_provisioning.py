from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ProvisioningError
from fasterrag.services import provisioning
from fasterrag.services.doctor import DoctorCheck, DoctorReport
from fasterrag.services.provisioning import (
    QDRANT_CONTAINER,
    QDRANT_SERVER_KEY_VAR,
    ContainerState,
    DockerResult,
    provision_qdrant,
    qdrant_status,
    stop_qdrant,
)

OK = DockerResult(returncode=0, stdout="", stderr="")
FAILED = DockerResult(returncode=1, stdout="", stderr="boom")


@dataclass
class DockerRecorder:
    """Stands in for run_docker, recording arguments and injected environments."""

    responses: dict[str, DockerResult] = field(default_factory=dict)
    calls: list[tuple[list[str], dict[str, str]]] = field(default_factory=list)

    async def __call__(
        self,
        args: Sequence[str],
        *,
        timeout: float = 60.0,
        env: Mapping[str, str] | None = None,
    ) -> DockerResult:
        self.calls.append((list(args), dict(env or {})))
        return self.responses.get(args[0], OK)

    def command(self, verb: str) -> list[str]:
        return next(args for args, _ in self.calls if args[0] == verb)

    def environment(self, verb: str) -> dict[str, str]:
        return next(env for args, env in self.calls if args[0] == verb)

    def verbs(self) -> list[str]:
        return [args[0] for args, _ in self.calls]


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> DockerRecorder:
    recorder = DockerRecorder()
    monkeypatch.setattr(provisioning, "run_docker", recorder)
    return recorder


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instantly_ready(host: str, ports: Sequence[int]) -> None:
        return None

    monkeypatch.setattr(provisioning, "_wait_until_ready", instantly_ready)


@pytest.fixture
def gate_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def healthy(settings: Settings) -> DoctorReport:
        return DoctorReport(checks=[DoctorCheck(name="docker", passed=True, detail="ok")])

    monkeypatch.setattr("fasterrag.services.doctor.run_doctor", healthy)


def absent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def state(name: str = QDRANT_CONTAINER) -> ContainerState:
        return ContainerState(name=name, exists=False)

    monkeypatch.setattr(provisioning, "container_state", state)


def existing(monkeypatch: pytest.MonkeyPatch, *, running: bool, image: str) -> None:
    async def state(name: str = QDRANT_CONTAINER) -> ContainerState:
        return ContainerState(name=name, exists=True, running=running, image=image)

    monkeypatch.setattr(provisioning, "container_state", state)


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_provisioning_creates_the_container_when_absent(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent(monkeypatch)
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    result = await provision_qdrant(Settings())

    assert result.status == "running"
    assert result.url == "http://localhost:6333"
    assert "run" in docker.verbs()


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_run_publishes_both_ports_and_mounts_the_named_volume(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent(monkeypatch)
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    await provision_qdrant(Settings())
    command = " ".join(docker.command("run"))

    assert "6333:6333" in command
    assert "6334:6334" in command
    assert "fasterrag_qdrant_storage:/qdrant/storage" in command
    assert "qdrant/qdrant:v1.9.0" in command


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_the_server_key_is_passed_by_name_never_on_the_command_line(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent(monkeypatch)
    monkeypatch.setenv("QDRANT_API_KEY", "super-secret-key")

    await provision_qdrant(Settings())

    command = " ".join(docker.command("run"))
    assert QDRANT_SERVER_KEY_VAR in command
    assert "super-secret-key" not in command
    assert docker.environment("run")[QDRANT_SERVER_KEY_VAR] == "super-secret-key"


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_a_stopped_container_is_started_not_recreated(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(monkeypatch, running=False, image="qdrant/qdrant:v1.9.0")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    await provision_qdrant(Settings())

    assert "start" in docker.verbs()
    assert "run" not in docker.verbs()


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_a_running_correct_container_is_left_alone(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(monkeypatch, running=True, image="qdrant/qdrant:v1.9.0")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    result = await provision_qdrant(Settings())

    assert result.status == "running"
    assert "run" not in docker.verbs()
    assert "start" not in docker.verbs()


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_an_image_change_replaces_the_container(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(monkeypatch, running=True, image="qdrant/qdrant:v1.8.0")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    await provision_qdrant(Settings())

    assert docker.verbs().count("rm") == 1
    assert "run" in docker.verbs()


@pytest.mark.usefixtures("ready")
async def test_provisioning_refuses_when_a_blocking_check_fails(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unhealthy(settings: Settings) -> DoctorReport:
        return DoctorReport(
            checks=[
                DoctorCheck(
                    name="docker",
                    passed=False,
                    detail="docker daemon is not responding",
                    fix="Start Docker Desktop.",
                    blocks_provisioning=True,
                )
            ]
        )

    monkeypatch.setattr("fasterrag.services.doctor.run_doctor", unhealthy)
    absent(monkeypatch)

    with pytest.raises(ProvisioningError, match="preflight checks failed") as caught:
        await provision_qdrant(Settings())

    assert caught.value.fix == "Start Docker Desktop."
    assert docker.verbs() == []


@pytest.mark.usefixtures("ready")
async def test_unreachable_backend_does_not_block_a_cold_start(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def not_yet_running(settings: Settings) -> DoctorReport:
        return DoctorReport(
            checks=[
                DoctorCheck(
                    name="vector_db",
                    passed=False,
                    detail="not reachable",
                    fix="Provision it.",
                    blocks_provisioning=False,
                )
            ]
        )

    monkeypatch.setattr("fasterrag.services.doctor.run_doctor", not_yet_running)
    absent(monkeypatch)
    monkeypatch.setenv("QDRANT_API_KEY", "secret-value")

    result = await provision_qdrant(Settings())

    assert result.status == "running"


async def test_provisioning_refuses_in_external_mode() -> None:
    settings = Settings.model_validate({"vector_db": {"mode": "external"}})

    with pytest.raises(ProvisioningError, match="not managed by fasterRag") as caught:
        await provision_qdrant(settings)
    assert caught.value.fix


@pytest.mark.usefixtures("gate_passes", "ready")
async def test_a_referenced_but_unset_key_is_reported_before_anything_runs(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent(monkeypatch)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(ProvisioningError, match="QDRANT_API_KEY") as caught:
        await provision_qdrant(Settings())

    assert "super" not in caught.value.detail
    assert docker.verbs() == []


async def test_stopping_preserves_the_data_volume(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(monkeypatch, running=True, image="qdrant/qdrant:v1.9.0")

    result = await stop_qdrant(Settings())

    assert result.status == "stopped"
    assert "fasterrag_qdrant_storage" in (result.detail or "")
    assert "stop" in docker.verbs()
    assert "rm" not in docker.verbs()


async def test_stopping_an_absent_container_is_not_an_error(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent(monkeypatch)

    assert (await stop_qdrant(Settings())).status == "absent"


async def test_status_reports_an_external_instance_as_unmanaged() -> None:
    settings = Settings.model_validate({"vector_db": {"mode": "external"}})

    result = await qdrant_status(settings)

    assert result.status == "external"
    assert result.url == "http://localhost:6333"


async def test_status_reports_a_running_container(
    docker: DockerRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(monkeypatch, running=True, image="qdrant/qdrant:v1.9.0")

    result = await qdrant_status(Settings())

    assert result.status == "running"
    assert result.detail == "image qdrant/qdrant:v1.9.0"


async def test_container_state_parses_docker_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inspect(args: Sequence[str], **kwargs: Any) -> DockerResult:
        return DockerResult(returncode=0, stdout="true\tqdrant/qdrant:v1.9.0", stderr="")

    monkeypatch.setattr(provisioning, "run_docker", inspect)
    state = await provisioning.container_state()

    assert state.exists is True
    assert state.running is True
    assert state.image == "qdrant/qdrant:v1.9.0"


async def test_container_state_is_absent_when_inspect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inspect(args: Sequence[str], **kwargs: Any) -> DockerResult:
        return FAILED

    monkeypatch.setattr(provisioning, "run_docker", inspect)

    assert (await provisioning.container_state()).exists is False


async def test_a_missing_docker_executable_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("docker")

    monkeypatch.setattr("asyncio.create_subprocess_exec", missing)

    with pytest.raises(ProvisioningError, match="docker executable was not found") as caught:
        await provisioning.run_docker(["info"])
    assert "PATH" in (caught.value.fix or "")
