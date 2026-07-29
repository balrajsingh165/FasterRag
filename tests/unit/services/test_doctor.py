import shutil
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import pytest

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.services import doctor
from fasterrag.services.doctor import (
    DoctorCheck,
    DoctorReport,
    diagnose,
    format_report,
    run_doctor,
)
from fasterrag.services.provisioning import ContainerState


class FakeAdapter:
    """Minimal stand-in for a vector database adapter."""

    def __init__(self, healthy: bool = True, detail: str | None = None) -> None:
        self.healthy = healthy
        self.detail = detail
        self.closed = False

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=self.healthy, detail=self.detail, latency_ms=1.0)

    async def close(self) -> None:
        self.closed = True


class Usage(NamedTuple):
    total: int
    used: int
    free: int


class Memory(NamedTuple):
    total: int
    available: int


@pytest.fixture
def healthy_environment(monkeypatch: pytest.MonkeyPatch) -> FakeAdapter:
    """Patch every probe so the checks describe a working machine."""
    adapter = FakeAdapter()

    async def docker_up() -> bool:
        return True

    async def no_container(name: str = "fasterrag-qdrant") -> ContainerState:
        return ContainerState(name=name, exists=False)

    monkeypatch.setattr(doctor, "docker_available", docker_up)
    monkeypatch.setattr(doctor, "container_state", no_container)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: True)
    monkeypatch.setattr(doctor, "create_vector_db_adapter", lambda settings: adapter)
    monkeypatch.setenv("QDRANT_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("FASTERRAG_API_KEY", "key")
    return adapter


@pytest.mark.usefixtures("healthy_environment")
async def test_a_healthy_environment_passes_every_check() -> None:
    report = await run_doctor(Settings())

    assert report.passed is True
    assert report.failures == []


@pytest.mark.usefixtures("healthy_environment")
async def test_the_documented_checks_are_all_present() -> None:
    names = {check.name for check in (await run_doctor(Settings())).checks}

    assert {
        "config",
        "python",
        "disk",
        "memory",
        "gpu",
        "docker",
        "secrets",
        "vector_db_ports",
        "vector_db",
        "api_port",
    } <= names


async def test_the_adapter_is_closed_after_probing(healthy_environment: FakeAdapter) -> None:
    await run_doctor(Settings())

    assert healthy_environment.closed is True


@pytest.mark.usefixtures("healthy_environment")
async def test_every_failing_check_carries_a_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(total=100, used=99, free=1))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: Memory(total=1, available=1))
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def docker_down() -> bool:
        return False

    monkeypatch.setattr(doctor, "docker_available", docker_down)
    monkeypatch.setattr(
        doctor, "create_vector_db_adapter", lambda settings: FakeAdapter(healthy=False, detail="no")
    )

    report = await run_doctor(Settings())

    assert report.failures
    for check in report.failures:
        assert check.fix, f"failing check {check.name!r} has no fix string"


@pytest.mark.usefixtures("healthy_environment")
async def test_docker_is_not_required_in_external_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def docker_down() -> bool:
        return False

    monkeypatch.setattr(doctor, "docker_available", docker_down)
    settings = Settings.model_validate({"vector_db": {"mode": "external"}})

    report = await run_doctor(settings)
    check = next(item for item in report.checks if item.name == "docker")

    assert check.passed is True
    assert "not required" in check.detail


@pytest.mark.usefixtures("healthy_environment")
async def test_docker_down_blocks_provisioning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def docker_down() -> bool:
        return False

    monkeypatch.setattr(doctor, "docker_available", docker_down)

    report = await run_doctor(Settings())
    check = next(item for item in report.checks if item.name == "docker")

    assert check.passed is False
    assert check.blocks_provisioning is True


@pytest.mark.usefixtures("healthy_environment")
async def test_a_blocked_grpc_port_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: port != 6334)
    settings = Settings.model_validate({"vector_db": {"mode": "external"}})

    report = await run_doctor(settings)
    check = next(item for item in report.checks if item.name == "vector_db_ports")

    assert check.passed is False
    assert "gRPC 6334" in check.detail
    assert "6334" in check.fix


@pytest.mark.usefixtures("healthy_environment")
async def test_ports_held_by_our_own_container_are_not_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def running(name: str = "fasterrag-qdrant") -> ContainerState:
        return ContainerState(name=name, exists=True, running=True, image="qdrant/qdrant:v1.9.0")

    monkeypatch.setattr(doctor, "container_state", running)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: False)

    report = await run_doctor(Settings())
    check = next(item for item in report.checks if item.name == "vector_db_ports")

    assert check.passed is True


@pytest.mark.usefixtures("healthy_environment")
async def test_backend_reachability_never_blocks_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor,
        "create_vector_db_adapter",
        lambda settings: FakeAdapter(healthy=False, detail="connection refused"),
    )

    report = await run_doctor(Settings())
    check = next(item for item in report.checks if item.name == "vector_db")

    assert check.passed is False
    assert check.blocks_provisioning is False


@pytest.mark.usefixtures("healthy_environment")
async def test_a_missing_secret_is_named_without_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = await run_doctor(Settings())
    check = next(item for item in report.checks if item.name == "secrets")

    assert check.passed is False
    assert "OPENAI_API_KEY" in check.detail
    assert "llm.api_key_env" in check.detail


async def test_invalid_config_is_a_failed_check_not_an_exception(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("app:\n  port: 70000\n", encoding="utf-8")

    report = await diagnose(config, env_file=None)

    assert report.passed is False
    assert report.checks[0].name == "config"
    assert report.checks[0].blocks_provisioning is True
    assert report.checks[0].fix


async def test_diagnose_reports_a_missing_config_file(tmp_path: Path) -> None:
    report = await diagnose(tmp_path / "absent.yaml", env_file=None)

    assert report.passed is False
    assert "not found" in report.checks[0].detail


def test_report_serializes_for_json_output() -> None:
    report = DoctorReport(
        checks=[DoctorCheck(name="docker", passed=False, detail="down", fix="Start it.")]
    )
    payload: dict[str, Any] = report.as_dict()

    assert payload["passed"] is False
    assert payload["checks"][0] == {
        "name": "docker",
        "passed": False,
        "detail": "down",
        "fix": "Start it.",
        "blocks_provisioning": False,
    }


def test_formatted_output_shows_fixes_for_failures() -> None:
    report = DoctorReport(
        checks=[
            DoctorCheck(name="python", passed=True, detail="3.12"),
            DoctorCheck(name="docker", passed=False, detail="down", fix="Start it."),
        ]
    )
    lines = "\n".join(format_report(report))

    assert "[PASS] python" in lines
    assert "[FAIL] docker" in lines
    assert "fix: Start it." in lines
