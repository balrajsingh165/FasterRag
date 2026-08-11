import shutil
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import pytest

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ProvisioningError
from fasterrag.services import doctor
from fasterrag.services.doctor import (
    DoctorCheck,
    DoctorReport,
    FixAttempt,
    FixOutcome,
    apply_fixes,
    diagnose,
    diagnose_and_fix,
    format_fix_outcome,
    format_report,
    run_doctor,
)
from fasterrag.services.provisioning import CONTAINER_LABEL, ContainerState


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

    async def volume_present(volume: str) -> bool:
        return True

    monkeypatch.setattr(doctor, "docker_available", docker_up)
    monkeypatch.setattr(doctor, "container_state", no_container)
    # CRITICAL: without this the volume check shells out to a real Docker daemon, so the
    # unit tier would pass or fail on whether the developer's machine happens to have the
    # configured volume — and fail outright in CI, which has no daemon at all.
    monkeypatch.setattr(doctor, "volume_exists", volume_present)
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


VOLUME = Settings().vector_db.docker.volume


class FakeDocker:
    """A daemon that remembers named volumes and one container, and records mutations.

    The vector database is healthy exactly when the container is running, so a repair that
    starts the container is visible to the re-check the same way it would be on a real
    machine. A stub where the backend is healthy regardless would make every assertion
    about re-checking vacuous.
    """

    def __init__(self, *, volumes: set[str] | None = None, container: ContainerState) -> None:
        self.volumes = volumes if volumes is not None else set()
        self.container = container
        self.created: list[str] = []
        self.started: list[str] = []
        self.waited = 0

    async def volume_exists(self, volume: str) -> bool:
        return volume in self.volumes

    async def ensure_volume(self, volume: str) -> None:
        self.created.append(volume)
        self.volumes.add(volume)

    async def container_state(self, name: str = "fasterrag-qdrant") -> ContainerState:
        return self.container

    async def start_container(self, name: str = "fasterrag-qdrant") -> None:
        self.started.append(name)
        self.container = ContainerState(
            name=self.container.name,
            exists=True,
            running=True,
            image=self.container.image,
            managed=self.container.managed,
        )

    async def await_backend_ready(self, settings: Settings) -> None:
        self.waited += 1


def wire(monkeypatch: pytest.MonkeyPatch, fake: FakeDocker) -> FakeDocker:
    """Point every Docker-touching name doctor imported at the fake, and nothing else."""
    monkeypatch.setattr(doctor, "volume_exists", fake.volume_exists)
    monkeypatch.setattr(doctor, "ensure_volume", fake.ensure_volume)
    monkeypatch.setattr(doctor, "container_state", fake.container_state)
    monkeypatch.setattr(doctor, "start_container", fake.start_container)
    monkeypatch.setattr(doctor, "await_backend_ready", fake.await_backend_ready)
    monkeypatch.setattr(
        doctor, "create_vector_db_adapter", lambda settings: FakeAdapter(fake.container.running)
    )
    return fake


@pytest.fixture
def docker_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A responding daemon, free ports, and populated secrets."""

    async def up() -> bool:
        return True

    monkeypatch.setattr(doctor, "docker_available", up)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: True)
    monkeypatch.setenv("QDRANT_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("FASTERRAG_API_KEY", "key")


def absent_container() -> ContainerState:
    return ContainerState(name="fasterrag-qdrant", exists=False)


def stopped_managed_container() -> ContainerState:
    return ContainerState(
        name="fasterrag-qdrant", exists=True, running=False, image="qdrant", managed=True
    )


def named(report: DoctorReport, name: str) -> DoctorCheck:
    """Return one check by name, failing loudly when the report has no such check."""
    return next(check for check in report.checks if check.name == name)


def attempt_for(outcome: FixOutcome, check: str) -> FixAttempt:
    """Return the attempt recorded for one check, failing loudly when there is none."""
    return next(item for item in outcome.attempts if item.check == check)


@pytest.mark.usefixtures("docker_up")
async def test_a_missing_storage_volume_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The volume the index lives on is invisible until the container is replaced."""
    wire(monkeypatch, FakeDocker(container=absent_container()))

    check = named(await run_doctor(Settings()), "vector_db_volume")

    assert check.passed is False
    assert VOLUME in check.detail
    assert check.fix


@pytest.mark.usefixtures("docker_up")
async def test_an_existing_storage_volume_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    wire(monkeypatch, FakeDocker(volumes={VOLUME}, container=absent_container()))

    assert named(await run_doctor(Settings()), "vector_db_volume").passed is True


@pytest.mark.usefixtures("docker_up")
async def test_a_missing_volume_does_not_block_provisioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`provision qdrant` creates the volume, so gating it on one would refuse the fix."""
    wire(monkeypatch, FakeDocker(container=absent_container()))

    assert named(await run_doctor(Settings()), "vector_db_volume").blocks_provisioning is False


async def test_the_volume_check_is_omitted_rather_than_passed_when_docker_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A check that could not run must never report PASS — that is a gate passing blind."""

    async def down() -> bool:
        return False

    monkeypatch.setattr(doctor, "docker_available", down)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: False)
    wire(monkeypatch, FakeDocker(container=absent_container()))

    report = await run_doctor(Settings())

    assert "vector_db_volume" not in {check.name for check in report.checks}


@pytest.mark.usefixtures("docker_up")
async def test_the_volume_is_not_required_in_external_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire(monkeypatch, FakeDocker(container=absent_container()))
    settings = Settings.model_validate({"vector_db": {"mode": "external"}})

    check = named(await run_doctor(settings), "vector_db_volume")

    assert check.passed is True
    assert "not required" in check.detail


@pytest.mark.usefixtures("docker_up")
async def test_fix_creates_the_missing_volume_and_verifies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = wire(monkeypatch, FakeDocker(container=absent_container()))

    outcome = await apply_fixes(Settings())

    assert fake.created == [VOLUME]
    assert outcome.rechecked is True
    assert "vector_db_volume" in outcome.repaired
    assert attempt_for(outcome, "vector_db_volume").status == "fixed"
    assert named(outcome.after, "vector_db_volume").passed is True


@pytest.mark.usefixtures("docker_up")
async def test_a_repair_that_did_not_take_is_reported_not_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-check is the only evidence that counts.

    A repair that returns without raising and leaves the check failing must never be
    reported as a success — that is precisely the "reports success without re-checking"
    defect this command exists to avoid.
    """
    fake = wire(monkeypatch, FakeDocker(container=absent_container()))

    async def pretend_to_create(volume: str) -> None:
        return None

    monkeypatch.setattr(doctor, "ensure_volume", pretend_to_create)

    outcome = await apply_fixes(Settings())

    assert fake.volumes == set()
    assert attempt_for(outcome, "vector_db_volume").status == "failed"
    assert outcome.repaired == []
    assert outcome.passed is False


@pytest.mark.usefixtures("docker_up")
async def test_a_repair_that_raises_is_reported_not_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire(monkeypatch, FakeDocker(container=absent_container()))

    async def refuse(volume: str) -> None:
        raise ProvisioningError("docker refused", fix="Check permissions.")

    monkeypatch.setattr(doctor, "ensure_volume", refuse)

    outcome = await apply_fixes(Settings())
    attempt = attempt_for(outcome, "vector_db_volume")

    assert attempt.status == "failed"
    assert "docker refused" in attempt.detail


@pytest.mark.usefixtures("docker_up")
async def test_a_stopped_container_fasterrag_created_is_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = wire(monkeypatch, FakeDocker(volumes={VOLUME}, container=stopped_managed_container()))

    outcome = await apply_fixes(Settings())

    assert fake.started == ["fasterrag-qdrant"]
    assert fake.waited == 1
    assert attempt_for(outcome, "vector_db").status == "fixed"
    assert outcome.passed is True


@pytest.mark.usefixtures("docker_up")
async def test_a_container_fasterrag_did_not_create_is_never_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting somebody else's stopped service is the surprising action --fix must not take."""
    fake = wire(
        monkeypatch,
        FakeDocker(
            volumes={VOLUME},
            container=ContainerState(
                name="fasterrag-qdrant", exists=True, running=False, managed=False
            ),
        ),
    )

    outcome = await apply_fixes(Settings())
    attempt = attempt_for(outcome, "vector_db")

    assert fake.started == []
    assert attempt.status == "manual"
    assert CONTAINER_LABEL in attempt.detail


@pytest.mark.usefixtures("docker_up")
async def test_an_absent_container_is_not_created_by_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating one pulls an image and publishes ports; that is provisioning, not diagnosis."""
    fake = wire(monkeypatch, FakeDocker(volumes={VOLUME}, container=absent_container()))

    outcome = await apply_fixes(Settings())
    attempt = attempt_for(outcome, "vector_db")

    assert fake.started == []
    assert attempt.status == "manual"
    assert "fasterrag provision qdrant" in attempt.detail


@pytest.mark.usefixtures("docker_up")
async def test_a_running_container_is_not_restarted(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable backend behind a running container is a credential or transport fault."""
    fake = wire(
        monkeypatch,
        FakeDocker(
            volumes={VOLUME},
            container=ContainerState(
                name="fasterrag-qdrant", exists=True, running=True, managed=True
            ),
        ),
    )
    monkeypatch.setattr(
        doctor,
        "create_vector_db_adapter",
        lambda settings: FakeAdapter(healthy=False, detail="401"),
    )

    outcome = await apply_fixes(Settings())

    assert fake.started == []
    assert attempt_for(outcome, "vector_db").status == "manual"


@pytest.mark.usefixtures("docker_up")
async def test_a_held_port_needs_a_human_and_nothing_is_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = wire(monkeypatch, FakeDocker(volumes={VOLUME}, container=absent_container()))
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: port != 8000)

    outcome = await apply_fixes(Settings())
    attempt = attempt_for(outcome, "api_port")

    assert attempt.status == "manual"
    assert "killing whatever holds it" in attempt.detail
    assert fake.created == []
    assert fake.started == []


@pytest.mark.usefixtures("docker_up")
async def test_nothing_is_rechecked_when_no_repair_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running the checks would be honest but pointless; saying so is what matters."""
    wire(monkeypatch, FakeDocker(volumes={VOLUME}, container=absent_container()))
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: port != 8000)
    monkeypatch.setattr(
        doctor, "create_vector_db_adapter", lambda settings: FakeAdapter(healthy=True)
    )

    outcome = await apply_fixes(Settings())

    assert outcome.rechecked is False
    assert outcome.after is outcome.before
    assert [attempt.status for attempt in outcome.attempts] == ["manual"]


@pytest.mark.usefixtures("docker_up")
async def test_every_failing_check_is_accounted_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure with no attempt is a failure the operator is never told about."""
    wire(monkeypatch, FakeDocker(container=absent_container()))
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(total=100, used=99, free=1))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: Memory(total=1, available=1))
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: False)

    outcome = await apply_fixes(Settings())

    assert {check.name for check in outcome.before.failures} == {
        attempt.check for attempt in outcome.attempts
    }
    for attempt in outcome.attempts:
        assert attempt.detail, f"{attempt.check} was reported with no explanation"


async def test_invalid_config_is_never_rewritten_by_fix(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    original = "app:\n  port: 70000\n"
    config.write_text(original, encoding="utf-8")

    outcome = await diagnose_and_fix(config, env_file=None)

    assert config.read_text(encoding="utf-8") == original
    assert outcome.passed is False
    assert outcome.rechecked is False
    assert attempt_for(outcome, "config").status == "manual"


@pytest.mark.usefixtures("docker_up")
async def test_fix_json_is_a_superset_of_the_plain_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--json` consumers must find `passed` and `checks` in the same place either way.

    They must also describe the machine *after* the repairs. A payload reporting the
    failures that have since been fixed would send an automated caller after work already
    done.
    """
    wire(monkeypatch, FakeDocker(container=stopped_managed_container()))

    payload = (await apply_fixes(Settings())).as_dict()

    assert set(payload) == {"passed", "checks", "fixes"}
    assert payload["passed"] is True
    assert [check for check in payload["checks"] if not check["passed"]] == []
    assert payload["fixes"]["rechecked"] is True
    assert payload["fixes"]["still_failing"] == []
    assert sorted(payload["fixes"]["repaired"]) == ["vector_db", "vector_db_volume"]
    assert {"check", "status", "detail"} == set(payload["fixes"]["attempts"][0])


@pytest.mark.usefixtures("docker_up")
async def test_fix_output_names_what_changed_and_what_needs_a_human(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire(monkeypatch, FakeDocker(container=absent_container()))
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: port != 8000)

    lines = "\n".join(format_fix_outcome(await apply_fixes(Settings())))

    assert "[fixed" in lines
    assert "[needs human" in lines
    assert "re-checked after fixing:" in lines
    assert "repaired: vector_db_volume" in lines


def test_every_check_has_a_repair_policy() -> None:
    """A check added later with no policy would silently report the generic fallback."""
    emitted = {
        "config",
        "python",
        "disk",
        "memory",
        "gpu",
        "docker",
        "secrets",
        "vector_db_ports",
        "vector_db_volume",
        "vector_db",
        "api_port",
        "dashboard_port",
    }
    covered = set(doctor._NO_SAFE_FIX) | set(doctor._FIXERS)

    assert emitted <= covered, f"no repair policy for: {sorted(emitted - covered)}"
