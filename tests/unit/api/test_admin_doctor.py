"""``GET /v1/admin/doctor`` reports; ``POST /v1/admin/doctor/fix`` repairs (TASK-0258).

The CLI's ``doctor --fix`` and the REST doctor report call into the same service module,
and two of that module's functions differ only in whether they touch the machine:
``run_doctor`` probes, ``apply_fixes`` probes, repairs, and probes again. Nothing in the
type system stops the router from being pointed at the second one, and nothing in the
response would look wrong afterwards — the ``--fix`` document is a deliberate superset of
the plain report, so a mutated endpoint would answer with the same ``passed`` and
``checks`` keys a client reads today while having created a volume and started a container
on the host behind a ``GET``.

These tests therefore drive the real router over a machine that is *repairable* — the
named volume is missing and fasterRag's own container is stopped and labelled, which are
exactly the two failures ``--fix`` is allowed to act on — and assert that nothing was
touched. Only the Docker boundary is faked; the router, the dependency wiring, and
``run_doctor`` itself are real, so an endpoint wired to the repairing half fails here.

The repairing route is driven over the same fakes, because the two things that make it the
same surface as the CLI are both invisible in a happy path: an attempt is classified from a
second doctor report rather than from a repair returning without raising, and a container
holding the configured name without the ``fasterrag.managed=true`` label is somebody
else's. Both are asserted here, against a machine whose repairs deliberately do not take
and against one whose container is unlabelled.
"""

from typing import Any

import pytest
from check_openapi_drift import documented_routes
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.services import doctor
from fasterrag.services.provisioning import ContainerState

VOLUME = Settings().vector_db.docker.volume


class StoppedBackend:
    """A vector database that answers unhealthy, as a stopped container's would."""

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=False, detail="connection refused", latency_ms=0.0)

    async def close(self) -> None:
        return None


@pytest.fixture
def repairable_machine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Describe a machine whose two repairable failures are both present."""
    touched: list[str] = []

    async def docker_up() -> bool:
        return True

    async def stopped_managed(name: str = "fasterrag-qdrant") -> ContainerState:
        return ContainerState(name=name, exists=True, running=False, image="qdrant", managed=True)

    async def volume_absent(volume: str) -> bool:
        return False

    async def record_volume(volume: str) -> None:
        touched.append(f"ensure_volume:{volume}")

    async def record_start(name: str = "fasterrag-qdrant") -> None:
        touched.append(f"start_container:{name}")

    async def record_wait(settings: Settings) -> None:
        touched.append("await_backend_ready")

    monkeypatch.setattr(doctor, "docker_available", docker_up)
    monkeypatch.setattr(doctor, "container_state", stopped_managed)
    monkeypatch.setattr(doctor, "volume_exists", volume_absent)
    monkeypatch.setattr(doctor, "ensure_volume", record_volume)
    monkeypatch.setattr(doctor, "start_container", record_start)
    monkeypatch.setattr(doctor, "await_backend_ready", record_wait)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: True)
    monkeypatch.setattr(doctor, "create_vector_db_adapter", lambda settings: StoppedBackend())
    monkeypatch.setenv("QDRANT_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("FASTERRAG_API_KEY", "not-a-real-key")
    return touched


def report(client: TestClient) -> dict[str, Any]:
    """Fetch the doctor report over REST."""
    response = client.get("/v1/admin/doctor")

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def named(body: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one check from the wire format, failing loudly when it is absent."""
    return next(check for check in body["checks"] if check["name"] == name)


def test_a_repairable_failure_is_reported_and_not_repaired(
    client: TestClient, repairable_machine: list[str]
) -> None:
    """The missing volume is the one failure ``--fix`` creates outright. A read must not."""
    volume = named(report(client), "vector_db_volume")

    assert volume["passed"] is False
    assert VOLUME in volume["detail"]
    assert volume["fix"]
    assert repairable_machine == []


def test_a_stopped_managed_container_is_not_started_by_a_read(
    client: TestClient, repairable_machine: list[str]
) -> None:
    """The label guard decides whether ``--fix`` may start a container; a read never may."""
    assert named(report(client), "vector_db")["passed"] is False
    assert repairable_machine == []


@pytest.mark.usefixtures("repairable_machine")
def test_the_response_is_the_report_and_not_a_fix_outcome(client: TestClient) -> None:
    """``--fix``'s document is a superset of this one, so its extra key is the tell."""
    body = report(client)

    assert set(body) == {"passed", "checks"}
    assert body["passed"] is False


def test_the_rest_repair_route_is_documented_as_shipped_not_as_planned() -> None:
    """TASK-0258 is closed in the code, so it must read as closed in the reference."""
    shipped, planned = documented_routes()

    assert "POST /v1/admin/doctor/fix" in shipped
    assert "POST /v1/admin/doctor/fix" not in planned
    assert "GET /v1/admin/doctor" in shipped


@pytest.fixture
def repairing_machine(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Describe the same repairable machine, with repairs that do take effect."""
    state: dict[str, Any] = {"volume": False, "running": False, "managed": True, "touched": []}

    async def docker_up() -> bool:
        return True

    async def current_state(name: str = "fasterrag-qdrant") -> ContainerState:
        return ContainerState(
            name=name,
            exists=True,
            running=bool(state["running"]),
            image="qdrant",
            managed=bool(state["managed"]),
        )

    async def volume_present(volume: str) -> bool:
        return bool(state["volume"])

    async def create_volume(volume: str) -> None:
        state["touched"].append(f"ensure_volume:{volume}")
        state["volume"] = True

    async def start(name: str = "fasterrag-qdrant") -> None:
        state["touched"].append(f"start_container:{name}")
        state["running"] = True

    async def wait(settings: Settings) -> None:
        state["touched"].append("await_backend_ready")

    class Backend:
        """A backend that answers once the container holding it has been started."""

        async def health(self) -> HealthStatus:
            running = bool(state["running"])
            return HealthStatus(
                healthy=running,
                detail=None if running else "connection refused",
                latency_ms=1.0,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(doctor, "docker_available", docker_up)
    monkeypatch.setattr(doctor, "container_state", current_state)
    monkeypatch.setattr(doctor, "volume_exists", volume_present)
    monkeypatch.setattr(doctor, "ensure_volume", create_volume)
    monkeypatch.setattr(doctor, "start_container", start)
    monkeypatch.setattr(doctor, "await_backend_ready", wait)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: True)
    monkeypatch.setattr(doctor, "create_vector_db_adapter", lambda settings: Backend())
    monkeypatch.setenv("QDRANT_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("FASTERRAG_API_KEY", "not-a-real-key")
    return state


def fix(client: TestClient) -> dict[str, Any]:
    """Run the repairs over REST."""
    response = client.post("/v1/admin/doctor/fix")

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def attempt(body: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one fix attempt from the wire format, failing loudly when it is absent."""
    return next(entry for entry in body["fixes"]["attempts"] if entry["check"] == name)


def test_the_repairs_the_cli_makes_are_reachable_over_rest(
    client: TestClient, repairing_machine: dict[str, Any]
) -> None:
    """Both repairable failures are repaired, and the body is the CLI's ``--json`` shape."""
    body = fix(client)

    assert f"ensure_volume:{VOLUME}" in repairing_machine["touched"]
    assert "start_container:fasterrag-qdrant" in repairing_machine["touched"]
    assert {"passed", "checks", "fixes"} <= set(body)
    assert body["fixes"]["rechecked"] is True
    assert attempt(body, "vector_db_volume")["status"] == "fixed"
    assert attempt(body, "vector_db")["status"] == "fixed"
    assert named(body, "vector_db_volume")["passed"] is True


def test_a_container_without_the_managed_label_is_not_started(
    client: TestClient, repairing_machine: dict[str, Any]
) -> None:
    """A container holding the configured name but not the label belongs to somebody else.

    The volume repair still runs in this request, so the refusal is not an artefact of a run
    that changed nothing: the guard has to hold check by check, not request by request.
    """
    repairing_machine["managed"] = False

    body = fix(client)
    refusal = attempt(body, "vector_db")

    assert refusal["status"] == "manual"
    assert "fasterrag.managed=true" in refusal["detail"]
    assert not [entry for entry in repairing_machine["touched"] if entry.startswith("start_")]
    assert f"ensure_volume:{VOLUME}" in repairing_machine["touched"]


def test_a_repair_that_leaves_the_check_failing_is_not_reported_as_fixed(
    client: TestClient, repairable_machine: list[str]
) -> None:
    """The status comes from the second report, never from the repair returning (T12)."""
    body = fix(client)

    assert f"ensure_volume:{VOLUME}" in repairable_machine
    assert attempt(body, "vector_db_volume")["status"] == "failed"
    assert "still fails" in attempt(body, "vector_db_volume")["detail"]
    assert attempt(body, "vector_db")["status"] == "failed"
    assert body["passed"] is False


def test_a_check_with_no_safe_repair_comes_back_as_needing_a_human(
    client: TestClient,
    repairing_machine: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing credential cannot be invented, and the reason travels with the refusal."""
    monkeypatch.delenv("OPENAI_API_KEY")

    body = fix(client)
    secrets = attempt(body, "secrets")

    assert secrets["status"] == "manual"
    assert secrets["detail"]
    assert named(body, "secrets")["passed"] is False
    assert named(body, "secrets")["fix"]


def test_reading_the_report_still_does_not_repair_now_that_the_fix_route_exists(
    client: TestClient, repairing_machine: dict[str, Any]
) -> None:
    """The two verbs share a service module; only one of them may touch the machine."""
    body = report(client)

    assert set(body) == {"passed", "checks"}
    assert repairing_machine["touched"] == []
