"""``GET /v1/admin/doctor`` reports; it must never repair (TASK-0258).

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

The planned ``/v1/admin/doctor/fix`` route is pinned too, as *planned*: the reference
carries it behind a "not yet implemented" marker, and the drift gate turns that marker
into a failure the day the route is served. Without this the note outlives the gap.
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
    """Describe a machine whose two repairable failures are both present.

    Returns the list every repair primitive appends to. An empty list is the assertion the
    whole module exists to make, so the primitives are replaced by recorders rather than
    by raisers: a recorder lets the request finish and fail on the evidence, while a raiser
    would surface as a ``500`` and prove only that something went wrong.
    """
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
    """The label guard decides whether ``--fix`` may start a container; a read never may.

    The container here carries ``fasterrag.managed=true`` and is stopped, which is the one
    state the repair is permitted to act on — so if the endpoint ever repaired, this is the
    case where it would.
    """
    assert named(report(client), "vector_db")["passed"] is False
    assert repairable_machine == []


@pytest.mark.usefixtures("repairable_machine")
def test_the_response_is_the_report_and_not_a_fix_outcome(client: TestClient) -> None:
    """``--fix``'s document is a superset of this one, so its extra key is the tell."""
    body = report(client)

    assert set(body) == {"passed", "checks"}
    assert body["passed"] is False


def test_the_rest_repair_route_is_documented_as_planned_not_as_shipped() -> None:
    """TASK-0258's asymmetry is recorded, and recorded in the form the drift gate reads.

    ``scripts/check_openapi_drift.py`` splits the reference into shipped and planned on the
    "not yet implemented" marker and fails either way round, so this assertion is what ties
    the written decision to a gate: drop the marker and the gate reports a route promised
    but not served; build the route and leave the marker and it reports the reverse.
    """
    shipped, planned = documented_routes()

    assert "POST /v1/admin/doctor/fix" in planned
    assert "POST /v1/admin/doctor/fix" not in shipped
    assert "GET /v1/admin/doctor" in shipped
