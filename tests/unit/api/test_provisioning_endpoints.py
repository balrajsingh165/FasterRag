"""The REST provisioning surface must offer exactly what ``fasterrag provision`` offers.

TASK-0251: the admin router held ``_PROVISIONABLE = {"qdrant"}`` while the CLI could stand up
Langfuse and Grafana too, so ``POST /v1/admin/provision/langfuse`` answered ``NOT_FOUND`` for
nine days while ``docs/api-reference.md`` documented all three. No gate could see it — the
route existed and was served, and the disagreement lived in a value inside it.

The tests here drive both control planes over the *same* faked docker boundary and compare
what each returns. Only the boundary is faked: ``run_docker``, container state, port probes,
the doctor gate, and Grafana's HTTP verification. The routers, the registry, and the
provisioners themselves are the real ones, so a route wired to the wrong provisioner, or to
none, fails here rather than passing on a hand-built result object.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fasterrag.cli.main import main as cli_main
from fasterrag.cli.parser import build_parser
from fasterrag.config.schema import Settings
from fasterrag.services import grafana, langfuse, provisioning
from fasterrag.services.grafana import GrafanaPlan, Verification
from fasterrag.services.provision_registry import PROVISIONABLE_TOOLS
from fasterrag.services.provisioning import ContainerState, DockerResult

OK = DockerResult(returncode=0, stdout="true\tan-image", stderr="")

VERBS = ("provision", "status", "down")
COMPARED = ("tool", "status", "url")


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Replace every docker touchpoint of all three provisioners, recording the commands.

    Everything above this boundary runs for real. The working directory moves to a temporary
    one because the Langfuse provisioner writes generated secrets to ``./.env`` and Grafana
    writes its manifests under ``./.fasterrag`` — a test must not put either in the tree.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_API_KEY", "not-a-real-key")
    calls: list[list[str]] = []

    async def run_docker(
        args: Sequence[str],
        *,
        timeout: float = 60.0,
        env: Mapping[str, str] | None = None,
    ) -> DockerResult:
        calls.append(list(args))
        return OK

    async def available() -> bool:
        return True

    async def state(name: str = provisioning.QDRANT_CONTAINER) -> ContainerState:
        return ContainerState(name=name, exists=True, running=True, image="an-image")

    async def gate(settings: Settings) -> None:
        return None

    async def ready(host: str, ports: Sequence[int]) -> None:
        return None

    async def answering(settings: Settings) -> None:
        return None

    async def verified(plan: GrafanaPlan) -> Verification:
        return Verification(True, "verified")

    for module in (provisioning, langfuse, grafana):
        monkeypatch.setattr(module, "run_docker", run_docker)
        monkeypatch.setattr(module, "require_provisioning_gate", gate)
    for module in (langfuse, grafana):
        monkeypatch.setattr(module, "docker_available", available)
        monkeypatch.setattr(module, "port_is_reachable", lambda host, port, timeout=1.0: True)
    for module in (provisioning, grafana):
        monkeypatch.setattr(module, "container_state", state)

    monkeypatch.setattr(provisioning, "_wait_until_ready", ready)
    monkeypatch.setattr(provisioning, "_wait_until_answering", answering)
    monkeypatch.setattr(grafana, "verify_grafana", verified)

    return calls


def over_rest(client: TestClient, tool: str, verb: str) -> dict[str, Any]:
    """Drive one provisioning verb over REST and return the response body."""
    if verb == "status":
        response = client.get(f"/v1/admin/provision/{tool}/status")
    elif verb == "down":
        response = client.delete(f"/v1/admin/provision/{tool}")
    else:
        response = client.post(f"/v1/admin/provision/{tool}")

    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def over_cli(tool: str, verb: str, config: Path, capsys: pytest.CaptureFixture[str]) -> Any:
    """Drive the same verb through ``fasterrag provision`` and return its JSON document."""
    flags = {"status": ["--status"], "down": ["--down"], "provision": []}[verb]
    code = cli_main(["--json", "provision", tool, "--config", str(config), *flags])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the schema defaults the REST client is built from, as a file the CLI can load."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "not-a-real-key")
    path = tmp_path / "config.yaml"
    path.write_text(
        "vector_db:\n  provider: qdrant\n  mode: docker\n  api_key_env: null\n"
        "llm:\n  provider: ollama\n  api_key_env: null\n",
        encoding="utf-8",
    )
    return path


def test_the_registry_holds_exactly_the_documented_tools() -> None:
    """Pins the set the parametrised tests below iterate over.

    Without this, deleting a tool from the registry would make every parity test quietly
    check less rather than fail — the failure mode that let this bug live.
    """
    assert set(PROVISIONABLE_TOOLS) == {"qdrant", "langfuse", "grafana"}


def test_provisioning_langfuse_over_rest_is_not_a_404(
    client: TestClient, fake_docker: list[list[str]]
) -> None:
    """The exact request TASK-0251 was filed for."""
    response = client.post("/v1/admin/provision/langfuse")

    assert response.status_code == 200
    assert response.json()["tool"] == "langfuse"
    assert any("compose" in argument for call in fake_docker for argument in call)


def test_provisioning_grafana_over_rest_is_not_a_404(
    client: TestClient, fake_docker: list[list[str]]
) -> None:
    response = client.post("/v1/admin/provision/grafana")

    assert response.status_code == 200
    assert response.json()["tool"] == "grafana"


@pytest.mark.parametrize("tool", PROVISIONABLE_TOOLS)
@pytest.mark.parametrize("verb", VERBS)
def test_every_tool_answers_every_verb_over_rest(
    client: TestClient, fake_docker: list[list[str]], tool: str, verb: str
) -> None:
    """A tool the CLI can drive that REST cannot is the defect this suite exists for."""
    body = over_rest(client, tool, verb)

    assert body["tool"] == tool
    assert body["status"]


@pytest.mark.parametrize("tool", PROVISIONABLE_TOOLS)
@pytest.mark.parametrize("verb", VERBS)
def test_the_two_control_planes_report_the_same_thing(
    client: TestClient,
    fake_docker: list[list[str]],
    config: Path,
    capsys: pytest.CaptureFixture[str],
    tool: str,
    verb: str,
) -> None:
    """Same fakes, same settings, same answer — or the two surfaces have diverged again.

    ``detail`` is compared for presence rather than equality, because the REST call here runs
    first and provisioning converges: the second caller is truthfully told that the work was
    already done. Requiring the two strings to match would be requiring provisioning not to
    be idempotent.
    """
    rest = over_rest(client, tool, verb)
    cli = over_cli(tool, verb, config, capsys)

    assert set(cli) == set(rest)
    assert {name: cli[name] for name in COMPARED} == {name: rest[name] for name in COMPARED}


def test_a_cli_run_after_a_rest_run_preserves_the_generated_secrets(
    client: TestClient,
    fake_docker: list[list[str]],
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Trap T11, across surfaces: both planes converge on one ``.env``.

    Rotating ``SALT``/``ENCRYPTION_KEY``/``NEXTAUTH_SECRET`` invalidates every credential the
    stack has issued. Provisioning from REST and then from the CLI must therefore preserve,
    not regenerate — and only a test that drives both can show that the two agree on the file.
    """
    client.post("/v1/admin/provision/langfuse")

    second = over_cli("langfuse", "provision", config, capsys)

    assert "preserved" in second["detail"]


@pytest.mark.parametrize("tool", PROVISIONABLE_TOOLS)
def test_the_cli_accepts_every_tool_rest_serves(tool: str) -> None:
    """Two hand-written lists is how the surfaces diverged; there is now one."""
    assert build_parser().parse_args(["provision", tool]).tool == tool


def test_the_cli_refuses_a_tool_the_registry_does_not_hold() -> None:
    """Otherwise the parser would accept a name that dies later in dispatch."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["provision", "redis"])


def test_an_unknown_tool_is_refused_with_a_problem_document(client: TestClient) -> None:
    """The refusal must name what *is* supported, from the same table that dispatches."""
    response = client.post("/v1/admin/provision/redis")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "NOT_FOUND"
    assert response.headers["content-type"].startswith("application/problem+json")
    for tool in PROVISIONABLE_TOOLS:
        assert tool in body["detail"]


def test_a_provisioning_response_never_carries_a_generated_secret(
    client: TestClient, fake_docker: list[list[str]], tmp_path: Path
) -> None:
    """Trap T11: the Langfuse provisioner mints nine secrets while answering this request.

    They belong in ``.env`` and in the subprocess environment, nowhere else. A response body
    is logged by proxies and printed by clients, so a value leaking into ``detail`` would be
    a secret published to every hop between here and the operator.
    """
    body = client.post("/v1/admin/provision/langfuse").json()

    rendered = json.dumps(body)
    written = langfuse.read_env(tmp_path / ".env")
    assert set(langfuse.GENERATED_SECRETS) <= set(written)
    for name, value in written.items():
        assert value not in rendered, f"{name} reached the response body"


def test_the_grafana_verification_reason_reaches_the_caller(
    client: TestClient, fake_docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded stack must say why; ``status`` alone leaves nothing to act on."""

    async def unverified(plan: GrafanaPlan) -> Verification:
        return Verification(False, "the datasource cannot reach prometheus")

    monkeypatch.setattr(grafana, "verify_grafana", unverified)

    body = client.post("/v1/admin/provision/grafana").json()

    assert body["status"] == "degraded"
    assert "cannot reach prometheus" in body["detail"]
