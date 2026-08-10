"""The ``serve`` and ``worker`` wrappers: what they refuse, and what they hand uvicorn."""

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.api.main import CONFIG_PATH_VAR
from fasterrag.cli.commands import processes
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.errors import FasterRagError
from tests.unit.cli.conftest import write_config


class FakeConfig:
    """Records the arguments ``uvicorn.Config`` was built with."""

    built: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, target: Any, **kwargs: Any) -> None:
        self.target = target
        self.kwargs = kwargs
        FakeConfig.built.append({"target": target, **kwargs})


class FakeServer:
    """A uvicorn server that returns the moment it is served."""

    served = 0

    def __init__(self, config: FakeConfig) -> None:
        self.config = config

    async def serve(self) -> None:
        FakeServer.served += 1


@pytest.fixture
def uvicorn_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Replace uvicorn so ``serve`` returns instead of binding a port."""
    FakeConfig.built = []
    FakeServer.served = 0
    module = types.ModuleType("uvicorn")
    module.Config = FakeConfig  # type: ignore[attr-defined]
    # CRITICAL: the assignments need the ignore because a ModuleType has no declared
    # attributes; there is no typed way to build a stub module.
    module.Server = FakeServer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return module


class FakeAdapter:
    """A vector DB whose health the test decides."""

    def __init__(self, healthy: bool = True) -> None:
        self.status = HealthStatus(healthy=healthy, detail="connection refused")

    async def health(self) -> HealthStatus:
        return self.status


class FakeIngestion:
    """Stands in for the service that owns the pools."""

    def __init__(self, healthy: bool = True, error: Exception | None = None) -> None:
        self.adapter = FakeAdapter(healthy)
        self.error = error
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeIngestion:
    built = FakeIngestion()
    monkeypatch.setattr(processes, "IngestionService", lambda *args, **kwargs: built)
    monkeypatch.setattr(processes, "create_journal", lambda settings: object())
    return built


def test_serve_refuses_an_invalid_config_before_binding_a_port(
    bad_config: str, uvicorn_stub: types.ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["serve", "--config", bad_config])

    assert code == ExitCode.USAGE
    assert FakeConfig.built == []
    assert "error:" in capsys.readouterr().err


def test_serve_announces_where_it_is_listening(
    config: str, uvicorn_stub: types.ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["serve", "--config", config])

    assert code == ExitCode.SUCCESS
    assert "serving on http://127.0.0.1:8000" in capsys.readouterr().out


def test_the_host_and_port_flags_override_the_configured_ones(
    config: str, uvicorn_stub: types.ModuleType
) -> None:
    main(["serve", "--config", config, "--host", "0.0.0.0", "--port", "9999"])

    assert (FakeConfig.built[0]["host"], FakeConfig.built[0]["port"]) == ("0.0.0.0", 9999)


def test_without_reload_the_application_is_built_in_process(
    config: str, uvicorn_stub: types.ModuleType
) -> None:
    main(["serve", "--config", config])

    assert FakeConfig.built[0]["factory"] is False
    assert not isinstance(FakeConfig.built[0]["target"], str)


def test_reload_hands_the_config_path_over_through_the_environment(
    config: str, uvicorn_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload child cannot receive a Settings object; without this it serves ./config.yaml."""
    monkeypatch.delenv(CONFIG_PATH_VAR, raising=False)

    main(["serve", "--config", config, "--reload"])

    assert FakeConfig.built[0]["target"] == "fasterrag.api.main:create_app"
    assert FakeConfig.built[0]["factory"] is True


def test_the_dashboard_runs_beside_the_api_on_its_own_port(
    tmp_path: Path,
    uvicorn_stub: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(tmp_path, "observability:\n  dashboard: true\n  dashboard_port: 8123\n")

    main(["serve", "--config", config])

    ports = [built["port"] for built in FakeConfig.built]
    assert ports == [8000, 8123]
    assert "dashboard on http://127.0.0.1:8123 (read-only)" in capsys.readouterr().out


def test_the_dashboard_is_off_unless_configured_on(
    config: str, uvicorn_stub: types.ModuleType
) -> None:
    main(["serve", "--config", config])

    assert len(FakeConfig.built) == 1


def test_worker_refuses_an_invalid_config(bad_config: str) -> None:
    assert main(["worker", "--config", bad_config]) == ExitCode.USAGE


def test_an_unknown_pool_is_named_rather_than_ignored(
    config: str, service: FakeIngestion, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["worker", "--config", config, "--pools", "cpu,gpu,quantum"])

    err = capsys.readouterr().err
    assert code == ExitCode.USAGE
    assert "gpu, quantum" in err


def test_pool_names_are_trimmed_and_blanks_dropped(
    config: str, service: FakeIngestion, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--pools cpu, embed`` is what a person types; rejecting " embed" would be pedantry."""
    with pytest.raises(TimeoutError):
        asyncio.run(_worker(config, "cpu, embed,"))

    assert '"pools": [' in capsys.readouterr().out


def test_an_unreachable_backend_exits_three_with_a_fix(
    config: str, service: FakeIngestion, capsys: pytest.CaptureFixture[str]
) -> None:
    service.adapter.status = HealthStatus(healthy=False, detail="connection refused")

    code = main(["worker", "--config", config])

    err = capsys.readouterr().err
    assert code == ExitCode.UNREACHABLE
    assert "fasterrag provision qdrant" in err


def test_an_unreachable_backend_still_closes_the_service(
    config: str, service: FakeIngestion
) -> None:
    service.adapter.status = HealthStatus(healthy=False, detail="connection refused")

    main(["worker", "--config", config])

    assert service.closed == 1


def test_a_typed_error_from_the_health_check_maps_to_its_code(
    config: str, service: FakeIngestion, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def raising() -> HealthStatus:
        raise FasterRagError("qdrant is not answering", retryable=True)

    monkeypatch.setattr(service.adapter, "health", raising)

    assert main(["worker", "--config", config]) == ExitCode.UNREACHABLE


async def _worker(config: str, pools: str) -> None:
    """Run the worker until it blocks on its idle wait, then give up on it.

    The happy path never returns by design — it reports the pools it would run and waits to
    be stopped. The reporting is the part worth asserting, so the wait is cut short.
    """
    from fasterrag.cli.output import Console
    from fasterrag.cli.parser import build_parser

    args = build_parser().parse_args(["worker", "--config", config, "--pools", pools])
    await asyncio.wait_for(processes.run_worker(args, Console(as_json=True)), timeout=0.5)


def test_a_ready_worker_reports_the_pool_sizes_it_would_run(
    config: str, service: FakeIngestion, capsys: pytest.CaptureFixture[str]
) -> None:
    """The configured 0 means "auto"; reporting it verbatim claimed no parse workers at all."""
    with pytest.raises(TimeoutError):
        asyncio.run(_worker(config, "cpu,embed,index"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["pools"] == ["cpu", "embed", "index"]
    assert payload["cpu"] == (os.cpu_count() or 1)


def test_the_worker_count_overrides_reach_the_report(
    config: str, service: FakeIngestion, capsys: pytest.CaptureFixture[str]
) -> None:
    from fasterrag.cli.output import Console
    from fasterrag.cli.parser import build_parser

    args = build_parser().parse_args(
        ["worker", "--config", config, "--cpu-workers", "3", "--embed-workers", "7", "--json"]
    )

    async def run() -> None:
        await asyncio.wait_for(processes.run_worker(args, Console(as_json=True)), timeout=0.5)

    with pytest.raises(TimeoutError):
        asyncio.run(run())

    payload = json.loads(capsys.readouterr().out)
    assert (payload["cpu"], payload["embed"]) == (3, 7)
