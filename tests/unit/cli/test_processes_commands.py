"""The ``serve`` wrapper: what it refuses, what it hands uvicorn; and the removed ``worker``."""

import re
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fasterrag.api.main import CONFIG_PATH_VAR
from fasterrag.cli.commands.processes import (
    BUILTIN_PROCESSES,
    PLANNED_PROCESSES,
    exposed_without_auth,
)
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.cli.parser import (
    PENDING_COMMANDS,
    REMOVED_COMMANDS,
    build_parser,
    removed_command,
)
from fasterrag.config.schema import Settings
from tests.unit.cli.conftest import write_config
from tests.unit.docs.test_cli_flag_parity import parsers_by_path


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
    tmp_path: Path, uvicorn_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(tmp_path, "security:\n  allow_unauthenticated: true\n")

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


def test_the_dashboard_keeps_its_own_loopback_bind_when_the_api_widens(
    tmp_path: Path, uvicorn_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It reused the API's host, so `--host 0.0.0.0` exposed the corpus view too (TASK-0460)."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key-long-enough")
    config = write_config(tmp_path, "security:\n  auth: true\nobservability:\n  dashboard: true\n")

    main(["serve", "--config", config, "--host", "10.0.0.5"])

    hosts = [built["host"] for built in FakeConfig.built]
    assert hosts == ["10.0.0.5", "127.0.0.1"]


def test_the_dashboard_binds_the_host_configured_for_it(
    tmp_path: Path, uvicorn_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key-long-enough")
    config = write_config(
        tmp_path,
        "security:\n  auth: true\nobservability:\n  dashboard: true\n  dashboard_host: 10.0.0.9\n",
    )

    main(["serve", "--config", config])

    assert [built["host"] for built in FakeConfig.built] == ["127.0.0.1", "10.0.0.9"]


def test_the_dashboard_is_off_unless_configured_on(
    config: str, uvicorn_stub: types.ModuleType
) -> None:
    main(["serve", "--config", config])

    assert len(FakeConfig.built) == 1


def test_the_removed_worker_command_says_why_it_went_and_where_it_is_tracked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``worker`` checked the database, printed "ready" and waited; it ran no pipeline work."""
    code = main(["worker"])

    err = capsys.readouterr().err
    assert code == ExitCode.USAGE
    assert "'worker' was removed" in err
    assert "ingestion runs inside 'fasterrag serve'" in err
    assert "TASK-0130" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["worker", "--pools", "cpu,embed", "--cpu-workers", "4"],
        ["worker", "--embed-workers", "2", "--json"],
        ["--quiet", "worker", "--help"],
        ["--config", "elsewhere.yaml", "worker"],
    ],
)
def test_a_removed_command_explains_itself_whatever_was_typed_around_it(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Every invocation that used to start the idle process gets the reason, not argparse's."""
    code = main(argv)

    captured = capsys.readouterr()
    assert code == ExitCode.USAGE
    assert "'worker' was removed" in captured.err
    assert "unrecognized arguments" not in captured.err
    assert captured.out == ""


def test_a_removed_command_is_neither_registered_nor_listed_in_help() -> None:
    """A tombstone in ``--help`` is still a surface that does nothing."""
    parser = build_parser()

    assert "worker" not in parser.format_help()
    assert not set(REMOVED_COMMANDS) & set(parsers_by_path())


def test_every_removed_command_names_the_task_that_tracks_it() -> None:
    """Without the id the message says what went and not where to follow it."""
    assert REMOVED_COMMANDS
    for command, reason in REMOVED_COMMANDS.items():
        assert "TASK-" in reason, command


@pytest.mark.parametrize(
    "argv",
    [
        ["--collection", "worker", "status"],
        ["ingest", "worker"],
        ["query", "what does a worker do", "--collection", "worker"],
        ["--set", "worker", "doctor"],
        [],
    ],
)
def test_a_value_that_spells_a_removed_command_is_not_mistaken_for_it(argv: list[str]) -> None:
    """Only the command position counts: a collection or a source path may be named anything."""
    assert removed_command(argv) is None


def test_every_registered_process_is_a_command_and_every_planned_one_explains_itself() -> None:
    """The gate reads the registry, so the registry has to be the CLI's truth too."""
    registered = set(parsers_by_path())

    assert set(BUILTIN_PROCESSES) <= registered
    assert not PLANNED_PROCESSES & registered
    assert set(REMOVED_COMMANDS) | set(PENDING_COMMANDS) >= PLANNED_PROCESSES


def test_no_shipped_compose_service_runs_a_command_the_cli_does_not_have() -> None:
    """``docker-compose.yml`` ran ``worker`` in a container that only waited."""
    root = Path(__file__).resolve().parents[3]
    presets = [root / "deploy" / f"compose.{size}.yml" for size in ("small", "medium", "large")]
    commands = [
        (path.name, match.group(1))
        for path in [root / "docker-compose.yml", *presets]
        for match in re.finditer(
            r'^\s*command:\s*\[\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    assert commands

    registered = set(parsers_by_path())
    assert [(name, command) for name, command in commands if command not in registered] == []


@pytest.mark.parametrize(
    ("host", "auth", "exposed"),
    [
        ("127.0.0.1", False, False),
        ("::1", False, False),
        ("localhost", False, False),
        ("0.0.0.0", False, True),
        ("10.0.0.5", False, True),
        ("api.internal", False, True),
        ("0.0.0.0", True, False),
    ],
)
def test_an_unauthenticated_non_loopback_bind_is_recognised(
    host: str, auth: bool, exposed: bool
) -> None:
    configured = Settings.model_validate({"security": {"auth": auth}})

    assert exposed_without_auth(host, configured) is exposed


def test_the_default_bind_is_loopback() -> None:
    """Auth defaults to off, so the default bind must not be reachable from other machines."""
    configured = Settings.model_validate({})

    assert configured.app.host == "127.0.0.1"
    assert not exposed_without_auth(configured.app.host, configured)


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "api.internal"])
def test_an_unauthenticated_non_loopback_bind_is_refused(
    config: str, uvicorn_stub: types.ModuleType, capsys: pytest.CaptureFixture[str], host: str
) -> None:
    """It was only a warning, so the open-by-default service still started (TASK-0460)."""
    code = main(["serve", "--config", config, "--host", host])

    err = capsys.readouterr().err
    assert code == ExitCode.USAGE
    assert FakeConfig.built == []
    assert f"refusing to bind {host} with security.auth off" in err
    assert "security.allow_unauthenticated: true" in err


def test_the_explicit_opt_in_binds_and_still_warns(
    tmp_path: Path,
    uvicorn_stub: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A proxy that authenticates in front is legitimate, but it has to be said in config."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(tmp_path, "security:\n  allow_unauthenticated: true\n")

    code = main(["serve", "--config", config, "--host", "0.0.0.0"])

    assert code == ExitCode.SUCCESS
    assert FakeConfig.built[0]["host"] == "0.0.0.0"
    assert "WARNING: binding 0.0.0.0 with security.auth off" in capsys.readouterr().err


def test_an_unauthenticated_dashboard_bind_is_refused_too(
    tmp_path: Path,
    uvicorn_stub: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dashboard renders corpus text, so its bind meets the same rule as the API's."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    config = write_config(
        tmp_path, "observability:\n  dashboard: true\n  dashboard_host: 0.0.0.0\n"
    )

    code = main(["serve", "--config", config])

    assert code == ExitCode.USAGE
    assert FakeConfig.built == []
    assert "refusing to bind 0.0.0.0" in capsys.readouterr().err


def test_an_authenticated_wide_bind_starts_without_a_warning(
    tmp_path: Path,
    uvicorn_stub: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key-long-enough")
    config = write_config(tmp_path, "security:\n  auth: true\n")

    code = main(["serve", "--config", config, "--host", "0.0.0.0"])

    assert code == ExitCode.SUCCESS
    assert "WARNING" not in capsys.readouterr().err


def test_every_compose_service_that_binds_wide_without_auth_opts_in() -> None:
    """The image runs `serve --host 0.0.0.0`; without the opt-in no compose stack starts."""
    root = Path(__file__).resolve().parents[3]
    files = [root / "docker-compose.yml"] + [
        root / "deploy" / f"compose.{size}.yml" for size in ("small", "medium", "large")
    ]
    overrides = [
        line
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("FASTERRAG_SET:")
    ]

    assert overrides
    assert all("security.allow_unauthenticated=true" in line for line in overrides), overrides
