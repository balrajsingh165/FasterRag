"""``cli/main``: the single place a typed error becomes an exit code.

Nothing else in the CLI catches broadly, so these paths are the difference between an
operator seeing a problem line and seeing a traceback.

The module is fetched by name rather than imported as ``from fasterrag.cli import main``:
the package re-exports the *function* under that name, so the plain import binds the entry
point and monkeypatching its module-level tables silently fails.
"""

import argparse
from importlib import import_module

import pytest

from fasterrag.cli.commands.autopilot import run_autopilot, run_generate_golden_set
from fasterrag.cli.commands.pipeline import run_index
from fasterrag.cli.commands.portability import run_export, run_import
from fasterrag.cli.commands.traces import run_traces
from fasterrag.cli.main import _dispatch, _resolve, main
from fasterrag.cli.output import Console, ExitCode
from fasterrag.errors import FasterRagError, ProvisioningError

cli_main = import_module("fasterrag.cli.main")


def dispatch(command: str, console: Console, **values: object) -> ExitCode:
    """Run the dispatcher against a hand-built namespace, bypassing the parser."""
    import asyncio

    args = argparse.Namespace(command=command, **values)
    return asyncio.run(_dispatch(args, console))


@pytest.mark.parametrize(
    ("command", "handler"),
    [
        ("export", run_export),
        ("import", run_import),
        ("index", run_index),
        ("traces", run_traces),
    ],
)
def test_the_commands_resolved_by_name_reach_their_handler(command: str, handler: object) -> None:
    assert _resolve(argparse.Namespace(command=command, action="list")) is handler


def test_autopilot_run_and_its_generator_are_different_handlers() -> None:
    """Both are ``autopilot``; only the action separates tuning from producing its input."""
    assert _resolve(argparse.Namespace(command="autopilot", action="run")) is run_autopilot
    assert (
        _resolve(argparse.Namespace(command="autopilot", action="generate-golden-set"))
        is run_generate_golden_set
    )


def test_a_command_with_no_handler_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = dispatch("nonexistent", Console(), action=None)

    assert code == ExitCode.USAGE
    assert "not implemented yet" in capsys.readouterr().err


def test_an_unknown_subcommand_is_a_usage_error() -> None:
    assert dispatch("config", Console(), action="teleport") == ExitCode.USAGE


def test_a_pending_command_names_the_slice_that_will_ship_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The list is empty today; the mechanism is what stops the next one printing nothing useful."""
    monkeypatch.setattr(cli_main, "PENDING_COMMANDS", {"dashboard": "TASK-0123"})

    code = dispatch("dashboard", Console(), action=None)

    assert code == ExitCode.USAGE
    assert "ships with TASK-0123" in capsys.readouterr().err


def test_a_provisioning_error_exits_four_with_its_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def raising(args: argparse.Namespace, console: Console) -> ExitCode:
        raise ProvisioningError("docker is not running", fix="start Docker Desktop")

    monkeypatch.setitem(cli_main._HANDLERS, "doctor", raising)

    code = dispatch("doctor", Console(), action=None)

    err = capsys.readouterr().err
    assert code == ExitCode.PREFLIGHT
    assert "fix: start Docker Desktop" in err


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_typed_error_escaping_a_handler_becomes_its_documented_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    retryable: bool,
    expected: ExitCode,
) -> None:
    async def raising(args: argparse.Namespace, console: Console) -> ExitCode:
        raise FasterRagError("the backend is not answering", retryable=retryable)

    monkeypatch.setitem(cli_main._HANDLERS, "status", raising)

    code = dispatch("status", Console(), action=None)

    err = capsys.readouterr().err
    assert code == expected
    assert "the backend is not answering" in err
    assert "Traceback" not in err


def test_an_interrupt_exits_one_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator pressing Ctrl-C knows what happened; a stack trace only hides their output."""

    async def interrupted(args: argparse.Namespace, console: Console) -> ExitCode:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main, "_dispatch", interrupted)

    code = main(["status"])

    assert code == ExitCode.FAILURE
    assert "interrupted" in capsys.readouterr().err


def test_the_json_flag_reaches_the_console(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Console] = []

    async def record(args: argparse.Namespace, console: Console) -> ExitCode:
        seen.append(console)
        return ExitCode.SUCCESS

    monkeypatch.setattr(cli_main, "_dispatch", record)

    main(["--json", "status"])

    assert seen[0].as_json is True


def test_the_quiet_flag_reaches_the_console(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Console] = []

    async def record(args: argparse.Namespace, console: Console) -> ExitCode:
        seen.append(console)
        return ExitCode.SUCCESS

    monkeypatch.setattr(cli_main, "_dispatch", record)

    main(["status", "--quiet"])

    assert seen[0].quiet is True


def test_main_returns_a_plain_int_a_shell_can_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sys.exit`` on an IntEnum works, but a caller comparing to 0 should not need to know."""

    async def succeed(args: argparse.Namespace, console: Console) -> ExitCode:
        return ExitCode.SUCCESS

    monkeypatch.setattr(cli_main, "_dispatch", succeed)

    assert type(main(["status"])) is int
