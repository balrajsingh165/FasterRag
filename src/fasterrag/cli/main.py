"""The ``fasterrag`` entry point.

Dispatch only. Every command delegates to the same service layer the REST API calls, which
is what makes ``docs/cli-reference.md``'s claim that "both call the same service layer, so
behavior is identical" true rather than aspirational — a command that reimplemented a
service would drift from it the first time either changed.

One place catches ``FasterRagError``: an uncaught traceback is a terrible CLI error message,
and the taxonomy already carries the stable code, the trace id, and often the fix. Nothing
else in the CLI catches broadly, so a typed error reaches here with its classification
intact.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Final

from fasterrag.cli.commands.autopilot import run_autopilot, run_generate_golden_set
from fasterrag.cli.commands.backup import run_backup, run_restore
from fasterrag.cli.commands.benchmark import run_benchmark
from fasterrag.cli.commands.diagnostics import (
    run_config_init,
    run_config_validate,
    run_doctor_command,
    run_status,
)
from fasterrag.cli.commands.infrastructure import run_estimate, run_provision
from fasterrag.cli.commands.pipeline import run_index, run_ingest, run_query
from fasterrag.cli.commands.portability import run_export, run_import
from fasterrag.cli.commands.processes import run_serve, run_worker
from fasterrag.cli.commands.replay import run_replay
from fasterrag.cli.commands.traces import run_traces
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import PENDING_COMMANDS, build_parser
from fasterrag.errors import FasterRagError, ProvisioningError
from fasterrag.observability.logging import configure_logging

Handler = Callable[[argparse.Namespace, Console], Awaitable[ExitCode]]

__all__ = ["main"]

_HANDLERS: Final[dict[str, Handler]] = {
    "backup": run_backup,
    "benchmark": run_benchmark,
    "restore": run_restore,
    "doctor": run_doctor_command,
    "estimate": run_estimate,
    "ingest": run_ingest,
    "provision": run_provision,
    "query": run_query,
    "replay": run_replay,
    "serve": run_serve,
    "status": run_status,
    "worker": run_worker,
}

# CRITICAL: `index` dispatches on its own action rather than through this table, because its
# three subcommands share one adapter and one config load. A `config`-style entry per action
# would open and close a connection three times over for one command.
_SUBCOMMAND_HANDLERS: Final[dict[tuple[str, str], Handler]] = {
    ("config", "init"): run_config_init,
    ("config", "validate"): run_config_validate,
}


def _resolve(args: argparse.Namespace) -> Handler | None:
    """Return the handler for the parsed command, or ``None`` if there is none."""
    if args.command == "export":
        return run_export
    if args.command == "import":
        return run_import
    if args.command == "index":
        return run_index
    if args.command == "traces":
        return run_traces
    if args.command == "autopilot":
        return (
            run_generate_golden_set
            if getattr(args, "action", None) == "generate-golden-set"
            else run_autopilot
        )

    action = getattr(args, "action", None)
    if action is not None:
        return _SUBCOMMAND_HANDLERS.get((args.command, action))
    return _HANDLERS.get(args.command)


async def _dispatch(args: argparse.Namespace, console: Console) -> ExitCode:
    """Run the requested command, translating a typed error into an exit code."""
    if args.command in PENDING_COMMANDS:
        console.error(
            f"'{args.command}' is not implemented yet; it ships with "
            f"{PENDING_COMMANDS[args.command]}"
        )
        return ExitCode.USAGE

    handler = _resolve(args)
    if handler is None:
        console.error(f"'{args.command}' is not implemented yet")
        return ExitCode.USAGE

    try:
        return await handler(args, console)
    except ProvisioningError as exc:
        console.problem(exc.code.value, exc.detail, exc.fix)
        return ExitCode.PREFLIGHT
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the command, and return a process exit code.

    ``KeyboardInterrupt`` exits 1 rather than dumping a traceback: an operator pressing
    Ctrl-C already knows what happened, and a stack trace only obscures whatever partial
    output they were reading.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console(as_json=args.as_json, quiet=args.quiet)
    configure_logging(level="debug" if args.verbose else "warning")

    try:
        return int(asyncio.run(_dispatch(args, console)))
    except KeyboardInterrupt:
        console.error("interrupted")
        return int(ExitCode.FAILURE)


if __name__ == "__main__":
    sys.exit(main())
