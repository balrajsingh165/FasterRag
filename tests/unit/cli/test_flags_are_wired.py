"""Every CLI flag must be read by the command that declares it.

`--recursive` shipped declared-and-never-read: argparse accepted it, `cli-reference.md`
documented it, and directory ingestion silently indexed nothing (TASK-0188). Auditing the
parser afterwards found five more in the same state. This is the guard that stops a sixth.
"""

import re
from pathlib import Path

CLI = Path(__file__).resolve().parents[3] / "src" / "fasterrag" / "cli"

# Flags argparse fills for every command, read by the shared plumbing rather than by any
# one command, plus the suppressed catch-all positional for unimplemented commands.
SHARED = {"config", "collection", "overrides", "as_json", "quiet", "verbose", "rest"}

# Declared, documented as "(not yet implemented)", and answered with a message at the point
# of use rather than silently ignored. Each cites the task that will implement it.
# `fix` left this set when TASK-0197 landed; `watch` is still waiting on TASK-0196.
PENDING = {"watch": "TASK-0196"}


def declared() -> dict[str, str]:
    """Return every argparse destination the parser declares, mapped to its first flag."""
    source = (CLI / "parser.py").read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for match in re.finditer(
        r'add_argument\(\s*("[^"]+"(?:\s*,\s*"[^"]+")*)([^)]*)\)', source, re.S
    ):
        names = re.findall(r'"([^"]+)"', match.group(1))
        explicit = re.search(r'dest="([^"]+)"', match.group(2))
        dest = explicit.group(1) if explicit else max(names, key=len).lstrip("-").replace("-", "_")
        found[dest] = names[0]
    return found


def command_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in CLI.rglob("*.py") if path.name != "parser.py"
    )


def test_the_parser_declares_something() -> None:
    """A regex that silently matched nothing would make every test below vacuous."""
    assert len(declared()) > 40


def test_every_declared_flag_is_read_somewhere() -> None:
    consumers = command_source()
    dead = [
        f"{flag} (args.{dest})"
        for dest, flag in sorted(declared().items())
        if dest not in SHARED
        and not re.search(rf"\b(args\.{dest}|getattr\(args,\s*\"{dest}\")", consumers)
    ]

    assert dead == [], f"declared but never read: {dead}"


def test_a_pending_flag_says_so_in_its_help() -> None:
    """A flag that does nothing must not read as a working one in --help."""
    source = (CLI / "parser.py").read_text(encoding="utf-8")
    for dest in PENDING:
        pattern = rf'"--{dest}"[^)]*?\(not yet implemented\)'
        assert re.search(pattern, source, re.S), f"--{dest} does not declare itself unimplemented"


def test_a_pending_flag_cites_the_task_that_will_land_it() -> None:
    source = (CLI / "parser.py").read_text(encoding="utf-8")
    for dest, task in PENDING.items():
        assert task in source, f"--{dest} is pending with no task cited"


def test_a_pending_flag_is_answered_at_the_point_of_use() -> None:
    """Help text is read once; the message is what reaches somebody who passed the flag."""
    consumers = command_source()
    for dest in PENDING:
        assert re.search(rf'getattr\(args,\s*"{dest}"', consumers), (
            f"--{dest} is not checked by any command, so passing it is silent"
        )
        assert f"--{dest} is not implemented yet" in consumers
