"""The two reference documents must describe the surface that exists.

`fasterrag backup` and `restore` shipped absent from `cli-reference.md` while the parser's
own docstring claimed "the surface mirrors docs/cli-reference.md exactly" (TASK-0138). Both
references are currently accurate, and nothing kept them that way — every row and heading
this session was added by hand, which is exactly how the last gap appeared.

Drift in either direction is a defect. A setting with no row is undiscoverable; a row with
no setting sends an operator to configure something that will be rejected as an unknown key.
"""

import argparse
import re
from pathlib import Path

from pydantic import BaseModel

from fasterrag.cli.parser import build_parser
from fasterrag.config.schema import Settings

DOCS = Path(__file__).resolve().parents[3] / "docs"

# Dotted names, allowing digits — `bm25_k1` is a real setting, and a pattern that assumed
# letters-and-underscores silently skipped three rows when this check was first written.
_DOTTED = re.compile(r"^\|\s*`([a-z0-9_]+(?:\.[a-z0-9_]+)+)`\s*\|", re.M)
_HEADING = re.compile(r"^## `fasterrag ([^`<]+)", re.M)


def settings_leaves() -> list[str]:
    """Return every dotted setting the schema declares."""

    def walk(model: BaseModel, prefix: str = "") -> list[str]:
        found: list[str] = []
        for name in type(model).model_fields:
            value = getattr(model, name)
            dotted = f"{prefix}{name}"
            if isinstance(value, BaseModel):
                found.extend(walk(value, f"{dotted}."))
            else:
                found.append(dotted)
        return found

    return sorted(walk(Settings()))


def documented_settings() -> set[str]:
    return set(_DOTTED.findall((DOCS / "config-reference.md").read_text(encoding="utf-8")))


def implemented_commands() -> list[str]:
    """Return every terminal subcommand path the parser accepts."""

    def walk(parser: argparse.ArgumentParser, prefix: str = "") -> list[str]:
        found: list[str] = []
        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in action.choices.items():
                full = f"{prefix}{name}"
                children = walk(sub, f"{full} ")
                found.extend(children or [full])
        return found

    return sorted(set(walk(build_parser())))


def documented_commands() -> set[str]:
    return {
        heading.strip()
        for heading in _HEADING.findall((DOCS / "cli-reference.md").read_text(encoding="utf-8"))
    }


def test_the_schema_walk_finds_settings() -> None:
    """A walk that quietly found nothing would make the parity checks vacuous."""
    assert len(settings_leaves()) > 90


def test_the_reference_table_was_parsed() -> None:
    """Likewise: a pattern matching nothing would report perfect parity with an empty set."""
    assert len(documented_settings()) > 90


def test_every_setting_has_a_reference_row() -> None:
    missing = [name for name in settings_leaves() if name not in documented_settings()]

    assert missing == [], f"settings with no row in config-reference.md: {missing}"


def test_no_reference_row_describes_a_setting_that_does_not_exist() -> None:
    """The loader rejects unknown keys, so a phantom row sends an operator into an error."""
    declared = set(settings_leaves())
    phantom = sorted(name for name in documented_settings() if name not in declared)

    assert phantom == [], f"rows in config-reference.md with no setting: {phantom}"


def test_the_parser_exposes_commands() -> None:
    assert len(implemented_commands()) > 20


def test_every_command_has_a_reference_heading() -> None:
    """A heading may cover a group (`index <subcommand>`), so a prefix match is enough."""
    headings = documented_commands()
    missing = [
        command
        for command in implemented_commands()
        if not any(
            heading == command or command.startswith(f"{heading} ") or heading.startswith(command)
            for heading in headings
        )
    ]

    assert missing == [], f"commands with no heading in cli-reference.md: {missing}"


def test_no_reference_heading_describes_a_command_that_does_not_exist() -> None:
    """A documented command that argparse rejects is worse than an undocumented one."""
    commands = implemented_commands()
    phantom = [
        heading
        for heading in sorted(documented_commands())
        if not any(
            command == heading
            or command.startswith(f"{heading} ")
            or heading.startswith(command.split()[0])
            for command in commands
        )
    ]

    assert phantom == [], f"headings in cli-reference.md with no command: {phantom}"
