"""Every flag in ``cli-reference.md`` must exist, and every flag that exists must be in it.

``test_reference_parity.py`` gates the CLI reference at the subcommand level — a command with
no heading, or a heading with no command, fails — and reported the reference accurate while
its flags were stale. Flags were not gated at all: ``index reembed --watch`` shipped
accepted-and-ignored (TASK-0249), and ``test_flags_are_wired.py`` could not see it because it
searches every CLI module at once, so ``ingest``'s notice satisfied the assertion on
``reembed``'s behalf. Flags are where drift hides, so this gate is per command.

Salvaged from an abandoned 2026-09-13 worktree under TASK-0284 and landed 2026-11-09 after the
seven defects it reports were fixed: ``autopilot run --golden-set`` documented and nonexistent,
``--dataset``/``--out``/``--iterations``/``--query``/``--to`` accepted and undocumented, and
``index reembed``'s source positional missing from its usage line.

Both sides come from the real thing. The implemented surface is walked out of
``build_parser()`` through ``parser._actions``, never a hand-written list, because a
hand-written list is a third copy that drifts on its own. The documented surface is parsed
out of ``cli-reference.md`` per command, under rules chosen deliberately:

* A command's section runs from its ``## `fasterrag ...``` heading to the next ``##``
  heading, subsections included. A flag mentioned anywhere in it — a table row, backticked
  prose, a fenced example, the heading itself — is a claim that this command accepts it.
* A group heading (``index <subcommand>``) documents its subcommands as table rows. The row
  whose first cell is ``reembed NAME`` is ``index reembed``'s section, and a flag mentioned
  in that row belongs to ``index reembed`` alone; a flag mentioned in the group's prose
  belongs to the group parser itself. Every such row must name a real subcommand.
* A backticked span or example line that opens with a different command is that command's
  example, not a claim about this one, so ``estimate`` may say "exactly as
  ``ingest --recursive`` does" without crediting ``estimate`` with ``ingest``'s flag.
* The global flags — the ones ``build_parser()`` attaches to the root — are documented once,
  in the table before the first heading, and never required again per command. That table
  must match the root parser exactly, and every parser below the root must accept every
  global flag: that is the premise that makes documenting them once honest, and it is
  asserted rather than assumed. A section may still mention a global flag where its
  behaviour is command-specific (``doctor --json``), and the mention is checked like any
  other.
* ``-h``/``--help`` belongs to argparse and is not a documented flag. An action whose help
  is ``argparse.SUPPRESS`` is hidden from ``--help`` on purpose and is hidden here for the
  same reason.
* Positionals are compared by count and required-ness rather than by name: the reference
  writes ``<path|url> [...]`` where the parser says ``sources``, and a metavar is not a
  contract. ``[...]`` continues the slot before it, ``[NAME]`` is an optional slot, and a
  value after an option that takes one (``export --out <archive>``) is that option's value.
  A group's ``<subcommand>`` placeholder is the slot its subparser action occupies.
* A ``##`` heading that is not a command (there are none today) is nobody's section and is
  not read.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import takewhile

from fasterrag.cli.parser import build_parser
from tests.unit.docs.test_reference_parity import DOCS, implemented_commands

REFERENCE = DOCS / "cli-reference.md"
HELP = frozenset({"-h", "--help"})
PREAMBLE = ""

_H2 = re.compile(r"^## ", re.M)
_COMMAND_HEADING = re.compile(r"`fasterrag ([^`]+)`")
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
_SPAN = re.compile(r"`([^`\n]+)`")
_ROW = re.compile(r"^\|\s*`([^`\n]+)`\s*(\|.*)$", re.M)
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_WORD = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class Surface:
    """What one parser accepts.

    ``options`` maps each option string to whether it takes a value; ``positionals`` is the
    required-ness of each positional in declaration order, a subparser action counting as
    one required slot; ``group`` says whether the parser dispatches to subcommands.
    """

    options: dict[str, bool]
    positionals: tuple[bool, ...]
    group: bool


@dataclass(frozen=True)
class Documented:
    """What the reference says about one command.

    ``usage`` is the tail of its heading (or row cell) after the command words; ``spans`` is
    every backticked span and fenced-example line in its section, the heading included.
    """

    usage: tuple[str, ...]
    spans: tuple[str, ...]


def _surface(parser: argparse.ArgumentParser) -> Surface:
    """Read one parser's option strings and positionals off its actions."""
    options: dict[str, bool] = {}
    positionals: list[bool] = []
    group = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            group = True
            positionals.append(True)
        elif action.help == argparse.SUPPRESS:
            continue
        elif action.option_strings:
            for flag in action.option_strings:
                if flag not in HELP:
                    options[flag] = action.nargs != 0
        else:
            positionals.append(action.nargs not in ("?", "*"))
    return Surface(options, tuple(positionals), group)


def parsers_by_path() -> dict[str, argparse.ArgumentParser]:
    """Return every parser below the root keyed by its command path, groups included."""

    def walk(parser: argparse.ArgumentParser, prefix: str) -> dict[str, argparse.ArgumentParser]:
        found: dict[str, argparse.ArgumentParser] = {}
        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for name, sub in action.choices.items():
                path = f"{prefix}{name}"
                found[path] = sub
                found.update(walk(sub, f"{path} "))
        return found

    return walk(build_parser(), "")


def implemented_surface() -> dict[str, Surface]:
    """Return what every parser below the root accepts, keyed by command path."""
    return {path: _surface(parser) for path, parser in parsers_by_path().items()}


def global_flags() -> dict[str, bool]:
    """Return the root parser's option strings: the flags documented once for every command."""
    return _surface(build_parser()).options


def root_commands() -> frozenset[str]:
    """Return the first word of every command, for recognising another command's example."""
    return frozenset(path.split()[0] for path in parsers_by_path())


def _spans(text: str) -> list[str]:
    """Return every fenced-example line and every backticked span outside a fence."""
    found: list[str] = []
    for fence in _FENCE.findall(text):
        found.extend(line for line in fence.splitlines() if line.strip())
    found.extend(_SPAN.findall(_FENCE.sub("", text)))
    return found


def _split_usage(tokens: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Split ``reembed NAME <path> [...]`` into its command words and the usage tail."""
    words = list(takewhile(_WORD.match, tokens))
    return " ".join(words), tuple(tokens[len(words) :])


def documented_surface(groups: Collection[str]) -> dict[str, Documented]:
    """Parse the reference into one entry per command it documents.

    ``groups`` names the commands whose section documents subcommands as table rows; the
    parser decides that, so a heading that is a group is read as one. The preamble, which
    holds the global-flag table, is filed under :data:`PREAMBLE`.
    """
    chunks = _H2.split(REFERENCE.read_text(encoding="utf-8"))
    found: dict[str, Documented] = {PREAMBLE: Documented((), tuple(_spans(chunks[0])))}
    for chunk in chunks[1:]:
        heading, _, body = chunk.partition("\n")
        match = _COMMAND_HEADING.match(heading)
        if match is None:
            continue
        path, usage = _split_usage(match.group(1).split())
        if path not in groups:
            found[path] = Documented(usage, tuple(_spans(heading) + _spans(body)))
            continue
        prose = [heading]
        for row in _ROW.finditer(body):
            cell, rest = row.group(1), row.group(2)
            tokens = cell.split()
            if not tokens or not _WORD.match(tokens[0]):
                prose.append(row.group(0))
                continue
            sub_path, sub_usage = _split_usage(tokens)
            found[f"{path} {sub_path}"] = Documented(
                sub_usage, tuple(_spans(f"`{path} {cell}`") + _spans(rest))
            )
        found[path] = Documented(usage, tuple(_spans("\n".join(prose) + _ROW.sub("", body))))
    return found


def _related(one: str, other: str) -> bool:
    """Return whether one command path is a word-wise prefix of the other."""
    first, second = one.split(), other.split()
    shorter = min(len(first), len(second))
    return first[:shorter] == second[:shorter]


def mentioned_flags(path: str, spans: Iterable[str], roots: Collection[str]) -> set[str]:
    """Return the flags the spans claim for ``path``, skipping another command's examples."""
    found: set[str] = set()
    for span in spans:
        tokens = span.split()
        if tokens and tokens[0] == "$":
            tokens = tokens[1:]
        if tokens and tokens[0] == "fasterrag":
            tokens = tokens[1:]
        if tokens and tokens[0] in roots:
            words, _ = _split_usage(tokens)
            if not _related(words, path):
                continue
        found.update(_FLAG.findall(span))
    return found


def documented_positionals(usage: Sequence[str], options: Mapping[str, bool]) -> tuple[bool, ...]:
    """Reduce a usage tail to the required-ness of each positional slot it shows."""
    slots: list[bool] = []
    tokens = list(usage)
    while tokens:
        token = tokens.pop(0)
        if token.startswith("-"):
            if options.get(token, False) and tokens and not tokens[0].startswith("-"):
                tokens.pop(0)
            continue
        if token == "[...]":
            continue
        slots.append(not token.startswith("["))
    return tuple(slots)


def _documented() -> dict[str, Documented]:
    """The reference, parsed with the parser's own notion of which commands are groups."""
    return documented_surface(
        {path for path, surface in implemented_surface().items() if surface.group}
    )


def _documented_flags() -> dict[str, set[str]]:
    """Every command's documented flags, the preamble's being the global table."""
    roots = root_commands()
    return {
        path: mentioned_flags(path, entry.spans, roots) for path, entry in _documented().items()
    }


def test_the_parser_walk_finds_flags() -> None:
    """A walk that quietly found nothing would make every check below vacuous."""
    shared = global_flags().keys()
    own = sum(len(surface.options.keys() - shared) for surface in implemented_surface().values())

    assert own > 40


def test_the_reference_was_parsed_per_command() -> None:
    """Likewise: a pattern matching nothing reports perfect parity against an empty set."""
    documented = _documented_flags()

    assert len(documented) > 20
    assert sum(len(flags) for flags in documented.values()) > 50


def test_the_walk_agrees_with_the_subcommand_gate() -> None:
    """The two walks read the same parser; disagreeing would mean one of them is wrong."""
    terminal = sorted(path for path, surface in implemented_surface().items() if not surface.group)

    assert terminal == implemented_commands()


def test_every_parser_below_the_root_accepts_every_global_flag() -> None:
    """The premise of documenting a global flag once: it must really be valid everywhere."""
    shared = set(global_flags())
    refusing = {
        path: sorted(shared - surface.options.keys())
        for path, surface in implemented_surface().items()
        if not shared <= surface.options.keys()
    }

    assert refusing == {}, f"parsers that do not accept every global flag: {refusing}"


def test_the_global_flag_table_matches_the_root_parser() -> None:
    """The preamble is the one place a global flag is documented, so it must be exact."""
    documented = _documented_flags()[PREAMBLE]
    declared = set(global_flags())

    assert sorted(declared - documented) == [], "global flags missing from the preamble"
    assert sorted(documented - declared) == [], "preamble flags the root parser lacks"


def test_every_row_in_a_group_section_names_a_subcommand() -> None:
    """A row for a subcommand argparse rejects is worse than a missing one."""
    implemented = implemented_surface()
    phantom = sorted(path for path in _documented() if path and path not in implemented)

    assert phantom == [], f"cli-reference.md documents commands that do not exist: {phantom}"


def test_every_flag_is_documented_under_its_command() -> None:
    """Direction one: a flag with no mention in its own command's section is undiscoverable."""
    shared = global_flags().keys()
    documented = _documented_flags()
    missing = {
        path: sorted(surface.options.keys() - shared - documented.get(path, set()))
        for path, surface in sorted(implemented_surface().items())
        if not surface.options.keys() - shared <= documented.get(path, set())
    }

    assert missing == {}, (
        f"flags with no mention under their command in cli-reference.md: {missing}"
    )


def test_no_documented_flag_is_missing_from_its_command() -> None:
    """Direction two: a documented flag argparse rejects sends an operator into exit 2."""
    implemented = implemented_surface()
    phantom = {
        path: sorted(flags - implemented[path].options.keys())
        for path, flags in sorted(_documented_flags().items())
        if path in implemented and not flags <= implemented[path].options.keys()
    }

    assert phantom == {}, f"flags cli-reference.md documents that the command lacks: {phantom}"


def test_every_command_documents_its_positionals() -> None:
    """A usage line showing fewer (or more) arguments than the parser takes is a lie."""
    implemented = implemented_surface()
    wrong = {
        path: (
            documented_positionals(entry.usage, implemented[path].options),
            " ".join(entry.usage),
        )
        for path, entry in sorted(_documented().items())
        if path in implemented
        and documented_positionals(entry.usage, implemented[path].options)
        != implemented[path].positionals
    }

    assert wrong == {}, f"usage lines whose positionals differ from the parser's: {wrong}"
