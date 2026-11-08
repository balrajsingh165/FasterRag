"""Fail when a document denies something whose existence the tree can decide for itself.

`check_claims.py` asks one question of a denial: is the task id it cites ticked? That
question has a hole its own docstring names and declares unclosable by regex — **an
umbrella id keeps every denial citing it green**. TASK-0049 is one open id covering six
vector-DB adapters. Chroma shipped under it on 2026-11-03, Milvus on 2026-11-07, Weaviate
on 2026-11-08, and the id correctly stayed open for Pinecone the whole time, so five
documents went on calling built adapters unbuilt and every gate passed them. The same hole
produced findings on three separate days in six days.

It is unclosable by regex and it is not unclosable. The fix is to stop asking the ledger
and ask the tree, for the narrow set of claims where the tree gives a yes or no:

* **adapter denials.** ``adapters/*/factory.py`` registers its providers in
  ``_BUILTIN_ADAPTERS`` and names the unwritten ones in ``_PLANNED_ADAPTERS``. That mapping
  is executable truth, maintained by the person who ships an adapter because the adapter
  does not load without it. A denial clause naming a registered provider is false whatever
  id it cites;
* **the structure map.** ``docs/structure.md`` states its own contract in its second line:
  "paths marked *(pending)* do not exist yet; everything else is in the tree today". Both
  halves are decidable against the filesystem, and both halves were wrong — six stale
  ``(pending)`` markers on modules that shipped, and three listed paths that do not exist.

What this gate CANNOT do, stated plainly because a check that oversells its coverage is
the defect it exists to catch:

* **it closes the umbrella hole for providers only.** The hole is general — "the load
  layer is unbuilt (TASK-0243)" was false the same way, and no registry decides a test
  layer. Every capability needs its own executable source of truth or its own id;
* it reads a clause, not a meaning. A denial whose subject sits in a different clause from
  its name is missed, and a column header that denies a whole table column (README's
  "Not built" column lists adapters with no denial verb anywhere) is invisible;
* it cannot see a stale *count* — "four of them are unbuilt", "two adapters pass the
  contract suite" name nothing and decide nothing. That is the counts class;
* "registered" is not "proven". Milvus and Weaviate are registered and have never reached a
  server, so this gate would pass a document calling them production-ready. It decides
  existence, which is the only thing a registry knows.

Exit codes: ``0`` when every decidable denial matches the tree, ``1`` on any finding or if
a vacuity guard fires.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# CRITICAL: the sentence splitter is imported, not reimplemented, private name and all. Two
# gates reading the same documents must agree on where a sentence ends or they disagree about
# which claims exist, and that splitter carries three incidents' worth of table-cell and
# colon handling. A second copy would drift away from it silently.
from check_claims import ROOT_DOCUMENTS, _sentences, scanned_documents

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
ADAPTERS: Final = Path("src/fasterrag/adapters")
STRUCTURE_MAP: Final = Path("docs/structure.md")

ADAPTER_DENIAL: Final = "adapter-denial"
STRUCTURE_PENDING: Final = "structure-pending"
STRUCTURE_ABSENT: Final = "structure-absent"
VACUOUS: Final = "vacuous"

# CRITICAL: every phrasing here denies that a thing EXISTS. "never run", "unverified",
# "not proven" and "skips" are deliberately absent, because they are true of Milvus and
# Weaviate today — both are registered and neither has reached a server. A gate that
# conflates "not built" with "not proven" fires on the most careful sentences in the
# repository and gets switched off.
DENIAL_PATTERNS: Final[tuple[str, ...]] = (
    r"\bun-?built\b",
    r"\bnot\s+(?:yet\s+)?built\b",
    r"\bnot\s+(?:yet\s+)?implemented\b",
    r"\bnot\s+(?:yet\s+)?written\b",
    r"\bnot\s+(?:yet\s+)?shipped\b",
    r"\b(?:does|do)\s+not\s+(?:yet\s+)?exist\b",
    r"\bspecified\s+but\b",
    r"\bpending\b",
)

# CRITICAL: an affirmation immediately before a provider's name means the name is the
# subject of a claim and not of the denial beside it. "doctor-gating shipped for Qdrant
# and pending for Langfuse/Grafana" is the live sentence that proved this necessary: it is
# the gate's only false positive without this rule, and it is entirely true.
AFFIRMATION: Final = re.compile(
    r"\b(?:ship(?:s|ped|ping)?|land(?:s|ed)|passes|pass|built|supported|available)\b",
    re.IGNORECASE,
)
AFFIRMATION_WINDOW: Final = 28

# CRITICAL: clause boundaries, not sentence boundaries. One sentence routinely says what
# shipped and what did not — "Qdrant, pgvector and Chroma ship; Milvus is unbuilt" — and
# pairing a denial with every name in the sentence fails the honest half. Splitting on
# " and " instead would be worse: it breaks the enumeration in "Milvus, Weaviate, and
# Pinecone are specified but not built", which is the exact shape this gate is for.
CLAUSE_BREAK: Final = re.compile(r"\s*[;:]\s*|\s+[\u2014\u2013]\s+|,\s+with\s+")

TREE_ENTRY: Final = re.compile(r"^(?P<lead>[^├└]*)(?:├──|└──)\s*(?P<name>\S+)(?P<rest>.*)$")
PENDING_MARKER: Final = "(pending)"

# CRITICAL: ratcheted by path, never by line. Line numbers in these documents move every
# time a paragraph above them is edited, and a ratchet pinned to a line turns an unrelated
# edit into a failure, which is how a ratchet gets deleted. Empty is the correct state:
# every finding this gate has ever reported was a real one and was fixed, so an entry here
# is a claim that something cannot be fixed and needs the reason written beside it.
RATCHET: Final[dict[str, int]] = {}


@dataclass(frozen=True)
class Finding:
    """One decidable denial the tree contradicts."""

    rule: str
    path: str
    detail: str


def registered_providers(root: Path = REPOSITORY_ROOT) -> tuple[set[str], set[str]]:
    """Return the provider names the adapter factories register, and the ones they plan."""
    shipped: set[str] = set()
    planned: set[str] = set()
    for factory in sorted((root / ADAPTERS).rglob("factory.py")):
        tree = ast.parse(factory.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if node.target.id == "_BUILTIN_ADAPTERS" and isinstance(node.value, ast.Dict):
                shipped |= {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            if node.target.id == "_PLANNED_ADAPTERS" and node.value is not None:
                planned |= {
                    item.value
                    for item in ast.walk(node.value)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
    return shipped, planned - shipped


def _name_pattern(provider: str) -> re.Pattern[str]:
    """Match a provider's registry key as prose spells it, underscore or space or hyphen."""
    return re.compile(rf"\b{re.escape(provider).replace('_', '[ _-]?')}\b", re.IGNORECASE)


def _denied_providers(clause: str, names: dict[str, re.Pattern[str]]) -> list[str]:
    """Return the providers this clause denies, empty when it denies none of them."""
    denial = re.compile("|".join(DENIAL_PATTERNS), re.IGNORECASE).search(clause)
    if denial is None:
        return []
    denied: list[str] = []
    for provider, pattern in names.items():
        match = pattern.search(clause)
        if match is None or match.start() > denial.start():
            continue
        lead = clause[max(0, match.start() - AFFIRMATION_WINDOW) : match.start()]
        if AFFIRMATION.search(lead):
            continue
        denied.append(provider)
    return denied


def adapter_findings(root: Path = REPOSITORY_ROOT) -> tuple[list[Finding], int]:
    """Return a finding per denial of a registered provider, with the mentions scanned."""
    shipped, _ = registered_providers(root)
    names = {provider: _name_pattern(provider) for provider in sorted(shipped)}
    findings: list[Finding] = []
    mentions = 0

    for path in scanned_documents(root):
        relative = path.relative_to(root).as_posix()
        for line, sentence in _sentences(path.read_text(encoding="utf-8")):
            for clause in CLAUSE_BREAK.split(sentence):
                if any(pattern.search(clause) for pattern in names.values()):
                    mentions += 1
                denied = _denied_providers(clause, names)
                if denied:
                    findings.append(
                        Finding(
                            ADAPTER_DENIAL,
                            relative,
                            f"line {line} calls {', '.join(denied)} unbuilt, but "
                            f"adapters/*/factory.py registers it: {clause.strip()[:120]}",
                        )
                    )
    return findings, mentions


def _mapped_paths(body: str) -> list[tuple[int, str, bool]]:
    """Return each tree-diagram entry as its repository path and its pending marker."""
    depths: dict[int, str] = {}
    entries: list[tuple[int, str, bool]] = []
    for number, line in enumerate(body.split("\n"), 1):
        match = TREE_ENTRY.match(line)
        if match is None:
            continue
        depth = len(match.group("lead"))
        name = match.group("name")
        depths = {level: value for level, value in depths.items() if level < depth}
        parent = depths[max(depths)] if depths else ""
        path = f"{parent}{name}"
        if name.endswith("/"):
            depths[depth] = path
        entries.append((number, path, PENDING_MARKER in match.group("rest")))
    return entries


def structure_findings(root: Path = REPOSITORY_ROOT) -> tuple[list[Finding], int]:
    """Return a finding per tree-diagram entry the filesystem contradicts."""
    document = root / STRUCTURE_MAP
    if not document.is_file():
        return [], 0

    relative = STRUCTURE_MAP.as_posix()
    findings: list[Finding] = []
    entries = _mapped_paths(document.read_text(encoding="utf-8"))
    for number, path, pending in entries:
        exists = (root / path).exists()
        if pending and exists:
            findings.append(
                Finding(
                    STRUCTURE_PENDING,
                    relative,
                    f"line {number} marks {path} (pending), but it is in the tree",
                )
            )
        elif not pending and not exists:
            findings.append(
                Finding(
                    STRUCTURE_ABSENT,
                    relative,
                    f"line {number} lists {path} unmarked, but it is not in the tree",
                )
            )
    return findings, len(entries)


def check(root: Path = REPOSITORY_ROOT) -> list[Finding]:
    """Return every decidable denial the tree contradicts, vacuity guards included."""
    shipped, planned = registered_providers(root)
    if not shipped:
        return [
            Finding(
                VACUOUS,
                ADAPTERS.as_posix(),
                "no factory registers any provider, so the gate decided nothing; "
                "_BUILTIN_ADAPTERS is the whole source of truth here",
            )
        ]

    missing = [name for name in ROOT_DOCUMENTS if not (root / name).is_file()]
    if missing:
        return [
            Finding(
                VACUOUS,
                ", ".join(missing),
                "absent from the scan root, so the documents read most often went "
                "unread; a narrowed scan root is the defect TASK-0310 closed",
            )
        ]

    adapters, mentions = adapter_findings(root)
    structure, entries = structure_findings(root)

    if mentions == 0:
        return [
            Finding(
                VACUOUS,
                "docs",
                f"scanned every document and found no clause naming any of the "
                f"{len(shipped)} registered providers; either the documentation stopped "
                "naming them or the registry keys stopped matching how prose spells them",
            )
        ]
    if entries == 0:
        return [
            Finding(
                VACUOUS,
                STRUCTURE_MAP.as_posix(),
                "parsed no tree-diagram entries, so the structure map proved nothing; "
                "the diagram's box-drawing characters are the whole parse",
            )
        ]
    if not planned:
        return [
            Finding(
                VACUOUS,
                ADAPTERS.as_posix(),
                "no factory names a planned provider, so every denial this gate reads "
                "is automatically false and the rule stopped discriminating",
            )
        ]
    return adapters + structure


def over_ratchet(findings: list[Finding], ratchet: dict[str, int] = RATCHET) -> list[str]:
    """Return one message per path over its allowance, and per allowance nothing uses."""
    counted: dict[str, int] = {}
    for finding in findings:
        counted[finding.path] = counted.get(finding.path, 0) + 1

    messages = [
        f"{path}: {count} findings, ratchet allows {ratchet.get(path, 0)}"
        for path, count in sorted(counted.items())
        if count > ratchet.get(path, 0)
    ]
    messages += [
        f"{path}: ratchet allows {allowed} findings and the gate reports none; "
        "delete the entry rather than leaving a licence nothing needs"
        for path, allowed in sorted(ratchet.items())
        if counted.get(path, 0) == 0 and allowed > 0
    ]
    return messages


def main(argv: list[str] | None = None) -> int:
    """Run the gate, reporting every decidable contradiction at once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    findings = check()
    messages = over_ratchet(findings)
    if not messages:
        return 0

    sys.stderr.write("a document denies something the tree says exists:\n")
    for finding in findings:
        sys.stderr.write(f"  [{finding.rule}] {finding.path}: {finding.detail}\n")
    for message in messages:
        sys.stderr.write(f"  {message}\n")
    sys.stderr.write(
        "\nthe registry and the filesystem are the authority here, not the cited task id: "
        "an umbrella id stays open while the thing it covers ships. Rewrite the sentence "
        "to describe what is in the tree.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
