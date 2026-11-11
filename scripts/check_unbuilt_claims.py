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
  ``(pending)`` markers on modules that shipped, and three listed paths that do not exist;
* **test layers** (TASK-0369). ``[tool.pytest.ini_options].markers`` registers every tier
  the suite knows, and ``--strict-markers`` rejects a marker that is not there, so the list
  is executable truth the same way a factory registry is. A tier exists when
  ``tests/<tier>/`` holds a test module or any test carries its marker, which is why
  ``reliability.md`` calling the load harness "specified but not built (TASK-0243)" was
  false while ``tests/load/`` held five modules;
* **an inherited denial** (TASK-0371), in a table column header or in a prose lead-in
  label. README's capability table puts the denial in the header — "Not yet" — and no cell
  under it carries a denial verb, so both gates read those cells as ordinary prose; that is
  how "Milvus/Weaviate/Pinecone adapters" and "load and soak test layers" sat in that
  column after they shipped. "**Specified but not built** (TASK-0243): load … and 24 h
  soak …" is the same construction written as prose. The list is split into items, and an
  item is decided only when the item IS the thing: a list of registry or tier names, one
  optional count before each and one optional noun after the last, and nothing else.

What this gate CANNOT do, stated plainly because a check that oversells its coverage is
the defect it exists to catch:

* **the umbrella hole stays open for every capability with no registry.** Providers and
  test tiers have one; degradation rungs and D-features do not. ``hybrid_only`` is a module
  constant and ``cache_only`` appears nowhere, so a grep decides a rung by whether a
  docstring happens to spell it — which is not a registry, and a wrong *yes* there fails an
  honest sentence. A per-item sub-ledger was considered and rejected: it is a second
  document asserting things about the tree, with nothing checking it, and anything able to
  check it could read the tree directly and not need it. Those classes need review;
* it reads a clause, not a meaning. A denial whose subject sits in a different clause from
  its name is missed, and a denied column item is decided only when the whole item is the
  thing — "a Milvus or Weaviate contract run against a real server" is correctly left alone,
  and so is any real defect worded that loosely;
* it cannot see a stale *count* — "four of them are unbuilt", "two adapters pass the
  contract suite" name nothing and decide nothing. That is the counts class;
* "registered" is not "proven". Milvus and Weaviate are registered and have never reached a
  server, and ``tests/load/`` has never been pointed at fasterRag. This gate decides
  existence, which is the only thing a registry knows;
* it does not guard the *number* of denying column headers, deliberately. The vacuity guard
  fires when no table parses at all, because the parse is the machinery; moving README's
  denial into its cells is one of the two fixes TASK-0371 names, and a guard on the header
  count would fail the repository for taking it.

Measured over four trees (237f6f5, 671156a, 650943b, HEAD): 574 tables, 19 items under a
denying header, 1,177 tier clause-mentions, **3 findings and no false positive**. All three
are on 650943b — the tree the previous sweep produced, which every gate scored green: two
README items under "Not yet" (a registered adapter, a tier with tests) and reliability.md:56.

Exit codes: ``0`` when every decidable denial matches the tree, ``1`` on any finding or if
a vacuity guard fires.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
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
PROJECT_FILE: Final = Path("pyproject.toml")
TESTS: Final = Path("tests")

PROVIDER: Final = "provider"
TIER: Final = "tier"

ADAPTER_DENIAL: Final = "adapter-denial"
LAYER_DENIAL: Final = "layer-denial"
COLUMN_DENIAL: Final = "column-denial"
LABEL_DENIAL: Final = "label-denial"
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

# CRITICAL: a table may sit inside a blockquote, and README's capability table — the one
# TASK-0371 is about — does. Stripping the quote marker before matching is the difference
# between parsing 152 tables and parsing 151, with the missing one being the only table in
# the repository whose header denies a column.
QUOTE_PREFIX: Final = re.compile(r"^\s*(?:>\s*)*")
TABLE_RULE: Final = re.compile(r"^\|[\s:|-]+\|$")
TABLE_ROW: Final = re.compile(r"^\|.*\|$")

# CRITICAL: matched against a whole header cell, not searched inside a sentence, and only
# for cells short enough to be a label. A header is "Not yet"; a paragraph that happens to
# contain "planned" is prose and denies nothing. The corpus measurement that set this list
# is one denying header over 152 tables, so the rule's entire input is README's third
# column and widening it buys nothing but exposure.
DENYING_HEADER: Final = re.compile(
    r"^(?:not\s+yet(?:\s+(?:built|implemented|shipped|started))?|not\s+(?:built|implemented|"
    r"shipped)|un-?built|planned|unimplemented|to\s*-?\s*build|to\s*do|backlog|missing)$",
    re.IGNORECASE,
)
HEADER_LABEL_LIMIT: Final = 24
LABEL_LIMIT: Final = 48

# CRITICAL: a denied column cell is a list of capabilities, and the separator this
# repository uses between them is the middle dot. Commas are deliberately absent: the cells
# enumerate adapters with commas inside one item ("Milvus, Weaviate and Pinecone adapters"),
# which is the shape the adapter rule exists for, and splitting there would destroy it.
ITEM_BREAK: Final = re.compile(r"\s*[·•;]\s*")

DISCARD: Final = " .,:;*-" + chr(0x2014) + chr(0x2013)
MARKUP: Final = re.compile(r"[*_`]+")
LINK: Final = re.compile(r"\[([^\]]*)\]\([^)]*\)")
PARENTHETICAL: Final = re.compile(r"\s*\([^)]*\)")
ARTICLE: Final = re.compile(r"^(?:the|a|an|any|its|our|their)\s+", re.IGNORECASE)
COPULA: Final = re.compile(
    r"\s+(?:is|are|was|were|remains?|stays?|has|have|had|been|still|all|both)$",
    re.IGNORECASE,
)
JOINER: Final = r"\s*(?:/|,\s*|\+|\band\b|\bor\b|\bnor\b)\s*"

# CRITICAL: a count or duration may sit in front of a name and nothing else may. "24 h soak"
# is the live reliability.md wording, and without it the one sentence TASK-0369 is named for
# is undecidable. General leading modifiers were rejected: they would admit "a semantic cache
# load", where the name is the last word of something that is not the thing.
QUALIFIER: Final = r"(?:\d+\s*[a-z]{0,2}[\s-]+)?"

# CRITICAL: the noun an item may end with, and the whole precision of the column rule. An
# item is decided only when it is a list of registry names plus one of these and nothing
# else, so "Pinecone adapter" is decided and "a Milvus or Weaviate contract run against a
# real server" is not — the second denies a contract run, not an adapter, and it is the
# live sentence that a header-inheritance rule gets wrong if it merely looks for a name.
ADAPTER_NOUNS: Final = (
    r"(?:adapter|backend|provider|integration|driver|client|store|connector)s?|support"
)
LAYER_NOUNS: Final = r"(?:(?:test[\s-]+)?(?:layer|tier|suite|harness|run)e?s?|tests)"

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


# CRITICAL: parsed as code, never grepped. This gate's own tests write "@pytest.mark.soak"
# inside a string fixture, and the substring version of this function read that as the soak
# tier existing: a gate fooled by a test about itself, which is the shape of defect the gate
# is for. A string constant is not an ast.Attribute, so the parse cannot be fooled that way.
def _markers_carried(path: Path) -> set[str]:
    """Return the pytest markers this test module applies, read as code and not as text."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    decorated: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            decorated += node.decorator_list
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        ):
            decorated.append(node.value)
    return {
        node.attr
        for expression in decorated
        for node in ast.walk(expression)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
    }


# CRITICAL: the vocabulary is every registered marker, not the four `addopts` deselects. At
# 650943b `addopts` carried no `-m` expression at all, so deriving tiers from it would have
# dropped `soak` out of the vocabulary on the exact tree whose reliability.md was wrong about
# soak's neighbour in the same enumeration, and the rule would have decided nothing there.
def registered_layers(root: Path = REPOSITORY_ROOT) -> tuple[set[str], set[str]]:
    """Return the pytest tiers that have test code, and the registered ones that do not."""
    project = root / PROJECT_FILE
    if not project.is_file():
        return set(), set()
    with project.open("rb") as handle:
        options = tomllib.load(handle).get("tool", {}).get("pytest", {}).get("ini_options", {})
    names = {str(entry).split(":", 1)[0].strip() for entry in options.get("markers", [])}
    names = {name for name in names if name}

    carried = {
        marker for path in (root / TESTS).rglob("test_*.py") for marker in _markers_carried(path)
    }
    built = {
        name for name in names if name in carried or any((root / TESTS / name).glob("test_*.py"))
    }
    return built, names - built


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


def _normalised(text: str) -> str:
    """Strip markdown, links, parentheticals, articles and a trailing copula from an item."""
    plain = PARENTHETICAL.sub("", MARKUP.sub("", LINK.sub(r"\1", text)))
    plain = re.sub(r"\s+", " ", plain).strip(DISCARD)
    previous = ""
    while plain != previous:
        previous = plain
        plain = COPULA.sub("", ARTICLE.sub("", plain)).strip()
    return plain


@dataclass(frozen=True)
class Oracle:
    """One vocabulary the tree decides: every name it knows, the built ones, their noun."""

    name: str
    known: frozenset[str]
    built: frozenset[str]
    noun: str
    source: str


def oracles(root: Path = REPOSITORY_ROOT) -> tuple[Oracle, ...]:
    """Return the vocabularies a denial can be decided against: providers and test tiers."""
    shipped, planned = registered_providers(root)
    built, unbuilt = registered_layers(root)
    return (
        Oracle(
            PROVIDER,
            frozenset(shipped | planned),
            frozenset(shipped),
            ADAPTER_NOUNS,
            "adapters/*/factory.py registers it",
        ),
        Oracle(
            TIER,
            frozenset(built | unbuilt),
            frozenset(built),
            LAYER_NOUNS,
            "pyproject registers that tier and tests/ holds it",
        ),
    )


def _subject_names(text: str, known: frozenset[str], built: frozenset[str], noun: str) -> list[str]:
    """Return the built names an item denies, empty unless the item IS a list of names.

    Parsed over every known name and reported over the built ones: the enumerations that
    caused this mix the two, and only a trailing noun may follow them.
    """
    if not known:
        return []
    plain = _normalised(text)
    alternatives = "|".join(
        sorted((re.escape(name).replace("_", "[ _-]?") for name in known), key=len, reverse=True)
    )
    whole = re.compile(
        rf"^{QUALIFIER}(?:{alternatives})(?:{JOINER}{QUALIFIER}(?:{alternatives}))*"
        rf"(?:\s+(?:{noun}))?$",
        re.IGNORECASE,
    )
    if whole.match(plain) is None:
        return []
    return sorted(name for name in built if _name_pattern(name).search(plain))


def denied_columns(root: Path = REPOSITORY_ROOT) -> tuple[list[tuple[str, int, str]], int]:
    """Return every item sitting under a denying column header, with the tables parsed."""
    items: list[tuple[str, int, str]] = []
    tables = 0
    for path in scanned_documents(root):
        relative = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").split("\n")
        columns: list[int] = []
        for number, line in enumerate(lines, 1):
            stripped = QUOTE_PREFIX.sub("", line).strip()
            if not TABLE_ROW.match(stripped):
                columns = []
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if TABLE_RULE.match(stripped):
                continue
            if number < len(lines) and TABLE_RULE.match(
                QUOTE_PREFIX.sub("", lines[number]).strip()
            ):
                tables += 1
                columns = [
                    index
                    for index, cell in enumerate(cells)
                    if len(cell) <= HEADER_LABEL_LIMIT and DENYING_HEADER.match(_normalised(cell))
                ]
                continue
            for index in columns:
                if index >= len(cells):
                    continue
                items += [
                    (relative, number, item)
                    for item in ITEM_BREAK.split(cells[index])
                    if item.strip()
                ]
    return items, tables


def _decided(item: str, known: tuple[Oracle, ...]) -> list[tuple[Oracle, list[str]]]:
    """Return each oracle that decides this item against it, skipping the ones that cannot."""
    return [
        (oracle, names)
        for oracle in known
        if (names := _subject_names(item, oracle.known, oracle.built, oracle.noun))
    ]


def column_findings(
    root: Path = REPOSITORY_ROOT, known: tuple[Oracle, ...] | None = None
) -> tuple[list[Finding], int]:
    """Return a finding per denied column item the tree says exists."""
    known = oracles(root) if known is None else known
    denial = re.compile("|".join(DENIAL_PATTERNS), re.IGNORECASE)
    items, tables = denied_columns(root)
    findings = [
        Finding(
            COLUMN_DENIAL,
            relative,
            f"line {number} lists {', '.join(names)} under a column headed as unbuilt, "
            f"but {oracle.source}: {item.strip()[:120]}",
        )
        for relative, number, item in items
        if not denial.search(item)
        for oracle, names in _decided(item, known)
    ]
    return findings, tables


def _labelled_items(sentence: str, denial: re.Pattern[str]) -> list[str]:
    """Return the items a denying lead-in label governs, empty when there is no such label.

    "Specified but not built (TASK-0243): load ... and 24 h soak ..." is reliability.md:56,
    a column header written as prose: the label denies and no item in the list does.
    """
    head, separator, tail = sentence.partition(": ")
    if not separator or denial.search(head) is None:
        return []
    if len(_normalised(head)) > LABEL_LIMIT or denial.search(tail) is not None:
        return []
    return [piece for clause in CLAUSE_BREAK.split(tail) for piece in ITEM_BREAK.split(clause)]


def prose_findings(
    root: Path = REPOSITORY_ROOT, known: tuple[Oracle, ...] | None = None
) -> tuple[list[Finding], int]:
    """Return a finding per prose denial the tree decides, by subject or by lead-in label."""
    known = oracles(root) if known is None else known
    # CRITICAL: the provider oracle is excluded from the subject rule and only from it.
    # adapter_findings already reads every clause for a provider denial, with an affirmation
    # guard this rule does not have, so including it here reports the same sentence twice and
    # loses that guard on the duplicate.
    subjects = tuple(oracle for oracle in known if oracle.name != PROVIDER)
    tiers = {name: _name_pattern(name) for oracle in subjects for name in sorted(oracle.built)}
    denial = re.compile("|".join(DENIAL_PATTERNS), re.IGNORECASE)
    findings: list[Finding] = []
    mentions = 0

    for path in scanned_documents(root):
        relative = path.relative_to(root).as_posix()
        for line, sentence in _sentences(path.read_text(encoding="utf-8")):
            for item in _labelled_items(sentence, denial):
                findings += [
                    Finding(
                        LABEL_DENIAL,
                        relative,
                        f"line {line} lists {', '.join(names)} after a denial that labels "
                        f"the list, but {oracle.source}: {item.strip()[:120]}",
                    )
                    for oracle, names in _decided(item, known)
                ]
            for clause in CLAUSE_BREAK.split(sentence):
                if any(pattern.search(clause) for pattern in tiers.values()):
                    mentions += 1
                match = denial.search(clause)
                if match is None:
                    continue
                findings += [
                    Finding(
                        LAYER_DENIAL,
                        relative,
                        f"line {line} denies {', '.join(names)}, but {oracle.source}: "
                        f"{clause.strip()[:120]}",
                    )
                    for oracle, names in _decided(clause[: match.start()], subjects)
                ]
    return findings, mentions


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
    known = oracles(root)
    columns, tables = column_findings(root, known)
    prose, layer_mentions = prose_findings(root, known)
    built = {name for oracle in known if oracle.name == TIER for name in oracle.built}
    unbuilt = {
        name for oracle in known if oracle.name == TIER for name in oracle.known - oracle.built
    }

    if not built:
        return [
            Finding(
                VACUOUS,
                PROJECT_FILE.as_posix(),
                "no pytest marker names a tier with test code, so the tier rule decided "
                "nothing; [tool.pytest.ini_options].markers is its whole vocabulary",
            )
        ]
    if not unbuilt:
        return [
            Finding(
                VACUOUS,
                PROJECT_FILE.as_posix(),
                "every registered tier has test code, so every tier denial this gate reads "
                "is automatically false and the rule stopped discriminating",
            )
        ]
    if tables == 0:
        return [
            Finding(
                VACUOUS,
                "docs",
                "parsed no markdown table at all, so the column rule read nothing; a table "
                "inside a blockquote is the one TASK-0371 is about and is easy to lose",
            )
        ]
    if layer_mentions == 0:
        return [
            Finding(
                VACUOUS,
                "docs",
                f"scanned every document and found no clause naming any of the {len(built)} "
                "tiers that have tests; either the documentation stopped naming them or the "
                "marker names stopped matching how prose spells them",
            )
        ]
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
    return adapters + structure + columns + prose


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
