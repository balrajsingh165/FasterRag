"""Fail when a module imports from a layer above its own.

``docs/structure.md`` states the rule in one line: imports point one way, from the entry
points down to the foundation. Nothing checked it. ``ruff`` sorts imports without asking
where they point, ``mypy`` follows them without objecting, and a cycle between two packages
imports fine for as long as the two modules involved happen to be loaded in a workable
order. So the rule held as far as whoever was typing remembered it, and a 2026-11-12 scan
found eighteen module pairs that broke it: ``workers`` importing ``services`` while
``services`` imports ``workers``, ``core`` importing ``adapters`` while ``adapters`` imports
``core``, and the facade and a service importing their builders from the REST layer — which
is why a library user who never starts a server still needs the web framework installed.

The contract, top to bottom. A module may import its own layer and any layer below it::

    cli
    api | facade | dashboard        (siblings: none of the three imports another)
    services
    workers
    adapters
    core
    observability
    config
    errors

``facade`` is the importable surface: ``fasterrag/__init__.py``, ``facade.py``, ``sync.py``
and the five re-export modules beside them. ``dashboard`` is
``observability/dashboard.py``, which is a second web application rather than
instrumentation, so it sits with the entry points instead of dragging the whole
``observability`` package up to their level. ``cli`` is above the other three because the
terminal is what starts the API and the dashboard.

What counts as an import:

* every ``import`` and ``from ... import`` statement wherever it stands — module level,
  inside a function, under ``if TYPE_CHECKING:``, inside a ``try``. The function-level ones
  are the point: that is how ``facade.py`` and ``services/evaluation.py`` reached into
  ``api/`` without a module-level scan ever seeing it, and a type-only import is still a
  statement about which layer knows about which;
* ``from package import name`` where ``name`` is itself a module counts as an import of that
  module, not of the package;
* a string literal that is exactly the dotted path of a module in this package, optionally
  followed by ``:attribute``. The adapter factories and the degradation ladder import through
  such strings, and a registry entry pointing upward is the same defect as an ``import``.

The rules:

* ``upward-import`` (error) — a module imports a layer ranked above its own.
* ``sideways-import`` (error) — a module imports a sibling layer of the same rank.
* ``unplaced-module`` (error) — a module no layer claims. A new top-level package has to be
  placed in ``LAYERS`` before it can be imported, or it would sit outside the contract and
  every import of it would pass.
* ``stale-exception`` (error) — an entry in ``EXCEPTIONS`` that no import matches any more.
  Without it a fixed edge leaves its waiver behind, and the waiver then excuses the edge
  when somebody reintroduces it.
* ``unreadable`` (error) — a file that could not be parsed, so its imports went unread.
* ``vacuous-scan`` (error) — the scan matched no modules, resolved no import inside the
  package, or found a layer with no module in it. Any of the three means the gate proved
  nothing (rebuild-playbook trap T12).

What this gate CANNOT do, said plainly:

* it cannot see an import assembled at run time — ``import_module(prefix + name)`` — only a
  literal;
* it reads ``src/fasterrag``. Tests and scripts import whatever they need to;
* it checks direction, not coupling. A service that reaches into the private parts of a
  worker is importing downward and passes;
* an exception is keyed by the importing and the imported module, so a second import of the
  same module from the same file rides on the first one's waiver.

Exit codes: ``0`` when every violation is a named exception and no exception is stale,
``1`` otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_TREE: Final = "src"
PACKAGE: Final = "fasterrag"

UPWARD: Final = "upward-import"
SIDEWAYS: Final = "sideways-import"
UNPLACED: Final = "unplaced-module"
STALE: Final = "stale-exception"
UNREADABLE: Final = "unreadable"
VACUOUS: Final = "vacuous-scan"

# CRITICAL: the order is the contract. A layer may import the layers listed after it and
# none listed before it; layers sharing a row may not import each other. Moving a name up a
# row to make an import legal changes the architecture, not the gate, so it happens only
# with the document that states the rule (docs/structure.md) changing in the same commit.
RANKS: Final[tuple[tuple[str, ...], ...]] = (
    ("cli",),
    ("api", "facade", "dashboard"),
    ("services",),
    ("workers",),
    ("adapters",),
    ("core",),
    ("observability",),
    ("config",),
    ("errors",),
)

LAYERS: Final[dict[str, tuple[str, ...]]] = {
    "cli": ("fasterrag.cli",),
    "api": ("fasterrag.api",),
    "facade": (
        "fasterrag.facade",
        "fasterrag.sync",
        "fasterrag.chunking",
        "fasterrag.evals",
        "fasterrag.parsing",
        "fasterrag.rerank",
        "fasterrag.retrieval",
    ),
    "dashboard": ("fasterrag.observability.dashboard",),
    "services": ("fasterrag.services",),
    "workers": ("fasterrag.workers",),
    "adapters": ("fasterrag.adapters",),
    "core": ("fasterrag.core",),
    "observability": ("fasterrag.observability",),
    "config": ("fasterrag.config",),
    "errors": ("fasterrag.errors",),
}

ROOT_LAYER: Final = "facade"
"""The layer ``fasterrag/__init__.py`` imports as: it lazily exports the facade.

As a *target* the package root is skipped. Importing any submodule already runs it, so
``from fasterrag import __version__`` adds no dependency the importer did not have.
"""

# CRITICAL: a backlog, not a permission. Every entry is an import that breaks the contract
# today and names the work that removes it; the gate fails on an entry nothing matches any
# more, so an edge and its waiver leave together. Adding an entry to make a NEW import green
# is the one thing this table must never be used for — move the code, or move the interface
# it needs down a layer. Keyed by module pair and never by line number: several people edit
# this tree at once and a line-keyed waiver would excuse a different import within the hour.
EXCEPTIONS: Final[dict[str, str]] = {
    "fasterrag.facade -> fasterrag.cli.sources": (
        "TASK-0456: directory expansion lives in the CLI; it belongs beside services/sources.py"
    ),
    "fasterrag.workers.cpu_pool -> fasterrag.services.journal": (
        "TASK-0456 step 3: the journal a worker writes to is defined by a service"
    ),
    "fasterrag.workers.embed_pool -> fasterrag.services.budget": (
        "TASK-0456 step 3: the cost governor a worker charges is defined by a service"
    ),
    "fasterrag.workers.embed_pool -> fasterrag.services.estimation": (
        "TASK-0456 step 3: a function-level import of the price table"
    ),
    "fasterrag.workers.indexer -> fasterrag.services.tenancy": (
        "TASK-0456 step 3: the tenancy helpers a worker stamps payloads with are a service's"
    ),
    "fasterrag.core.cache.embedding -> fasterrag.adapters.embeddings.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.core.cache.embedding -> fasterrag.adapters.vectordb.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.core.chunking.enrichment -> fasterrag.adapters.llm.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.core.evals.generator -> fasterrag.adapters.llm.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.core.faithfulness -> fasterrag.adapters.llm.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.core.retrieval.bm25 -> fasterrag.adapters.vectordb.base": (
        "TASK-0456 step 4: core names an adapter base interface; the rule is undecided"
    ),
    "fasterrag.config.loader -> fasterrag.observability.logging": (
        "TASK-0456: the loader logs through a layer above it"
    ),
    "fasterrag.errors -> fasterrag.observability.logging": (
        "TASK-0456: the error base class reads the trace id from a layer above it"
    ),
    "fasterrag.observability.langfuse_export -> fasterrag.core.tracing": (
        "TASK-0456: a trace exporter needs the trace model, which lives in core"
    ),
    "fasterrag.observability.otel_export -> fasterrag.core.tracing": (
        "TASK-0456: a trace exporter needs the trace model, which lives in core"
    ),
    "fasterrag.observability.dashboard -> fasterrag.api.auth": (
        "TASK-0456: the dashboard reuses the REST layer's authentication middleware"
    ),
}

_REFERENCE: Final = re.compile(
    r"^" + PACKAGE + r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?::[A-Za-z_][A-Za-z0-9_.]*)?$"
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One defect in the layering, located precisely enough for a human to judge it."""

    rule: str
    path: str
    line: int
    detail: str
    key: str = ""

    def __str__(self) -> str:
        """Return the one-line report form."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


@dataclass(frozen=True, slots=True)
class Module:
    """One source file, under the dotted name the interpreter knows it by."""

    name: str
    path: Path
    is_package: bool

    @property
    def package(self) -> str:
        """Return the package a relative import in this module is resolved against."""
        return self.name if self.is_package else self.name.rpartition(".")[0]


@dataclass(frozen=True, slots=True)
class Reference:
    """One place a module names another module of the package."""

    target: str
    line: int


def _modules(root: Path) -> list[Module]:
    """Return every module of the package under ``root``, excluding bytecode caches."""
    base = root / SOURCE_TREE
    package = base / PACKAGE
    if not package.is_dir():
        return []
    found: list[Module] = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(base).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        found.append(Module(".".join(parts), path, is_package))
    return found


def layer_of(module: str) -> str | None:
    """Return the layer that claims ``module``: the longest matching prefix wins."""
    if module == PACKAGE:
        return ROOT_LAYER
    best: tuple[int, str] | None = None
    for layer, prefixes in LAYERS.items():
        for prefix in prefixes:
            if module != prefix and not module.startswith(prefix + "."):
                continue
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), layer)
    return None if best is None else best[1]


def rank_of(layer: str) -> int:
    """Return how far down the contract ``layer`` sits; zero is the top."""
    for index, row in enumerate(RANKS):
        if layer in row:
            return index
    raise KeyError(layer)


def _resolved(module: Module, node: ast.ImportFrom) -> str | None:
    """Return the absolute module a ``from`` import names, or ``None`` if it leaves the tree."""
    if node.level == 0:
        return node.module
    parts = module.package.split(".")
    if node.level - 1 >= len(parts):
        return None
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _targets(module: Module, node: ast.AST, known: frozenset[str]) -> Iterator[str]:
    """Yield every module one syntax node names, whether by statement or by string."""
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        base = _resolved(module, node)
        if base is None:
            return
        for alias in node.names:
            submodule = f"{base}.{alias.name}"
            yield submodule if submodule in known else base
    elif (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _REFERENCE.match(node.value)
    ):
        named = node.value.partition(":")[0]
        if named in known:
            yield named


def references(module: Module, source: str, known: frozenset[str]) -> list[Reference]:
    """Return every in-package module ``module`` names, at any depth of its syntax tree.

    Raises ``SyntaxError`` for a file that does not parse; the caller reports it.
    """
    found: list[Reference] = []
    for node in ast.walk(ast.parse(source)):
        for target in _targets(module, node, known):
            if target == PACKAGE or not target.startswith(PACKAGE + "."):
                continue
            found.append(Reference(target, getattr(node, "lineno", 0)))
    return found


def _judge(importer: str, imported: str) -> str | None:
    """Return the rule one import breaks, or ``None`` when the contract allows it."""
    above, below = layer_of(importer), layer_of(imported)
    if above is None or below is None or above == below:
        return None
    if rank_of(below) < rank_of(above):
        return UPWARD
    if rank_of(below) == rank_of(above):
        return SIDEWAYS
    return None


def _vacuous(detail: str) -> list[Finding]:
    """Return the single finding that says the scan proved nothing."""
    return [Finding(VACUOUS, f"{SOURCE_TREE}/{PACKAGE}", 0, detail)]


def check(
    root: Path = REPOSITORY_ROOT, exceptions: Mapping[str, str] = EXCEPTIONS
) -> list[Finding]:
    """Return every finding in ``root``: violations, stale exceptions and vacuity."""
    modules = _modules(root)
    if not modules:
        return _vacuous("the scan matched no Python module; it proved nothing")

    known = frozenset(module.name for module in modules)
    findings: list[Finding] = []
    populated: set[str] = set()
    resolved = 0

    for module in modules:
        relative = module.path.relative_to(root).as_posix()
        layer = layer_of(module.name)
        if layer is None:
            findings.append(
                Finding(
                    UNPLACED,
                    relative,
                    0,
                    f"{module.name} belongs to no layer; place its package in LAYERS in "
                    "scripts/check_layering.py before anything imports it",
                )
            )
            continue
        populated.add(layer)
        try:
            found = references(module, module.path.read_text(encoding="utf-8"), known)
        except SyntaxError as error:
            findings.append(
                Finding(
                    UNREADABLE,
                    relative,
                    error.lineno or 0,
                    "the file does not parse, so every import in it went unread and it is "
                    "clean only as far as the gate could see",
                )
            )
            continue
        resolved += len(found)
        for reference in found:
            rule = _judge(module.name, reference.target)
            if rule is None:
                continue
            findings.append(
                Finding(
                    rule,
                    relative,
                    reference.line,
                    f"{module.name} ({layer}) imports {reference.target} "
                    f"({layer_of(reference.target)})",
                    key=f"{module.name} -> {reference.target}",
                )
            )

    if resolved == 0:
        return _vacuous(
            f"parsed {len(modules)} modules and resolved no import inside the package; "
            "either nothing imports anything, or this scan no longer reads imports"
        )
    empty = sorted(set(LAYERS) - populated)
    if empty:
        return _vacuous(
            f"no module belongs to the layer(s) {', '.join(empty)}; a contract about a "
            "layer that is not there checks nothing, so fix LAYERS or remove the layer"
        )

    matched = {finding.key for finding in findings}
    findings.extend(
        Finding(
            STALE,
            "scripts/check_layering.py",
            0,
            f"the exception '{key}' matches no import any more; delete it, so the edge "
            "cannot come back under a waiver written for the old one",
            key=key,
        )
        for key in sorted(set(exceptions) - matched)
    )
    return findings


def unexcused(findings: list[Finding], exceptions: Mapping[str, str] = EXCEPTIONS) -> list[Finding]:
    """Return the findings no named exception covers."""
    return [
        finding
        for finding in findings
        if finding.rule not in (UPWARD, SIDEWAYS) or finding.key not in exceptions
    ]


def _report(write: Callable[[str], int], heading: str, lines: list[str]) -> None:
    """Write one headed block, or nothing when the block is empty."""
    if not lines:
        return
    write(f"{heading}\n")
    for line in lines:
        write(f"  {line}\n")
    write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the gate, separating new violations from the named exceptions."""
    parser = argparse.ArgumentParser(description="Check that imports point down the layers.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root to scan (default: the checkout this script lives in)",
    )
    arguments = parser.parse_args(argv)

    findings = check(arguments.root)
    excess = unexcused(findings)
    known = sorted(
        {
            f"{finding.key}  [{finding.rule}] {EXCEPTIONS[finding.key]}"
            for finding in findings
            if finding not in excess
        }
    )

    _report(sys.stderr.write, "imports the layering does not allow:", [str(f) for f in excess])
    _report(sys.stdout.write, "named exceptions (a backlog, not a permission):", known)

    if excess:
        sys.stderr.write(
            "Imports point one way: cli -> api | facade | dashboard -> services -> workers "
            "-> adapters -> core -> observability -> config -> errors. Move the code that "
            "needs the import up, or move the interface it needs down. Do not add an entry "
            "to EXCEPTIONS in scripts/check_layering.py to make a new import pass.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
