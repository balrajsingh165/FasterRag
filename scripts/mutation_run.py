"""Mutation-test the modules whose correctness this repository argues about.

Mutation checking has been this project's habit rather than its infrastructure: several
slices (TASK-0193, TASK-0208, TASK-0211, TASK-0229, TASK-0232) record a maintainer breaking
a line by hand and confirming exactly one test went red. A habit protects whatever somebody
remembered to break. This automates it for a named target set, which is the mutation layer
of TASK-0243 — the load and soak layers still need reference hardware and remain open.

**Why a purpose-built runner rather than mutmut or cosmic-ray.** Both were evaluated here on
2026-11-04 against Python 3.14.7 on Windows, which is the machine this repository is
developed on:

* ``mutmut`` 3.8.0 installs and then refuses to run at all — it prints "To run mutmut on
  Windows, please use the WSL" and exits. A nightly survivor nobody can reproduce locally is
  a report, not a gate.
* ``cosmic-ray`` 8.7.0 installs cleanly on 3.14, but it parses with ``parso`` 0.8.7, which
  reports a false ``SyntaxError`` on the parenthesized ``with`` statements in
  ``src/fasterrag/services/archive.py`` — valid since Python 3.10 and accepted by
  ``ast.parse`` here. It also carries one global ``test-command`` per session, so running
  only the tests that cover a module means one session, one SQLite database and one report
  per target, stitched back together by a script of our own.

Both facts point the same way: this repository already has hand-rolled AST tooling
(``scripts/trim_docstrings.py``, with its before/after structure comparison), the mutation
budget is only affordable if each mutant runs *its own module's* tests, and the output has
to be a survivor list rather than a score. ``ast`` parses exactly what the interpreter
parses, so the parso class of defect cannot occur.

**A surviving mutant is a claim, not a verdict.** It says some line can be changed without
any test noticing. That is usually a missing test and sometimes an equivalent mutant, so
this prints every survivor with its file, line, operator and the exact text substitution,
and never prints a bare score. Exit codes: ``0`` every mutant killed, ``1`` survivors
found, ``2`` the run could not be performed. ``2`` is the vacuity case this repository
treats as a bug — a baseline suite that was already red kills every mutant for free, so it
is never allowed to look like a pass.

**Nothing under ``src/`` is ever written.** The tree is copied once per run and the copy is
mutated; pytest is pointed at it with ``-o pythonpath=`` (which empties the ini setting that
would otherwise put the real ``src`` first) plus ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ARITHMETIC_SWAPS",
    "BOOLEAN_SWAPS",
    "COMPARISON_SWAPS",
    "DEFAULT_TARGETS",
    "EXIT_CODES",
    "Mutation",
    "RunReport",
    "Target",
    "TargetResult",
    "Verdict",
    "apply",
    "decide",
    "generate",
    "load_targets",
    "main",
    "markdown",
    "render",
    "restrict",
    "run",
    "select",
    "source_files",
]

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

DEFAULT_TIMEOUT_S: Final = 180.0
DEFAULT_SEED: Final = 0


class Verdict(StrEnum):
    """What a mutation run concluded."""

    CLEAN = "clean"
    SURVIVORS = "survivors"
    NOT_RUN = "not-run"


EXIT_CODES: Final[dict[Verdict, int]] = {
    Verdict.CLEAN: 0,
    Verdict.SURVIVORS: 1,
    Verdict.NOT_RUN: 2,
}


@dataclass(frozen=True)
class Target:
    """One mutated source path and the tests that are supposed to defend it."""

    name: str
    source: str
    tests: tuple[str, ...]


# CRITICAL: this set is deliberately small. Mutating all 27k lines produces a run that never
# finishes and a report nobody reads; these are the modules whose behaviour the docs argue
# about — chunk boundaries, retrieval scoring and fusion, the index identity digest, the
# breaker's state machine, the eval regression gate and the rate limiter. Widen it with
# --targets-file rather than by quietly adding paths nobody budgeted for.
DEFAULT_TARGETS: Final[tuple[Target, ...]] = (
    Target(
        name="retrieval",
        source="src/fasterrag/core/retrieval",
        tests=("tests/unit/core/retrieval",),
    ),
    Target(
        name="chunking",
        source="src/fasterrag/core/chunking",
        tests=("tests/unit/core/chunking",),
    ),
    Target(
        name="identity",
        source="src/fasterrag/core/identity.py",
        tests=(
            "tests/unit/core/test_identity.py",
            "tests/unit/core/test_identity_properties.py",
            "tests/unit/services/test_identity_drift.py",
        ),
    ),
    Target(
        name="breaker",
        source="src/fasterrag/core/breaker.py",
        tests=("tests/unit/core/test_breaker.py",),
    ),
    Target(
        name="regression",
        source="src/fasterrag/services/regression.py",
        tests=(
            "tests/unit/services/test_regression.py",
            "tests/unit/services/test_comparability.py",
        ),
    ),
    Target(
        name="rate-limit",
        source="src/fasterrag/core/rate_limit.py",
        tests=("tests/unit/core/test_rate_limit.py",),
    ),
)

COMPARISON_SWAPS: Final[dict[type[ast.cmpop], tuple[str, str]]] = {
    ast.Lt: ("<", "<="),
    ast.LtE: ("<=", "<"),
    ast.Gt: (">", ">="),
    ast.GtE: (">=", ">"),
    ast.Eq: ("==", "!="),
    ast.NotEq: ("!=", "=="),
    ast.Is: ("is", "is not"),
    ast.IsNot: ("is not", "is"),
    ast.In: ("in", "not in"),
    ast.NotIn: ("not in", "in"),
}

ARITHMETIC_SWAPS: Final[dict[type[ast.operator], tuple[str, str]]] = {
    ast.Add: ("+", "-"),
    ast.Sub: ("-", "+"),
    ast.Mult: ("*", "/"),
    ast.Div: ("/", "*"),
    ast.FloorDiv: ("//", "/"),
    ast.Mod: ("%", "//"),
}

BOOLEAN_SWAPS: Final[dict[type[ast.boolop], tuple[str, str]]] = {
    ast.And: ("and", "or"),
    ast.Or: ("or", "and"),
}

Candidate = tuple[int, int, str, str, str]
"""One substitution: start offset, end offset, operator name, text before, text after."""


@dataclass(frozen=True)
class Mutation:
    """One text substitution, located precisely enough for a reader to act on it."""

    path: str
    line: int
    column: int
    operator: str
    before: str
    after: str
    start: int
    end: int

    def describe(self) -> str:
        """Render the mutation the way a survivor report has to read."""
        return (
            f"{self.path}:{self.line}:{self.column} [{self.operator}] "
            f"{self.before!r} -> {self.after!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "operator": self.operator,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class TargetResult:
    """What one target's mutants did."""

    name: str
    generated: int = 0
    executed: int = 0
    killed: int = 0
    survived: int = 0
    timed_out: int = 0
    duration_s: float = 0.0
    survivors: list[Mutation] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "name": self.name,
            "generated": self.generated,
            "executed": self.executed,
            "killed": self.killed,
            "survived": self.survived,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 1),
            "survivors": [mutation.as_dict() for mutation in self.survivors],
            "problems": list(self.problems),
        }


@dataclass
class RunReport:
    """Every target's result plus the verdict they add up to."""

    verdict: Verdict
    results: list[TargetResult]

    @property
    def executed(self) -> int:
        """Total mutants actually run."""
        return sum(result.executed for result in self.results)

    @property
    def survived(self) -> int:
        """Total mutants no test noticed."""
        return sum(result.survived for result in self.results)

    @property
    def killed(self) -> int:
        """Total mutants some test caught."""
        return sum(result.killed for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the JSON artifact."""
        return {
            "verdict": self.verdict.value,
            "executed": self.executed,
            "killed": self.killed,
            "survived": self.survived,
            "targets": [result.as_dict() for result in self.results],
        }


def _line_offsets(source: str) -> list[int]:
    """Character offset at which each 1-based line starts."""
    offsets = [0, 0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _offset(offsets: list[int], line: int, column: int) -> int:
    """Translate an ``ast`` (line, column) position into a character offset."""
    return offsets[line] + column


def _token_span(source: str, start: int, end: int, token: str) -> tuple[int, int] | None:
    """Locate ``token`` in the gap between two child nodes, tolerating odd whitespace."""
    pattern = r"\s+".join(re.escape(part) for part in token.split())
    found = re.search(pattern, source[start:end])
    if found is None:
        return None
    return start + found.start(), start + found.end()


def _position(offsets: list[int], offset: int) -> tuple[int, int]:
    """Translate a character offset back into a 1-based line and 0-based column."""
    line = 1
    while line + 1 < len(offsets) and offsets[line + 1] <= offset:
        line += 1
    return line, offset - offsets[line]


def _number_replacement(value: int | float) -> str:
    """Perturb a numeric literal so a boundary test has something to catch."""
    if isinstance(value, bool):
        return "False" if value else "True"
    if value == 0:
        return "1"
    if value == 1:
        return "0"
    return repr(value + 1)


def _gap(offsets: list[int], before: ast.expr, after: ast.expr) -> tuple[int, int] | None:
    """Offsets of the text between two sibling expressions, where their operator sits."""
    if before.end_lineno is None or before.end_col_offset is None:
        return None
    return (
        _offset(offsets, before.end_lineno, before.end_col_offset),
        _offset(offsets, after.lineno, after.col_offset),
    )


def _infix(
    source: str, offsets: list[int], left: ast.expr, right: ast.expr, swap: tuple[str, str]
) -> tuple[int, int] | None:
    """Locate an infix operator's own characters between the operands it joins."""
    gap = _gap(offsets, left, right)
    if gap is None:
        return None
    return _token_span(source, gap[0], gap[1], swap[0])


def _candidates(source: str, tree: ast.Module, offsets: list[int]) -> list[Candidate]:
    """Collect every (start, end, operator, before, after) substitution the catalogue allows."""
    found: list[Candidate] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left: ast.expr = node.left
            for operation, comparator in zip(node.ops, node.comparators, strict=True):
                swap = COMPARISON_SWAPS.get(type(operation))
                span = None if swap is None else _infix(source, offsets, left, comparator, swap)
                if swap is not None and span is not None:
                    found.append((span[0], span[1], "comparison", swap[0], swap[1]))
                left = comparator
        elif isinstance(node, ast.BinOp):
            swap = ARITHMETIC_SWAPS.get(type(node.op))
            span = None if swap is None else _infix(source, offsets, node.left, node.right, swap)
            if swap is not None and span is not None:
                found.append((span[0], span[1], "arithmetic", swap[0], swap[1]))
        elif isinstance(node, ast.BoolOp):
            swap = BOOLEAN_SWAPS[type(node.op)]
            for first, second in zip(node.values, node.values[1:], strict=False):
                span = _infix(source, offsets, first, second, swap)
                if span is not None:
                    found.append((span[0], span[1], "boolean-op", swap[0], swap[1]))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            start = _offset(offsets, node.lineno, node.col_offset)
            end = _offset(offsets, node.operand.lineno, node.operand.col_offset)
            if source[start:end].strip() == "not":
                found.append((start, end, "unary-not", source[start:end], ""))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            if node.end_lineno is None or node.end_col_offset is None:
                continue
            start = _offset(offsets, node.lineno, node.col_offset)
            end = _offset(offsets, node.end_lineno, node.end_col_offset)
            replacement = _number_replacement(node.value)
            kind = "boolean-constant" if isinstance(node.value, bool) else "number"
            found.append((start, end, kind, source[start:end], replacement))
    return found


def generate(source: str, path: str) -> list[Mutation]:
    """Generate every viable mutation of one source file, guards included.

    Each candidate is spliced and kept only if the result still parses, differs
    structurally from the original, and occupies the same number of lines.
    """
    tree = ast.parse(source)
    original = ast.dump(tree)
    offsets = _line_offsets(source)
    line_count = source.count("\n")

    mutations: list[Mutation] = []
    for start, end, operator, before, after in _candidates(source, tree, offsets):
        if before == after:
            continue
        mutated = source[:start] + after + source[end:]
        if mutated.count("\n") != line_count:
            continue
        try:
            candidate = ast.parse(mutated)
        except SyntaxError:
            continue
        if ast.dump(candidate) == original:
            continue
        line, column = _position(offsets, start)
        mutations.append(
            Mutation(
                path=path,
                line=line,
                column=column,
                operator=operator,
                before=before,
                after=after,
                start=start,
                end=end,
            )
        )
    mutations.sort(key=lambda item: (item.line, item.column, item.operator, item.after))
    return mutations


def apply(source: str, mutation: Mutation) -> str:
    """Splice one mutation into the source it was generated from."""
    return source[: mutation.start] + mutation.after + source[mutation.end :]


def select(mutations: list[Mutation], budget: int | None, seed: int) -> list[Mutation]:
    """Take a reproducible sample when the full set exceeds the budget.

    Sampling is seeded and the result re-sorted, so a bounded nightly run is as
    repeatable as a full one and two runs of the same seed report the same mutants.
    """
    if budget is None or budget <= 0 or len(mutations) <= budget:
        return list(mutations)
    chosen = random.Random(seed).sample(range(len(mutations)), budget)
    return [mutations[index] for index in sorted(chosen)]


def source_files(root: Path, source: str) -> list[Path]:
    """Every Python file a target names, whether it points at a module or a package."""
    path = root / source
    if path.is_dir():
        return sorted(path.rglob("*.py"))
    return [path] if path.is_file() else []


def load_targets(document: Any) -> list[Target]:
    """Read a --targets-file document, refusing a shape that cannot be run."""
    if not isinstance(document, list):
        raise ValueError("a targets file must hold a list of targets")
    targets: list[Target] = []
    for entry in document:
        if not isinstance(entry, dict):
            raise ValueError("each target must be an object with name, source and tests")
        name = entry.get("name")
        source = entry.get("source")
        tests = entry.get("tests")
        if not isinstance(name, str) or not isinstance(source, str):
            raise ValueError("each target needs a string name and a string source path")
        if not isinstance(tests, list) or not tests or not all(isinstance(t, str) for t in tests):
            raise ValueError(f"target {name!r} needs a non-empty list of test paths")
        targets.append(Target(name=name, source=source, tests=tuple(tests)))
    return targets


def restrict(targets: tuple[Target, ...] | list[Target], only: list[str]) -> list[Target]:
    """Keep the named targets, refusing a name that does not exist rather than running nothing."""
    if not only:
        return list(targets)
    known = {target.name: target for target in targets}
    missing = sorted(set(only) - set(known))
    if missing:
        raise ValueError(f"unknown target(s): {', '.join(missing)}")
    return [known[name] for name in only]


def decide(results: list[TargetResult]) -> Verdict:
    """Decide the run's verdict; a run that could not be performed is never clean."""
    if not results or any(result.problems for result in results):
        return Verdict.NOT_RUN
    if not any(result.executed for result in results):
        return Verdict.NOT_RUN
    if any(result.survivors for result in results):
        return Verdict.SURVIVORS
    return Verdict.CLEAN


def render(report: RunReport) -> str:
    """Render the human report: counts per target, then every survivor in full."""
    lines = [f"mutation run: {report.verdict.value}"]
    for result in report.results:
        lines.append(
            f"  {result.name}: {result.executed}/{result.generated} run, "
            f"{result.killed} killed, {result.survived} survived, "
            f"{result.timed_out} timed out, {result.duration_s:.0f}s"
        )
        for problem in result.problems:
            lines.append(f"    [not-run] {problem}")
    survivors = [(result.name, m) for result in report.results for m in result.survivors]
    if survivors:
        lines.append("")
        lines.append("surviving mutants (each one is a line no test defends):")
        lines.extend(f"  [{name}] {mutation.describe()}" for name, mutation in survivors)
    return "\n".join(lines)


def markdown(report: RunReport) -> str:
    """Render the same report for the job summary, where a reader cannot avoid it."""
    lines = [
        "## Mutation testing",
        "",
        f"**Verdict: `{report.verdict.value}`** — "
        f"{report.killed} killed, {report.survived} survived of {report.executed} run.",
        "",
        "| Target | Run | Killed | Survived | Timed out | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {result.name} | {result.executed} | {result.killed} | {result.survived} "
        f"| {result.timed_out} | {result.duration_s:.0f} |"
        for result in report.results
    )
    problems = [p for result in report.results for p in result.problems]
    if problems:
        lines.extend(["", "**Could not run:**", ""])
        lines.extend(f"- {problem}" for problem in problems)
    survivors = [(result.name, m) for result in report.results for m in result.survivors]
    if survivors:
        lines.extend(["", "**Surviving mutants** — each is a line no test defends:", ""])
        lines.extend(f"- `{name}` {mutation.describe()}" for name, mutation in survivors)
    return "\n".join(lines) + "\n"


def _pytest(tests: tuple[str, ...], workspace: Path, timeout: float) -> int | None:
    """Run one target's tests against the mutated copy; ``None`` means it timed out.

    ``-o pythonpath=`` empties the ini setting that would otherwise put the real ``src``
    ahead of the copy, which is what keeps this run off the working tree entirely.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workspace)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-x",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "pythonpath=",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return completed.returncode


def _run_target(
    target: Target, workspace: Path, *, budget: int | None, seed: int, timeout: float
) -> TargetResult:
    """Mutate one target in the workspace copy and record what its tests caught.

    Mutations are generated from the copy, so a mutation's offsets always belong to the
    text it is spliced into.
    """
    result = TargetResult(name=target.name)
    started = time.monotonic()

    files = source_files(REPOSITORY_ROOT, target.source)
    if not files:
        result.problems.append(f"{target.name}: {target.source} matched no Python file")
        return result

    baseline = _pytest(target.tests, workspace, timeout)
    if baseline != 0:
        result.problems.append(
            f"{target.name}: the unmutated suite exited {baseline}, so every mutant would "
            "have died for free"
        )
        return result

    mutations: list[Mutation] = []
    for file in files:
        relative = file.relative_to(REPOSITORY_ROOT)
        copy = workspace / relative.relative_to("src")
        mutations.extend(generate(copy.read_text(encoding="utf-8"), relative.as_posix()))
    result.generated = len(mutations)
    if not mutations:
        result.problems.append(f"{target.name}: no mutant could be generated from {target.source}")
        return result

    for mutation in select(mutations, budget, seed):
        copied = workspace / Path(mutation.path).relative_to("src")
        original = copied.read_text(encoding="utf-8")
        copied.write_text(apply(original, mutation), encoding="utf-8")
        try:
            code = _pytest(target.tests, workspace, timeout)
        finally:
            copied.write_text(original, encoding="utf-8")
        result.executed += 1
        if code is None:
            result.timed_out += 1
            result.killed += 1
        elif code == 0:
            result.survived += 1
            result.survivors.append(mutation)
        else:
            result.killed += 1

    result.duration_s = time.monotonic() - started
    return result


def run(
    targets: list[Target], *, budget: int | None, seed: int, timeout: float, workspace: Path
) -> RunReport:
    """Copy ``src/`` into the workspace, then run every target against the copy."""
    shutil.copytree(REPOSITORY_ROOT / "src", workspace, dirs_exist_ok=True)
    results = [
        _run_target(target, workspace, budget=budget, seed=seed, timeout=timeout)
        for target in targets
    ]
    return RunReport(decide(results), results)


def _write(path: Path, body: str, *, append: bool) -> None:
    """Write a report file, reporting a write failure without altering the verdict."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(body)
    except OSError as exc:
        sys.stderr.write(f"could not write {path}: {exc}\n")


def _parser() -> argparse.ArgumentParser:
    """Build the command line."""
    parser = argparse.ArgumentParser(
        description="Mutation-test the targeted modules and report every surviving mutant."
    )
    parser.add_argument(
        "--only", action="append", default=[], help="run one named target (repeatable)"
    )
    parser.add_argument(
        "--targets-file", type=Path, default=None, help="JSON target set replacing the default"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="maximum mutants per target; a seeded sample is taken above it",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds allowed per mutant"
    )
    parser.add_argument("--list", action="store_true", help="count the mutants and run none")
    parser.add_argument("--json-out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument(
        "--summary", type=Path, default=None, help="append a markdown summary to this file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mutation layer and exit with the verdict's code."""
    args = _parser().parse_args(argv)

    try:
        configured: list[Target] = (
            load_targets(json.loads(args.targets_file.read_text(encoding="utf-8")))
            if args.targets_file is not None
            else list(DEFAULT_TARGETS)
        )
        targets = restrict(configured, args.only)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"target set unusable: {exc}\n")
        return EXIT_CODES[Verdict.NOT_RUN]

    if args.list:
        for target in targets:
            total = sum(
                len(generate(file.read_text(encoding="utf-8"), file.as_posix()))
                for file in source_files(REPOSITORY_ROOT, target.source)
            )
            sys.stdout.write(f"{target.name}: {total} mutants from {target.source}\n")
        return EXIT_CODES[Verdict.CLEAN]

    with tempfile.TemporaryDirectory(prefix="fasterrag-mutation-") as directory:
        report = run(
            targets,
            budget=args.budget,
            seed=args.seed,
            timeout=args.timeout,
            workspace=Path(directory),
        )

    sys.stdout.write(f"{render(report)}\n")
    if args.json_out is not None:
        payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
        _write(args.json_out, f"{payload}\n", append=False)
    if args.summary is not None:
        _write(args.summary, markdown(report), append=True)
    return EXIT_CODES[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
