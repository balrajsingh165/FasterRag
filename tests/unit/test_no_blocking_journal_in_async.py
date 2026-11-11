"""Proves no coroutine in ``src/`` reaches a blocking journal method (TASK-0399).

``Journal`` offers every filesystem operation twice: an ``async`` method and the
``*_blocking`` twin it runs in a worker thread. The offload is therefore inside the store
and a caller cannot forget it by accident — but it can still be *overridden* on purpose, by
naming the blocking twin from a coroutine, and nothing about that reads as wrong at the call
site. It would simply put the write back on the event loop, silently, which is the defect
TASK-0358 measured at 633 ms p50 and the reason this file exists.

The check is lexical rather than a call-graph: a sync helper that a coroutine calls is not
flagged here, because deciding that statically needs type resolution this does not have. The
shape it does catch is the one the defect actually took — ``await``-less journal I/O written
straight into an ``async def`` — and the matching ``mypy`` error catches the rest, since
every async method now returns a coroutine a synchronous caller cannot use.
"""

from __future__ import annotations

import ast
from pathlib import Path

from fasterrag.services.journal import Journal

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "fasterrag"

BLOCKING_METHODS = frozenset(
    name
    for name in vars(Journal)
    if name.endswith("_blocking") and callable(getattr(Journal, name))
)


class _AsyncBlockingCalls(ast.NodeVisitor):
    """Collects every ``*_blocking`` call written inside an ``async def`` body."""

    def __init__(self) -> None:
        """Start outside any function."""
        self.depth = 0
        self.stack: list[str] = []
        self.found: list[tuple[int, str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Descend into a synchronous function, which resets the async context."""
        self.stack.append(node.name)
        outer, self.depth = self.depth, 0
        self.generic_visit(node)
        self.depth = outer
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Descend into a coroutine, where a blocking call is the defect."""
        self.stack.append(node.name)
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Record a call to a blocking twin made directly from a coroutine."""
        func = node.func
        if self.depth > 0 and isinstance(func, ast.Attribute) and func.attr in BLOCKING_METHODS:
            self.found.append((node.lineno, "->".join(self.stack), func.attr))
        self.generic_visit(node)


def _offenders(path: Path) -> list[str]:
    """Return one line per blocking journal call a coroutine in ``path`` makes."""
    visitor = _AsyncBlockingCalls()
    visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
    return [
        f"{path.as_posix()}:{lineno} async {where} calls {method}"
        for lineno, where, method in visitor.found
    ]


def test_the_journal_offers_a_blocking_twin_for_every_async_method() -> None:
    """Without the pairing there is nothing for the guard below to be about."""
    asynchronous = {
        name.removeprefix("_")
        for name, value in vars(Journal).items()
        if not name.startswith("_") and getattr(value, "__code__", None) is not None
    }
    paired = {name.removesuffix("_blocking") for name in BLOCKING_METHODS}

    assert paired, "Journal exposes no blocking twins"
    assert paired <= asynchronous
    assert "create_job" in paired
    assert "save_job" in paired


def test_no_coroutine_calls_a_blocking_journal_method() -> None:
    """A coroutine that calls a ``*_blocking`` twin runs file I/O on the event loop."""
    offenders = [
        line for module in sorted(SOURCE_ROOT.rglob("*.py")) for line in _offenders(module)
    ]

    assert not offenders, (
        "these coroutines do journal file I/O on the event loop, which stalls every other "
        "request on the worker (TASK-0399); await the async method instead of its "
        "*_blocking twin:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_catches_the_defect_it_was_written_for(tmp_path: Path) -> None:
    """A guard nothing can fail is a guard nobody can trust."""
    offending = tmp_path / "regression.py"
    offending.write_text(
        "async def accept(journal: object) -> None:\n    journal.create_job_blocking('c', [])\n",
        encoding="utf-8",
    )

    assert _offenders(offending) == [
        f"{offending.as_posix()}:2 async accept calls create_job_blocking"
    ]


def test_a_synchronous_caller_is_left_alone(tmp_path: Path) -> None:
    """The CLI, the benchmark harness and the tests are what the twins exist for."""
    allowed = tmp_path / "allowed.py"
    allowed.write_text(
        "async def outer() -> None:\n"
        "    pass\n"
        "\n"
        "def populate(journal: object) -> None:\n"
        "    journal.save_job_blocking(None)\n",
        encoding="utf-8",
    )

    assert _offenders(allowed) == []
