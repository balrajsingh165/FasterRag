"""Fail when the error-taxonomy rules CLAUDE.md calls mandatory are broken.

``CLAUDE.md`` states two absolutes: *"No bare ``except``, no silently swallowed exceptions,
anywhere, ever. Every caught exception is either handled with a logged correlation/trace id
or rethrown"*, and *"Any ``# type: ignore`` requires an adjacent ``# CRITICAL:``
justification"*. Until this gate, nothing checked either one. ``ruff`` catches a bare
``except:`` (E722) and stops there, which leaves the commoner and more dangerous shape
untouched: a handler whose whole body is ``pass``, ``continue``, or ``return None``. In a
RAG pipeline that shape does not produce an error, it produces a confidently wrong answer
from a silently shortened context — the worst outcome this system has.

The gate reads the tree with CPython's own ``ast``, so it sees structure rather than text,
and separates what it is certain about from what a human must judge:

* ``swallow`` (error) — a handler whose entire body is ``pass``/``continue``/``break``/
  ``return``/``return None``/``...``. Nothing is logged, nothing is rethrown, and the
  failure leaves no trace anywhere. Three exemptions, each structural and each measured
  against this tree: ``StopIteration``/``StopAsyncIteration`` (the iterator protocol's own
  termination signal, not a failure), a ``pass`` arm on a ``try``/``except``/``else`` (the
  handler is the false branch of a probe, and the ``else`` carries the real work), and the
  ``swallow-documented`` split below.
* ``swallow-documented`` (warning) — the same trivial ``return None``, but inside a
  function whose return annotation already admits ``None``. The absence is part of the
  published contract, so the caller cannot mistake it for success. It stays a warning
  rather than disappearing, because the annotation cannot distinguish "absent" from
  "corrupt" and some of these sites conflate the two.
* ``swallow-fallback`` (error) — a handler whose entire body is ``return <expr>`` for a
  substantive ``expr``, catching ``Exception``/``BaseException``, where ``expr`` does not
  mention the caught exception. Nothing is logged, nothing is rethrown, and the caller gets
  a value it cannot tell from a real result: the ``swallow`` shape with a fallback in place
  of ``pass``, and the commonest form of it. The scope is measured rather than assumed. On
  2026-11-08 ``src/`` and ``scripts/`` held 34 handlers whose body is a lone fallback
  ``return``; 20 of those return a value carrying no trace of the exception, and hand
  classification of those 20 gave 7 true positives against 13 false positives — 65%, far
  too noisy to block. The false positives are predicates and probes where the exception *is*
  the answer (``except ValueError: return False`` in ``_is_ip_address``), or values that
  report the failure themselves (``HealthStatus(healthy=False, detail=exc.detail)``,
  ``"unknown (git unavailable)"``). Narrowing to a *broad* catch left exactly one site,
  ``ModelTokenCounter.count``, which was a true positive and is fixed: 0 false positives of
  1, so this rule blocks. The six remaining true positives all catch a named type and are
  filed in TASK-0333 rather than swept into the ratchet.
* ``broad-except`` (warning) — ``except Exception``/``except BaseException`` that does not
  rethrow. A broad catch that immediately translates the vendor's error into a typed one is
  the adapter boundary working exactly as ``docs/reliability.md`` specifies, so only the
  non-translating ones are reported. A handler already reported as ``swallow-fallback`` is
  not reported here as well, so a line judged blocking never also sits in a warning bucket.
  This bucket's "7 of 7 false positives" reading from 2026-11-07 was itself wrong:
  ``ModelTokenCounter.count`` was one of those seven and was a real swallow, which is why
  ``swallow-fallback`` exists (TASK-0333). Six of seven, not seven.
* ``untyped-raise`` (warning) — a builtin exception raised out of a *public* function in
  ``services/`` or ``adapters/`` with no enclosing ``try`` to translate it. That is an
  untyped error crossing a boundary ``docs/reliability.md`` says only typed errors cross.
* ``unjustified-ignore`` (error) — ``# type: ignore`` with no ``# CRITICAL:`` on the same
  line or in the comment block directly above it.

What it CANNOT do, said plainly because a gate that oversells its coverage is the same
defect in miniature:

* it cannot tell a deliberate swallow from a careless one. ``swallow`` fires on the shape,
  and the remedy for a deliberate one is a one-line log, not an exemption;
* ``untyped-raise`` resolves names lexically. A factory call (``raise _translate(exc)``) or
  a locally defined subclass is invisible to it, by design — it only reports a *builtin*
  exception, where it can be certain the type is outside the hierarchy;
* it cannot see runtime behaviour at all. A handler that logs through something this file
  does not recognise as a logger is still a handler whose body is not trivial, so it is
  never reported; a handler that logs to ``/dev/null`` is never reported either;
* ``swallow-fallback`` reads the returned expression lexically. ``return _recover(exc)`` is
  exempt because it mentions the exception, whether or not ``_recover`` reports anything,
  and a fallback assembled over two statements is not a lone ``return`` at all;
* every handler rule reads ``src/fasterrag`` and nothing else. ``scripts/`` is parsed only
  for ``# type: ignore`` comments, so three of the seven true positives the 2026-11-08
  fallback measurement found live in gate scripts this gate cannot see (TASK-0333).

A vacuity guard is mandatory (rebuild-playbook trap T12): a run that matches no files,
parses no handlers, or never puts a lone fallback ``return`` in front of the
``swallow-fallback`` matcher, fails rather than reporting a clean tree. A blocking rule
whose matcher never runs is the vacuous pass it exists to prevent.

Exit codes: ``0`` when no error-severity finding exceeds the ratchet, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import sys
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

PACKAGE_PARTS: Final = ("src", "fasterrag")
IGNORE_SCAN_ROOTS: Final[tuple[str, ...]] = ("src", "scripts")
BOUNDARY_PARTS: Final[tuple[str, ...]] = ("services", "adapters")

SWALLOW: Final = "swallow"
SWALLOW_DOCUMENTED: Final = "swallow-documented"
SWALLOW_FALLBACK: Final = "swallow-fallback"
BROAD_EXCEPT: Final = "broad-except"
UNTYPED_RAISE: Final = "untyped-raise"
UNJUSTIFIED_IGNORE: Final = "unjustified-ignore"
VACUOUS: Final = "vacuous-scan"

ERROR_RULES: Final[frozenset[str]] = frozenset(
    {SWALLOW, SWALLOW_FALLBACK, UNJUSTIFIED_IGNORE, VACUOUS}
)

BROAD_EXCEPTIONS: Final[frozenset[str]] = frozenset({"Exception", "BaseException"})

FALLBACK_EXCERPT: Final = 60

CONTROL_FLOW_SIGNALS: Final[frozenset[str]] = frozenset({"StopIteration", "StopAsyncIteration"})

RAISE_ALLOWED_BUILTINS: Final[frozenset[str]] = frozenset({"NotImplementedError"})

BUILTIN_EXCEPTIONS: Final[frozenset[str]] = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
)

# CRITICAL: empty since 2026-11-08, and an empty ratchet is the only state that makes this
# gate mean what it says. The six entries it shipped with on 2026-11-07 were violations the
# gate's author was not authorised to edit; TASK-0317 fixed all six in `src/` and `scripts/`
# rather than annotating them. An entry here means "this many are known", so the gate still
# fails the moment a file grows one more. Never raise a number to make a new violation green
# — fix the violation, and lower the number when one goes.
RATCHET: Final[dict[str, int]] = {}


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule violation, located precisely enough for a human to judge it."""

    rule: str
    path: str
    line: int
    detail: str

    @property
    def key(self) -> str:
        """Return the ratchet key this finding counts against."""
        return f"{self.rule}:{self.path}"

    def __str__(self) -> str:
        """Return the one-line report form."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


def _sources(root: Path, *parts: str) -> list[Path]:
    """Return the Python modules under ``root/parts``, excluding bytecode caches."""
    base = root.joinpath(*parts)
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _handler_types(handler: ast.ExceptHandler) -> list[str]:
    """Return the exception expressions this handler catches, as source text."""
    if handler.type is None:
        return []
    if isinstance(handler.type, ast.Tuple):
        return [ast.unparse(element) for element in handler.type.elts]
    return [ast.unparse(handler.type)]


def _body_without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return ``body`` with a leading string expression removed."""
    return [
        statement
        for statement in body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def _trivial_body(body: list[ast.stmt]) -> str | None:
    """Return the name of the no-op this handler body is, or ``None`` if it does work."""
    statements = _body_without_docstring(body)
    if len(statements) != 1:
        return None

    only = statements[0]
    if isinstance(only, ast.Pass):
        return "pass"
    if isinstance(only, ast.Continue):
        return "continue"
    if isinstance(only, ast.Break):
        return "break"
    if isinstance(only, ast.Return) and (
        only.value is None or (isinstance(only.value, ast.Constant) and only.value.value is None)
    ):
        return "return None"
    if (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Constant)
        and only.value.value is Ellipsis
    ):
        return "..."
    return None


def _fallback_return(body: list[ast.stmt]) -> ast.Return | None:
    """Return the lone ``return <expr>`` this handler body is, for a substantive ``expr``."""
    statements = _body_without_docstring(body)
    if len(statements) != 1:
        return None
    only = statements[0]
    if not isinstance(only, ast.Return) or only.value is None:
        return None
    if isinstance(only.value, ast.Constant) and only.value.value is None:
        return None
    return only


def _mentions(node: ast.AST, name: str | None) -> bool:
    """Return whether ``name`` is read anywhere inside ``node``."""
    if name is None:
        return False
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _excerpt(node: ast.AST) -> str:
    """Return the source of ``node``, shortened to stay inside one report line."""
    text = " ".join(ast.unparse(node).split())
    return text if len(text) <= FALLBACK_EXCERPT else f"{text[:FALLBACK_EXCERPT]}..."


def _admits_none(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    """Return whether the function's return annotation already publishes ``None``."""
    if function is None or function.returns is None:
        return False
    annotation = ast.unparse(function.returns).strip()
    return annotation != "None" and "None" in annotation


def _noreturn_names(tree: ast.Module) -> frozenset[str]:
    """Return the names of functions in this module annotated ``NoReturn``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.returns is not None
            and ast.unparse(node.returns).split(".")[-1] == "NoReturn"
        ):
            names.add(node.name)
    return frozenset(names)


def _probe_handlers(tree: ast.Module) -> frozenset[tuple[int, int]]:
    """Return the positions of handlers on a ``try``/``except``/``else``: branches, not catches."""
    probes: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.orelse:
            probes.update((handler.lineno, handler.col_offset) for handler in node.handlers)
    return frozenset(probes)


def _escapes(body: list[ast.stmt], noreturn: frozenset[str]) -> bool:
    """Return whether this handler body rethrows or hands off to a ``NoReturn`` callee."""
    module = ast.Module(body=list(body), type_ignores=[])
    for node in ast.walk(module):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name is None and isinstance(func, ast.Attribute):
                name = func.attr
            if name is not None and name in noreturn:
                return True
    return False


def _raised_name(node: ast.Raise) -> str | None:
    """Return the name of the class a ``raise`` instantiates, if it names one directly."""
    if node.exc is None:
        return None
    if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
        return node.exc.func.id
    if isinstance(node.exc, ast.Name):
        return node.exc.id
    return None


def _is_public(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    """Return whether a raise in this function can reach a caller outside the module."""
    if function is None:
        return True
    return not function.name.startswith("_") or function.name.startswith("__")


@dataclass(frozen=True, slots=True)
class _Scope:
    """The lexical facts a node needs from its ancestors."""

    function: ast.FunctionDef | ast.AsyncFunctionDef | None
    try_depth: int


class _ModuleScan:
    """Collects every finding in one parsed module."""

    def __init__(self, relative: str, tree: ast.Module) -> None:
        """Prepare a scan of one module."""
        self.relative = relative
        self.noreturn = _noreturn_names(tree)
        self.probes = _probe_handlers(tree)
        self.at_boundary = any(part in relative.split("/") for part in BOUNDARY_PARTS)
        self.findings: list[Finding] = []
        self.handlers = 0
        self.fallbacks = 0
        self._walk(tree, _Scope(function=None, try_depth=0))

    def _walk(self, node: ast.AST, scope: _Scope) -> None:
        """Visit ``node`` then its children, carrying the lexical scope down."""
        inner = scope
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            inner = _Scope(function=node, try_depth=0)
        elif isinstance(node, ast.Try):
            inner = _Scope(function=scope.function, try_depth=scope.try_depth + 1)

        if isinstance(node, ast.ExceptHandler):
            self.handlers += 1
            self._inspect_handler(node, scope)
        elif isinstance(node, ast.Raise):
            self._inspect_raise(node, scope)

        for child in ast.iter_child_nodes(node):
            self._walk(child, inner)

    def _inspect_handler(self, handler: ast.ExceptHandler, scope: _Scope) -> None:
        """Apply the swallow and broad-catch rules to one handler."""
        caught = _handler_types(handler)
        shown = " | ".join(caught) if caught else "<bare>"

        trivial = _trivial_body(handler.body)
        signal = any(name.split(".")[-1] in CONTROL_FLOW_SIGNALS for name in caught)
        probe = trivial == "pass" and (handler.lineno, handler.col_offset) in self.probes
        if trivial is not None and not signal and not probe:
            rule = (
                SWALLOW_DOCUMENTED
                if trivial == "return None" and _admits_none(scope.function)
                else SWALLOW
            )
            reason = (
                "the absence is in the return annotation, but it cannot tell a missing "
                "value from a corrupt one"
                if rule is SWALLOW_DOCUMENTED
                else "nothing is logged and nothing is rethrown, so the failure leaves no trace"
            )
            self.findings.append(
                Finding(
                    rule,
                    self.relative,
                    handler.lineno,
                    f"'except {shown}' does only '{trivial}': {reason}",
                )
            )

        broad = any(name in BROAD_EXCEPTIONS for name in caught)
        if self._inspect_fallback(handler, shown, broad=broad, signal=signal):
            return

        if broad and not _escapes(handler.body, self.noreturn):
            self.findings.append(
                Finding(
                    BROAD_EXCEPT,
                    self.relative,
                    handler.lineno,
                    f"'except {shown}' catches everything and rethrows nothing; name the "
                    "taxonomy types it is meant to absorb, or translate and raise",
                )
            )

    def _inspect_fallback(
        self,
        handler: ast.ExceptHandler,
        shown: str,
        *,
        broad: bool,
        signal: bool,
    ) -> bool:
        """Report a lone fallback ``return``, counting every candidate the matcher saw."""
        fallback = _fallback_return(handler.body)
        if fallback is None or signal:
            return False

        self.fallbacks += 1
        if not broad or _mentions(fallback, handler.name):
            return False

        self.findings.append(
            Finding(
                SWALLOW_FALLBACK,
                self.relative,
                handler.lineno,
                f"'except {shown}' does only '{_excerpt(fallback)}': the caller cannot tell "
                "that fallback from a real result, and the failure leaves no trace",
            )
        )
        return True

    def _inspect_raise(self, node: ast.Raise, scope: _Scope) -> None:
        """Apply the untyped-raise rule to one ``raise`` at a service or adapter boundary."""
        if not self.at_boundary or scope.try_depth > 0 or not _is_public(scope.function):
            return
        name = _raised_name(node)
        if name is None or name not in BUILTIN_EXCEPTIONS or name in RAISE_ALLOWED_BUILTINS:
            return
        where = scope.function.name if scope.function is not None else "<module>"
        self.findings.append(
            Finding(
                UNTYPED_RAISE,
                self.relative,
                node.lineno,
                f"'raise {name}' escapes {where}(); only the typed hierarchy in "
                "docs/reliability.md crosses a service or adapter boundary",
            )
        )


def _comments(source: str) -> dict[int, tuple[str, bool]]:
    """Return each real comment by line as (text, stands alone), so literals never match."""
    found: dict[int, tuple[str, bool]] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                found[token.start[0]] = (token.string, token.line.strip().startswith("#"))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return found
    return found


def _unjustified_ignores(path: Path, relative: str) -> list[Finding]:
    """Return one finding per ``# type: ignore`` comment with no ``# CRITICAL:`` beside it."""
    comments = _comments(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    for line, (text, _) in sorted(comments.items()):
        if "type: ignore" not in text or "CRITICAL:" in text:
            continue
        above = line - 1
        justified = False
        while above in comments and comments[above][1]:
            if comments[above][0].startswith("# CRITICAL:"):
                justified = True
                break
            above -= 1
        if not justified:
            findings.append(
                Finding(
                    UNJUSTIFIED_IGNORE,
                    relative,
                    line,
                    "'# type: ignore' comment with no adjacent '# CRITICAL:' saying why "
                    "the checker is wrong here",
                )
            )
    return findings


def check(root: Path = REPOSITORY_ROOT) -> list[Finding]:
    """Return every finding in ``root``, including the vacuity guard's own."""
    modules = _sources(root, *PACKAGE_PARTS)
    if not modules:
        where = Path(*PACKAGE_PARTS).as_posix()
        return [Finding(VACUOUS, where, 0, "the scan matched no Python module; it proved nothing")]

    findings: list[Finding] = []
    handlers = 0
    fallbacks = 0
    for path in modules:
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scan = _ModuleScan(relative, tree)
        findings.extend(scan.findings)
        handlers += scan.handlers
        fallbacks += scan.fallbacks

    where = Path(*PACKAGE_PARTS).as_posix()
    if handlers == 0:
        return [
            Finding(
                VACUOUS,
                where,
                0,
                f"parsed {len(modules)} modules and found no 'except' handler at all; "
                "either error handling has left this package, or the scan is broken",
            )
        ]

    if fallbacks == 0:
        return [
            Finding(
                VACUOUS,
                where,
                0,
                f"parsed {handlers} handlers and put none of them in front of the "
                "swallow-fallback matcher; a blocking rule that never runs reports clean "
                "for the same reason a broken one does",
            )
        ]

    for name in IGNORE_SCAN_ROOTS:
        for path in _sources(root, name):
            findings.extend(_unjustified_ignores(path, path.relative_to(root).as_posix()))

    return findings


def over_ratchet(findings: list[Finding]) -> list[Finding]:
    """Return the error-severity findings beyond what the ratchet already records."""
    counted: dict[str, int] = {}
    excess: list[Finding] = []
    for finding in findings:
        if finding.rule not in ERROR_RULES:
            continue
        counted[finding.key] = counted.get(finding.key, 0) + 1
        if counted[finding.key] > RATCHET.get(finding.key, 0):
            excess.append(finding)
    return excess


def _report(write: Callable[[str], int], heading: str, findings: list[Finding]) -> None:
    """Write one headed block of findings, or nothing when the block is empty."""
    if not findings:
        return
    write(f"{heading}\n")
    for finding in findings:
        write(f"  {finding}\n")
    write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the gate, reporting errors, warnings, and the ratcheted backlog separately."""
    parser = argparse.ArgumentParser(description="Check the typed error taxonomy.")
    parser.add_argument(
        "--strict", action="store_true", help="fail on warnings as well as on errors"
    )
    arguments = parser.parse_args(argv)

    findings = check()
    excess = over_ratchet(findings)
    warnings = [finding for finding in findings if finding.rule not in ERROR_RULES]
    known = [
        finding for finding in findings if finding.rule in ERROR_RULES and finding not in excess
    ]

    _report(sys.stderr.write, "the typed error taxonomy is broken:", excess)
    _report(sys.stdout.write, "warnings (judge these, they do not fail the build):", warnings)
    _report(sys.stdout.write, "known and ratcheted (a backlog, not a permission):", known)

    if excess:
        sys.stderr.write(
            "a swallowed exception in this pipeline produces a confidently wrong answer "
            "rather than an error. Log the failure with its trace id, or rethrow it as a "
            "typed error. Do not widen the ratchet in scripts/check_error_taxonomy.py.\n"
        )
        return 1
    if arguments.strict and warnings:
        sys.stderr.write("--strict: warnings are failures in this run.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
