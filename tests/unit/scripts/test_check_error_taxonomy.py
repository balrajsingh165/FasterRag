from pathlib import Path

from check_error_taxonomy import (
    BROAD_EXCEPT,
    RATCHET,
    REPOSITORY_ROOT,
    SWALLOW,
    SWALLOW_DOCUMENTED,
    UNJUSTIFIED_IGNORE,
    UNTYPED_RAISE,
    VACUOUS,
    Finding,
    check,
    over_ratchet,
)


def build(root: Path, **modules: str) -> Path:
    """Write a fake package, mapping a dotted-free relative path to its source."""
    for relative, source in modules.items():
        path = root / "src" / "fasterrag" / relative.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def rules(findings: list[Finding], rule: str) -> list[Finding]:
    return [finding for finding in findings if finding.rule == rule]


HANDLER = """
def anchor() -> None:
    \"\"\"Give every tree one honest handler so the vacuity guard stays quiet.\"\"\"
    try:
        pass
    except ValueError:
        raise
"""


def test_a_handler_that_only_passes_is_a_swallow(tmp_path: Path) -> None:
    """The canonical case: the failure leaves no trace anywhere."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def work() -> None:
    \"\"\"Do the work.\"\"\"
    try:
        step()
    except OSError:
        pass
"""
        },
    )

    found = rules(check(root), SWALLOW)

    assert len(found) == 1
    assert found[0].path == "src/fasterrag/a.py"
    assert found[0].line == 13


def test_a_handler_that_only_continues_is_a_swallow(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def work() -> None:
    \"\"\"Do the work.\"\"\"
    for item in items():
        try:
            step(item)
        except OSError:
            continue
"""
        },
    )

    assert len(rules(check(root), SWALLOW)) == 1


def test_a_handler_that_returns_none_from_a_plain_function_is_a_swallow(tmp_path: Path) -> None:
    """``-> None`` publishes nothing about failure, so the caller cannot see it."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def work() -> None:
    \"\"\"Do the work.\"\"\"
    try:
        step()
    except OSError:
        return
"""
        },
    )

    assert len(rules(check(root), SWALLOW)) == 1


def test_a_handler_that_logs_is_not_a_swallow(tmp_path: Path) -> None:
    """The log line carries the trace id, which is what the rule asks for."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def work() -> None:
    \"\"\"Do the work.\"\"\"
    try:
        step()
    except OSError as exc:
        _logger.warning("step failed", extra={"error": str(exc)})
"""
        },
    )

    assert rules(check(root), SWALLOW) == []


def test_a_handler_that_rethrows_is_not_a_swallow(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def work() -> None:
    \"\"\"Do the work.\"\"\"
    try:
        step()
    except OSError as exc:
        raise RetrievalError("step failed") from exc
"""
        },
    )

    assert rules(check(root), SWALLOW) == []


def test_the_iterator_protocols_own_signal_is_not_a_swallow(tmp_path: Path) -> None:
    """``StopAsyncIteration`` says the stream ended; it is not a failure to report."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def drain() -> None:
    \"\"\"Pull every event.\"\"\"
    while True:
        try:
            yield run(events.__anext__())
        except StopAsyncIteration:
            return
"""
        },
    )

    assert rules(check(root), SWALLOW) == []


def test_a_pass_arm_on_a_try_except_else_is_a_probe_not_a_swallow(tmp_path: Path) -> None:
    """The ``else`` carries the work, so the handler is the false branch of a question."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def enter() -> None:
    \"\"\"Refuse to run inside a loop.\"\"\"
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise FasterRagError("already inside a running event loop")
"""
        },
    )

    assert rules(check(root), SWALLOW) == []


def test_returning_none_from_an_optional_function_is_only_a_warning(tmp_path: Path) -> None:
    """The absence is published in the annotation, so it downgrades but never vanishes."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def read() -> dict[str, int] | None:
    \"\"\"Return the payload, or None when it is unreadable.\"\"\"
    try:
        return load()
    except OSError:
        return None
"""
        },
    )

    findings = check(root)

    assert rules(findings, SWALLOW) == []
    assert len(rules(findings, SWALLOW_DOCUMENTED)) == 1


def test_a_broad_catch_that_translates_is_the_adapter_boundary_working(tmp_path: Path) -> None:
    """Catching every vendor type and raising a typed one is the specified pattern."""
    root = build(
        tmp_path,
        **{
            "adapters__llm.py": HANDLER
            + """
def call() -> str:
    \"\"\"Call the provider.\"\"\"
    try:
        return vendor.complete()
    except Exception as exc:
        raise classify_llm_failure(exc) from exc
"""
        },
    )

    assert rules(check(root), BROAD_EXCEPT) == []


def test_a_broad_catch_that_rethrows_nothing_is_reported(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
def call() -> None:
    \"\"\"Call the provider.\"\"\"
    try:
        vendor.complete()
    except Exception as exc:
        _logger.warning("provider failed", extra={"error": str(exc)})
"""
        },
    )

    found = rules(check(root), BROAD_EXCEPT)

    assert len(found) == 1
    assert "except Exception" in found[0].detail


def test_handing_off_to_a_noreturn_callee_counts_as_rethrowing(tmp_path: Path) -> None:
    """``_abandon`` is annotated NoReturn; the handler does end in a raise."""
    root = build(
        tmp_path,
        **{
            "services__importer.py": HANDLER
            + """
async def _abandon(cause: Exception) -> NoReturn:
    \"\"\"Undo the half-written import and rethrow.\"\"\"
    raise cause


async def run() -> None:
    \"\"\"Import the archive.\"\"\"
    try:
        await write()
    except Exception as exc:
        await _abandon(exc)
"""
        },
    )

    assert rules(check(root), BROAD_EXCEPT) == []


def test_a_builtin_raised_out_of_a_public_service_function_is_reported(tmp_path: Path) -> None:
    """Only the typed hierarchy crosses a service boundary."""
    root = build(
        tmp_path,
        **{
            "services__regression.py": HANDLER
            + """
def validate(payload: object) -> None:
    \"\"\"Refuse a baseline of the wrong shape.\"\"\"
    if not isinstance(payload, dict):
        raise ValueError("a baseline is a JSON object")
"""
        },
    )

    found = rules(check(root), UNTYPED_RAISE)

    assert len(found) == 1
    assert "raise ValueError" in found[0].detail


def test_a_builtin_raised_from_a_private_helper_is_not_reported(tmp_path: Path) -> None:
    """A private helper's ValueError is translated by the public caller that wraps it."""
    root = build(
        tmp_path,
        **{
            "services__regression.py": HANDLER
            + """
def _validate(payload: object) -> None:
    \"\"\"Refuse a baseline of the wrong shape.\"\"\"
    raise ValueError("a baseline is a JSON object")
"""
        },
    )

    assert rules(check(root), UNTYPED_RAISE) == []


def test_a_builtin_raised_inside_a_try_is_not_reported(tmp_path: Path) -> None:
    """An enclosing try is the translation point; the builtin never leaves."""
    root = build(
        tmp_path,
        **{
            "services__regression.py": HANDLER
            + """
def validate(payload: object) -> None:
    \"\"\"Refuse a baseline of the wrong shape.\"\"\"
    try:
        raise ValueError("a baseline is a JSON object")
    except ValueError as exc:
        raise FasterRagError("invalid baseline") from exc
"""
        },
    )

    assert rules(check(root), UNTYPED_RAISE) == []


def test_not_implemented_error_is_the_abstract_method_idiom(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        **{
            "adapters__base.py": HANDLER
            + """
def search() -> None:
    \"\"\"Subclasses implement this.\"\"\"
    raise NotImplementedError
"""
        },
    )

    assert rules(check(root), UNTYPED_RAISE) == []


def test_the_raise_rule_applies_only_at_a_service_or_adapter_boundary(tmp_path: Path) -> None:
    """``core/`` is internal; the rule is about what crosses a published boundary."""
    root = build(
        tmp_path,
        **{
            "core__chunking.py": HANDLER
            + """
def split() -> None:
    \"\"\"Split the document.\"\"\"
    raise ValueError("bad offsets")
"""
        },
    )

    assert rules(check(root), UNTYPED_RAISE) == []


def test_a_type_ignore_without_a_critical_is_reported(tmp_path: Path) -> None:
    root = build(
        tmp_path, **{"a.py": HANDLER + "\nvalue = untyped()  # type: ignore[no-any-return]\n"}
    )

    found = rules(check(root), UNJUSTIFIED_IGNORE)

    assert len(found) == 1
    assert found[0].path == "src/fasterrag/a.py"


def test_a_type_ignore_justified_by_the_block_above_it_passes(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + """
# CRITICAL: the stub ships no annotations, so the checker infers Any here.
# Removing the ignore does not fix the stub, it only reddens the build.
value = untyped()  # type: ignore[no-any-return]
"""
        },
    )

    assert rules(check(root), UNJUSTIFIED_IGNORE) == []


def test_a_type_ignore_justified_on_its_own_line_passes(tmp_path: Path) -> None:
    """``adjacent`` includes the same line, which is where a short reason belongs."""
    root = build(
        tmp_path,
        **{
            "a.py": HANDLER
            + "\nvalue = untyped()  # type: ignore[no-any-return]  # CRITICAL: the stub is Any\n"
        },
    )

    assert rules(check(root), UNJUSTIFIED_IGNORE) == []


def test_a_type_ignore_inside_a_string_literal_is_not_a_comment(tmp_path: Path) -> None:
    """The gate reads comment tokens, so its own error message never trips it."""
    root = build(
        tmp_path,
        **{"a.py": HANDLER + '\nMESSAGE = "a # type: ignore needs a # CRITICAL: beside it"\n'},
    )

    assert rules(check(root), UNJUSTIFIED_IGNORE) == []


def test_an_empty_tree_fails_rather_than_reporting_a_clean_one(tmp_path: Path) -> None:
    """Trap T12: a check that passes without running is a defect in its own right."""
    findings = check(tmp_path)

    assert len(findings) == 1
    assert findings[0].rule == VACUOUS


def test_a_tree_with_no_handler_at_all_fails(tmp_path: Path) -> None:
    """Modules but no ``except`` means the scan is broken, not that the code is perfect."""
    root = build(tmp_path, **{"a.py": 'def work() -> None:\n    """Work."""\n    return\n'})

    findings = check(root)

    assert len(findings) == 1
    assert findings[0].rule == VACUOUS
    assert "no 'except' handler" in findings[0].detail


def test_the_ratchet_absorbs_a_known_violation_but_not_a_second_one() -> None:
    """A ratcheted file may not grow; the entry records a backlog, not a permission."""
    known = Finding(SWALLOW, "src/fasterrag/services/traces.py", 146, "known")
    extra = Finding(SWALLOW, "src/fasterrag/services/traces.py", 999, "new")

    assert over_ratchet([known, known]) == []
    assert over_ratchet([known, known, extra]) == [extra]


def test_an_unratcheted_file_fails_on_its_first_violation() -> None:
    fresh = Finding(SWALLOW, "src/fasterrag/services/brand_new.py", 10, "new")

    assert over_ratchet([fresh]) == [fresh]


def test_warnings_never_fail_the_build() -> None:
    warning = Finding(BROAD_EXCEPT, "src/fasterrag/a.py", 10, "broad")

    assert over_ratchet([warning]) == []


def test_every_ratchet_entry_still_describes_the_real_tree() -> None:
    """A stale ratchet entry hides the next violation in that file behind a dead number."""
    findings = check(REPOSITORY_ROOT)
    counted: dict[str, int] = {}
    for finding in findings:
        counted[finding.key] = counted.get(finding.key, 0) + 1

    assert {key: counted.get(key, 0) for key in RATCHET} == RATCHET


def test_the_real_tree_is_scanned_rather_than_skipped() -> None:
    """The vacuity guard must never be what a green run actually reports."""
    findings = check(REPOSITORY_ROOT)

    assert rules(findings, VACUOUS) == []
