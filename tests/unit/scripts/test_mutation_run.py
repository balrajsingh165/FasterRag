import ast
import importlib.util
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from mutation_run import (
    ARITHMETIC_SWAPS,
    CEILING,
    COMPARISON_SWAPS,
    DEFAULT_TARGETS,
    EXIT_CODES,
    REPOSITORY_ROOT,
    STAGED_ROOTS,
    Ceiling,
    Mutation,
    RunReport,
    Target,
    TargetResult,
    Verdict,
    apply,
    child_peak_rss,
    decide,
    gate,
    generate,
    load_targets,
    main,
    markdown,
    memory_limiter,
    ratchet_block,
    read_report,
    render,
    resource_module,
    restrict,
    run,
    select,
    source_files,
    stage_suite,
    staged_tests,
)


def mutation(
    *,
    path: str = "src/fasterrag/core/breaker.py",
    line: int = 7,
    column: int = 3,
    operator: str = "comparison",
    before: str = ">=",
    after: str = ">",
    start: int = 0,
    end: int = 2,
) -> Mutation:
    """Build a Mutation without binding the test to the field order."""
    return Mutation(
        path=path,
        line=line,
        column=column,
        operator=operator,
        before=before,
        after=after,
        start=start,
        end=end,
    )


def operators(source: str) -> set[str]:
    return {item.operator for item in generate(source, "m.py")}


def substitutions(source: str) -> set[tuple[str, str]]:
    return {(item.before, item.after) for item in generate(source, "m.py")}


class TestGeneration:
    def test_comparison_operators_are_swapped_for_their_boundary_neighbour(self) -> None:
        assert ("<", "<=") in substitutions("def f(a, b):\n    return a < b\n")
        assert (">=", ">") in substitutions("def f(a, b):\n    return a >= b\n")
        assert ("==", "!=") in substitutions("def f(a, b):\n    return a == b\n")

    def test_identity_and_membership_operators_tolerate_their_two_word_spelling(self) -> None:
        assert ("is not", "is") in substitutions("def f(a, b):\n    return a is not b\n")
        assert ("not in", "in") in substitutions("def f(a, b):\n    return a not in b\n")

    def test_a_chained_comparison_yields_one_mutation_per_operator(self) -> None:
        found = generate("def f(a, b, c):\n    return a < b <= c\n", "m.py")
        assert sorted(item.before for item in found if item.operator == "comparison") == ["<", "<="]

    def test_arithmetic_and_boolean_operators_are_swapped(self) -> None:
        assert ("+", "-") in substitutions("def f(a, b):\n    return a + b\n")
        assert ("and", "or") in substitutions("def f(a, b):\n    return a and b\n")

    def test_a_not_is_removed_rather_than_replaced(self) -> None:
        found = generate("def f(a):\n    return not a\n", "m.py")
        removal = next(item for item in found if item.operator == "unary-not")
        assert removal.after == ""
        assert apply("def f(a):\n    return not a\n", removal) == "def f(a):\n    return a\n"

    def test_numeric_and_boolean_literals_are_perturbed(self) -> None:
        assert ("0", "1") in substitutions("def f():\n    return 0\n")
        assert ("1", "0") in substitutions("def f():\n    return 1\n")
        assert ("7", "8") in substitutions("def f():\n    return 7\n")
        assert ("True", "False") in substitutions("def f():\n    return True\n")

    def test_string_literals_are_left_alone(self) -> None:
        assert generate('def f():\n    """Doc."""\n    return "code"\n', "m.py") == []

    def test_every_mutation_reports_the_line_it_sits_on(self) -> None:
        source = "def f(a, b):\n    x = 1\n    if a >= b:\n        return x\n    return 0\n"
        comparison = next(item for item in generate(source, "m.py") if item.before == ">=")
        assert comparison.line == 3
        assert source.splitlines()[comparison.line - 1][comparison.column :].startswith(">=")

    def test_a_mutation_applies_as_the_exact_text_substitution_it_reports(self) -> None:
        source = "def f(a, b):\n    return a >= b\n"
        comparison = next(item for item in generate(source, "m.py") if item.before == ">=")
        assert apply(source, comparison) == "def f(a, b):\n    return a > b\n"

    def test_no_generated_mutation_changes_the_line_count(self) -> None:
        source = (REPOSITORY_ROOT / "src/fasterrag/core/breaker.py").read_text(encoding="utf-8")
        expected = source.count("\n")
        for item in generate(source, "breaker.py"):
            assert apply(source, item).count("\n") == expected

    def test_every_generated_mutation_still_parses_and_differs_structurally(self) -> None:
        source = (REPOSITORY_ROOT / "src/fasterrag/core/rate_limit.py").read_text(encoding="utf-8")
        original = ast.dump(ast.parse(source))
        found = generate(source, "rate_limit.py")
        assert found
        for item in found:
            assert ast.dump(ast.parse(apply(source, item))) != original

    def test_a_no_op_substitution_is_never_emitted(self) -> None:
        assert all(item.before != item.after for item in generate("x = 1 < 2 and 3\n", "m.py"))

    def test_a_substitution_that_leaves_the_tree_unchanged_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(COMPARISON_SWAPS, ast.Lt, ("<", "< "))
        source = "def f(a, b):\n    return a < b\n"
        assert [item for item in generate(source, "m.py") if item.operator == "comparison"] == []

    def test_a_substitution_that_adds_a_line_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(ARITHMETIC_SWAPS, ast.Add, ("+", "-\n"))
        source = "def f(a, b):\n    return (a + b)\n"
        assert [item for item in generate(source, "m.py") if item.operator == "arithmetic"] == []

    def test_the_whole_catalogue_is_reachable(self) -> None:
        source = (
            "def f(a, b):\n"
            "    c = a + b\n"
            "    if a >= b and not c:\n"
            "        return 0\n"
            "    return True\n"
        )
        assert operators(source) == {
            "arithmetic",
            "boolean-op",
            "comparison",
            "number",
            "boolean-constant",
            "unary-not",
        }


class TestSelection:
    def test_a_set_within_budget_is_returned_whole_and_in_order(self) -> None:
        found = [mutation(line=index) for index in range(5)]
        assert select(found, 10, 0) == found

    def test_a_budget_caps_the_sample(self) -> None:
        found = [mutation(line=index) for index in range(50)]
        assert len(select(found, 7, 0)) == 7

    def test_the_sample_is_reproducible_for_a_seed_and_moves_with_it(self) -> None:
        found = [mutation(line=index) for index in range(50)]
        assert select(found, 7, 0) == select(found, 7, 0)
        assert select(found, 7, 0) != select(found, 7, 99)

    def test_the_sample_keeps_the_source_order(self) -> None:
        found = [mutation(line=index) for index in range(50)]
        sampled = select(found, 9, 3)
        assert [item.line for item in sampled] == sorted(item.line for item in sampled)

    def test_an_absent_or_non_positive_budget_runs_everything(self) -> None:
        found = [mutation(line=index) for index in range(50)]
        assert select(found, None, 0) == found
        assert select(found, 0, 0) == found


class TestTargets:
    def test_the_default_set_names_only_paths_that_exist(self) -> None:
        root = REPOSITORY_ROOT
        for target in DEFAULT_TARGETS:
            assert source_files(root, target.source), target.source
            for test in target.tests:
                assert (root / test).exists(), test

    def test_a_package_target_collects_its_modules_and_a_file_target_itself(self) -> None:
        root = REPOSITORY_ROOT
        assert len(source_files(root, "src/fasterrag/core/retrieval")) > 1
        assert source_files(root, "src/fasterrag/core/breaker.py") == [
            root / "src/fasterrag/core/breaker.py"
        ]

    def test_a_missing_source_path_collects_nothing_rather_than_raising(self) -> None:
        assert source_files(REPOSITORY_ROOT, "src/fasterrag/core/not_a_module.py") == []

    def test_restrict_keeps_the_named_targets_in_the_order_asked_for(self) -> None:
        kept = restrict(DEFAULT_TARGETS, ["breaker", "identity"])
        assert [target.name for target in kept] == ["breaker", "identity"]

    def test_restrict_without_names_keeps_everything(self) -> None:
        assert restrict(DEFAULT_TARGETS, []) == list(DEFAULT_TARGETS)

    def test_an_unknown_target_name_is_refused_rather_than_silently_running_nothing(self) -> None:
        with pytest.raises(ValueError, match="unknown target"):
            restrict(DEFAULT_TARGETS, ["breaker", "nope"])

    def test_a_targets_file_is_loaded_into_targets(self) -> None:
        loaded = load_targets([{"name": "x", "source": "src/a.py", "tests": ["tests/a.py"]}])
        assert loaded == [Target(name="x", source="src/a.py", tests=("tests/a.py",))]

    @pytest.mark.parametrize(
        "document",
        [
            {"name": "x"},
            ["not-an-object"],
            [{"name": "x", "source": "src/a.py"}],
            [{"name": "x", "source": "src/a.py", "tests": []}],
            [{"name": "x", "source": "src/a.py", "tests": "tests/a.py"}],
            [{"name": 1, "source": "src/a.py", "tests": ["tests/a.py"]}],
        ],
    )
    def test_an_unrunnable_target_document_is_refused(self, document: object) -> None:
        with pytest.raises(ValueError, match="target"):
            load_targets(document)


def result(
    name: str = "breaker",
    *,
    executed: int = 4,
    killed: int = 4,
    survived: int = 0,
    survivors: list[Mutation] | None = None,
    timeouts: list[Mutation] | None = None,
    signals: list[tuple[Mutation, int]] | None = None,
    peak: int | None = None,
    problems: list[str] | None = None,
) -> TargetResult:
    """Build a TargetResult for the verdict and rendering tests."""
    return TargetResult(
        name=name,
        generated=executed,
        executed=executed,
        killed=killed,
        survived=survived,
        timed_out=len(timeouts or []),
        signalled=len(signals or []),
        peak_child_rss_bytes=peak,
        survivors=survivors or [],
        timeouts=timeouts or [],
        signals=signals or [],
        problems=problems or [],
    )


class TestVerdict:
    def test_every_mutant_killed_is_clean(self) -> None:
        assert decide([result()]) is Verdict.CLEAN

    def test_a_survivor_is_reported_as_survivors(self) -> None:
        assert decide([result(killed=3, survived=1, survivors=[mutation()])]) is Verdict.SURVIVORS

    def test_a_target_that_could_not_run_is_not_run_even_beside_a_clean_one(self) -> None:
        assert decide([result(), result("x", problems=["baseline red"])]) is Verdict.NOT_RUN

    def test_a_problem_outranks_a_survivor(self) -> None:
        broken = result("x", problems=["baseline red"])
        survived = result(killed=3, survived=1, survivors=[mutation()])
        assert decide([survived, broken]) is Verdict.NOT_RUN

    def test_no_targets_at_all_is_not_run_rather_than_clean(self) -> None:
        assert decide([]) is Verdict.NOT_RUN

    def test_targets_that_executed_nothing_are_not_run_rather_than_clean(self) -> None:
        assert decide([result(executed=0, killed=0)]) is Verdict.NOT_RUN

    def test_the_exit_codes_separate_the_three_verdicts(self) -> None:
        assert EXIT_CODES[Verdict.CLEAN] == 0
        assert EXIT_CODES[Verdict.SURVIVORS] == 1
        assert EXIT_CODES[Verdict.NOT_RUN] == 2


class TestReporting:
    def test_a_survivor_is_printed_with_its_file_line_and_substitution(self) -> None:
        survivor = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        text = render(RunReport(Verdict.SURVIVORS, [result(survived=1, survivors=[survivor])]))
        assert "src/fasterrag/core/breaker.py:82:67" in text
        assert "[comparison]" in text
        assert "'>=' -> '>'" in text

    def test_a_clean_run_prints_counts_and_no_survivor_section(self) -> None:
        text = render(RunReport(Verdict.CLEAN, [result()]))
        assert "mutation run: clean" in text
        assert "4 killed, 0 survived" in text
        assert "surviving mutants" not in text

    def test_a_target_that_could_not_run_prints_why(self) -> None:
        text = render(RunReport(Verdict.NOT_RUN, [result("x", problems=["the suite was red"])]))
        assert "[not-run] the suite was red" in text

    def test_the_markdown_summary_carries_the_survivors_not_just_the_score(self) -> None:
        survivor = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        body = markdown(RunReport(Verdict.SURVIVORS, [result(survived=1, survivors=[survivor])]))
        assert "`survivors`" in body
        assert "| breaker | 4 | 4 | 1 |" in body
        assert "src/fasterrag/core/breaker.py:82:67" in body

    def test_the_run_report_totals_every_target(self) -> None:
        report = RunReport(
            Verdict.SURVIVORS,
            [result(), result("x", executed=6, killed=5, survived=1, survivors=[mutation()])],
        )
        assert (report.executed, report.killed, report.survived) == (10, 9, 1)

    def test_the_json_document_carries_each_survivor_as_structured_fields(self) -> None:
        survivor = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        report = RunReport(Verdict.SURVIVORS, [result(survived=1, survivors=[survivor])])
        document = report.as_dict()
        assert document["verdict"] == "survivors"
        assert document["targets"][0]["survivors"] == [
            {
                "path": "src/fasterrag/core/breaker.py",
                "line": 82,
                "column": 67,
                "operator": "comparison",
                "before": ">=",
                "after": ">",
            }
        ]


class TestCommandLine:
    def test_list_counts_mutants_without_running_any_test(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--list", "--only", "breaker"]) == EXIT_CODES[Verdict.CLEAN]
        printed = capsys.readouterr().out
        assert printed.startswith("breaker: ")
        assert "mutants from src/fasterrag/core/breaker.py" in printed

    def test_an_unknown_target_exits_not_run_rather_than_clean(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--list", "--only", "nope"]) == EXIT_CODES[Verdict.NOT_RUN]
        assert "unknown target" in capsys.readouterr().err

    def test_an_unreadable_targets_file_exits_not_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--list", "--targets-file", str(tmp_path / "gone.json")]) == 2
        assert "target set unusable" in capsys.readouterr().err

    def test_a_malformed_targets_file_exits_not_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = tmp_path / "targets.json"
        document.write_text("{not json", encoding="utf-8")
        assert main(["--list", "--targets-file", str(document)]) == 2
        assert "target set unusable" in capsys.readouterr().err

    def test_a_targets_file_replaces_the_default_set(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        document = tmp_path / "targets.json"
        document.write_text(
            json.dumps(
                [
                    {
                        "name": "only-breaker",
                        "source": "src/fasterrag/core/breaker.py",
                        "tests": ["tests/unit/core/test_breaker.py"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        assert main(["--list", "--targets-file", str(document)]) == 0
        assert capsys.readouterr().out.startswith("only-breaker: ")


COMPARISON_MODULE = '''"""One comparison and no literal, so the module has exactly one mutant."""


def check(a: int, b: int) -> bool:
    """Report whether a is below b."""
    return a < b
'''

CONSTANT_MODULE = '''"""One integer literal, so the module's only mutant preserves its length."""


def limit() -> int:
    """Report the limit its test pins."""
    return 0
'''

PINNING_TEST = '''"""Pin the constant, so the mutant dies the moment it is actually executed."""

from pkgy.mod import limit


def test_limit() -> None:
    assert limit() == 0
'''

STRICT_TEST = '''"""The stricter test an edit lands mid-run; it notices ``<`` becoming ``<=``."""

from pkgx.mod import check


def test_check() -> None:
    assert check(1, 2) is True
    assert check(2, 2) is False
'''

LENIENT_TEST = '''"""The test the run starts with; it does not notice the mutant.

Executing it overwrites the live copy of itself with the stricter test, which is the
mid-run edit this regression is about.
"""

from pathlib import Path

from pkgx.mod import check

LIVE = Path({live})
STRICT = {strict}


def test_check() -> None:
    LIVE.write_text(STRICT, encoding="utf-8")
    assert check(1, 2) is True
'''


PYPROJECT = """[tool.pytest.ini_options]
addopts = ""
"""


def miniature(root: Path, package: str, module: str, test: str) -> None:
    """Lay out a miniature repository: one package, one module, one test, one ini."""
    for relative, body in (
        ("pyproject.toml", PYPROJECT),
        (f"src/{package}/__init__.py", ""),
        (f"src/{package}/mod.py", module),
        ("tests/__init__.py", ""),
        ("tests/test_mod.py", test),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")


def mutate(
    root: Path,
    workspace: Path,
    package: str,
    monkeypatch: pytest.MonkeyPatch,
    tests: tuple[str, ...] = ("tests/test_mod.py",),
) -> TargetResult:
    """Point the runner at a miniature repository and run its single target."""
    monkeypatch.setattr("mutation_run.REPOSITORY_ROOT", root)
    report = run(
        [Target(name="x", source=f"src/{package}/mod.py", tests=tests)],
        budget=None,
        seed=0,
        timeout=120.0,
        workspace=workspace,
    )
    return report.results[0]


class TestIsolation:
    """TASK-0303: a verdict may depend on nothing but the snapshot the run was given."""

    def test_a_test_edited_mid_run_cannot_change_a_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        live = root / "tests/test_mod.py"
        miniature(
            root,
            "pkgx",
            COMPARISON_MODULE,
            LENIENT_TEST.format(live=repr(str(live)), strict=repr(STRICT_TEST)),
        )

        outcome = mutate(root, tmp_path / "workspace", "pkgx", monkeypatch)

        assert live.read_text(encoding="utf-8") == STRICT_TEST
        assert outcome.problems == []
        assert (outcome.generated, outcome.executed) == (1, 1)
        assert outcome.survived == 1

    def test_a_stale_bytecode_cache_cannot_shadow_a_length_preserving_mutant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        module = root / "src/pkgy/mod.py"
        py_compile.compile(
            str(module),
            cfile=importlib.util.cache_from_source(str(module)),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert outcome.problems == []
        assert (outcome.generated, outcome.executed) == (1, 1)
        assert outcome.killed == 1
        assert outcome.survivors == []

    def test_the_workspace_keeps_no_bytecode_for_a_later_mutant_to_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        workspace = tmp_path / "workspace"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        module = root / "src/pkgy/mod.py"
        py_compile.compile(str(module), doraise=True)

        mutate(root, workspace, "pkgy", monkeypatch)

        assert list(workspace.rglob("__pycache__")) == []
        assert list(workspace.rglob("*.pyc")) == []

    def test_staging_copies_the_whole_test_tree_a_named_test_may_reach_into(
        self, tmp_path: Path
    ) -> None:
        stage_suite(tmp_path)
        resolved = staged_tests(("tests/unit/core/test_breaker.py", "tests/unit/scripts"), tmp_path)

        assert resolved == (
            str(tmp_path / "tests/unit/core/test_breaker.py"),
            str(tmp_path / "tests/unit/scripts"),
        )
        for neighbour in (
            "pyproject.toml",
            "tests/conftest.py",
            "tests/unit/scripts/test_mutation_run.py",
            "tests/eval/datasets/handbook/baseline.json",
        ):
            assert (tmp_path / neighbour).is_file(), neighbour
        assert list(tmp_path.rglob("__pycache__")) == []

    def test_a_test_path_outside_the_snapshot_is_refused(self, tmp_path: Path) -> None:
        stage_suite(tmp_path)
        with pytest.raises(ValueError, match="outside the snapshot"):
            staged_tests((str(REPOSITORY_ROOT / "tests/unit/core/test_breaker.py"),), tmp_path)

    def test_a_test_path_that_cannot_be_staged_is_reported_rather_than_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch, ("tests/test_gone.py",))

        assert outcome.executed == 0
        assert any("could not be snapshotted" in problem for problem in outcome.problems)


def fake_limits(recorder: list[tuple[int, tuple[int, int]]], maxrss: int = 0) -> ModuleType:
    """A stand-in for POSIX ``resource``, so the branch Windows cannot run is still tested."""
    module = ModuleType("fake_resource")
    module.RLIMIT_AS = 9  # type: ignore[attr-defined]
    module.RUSAGE_CHILDREN = -1  # type: ignore[attr-defined]
    module.setrlimit = lambda which, pair: recorder.append((which, pair))  # type: ignore[attr-defined]
    module.getrusage = lambda who: SimpleNamespace(ru_maxrss=maxrss)  # type: ignore[attr-defined]
    return module


def spy(seen: dict[str, Any]) -> Any:
    """Replace ``subprocess.run`` with a recorder, so the child's arguments are inspectable."""

    def record(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    return record


class TestResourceLimits:
    """TASK-0321: the POSIX branch is injected, because this machine cannot execute it."""

    def test_no_limit_is_asked_for_means_no_preexec_hook(self) -> None:
        assert memory_limiter(None, fake_limits([])) is None
        assert memory_limiter(0, fake_limits([])) is None
        assert memory_limiter(-1, fake_limits([])) is None

    def test_a_limit_caps_the_address_space_in_both_rlimit_fields(self) -> None:
        recorded: list[tuple[int, tuple[int, int]]] = []
        limiter = memory_limiter(64, fake_limits(recorded))

        assert limiter is not None
        limiter()

        assert recorded == [(9, (64 * 1024 * 1024, 64 * 1024 * 1024))]

    def test_a_limit_on_a_platform_without_resource_is_refused_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="POSIX resource limits"):
            memory_limiter(64, None)

    def test_the_resource_module_is_present_exactly_where_the_platform_has_it(self) -> None:
        assert (resource_module() is None) is (sys.platform == "win32")

    def test_the_peak_rss_is_scaled_by_the_unit_the_platform_reports(self) -> None:
        limits = fake_limits([], maxrss=2048)
        assert child_peak_rss(limits, "linux") == 2048 * 1024
        assert child_peak_rss(limits, "darwin") == 2048
        assert child_peak_rss(None, "linux") is None

    def test_the_limiter_reaches_the_child_as_its_preexec_hook(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import mutation_run

        seen: dict[str, Any] = {}
        monkeypatch.setattr("mutation_run.subprocess.run", spy(seen))
        limiter = memory_limiter(64, fake_limits([]))

        assert mutation_run._pytest(("tests",), tmp_path, 1.0, limiter) == 0
        assert seen["preexec_fn"] is limiter

    def test_no_limit_leaves_the_hook_unset_so_windows_can_spawn_at_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import mutation_run

        seen: dict[str, Any] = {}
        monkeypatch.setattr("mutation_run.subprocess.run", spy(seen))

        assert mutation_run._pytest(("tests",), tmp_path, 1.0, None) == 0
        assert seen["preexec_fn"] is None

    def test_a_memory_limit_off_posix_exits_not_run_rather_than_running_uncapped(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        monkeypatch.setattr(mutation_run, "resource_module", lambda: None)

        assert main(["--only", "breaker", "--memory-mb", "512"]) == EXIT_CODES[Verdict.NOT_RUN]
        assert "POSIX resource limits" in capsys.readouterr().err

    def test_the_baseline_runs_under_the_same_cap_as_every_mutant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        monkeypatch.setattr("mutation_run.REPOSITORY_ROOT", root)
        limiter = memory_limiter(64, fake_limits([]))
        given: list[Any] = []

        def record(
            tests: tuple[str, ...], workspace: Path, timeout: float, hook: Any = None
        ) -> int | None:
            given.append(hook)
            return 1 if given[1:] else 0

        monkeypatch.setattr(mutation_run, "_pytest", record)
        run(
            [Target(name="x", source="src/pkgy/mod.py", tests=("tests/test_mod.py",))],
            budget=None,
            seed=0,
            timeout=120.0,
            workspace=tmp_path / "workspace",
            limiter=limiter,
        )

        assert given == [limiter, limiter]


class TestTimeouts:
    """A timeout is counted as a kill and named as a timeout, never absorbed into a score."""

    def test_a_timed_out_mutant_is_named_as_well_as_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        calls: list[int] = []

        def never_finishes(
            tests: tuple[str, ...], workspace: Path, timeout: float, limiter: Any = None
        ) -> int | None:
            calls.append(1)
            return 0 if len(calls) == 1 else None

        monkeypatch.setattr(mutation_run, "_pytest", never_finishes)

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert (outcome.executed, outcome.killed, outcome.survived) == (1, 1, 0)
        assert outcome.timed_out == 1
        assert [item.line for item in outcome.timeouts] == [6]

    def test_the_killed_count_still_accounts_for_every_mutant_executed(self) -> None:
        outcome = result(executed=5, killed=3, survived=2, timeouts=[mutation()])
        assert outcome.killed + outcome.survived == outcome.executed

    def test_the_human_report_says_which_kills_were_the_clock(self) -> None:
        timed = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        text = render(RunReport(Verdict.CLEAN, [result(timeouts=[timed])]))
        assert "timed-out mutants" in text
        assert "src/fasterrag/core/breaker.py:82:67" in text

    def test_the_markdown_summary_lists_the_timed_out_mutants(self) -> None:
        timed = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        body = markdown(RunReport(Verdict.CLEAN, [result(timeouts=[timed])]))
        assert "**Timed out**" in body
        assert "src/fasterrag/core/breaker.py:82:67" in body

    def test_the_json_artifact_names_the_timed_out_mutants(self) -> None:
        timed = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        document = RunReport(Verdict.CLEAN, [result(timeouts=[timed])]).as_dict()
        assert document["timed_out"] == 1
        assert document["targets"][0]["timeouts"] == [timed.as_dict()]


class TestProgress:
    """A SIGTERM must leave the name of the mutant that was in flight when it arrived."""

    def test_every_mutant_is_announced_before_it_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)

        mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        printed = capsys.readouterr().out.splitlines()
        assert "[x] baseline: 1 staged test path(s)" in printed
        assert any(line.startswith("[x] 1/1 src/pkgy/mod.py:6:") for line in printed)
        assert any(
            line.startswith("[x] done: 1/1 run, 1 killed (0 by the clock, 0 by a signal)")
            for line in printed
        )

    def test_the_announcement_precedes_the_run_it_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        announced: list[str] = []
        real = mutation_run._pytest

        def watched(
            tests: tuple[str, ...], workspace: Path, timeout: float, limiter: Any = None
        ) -> int | None:
            announced.append(capsys.readouterr().out)
            return real(tests, workspace, timeout, limiter)

        monkeypatch.setattr(mutation_run, "_pytest", watched)
        mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert "1/1 src/pkgy/mod.py" in announced[-1]


class TestIncrementalArtifact:
    """A campaign that is killed leaves the targets it finished rather than nothing."""

    def test_the_report_grows_one_target_at_a_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        monkeypatch.setattr("mutation_run.REPOSITORY_ROOT", root)
        targets = [
            Target(name=name, source="src/pkgy/mod.py", tests=("tests/test_mod.py",))
            for name in ("a", "b")
        ]
        seen: list[dict[str, Any]] = []

        report = run(
            targets,
            budget=None,
            seed=0,
            timeout=120.0,
            workspace=tmp_path / "workspace",
            after_target=lambda partial: seen.append(partial.as_dict()),
        )

        assert [len(document["targets"]) for document in seen] == [1, 2]
        assert seen[0]["complete"] is False
        assert seen[0]["planned"] == 2
        assert seen[-1]["complete"] is True
        assert report.complete is True

    def test_a_partial_report_says_so_in_both_renderings(self) -> None:
        report = RunReport(Verdict.CLEAN, [result()], planned=3)
        assert "PARTIAL: 1 of 3 targets ran" in render(report)
        assert "**Partial:** 1 of 3 targets ran." in markdown(report)
        assert report.as_dict()["complete"] is False

    def test_a_whole_run_is_never_labelled_partial(self) -> None:
        report = RunReport(Verdict.CLEAN, [result()], planned=1)
        assert "PARTIAL" not in render(report)
        assert report.as_dict()["complete"] is True

    def test_the_peak_child_rss_reaches_the_artifact_when_the_platform_reports_one(self) -> None:
        report = RunReport(Verdict.CLEAN, [result()], peak_child_rss_bytes=3 * 1024 * 1024)
        assert report.as_dict()["peak_child_rss_bytes"] == 3 * 1024 * 1024
        assert "peak child RSS: 3 MiB" in render(report)

    def test_a_platform_without_rusage_reports_no_peak_rather_than_zero(self) -> None:
        document = RunReport(Verdict.CLEAN, [result()]).as_dict()
        assert document["peak_child_rss_bytes"] is None

    def test_the_artifact_is_replaced_whole_and_leaves_no_half_written_file(
        self, tmp_path: Path
    ) -> None:
        import mutation_run

        path = tmp_path / "mutation.json"
        mutation_run._write_json(path, RunReport(Verdict.CLEAN, [result()], planned=2))
        mutation_run._write_json(path, RunReport(Verdict.SURVIVORS, [result(), result("y")]))

        assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == "survivors"
        assert list(tmp_path.iterdir()) == [path]

    def test_an_artifact_that_cannot_be_written_is_reported_and_never_raises(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import mutation_run

        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")

        mutation_run._write_json(
            tmp_path / "blocked" / "mutation.json", RunReport(Verdict.CLEAN, [result()])
        )

        printed = capsys.readouterr().err
        assert "could not write" in printed
        assert "could not replace" in printed


class TestStagingScope:
    """TASK-0331: what the snapshot holds, and what a test reaching past itself will not find."""

    def test_the_docs_tree_is_staged_so_a_test_that_reads_a_spec_finds_it(
        self, tmp_path: Path
    ) -> None:
        stage_suite(tmp_path)

        assert (tmp_path / "docs" / "archive-format.md").is_file()

    def test_only_the_named_roots_and_the_ini_are_staged(self, tmp_path: Path) -> None:
        stage_suite(tmp_path)

        assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
            [*STAGED_ROOTS, "pyproject.toml"]
        )
        for absent in ("config.yaml", "scripts", ".github", "README.md", "src"):
            assert not (tmp_path / absent).exists(), absent


class TestSignalledChildren:
    """TASK-0362: a cgroup OOM kill is not a test failure and may not read as one."""

    def test_a_signalled_child_is_counted_as_a_kill_and_named_as_a_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        calls: list[int] = []

        def oom_killed(
            tests: tuple[str, ...], workspace: Path, timeout: float, limiter: Any = None
        ) -> int | None:
            calls.append(1)
            return 0 if len(calls) == 1 else -9

        monkeypatch.setattr(mutation_run, "_pytest", oom_killed)

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert (outcome.executed, outcome.killed, outcome.survived) == (1, 1, 0)
        assert outcome.signalled == 1
        assert outcome.timed_out == 0
        assert [code for _, code in outcome.signals] == [-9]

    def test_an_ordinary_test_failure_is_still_an_unremarkable_kill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        calls: list[int] = []

        def failing(
            tests: tuple[str, ...], workspace: Path, timeout: float, limiter: Any = None
        ) -> int | None:
            calls.append(1)
            return 0 if len(calls) == 1 else 1

        monkeypatch.setattr(mutation_run, "_pytest", failing)

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert (outcome.killed, outcome.signalled, outcome.signals) == (1, 0, [])

    def test_the_unproven_kills_are_the_clock_plus_the_signals(self) -> None:
        outcome = result(executed=5, killed=5, timeouts=[mutation()], signals=[(mutation(), -9)])
        assert outcome.unproven == 2

    def test_a_signalled_baseline_says_a_memory_cap_could_not_have_caught_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        monkeypatch.setattr(
            mutation_run, "_pytest", lambda tests, workspace, timeout, limiter=None: -9
        )

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert outcome.executed == 0
        assert any("SIGKILL" in problem for problem in outcome.problems)
        assert any("--memory-mb" in problem for problem in outcome.problems)

    def test_a_baseline_that_never_finished_is_described_as_a_timeout_not_an_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        monkeypatch.setattr(
            mutation_run, "_pytest", lambda tests, workspace, timeout, limiter=None: None
        )

        outcome = mutate(root, tmp_path / "workspace", "pkgy", monkeypatch)

        assert any("did not finish inside" in problem for problem in outcome.problems)

    def test_an_unknown_signal_number_is_still_named_rather_than_dropped(self) -> None:
        document = result(executed=1, killed=1, signals=[(mutation(), -999)]).as_dict()
        assert document["signals"][0]["signal_name"] == "signal 999"

    def test_both_renderings_separate_a_signal_from_a_kill_a_test_earned(self) -> None:
        killed = mutation(path="src/fasterrag/core/breaker.py", line=82, column=67)
        report = RunReport(Verdict.CLEAN, [result(executed=4, killed=4, signals=[(killed, -9)])])

        assert "signalled mutants" in render(report)
        assert "SIGKILL src/fasterrag/core/breaker.py:82:67" in render(report)
        assert "**Signalled**" in markdown(report)
        assert report.as_dict()["signalled"] == 1


class TestPerTargetPeak:
    """TASK-0363: which target approached the cap is the question a calibrator has."""

    def test_each_target_records_the_mark_as_it_stood_when_it_finished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "root"
        miniature(root, "pkgy", CONSTANT_MODULE, PINNING_TEST)
        monkeypatch.setattr("mutation_run.REPOSITORY_ROOT", root)
        rising = iter([64, 4096])
        limits = fake_limits([])
        limits.getrusage = lambda who: SimpleNamespace(ru_maxrss=next(rising))  # type: ignore[attr-defined]

        report = run(
            [
                Target(name=name, source="src/pkgy/mod.py", tests=("tests/test_mod.py",))
                for name in ("a", "b")
            ],
            budget=None,
            seed=0,
            timeout=120.0,
            workspace=tmp_path / "workspace",
            limits=limits,
        )

        assert [target.peak_child_rss_bytes for target in report.results] == [
            64 * 1024,
            4096 * 1024,
        ]
        assert report.peak_child_rss_bytes == 4096 * 1024

    def test_the_reports_name_the_target_that_reached_the_figure(self) -> None:
        report = RunReport(
            Verdict.CLEAN,
            [result("a", peak=1 * 1024 * 1024), result("b", peak=9 * 1024 * 1024)],
            peak_child_rss_bytes=9 * 1024 * 1024,
        )

        assert "peak child RSS: 9 MiB, first reached in b" in render(report)
        assert "first reached in `b`" in markdown(report)
        assert "| b | 4 | 4 | 0 | 0 | 0 | 9 |" in markdown(report)

    def test_a_platform_without_rusage_shows_a_dash_rather_than_a_zero(self) -> None:
        assert "| breaker | 4 | 4 | 0 | 0 | 0 | - |" in markdown(
            RunReport(Verdict.CLEAN, [result()])
        )

    def test_the_per_target_mark_reaches_the_artifact(self) -> None:
        document = RunReport(Verdict.CLEAN, [result(peak=7 * 1024 * 1024)]).as_dict()
        assert document["targets"][0]["peak_child_rss_bytes"] == 7 * 1024 * 1024


MEASURED_NIGHTLY: dict[str, int] = {
    "retrieval": 1,
    "chunking": 5,
    "identity": 2,
    "breaker": 1,
    "regression": 0,
    "rate-limit": 2,
}
"""The per-target survivor split of nightly runs 42419161973, 42468563494, 42603353729 and
42800995556 (2026-11-07 to 11-10, --budget 10 --seed 0), which is what CEILING records."""


def nightly(
    survived: dict[str, int] | None = None,
    *,
    timeouts: dict[str, int] | None = None,
    planned: int | None = None,
) -> RunReport:
    """Build the report a whole nightly run writes, so a test can move one number."""
    counts = dict(MEASURED_NIGHTLY if survived is None else survived)
    clock = timeouts or {}
    results = [
        result(
            name,
            executed=10,
            killed=10 - count,
            survived=count,
            survivors=[mutation(line=index + 1) for index in range(count)],
            timeouts=[mutation(line=90 + index) for index in range(clock.get(name, 0))],
        )
        for name, count in counts.items()
    ]
    return RunReport(decide(results), results, planned=planned or len(results))


def written(tmp_path: Path, report: RunReport) -> Path:
    """Write a report where the gate reads one, through the artifact's own serialiser."""
    path = tmp_path / "mutation.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


class TestRatchetBaseline:
    """TASK-0382: the ceiling is a measurement, and nothing may quietly raise it."""

    def test_the_ceiling_covers_every_target_the_nightly_runs(self) -> None:
        assert set(CEILING) == {target.name for target in DEFAULT_TARGETS}

    def test_no_ceiling_stands_above_the_four_nights_that_measured_it(self) -> None:
        for name, measured in MEASURED_NIGHTLY.items():
            assert CEILING[name].survived <= measured, name
            assert CEILING[name].unproven == 0, name
            assert CEILING[name].sample == 10, name

    def test_the_measured_run_itself_holds(self, tmp_path: Path) -> None:
        verdict, text = gate(written(tmp_path, nightly()))

        assert verdict is Verdict.CLEAN
        assert "the mutation ratchet holds: 11 survived of 60 executed" in text


class TestRatchetFailures:
    """Every way the count may worsen, and the message that has to name it."""

    def test_one_more_survivor_in_one_target_fails_the_whole_run(self, tmp_path: Path) -> None:
        regressed = dict(MEASURED_NIGHTLY, chunking=6)

        verdict, text = gate(written(tmp_path, nightly(regressed)))

        assert verdict is Verdict.SURVIVORS
        assert EXIT_CODES[verdict] == 1
        assert "chunking: 6 mutants survived against a ceiling of 5 (1 more)" in text

    def test_a_target_that_rots_behind_another_that_improves_is_still_caught(
        self, tmp_path: Path
    ) -> None:
        offset = dict(MEASURED_NIGHTLY, identity=0, chunking=7)

        verdict, text = gate(written(tmp_path, nightly(offset)))

        assert verdict is Verdict.SURVIVORS
        assert "chunking" in text
        assert nightly(offset).survived == 11

    def test_the_worst_overshoot_is_named_first(self, tmp_path: Path) -> None:
        regressed = dict(MEASURED_NIGHTLY, breaker=2, chunking=9)

        _, text = gate(written(tmp_path, nightly(regressed)))
        reported = [line.strip().split(":")[0] for line in text.splitlines()[1:]]

        assert reported == ["chunking", "breaker"]

    def test_a_timeout_inside_a_perfect_kill_count_still_fails(self, tmp_path: Path) -> None:
        verdict, text = gate(written(tmp_path, nightly(timeouts={"breaker": 1})))

        assert verdict is Verdict.SURVIVORS
        assert "breaker: 1 kills were the clock and 0 were a signal" in text

    def test_a_regression_is_never_handed_a_loosened_ceiling_to_paste(self, tmp_path: Path) -> None:
        _, text = gate(written(tmp_path, nightly(dict(MEASURED_NIGHTLY, chunking=6))))

        assert "CEILING" not in text
        with pytest.raises(ValueError, match="never printed to raise one"):
            ratchet_block(nightly(dict(MEASURED_NIGHTLY, chunking=6)))


class TestRatchetVacuity:
    """Trap T12: an artifact that proves nothing must never read as a held ratchet."""

    def test_a_missing_artifact_fails_rather_than_passes(self, tmp_path: Path) -> None:
        verdict, text = gate(tmp_path / "never-written.json")

        assert verdict is Verdict.NOT_RUN
        assert EXIT_CODES[verdict] == 2
        assert "could not be read" in text

    def test_an_empty_artifact_fails_rather_than_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "mutation.json"
        path.write_text("", encoding="utf-8")

        verdict, text = gate(path)

        assert verdict is Verdict.NOT_RUN
        assert "could not be read" in text

    @pytest.mark.parametrize(
        ("document", "expected"),
        [
            ({}, "lists no target"),
            ({"targets": []}, "lists no target"),
            ([], "not a JSON object"),
        ],
    )
    def test_a_document_with_nothing_in_it_fails(
        self, tmp_path: Path, document: object, expected: str
    ) -> None:
        path = tmp_path / "mutation.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        verdict, text = gate(path)

        assert verdict is Verdict.NOT_RUN
        assert expected in text

    def test_a_run_that_executed_no_mutant_fails(self, tmp_path: Path) -> None:
        empty = RunReport(Verdict.NOT_RUN, [result("breaker", executed=0, killed=0)], planned=1)

        verdict, text = gate(written(tmp_path, empty))

        assert verdict is Verdict.NOT_RUN
        assert "no target executed a single mutant" in text

    def test_a_red_baseline_fails_as_a_run_that_could_not_be_performed(
        self, tmp_path: Path
    ) -> None:
        broken = RunReport(
            Verdict.NOT_RUN,
            [result("breaker", executed=0, killed=0, problems=["breaker: the suite was red"])],
            planned=1,
        )

        verdict, text = gate(written(tmp_path, broken))

        assert verdict is Verdict.NOT_RUN
        assert "the suite was red" in text

    def test_a_partial_artifact_fails_rather_than_judging_part_of_a_run(
        self, tmp_path: Path
    ) -> None:
        verdict, text = gate(written(tmp_path, nightly(planned=7)))

        assert verdict is Verdict.NOT_RUN
        assert "the artifact is a partial: 6 of 7 targets finished" in text

    def test_a_run_that_skipped_a_target_cannot_satisfy_the_ceiling(self, tmp_path: Path) -> None:
        verdict, text = gate(written(tmp_path, nightly({"breaker": 1})))

        assert verdict is Verdict.NOT_RUN
        assert "the ceiling covers chunking, identity, rate-limit, regression, retrieval" in text

    def test_a_target_with_no_ceiling_fails_rather_than_running_ungated(
        self, tmp_path: Path
    ) -> None:
        extra = dict(MEASURED_NIGHTLY, archive=3)

        verdict, text = gate(written(tmp_path, nightly(extra)))

        assert verdict is Verdict.NOT_RUN
        assert "archive ran with no ceiling" in text

    def test_a_verdict_that_disagrees_with_its_own_counts_fails(self, tmp_path: Path) -> None:
        document = nightly().as_dict()
        document["verdict"] = "clean"
        path = tmp_path / "mutation.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        verdict, text = gate(path)

        assert verdict is Verdict.NOT_RUN
        assert "calls itself 'clean' while its own counts say 'survivors'" in text

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        [
            ("executed", 11, "is not"),
            ("survivors", [], "are named"),
            ("timed_out", 99, "unproven kills"),
            ("killed", "nine", "not an integer"),
            ("signalled", None, "not an integer"),
            ("name", 7, "not a string"),
        ],
    )
    def test_an_artifact_whose_arithmetic_cannot_be_trusted_fails(
        self, tmp_path: Path, key: str, value: object, expected: str
    ) -> None:
        document = nightly().as_dict()
        document["targets"][1][key] = value
        path = tmp_path / "mutation.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        verdict, text = gate(path)

        assert verdict is Verdict.NOT_RUN
        assert expected in text

    def test_an_artifact_written_before_signals_were_told_apart_cannot_be_judged(
        self, tmp_path: Path
    ) -> None:
        document = nightly().as_dict()
        for target in document["targets"]:
            del target["signalled"]
        path = tmp_path / "mutation.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        verdict, text = gate(path)

        assert verdict is Verdict.NOT_RUN
        assert "signalled is missing" in text


class TestRatchetTightening:
    """A ratchet nobody can lower is deleted; a ratchet anyone can raise is not one."""

    def test_an_improvement_prints_the_block_that_records_it(self, tmp_path: Path) -> None:
        better = dict(MEASURED_NIGHTLY, chunking=3)

        verdict, text = gate(written(tmp_path, nightly(better)))

        assert verdict is Verdict.CLEAN
        assert "tighten the ceiling in scripts/mutation_run.py" in text
        assert '"chunking": Ceiling(survived=3, sample=10),' in text
        assert '"breaker": Ceiling(survived=1, sample=10),' in text

    def test_a_run_exactly_on_the_ceiling_asks_for_nothing(self, tmp_path: Path) -> None:
        _, text = gate(written(tmp_path, nightly()))

        assert "tighten" not in text

    def test_the_printed_block_records_what_ran_not_what_was_allowed(self) -> None:
        allowed = {"breaker": Ceiling(survived=1, sample=10, unproven=2)}
        report = RunReport(
            Verdict.SURVIVORS,
            [result("breaker", executed=10, killed=9, survived=1, survivors=[mutation()])],
            planned=1,
        )

        assert ratchet_block(report, allowed) == (
            "CEILING: Final[dict[str, Ceiling]] = {\n"
            '    "breaker": Ceiling(survived=1, sample=10),\n}'
        )

    def test_a_block_that_carries_an_unproven_kill_says_so(self) -> None:
        allowed = {"breaker": Ceiling(survived=0, sample=4, unproven=3)}
        report = RunReport(Verdict.CLEAN, [result("breaker", timeouts=[mutation()])], planned=1)

        assert '"breaker": Ceiling(survived=0, sample=4, unproven=1),' in ratchet_block(
            report, allowed
        )


class TestRatchetCommandLine:
    """The gate has to be runnable against a downloaded artifact, not only inside a run."""

    def test_a_held_ratchet_exits_zero_and_says_so_on_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = written(tmp_path, nightly())

        assert main(["--check-ratchet", str(path)]) == 0
        assert "the mutation ratchet holds" in capsys.readouterr().out

    def test_a_regression_exits_one_and_says_so_on_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = written(tmp_path, nightly(dict(MEASURED_NIGHTLY, retrieval=4)))

        assert main(["--check-ratchet", str(path)]) == 1
        assert "the mutation ratchet moved the wrong way" in capsys.readouterr().err

    def test_a_vacuous_artifact_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--check-ratchet", str(tmp_path / "gone.json")]) == 2
        assert "could not be read" in capsys.readouterr().err

    def test_checking_a_ratchet_runs_no_mutant_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mutation_run

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("a ratchet check must never start a campaign")

        monkeypatch.setattr(mutation_run, "run", refuse)

        assert main(["--check-ratchet", str(written(tmp_path, nightly()))]) == 0

    def test_the_round_trip_through_the_artifact_preserves_what_the_gate_judges(
        self, tmp_path: Path
    ) -> None:
        report = nightly(timeouts={"identity": 1})

        restored = read_report(json.loads(written(tmp_path, report).read_text(encoding="utf-8")))

        assert [target.name for target in restored.results] == list(MEASURED_NIGHTLY)
        assert [target.survived for target in restored.results] == list(MEASURED_NIGHTLY.values())
        assert restored.timed_out == 1
        assert restored.verdict is report.verdict

    def test_a_shrunken_budget_cannot_satisfy_a_ceiling_measured_on_a_bigger_sample(
        self, tmp_path: Path
    ) -> None:
        thin = [result(name, executed=2, killed=2, survived=0) for name in MEASURED_NIGHTLY]
        report = RunReport(decide(thin), thin, planned=len(thin))

        verdict, text = gate(written(tmp_path, report))

        assert verdict is Verdict.NOT_RUN
        assert "ran 2 of its mutants against a ceiling measured on 10" in text

    def test_an_exhaustive_run_is_refused_rather_than_failed_for_nothing(
        self, tmp_path: Path
    ) -> None:
        wide = [
            result(
                name,
                executed=52,
                killed=52 - 9,
                survived=9,
                survivors=[mutation(line=index) for index in range(9)],
            )
            for name in MEASURED_NIGHTLY
        ]
        report = RunReport(decide(wide), wide, planned=len(wide))

        verdict, text = gate(written(tmp_path, report))

        assert verdict is Verdict.NOT_RUN
        assert "cannot judge a different sample, in either direction" in text


class TestNightlyWiring:
    """A gate nobody wired is a gate that does not exist, which is the whole of TASK-0382."""

    def mutation_job(self) -> str:
        """The shell every step of the nightly mutation job runs, as one block of text."""
        import yaml

        workflow = yaml.safe_load(
            (REPOSITORY_ROOT / ".github/workflows/nightly.yml").read_text(encoding="utf-8")
        )
        return "\n".join(step.get("run", "") for step in workflow["jobs"]["mutation"]["steps"])

    def test_the_nightly_judges_its_own_artifact_against_the_ceiling(self) -> None:
        assert "--check-ratchet artifacts/mutation.json" in self.mutation_job()

    def test_the_nightly_caps_the_memory_of_every_child_it_spawns(self) -> None:
        assert "--memory-mb" in self.mutation_job()

    def test_the_ratchet_reads_the_artifact_the_campaign_actually_writes(self) -> None:
        shell = self.mutation_job()
        assert "--json-out artifacts/mutation.json" in shell
        assert shell.index("--json-out") < shell.index("--check-ratchet")
