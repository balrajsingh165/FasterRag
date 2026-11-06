import ast
import importlib.util
import json
import py_compile
from pathlib import Path

import pytest
from mutation_run import (
    ARITHMETIC_SWAPS,
    COMPARISON_SWAPS,
    DEFAULT_TARGETS,
    EXIT_CODES,
    REPOSITORY_ROOT,
    Mutation,
    RunReport,
    Target,
    TargetResult,
    Verdict,
    apply,
    decide,
    generate,
    load_targets,
    main,
    markdown,
    render,
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
    problems: list[str] | None = None,
) -> TargetResult:
    """Build a TargetResult for the verdict and rendering tests."""
    return TargetResult(
        name=name,
        generated=executed,
        executed=executed,
        killed=killed,
        survived=survived,
        survivors=survivors or [],
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
