from pathlib import Path

import pytest
from check_layering import (
    EXCEPTIONS,
    LAYERS,
    RANKS,
    REPOSITORY_ROOT,
    SIDEWAYS,
    STALE,
    UNPLACED,
    UNREADABLE,
    UPWARD,
    VACUOUS,
    Finding,
    check,
    layer_of,
    main,
    unexcused,
)

# CRITICAL: every layer the contract names needs a module in a fixture tree, or the vacuity
# guard fires and the test reads a VACUOUS finding instead of the rule it meant to exercise.
SKELETON = {
    "__init__.py": '__version__ = "0"\n',
    "errors.py": "",
    "config/__init__.py": "",
    "observability/__init__.py": "",
    "observability/logging.py": "from fasterrag.errors import X\n",
    "observability/dashboard.py": "",
    "core/__init__.py": "",
    "adapters/__init__.py": "",
    "workers/__init__.py": "",
    "services/__init__.py": "",
    "api/__init__.py": "",
    "cli/__init__.py": "",
    "facade.py": "",
}


def build(root: Path, **modules: str) -> Path:
    """Write a fake package holding one module per layer, plus the ones a test adds."""
    for relative, source in {**SKELETON, **modules}.items():
        path = root / "src" / "fasterrag" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def broken(root: Path, **modules: str) -> list[Finding]:
    """Return the import violations of a fixture tree that excuses nothing."""
    found = check(build(root, **modules), exceptions={})
    return [finding for finding in found if finding.rule in (UPWARD, SIDEWAYS)]


def test_the_repository_breaks_the_contract_only_where_it_says_it_does() -> None:
    """The gate itself, as a unit test: a new upward import fails here before it fails in CI."""
    assert [str(finding) for finding in unexcused(check(REPOSITORY_ROOT))] == []


def test_the_gate_exits_zero_on_the_repository() -> None:
    assert main(["--root", str(REPOSITORY_ROOT)]) == 0


def test_every_layer_has_exactly_one_rank() -> None:
    """A layer missing from RANKS would raise on its first import; one listed twice is ambiguous."""
    ranked = [layer for row in RANKS for layer in row]

    assert sorted(ranked) == sorted(LAYERS)
    assert len(ranked) == len(set(ranked))


def test_every_exception_names_the_task_that_removes_it() -> None:
    """A waiver with no owner is a permission; these are a backlog."""
    assert all("TASK-" in reason for reason in EXCEPTIONS.values())


def test_a_downward_import_is_allowed(tmp_path: Path) -> None:
    assert broken(tmp_path, **{"services/a.py": "from fasterrag.core import thing\n"}) == []


def test_an_upward_import_is_a_violation(tmp_path: Path) -> None:
    """The defect the gate exists for: a worker reaching up into the service that drives it."""
    found = broken(tmp_path, **{"workers/pool.py": "import fasterrag.services.journal\n"})

    assert [finding.rule for finding in found] == [UPWARD]
    assert found[0].path == "src/fasterrag/workers/pool.py"
    assert found[0].line == 1
    assert found[0].key == "fasterrag.workers.pool -> fasterrag.services.journal"


def test_an_import_inside_a_function_is_seen(tmp_path: Path) -> None:
    """facade.py and services/evaluation.py imported api/ this way, below a module-level scan."""
    source = "def build() -> None:\n    from fasterrag.api.dependencies import make\n"

    found = broken(tmp_path, **{"services/evaluation.py": source})

    assert [finding.key for finding in found] == [
        "fasterrag.services.evaluation -> fasterrag.api.dependencies"
    ]
    assert found[0].line == 2


def test_a_type_checking_import_is_seen(tmp_path: Path) -> None:
    """A type-only import still says which layer knows about which."""
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from fasterrag.services.budget import Governor\n"
    )

    assert len(broken(tmp_path, **{"workers/pool.py": source})) == 1


def test_a_relative_import_is_resolved(tmp_path: Path) -> None:
    """`from ..services import x` is the same edge spelled so a prefix match cannot see it."""
    found = broken(tmp_path, **{"workers/pool.py": "from ..services import journal\n"})

    assert [finding.key for finding in found] == ["fasterrag.workers.pool -> fasterrag.services"]


def test_a_relative_import_from_a_package_init_is_resolved(tmp_path: Path) -> None:
    """One dot in an `__init__` is the package itself, not its parent."""
    modules = {"core/__init__.py": "from . import sibling\n", "core/sibling.py": ""}

    assert broken(tmp_path, **modules) == []


def test_importing_a_submodule_by_name_points_at_the_submodule(tmp_path: Path) -> None:
    """`from fasterrag.observability import dashboard` imports the dashboard, not the package."""
    modules = {"api/routes.py": "from fasterrag.observability import dashboard\n"}

    found = broken(tmp_path, **modules)

    assert [(finding.rule, finding.key) for finding in found] == [
        (SIDEWAYS, "fasterrag.api.routes -> fasterrag.observability.dashboard")
    ]


def test_sibling_entry_points_may_not_import_each_other(tmp_path: Path) -> None:
    """The facade importing the REST layer is what makes a library user install a web server."""
    found = broken(tmp_path, **{"facade.py": "from fasterrag.api.dependencies import build\n"})

    assert [finding.rule for finding in found] == [SIDEWAYS]


def test_the_cli_may_import_the_api_it_starts(tmp_path: Path) -> None:
    assert broken(tmp_path, **{"cli/serve.py": "from fasterrag.api.main import app\n"}) == []


def test_reading_the_version_from_the_package_root_is_not_an_edge(tmp_path: Path) -> None:
    """Importing any submodule already runs the root, so naming it adds no dependency."""
    assert broken(tmp_path, **{"errors.py": "from fasterrag import __version__\n"}) == []


def test_the_package_root_imports_as_the_facade(tmp_path: Path) -> None:
    modules = {"__init__.py": "def lazy() -> None:\n    from fasterrag.api import app\n"}

    assert [finding.rule for finding in broken(tmp_path, **modules)] == [SIDEWAYS]


def test_a_registry_string_naming_a_module_is_an_import(tmp_path: Path) -> None:
    """The factories import through strings; a registry entry pointing upward is the same defect."""
    modules = {
        "adapters/factory.py": 'BUILTIN = {"x": "fasterrag.services.rung:serve"}\n',
        "services/rung.py": "",
    }

    found = broken(tmp_path, **modules)

    assert [finding.key for finding in found] == [
        "fasterrag.adapters.factory -> fasterrag.services.rung"
    ]


def test_a_string_that_names_no_module_is_not_an_import(tmp_path: Path) -> None:
    """An entry-point group is spelled like a module path and is not one."""
    assert broken(tmp_path, **{"adapters/factory.py": 'GROUP = "fasterrag.vectordb"\n'}) == []


def test_a_named_exception_excuses_its_import_and_nothing_else(tmp_path: Path) -> None:
    modules = {
        "workers/pool.py": "import fasterrag.services.journal\n",
        "workers/indexer.py": "import fasterrag.services.tenancy\n",
    }
    excused = {"fasterrag.workers.pool -> fasterrag.services.journal": "TASK-0456"}

    found = unexcused(check(build(tmp_path, **modules), exceptions=excused), excused)

    assert [finding.key for finding in found] == [
        "fasterrag.workers.indexer -> fasterrag.services.tenancy"
    ]


def test_an_exception_that_matches_nothing_fails_the_gate(tmp_path: Path) -> None:
    """A fixed edge must take its waiver with it, or the waiver excuses the edge's return."""
    excused = {"fasterrag.workers.pool -> fasterrag.services.journal": "TASK-0456"}

    found = unexcused(check(build(tmp_path), exceptions=excused), excused)

    assert [finding.rule for finding in found] == [STALE]
    assert "fasterrag.workers.pool -> fasterrag.services.journal" in found[0].detail


def test_a_module_outside_every_layer_is_a_violation(tmp_path: Path) -> None:
    """A new top-level package must be placed, or every import of it would pass unchecked."""
    found = check(build(tmp_path, **{"plugins/__init__.py": ""}), exceptions={})

    assert [(finding.rule, finding.path) for finding in found] == [
        (UNPLACED, "src/fasterrag/plugins/__init__.py")
    ]


def test_a_file_that_does_not_parse_is_reported_not_skipped(tmp_path: Path) -> None:
    found = check(build(tmp_path, **{"core/broken.py": "def (:\n"}), exceptions={})

    assert [(finding.rule, finding.path) for finding in found] == [
        (UNREADABLE, "src/fasterrag/core/broken.py")
    ]


def test_an_empty_tree_is_vacuous_not_clean(tmp_path: Path) -> None:
    assert [finding.rule for finding in check(tmp_path, exceptions={})] == [VACUOUS]


def test_a_tree_with_no_internal_import_is_vacuous(tmp_path: Path) -> None:
    """A scan that stops reading imports would otherwise report a perfectly layered tree."""
    found = check(build(tmp_path, **{"observability/logging.py": ""}), exceptions={})

    assert [finding.rule for finding in found] == [VACUOUS]


def test_a_layer_with_no_module_is_vacuous(tmp_path: Path) -> None:
    """Renaming a package must not leave the contract guarding a name nothing answers to."""
    root = build(tmp_path)
    (root / "src" / "fasterrag" / "workers" / "__init__.py").unlink()

    found = check(root, exceptions={})

    assert [finding.rule for finding in found] == [VACUOUS]
    assert "workers" in found[0].detail


@pytest.mark.parametrize(
    ("module", "layer"),
    [
        ("fasterrag", "facade"),
        ("fasterrag.sync", "facade"),
        ("fasterrag.observability.dashboard", "dashboard"),
        ("fasterrag.observability.logging", "observability"),
        ("fasterrag.api.dependencies", "api"),
        ("fasterrag.apiary", None),
    ],
)
def test_a_module_is_placed_by_its_longest_prefix(module: str, layer: str | None) -> None:
    assert layer_of(module) == layer


def test_a_new_violation_fails_the_gate_and_says_how_to_fix_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = build(tmp_path, **{"core/thing.py": "import fasterrag.services.journal\n"})

    assert main(["--root", str(root)]) == 1
    reported = capsys.readouterr().err
    assert "fasterrag.core.thing (core) imports fasterrag.services.journal (services)" in reported
    assert "Do not add an entry to EXCEPTIONS" in reported
