import sys
import tomllib
import venv
from pathlib import Path

import pytest
from check_package import (
    EXTRA_ONLY_MODULES,
    PUBLIC_IMPORTS,
    REPOSITORY_ROOT,
    _check_cli,
    _check_core_stays_small,
    _check_imports,
    _check_named_extras,
    _python_in,
    _run,
    clean_environment,
)


def test_the_venv_interpreter_is_found_where_this_platform_puts_it(tmp_path: Path) -> None:
    """TASK-0263's trap lives in _python_in; only a real venv proves the layout."""
    environment = tmp_path / "venv"
    venv.create(environment, with_pip=False)

    assert _python_in(environment).is_file()


def test_a_broken_documented_import_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing statement must surface, quoted, or the gate reports nothing it finds."""
    monkeypatch.setattr("check_package.PUBLIC_IMPORTS", ("import fasterrag_not_a_module",))

    problems = _check_imports(Path(sys.executable))

    assert len(problems) == 1
    assert "import fasterrag_not_a_module" in problems[0]


def test_a_working_documented_import_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the polarity: an importable statement must not be flagged."""
    monkeypatch.setattr("check_package.PUBLIC_IMPORTS", ("import sys; assert sys.executable",))

    assert _check_imports(Path(sys.executable)) == []


def test_a_present_extra_only_module_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inverting this comparison would pass a bloated install and fail a clean one."""
    monkeypatch.setattr("check_package.EXTRA_ONLY_MODULES", ("sys",))

    problems = _check_core_stays_small(Path(sys.executable))

    assert len(problems) == 1
    assert "'sys'" in problems[0]


def test_an_absent_extra_only_module_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A core install that lacks the heavy stack is the passing case, not a problem."""
    monkeypatch.setattr("check_package.EXTRA_ONLY_MODULES", ("fasterrag_not_a_module",))

    assert _check_core_stays_small(Path(sys.executable)) == []


def test_a_missing_console_script_is_reported(tmp_path: Path) -> None:
    """An empty environment ships no entry point, and the gate must say which path it tried."""
    interpreter = tmp_path / "bin" / "python"
    interpreter.parent.mkdir(parents=True)

    problems = _check_cli(interpreter)

    assert len(problems) == 1
    assert "console script" in problems[0]


def test_every_documented_import_names_the_package() -> None:
    """A manifest entry that never touches fasterrag would pass while proving nothing."""
    assert PUBLIC_IMPORTS
    assert all("fasterrag" in statement for statement in PUBLIC_IMPORTS)


def test_the_extra_only_modules_are_absent_from_the_core_dependencies() -> None:
    """Promoting one of these into core deps turns the absence check into a certain failure."""
    manifest = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = " ".join(manifest["project"]["dependencies"]).replace("-", "_").lower()

    assert [module for module in EXTRA_ONLY_MODULES if module in core] == []


def test_a_failing_command_is_captured_as_text_rather_than_raised() -> None:
    """TASK-0317: the contract the removed `# type: ignore` was papering over."""
    completed = _run([sys.executable, "-c", "import sys; sys.stderr.write('broken'); sys.exit(3)"])

    assert completed.returncode == 3
    assert completed.stderr == "broken"
    assert completed.stdout == ""


def test_the_clean_environment_drops_the_import_path_and_every_credential() -> None:
    """Run as `PYTHONPATH=src`, the venv imported the checkout instead of the wheel."""
    cleaned = clean_environment(
        {
            "PATH": "/bin",
            "PYTHONPATH": "src",
            "OPENAI_API_KEY": "sk-secret",
            "PGVECTOR_DSN": "postgresql://user:pw@host/db",
            "FASTERRAG_CONFIG": "/etc/fasterrag.yaml",
        }
    )

    assert cleaned == {"PATH": "/bin"}


def test_a_missing_extra_that_is_not_named_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "check_package.NAMED_EXTRAS", (("a thing", "print('no hint here')", "fasterrag[x]"),)
    )

    problems = _check_named_extras(Path(sys.executable))

    assert len(problems) == 1
    assert "fasterrag[x]" in problems[0]


def test_a_missing_extra_that_is_named_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "check_package.NAMED_EXTRAS",
        (("a thing", "print('pip install \"fasterrag[x]\"')", 'pip install "fasterrag[x]"'),),
    )

    assert _check_named_extras(Path(sys.executable)) == []


def test_every_extra_the_slim_core_moved_out_exists_and_is_in_all() -> None:
    """The error messages name these extras; one missing from pyproject is an install that fails."""
    manifest = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = manifest["project"]["optional-dependencies"]

    for name in ("server", "qdrant", "pdf", "docx", "html", "chroma", "huggingface"):
        assert extras[name], name
        assert set(extras[name]) <= set(extras["all"]), name
