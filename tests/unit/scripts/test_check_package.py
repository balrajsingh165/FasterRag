import re
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


def _requirements_users_install() -> dict[str, list[str]]:
    """Return the core requirements and every extra's, keyed by where they are declared."""
    manifest = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest["project"]
    declared = {"dependencies": list(project["dependencies"])}
    for extra, requirements in project["optional-dependencies"].items():
        if extra != "dev":
            declared[f"[{extra}]"] = list(requirements)
    return declared


def test_every_requirement_a_user_installs_is_capped_below_its_next_major() -> None:
    """TASK-0556: an uncapped httpx let `pip install --pre` pull 1.0.dev6 and break the CLI."""
    uncapped = [
        f"{where}: {requirement}"
        for where, requirements in _requirements_users_install().items()
        for requirement in requirements
        if "<" not in requirement
    ]

    assert uncapped == []


def test_a_requirement_listed_twice_carries_the_same_bounds_everywhere() -> None:
    """The `all` and `dev` copies of an extra's requirement must not drift from the extra's."""
    manifest = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest["project"]
    seen: dict[str, set[str]] = {}
    for requirements in [project["dependencies"], *project["optional-dependencies"].values()]:
        for requirement in requirements:
            name = re.split(r"[\[<>=!~ ]", requirement, maxsplit=1)[0].lower()
            seen.setdefault(name, set()).add(requirement.split(name, 1)[-1].split("]")[-1])

    assert {name: specs for name, specs in seen.items() if len(specs) > 1} == {}


def test_the_local_profile_note_is_several_chunks_and_only_one_answers() -> None:
    """TASK-0542: a one-chunk note proved ingested text returns, not that the right chunk ranks."""
    from local_profile_probe import ANSWER, NOTE

    from fasterrag.chunking import RecursiveChunker
    from fasterrag.parsing import parse_bytes

    chunks = RecursiveChunker(chunk_size=768, overlap=64).split(
        parse_bytes(NOTE.encode("utf-8"), filename="note.md")
    )
    answering = [chunk for chunk in chunks if ANSWER in chunk.text]

    assert len(chunks) >= 3
    assert len(answering) == 1


def test_the_release_smoke_and_the_package_gate_run_the_same_local_profile_probe() -> None:
    """One probe, two callers: the post-publish smoke cannot prove less than the build gate."""
    smoke = (REPOSITORY_ROOT / ".github" / "scripts" / "release_smoke.sh").read_text(
        encoding="utf-8"
    )
    gate = (REPOSITORY_ROOT / "scripts" / "check_package.py").read_text(encoding="utf-8")

    assert "LOCAL_PROFILE_PROBE" in smoke
    assert "local_profile_probe.py" in gate
