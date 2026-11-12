"""The entry-point lookup configuration validates provider names against (TASK-0439)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fasterrag.config import plugins
from fasterrag.config.plugins import registered_names
from tests.support.plugins import install_distribution

GROUP = "fasterrag.test_only_group"


def imported_modules(path: Path) -> set[str]:
    """Return every module a source file imports, read as code."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_a_group_nothing_registers_under_is_empty() -> None:
    assert registered_names(GROUP) == frozenset()


def test_an_installed_distribution_is_found_by_the_name_it_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection starts here: a name configuration cannot see is a name it must refuse."""
    install_distribution(tmp_path, monkeypatch, {GROUP: {"acme": "acme_pkg.adapter:Adapter"}})

    assert registered_names(GROUP) == frozenset({"acme"})


def test_only_the_asked_group_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A vector database plugin must not become a selectable LLM by sharing a distribution."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {GROUP: {"acme": "acme_pkg:A"}, f"{GROUP}.other": {"globex": "acme_pkg:B"}},
    )

    assert registered_names(GROUP) == frozenset({"acme"})
    assert registered_names(f"{GROUP}.other") == frozenset({"globex"})


def test_a_name_is_reported_without_importing_what_it_points_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation must not run third-party code; a broken plugin fails where it is built."""
    install_distribution(tmp_path, monkeypatch, {GROUP: {"broken": "no_such_module_anywhere:X"}})

    assert registered_names(GROUP) == frozenset({"broken"})


def test_the_lookup_imports_nothing_from_this_package() -> None:
    """``config/`` validating through ``adapters/`` would close an import cycle."""
    imported = imported_modules(Path(plugins.__file__))

    assert "importlib.metadata" in imported
    assert not {name for name in imported if name.startswith("fasterrag")}
