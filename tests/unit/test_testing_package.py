"""The adapter contract suite ships in the wheel as ``fasterrag.testing`` (TASK-0551).

It lived only in ``tests/contract/``, which is in the sdist but not the wheel, so a plugin
author had to run it from a fasterRag checkout. It now installs with the package; pytest
comes from the ``testing`` extra, and the slim core never imports it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import fasterrag.testing.vectordb as installed_suite
import tests.contract.vectordb as in_repo_suite

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_the_in_repo_contract_module_is_the_installed_suite() -> None:
    """``tests/contract/`` re-exports, so in-repo adapters run exactly what plugins run."""
    assert in_repo_suite.VectorDBContract is installed_suite.VectorDBContract
    assert in_repo_suite.DIMENSIONS == installed_suite.DIMENSIONS
    assert in_repo_suite.point is installed_suite.point


def test_the_suite_lives_in_the_package_the_wheel_is_built_from() -> None:
    """The wheel packages ``src/fasterrag``; a suite outside it is not installable."""
    package = Path(installed_suite.__file__).resolve()

    assert package.is_relative_to(REPOSITORY_ROOT / "src" / "fasterrag")


def test_the_testing_extra_brings_pytest_and_its_asyncio_plugin() -> None:
    """The suite's cases are coroutines; pytest alone would not run them."""
    manifest = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    testing = manifest["project"]["optional-dependencies"]["testing"]

    assert {requirement.split(">")[0] for requirement in testing} == {"pytest", "pytest-asyncio"}
    assert "pytest" not in " ".join(manifest["project"]["dependencies"])


def test_without_pytest_the_import_names_the_extra_to_install() -> None:
    """A plugin author without pytest gets the install line, not a bare ModuleNotFoundError."""
    script = (
        "import sys\n"
        "sys.modules['pytest'] = None\n"
        "try:\n"
        "    import fasterrag.testing\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    ran = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPOSITORY_ROOT,
        env={**_environment(), "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )

    assert 'pip install "fasterrag[testing]"' in ran.stdout


def test_the_core_package_does_not_import_the_testing_package() -> None:
    """Importing ``fasterrag`` and its facade must not need pytest."""
    script = (
        "import sys\n"
        "sys.modules['pytest'] = None\n"
        "import fasterrag, fasterrag.facade, fasterrag.parsing, fasterrag.chunking\n"
        "import fasterrag.adapters.vectordb.factory\n"
        "assert 'fasterrag.testing' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=REPOSITORY_ROOT,
        env={**_environment(), "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )


def _environment() -> dict[str, str]:
    """Return the current environment without an inherited import path."""
    return {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
