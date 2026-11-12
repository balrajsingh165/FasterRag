"""A test's git calls must act on the directory the test is in, never on this repository.

Not a property of the code under test — a property of the test run, which is why it sits at
the root of ``tests/`` beside the other checks on the run itself.

git exports ``GIT_DIR`` and its siblings to a hook, and the pre-commit hook runs the unit tier.
In a linked worktree ``GIT_DIR`` names the real repository. Tests that build a throwaway
repository in ``tmp_path`` inherited it, so ``git init`` and ``git remote add`` ran against the
real one: five tests failed in every worktree, and ``git init`` set ``core.bare`` in the shared
configuration, leaving the main checkout unusable until it was reset by hand (2026-11-12).
``tests/conftest.py`` now removes those variables at import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import GIT_HOOK_VARIABLES, REPO_ROOT

_CHECK = (
    "import os, tests.conftest as c; "
    "raise SystemExit(int(any(n in os.environ for n in c.GIT_HOOK_VARIABLES)))"
)


def test_the_variables_a_hook_exports_are_removed_before_any_test_runs() -> None:
    """A process started as a hook would be must come out with none of them set."""
    hooked = {name: str(REPO_ROOT / "not-a-git-dir") for name in GIT_HOOK_VARIABLES}
    completed = subprocess.run(
        [sys.executable, "-c", _CHECK],
        cwd=REPO_ROOT,
        env={**os.environ, **hooked},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_git_init_in_a_temporary_directory_creates_the_repository_there(tmp_path: Path) -> None:
    """The defect itself: with GIT_DIR inherited, this initialised the real repository."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=60)

    assert (tmp_path / ".git").is_dir()
    assert not any(name in os.environ for name in GIT_HOOK_VARIABLES)
