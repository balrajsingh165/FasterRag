"""The suite must test the tree it is run from.

Not a property of the code under test — a property of the test run itself, which is why it
sits at the root of ``tests/`` rather than under ``unit/``.

The editable install writes an absolute path into
``site-packages/_editable_impl_fasterrag.pth``, so a bare ``import fasterrag`` resolves to
whichever checkout ran ``pip install -e .``, whatever the working directory. Every agent
worktree in this repo therefore ran its own tests against *main's* source until
``pythonpath = ["src", ...]`` was added: new tests for an unlanded fix failed confusingly,
and — the dangerous direction — a change whose behaviour main already had could be committed
green without its own code ever executing. The pre-commit hook runs pytest, so the gate
agreed with the mistake.

A green suite is only evidence about the code it imported.
"""

from pathlib import Path

import fasterrag

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_imported_package_is_the_one_in_this_checkout() -> None:
    """``import fasterrag`` must resolve inside this repository, not another checkout."""
    imported = Path(fasterrag.__file__).resolve()
    expected = REPO_ROOT / "src" / "fasterrag" / "__init__.py"

    assert imported == expected, (
        f"the suite imported {imported}, but this checkout is {REPO_ROOT}. "
        "Tests are running against a different tree's source, so passing proves nothing "
        "about the code here."
    )
