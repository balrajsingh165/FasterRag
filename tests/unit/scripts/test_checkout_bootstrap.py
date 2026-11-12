"""A gate script reads the checkout it lives in, not the virtualenv's editable install.

Run from a linked worktree, ``python scripts/check_openapi_drift.py`` imported ``fasterrag``
from the shared venv, whose editable install points at the main checkout, and reported green
on a worktree whose new route was undocumented (TASK-0522).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import _checkout

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
ONLY_HERE = "/v1/only-in-the-second-checkout"

FAKE_PACKAGE = {
    "__init__.py": '"""A second checkout that serves one route no other tree does."""\n',
    "api/__init__.py": "",
    "api/main.py": (
        "class _App:\n"
        "    def openapi(self):\n"
        f'        return {{"paths": {{"{ONLY_HERE}": {{"get": {{}}}}}}}}\n\n\n'
        "def create_app(settings):\n"
        "    return _App()\n"
    ),
    "config/__init__.py": "",
    "config/schema.py": (
        "class Settings:\n"
        "    @classmethod\n"
        "    def model_validate(cls, payload):\n"
        "        return cls()\n"
    ),
}


def second_checkout(root: Path, reference: str) -> Path:
    """Lay out a checkout holding the real gate script and a fake ``fasterrag``."""
    (root / "scripts").mkdir(parents=True)
    for script in ("_checkout.py", "check_openapi_drift.py"):
        shutil.copy2(SCRIPTS / script, root / "scripts" / script)
    for relative, body in FAKE_PACKAGE.items():
        target = root / "src" / "fasterrag" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "api-reference.md").write_text(reference, encoding="utf-8")
    return root


def run_gate(checkout: Path, elsewhere: Path) -> subprocess.CompletedProcess[str]:
    """Run the copied gate as CI does, from an unrelated directory, with no ``PYTHONPATH``."""
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(checkout / "scripts" / "check_openapi_drift.py")],
        cwd=elsewhere,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_gate_run_from_a_second_checkout_reads_that_checkouts_code(tmp_path: Path) -> None:
    """The gate imported the installed fasterrag, so it judged a tree it was not run from."""
    checkout = second_checkout(tmp_path / "second", reference="# API\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = run_gate(checkout, elsewhere)

    assert result.returncode == 1, result.stderr
    assert result.stderr.count("served but undocumented") == 1, result.stderr
    assert f"GET {ONLY_HERE}" in result.stderr


def test_the_second_checkouts_own_reference_makes_it_green(tmp_path: Path) -> None:
    """The documents and the code a gate compares come from the same checkout."""
    checkout = second_checkout(tmp_path / "second", reference=f"| `GET {ONLY_HERE}` | probe |\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = run_gate(checkout, elsewhere)

    assert result.returncode == 0, result.stderr


def test_putting_the_checkout_first_twice_leaves_one_entry_in_front() -> None:
    """Re-running the bootstrap must not grow sys.path or let another entry overtake it."""
    _checkout.put_this_checkout_first()
    _checkout.put_this_checkout_first()

    assert sys.path[0] == str(_checkout.SOURCE_ROOT)
    assert sys.path.count(str(_checkout.SOURCE_ROOT)) == 1
