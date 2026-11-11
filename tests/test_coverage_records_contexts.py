"""Per-test coverage contexts must actually record which test entered a line.

Not a property of the code under test — a property of the measuring instrument, which is
why it sits at the root of ``tests/`` rather than under ``unit/``.

``--cov-context=test`` asks pytest-cov to label every recorded line with the test that
executed it. Under coverage's default ``sysmon`` core on CPython 3.12+ that labelling is
silently incomplete: ``sys.monitoring`` disables a code object's line events once a line has
been seen, so only the *first* context to enter a line keeps credit for it. When that first
visitor is module import or a session fixture, the context is the empty string and a query
for "which tests entered this line" returns **zero** while the truth is many. Coverage says
so only as a ``no-sysmon-context`` warning in a run that otherwise exits clean and green.

Two passes of the TASK-0335 vacuity sweep reported false "0 tests" results from exactly
this, which is the vacuity defect class arriving inside the instrument used to find it.
``[tool.coverage.run] core`` in ``pyproject.toml`` pins a core that records contexts; these
tests prove the pinned value does, and that the trap it exists for is still real.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
PYPROJECT: Final = REPO_ROOT / "pyproject.toml"

TRAPPED_CORE: Final = "sysmon"
PROBE_TIMEOUT_SECONDS: Final = 120

_SUBJECT: Final = """def shared() -> int:
    return 1
"""

_PROBE: Final = """
import sys

import coverage
import subject

data_file = sys.argv[1]
cov = coverage.Coverage(data_file=data_file, branch=True, config_file=False)
cov.start()
for name in ("alpha", "beta"):
    cov.switch_context(name)
    subject.shared()
cov.stop()
cov.save()

loaded = coverage.Coverage(data_file=data_file, config_file=False)
loaded.load()
data = loaded.get_data()
measured = [path for path in data.measured_files() if path.endswith("subject.py")]
source = open("subject.py", encoding="utf-8").read().splitlines()
body = next(i for i, line in enumerate(source, 1) if "return 1" in line)
crediting = []
for context in sorted(data.measured_contexts()):
    data.set_query_context(context)
    if body in (data.lines(measured[0]) or ()):
        crediting.append(context)
print("CREDITING", ",".join(crediting))
"""


def committed_core() -> str | None:
    """Return the coverage core pinned in ``pyproject.toml``, or ``None`` if unpinned."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    run = config.get("tool", {}).get("coverage", {}).get("run", {})
    pinned = run.get("core")
    return str(pinned) if pinned is not None else None


def contexts_crediting_a_shared_line(tmp_path: Path, core: str) -> list[str]:
    """Run two labelled contexts through one line and return those the core credits."""
    (tmp_path / "subject.py").write_text(_SUBJECT, encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(probe), str(tmp_path / "probe.coverage")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        env={**_probe_env(), "COVERAGE_CORE": core},
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"the context probe failed under core={core!r}:\n{completed.stdout}\n{completed.stderr}"
        )
    reported = [line for line in completed.stdout.splitlines() if line.startswith("CREDITING")]
    assert reported, (
        f"the probe printed no result under core={core!r}, so this test measured nothing:\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    payload = reported[-1].removeprefix("CREDITING").strip()
    return [name for name in payload.split(",") if name]


def _probe_env() -> dict[str, str]:
    """Return the parent environment with the outer run's coverage settings removed."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"COVERAGE_FILE", "COVERAGE_PROCESS_START", "COVERAGE_CORE"}
    }


def test_the_coverage_config_pins_a_core_rather_than_taking_the_default() -> None:
    """An unpinned core is the default ``sysmon``, under which contexts under-report."""
    pinned = committed_core()

    assert pinned is not None, (
        "[tool.coverage.run] in pyproject.toml sets no 'core', so coverage takes its "
        f"default ({TRAPPED_CORE!r}) and any --cov-context run silently under-reports. "
        "See docs/testing-strategy.md."
    )
    assert pinned != TRAPPED_CORE, (
        f"[tool.coverage.run] core is pinned to {pinned!r}, which does not record dynamic "
        "contexts; a --cov-context run would report cleanly while measuring almost nothing."
    )


def test_the_pinned_core_credits_every_context_that_entered_a_line(tmp_path: Path) -> None:
    """The pin is only worth having if the core it names really labels every visit."""
    pinned = committed_core()
    assert pinned is not None

    crediting = contexts_crediting_a_shared_line(tmp_path, pinned)

    assert crediting == ["alpha", "beta"], (
        f"core={pinned!r} credited {crediting} for a line both contexts executed. "
        "Per-test coverage contexts measured with this core cannot be trusted."
    )


def test_the_default_core_still_loses_contexts_so_the_pin_is_still_needed(
    tmp_path: Path,
) -> None:
    """The pin's justification, reported rather than enforced.

    Whether the default core loses contexts depends on the interpreter and the coverage
    build, not on this repository, so a tree where it behaves must not fail. It skips
    with the reason instead, which is what asks for the re-measure.
    """
    crediting = contexts_crediting_a_shared_line(tmp_path, TRAPPED_CORE)

    if crediting == ["alpha", "beta"]:
        pytest.skip(
            f"core={TRAPPED_CORE!r} credits both contexts on Python "
            f"{platform.python_version()}, so the trap the pyproject pin exists for does "
            "not reproduce here. Whether it reproduces depends on the interpreter and the "
            "coverage build rather than on this repository, so the pin stays until it is "
            "measured absent on every supported interpreter (TASK-0398)."
        )

    assert crediting != ["alpha", "beta"]
