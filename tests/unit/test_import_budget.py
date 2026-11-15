"""What ``from fasterrag import FasterRag`` costs, held to a budget (TASK-0488).

The package root is lazy (``tests/unit/test_facade.py``) and a core install carries no extra
(``tests/unit/test_slim_core.py``), but nothing stopped the facade itself from growing: one
eager import of a model library or a vendor client in a module the facade reaches would add
seconds to every CLI command and every ``import`` in a user's application.

Two budgets, measured in a child interpreter so this process's imports cannot hide a cost.
The **module budget** runs in the unit tier: which packages are loaded is deterministic, so
it can gate every push. The **time budget** carries the ``load`` marker and runs in the
weekly workflow, because wall time on a shared runner moves 2x between identical runs
(TASK-0350); its ceiling is a tripwire for an order-of-magnitude regression, not a target,
and no figure here is a measurement anyone may cite (``docs/benchmarks.md``).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

import fasterrag

SOURCE_TREE: Final = Path(fasterrag.__file__).resolve().parents[1]

MODULE_BUDGET: Final = 900
"""Modules ``from fasterrag import FasterRag`` may load, the interpreter's own excluded.

656 on 2026-11-15 (Python 3.14, the development venv). The headroom absorbs dependency and
interpreter versions; an eager model library or vendor SDK costs far more than it.
"""

SECONDS_BUDGET: Final = 5.0
"""Median wall time the facade import may take in a fresh interpreter.

About 1.4 s on 2026-11-15 on a loaded development workstation, and not citable. An eager
``torch`` or ``transformers`` import alone costs several seconds, which is what this catches.
"""

HEAVY: Final = (
    "torch",
    "transformers",
    "sentence_transformers",
    "numpy",
    "qdrant_client",
    "chromadb",
    "pymilvus",
    "weaviate",
    "psycopg",
    "openai",
    "anthropic",
    "cohere",
    "ollama",
    "fastapi",
    "uvicorn",
    "langfuse",
)

PROBE: Final = """
import json, sys, time
before = set(sys.modules)
started = time.perf_counter()
from fasterrag import FasterRag
elapsed = time.perf_counter() - started
loaded = sorted(set(sys.modules) - before)
print(json.dumps({"seconds": elapsed, "modules": loaded}))
"""


def probe() -> tuple[float, list[str]]:
    """Import the facade in a fresh interpreter; return the seconds and the modules it loaded."""
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        cwd=SOURCE_TREE,
        timeout=120,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return float(payload["seconds"]), list(payload["modules"])


def test_the_facade_import_stays_within_its_module_budget() -> None:
    """A module the facade reaches importing a heavy library eagerly had no gate."""
    _, modules = probe()

    assert len(modules) <= MODULE_BUDGET, f"{len(modules)} modules"
    packages = {name.partition(".")[0] for name in modules}
    assert sorted(packages & set(HEAVY)) == []


@pytest.mark.load
def test_the_facade_import_stays_within_its_time_budget() -> None:
    """The median of five fresh imports stays under the ceiling a heavy eager import breaks."""
    seconds = statistics.median(probe()[0] for _ in range(5))

    assert seconds <= SECONDS_BUDGET, f"median facade import took {seconds:.2f} s"
