"""Print each soak verdict in the terminal summary, so a passing run still reports.

A soak run's numbers are the point of running it, and a green pytest line says nothing
about how much growth the run could have ruled out. The repository's own provenance
header uses this hook (``tests/conftest.py``), so the numbers go out the same way rather
than through a ``print`` the lint rules correctly forbid.
"""

from __future__ import annotations

import pytest

from tests.soak.workload import VERDICTS


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Write every verdict the run produced, pass or fail."""
    for line in VERDICTS:
        terminalreporter.write_line(line)
