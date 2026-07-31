"""The terminal half of the control plane.

One of the two ways fasterRag is operated, the other being the REST API
(``docs/cli-reference.md``). Both call the same service layer; there is deliberately no third
way, and no GUI that controls anything — the dashboard observes only.
"""

from __future__ import annotations

from fasterrag.cli.main import main
from fasterrag.cli.output import Console, ExitCode

__all__ = ["Console", "ExitCode", "main"]
