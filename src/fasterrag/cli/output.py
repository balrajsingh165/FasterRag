"""CLI output and exit codes.

The exit-code table of ``docs/cli-reference.md``. Codes are the CLI's real contract: the
text a command prints is for a person, but the code is what a shell script, a CI job, and a
Makefile branch on. A command that returns 1 for everything is unscriptable, so every failure
mode maps to a distinct code.

``--json`` prints one JSON document to stdout and nothing else, which is why every human
message goes through ``emit`` rather than ``print``. A stray progress line on stdout turns a
parseable document into a parse error at the far end of a pipe.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from enum import IntEnum
from typing import Any

__all__ = ["Console", "ExitCode"]


class ExitCode(IntEnum):
    """Process exit codes, per ``docs/cli-reference.md``."""

    SUCCESS = 0
    FAILURE = 1
    USAGE = 2
    UNREACHABLE = 3
    PREFLIGHT = 4
    REGRESSION = 5


class Console:
    """Writes command output in the mode the global flags selected."""

    def __init__(self, *, as_json: bool = False, quiet: bool = False) -> None:
        """Build a console.

        Args:
            as_json: Emit machine-readable JSON instead of human text.
            quiet: Suppress everything but errors.
        """
        self.as_json = as_json
        self.quiet = quiet

    def emit(self, message: str) -> None:
        """Write one human-readable line, unless the mode suppresses it.

        Silent under ``--json`` as well as ``--quiet``: mixing prose into a stream a caller
        is piping to a JSON parser breaks it, and the JSON document says the same thing.
        """
        if self.quiet or self.as_json:
            return
        sys.stdout.write(f"{message}\n")

    def lines(self, messages: Iterable[str]) -> None:
        """Write several human-readable lines."""
        for message in messages:
            self.emit(message)

    def document(self, payload: Mapping[str, Any]) -> None:
        """Write the machine-readable result, if the mode asked for one."""
        if self.as_json:
            sys.stdout.write(f"{json.dumps(payload, indent=2, default=str)}\n")

    def error(self, message: str) -> None:
        """Write an error to stderr.

        Errors go to stderr in every mode, including ``--quiet`` and ``--json``: silencing
        the reason for a non-zero exit leaves an operator with a bare code and no cause.
        """
        sys.stderr.write(f"error: {message}\n")

    def problem(self, code: str, detail: str, fix: str | None = None) -> None:
        """Write a typed failure, matching the error taxonomy's shape."""
        self.error(f"[{code}] {detail}")
        if fix:
            sys.stderr.write(f"  fix: {fix}\n")
