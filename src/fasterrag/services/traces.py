"""Local trace persistence (D8).

One file per trace under ``.fasterrag/traces``, written atomically. A trace is written after
the query it describes has already returned, so a failure to store one must never surface to
the caller — an observability record that can fail the request it observes is worse than no
record.

Retention is enforced on write rather than by a background sweeper. There is no scheduler in
the process to hang a sweeper off, and pruning while writing means the window is honoured by
anything that stores traces at all, including a short-lived CLI invocation.

Traces hold the full prompt and response, which is exactly what makes replay possible and
also what makes them worth keeping out of a shared directory. They live beside the ingestion
journal under ``.fasterrag/``, which is gitignored.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Trace
from fasterrag.observability.logging import get_logger

__all__ = ["DEFAULT_TRACE_ROOT", "TraceStore", "create_trace_store"]

DEFAULT_TRACE_ROOT: Final = Path(".fasterrag") / "traces"

_SUFFIX: Final = ".json"

# CRITICAL: pruning walks the directory, so doing it on every write would make trace storage
# cost grow with the number of traces retained. Once per this many writes keeps the retention
# window honoured without turning a constant-time append into a linear scan.
_PRUNE_EVERY: Final = 50

_logger = get_logger(__name__)


class TraceStore:
    """Stores and retrieves query traces."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        enabled: bool = True,
        retention_days: int = 30,
    ) -> None:
        """Build a store.

        Args:
            root: Directory holding trace files. There is no configuration key for the
                path; callers pass the deployment's data directory.
            enabled: From ``traces.store``. When false, nothing is written and every read
                reports the trace as absent, which is the honest answer.
            retention_days: From ``traces.retention_days``.
        """
        self.root = root or DEFAULT_TRACE_ROOT
        self.enabled = enabled
        self.retention_days = retention_days
        self._writes = 0

    def _path(self, trace_id: str) -> Path:
        """Return the file holding a trace."""
        return self.root / f"{trace_id}{_SUFFIX}"

    def store(self, trace: Trace) -> None:
        """Persist a trace, swallowing any storage failure.

        The query has already been answered by the time this runs. Raising here would turn a
        full disk into a failed request, so a write failure is logged and dropped — the
        record is lost, which is the smaller harm.
        """
        if not self.enabled:
            return

        path = self._path(trace.trace_id)
        temporary = path.with_suffix(".tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(trace.as_dict()), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            _logger.warning(
                "could not persist the query trace; the answer was returned regardless",
                extra={"trace_id": trace.trace_id, "error": str(exc)},
            )
            return

        self._writes += 1
        if self._writes % _PRUNE_EVERY == 0:
            self.prune()

    def load(self, trace_id: str) -> Trace | None:
        """Return a stored trace, or ``None`` if it is absent or unreadable."""
        path = self._path(trace_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        try:
            return Trace.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _logger.warning("a stored trace is malformed", extra={"trace_id": trace_id})
            return None

    def recent(self, limit: int = 50) -> list[str]:
        """Return the most recently written trace ids, newest first."""
        if not self.root.exists():
            return []

        files = sorted(
            self.root.glob(f"*{_SUFFIX}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [path.stem for path in files[:limit]]

    def prune(self) -> int:
        """Delete traces older than the retention window, returning how many went."""
        if not self.root.exists():
            return 0

        cutoff = time.time() - self.retention_days * 86400
        removed = 0
        for path in self.root.glob(f"*{_SUFFIX}"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue

        if removed:
            _logger.info(
                "pruned traces past the retention window",
                extra={"removed": removed, "retention_days": self.retention_days},
            )
        return removed


def now() -> str:
    """Return the current UTC timestamp in ISO 8601."""
    return datetime.now(tz=UTC).isoformat()


def create_trace_store(settings: Settings, root: Path | None = None) -> TraceStore:
    """Build a trace store from validated configuration."""
    return TraceStore(
        root,
        enabled=settings.traces.store,
        retention_days=settings.traces.retention_days,
    )
