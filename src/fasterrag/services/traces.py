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

import asyncio
import json
import os
import time
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from fasterrag import __version__
from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Trace
from fasterrag.errors import ConfigError
from fasterrag.observability.langfuse_export import LangfuseExporter
from fasterrag.observability.logging import get_logger
from fasterrag.observability.otel_export import OtelExporter

__all__ = [
    "DEFAULT_LANGFUSE_HOST",
    "DEFAULT_TRACE_ROOT",
    "LANGFUSE_HOST_VAR",
    "LANGFUSE_PUBLIC_KEY_VAR",
    "LANGFUSE_SECRET_KEY_VAR",
    "TraceExporter",
    "TraceStore",
    "create_exporters",
    "create_langfuse_exporter",
    "create_otel_exporter",
    "create_trace_store",
]

# Names only. The values are generated into `.env` by the provisioner and read from the
# environment, never from config.yaml.
LANGFUSE_PUBLIC_KEY_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_VAR = "LANGFUSE_SECRET_KEY"
LANGFUSE_HOST_VAR = "LANGFUSE_HOST"
DEFAULT_LANGFUSE_HOST = "http://localhost:3000"

DEFAULT_TRACE_ROOT: Final = Path(".fasterrag") / "traces"

_SUFFIX: Final = ".json"

# CRITICAL: pruning walks the directory, so doing it on every write would make trace storage
# cost grow with the number of traces retained. Once per this many writes keeps the retention
# window honoured without turning a constant-time append into a linear scan.
_PRUNE_EVERY: Final = 50

_logger = get_logger(__name__)


@runtime_checkable
class TraceExporter(Protocol):
    """Ships a finished trace somewhere outside this process."""

    async def export(self, trace: Trace) -> bool:
        """Send one trace, reporting whether it was accepted. Never raises."""
        ...

    async def close(self) -> None:
        """Release whatever the exporter holds open."""
        ...


class TraceStore:
    """Stores and retrieves query traces."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        enabled: bool = True,
        retention_days: int = 30,
        exporters: Sequence[TraceExporter] = (),
    ) -> None:
        """Build a store.

        Args:
            root: Directory holding trace files. There is no configuration key for the
                path; callers pass the deployment's data directory.
            enabled: From ``traces.store``. When false, nothing is written and every read
                reports the trace as absent, which is the honest answer.
            retention_days: From ``traces.retention_days``.
            exporters: Trace exporters to ship to, if any. Traces are written locally *and*
                shipped; the local copy is what replay and ``GET /v1/traces/{id}`` read, so
                an investigation never depends on the observability stack being healthy.
                A list rather than one slot because Langfuse and OTLP answer different
                questions and a deployment may reasonably want both.
        """
        self.root = root or DEFAULT_TRACE_ROOT
        self.enabled = enabled
        self.retention_days = retention_days
        self.exporters = list(exporters)
        self._writes = 0
        self._exports: set[asyncio.Task[bool]] = set()

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
            # CRITICAL: the cleanup is itself suppressed. On Linux, when the parent is a
            # file rather than a directory, `unlink` raises NotADirectoryError from inside
            # this handler — replacing the error being handled with a second one that
            # escapes, so a degradation path becomes a crash. Windows does not raise here,
            # which is why this only ever failed on the Linux CI leg.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            _logger.warning(
                "could not persist the query trace; the answer was returned regardless",
                extra={"trace_id": trace.trace_id, "error": str(exc)},
            )
            return

        self._writes += 1
        if self._writes % _PRUNE_EVERY == 0:
            self.prune()

        self._export(trace)

    def _export(self, trace: Trace) -> None:
        """Ship a trace to Langfuse without waiting for it.

        Fire-and-forget on purpose: ``store`` is called on the query path after the answer is
        ready, and awaiting a network round-trip here would add the exporter's latency to
        every request that succeeds.

        # CRITICAL: the task is kept in a set until it finishes. asyncio holds only a weak
        # reference to a running task, so one that nothing retains can be garbage-collected
        # mid-flight and the export vanishes with no error anywhere.
        """
        if not self.exporters:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: a CLI one-shot rather than the server. Exporting would need one
            # started and torn down per trace, which costs more than the record is worth.
            return

        for exporter in self.exporters:
            task = loop.create_task(exporter.export(trace))
            self._exports.add(task)
            task.add_done_callback(self._exports.discard)

    async def drain(self) -> None:
        """Wait for in-flight exports, so shutdown does not drop the last traces."""
        if self._exports:
            await asyncio.gather(*tuple(self._exports), return_exceptions=True)
        for exporter in self.exporters:
            await exporter.close()

    def load(self, trace_id: str, *, tenant: str | None = None) -> Trace | None:
        """Return a stored trace, or ``None`` if it is absent or unreadable.

        # CRITICAL: a trace belonging to another tenant is reported as *absent*, not as
        # forbidden. A trace records the query text and every retrieved chunk, so telling a
        # caller "that exists but is not yours" confirms the id is real — and trace ids are
        # the one identifier a caller can guess at by listing their own.
        """
        path = self._path(trace_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        try:
            trace = Trace.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _logger.warning("a stored trace is malformed", extra={"trace_id": trace_id})
            return None

        if tenant is not None and trace.tenant != tenant:
            return None
        return trace

    def recent(self, limit: int = 50, *, tenant: str | None = None) -> list[str]:
        """Return the most recently written trace ids, newest first.

        Filtering happens after sorting rather than by scanning until ``limit`` is reached,
        so a tenant's newest traces are returned even when another tenant wrote more
        recently — the alternative silently hides a tenant behind a noisier one.
        """
        if not self.root.exists():
            return []

        files = sorted(
            self.root.glob(f"*{_SUFFIX}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if tenant is None:
            return [path.stem for path in files[:limit]]

        owned: list[str] = []
        for path in files:
            if len(owned) >= limit:
                break
            trace = self.load(path.stem, tenant=tenant)
            if trace is not None:
                owned.append(path.stem)
        return owned

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


def create_langfuse_exporter(settings: Settings) -> LangfuseExporter | None:
    """Build the Langfuse exporter when the toggle is on and its keys are present.

    The keys are read from the environment by *name*, exactly like every other credential —
    they are generated into ``.env`` by ``fasterrag provision langfuse`` and never appear in
    ``config.yaml``.

    Returns:
        The exporter, or ``None`` when the toggle is off or the keys are absent. A missing key
        is a warning rather than a failure: the toggle also provisions the stack, and refusing
        to serve queries because a dashboard has no credentials inverts the dependency.
    """
    if not settings.observability.langfuse:
        return None

    public = os.environ.get(LANGFUSE_PUBLIC_KEY_VAR, "").strip()
    secret = os.environ.get(LANGFUSE_SECRET_KEY_VAR, "").strip()
    if not (public and secret):
        _logger.warning(
            "observability.langfuse is on but its keys are unset, so no trace will be "
            "exported; run 'fasterrag provision langfuse' to generate them",
            extra={"expected": [LANGFUSE_PUBLIC_KEY_VAR, LANGFUSE_SECRET_KEY_VAR]},
        )
        return None

    return LangfuseExporter(
        os.environ.get(LANGFUSE_HOST_VAR, "").strip() or DEFAULT_LANGFUSE_HOST,
        public,
        secret,
        release=__version__,
    )


def create_otel_exporter(settings: Settings) -> OtelExporter | None:
    """Build the OTLP exporter when ``observability.otel`` is on.

    Returns:
        The exporter, or ``None`` when the toggle is off or the SDK is not installed. A
        missing SDK is a warning rather than a failure, for the same reason a missing
        Langfuse key is: refusing to serve queries because a trace backend is unavailable
        inverts the dependency between the system and the thing watching it.
    """
    if not settings.observability.otel:
        return None

    endpoint = settings.observability.otel_endpoint
    if not endpoint:
        return None

    try:
        return OtelExporter(endpoint)
    except ConfigError as exc:
        _logger.warning(
            "observability.otel is on but the OpenTelemetry SDK is not installed, so no "
            "trace will be exported; install it with 'pip install fasterrag[otel]'",
            extra={"endpoint": endpoint, "detail": exc.detail},
        )
        return None


def create_exporters(settings: Settings) -> list[TraceExporter]:
    """Build every trace exporter the configuration turns on.

    Both may be on at once: Langfuse answers "what did the model see and say", OTLP answers
    "where did the time go across the whole system", and a deployment that wants both should
    not have to choose.
    """
    candidates = (create_langfuse_exporter(settings), create_otel_exporter(settings))
    return [exporter for exporter in candidates if exporter is not None]


def create_trace_store(settings: Settings, root: Path | None = None) -> TraceStore:
    """Build a trace store from validated configuration."""
    return TraceStore(
        root,
        enabled=settings.traces.store,
        retention_days=settings.traces.retention_days,
        exporters=create_exporters(settings),
    )
