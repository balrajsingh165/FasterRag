"""Pins on the trace store's failure reporting (TASK-0317).

Both paths here used to fail in silence: a prune that could delete nothing logged nothing
because its only log line was gated on a non-zero removal count, and a store with exporters
but no running loop dropped every export without a word.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Trace, config_snapshot
from fasterrag.services.traces import TraceStore

LOGGER = "fasterrag.services.traces"


def trace(**overrides: Any) -> Trace:
    """Build a trace with everything the store writes populated."""
    defaults: dict[str, Any] = {
        "trace_id": "t_1",
        "query": "what is the notice period?",
        "collection": "docs",
        "config_snapshot": config_snapshot(Settings()),
        "retrieved": [],
        "prompt": "<context>...</context>",
        "response": "thirty days",
        "result": {"answer": "thirty days", "citations": []},
        "created_at": "2026-11-08T00:00:00+00:00",
    }
    return Trace(**{**defaults, **overrides})


class RecordingExporter:
    """A trace exporter that remembers what it was handed."""

    def __init__(self) -> None:
        """Start with nothing sent."""
        self.sent: list[str] = []

    async def export(self, trace: Trace) -> bool:
        """Record the trace and report it accepted."""
        self.sent.append(trace.trace_id)
        return True

    async def close(self) -> None:
        """Hold nothing open."""
        return


def expired_store(root: Path, *, count: int = 2) -> TraceStore:
    """Return a store holding ``count`` traces already past the retention window."""
    store = TraceStore(root, retention_days=1)
    for index in range(count):
        store.store(trace(trace_id=f"t_{index}"))
    stale = time.time() - 30 * 86400
    for path in root.glob("*.json"):
        os.utime(path, (stale, stale))
    return store


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return the warning records captured so far."""
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def test_a_prune_that_removed_nothing_because_everything_failed_still_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The case the old code could not report: retention wholly broken, removal count zero."""
    store = expired_store(tmp_path)

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("access is denied")

    monkeypatch.setattr(Path, "unlink", refuse)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        removed = store.prune()

    assert removed == 0
    reported = warnings_in(caplog)
    assert len(reported) == 1
    assert "retention window is not being enforced" in reported[0].getMessage()
    assert getattr(reported[0], "failed", None) == 2
    assert getattr(reported[0], "removed", None) == 0
    assert "access is denied" in getattr(reported[0], "first_failure", "")


def test_a_prune_that_removed_some_still_reports_the_ones_it_could_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A partial failure is invisible in the return value, which is only a success count."""
    store = expired_store(tmp_path, count=4)
    real = Path.unlink
    refused: list[str] = []

    def sometimes(self: Path, missing_ok: bool = False) -> None:
        if len(refused) < 2:
            refused.append(self.name)
            raise OSError("the volume is read-only")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", sometimes)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        removed = store.prune()

    assert removed == 2
    reported = warnings_in(caplog)
    assert len(reported) == 1
    assert getattr(reported[0], "failed", None) == 2
    assert getattr(reported[0], "removed", None) == 2


def test_a_prune_with_nothing_to_report_stays_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The other polarity: a working retention sweep must not cry wolf."""
    store = expired_store(tmp_path)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        removed = store.prune()

    assert removed == 2
    assert warnings_in(caplog) == []


def test_dropping_an_export_for_want_of_a_loop_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A CLI one-shot exports nothing; it may not do that without saying so once."""
    exporter = RecordingExporter()
    store = TraceStore(tmp_path, exporters=[exporter])

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        store.store(trace(trace_id="t_cli"))

    assert exporter.sent == []
    reported = warnings_in(caplog)
    assert len(reported) == 1
    assert "every configured trace exporter received nothing" in reported[0].getMessage()
    assert getattr(reported[0], "trace_id", None) == "t_cli"
    assert getattr(reported[0], "exporters", None) == ["RecordingExporter"]


def test_the_dropped_exports_after_the_first_are_counted_not_repeated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One warning per process, then a running count: a per-trace warning would be noise."""
    store = TraceStore(tmp_path, exporters=[RecordingExporter()])

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for index in range(3):
            store.store(trace(trace_id=f"t_{index}"))

    assert len(warnings_in(caplog)) == 1
    counted = [
        getattr(record, "unexported", None) for record in caplog.records if record.name == LOGGER
    ]
    assert counted == [1, 2, 3]


async def test_a_store_with_a_running_loop_exports_without_complaint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The polarity that proves the warning is about the loop and not about exporting."""
    exporter = RecordingExporter()
    store = TraceStore(tmp_path, exporters=[exporter])

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        store.store(trace(trace_id="t_server"))
        await store.drain()

    assert exporter.sent == ["t_server"]
    assert warnings_in(caplog) == []
