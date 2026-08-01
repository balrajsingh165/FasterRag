"""The ``traces`` command: list and inspect stored query traces (D8).

The read-only half of replay. Listing exists because a trace id is a 32-character hex string
that nobody retains between the query and the investigation — without a way to find the
recent ones, every stored trace is unreachable unless the id was copied at the time.
"""

from __future__ import annotations

import argparse

from fasterrag.cli.commands.pipeline import _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.services.traces import create_trace_store

__all__ = ["run_traces"]


async def run_traces(args: argparse.Namespace, console: Console) -> ExitCode:
    """List recent traces, or print one in full."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    store = create_trace_store(settings)

    if args.action == "list":
        recent = store.recent(args.limit)
        if not recent:
            console.emit(
                "no stored traces" + ("" if settings.traces.store else " (traces.store is false)")
            )
        for trace_id in recent:
            console.emit(trace_id)
        console.document({"traces": recent})
        return ExitCode.SUCCESS

    trace = store.load(args.trace_id)
    if trace is None:
        console.error(
            f"no stored trace {args.trace_id!r}; it may have expired past "
            f"traces.retention_days ({settings.traces.retention_days})"
        )
        return ExitCode.FAILURE

    console.emit(f"trace           {trace.trace_id}")
    console.emit(f"query           {trace.query}")
    console.emit(f"collection      {trace.collection}")
    console.emit(f"created         {trace.created_at}")
    console.emit(f"candidates      {len(trace.retrieved)}")
    console.emit("spans")
    for span in trace.spans:
        console.emit(f"  {span.name:<18} {span.duration_ms:8.1f} ms  {span.attributes}")
    console.document(trace.as_dict())
    return ExitCode.SUCCESS
