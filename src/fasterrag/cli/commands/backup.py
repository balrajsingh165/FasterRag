"""The ``backup`` and ``restore`` commands.

The tooling the restore drill of ``docs/disaster-recovery.md`` §2 exercises. Both print what
they actually did rather than a success banner: a drill's whole value is discovering that a
step needed improvisation, and a command that reports only "done" hides exactly that.

``restore`` refuses a directory without a manifest. During an incident is the worst possible
moment to be guessing whether a folder is a complete backup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.commands.pipeline import _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.backup import back_up, restore

__all__ = ["run_backup", "run_restore"]


async def run_backup(args: argparse.Namespace, console: Console) -> ExitCode:
    """Capture every documented artifact into a backup directory."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    adapter = create_vector_db_adapter(settings)
    try:
        manifest = await back_up(
            Path(args.destination),
            adapter,
            config_path=Path(args.config),
            collections=args.collections or None,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await adapter.close()

    console.emit(f"destination     {args.destination}")
    for name, snapshot in manifest.collections.items():
        console.emit(f"  {name}: {snapshot} ({manifest.vector_counts.get(name, 0)} vectors)")
    console.emit(f"artifacts       {', '.join(manifest.artifacts) or 'none on disk yet'}")
    console.emit(f"config          {manifest.config or 'not copied'}")
    console.emit("secrets         .env excluded by design; restore it from your secret store")
    console.document(manifest.as_dict())
    return ExitCode.SUCCESS


async def run_restore(args: argparse.Namespace, console: Console) -> ExitCode:
    """Restore a deployment from a backup directory."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    adapter = create_vector_db_adapter(settings)
    try:
        report = await restore(
            Path(args.source),
            adapter,
            collections=args.collections or None,
            restore_files=not args.collections_only,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await adapter.close()

    console.emit(f"from backup     {report['created_at']}")
    console.emit(f"collections     {', '.join(report['collections']) or 'none'}")
    console.emit(f"artifacts       {', '.join(report['artifacts']) or 'none'}")

    if report["mismatches"]:
        console.error("the restore completed with mismatches:")
        for mismatch in report["mismatches"]:
            console.error(f"  {mismatch}")
        console.document(report)
        # CRITICAL: a mismatch exits non-zero. A restore that reports success while holding
        # a different number of vectors than it backed up is how a broken recovery gets
        # signed off, and the drill exists precisely to catch that.
        return ExitCode.FAILURE

    console.emit("verified        restored vector counts match the manifest")
    console.document(report)
    return ExitCode.SUCCESS
