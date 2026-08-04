"""The commands that move a collection between deployments: ``export`` and ``import`` (D11).

Both are thin: verification and writing live in ``services/archive`` and
``services/archive_import``, because the REST endpoints must behave identically and a rule
enforced in a CLI handler is a rule the API does not have.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fasterrag.adapters.embeddings.tiering import create_embedding_router
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.settings import settings_from
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.archive import export_archive
from fasterrag.services.archive_import import import_archive, open_archive
from fasterrag.services.lockfile import create_lock_store

__all__ = ["run_export", "run_import"]


async def run_export(args: argparse.Namespace, console: Console) -> ExitCode:
    """Write a collection to a portable archive."""
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    collection = args.collection or settings.vector_db.collection.default_name
    store = create_lock_store(settings)
    lock = store.read(collection) if store is not None and store.enabled else None

    adapter = create_vector_db_adapter(settings)
    try:
        counts = await export_archive(
            settings,
            adapter,
            collection=collection,
            destination=Path(args.out),
            include_vectors=args.include_vectors,
            lock=lock,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE
    finally:
        await adapter.close()

    console.emit(f"collection      {collection}")
    console.emit(f"documents       {counts.documents}")
    console.emit(f"chunks          {counts.chunks}")
    console.emit(f"vectors         {counts.vectors}")
    if not args.include_vectors:
        console.emit("note            no vectors; import will need --reembed")
    console.document({"collection": collection, **counts.as_dict()})
    return ExitCode.SUCCESS


async def run_import(args: argparse.Namespace, console: Console) -> ExitCode:
    """Import a previously exported archive.

    Verification happens before anything is written, so a refused archive leaves the target
    untouched rather than half-populated.
    """
    try:
        settings = settings_from(args)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    try:
        reader = open_archive(Path(args.archive))
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.USAGE

    target = args.target_collection or args.collection or reader.collection
    adapter = create_vector_db_adapter(settings)
    router = create_embedding_router(settings) if args.reembed else None

    try:
        counts = await import_archive(
            settings,
            adapter,
            reader,
            collection=target,
            reembed=args.reembed,
            router=router,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE
    finally:
        await adapter.close()
        if router is not None:
            await router.close()

    console.emit(f"collection      {target}")
    console.emit(f"documents       {counts.documents}")
    console.emit(f"chunks          {counts.chunks}")
    console.emit(f"mode            {'re-embedded' if args.reembed else 'vector copy'}")
    console.document({"collection": target, "reembed": args.reembed, **counts.as_dict()})
    return ExitCode.SUCCESS
