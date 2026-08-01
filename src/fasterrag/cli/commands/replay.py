"""The ``replay`` command (D8).

Re-executes a stored query under a candidate configuration and prints the diff. The value is
in what it *does not* print: an unchanged retrieval set reports "identical" rather than a
wall of chunk ids, so the interesting case stands out.

The candidate config is loaded and validated before anything runs. Replaying under a config
that turns out to be invalid halfway through would leave the operator unsure whether the
difference they are looking at came from the change or from the failure.
"""

from __future__ import annotations

import argparse

from fasterrag.adapters.embeddings.tiering import create_embedding_router
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.commands.pipeline import _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.config.loader import load_settings
from fasterrag.core.rerank import CrossEncoderReranker
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.generation import GenerationService
from fasterrag.services.querying import RetrievalService
from fasterrag.services.replay import ReplayResult, replay_trace
from fasterrag.services.traces import create_trace_store

__all__ = ["run_replay"]


def _print(result: ReplayResult, console: Console, diff_only: bool) -> None:
    """Render a replay result for a person."""
    console.emit(f"trace           {result.trace_id}")
    console.emit(f"query           {result.query}")

    if result.config_changes:
        console.emit(f"\nconfig changes ({len(result.config_changes)})")
        for change in result.config_changes:
            console.emit(f"  {change['key']}: {change['was']} -> {change['now']}")
    else:
        console.emit("\nconfig          unchanged")

    retrieval = result.retrieval
    if retrieval.identical:
        console.emit("retrieval       identical")
    else:
        console.emit("\nretrieval")
        for chunk_id in retrieval.added:
            console.emit(f"  + {chunk_id}")
        for chunk_id in retrieval.removed:
            console.emit(f"  - {chunk_id}")
        for move in retrieval.reordered:
            console.emit(f"  ~ {move['chunk_id']}: rank {move['was']} -> {move['now']}")

    if not result.answer_changed:
        console.emit("answer          unchanged")
        return

    console.emit("\nanswer changed")
    if not diff_only:
        console.emit(f"  was: {result.original_answer}")
        console.emit(f"  now: {result.replayed_answer}")
    console.emit(f"  citations was: {', '.join(result.original_citations) or 'none'}")
    console.emit(f"  citations now: {', '.join(result.replayed_citations) or 'none'}")


async def run_replay(args: argparse.Namespace, console: Console) -> ExitCode:
    """Re-execute a past query under a candidate configuration."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    if not settings.traces.replay:
        console.error("replay is disabled; set traces.replay to true to enable it")
        return ExitCode.USAGE

    candidate = settings
    if args.candidate:
        try:
            candidate = load_settings(args.candidate)
        except ConfigError as exc:
            console.problem(exc.code.value, exc.detail)
            return ExitCode.USAGE

    trace = create_trace_store(settings).load(args.trace)
    if trace is None:
        console.error(
            f"no stored trace {args.trace!r}; it may have expired, or traces.store may be false"
        )
        return ExitCode.FAILURE

    adapter = create_vector_db_adapter(candidate)
    router = create_embedding_router(candidate)
    reranker = CrossEncoderReranker(candidate) if candidate.retrieval.rerank else None
    service = GenerationService(
        candidate,
        RetrievalService(candidate, adapter, router, reranker),
        create_llm_adapter(candidate),
        embedder=router.default,
    )

    try:
        result = await replay_trace(trace, candidate, service)
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()
        await router.close()
        await adapter.close()

    _print(result, console, args.diff_only)
    console.document(result.as_dict())
    return ExitCode.SUCCESS
