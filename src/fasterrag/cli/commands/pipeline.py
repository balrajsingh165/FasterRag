"""The commands that move data: ``ingest``, ``query``, and ``index``.

Each is a thin front for the service the REST API calls, so ``fasterrag query`` and
``POST /v1/query`` cannot diverge — the CLI parses flags and renders results, and nothing
else. Any retrieval or generation logic that appeared here would be a second implementation
to keep in step with the first.

``--dry-run`` on ``ingest`` is the exception that proves the rule: it calls the estimator
rather than the ingestion service, because reporting what *would* be indexed without
embedding anything is exactly what the estimator already does.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.embeddings.tiering import create_embedding_router
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.adapters.vectordb.base import CollectionSpec, Distance, VectorDBAdapter
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.output import Console, ExitCode
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.core.rerank import CrossEncoderReranker
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.services.estimation import estimate_sources
from fasterrag.services.evaluation import run_eval
from fasterrag.services.generation import GenerationService
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import create_journal
from fasterrag.services.lockfile import create_lock_store, detect_drift
from fasterrag.services.querying import RetrievalService
from fasterrag.services.regression import GateResult
from fasterrag.services.reindex import plan_reindex, retire, rollback, swap
from fasterrag.services.traces import create_trace_store

__all__ = ["run_index", "run_ingest", "run_query"]

_DISTANCES: frozenset[str] = frozenset({"cosine", "dot", "euclid"})


def _settings_or_none(args: argparse.Namespace, console: Console) -> Settings | None:
    """Load configuration, reporting an invalid file rather than raising."""
    try:
        return load_settings(args.config)
    except ConfigError as exc:
        console.problem(exc.code.value, exc.detail)
        return None


def _pairs(values: Sequence[str], console: Console) -> dict[str, str] | None:
    """Parse repeated ``KEY=VALUE`` flags, or report the one that is malformed."""
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, rest = value.partition("=")
        if not separator or not key:
            console.error(f"expected KEY=VALUE, got {value!r}")
            return None
        parsed[key] = rest
    return parsed


async def run_ingest(args: argparse.Namespace, console: Console) -> ExitCode:
    """Ingest sources, or report what ingesting them would involve under ``--dry-run``."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    metadata = _pairs(args.metadata, console)
    if metadata is None:
        return ExitCode.USAGE

    if args.dry_run:
        estimate = estimate_sources(args.sources, settings)
        console.emit(f"would index {estimate.chunks} chunks from {estimate.documents} documents")
        console.emit(f"tokens          {estimate.tokens}")
        console.emit(f"unreadable      {estimate.unreadable}")
        console.document({"dry_run": True, **estimate.as_dict()})
        return ExitCode.SUCCESS

    service = IngestionService(
        settings, journal=create_journal(settings), locks=create_lock_store(settings)
    )
    try:
        record = await service.ingest(
            args.sources,
            collection=args.collection,
            metadata=metadata or None,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()

    counts = record.counts
    console.emit(f"job             {record.job_id}")
    console.emit(f"status          {record.status}")
    console.emit(f"indexed         {counts.get('indexed', 0)} chunks")
    console.emit(f"dead-lettered   {counts.get('dead_lettered', 0)}")
    console.document({"job_id": record.job_id, "status": record.status, "counts": dict(counts)})

    return ExitCode.SUCCESS if record.status != "failed" else ExitCode.FAILURE


def _build_generation(settings: Settings, adapter: VectorDBAdapter) -> GenerationService:
    """Assemble the query path exactly as the API assembles it."""
    router = create_embedding_router(settings)
    reranker = CrossEncoderReranker(settings) if settings.retrieval.rerank else None
    retrieval = RetrievalService(settings, adapter, router, reranker)
    cache = SemanticCache(settings, create_semantic_store(settings))
    return GenerationService(
        settings,
        retrieval,
        create_llm_adapter(settings),
        cache=cache,
        traces=create_trace_store(settings),
        embedder=router.default,
    )


async def run_query(args: argparse.Namespace, console: Console) -> ExitCode:
    """Answer a question, streaming tokens unless ``--no-stream`` was given."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    filters = _pairs(args.filter, console)
    if filters is None:
        return ExitCode.USAGE

    adapter = create_vector_db_adapter(settings)
    service = _build_generation(settings, adapter)

    try:
        answer = await service.answer(
            args.question,
            collection=args.collection,
            top_k=args.top_k,
            filters=filters or None,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()
        await adapter.close()

    if answer.insufficient_evidence:
        console.emit("INSUFFICIENT EVIDENCE — the answer was withheld (D5)")
        console.emit(f"  faithfulness {answer.faithfulness} < threshold {answer.threshold}")
        for candidate in answer.best_candidates:
            console.emit(f"  candidate {candidate['chunk_id']} ({candidate['source']})")
        console.document(answer.as_dict())
        return ExitCode.SUCCESS

    console.emit(answer.answer or "")
    if answer.citations:
        console.emit("")
        for citation in answer.citations:
            page = f", page {citation.page}" if citation.page is not None else ""
            console.emit(f"  [^{citation.chunk_id}] {citation.source or 'unknown source'}{page}")

    if answer.degraded:
        console.emit(f"\ndegraded: mode={answer.mode}")
    if args.show_timings:
        console.emit(
            "\ntimings: "
            + ", ".join(f"{stage}={value}ms" for stage, value in answer.timings_ms.items())
        )

    console.document(answer.as_dict())
    return ExitCode.SUCCESS


async def _index_list(settings: Settings, adapter: VectorDBAdapter, console: Console) -> ExitCode:
    """Print every collection the backend holds, with its drift status (D1)."""
    collections = await adapter.list_collections()
    if not collections:
        console.emit("no collections")

    locks = create_lock_store(settings)
    journal = create_journal(settings)
    payload: list[dict[str, Any]] = []

    for info in collections:
        lock = locks.read(info.name)
        drift = detect_drift(
            lock,
            settings,
            collection=info.name,
            document_hashes=journal.document_hashes(info.name) or None,
        )
        status = "no lockfile" if drift.missing_lock else ("DRIFT" if drift.detected else "ok")
        console.emit(
            f"{info.name:<24} {info.vectors:>10} vectors  "
            f"dim={info.dimensions}  distance={info.distance}  "
            f"sparse={'yes' if info.sparse else 'no'}  {status}"
        )
        payload.append({**info.as_dict(), "drift": drift.as_dict()})

    console.document({"collections": payload})
    return ExitCode.SUCCESS


async def _index_lock_verify(
    args: argparse.Namespace, settings: Settings, console: Console
) -> ExitCode:
    """Verify a collection against its lockfile, exiting non-zero on drift (D1)."""
    collection = args.name or args.collection or settings.vector_db.collection.default_name
    lock = create_lock_store(settings).read(collection)
    drift = detect_drift(
        lock,
        settings,
        collection=collection,
        document_hashes=create_journal(settings).document_hashes(collection) or None,
    )

    console.document(drift.as_dict())

    if drift.missing_lock:
        console.error(
            f"no lockfile for {collection!r}; ingest into it, or set index.lockfile to false"
        )
        return ExitCode.FAILURE

    if not drift.detected:
        console.emit(f"{collection}: no drift; the index matches its lockfile")
        return ExitCode.SUCCESS

    console.error(f"{collection}: index drift detected")
    for detail in drift.details:
        console.error(
            f"  {detail['field']}: locked {detail['locked']!r} -> live {detail['live']!r}"
        )
    for document in drift.documents_added:
        console.error(f"  document added since the lock: {document}")
    for document in drift.documents_removed:
        console.error(f"  document removed since the lock: {document}")
    for document in drift.documents_changed:
        console.error(f"  document content changed since the lock: {document}")

    # CRITICAL: exit 1, not 0. Verify exists to be a gate in a pipeline, and a drift check
    # that succeeds while reporting drift is a check nothing can branch on.
    return ExitCode.FAILURE


def _distance_for(
    args: argparse.Namespace, settings: Settings, console: Console
) -> Distance | None:
    """Return the distance to create with, or ``None`` if the flag is invalid."""
    if args.distance is None:
        return settings.vector_db.collection.distance
    if args.distance not in _DISTANCES:
        console.error(f"--distance must be one of {', '.join(sorted(_DISTANCES))}")
        return None
    return args.distance  # type: ignore[no-any-return]


async def _dimensions_of(embedder: EmbeddingAdapter) -> int | None:
    """Return the model's vector size, embedding a probe if it is not known yet.

    A local model reports no dimension until its weights are loaded, and creating a
    collection is precisely the moment that number has to be right. One throwaway embedding
    is cheaper than a collection created at the wrong width, which cannot be widened later
    and forces a full re-embed to correct.
    """
    if embedder.dimensions is not None:
        return embedder.dimensions

    return len(await embedder.embed_query("dimension probe")) or None


async def _index_create(
    args: argparse.Namespace, settings: Settings, adapter: VectorDBAdapter, console: Console
) -> ExitCode:
    """Create a collection sized from the configured embedding model."""
    distance = _distance_for(args, settings, console)
    if distance is None:
        return ExitCode.USAGE

    router = create_embedding_router(settings)
    try:
        dimensions = await _dimensions_of(router.default)
        if dimensions is None:
            console.error(
                "the configured embedding model did not report a vector size; "
                "ingest one document instead, which creates the collection automatically"
            )
            return ExitCode.FAILURE

        collection = settings.vector_db.collection
        await adapter.create_collection(
            CollectionSpec(
                name=args.name,
                dimensions=dimensions,
                distance=distance,
                shard_number=args.shards or collection.shard_number,
                replication_factor=args.replicas or collection.replication_factor,
                sparse=settings.retrieval.hybrid,
            )
        )
    finally:
        await router.close()

    console.emit(f"created {args.name} ({dimensions} dimensions, {distance})")
    console.document({"name": args.name, "dimensions": dimensions, "distance": distance})
    return ExitCode.SUCCESS


async def _index_delete(
    args: argparse.Namespace, adapter: VectorDBAdapter, console: Console
) -> ExitCode:
    """Drop a collection, requiring ``--force`` because the data does not come back."""
    if not args.force:
        console.error(f"deleting {args.name!r} destroys its vectors; pass --force to confirm")
        return ExitCode.USAGE

    dropped = await adapter.drop_collection(args.name)
    console.emit(f"{'deleted' if dropped else 'no such collection:'} {args.name}")
    console.document({"name": args.name, "deleted": dropped})
    return ExitCode.SUCCESS


async def run_index(args: argparse.Namespace, console: Console) -> ExitCode:
    """Dispatch an ``index`` subcommand."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    adapter = create_vector_db_adapter(settings)
    try:
        if args.action == "list":
            return await _index_list(settings, adapter, console)
        if args.action == "lock":
            return await _index_lock_verify(args, settings, console)
        if args.action == "reembed":
            return await _index_reembed(args, settings, adapter, console)
        if args.action == "rollback":
            return await _index_rollback(args, adapter, console)
        if args.action == "create":
            return await _index_create(args, settings, adapter, console)
        return await _index_delete(args, adapter, console)
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        console.document({"error": exc.code.value, "detail": exc.detail})
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await adapter.close()


async def _index_reembed(
    args: argparse.Namespace, settings: Settings, adapter: VectorDBAdapter, console: Console
) -> ExitCode:
    """Rebuild an index behind its alias, swapping only if the gate allows it (D2)."""
    plan = await plan_reindex(args.name, settings, adapter)
    console.emit(f"building        {plan.green}")
    console.emit(f"replacing       {plan.blue or '(first build)'}")

    service = IngestionService(
        settings,
        journal=create_journal(settings),
        adapter=adapter,
        locks=create_lock_store(settings),
    )
    try:
        record = await service.ingest(args.sources, collection=plan.green)
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()

    if record.status == "failed":
        console.error(f"the build failed; {plan.blue or 'nothing'} is still live")
        console.document({**plan.as_dict(), "swapped": False, "reason": "the build failed"})
        return ExitCode.FAILURE

    gate = (
        None
        if args.no_eval_gate
        else await _run_eval_gate(args, settings, adapter, plan.green, console)
    )
    result = await swap(
        plan,
        adapter,
        eval_passed=gate.passed if gate else None,
        eval_report=gate.as_dict() if gate else {},
    )

    if not result.swapped:
        # CRITICAL: exit 5, the documented regression-gate code. A blocked swap is the gate
        # working, and a pipeline needs to distinguish it from a build that crashed.
        console.error(result.reason)
        console.document(result.as_dict())
        return ExitCode.REGRESSION

    console.emit(f"swapped         {plan.alias} -> {plan.green} in {result.swap_ms} ms")
    retired = await retire(plan.alias, adapter, settings, locks=create_lock_store(settings))
    for name in retired:
        console.emit(f"retired         {name}")

    console.document({**result.as_dict(), "retired": retired})
    return ExitCode.SUCCESS


async def _run_eval_gate(
    args: argparse.Namespace,
    settings: Settings,
    adapter: VectorDBAdapter,
    green: str,
    console: Console,
) -> GateResult | None:
    """Score the freshly built collection against a dataset, or report why it cannot be.

    Returns ``None`` when no dataset was named, which the swap records as ungated rather
    than as a pass — a gate that did not run has established nothing.
    """
    if not args.dataset:
        console.emit(
            "eval gate       not run: pass --dataset to score the new build before swapping"
        )
        return None

    router = create_embedding_router(settings)
    try:
        report, gate = await run_eval(
            Path(args.dataset), settings, adapter, router, collection=green
        )
    except FasterRagError as exc:
        console.error(f"eval gate       could not run: {exc.detail}")
        return None
    finally:
        await router.close()

    console.emit(
        f"eval gate       recall@{report.k}={report.recall_at_k:.4f} "
        f"mrr={report.mrr:.4f} ndcg@{report.k}={report.ndcg_at_k:.4f}"
    )
    return gate


async def _index_rollback(
    args: argparse.Namespace, adapter: VectorDBAdapter, console: Console
) -> ExitCode:
    """Flip the alias back to a retained build (D2)."""
    result = await rollback(args.name, adapter, to=args.to)

    console.emit(f"restored        {result.alias} -> {result.restored}")
    console.emit(f"replaced        {result.replaced or '(nothing)'}")
    console.document(result.as_dict())
    return ExitCode.SUCCESS
