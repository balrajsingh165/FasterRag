"""The ``benchmark`` command.

Runs the suites of ``docs/performance.md`` and, with ``--ledger``, emits an entry ready to
paste into ``docs/benchmarks.md``. The entry is generated rather than hand-written because
the ledger rules require seven fields on every entry, and the reliable way to satisfy that is
to make omitting one impossible.

The suites measure a *running deployment*, not a synthetic loop: the query suite issues real
queries through the real service against the configured backend. A benchmark that measured a
mock would produce numbers no operator could ever reproduce.
"""

from __future__ import annotations

import argparse

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.commands.pipeline import _build_generation, _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.benchmark import (
    REPETITIONS,
    SuiteResult,
    fingerprint,
    ledger_entry,
    measure,
)
from fasterrag.services.estimation import estimate_sources

__all__ = ["run_benchmark"]

_DEFAULT_QUERY = "What does this corpus say about the subject it covers?"


async def _query_suite(
    args: argparse.Namespace, console: Console
) -> tuple[SuiteResult | None, ExitCode]:
    """Measure end-to-end query latency against the configured backend."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return None, ExitCode.USAGE

    adapter = create_vector_db_adapter(settings)
    service = _build_generation(settings, adapter)
    # CRITICAL: the suite issues the same question repeatedly, so with the semantic cache
    # live every repetition after the first would be a cache hit — the run would report
    # cache latency under the name "query latency". docs/performance.md is explicit that
    # cache hits are never mixed into non-cached percentiles.
    service.cache = None
    question = args.query or _DEFAULT_QUERY

    async def once() -> None:
        await service.answer(question, collection=args.collection)

    try:
        cold, runs = await measure(once, iterations=args.iterations)
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return None, ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await service.close()
        await adapter.close()

    return (
        SuiteResult(
            suite="query",
            dataset=args.dataset or (args.collection or settings.vector_db.collection.default_name),
            cold=cold,
            runs=runs,
            notes=(
                "end-to-end POST /v1/query equivalent through the service layer, including "
                "retrieval, assembly, and generation. The semantic cache is disabled for the "
                "run: the suite repeats one question, so a live cache would report hit "
                "latency as query latency. These are non-cached, and include provider "
                "round-trip time, which dominates and is not fasterRag's to control"
            ),
        ),
        ExitCode.SUCCESS,
    )


async def _ingest_suite(
    args: argparse.Namespace, console: Console
) -> tuple[SuiteResult | None, ExitCode]:
    """Measure parse-and-chunk throughput, which is what the estimator already does."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return None, ExitCode.USAGE

    if not args.sources:
        console.error("the ingest suite needs --sources naming a corpus to measure against")
        return None, ExitCode.USAGE

    async def once() -> None:
        estimate_sources(args.sources, settings)

    # CRITICAL: the iteration count is used as given. An earlier version quietly divided it
    # because an ingest iteration is expensive, which produced a single-sample run whose p50,
    # p95, and p99 were the same number — a percentile that looks measured and is not.
    cold, runs = await measure(once, iterations=args.iterations)

    estimate = estimate_sources(args.sources, settings)
    return (
        SuiteResult(
            suite="ingest",
            dataset=args.dataset or f"{estimate.documents} documents, {estimate.tokens} tokens",
            cold=cold,
            runs=runs,
            notes=(
                "parse and chunk only — embedding and indexing are excluded, because their "
                "cost belongs to the provider and the backend rather than to fasterRag, and "
                "mixing them in would measure a network round trip"
            ),
        ),
        ExitCode.SUCCESS,
    )


async def run_benchmark(args: argparse.Namespace, console: Console) -> ExitCode:
    """Run the requested suite and report, optionally as a ledger entry."""
    suites = ["query", "ingest"] if args.suite == "all" else [args.suite]

    if "eval" in suites:
        # TODO: the eval suite needs the golden-set harness of TASK-0077.
        console.error(
            "the eval suite needs a committed golden set (TASK-0077); "
            "use --suite query or --suite ingest"
        )
        return ExitCode.USAGE

    machine = fingerprint()
    console.emit(f"hardware        {machine.describe()}")
    console.emit(f"repetitions     {REPETITIONS} warmed, plus one cold start")

    results: list[SuiteResult] = []
    for suite in suites:
        console.emit(f"\nrunning         {suite}")
        result, code = (
            await _query_suite(args, console)
            if suite == "query"
            else await _ingest_suite(args, console)
        )
        if result is None:
            return code

        median = result.median
        console.emit(f"  cold start    {result.cold.as_dict()['p50_ms'] if result.cold else 0} ms")
        console.emit(f"  p50           {median.get('p50_ms', 0)} ms")
        console.emit(f"  p95           {median.get('p95_ms', 0)} ms")
        console.emit(f"  p99           {median.get('p99_ms', 0)} ms")
        console.emit(f"  throughput    {median.get('throughput_per_sec', 0)}/s")
        results.append(result)

    if args.ledger:
        console.emit("\n--- ledger entries, ready to paste into docs/benchmarks.md ---\n")
        for index, result in enumerate(results, start=1):
            console.emit(
                ledger_entry(
                    f"BENCH-{index:04d}",
                    f"{result.suite} suite measured on the hardware recorded below",
                    result,
                    machine=machine,
                )
            )
            console.emit("")

    console.document(
        {"hardware": machine.as_dict(), "suites": [result.as_dict() for result in results]}
    )
    return ExitCode.SUCCESS
