"""The ``autopilot run`` command (D6).

Prints every trial's measurement, not just the winner. A tuner that reported only its
conclusion would be asking for trust; reporting the whole search lets a reader see that the
second-best was close, or that the grid barely moved the numbers at all — which is itself
the most useful finding a tuning run can produce.

The suggestion is written to a file *beside* ``config.yaml``, never into it. The command
verifies that afterwards rather than merely intending it, because "never writes your config"
is the promise D6 is built on and an unchecked promise is a hope.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fasterrag.adapters.embeddings.tiering import create_embedding_router
from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.commands.pipeline import _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.autopilot import SUGGESTION_FILE, render_suggestion, tune
from fasterrag.services.evaluation import load_dataset
from fasterrag.services.golden_sets import generate_from_sources

__all__ = ["run_autopilot", "run_generate_golden_set"]


async def run_autopilot(args: argparse.Namespace, console: Console) -> ExitCode:
    """Search query-time configurations and write a suggested diff."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    if not args.dataset:
        console.error(
            "autopilot needs --dataset naming a directory with a golden set; generate one "
            "from your corpus first, or point at tests/eval/datasets/handbook to see it work"
        )
        return ExitCode.USAGE

    config_path = Path(args.config)
    fingerprint = config_path.read_bytes() if config_path.is_file() else None

    collection = args.collection or settings.vector_db.collection.default_name
    adapter = create_vector_db_adapter(settings)
    router = create_embedding_router(settings)

    try:
        dataset = load_dataset(Path(args.dataset))
        suggestion = await tune(
            dataset,
            settings,
            adapter,
            router,
            collection=collection,
            budget_seconds=args.budget_minutes * 60,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await router.close()
        await adapter.close()

    console.emit(f"collection      {collection}")
    console.emit(f"evaluated       {suggestion.evaluated} candidates in {suggestion.seconds:.1f}s")
    if suggestion.skipped:
        console.emit(f"skipped         {suggestion.skipped} at the budget")

    console.emit("\ntrials (ndcg / recall / mrr):")
    for trial in sorted(suggestion.trials, key=lambda item: -item.ndcg_at_k):
        marker = "*" if trial is suggestion.best else " "
        console.emit(
            f" {marker} {trial.ndcg_at_k:.4f} / {trial.recall_at_k:.4f} / {trial.mrr:.4f}"
            f"   {trial.candidate.label}"
        )

    output = Path(args.out or SUGGESTION_FILE)
    output.write_text(render_suggestion(suggestion), encoding="utf-8")

    if suggestion.improves:
        deltas = suggestion.deltas
        console.emit(
            f"\nsuggested       {suggestion.best.candidate.label}"
            f"\n                ndcg {deltas['ndcg_at_k']:+.4f}, "
            f"recall {deltas['recall_at_k']:+.4f}, mrr {deltas['mrr']:+.4f}"
        )
    else:
        console.emit("\nsuggested       nothing — no candidate beat the current configuration")

    console.emit(f"written         {output} (NOT applied; review and apply it yourself)")

    # CRITICAL: the promise is checked, not assumed. D6 is built on "never auto-applies", and
    # a regression that started writing config.yaml would otherwise be invisible until it had
    # already overwritten somebody's production settings.
    if fingerprint is not None and config_path.read_bytes() != fingerprint:
        console.error(f"{config_path} changed during the run; autopilot must never write it")
        return ExitCode.FAILURE

    console.document({**suggestion.as_dict(), "suggestion_file": str(output)})
    return ExitCode.SUCCESS


async def run_generate_golden_set(args: argparse.Namespace, console: Console) -> ExitCode:
    """Generate a golden Q&A set from a corpus and write it to disk (P4).

    The shared machinery behind both the eval harness and Autopilot's search. It was
    reachable from Python but not from the terminal, which meant the one prerequisite for
    running D6 had no supported way to produce it.

    ``--size`` falls back to ``autopilot.golden_set_size`` rather than to a second hardcoded
    number. The two happened to agree at 100, which is why the configured value being
    ignored looked like nothing at all until somebody changed it.
    """
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    size = args.size if args.size is not None else settings.autopilot.golden_set_size

    destination = Path(args.out)
    if destination.exists():
        console.error(
            f"{destination} already exists; a golden set is hand-curated after generation, "
            "so it is never overwritten. Delete it or pass a different --out"
        )
        return ExitCode.USAGE

    try:
        records, tally = await generate_from_sources(
            args.sources,
            settings,
            destination=destination,
            size=size,
            seed=args.seed,
        )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        return ExitCode.FAILURE

    console.emit(f"wrote {destination}")
    console.emit(f"records         {len(records)}")
    for name, count in sorted(tally.items()):
        console.emit(f"  {name:<14}{count}")
    console.emit("review the questions before using them as a baseline; they are generated")
    console.document({"path": str(destination), "records": len(records), "size": size, **tally})
    return ExitCode.SUCCESS
