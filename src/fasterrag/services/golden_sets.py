"""Building a golden set from a corpus on disk (P4).

The generator in ``core/evals`` turns candidate chunks into evaluated queries. This is the
service that finds the candidates: it parses and chunks the corpus exactly as ingestion
would, then hands the result to the generator.

Chunked here rather than read back from the vector database on purpose. A golden set is
usually wanted *before* a collection exists — that is the point of tuning against one — and
the adapter contract has no scroll operation to read chunks back with anyway.

Both the eval harness and Autopilot (D6) consume what this produces, so the same machinery
is reachable from Python and from the terminal instead of only from Python.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.config.schema import Settings
from fasterrag.core.evals.generator import CandidateChunk, generate_golden_set
from fasterrag.core.evals.golden import GoldenRecord, write_golden_set
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk

__all__ = ["candidates_from_sources", "generate_from_sources"]

_logger = get_logger(__name__)


def candidates_from_sources(
    sources: Sequence[str], settings: Settings
) -> tuple[list[CandidateChunk], int]:
    """Parse and chunk a corpus into generator candidates.

    Returns:
        The candidate chunks, and how many sources could not be read. Unreadable sources are
        counted rather than raised: a golden set built from ninety-nine of a hundred
        documents is useful, and failing the whole run over one corrupt file is not.
    """
    candidates: list[CandidateChunk] = []
    unreadable = 0

    for task in CpuWorkerPool.tasks_for(list(sources)):
        try:
            outcome = parse_and_chunk(task, settings)
        except FasterRagError as exc:
            unreadable += 1
            _logger.info(
                "source skipped while building a golden set",
                extra={"source": task.source, "code": exc.code.value},
            )
            continue

        candidates.extend(
            CandidateChunk(
                chunk_id=payload.chunk_id,
                document_id=payload.document_id,
                text=payload.chunk.text,
                metadata={"source": payload.source},
            )
            for payload in outcome.chunks
        )

    return candidates, unreadable


async def generate_from_sources(
    sources: Sequence[str],
    settings: Settings,
    *,
    destination: Path,
    size: int,
    seed: int = 0,
) -> tuple[list[GoldenRecord], dict[str, int]]:
    """Generate a golden set from a corpus and write it to ``destination``.

    Returns:
        The records and the generator's tally of what happened.

    Raises:
        FasterRagError: If the corpus yields no readable chunk, which would otherwise
            produce an empty golden set that every later eval would score against silently.
    """
    candidates, unreadable = candidates_from_sources(sources, settings)
    if not candidates:
        raise FasterRagError(
            f"no readable chunks were produced from {len(sources)} source(s); "
            f"{unreadable} could not be parsed, so there is nothing to generate questions from"
        )

    generator = create_llm_adapter(settings)
    try:
        records, tally = await generate_golden_set(candidates, generator, size=size, seed=seed)
    finally:
        await generator.close()

    write_golden_set(destination, records)
    _logger.info(
        "golden set written",
        extra={
            "path": str(destination),
            "records": len(records),
            "unreadable_sources": unreadable,
            **tally,
        },
    )
    return records, {**tally, "unreadable_sources": unreadable}
