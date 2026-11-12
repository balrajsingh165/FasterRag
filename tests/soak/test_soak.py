"""The soak run: sustained ingestion, sampled resources, a trend test on the series.

``pytest -m soak`` runs this. The duration is ``FASTERRAG_SOAK_SECONDS`` and nothing else
changes between the two-minute run that proved the harness and the 24 h run the
documentation specifies, which is the point: the decision rule, the sampler and the
workload are the same code at both lengths, so the long run is a booking of hardware
rather than a piece of engineering.

**What it asserts**, in the order the assertions have to hold:

1. The run did work. A soak that ingested nothing has a perfectly flat RSS series and
   would pass every leak test ever written, so the cycle count and the number of points
   accepted are asserted before any series is looked at.
2. The sampler sampled. Enough readings for the trend test to mean something, and readings
   that are not all identical.
3. No memory growth — the RSS series shows no trend distinguishable from a reordering of
   itself that also accounts for more than a tenth of the process's own baseline.
4. No handle growth — the same test with a budget of zero, because handles do not scale
   with the machine and any real upward trend is a descriptor nobody closed.

**What it does not assert: any time.** There is no latency threshold here and there will
not be one. TASK-0350 measured this project's hosted runner at 2.07x variance on an
identical commit, so a wall-clock bound would be a measurement of the runner's neighbours.
Throughput is recorded in the report and graded by nobody.

**What a short run cannot do** is in the report rather than in a docstring that would go
stale: every verdict prints the smallest growth that run had the power to detect, so "no
growth" is always read next to "down to what".
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import psutil
import pytest

from tests.soak.trend import (
    HANDLE_BUDGET,
    MINIMUM_SAMPLES,
    RSS_BUDGET_FRACTION,
    Trend,
    assess,
)
from tests.soak.workload import (
    VERDICTS,
    Sample,
    build_service,
    duration_s,
    interval_s,
    sample,
    write_corpus,
)

pytestmark = [pytest.mark.soak, pytest.mark.timeout(0)]

MINIMUM_CYCLES = 3


async def _soak(root: Path) -> list[Sample]:
    """Ingest the corpus repeatedly for the configured duration, sampling as it goes."""
    sources = write_corpus(root / "corpus")
    service = build_service(root / "journal")
    process = psutil.Process()

    started = time.monotonic()
    deadline = started + duration_s()
    gap = interval_s()
    samples = [sample(process, started=started, cycles=0)]
    cycles = 0
    next_sample = started + gap

    while time.monotonic() < deadline:
        record = await service.accept(sources)
        settled = await service.run(record)
        assert settled.status in {"completed", "partial"}, settled.status
        cycles += 1
        now = time.monotonic()
        if now >= next_sample:
            samples.append(sample(process, started=started, cycles=cycles))
            next_sample = now + gap

    samples.append(sample(process, started=started, cycles=cycles))
    return samples


def _report(path: Path, samples: list[Sample], trends: list[Trend]) -> None:
    """Write the run's series and verdicts, so a failure is diagnosable after the fact."""
    path.write_text(
        json.dumps(
            {
                "duration_s": samples[-1].at_s,
                "cycles": samples[-1].cycles,
                "samples": [
                    {
                        "at_s": item.at_s,
                        "rss_bytes": item.rss_bytes,
                        "handles": item.handles,
                        "cycles": item.cycles,
                    }
                    for item in samples
                ],
                "trends": [trend.summary() for trend in trends],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_a_sustained_ingest_run_grows_neither_memory_nor_handles(tmp_path: Path) -> None:
    """The soak layer: run, sample, regress, and refuse a run that measured nothing."""
    samples = asyncio.run(_soak(tmp_path))

    assert samples[-1].cycles >= MINIMUM_CYCLES, (
        f"the run completed {samples[-1].cycles} ingestion cycles in {samples[-1].at_s:.1f}s; "
        "a soak that did no work proves nothing about leaks"
    )
    assert len(samples) >= MINIMUM_SAMPLES, (
        f"{len(samples)} samples is too few to regress on; raise FASTERRAG_SOAK_SECONDS "
        "or lower FASTERRAG_SOAK_SAMPLE_INTERVAL_S"
    )
    assert len({item.rss_bytes for item in samples}) > 1, (
        "every RSS reading was identical, which means the sampler is not reading this process"
    )

    times = [item.at_s for item in samples]
    baseline_rss = min(item.rss_bytes for item in samples)
    memory = assess(
        "rss_bytes",
        times,
        [item.rss_bytes for item in samples],
        budget=baseline_rss * RSS_BUDGET_FRACTION,
    )
    handles = assess(
        "open_handles",
        times,
        [item.handles for item in samples],
        budget=HANDLE_BUDGET,
    )
    _report(tmp_path / "soak-report.json", samples, [memory, handles])

    VERDICTS.extend((memory.summary(), handles.summary()))
    assert not memory.growing, memory.summary()
    assert not handles.growing, handles.summary()
