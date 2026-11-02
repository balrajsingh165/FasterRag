# performance.md — Performance Goals & Measurement Methodology

> Provable-claims policy applies: **a claim without a measurement is a bug.** Everything in this file is a **goal** or a **method** until the [benchmark ledger](benchmarks.md) holds a measured entry for it. SLO targets derive from these measurements only after the baseline run on documented reference hardware ([slo.md](slo.md)).

## 1. What we measure

| Metric | Definition | Measured by |
|---|---|---|
| End-to-end latency p50/p95 | `POST /v1/query` receipt → final token (non-cached, non-degraded) | query benchmark suite; histogram `fasterrag_request_duration_seconds` |
| Per-stage latency | embed / retrieve(dense) / retrieve(bm25) / fuse / rerank / assemble / generate, p50/p95 — the **retrieval vs generation** split | `fasterrag_stage_duration_seconds`; per-query `timings_ms` |
| Time-to-first-token (TTFT) | request receipt → first SSE `token` event | `fasterrag_ttft_seconds` |
| Ingestion throughput | docs/sec and tokens/sec sustained over ≥ 10 min | ingest benchmark suite; `fasterrag_ingest_throughput` |
| Retrieval quality | recall@k, MRR, nDCG on named eval datasets | eval harness (`pytest -m eval`, `fasterrag benchmark --suite eval`) |
| Faithfulness | grounding score distribution of generated answers | eval harness |
| Cache hit rate | semantic + embedding cache hits / lookups | `fasterrag_cache_events_total` |
| Cost per query | Σ provider token costs / queries | `fasterrag_cost_usd_total` |

## 2. Beta targets (goals — TBD-until-measured)

| Metric | Target | Status |
|---|---|---|
| Retrieval-only p50 (10M chunks, hybrid, no rerank) | ≤ 150 ms | TBD-until-measured |
| End-to-end p95 excl. LLM generation (with rerank) | ≤ 600 ms | TBD-until-measured |
| Rerank stage cost | no target set — the "100–300 ms band" this row used to cite was never measured and has been withdrawn | TBD-until-measured |
| Ingestion throughput (reference rig) | ≥ 50 docs/sec; ≥ 200k tokens/sec embedded | TBD-until-measured |
| Top-20 retrieval failure rate (hybrid + contextual + rerank) | ≤ 2% | TBD-until-measured (literature anchor: 1.9% in Anthropic's Contextual Retrieval post) |
| Semantic-cache hit response | ≤ 50 ms | TBD-until-measured |
| Soak (24 h) | zero memory/fd growth trend | TBD-until-measured |

Every row flips from goal → claim only by landing a ledger entry with numbers, hardware, dataset, date, and commit hash.

## 3. Methodology (the rules every measurement follows)

1. **Hardware is part of the number.** Every run records the exact spec: CPU model + cores, RAM, GPU model + VRAM, storage class, OS; recorded automatically by `fasterrag benchmark --ledger`.
2. **Datasets are named and versioned.** Eval/benchmark datasets live under `tests/eval/datasets/` with a manifest (doc count, token count, source, license). No number without its dataset name.
3. **Warm/cold discipline.** Latency suites report cold-start and warmed (caches primed, model loaded) separately; cache-hit metrics are never mixed into non-cached latency percentiles.
4. **Percentiles over averages.** p50/p95 (p99 where sample size permits); minimum 1,000 samples per latency figure.
5. **Isolation.** Benchmarks run with no co-tenant load; background jobs disabled; three repetitions, median run reported, all runs attached.
6. **Baselines are measured, not quoted.** "Fastest" claims require a named competitor/baseline measured by us with the same harness, dataset, and hardware, harness committed to this repo.
7. **Regression tracking.** A nightly workflow (`.github/workflows/nightly.yml`) runs the eval harness and the benchmark suite, fetches the previous night's artifact, compares the two with `scripts/compare_nightly.py`, and uploads ledger-format artifacts — tonight's numbers plus the comparison — with 30-day retention. What it compares, exactly: the warmed median `p50_ms`, `p95_ms`, `p99_ms` (lower is better) and `throughput_per_sec` (higher is better) of every suite the two runs share. A metric that moved more than **10%** in the worse direction fails the job; an improvement never does.
   - **Three outcomes, not two.** *Regressed*, *within tolerance*, and *not comparable* — no previous artifact, a different runner, a different dataset, a missing or zero metric. "Not comparable" is surfaced as a job-summary warning and is never reported as a pass, because a gate that goes green having compared nothing is worse than no gate. This mirrors the retrieval regression gate (`services/regression.py`), which refuses a baseline recorded under a different embedding model and blocks rather than passes when no baseline exists.
   - **It does not append to the ledger and does not file a bug.** That is a decision, not an omission. [todo.md](todo.md) is this repository's only task file, and these numbers are not citable (shared runner — see rule 5 above), so a job writing entries from them would be a machine building a second task list out of measurements the methodology forbids citing. A regression fails the nightly job, which is already a notification; a human decides whether it earns a task or a run on real hardware.
   - **Verification status.** `scripts/compare_nightly.py` is covered by unit tests in `tests/unit/scripts/test_compare_nightly.py`, including the not-comparable cases. The workflow wiring around it — the previous-run lookup via the Actions API and the cross-run artifact download — has not been observed on a real scheduled run; the first nightly after it lands is its first evidence. Treat the three outcomes as specified-and-tested and the fetch as untested until a run shows otherwise.

## 4. Harness

- `fasterrag benchmark --suite ingest|query|eval|all --dataset <name> --ledger` orchestrates the suites and emits ready-to-commit ledger entries. **Built.**
- `scripts/compare_nightly.py --previous <a.json> --current <b.json> [--tolerance 0.10]` compares two `fasterrag benchmark --json` documents and exits `0` within tolerance, `1` regressed, `2` not comparable. **Built**, and usable by hand on any two artifacts, not only by the nightly job. It detects a *change* between two runs; it produces no ledger entry and makes neither run citable.
- Load generation for concurrency scenarios is specified to use k6 or Locust scripts committed under the benchmark suite ([testing-strategy.md](testing-strategy.md) §1.7). **Not built — no such script exists in this repository** (TASK-0243), so no concurrency scenario has ever been run.
- The reference-hardware baseline run happens at build-phase slice S11; until then this document contains **no measured numbers by design**.
