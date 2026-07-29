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
| Rerank stage cost | within the documented 100–300 ms band | TBD-until-measured |
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
7. **Regression tracking.** Nightly benchmark runs append to the ledger; a >10% p95 regression or any eval-gate breach files a bug in [todo.md](todo.md) automatically.

## 4. Harness

- `fasterrag benchmark --suite ingest|query|eval|all --dataset <name> --ledger` orchestrates the suites and emits ready-to-commit ledger entries.
- Load generation for concurrency scenarios uses k6 or Locust scripts committed under the benchmark suite ([testing-strategy.md](testing-strategy.md) §1.7).
- The reference-hardware baseline run happens at build-phase slice S11; until then this document contains **no measured numbers by design**.
