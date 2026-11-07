# performance.md — Performance Goals & Measurement Methodology

> Provable-claims policy applies: **a claim without a measurement is a bug.** Everything in this file is a **goal** or a **method** until the [benchmark ledger](benchmarks.md) holds a measured entry for it. SLO targets derive from these measurements only after the baseline run on documented reference hardware ([slo.md](slo.md)).

## 1. What we measure

| Metric | Definition | Measured by |
|---|---|---|
| End-to-end latency p50/p95 | `POST /v1/query` receipt → final token (non-cached, non-degraded) | query benchmark suite; histogram `fasterrag_request_duration_seconds` |
| Per-stage latency | embed / retrieve(dense) / retrieve(bm25) / fuse / rerank / assemble / generate, p50/p95 — the **retrieval vs generation** split | `fasterrag_stage_duration_seconds`; per-query `timings_ms` |
| Time-to-first-token (TTFT) | request receipt → first SSE `token` event | `fasterrag_ttft_seconds` |
| Ingestion throughput | docs/sec and tokens/sec sustained over ≥ 10 min | ingest benchmark suite; `fasterrag_ingest_throughput` |
| Retrieval quality | recall@k, MRR, nDCG on named eval datasets | eval harness (`pytest -m eval`, `fasterrag benchmark --suite eval`, `scripts/measure_rerank.py`) |
| Faithfulness | grounding score distribution of generated answers | eval harness |
| Cache hit rate | semantic + embedding cache hits / lookups | `fasterrag_cache_events_total` |
| Cost per query | Σ provider token costs / queries | `fasterrag_cost_usd_total` |

## 2. Beta targets (goals — TBD-until-measured)

| Metric | Target | Status |
|---|---|---|
| Retrieval-only p50 (10M chunks, hybrid, no rerank) | ≤ 150 ms | TBD-until-measured |
| End-to-end p95 excl. LLM generation (with rerank) | ≤ 600 ms | TBD-until-measured |
| Rerank stage cost | no target set — the "100–300 ms band" this row used to cite was never measured and has been withdrawn | TBD-until-measured (BENCH-0003 priced the stage on a **non-isolated, CPU-only** machine: 12.9 s p50 for the shipped reranker over a 12-candidate shortlist, 86.9 s at the shipped `rerank_top_n: 100`; indicative only, and no GPU was measured) |
| Ingestion throughput (reference rig) | ≥ 50 docs/sec; ≥ 200k tokens/sec embedded | TBD-until-measured |
| Top-20 retrieval failure rate (hybrid + contextual + rerank) | ≤ 2% | TBD-until-measured (literature anchor: 1.9% in Anthropic's Contextual Retrieval post) |
| Retrieval-quality gain from reranking | no target set — the gain has never been measured on a dataset able to show one | TBD-until-measured (BENCH-0003 measured it on the committed golden sets and found it **inside the noise**: +0.09 MRR ± 0.05 over 16 queries, with recall@5 already 1.0 and the two candidate rerankers indistinguishable. The blocker is dataset size, not hardware) |
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
- `scripts/measure_rerank.py [--dataset <dir>] [--model <cross-encoder>] [--chunk-size <n>] [--json <path>]` prices the rerank stage across candidate `retrieval.rerank` defaults: the harness's own recall@k / MRR / nDCG@k per configuration, plus each cross-encoder's on-disk size, cold load time, and per-query rerank latency. **Built**, and run — see BENCH-0003 in [benchmarks.md](benchmarks.md), which is **not citable**, for two separate reasons: a shared developer workstation, *and* quality deltas that fall inside the committed golden sets' own noise. It reuses `fasterrag.core.evals.evaluate` rather than defining a second notion of retrieval quality, substitutes an exact in-process store for the vector database (so the comparison between configurations holds while the absolute numbers are not a Qdrant reproduction), never downloads weights without `--allow-download`, and decides nothing: it exists to give TASK-0175 a number instead of an expectation, and its most useful finding is which number the repository still cannot produce.
- `python -m tests.load.run_load --target <url> --scenario query|ingest|mixed --users <n> --duration <s> --dataset <dir> --sut-hardware "<spec>" [--isolated] [--repetitions 3] [--processes <n>] [--json <path>] [--ledger-draft]` drives the concurrency scenarios with Locust and reports the [slo.md](slo.md) §1 SLIs. **Built 2026-11-07 and never run against fasterRag** (TASK-0243): there is no isolated reference hardware (TASK-0084) and no load number anywhere in this repository. The tool exists because the provable-claims policy requires the harness to be committed before a comparative claim can be made from it, not because a measurement has been taken. What it measures, what it refuses, and why Locust rather than k6 are in [testing-strategy.md](testing-strategy.md) §1.7; the procedure for running it is §5 below. It never reports a pass — its verdict is `valid-sample` or `invalid` — because every SLO target is still TBD-until-measured.
- The reference-hardware baseline run happens at build-phase slice S11; until then this document contains **no measured numbers by design**.

## 5. Load-run operator procedure

Written for whoever has the hardware, which at the time of writing is nobody. Follow it in order; a step skipped is a run that cannot be cited, and §5.4 says so explicitly rather than leaving it to judgement.

### 5.1 The machines

Three roles, and the first rule is that the first two are **not the same machine**. A load generator competing with the system for cores measures the contention, and the error runs in the direction that makes fasterRag look slow.

| Role | Requirement |
|---|---|
| System under test | Dedicated, idle except for fasterRag and its backends. No desktop session, no other containers, no co-tenant VM. Fixed CPU governor (`performance`, not `ondemand`), swap off or irrelevant, and the exact spec recorded — it is passed to `--sut-hardware` by hand, because nothing on the generator can detect the machine at the other end of a URL. |
| Load generator | A separate box on the same wired LAN segment, with headroom to spare. Watch its own CPU during the run (`top`, `psutil`, anything): if the generator saturates, raise `--processes` until it does not, or the numbers describe Locust. |
| Backends | Qdrant (or the configured vector DB), any embedding/LLM providers, and the journal store. A remote LLM makes the end-to-end figures provider-dominated, which is exactly why the headline SLI excludes generation. Record whether providers were local or remote; it changes what the run means. |

A run across a VPN, a laptop on battery, a cloud instance with burst credits, or any machine shared with another workload is a tool exercise, not a measurement. Say so in the report rather than discarding it quietly — a known-bad run is evidence about the harness.

### 5.2 Running it

1. Install the generator: `pip install -r tests/load/requirements.txt`. Nothing else is needed; the harness's other modules are standard library only.
2. Put a key in the environment if the target has auth on: `export FASTERRAG_LOAD_API_KEY=...`. It is never a flag — a command line is visible in shell history and in every process listing on the box.
3. Pick the dataset. Any directory under `tests/eval/datasets/` works: the queries come from its committed `golden.jsonl` and the ingest corpus from its `corpus/`. The dataset name is part of the number, so a run against an uncommitted corpus cannot be recorded.
4. Point it at a **throwaway collection**. The ingest and mixed scenarios write, and the preflight writes one document of its own. Never run against an index whose contents matter.
5. Run each scenario separately — `query`, then `ingest`, then `mixed` — at each concurrency step you care about. Stepping is done by re-running with a larger `--users`, not by a load shape inside the harness. Record every step, including the ones past the knee; where throughput stops rising is the most useful thing a load test tells you.
6. Pass `--isolated` only if §5.1 actually held. It is an assertion, and it is the switch that decides whether a ledger entry may exist.
7. Keep the defaults for `--repetitions` (3), `--warmup` (60 s) and `--min-samples` (1,000) unless you are deliberately producing a non-citable exploratory run. They are §3's rules 3, 4 and 5 in mechanical form.
8. Read the exit code. `0` means the run produced a valid sample — *not* that the system is fast. `2` preflight refused, `3` the run was vacuous, `4` too many failures or the rate limiter shaped the run, `5` Locust is not installed.

### 5.3 Recording a result in the ledger

Only once real hardware exists. `--ledger-draft` prints either a [benchmarks.md](benchmarks.md) entry in this repository's format or an itemised refusal naming every condition that is not met.

1. Run the scenario to a `valid-sample` verdict with `--isolated` and three warmed repetitions.
2. Take the draft, allocate the next `BENCH-` id, and **rewrite the claim line yourself.** The harness fills in method, dataset, hardware, date, numbers and commit; it deliberately does not write the claim, because what a number licenses is a human judgement. A p95 is a p95: it is not "the fastest RAG" unless a named competitor was measured by us, on this hardware, with this dataset, and that harness is committed too.
3. Attach all three repetitions (`--json`), not only the median. A reader needs the spread.
4. Only then may a `TBD-until-measured` row in §2 above or in [slo.md](slo.md) §1 be replaced, in the same change, with the target derived from it and a link to the entry. [todo.md](todo.md) tick required.

### 5.4 What makes a run invalid

Any one of these, and the run is not a measurement. The first six the harness detects and refuses on its own; the last four it cannot see, and they are the operator's responsibility.

- The preflight failed: the target was unreachable, not ready, not fasterRag, or could not answer the scenario's requests.
- Zero requests were recorded, or a scenario recorded none against an endpoint it requires.
- A latency figure rests on fewer than 1,000 samples, or a p95 came back as zero over real requests.
- The failure ratio exceeded `--max-failure-ratio` (1% by default).
- More than 5% of requests were rate-limited — the run measured `security.rate_limit`, not the system.
- Fewer than three warmed repetitions, or no `--isolated` assertion: both block the ledger draft.
- **Anything else was running on the system under test.** Undetectable from here; this is what `--isolated` asserts.
- **The generator saturated.** A p95 that stops moving as `--users` rises, with the generator's CPU pinned, is Locust's number.
- **The index was not in a known state.** Warm caches from a previous run, or a collection of a different size, make two runs incomparable even on identical hardware.
- **The configuration moved.** Chunking, retrieval or rerank settings are part of the measurement; `config.yaml` must be committed or recorded with the entry.
