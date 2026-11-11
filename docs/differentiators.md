# differentiators.md — The Twelve Flagship Capabilities (Uniqueness Contract)

This file is the uniqueness contract: each capability below is what makes fasterRag defensibly different from assembling a RAG stack by hand — **not by adjectives, but by specified behavior that a test or a benchmark must prove.**

> **How to read this file** (clarified by the 2026-08-09 claims audit, TASK-0238):
> - **"Acceptance test"** and **"Proof metric"** state the bar a capability *must* clear. They are the contract, **not a report that it has been cleared.** Where the named test does not exist yet, the bullet says so.
> - **"Proof metric ... (ledger)"** means the number belongs in [benchmarks.md](benchmarks.md) once measured. The ledger currently holds **four entries, all explicitly non-citable** — BENCH-0001, BENCH-0002 and BENCH-0004 for non-isolated hardware, BENCH-0003 for that *and* for quality deltas smaller than their own standard error — so **no proof metric in this file has a measured value yet**. TASK-0084 is the entry that needs real hardware; BENCH-0003's second reason needs a larger golden set instead (TASK-0297).
> - **"Why it is unique"** is positioning based on our own reading of other frameworks. It is **not** a measured competitive comparison, and ledger rule 3 forbids treating it as one: a "faster/better than X" claim may only cite an entry that measured X itself.

Pain-point numbers reference the catalogue in [scope.md](scope.md). Config keys live in [config-reference.md](config-reference.md); CLI/API surfaces in [cli-reference.md](cli-reference.md) / [api-reference.md](api-reference.md); tests in [testing-strategy.md](testing-strategy.md) and [failure-modes.md](failure-modes.md).

---

## D1 — Index Lockfile & Reproducible Builds

- **What it is.** Every index build writes `index.lock`: config hash, embedding model name + version, chunker strategy + version, and per-document content hashes. Every subsequent operation compares live state to the lockfile; any drift (model version change, config change, corpus change) is detected and reported. A stale or irreproducible index is hard to have *silently*: every single-field edit of the hashed config subset is covered by a schema-derived mutation suite, while two-field edits and non-config drift routes are sampled rather than enumerated (see Proof metric).
- **Pain point it kills.** #4 stale indexes, #20 embedding drift.
- **Why it is unique.** Existing frameworks let you re-embed with a different model or tweak chunking and keep serving mixed vectors without a word; drift discovery is manual archaeology. fasterRag makes the index a reproducible build artifact with a lockfile, like a package manager.
- **Config keys.** `index.lockfile` (bool, default `true`).
- **CLI / API surface.** `fasterrag index lock verify [NAME]` (exit 1 on drift); drift status in `GET /v1/collections/{name}` and `fasterrag index list`.
- **Acceptance test.** Build an index; change `embeddings.model` in config; `index lock verify` exits 1 naming the drifted field; queries against the drifted collection log a drift warning. Covered in the adapter/integration suites.
- **Proof metric.** *(One half met, one half still a target.)* Drift-detection latency (verify runtime on a 1M-chunk lockfile) is **still unrecorded** — no ledger entry exists, and the measurement waits on the isolated hardware TASK-0084 is blocked on. The config-mutation scenarios now exist (`tests/chaos/test_config_mutation.py`, TASK-0241 ✅) and run under `pytest -m chaos`. Read precisely, they establish this much: the mutation set is **derived from the live schema** rather than listed by hand, so no present or future leaf of the hashed subset can quietly stop being covered, and across every **single-field** edit of that subset the detector both reports drift and names a field, while legal edits outside the subset are asserted *not* to report. That is zero false negatives for single-field edits at the current schema — not the blanket claim. Edits moving two fields at once are not enumerated, and drift arriving by a route other than configuration (corpus content, embedding weights behind a stable model name, identity scheme) is **sampled, one scenario each**.

## D2 — Zero-Downtime Reindexing

- **What it is.** Re-embedding a huge corpus never takes queries down: build the new collection in parallel (blue/green), validate it against the eval set, then atomically switch via a collection alias. The old collection is retained for `index.reindex.rollback_retention_hours`, so rolling back is one alias flip and no rebuild — the flip's own latency is this differentiator's proof metric and is unmeasured.
- **Pain point it kills.** #4 stale indexes, #20 embedding drift, #13 very large datasets (reindex at scale).
- **Why it is unique.** The common path elsewhere is "drop and re-ingest" with hours of downtime, or hand-rolled alias juggling. Here blue/green + eval gate + one-command rollback is the built-in default.
- **Config keys.** `index.reindex.strategy` (`blue_green`|`in_place`, default `blue_green`), `index.reindex.eval_gate` (default `true`), `index.reindex.rollback_retention_hours` (default `72`).
- **CLI / API surface.** `fasterrag index reembed NAME`, `fasterrag index rollback NAME`; `POST /v1/collections/{name}/reindex`, `POST /v1/collections/{name}/rollback`.
- **Acceptance test.** Alias swap atomicity and rollback are covered by the shared adapter contract suite and `tests/unit/services/test_reindex.py`. The **continuous-query-load-during-reindex** drill landed 2026-11-02 as `tests/chaos/test_reindex_under_load.py` (8 cases), closing TASK-0241. One bound it measured rather than fixed: a *hybrid* query issues its dense and sparse legs as two searches that resolve the alias independently, so a swap landing between them returns one result fused across both generations — the either-old-or-new guarantee holds per retrieval leg, not per hybrid query. Recorded in [failure-modes.md](failure-modes.md) with a `# TODO:` at the seam.
- **Proof metric.** *(Target, not a result.)* Queries dropped during reindex — target 0. The drill above observes zero against an **in-process adapter double** that resolves the alias per search call the way Qdrant does; that is evidence about fasterRag's orchestration, not about a live backend under concurrent writes. Unmeasured against real infrastructure, and no swap-latency or corpus-size figure exists; both need a ledger entry.

## D3 — Checkpointed, Idempotent Ingestion

- **What it is.** Content-hash deduplication; a journal checkpoint every N documents; a crash mid-ingest of millions of docs resumes exactly where it stopped — never restarts. Failed documents go to a dead-letter queue with machine-readable reason codes and a per-document status API. Re-running the same ingest is a no-op (exactly-once index effects).
- **Pain point it kills.** #5 incremental updates & dedup, #13 very large datasets, #22 debugging (per-doc status).
- **Why it is unique.** Typical pipelines restart from zero after a crash and silently double-index on retry. Journal + dedup + DLQ turns ingestion into a resumable, auditable batch system.
- **Config keys.** `ingestion.dedup` (default `true`), `ingestion.journal.enabled` (default `true`), `ingestion.journal.checkpoint_every` (default `100`), `ingestion.dlq.enabled` (default `true`), `ingestion.dlq.max_retries` (default `3`).
- **CLI / API surface.** `fasterrag ingest` (`--watch` is implemented, TASK-0196 ✅: it prints the job id before the work starts, then one appended `progress` line per change in the job's per-document tally, polled from the ingestion journal — the same state `GET /v1/ingest/{job_id}` reports. It is refused with a note under `--json`/`--quiet`, which suppress the lines it would print, and under `--dry-run`, which creates no job); `GET /v1/ingest/{job_id}`, `GET /v1/ingest/{job_id}/documents?status=dead_lettered`, `POST /v1/ingest/{job_id}/retry-dlq`.
- **Acceptance test.** Chaos: kill workers mid-ingest → resume from checkpoint with no duplicate vectors; corrupt document → DLQ with `PARSE_FAILED`, pipeline continues; re-run identical ingest → zero new vectors.
- **Proof metric.** Recovery time from kill (ledger); duplicate-vector count after chaos run (must be 0); dedup hit-rate on re-ingest (must be 100%).

## D4 — Degradation Ladder

- **What it is.** A documented table of graceful fallbacks: reranker down → hybrid-only retrieval; vector DB down → semantic-cache-only answers; LLM down → extractive, retrieval-only answers. Every degraded response carries explicit `degraded: true` + `mode`. **As built, two of the three rungs exist**: `hybrid_only` and `extractive` are served; the **`cache_only` rung is specified and not implemented** (TASK-0159), so a vector-DB outage currently raises a retryable `RETRIEVAL_FAILED` rather than serving a degraded answer.
- **Pain point it kills.** #12 latency (fail-fast fallbacks), #16 observability gaps (explicit modes), #22 debugging.
- **Why it is unique.** Standard stacks return 500s or silently worse answers when a component dies. The ladder makes partial availability an explicit, tested product state.
- **Config keys.** `reliability.degradation_ladder` (default `true`), plus breaker/timeout keys under `reliability.*`.
- **CLI / API surface.** `mode`/`degraded` fields on every `POST /v1/query` response and SSE `meta` event; `fasterrag status` shows current ladder state; `fasterrag_degraded_responses_total` metric.
- **Acceptance test.** *(Partially satisfied.)* The chaos suite exercises the slow-LLM rung (`extractive` served) and stop-vector-DB — but the latter asserts a retryable `RETRIEVAL_FAILED`, which is the built behaviour, **not** the `cache_only` rung this row specifies. That rung cannot be acceptance-tested until TASK-0159 builds it.
- **Proof metric.** *(Target, not a result.)* Availability under single-component failure (ledger — no entry yet); every degraded response flagged, which holds for the two rungs that exist.

## D5 — Grounded-or-Refuse Answering

- **What it is.** Span-level citations are mandatory on every generated answer. Each answer gets a faithfulness score; below `generation.faithfulness_threshold` the system returns a structured `insufficient_evidence` response instead of guessing. Hallucination is treated as an availability failure, not a cosmetic one.
- **Pain point it kills.** #3 hallucinations/weak grounding, #19 citation/provenance.
- **Why it is unique.** Citations elsewhere are optional decoration; no mainstream framework refuses to answer on low faithfulness by default contract.
- **Config keys.** `generation.grounded_or_refuse` (default `false` — risky-feature flag), `generation.faithfulness_threshold` (default `0.7`), `generation.citations` (default `true`, cannot be off while grounded_or_refuse is on).
- **CLI / API surface.** `INSUFFICIENT_EVIDENCE` structured response on `/v1/query` (with `best_candidates`); `faithfulness` on every response; `fasterrag query` prints refusals distinctly.
- **Acceptance test.** Eval-harness adversarial set (questions unanswerable from the corpus): 0 fabricated answers when enabled; every returned answer carries ≥ 1 span citation resolving to a real chunk offset.
- **Proof metric.** Hallucination rate on the adversarial set with/without the feature (ledger); faithfulness distribution.

## D6 — Autopilot (Eval-Driven Auto-Tuning)

- **What it is.** Generates a golden Q&A set from the user's own corpus, then searches configurations against it and emits a suggested config diff with measured deltas (e.g., recall@10 before/after). It **NEVER auto-applies** changes — the human approves. The search is split in two because the two halves cost different things:
  - **Query-time** (`tune`, behind `fasterrag autopilot run`): hybrid weights, `rrf_k`, rerank. One index serves every candidate, so a candidate costs one evaluation. `top_k` is excluded by construction — measuring recall@k fixes the retrieval depth, so a `top_k` candidate cannot change a score.
  - **Index-time** (`plan_index_sweep` + `sweep_index_time`, 2026-11-04): `chunking.chunk_size` and `chunking.overlap`. Each candidate is a *different index*, so each one costs a re-chunk and re-embed of the dataset's corpus. The plan prices every rebuild with D9's estimator before any of them runs, the sweep refuses to start unless the caller approves that rebuild count (default `0`), each candidate is built into a scratch `autopilot-sweep-*` collection that is dropped in a `finally` over names recorded before each build began, and a drop that fails is reported as a leak rather than swallowed. The live collection is scored as-is and no alias is ever moved.
- **Pain point it kills.** #1 chunking quality, #2 retrieval accuracy, #10 evaluation difficulty.
- **Why it is unique.** Every other framework hands users a dozen knobs and a shrug; tuning is folklore. Autopilot replaces guesswork with measured suggestions on the user's own data — and prices the expensive half before spending the operator's money on it.
- **Config keys.** `autopilot.enabled` (default `false`), `autopilot.golden_set_size` (default `100`). The index-time sweep has no key of its own yet (TASK-0145); its opt-in is the `approved_rebuilds` argument, which defaults to zero rebuilds.
- **CLI / API surface.** `fasterrag autopilot run --budget-minutes N [--golden-set PATH]` → suggested diff + `autopilot-suggestion.yaml`. The index-time sweep is a Python surface only so far (TASK-0145): `plan_index_sweep` → `render_plan` → `sweep_index_time`, the caller supplying the `rebuild` callable that builds one scratch collection.
- **Acceptance test.** On a fixture corpus with a known-better config, autopilot's suggestion matches or beats the known config; asserting the tool made zero writes to `config.yaml`. **Executed 2026-08-03** against `tests/eval/datasets/policies`, where the known-better configuration is BM25-weighted retrieval at nDCG@5 0.9308 / MRR 0.9062. Starting from the default 1.0/1.0 weights, autopilot evaluated 7 candidates (in tens of seconds on a developer laptop — an illustration, not a ledger entry) and suggested `bm25_weight=1.0, dense_weight=0.5` — **nDCG 0.9308, matching the known-better score exactly** — and `config.yaml` was byte-unmodified afterwards.
- **Acceptance test, index-time half (TASK-0145).** Unverified end to end, stated plainly: the sweep's seams are unit-tested — the plan's pricing and rejections, the zero-rebuild default, the scratch naming, the cleanup after a rebuild that dies mid-build, a drop that fails being reported as a leak rather than hidden, the budget stopping before a rebuild rather than after, and the refusal to score an embedding candidate with the wrong router. What has **never run** is a real sweep: no vector database and no Docker were available on the machine where it was written, so no scratch collection has ever been created or dropped against a live backend, and no chunk-size candidate has ever been scored on an index built for it. The first real run is the acceptance test, and until it happens the measured-delta column below stays empty for this half.
- **Proof metric.** Measured nDCG/MRR delta of suggested vs default on a named dataset. On `policies`: **nDCG +0.0231, MRR +0.0312**. Note that **recall@5 stayed at 1.0000 across all seven trials** — a fixture whose recall is saturated is the only kind that can prove this differentiator, because a tuner that only moves ranking is invisible to recall. No index-time delta is recorded: the sweep has not been run against a live backend (TASK-0145).

## D7 — Continuous Retrieval Regression Gate

- **What it is.** Every config or index change runs the eval harness; a recall/nDCG regression beyond tolerance blocks the change (the `eval-gate` CI job, and enforced on `reindex` via the eval gate). **Scope bound:** the CI job pins the configuration the committed baseline was recorded under — `top_k=5`, `rerank=false`, `token_counter=estimate` — so it does not cover the reranker, and the `policies` dataset is not scored by it. A baseline it cannot compare fails the job rather than passing it.
- **Pain point it kills.** #10 evaluation difficulty, #2 retrieval accuracy (protects it over time).
- **Why it is unique.** Retrieval quality elsewhere is measured ad hoc, if ever; here it is a blocking CI gate like type checks.
- **Config keys.** `eval.regression_gate` (default `false`), `eval.recall_tolerance` (default `0.02`), `eval.ndcg_tolerance` (default `0.02`).
- **CLI / API surface.** `fasterrag benchmark --suite eval` (exit 5 on breach); the same gate guards `index reembed` when `index.reindex.eval_gate: true`.
- **Acceptance test.** CI fixture: a deliberately-degrading config change (e.g. chunk_size 2400, hybrid off) is blocked; a neutral change passes.
- **Proof metric.** Gate correctness on fixture suite (0 false passes); harness runtime (ledger).

## D8 — Time-Travel Replay

- **What it is.** Every query's full trace (retrieved chunks, scores, prompt, response) is persisted locally. `fasterrag replay --trace <id> --config candidate.yaml` re-executes a past query under a new config and shows a side-by-side diff of retrieval sets and answers. "Why did this answer change last week?" becomes answerable.
- **Pain point it kills.** #22 debugging retrieval-vs-generation failures, #16 observability gaps.
- **Why it is unique.** Trace viewers elsewhere show what happened; none re-execute history under a candidate config with a structured diff.
- **Config keys.** `traces.store` (default `true`), `traces.retention_days` (default `30`), `traces.replay` (default `true`).
- **CLI / API surface.** `fasterrag replay --trace <id> --config candidate.yaml [--diff-only]`; `GET /v1/traces/{id}`; `POST /v1/replay`.
- **Acceptance test.** Integration: replay under identical config reproduces the retrieval set exactly; replay under changed `rrf_k` shows a deterministic, correctly-computed diff.
- **Proof metric.** Replay determinism rate (must be 100% for identical config); trace storage overhead per query (ledger).

## D9 — Cost Governor & Preflight Estimator

- **What it is.** `fasterrag estimate <path>` reports token counts and projected embedding cost per configured provider BEFORE ingestion (wall-clock time is deliberately **not** projected — throughput has never been measured). Tiered embedding routing sends low-priority document classes to cheaper models. **The runtime cost governor is built** (`services/budget.py`, TASK-0242 ✅): both token budgets are checked before the provider is called, against the prompt's size plus `llm.max_tokens` rather than the completion's unknown real size, and a refusal raises non-retryable `BUDGET_EXCEEDED`. One caveat, marked `# CRITICAL:` in that module: the per-tenant window is counted **per replica**, so N replicas grant N times the configured budget until a shared counter lands. The per-query budget is stateless and identical everywhere.
- **Pain point it kills.** #11 cost control.
- **Why it is unique.** Everyone else discovers embedding costs on the invoice. Preflight estimates + hard runtime budgets + tier routing make spend a first-class, enforced dimension.
- **Config keys.** `cost.estimator` (default `true`), `cost.per_query_token_budget` (default `0` = unlimited), `cost.per_tenant_token_budget` (default `0`), `embeddings.tiering.*`.
- **CLI / API surface.** `fasterrag estimate [--all-providers]`; `POST /v1/estimate`. The runtime governor ships (TASK-0242 ✅): both budgets are enforced before the provider is called and refuse with `BUDGET_EXCEEDED` (402). `estimated_cost_usd` is emitted on every query response (TASK-0260 ✅), nullable so an unpriced model reports nothing rather than a fabricated zero.
- **Acceptance test.** *(Budget half written; estimator half not.)* The budget test exists: `tests/unit/services/test_generation.py` engineers a query over the cap and asserts non-retryable `BUDGET_EXCEEDED` — which the problem table maps to 402 — with the provider recording no call, on both the answering and the streaming paths. Estimator accuracy (projected vs actual tokens on a fixture corpus within ±5%) is **still unwritten**.
- **Proof metric.** *(Target, not a result.)* Estimator error % on named corpora (ledger); tier-routing cost delta on a mixed corpus (ledger). No entry for either.

## D10 — `fasterrag doctor`

- **What it is.** Preflight diagnostics: Docker present and running, required ports free, disk space, RAM/GPU availability, vector DB reachable (all three Qdrant modes), API keys valid, config schema valid. Every failed check prints a concrete fix-it instruction. `doctor` must pass before any auto-provisioning runs — this is what makes `langfuse: true`-style provisioning survivable on arbitrary machines.
- **Pain point it kills.** #21 cold-start.
- **Why it is unique.** Frameworks assume a working environment and fail with stack traces; doctor turns environment problems into named checks with fixes, and gates provisioning on them.
- **Config keys.** None (always available); provisioning honors it implicitly.
- **CLI / API surface.** `fasterrag doctor [--fix] [--json]`; `GET /v1/admin/doctor`; exit code 4 on failure. `--fix` repairs only what is idempotent and non-destructive — the missing named volume, and fasterRag's *own* stopped container — then re-runs every check and classifies each attempt from that second report, so a repair that did not take is reported `not fixed` rather than claimed. Everything else keeps its concrete fix-it line and is reported as needing a human. Over REST the repairs are `POST /v1/admin/doctor/fix`, which calls the same service function under the same guards; `GET /v1/admin/doctor` stays read-only because a GET must not mutate.
- **Acceptance test.** Matrix test: each simulated broken precondition (port taken, Docker stopped, missing env var, unreachable remote Qdrant on 6334 only) produces the correct failing check **with a non-empty fix string**; provisioning refuses to run while doctor fails. Covered by `tests/unit/services/test_doctor.py`.
- **Proof metric.** *(Target, not a result.)* Check coverage vs FMEA rows (every environment-class failure mode has a doctor check); doctor runtime — no ledger entry.

## D11 — Portability & Anti-Lock-In

- **What it is.** First-class export / import of documents, chunks, metadata, and the index manifest; a documented migration path between vector DBs (re-embed, or direct vector copy where dimensions match). Leaving fasterRag — or any vendor underneath it — is a supported feature, which is precisely why people can trust adopting it.
- **Pain point it kills.** #17 vendor lock-in.
- **Why it is unique.** Export elsewhere means writing a scraper against the vector DB. Here the portable archive + migration path is part of the product contract.
- **Config keys.** None required (always available); `--include-vectors` at call time. Archive format specified in [archive-format.md](archive-format.md).
- **CLI / API surface.** `fasterrag export --out <archive> [--include-vectors]`, `fasterrag import <archive> [--reembed] [--target-collection]`; `POST /v1/admin/export`, `POST /v1/admin/import`.
- **Acceptance test.** *(Half satisfied.)* Qdrant → Qdrant vector copy ran once against a **live** backend (TASK-0214 ✅) and that run found three defects the fake adapter had hidden: a vector-copy guard that compared the archive to itself, provenance (`source_uri`, `content_hash`) discarded on import, and no `index.lock` written. All three are fixed, and the fixes are **re-verified against a fake adapter only** — `tests/integration/test_archive_round_trip.py` (12 cases) skips without a Docker daemon, so nothing has round-tripped through a real Qdrant since the change (TASK-0079). The **Qdrant → pgvector `--reembed`** leg is still unrun.
- **Proof metric.** *(Not yet a result.)* Round-trip fidelity target: chunk/metadata loss = 0. The one live Qdrant run measured loss **greater than zero** — `source_uri` and `content_hash` were dropped on every import — and the fix for it has only been exercised against a fake adapter (TASK-0079). Export/import throughput: **no ledger entry**.

## D12 — Chaos-Certified

- **What it is.** The chaos suite ([testing-strategy.md](testing-strategy.md) §1.9) is public, scripted, and repeatable. [failure-modes.md](failure-modes.md) lists every injected fault and the observed behavior. Reliability anyone can re-run is the differentiator no marketing page can fake.
- **Pain point it kills.** #16 observability gaps, and it is the proof mechanism for D2/D3/D4.
- **Why it is unique.** RAG frameworks do not ship fault-injection suites; reliability claims are prose. Here they are executable.
- **Config keys.** None (test-side).
- **CLI / API surface.** Chaos scripts in-repo (`tests/chaos/`), runnable against a local stack. Nothing appends results to the ledger automatically — that is a manual step.
- **Acceptance test.** *(Partly satisfied.)* Every scripted scenario passes. Partition injection, long missing, now exists (TASK-0241, partly), as does the config-mutation scenario; **rows citing a load test or a soak test still name suites that have never been written** (TASK-0243). The self-audit test that asserts the suite covers each §1.9 row currently enumerates only the five scenarios living in `tests/chaos/test_chaos.py`, so it does not yet cover the two that own their own modules — the gap is in the audit, not in the scenarios, and closing it is an open task.
- **Proof metric.** *(Target, not a result.)* Scenario pass rate — currently 100% of the scripted scenarios. Recovery times per scenario are **not measured** (TASK-0136, which needs the isolated hardware of TASK-0084).
