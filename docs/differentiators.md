# differentiators.md — The Twelve Flagship Capabilities (Uniqueness Contract)

This file is the uniqueness contract: each capability below is what makes fasterRag defensibly different from assembling a RAG stack by hand — **not by adjectives, but by specified behavior proven by a test or a benchmark.** Provable-claims policy applies: uniqueness statements here describe *specified behavior*; any comparative performance claim must link to the [benchmark ledger](benchmarks.md).

Pain-point numbers reference the catalogue in [scope.md](scope.md). Config keys live in [config-reference.md](config-reference.md); CLI/API surfaces in [cli-reference.md](cli-reference.md) / [api-reference.md](api-reference.md); tests in [testing-strategy.md](testing-strategy.md) and [failure-modes.md](failure-modes.md).

---

## D1 — Index Lockfile & Reproducible Builds

- **What it is.** Every index build writes `index.lock`: config hash, embedding model name + version, chunker strategy + version, and per-document content hashes. Every subsequent operation compares live state to the lockfile; any drift (model version change, config change, corpus change) is detected and reported. A stale or irreproducible index becomes impossible to have *silently*.
- **Pain point it kills.** #4 stale indexes, #20 embedding drift.
- **Why it is unique.** Existing frameworks let you re-embed with a different model or tweak chunking and keep serving mixed vectors without a word; drift discovery is manual archaeology. fasterRag makes the index a reproducible build artifact with a lockfile, like a package manager.
- **Config keys.** `index.lockfile` (bool, default `true`).
- **CLI / API surface.** `fasterrag index lock verify [NAME]` (exit 1 on drift); drift status in `GET /v1/collections/{name}` and `fasterrag index list`.
- **Acceptance test.** Build an index; change `embeddings.model` in config; `index lock verify` exits 1 naming the drifted field; queries against the drifted collection log a drift warning. Covered in the adapter/integration suites.
- **Proof metric.** Drift-detection latency (verify runtime on a 1M-chunk lockfile) recorded in the ledger; zero false negatives across the chaos suite's config-mutation scenarios.

## D2 — Zero-Downtime Reindexing

- **What it is.** Re-embedding a huge corpus never takes queries down: build the new collection in parallel (blue/green), validate it against the eval set, then atomically switch via a collection alias. The old collection is retained for instant rollback for `index.reindex.rollback_retention_hours`.
- **Pain point it kills.** #4 stale indexes, #20 embedding drift, #13 very large datasets (reindex at scale).
- **Why it is unique.** The common path elsewhere is "drop and re-ingest" with hours of downtime, or hand-rolled alias juggling. Here blue/green + eval gate + one-command rollback is the built-in default.
- **Config keys.** `index.reindex.strategy` (`blue_green`|`in_place`, default `blue_green`), `index.reindex.eval_gate` (default `true`), `index.reindex.rollback_retention_hours` (default `72`).
- **CLI / API surface.** `fasterrag index reembed NAME`, `fasterrag index rollback NAME`; `POST /v1/collections/{name}/reindex`, `POST /v1/collections/{name}/rollback`.
- **Acceptance test.** Integration: continuous query load during a full reindex — zero failed queries, alias swap atomic, rollback restores prior answers byte-identically within the retention window.
- **Proof metric.** Queries dropped during reindex (target and measured: 0); swap latency; ledger entry with corpus size + hardware.

## D3 — Checkpointed, Idempotent Ingestion

- **What it is.** Content-hash deduplication; a journal checkpoint every N documents; a crash mid-ingest of millions of docs resumes exactly where it stopped — never restarts. Failed documents go to a dead-letter queue with machine-readable reason codes and a per-document status API. Re-running the same ingest is a no-op (exactly-once index effects).
- **Pain point it kills.** #5 incremental updates & dedup, #13 very large datasets, #22 debugging (per-doc status).
- **Why it is unique.** Typical pipelines restart from zero after a crash and silently double-index on retry. Journal + dedup + DLQ turns ingestion into a resumable, auditable batch system.
- **Config keys.** `ingestion.dedup` (default `true`), `ingestion.journal.enabled` (default `true`), `ingestion.journal.checkpoint_every` (default `100`), `ingestion.dlq.enabled` (default `true`), `ingestion.dlq.max_retries` (default `3`).
- **CLI / API surface.** `fasterrag ingest --watch`; `GET /v1/ingest/{job_id}`, `GET /v1/ingest/{job_id}/documents?status=dead_lettered`, `POST /v1/ingest/{job_id}/retry-dlq`.
- **Acceptance test.** Chaos: kill workers mid-ingest → resume from checkpoint with no duplicate vectors; corrupt document → DLQ with `PARSE_FAILED`, pipeline continues; re-run identical ingest → zero new vectors.
- **Proof metric.** Recovery time from kill (ledger); duplicate-vector count after chaos run (must be 0); dedup hit-rate on re-ingest (must be 100%).

## D4 — Degradation Ladder

- **What it is.** A documented, tested table of graceful fallbacks: reranker down → hybrid-only retrieval; vector DB down → semantic-cache-only answers; LLM down → extractive, retrieval-only answers. Every degraded response carries explicit `degraded: true` + `mode`. There is never a silent quality drop.
- **Pain point it kills.** #12 latency (fail-fast fallbacks), #16 observability gaps (explicit modes), #22 debugging.
- **Why it is unique.** Standard stacks return 500s or silently worse answers when a component dies. The ladder makes partial availability an explicit, tested product state.
- **Config keys.** `reliability.degradation_ladder` (default `true`), plus breaker/timeout keys under `reliability.*`.
- **CLI / API surface.** `mode`/`degraded` fields on every `POST /v1/query` response and SSE `meta` event; `fasterrag status` shows current ladder state; `fasterrag_degraded_responses_total` metric.
- **Acceptance test.** Chaos suite: each rung exercised (stop reranker/vector DB/LLM) → correct mode served, flag present, automatic recovery on restore.
- **Proof metric.** Availability under single-component failure (ledger); 100% of degraded responses flagged in chaos runs.

## D5 — Grounded-or-Refuse Answering

- **What it is.** Span-level citations are mandatory on every generated answer. Each answer gets a faithfulness score; below `generation.faithfulness_threshold` the system returns a structured `insufficient_evidence` response instead of guessing. Hallucination is treated as an availability failure, not a cosmetic one.
- **Pain point it kills.** #3 hallucinations/weak grounding, #19 citation/provenance.
- **Why it is unique.** Citations elsewhere are optional decoration; no mainstream framework refuses to answer on low faithfulness by default contract.
- **Config keys.** `generation.grounded_or_refuse` (default `false` — risky-feature flag), `generation.faithfulness_threshold` (default `0.7`), `generation.citations` (default `true`, cannot be off while grounded_or_refuse is on).
- **CLI / API surface.** `INSUFFICIENT_EVIDENCE` structured response on `/v1/query` (with `best_candidates`); `faithfulness` on every response; `fasterrag query` prints refusals distinctly.
- **Acceptance test.** Eval-harness adversarial set (questions unanswerable from the corpus): 0 fabricated answers when enabled; every returned answer carries ≥ 1 span citation resolving to a real chunk offset.
- **Proof metric.** Hallucination rate on the adversarial set with/without the feature (ledger); faithfulness distribution.

## D6 — Autopilot (Eval-Driven Auto-Tuning)

- **What it is.** Generates a golden Q&A set from the user's own corpus, then searches chunk size, top_k, hybrid weights, and rerank settings against it. Output is a suggested config diff with measured deltas (e.g., recall@10 before/after). It **NEVER auto-applies** changes — the human approves.
- **Pain point it kills.** #1 chunking quality, #2 retrieval accuracy, #10 evaluation difficulty.
- **Why it is unique.** Every other framework hands users a dozen knobs and a shrug; tuning is folklore. Autopilot replaces guesswork with measured suggestions on the user's own data.
- **Config keys.** `autopilot.enabled` (default `false`), `autopilot.golden_set_size` (default `100`).
- **CLI / API surface.** `fasterrag autopilot run --budget-minutes N [--golden-set PATH]` → suggested diff + `autopilot-suggestion.yaml`.
- **Acceptance test.** On a fixture corpus with a known-better config, autopilot's suggestion matches or beats the known config; asserting the tool made zero writes to `config.yaml`. **Executed 2026-08-03** against `tests/eval/datasets/policies`, where the known-better configuration is BM25-weighted retrieval at nDCG@5 0.9308 / MRR 0.9062. Starting from the default 1.0/1.0 weights, autopilot evaluated 7 candidates in 33 s and suggested `bm25_weight=1.0, dense_weight=0.5` — **nDCG 0.9308, matching the known-better score exactly** — and `config.yaml` was byte-unmodified afterwards.
- **Proof metric.** Measured nDCG/MRR delta of suggested vs default on a named dataset. On `policies`: **nDCG +0.0231, MRR +0.0312**. Note that **recall@5 stayed at 1.0000 across all seven trials** — a fixture whose recall is saturated is the only kind that can prove this differentiator, because a tuner that only moves ranking is invisible to recall.

## D7 — Continuous Retrieval Regression Gate

- **What it is.** Every config or index change runs the eval harness; a recall/nDCG regression beyond tolerance blocks the change (CI-integrated and enforced on `reindex` via the eval gate). You cannot accidentally ship a worse RAG.
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

- **What it is.** `fasterrag estimate <path>` reports token counts, projected embedding cost, and projected time per configured provider BEFORE ingestion. Per-query and per-tenant token budgets enforce spend at runtime; tiered embedding routing sends low-priority document classes to cheaper models.
- **Pain point it kills.** #11 cost control.
- **Why it is unique.** Everyone else discovers embedding costs on the invoice. Preflight estimates + hard runtime budgets + tier routing make spend a first-class, enforced dimension.
- **Config keys.** `cost.estimator` (default `true`), `cost.per_query_token_budget` (default `0` = unlimited), `cost.per_tenant_token_budget` (default `0`), `embeddings.tiering.*`.
- **CLI / API surface.** `fasterrag estimate [--all-providers]`; `POST /v1/estimate`; `BUDGET_EXCEEDED` (402) problem on budget breach; `estimated_cost_usd` in every query response.
- **Acceptance test.** Estimator accuracy test: projected vs actual tokens on a fixture corpus within ±5%; budget test: a query engineered to exceed the budget returns 402 before the provider call.
- **Proof metric.** Estimator error % on named corpora (ledger); tier-routing cost delta on a mixed corpus (ledger).

## D10 — `fasterrag doctor`

- **What it is.** Preflight diagnostics: Docker present and running, required ports free, disk space, RAM/GPU availability, vector DB reachable (all three Qdrant modes), API keys valid, config schema valid. Every failed check prints a concrete fix-it instruction. `doctor` must pass before any auto-provisioning runs — this is what makes `langfuse: true`-style provisioning survivable on arbitrary machines.
- **Pain point it kills.** #21 cold-start.
- **Why it is unique.** Frameworks assume a working environment and fail with stack traces; doctor turns environment problems into named checks with fixes, and gates provisioning on them.
- **Config keys.** None (always available); provisioning honors it implicitly.
- **CLI / API surface.** `fasterrag doctor [--fix] [--json]`; `GET /v1/admin/doctor`; exit code 4 on failure.
- **Acceptance test.** Matrix test: each simulated broken precondition (port taken, Docker stopped, missing env var, unreachable remote Qdrant on 6334 only) produces the correct failing check **with a non-empty fix string**; provisioning refuses to run while doctor fails.
- **Proof metric.** Check coverage vs FMEA rows (every environment-class failure mode has a doctor check); doctor runtime (ledger).

## D11 — Portability & Anti-Lock-In

- **What it is.** First-class export / import of documents, chunks, metadata, and the index manifest; a documented migration path between vector DBs (re-embed, or direct vector copy where dimensions match). Leaving fasterRag — or any vendor underneath it — is a supported feature, which is precisely why people can trust adopting it.
- **Pain point it kills.** #17 vendor lock-in.
- **Why it is unique.** Export elsewhere means writing a scraper against the vector DB. Here the portable archive + migration path is part of the product contract.
- **Config keys.** None required (always available); `--include-vectors` at call time. Archive format specified in [archive-format.md](archive-format.md).
- **CLI / API surface.** `fasterrag export --out <archive> [--include-vectors]`, `fasterrag import <archive> [--reembed] [--target-collection]`; `POST /v1/admin/export`, `POST /v1/admin/import`.
- **Acceptance test.** Round-trip: export from Qdrant → import to pgvector (`--reembed`) and to a second Qdrant (vector copy) → eval metrics within tolerance of the source; archive validated against [archive-format.md](archive-format.md).
- **Proof metric.** Round-trip fidelity (chunk/metadata loss = 0; eval delta ≤ gate tolerance); export/import throughput (ledger).

## D12 — Chaos-Certified

- **What it is.** The chaos suite ([testing-strategy.md](testing-strategy.md) §1.9) is public, scripted, and repeatable. [failure-modes.md](failure-modes.md) lists every injected fault and the observed behavior. Reliability anyone can re-run is the differentiator no marketing page can fake.
- **Pain point it kills.** #16 observability gaps, and it is the proof mechanism for D2/D3/D4.
- **Why it is unique.** RAG frameworks do not ship fault-injection suites; reliability claims are prose. Here they are executable.
- **Config keys.** None (test-side).
- **CLI / API surface.** Chaos scripts in-repo, runnable against a local stack; results append to the ledger.
- **Acceptance test.** The suite itself: every scenario in §1.9 passes; every FMEA row's "test that proves it" exists and runs.
- **Proof metric.** Scenario pass rate (must be 100% at release); recovery times per scenario (ledger).
