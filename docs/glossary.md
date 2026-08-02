# glossary.md — Canonical Terminology

Pinned definitions for every loaded term in this repository. Documentation, code, and future AI sessions must use these terms with exactly these meanings — if a doc appears to use a term differently, that is a bug ([todo.md](todo.md)). Alphabetical.

| Term | Canonical meaning in fasterRag |
|---|---|
| **Adapter** | A class implementing a vendor-neutral base interface (`VectorDBAdapter`, `EmbeddingAdapter`, `LLMAdapter`) inside `adapters/`. Vendor types never escape an adapter. Selected by a factory from config ([ADR-0002](adr/ADR-0002-adapter-factory-pluggability.md)). |
| **Alias swap** | The atomic re-pointing of a collection alias from the old (blue) to the new (green) collection at the end of a zero-downtime reindex (D2). |
| **Autopilot** | D6: eval-driven tuning that outputs a *suggested* config diff with measured deltas. It never applies changes. |
| **Backpressure** | Bounded queues rejecting new work when full; at the API this surfaces as `429` + `Retry-After` (`QUEUE_FULL`), never as unbounded memory growth. |
| **Benchmark ledger** | The append-only record in [benchmarks.md](benchmarks.md) (claim, method, dataset, hardware, date, numbers, commit hash) that must back every performance/superiority claim. |
| **Blue/green (reindex)** | Building a complete new collection in parallel with the serving one, validating it, then alias-swapping. `blue` = currently serving, `green` = candidate. |
| **BM25** | The sparse lexical ranking function used for the keyword leg of hybrid retrieval. |
| **Bulkhead** | Structural isolation: ingestion and query paths use separate worker pools and bounded queues so one cannot starve the other. Not configurable off. |
| **Chunk** | The indexed retrieval unit: text span + metadata (source URI, page/section, offsets, version, content hash, embedding model+version). |
| **Chunking strategies** | `fixed` (token windows), `recursive` (hierarchical separator splitting), `semantic` (similarity-boundary splitting), `layout` (document-structure-aware), `late` (embed long context first, derive chunk embeddings after). Contextual enrichment composes with any of them. |
| **Circuit breaker** | Per-provider failure guard: opens after N consecutive failures, half-open probes after a reset timeout; state exported as `fasterrag_circuit_state` (0 closed / 1 half-open / 2 open). |
| **Citation (span-level)** | A structured reference on an answer: `chunk_id`, source, page, character `span {start, end}`, score — resolvable to a real chunk offset. |
| **Context assembly** | The stage that packs top-K chunks into the prompt within a token budget, deduplicates near-identical chunks, and attaches citations. |
| **Context cliff** | The ~2,500-token chunk-size ceiling beyond which retrieval quality degrades. Directional (single January 2026 preprint), enforced as the hard upper bound of `chunking.chunk_size`. |
| **Contextual enrichment** | Prepending a short (~50–100 token) LLM-generated document-level context to each chunk before embedding and BM25 indexing (Anthropic's Contextual Retrieval; see [references.md](references.md)). |
| **Control plane** | The surfaces that can change system state: REST API, CLI, and the Python package. Never a GUI ([ADR-0005](adr/ADR-0005-api-cli-only-control-plane.md)). |
| **Degradation ladder** | D4: the tested fallback table — reranker down → `hybrid_only`; vector DB down → `cache_only`; LLM down → `extractive`. Every degraded response carries `degraded: true` + `mode`. |
| **Degraded modes** | `full` (normal), `hybrid_only` (fusion results unreranked), `cache_only` (semantic-cache answers only), `extractive` (retrieval-only answers, no generation). As built, `cache_only` is specified but not yet served (TASK-0159). |
| **DLQ (dead-letter queue)** | Where documents land after exhausting `ingestion.dlq.max_retries`, each with a machine-readable reason code (e.g. `PARSE_FAILED`); inspectable per document and re-runnable. |
| **Doctor** | D10: `fasterrag doctor` preflight diagnostics; every failed check prints a concrete fix-it instruction. Must pass before any auto-provisioning runs. |
| **Drift** | Any divergence between the serving index and its `index.lock` (embedding model/version, config hash, corpus content hashes). Detected and reported, never silent (D1). |
| **Error budget** | The allowed SLO shortfall per window; when burning, feature work freezes and reliability tasks take priority ([slo.md](slo.md)). |
| **Eval harness** | The measurement machinery for recall@k, MRR, nDCG, and faithfulness over named datasets ([testing-strategy.md](testing-strategy.md) §1.6). |
| **Faithfulness** | A 0–1 score of how well a generated answer is supported by its retrieved context; below `generation.faithfulness_threshold`, grounded-or-refuse returns `INSUFFICIENT_EVIDENCE`. |
| **Golden set** | A versioned JSONL set of query → relevant-chunk/answer records used by the eval harness, the regression gate, and Autopilot (schema in [testing-strategy.md](testing-strategy.md) §1.6). |
| **Grounded-or-refuse** | D5: answers below the faithfulness threshold are replaced by a structured `INSUFFICIENT_EVIDENCE` response instead of a guess. |
| **Hybrid retrieval** | Dense ANN + sparse BM25 legs run in parallel with the same pushed-down filters, then fused. |
| **Idempotency key** | The `Idempotency-Key` header on mutating endpoints; replaying it returns the original result and performs no duplicate work. |
| **Index lockfile (`index.lock`)** | D1: per-index record of config hash, embedding model name+version, chunker strategy+version, and per-document content hashes — the reproducibility contract of the index. |
| **Ingestion journal** | D3: the checkpointed record (every `checkpoint_every` documents) that lets a crashed ingest resume exactly where it stopped. Written with atomic write-temp-then-rename. |
| **Ledger entry** | One record in the benchmark ledger; the only thing that can turn a goal into a claim. |
| **Problem (`problem+json`)** | The RFC 9457 error body every API error returns, always carrying a stable machine-readable `code`, a `trace_id`, and `retryable`. |
| **Provisioning (config-driven)** | Flipping a config toggle (e.g. `observability.langfuse: true`) → doctor gate → auto-install → configure → running URL, with zero application-code changes at toggle time. Idempotent. |
| **Regression gate** | D7: the CI/reindex check that blocks any change whose recall@k or nDCG drop exceeds `eval.*_tolerance`. |
| **Reranker** | The cross-encoder model that re-scores fused candidates (top 100–1000 → rerank → truncate to `top_k`); ~100–300 ms; the biggest single quality lever. |
| **RRF (Reciprocal Rank Fusion)** | Rank-based fusion `Σ 1/(k + rank)`, default k=60 per Cormack/Clarke/Büttcher SIGIR 2009 ([references.md](references.md)). |
| **RPO / RTO** | Recovery Point Objective (max data-loss window) / Recovery Time Objective (clean host → serving). Both TBD-until-measured by the executed restore drill ([disaster-recovery.md](disaster-recovery.md)). |
| **Semantic cache** | The response cache keyed by query-embedding cosine similarity (threshold ~0.92–0.97, default 0.95); TTL + corpus-change invalidation; tenant-scoped. |
| **SLI / SLO** | Service Level Indicator (the measured quantity) / Objective (the target set only after baselining on reference hardware). |
| **Stateful worker** | An embedding worker that loads its model into memory once and reuses it across all batches — reloading per task is prohibited by design. |
| **TBD-until-measured** | The mandatory marker for any target whose value awaits a baseline measurement; removing it requires a ledger entry. |
| **Tenant** | An isolation domain under `security.multi_tenancy`: scoped collections, API keys, caches, traces, metrics, and budgets. |
| **Tiered embedding** | Routing document classes to different embedding models by cost/priority (`embeddings.tiering.rules`, first match wins). |
| **Time-travel replay** | D8: re-executing a persisted query trace under a candidate config and diffing retrieval sets and answers. |
| **Trace / trace ID** | The correlated record of one request across logs, metrics, spans, problems, and the trace store. Four RAG span types: `retrieval`, `reranker`, `context-assembly`, `generation`. |
| **TTFT** | Time-to-first-token: request receipt → first SSE `token` event. |
| **Vector DB modes** | `docker` (system-managed container), `external` local (user-run, same host), `external` remote (user-run, `host:port` on another machine). |
