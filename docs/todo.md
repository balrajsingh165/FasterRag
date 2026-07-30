# todo.md — Universal Task File

> **This is the ONLY task file in fasterRag. No other todo/task/tracking file may ever be created. All tasks live here.**
>
> Rules:
> - Open task: `- [ ] TASK-0001: <description>`
> - Completed task: `- [x] TASK-0001: <description> — ✅ YYYY-MM-DD`
> - **Append-only rule:** when a task is completed, tick the checkbox and append the completion date in `YYYY-MM-DD` format. After ticking, that entry is append-only and must NOT be edited further.
> - Task IDs (`TASK-`/`AUDIT-`) are sequential and never reused. Audit-gap tasks use the `AUDIT-` prefix.

## Done

### Phase 1 — documentation set

- [x] TASK-0001: Author CLAUDE.md (always-loaded rules, tech stack, standards, pointers) — ✅ 2026-07-29
- [x] TASK-0002: Author scope.md (vision, non-goals, pain-point catalogue, success criteria) — ✅ 2026-07-29
- [x] TASK-0003: Author todo.md (this universal task file with entry format and append-only rule) — ✅ 2026-07-29
- [x] TASK-0004: Author structure.md (proposed repository layout and layer responsibilities) — ✅ 2026-07-29
- [x] TASK-0005: Author flow.md (Mermaid flows: ingestion, worker hand-off, query, cache, provisioning) — ✅ 2026-07-29
- [x] TASK-0006: Author architecture.md (workers, adapters, batching, caching, fault tolerance, scaling) — ✅ 2026-07-29
- [x] TASK-0007: Author config-reference.md (full config.yaml schema with types, defaults, validation) — ✅ 2026-07-29
- [x] TASK-0008: Author api-reference.md (REST endpoints, schemas, status codes, streaming semantics) — ✅ 2026-07-29
- [x] TASK-0009: Author cli-reference.md (all commands, subcommands, flags) — ✅ 2026-07-29
- [x] TASK-0010: Author observability.md (dashboard spec, metrics catalogue, Langfuse/Grafana provisioning) — ✅ 2026-07-29
- [x] TASK-0011: Author deployment.md (self-hosting, Docker modes, remote connections, sizing) — ✅ 2026-07-29
- [x] TASK-0012: Author testing-strategy.md (unit/integration tests, eval harness, CI gates) — ✅ 2026-07-29
- [x] TASK-0013: Author security.md (secrets, .env policy, auth, multi-tenancy, Qdrant API key) — ✅ 2026-07-29
- [x] TASK-0014: Author performance.md (benchmark targets and measurement methodology) — ✅ 2026-07-29
- [x] TASK-0015: Author integrations.md (vector DBs, embedding/LLM providers, observability tools) — ✅ 2026-07-29
- [x] TASK-0016: Author CHANGELOG.md (Keep a Changelog + SemVer 2.0.0) — ✅ 2026-07-29
- [x] TASK-0017: Author CONTRIBUTING.md (commit, comment, one-todo-file, incremental-shipping rules) — ✅ 2026-07-29
- [x] TASK-0018: Seed docs/adr/ with ADR-0001..ADR-0005 (MADR style) — ✅ 2026-07-29
- [x] TASK-0019: Review documentation set against Phase 1 acceptance criteria and fix gaps (executed as the Gate A audit, see AUDIT tasks) — ✅ 2026-07-29

### Phase 2 — restructure, framework surface, Gate B hardening

- [x] TASK-0056: Relocate all documentation into docs/ keeping only CLAUDE.md and README.md at root — ✅ 2026-07-29
- [x] TASK-0057: Author python-api.md (importable package surface, sync facade, standalone components, entry-point plugin contract) — ✅ 2026-07-29
- [x] TASK-0058: Author ADR-0006 (ship fasterRag as an importable Python package) — ✅ 2026-07-29
- [x] TASK-0059: Author detailed root README.md positioning fasterRag as a RAG framework — ✅ 2026-07-29
- [x] TASK-0060: Author differentiators.md (D1–D12, each with pain point, uniqueness, config keys, CLI/API surface, acceptance test, proof metric) — ✅ 2026-07-29
- [x] TASK-0061: Author reliability.md (doctrine sentence, error taxonomy, resilience patterns, data safety, static discipline) — ✅ 2026-07-29
- [x] TASK-0062: Author failure-modes.md (FMEA, 37 rows, every row naming a proving test) — ✅ 2026-07-29
- [x] TASK-0063: Author slo.md (SLI definitions, TBD-until-measured targets, error budget policy) — ✅ 2026-07-29
- [x] TASK-0064: Author disaster-recovery.md (backup inventory, executable restore drill, RPO/RTO TBD-until-measured) — ✅ 2026-07-29
- [x] TASK-0065: Author benchmarks.md as the append-only benchmark ledger with entry schema and rules — ✅ 2026-07-29
- [x] TASK-0066: Weave Phase 2 surfaces into references (RFC 9457 error model in api-reference; doctor/estimate/replay/benchmark/export/import in cli-reference; D1–D12 + reliability keys in config-reference; full pyramid in testing-strategy; supply chain in security) — ✅ 2026-07-29
- [x] TASK-0067: Update CLAUDE.md with Phase 2 permanent constraints (provable claims, error taxonomy, flags-default-false) — ✅ 2026-07-29

### Gate A — audit (zero open AUDIT tasks = gate complete)

- [x] AUDIT-0001: architecture.md contained unmeasured superlative "fastest" for the fixed chunker — rephrased to a factual statement — ✅ 2026-07-29
- [x] AUDIT-0002: scope.md contained two "best-in-class" claims — rephrased as goals per the provable-claims policy — ✅ 2026-07-29
- [x] AUDIT-0003: scope.md control-plane reconciliation predated the Python package surface — updated to API + CLI + library — ✅ 2026-07-29
- [x] AUDIT-0004: CLAUDE.md pointer list was missing python-api.md — added — ✅ 2026-07-29
- [x] AUDIT-0005: structure.md directory tree was missing python-api.md — added — ✅ 2026-07-29
- [x] TASK-0068: Gate A audit complete (files/sections verified; commit log 100% single-line with zero attribution; exactly one todo file; config-reference parameter coverage verified; pain points and reconciliation verified; ADRs MADR-style and sequential) — ✅ 2026-07-29
- [x] TASK-0069: Gate B documentation hardening complete (all Section 3 files exist with required sections) — ✅ 2026-07-29
- [x] TASK-0070: Merge documentation branch to main and push — ✅ 2026-07-29

### Phase 3 — repository hardening

- [x] TASK-0088: Add .gitignore (secrets, caches, build artifacts) — ✅ 2026-07-29
- [x] TASK-0089: Add .env.example matching the security.md variable inventory — ✅ 2026-07-29
- [x] TASK-0090: Commit canonical default config.yaml, byte-consistent with the config-reference.md example — ✅ 2026-07-29
- [x] TASK-0091: Add GitHub PR template (rule-enforcement checklist) and bug/feature issue templates — ✅ 2026-07-29
- [x] TASK-0092: Add .github/SECURITY.md private vulnerability-reporting policy — ✅ 2026-07-29
- [x] TASK-0093: Author docs/glossary.md pinning canonical terminology — ✅ 2026-07-29
- [x] TASK-0094: Author docs/references.md consolidating external evidence sources (R1–R14) — ✅ 2026-07-29
- [x] TASK-0095: Author docs/archive-format.md specifying the D11 portability archive (v1.0.0) — ✅ 2026-07-29
- [x] TASK-0096: Add the golden-set JSONL schema to testing-strategy.md (shared by eval harness, D6, D7) — ✅ 2026-07-29
- [x] TASK-0097: Merge repo-hardening branch to main and push — ✅ 2026-07-29

### Phase 4 — adoption guides and build-de-risking specs

- [x] TASK-0099: Documentation consistency audit (relative links, config-key drift vs config.yaml, error codes, metric names, CLI commands) — zero real defects found — ✅ 2026-07-29
- [x] TASK-0100: Author quickstart.md (zero-to-answered-query for CLI, Python, and REST paths) — ✅ 2026-07-29
- [x] TASK-0101: Author data-model.md (canonical entity schemas, ID scheme, cross-entity invariants) — ✅ 2026-07-29
- [x] TASK-0102: Author prompts.md (P1 generation, P2 contextual enrichment, P3 faithfulness, P4 golden-set contracts) — ✅ 2026-07-29
- [x] TASK-0103: Author troubleshooting.md (symptom → cause → fix, user-facing inverse of the FMEA) — ✅ 2026-07-29
- [x] TASK-0104: Author cookbook.md (nine composable configuration recipes) — ✅ 2026-07-29
- [x] TASK-0105: Author migration-guide.md (concept mapping, migration procedure, honest trade-offs) — ✅ 2026-07-29
- [x] TASK-0106: Cross-link Phase 4 docs from README, CLAUDE.md, and structure.md — ✅ 2026-07-29
- [x] TASK-0107: Merge adoption-guides branch to main and push — ✅ 2026-07-29

### Slice S1 — skeleton (build phase)

- [x] TASK-0021: Scaffold repository per structure.md (pyproject, src layout, tooling: ruff, mypy strict, pytest, pre-commit) — ✅ 2026-07-29
- [x] TASK-0072: Implement error-taxonomy base classes, RFC 9457 problem responses, structured logging with correlation ids — ✅ 2026-07-29
- [x] TASK-0022: Implement config loader (pydantic-settings + YAML source, .env secrets, fail-fast validation incl. all cross-field rules) — ✅ 2026-07-29
- [x] TASK-0039: Implement REST API app factory + routers (endpoints land incrementally per slice; /healthz + /readyz first) — ✅ 2026-07-29

### Slice S2 — Qdrant adapter + doctor v1 (build phase)

- [x] TASK-0023: Implement VectorDBAdapter base interface + factory (create_collection, upsert, search, update, delete, health) — ✅ 2026-07-30
- [x] TASK-0024: Implement Qdrant reference adapter (docker | external | remote modes, 6333/6334 both handled, prefer_grpc, API key) — ✅ 2026-07-30
- [x] TASK-0025: Implement Qdrant system-managed Docker provisioning (named volume enforced on Windows/WSL, both ports exposed) — ✅ 2026-07-30
- [x] TASK-0073: Implement the shared adapter contract test suite (run against Qdrant in all three modes) — ✅ 2026-07-30
- [x] TASK-0074: Implement fasterrag doctor v1 (Docker, ports, disk, RAM/GPU, DB reachability all modes, key validity, config validity; fix-it strings; provisioning gate) — ✅ 2026-07-30

### Slice S3 — ingestion core (build phase, in progress)

- [x] TASK-0110: Register the vector database health check with `/readyz` — ✅ 2026-07-30
- [x] TASK-0028: Implement parsing pipeline (PDF incl. tables/OCR, HTML, Markdown, DOCX, TXT, CSV/JSON; golden-file tests) — ✅ 2026-07-30
- [x] TASK-0029: Implement chunking pipeline (recursive baseline first; then fixed, semantic, layout, late; Hypothesis invariants) — ✅ 2026-07-30
- [x] TASK-0026: Implement embedding provider adapters (HuggingFace local first; OpenAI, Cohere, Ollama) + tiering router — ✅ 2026-07-30

### Workflow

- [x] TASK-0108: Land slice S1 on main — ✅ 2026-07-30
- [x] TASK-0109: Land slice S2 on main — ✅ 2026-07-30
- [x] TASK-0112: Adopt trunk-based development on main and update CLAUDE.md, CONTRIBUTING.md, and the revert playbook (maintainer instruction) — ✅ 2026-07-30

## In Progress

_(empty)_

## Pending

- [ ] TASK-0020: Decide beta version number and stamp CHANGELOG Unreleased → 0.1.0-beta.1 on first release
- [ ] TASK-0098: (maintainer action, GitHub settings) Enable branch protection on main — require PRs, block direct pushes
- [ ] TASK-0111: (maintainer action) Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2` on main
- [ ] TASK-0110: Register the vector database health check with `/readyz` when the API first uses the adapter (S3/S4); the readiness registry exists and the adapter's `health()` is ready, but nothing wires them together yet
- [ ] AUDIT-0007: deployment.md states every provisioned container runs non-root, but the Qdrant provisioner does not pass `--user` — the official image expects to own its storage volume and forcing a uid risks an unstartable container. Decide whether to verify a working non-root uid for the pinned image or to narrow the documented claim to the containers fasterRag builds itself
- [ ] AUDIT-0006: `trace_id` format conflict — data-model.md gives Trace ids a `t_` prefix and states the id equals the value propagated through logs, spans, and problem responses, while api-reference.md's example and the "matches OTel trace" requirement imply a bare 32-hex OpenTelemetry trace id. S1 implements 32-hex lowercase for OTel compatibility; reconcile the two documents (either drop the `t_` prefix from the Trace row, or distinguish the Trace entity id from the propagated correlation id)

## Todo

> **Gate C authorized (2026-07-29).** The maintainer opened the build phase; slice S1 has shipped (see Done). The slices below remain fully specified and strictly ordered, and work on each one begins only on explicit maintainer authorization. Every slice ships on its own feature branch, meets the per-slice Definition of Done ([testing-strategy.md](testing-strategy.md) §4), and ends with a tag `v0.x.0-sN`.

### S2 — Qdrant adapter + doctor v1

- [ ] TASK-0023: Implement VectorDBAdapter base interface + factory (create_collection, upsert, search, update, delete, health)
- [ ] TASK-0024: Implement Qdrant reference adapter (docker | external | remote modes, 6333/6334 both handled, prefer_grpc, API key)
- [ ] TASK-0025: Implement Qdrant system-managed Docker provisioning (named volume enforced on Windows/WSL, both ports exposed)
- [ ] TASK-0073: Implement the shared adapter contract test suite (run against Qdrant in all three modes)
- [ ] TASK-0074: Implement fasterrag doctor v1 (Docker, ports, disk, RAM/GPU, DB reachability all modes, key validity, config validity; fix-it strings; provisioning gate)

### S3 — Ingestion core

- [ ] TASK-0030: Implement contextual enrichment stage (~50–100-token chunk context, parent-doc prompt caching)
- [ ] TASK-0113: Implement the pooling half of late chunking in the embedding pool (embed the document in one long-context pass, then pool token representations over each chunk's span); the boundary half and the `late_pooling` marker already ship
- [ ] TASK-0114: Replace the estimating token counter with the embedding provider's real tokenizer once an embedding adapter is configured, so `chunking.chunk_size` counts true tokens rather than a four-characters-per-token estimate
- [ ] TASK-0115: (maintainer decision) `sentence-transformers` ships as the `huggingface` extra rather than in the core install, because it pulls a multi-gigabyte deep-learning runtime that a hosted-provider deployment should not have to download. `docs/python-api.md` was updated to match. Reverting to core is a one-line change in `pyproject.toml` if the original packaging promise should stand
- [ ] TASK-0031: Implement CPU worker pool (load/parse/chunk) with bounded queue hand-off and backpressure
- [ ] TASK-0032: Implement stateful embedding worker pool (model loaded once per worker, batched, retryable)
- [ ] TASK-0033: Implement indexer (batch dense upsert + BM25 index + metadata, deterministic chunk IDs)
- [ ] TASK-0075: Implement checkpointed journal, content-hash dedup, DLQ with reason codes, per-document status API (D3)
- [ ] TASK-0076: Implement fasterrag estimate / POST /v1/estimate preflight cost estimator (D9)

### S4 — Retrieval

- [ ] TASK-0034: Implement hybrid retrieval (dense + BM25 parallel legs), RRF fusion (k=60), metadata filter push-down

### S5 — Rerank + eval harness v1

- [ ] TASK-0035: Implement config-gated cross-encoder reranker stage (top 100–1000 → rerank → top_k)
- [ ] TASK-0047: Implement eval harness (recall@k, MRR, nDCG, faithfulness) with dataset fixtures
- [ ] TASK-0077: Implement golden-set generator + retrieval regression gate wired into CI (D7)

### S6 — Generation

- [ ] TASK-0027: Implement LLM provider adapters (OpenAI, Anthropic, Cohere, Ollama, OpenAI-compatible; streaming)
- [ ] TASK-0036: Implement context assembly (token budgeter, dedup, span-level citations)
- [ ] TASK-0037: Implement generation service with SSE streaming (meta/token/citations/usage/done/error events)
- [ ] TASK-0078: Implement grounded-or-refuse mode with faithfulness scoring and INSUFFICIENT_EVIDENCE responses (D5)

### S7 — Semantic cache

- [ ] TASK-0038: Implement embedding cache + semantic response cache (cosine threshold, TTL + corpus-change invalidation, hit/miss metrics)

### S8 — CLI complete

- [ ] TASK-0040: Implement CLI (serve, worker, ingest, query, index, provision, status, doctor, estimate, replay, benchmark, config validate)
- [ ] TASK-0079: Implement export/import portability archives + vector-copy and re-embed migration paths (D11)
- [ ] TASK-0046: Implement security layer (API-key auth with scopes, rate limiting, multi-tenancy isolation, tenant-scoped caches)

### S9 — Trace store + replay

- [ ] TASK-0041: Implement OTel instrumentation (retrieval/reranker/context-assembly/generation trace types, trace-id correlation)
- [ ] TASK-0042: Implement metrics catalogue export (RED, per-stage latency, tokens, cost/query, cache ratio, queue/DLQ depth, breaker state)
- [ ] TASK-0080: Implement local trace store + fasterrag replay with side-by-side retrieval/answer diff (D8)

### S10 — Zero-downtime reindex + lockfile

- [ ] TASK-0081: Implement blue/green reindexing with eval-gated atomic alias swap and rollback retention (D2)
- [ ] TASK-0082: Implement index.lock writing + drift detection + `index lock verify` (D1)

### S11 — Chaos, load, soak; baselines

- [ ] TASK-0083: Implement the scripted chaos suite (kill-worker, stop-Qdrant, corrupt-doc, slow-LLM, disk-full) and degradation ladder verification (D4/D12)
- [ ] TASK-0048: Implement benchmark suite per performance.md (p50/p95, throughput, cache hit rate, cost/query; --ledger output)
- [ ] TASK-0084: Run load + soak + chaos on documented reference hardware; record first benchmark-ledger entries; replace every TBD-until-measured in slo.md
- [ ] TASK-0085: Execute the disaster-recovery restore drill for real; record RPO/RTO from measurements

### S12 — Autopilot v1

- [ ] TASK-0086: Implement suggest-only eval-driven auto-tuning with measured deltas; assert zero writes to config.yaml (D6)

### S13 — Langfuse + Grafana integrations

- [ ] TASK-0043: Implement Langfuse auto-provisioning (compose stack, secrets generated once and preserved, LANGFUSE_INIT_* headless bootstrap without double quotes, doctor-gated, return http://host:3000, zero code changes at toggle time)
- [ ] TASK-0044: Implement Grafana auto-provisioning (provisioning-as-code datasources/dashboards, editable:false, allowUiUpdates:false, 30 s reload)

### S14 — Observability dashboard (last; after Langfuse proves the trace pipeline)

- [ ] TASK-0045: Implement read-only self-hosted dashboard (cache stats, tokens, costs, latencies, full LLM I/O history; zero mutating routes, asserted by test)

### Cross-slice (scheduled opportunistically after their dependencies)

- [ ] TASK-0049: Implement remaining vector DB adapters (Milvus, Weaviate, Pinecone, pgvector, Chroma) passing the contract suite
- [ ] TASK-0050: Ship Docker deployment artifacts per deployment.md (compose profiles, sizing presets)
- [ ] TASK-0087: Publish the fasterrag package to PyPI (wheel, extras, hash-locked deps, SBOM) at the first tagged beta

## Future

- [ ] TASK-0051: Multi-modal ingestion (images, audio) exploration
- [ ] TASK-0052: Distributed multi-node worker orchestration (Kubernetes operator)
- [ ] TASK-0053: Additional adapters (Elasticsearch, Vespa, LanceDB) via community entry-point contributions
- [ ] TASK-0054: GraphRAG / knowledge-graph retrieval exploration
- [ ] TASK-0055: Managed cloud offering feasibility study (post-beta)
