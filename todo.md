# todo.md — Universal Task File

> **This is the ONLY task file in fasterRag. No other todo/task/tracking file may ever be created. All tasks live here.**
>
> Rules:
> - Open task: `- [ ] TASK-0001: <description>`
> - Completed task: `- [x] TASK-0001: <description> — ✅ YYYY-MM-DD`
> - **Append-only rule:** when a task is completed, tick the checkbox and append the completion date in `YYYY-MM-DD` format. After ticking, that entry is append-only and must NOT be edited further.
> - Task IDs are sequential and never reused.

## Done

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

## In Progress

_(empty)_

## Pending

- [ ] TASK-0019: Review documentation set against Section 10 acceptance criteria and fix gaps
- [ ] TASK-0020: Decide beta version number and stamp CHANGELOG Unreleased → 0.1.0-beta.1 on first release

## Todo

Beta build tasks derived from the architecture (each ships incrementally on its own feature branch):

- [ ] TASK-0021: Scaffold repository per structure.md (pyproject, src layout, tooling: ruff, mypy, pytest)
- [ ] TASK-0022: Implement config loader (pydantic-settings + YAML source, .env secrets, fail-fast validation)
- [ ] TASK-0023: Implement VectorDBAdapter base interface + factory (create_collection, upsert, search, update, delete, health)
- [ ] TASK-0024: Implement Qdrant reference adapter (docker | external modes, 6333/6334, prefer_grpc, API key)
- [ ] TASK-0025: Implement Qdrant system-managed Docker provisioning (named volume on Windows/WSL, both ports exposed)
- [ ] TASK-0026: Implement embedding provider adapters (OpenAI, Cohere, HuggingFace/sentence-transformers, Ollama) + tiering
- [ ] TASK-0027: Implement LLM provider adapters (config-selected, streaming support)
- [ ] TASK-0028: Implement parsing pipeline (PDF incl. tables/OCR, HTML, Markdown, DOCX, TXT, CSV/JSON)
- [ ] TASK-0029: Implement chunking pipeline (fixed, recursive, semantic, layout-aware, late chunking)
- [ ] TASK-0030: Implement contextual enrichment stage (~50–100-token chunk context, parent-doc prompt caching)
- [ ] TASK-0031: Implement CPU worker pool (load/parse/chunk) with bounded queue hand-off
- [ ] TASK-0032: Implement stateful embedding worker pool (model loaded once per worker, batched, backpressure)
- [ ] TASK-0033: Implement indexer (dense upsert + BM25 index + metadata, dedup, versioning)
- [ ] TASK-0034: Implement hybrid retrieval (dense + BM25 parallel legs) with RRF fusion (k=60)
- [ ] TASK-0035: Implement cross-encoder reranker stage (top 100–1000 → rerank → top-K)
- [ ] TASK-0036: Implement context assembly (token budgeter, dedup, citations)
- [ ] TASK-0037: Implement generation service with SSE streaming
- [ ] TASK-0038: Implement embedding cache and semantic response cache (cosine threshold, TTL + event invalidation, hit/miss metrics)
- [ ] TASK-0039: Implement REST API (ingest, query, collections CRUD, health/readiness, admin/provisioning)
- [ ] TASK-0040: Implement CLI (ingest, query, index, provision, status, benchmark, serve, worker)
- [ ] TASK-0041: Implement OTel instrumentation (retrieval/reranker/context-assembly/generation trace types)
- [ ] TASK-0042: Implement metrics catalogue export (latency, tokens, cost/query, cache hit ratio, error rates)
- [ ] TASK-0043: Implement Langfuse auto-provisioning (compose stack, secrets generation, LANGFUSE_INIT_* headless bootstrap, return http://host:3000)
- [ ] TASK-0044: Implement Grafana auto-provisioning (provisioning-as-code datasources/dashboards, read-only UI)
- [ ] TASK-0045: Implement observability dashboard (read-only inspection UI: cache stats, tokens, costs, latencies, LLM I/O history)
- [ ] TASK-0046: Implement security layer (API-key auth, multi-tenancy isolation, tenant-scoped collections)
- [ ] TASK-0047: Implement eval harness (recall@k, MRR, nDCG, faithfulness) + CI quality gates
- [ ] TASK-0048: Implement benchmark suite per performance.md (p50/p95, throughput, cache hit rate, cost/query)
- [ ] TASK-0049: Implement remaining vector DB adapters (Milvus, Weaviate, Pinecone, pgvector, Chroma)
- [ ] TASK-0050: Ship Docker deployment artifacts per deployment.md (compose files, sizing presets)

## Future

- [ ] TASK-0051: Multi-modal ingestion (images, audio) exploration
- [ ] TASK-0052: Distributed multi-node worker orchestration (Kubernetes operator)
- [ ] TASK-0053: Additional adapters (Elasticsearch, Vespa, LanceDB) via community contributions
- [ ] TASK-0054: GraphRAG / knowledge-graph retrieval exploration
- [ ] TASK-0055: Managed cloud offering feasibility study (post-beta)
