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
- [x] TASK-0075: Implement checkpointed journal, content-hash dedup, DLQ with reason codes, per-document status API (D3) — journal, dedup, DLQ, and the query methods; the REST endpoints land with the ingest router — ✅ 2026-07-30
- [x] TASK-0031: Implement CPU worker pool (load/parse/chunk) with bounded queue hand-off and backpressure — ✅ 2026-07-30
- [x] TASK-0032: Implement stateful embedding worker pool (model loaded once per worker, batched, retryable) — ✅ 2026-07-30
- [x] TASK-0117: Decide the sparse-retrieval design before the indexer lands — resolved as ADR-0007: BM25 lives in the vector database as sparse vectors, with term encoding in fasterRag and IDF delegated to the backend — ✅ 2026-07-30
- [x] TASK-0033: Implement indexer (batch dense upsert + BM25 index + metadata, deterministic chunk IDs) — ✅ 2026-07-30
- [x] TASK-0118: Wire the two pools into an ingestion service that owns the job lifecycle — create the job, run both pools concurrently against one bounded queue, checkpoint as documents complete, and mark the job completed, failed, or partial — ✅ 2026-07-30
- [x] TASK-0076: Implement fasterrag estimate / POST /v1/estimate preflight cost estimator (D9) — the estimation service; the CLI and REST surfaces land with their own slices — ✅ 2026-07-30

### Slice S4 — retrieval (build phase)

- [x] TASK-0034: Implement hybrid retrieval (dense + BM25 parallel legs), RRF fusion (k=60), metadata filter push-down — ✅ 2026-07-30

### Slice S5 — rerank and eval harness (build phase, in progress)

- [x] TASK-0047: Implement eval harness (recall@k, MRR, nDCG, faithfulness) with dataset fixtures — retrieval metrics and the golden-set format; faithfulness needs the LLM call site and lands with the generation slice — ✅ 2026-07-30
- [x] TASK-0035: Implement config-gated cross-encoder reranker stage (top 100–1000 → rerank → top_k) — ✅ 2026-07-30
- [x] TASK-0121: Verify the embedder and reranker against real downloaded models — `tests/eval/test_real_models.py`, marked `eval`, wired into CI as its own job; it found and fixed a sentence-transformers 5.x accessor deprecation — ✅ 2026-07-30
- [x] TASK-0122: Implement the retrieval regression gate (D7) — tolerance comparison against a committed baseline, refusing a baseline recorded under a different embedding model or retrieval configuration, and blocking rather than passing when none exists — ✅ 2026-07-30

### Slice S6 — generation (build phase, in progress)

- [x] TASK-0036: Implement context assembly (token budgeter, dedup, span-level citations) — ✅ 2026-07-30
- [x] TASK-0027: Implement LLM provider adapters (OpenAI, Anthropic, Cohere, Ollama, OpenAI-compatible; streaming) — ✅ 2026-07-30

### Workflow

- [x] TASK-0108: Land slice S1 on main — ✅ 2026-07-30
- [x] TASK-0109: Land slice S2 on main — ✅ 2026-07-30
- [x] TASK-0112: Adopt trunk-based development on main and update CLAUDE.md, CONTRIBUTING.md, and the revert playbook (maintainer instruction) — ✅ 2026-07-30

### First formal audit (2026-08-02)

- [x] TASK-0169: Execute the first formal production-grade audit — gates re-run green (ruff, format, mypy strict over 216 files, 1224 unit tests, 87% branch coverage on the gated packages); progress scored 68/100 and implementation-vs-requirements 71/100; findings filed as TASK-0155–TASK-0168 plus two maintainer-decision items (TASK-0164/0165); stale-state corrections applied across CLAUDE.md, README, reliability, api-reference, glossary, failure-modes, quickstart, cookbook, python-api, integrations, data-model, deployment, and CONTRIBUTING; AUDIT-0006/0007 resolved; ledger hygiene repaired (duplicate S2 block removed, duplicated TASK-0110 renumbered to TASK-0155) — ✅ 2026-08-02

## In Progress

_(empty)_

## Pending

- [ ] TASK-0020: Decide beta version number and stamp CHANGELOG Unreleased → 0.1.0-beta.1 on first release
- [ ] TASK-0098: (maintainer action, GitHub settings) Enable branch protection on main — require PRs, block direct pushes
- [ ] TASK-0111: (maintainer action) Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2` on main
- [x] TASK-0155: Register the vector database health check with `/readyz` — already wired by `_vector_db_check` in `api/main.py`; the task text was stale, carried forward from the renumbering. Verified live rather than by reading: healthy returns `200` with `vector_db reachable in 17.982 ms`; with the Qdrant container stopped `/readyz` returns `503 NOT_READY` naming `vector_db` and `qdrant was unreachable during health: ResponseHandlingException`; restarting recovers to `200`. `tests/unit/api/test_health.py` already covers the failing path — ✅ 2026-08-02
- [x] AUDIT-0007: deployment.md states every provisioned container runs non-root, but the Qdrant provisioner does not pass `--user` — the official image expects to own its storage volume and forcing a uid risks an unstartable container. Decide whether to verify a working non-root uid for the pinned image or to narrow the documented claim to the containers fasterRag builds itself — resolved by narrowing the claim in deployment.md — ✅ 2026-08-02
- [x] AUDIT-0006: `trace_id` format conflict — data-model.md gives Trace ids a `t_` prefix and states the id equals the value propagated through logs, spans, and problem responses, while api-reference.md's example and the "matches OTel trace" requirement imply a bare 32-hex OpenTelemetry trace id. S1 implements 32-hex lowercase for OTel compatibility; reconcile the two documents — resolved by dropping the `t_` prefix from data-model.md, matching the implementation — ✅ 2026-08-02
- [ ] TASK-0164: (maintainer decision) Write an ADR making the license choice explicit — GPL-3.0-or-later (current) vs a permissive license, given the adoption thesis is embedding fasterRag inside other products; either answer is fine, undocumented default is not
- [ ] TASK-0165: (maintainer decision) ADR-0008 on degradation-ladder scope — implement the circuit breaker (TASK-0148) and `cache_only` rung (TASK-0159) as specified, or formally narrow D4 to the two shipped rungs; reliability.md, api-reference.md, glossary.md, and failure-modes.md carry specified-not-built annotations until this is decided

## Todo

> **Gate C authorized (2026-07-29).** The maintainer opened the build phase; slices S1–S13 have landed (see Done and the ticked entries below). Remaining work is fully specified and strictly ordered, and a new slice begins only on explicit maintainer authorization. Work lands directly on `main` (trunk-based, maintainer instruction 2026-07-30) — every commit leaves `main` green, meets the per-slice Definition of Done ([testing-strategy.md](testing-strategy.md) §4), and slice boundaries are tagged `v0.x.0-sN`. *(The duplicate unticked S2 block that previously sat here was removed 2026-08-02 — those five tasks were already ticked in Done under their same ids.)*

### S3 — Ingestion core

- [ ] TASK-0030: Implement contextual enrichment stage (~50–100-token chunk context, parent-doc prompt caching)
- [ ] TASK-0113: Implement the pooling half of late chunking in the embedding pool (embed the document in one long-context pass, then pool token representations over each chunk's span); the boundary half and the `late_pooling` marker already ship
- [ ] TASK-0114: Replace the estimating token counter with the embedding provider's real tokenizer once an embedding adapter is configured, so `chunking.chunk_size` counts true tokens rather than a four-characters-per-token estimate
- [ ] TASK-0115: (maintainer decision) `sentence-transformers` ships as the `huggingface` extra rather than in the core install, because it pulls a multi-gigabyte deep-learning runtime that a hosted-provider deployment should not have to download. `docs/python-api.md` was updated to match. Reverting to core is a one-line change in `pyproject.toml` if the original packaging promise should stand
- [ ] TASK-0119: Rebuilding a collection is the only way to add a keyword leg — Qdrant cannot attach a sparse vector to an existing dense-only collection, and the two layouts (unnamed dense versus named dense plus named sparse) are not interchangeable. Flipping `retrieval.hybrid` on an indexed corpus therefore needs the D2 blue/green reindex path, and until that exists the adapter refuses the mismatch rather than dropping the sparse vector silently
- [x] TASK-0116: Expose the per-document status endpoints over REST (`GET /v1/ingest/{job_id}`, `GET /v1/ingest/{job_id}/documents?status=…`, `POST /v1/ingest/{job_id}/retry-dlq`); the journal already answers these queries — ✅ 2026-08-01
- [x] TASK-0128: Implement the ingest, query, retrieve, collections, and admin routers, completing the REST half of the control plane — ✅ 2026-08-01
- [ ] TASK-0129: `POST /v1/ingest` accepts only `path` sources; `url` and `inline` are documented in api-reference.md but rejected with `VALIDATION_FAILED` naming what is supported, because no loader fetches them yet. Ships with the source loader
- [ ] TASK-0130: Ingestion runs as an in-process background task rather than through a queue backend. The documented contract (`202` + job id, poll `GET /v1/ingest/{job_id}`) is satisfied identically by a distributed queue, so moving to one changes no response — but a restart mid-job currently relies on journal resume rather than queue redelivery. Maintainer decision on the queue backend

### S5 — Rerank + eval harness v1

- [x] TASK-0077: Implement the golden-set generator (P4) — stratified sampling, adversarial records, `source: autopilot` provenance; verified against a real 21-chunk PDF corpus producing 10/10 records with 2 adversarial and 0 dropped — ✅ 2026-08-01
- [x] TASK-0139: Commit a golden set and a measured baseline for a fixture corpus under `tests/eval/datasets/` — the `handbook` fixture: 6 documents, 15 hand-authored records (12 answerable, 3 adversarial), document-level ground truth, and a baseline measured against a real Qdrant at recall@5 1.0 / MRR 1.0 / nDCG@5 1.0 — ✅ 2026-08-01
- [x] TASK-0141: On Windows a document id hashes the source path string, so the same file ingested as `d:\...` and `D:\...` gets two different document ids. Content-hash dedup still prevents a double index, but `index.lock` document hashes and drift detection would treat them as different documents. Found while building the eval baseline, where a resolved-vs-globbed path produced a recall of 0.0 that looked like a retrieval failure. Consider normalising the source URI before hashing, platform-aware — `normalise_source` folds Windows paths (drive letter, separator, and case) before hashing, so `d:\docs.md`, `D:\docs.md`, and `D:/docs/a.md` are one document. URLs are left untouched: their paths are case-sensitive by specification, and folding one would merge two genuinely different resources. POSIX paths keep their case for the same reason — ✅ 2026-08-03
- [ ] TASK-0142: The handbook baseline detects large regressions but not small ones. Verified to catch an index missing four of six documents (recall 1.0 → 0.3333, gate blocked, exit 5), but at six documents and k=5 a retriever returns five of six, so a *subtle* ranking regression would pass. A discriminating baseline needs a larger fixture with topically overlapping documents
- [ ] TASK-0143: Dropping a collection without clearing the journal makes a re-ingest a correct no-op — the content hashes are remembered per collection, so nothing is re-indexed and the collection is never recreated. That is D3 behaving as designed, but it is a sharp edge during recovery; `disaster-recovery.md` §4 documents restore-from-snapshot as the path, and should say explicitly that re-ingest is not an alternative unless the journal is cleared too
- [x] TASK-0140: Expose golden-set generation on the CLI (`fasterrag autopilot generate-golden-set` or similar). Today it is a library call only, so the shared machinery Autopilot (D6) depends on is reachable from Python but not from the terminal — `fasterrag autopilot generate-golden-set <sources...>` parses and chunks the corpus exactly as ingestion would, so no collection is needed and none is read; a golden set is usually wanted before the index exists. Refuses to overwrite an existing file, since a generated set is hand-curated afterwards. Verified live against the handbook corpus: 6 records, 1 adversarial, 0 dropped, with sensible questions — ✅ 2026-08-03
- [ ] TASK-0120: Add the faithfulness metric to the eval harness once the generation slice provides the P3 call site; the harness reports retrieval metrics only until then

### S6 — Generation

- [x] TASK-0037: Implement generation service with SSE streaming (meta/token/citations/usage/done/error events) — ✅ 2026-07-31
- [x] TASK-0078: Implement grounded-or-refuse mode with faithfulness scoring and INSUFFICIENT_EVIDENCE responses (D5) — ✅ 2026-07-31
- [ ] TASK-0123: Maintainer review — no doc specified how grounded-or-refuse behaves while streaming, and refusing after tokens have been sent is impossible. Implemented as: with `generation.grounded_or_refuse: true` the stream buffers the answer, grades it, then emits either the whole answer or one `insufficient_evidence` event followed by `done`. This trades time-to-first-token for the guarantee, only when the flag is on. Recorded in `docs/api-reference.md` §Streaming semantics; confirm or choose the alternative (stream tokens and emit a retraction event)

### S7 — Semantic cache

- [x] TASK-0038: Implement embedding cache + semantic response cache (cosine threshold, TTL + corpus-change invalidation, hit/miss metrics) — ✅ 2026-07-31
- [ ] TASK-0124: Implement the `redis` cache backend behind a `.[redis]` extra. `embeddings.cache.backend` and `cache.backend` both accept `redis` per config-reference.md, but only `memory` and `disk` are implemented; selecting `redis` raises a `ConfigError` naming the alternatives rather than silently falling back. Needs a Docker Redis for the integration test

- [x] TASK-0125: The error-code table has no code for "the vector database rejected our credentials". The Qdrant adapter currently reports `EMBED_PROVIDER_ERROR`, documented as "Embedding provider timeout / hard failure", so a vector-DB auth failure surfaces under an embedding label — misleading when grepping logs. `AUTH_INVALID` is reserved for fasterRag's own API auth and `RETRIEVAL_FAILED` is documented retryable, which this is not. Add a `VECTOR_DB_AUTH` code to `errors.py` and `api-reference.md`, then use it in `_auth_error`. Found by running `fasterrag index list` against a key-protected Qdrant, 2026-07-31 — added `VECTOR_DB_AUTH_FAILED` (503, non-retryable). Sending an operator to check their embedding provider over a rejected Qdrant key is a wrong answer with a confident tone; the taxonomy exists to point at the right knob — ✅ 2026-08-03

### S8 — CLI complete

- [x] TASK-0040: Implement CLI (serve, worker, ingest, query, index, provision, status, doctor, estimate, config validate) — `replay`, `benchmark`, `export`, `import`, and `autopilot` are registered but report the slice that ships them, since their services do not exist yet — ✅ 2026-07-31
- [ ] TASK-0126: Maintainer review — `VectorDBAdapter` gained `list_collections()` and `drop_collection()` under TASK-0040, with a vendor-neutral `CollectionInfo` result. `adapters/` is a folder-boundary-sensitive package; the addition is judged to strengthen the boundary rather than weaken it, because the documented `fasterrag index list` and `index delete` are otherwise unimplementable without the CLI reaching into a vendor client directly. Both are covered by the shared contract suite. Confirm or revert
- [x] TASK-0127: The semantic cache is unusable from the CLI: `cache.backend` accepts only `memory` and `redis`, and a memory cache dies with each short-lived CLI process, so no two `fasterrag query` invocations can ever share one. Verified in-process instead (16.7s cold vs 69ms on a hit against a real corpus). TASK-0124's redis backend resolves it; until then consider documenting in cookbook.md that the semantic cache benefits `fasterrag serve` and the library, not one-shot CLI calls — `cache.backend` now accepts `disk`, the only backend that survives between two invocations of a command-line tool. `memory` remains the default for the server, and `redis` still fails fast until TASK-0124 — ✅ 2026-08-03
- [ ] TASK-0079: Implement export/import portability archives + vector-copy and re-embed migration paths (D11)
- [ ] TASK-0046: Implement security layer (API-key auth with scopes, rate limiting, multi-tenancy isolation, tenant-scoped caches)

### S9 — Trace store + replay

- [x] TASK-0041: Implement OTel instrumentation (retrieval/reranker/context-assembly/generation trace types, trace-id correlation) — the four span types are recorded against one clock origin and correlated by trace id; OTLP export itself is TASK-0132 — ✅ 2026-08-01
- [x] TASK-0042: Implement metrics catalogue export (RED, per-stage latency, tokens, cost/query, cache ratio, queue/DLQ depth, breaker state) — every documented metric declared and served at `GET /metrics`; the gauges fed by the worker pools and circuit breaker are declared but only populated once those slices report into them — ✅ 2026-08-01
- [x] TASK-0080: Implement local trace store + fasterrag replay with side-by-side retrieval/answer diff (D8) — ✅ 2026-08-01
- [ ] TASK-0131: `fasterrag traces list|show` was added beyond `docs/cli-reference.md`, because a 32-hex trace id is unreachable without a way to list recent ones. Documented in cli-reference.md; confirm the addition
- [ ] TASK-0132: OTel export is not wired. The four RAG spans are recorded and persisted by the trace store, which is what replay and the dashboard read, but `observability.otel: true` does not yet emit them over OTLP to `observability.otel_endpoint`. Needs the opentelemetry SDK dependency and an exporter

### S10 — Zero-downtime reindex + lockfile

- [x] TASK-0081: Implement blue/green reindexing with eval-gated atomic alias swap and rollback retention (D2) — the alias swap, retention, and rollback are complete and verified against a real Qdrant; the eval gate reports that it *could not run* rather than passing, because scoring a green build needs TASK-0077's golden-set harness — ✅ 2026-08-01
- [ ] TASK-0133: Maintainer review — `VectorDBAdapter` gained `set_alias`, `alias_target`, and `delete_alias` under TASK-0081. Aliases are the primitive zero-downtime reindexing is built on and cannot be emulated above the adapter, so the contract is the only place they can live; all three are covered by the shared contract suite, including that a swap is atomic and an alias is searchable as if it were the collection. Same boundary question as TASK-0126; confirm or revert
- [x] TASK-0134: Wire the eval gate into `index reembed` — `--dataset` scores the green collection before the swap; without it the swap still records `gate_ran: false` rather than a pass — ✅ 2026-08-02
- [x] TASK-0082: Implement index.lock writing + drift detection + `index lock verify` (D1) — ✅ 2026-08-01

### S11 — Chaos, load, soak; baselines

- [x] TASK-0083: Implement the scripted chaos suite (kill-worker, stop-Qdrant, corrupt-doc, slow-LLM, disk-full) and degradation ladder verification (D4/D12) — all five scenarios scripted and passing; observed behavior recorded in failure-modes.md — ✅ 2026-08-01
- [ ] TASK-0135: The chaos suite injects the stop-Qdrant and disk-full faults at the adapter and filesystem seams rather than by stopping a real container or filling a real disk. That verifies fasterRag's response but not the operating system's reporting of those conditions; add container-stop and disk-quota variants under the `integration` marker
- [ ] TASK-0136: Recovery time per chaos scenario is not measured — D12's proof metric asks for it, and it needs the isolated hardware TASK-0084 is blocked on
- [x] TASK-0137: The error-code table has no timeout code for generation. Embeddings distinguish `EMBED_PROVIDER_TIMEOUT` from `EMBED_PROVIDER_ERROR`, but a generation timeout can only be reported as `GENERATION_FAILED`, so `reliability.timeouts.llm_ms` firing is indistinguishable from any other provider failure in logs and metrics. Consider `LLM_TIMEOUT`, alongside TASK-0125 — added `GENERATION_TIMEOUT` (504, retryable), mirroring the embedding split. A timeout says raise `reliability.timeouts.llm_ms` or shrink the context; a hard failure says look at the provider, and one code for both cannot say which — ✅ 2026-08-03
- [x] TASK-0048: Implement benchmark suite per performance.md (p50/p95, throughput, cache hit rate, cost/query; --ledger output) — ingest and query suites; the eval suite needs TASK-0077's golden set — ✅ 2026-08-01
- [ ] TASK-0084: Run load + soak + chaos on documented reference hardware; record first benchmark-ledger entries; replace every TBD-until-measured in slo.md — BLOCKED on isolated hardware. BENCH-0001 and BENCH-0002 are committed but explicitly marked not citable: they were taken on a developer laptop with Docker, an IDE, and other co-tenant load, failing ledger rule 5's isolation requirement. Run-to-run variance exceeded the effect being measured (p50 5.5 s vs 8.7 s across two invocations of the same commit), and cold start came out *faster* than the warmed median — both symptoms of the missing isolation. slo.md's TBDs stay TBD until superseding entries come from a quiet machine
- [ ] TASK-0085: Execute the disaster-recovery restore drill for real; record RPO/RTO from measurements — PARTIALLY EXECUTED 2026-08-01. Steps 4, 5, and 7 ran for real against a live Qdrant: a collection was snapshotted, deleted, restored, and verified to answer with an identical citation. Steps 1–3 (clean host, `.env`, doctor) and step 6 (`index lock verify`) did not run, because the restore was onto the same host — that makes it the §4 single-collection shortcut, not the §2 clean-host drill. RPO/RTO stay TBD: the measured 4,311 ms is a restore duration for one tiny collection, not `T1 − T0` from a clean host. Execution log in disaster-recovery.md
- [ ] TASK-0138: Implement backup/restore for the remaining §4 shortcuts and a documented default schedule (daily snapshots retained 14 days); today `fasterrag backup` is a manual command with no cadence, so the documented default is a statement about tooling that does not exist yet

### S12 — Autopilot v1

- [x] TASK-0086: Implement suggest-only eval-driven auto-tuning with measured deltas; assert zero writes to config.yaml (D6) — `fasterrag autopilot run` searches query-time parameters against a golden set, writes `autopilot-suggestion.yaml`, and verifies config.yaml is byte-identical afterwards; searching index-time parameters is TASK-0145 — ✅ 2026-08-02
- [ ] TASK-0144: D6's acceptance test — "on a fixture corpus with a known-better config, autopilot's suggestion matches or beats it" — is only verified by unit test, not against a real corpus. On the handbook fixture every candidate scores 1.0, so there is no improvement to find and Autopilot correctly suggests nothing. Demonstrating the improvement path end to end needs the harder fixture of TASK-0142
- [ ] TASK-0145: Autopilot searches query-time parameters only. Chunk size, overlap, and the embedding model change how the index is *built*, so each candidate costs a full re-chunk and re-embed of the corpus — hours on a real one. That search needs its own budget model and a temporary collection per candidate

### S13 — Langfuse + Grafana integrations

- [x] TASK-0043: Implement Langfuse auto-provisioning (compose stack, secrets generated once and preserved, LANGFUSE_INIT_* headless bootstrap without double quotes, doctor-gated, return http://host:3000, zero code changes at toggle time) — verified end to end: the six-container stack came up, `/api/public/health` returned `{"status":"OK","version":"3.224.4"}`, and the bootstrapped keys authenticated against `/api/public/projects`, returning the `fasterRag` org and project with no quote characters in either name; doctor-gating is TASK-0149 — ✅ 2026-08-02
- [x] TASK-0044: Implement Grafana auto-provisioning (provisioning-as-code datasources/dashboards, editable:false, allowUiUpdates:false, 30 s reload) — verified end to end: fasterRag → Prometheus → Grafana's provisioned datasource returned real series — ✅ 2026-08-02
- [x] TASK-0154: The provisioned dashboard rendered six empty panels. Three causes, all fixed: five catalogue metrics (`queue_depth`, `dlq_depth`, `ingest_documents_total`, `ingest_throughput`, `cost_usd_total`) were declared but written by nothing; the Prometheus host port and container port had been collapsed into one constant, binding 9099:9099 where nothing listens; and no panel carried a `legendFormat`, so each legend rendered the whole label set and truncated before the only label that varied. Verified live: ingest and query over HTTP moved every wired series, and Grafana's datasource proxy returned them — ✅ 2026-08-02
- [x] TASK-0146: `observability.grafana: true` does not yet trigger provisioning at startup — `fasterrag provision grafana` does. The doc frames the toggle as the trigger, and the config-driven path should call the same provisioner so flipping the flag is genuinely all it takes — `provision_enabled_observability` runs in the API lifespan, so the toggle really is the trigger the doc describes. A provisioning failure is logged and swallowed: refusing to serve queries because a dashboard would not start inverts the dependency — ✅ 2026-08-03
- [x] TASK-0147: Grafana provisioning is not doctor-gated. The Qdrant provisioner checks `fasterrag doctor` before mutating anything; the Grafana one should use the same gate so a port conflict is reported with a fix rather than as a container that fails to start — `require_provisioning_gate` extracted from the Qdrant provisioner and called by Grafana before any manifest is written — ✅ 2026-08-03
- [ ] TASK-0148: `fasterrag_circuit_state` is the one catalogue metric nothing writes, because no circuit breaker exists — only its configuration under `reliability.circuit_breaker`. The dashboard's circuit panel is empty by design until the breaker in `docs/reliability.md` §3 is implemented, and says so in its panel description; `test_no_catalogue_metric_is_declared_and_never_written` pins this as the only exemption
- [x] TASK-0149: Langfuse provisioning is not doctor-gated either, and needs the same gate as TASK-0147. Its eight published ports are the widest surface any toggle claims, and a conflict currently surfaces as a Compose failure after some containers have already started — gated before any secret is generated, since a conflict discovered mid-`compose up` leaves some of the eight containers running and some not — ✅ 2026-08-03
- [x] TASK-0150: `observability.langfuse: true` does not trigger provisioning at startup — `fasterrag provision langfuse` does. Same gap as TASK-0146 on the Grafana side, and both should be fixed by the same config-driven path — same startup path as TASK-0146, both toggles handled together — ✅ 2026-08-03
- [ ] TASK-0151: Nothing exports traces to the provisioned Langfuse yet. The stack is stood up, bootstrapped, and reachable, but `core/tracing.py` still only writes to the local trace store — the export path is what S14 means by "after Langfuse proves the trace pipeline"
- [ ] TASK-0152: `GENERATION_PRICES_USD_PER_MILLION_TOKENS` covers only `gpt-4o-mini` and `gpt-4o`. Every other generation model contributes nothing to `fasterrag_cost_usd_total`, so the cost panel silently understates spend rather than reporting an unknown. The table needs the remaining configured providers, and the panel needs a way to say "some traffic is unpriced". (The metric itself is now verified live: a real query recording 386 prompt and 51 completion tokens on `gpt-4o-mini` published `8.85e-05`, matching `386×0.15/1e6 + 51×0.60/1e6` exactly. Superseded in scope by TASK-0168.)
- [x] TASK-0153: Filed on a mis-read: a query was reported as hanging past ten minutes. It does not. Measured on the same deployment, a single query returns `http=200` in **10.78 s**; the ten minutes were my own shell budget — an ingest, a 25 s sleep, and three curls each capped at `-m 120` — not one unbounded call. `reliability.timeouts.llm_ms` is applied by the adapter at construction (`adapters/llm/base.py`), and the openai adapter passes it to its client. No defect; entry kept rather than deleted so the ledger records the correction — ✅ 2026-08-02

### S14 — Observability dashboard (last; after Langfuse proves the trace pipeline)

- [ ] TASK-0045: Implement read-only self-hosted dashboard (cache stats, tokens, costs, latencies, full LLM I/O history; zero mutating routes, asserted by test)

### Cross-slice (scheduled opportunistically after their dependencies)

- [ ] TASK-0049: Implement remaining vector DB adapters (Milvus, Weaviate, Pinecone, pgvector, Chroma) passing the contract suite — pgvector first: it proves the contract against a genuinely different (SQL) paradigm and is the cheapest to CI
- [ ] TASK-0050: Ship Docker deployment artifacts per deployment.md (compose profiles, sizing presets)
- [ ] TASK-0087: Publish the fasterrag package to PyPI (wheel, extras, hash-locked deps, SBOM) at the first tagged beta. Distribution mechanics are now verified: `python -m build` produces both artifacts, `twine check` passes on both, and the wheel installs into a clean venv and runs. **Remaining blockers before a first publish** — TASK-0164 (license decision; the only irreversible one, since a PyPI version number can never be reused), TASK-0020 (beta version stamp), TASK-0158 (lockfile, SBOM, secret scan)
- [x] TASK-0170: `pip install fasterrag` was unusable out of the box — every command needs a `config.yaml`, and the missing-config error told the user to "copy the canonical config.yaml from the repository root", which an installed package does not have. Added `fasterrag config init`: it writes the canonical config plus `.env.example`, force-includes both into the wheel so the template cannot drift from the documented one, refuses to overwrite without `--force`, and never writes `.env` itself. Verified against a built wheel in a clean venv from an empty directory: `doctor` → names the command → `config init` → `config validate` passes — ✅ 2026-08-02

### Audit follow-ups (first formal audit, 2026-08-02)

- [x] TASK-0156: Fail fast on accepted-but-unenforced config — `security.auth`, `security.multi_tenancy`, and both `cost.*_token_budget` keys validate today and are consumed by nothing; enabling any of them must raise `ConfigError` naming the missing slice (mirror the implemented `cache.backend: redis` pattern) until enforcement lands with TASK-0046 — `_reject_unenforced_settings` runs in `load_settings` before the env-var check, lists every enabled key at once so an operator learns all of them in one restart, and leaves `0` (the documented "unlimited") accepted; the canonical config is asserted to still start — ✅ 2026-08-02
- [x] TASK-0157: Enforce the coverage gate in CI — `--cov-fail-under=85` scoped to `core/`, `adapters/`, `workers/`; the 87% measured today is met by discipline, not by a gate — added as its own CI step after the unrestricted run. Scoped deliberately: a repository-wide threshold is diluted by CLI plumbing and API wiring, so it can stay green while the retrieval and adapter code it exists to protect regresses. Re-measured before setting it at 87% branch coverage over 3525 statements — ✅ 2026-08-02
- [ ] TASK-0158: Supply chain per security.md §6 — **partially done 2026-08-03**: a `supply-chain` CI job now runs gitleaks over full history (a secret committed and later removed is still compromised, so scanning only the tip would call it clean) and pip-audit as advisory-only. Still open: the uv lockfile with `uv sync --locked`, which is also what makes pip-audit safe to make blocking, and the SBOM (TASK-0087)
- [ ] TASK-0159: Implement the `cache_only` degradation rung — consult the semantic cache when retrieval raises, serving `degraded: true, mode: cache_only` instead of a bare `RETRIEVAL_FAILED`; pairs with the breaker (TASK-0148) and the scope decision (TASK-0165)
- [x] TASK-0160: Add a Windows CI leg running the fast suite — the project is developed on Windows and TASK-0141's path-case defect is exactly the class ubuntu-only CI cannot catch — added as a `windows-latest` job running the fast suite only; lint, format, and typecheck are not repeated because they are platform-independent and already gate on the quality job. Two shipped defects were Windows-only: the path-case document id and the UTF-8 BOM the config loader rejected — ✅ 2026-08-02
- [x] TASK-0161: Add a nightly scheduled CI job running the eval and benchmark suites with ledger-format artifacts, so the provable-claims policy is infrastructure rather than habit — `.github/workflows/nightly.yml` runs the eval and benchmark suites at 03:00 and uploads ledger-format artifacts with 30-day retention. The workflow states in its own header that its numbers are **not citable**: a shared GitHub runner has neighbours, and performance.md requires an isolated documented machine. It is a regression signal; TASK-0084 still needs real hardware — ✅ 2026-08-03
- [x] TASK-0162: Add a CI check diffing the FastAPI-generated OpenAPI schema against api-reference.md's endpoint table, catching endpoint drift automatically (the `traces list` drift was caught by hand) — `scripts/check_openapi_drift.py`, wired into the quality job. It found seven real discrepancies on its first run: `GET /v1/traces` and `POST /v1/retrieve` served but undocumented, and five documented endpoints nothing serves. All seven are now resolved — the two are documented, and the five carry an explicit **Not yet implemented** marker that the gate honours, so a reader sees the gap rather than the script merely tolerating it. It also catches the reverse drift: an endpoint that got built while the marker was left behind — ✅ 2026-08-03
- [x] TASK-0163: Implement the `FasterRag` facade per the python-api.md status table — `from_config`, `from_settings`, async context manager, `ingest`, `query`, `query_stream`, `retrieve`, `estimate`, `index_lock`. A composition layer only: it builds the same services `api/dependencies.py` builds, so no behavior can exist here that the REST API lacks, and a test asserts no pipeline logic crept in. Exported lazily through `__getattr__` — `import fasterrag` stays at 6 ms while resolving the facade costs 4.4 s. Verified end to end against live Qdrant and OpenAI: estimate 3 docs/189 tokens, ingest 3 indexed, retrieve 3 hits, query answered with a citation, stream 59 events `meta`→`done`, lockfile present — ✅ 2026-08-03
- [ ] TASK-0171: `FasterRag.collections`, `.doctor`, `.replay`, and `.export_archive`/`.import_archive` are still unimplemented on the facade; the CLI and REST surfaces cover them today. The python-api.md table marks each one
- [x] TASK-0172: `fasterrag.sync` blocking facade — the documented wrapper over the async one, for callers not running an event loop. Owns its event loop (created on `__enter__`, closed in a `finally` so a shutdown failure cannot leak it), refuses to start inside a running loop rather than deadlocking, and keeps `query_stream` incremental by pulling one event at a time — measured first token at 0.91 s over a 51-event response. Verified live: retrieve 3 hits, query answered in 4.30 s with a citation, stream `meta`→`done` — ✅ 2026-08-03
- [x] TASK-0174: The shipped `config.yaml` defaults to `retrieval.rerank: true` with `BAAI/bge-reranker-v2-m3`, a **2.2 GB** cross-encoder. A fresh `pip install` user's first query therefore downloads 2.2 GB and then loads it — measured as **not completing within 400 s** on a developer laptop, presenting as a hang with no output. Two things to decide: whether the out-of-box default should be a small reranker (or `rerank: false`) with the large model as an opt-in, and whether model loading should emit progress so a first run looks slow rather than broken. Found while verifying the sync facade, where it masqueraded as a facade bug — the *silent* half is fixed: the first-load message is now a **warning**, not an info line, because nothing configures logging in a plain script and Python's last-resort handler only emits warnings. It names the cost and the two ways out (`retrieval.rerank: false`, or a smaller `retrieval.reranker_model`), and the load duration is logged on completion. The **default** half is deliberately left open as TASK-0175: whether to ship a smaller reranker out of the box is a retrieval-quality decision for the maintainer, not a bug fix — ✅ 2026-08-03
- [ ] TASK-0175: (maintainer decision) The shipped default is `retrieval.rerank: true` with `BAAI/bge-reranker-v2-m3` (2.2 GB, measured at over 400 s to load on a laptop). Reranking is the single biggest quality lever in the stack, so turning it off by default trades the framework's best feature for a faster first run. The alternatives are a smaller default cross-encoder, or keeping it and treating the first-run cost as documented. TASK-0174 made the wait visible; this decides whether it should exist
- [ ] TASK-0173: Entry-point plugin groups (`fasterrag.vectordb` / `.embeddings` / `.llm`); the factories resolve built-ins only. Split out of TASK-0163, which shipped the facade half
- [x] TASK-0166: Add a CI self-truth check that fails when CLAUDE.md or README claims a documentation-only state while `src/` exists — the audit found exactly that inversion — `scripts/check_doc_truth.py`, wired into CI. Deliberately narrow: it asserts only the cheap-to-verify, expensive-to-get-wrong facts and ignores fenced code blocks, because a doc gate that fires on judgement calls gets disabled. Parameterised by root so the gate itself is tested — eight tests, including a reconstruction of the audit's actual inversion. Left out of pre-commit at the maintainer's direction; CI-only — ✅ 2026-08-02
- [x] TASK-0167: Run `fasterrag doctor --json` as a CI smoke job against the CI environment itself (dogfoods D10 on every push) — added to the quality job, and it exercises the packaged-template path with it: `config init` in an empty directory, then `doctor --json`. Placeholder keys are set so doctor gets past config validation and actually runs its environment checks; without them it would exit 4 on the first one and pass trivially. Exit 4 is accepted afterwards (CI has no Docker daemon, and reporting that is doctor working); a crash still fails the job — ✅ 2026-08-02
- [ ] TASK-0168: Turn the generation price table into dated data with an unpriced-traffic counter metric so the cost panel can say "some traffic unpriced" instead of silently understating (extends TASK-0152)

## Future

- [ ] TASK-0051: Multi-modal ingestion (images, audio) exploration
- [ ] TASK-0052: Distributed multi-node worker orchestration (Kubernetes operator)
- [ ] TASK-0053: Additional adapters (Elasticsearch, Vespa, LanceDB) via community entry-point contributions
- [ ] TASK-0054: GraphRAG / knowledge-graph retrieval exploration
- [ ] TASK-0055: Managed cloud offering feasibility study (post-beta)
