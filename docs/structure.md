# structure.md — Repository Structure

> This documents the layout for the beta build. Directories are created only when their build tasks in [todo.md](todo.md) begin, so paths marked *(pending)* do not exist yet; everything else is in the tree today.

## Proposed directory tree

```text
fasterRag/
├── CLAUDE.md                     # Always-loaded Claude Code instructions (ONLY doc kept at root)
├── README.md                     # Project overview + entry point into docs/
├── LICENSE
├── .gitignore                    # Ignores .env, caches, build artifacts
├── config.yaml                   # All behavior (no secrets) — MUST match docs/config-reference.md example
├── .env.example                  # Template for secrets; real .env is git-ignored
├── pyproject.toml                # Project metadata, deps, tooling config (ruff, mypy, pytest, coverage)
├── .pre-commit-config.yaml       # Local mirror of the CI quality gates
├── scripts/
│   └── check_commit_message.py   # Enforces the single-line, no-trailer commit rule
├── .github/                      # PR template (rule enforcement), issue templates, SECURITY.md
│   ├── dependabot.yml            # Weekly grouped update PRs: uv.lock, Actions SHA pins, Docker base digests
│   ├── scripts/                  # Release decisions, release smoke test, dependency licence check
│   └── workflows/
│       ├── ci.yml                # Blocking gates: ruff, ruff format, mypy --strict, pytest
│       └── release.yml           # v* tag: gates, build, SBOM, PyPI (trusted publishing), GHCR, GitHub release
├── docs/                         # ALL documentation lives here (rule: only CLAUDE.md stays at root)
│   ├── quickstart.md  scope.md  structure.md  flow.md  architecture.md
│   ├── config-reference.md  api-reference.md  cli-reference.md  python-api.md
│   ├── data-model.md  prompts.md  archive-format.md
│   ├── observability.md  deployment.md  security.md
│   ├── testing-strategy.md  performance.md  benchmarks.md  integrations.md
│   ├── cookbook.md  troubleshooting.md  migration-guide.md  how-to-use.md
│   ├── internals.md  rebuild-playbook.md  verification.md
│   ├── release.md  release-readiness.md
│   ├── glossary.md  references.md
│   ├── differentiators.md        # The twelve flagship capabilities (uniqueness contract)
│   ├── reliability.md            # Reliability doctrine + resilience patterns
│   ├── failure-modes.md          # FMEA table (component failure analysis)
│   ├── slo.md                    # SLIs, SLO targets (TBD-until-measured), error budget
│   ├── disaster-recovery.md      # Backups, restore drill, RPO/RTO
│   ├── CHANGELOG.md              # Keep a Changelog + SemVer
│   ├── CONTRIBUTING.md           # Contributor rules
│   ├── todo.md                   # THE one and only task file
│   ├── blockers.md               # Read-only view of the blocked subset of todo.md
│   ├── engineering-scorecard.md  # Rubric, current score, the bar for 90, working rules
│   ├── engineering-audit.md      # The audit behind the score (dated record)
│   ├── task-briefs.md            # Step-by-step brief per open task (guidance, not a tracker)
│   └── adr/                      # Architecture Decision Records (MADR, never deleted)
│       ├── ADR-0001-qdrant-as-reference-vector-db.md
│       ├── ADR-0002-adapter-factory-pluggability.md
│       ├── ADR-0003-config-yaml-env-split.md
│       ├── ADR-0004-hybrid-search-plus-reranking.md
│       ├── ADR-0005-api-cli-only-control-plane.md
│       ├── ADR-0006-python-package-surface.md
│       └── ADR-0007-bm25-as-backend-sparse-vectors.md
├── src/
│   └── fasterrag/
│       ├── errors.py             # Typed error taxonomy + stable error-code table
│       ├── api/                  # FastAPI routers (thin; zero business logic)
│       │   ├── main.py           # App factory, lifespan, correlation middleware
│       │   ├── problems.py       # RFC 9457 problem documents + exception handlers
│       │   ├── health.py         # /healthz, /readyz (registry of dependency checks)
│       │   ├── ingest.py         # POST /v1/ingest, job status
│       │   ├── query.py          # POST /v1/query (+ SSE streaming)
│       │   ├── collections.py    # Collections CRUD
│       │   └── admin.py          # Provisioning + admin endpoints
│       ├── services/             # Business logic / use cases (orchestration)
│       │   ├── provisioning.py   # Config-driven auto-provisioning (Qdrant/Langfuse/Grafana)
│       │   ├── doctor.py         # D10 preflight checks; gates provisioning
│       │   ├── journal.py        # D3 checkpoints, dedup, DLQ; async over *_blocking twins
│       │   ├── estimation.py     # D9 preflight token counts and projected cost
│       │   ├── composition.py    # Composition root: the service builders the API, CLI, facade and eval harness share
│       │   ├── embedding_router.py # The embedding router every entry point uses, cached when configured
│       │   ├── ingestion.py      # Job lifecycle: run both pools, checkpoint, settle
│       │   ├── documents.py      # Delete one document: chunks, dedup tombstone, lockfile, cache
│       │   ├── querying.py       # Retrieve → fuse → rerank → assemble → generate
│       │   └── collections.py    # Collection lifecycle (pending)
│       ├── core/                 # RAG pipeline (pure domain logic)
│       │   ├── identity.py       # Deterministic document, chunk, and config hashes
│       │   ├── tenancy.py        # The payload field naming a chunk's owning tenant
│       │   ├── parsing/          # PDF/HTML/MD/DOCX/OCR parsers, table extraction
│       │   ├── chunking/         # fixed, recursive, semantic, layout, late, contextual
│       │   ├── retrieval/        # bm25.py sparse encoding; fusion and filters
│       │   ├── cache/            # Embedding cache, semantic response cache
│       │   ├── rerank/           # cross-encoder reranking
│       │   ├── context.py        # Context assembly, token budgeting, citations
│       │   └── generation.py     # Prompt building + streaming generation
│       ├── adapters/             # Vendor isolation (nothing vendor-specific escapes)
│       │   ├── vectordb/         # base.py (VectorDBAdapter), factory.py,
│       │   │                     # qdrant.py, milvus.py, weaviate.py,
│       │   │                     # pgvector.py, chroma.py (pinecone.py pending)
│       │   ├── embeddings/       # base, factory, huggingface, openai, cohere, ollama,
│       │   │                     # tiering (metadata-routed model selection), sync bridge
│       │   └── llm/              # base, factory, provider clients (streaming)
│       ├── workers/              # Parallel pools
│       │   ├── cpu_pool.py       # load/parse/chunk workers (CPU-bound)
│       │   ├── embed_pool.py     # stateful embedding workers (model loaded once)
│       │   ├── queues.py         # Bounded queues, backpressure, retry policy
│       │   ├── ledger.py         # Per-document completion: "indexed" only once every chunk is written
│       │   └── indexer.py        # Batch upsert: dense vector + BM25 sparse vector + payload
│       ├── config/               # Loaders + schema
│       │   ├── schema.py         # Pydantic v2 models mirroring config-reference.md
│       │   ├── plugins.py        # Entry-point names by group, read from metadata only
│       │   └── loader.py         # pydantic-settings YAML source + .env; fail-fast
│       ├── observability/        # Metrics, tracing, dashboard
│       │   ├── logging.py        # Structured JSON logging + correlation ids
│       │   ├── otel_export.py    # Span helpers (retrieval/reranker/context/generation)
│       │   ├── metrics.py        # Metrics catalogue export
│       │   └── dashboard.py      # Read-only inspection UI (observability ONLY)
│       └── cli/                  # Terminal entry points
│           └── main.py           # fasterrag ingest|query|index|provision|status|benchmark|serve
└── tests/
    ├── unit/                     # Fast, isolated (mirrors src layout)
    ├── contract/                 # Shared adapter contract suite every provider must pass
    ├── integration/              # Real adapters against containerized backends
    ├── eval/                     # Retrieval eval harness + datasets (recall@k, MRR, nDCG, faithfulness)
    ├── chaos/                    # Scripted fault injection (D12); `chaos` marker, not in CI
    ├── load/                     # Locust benchmark + the `load` saturation suite; not in CI
    └── soak/                     # Long-run resource-trend harness; `soak` marker, not in CI
```

## Directory responsibilities

| Directory | Responsibility |
|---|---|
| `api/` | HTTP surface only. Routers validate requests (Pydantic), call one service function, shape the response. **No business logic in routers.** |
| `services/` | Use-case orchestration. Compose core pipeline pieces and adapters into ingestion, querying, collection, provisioning, and diagnostic workflows. Own transactions/job state. |
| `tests/contract/` | The shared adapter contract suite. Vendor-neutral by construction: a subclass supplies the adapter and collection fixtures, so third-party providers registered through entry points run the identical suite. |
| `core/` | The RAG pipeline itself — parsing, chunking, retrieval, rerank, context assembly, generation. Pure domain logic; depends on adapter *interfaces*, never concrete vendors. |
| `adapters/` | All vendor code (vector DBs, embedding providers, LLM providers). Each adapter implements a base interface; a factory instantiates the concrete one from config. Vendor types never leak past this boundary. |
| `workers/` | Parallel execution: CPU pool for parse/chunk, stateful embedding pool, bounded queues, batch indexer. Owns backpressure and retry policy. |
| `config/` | Pydantic schema + pydantic-settings loader (YAML config, `.env` secrets). Validates on startup and fails fast with a clear error. Sensitive area — see [CLAUDE.md](https://github.com/balrajsingh165/fasterrag/blob/main/CLAUDE.md) folder boundaries. |
| `core/cache/` | Embedding cache and semantic response cache (similarity-keyed), TTL + event-driven invalidation. |
| `observability/` | OTel spans, metrics export, and the read-only dashboard. The dashboard renders data; it exposes zero control endpoints. |
| `cli/` | Terminal control plane. Thin command layer calling the same services as the API. |
| `docs/adr/` | MADR decision records, sequentially numbered, never deleted (superseded ADRs get a "Superseded by ADR-XXXX" status). |
| `tests/` | Unit, integration, and eval suites per [testing-strategy.md](testing-strategy.md). |

## The import rule

Imports point one way. A module may import its own layer and any layer listed after it, never one listed before it:

`cli/` → `api/` | the facade | the dashboard → `services/` → `workers/` → `adapters/` → `core/` → `observability/` → `config/` → `errors.py`

The three in the second position are siblings and may not import each other: the facade is `__init__.py`, `facade.py`, `sync.py` and the re-export modules beside them, and the dashboard is `observability/dashboard.py`, a second web application rather than instrumentation. `cli/` sits above them because the terminal is what starts the API and the dashboard.

`scripts/check_layering.py` enforces this in CI's `quality` job and in the unit tier, reading every import statement at any depth — a function-level import is how the facade used to reach into `api/` unseen. **The tree does not satisfy the rule yet** (TASK-0456): the imports that break it are listed by name in that script's `EXCEPTIONS` table, each with the step that removes it, and the gate fails on any import not in the table and on any entry the tree no longer needs. Whether `core/` may name the adapter base interfaces — which the table below says it does, and which the order above does not allow — is the open decision in that task.

## Separation-of-concerns rationale

- **Endpoints orchestrate use cases, nothing more.** A router that grows an `if` about chunking strategy is a bug; that decision belongs in `core/chunking` driven by config.
- **Services are the only writers of state.** Job records, collection metadata, and cache invalidation all flow through services, so both API and CLI share identical behavior.
- **Adapters isolate vendor code** so a `vector_db.provider` change is a config edit, not a refactor. Core code imports `VectorDBAdapter`, never `qdrant_client`.
- **Workers are separate from services** so ingestion acceptance (fast, async, HTTP-facing) is decoupled from heavy processing (CPU/GPU-bound, queue-fed) — the API never blocks on embedding.
- **Observability is one-directional.** `observability/` reads events and metrics from the pipeline; nothing in the pipeline depends on it, and the dashboard cannot mutate system state.
