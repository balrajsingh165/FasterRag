# structure.md — Repository Structure

> This documents the **intended** layout for the beta build. No implementation code exists yet; directories below are created only when their build tasks in [todo.md](todo.md) begin.

## Proposed directory tree

```text
fasterRag/
├── CLAUDE.md                     # Always-loaded Claude Code instructions (ONLY doc kept at root)
├── README.md                     # Project overview + entry point into docs/
├── LICENSE
├── .gitignore                    # Ignores .env, caches, build artifacts
├── config.yaml                   # All behavior (no secrets) — MUST match docs/config-reference.md example
├── .env.example                  # Template for secrets; real .env is git-ignored
├── pyproject.toml                # Project metadata, deps, tooling config
├── .github/                      # PR template (rule enforcement), issue templates, SECURITY.md
├── docs/                         # ALL documentation lives here (rule: only CLAUDE.md stays at root)
│   ├── quickstart.md  scope.md  structure.md  flow.md  architecture.md
│   ├── config-reference.md  api-reference.md  cli-reference.md  python-api.md
│   ├── data-model.md  prompts.md  archive-format.md
│   ├── observability.md  deployment.md  security.md
│   ├── testing-strategy.md  performance.md  benchmarks.md  integrations.md
│   ├── cookbook.md  troubleshooting.md  migration-guide.md
│   ├── glossary.md  references.md
│   ├── differentiators.md        # The twelve flagship capabilities (uniqueness contract)
│   ├── reliability.md            # Reliability doctrine + resilience patterns
│   ├── failure-modes.md          # FMEA table (component failure analysis)
│   ├── slo.md                    # SLIs, SLO targets (TBD-until-measured), error budget
│   ├── disaster-recovery.md      # Backups, restore drill, RPO/RTO
│   ├── CHANGELOG.md              # Keep a Changelog + SemVer
│   ├── CONTRIBUTING.md           # Contributor rules
│   ├── todo.md                   # THE one and only task file
│   └── adr/                      # Architecture Decision Records (MADR, never deleted)
│       ├── ADR-0001-qdrant-as-reference-vector-db.md
│       ├── ADR-0002-adapter-factory-pluggability.md
│       ├── ADR-0003-config-yaml-env-split.md
│       ├── ADR-0004-hybrid-search-plus-reranking.md
│       └── ADR-0005-api-cli-only-control-plane.md
├── src/
│   └── fasterrag/
│       ├── api/                  # FastAPI routers (thin; zero business logic)
│       │   ├── main.py           # App factory, lifespan, middleware
│       │   ├── ingest.py         # POST /v1/ingest, job status
│       │   ├── query.py          # POST /v1/query (+ SSE streaming)
│       │   ├── collections.py    # Collections CRUD
│       │   ├── health.py         # /healthz, /readyz
│       │   └── admin.py          # Provisioning + admin endpoints
│       ├── services/             # Business logic / use cases (orchestration)
│       │   ├── ingestion.py      # Accept → enqueue → track jobs
│       │   ├── querying.py       # Retrieve → fuse → rerank → assemble → generate
│       │   ├── collections.py    # Collection lifecycle
│       │   └── provisioning.py   # Config-driven auto-provisioning (Qdrant/Langfuse/Grafana)
│       ├── core/                 # RAG pipeline (pure domain logic)
│       │   ├── parsing/          # PDF/HTML/MD/DOCX/OCR parsers, table extraction
│       │   ├── chunking/         # fixed, recursive, semantic, layout, late, contextual
│       │   ├── retrieval/        # dense + BM25 legs, RRF fusion, filters
│       │   ├── rerank/           # cross-encoder reranking
│       │   ├── context.py        # Context assembly, token budgeting, citations
│       │   └── generation.py     # Prompt building + streaming generation
│       ├── adapters/             # Vendor isolation (nothing vendor-specific escapes)
│       │   ├── vectordb/         # base.py (VectorDBAdapter), factory.py,
│       │   │                     # qdrant.py, milvus.py, weaviate.py,
│       │   │                     # pinecone.py, pgvector.py, chroma.py
│       │   ├── embeddings/       # base, factory, openai, cohere, huggingface, ollama
│       │   └── llm/              # base, factory, provider clients (streaming)
│       ├── workers/              # Parallel pools
│       │   ├── cpu_pool.py       # load/parse/chunk workers (CPU-bound)
│       │   ├── embed_pool.py     # stateful embedding workers (model loaded once)
│       │   ├── queues.py         # Bounded queues, backpressure, retry policy
│       │   └── indexer.py        # Batch upsert to vector DB + BM25
│       ├── config/               # Loaders + schema
│       │   ├── schema.py         # Pydantic v2 models mirroring config-reference.md
│       │   └── loader.py         # pydantic-settings YAML source + .env; fail-fast
│       ├── cache/                # Embedding cache, semantic response cache
│       ├── observability/        # Metrics, tracing, dashboard
│       │   ├── otel.py           # Span helpers (retrieval/reranker/context/generation)
│       │   ├── metrics.py        # Metrics catalogue export
│       │   └── dashboard/        # Read-only inspection UI (observability ONLY)
│       └── cli/                  # Terminal entry points
│           └── main.py           # fasterrag ingest|query|index|provision|status|benchmark|serve|worker
└── tests/
    ├── unit/                     # Fast, isolated (mirrors src layout)
    ├── integration/              # Real adapters against containerized backends
    └── eval/                     # Retrieval eval harness + datasets (recall@k, MRR, nDCG, faithfulness)
```

## Directory responsibilities

| Directory | Responsibility |
|---|---|
| `api/` | HTTP surface only. Routers validate requests (Pydantic), call one service function, shape the response. **No business logic in routers.** |
| `services/` | Use-case orchestration. Compose core pipeline pieces and adapters into ingestion, querying, collection, and provisioning workflows. Own transactions/job state. |
| `core/` | The RAG pipeline itself — parsing, chunking, retrieval, rerank, context assembly, generation. Pure domain logic; depends on adapter *interfaces*, never concrete vendors. |
| `adapters/` | All vendor code (vector DBs, embedding providers, LLM providers). Each adapter implements a base interface; a factory instantiates the concrete one from config. Vendor types never leak past this boundary. |
| `workers/` | Parallel execution: CPU pool for parse/chunk, stateful embedding pool, bounded queues, batch indexer. Owns backpressure and retry policy. |
| `config/` | Pydantic schema + pydantic-settings loader (YAML config, `.env` secrets). Validates on startup and fails fast with a clear error. Sensitive area — see [CLAUDE.md](../CLAUDE.md) folder boundaries. |
| `cache/` | Embedding cache and semantic response cache (similarity-keyed), TTL + event-driven invalidation. |
| `observability/` | OTel spans, metrics export, and the read-only dashboard. The dashboard renders data; it exposes zero control endpoints. |
| `cli/` | Terminal control plane. Thin command layer calling the same services as the API. |
| `docs/adr/` | MADR decision records, sequentially numbered, never deleted (superseded ADRs get a "Superseded by ADR-XXXX" status). |
| `tests/` | Unit, integration, and eval suites per [testing-strategy.md](testing-strategy.md). |

## Separation-of-concerns rationale

- **Endpoints orchestrate use cases, nothing more.** A router that grows an `if` about chunking strategy is a bug; that decision belongs in `core/chunking` driven by config.
- **Services are the only writers of state.** Job records, collection metadata, and cache invalidation all flow through services, so both API and CLI share identical behavior.
- **Adapters isolate vendor code** so a `vector_db.provider` change is a config edit, not a refactor. Core code imports `VectorDBAdapter`, never `qdrant_client`.
- **Workers are separate from services** so ingestion acceptance (fast, async, HTTP-facing) is decoupled from heavy processing (CPU/GPU-bound, queue-fed) — the API never blocks on embedding.
- **Observability is one-directional.** `observability/` reads events and metrics from the pipeline; nothing in the pipeline depends on it, and the dashboard cannot mutate system state.
