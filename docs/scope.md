# scope.md — Vision, Goals, and Scope

> **Assumption (control vs dashboard).** The requirements say the system is "terminal/API only, no GUI" and also "ships a web dashboard." These are reconciled as follows: the **control plane is exclusively programmatic — the REST API, the CLI, and the importable Python package** ([python-api.md](python-api.md), [ADR-0006](adr/ADR-0006-python-package-surface.md)) — no graphical interface can create, modify, or drive the RAG. The **observability dashboard is a separate, optional, self-hosted web GUI used strictly for inspection** (metrics, traces, LLM input/output history), never for control. This reconciliation is treated as a project assumption throughout the documentation. See also [observability.md](observability.md) and [ADR-0005](adr/ADR-0005-api-cli-only-control-plane.md).

## Vision

fasterRag is a FastAPI-based, backend-only, one-stop Retrieval-Augmented Generation (RAG) solution engineered for very, very large datasets. The goal of this repository is to ship the **first beta version** of an all-in-one RAG system that aims to be the fastest, most efficient, and most optimized RAG project available.

Speed and efficiency come from three pillars:

1. **Multi-worker parallel processing** across ingestion, chunking, embedding, and indexing — a CPU worker pool streams into a GPU/embedding worker pool, so expensive workers wait on parsing only when the bounded queue runs dry. How often that happens is unmeasured (TASK-0084).
2. **Maximum chunking quality as the goal**, via a configurable chunking pipeline (fixed, recursive, semantic, layout-aware, late chunking, contextual enrichment) whose defaults are drawn from the published evidence catalogued in [references.md](references.md) — noting that the chunk-size guidance rests on a single preliminary source (R3) and is treated as directional, not settled.
3. Aggressive **caching, batching, streaming, and async I/O** at every stage.

**Total pluggability**: any vector database, any embedding model, any LLM provider — all selected purely through configuration (`config.yaml`), with secrets isolated in `.env`.

## Goals

- Ship a beta that ingests, indexes, and serves retrieval-augmented generation over very large corpora (hundreds of GB of documents, hundreds of millions of chunks) on self-hosted hardware.
- Retrieval quality that targets the strongest published results out of the box: hybrid dense + BM25 retrieval, Reciprocal Rank Fusion (k=60), and cross-encoder reranking enabled by default (measured by our own eval harness before any claim is made — see [benchmarks.md](benchmarks.md)).
- Release fasterRag as an importable Python package (`pip install fasterrag`) with a stable public API, standalone components, and an entry-point plugin contract ([python-api.md](python-api.md)).
- One config file drives everything; flipping a boolean provisions entire subsystems (Langfuse, Grafana, Qdrant) with no code changes.
- Deterministic, measurable performance: every release ships benchmark numbers per [performance.md](performance.md).

## Non-goals (explicit)

- **No control GUI.** No web or desktop interface ever controls the RAG. Control plane = REST API + CLI only.
- **Not a managed cloud service in beta.** fasterRag beta is self-hosted software; no hosted/SaaS offering, billing, or tenant onboarding portal.
- **Not a document authoring/annotation tool.** We ingest documents; we do not edit them.
- **Not an agent framework.** fasterRag answers queries over corpora; it does not orchestrate multi-step autonomous agents.
- **Not a model trainer.** No fine-tuning or embedding-model training in beta.

## In-scope beta features

- Ingestion API + CLI: files, directories, URLs; async accept with job tracking.
- Parsing pipeline for PDF (incl. tables and scanned/OCR), HTML, Markdown, DOCX, TXT, CSV/JSON.
- Configurable chunking: fixed, recursive/hierarchical, semantic, layout/structure-aware, late chunking, contextual-retrieval-style enrichment.
- Parallel workers: CPU pool (load/parse/chunk) → queue → stateful GPU/embedding pool (batch embed) → indexer.
- Vector DB adapters: Qdrant (reference) and pgvector are **built and pass the shared contract suite**; Milvus, Weaviate, Pinecone, and Chroma are in scope for beta but **not yet built** (TASK-0049) and are refused by name at config load.
- Embedding providers: OpenAI, Cohere, HuggingFace/sentence-transformers, Ollama/local; tiered embedding.
- LLM providers: any, via config (OpenAI, Anthropic, Cohere, Ollama/local, OpenAI-compatible endpoints).
- Hybrid retrieval (dense + BM25), RRF fusion (k=60), cross-encoder reranking, metadata filtering.
- Embedding cache + semantic response cache + provider prompt-cache support.
- Streaming generation (SSE), citations/provenance in responses.
- Incremental updates: dedup, versioned metadata, re-embedding workflows.
- Observability: OTel traces, metrics catalogue, optional dashboard, Langfuse/Grafana auto-provisioning.
- Security: API-key auth, multi-tenancy isolation, Qdrant API-key support.
- Retrieval eval harness (recall@k, MRR, nDCG, faithfulness) and benchmark suite.

## Out-of-scope / future features

- Managed cloud/SaaS offering, billing, quota portal.
- Multi-modal retrieval (images, audio, video embeddings).
- Distributed multi-node worker orchestration (Kubernetes operator, autoscaling controllers).
- Fine-tuning pipelines and embedding-model training.
- GraphRAG / knowledge-graph construction.
- Real-time collaborative index editing.
- Additional vector DB adapters beyond the six scoped for beta (e.g. Elasticsearch, Vespa, LanceDB) — community adapters welcome post-beta via the entry-point plugin contract.

## Target users

- **Platform/ML engineers** embedding RAG into products who need vendor freedom and predictable latency.
- **Data teams** with very large private corpora (legal, medical, finance, engineering docs) that exceed the comfort zone of hosted RAG APIs.
- **Self-hosters** with compliance constraints who need everything (vector DB, dashboard, tracing) running on their own machines.

## Primary use cases

1. Enterprise knowledge-base Q&A over millions of documents with citations.
2. High-throughput document ingestion pipelines (continuous feeds, nightly batches).
3. Search/retrieval backends for downstream applications via REST.
4. Evaluation-driven RAG tuning: swap chunkers/embedders/rerankers via config and compare eval metrics.

## Measurable success criteria (beta)

| Criterion | Target |
|---|---|
| Ingestion throughput (8-core CPU + 1 GPU reference rig) | ≥ 50 docs/sec sustained; ≥ 200k tokens/sec embedded |
| Query latency p50 (retrieval only, 10M chunks) | ≤ 150 ms |
| Query latency p95 end-to-end (with rerank, excluding LLM generation) | ≤ 600 ms |
| Retrieval failure rate (top-20, eval corpus, hybrid + contextual + rerank) | ≤ 2% |
| Semantic cache hit latency | ≤ 50 ms |
| Backend swap (e.g. Qdrant → pgvector) | config-only change, zero code edits |
| Langfuse/Grafana provisioning | single config toggle → running URL, zero code edits |
| Eval harness | recall@k, MRR, nDCG, faithfulness reported in CI on every PR |

## RAG pain-points catalogue and how fasterRag addresses each

Every known RAG pain point, with fasterRag's concrete mitigation. Architecture-level detail in [architecture.md](architecture.md).

| # | Pain point | fasterRag mitigation |
|---|---|---|
| 1 | **Poor chunking quality** | Configurable chunking pipeline with six strategies (fixed, recursive, semantic, layout-aware, late chunking, contextual enrichment); practical default range ~512–1024 tokens; documented "context cliff" ceiling near ~2,500 tokens (directional, from a January 2026 preprint); per-collection chunking config. |
| 2 | **Bad retrieval accuracy** | Hybrid dense + BM25 retrieval fused with RRF (k=60) plus cross-encoder reranking — the two upgrades this stack expects to matter most, unmeasured by us — and Contextual Retrieval enrichment (Anthropic, Sept 2024: 49% fewer failed retrievals; 67% fewer combined with reranking — their measurement, not ours). |
| 3 | **Hallucinations / weak grounding** | Retrieval-grounded prompts with mandatory citation assembly; faithfulness scoring in the eval harness; configurable "answer only from context" prompting; low default temperature. |
| 4 | **Stale / out-of-date indexes** | Incremental ingestion with versioned metadata; event-driven cache invalidation on corpus change; re-embedding workflows; `index reembed` CLI command. |
| 5 | **Incremental updates & deduplication** | Content-hash dedup at ingestion; document version fields; upsert semantics in all adapters; partial re-index of changed documents only. |
| 6 | **Parsing messy PDFs / tables / scanned docs** | Layout-aware parsing preserving reading order, tables, headings, and metadata; OCR path for scanned documents; table serialization strategies; parse-quality flags stored in metadata. |
| 7 | **Metadata filtering** | First-class metadata schema per collection; filter expressions in query API/CLI pushed down to the vector DB adapter; BM25 index carries the same filters. |
| 8 | **Hybrid-search needs** | Dense ANN + sparse BM25 run in parallel on every query (when `retrieval.hybrid: true`), fused with Reciprocal Rank Fusion using k=60 per Cormack, Clarke & Büttcher (SIGIR 2009). |
| 9 | **Reranking** | Cross-encoder reranker stage (retrieve top 100–1000 → rerank → truncate to top-K); toggleable via config. Its latency cost and quality benefit are both unmeasured (TASK-0084). |
| 10 | **Evaluation difficulty** | Built-in eval harness: recall@k, MRR, nDCG, faithfulness; dataset fixtures; CI quality gates; `benchmark` CLI command. See [testing-strategy.md](testing-strategy.md). |
| 11 | **Cost control** | Embedding cache; semantic response cache; batched embedding and LLM calls; tiered embedding (cheap models for low-priority classes); provider prompt caching; cost-per-query metric on the dashboard. |
| 12 | **Latency** | Async FastAPI; streaming responses (time-to-first-token); parallel retrieval legs; batch inference; semantic cache short-circuit; p50/p95 tracked per stage. |
| 13 | **Scaling to very large datasets** | Decoupled multi-worker pipeline with bounded queues; stateful embedding workers (model loaded once per worker); vector DB sharding/replication via adapter config; streaming ingestion that never materializes whole corpora in memory. |
| 14 | **Multi-tenancy** | Tenant-scoped collections and API keys; per-tenant isolation enforced at the service layer; tenant tag on every trace and metric. See [security.md](security.md). |
| 15 | **Security** | `.env`-only secrets; API-key auth on the control plane; Qdrant `QDRANT__SERVICE__API_KEY` support; TLS guidance; no secrets in config or logs. |
| 16 | **Observability gaps** | OTel spans wrapping retrieval and generation correlated by trace ID; four RAG trace types (retrieval, reranker, context-assembly, generation); metrics catalogue; optional Langfuse-like dashboard; Grafana provisioning-as-code. |
| 17 | **Vendor lock-in** | Adapter/factory pattern for vector DBs, embeddings, and LLMs; one-line `vector_db.provider` swap; no vendor types outside `adapters/`. |
| 18 | **Prompt/context-window management (context rot)** | Context assembly budgeter that packs top-K chunks within a token budget; chunk-size guidance (512–1024 working range, ~2,500-token context cliff as a directional ceiling); dedup of near-identical chunks before prompting. |
| 19 | **Citation/grounding & provenance** | Every chunk carries source URI, page/section, version, and hash; responses return citation objects alongside streamed text; dashboard shows full LLM input/output history for audit. |
| 20 | **Embedding drift** | Embedding model + version recorded per chunk; drift detection when config model ≠ indexed model; guided re-embedding (`index reembed`) with zero-downtime collection swap. |
| 21 | **Cold-start** | Auto-provisioning of Qdrant/Langfuse/Grafana from config toggles; sensible defaults for every parameter; `fasterrag provision` + `fasterrag status` get a working stack in minutes. |
| 22 | **Debugging retrieval-vs-generation failures** | Per-stage traces and latency split (retrieval vs generation); dashboard drill-down from answer → context → retrieved candidates → fused/reranked scores; eval harness isolates retrieval metrics from generation faithfulness. |
