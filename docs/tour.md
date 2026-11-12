---
title: A visual tour of fasterRag
description: >-
  Follow a document into fasterRag and a question out of it, stage by stage: which module owns
  each step, what happens when a step fails, the mechanisms that keep the system dependable, and
  every package module by module.
---

# A visual tour

Two paths run through fasterRag: a document going in, and a question coming out. This page follows
both, names the module that owns each stage, and says what happens when that stage fails. For the
picture of how the packages fit together, see the map on the [home page](index.md#how-it-fits-together).

<div class="fr-term" aria-hidden="true">
<div class="fr-term-bar"><i></i><i></i><i></i><span>the whole lifecycle, from a terminal</span></div>
<div class="fr-term-body"><span class="fr-l fr-l1">$ fasterrag config init</span><span class="fr-l fr-l2">$ fasterrag doctor</span><span class="fr-l fr-l3">$ fasterrag ingest ./handbook --recursive --watch</span><span class="fr-l fr-l4">$ fasterrag query "What is the refund window?" --show-chunks</span></div>
</div>

## A document goes in

Eight stages, each owned by one module. The second line of each card is what happens when that
stage goes wrong.

<ol class="fr-track fr-track--in fr-track--8"><li style="--i:0"><span class="fr-step">01</span><strong>Accept</strong><code>services/ingestion.py</code><p>A job is recorded in the journal and its id returned at once. An idempotency key replayed returns the same job.</p><p class="fr-guard">A bad source is refused here, before any work.</p></li><li style="--i:1"><span class="fr-step">02</span><strong>Resolve</strong><code>services/sources.py</code><p>Paths, URLs and inline payloads become readable bytes. Paths can be confined to allowed roots; URLs are fetched through an SSRF guard.</p><p class="fr-guard">A source that cannot be read fails the request, not the corpus.</p></li><li style="--i:2"><span class="fr-step">03</span><strong>Parse</strong><code>core/parsing/</code><p>A worker process turns the file into structured blocks: headings, paragraphs, tables, code.</p><p class="fr-guard">A corrupt file costs one document. It goes to the dead-letter queue with a reason code.</p></li><li style="--i:3"><span class="fr-step">04</span><strong>Chunk</strong><code>core/chunking/</code><p>Blocks are packed into chunks inside a token budget. Each chunk gets a deterministic id.</p><p class="fr-guard">Re-ingesting the same content produces the same ids, so a retry overwrites.</p></li><li style="--i:4"><span class="fr-step">05</span><strong>Queue</strong><code>workers/queues.py</code><p>Chunks cross a bounded queue. When embedding is slower than parsing, the parser waits.</p><p class="fr-guard">Memory stays bounded however large the corpus.</p></li><li style="--i:5"><span class="fr-step">06</span><strong>Embed</strong><code>workers/embed_pool.py</code><p>Workers batch chunks and call the embedding adapter behind a circuit breaker.</p><p class="fr-guard">Retryable provider errors are retried with backoff; the rest fail only their own document.</p></li><li style="--i:6"><span class="fr-step">07</span><strong>Index</strong><code>workers/indexer.py</code><p>Dense vectors and BM25 sparse vectors are written together through the vector-database adapter.</p><p class="fr-guard">A refused write is retried, then recorded against the document.</p></li><li style="--i:7"><span class="fr-step">08</span><strong>Commit</strong><code>workers/ledger.py</code><p>When the last chunk of a document is written, the journal records it as indexed and the checkpoint advances.</p><p class="fr-guard">A document with any failed chunk is never recorded as indexed.</p></li></ol>

A job always ends: an error, a cancellation or a shutdown settles it in a terminal status. And the
journal never lies about the index: a document's dedup hash and its `indexed` record are written
after its chunks, not before.

!!! note "What is not built yet"
    Nothing resumes an interrupted job on its own: it is settled as interrupted when the server
    next starts, and runs again from its checkpoint when you ask (`fasterrag ingest --resume`).
    Ingestion runs inside the process that accepts the job; a separate queue-backed worker is
    not built (TASK-0130).

## A question comes out

Nine stages. Retrieval is hybrid by default — meaning and keywords are searched together — and the
stages that can fail have a defined fallback.

<ol class="fr-track fr-track--q fr-track--9"><li style="--i:0"><span class="fr-step">01</span><strong>Ask</strong><code>api/query.py · cli · facade</code><p>A question arrives with an optional collection, filter and top-k.</p><p class="fr-guard">Validated at the boundary; errors are problem+json with a stable code.</p></li><li style="--i:1"><span class="fr-step">02</span><strong>Cache</strong><code>core/cache/semantic.py</code><p>A paraphrase of an earlier question can be answered from the semantic cache, scoped by collection and filter.</p><p class="fr-guard">A miss falls through to retrieval.</p></li><li style="--i:2"><span class="fr-step">03</span><strong>Embed</strong><code>adapters/embeddings/</code><p>The question is embedded with the configured embedding model.</p><p class="fr-guard">Behind the embeddings circuit breaker.</p></li><li style="--i:3"><span class="fr-step">04</span><strong>Search</strong><code>services/querying.py</code><p>Two searches run at once: dense vectors and BM25. Both receive the same pushed-down filter.</p><p class="fr-guard">If the keyword leg has no terms it is skipped and the dense leg answers.</p></li><li style="--i:4"><span class="fr-step">05</span><strong>Fuse</strong><code>core/retrieval/fusion.py</code><p>Reciprocal Rank Fusion merges the two rankings into one shortlist.</p><p class="fr-guard">Each result keeps the rank and score of both legs.</p></li><li style="--i:5"><span class="fr-step">06</span><strong>Rerank</strong><code>core/rerank/</code><p>A cross-encoder reads the question and each chunk together and reorders the shortlist.</p><p class="fr-guard">If the reranker fails, the fused order is used and the degradation is recorded.</p></li><li style="--i:6"><span class="fr-step">07</span><strong>Assemble</strong><code>core/context.py</code><p>Chunks are packed into the prompt inside a token budget; near-duplicates are dropped.</p><p class="fr-guard">Chunks that do not fit the budget are left out.</p></li><li style="--i:7"><span class="fr-step">08</span><strong>Generate</strong><code>services/generation.py</code><p>The LLM adapter answers, streamed as server-sent events, after the cost budget is charged.</p><p class="fr-guard">A provider failure before the first token degrades to an extractive answer built from the retrieved chunks, streamed or not.</p></li><li style="--i:8"><span class="fr-step">09</span><strong>Cite and trace</strong><code>core/generation.py · services/traces.py</code><p>Citation markers are resolved against the chunks that were supplied, and the whole query is stored as a trace.</p><p class="fr-guard">A stored trace can be replayed later under a different configuration.</p></li></ol>

The same path serves `POST /v1/query`, `fasterrag query` and `FasterRag.query`. `POST /v1/retrieve`
stops after reranking and returns the chunks with the rank and score each stage gave them.

## What holds it together

Six mechanisms do most of the work of making the system dependable. Each is one module you can read.

<div class="fr-mech"><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w" style="--i:0" x="8" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:1" x="25" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:2" x="42" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:3" x="59" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:4" x="76" y="14" width="12" height="16" rx="3"/><rect class="fr-w" style="--i:5" x="93" y="14" width="12" height="16" rx="3"/></svg><strong>The ledger</strong><code>workers/ledger.py</code><p>A document is recorded as indexed only when the vector database holds every chunk of it. The checkpoint is a watermark over settled documents, so a resumed job never skips one still in flight.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w" style="--i:0" x="8" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:1" x="29" y="10" width="15" height="12" rx="3"/><rect class="fr-drop" style="--i:2" x="50" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:3" x="71" y="10" width="15" height="12" rx="3"/><rect class="fr-w" style="--i:4" x="92" y="10" width="15" height="12" rx="3"/></svg><strong>The dead-letter queue</strong><code>services/journal.py</code><p>One bad document costs one document. It is set aside with a reason code and can be retried on its own; the rest of the job carries on.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><circle class="fr-w fr-w--3" style="--i:0" cx="24" cy="22" r="9"/><circle class="fr-w fr-w--3" style="--i:1" cx="60" cy="22" r="9"/><circle class="fr-w fr-w--3" style="--i:2" cx="96" cy="22" r="9"/><path d="M33 22H51M69 22H87" class="fr-line"/></svg><strong>Circuit breakers</strong><code>core/breaker.py</code><p>One breaker per provider, shared across the process. Repeated failures open it, and calls are refused quickly instead of queueing behind a dead dependency.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-w fr-w--3" style="--i:0" x="14" y="8" width="80" height="7" rx="3.5"/><rect class="fr-w fr-w--3" style="--i:1" x="28" y="19" width="66" height="7" rx="3.5"/><rect class="fr-w fr-w--3" style="--i:2" x="42" y="30" width="52" height="7" rx="3.5"/></svg><strong>The degradation ladder</strong><code>services/degradation.py</code><p>When a stage fails the answer steps down, visibly: full, then fused retrieval without reranking, then an extractive answer from the retrieved chunks.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><path class="fr-line" d="M10 12H44Q56 12 60 22M10 32H44Q56 32 60 22M60 22H82"/><rect class="fr-w" style="--i:2" x="84" y="13" width="26" height="18" rx="5"/><circle class="fr-w" style="--i:0" cx="10" cy="12" r="4"/><circle class="fr-w" style="--i:1" cx="10" cy="32" r="4"/></svg><strong>Idempotency keys</strong><code>services/idempotency.py</code><p>A mutating request replayed with the same key returns the first result. A retry after a timeout cannot start the work twice.</p></div><div class="fr-mcard"><svg viewBox="0 0 120 44" aria-hidden="true"><rect class="fr-rail" x="8" y="17" width="104" height="10" rx="5"/><rect class="fr-grow" x="8" y="17" width="80" height="10" rx="5"/><path class="fr-line" d="M88 9V35"/></svg><strong>The cost governor</strong><code>services/budget.py</code><p>A token budget per tenant is charged before the provider is called. A request over the budget is refused, not billed.</p></div></div>

## Every package, module by module

Open a package to see each module and what it is for. Each description is the first line of the
module's own docstring, as of the commit this page was generated from.

??? abstract "api/ — entry surfaces"

    The REST surface. Routers hold no business logic: they validate, call a service and render the result or a problem document.

    | Module | What it does |
    | --- | --- |
    | `api/__init__.py` | FastAPI surface: routers, problem responses, and the application factory. |
    | `api/admin.py` | Administrative endpoints: doctor, provisioning, and preflight estimation. |
    | `api/auth.py` | Control-plane authentication, scopes, and per-key rate limiting. |
    | `api/collections.py` | Collection management endpoints. |
    | `api/dependencies.py` | FastAPI dependencies: what a router reads from `app.state` and the request. |
    | `api/health.py` | Liveness and readiness endpoints. |
    | `api/idempotency.py` | `Idempotency-Key` handling for every mutating endpoint (TASK-0318). |
    | `api/ingest.py` | Ingestion endpoints. |
    | `api/limits.py` | Request body size limiting (`security.max_request_mb`). |
    | `api/main.py` | Application factory, lifespan, and middleware. |
    | `api/metrics.py` | The metrics scrape endpoint and the RED middleware behind it. |
    | `api/problems.py` | RFC 9457 `application/problem+json` responses. |
    | `api/query.py` | Query endpoints. |
    | `api/schemas.py` | Request and response models for the REST surface. |
    | `api/traces.py` | Trace inspection and replay endpoints (D8). |

??? abstract "cli/ — entry surfaces"

    The `fasterrag` command. Every command turns its arguments into the same validated settings the API uses.

    | Module | What it does |
    | --- | --- |
    | `cli/__init__.py` | The terminal half of the control plane. |
    | `cli/commands/autopilot.py` | The `autopilot run` command (D6). |
    | `cli/commands/backup.py` | The `backup` and `restore` commands. |
    | `cli/commands/benchmark.py` | The `benchmark` command. |
    | `cli/commands/diagnostics.py` | The commands that inspect an installation: `config validate`, `doctor`, `status`. |
    | `cli/commands/documents.py` | The `documents` command: per-document operations on a collection. |
    | `cli/commands/infrastructure.py` | The commands that stand things up or price them out: `provision` and `estimate`. |
    | `cli/commands/pipeline.py` | The commands that move data: `ingest`, `query`, and `index`. |
    | `cli/commands/portability.py` | The commands that move a collection between deployments: `export` and `import` (D11). |
    | `cli/commands/processes.py` | The command that runs a long-lived process: `serve`. |
    | `cli/commands/replay.py` | The `replay` command (D8). |
    | `cli/commands/traces.py` | The `traces` command: list and inspect stored query traces (D8). |
    | `cli/main.py` | The `fasterrag` entry point. |
    | `cli/output.py` | CLI output and exit codes. |
    | `cli/parser.py` | The `fasterrag` argument parser. |
    | `cli/settings.py` | The single way a CLI command turns parsed arguments into validated settings. |
    | `cli/sources.py` | Expansion of directory arguments into the files a job actually ingests. |

??? abstract "facade.py · sync.py · public modules — entry surfaces"

    The importable package: the async facade, its blocking twin, and the parts usable on their own.

    | Module | What it does |
    | --- | --- |
    | `facade.py` | The `FasterRag` facade: fasterRag as a library rather than a service. |
    | `sync.py` | The blocking facade: fasterRag from a script, a notebook, or a WSGI app. |
    | `chunking.py` | Public chunking surface: `from fasterrag.chunking import RecursiveChunker`. |
    | `parsing.py` | Public parsing surface: `from fasterrag.parsing import parse_document`. |
    | `retrieval.py` | Public retrieval surface: `from fasterrag.retrieval import rrf_fuse`. |
    | `rerank.py` | Public reranking surface: `from fasterrag.rerank import CrossEncoderReranker`. |
    | `evals.py` | Public evaluation surface: `from fasterrag.evals import evaluate`. |
    | `__init__.py` | fasterRag — a backend-only Retrieval-Augmented Generation framework. |

??? abstract "services/ — orchestration"

    Use cases. A service owns a lifecycle — a job, a query, a replay — and calls workers, core and adapters to carry it out.

    | Module | What it does |
    | --- | --- |
    | `services/__init__.py` | Use-case orchestration. |
    | `services/archive.py` | Portable archive export (D11, `docs/archive-format.md`). |
    | `services/archive_import.py` | Portable archive import (D11, `docs/archive-format.md` §Import semantics). |
    | `services/autopilot.py` | Eval-driven auto-tuning that only ever suggests (D6). |
    | `services/backfill.py` | Payload backfill for the tenant tag, and the audit that must precede it. |
    | `services/backup.py` | Backup and restore of a whole deployment. |
    | `services/benchmark.py` | The benchmark suite and ledger emitter. |
    | `services/budget.py` | The D9 runtime cost governor: token budgets enforced before a provider is called. |
    | `services/ci_status.py` | Reads the latest CI verdict for a branch from the GitHub Actions API. |
    | `services/composition.py` | Composition root: where an entry point gets the pieces it runs on. |
    | `services/degradation.py` | The degradation ladder's registry: which rungs exist, and what serves each one. |
    | `services/doctor.py` | Preflight diagnostics (D10). |
    | `services/documents.py` | Document lifecycle: taking one document out of a collection (TASK-0443). |
    | `services/embedding_router.py` | The embedding router every entry point uses, behind the embedding cache when configured. |
    | `services/estimation.py` | Preflight cost estimation (D9). |
    | `services/evaluation.py` | Running the eval harness against a live collection. |
    | `services/generation.py` | Answer generation: retrieve, assemble, generate, cite. |
    | `services/golden_sets.py` | Building a golden set from a corpus on disk (P4). |
    | `services/grafana.py` | Grafana auto-provisioning as code (`observability.grafana: true`). |
    | `services/idempotency.py` | Durable `Idempotency-Key` records for the mutating REST surface (TASK-0318). |
    | `services/ingestion.py` | Ingestion: accept a job, run the pipeline, own its lifecycle. |
    | `services/jobs.py` | The handle on every ingestion job this process is running (TASK-0436). |
    | `services/journal.py` | Checkpointed, idempotent ingestion state (D3). |
    | `services/langfuse.py` | Langfuse v3 auto-provisioning (`observability.langfuse: true`). |
    | `services/lockfile.py` | The index lockfile and drift detection (D1). |
    | `services/provision_registry.py` | The one table of what fasterRag can provision, and the three verbs each tool answers. |
    | `services/provisioning.py` | Config-driven provisioning of system-managed containers. |
    | `services/querying.py` | Retrieval: hybrid dense and keyword legs, fused. |
    | `services/regression.py` | The retrieval regression gate (D7). |
    | `services/reindex.py` | Zero-downtime reindexing (D2). |
    | `services/replay.py` | Time-travel replay (D8). |
    | `services/sources.py` | Resolving the three documented source types into readable bytes. |
    | `services/tenancy.py` | Tenant scoping for collection names. |
    | `services/traces.py` | Local trace persistence (D8). |

??? abstract "workers/ — engine"

    The ingestion pipeline: a process pool that parses and chunks, a bounded queue, embedding workers, the indexer and the ledger.

    | Module | What it does |
    | --- | --- |
    | `workers/__init__.py` | Parallel execution: the CPU pool, the embedding pool, bounded queues, and the indexer. |
    | `workers/cpu_pool.py` | CPU worker pool: load, parse, and chunk. |
    | `workers/embed_pool.py` | Embedding worker pool: stateful, batched, retryable. |
    | `workers/indexer.py` | Indexer: writes embedded chunks to the vector database. |
    | `workers/ledger.py` | Per-document completion tracking between the parse side and the index side. |
    | `workers/queues.py` | Bounded queues, backpressure, and the payloads that cross the pool boundary. |

??? abstract "core/ — engine"

    The vendor-free library: parsing, chunking, sparse encoding and fusion, reranking, context assembly, prompt contracts, caches, evaluation, the breaker and the rate limiter.

    | Module | What it does |
    | --- | --- |
    | `core/__init__.py` | The RAG pipeline: parsing, chunking, retrieval, reranking, assembly, generation. |
    | `core/breaker.py` | Per-provider circuit breakers (`reliability.circuit_breaker`, docs/reliability.md §3). |
    | `core/cache/embedding.py` | Embedding cache: never pay twice for the same text under the same model. |
    | `core/cache/redis.py` | Redis-backed cache store: one cache shared by every process that reaches the same server. |
    | `core/cache/semantic.py` | Semantic response cache: answer a paraphrase from the answer already given. |
    | `core/cache/stats.py` | Cache hit/miss accounting. |
    | `core/cache/store.py` | Cache storage backends. |
    | `core/chunking/enrichment.py` | Contextual enrichment (P2, `docs/prompts.md`). |
    | `core/chunking/fixed.py` | Fixed-size chunker. |
    | `core/chunking/late.py` | Late chunker. |
    | `core/chunking/layout.py` | Layout-aware chunker. |
    | `core/chunking/models.py` | Chunk structures, token counting, and the shared assembler that enforces invariants. |
    | `core/chunking/pooling.py` | The pooling half of late chunking (D-late, `chunking.strategy: late`). |
    | `core/chunking/recursive.py` | Recursive chunker — the default strategy. |
    | `core/chunking/semantic.py` | Semantic chunker. Usable from the library; the ingestion pipeline refuses it for now (TASK-0431). |
    | `core/chunking/tokenizer.py` | The embedding model's own tokenizer, used to make `chunking.chunk_size` mean tokens. |
    | `core/context.py` | Context assembly: pack retrieved chunks into a prompt, with citations. |
    | `core/evals/generator.py` | P4 golden-set generation from a real corpus. |
    | `core/evals/golden.py` | Golden sets: the ground truth retrieval quality is measured against. |
    | `core/evals/harness.py` | The retrieval eval harness. |
    | `core/evals/metrics.py` | Retrieval quality metrics. |
    | `core/faithfulness.py` | P3 faithfulness scoring: how much of an answer its context actually supports. |
    | `core/generation.py` | P1 answer generation: prompt building and citation resolution. |
    | `core/identity.py` | Deterministic identifiers and content hashes. |
    | `core/parsing/docx.py` | DOCX parser. |
    | `core/parsing/html.py` | HTML parser. |
    | `core/parsing/markdown.py` | Markdown parser. |
    | `core/parsing/models.py` | Parsed-document structures shared by every parser. |
    | `core/parsing/options.py` | Tunable parser thresholds, and how configuration reaches a parser. |
    | `core/parsing/pdf.py` | PDF parser with table extraction, layout-based heading inference, and an OCR path. |
    | `core/parsing/plaintext.py` | Plain-text parser. |
    | `core/parsing/tabular.py` | CSV and JSON parsers. |
    | `core/rate_limit.py` | Per-key request limiting, with the window kept wherever the deployment chooses. |
    | `core/rerank/cross_encoder.py` | Cross-encoder reranking of the retrieved shortlist. |
    | `core/retrieval/bm25.py` | BM25 sparse encoding for the keyword retrieval leg. |
    | `core/retrieval/fusion.py` | Reciprocal Rank Fusion. |
    | `core/retrieval/models.py` | Retrieval result types. |
    | `core/tenancy.py` | The payload field that names a chunk's owning tenant. |
    | `core/tracing.py` | Query traces and the four RAG spans (D8). |

??? abstract "adapters/ — vendor boundary"

    One interface and one factory per kind of vendor. Every vendor exception is translated into the project's error taxonomy here.

    | Module | What it does |
    | --- | --- |
    | `adapters/__init__.py` | Vendor isolation boundary. |
    | `adapters/embeddings/base.py` | Vendor-neutral embedding provider contract. |
    | `adapters/embeddings/cohere.py` | Cohere embedding provider. |
    | `adapters/embeddings/factory.py` | Selects the concrete embedding adapter from configuration. |
    | `adapters/embeddings/huggingface.py` | Local sentence-transformers embedding provider — the default. |
    | `adapters/embeddings/ollama.py` | Ollama embedding provider — a local server, so no credential is required. |
    | `adapters/embeddings/openai.py` | OpenAI embedding provider. |
    | `adapters/embeddings/sync.py` | Bridge from an async embedding adapter to the synchronous chunker interface. |
    | `adapters/embeddings/tiering.py` | Tiered embedding: route document classes to different models. |
    | `adapters/llm/anthropic.py` | Anthropic generation. |
    | `adapters/llm/base.py` | Vendor-neutral LLM provider contract. |
    | `adapters/llm/cohere.py` | Cohere generation through the v2 chat API. |
    | `adapters/llm/factory.py` | Selects the concrete LLM adapter from configuration. |
    | `adapters/llm/ollama.py` | Ollama generation against a local server, so no credential is required. |
    | `adapters/llm/openai.py` | OpenAI and OpenAI-compatible generation. |
    | `adapters/vectordb/base.py` | Vendor-neutral vector database contract. |
    | `adapters/vectordb/chroma.py` | Chroma adapter — the contract proved against an in-process, file-backed store. |
    | `adapters/vectordb/factory.py` | Selects the concrete vector database adapter from configuration. |
    | `adapters/vectordb/milvus.py` | Milvus adapter — the contract proved against a distributed, schema-first store. |
    | `adapters/vectordb/pgvector.py` | pgvector adapter — the vendor-neutral contract proved against a SQL backend. |
    | `adapters/vectordb/qdrant.py` | Qdrant reference adapter. |
    | `adapters/vectordb/weaviate.py` | Weaviate adapter — the contract proved against a store with its own opinion of search. |

??? abstract "config/ · errors.py · observability/ — foundations"

    The validated configuration, the typed error hierarchy, and logs, metrics, traces and the read-only dashboard.

    | Module | What it does |
    | --- | --- |
    | `config/__init__.py` | Configuration schema and loader. |
    | `config/loader.py` | Fail-fast configuration loader. |
    | `config/plugins.py` | Which provider names configuration may select, decided without loading a provider. |
    | `config/schema.py` | Pydantic models mirroring `docs/config-reference.md`. |
    | `config/template.py` | Access to the canonical `config.yaml` that ships inside the package. |
    | `errors.py` | Typed error taxonomy shared by the API, the CLI, and the Python package. |
    | `observability/__init__.py` | Observability: correlation ids, structured logging, spans, metrics, and the dashboard. |
    | `observability/dashboard.py` | The read-only observability dashboard (`observability.dashboard: true`). |
    | `observability/langfuse_export.py` | Exporting query traces to a provisioned Langfuse instance. |
    | `observability/logging.py` | Structured JSON logging and request correlation ids. |
    | `observability/metrics.py` | The metrics catalogue of `docs/observability.md` §2. |
    | `observability/otel_export.py` | OTLP export of query traces (`observability.otel`). |
    | `observability/otel_metrics.py` | OTLP export of the metric catalogue (`observability.otel`). |


Around the package: `tests/` (unit, contract, integration, eval, chaos, load and soak tiers),
`docs/`, `scripts/` (the gates that fail the build when a document and the code disagree), and
`deploy/` with `.github/` (compose profiles and the CI, nightly and security workflows).

## Where next

[Architecture](architecture.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }
[Engineering notes](engineering-notes.md){ .md-button }
