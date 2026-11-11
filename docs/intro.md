---
title: Introduction — the whole of fasterRag on one page
description: >-
  A guided introduction to fasterRag: what it is, how a document and a question travel
  through it, and a map of the entire repository — every package, what it owns, and what
  it is allowed to depend on.
hide:
  - toc
---

# Introduction

<div class="fr-stage">
<div class="fr-orb fr-orb--a"></div>
<div class="fr-orb fr-orb--b"></div>
<div class="fr-orb fr-orb--c"></div>
<div class="fr-hero">
<img src="../assets/logo.svg" alt="The fasterRag logo">
<p class="fr-title">faster<span>Rag</span>, on one page</p>
<p class="fr-lede">A backend-only RAG system you drive from a REST API, a terminal or Python. Below is the entire repository as one map: three ways in, one service layer, an engine, and a vendor boundary nothing crosses.</p>
</div>
<div class="fr-map">
<svg viewBox="0 0 1000 950" role="img" aria-labelledby="fr-map-title fr-map-desc" xmlns="http://www.w3.org/2000/svg">
<title id="fr-map-title">Map of the fasterRag repository</title>
<desc id="fr-map-desc">Three entry surfaces — REST API, CLI and Python package — call one service layer. Services drive the worker pools and the core library, which reach vector databases, embedding providers and LLM providers only through adapters. Tests, documentation, scripts and deployment files sit around the package.</desc>
<defs>
<linearGradient id="fr-fill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#38bdf8"/></linearGradient>
</defs>
<text class="fr-band" x="40" y="34">HOW YOU DRIVE IT</text>
<text class="fr-band" x="40" y="186">ONE SERVICE LAYER — SAME BEHAVIOUR ON EVERY SURFACE</text>
<text class="fr-band" x="40" y="348">THE ENGINE</text>
<text class="fr-band" x="40" y="524">THE VENDOR BOUNDARY — NO VENDOR TYPE LEAKS PAST IT</text>
<text class="fr-band" x="40" y="682">OUTSIDE THE PROCESS</text>
<text class="fr-band" x="40" y="826">AROUND THE PACKAGE</text>
<path class="fr-wire" d="M180 120 V214"/>
<path class="fr-wire" d="M500 120 V214"/>
<path class="fr-wire" d="M820 120 V214"/>
<path class="fr-wire" d="M180 282 V376"/>
<path class="fr-wire" d="M500 282 V376"/>
<path class="fr-wire fr-wire--quiet" d="M820 282 V376"/>
<path class="fr-wire" d="M180 458 V552"/>
<path class="fr-wire" d="M240 458 C240 510 440 500 460 552"/>
<path class="fr-wire" d="M500 458 V552"/>
<path class="fr-wire" d="M660 282 V496 Q660 516 690 516 H720 Q740 516 740 552"/>
<path class="fr-wire" d="M180 616 V710"/>
<path class="fr-wire" d="M500 616 V710"/>
<path class="fr-wire" d="M820 616 V710"/>
<path class="fr-wire fr-wire--quiet" d="M40 800 H960"/>
<circle class="fr-spark" r="4.5"><animateMotion dur="5.2s" repeatCount="indefinite" path="M180 120 V710"/></circle>
<circle class="fr-spark" r="4.5"><animateMotion dur="6.4s" begin="-2s" repeatCount="indefinite" path="M500 120 V710"/></circle>
<circle class="fr-spark" r="4.5"><animateMotion dur="5.8s" begin="-3.4s" repeatCount="indefinite" path="M820 120 V282 M660 282 V496 Q660 516 690 516 H720 Q740 516 740 552 M820 616 V710"/></circle>
<circle class="fr-spark" r="3.5"><animateMotion dur="4.6s" begin="-1.2s" repeatCount="indefinite" path="M240 458 C240 510 440 500 460 552"/></circle>
<g class="fr-node fr-node--surface fr-float"><rect x="40" y="48" width="280" height="86" rx="16"/><text class="fr-name" x="180" y="82" text-anchor="middle">REST API</text><text class="fr-path" x="180" y="103" text-anchor="middle">api/</text><text class="fr-detail" x="180" y="124" text-anchor="middle">routers · auth · limits · problem+json</text></g>
<g class="fr-node fr-node--surface fr-float"><rect x="360" y="48" width="280" height="86" rx="16"/><text class="fr-name" x="500" y="82" text-anchor="middle">CLI</text><text class="fr-path" x="500" y="103" text-anchor="middle">cli/</text><text class="fr-detail" x="500" y="124" text-anchor="middle">ingest · query · doctor · replay</text></g>
<g class="fr-node fr-node--surface fr-float"><rect x="680" y="48" width="280" height="86" rx="16"/><text class="fr-name" x="820" y="82" text-anchor="middle">Python package</text><text class="fr-path" x="820" y="103" text-anchor="middle">facade.py · sync.py</text><text class="fr-detail" x="820" y="124" text-anchor="middle">async facade and a blocking twin</text></g>
<g class="fr-node fr-float"><rect x="40" y="200" width="920" height="96" rx="16"/><text class="fr-name" x="500" y="236" text-anchor="middle">Services</text><text class="fr-path" x="500" y="257" text-anchor="middle">services/</text><text class="fr-detail" x="500" y="278" text-anchor="middle">ingestion · querying · generation · journal · idempotency · evaluation · replay · reindex · budget · doctor</text></g>
<g class="fr-node fr-float"><rect x="40" y="362" width="280" height="110" rx="16"/><text class="fr-name" x="180" y="397" text-anchor="middle">Workers</text><text class="fr-path" x="180" y="418" text-anchor="middle">workers/</text><text class="fr-detail" x="180" y="439" text-anchor="middle">parse pool → queue → embed pool</text><text class="fr-detail" x="180" y="458" text-anchor="middle">ledger · indexer</text></g>
<g class="fr-node fr-float"><rect x="360" y="362" width="280" height="110" rx="16"/><text class="fr-name" x="500" y="397" text-anchor="middle">Core</text><text class="fr-path" x="500" y="418" text-anchor="middle">core/</text><text class="fr-detail" x="500" y="439" text-anchor="middle">parsing · chunking · retrieval · rerank</text><text class="fr-detail" x="500" y="458" text-anchor="middle">context · cache · evals · breaker</text></g>
<g class="fr-node fr-float"><rect x="680" y="362" width="280" height="110" rx="16"/><text class="fr-name" x="820" y="397" text-anchor="middle">Foundations</text><text class="fr-path" x="820" y="418" text-anchor="middle">config/ · observability/</text><text class="fr-detail" x="820" y="439" text-anchor="middle">validated config · typed errors</text><text class="fr-detail" x="820" y="458" text-anchor="middle">metrics · traces · dashboard</text></g>
<g class="fr-node fr-float"><rect x="40" y="538" width="280" height="92" rx="16"/><text class="fr-name" x="180" y="573" text-anchor="middle">Vector DB adapters</text><text class="fr-path" x="180" y="594" text-anchor="middle">adapters/vectordb/</text><text class="fr-detail" x="180" y="615" text-anchor="middle">one contract, five backends</text></g>
<g class="fr-node fr-float"><rect x="360" y="538" width="280" height="92" rx="16"/><text class="fr-name" x="500" y="573" text-anchor="middle">Embedding adapters</text><text class="fr-path" x="500" y="594" text-anchor="middle">adapters/embeddings/</text><text class="fr-detail" x="500" y="615" text-anchor="middle">hosted and local models</text></g>
<g class="fr-node fr-float"><rect x="680" y="538" width="280" height="92" rx="16"/><text class="fr-name" x="820" y="573" text-anchor="middle">LLM adapters</text><text class="fr-path" x="820" y="594" text-anchor="middle">adapters/llm/</text><text class="fr-detail" x="820" y="615" text-anchor="middle">complete and stream</text></g>
<g class="fr-node fr-node--outside fr-float"><rect x="40" y="696" width="280" height="78" rx="16"/><text class="fr-detail" x="180" y="730" text-anchor="middle">Qdrant · pgvector · Chroma</text><text class="fr-detail" x="180" y="749" text-anchor="middle">Milvus · Weaviate</text></g>
<g class="fr-node fr-node--outside fr-float"><rect x="360" y="696" width="280" height="78" rx="16"/><text class="fr-detail" x="500" y="730" text-anchor="middle">OpenAI · Cohere</text><text class="fr-detail" x="500" y="749" text-anchor="middle">Hugging Face · Ollama</text></g>
<g class="fr-node fr-node--outside fr-float"><rect x="680" y="696" width="280" height="78" rx="16"/><text class="fr-detail" x="820" y="730" text-anchor="middle">OpenAI · Anthropic · Cohere</text><text class="fr-detail" x="820" y="749" text-anchor="middle">Ollama · OpenAI-compatible</text></g>
<g class="fr-node fr-node--satellite fr-float"><rect x="40" y="840" width="215" height="92" rx="16"/><text class="fr-name" x="147" y="872" text-anchor="middle">tests/</text><text class="fr-detail" x="147" y="893" text-anchor="middle">unit · contract · integration</text><text class="fr-detail" x="147" y="912" text-anchor="middle">eval · chaos · load · soak</text></g>
<g class="fr-node fr-node--satellite fr-float"><rect x="275" y="840" width="215" height="92" rx="16"/><text class="fr-name" x="382" y="872" text-anchor="middle">docs/</text><text class="fr-detail" x="382" y="893" text-anchor="middle">reference · decision records</text><text class="fr-detail" x="382" y="912" text-anchor="middle">benchmark ledger · task ledger</text></g>
<g class="fr-node fr-node--satellite fr-float"><rect x="510" y="840" width="215" height="92" rx="16"/><text class="fr-name" x="617" y="872" text-anchor="middle">scripts/</text><text class="fr-detail" x="617" y="893" text-anchor="middle">document and code gates</text><text class="fr-detail" x="617" y="912" text-anchor="middle">release stamping</text></g>
<g class="fr-node fr-node--satellite fr-float"><rect x="745" y="840" width="215" height="92" rx="16"/><text class="fr-name" x="852" y="872" text-anchor="middle">deploy/ · .github/</text><text class="fr-detail" x="852" y="893" text-anchor="middle">compose profiles</text><text class="fr-detail" x="852" y="912" text-anchor="middle">CI workflows</text></g>
</svg>
</div>
<div class="fr-legend">
<span><i class="fr-key--surface"></i>Entry surface</span>
<span><i></i>Package inside <code>src/fasterrag/</code></span>
<span><i class="fr-key--outside"></i>External system</span>
<span><i class="fr-key--satellite"></i>Repository, outside the package</span>
</div>
</div>

## How to read the map

Dependencies point **down**. A surface calls services; services drive the engine; the engine reaches
the outside world only through an adapter. Nothing in the engine knows which vector database or model
provider is configured, which is what makes them a line in `config.yaml` and not a code change.

The three surfaces at the top are deliberately thin. The REST routers, the CLI commands and the
Python facade each build the *same* services from the *same* validated settings, so a query behaves
identically — same retrieval, same errors, same trace — whichever way it arrived.

## The repository, package by package

<div class="fr-cards">
<div class="fr-card"><strong>Entry surfaces</strong><code>api/</code> · <code>cli/</code> · <code>facade.py</code> · <code>sync.py</code><p>FastAPI routers with authentication, body limits and RFC 9457 errors; the <code>fasterrag</code> command; and the importable <code>FasterRag</code> class in async and blocking forms. No business logic lives here.</p></div>
<div class="fr-card"><strong>Services</strong><code>services/</code><p>Orchestration: accepting and running ingestion jobs, retrieval, generation, the job journal, idempotency keys, evaluation, replay, blue/green re-embedding, cost budgets and the <code>doctor</code> checks.</p></div>
<div class="fr-card"><strong>Workers</strong><code>workers/</code><p>The ingestion pipeline: a process pool that parses and chunks, a bounded queue, embedding workers, the indexer, and the ledger that records a document as indexed only once every chunk is written.</p></div>
<div class="fr-card"><strong>Core</strong><code>core/</code><p>The vendor-free library: parsers, chunkers, BM25 encoding and rank fusion, the reranker, context assembly, prompt contracts, caches, evaluation metrics, the circuit breaker and the rate limiter.</p></div>
<div class="fr-card"><strong>Adapters</strong><code>adapters/vectordb/</code> · <code>adapters/embeddings/</code> · <code>adapters/llm/</code><p>One interface and one factory per kind of vendor. Every vendor exception is translated into the project's error taxonomy at this boundary.</p></div>
<div class="fr-card"><strong>Foundations</strong><code>config/</code> · <code>errors.py</code> · <code>observability/</code><p>The validated configuration schema and loader, the typed error hierarchy with stable codes, and metrics, structured logs, trace export and the read-only dashboard.</p></div>
<div class="fr-card"><strong>Tests</strong><code>tests/</code><p>Unit tests, a contract suite shared by every vector-database adapter, integration tests against real backends, a retrieval evaluation harness, and chaos, load and soak tiers.</p></div>
<div class="fr-card"><strong>Docs, scripts, deploy</strong><code>docs/</code> · <code>scripts/</code> · <code>deploy/</code> · <code>.github/</code><p>This documentation, the gates that keep it true to the code, the container image and compose profiles, and the CI workflows.</p></div>
</div>

## Follow a document in

1. **A source arrives** through `POST /v1/ingest`, `fasterrag ingest` or `FasterRag.ingest` — a path, a URL or an inline payload — and `services/ingestion.py` records a job in the journal.
2. **It is parsed and chunked** in a worker process (`workers/cpu_pool.py`, using `core/parsing/` and `core/chunking/`). A file that cannot be parsed costs that one document: it goes to the dead-letter queue with a reason code and the job continues.
3. **Chunks stream through a bounded queue** to the embedding workers (`workers/embed_pool.py`), which batch them and call the configured embedding adapter, retrying what the provider says is retryable.
4. **The indexer writes** dense and BM25 sparse vectors together through the vector-database adapter (`workers/indexer.py`).
5. **Only then is the document recorded as indexed.** `workers/ledger.py` counts the chunks actually written and commits the document to the journal when the last one lands.

## Follow a question out

1. **A question arrives** through `POST /v1/query`, `fasterrag query` or `FasterRag.query`.
2. **Retrieval runs two searches at once** — dense vectors and BM25 — and fuses them with Reciprocal Rank Fusion (`services/querying.py`, `core/retrieval/`). A cross-encoder can then rerank the shortlist (`core/rerank/`).
3. **Context is assembled** within a token budget, dropping near-duplicates (`core/context.py`).
4. **The LLM adapter generates**, streamed as server-sent events, and citation markers are resolved against the chunks that were supplied (`services/generation.py`, `core/generation.py`).
5. **A trace is stored**, so the same question can later be replayed under a different configuration and the two results compared.

## Where to go next

[Run the quickstart](quickstart.md){ .md-button .md-button--primary }
[Architecture in depth](architecture.md){ .md-button }
[Repository structure](structure.md){ .md-button }
[What is and is not built](release-readiness.md){ .md-button }
