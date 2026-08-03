# fasterRag

**A FastAPI-based, backend-only, all-in-one Retrieval-Augmented Generation (RAG) framework engineered for very large datasets.** Install it as a Python package, drive it from a CLI or REST API, plug in any vector database, any embedding model, and any LLM provider — all through one config file. Its goal is to be the fastest, most efficient, and most reliable RAG framework available, and this repository holds itself to a standing rule: **a claim without a measurement is a bug** ([docs/benchmarks.md](docs/benchmarks.md)).

> **Status: build phase, pre-beta (not yet on PyPI — install from source).** The engineering specification in [`docs/`](docs/) is being implemented in tracked vertical slices; slices **S1–S13 have landed** and the full ledger lives in [docs/todo.md](docs/todo.md).
>
> | Shipped | Partial | Not yet |
> |---|---|---|
> | Config loader (fail-fast) · Qdrant adapter (docker/external/remote) + contract suite · `doctor` · parsing (PDF/OCR, HTML, MD, DOCX, tabular) · five chunkers with property-tested invariants · checkpointed ingestion with dedup + DLQ from `path`/`url`/`inline` sources · hybrid retrieval + RRF(k=60) · cross-encoder rerank · SSE generation with span citations · grounded-or-refuse · embedding + semantic caches · eval harness (recall/MRR/nDCG + faithfulness) · `FasterRag` facade (async + sync) + entry-point plugins · CLI · REST routers · trace store + replay · index lockfile + drift · blue/green reindex + rollback · chaos suite · benchmark suite + ledger · autopilot (suggest-only) · Langfuse/Grafana provisioning | Degradation ladder (2 of 3 rungs; no circuit breaker yet) · cost estimator (budgets not enforced) · OTel spans recorded locally (OTLP export pending) | Security layer (auth/tenancy/rate limits — config validates but is **not enforced**) · D11 export/import · Milvus/Weaviate/Pinecone/pgvector/Chroma adapters · observability dashboard (S14) · **citable benchmarks — the [ledger](docs/benchmarks.md) holds only non-citable entries until the isolated-hardware baseline run** |

---

## What it is

fasterRag is a **framework**, not a demo: you adopt it three ways, all backed by one engine and one config contract, with identical behavior and identical typed errors everywhere.

| Surface | How | For |
|---|---|---|
| **Python package** *(shipped — `FasterRag` facade async + sync, standalone components, entry-point plugins)* | `pip install -e .` → `from fasterrag import FasterRag`, or piecemeal: `from fasterrag.chunking import RecursiveChunker` | Embedding RAG in your own application; using components standalone (chunkers, hybrid fusion, evals) |
| **CLI** *(shipped)* | `fasterrag ingest / query / doctor / estimate / benchmark / replay ...` | Operations, scripting, CI |
| **REST API** *(shipped)* | `fasterrag serve` → `/v1/ingest`, `/v1/query` (SSE streaming) | Services and non-Python clients |

There is deliberately **no control GUI**. A separate, optional, self-hosted dashboard exists for *observability only* — metrics, costs, latencies, and full LLM input/output history — and cannot control the system ([docs/adr/ADR-0005](docs/adr/ADR-0005-api-cli-only-control-plane.md)).

## Why another RAG framework

Most RAG stacks make you assemble chunking, retrieval, reranking, caching, evaluation, and observability by hand — then leave you guessing whether any of it works. fasterRag's answer is an **engineered pipeline with proof built in**:

- **Speed by architecture** *(shipped; unmeasured — no citable benchmark yet)*: decoupled CPU parse/chunk pool streaming into stateful embedding workers (model loaded once per worker), bounded queues with backpressure, batched embedding and upserts, async-everything API, SSE streaming for time-to-first-token. See [docs/architecture.md](docs/architecture.md).
- **Retrieval quality by default** *(shipped except contextual enrichment, which is pending)*: hybrid dense + BM25 retrieval fused with Reciprocal Rank Fusion (k=60, per Cormack/Clarke/Büttcher SIGIR 2009) and cross-encoder reranking; contextual-retrieval-style chunk enrichment (Anthropic's Sept 2024 results: −49% failed retrievals, −67% with reranking) is specified and tracked in [docs/todo.md](docs/todo.md). See [docs/adr/ADR-0004](docs/adr/ADR-0004-hybrid-search-plus-reranking.md).
- **Total pluggability** *(partial)*: Qdrant (reference adapter, shipped, contract-suite-tested in all three modes); Milvus/Weaviate/Pinecone/pgvector/Chroma adapters pending. OpenAI/Cohere/HuggingFace/Ollama embeddings and OpenAI/Anthropic/Cohere/Ollama/OpenAI-compatible LLMs: shipped. Entry-point plugin registration: pending. See [docs/integrations.md](docs/integrations.md).
- **Reliability as a feature** *(partial)*: typed error taxonomy, RFC 9457 problem responses, bulkheads, timeouts + backoff retries, checkpointed exactly-once ingestion with a dead-letter queue, zero-downtime blue/green reindexing, and a public chaos suite: shipped. Circuit breaker and the `cache_only` degradation rung: specified, not yet built. See [docs/reliability.md](docs/reliability.md), [docs/differentiators.md](docs/differentiators.md).

## Sixty-second tour *(works today, from a source checkout)*

```bash
pip install -e ".[all]"          # PyPI publication lands with the first tagged beta
fasterrag doctor                 # preflight: Docker, ports, disk, keys — every failure prints a fix
fasterrag provision qdrant       # config-driven; system-managed container, named volume
fasterrag estimate ./my-docs/    # token counts + projected embedding cost BEFORE ingesting
fasterrag ingest ./my-docs/ --watch
fasterrag query "What does the vendor agreement say about termination?"
```

Or in Python — the `FasterRag` facade (async here; a blocking twin lives at `fasterrag.sync`) and the standalone components (`fasterrag.parsing`, `.chunking`, `.retrieval`, `.rerank`, `.evals`) are both shipped ([docs/python-api.md](docs/python-api.md)):

```python
from fasterrag import FasterRag

async with FasterRag.from_config("config.yaml") as rag:
    await (await rag.ingest(["./my-docs/"])).wait()
    result = await rag.query("What does the vendor agreement say about termination?")
    print(result.answer, result.citations)
```

Everything above is driven by two files: `config.yaml` (all behavior, committable, no secrets) and `.env` (secrets only, referenced by env-var name). Flipping `observability.langfuse: true` auto-provisions a full self-hosted Langfuse v3 stack and returns `http://<host>:3000` — zero code changes ([docs/observability.md](docs/observability.md)).

## The twelve differentiators

Specified with acceptance tests and proof metrics in [docs/differentiators.md](docs/differentiators.md):

1. **Index lockfile & reproducible builds** — drift becomes impossible to have silently
2. **Zero-downtime reindexing** — blue/green + eval gate + instant rollback
3. **Checkpointed, idempotent ingestion** — crash-resume, dedup, DLQ with reason codes
4. **Degradation ladder** — explicit degraded modes, never a silent quality drop
5. **Grounded-or-refuse answering** — span citations mandatory; refuses instead of hallucinating
6. **Autopilot** — eval-driven tuning suggestions from *your* corpus; never auto-applies
7. **Continuous retrieval regression gate** — you cannot accidentally ship a worse RAG
8. **Time-travel replay** — re-execute past queries under a candidate config, diff the results
9. **Cost governor & preflight estimator** — know the bill before ingesting; enforce budgets at runtime
10. **`fasterrag doctor`** — preflight diagnostics with concrete fix-it instructions
11. **Portability & anti-lock-in** — export/import everything; leaving is a supported feature
12. **Chaos-certified** — the fault-injection suite is public and repeatable

## Documentation map

All documentation lives in [`docs/`](docs/) (only `CLAUDE.md` and this README sit at root).

| Read this | For |
|---|---|
| **[quickstart.md](docs/quickstart.md)** | **Start here** — zero to an answered query via CLI, Python, or REST |
| [cookbook.md](docs/cookbook.md) | Ready-made config recipes (local/air-gapped, max accuracy, cost-optimized, multi-tenant, …) |
| [migration-guide.md](docs/migration-guide.md) | Coming from LangChain / LlamaIndex / Haystack / a hand-rolled stack |
| [troubleshooting.md](docs/troubleshooting.md) | Symptom → cause → fix |
| [scope.md](docs/scope.md) | Vision, goals, non-goals, the 22-point RAG pain-point catalogue and mitigations |
| [architecture.md](docs/architecture.md) | Workers, batching, chunking strategies, hybrid retrieval, caching, fault tolerance |
| [flow.md](docs/flow.md) | End-to-end Mermaid flows (ingestion, worker hand-off, query, cache, provisioning) |
| [structure.md](docs/structure.md) | Intended repository layout and layer responsibilities |
| [config-reference.md](docs/config-reference.md) | Every `config.yaml` key: type, default, validation, description |
| [python-api.md](docs/python-api.md) | The importable package surface, standalone components, plugin contract |
| [api-reference.md](docs/api-reference.md) | REST endpoints, SSE streaming semantics, RFC 9457 error model |
| [cli-reference.md](docs/cli-reference.md) | Every command and flag |
| [data-model.md](docs/data-model.md) | Canonical entity schemas (Document, Chunk, Job, Trace, IndexLock) and their invariants |
| [prompts.md](docs/prompts.md) | The four LLM call-site contracts: generation, enrichment, faithfulness, golden-set |
| [integrations.md](docs/integrations.md) | Vector DBs, embedding/LLM providers, observability tools |
| [observability.md](docs/observability.md) | Metrics catalogue, tracing, dashboard, Langfuse/Grafana auto-provisioning |
| [deployment.md](docs/deployment.md) | Self-hosting modes, sizing, revert playbook |
| [security.md](docs/security.md) | Secrets policy, auth/scopes, multi-tenancy, supply chain |
| [reliability.md](docs/reliability.md) · [failure-modes.md](docs/failure-modes.md) · [slo.md](docs/slo.md) · [disaster-recovery.md](docs/disaster-recovery.md) | Error taxonomy, FMEA, SLIs/SLOs, backup + restore drill |
| [testing-strategy.md](docs/testing-strategy.md) · [performance.md](docs/performance.md) · [benchmarks.md](docs/benchmarks.md) | Testing pyramid incl. chaos suite, measurement methodology, the benchmark ledger |
| [differentiators.md](docs/differentiators.md) | The twelve capabilities, each with acceptance test + proof metric |
| [glossary.md](docs/glossary.md) · [references.md](docs/references.md) · [archive-format.md](docs/archive-format.md) | Canonical terminology, external evidence sources, portability archive spec |
| [adr/](docs/adr/) | Architecture Decision Records (MADR) |
| [todo.md](docs/todo.md) | The ONE task file: done, in progress, pending, todo, future |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) · [CHANGELOG.md](docs/CHANGELOG.md) | Rules and history |

## Project principles (short version)

1. **Config drives everything; secrets live only in `.env`.** Every integration toggle defaults to `false`.
2. **Control plane is programmatic only** (API/CLI/library); the dashboard is read-only observability.
3. **A claim without a measurement is a bug** — unmeasured statements are goals, and "fastest" is only ever claimed against a baseline we measured ourselves.
4. **Reliability is tested, not asserted** — FMEA rows name their proving tests; the chaos suite is public.
5. **Incremental, revertible shipping** — feature branches, single-line commits, slice tags, a written revert playbook.

## Contributing

Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) first — commit format (single-line, no trailers), comment policy (docstrings only), the one-todo-file rule, and the quality gates are strictly enforced.

## License

See [LICENSE](LICENSE).
