# fasterRag

**A FastAPI-based, backend-only, all-in-one Retrieval-Augmented Generation (RAG) framework engineered for very large datasets.** Install it as a Python package, drive it from a CLI or REST API, plug in any vector database, any embedding model, and any LLM provider — all through one config file. Its goal is to be the fastest, most efficient, and most reliable RAG framework available, and this repository holds itself to a standing rule: **a claim without a measurement is a bug** ([docs/benchmarks.md](docs/benchmarks.md)).

> **Status: documentation-first, pre-beta.** The complete engineering specification lives in [`docs/`](docs/); implementation ships in tracked vertical slices ([docs/todo.md](docs/todo.md)). No implementation code exists yet — everything below marked *(planned)* describes the specified beta contract.

---

## What it is

fasterRag is a **framework**, not a demo: you adopt it three ways, all backed by one engine and one config contract, with identical behavior and identical typed errors everywhere.

| Surface | How | For |
|---|---|---|
| **Python package** *(planned)* | `pip install fasterrag` → `from fasterrag import FasterRag` | Embedding RAG in your own application; using components standalone (chunkers, hybrid fusion, evals) |
| **CLI** *(planned)* | `fasterrag ingest / query / doctor / benchmark ...` | Operations, scripting, CI |
| **REST API** *(planned)* | `fasterrag serve` → `/v1/ingest`, `/v1/query` (SSE streaming) | Services and non-Python clients |

There is deliberately **no control GUI**. A separate, optional, self-hosted dashboard exists for *observability only* — metrics, costs, latencies, and full LLM input/output history — and cannot control the system ([docs/adr/ADR-0005](docs/adr/ADR-0005-api-cli-only-control-plane.md)).

## Why another RAG framework

Most RAG stacks make you assemble chunking, retrieval, reranking, caching, evaluation, and observability by hand — then leave you guessing whether any of it works. fasterRag's answer is an **engineered pipeline with proof built in**:

- **Speed by architecture** *(planned)*: decoupled CPU parse/chunk pool streaming into stateful GPU embedding workers (model loaded once per worker), bounded queues with backpressure, batched embedding and upserts, async-everything API, SSE streaming for time-to-first-token. See [docs/architecture.md](docs/architecture.md).
- **Retrieval quality by default** *(planned)*: hybrid dense + BM25 retrieval fused with Reciprocal Rank Fusion (k=60, per Cormack/Clarke/Büttcher SIGIR 2009), cross-encoder reranking, and contextual-retrieval-style chunk enrichment (Anthropic's Sept 2024 results: −49% failed retrievals, −67% with reranking). See [docs/adr/ADR-0004](docs/adr/ADR-0004-hybrid-search-plus-reranking.md).
- **Total pluggability** *(planned)*: Qdrant (reference), Milvus, Weaviate, Pinecone, pgvector, Chroma; OpenAI/Cohere/HuggingFace/Ollama embeddings; any LLM incl. OpenAI-compatible endpoints. One config line swaps a backend; third-party providers register via entry points and must pass a shared contract test suite. See [docs/integrations.md](docs/integrations.md).
- **Reliability as a feature** *(planned)*: typed error taxonomy, RFC 9457 problem responses, circuit breakers, bulkheads, a tested degradation ladder (never a silent quality drop), checkpointed exactly-once ingestion with a dead-letter queue, zero-downtime blue/green reindexing, and a public chaos suite. See [docs/reliability.md](docs/reliability.md), [docs/differentiators.md](docs/differentiators.md).

## Sixty-second tour *(planned surface)*

```bash
pip install "fasterrag[all]"
fasterrag doctor                 # preflight: Docker, ports, disk, keys — every failure prints a fix
fasterrag provision qdrant       # config-driven; system-managed container, named volume
fasterrag estimate ./my-docs/    # token counts + projected embedding cost BEFORE ingesting
fasterrag ingest ./my-docs/ --watch
fasterrag query "What does the vendor agreement say about termination?"
```

Or in Python:

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
| [scope.md](docs/scope.md) | Vision, goals, non-goals, the 22-point RAG pain-point catalogue and mitigations |
| [architecture.md](docs/architecture.md) | Workers, batching, chunking strategies, hybrid retrieval, caching, fault tolerance |
| [flow.md](docs/flow.md) | End-to-end Mermaid flows (ingestion, worker hand-off, query, cache, provisioning) |
| [structure.md](docs/structure.md) | Intended repository layout and layer responsibilities |
| [config-reference.md](docs/config-reference.md) | Every `config.yaml` key: type, default, validation, description |
| [python-api.md](docs/python-api.md) | The importable package surface, standalone components, plugin contract |
| [api-reference.md](docs/api-reference.md) | REST endpoints, SSE streaming semantics, RFC 9457 error model |
| [cli-reference.md](docs/cli-reference.md) | Every command and flag |
| [integrations.md](docs/integrations.md) | Vector DBs, embedding/LLM providers, observability tools |
| [observability.md](docs/observability.md) | Metrics catalogue, tracing, dashboard, Langfuse/Grafana auto-provisioning |
| [deployment.md](docs/deployment.md) | Self-hosting modes, sizing, revert playbook |
| [security.md](docs/security.md) | Secrets policy, auth/scopes, multi-tenancy, supply chain |
| [reliability.md](docs/reliability.md) · [failure-modes.md](docs/failure-modes.md) · [slo.md](docs/slo.md) · [disaster-recovery.md](docs/disaster-recovery.md) | Error taxonomy, FMEA, SLIs/SLOs, backup + restore drill |
| [testing-strategy.md](docs/testing-strategy.md) · [performance.md](docs/performance.md) · [benchmarks.md](docs/benchmarks.md) | Testing pyramid incl. chaos suite, measurement methodology, the benchmark ledger |
| [differentiators.md](docs/differentiators.md) | The twelve capabilities, each with acceptance test + proof metric |
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
