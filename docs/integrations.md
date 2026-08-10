# integrations.md — Supported Providers & How Config Enables Them

Everything is selected purely through `config.yaml`; secrets by env-var name only; **every integration toggle defaults to `false`**. Each row below carries a **Status**: *shipped* means implemented and tested in this repo today; *specified* means the config surface and behavior are designed but the adapter has not been built (its task is in [todo.md](todo.md)) — selecting a specified provider fails fast with a `ConfigError`, never silently. Conformance for any adapter, in-repo or third-party, means passing the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). The entry-point extension contract ([python-api.md](python-api.md)) is **shipped** (TASK-0173 ✅): all three factories resolve registered plugins, and a built-in name always wins so an installed package cannot silently take over a configured provider.

## 1. Vector databases (`vector_db.provider`)

| Provider | Value | Status | Modes | Secret (env var) | Notes |
|---|---|---|---|---|---|
| **Qdrant** (reference) | `qdrant` | **Shipped** — contract suite passes in all three modes | docker (system-managed) · external local · remote host:port | `QDRANT_API_KEY` | Reference implementation. REST 6333 + gRPC 6334 (both must be reachable; `prefer_grpc` default `false`). Named Docker volume mandatory on Windows/WSL. |
| Milvus | `milvus` | Specified (TASK-0049) | external · remote | `MILVUS_API_KEY` | gRPC 19530 (default). |
| Weaviate | `weaviate` | Specified (TASK-0049) | external · remote | `WEAVIATE_API_KEY` | HTTP/gRPC per instance config. |
| Pinecone | `pinecone` | Specified (TASK-0049) | remote (managed SaaS) | `PINECONE_API_KEY` | Serverless/pod indexes; region via adapter options. |
| **pgvector** | `pgvector` | **Shipped** — contract suite passes against a live PostgreSQL 17 + pgvector (TASK-0232 ✅) | external · remote | DSN in the env var named by `vector_db.pgvector.dsn_env` (no default — required only when the provider is pgvector) | Runs inside existing PostgreSQL. The BM25 leg stores fasterRag's term frequencies in a term table with IDF computed per query — **not** `tsvector`, which would re-tokenize server-side and make one query hit different terms than on Qdrant. Aliases are catalog rows, not table renames. Snapshots are logical copies in the same database, not physical backups. |
| Chroma | `chroma` | Specified (TASK-0049) | external · remote | `CHROMA_API_KEY` | Lightweight; dev/small corpora. |

Enable by config only:

```yaml
vector_db:
  provider: pgvector          # was: qdrant — this line is the entire migration trigger
```

Backend swap = config change + D2 zero-downtime reindex (or D11 vector copy where dimensions match). No application code changes.

## 2. Embedding providers (`embeddings.provider`)

| Provider | Value | Secret (env var) | Local? | Notes |
|---|---|---|---|---|
| HuggingFace / sentence-transformers | `huggingface` | — | ✅ | Default; runs on CPU/GPU in the embedding pool; no data leaves the host. |
| OpenAI | `openai` | `OPENAI_API_KEY` | ❌ | Batched API calls; prompt caching honored where offered. |
| Cohere | `cohere` | `COHERE_API_KEY` | ❌ | Batched. |
| Ollama (local server) | `ollama` | — | ✅ | `llm.base_url`-style endpoint override supported. |

**Tiered embedding** (`embeddings.tiering`): route document classes to different models — cheap models for high-volume/low-priority classes, higher-cost models where retrieval precision matters:

```yaml
embeddings:
  tiering:
    enabled: true
    rules:
      - match: {priority_class: archive}
        provider: huggingface
        model: BAAI/bge-small-en-v1.5
      - match: {priority_class: legal}
        provider: openai
        model: text-embedding-3-large
```

All four embedding providers above are **shipped** (the `huggingface` local default requires the `.[huggingface]` extra; a missing extra fails fast naming the install command).

## 3. LLM providers (`llm.provider`)

All five LLM providers below are **shipped**, with streaming.

| Provider | Value | Secret (env var) | Streaming | Notes |
|---|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | ✅ | |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | ✅ | e.g. `model: claude-opus-5`; provider prompt caching used for shared prefixes. |
| Cohere | `cohere` | `COHERE_API_KEY` | ✅ | |
| Ollama (local) | `ollama` | — | ✅ | Fully local generation. |
| Any OpenAI-compatible endpoint | `openai_compatible` | per endpoint | ✅ | `llm.base_url` required — covers vLLM, LM Studio, llama.cpp servers, TGI, and most hosted gateways. |

**Cost coverage.** OpenAI, Anthropic, and part of the Cohere lineup have dated list prices recorded, so their traffic reaches `fasterrag_cost_usd_total`; Ollama is priced at zero because it is local. `openai_compatible` never is — a gateway sets its own prices, and charging its traffic at the upstream vendor's rate would be a fabricated bill. Unpriced traffic is counted by `fasterrag_unpriced_tokens_total` rather than silently dropped. The exact per-model coverage and its check dates are in [config-reference.md](config-reference.md#which-models-carry-a-price).

## 4. Rerankers (`retrieval.reranker_model`)

Cross-encoder models loaded locally in the query path (e.g. `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`). Selected by model id; toggle with `retrieval.rerank`. Stage latency is unmeasured (TASK-0084) — the "~100–300 ms" this row previously quoted had no source and has been withdrawn.

## 5. Observability tools (all default `false`)

| Tool | Toggle | Status | What flipping it does | Result |
|---|---|---|---|---|
| fasterRag dashboard | `observability.dashboard: true` | **Shipped** (TASK-0045 ✅, tenant-scoped by TASK-0184 ✅) — a separate ASGI app on its own port, declaring no write route at all | Starts the read-only inspection UI | `http://<host>:8080` |
| OpenTelemetry | `observability.otel: true` | Shipped and verified once by hand against a running collector (a manual run, not a committed regression test) — `otel/opentelemetry-collector-contrib` 0.140.1 accepted both signals with no warning, refusal, or rejection in its log (TASK-0231): five spans carrying the fasterRag trace id with the root/child structure intact, both signals scoped `fasterrag`, the counter as a monotonic `Sum`, the gauge as a `Gauge`, and the histogram's bucket counts summing to its observation count. Needs the `otel` extra | Exports spans and metrics via OTLP to `otel_endpoint`, treated as a base URL from which `/v1/traces` and `/v1/metrics` are derived | your collector |
| **Langfuse** | `observability.langfuse: true` | **Shipped**: provisioning verified end-to-end once by hand (six containers up, `/api/public/health` OK, bootstrapped keys authenticated — TASK-0043 ✅; a manual run, not a committed regression test), the toggle triggers it at startup (TASK-0150 ✅), it is doctor-gated (TASK-0149 ✅), and trace export is implemented (TASK-0151 ✅). **Export is not yet verified against a live Langfuse** — the wire format is unit- and socket-tested only, so field names are checked against the documented API rather than against the server (TASK-0178) | Auto-installs the self-hosted Langfuse v3 compose stack (web, worker, Postgres, ClickHouse ≥ 24.3, Redis, MinIO), generates + persists secrets, headless-bootstraps org/project/keys/user via `LANGFUSE_INIT_*`, health-checks | **`http://<host>:3000`** |
| **Grafana** | `observability.grafana: true` | **Shipped**: provisioning verified end-to-end once by hand — fasterRag → Prometheus → Grafana's provisioned datasource returned real series (TASK-0044 ✅, panels fixed under TASK-0154 ✅); a manual run, not a committed regression test. Toggle triggers it at startup (TASK-0146 ✅) and it is doctor-gated (TASK-0147 ✅) | Provisioning-as-code: datasource + dashboard YAML/JSON under `/etc/grafana/provisioning/`, `editable: false`, `allowUiUpdates: false`, 30 s file reload | Grafana with fasterRag dashboards, read-only for provisioned resources |

The target contract for every row is: config toggle → doctor-gated auto-provisioning → running URL, with **zero application-code changes at toggle time** ([observability.md](observability.md)). **As built, that contract holds for Langfuse and Grafana**: both are reachable from the CLI command *and* from the toggle at startup, and both are doctor-gated before anything is mutated. What remains open is verification against a live Langfuse (TASK-0178), not construction.

## 6. Adding a provider that isn't listed

Implement the adapter base class, register it via the `fasterrag.vectordb` / `fasterrag.embeddings` / `fasterrag.llm` entry point, pass the contract suite, and it becomes selectable in `config.yaml` like any built-in ([python-api.md](python-api.md) §Extending). This is the supported path for the long tail of vector DBs, embedders, and LLM gateways. **Status: shipped** (TASK-0173 ✅). All three factories resolve entry points, report a plugin's origin in `available_providers()`, reject a registered object that is not an adapter, and always prefer a built-in name — so an installed package cannot silently take over a configured provider. Verified with 15 tests registering a fake plugin against each group.
