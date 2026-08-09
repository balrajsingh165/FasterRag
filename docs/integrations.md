# integrations.md — Supported Providers & How Config Enables Them

Everything is selected purely through `config.yaml`; secrets by env-var name only; **every integration toggle defaults to `false`**. Each row below carries a **Status**: *shipped* means implemented and tested in this repo today; *specified* means the config surface and behavior are designed but the adapter has not been built (its task is in [todo.md](todo.md)) — selecting a specified provider fails fast with a `ConfigError`, never silently. Conformance for any adapter, in-repo or third-party, means passing the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). The entry-point extension contract ([python-api.md](python-api.md)) is itself not yet implemented (TASK-0163).

## 1. Vector databases (`vector_db.provider`)

| Provider | Value | Status | Modes | Secret (env var) | Notes |
|---|---|---|---|---|---|
| **Qdrant** (reference) | `qdrant` | **Shipped** — contract suite passes in all three modes | docker (system-managed) · external local · remote host:port | `QDRANT_API_KEY` | Reference implementation. REST 6333 + gRPC 6334 (both must be reachable; `prefer_grpc` default `false`). Named Docker volume mandatory on Windows/WSL. |
| Milvus | `milvus` | Specified (TASK-0049) | external · remote | `MILVUS_API_KEY` | gRPC 19530 (default). |
| Weaviate | `weaviate` | Specified (TASK-0049) | external · remote | `WEAVIATE_API_KEY` | HTTP/gRPC per instance config. |
| Pinecone | `pinecone` | Specified (TASK-0049) | remote (managed SaaS) | `PINECONE_API_KEY` | Serverless/pod indexes; region via adapter options. |
| pgvector | `pgvector` | Specified (TASK-0049; recommended next — proves the contract against SQL) | external · remote | `PGVECTOR_DSN_ENV` → DSN in env | Runs inside existing PostgreSQL; BM25 leg uses PG full-text. |
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

Cross-encoder models loaded locally in the query path (e.g. `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`). Selected by model id; ~100–300 ms per query; toggle with `retrieval.rerank`.

## 5. Observability tools (all default `false`)

| Tool | Toggle | Status | What flipping it does | Result |
|---|---|---|---|---|
| fasterRag dashboard | `observability.dashboard: true` | Not built (S14, last by design) | Starts the read-only inspection UI | `http://<host>:8080` |
| OpenTelemetry | `observability.otel: true` | Shipped — the four RAG spans and the metric catalogue both export over OTLP/HTTP to `otel_endpoint`, spans preserving the fasterRag trace id. Not yet verified against a running collector (B7). Needs the `otel` extra | Exports spans and metrics via OTLP to `otel_endpoint` | your collector |
| **Langfuse** | `observability.langfuse: true` | **Provisioning shipped and verified end-to-end** via `fasterrag provision langfuse`; the config toggle does not yet trigger it at startup (TASK-0150), it is not yet doctor-gated (TASK-0149), and trace export into the stack is pending (TASK-0151) | Auto-installs the self-hosted Langfuse v3 compose stack (web, worker, Postgres, ClickHouse ≥ 24.3, Redis, MinIO), generates + persists secrets, headless-bootstraps org/project/keys/user via `LANGFUSE_INIT_*`, health-checks | **`http://<host>:3000`** |
| **Grafana** | `observability.grafana: true` | **Provisioning shipped and verified end-to-end** via `fasterrag provision grafana` (real series rendered); toggle-at-startup pending (TASK-0146), doctor gate pending (TASK-0147) | Provisioning-as-code: datasource + dashboard YAML/JSON under `/etc/grafana/provisioning/`, `editable: false`, `allowUiUpdates: false`, 30 s file reload | Grafana with fasterRag dashboards, read-only for provisioned resources |

The target contract for every row is: config toggle → doctor-gated auto-provisioning → running URL, with **zero application-code changes at toggle time** ([observability.md](observability.md)). As built, provisioning runs via the CLI command; the toggle-triggered and doctor-gated halves are the open tasks named above.

## 6. Adding a provider that isn't listed

Implement the adapter base class, register it via the `fasterrag.vectordb` / `fasterrag.embeddings` / `fasterrag.llm` entry point, pass the contract suite, and it becomes selectable in `config.yaml` like any built-in ([python-api.md](python-api.md) §Extending). This is the supported path for the long tail of vector DBs, embedders, and LLM gateways. **Status: the entry-point registration mechanism is not yet implemented (TASK-0163) — today the factories resolve built-in adapters only, so third-party providers require a fork until it lands.**
