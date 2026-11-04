# integrations.md — Supported Providers & How Config Enables Them

Everything is selected purely through `config.yaml`; secrets by env-var name only; **every integration toggle defaults to `false`**. Each row below carries a **Status**: *shipped* means implemented and tested in this repo today; *specified* means the config surface and behavior are designed but the adapter has not been built (its task is in [todo.md](todo.md)) — selecting a specified provider fails fast with a `ConfigError`, never silently. Conformance for any adapter, in-repo or third-party, means passing the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). The entry-point extension contract ([python-api.md](python-api.md)) is **shipped** (TASK-0173 ✅): all three factories resolve registered plugins, and a built-in name always wins so an installed package cannot silently take over a configured provider.

## 1. Vector databases (`vector_db.provider`)

> **Security note (2026-11-04).** `chromadb` 1.5.9 — the latest release — carries four unfixed advisories (PYSEC-2026-311, -3813, -3814, -3815). All four are in Chroma's HTTP server, which fasterRag never starts; the adapter runs embedded against a local directory. See [security.md](security.md) §6 for the reasoning and its limits. If you run a Chroma server of your own, those advisories apply to it regardless of fasterRag.

| Provider | Value | Status | Modes | Secret (env var) | Notes |
|---|---|---|---|---|---|
| **Qdrant** (reference) | `qdrant` | **Shipped** — contract suite passes in all three modes | docker (system-managed) · external local · remote host:port | `QDRANT_API_KEY` | Reference implementation. REST 6333 + gRPC 6334 (both must be reachable; `prefer_grpc` default `false`). Named Docker volume mandatory on Windows/WSL. |
| Milvus | `milvus` | Specified (TASK-0049) | external · remote | `MILVUS_API_KEY` | gRPC 19530 (default). |
| Weaviate | `weaviate` | Specified (TASK-0049) | external · remote | `WEAVIATE_API_KEY` | HTTP/gRPC per instance config. |
| Pinecone | `pinecone` | Specified (TASK-0049) | remote (managed SaaS) | `PINECONE_API_KEY` | Serverless/pod indexes; region via adapter options. |
| **pgvector** | `pgvector` | **Shipped** — contract suite passes against a live PostgreSQL 17 + pgvector (TASK-0232 ✅) | external · remote | DSN in the env var named by `vector_db.pgvector.dsn_env` (no default — required only when the provider is pgvector) | Runs inside existing PostgreSQL. The BM25 leg stores fasterRag's term frequencies in a term table with IDF computed per query — **not** `tsvector`, which would re-tokenize server-side and make one query hit different terms than on Qdrant. Aliases are catalog rows, not table renames. Snapshots are logical copies in the same database, not physical backups. |
| **Chroma** | `chroma` | **Shipped** — contract suite passes against a live embedded Chroma 1.5 store (TASK-0049) | embedded (in-process, file-backed) | none — there is no credential to supply | Runs inside the fasterRag process against `vector_db.chroma.path`; `mode`, `host`, `port`, and `api_key_env` are ignored, so a Chroma deployment needs no server and no secret. Aliases are records in a `fasterrag-aliases` collection, not renames, so both sides of a blue/green reindex stay addressable. The BM25 leg stores fasterRag's term frequencies as per-document metadata keys and computes IDF over the live corpus in the adapter, because Chroma has no corpus-rarity scoring primitive — correct, but it reads every posting a query's terms match into the process, so it does not scale the way Qdrant's server-side sparse index does. Snapshots are copies into a second collection in the same directory, not physical backups. Exercised so far only as an embedded store on one developer machine, with no Chroma-server or multi-process-writer run behind it. |

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

## 3a. Live provider verification — what `pytest -m eval` does and does not prove

Every other provider test in this repository substitutes a fake client. That proves the adapter issues the request it meant to; it proves nothing about whether the vendor still accepts it, which is the drift that actually breaks a release — a renamed response field, a changed usage block, a retired model id. `tests/eval/test_live_providers.py` closes that gap with one real call per advertised provider, through the adapter and the factory rather than through the vendor SDK, so our own integration is what is under test.

Each provider is gated on its own credential and skips, never fails, when it is absent. A full run with everything present is **eight requests** of a few dozen tokens each — five generations and three embeddings:

| Check | Gate | Notes |
|---|---|---|
| `openai llm`, `openai embeddings` | `OPENAI_API_KEY` | |
| `anthropic llm` | `ANTHROPIC_API_KEY` | |
| `cohere llm`, `cohere embeddings` | `COHERE_API_KEY` | |
| `ollama llm`, `ollama embeddings` | `OLLAMA_HOST` (an absolute `http(s)://` URL) | Local server; costs nothing. Gated on an explicit opt-in because gating it on nothing would mean a connection *failure*, not a skip, on every machine without Ollama. |
| `openai_compatible llm` | `FASTERRAG_LIVE_OPENAI_COMPATIBLE_URL` + `FASTERRAG_LIVE_OPENAI_COMPATIBLE_KEY` + `FASTERRAG_LIVE_MODEL_OPENAI_COMPATIBLE_LLM` | A gateway names its own models, so there is no default model id to guess; without all three the check skips and says which one is missing. |

`FASTERRAG_LIVE_MODEL_<PROVIDER>_<KIND>` overrides any provider's probe model, so a retired id is replaced by an environment variable rather than a code change. A missing provider SDK skips the same way, naming the extra to install.

**What a green `pytest -m eval` proves: only the providers whose own checks ran.** Read the skip lines — `pytest -m eval -rs` — not the exit code.

**What it does not prove: anything at all about a provider that skipped.** On a machine with no credentials every probe skips and the file is green while having contacted nobody. A skipped probe counted as a passed one is trap T12 ([rebuild-playbook.md](rebuild-playbook.md) §Traps). Three things keep the two readings apart:

- Every provider has its own test, so pytest's own `SKIPPED` line naming the missing variable is the record, and no aggregate can launder it.
- A summary test at the end of the module is green only when *every* provider was reached; otherwise it skips carrying a per-provider table marking each one `VERIFIED`, `NOT CHECKED` (with the reason) or `NOT RUN`. The honest outcome of a keyless run is a skip, not a pass.
- The gating helpers are proved by `tests/eval/test_live_provider_gating.py`, which carries **no** `eval` marker and so runs in the ordinary `pytest -m "not integration and not eval"` CI job — the job that cannot skip. Without it, the only evidence that a skip is a real gate rather than a silent bug would come from the job that skips.

**Status, as of 2026-10-29.** The gating logic is verified on every commit. The live call bodies have **never been executed against a real provider**, in CI or by hand, because no credentials have been available to any run — so no row in the table above is backed by a recorded response today (TASK-0205). The HuggingFace embedder is the one advertised provider with no entry here and needs none: it is local, and `tests/eval/test_real_models.py` exercises it against real downloaded weights.

[testing-strategy.md](testing-strategy.md) §1.11 holds the fuller treatment of this layer, including the known gap that the Cohere and Ollama embedding adapters surface no token usage.

## 4. Rerankers (`retrieval.reranker_model`)

Cross-encoder models loaded locally in the query path (e.g. `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-6-v2`). Selected by model id; toggle with `retrieval.rerank`. Stage latency is unmeasured (TASK-0084) — the "~100–300 ms" this row previously quoted had no source and has been withdrawn.

## 5. Observability tools (all default `false`)

| Tool | Toggle | Status | What flipping it does | Result |
|---|---|---|---|---|
| fasterRag dashboard | `observability.dashboard: true` | **Shipped** (TASK-0045 ✅, tenant-scoped by TASK-0184 ✅) — a separate ASGI app on its own port, declaring no write route at all | Starts the read-only inspection UI | `http://<host>:8080` |
| OpenTelemetry | `observability.otel: true` | Shipped and verified once by hand against a running collector (a manual run, not a committed regression test) — `otel/opentelemetry-collector-contrib` 0.140.1 accepted both signals with no warning, refusal, or rejection in its log (TASK-0231): five spans carrying the fasterRag trace id with the root/child structure intact, both signals scoped `fasterrag`, the counter as a monotonic `Sum`, the gauge as a `Gauge`, and the histogram's bucket counts summing to its observation count. Needs the `otel` extra | Exports spans and metrics via OTLP to `otel_endpoint`, treated as a base URL from which `/v1/traces` and `/v1/metrics` are derived | your collector |
| **Langfuse** | `observability.langfuse: true` | **Shipped**: provisioning verified end-to-end once by hand (six containers up, `/api/public/health` OK, bootstrapped keys authenticated — TASK-0043 ✅; a manual run, not a committed regression test), the toggle triggers it at startup (TASK-0150 ✅), it is doctor-gated (TASK-0149 ✅), and trace export is implemented (TASK-0151 ✅) and **verified against a real self-hosted Langfuse 3.225.1** (TASK-0236 ✅): the batch was accepted, the trace came back under fasterRag's own id with all four stage observations and the generation's token usage. That run also found a defect it was raised to find — a `207` carrying rejected events was being reported as a successful export, and now is not | Auto-installs the self-hosted Langfuse v3 compose stack (web, worker, Postgres, ClickHouse ≥ 24.3, Redis, MinIO), generates + persists secrets, headless-bootstraps org/project/keys/user via `LANGFUSE_INIT_*`, health-checks | **`http://<host>:3000`** |
| **Grafana** | `observability.grafana: true` | **Shipped**: provisioning verified end-to-end once by hand — fasterRag → Prometheus → Grafana's provisioned datasource returned real series (TASK-0044 ✅, panels fixed under TASK-0154 ✅); a manual run, not a committed regression test. Toggle triggers it at startup (TASK-0146 ✅) and it is doctor-gated (TASK-0147 ✅) | Provisioning-as-code: datasource + dashboard YAML/JSON under `/etc/grafana/provisioning/`, `editable: false`, `allowUiUpdates: false`, 30 s file reload | Grafana with fasterRag dashboards, read-only for provisioned resources |

The target contract for every row is: config toggle → doctor-gated auto-provisioning → running URL, with **zero application-code changes at toggle time** ([observability.md](observability.md)). **As built, that contract holds for Langfuse and Grafana**: both are reachable from the CLI command *and* from the toggle at startup, and both are doctor-gated before anything is mutated. What remains open is verification against a live Langfuse (TASK-0178), not construction.

## 6. Adding a provider that isn't listed

Implement the adapter base class, register it via the `fasterrag.vectordb` / `fasterrag.embeddings` / `fasterrag.llm` entry point, pass the contract suite, and it becomes selectable in `config.yaml` like any built-in ([python-api.md](python-api.md) §Extending). This is the supported path for the long tail of vector DBs, embedders, and LLM gateways. **Status: shipped** (TASK-0173 ✅). All three factories resolve entry points, report a plugin's origin in `available_providers()`, reject a registered object that is not an adapter, and always prefer a built-in name — so an installed package cannot silently take over a configured provider. Verified with 15 tests registering a fake plugin against each group.
