# integrations.md — Supported Providers & How Config Enables Them

Everything is selected purely through `config.yaml`; secrets by env-var name only; **every integration toggle defaults to `false`**. Each row below carries a **Status**: *shipped* means implemented and tested in this repo today; *specified* means the config surface and behavior are designed but the adapter has not been built (its task is in [todo.md](todo.md)) — selecting a specified provider fails fast with a `ConfigError`, never silently. Conformance for any adapter, in-repo or third-party, means passing the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). The entry-point extension contract ([python-api.md](python-api.md)) is **shipped** (TASK-0173 ✅): all three factories resolve registered plugins, and a built-in name always wins so an installed package cannot silently take over a configured provider.

## 1. Vector databases (`vector_db.provider`)

> **Security note (2026-11-04).** `chromadb` 1.5.9 — the latest release — carries four unfixed advisories (PYSEC-2026-311, -3813, -3814, -3815). All four are in Chroma's HTTP server, which fasterRag never starts; the adapter runs embedded against a local directory. See [security.md](security.md) §6 for the reasoning and its limits. If you run a Chroma server of your own, those advisories apply to it regardless of fasterRag.

| Provider | Value | Status | Modes | Secret (env var) | Notes |
|---|---|---|---|---|---|
| **Qdrant** (reference) | `qdrant` | **Shipped** — contract suite passes in all three modes | docker (system-managed) · external local · remote host:port | `QDRANT_API_KEY` | Reference implementation. REST 6333 + gRPC 6334 (both must be reachable; `prefer_grpc` default `false`). Named Docker volume mandatory on Windows/WSL. |
| **Milvus** | `milvus` | **Shipped, and run against a real server since 2026-11-11** — the shared contract suite passes **46 of 46** against `milvusdb/milvus` from `deploy/compose.test-backends.yml` (TASK-0396 ✅). Its first run failed **39 of the 46** on one real defect the fakes could not show: `schema.description` is read-only in pymilvus 3.0.2, so the adapter could not create a collection at all. It runs on a developer machine and **in no CI job**, so nothing re-proves it on a change (TASK-0412). 105 unit cases still run against a fake client, which proves the adapter issues the calls it means to and nothing about what Milvus does with them. | external · remote | token in the env var named by `vector_db.api_key_env` (set it to `MILVUS_API_KEY`; the default is the Qdrant variable) | gRPC 19530 — **set `vector_db.port: 19530`**, because the default 6333 is Qdrant's. Every field is declared: the payload lives in one JSON column rather than Milvus's dynamic `$meta`, so a payload key can be anything, at the cost that a filtered search scans the JSON column unless a deployment adds its own JSON path index. Aliases are native Milvus aliases under derived names (`fragalias_…`), because a Milvus identifier forbids the `-` a fasterRag collection name allows. The BM25 leg stores fasterRag's term frequencies as a `SPARSE_FLOAT_VECTOR`; Milvus has no IDF modifier for a caller-supplied sparse vector, so IDF is built from one server-side `count(*)` per query term and carried into the *query* weights, which makes Milvus's inner product compute the BM25 score and do the top-k itself. That is `len(terms) + 1` counts per keyword query — unlike Chroma, no posting is read into the process — but it is more round trips than Qdrant's single server-side sparse search. Snapshots are copies into a second collection, not physical backups: pymilvus 3.0's `create_snapshot`/`restore_snapshot` are documented only as a Zilliz Cloud private preview and are not confirmed to exist on a self-hosted server. Reads ask for `Strong` consistency, because Milvus defaults to `Bounded` and a point just written is otherwise not necessarily searchable. `shard_number` and `replication_factor` are ignored rather than reinterpreted, `update()` is a read-modify-write because Milvus has no partial update, and a chunk with more than 4096 distinct terms is refused rather than truncated. |
| **Weaviate** | `weaviate` | **Shipped, and run against a real server since 2026-11-11** — the shared contract suite passes **54 of 54** against a real Weaviate from `deploy/compose.test-backends.yml` (TASK-0396 ✅), and found three defects the fakes could not: `iterate_points` never terminated, because Weaviate's iterator rewinds to page one when `islice` re-enters it; `restore_snapshot` could never clear a collection, because its match-everything filter tokenised to nothing; and an HTTP 401 was classified retryable, so a rejected credential would have been retried forever. It runs on a developer machine and **in no CI job** (TASK-0412). 124 unit cases still run against a fake client that evaluates the adapter's real filter objects, which proves the adapter issues the calls it means to and nothing about what Weaviate does with them. | external · remote | key in the env var named by `vector_db.api_key_env` (set it to `WEAVIATE_API_KEY`; the default is the Qdrant variable) | HTTP 8080 **and** gRPC 50051 — set `vector_db.port: 8080` and `vector_db.grpc_port: 50051`, because the defaults are Qdrant's and the v4 client refuses to work over HTTP alone. **Weaviate's native BM25 and hybrid search are deliberately unused.** The adapter is handed an encoded `SparseVector`, never the query text `query.bm25()` wants; Weaviate's own tokenizer would hit different terms than every other backend, which is the drift `docs/adr/ADR-0007` exists to prevent; and its inverted index has no read path, so a collection scored by it could not be exported (TASK-0291). Storing our stemmed tokens in a text property and tuning the collection's `bm25_k1`/`bm25_b` was considered and is impossible: `encode_document` stores *saturated* term frequencies and saturation is not invertible, so the raw count a native index needs cannot be recovered. Postings are therefore two array properties (`fr_terms`, `fr_weights`) and IDF is computed in the adapter over the live corpus — **which reads every document carrying any query term into the process**, exactly Chroma's limit and not scalable the way Qdrant's server-side sparse index is; candidate paging is deliberately unbounded, because a truncated candidate set returns a wrong top-k silently. Payloads are split: a scalar under a legal property name is flattened so filters push down, everything else is JSON-encoded into `fr_extra` and decoded back, and a filter on a key that could not be flattened is refused with a typed error rather than silently matching nothing. Flattening relies on the server's default `AUTOSCHEMA_ENABLED`. Collections are `Frag_<slug>_<digest>` because a Weaviate name must start with a capital, point ids become UUIDv5 because an object id is a UUID and a chunk id is not, and the declared vector width is recorded in the collection description because Weaviate does not declare one. Aliases are records in a `Fasterrag_aliases` collection rather than Weaviate 1.32's native aliases, so the adapter does not work on one server version and fail on another. Snapshots are copies into a second collection, not Weaviate backups, which need a backup module enabled in the server's own environment. `update()` is a read-modify-write because the JSON remainder has to be rewritten whole. |
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
| Cohere | `cohere` | `COHERE_API_KEY` | ❌ | Batched, through the v1 `embed` call. |
| Ollama (local server) | `ollama` | — | ✅ | One request per batch on `/api/embed`, so the server must serve that route; the adapter does not fall back to the superseded per-text `/api/embeddings`. Needs `ollama>=0.4`. There is no `embeddings.base_url`: the server is `llm.base_url` when `llm.provider` is also `ollama`, and otherwise the client's own default — the `OLLAMA_HOST` environment variable, else `http://127.0.0.1:11434`. `embeddings.dimensions` is not sent to the server. |

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
| Cohere | `cohere` | `COHERE_API_KEY` | ✅ | The v2 chat API, through `cohere.AsyncClientV2`. Needs `cohere>=5.8`. Token counts are the billed units, falling back to the consumed tokens when the reply states no bill. |
| Ollama (local) | `ollama` | — | ✅ | Fully local generation; `llm.base_url` names the server, else `OLLAMA_HOST`, else `http://127.0.0.1:11434`. Needs `ollama>=0.4`. |
| Any OpenAI-compatible endpoint | `openai_compatible` | per endpoint | ✅ | `llm.base_url` required — covers vLLM, LM Studio, llama.cpp servers, TGI, and most hosted gateways. |

**Cost coverage.** OpenAI, Anthropic, and part of the Cohere lineup have dated list prices recorded, so their traffic reaches `fasterrag_cost_usd_total`; Ollama is priced at zero because it is local. Embedding spend reaches the same counter as of TASK-0312 (2026-11-08), priced from the **provider's own reported token count** where the adapter surfaces one — OpenAI's `usage.total_tokens`, Cohere's `meta.billed_units.input_tokens` and, as of TASK-0441, Ollama's `prompt_eval_count` (counted, and still priced at zero) — and from our local count only where the provider states nothing. Where both exist and disagree, the difference is reported as `EmbedReport.token_drift` and logged at the end of the ingest rather than silently resolved in either direction. `openai_compatible` never is — a gateway sets its own prices, and charging its traffic at the upstream vendor's rate would be a fabricated bill. Unpriced traffic is counted by `fasterrag_unpriced_tokens_total` rather than silently dropped. The exact per-model coverage and its check dates are in [config-reference.md](config-reference.md#which-models-carry-a-price).

## 3a. Live provider verification — what `pytest -m eval` does and does not prove

Every other provider test in this repository substitutes a fake client or, for Cohere and Ollama, replays a recorded reply through the real client library (§3b). That proves the adapter issues the request it meant to; it proves nothing about whether the vendor still accepts it, which is the drift that actually breaks a release — a renamed response field, a changed usage block, a retired model id. `tests/eval/test_live_providers.py` closes that gap with one real call per advertised provider, through the adapter and the factory rather than through the vendor SDK, so our own integration is what is under test.

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

[testing-strategy.md](testing-strategy.md) §1.11 holds the fuller treatment of this layer, including why `reports_token_usage` is `True` for the Cohere embedder and, since the move to `/api/embed`, for the Ollama one.

## 3b. Recorded-response tests — the Cohere and Ollama adapters against the real client libraries

A fake client proves the adapter and the fake agree. It says nothing about the library, and three adapters were wrong against theirs with every test green (TASK-0441, found 2026-11-12 by replaying a recorded reply through `cohere` 7.3.0 and `ollama` 0.6.3 — the versions `uv.lock` resolves — with no network and no key):

| Adapter | What the real client did | What a user saw |
|---|---|---|
| Cohere LLM | The adapter built `cohere.AsyncClient` (the v1 client) and called `chat(messages=…)`. v1 `chat` takes `message`, so the call raised `TypeError` before any request was sent. | Every generation failed as `GENERATION_FAILED`, "cohere was unreachable during complete: TypeError", marked retryable — so it was retried and then failed over as if Cohere were down. |
| Ollama LLM | `chat` returns a typed `ChatResponse`, not a `dict`; the adapter read text only from a `dict`. | An empty answer beside correct token counts, and a stream that yielded nothing. No error anywhere. |
| Ollama embeddings | `embeddings` returns a typed `EmbeddingsResponse`, not a `dict`. | Every embedding failed as `EMBED_PROVIDER_ERROR`, "ollama returned an empty embedding". |
| Cohere embeddings | `AsyncClient.embed` accepted the call and returned vectors on `embeddings.float_` with `meta.billed_units`. | Nothing wrong — it was not a bug. |

The tests that would have caught each are `tests/unit/adapters/llm/test_real_clients.py` and `tests/unit/adapters/embeddings/test_real_clients.py`. The adapter builds its own genuine client; `tests/unit/adapters/recordings.py` answers it at the HTTP transport from a body in the vendor's documented wire format, so the library's request building and response parsing both run and no socket is opened. The same module holds the bodies the hand-written fakes are built from, and a fidelity test per reply fails if a fake carries anything the real client's object does not.

`cohere` and `ollama` are in the `dev` extra so these run in CI rather than skip, and one test per library in each module is marked `must_run` ([testing-strategy.md](testing-strategy.md) §3.2). Without the library a test skips, naming the extra to install.

**What this does not prove.** The bodies were written from each vendor's API reference, not captured from a live server, so a server that has drifted from its reference is still invisible here; only the billed probes in §3a can see that, and they have never run. The lower bounds in `pyproject.toml` come from replaying the same recordings through older releases: `cohere` 5.6.2 fails and 5.8.0 passes; `ollama` 0.3.3 fails and 0.4.0 passes.

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

Implement the adapter base class, register it via the `fasterrag.vectordb` / `fasterrag.embeddings` / `fasterrag.llm` entry point, pass the contract suite, and it becomes selectable in `config.yaml` like any built-in ([python-api.md](python-api.md) §Extending). This is the supported path for the long tail of vector DBs, embedders, and LLM gateways. **Status: shipped** (TASK-0173 ✅). All three factories resolve entry points, report a plugin's origin in `available_providers()`, reject a registered object that is not an adapter, and always prefer a built-in name — so an installed package cannot silently take over a configured provider. Verified with 15 tests registering a fake plugin against each group. **Status (2026-11-11):** the entry point is discovered by the factory, but the provider name is refused by the config schema's closed list before the factory runs, so a plugin is not yet selectable from `config.yaml` (TASK-0439).
