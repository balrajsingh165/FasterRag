# config-reference.md — Full `config.yaml` Schema

`config.yaml` drives ALL behavior. Any key here can be overridden per invocation with `--set dotted.key=value`, or through the `FASTERRAG_SET` environment variable for containers ([cli-reference.md](cli-reference.md)); both are validated exactly as a file value is. `.env` holds ONLY credentials/secrets — config references secrets exclusively **by environment-variable name** (e.g. `api_key_env: OPENAI_API_KEY`); secret values are NEVER inlined in YAML. The loader is **pydantic-settings** with a YAML source for config and env/`.env` for secrets; it validates the entire schema at startup and **fails fast** with a clear error naming the offending key. **Every integration toggle defaults to `false`.**

Conventions used below:

- **Type** uses Python/Pydantic notation. `str|null` means optional/nullable.
- **Validation** is enforced at startup by the Pydantic schema; violations abort startup with `ConfigError`.
- Durations are integer seconds unless the key name says otherwise (`*_ms` = milliseconds).

## Complete example (all defaults shown; secrets referenced by env-var name only)

```yaml
app:
  host: 127.0.0.1
  port: 8000
  log_level: info

vector_db:
  provider: qdrant
  mode: docker
  host: localhost
  port: 6333
  grpc_port: 6334
  prefer_grpc: false
  https: false
  api_key_env: QDRANT_API_KEY
  docker:
    image: qdrant/qdrant:v1.18.1
    volume: fasterrag_qdrant_storage
  pgvector:
    dsn_env: null
    db_schema: fasterrag
    pool_max_size: 8
    maintenance_timeout_ms: 600000
  chroma:
    path: .fasterrag/chroma
    batch_size: 1000
  collection:
    default_name: default
    distance: cosine
    shard_number: 1
    replication_factor: 1

embeddings:
  provider: huggingface
  model: BAAI/bge-small-en-v1.5
  api_key_env: null
  batch_size: 64
  dimensions: null
  cache:
    enabled: true
    backend: disk
    max_entries: 10000
    redis_url: redis://localhost:6379/0
  tiering:
    enabled: false
    rules: []

llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  base_url: null
  temperature: 0.1
  max_tokens: 1024
  streaming: true

parsing:
  minimum_chars_per_page: 40
  ocr_resolution: 200
  heading_size_ratio: 1.15
  max_heading_chars: 120
  rows_per_block: 20

chunking:
  strategy: recursive
  chunk_size: 768
  overlap: 64
  token_counter: auto
  chars_per_token: 4
  semantic_percentile: 0.95
  contextual_enrichment: false
  context_tokens: 75

retrieval:
  top_k: 10
  hybrid: true
  bm25_weight: 1.0
  dense_weight: 1.0
  rrf_k: 60
  bm25_k1: 1.2
  bm25_b: 0.75
  rerank: true
  reranker_model: BAAI/bge-reranker-v2-m3
  rerank_top_n: 100

generation:
  grounded_or_refuse: false
  faithfulness_threshold: 0.7
  citations: true
  context_budget_tokens: 4000

cache:
  semantic: false
  similarity_threshold: 0.95
  ttl: 3600
  backend: memory
  max_entries: 10000
  redis_url: redis://localhost:6379/0

workers:
  cpu_pool_size: 0
  embedding_pool_size: 1
  queue_depth: 1000

ingestion:
  dedup: true
  journal:
    enabled: true
    checkpoint_every: 100
  dlq:
    enabled: true
    max_retries: 3
  max_document_mb: 100
  # Seconds one document may spend being parsed and chunked before it is
  # dead-lettered as PARSE_TIMEOUT and its worker is replaced. 0 = no limit.
  parse_timeout_seconds: 600
  # Directories a `path` source may read from. Empty is unconfined, which is
  # refused outright once security.auth is on -- see docs/security.md.
  allowed_source_roots: []

index:
  lockfile: true
  reindex:
    strategy: blue_green
    eval_gate: true
    rollback_retention_hours: 72

reliability:
  timeouts:
    vector_db_ms: 5000
    embeddings_ms: 30000
    llm_ms: 120000
    redis_ms: 1000
  retries:
    max_attempts: 3
    backoff_base_ms: 250
    backoff_max_ms: 10000
    jitter: true
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_ms: 30000
  idempotency:
    enabled: true
    ttl_seconds: 86400
    in_flight_lease_seconds: 900
  shutdown_grace_seconds: 20
  stale_job_seconds: 120

traces:
  store: true
  retention_days: 30
  replay: true

cost:
  estimator: true
  per_query_token_budget: 0
  per_tenant_token_budget: 0

autopilot:
  enabled: false
  golden_set_size: 100

eval:
  regression_gate: false
  recall_tolerance: 0.02
  ndcg_tolerance: 0.02

observability:
  dashboard: false
  dashboard_port: 8080
  otel: false
  otel_endpoint: null
  langfuse: false
  grafana: false

security:
  auth: false
  api_key_env: FASTERRAG_API_KEY
  multi_tenancy: false
  tenant_header: X-Tenant-ID
  # Push the caller's tenant into both retrieval legs as a mandatory filter. Requires auth
  # and multi_tenancy. Off by default: chunks indexed before multi_tenancy was enabled
  # carry no tenant field, and an absent field never matches -- see docs/security.md.
  tenant_scoped_retrieval: false
  rate_limit_per_minute: 600
  rate_limit_backend: memory
  rate_limit_redis_url: redis://localhost:6379/0
  max_request_mb: 25
  # Directories admin export may write into and admin import may read from. Empty is
  # unconfined, which is refused outright once auth is on -- see docs/security.md.
  allowed_archive_roots: []
```

---

## `app`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `app.host` | str | `127.0.0.1` | valid IPv4/IPv6 address or hostname | Bind address for the API server and the dashboard. Loopback by default, because `security.auth` defaults to `false`: an unauthenticated API must not be reachable from other machines unless an operator says so. The container image passes `--host 0.0.0.0` explicitly and publishes on `127.0.0.1`; set this to a routable address only together with `security.auth: true`. `fasterrag serve` logs a warning when it binds a non-loopback address with auth off. |
| `app.port` | int | `8000` | 1–65535 | API server port. |
| `app.log_level` | str | `info` | one of `debug`, `info`, `warning`, `error` | Structured-log verbosity. |

**`app.workers` was removed on 2026-11-12.** It was validated and read by nothing: `fasterrag serve` runs one server process whatever it said. A `config.yaml` that still sets it fails at load with a message naming the key and the reason; delete the line. The pipeline pools are sized by [`workers.*`](#workers), a different section that is unchanged. Several API processes would also be unsafe today, because rate limits, token budgets, circuit breakers and the file journal are per process (TASK-0437, TASK-0426).

## `vector_db`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `vector_db.provider` | str | `qdrant` | a built-in (`qdrant`, `milvus`, `weaviate`, `pgvector`, `chroma`) or a name an installed package registers under the `fasterrag.vectordb` entry-point group | Selects the concrete `VectorDBAdapter` via the factory. Changing this swaps backends with zero code changes. Any other name stops the load, and the error lists the built-in names and the plugins that are installed ([python-api.md](python-api.md), "Extending the framework"). A built-in name always wins over a plugin registered under the same name. `pinecone` is specified and its adapter is not built (TASK-0049): naming it stops the load with that task id, where it used to validate and fail later inside the factory. An installed package that registers `pinecone` itself is selectable. |
| `vector_db.mode` | str | `docker` | one of `docker`, `external` | `docker` = system-managed container (fasterRag launches/manages it). `external` = user-run instance, local (no-Docker) **or** remote via `host`/`port` (covers the remote-IP mode). |
| `vector_db.host` | str | `localhost` | non-empty hostname or IP | Backend host. For remote Qdrant set the remote machine's IP/hostname. |
| `vector_db.port` | int | `6333` | 1–65535 | REST port. Qdrant client default is 6333. **The Milvus adapter reads this field too and Milvus serves gRPC on 19530**, so a Milvus deployment must set `port: 19530` — the default is Qdrant's and nothing will answer on it. Milvus has no second port here: `grpc_port` and `prefer_grpc` are Qdrant's and are ignored. The **Weaviate** adapter reads it as the HTTP port and must be set to `8080` for a default Weaviate; unlike Milvus it also reads `grpc_port`, because the v4 client requires both channels. |
| `vector_db.grpc_port` | int | `6334` | 1–65535; must differ from `port` | gRPC port. Qdrant client default is 6334. **Both 6333 and 6334 must be exposed/reachable** — Qdrant GitHub Discussion #2195 records connection failures when only 6333 is exposed and the client attempts gRPC. **The Weaviate adapter reads this field too and Weaviate serves gRPC on 50051**, so a Weaviate deployment must set `grpc_port: 50051`; a server exposing only 8080 cannot be used at all, because the v4 client issues every query over gRPC. |
| `vector_db.prefer_grpc` | bool | `false` | — | Matches the Qdrant Python client default (`prefer_grpc=False`). Set `true` for gRPC-first traffic (requires `grpc_port` reachable). |
| `vector_db.https` | bool | `false` | — | Whether to reach the backend over TLS. **Must be set explicitly rather than inferred**: the Qdrant Python client silently switches to HTTPS whenever an `api_key` is supplied, which fails against the plain-HTTP listener a container serves by default. `false` matches the documented private-network deployment (an API key over a private link); set `true` for a TLS-terminated remote instance ([deployment.md](deployment.md) §2). |
| `vector_db.api_key_env` | str\|null | `QDRANT_API_KEY` | valid env-var name or null | Name of the env var holding the backend API key (Qdrant: consumed as `QDRANT__SERVICE__API_KEY` on the server side). Never the key itself. Null = unauthenticated (local dev only). The Milvus adapter reads the same field and passes the value as pymilvus's `token`, which for a self-hosted server is `user:password` — point it at `MILVUS_API_KEY` rather than leaving it on the Qdrant default. The Weaviate adapter reads the same field and passes the value as the client's API-key credential; point it at `WEAVIATE_API_KEY`. Null means the adapter connects anonymously, which only works against a server with anonymous access enabled. **For a plugin provider this key must be written** — a variable name, or `null` — because the default names Qdrant's key and a plugin is never handed it implicitly (cross-field rule 3). |
| `vector_db.docker.image` | str | `qdrant/qdrant:v1.18.1` | non-empty image ref; pinned tag required (no `latest`) | Image used in `docker` mode. |
| `vector_db.docker.volume` | str | `fasterrag_qdrant_storage` | valid Docker volume name | Storage volume for persistence. **On Windows/WSL this MUST be a named Docker volume** — bind mounts have known file-system data-loss issues per Qdrant's install docs. The loader rejects bind-mount paths on Windows/WSL. |
| `vector_db.pgvector.dsn_env` | str\|null | `null` | valid env-var name | Name of the environment variable holding the PostgreSQL DSN. **Required when `provider: pgvector`** (cross-field rule 11) and left unset otherwise — it defaults to `null` rather than to a name because a name here is a promise that the variable exists; since TASK-0444 rule 9 requires it only for `provider: pgvector`, but a default would still teach every Qdrant deployment a DSN variable it does not have. A DSN carries the password, so it is named here and stored in `.env`, never in `config.yaml`. `host`, `port`, and `grpc_port` are ignored by this provider: a usable connection string already carries the host, the database, the role, and the SSL mode. |
| `vector_db.pgvector.db_schema` | str | `fasterrag` | `^[a-z_][a-z0-9_]{0,62}$` | PostgreSQL schema the adapter owns. It holds the three catalog tables (`fasterrag_collections`, `fasterrag_aliases`, `fasterrag_snapshots`) plus one table per collection, so pointing two deployments at one database with different schemas keeps them isolated. Created on first use if absent. |
| `vector_db.pgvector.pool_max_size` | int | `8` | 1–256 | Maximum connections the adapter's pool opens. It held a single connection before TASK-0239, which serialized every operation on it — one blocked statement stalled every other caller, including a health check needing no lock at all. Size it to the concurrency the API and worker pools actually offer; PostgreSQL's own `max_connections` is the ceiling, and this is per process, so N replicas open N times this many. |
| `vector_db.pgvector.maintenance_timeout_ms` | int | `600000` | ≥ 1000 | Statement budget for schema changes, snapshot copies, and restores, applied with `SET LOCAL` so it cannot leak back to the pool. Deliberately separate from `reliability.timeouts.vector_db_ms`, which bounds ordinary queries: a bound tight enough for a query aborts the HNSW build on any collection large enough to need one, so the two budgets exist to be told apart. It is a longer bound, not the absence of one. |
| `vector_db.chroma.path` | str | `.fasterrag/chroma` | non-empty path | Directory the embedded Chroma store persists into, created on first use. Chroma runs in the fasterRag process rather than over a socket, so `mode`, `host`, `port`, `grpc_port`, and `api_key_env` are all ignored by this provider — this path is the entire connection. Two deployments sharing one path share one store; give them separate directories to isolate them. A path that already exists and is not a directory is rejected at first use rather than reported later as a store failure. |
| `vector_db.chroma.batch_size` | int | `1000` | 1–5000 | Records the adapter writes per Chroma call, for upserts, metadata updates, and snapshot copies. It is capped at whatever the client reports as its own maximum batch, so raising it past that has no effect. Lower it if a single batch of large payloads is using more memory than the process has to spare. |
| *(no `vector_db.weaviate.*` section)* | — | — | — | The Weaviate adapter is configured entirely from the shared `vector_db` fields above — `host`, `port` (HTTP), `grpc_port`, `https`, and `api_key_env`. It has no section of its own for the same reason Milvus has none: adding one is a change to the `config/` validation contract, which is a separately authorized boundary. `https` applies to **both** channels, so a deployment terminating TLS on HTTP only is not expressible yet. Multi-tenancy, named vectors, and Weaviate's own backup backends are likewise not configurable here. Tracked under TASK-0049. |
| *(Weaviate: `collection.shard_number` / `replication_factor`)* | — | — | — | **Passed through**, unlike Milvus's, because Weaviate takes both directly: `shard_number` becomes the sharding config's desired count and `replication_factor` becomes the replication factor. Both default to `1`, so the mapping is inert on a single node; a value above `1` against a single-node server is refused by the server, not by the adapter, and has never been exercised here. |
| *(no `vector_db.milvus.*` section)* | — | — | — | The Milvus adapter is configured entirely from the shared `vector_db` fields above — `host`, `port`, `https`, and `api_key_env`. It has no section of its own because adding one is a change to the `config/` validation contract, which is a separately authorized boundary; until it has one, a Milvus deployment sets `port: 19530` by hand and the adapter uses the default Milvus database. Tracked under TASK-0049. |
| *(Milvus: `collection.shard_number` / `replication_factor`)* | — | — | — | Ignored by the Milvus adapter rather than reinterpreted. Milvus has its own sharding and replica model, and guessing a mapping onto it would quietly create collections shaped differently from what the configuration asked for. |
| `vector_db.collection.default_name` | str | `default` | `^[a-zA-Z0-9_-]{1,64}$` | Default collection name. |
| `vector_db.collection.distance` | str | `cosine` | one of `cosine`, `dot`, `euclid` | Distance metric for dense vectors. |
| `vector_db.collection.shard_number` | int | `1` | ≥ 1 | Shards per collection (scaling knob; passed through the adapter). |
| `vector_db.collection.replication_factor` | int | `1` | ≥ 1 | Replicas per shard (availability knob; passed through the adapter). |

## `embeddings`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `embeddings.provider` | str | `huggingface` | a built-in (`openai`, `cohere`, `huggingface`, `ollama`) or a name registered under the `fasterrag.embeddings` entry-point group | Embedding adapter selection. `huggingface` = local sentence-transformers (no key needed). Any other name stops the load, naming what is installed. |
| `embeddings.model` | str | `BAAI/bge-small-en-v1.5` | non-empty | Model name/ID understood by the selected provider. Recorded per chunk for drift detection (see `index.lockfile`). |
| `embeddings.api_key_env` | str\|null | `null` | valid env-var name or null | Env var holding the provider key (e.g. `OPENAI_API_KEY`, `COHERE_API_KEY`). Required (non-null) when provider is `openai` or `cohere`. A plugin provider is asked for nothing here: the default is already `null`, and a plugin that needs a key enforces that itself when it is built. |
| `embeddings.batch_size` | int | `64` | 1–2048 | Texts per embedding request. Batched embedding is far cheaper than one-at-a-time calls. |
| `embeddings.dimensions` | int\|null | `null` | ≥ 8 when set | Output dimensionality; `null` = the model's native size, which is the right setting unless you have a reason. Behaviour differs by provider, because shortening a vector is only lossless for a model trained for it (Matryoshka representation learning). **Hosted providers that support it** (OpenAI `text-embedding-3-*`) receive the value as an API parameter and shorten server-side — verified against the live API: `text-embedding-3-small` returns 1536 natively and exactly 256 when asked (TASK-0207). **Local `huggingface` models** cannot be shortened, so a value that disagrees with what the model emits is refused at load time with an error naming both sizes — rather than left to fail later as a rejected upsert against a collection already created at the wrong width. Must match the collection's vector size. |
| `embeddings.cache.enabled` | bool | `true` | — | Embedding cache keyed by provider, model, model version, `embeddings.dimensions`, input type (document or query) and the content hash of the text, so changing any of them can never be answered with a vector produced under the old value. When `true`, every embedding adapter the API, the CLI, the facade and the ingestion service use is wrapped by the cache, so a text already embedded is not sent to the provider again; when `false`, every call reaches the provider. Vectors from late chunking (`chunking.strategy: late`) are never cached. A cache failure is logged and stepped past — the text is embedded directly. |
| `embeddings.cache.max_entries` | int | `10000` | ≥ 1 | Entry ceiling for the embedding cache. Raise when reingesting a large corpus repeatedly, which is exactly when the cache pays for itself. |
| `embeddings.cache.backend` | str | `disk` | one of `memory`, `disk`, `redis` | Where embedding-cache entries live. `disk` survives a restart on one host and writes one file per entry under `.fasterrag/cache/embedding/` in the working directory; `redis` is the only backend several workers or replicas can share, so a vector one of them paid for is not re-embedded by the next. |
| `embeddings.cache.redis_url` | str | `redis://localhost:6379/0` | starts with `redis://`, `rediss://`, or `unix://` | Connection URL, read only when `embeddings.cache.backend` is `redis`. Needs `pip install fasterrag[redis]`. Entries are namespaced under `fasterrag:embedding`, so one server can back both caches and `clear` never reaches the other's keys or a co-tenant application's. **A URL containing a password is a secret and must not be written here** — supply it through the environment instead (`FASTERRAG_SET=embeddings.cache.redis_url=...`), which is exactly what the `.env`-only policy requires. |
| `embeddings.tiering.enabled` | bool | `false` | — | Tiered embedding: route document classes to different models (cheap models for high-volume/low-priority classes; higher-cost models where retrieval precision matters). |
| `embeddings.tiering.rules` | list[TierRule] | `[]` | each rule: `{match: <metadata filter>, provider: <provider>, model: <model>}`; must be non-empty when tiering enabled | Ordered routing rules, first match wins. A rule's `provider` is validated exactly as `embeddings.provider` is: a built-in or a registered `fasterrag.embeddings` plugin. |

## `llm`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `llm.provider` | str | `openai` | a built-in (`openai`, `anthropic`, `cohere`, `ollama`, `openai_compatible`) or a name registered under the `fasterrag.llm` entry-point group | LLM adapter selection. `openai_compatible` targets any endpoint speaking the OpenAI API shape (set `base_url`). Any other name stops the load, naming what is installed. |
| `llm.model` | str | `gpt-4o-mini` | non-empty | Generation model ID understood by the selected provider (e.g. `claude-opus-5` for provider `anthropic`). |
| `llm.api_key_env` | str\|null | `OPENAI_API_KEY` | valid env-var name or null | Env var holding the provider key. Among the built-in providers, null is allowed only for `ollama`. **For a plugin provider this key must be written** — a variable name, or `null` for a plugin that needs none — because the default names OpenAI's key and a plugin is never handed it implicitly (cross-field rule 3). |
| `llm.base_url` | str\|null | `null` | valid URL when set | Endpoint override (required for `openai_compatible`, optional for `ollama`). When `llm.provider` is `ollama` this is also the server the `ollama` **embedding** provider uses — there is no `embeddings.base_url`. When the LLM is any other provider the Ollama embedder ignores it and uses the `OLLAMA_HOST` environment variable, else `http://127.0.0.1:11434`; before TASK-0441 it read this key whatever the LLM was, and so sent embedding requests to an `openai_compatible` gateway. |
| `llm.temperature` | float | `0.1` | 0.0–2.0 | Sampling temperature. Low default favors grounded answers. |
| `llm.max_tokens` | int | `1024` | 1–32768 | Max generated tokens per answer. |
| `llm.streaming` | bool | `true` | — | Stream tokens via SSE (time-to-first-token). |

## `parsing`

Thresholds the parsers apply before chunking sees anything. They are corpus-dependent: a corpus of scans needs a different OCR trigger and render than a born-digital one, and a PDF laid out with a large body face needs a different heading ratio. **Changing any of these alters the text that gets indexed, but changes neither the chunk ids nor the document content hash** — so re-ingesting an unchanged corpus is deduplicated and the new value has no effect on it. Rebuild with `fasterrag index reembed` to apply them to documents already indexed.

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `parsing.minimum_chars_per_page` | int | `40` | 0–10000 | Characters a PDF page must yield before it is accepted as born-digital text. Below it the page is rasterized and sent through OCR (which needs the `ocr` extra and the tesseract binary; without them the document is flagged `low_text_yield` instead). Raise it for a corpus of scans whose pages carry a text layer that is present but useless — a stamped header, a page number — and would otherwise suppress OCR; `0` disables the OCR path entirely. |
| `parsing.ocr_resolution` | int | `200` | 72–1200 (DPI) | Resolution the page is rendered at before OCR. Higher resolves small type and dense tables better; the rendered bitmap grows with the square of the value, so render time and per-page memory grow with it too. Raise it when OCR output is garbled on fine print, lower it when ingesting a large scanned corpus is memory-bound. |
| `parsing.heading_size_ratio` | float | `1.15` | 1.0–4.0 | How much larger than the document's median type size a PDF line must be to be inferred as a heading. A PDF carries no semantic headings, so this is a heuristic and it only ever affects the `section` label a chunk inherits. Raise it when body emphasis is being read as structure; lower it for documents whose headings are only slightly larger. |
| `parsing.max_heading_chars` | int | `120` | 1–1000 | Length ceiling for an inferred PDF heading. A long line is prose whatever its type size, so this stops a large-type paragraph from becoming the section label for everything under it. |
| `parsing.rows_per_block` | int | `20` | 1–1000 | CSV/TSV rows serialized into one block. Rows are emitted as `column: value` pairs so a retrieved chunk stays self-describing; this decides how many of them travel together. Lower it for wide tables whose rows are long, raise it for narrow tables where one row alone carries too little context to retrieve on. |

## `chunking`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `chunking.strategy` | str | `recursive` | one of `fixed`, `recursive`, `semantic`, `layout`, `late` | Chunking strategy (see [architecture.md](architecture.md) §5). `late` requires a local `embeddings.provider` for its pooling pass and falls back to ordinary embedding on a hosted one. `semantic` finds boundaries with the embedding model, which each parse worker loads for itself: it needs `embeddings.provider: huggingface`, and with any other provider the job is refused with `CONFIG_INVALID` before it is recorded. **Each worker holds its own copy of the model, so memory is `workers.cpu_pool_size` times the model's size**; the model is loaded in every worker before the first document, so start-up takes one model load. |
| `chunking.chunk_size` | int | `768` | 64–2500; warning logged above 1024 | Target chunk size in tokens. **The default and the bounds are configurable engineering choices, not measured optima — tune them against your own corpus with the eval harness ([testing-strategy.md](testing-strategy.md)).** Rationale for the shipped values: a chunk longer than the embedding model's maximum sequence length is silently truncated at embed time, so the default is kept modest and the warning above 1024 exists to prompt you to check `chunk_size` against the sequence limit of the model you configured; the 2500 ceiling is a guardrail against pathological values, not a measured quality threshold. The ~2,500-token *"context cliff"* of [references.md](references.md) R3 is **not** a chunk-size limit — it is a limit on the total retrieved context handed to the generator — and an earlier "~512–1024 practical working range" attributed to that source has been withdrawn, because the study swept chunk sizes of 50–500 only and recommended 150–300 (TASK-0247). R3 is a single January 2026 preprint on one setup (Natural Questions, SPLADE retrieval, Ministral-8B-Instruct-2410), and its authors note the drop-off point is model-dependent: treat it as directional, not settled. |
| `chunking.overlap` | int | `64` | 0 ≤ overlap < `chunk_size` | Token overlap between adjacent chunks: each chunk repeats the trailing tokens of its predecessor, so a sentence cut by a boundary is still present whole in one of the two. **Unresolved tension, stated openly:** the one chunking study this repository cites, [references.md](references.md) R3, ships **0% overlap** as its Table 1 default and reports "no measurable benefit" from overlap while it raises cost and complexity — on a single January 2026 preprint's single setup, with a model-dependent caveat from its own authors. The shipped default of 64 predates that source and rests on no measurement of ours either, so neither value is evidence-backed here. Whether the default should change is a maintainer decision to be settled against your own corpus with the eval harness ([testing-strategy.md](testing-strategy.md)); set `overlap: 0` to follow R3 today. |
| `chunking.token_counter` | str | `auto` | one of `auto`, `estimate`, `model` | How `chunk_size` and `overlap` are counted. `auto` loads the embedding model's own tokenizer when the provider ships a local one and estimates otherwise; `estimate` forces the `chars_per_token` ratio; `model` forces the real tokenizer for any provider, which requires the matching tokenizer in the local Hugging Face cache and falls back to the estimate with a logged warning when it is absent. Loading never reaches the network. |
| `chunking.chars_per_token` | int | `4` | 1–16 | Characters per token assumed by the estimate, and by the first split pass before real counts refine it. Lower it for corpora that tokenize densely (code, CJK) when running `token_counter: estimate`. |
| `chunking.semantic_percentile` | float | `0.95` | 0.50–0.99 | Only used by `strategy: semantic`. Distance percentile above which a gap between adjacent sentences becomes a chunk boundary. Lower splits more eagerly, giving smaller and more topically uniform chunks; higher keeps related passages together. |
| `chunking.contextual_enrichment` | bool | `false` | — | Contextual-retrieval-style enrichment: prepend an LLM-generated document-level context to each chunk before embedding and BM25 indexing (Anthropic, Sept 2024: −49% failed retrievals; −67% with reranking). Costs LLM calls at ingest; uses provider prompt caching of the parent document. |
| `chunking.context_tokens` | int | `75` | 25–150 | Target length of the generated per-chunk context. Recommended ~50–100 tokens. |

## `retrieval`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `retrieval.top_k` | int | `10` | 1–100; ≤ `rerank_top_n` | Final number of chunks passed to context assembly. |
| `retrieval.hybrid` | bool | `true` | — | Run dense ANN + sparse BM25 legs in parallel and fuse. `false` = dense-only. **Index-shaping, not a pure runtime toggle**: a keyword leg cannot be attached to an existing dense-only collection, so turning this on over one does not degrade to dense — the query fails. Run `fasterrag index reembed` to rebuild, and set it before importing an archive (TASK-0119). |
| `retrieval.bm25_weight` | float | `1.0` | 0.0–1.0 after normalization; `bm25_weight + dense_weight > 0` | Weight of the BM25 leg in weighted-fusion mode (RRF itself is rank-based; weights scale each leg's RRF contribution). |
| `retrieval.dense_weight` | float | `1.0` | 0.0–1.0 after normalization; see above | Weight of the dense leg. |
| `retrieval.rrf_k` | float | `60` | > 0 | Reciprocal Rank Fusion constant. Default 60 per Cormack, Clarke & Büttcher (SIGIR 2009). |
| `retrieval.bm25_k1` | float | `1.2` | 0.0–3.0 | BM25 term-frequency saturation. Lower flattens sooner, so a repeated term counts for less. **Changing this changes the stored sparse vectors — existing collections must be reindexed** (`fasterrag index reembed`) before the new value applies to anything already ingested. |
| `retrieval.bm25_b` | float | `0.75` | 0.0–1.0 | BM25 length normalization, `0` off to `1` full. Lower it for a corpus of uniformly sized chunks. **Requires a reindex, as `bm25_k1` does.** |
| `retrieval.rerank` | bool | `true` | — | Cross-encoder reranking of fused candidates. Expected to be the largest single contributor to query latency and a major retrieval-quality lever; **neither is measured** — benchmark it on your own hardware (TASK-0084). |
| `retrieval.reranker_model` | str | `BAAI/bge-reranker-v2-m3` | non-empty when `rerank: true` | Cross-encoder model ID. |
| `retrieval.rerank_top_n` | int | `100` | 10–1000; ≥ `top_k` | Candidates retrieved per leg and fed to the reranker (retrieve top 100–1000 → rerank → truncate to `top_k`). |

## `generation`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `generation.grounded_or_refuse` | bool | `false` | — | D5: when `true`, answers below `faithfulness_threshold` return a structured `insufficient_evidence` response instead of guessing. |
| `generation.faithfulness_threshold` | float | `0.7` | 0.0–1.0 | Minimum faithfulness score for an answer to be returned when `grounded_or_refuse` is on. |
| `generation.citations` | bool | `true` | — | Attach span-level citation objects to every answer. Cannot be `false` while `grounded_or_refuse` is `true`. |
| `generation.context_budget_tokens` | int | `4000` | ≥ 1 | Token budget for the **whole** assembled retrieved context — all packed chunks together, not any one chunk. See the note below. |


### The assembled retrieved-context budget

Retrieved chunks are packed in relevance order until `generation.context_budget_tokens` is
spent. Packing does not stop at the first chunk that does not fit: that chunk is counted as
dropped and the next, possibly smaller, one is still tried. Everything dropped for budget is
reported as `evidence_dropped` on the answer, and as `dropped_budget` on the
`context-assembly` trace span, so a budget that is doing the cutting is visible rather than
inferred.

**When the budget is smaller than a single chunk, nothing is packed.** Every chunk is charged
at least one token, so a budget below the first chunk's cost admits no chunk at all. The query
is still answered rather than failing, but the answer is generated with no retrieved context
and carries no citations, with `evidence_dropped` equal to the number of chunks retrieved. A
non-zero `evidence_dropped` alongside an empty `citations` list is the signature of a budget
set below chunk size.

**It interacts with `retrieval.top_k`.** `top_k` decides how many chunks compete for the
budget; the budget decides how many of them reach the model. `top_k` chunks of
`chunking.chunk_size` tokens each need roughly `top_k × chunk_size` tokens to all fit — at the
defaults, 10 × 768 is well past 4,000, so a full result set is routinely trimmed by the budget
and the lowest-ranked chunks never reach the prompt. That is the designed behaviour, not a
fault: dropping the least relevant material beats letting a provider truncate the prompt from
the middle. Raising `top_k` without raising the budget buys nothing; what it changes is the
reranker's shortlist, not what the model sees.

**On the default.** 4,000 is the value the service has shipped with since the generation slice
landed, and no measurement of ours stands behind it or behind any other value. The introducing
commit records only that it is deliberately conservative against silent provider-side
truncation; no rationale for the number itself was recorded. [references.md](references.md) R3
reports answer quality falling off beyond a ~2.5k-token retrieved context, but on a single
setup with a single 8B generator, and its authors note the drop-off point is model-dependent —
one preliminary preprint is not grounds to move a shipped default, which would swap one
unmeasured number for another. What would move it: a run of the eval harness across budgets on
a committed baseline, recorded in [benchmarks.md](benchmarks.md) (TASK-0285). Until then, treat
4,000 as a starting point to tune per deployment, not as a measured optimum.

## `cache`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `cache.semantic` | bool | `false` | — | Semantic response cache keyed by query-embedding similarity. |
| `cache.similarity_threshold` | float | `0.95` | 0.90–0.99 | Cosine similarity above which a cached response is served. Typical useful range ~0.92–0.97. |
| `cache.ttl` | int | `3600` | ≥ 1 (seconds) | Entry lifetime. Corpus-change events invalidate affected entries immediately regardless of TTL. |
| `cache.max_entries` | int | `10000` | ≥ 1 | Entry ceiling for the semantic cache; the oldest are evicted past it. Raise for a high-traffic deployment with repetitive queries, lower to cap memory or disk. |
| `cache.backend` | str | `memory` | one of `memory`, `disk`, `redis` | Semantic-cache storage backend. Use `disk` from the CLI: a `memory` cache dies with each short-lived process, so every invocation would pay for a query embedding and then discard the answer. Use `redis` for more than one API replica — `memory` gives each replica its own cache, so a hit depends on which one the load balancer picked. |
| `cache.redis_url` | str | `redis://localhost:6379/0` | starts with `redis://`, `rediss://`, or `unix://` | Connection URL, read only when `cache.backend` is `redis`. Needs `pip install fasterrag[redis]`. Entries are namespaced under `fasterrag:semantic`, separately from the embedding cache. **A URL containing a password is a secret and must not be written here** — supply it through the environment instead (`FASTERRAG_SET=cache.redis_url=...`). |

## `workers`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `workers.cpu_pool_size` | int | `0` | 0 (= auto: CPU count) or ≥ 1 | Parse/chunk worker processes (CPU-bound pool). |
| `workers.embedding_pool_size` | int | `1` | ≥ 1 | Stateful embedding workers (each loads the model once and reuses it across all batches). |
| `workers.queue_depth` | int | `1000` | ≥ 10 | Bounded chunk-queue capacity between the pools; provides backpressure. Overflow at the API returns 429 + `Retry-After`. |

Ingestion and query paths use **separate pools and queues** (bulkheads) — an ingestion storm can never starve live queries. This split is structural, not configurable.

## `ingestion`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `ingestion.dedup` | bool | `true` | — | Content-hash deduplication; re-running the same ingest is a no-op (exactly-once index effects). |
| `ingestion.journal.enabled` | bool | `true` | — | D3: checkpointed ingestion journal; crash mid-ingest resumes exactly where it stopped. |
| `ingestion.journal.checkpoint_every` | int | `100` | ≥ 1 | Journal checkpoint interval in documents. |
| `ingestion.dlq.enabled` | bool | `true` | — | Failed documents go to a dead-letter queue with machine-readable reason codes and per-document status. |
| `ingestion.dlq.max_retries` | int | `3` | 0–10 | Retries before a document is dead-lettered. |
| `ingestion.max_document_mb` | int | `100` | 1–1024 | Input size limit per document (security + stability). |
| `ingestion.parse_timeout_seconds` | float | `600` | ≥ 0 (0 = no limit) | Time limit per document for loading, parsing and chunking, counted from the moment the document is handed to the worker pool. A document still being parsed when it expires is dead-lettered with reason `PARSE_TIMEOUT` and the job continues. A running process-pool task cannot be cancelled, so the pool is recycled: its workers are killed, a new pool is started, and the documents that were in flight beside the overdue one are parsed again from the start with a fresh limit (one that was itself within 50 ms of its own limit is dead-lettered with it). Set it above the slowest legitimate document — OCR of a long scan is the usual candidate — because a document that needs longer can never be ingested while the limit stands. `0` restores the unbounded wait, in which one hung parser stalls its job for ever. |
| `ingestion.allowed_source_roots` | list[str] | `[]` | paths | Directories a `path` source may be read from. **Authorization boundary**: a `path` source names a file on the server, so an unconfined one lets any caller with the `ingest` scope read `.env` or an SSH key back out through a query. Paths are resolved before comparison, so `..` and symlinks cannot escape a root. A path written inside a root that a symbolic link makes resolve outside it is refused by the parse worker, as one dead-lettered document with `SOURCE_LINK_ESCAPES`, rather than failing the whole job (TASK-0504); a path written outside every root still refuses the job. Empty means unconfined, which is refused outright when `security.auth` is on — so forgetting it breaks a served deployment loudly instead of exposing it quietly. |

## `index`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `index.lockfile` | bool | `true` | — | D1: every index build writes `index.lock` (config hash, embedding model name+version, chunker strategy+version, per-document content hashes). Drift is detected and reported. |
| `index.reindex.strategy` | str | `blue_green` | one of `blue_green`, `in_place` | D2: `blue_green` builds the new collection in parallel, validates against the eval set, then atomically switches via collection alias. `in_place` is allowed only for dev. |
| `index.reindex.eval_gate` | bool | `true` | — | Require eval-set validation to pass before the alias swap. |
| `index.reindex.rollback_retention_hours` | int | `72` | ≥ 0 | How long the old collection is retained after a swap so `index rollback` can re-point the alias back to it instead of rebuilding. |

## `reliability`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `reliability.timeouts.vector_db_ms` | int | `5000` | ≥ 100 | Timeout for every vector-DB call. Every external interaction MUST have an explicit timeout. |
| `reliability.timeouts.embeddings_ms` | int | `30000` | ≥ 100 | Timeout per embedding batch call. |
| `reliability.timeouts.llm_ms` | int | `120000` | ≥ 1000 | Timeout per LLM call (streaming: per-connection). |
| `reliability.timeouts.redis_ms` | int | `1000` | ≥ 10 | Socket and connect timeout for every Redis client — the `redis` cache backends and the `redis` rate limiter — so one Redis call waits at most this long for a connection and at most this long for a reply. A Redis that accepts connections and never answers then degrades exactly as a stopped one does: the cache misses, and the limiter serves the request unthrottled with a `WARNING` (TASK-0453). |
| `reliability.retries.max_attempts` | int | `3` | 0–10 | Retries on errors flagged `retryable` only. |
| `reliability.retries.backoff_base_ms` | int | `250` | ≥ 1 | Exponential backoff base. |
| `reliability.retries.backoff_max_ms` | int | `10000` | ≥ `backoff_base_ms` | Backoff ceiling. |
| `reliability.retries.jitter` | bool | `true` | — | Randomized jitter on backoff. |
| `reliability.circuit_breaker.enabled` | bool | `true` | — | Per-provider circuit breakers (LLM, embeddings, vector DB); state exported as a metric. |
| `reliability.circuit_breaker.failure_threshold` | int | `5` | ≥ 1 | Consecutive failures that open the circuit. |
| `reliability.circuit_breaker.reset_timeout_ms` | int | `30000` | ≥ 1000 | Half-open probe interval. |
| `reliability.idempotency.enabled` | bool | `true` | — | Honour the `Idempotency-Key` header on mutating endpoints ([api-reference.md](api-reference.md#idempotency)). **Defaults to `true`, deliberately against the default-false rule for risky features** (TASK-0356): the feature starts nothing on its own — a request without the header touches no store and writes no byte — while defaulting it off would make a published contract false for every deployment that never read this setting, and a client whose retry silently executed twice has no way to notice. It is a kill switch for a store that is misbehaving, not a feature gate. Setting it to `false` passes keyed requests straight through: nothing is recorded, nothing is replayed, and no request is refused. |
| `reliability.idempotency.ttl_seconds` | int | `86400` | 1–604800 | How long a completed record stays replayable, and since TASK-0365 how long its file survives: a reclamation pass removes records a minute past this, so the directory holds roughly one window of keyed traffic rather than all of it. Raise it only as far as your clients actually retry. |
| `reliability.idempotency.in_flight_lease_seconds` | int | `900` | ≥ 1, ≤ `ttl_seconds` | How long an unfinished reservation holds its key before another request may take it over. A worker killed mid-request frees its key on this lease rather than holding it for the full lifetime — which would force the client to invent a new key, the exact duplicate the header exists to prevent. The loader refuses a lease longer than `ttl_seconds`, because an unfinished request outliving a completed one inverts the intent. It is also the reclamation interval (TASK-0365): one host runs at most one pass per lease, so shortening the lease tightens cleanup as well as takeover. |
| `reliability.shutdown_grace_seconds` | int | `20` | 0–3600 | How long a stopping API process waits for the ingestion jobs it is running before cancelling them (TASK-0436). During the wait it starts no new job (`POST /v1/ingest` answers `503 NOT_READY`, retryable); a job that finishes inside the window settles normally, and one that does not is cancelled and settled `failed` with `reason_code: INTERRUPTED`, keeping its checkpoint. Adapters are closed only after every job has settled. **Keep it below whatever kills the process**: `docker stop` sends `SIGKILL` after 10 s by default and Kubernetes after 30 s, and a process killed mid-wait settles nothing — its jobs stay `running` until the next start reaps them. `0` cancels at once. The wait is a floor on shutdown time only when jobs are running, and it is not a ceiling: a cancelled job still has to leave the parse pool, which waits for the documents already being parsed. |
| `reliability.stale_job_seconds` | int | `120` | 1–86400 | How long a `running` job must have been quiet before a starting process treats it as abandoned and settles it `failed` with `reason_code: INTERRUPTED` (TASK-0436). "Quiet" means neither the job record nor its per-document log has been written: the journal can be shared by several processes on one host, and a job still writing is the only evidence that some other process owns it. The look runs once at startup, in the background, and **once more this many seconds later** if any record was too recent to judge — a process restarted seconds after a crash finds the orphan still fresh, so without the second look it would stay `running` until some later restart. The cost of a low value: a live job that writes nothing for this long — one document that takes longer than this to parse, embed and index — is settled by *another* process starting on the same journal while it is still running. It corrects itself when the job finishes, because the final settle overwrites the record, but in between the job reads as interrupted. Raise it if a single document can take minutes. A job is only ever settled here, never resumed. |

**`reliability.degradation_ladder` was removed on 2026-11-12** (TASK-0507). It was validated and read by nothing: the degradation ladder ran whatever it said, so `false` turned nothing off. A `config.yaml` that still sets it fails at load with a message naming the key and the reason; delete the line. The ladder itself is unchanged — a degraded response still carries `degraded: true` and its `mode` ([reliability.md](reliability.md) §2).

## `traces`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `traces.store` | bool | `true` | — | D8: persist every query's full trace locally (retrieved chunks, scores, prompt, response). |
| `traces.retention_days` | int | `30` | ≥ 1 | Trace retention window. |
| `traces.replay` | bool | `true` | — | Enable `fasterrag replay` / `POST /v1/replay` (re-execute a past query under a candidate config, side-by-side diff). |

## `cost`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `cost.estimator` | bool | `true` | — | D9: enable `fasterrag estimate`, `POST /v1/estimate`, `fasterrag ingest --dry-run`, and `FasterRag.estimate` (token counts, projected embedding cost per provider, BEFORE ingestion). When `false` each refuses with `VALIDATION_FAILED` naming this setting, rather than reporting an estimate of zero. `benchmark --suite ingest` is unaffected: it times the same parse-and-chunk work and reports no cost. |
| `cost.per_query_token_budget` | int | `0` | ≥ 0 (0 = unlimited) | Hard token budget per query, checked before the provider is called against the prompt plus `llm.max_tokens` — the ceiling the call can reach, because a completion length is unknown until the tokens are spent. Exceeding it returns `BUDGET_EXCEEDED` (402), which is non-retryable so the degradation ladder cannot absorb it into an extractive answer. Exact and identical on every replica. |
| `cost.per_tenant_token_budget` | int | `0` | ≥ 0 (0 = unlimited) | Per-tenant token budget over a **rolling 24-hour window** (the documented "rolling" needs a period to mean anything). Charged the actual tokens a query cost, admitted against the ceiling. Without `security.multi_tenancy: true` all traffic shares one bucket, which is what a per-tenant budget means for a single-tenant deployment. **Counted in-process**, like the per-key rate limiter (TASK-0216): N replicas grant N times this budget, so size it for your replica count. **Embedding spend is charged as of TASK-0312**, from the provider's reported token count where there is one — an ingest records against the window but is never refused mid-job, because abandoning a half-indexed corpus is worse than letting the cap bite later. **Ingest and query share one window as of TASK-0337**: both charge the process-wide governor, so a query is refused on what ingestion already spent, and the cap is reachable through the REST API. Until then `GenerationService` built its own governor on every request, so the window was discarded between requests and the ceiling could not be reached at all. **The window became accurate to within one hour as of TASK-0366, and the error only ever refuses early.** It is held as at most 26 hourly buckets per tenant (24 whole hours plus the two partial ones the window's ends straddle; the measured run held 24) instead of one `(timestamp, tokens)` entry per charge, which was measured at 122.5 bytes per charge and 128 ms per check at 1M charges in a window (BENCH-0005) — unbounded, in a governor that TASK-0337 made process-lifetime. A bucket is discarded only once its **newest** charge has aged out, so spend is remembered for up to an hour longer than an exact window would remember it. **What changes for you: a tenant can stay refused up to an hour after an exact 24-hour window would have released its budget, and can never be admitted past the cap.** The amount counted inside the window is unchanged and still exact; only the moment spend leaves the window moved. If you set this expecting a to-the-second window, that is the behaviour that changed. |

### Which models carry a price

Costing is a lookup in the dated list-price tables in `services/estimation.py`, keyed by **provider *and* model** — a rate is never applied across providers, because the same model name behind a different vendor is a different bill. Priced today:

| Provider | Priced models | Checked |
|---|---|---|
| `openai` (embeddings) | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` | 2026-08-09 |
| `openai` (generation) | the GPT-5.x, GPT-4.1, GPT-4o, GPT-4, GPT-3.5-turbo and o-series chat models | 2026-08-09 |
| `anthropic` | the Claude Fable/Mythos 5, Opus 5, Opus 4.5–4.8, Sonnet 4.5/4.6/5 and Haiku 4.5 ids, alias and dated forms | 2026-08-09 |
| `cohere` (embeddings) | `embed-english-v3.0`, `embed-multilingual-v3.0`, `embed-english-light-v3.0` | 2026-07-30 |
| `cohere` (generation) | `command-r-plus-08-2024` | 2026-08-09 |
| `huggingface`, `ollama` | **every** model — locally served, so zero is the recorded price, not a missing one | — |

Anything else — `openai_compatible` gateways, Cohere's Command A / R7B lineup (published as instance-hour rates, which cannot be converted to a per-token rate without a throughput measurement fasterRag does not have), a model released after the dates above — is **deliberately unpriced**: it contributes nothing to `fasterrag_cost_usd_total` and is counted instead by `fasterrag_unpriced_tokens_total` ([observability.md](observability.md)). A wrong price produces a confident wrong bill, which is worse than a visible gap.

## `autopilot`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `autopilot.enabled` | bool | `false` | — | D6: eval-driven auto-tuning. Generates a golden Q&A set from the corpus, searches the query-time parameters it can score without a rebuild — hybrid weights, `rrf_k` and rerank settings (`top_k` is deliberately excluded: measuring recall@k retrieves exactly k, so a top_k candidate cannot change the score; chunk size and overlap need the separate, priced index-time sweep and have no CLI surface yet, TASK-0145), outputs a **suggested config diff with measured deltas**. **NEVER auto-applies** — human approves. |
| `autopilot.golden_set_size` | int | `100` | 10–10000 | Q&A pairs generated for the golden set; the default for `fasterrag autopilot generate-golden-set --size`, which overrides it per run. |

## `eval`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `eval.regression_gate` | bool | `false` | — | D7: run the eval harness on every config/index change; block the change if retrieval quality regresses beyond tolerance (CI-integrated). |
| `eval.recall_tolerance` | float | `0.02` | 0.0–1.0 | Max allowed recall@k drop before the gate blocks. |
| `eval.ndcg_tolerance` | float | `0.02` | 0.0–1.0 | Max allowed nDCG drop before the gate blocks. |

## `observability`

All integration toggles default `false`.

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `observability.dashboard` | bool | `false` | — | Self-hosted **read-only** inspection dashboard (cache stats, tokens, costs, latencies, full LLM I/O history). Never controls the RAG. |
| `observability.dashboard_port` | int | `8080` | 1–65535 | Dashboard bind port. |
| `observability.otel` | bool | `false` | — | OTLP export of the four RAG spans *and* the metric catalogue to `otel_endpoint`. Spans preserve the fasterRag trace id, so one id works in the API, the logs, and your trace viewer; metrics are pushed every 60s and read the same registry `/metrics` renders. Needs `pip install fasterrag[otel]`; without it the toggle warns and queries keep serving. |
| `observability.otel_endpoint` | str\|null | `null` | valid URL when `otel: true` | The collector's OTLP/HTTP endpoint, taken as a **base URL**: `/v1/traces` and `/v1/metrics` are derived from it, exactly as the OpenTelemetry specification treats `OTEL_EXPORTER_OTLP_ENDPOINT`. Set it to `http://collector:4318`, not to a signal path — one setting feeds both exporters and no single signal path can serve both. A value that already ends in `/v1/traces`, `/v1/metrics`, or `/v1/logs` has that path replaced rather than appended to, so configurations written against earlier docs keep working. |
| `observability.langfuse` | bool | `false` | — | Flipping to `true` auto-provisions self-hosted Langfuse v3 (Docker Compose stack), performs all configuration incl. headless bootstrap, and returns the running URL `http://<host>:3000`. **No application-code changes at toggle time.** Gated by `fasterrag doctor`. See [observability.md](observability.md). |
| `observability.grafana` | bool | `false` | — | Flipping to `true` auto-provisions Grafana via provisioning-as-code (datasources + dashboards read at startup; UI read-only for provisioned resources). No manual clicks; no code changes. Gated by `fasterrag doctor`. |

## `security`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `security.auth` | bool | `false` | — | API-key authentication on all control-plane endpoints (keys carry scopes; see [security.md](security.md)). |
| `security.api_key_env` | str | `FASTERRAG_API_KEY` | valid env-var name | Env var holding the API key material (never the key itself in YAML). Required to be set only when `security.auth: true` (rule 9). |
| `security.multi_tenancy` | bool | `false` | — | Resolves a tenant per request and scopes collection, trace, job, metric and semantic-cache access to it. It does **not** by itself scope retrieval — see `security.tenant_scoped_retrieval` and [security.md](security.md) §3. |
| `security.tenant_scoped_retrieval` | bool | `false` | requires `auth` and `multi_tenancy` | Pushes the caller's tenant into **both** retrieval legs as a mandatory `tenant` equality filter, injected below any `filters` in the request body; a body that names the `tenant` field is refused with `TENANT_FORBIDDEN` rather than overridden. **Off by default because turning it on changes what an existing corpus answers**: chunks indexed before `multi_tenancy` was enabled carry no `tenant` payload field, and an absent field never matches, so an unbackfilled collection returns empty results. Backfill first with `fasterrag index backfill-tenant <collection> --tenant <id>` (dry-run by default; `fasterrag index tenants` reports how many points are untagged before anything is written), and add a payload index on `tenant` in the backend or every query pays an unindexed filter. Combining it with `auth: false` or `multi_tenancy: false` is refused at load (TASK-0339). |
| `security.tenant_header` | str | `X-Tenant-ID` | valid HTTP header name | Header carrying the tenant identifier when multi-tenancy is on. |
| `security.rate_limit_per_minute` | int | `600` | ≥ 1 | Per-key request rate limit (on by default once `auth` is enabled). Counted in the window named by `security.rate_limit_backend`. |
| `security.rate_limit_backend` | str | `memory` | `memory`, `redis` | Where the per-key window is counted. `memory` is per process: with N API processes or replicas the effective ceiling is `rate_limit_per_minute × N`, which is correct only for a single-process deployment. `redis` counts every replica into one shared fixed window (`INCR` + `EXPIRE` in one transaction, keyed by a SHA-256 fingerprint of the API key, never the key). Needs `pip install fasterrag[redis]`. **The Redis backend has never been exercised against a live Redis server** — only against an in-memory double (TASK-0216); see [security.md](security.md) §2. |
| `security.rate_limit_redis_url` | str | `redis://localhost:6379/0` | starts with `redis://`, `rediss://`, or `unix://` | Connection URL, read only when `security.rate_limit_backend` is `redis`. Counters are namespaced under `fasterrag:ratelimit`, so the same server can also back the caches. **A URL containing a password is a secret and must not be written here** — supply it through the environment instead (`FASTERRAG_SET=security.rate_limit_redis_url=...`). |
| `security.max_request_mb` | int | `25` | 1–1024 | Input size limit for API request bodies (ingestion uploads are governed by `ingestion.max_document_mb`). |
| `security.allowed_archive_roots` | list[str] | `[]` | paths | Directories `POST /v1/admin/export` may write an archive into and `POST /v1/admin/import` may read one from. **Authorization boundary**: both paths arrive in the request body, so an unconfined export overwrites any file the process can write (`config.yaml`, a web root) and an unconfined import reads any archive on the host. One list covers both verbs because they name the same artifact in the same directory. Paths are resolved before comparison, so `..` and symlinks cannot escape a root. Empty means unconfined, which is refused outright when `security.auth` is on — the same rule as `ingestion.allowed_source_roots`. |

---

## Cross-field validation rules (fail-fast at startup)

1. `retrieval.top_k ≤ retrieval.rerank_top_n`.
2. `chunking.overlap < chunking.chunk_size`.
3. `embeddings.api_key_env` must be set (and the env var present) for `openai`/`cohere` providers; same for `llm.api_key_env` except `ollama`. These lists name built-in providers. A plugin provider is decided differently, because the schema validates a plugin's name without loading it and so cannot know whether it needs a key: **a plugin never inherits a built-in's default key name.** `llm.api_key_env` and `vector_db.api_key_env` must be written out for a plugin provider — a variable name, or `null` — and leaving either to its default (`OPENAI_API_KEY`, `QDRANT_API_KEY`) stops the load. `embeddings.api_key_env` already defaults to `null`, so nothing is asked. A variable a plugin names must be present like any other (rule 9), and a plugin that needs a key it was not given refuses when it is built.
4. `vector_db.grpc_port ≠ vector_db.port`.
5. `generation.citations` cannot be `false` while `generation.grounded_or_refuse` is `true`.
6. `observability.otel_endpoint` required when `observability.otel: true`.
7. On Windows/WSL, `vector_db.docker.volume` must be a named volume, not a path.
8. `embeddings.tiering.rules` non-empty when `embeddings.tiering.enabled: true`.
9. Any `*_env` variable that an enabled feature reads and that is missing from the environment/`.env` at startup is a fatal `ConfigError` naming the variable (its value is never logged). A variable that nothing reads under the loaded configuration is not required (TASK-0444): `security.api_key_env` only when `security.auth: true`; `vector_db.api_key_env` for every vector database except `chroma` and `pgvector`, which never read it; `vector_db.pgvector.dsn_env` only when `provider: pgvector`; `embeddings.api_key_env` except for `huggingface` and `ollama`; `llm.api_key_env` except for `ollama`. The LLM key has no other exemption, because no setting turns generation off. `fasterrag doctor`'s `secrets` check applies the same rule.
10. `index.reindex.strategy: in_place` logs a prominent warning (dev-only path; no zero-downtime guarantee).
11. `vector_db.pgvector.dsn_env` required when `vector_db.provider: pgvector`.
