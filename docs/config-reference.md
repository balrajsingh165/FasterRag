# config-reference.md — Full `config.yaml` Schema

`config.yaml` drives ALL behavior. Any key here can be overridden per invocation with `--set dotted.key=value`, or through the `FASTERRAG_SET` environment variable for containers ([cli-reference.md](cli-reference.md)); both are validated exactly as a file value is. `.env` holds ONLY credentials/secrets — config references secrets exclusively **by environment-variable name** (e.g. `api_key_env: OPENAI_API_KEY`); secret values are NEVER inlined in YAML. The loader is **pydantic-settings** with a YAML source for config and env/`.env` for secrets; it validates the entire schema at startup and **fails fast** with a clear error naming the offending key. **Every integration toggle defaults to `false`.**

Conventions used below:

- **Type** uses Python/Pydantic notation. `str|null` means optional/nullable.
- **Validation** is enforced at startup by the Pydantic schema; violations abort startup with `ConfigError`.
- Durations are integer seconds unless the key name says otherwise (`*_ms` = milliseconds).

## Complete example (all defaults shown; secrets referenced by env-var name only)

```yaml
app:
  host: 0.0.0.0
  port: 8000
  workers: 4
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
  retries:
    max_attempts: 3
    backoff_base_ms: 250
    backoff_max_ms: 10000
    jitter: true
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_ms: 30000
  degradation_ladder: true

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
  rate_limit_per_minute: 600
  max_request_mb: 25
```

---

## `app`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `app.host` | str | `0.0.0.0` | valid IPv4/IPv6 address or hostname | Bind address for the API server. |
| `app.port` | int | `8000` | 1–65535 | API server port. |
| `app.workers` | int | `4` | ≥ 1 | Uvicorn/Gunicorn worker process count for the API tier (not pipeline workers). |
| `app.log_level` | str | `info` | one of `debug`, `info`, `warning`, `error` | Structured-log verbosity. |

## `vector_db`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `vector_db.provider` | str | `qdrant` | one of `qdrant`, `milvus`, `weaviate`, `pinecone`, `pgvector`, `chroma` | Selects the concrete `VectorDBAdapter` via the factory. Changing this swaps backends with zero code changes. |
| `vector_db.mode` | str | `docker` | one of `docker`, `external` | `docker` = system-managed container (fasterRag launches/manages it). `external` = user-run instance, local (no-Docker) **or** remote via `host`/`port` (covers the remote-IP mode). |
| `vector_db.host` | str | `localhost` | non-empty hostname or IP | Backend host. For remote Qdrant set the remote machine's IP/hostname. |
| `vector_db.port` | int | `6333` | 1–65535 | REST port. Qdrant client default is 6333. |
| `vector_db.grpc_port` | int | `6334` | 1–65535; must differ from `port` | gRPC port. Qdrant client default is 6334. **Both 6333 and 6334 must be exposed/reachable** — Qdrant GitHub Discussion #2195 records connection failures when only 6333 is exposed and the client attempts gRPC. |
| `vector_db.prefer_grpc` | bool | `false` | — | Matches the Qdrant Python client default (`prefer_grpc=False`). Set `true` for gRPC-first traffic (requires `grpc_port` reachable). |
| `vector_db.https` | bool | `false` | — | Whether to reach the backend over TLS. **Must be set explicitly rather than inferred**: the Qdrant Python client silently switches to HTTPS whenever an `api_key` is supplied, which fails against the plain-HTTP listener a container serves by default. `false` matches the documented private-network deployment (an API key over a private link); set `true` for a TLS-terminated remote instance ([deployment.md](deployment.md) §2). |
| `vector_db.api_key_env` | str\|null | `QDRANT_API_KEY` | valid env-var name or null | Name of the env var holding the backend API key (Qdrant: consumed as `QDRANT__SERVICE__API_KEY` on the server side). Never the key itself. Null = unauthenticated (local dev only). |
| `vector_db.docker.image` | str | `qdrant/qdrant:v1.18.1` | non-empty image ref; pinned tag required (no `latest`) | Image used in `docker` mode. |
| `vector_db.docker.volume` | str | `fasterrag_qdrant_storage` | valid Docker volume name | Storage volume for persistence. **On Windows/WSL this MUST be a named Docker volume** — bind mounts have known file-system data-loss issues per Qdrant's install docs. The loader rejects bind-mount paths on Windows/WSL. |
| `vector_db.pgvector.dsn_env` | str\|null | `null` | valid env-var name | Name of the environment variable holding the PostgreSQL DSN. **Required when `provider: pgvector`** (cross-field rule 11) and left unset otherwise — it defaults to `null` rather than to a name because every populated `*_env` key is required to be present at startup by rule 9, so a default would demand a PostgreSQL DSN from every Qdrant deployment. A DSN carries the password, so it is named here and stored in `.env`, never in `config.yaml`. `host`, `port`, and `grpc_port` are ignored by this provider: a usable connection string already carries the host, the database, the role, and the SSL mode. |
| `vector_db.pgvector.db_schema` | str | `fasterrag` | `^[a-z_][a-z0-9_]{0,62}$` | PostgreSQL schema the adapter owns. It holds the three catalog tables (`fasterrag_collections`, `fasterrag_aliases`, `fasterrag_snapshots`) plus one table per collection, so pointing two deployments at one database with different schemas keeps them isolated. Created on first use if absent. |
| `vector_db.collection.default_name` | str | `default` | `^[a-zA-Z0-9_-]{1,64}$` | Default collection name. |
| `vector_db.collection.distance` | str | `cosine` | one of `cosine`, `dot`, `euclid` | Distance metric for dense vectors. |
| `vector_db.collection.shard_number` | int | `1` | ≥ 1 | Shards per collection (scaling knob; passed through the adapter). |
| `vector_db.collection.replication_factor` | int | `1` | ≥ 1 | Replicas per shard (availability knob; passed through the adapter). |

## `embeddings`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `embeddings.provider` | str | `huggingface` | one of `openai`, `cohere`, `huggingface`, `ollama` | Embedding adapter selection. `huggingface` = local sentence-transformers (no key needed). |
| `embeddings.model` | str | `BAAI/bge-small-en-v1.5` | non-empty | Model name/ID understood by the selected provider. Recorded per chunk for drift detection (see `index.lockfile`). |
| `embeddings.api_key_env` | str\|null | `null` | valid env-var name or null | Env var holding the provider key (e.g. `OPENAI_API_KEY`, `COHERE_API_KEY`). Required (non-null) when provider is `openai` or `cohere`. |
| `embeddings.batch_size` | int | `64` | 1–2048 | Texts per embedding request. Batched embedding is far cheaper than one-at-a-time calls. |
| `embeddings.dimensions` | int\|null | `null` | ≥ 8 when set | Output dimensionality; `null` = the model's native size, which is the right setting unless you have a reason. Behaviour differs by provider, because shortening a vector is only lossless for a model trained for it (Matryoshka representation learning). **Hosted providers that support it** (OpenAI `text-embedding-3-*`) receive the value as an API parameter and shorten server-side — verified against the live API: `text-embedding-3-small` returns 1536 natively and exactly 256 when asked (TASK-0207). **Local `huggingface` models** cannot be shortened, so a value that disagrees with what the model emits is refused at load time with an error naming both sizes — rather than left to fail later as a rejected upsert against a collection already created at the wrong width. Must match the collection's vector size. |
| `embeddings.cache.enabled` | bool | `true` | — | Embedding cache keyed by content hash + model + version. |
| `embeddings.cache.max_entries` | int | `10000` | ≥ 1 | Entry ceiling for the embedding cache. Raise when reingesting a large corpus repeatedly, which is exactly when the cache pays for itself. |
| `embeddings.cache.backend` | str | `disk` | one of `memory`, `disk`, `redis` | Where embedding-cache entries live. `disk` survives a restart on one host; `redis` is the only backend several workers or replicas can share, so a vector one of them paid for is not re-embedded by the next. |
| `embeddings.cache.redis_url` | str | `redis://localhost:6379/0` | starts with `redis://`, `rediss://`, or `unix://` | Connection URL, read only when `embeddings.cache.backend` is `redis`. Needs `pip install fasterrag[redis]`. Entries are namespaced under `fasterrag:embedding`, so one server can back both caches and `clear` never reaches the other's keys or a co-tenant application's. **A URL containing a password is a secret and must not be written here** — supply it through the environment instead (`FASTERRAG_SET=embeddings.cache.redis_url=...`), which is exactly what the `.env`-only policy requires. |
| `embeddings.tiering.enabled` | bool | `false` | — | Tiered embedding: route document classes to different models (cheap models for high-volume/low-priority classes; higher-cost models where retrieval precision matters). |
| `embeddings.tiering.rules` | list[TierRule] | `[]` | each rule: `{match: <metadata filter>, provider: <provider>, model: <model>}`; must be non-empty when tiering enabled | Ordered routing rules, first match wins. |

## `llm`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `llm.provider` | str | `openai` | one of `openai`, `anthropic`, `cohere`, `ollama`, `openai_compatible` | LLM adapter selection. `openai_compatible` targets any endpoint speaking the OpenAI API shape (set `base_url`). |
| `llm.model` | str | `gpt-4o-mini` | non-empty | Generation model ID understood by the selected provider (e.g. `claude-opus-5` for provider `anthropic`). |
| `llm.api_key_env` | str\|null | `OPENAI_API_KEY` | valid env-var name or null | Env var holding the provider key. Null allowed only for `ollama`/local. |
| `llm.base_url` | str\|null | `null` | valid URL when set | Endpoint override (required for `openai_compatible`, optional for `ollama`). |
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
| `chunking.strategy` | str | `recursive` | one of `fixed`, `recursive`, `semantic`, `layout`, `late` | Chunking strategy (see [architecture.md](architecture.md) §5). `late` requires a local `embeddings.provider` for its pooling pass and falls back to ordinary embedding on a hosted one. |
| `chunking.chunk_size` | int | `768` | 64–2500; warning logged above 1024 | Target chunk size in tokens. Practical working range is ~512–1024; the ~2,500-token "context cliff" (directional ceiling from a January 2026 preprint) is the hard upper bound. |
| `chunking.overlap` | int | `64` | 0 ≤ overlap < `chunk_size` | Token overlap between adjacent chunks. |
| `chunking.token_counter` | str | `auto` | one of `auto`, `estimate`, `model` | How `chunk_size` and `overlap` are counted. `auto` loads the embedding model's own tokenizer when the provider ships a local one and estimates otherwise; `estimate` forces the `chars_per_token` ratio; `model` forces the real tokenizer for any provider, which requires the matching tokenizer in the local Hugging Face cache and falls back to the estimate with a logged warning when it is absent. Loading never reaches the network. |
| `chunking.chars_per_token` | int | `4` | 1–16 | Characters per token assumed by the estimate, and by the first split pass before real counts refine it. Lower it for corpora that tokenize densely (code, CJK) when running `token_counter: estimate`. |
| `chunking.semantic_percentile` | float | `0.95` | 0.50–0.99 | Only used by `strategy: semantic`. Distance percentile above which a gap between adjacent sentences becomes a chunk boundary. Lower splits more eagerly, giving smaller and more topically uniform chunks; higher keeps related passages together. |
| `chunking.contextual_enrichment` | bool | `false` | — | Contextual-retrieval-style enrichment: prepend an LLM-generated document-level context to each chunk before embedding and BM25 indexing (Anthropic, Sept 2024: −49% failed retrievals; −67% with reranking). Costs LLM calls at ingest; uses provider prompt caching of the parent document. |
| `chunking.context_tokens` | int | `75` | 25–150 | Target length of the generated per-chunk context. Recommended ~50–100 tokens. |

## `retrieval`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `retrieval.top_k` | int | `10` | 1–100; ≤ `rerank_top_n` | Final number of chunks passed to context assembly. |
| `retrieval.hybrid` | bool | `true` | — | Run dense ANN + sparse BM25 legs in parallel and fuse. `false` = dense-only. |
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

## `index`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `index.lockfile` | bool | `true` | — | D1: every index build writes `index.lock` (config hash, embedding model name+version, chunker strategy+version, per-document content hashes). Drift is detected and reported. |
| `index.reindex.strategy` | str | `blue_green` | one of `blue_green`, `in_place` | D2: `blue_green` builds the new collection in parallel, validates against the eval set, then atomically switches via collection alias. `in_place` is allowed only for dev. |
| `index.reindex.eval_gate` | bool | `true` | — | Require eval-set validation to pass before the alias swap. |
| `index.reindex.rollback_retention_hours` | int | `72` | ≥ 0 | How long the old collection is retained for instant rollback after a swap. |

## `reliability`

| Name | Type | Default | Allowed values / validation | Description |
|---|---|---|---|---|
| `reliability.timeouts.vector_db_ms` | int | `5000` | ≥ 100 | Timeout for every vector-DB call. Every external interaction MUST have an explicit timeout. |
| `reliability.timeouts.embeddings_ms` | int | `30000` | ≥ 100 | Timeout per embedding batch call. |
| `reliability.timeouts.llm_ms` | int | `120000` | ≥ 1000 | Timeout per LLM call (streaming: per-connection). |
| `reliability.retries.max_attempts` | int | `3` | 0–10 | Retries on errors flagged `retryable` only. |
| `reliability.retries.backoff_base_ms` | int | `250` | ≥ 1 | Exponential backoff base. |
| `reliability.retries.backoff_max_ms` | int | `10000` | ≥ `backoff_base_ms` | Backoff ceiling. |
| `reliability.retries.jitter` | bool | `true` | — | Randomized jitter on backoff. |
| `reliability.circuit_breaker.enabled` | bool | `true` | — | Per-provider circuit breakers (LLM, embeddings, vector DB); state exported as a metric. |
| `reliability.circuit_breaker.failure_threshold` | int | `5` | ≥ 1 | Consecutive failures that open the circuit. |
| `reliability.circuit_breaker.reset_timeout_ms` | int | `30000` | ≥ 1000 | Half-open probe interval. |
| `reliability.degradation_ladder` | bool | `true` | — | D4: graceful fallbacks (reranker down → hybrid-only; vector DB down → semantic-cache-only; LLM down → extractive answers). Every degraded response carries `degraded: true` + `mode`. |

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
| `cost.per_query_token_budget` | int | `0` | ≥ 0 (0 = unlimited) | Hard token budget per query; exceeding returns a budget-exceeded problem response. |
| `cost.per_tenant_token_budget` | int | `0` | ≥ 0 (0 = unlimited) | Rolling per-tenant token budget (requires `security.multi_tenancy: true` to be meaningful). |

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
| `autopilot.enabled` | bool | `false` | — | D6: eval-driven auto-tuning. Generates a golden Q&A set from the corpus, searches chunk size / top_k / hybrid weights / rerank settings, outputs a **suggested config diff with measured deltas**. **NEVER auto-applies** — human approves. |
| `autopilot.golden_set_size` | int | `100` | 10–10000 | Q&A pairs generated for the golden set. |

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
| `security.api_key_env` | str | `FASTERRAG_API_KEY` | valid env-var name | Env var holding the API key material (never the key itself in YAML). |
| `security.multi_tenancy` | bool | `false` | — | Tenant-scoped collections and API keys; isolation enforced at the service layer; tenant tag on every trace/metric. |
| `security.tenant_header` | str | `X-Tenant-ID` | valid HTTP header name | Header carrying the tenant identifier when multi-tenancy is on. |
| `security.rate_limit_per_minute` | int | `600` | ≥ 1 | Per-key request rate limit (on by default once `auth` is enabled). |
| `security.max_request_mb` | int | `25` | 1–1024 | Input size limit for API request bodies (ingestion uploads are governed by `ingestion.max_document_mb`). |

---

## Cross-field validation rules (fail-fast at startup)

1. `retrieval.top_k ≤ retrieval.rerank_top_n`.
2. `chunking.overlap < chunking.chunk_size`.
3. `embeddings.api_key_env` must be set (and the env var present) for `openai`/`cohere` providers; same for `llm.api_key_env` except `ollama`.
4. `vector_db.grpc_port ≠ vector_db.port`.
5. `generation.citations` cannot be `false` while `generation.grounded_or_refuse` is `true`.
6. `observability.otel_endpoint` required when `observability.otel: true`.
7. On Windows/WSL, `vector_db.docker.volume` must be a named volume, not a path.
8. `embeddings.tiering.rules` non-empty when `embeddings.tiering.enabled: true`.
9. Any referenced `*_env` variable that is missing from the environment/`.env` at startup is a fatal `ConfigError` naming the variable (its value is never logged).
10. `index.reindex.strategy: in_place` logs a prominent warning (dev-only path; no zero-downtime guarantee).
11. `vector_db.pgvector.dsn_env` required when `vector_db.provider: pgvector`.
