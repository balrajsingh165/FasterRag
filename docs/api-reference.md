# api-reference.md — REST API

The REST API is one half of the control plane (the other is the [CLI](cli-reference.md)). All endpoints are versioned under `/v1`. All request/response bodies are JSON except SSE streams and problem responses (`application/problem+json`).

- Authentication: `Authorization: Bearer <api-key>` when `security.auth: true`. Keys carry scopes (`ingest`, `query`, `collections`, `admin`). See [security.md](security.md).
- Multi-tenancy: `X-Tenant-ID` header (configurable) when `security.multi_tenancy: true`.
- Idempotency: all mutating endpoints accept an `Idempotency-Key` header; replaying the same key returns the original result and performs no duplicate work.
- Backpressure: when queues are full the API returns `429` with a `Retry-After` header.

## Error model (RFC 9457 `application/problem+json`)

Every error response is a problem document with a **stable machine-readable `code`**:

```json
{
  "type": "https://fasterrag.dev/problems/provider-timeout",
  "title": "Embedding provider timed out",
  "status": 503,
  "detail": "OpenAI embeddings call exceeded 30000 ms (attempt 3/3).",
  "instance": "/v1/ingest/job_01J8Z3W7",
  "code": "EMBED_PROVIDER_TIMEOUT",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "retryable": true
}
```

| Field | Meaning |
|---|---|
| `type` / `title` / `status` / `detail` / `instance` | Standard RFC 9457 members. |
| `code` | Stable machine-readable error code (see table). Never renamed once released. |
| `trace_id` | Correlation id; matches OTel trace and log lines. |
| `retryable` | Whether the client may retry (mirrors the error-taxonomy flag). |

### Error code table (maps 1:1 to the typed exception taxonomy in [reliability.md](reliability.md))

| `code` | HTTP | Taxonomy class | When |
|---|---|---|---|
| `CONFIG_INVALID` | 500 | `ConfigError` | Startup/config contract violated (should never reach prod). |
| `AUTH_MISSING` / `AUTH_INVALID` / `AUTH_SCOPE` | 401 / 401 / 403 | — | Missing key / bad key / insufficient scope. |
| `TENANT_FORBIDDEN` | 403 | — | Key not authorized for the tenant/collection. |
| `VALIDATION_FAILED` | 422 | — | Request body failed schema validation (`errors[]` lists fields). |
| `NOT_FOUND` | 404 | — | Unknown job, collection, trace, or document id. |
| `CONFLICT` | 409 | — | e.g. collection already exists; concurrent alias swap. |
| `PAYLOAD_TOO_LARGE` | 413 | — | Body exceeds `security.max_request_mb` / `ingestion.max_document_mb`. |
| `RATE_LIMITED` | 429 | — | Per-key rate limit exceeded (`Retry-After` set). |
| `QUEUE_FULL` | 429 | `IngestionError` | Bounded queue overflow (`Retry-After` set). |
| `BUDGET_EXCEEDED` | 402 | — | Per-query or per-tenant token budget exhausted (D9). **Defined in the taxonomy and raised by nothing** — the runtime cost governor is not built (TASK-0242), so no request can currently receive this problem. |
| `PARSE_FAILED` | 422 | `ParseError` | Document unparseable; also the DLQ reason code. |
| `CHUNK_FAILED` | 500 | `ChunkError` | Chunker invariant violated. |
| `EMBED_PROVIDER_TIMEOUT` / `EMBED_PROVIDER_ERROR` | 503 | `EmbedError`/`ProviderError` | Embedding provider timeout / hard failure. |
| `VECTOR_DB_AUTH_FAILED` | 503 | `ProviderError` | The vector database rejected our credentials. Distinct from `EMBED_PROVIDER_ERROR` because the knob to turn is `vector_db.api_key_env`, not the embedding provider. Never retried. |
| `RETRIEVAL_FAILED` | 503 | `RetrievalError` | Vector DB/BM25 leg failed after retries. |
| `RERANK_FAILED` | 503 | `RetrievalError` | Reranker failure (degradation ladder may answer instead — see below). |
| `GENERATION_FAILED` | 503 | `GenerationError` | LLM provider failure after retries. |
| `GENERATION_TIMEOUT` | 504 | `GenerationError` | LLM provider timed out. Split from `GENERATION_FAILED` for the same reason embeddings split theirs: a timeout says raise `reliability.timeouts.llm_ms` or shrink the context, a hard failure says look at the provider. |
| `INSUFFICIENT_EVIDENCE` | 200 | — | Not an error transport-wise: structured grounded-or-refuse response (D5), see `/v1/query`. |
| `CIRCUIT_OPEN` | 503 | `ProviderError` | Circuit breaker open for the named provider (`detail` names it, `Retry-After` set). |
| `PROVISIONING_FAILED` | 500 | `ProvisioningError` | Auto-provisioning step failed; `detail` carries the doctor-style fix-it hint. |
| `CACHE_ERROR` | 500 | `CacheError` | Cache backend failure (system degrades to cache-off rather than failing queries; surfaced on admin endpoints). |
| `NOT_READY` | 503 | — | A readiness dependency check failed; returned only by `/readyz`, with a `dependencies[]` extension member naming each check and its state. |
| `INTERNAL` | 500 | `FasterRagError` | Anything unclassified. Always carries `trace_id`. Never returned without a problem body. |

---

## Ingestion

### `POST /v1/ingest`

Accepts an ingestion job **asynchronously** — validates, journals, enqueues, returns immediately. Never blocks on parsing/embedding/indexing.

Request:

```json
{
  "collection": "default",
  "sources": [
    {"type": "path", "value": "/data/contracts/"},
    {"type": "url", "value": "https://example.com/doc.pdf"}
  ],
  "metadata": {"department": "legal"},
  "priority_class": "standard"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `collection` | str | no (default collection) | Must exist. |
| `sources[]` | list | yes, ≥ 1 | `type` ∈ `path`, `url`, `inline`; `value` per type. |
| `metadata` | object | no | Merged into every produced chunk's metadata. |
| `priority_class` | str | no | Tiered-embedding routing class (D9). Merged into chunk metadata as `priority_class`, which is what `embeddings.tiering.rules[].match` matches on. |

Responses: `202 Accepted` `{"job_id": "job_01J8Z3W7", "status": "queued"}` · `422` · `429 QUEUE_FULL`.

### `GET /v1/ingest/{job_id}`

Job status: `{"job_id", "status": "queued|running|completed|failed|partial", "counts": {"total", "parsed", "chunked", "embedded", "indexed", "deduplicated", "dead_lettered"}, "started_at", "finished_at"}`. `404` if unknown.

### `GET /v1/ingest/{job_id}/documents?status=dead_lettered`

Per-document status (D3), filterable. Each entry: `{"document_id", "source", "status", "reason_code", "attempts", "content_hash"}`. Reason codes reuse the error `code` table (e.g. `PARSE_FAILED`).

### `POST /v1/ingest/{job_id}/retry-dlq`

Re-enqueues dead-lettered documents of the job. `202`.

## Query

### `POST /v1/query`

```json
{
  "collection": "default",
  "query": "What is the termination clause in the 2024 vendor agreement?",
  "top_k": 10,
  "filters": {"department": "legal", "year": {"$gte": 2024}},
  "stream": true,
  "include_chunks": false
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | str | yes | 1–8192 chars. |
| `collection` | str | no | Defaults to configured default. |
| `top_k` | int | no | Overrides `retrieval.top_k`; same bounds. |
| `filters` | object | no | Metadata filter expression, pushed down to both retrieval legs. |
| `stream` | bool | no (default per `llm.streaming`) | `true` → SSE response. |
| `include_chunks` | bool | no | Include full retrieved chunk texts in the (non-streamed) response. |

Non-streaming `200`:

```json
{
  "answer": "The termination clause allows either party to ...",
  "citations": [
    {"chunk_id": "c_9f2", "source": "s3://contracts/vendor-2024.pdf", "page": 12,
     "span": {"start": 128, "end": 342}, "score": 0.91}
  ],
  "usage": {"prompt_tokens": 3211, "completion_tokens": 187},
  "timings_ms": {"embed": 8, "retrieve": 41, "fuse": 1, "rerank": 143, "assemble": 6, "generate": 902, "total": 1101},
  "degraded": false,
  "mode": "full",
  "truncated": false,
  "evidence_dropped": 0,
  "faithfulness": 0.93,
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "cache": {"semantic_hit": false}
}
```

`truncated` is `true` when the model stopped at `llm.max_tokens` rather than finishing. The answer is still served — a partial answer with a flag beats no answer — but a caller cannot otherwise tell a complete answer from one that ran out mid-sentence. A reasoning model can spend the whole ceiling before emitting anything, which arrives as an **empty** `answer` with `truncated: true`; this was observed against a real OpenAI-compatible endpoint (TASK-0206).

`evidence_dropped` counts chunks that were retrieved and reranked but did not fit the context budget. Unlike `truncated`, nothing was cut off — the answer is simply grounded in a subset of the evidence the retriever chose. A non-zero value is what tells an operator to raise `chunking.chunk_size` or lower `retrieval.top_k`.

Degraded responses (D4) always carry `degraded: true` and `mode` ∈ `full`, `hybrid_only` (reranker down), `cache_only` (vector DB down), `extractive` (LLM down) — there is never a silent quality drop. **As built, only `hybrid_only` and `extractive` are served; `cache_only` is specified but not implemented (TASK-0159), and a vector-DB outage currently returns a retryable `RETRIEVAL_FAILED` problem instead.**

Grounded-or-refuse (D5): when faithfulness < threshold, `200` with:

```json
{"code": "INSUFFICIENT_EVIDENCE", "answer": null,
 "best_candidates": [{"chunk_id": "c_9f2", "source": "...", "score": 0.41}],
 "faithfulness": 0.38, "threshold": 0.7, "trace_id": "..."}
```

#### Streaming semantics (SSE)

`stream: true` → `Content-Type: text/event-stream`. Event order:

```
event: meta       data: {"trace_id": "...", "mode": "full", "degraded": false, "cache": {...}}
event: token      data: {"text": "The"}          (repeated; first token ASAP = TTFT)
event: citations  data: {"citations": [...]}      (once, after retrieval concludes)
event: usage      data: {"usage": {...}, "timings_ms": {...}, "faithfulness": 0.93}
event: done       data: {}
```

On mid-stream failure: `event: error` with a problem document, then the stream closes. Clients must treat a missing `done` as an incomplete answer.

Grounded-or-refuse while streaming: a token cannot be unsaid once sent, so with `generation.grounded_or_refuse: true` no `token` event is emitted until the whole answer has been graded — time-to-first-token is traded for the guarantee the flag exists to provide, and the flag defaults to `false` so the trade is only made on request. A withheld answer replaces the `token`, `citations`, and `usage` events with one `insufficient_evidence` event carrying the same body as the non-streamed refusal, followed by `done`:

```
event: meta                   data: {"trace_id": "...", "mode": "full", "degraded": false}
event: insufficient_evidence  data: {"code": "INSUFFICIENT_EVIDENCE", "answer": null, "best_candidates": [...], "faithfulness": 0.38, "threshold": 0.7, "trace_id": "..."}
event: done                   data: {}
```

`done` is still sent: the query completed and declined, which is a finished response, not a truncated one.

A semantic cache hit streams too. The whole answer is already known, so it arrives as a single `token` event after `meta`, and `meta` carries the `cache` member so a client learns it is being served from cache before the text. The event sequence is otherwise identical, so no client needs a separate code path for a cached response.

### `POST /v1/retrieve`

Retrieval without generation, for callers bringing their own LLM step. Same request body as `/v1/query` minus the generation controls; returns the ranked chunks with the rank and score every retrieval leg gave each one, so a client can see *why* a chunk placed where it did.

```json
{"query": "...", "collection": "docs", "top_k": 10, "filters": {"source": "handbook"}}
```

Responds `200` with `{"chunks": [...], "trace_id": "..."}`. No generation happens, so no LLM cost is incurred and `INSUFFICIENT_EVIDENCE` cannot apply.

## Collections

| Method + path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /v1/collections` | List collections (name, vectors count, embedding model+version, created_at, alias target). | 200 | — |
| `POST /v1/collections` | Create: `{"name", "distance", "shard_number", "replication_factor"}`. | 201 | 409 `CONFLICT`, 422 |
| `GET /v1/collections/{name}` | Detail incl. `index.lock` summary and drift status (D1). | 200 | 404 |
| `PATCH /v1/collections/{name}` | **Not yet implemented.** Mutable settings only (metadata schema, description). | 200 | 404, 422 |
| `DELETE /v1/collections/{name}` | Drop collection (requires `admin` scope; refuses while it is an active alias target unless `?force=true`). | 204 | 404, 409 |
| `POST /v1/collections/{name}/reindex` | **Not yet implemented over REST** — shipped on the CLI as `fasterrag index reindex`. D2 zero-downtime reindex: blue/green build → eval gate → alias swap. Returns a job id. | 202 | 404, 409 |
| `POST /v1/collections/{name}/rollback` | **Not yet implemented over REST** — shipped on the CLI as `fasterrag index rollback`. Alias flip back to the retained previous collection (within retention window). | 200 | 404, 409 |

## Health & readiness

| Method + path | Purpose |
|---|---|
| `GET /healthz` | Liveness: process is up. Always 200 if the process can answer. No dependency checks. |
| `GET /readyz` | Readiness: dependencies actually checked (vector DB `health()`, queue backend, config valid, circuit-breaker states). 200 with `{"status": "ready", "dependencies": [...]}` when every check passes; otherwise 503 with a `NOT_READY` problem body whose `dependencies[]` member lists each check and its state. Checks are registered by the slice that introduces the dependency, so the report always reflects what is actually verifiable. |

## Admin & provisioning

All require the `admin` scope.

| Method + path | Purpose | Success |
|---|---|---|
| `POST /v1/admin/provision/{tool}` | Trigger config-driven provisioning for `qdrant`, `langfuse`, or `grafana` (equivalent to the config toggle path; idempotent; doctor-gated). Returns `{"status", "url"}` — e.g. Langfuse returns `http://<host>:3000`. | 200 |
| `GET /v1/admin/provision/{tool}/status` | Provisioning/health state of the tool. | 200 |
| `GET /v1/admin/doctor` | Machine-readable doctor report (D10): every check with `pass|fail` and a concrete `fix` string. | 200 |
| `POST /v1/estimate` | D9 preflight: `{"sources": [...]}` → token counts, projected embedding cost and time per configured provider, BEFORE ingestion. Requires `cost.estimator: true`; with it off the endpoint answers `422 VALIDATION_FAILED` naming the setting rather than estimating zero. | 200 |
| `POST /v1/admin/export` | D11: export documents, chunks, metadata, and the index manifest to a portable archive. Body: `{"out", "collection", "include_vectors"}`. **Synchronous** — unlike ingestion it reads what is already indexed and writes a file, so there is no queue to wait behind and no partial state to poll. Returns the row counts written. | 200 |
| `POST /v1/admin/import` | D11: import a previously exported archive. Body: `{"archive", "target_collection", "reembed"}`. Checksums, manifest counts, and referential integrity are all verified **before anything is written**, so a refused archive leaves the target untouched. A vector copy is refused with `CONFLICT` unless the archive carries vectors and its model, model version, and dimensions all match; `reembed: true` is always legal. | 200 |

## Traces & replay

| Method + path | Purpose | Success |
|---|---|---|
| `GET /v1/traces` | D8: recent trace ids, newest first. A 32-hex trace id is unreachable without a way to list them, so listing is what makes stored traces usable at all. | 200 |
| `GET /v1/traces/{trace_id}` | D8: full persisted trace of a past query — retrieved chunks, leg scores, fused/reranked ranks, prompt, response, timings. | 200 / 404 |
| `POST /v1/replay` | `{"trace_id", "config_overrides": {...}}` → re-executes the past query under the candidate config; returns side-by-side diff of retrieval sets and answers. | 200 / 404 |

## Status code summary

`200` OK · `201` created · `202` accepted (async job) · `204` no content · `401`/`403` auth · `402` budget · `404` not found · `409` conflict · `413` too large · `422` validation · `429` rate/queue (`Retry-After`) · `500` internal (problem body mandatory) · `503` dependency/provider unavailable (`Retry-After` when known).
