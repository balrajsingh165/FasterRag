# api-reference.md — REST API

The REST API is one half of the control plane (the other is the [CLI](cli-reference.md)). All endpoints are versioned under `/v1`. All request/response bodies are JSON except SSE streams and problem responses (`application/problem+json`).

- Authentication: `Authorization: Bearer <api-key>` when `security.auth: true`. Keys carry scopes (`ingest`, `query`, `collections`, `admin`). See [security.md](security.md).
- Multi-tenancy: `X-Tenant-ID` header (configurable) when `security.multi_tenancy: true`.
- Idempotency: every mutating endpoint accepts an optional `Idempotency-Key` header (TASK-0318 ✅ 2026-11-08). See [Idempotency](#idempotency) below for its scope, lifetime, conflict codes and replay signal.
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
| `IDEMPOTENCY_KEY_REUSED` | 422 | — | The `Idempotency-Key` was already used for a *different* request (method, path, query string or body differ). Non-retryable: resend under a new key. |
| `IDEMPOTENCY_IN_PROGRESS` | 409 | — | A request with this `Idempotency-Key` is still in flight. **Retryable**, with `Retry-After: 1` — the original response is replayable once it lands. |
| `PAYLOAD_TOO_LARGE` | 413 | — | Body exceeds `security.max_request_mb` / `ingestion.max_document_mb`. |
| `RATE_LIMITED` | 429 | — | Per-key rate limit exceeded (`Retry-After` set). |
| `QUEUE_FULL` | 429 | `IngestionError` | Bounded queue overflow (`Retry-After` set). |
| `BUDGET_EXCEEDED` | 402 | — | Per-query or per-tenant token budget exhausted (D9), raised by the runtime cost governor before the provider is called (TASK-0242 ✅). **Non-retryable**, which is deliberate: the degradation ladder absorbs retryable failures into an extractive answer, so a retryable budget error would serve a degraded answer instead of reporting the cap. The check weighs the prompt plus `llm.max_tokens`, the ceiling the call can reach. |
| `PARSE_FAILED` | 422 | `ParseError` | Document unparseable; also the DLQ reason code. |
| `PARSE_TIMEOUT` | 422 | `ParseError` | A document was still being parsed after `ingestion.parse_timeout_seconds`. A DLQ reason code only: the job carries on without the document, so no request is answered with it. Non-retryable as it stands — the same document will run out the same limit — until the limit is raised or the source is fixed. |
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
| `NOT_READY` | 503 | — | A readiness dependency check failed: returned by `/readyz`, with a `dependencies[]` extension member naming each check and its state. Also returned, retryable, by `POST /v1/ingest` and `POST /v1/ingest/{job_id}/retry-dlq` while the server is shutting down and has stopped starting jobs (see [`POST /v1/ingest`](#post-v1ingest)); that refusal carries no `dependencies[]`. |
| `INTERNAL` | 500 | `FasterRagError` | Anything unclassified. Always carries `trace_id`. Never returned without a problem body. |

---

## Idempotency

Shipped by TASK-0318 (✅ 2026-11-08). Implemented in `api/idempotency.py` (transport) over `services/idempotency.py` (the store and every decision), covered by `tests/unit/api/test_idempotency.py`.

**The header.** Any `POST`, `PUT`, `PATCH` or `DELETE` may carry `Idempotency-Key: <1–255 printable ASCII characters>`. It is **optional**: a request without it is passed straight through and touches no store, so nothing changes for a client that does not use it. A key outside those bounds is refused `422 VALIDATION_FAILED`, and the offending value is never echoed back.

**Scope is `(tenant, key)`.** A key names *one request*, not one endpoint's namespace. The request's method, path, query string and body are fingerprinted and stored beside the response, so reusing a key for anything else — a different body, or the same key on another endpoint — is `422 IDEMPOTENCY_KEY_REUSED` rather than a silent wrong answer. The tenant (as `security.multi_tenancy` resolved it) is part of the record path, so two tenants that both pick `retry-1` never meet.

**Lifetime, and the kill switch.** A completed record is replayable for `reliability.idempotency.ttl_seconds` (**24 hours** by default); an unfinished reservation is held for `reliability.idempotency.in_flight_lease_seconds` (**15 minutes**), so a worker that died mid-request frees its key in minutes instead of holding it for a day. The loader refuses a lease longer than the lifetime. Expiry of a *key* is lazy — the next request to reach an expired record takes it over — and until TASK-0365 that was the only reclamation there was, so a record whose key never returned was never removed and a client using a fresh UUID per request left a file behind on every call. **Files are now reclaimed too** (✅ 2026-11-11): one pass runs beside a request, at most once per `in_flight_lease_seconds` per host, and removes records that have been expired for more than a minute along with any temporary file a crash left behind. The directory is therefore bounded by one lifetime of keyed traffic rather than by all of it, and `ttl_seconds` now bounds disk as well as replayability. See [Reclamation](#reclamation) below. Both became configurable with TASK-0356, together with `reliability.idempotency.enabled`, which **defaults to `true`**: it is a kill switch for a misbehaving store, and defaulting it off would make this section untrue for every deployment that never read the setting. With it `false`, a keyed request is passed straight through — nothing recorded, nothing replayed, nothing refused.

**Replay signal.** A replayed response is the original one byte for byte — same status, same body — plus `Idempotent-Replayed: true`. Every keyed response, replayed or not, echoes `Idempotency-Key`. A client can therefore tell "this already happened" from "this just happened", which a bare repeated `201` cannot.

**Only 2xx is remembered.** A refusal does not pin a key: a caller that hit a `503` must be able to retry the same key and get through, and a caller that hit a `422` will fix the body — which the fingerprint would then refuse if the failure had been stored. Error responses release the key. A success larger than 256 KB is also not stored; it is returned normally and its key is released.

**Concurrent duplicates.** Two copies of one request arriving before either finishes is the ordinary shape of a client retry, and it is the case a check-then-write implementation returns two `200`s for. The reservation is taken with `O_CREAT | O_EXCL`, so exactly one caller wins; the loser is refused `409 IDEMPOTENCY_IN_PROGRESS` with `Retry-After: 1` and executes nothing.

**Covered endpoints.** Everything mutating: `POST /v1/ingest`, `POST /v1/ingest/{job_id}/retry-dlq`, `POST /v1/collections`, `DELETE /v1/collections/{name}`, `DELETE /v1/collections/{name}/documents/{document_id}`, `POST /v1/admin/doctor/fix`, `POST /v1/admin/provision/{tool}`, `DELETE /v1/admin/provision/{tool}`, `POST /v1/admin/export`, `POST /v1/admin/import`. A mutating router added later is covered the moment it is mounted, because this is middleware rather than a per-route dependency.

**Deliberately not covered**, because they are `POST` for the body and not for an effect: `/v1/query`, `/v1/retrieve`, `/v1/estimate`, `/v1/replay`. A key sent to one of them is ignored rather than refused, so a client that sets the header on every request keeps working. `/v1/query` can also be a `text/event-stream`, which cannot be captured and handed back.

**Storage, and its limit.** Records are files under the ingestion journal root (`.fasterrag/journal/idempotency/`), *not* process memory — production runs Gunicorn with several Uvicorn workers, and a per-process map would make a key hold only for the worker that served the first request. A shared filesystem therefore covers every worker on one host. **Several hosts behind a load balancer need that directory on shared storage**; without it, idempotency is per host. This is a documented limit, not a solved problem (TASK-0357).

**The backend is substitutable.** `IdempotencyStore` is an abstract interface — `enabled`, `begin`, `commit`, `release`, `close` — and `FileIdempotencyStore` is the one implementation that ships, in the shape `security.rate_limit_backend` already uses for the other control with a one-host window. The interface is `async` and a reservation carries no path, so a shared backend is a second class and a branch in `create_idempotency_store`, with no change to the middleware or to any behaviour described above. Which backend to add, and whether to require shared storage instead, is TASK-0357's open decision; nothing in this section changes either way.

### Reclamation

Shipped by TASK-0365 (✅ 2026-11-11), in `FileIdempotencyStore.reclaim`, covered by `tests/unit/services/test_idempotency_reclaim.py`, measured by `scripts/measure_idempotency_reclaim.py` and recorded as BENCH-0004 in [benchmarks.md](benchmarks.md).

**What is removed.** A record that has been expired for more than a minute of grace — `ttl_seconds` past its completion, or `in_flight_lease_seconds` past a reservation nobody finished; a record that fails its checksum, which `claim` already treats as free; and a `*.tmp` file older than that same minute, which can only be a write a crash interrupted (`write_record` holds one for microseconds). Nothing else. Empty tenant directories are deliberately **left in place**: removing one races the `mkdir` every write performs, and turning a cleanup into an occasional `500` is a bad trade for a few inodes.

**When it runs.** Beside a request, never inside it: `begin` starts the pass as a task and does not wait for it, and every step of the pass runs in a worker thread, so neither the caller nor the event loop waits for a directory walk. At most one pass per `in_flight_lease_seconds` per host, and at most one at a time per process. The first check is an in-process timestamp, so a request that is not due performs no filesystem operation at all. **Why not inside the request**: a pass costs one cold file open per record removed — 18.3 ms each on the measured hardware, 36.6 s to clear 2,000 records — while a pass that finds nothing costs 21.8 ms over 8,200 records. Charging either to a caller was the alternative design and it is the one the numbers rejected (BENCH-0004). A concurrent claim pays +0.07 ms at p50 and +0.27 ms at p95 while a pass runs, with a 16 ms worst case in 200 samples.

**Why it is safe with several workers.** The right to sweep is taken as one more record under `O_CREAT | O_EXCL` — `.fasterrag/journal/idempotency/.sweep.json` — so one process of N walks the directory per interval instead of N. A lease whose holder died is taken over once its interval has passed, exactly as a record is. Two processes that find it stale in the same instant can both sweep; that costs a duplicated pass and nothing else, because every removal re-reads the envelope and tolerates a file another pass already took. The same applies when a pass outlives its own lease under extreme volume: a second worker may start one beside it, both converge, and neither removes a record the other needed.

**The grace margin, and the race it does not close.** `claim` takes an expired record over by rewriting it in place. A sweep that unlinked a record the instant it expired could therefore delete the fresh reservation a concurrent takeover had just written, freeing a key whose request is still running — a duplicate execution. The margin makes only records untouched for a full minute past their lifetime eligible, and the envelope is re-read immediately before the unlink, which narrows the window to a read and a syscall. It does not eliminate it: no filesystem offers unlink-if-unchanged. The residual exposure is one duplicate execution of a key abandoned for over a lifetime, and it is recorded here rather than left for someone to find.

**What it does not do.** It is not on the `IdempotencyStore` interface, because a backend that expires keys natively has nothing to sweep; it is a `FileIdempotencyStore` method, and a Redis backend (TASK-0357) would implement `close` and nothing else from this section. It does not bound the directory on its own — it bounds it to roughly one `ttl_seconds` window of keyed traffic plus one interval — so an operator wanting a smaller directory lowers `ttl_seconds`, which now bounds disk as well as replayability. And a pass that cannot remove a file logs one warning for the whole pass with a count and the first failure, never one line per record.

### Deprecated: the `idempotency_key` body field

`POST /v1/ingest` has always taken an `idempotency_key` **body field**, resolved by `Journal.find_by_idempotency_key`. It **still works and is not being broken now**; it is deprecated and removed in 1.0.

- Both mechanisms may be sent together. The header is resolved first, in middleware, so a header replay never reaches the route and the body field never sees it.
- A request that uses the body field gets `Deprecation: true` and `Link: <...>; rel="deprecation"` on its response, and the field is marked `deprecated` in `/openapi.json`.
- The two are not identical: the body field dedupes only the *job*, re-running the keyword-leg preflight and returning a fresh `202` each time, and it compares nothing about the rest of the request. The header replays the original response and refuses a changed one. Prefer the header.
- Until 2026-11-08 the body field's lookup was **not tenant-scoped**: one tenant replaying another tenant's key was handed that tenant's job id and its own ingest was silently dropped. Fixed with the rest of TASK-0318.
- **It costs a full scan of the job journal, measured.** `Journal.find_by_idempotency_key` globs `jobs/*.json` and opens each until it matches, so the cost is linear in how many ingest jobs the journal has ever held: **633 ms p50 at 10,000 jobs**, ~63 µs per job record (BENCH-0004, non-citable hardware). The header path is O(1) — one file addressed by `sha256(key)`. The scan also runs on a *new* key, which is the common case, because `create_job` resolves the key before creating the job. **This will not be indexed; it retires with the field at 1.0** (TASK-0358), because an index is a second source of truth whose disagreement modes are the very failures the field exists to prevent — a miss starts a second ingest of the same corpus, a stale hit returns a job id that no longer exists. If your journal is large and you still use the field, the fix available today is to send the header instead. **The scan no longer runs on the event loop** (TASK-0399): `Journal` offers every filesystem operation as an `async` method over a `*_blocking` twin that executes in a worker thread, so a request that sends the field waits for its own scan and no other request on that worker waits with it — measured as a drop from a multi-second p95 stall on the loop to the idle heartbeat floor (BENCH-0006).

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
| `idempotency_key` | str | no | **Deprecated, removed in 1.0** — send the [`Idempotency-Key` header](#idempotency) instead. Still honoured: replaying a key returns the original job and enqueues nothing, scoped to the caller's tenant — except that replaying the key of a job settled `failed` / `INTERRUPTED` runs that job again from its checkpoint (see [`GET /v1/ingest/{job_id}`](#get-v1ingestjob_id)). A request that uses it gets `Deprecation: true` on the response. |

Responses: `202 Accepted` `{"job_id": "job_01J8Z3W7", "status": "queued"}` · `409 CONFLICT` — the target collection was indexed without the keyword leg `retrieval.hybrid` asks for, and a sparse vector cannot be attached to an existing collection; the detail names the rebuild command and the opt-out (see [troubleshooting.md](troubleshooting.md)). Refused before a job id is issued, so nothing is written · `422` · `429 QUEUE_FULL` · `503 NOT_READY` (retryable) — the server is shutting down and has stopped starting jobs; nothing was recorded, so the same request, with the same `Idempotency-Key`, is safe to send again.

**Shutdown** (TASK-0436). When the process is asked to stop it refuses new jobs, waits `reliability.shutdown_grace_seconds` for the ones it is running, cancels what remains — each is settled `failed` with `reason_code: "INTERRUPTED"`, its checkpoint kept — and only then closes its adapters. What a client sees depends on the server in front of the application, and this was observed rather than assumed: under uvicorn 0.52.1 (measured 2026-11-12, Windows) the listening socket is closed and idle keep-alive connections are dropped *before* the application's shutdown runs, so a request sent during the drain fails to **connect** and the application sees no request at all. The `503` is therefore what a client gets from a server that still routes requests to an application that is draining; behind uvicorn, treat a refused connection during a deploy the same way — retry.

### `GET /v1/ingest/{job_id}`

Job status: `{"job_id", "status": "queued|running|completed|failed|partial", "reason_code", "counts": {"total", "parsed", "chunked", "embedded", "indexed", "deduplicated", "skipped", "dead_lettered"}, "started_at", "finished_at"}`. `404` if unknown. `skipped` counts documents that parsed without error and held nothing to index; they are not failures, so a job whose only non-indexed documents are skipped is `completed`, not `partial`.

`reason_code` is `null` except on a job the process stopped working on before it finished, which is settled `failed` with `reason_code: "INTERRUPTED"` (TASK-0436). That pair means *nothing was wrong with the documents*: the job was cancelled, and its checkpoint is kept. `INTERRUPTED` is a reason on a job, not an entry in the error-code table — no request fails with it.

**Running an interrupted job again.** Nothing resumes a job on its own: a starting server settles what an earlier process left `running` and stops there. To continue one from its checkpoint, either run `fasterrag ingest --resume <job_id>` ([cli-reference.md](cli-reference.md)), or send the original `POST /v1/ingest` again with the same `idempotency_key` **body field** — the response is `202` with the same `job_id` and `"status": "queued"`, and the run skips every document the checkpoint already covers. Replaying the key of a job that is not `INTERRUPTED` still returns the job and runs nothing.

**The limit, stated plainly:** this works through the body field only. A request keyed with the `Idempotency-Key` *header* is answered by the idempotency middleware from the stored `202` before it reaches the route, so replaying that header returns the original response with `Idempotent-Replayed: true` and starts nothing, however the job ended. A client that keys ingestion with the header alone resumes with the CLI, or submits the sources again under a new key and lets deduplication skip what is already indexed. Whether the header should resume is a decision about the middleware and has not been made.

### `GET /v1/ingest/{job_id}/documents?status=dead_lettered`

Per-document status (D3), filterable. Each entry: `{"document_id", "source", "status", "reason_code", "attempts", "content_hash"}`. Reason codes reuse the error `code` table (e.g. `PARSE_FAILED`). `status` is one of `indexed`, `deduplicated`, `skipped` or `dead_lettered`. A `skipped` entry carries the one reason code that is not an error code, `EMPTY_DOCUMENT`: a zero-byte or whitespace-only file, or any document whose parse produced no chunk. It is never written to the vector database, never remembered for deduplication, and `retry-dlq` does not pick it up — `?status=skipped` lists them.

### `POST /v1/ingest/{job_id}/retry-dlq`

Re-enqueues dead-lettered documents of the job. `202`, or `409 CONFLICT` under the same keyword-leg preflight as `POST /v1/ingest`.

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
| `filters` | object | no | Metadata filter expression, pushed down to both retrieval legs. With `security.tenant_scoped_retrieval` on, a `tenant` key here is refused with `403 TENANT_FORBIDDEN`: the tenant a retrieval runs as comes from the credential and is injected below this expression ([security.md](security.md) §3). |
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
event: usage      data: {"usage": {"prompt_tokens": 3211, "completion_tokens": 187, "estimated_cost_usd": null},
                         "timings_ms": {...}, "faithfulness": 0.93, "template_version": "...",
                         "mode": "full", "degraded": false, "truncated": false, "evidence_dropped": 0}
event: done       data: {}
```

**A stream reports everything the JSON body reports.** Both shapes are produced by one pipeline (TASK-0438), so `usage` closes the stream with the same `mode`, `degraded`, `truncated` and `evidence_dropped` the non-streamed body carries, `meta` always carries the `cache` member, a streamed query increments the same metrics (plus `fasterrag_ttft_seconds`, which only a stream can have), and it stores the same trace — including the trace of a withheld answer, which a streamed refusal did not store before 2026-11-12. Two members are **estimates on a stream and provider-reported otherwise**, because a provider stream carries text only: `prompt_tokens` / `completion_tokens` are counted locally (and `estimated_cost_usd` is priced from those), and `truncated` is inferred from the local completion count reaching `llm.max_tokens` rather than read from the provider's finish reason. An answer that ended on its own just under the ceiling can therefore be flagged, and one the provider cut can be missed when the local count runs low.

**A query refused before its first event is an ordinary problem response, not a stream.** Everything that can refuse a query outright — retrieval failing (`503 RETRIEVAL_FAILED`), the token budget (`402 BUDGET_EXCEEDED`) — happens before `meta` is produced, and the server produces `meta` before it sends the status line. Such a request gets the same status and `application/problem+json` body as `stream: false`, with no SSE frame at all. Before 2026-11-12 the budget check ran after `meta`, so a streamed request over its budget received `200` and then an `error` event (TASK-0438).

**A provider that is down degrades a stream the way it degrades a JSON body** (D4, TASK-0438). When generation fails with a *retryable* error — timeout, rate limit, 5xx, an open circuit breaker — **before the first `token` has been sent**, the stream serves the `extractive` rung instead of failing: a second `meta` announcing the mode, the retrieved passages as one `token`, then `citations`, `usage` and `done`.

```
event: meta       data: {"trace_id": "...", "mode": "full", "degraded": false, "cache": {...}}
event: meta       data: {"trace_id": "...", "mode": "extractive", "degraded": true}
event: token      data: {"text": "<the retrieved passages>"}
event: citations  data: {"citations": [...]}
event: usage      data: {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": null}, ..., "mode": "extractive", "degraded": true, ...}
event: done       data: {}
```

`meta` can therefore arrive twice, and the later one wins: the first was sent before the provider had been called. A client that reads only the first still gets the truth from `usage`. `fasterrag_degraded_responses_total{mode="extractive"}` is incremented exactly as for a non-streamed query. Before 2026-11-12 this case ended the stream with `error` and no `done`.

**The boundary is the first `token` sent.** After it, the caller has been shown text the retrieved passages cannot replace — appended, they would read as the answer's continuation — so there is no fallback: `event: error` carrying `code`, `detail`, `trace_id`, `retryable` and the `status` the problem would have had, then the stream closes. The same `error` event ends a stream on any *non-retryable* failure after `meta` (a rejected credential, an unknown model): the status is already `200` and cannot change. Clients must treat a missing `done` as an incomplete answer. With `generation.grounded_or_refuse: true` no `token` is sent until the answer has been graded, so there a retryable failure at any point of generation is still before the boundary and degrades.

Grounded-or-refuse while streaming: a token cannot be unsaid once sent, so with `generation.grounded_or_refuse: true` no `token` event is emitted until the whole answer has been graded — time-to-first-token is traded for the guarantee the flag exists to provide, and the flag defaults to `false` so the trade is only made on request. A withheld answer replaces the `token`, `citations`, and `usage` events with one `insufficient_evidence` event carrying the same body as the non-streamed refusal, followed by `done`:

```
event: meta                   data: {"trace_id": "...", "mode": "full", "degraded": false, "cache": {...}}
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

**Tenancy.** Both `/v1/query` and `/v1/retrieve` address the collection the body names, unprefixed — neither applies the `<tenant>__` scoping `/v1/collections` uses. Retrieval is filtered by tenant only when `security.tenant_scoped_retrieval` is on, and `403 TENANT_FORBIDDEN` is then also possible on both endpoints. See [security.md](security.md) §3 for what is enforced today and what the flag changes (TASK-0339). The pre-enable `tenant` backfill is **CLI-only** (`fasterrag index tenants`, `fasterrag index backfill-tenant`) and has no endpoint: it rewrites every point in a collection and chooses the tenant from an operator's argument rather than from a credential, which is not a decision a request-scoped surface should be able to make (TASK-0377).

## Collections

| Method + path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /v1/collections` | List collections (name, vectors count, embedding model+version, created_at, alias target). | 200 | — |
| `POST /v1/collections` | Create: `{"name", "distance", "shard_number", "replication_factor"}`. | 201 | 409 `CONFLICT`, 422 |
| `GET /v1/collections/{name}` | Detail incl. `index.lock` summary and drift status (D1). | 200 | 404 |
| `PATCH /v1/collections/{name}` | **Not yet implemented.** Mutable settings only (metadata schema, description). | 200 | 404, 422 |
| `DELETE /v1/collections/{name}` | Drop collection (requires `admin` scope; refuses while it is an active alias target unless `?force=true`). | 204 | 404, 409 |
| `DELETE /v1/collections/{name}/documents/{document_id}` | Delete one document: every chunk carrying that `document_id`, its dedup record, its lockfile entry, and the semantic cache. See [Deleting a document](#deleting-a-document). | 200 | 404, 422 |
| `POST /v1/collections/{name}/reindex` | **Not yet implemented over REST** — shipped on the CLI as `fasterrag index reindex`. D2 zero-downtime reindex: blue/green build → eval gate → alias swap. Returns a job id. | 202 | 404, 409 |
| `POST /v1/collections/{name}/rollback` | **Not yet implemented over REST** — shipped on the CLI as `fasterrag index rollback`. Alias flip back to the retained previous collection (within retention window). | 200 | 404, 409 |

### Deleting a document

`DELETE /v1/collections/{name}/documents/{document_id}` — `collections` scope. The `document_id` is the `d_…` id `GET /v1/ingest/{job_id}/documents` reports for each source.

```json
{
  "collection": "docs",
  "document_id": "d_0d06505c1a084fbc",
  "known": true,
  "deleted_at": "2026-11-12T02:14:07.512339+00:00"
}
```

What it does, in order: writes a tombstone to the ingestion journal so the document's content hash stops counting as a duplicate and the same file can be ingested again; deletes every chunk whose `document_id` payload matches, by one filter pushed down to the vector database; removes the document from the collection's `index.lock`; invalidates the semantic response cache. A failure after the tombstone leaves chunks the journal has already forgotten — the request returns the error, and repeating it removes them.

**It is idempotent, and it does not return 404 for a document that is not there.** `known` reports whether the journal held a live record of the document in that collection. Deleting a document nobody ingested, or deleting the same one twice, is `200` with `"known": false`: the vector database is still asked to delete, because the journal is not the index — with `ingestion.dedup: false`, or after an archive import, documents are indexed that the journal has never heard of — and no backend reports whether a filter matched anything. The response therefore never states how many chunks were removed. `404 NOT_FOUND` means the **collection** does not exist; `422 VALIDATION_FAILED` means the document id was blank.

**Tenancy.** The collection is resolved exactly as every other `/v1/collections` route resolves it — `<tenant>__<name>` when `security.multi_tenancy` is on — and the delete additionally matches only chunks whose `tenant` payload tag is the caller's, so knowing another tenant's document id (it is derived from the source path) removes nothing of theirs. `POST /v1/ingest` does not apply that prefix (TASK-0381), so with multi-tenancy on, a document ingested into `docs` lives in the unprefixed `docs` collection and this route, which addresses `<tenant>__docs`, does not reach it; which of the two is correct is the open decision in TASK-0339.

**Re-ingesting instead of deleting.** A source ingested again with different content is updated in place: its chunks are overwritten, and the chunks beyond its new length are deleted once the new ones are written. That needs the journal to know the document, so it applies when `ingestion.dedup` is on (the default).

## Health & readiness

| Method + path | Purpose |
|---|---|
| `GET /healthz` | Liveness: process is up. Always 200 if the process can answer. No dependency checks. |
| `GET /readyz` | Readiness: dependencies actually checked (vector DB `health()`, queue backend, config valid, circuit-breaker states). 200 with `{"status": "ready", "dependencies": [...]}` when every check passes; otherwise 503 with a `NOT_READY` problem body whose `dependencies[]` member lists each check and its state. Checks are registered by the slice that introduces the dependency, so the report always reflects what is actually verifiable. |

## Admin & provisioning

All require the `admin` scope.

| Method + path | Purpose | Success |
|---|---|---|
| `POST /v1/admin/provision/{tool}` | Trigger config-driven provisioning for `qdrant`, `langfuse`, or `grafana` (equivalent to the config toggle path; idempotent; doctor-gated). Returns `{"tool", "status", "url", "detail"}` — e.g. Langfuse returns `http://<host>:3000`. `detail` is where a Grafana run says why it came back `degraded` and a Langfuse run says how many secrets it generated; it never carries a secret value. Any other tool name is refused `404 NOT_FOUND` naming the three that are supported. | 200 |
| `GET /v1/admin/provision/{tool}/status` | Provisioning/health state of the tool. Same body shape. | 200 |
| `DELETE /v1/admin/provision/{tool}` | Stop the tool's managed containers, preserving data volumes and generated secrets — the REST half of `fasterrag provision <tool> --down`. Reversible by re-provisioning, but stopping `qdrant` stops the backend that answers queries. | 200 |
| `GET /v1/admin/doctor` | Machine-readable doctor report (D10): every check with `pass|fail` and a concrete `fix` string. **Read-only, and pinned as read-only by a test**: it never creates the storage volume and never starts a container, however repairable the failure it is reporting. | 200 |
| `POST /v1/admin/doctor/fix` | The repair half of `fasterrag doctor --fix`, calling the same service function: run the checks, apply only the repairs that are idempotent and non-destructive (create the configured named volume; start fasterRag's *own* container, guarded by the `fasterrag.managed=true` label), then run every check again and classify each attempt from that second report as `fixed`, `not fixed` (`failed`), or `needs human` (`manual`). A repair returning without an error never counts as evidence. Takes no body. The response is the `--json` document the CLI emits — `passed`, `checks`, and a `fixes` object carrying `rechecked`, `attempts[]`, `repaired[]`, and `still_failing[]` — and `200` can carry a report that still fails, because the failing checks are the payload exactly as they are on the `GET`. | 200 |
| `POST /v1/estimate` | D9 preflight: `{"sources": [...]}` → token counts, projected embedding cost and time per configured provider, BEFORE ingestion. Requires `cost.estimator: true`; with it off the endpoint answers `422 VALIDATION_FAILED` naming the setting rather than estimating zero. | 200 Sources are confined to `ingestion.allowed_source_roots` exactly as an ingest's path sources are (since 2026-11-11, TASK-0432): an estimate parses the files it is given, so a path outside every root is refused with `VALIDATION_FAILED`, and with `security.auth` on and no roots configured every path is refused. |
| `POST /v1/admin/export` | D11: export documents, chunks, metadata, and the index manifest to a portable archive. Body: `{"out", "collection", "include_vectors"}`. **Synchronous** — unlike ingestion it reads what is already indexed and writes a file, so there is no queue to wait behind and no partial state to poll. Returns the row counts written. The destination is confined to `security.allowed_archive_roots`; a path outside every root, or any path at all while that list is empty and `security.auth` is on, is refused with `422 VALIDATION_FAILED` before anything is written ([security.md](security.md) §5.6). | 200 |
| `POST /v1/admin/import` | D11: import a previously exported archive. Body: `{"archive", "target_collection", "reembed"}`. Checksums, manifest counts, and referential integrity are all verified **before anything is written**, so a refused archive leaves the target untouched. A vector copy is refused with `CONFLICT` unless the archive carries vectors and its model, model version, and dimensions all match; `reembed: true` is always legal. The archive path is confined to `security.allowed_archive_roots` by the same rule the export side applies ([security.md](security.md) §5.6). | 200 |

### Why the doctor repairs are exposed over HTTP

`GET /v1/admin/doctor` is read-only because a `GET` must not mutate, which is correct and is not the question. The question is whether the repairs should be reachable over HTTP at all, and the honest answer from [security.md](security.md) is that they should: the argument for keeping them off the API does not survive contact with what the `admin` scope already grants. Provisioning over REST starts containers, pulls pinned images, publishes ports, and writes provisioner-generated secrets to `.env`; stopping a tool over REST can stop the backend that answers every query. Against that, the two repairs are `docker volume create` on the configured name and starting a container fasterRag itself labelled — strictly less host authority than the provisioning verbs shipped beside them, and reversible in a way that stopping Qdrant is not. A caller who can reach these endpoints already holds the strongest credential the system issues, and declining this one verb would protect nothing while leaving the thesis — REST, CLI, and library as three surfaces over one service layer — false in a visible place. That is the T21 shape the trap registry names, and [rebuild-playbook.md](rebuild-playbook.md) records the rule it produced: wire both surfaces, or file the asymmetry the same day. The route exists as of TASK-0258, and it calls `services.doctor.apply_fixes` — the function the CLI reaches once it has loaded settings — so neither surface can gain or lose a repair without the other.

The argument for the other outcome, recorded because it is not empty: repairs touch Docker on the host rather than data inside fasterRag, and an API key is copied into CI runners, scripts, and `.env` files far more freely than a shell account on the box is handed out, so "who can call an admin endpoint" is a genuinely wider set than "who has a shell". If that argument were accepted it would have to be accepted for provisioning first, since provisioning does more; narrowing only the smaller verb would be security theatre. It was not accepted, and what bounds the exposure instead is the service layer: the repair set is two entries long, the container repair refuses anything that is not a stopped container carrying fasterRag's own label, and everything else — ports, disk, memory, `.env`, `config.yaml` — comes back as `manual` with its reason. The drift gate (`scripts/check_openapi_drift.py`) and the route-scope table both fail if that changes silently in either direction.

## Traces & replay

| Method + path | Purpose | Success |
|---|---|---|
| `GET /v1/traces` | D8: recent trace ids, newest first. A 32-hex trace id is unreachable without a way to list them, so listing is what makes stored traces usable at all. | 200 |
| `GET /v1/traces/{trace_id}` | D8: full persisted trace of a past query — retrieved chunks, leg scores, fused/reranked ranks, prompt, response, timings. | 200 / 404 |
| `POST /v1/replay` | `{"trace_id", "config_overrides": {...}}` → re-executes the past query under the candidate config; returns side-by-side diff of retrieval sets and answers. | 200 / 404 |

## Status code summary

`200` OK · `201` created · `202` accepted (async job) · `204` no content · `401`/`403` auth · `402` budget · `404` not found · `409` conflict · `413` too large · `422` validation · `429` rate/queue (`Retry-After`) · `500` internal (problem body mandatory) · `503` dependency/provider unavailable (`Retry-After` when known).
