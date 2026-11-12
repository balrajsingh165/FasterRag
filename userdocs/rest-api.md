# REST API

Start the server with `fasterrag serve` (install the `server` extra first). It listens on
`http://127.0.0.1:8000` by default; change that with `app.host` / `app.port` or `--host` /
`--port`. The OpenAPI schema is at `/openapi.json`.

## Endpoints

| Method and path | What it does |
|---|---|
| `POST /v1/ingest` | start an ingestion job; returns `202` with a `job_id` |
| `GET /v1/ingest/{job_id}` | job status and counts |
| `GET /v1/ingest/{job_id}/documents?status=dead_lettered` | per-document results, optionally filtered |
| `POST /v1/ingest/{job_id}/retry-dlq` | retry the documents that failed |
| `POST /v1/query` | answer a question, as JSON or as a stream |
| `POST /v1/retrieve` | return the matching chunks without generating an answer |
| `GET /v1/collections` · `POST /v1/collections` | list or create collections |
| `GET /v1/collections/{name}` · `DELETE /v1/collections/{name}` | inspect or drop one |
| `GET /v1/collections/{name}/documents` | documents in a collection |
| `DELETE /v1/collections/{name}/documents/{document_id}` | remove one document |
| `GET /v1/traces` · `GET /v1/traces/{trace_id}` | stored query traces |
| `POST /v1/replay` | rerun a past query, optionally with config overrides |
| `POST /v1/estimate` | token count and cost before ingesting |
| `GET /healthz` · `GET /readyz` · `GET /metrics` | liveness, readiness, Prometheus metrics |

## Ingest

```bash
curl -X POST localhost:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"sources":[{"type":"path","value":"/data/docs"}], "collection":"default"}'
```

A source is `{"type": "path" | "url" | "inline", "value": "..."}`. Optional fields:
`collection`, `metadata` (attached to every chunk). Send an `Idempotency-Key` header to make a
retry safe: the same key returns the original response.

## Query

```bash
curl -X POST localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are the payment terms?","top_k":5,"filters":{"department":"legal"},"stream":false}'
```

The JSON response carries `answer`, `citations` (`source`, `page`, `score`), `usage`,
`timings_ms`, `trace_id`, and `degraded` / `mode`, which tell you when a component was down
and a reduced answer was served.

With `"stream": true` the response is Server-Sent Events in this order: `meta`, `token`
(repeated), `citations`, `usage`, `done`. If the stream ends without `done`, the answer is
incomplete; an `error` event carries the reason.

`POST /v1/retrieve` takes the same body and returns the ranked chunks only, for when you run
your own LLM step.

## Authentication

Off by default. To turn it on, set `security.auth: true` and put the key in `.env`:

```bash
FASTERRAG_API_KEY=my-secret-key
```

Then send it on every request:

```bash
curl -H 'Authorization: Bearer my-secret-key' ...
```

Several keys are separated by `;`, and a key can be limited to scopes (`ingest`, `query`,
`collections`, `admin`) as `key:query,ingest`. With auth on, set
`ingestion.allowed_source_roots` too.

## Errors

Every error is an RFC 9457 `application/problem+json` document:

```json
{"type": "...", "title": "Not Found", "status": 404, "detail": "no collection named 'docs'",
 "code": "NOT_FOUND", "trace_id": "…", "retryable": false}
```

Branch on `code`, not on the message. Retry only when `retryable` is `true`, honouring any
`Retry-After` header. Quote the `trace_id` when reporting a problem.

## Full reference

Request and response schemas, every error code and every status:
[api-reference.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/api-reference.md).
