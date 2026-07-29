# observability.md — Dashboard, Metrics, Tracing, Auto-Provisioning

> **Assumption (control vs dashboard).** Restating the reconciliation from [scope.md](scope.md): the **control plane is exclusively the REST API, the CLI, and the Python library** — no graphical interface can create, modify, or drive the RAG. The **observability dashboard is a separate, optional, self-hosted web GUI for inspection only** (Langfuse-like). It renders data and exposes **zero** control endpoints. "Terminal only, no GUI" (control) and "give a dashboard" (observability) are therefore both true, and this is treated as a project assumption.

## 1. Self-hosted observability dashboard (`observability.dashboard: true`)

A read-only web GUI that lets the user inspect:

- **Cache stats** — semantic and embedding cache hit/miss ratios over time, entry counts, invalidation events.
- **Token usage** — prompt/completion tokens per query, per collection, per tenant.
- **Costs** — estimated cost per query and cumulative per provider/tenant.
- **Latencies** — p50/p95 end-to-end and per stage, split **retrieval vs generation**.
- **Complete LLM inputs/outputs history** — every prompt and response persisted by the trace store (D8), browsable and searchable for later review.
- Queue depth, DLQ depth, circuit-breaker states, degraded-mode counts, drift warnings.

Hard rules: toggleable via `observability.dashboard` (default `false`); binds `observability.dashboard_port` (default 8080); **observability-only and never controls the RAG** — it contains no button, form, or endpoint that mutates system state.

## 2. Metrics catalogue

| Metric | Type | Labels | Description |
|---|---|---|---|
| `fasterrag_requests_total` | counter | endpoint, method, status, tenant | Request volume (RED: rate). |
| `fasterrag_errors_total` | counter | endpoint, code, tenant | Error rate by problem `code` (RED: errors). |
| `fasterrag_request_duration_seconds` | histogram | endpoint | End-to-end latency; p50/p95 derived (RED: duration). |
| `fasterrag_stage_duration_seconds` | histogram | stage ∈ parse, chunk, embed, retrieve_dense, retrieve_bm25, fuse, rerank, assemble, generate | Per-stage latency — the retrieval-vs-generation split. |
| `fasterrag_ttft_seconds` | histogram | — | Time to first streamed token. |
| `fasterrag_tokens_total` | counter | kind ∈ prompt, completion; provider, tenant | Token counts. |
| `fasterrag_cost_usd_total` | counter | provider, tenant | Estimated cost per query, accumulated. |
| `fasterrag_cache_events_total` | counter | cache ∈ semantic, embedding; result ∈ hit, miss, invalidated | Cache hit/miss ratio source. |
| `fasterrag_retrieval_quality` | gauge | metric ∈ precision_at_k, recall_at_k, mrr, ndcg | Latest eval-harness scores. |
| `fasterrag_faithfulness` | histogram | — | Grounding/faithfulness score distribution (D5). |
| `fasterrag_ingest_documents_total` | counter | status ∈ indexed, deduplicated, dead_lettered | Ingestion outcomes. |
| `fasterrag_ingest_throughput` | gauge | unit ∈ docs_per_sec, tokens_per_sec | Live ingestion throughput. |
| `fasterrag_queue_depth` | gauge | queue ∈ ingest, chunk | Bounded-queue occupancy. |
| `fasterrag_dlq_depth` | gauge | collection | Dead-letter queue depth. |
| `fasterrag_circuit_state` | gauge | provider ∈ llm, embeddings, vector_db | 0 closed / 1 half-open / 2 open. |
| `fasterrag_degraded_responses_total` | counter | mode ∈ hybrid_only, cache_only, extractive | Degradation-ladder activations (D4). |

## 3. Tracing (OpenTelemetry)

- **OTel spans wrap retrieval and generation**: the retrieval span records source, number of documents, and latency; the generation span records model, temperature, prompt, and response length. Both are correlated by **trace ID**, which also appears in every log line and every API problem/response body.
- **Four RAG trace types**: `retrieval`, `reranker`, `context-assembly`, `generation` — every query produces all four (minus skipped stages), nested under one root span.
- Export via OTLP when `observability.otel: true` (`observability.otel_endpoint`); traces are additionally persisted locally by the trace store (D8) regardless of OTel, powering the dashboard and `fasterrag replay`.

## 4. Langfuse auto-provisioning (`observability.langfuse: true`)

Flipping the toggle causes fasterRag to read config, **auto-install**, perform **all required configuration** on the user's system, and return a **running URL** — with **no application-code changes at any point** (pure config-driven provisioning). Provisioning is idempotent and doctor-gated (D10).

### What gets installed

Self-hosted **Langfuse v3** runs via **Docker Compose** as a multi-container stack. Per Langfuse's official self-hosting docs (`langfuse.com/self-hosting`), the required components are:

| Component | Role | Port(s) |
|---|---|---|
| **Langfuse Web** | UI + API — **publicly exposed**; this is the returned URL | **3000** |
| **Langfuse Worker** | Async event processing | 3030 |
| **PostgreSQL** | Transactional store | 5432 |
| **ClickHouse** | "Stores traces, observations, and scores"; Langfuse v3 supports ClickHouse ≥ 24.3 | 8123 (HTTP), 9000 (native) |
| **Redis/Valkey** | Queue/cache — the shipped compose file uses stock `redis:7` | 6379 |
| **S3/Blob Storage** | Event/media storage — the shipped compose file uses a Chainguard MinIO image | 9090 (API), 9091 (console) |

**The running URL returned is `http://<host>:3000`.**

### Secrets (generated once, stored in `.env`, never in YAML)

| Secret | Generation (per Langfuse's own `.env` guidance) |
|---|---|
| `NEXTAUTH_SECRET` | `openssl rand -base64 32` |
| `SALT` | random; generated by the provisioner |
| `ENCRYPTION_KEY` | `openssl rand -hex 32` |

Critical handling rules (per Langfuse Discussion #1902):

- `SALT`, `ENCRYPTION_KEY`, and `NEXTAUTH_SECRET` **must be preserved across restarts/upgrades** — changing them invalidates existing passwords and API keys. The provisioner therefore writes them once and never regenerates on re-run.
- `SALT` and `ENCRYPTION_KEY` **must be identical between the web and worker containers** — the provisioner injects the same values into both.

### Headless bootstrap (no manual UI clicks)

The provisioner uses the `LANGFUSE_INIT_*` environment variables to auto-create an org, project, API keys, and user on first boot:

- `LANGFUSE_INIT_ORG_ID`, `LANGFUSE_INIT_ORG_NAME`
- `LANGFUSE_INIT_PROJECT_ID`, `LANGFUSE_INIT_PROJECT_NAME`, `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`, `LANGFUSE_INIT_PROJECT_SECRET_KEY`
- `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_PASSWORD`, `LANGFUSE_INIT_USER_NAME`

Caveats the provisioner encodes:

- **Do not double-quote these values in Docker Compose** (per Langfuse's headless-initialization docs / GitHub issue #3398) — quoted values are taken literally and bootstrap silently misbehaves.
- **Dependency tree**: a project/user cannot be created without an org — `ORG_*` must be present for `PROJECT_*`/`USER_*` to take effect.

The generated project public/secret keys are what fasterRag's tracing exporter uses; they land in `.env`, referenced by env-var name.

### Flow summary

`langfuse: true` → doctor preflight (ports 3000/3030/5432/6379/8123/9000/9090/9091 free, Docker running, disk) → generate compose + `.env` secrets (once) → `docker compose up -d` → health-check until ready → **return `http://<host>:3000`** → export traces. Re-running converges (no reinstall, secrets untouched). **Zero application-code changes.**

## 5. Grafana auto-provisioning (`observability.grafana: true`) — provisioning-as-code

No manual clicks; everything is version-controlled config:

- YAML manifests are written under Grafana's provisioning directory (default **`/etc/grafana/provisioning/`**, mounted into the container) with **`datasources/`** and **`dashboards/`** subdirectories, read by Grafana **at startup**.
- Datasource manifests set **`editable: false`**; dashboard providers set **`allowUiUpdates: false`** — the UI becomes read-only for provisioned resources, enforcing GitOps (the files in the repo are the source of truth).
- Dashboards are **JSON** files; the provider polls/reloads them on an interval (e.g. **`updateIntervalSeconds: 30`**) so file changes apply **without restart**.
- fasterRag ships dashboard JSON for: query latency (p50/p95 per stage), ingestion throughput, cache hit ratio, queue/DLQ depth, circuit-breaker state, cost per query.
- The provisioned datasource points at fasterRag's metrics endpoint (Prometheus format).

As with Langfuse: doctor-gated, idempotent, and **no application-code changes at toggle time**.

## 6. Reliability observability (summary; doctrine in [reliability.md](reliability.md))

- RED metrics per endpoint (rate, errors, duration).
- Per-stage spans (parse, chunk, embed, retrieve, rerank, generate) correlated by trace id.
- Exported gauges for queue depth, DLQ depth, circuit-breaker state, cache hit rate.
- `/healthz` (liveness) is distinct from `/readyz` (dependencies actually checked) — see [api-reference.md](api-reference.md).
