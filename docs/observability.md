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

**Shipped** (TASK-0045). Three properties are enforced rather than promised:

- **No write route exists.** A test walks every route the application declares and asserts none carries `POST`, `PUT`, `PATCH`, or `DELETE`. "We only added read endpoints" is a promise; an application that has none is a property, and a live `POST /` returns `405`.
- **Separate application, separate port.** It runs beside `fasterrag serve` rather than mounted into it, so the API and the dashboard can be bound to different interfaces — the dashboard displays prompts, responses, and corpus text, which usually belongs on an internal one.
- **One source of truth for metrics.** The page reads the same registry a Prometheus scrape renders, through `Registry.series`, so the two cannot drift apart.

Every interpolated value is HTML-escaped: a trace carries user-supplied query text and model output directly onto a page the operator trusts.

**Authentication and tenancy come from the API's own middleware**, not a second implementation. With `security.auth: false` the dashboard is open — the single-operator deployment it was built for. With auth on it requires a key carrying the `admin` scope, the same scope `/v1/traces` and `/metrics` need, because it reads the same data. With `security.multi_tenancy` on, both the page and `/api/traces` show only the calling tenant's traces; scoping the page alone would leave the leak one URL away.

It still has no transport security of its own — put it behind the same reverse proxy as the API ([deployment.md](deployment.md)).

## 2. Metrics catalogue

| Metric | Type | Labels | Description |
|---|---|---|---|
| `fasterrag_requests_total` | counter | endpoint, method, status, tenant | Request volume (RED: rate). |
| `fasterrag_errors_total` | counter | endpoint, code, tenant | Error rate by problem `code` (RED: errors). |
| `fasterrag_request_duration_seconds` | histogram | endpoint | End-to-end latency; p50/p95 derived (RED: duration). |
| `fasterrag_stage_duration_seconds` | histogram | stage ∈ parse, chunk, embed, retrieve_dense, retrieve_bm25, fuse, rerank, assemble, generate | Per-stage latency — the retrieval-vs-generation split. |
| `fasterrag_ttft_seconds` | histogram | — | Time to first streamed token. |
| `fasterrag_tokens_total` | counter | kind ∈ prompt, completion; provider, tenant | Token counts. |
| `fasterrag_cost_usd_total` | counter | provider, tenant | Estimated cost per query, accumulated. List prices, dated and sourced — not a measurement. |
| `fasterrag_unpriced_tokens_total` | counter | provider, model | Tokens spent on a model with no recorded list price, and therefore **absent from** `fasterrag_cost_usd_total`. A non-zero value means the cost figure understates real spend; without this counter that gap is invisible. |
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
- Export via OTLP when `observability.otel: true` (`observability.otel_endpoint`); traces are additionally persisted locally by the trace store (D8) regardless of OTel, powering the dashboard and `fasterrag replay`. Needs the optional extra: `pip install fasterrag[otel]`. If the SDK is absent the toggle logs a warning and queries keep serving — refusing to answer because a trace backend is unavailable inverts the dependency between the system and the thing watching it.
- **The trace id is preserved end to end.** fasterRag mints a 32-hex id, which is exactly the OpenTelemetry trace-id shape, so the id in a `problem+json` error body, in the logs, in `GET /v1/traces/{id}`, and in your trace viewer's search box are all the same id. Span ids are derived from the trace id and stage name, so a retried export is one trace in the backend rather than two overlapping copies.
- Export is fire-and-forget on the query path: `store` runs after the answer is ready, and an unreachable collector costs the record and nothing else.
- OTLP export and Langfuse export can both be on. They answer different questions — Langfuse "what did the model see and say", OTLP "where did the time go across the whole system" — and a deployment that wants both should not have to choose.
- **Metrics ship too.** The same toggle and endpoint push the whole catalogue on an interval (60s), so an OTLP-native stack sees fasterRag's counters as well as its traces. `/metrics` remains the Prometheus scrape endpoint and both views read one registry — a renderer that parsed another renderer's text would be a lossy copy that eventually reports a number neither of them got wrong.
- One caveat when comparing the two by eye: a `/metrics` scrape cannot count the request that is serving it, so a scrape taken during traffic reads one request behind an OTLP snapshot taken just after. At the same instant they are identical.

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

The generated project public/secret keys are what fasterRag's tracing exporter uses; they land in `.env`, referenced by env-var name — `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`, with `LANGFUSE_HOST` defaulting to `http://localhost:3000`.

**What actually gets exported.** Every query trace the store writes is also shipped to Langfuse: one `trace-create` carrying the question and the answer, then one observation per stage. The generation stage is sent as a `generation-create` rather than a `span-create`, because Langfuse derives model and token usage only from that type — sending it as a plain span leaves Langfuse's own cost view empty while the data sits in the payload. Span offsets are converted from milliseconds-since-query-start to absolute timestamps, or every trace would land on the epoch and stack on top of the others.

Three properties are deliberate:

- **The local trace store stays authoritative.** Replay and `GET /v1/traces/{id}` read the local copy, so investigating an incident never requires the observability stack to be healthy — precisely when it is least likely to be.
- **Export never fails a query.** The answer has already been returned; an unreachable or rejecting Langfuse is logged and dropped.
- **Missing keys are a warning, not a failure.** The toggle that enables export also provisions the stack, so refusing to serve because a dashboard has no credentials inverts the dependency.

### Flow summary

`langfuse: true` → doctor preflight (ports 3000/3030/5432/6379/8123/9000/9090/9091 free, Docker running, disk) → generate compose + `.env` secrets (once) → `docker compose up -d` → health-check until ready → **return `http://<host>:3000`** → export traces. Re-running converges (no reinstall, secrets untouched). **Zero application-code changes.**

Three implementation details are load-bearing, each learned from a failure that presents as something else:

- **The generated compose file is always run with `--env-file` pointing at the project-root `.env`.** Compose looks for a `.env` beside the compose file, which lives under `.fasterrag/langfuse/`. Without the flag every `${...}` interpolates to the empty string, the stack starts with blank passwords, and the web container fails with a Postgres *authentication* error that reads like a credentials bug rather than a wiring one.
- **`CLICKHOUSE_CLUSTER_ENABLED: "false"`** on both web and worker. This stack runs one ClickHouse node; left at its default the migration issues `CREATE TABLE ... ON CLUSTER default` with `ReplicatedMergeTree`, which a single node rejects for having no Zookeeper — *after* the Postgres migrations have already succeeded.
- **MinIO creates the `langfuse` bucket before its server starts.** MinIO does not create one on demand, and event uploads fail against a missing bucket long after the stack has reported itself healthy.

Only Langfuse's own variables are handed to the Compose subprocess. `.env` also holds LLM provider credentials, which the stack has no business seeing.

## 5. Grafana auto-provisioning (`observability.grafana: true`) — provisioning-as-code

No manual clicks; everything is version-controlled config:

- YAML manifests are written under Grafana's provisioning directory (default **`/etc/grafana/provisioning/`**, mounted into the container) with **`datasources/`** and **`dashboards/`** subdirectories, read by Grafana **at startup**.
- Datasource manifests set **`editable: false`**; dashboard providers set **`allowUiUpdates: false`** — the UI becomes read-only for provisioned resources, enforcing GitOps (the files in the repo are the source of truth).
- Dashboards are **JSON** files; the provider polls/reloads them on an interval (e.g. **`updateIntervalSeconds: 30`**) so file changes apply **without restart**.
- fasterRag ships dashboard JSON for: query latency (p50/p95 per stage), ingestion throughput, cache hit ratio, queue/DLQ depth, circuit-breaker state, cost per query.
- **A Prometheus instance is provisioned alongside Grafana**, scraping fasterRag's `/metrics` endpoint; the Grafana datasource points at *that*. Grafana's Prometheus datasource speaks PromQL to a Prometheus server, so pointing it straight at the exposition endpoint would yield a datasource that never returns a series — the hop is not optional.
- The datasource **uid is pinned** (`fasterrag-prometheus`). Left unset, Grafana mints a random uid per installation and every dashboard panel — which references the datasource by uid — resolves to nothing, rendering empty with no error.
- Prometheus is published on the host at **9099**, not its own default of 9090, because the Langfuse stack publishes MinIO there (§4) and both toggles must be able to be on at once. Inside the container it still listens on **9090**, and the datasource — which reaches it container-to-container over the `fasterrag` network — uses that. The two numbers are deliberately different; collapsing them binds a port nothing serves.
- Every panel carries a **`legendFormat`** naming the one label that varies (`{{stage}}`, `{{unit}}`, `{{cache}}`). Without it Grafana labels each series with its whole label set, and the identical `instance`/`job` scrape labels push the distinguishing one past the end of the line.
- Every panel also carries a **description** stating when it is legitimately empty. A panel with nothing to show yet looks exactly like a panel whose datasource is broken, and the panel itself is the only place that distinction can be made.

As with Langfuse: doctor-gated, idempotent, and **no application-code changes at toggle time**.

**A declared metric is not an emitted metric.** A panel over an instrument that no code path ever writes exports zero series and renders "No data" forever, with nothing in any log to say why. `fasterrag_circuit_state` is currently the one such metric — its configuration exists but the breaker does not (TASK-0148) — and the test suite pins it as the only permitted exemption.

## 6. Reliability observability (summary; doctrine in [reliability.md](reliability.md))

- RED metrics per endpoint (rate, errors, duration).
- Per-stage spans (parse, chunk, embed, retrieve, rerank, generate) correlated by trace id.
- Exported gauges for queue depth, DLQ depth, circuit-breaker state, cache hit rate.
- `/healthz` (liveness) is distinct from `/readyz` (dependencies actually checked) — see [api-reference.md](api-reference.md).
