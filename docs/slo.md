# slo.md — SLIs, SLO Targets, Error Budget Policy

> **All SLO targets in this file are TBD-until-measured.** Targets are set ONLY after baseline measurement on documented reference hardware (build-phase slice S11, see [todo.md](todo.md)); until then this file defines the SLIs and their measurement methodology. Setting a target without a baseline would violate the provable-claims policy ([reliability.md](reliability.md)).

## 1. Service Level Indicators (definitions + measurement)

| SLI | Definition | Measured from | SLO target |
|---|---|---|---|
| **Query availability** | successful `POST /v1/query` responses (2xx incl. flagged degraded modes; excl. client 4xx) ÷ all query requests, per rolling 30 days | `fasterrag_requests_total` / `fasterrag_errors_total` | TBD-until-measured |
| **p95 retrieval latency (excluding LLM generation)** | p95 of embed + retrieve + fuse + rerank + assemble stage durations per query | `fasterrag_stage_duration_seconds` (sum of pre-generate stages), `timings_ms` | TBD-until-measured |
| **p95 time-to-first-token** | p95 of request receipt → first SSE `token` event | `fasterrag_ttft_seconds` | TBD-until-measured |
| **Ingestion throughput** | docs/min and tokens/sec sustained over a 10-minute window | `fasterrag_ingest_throughput` | TBD-until-measured |
| **Error rate** | responses carrying a problem `code` in the 5xx class ÷ all requests | `fasterrag_errors_total` | TBD-until-measured |
| **Cache hit rate** | semantic-cache hits ÷ lookups (rolling 24 h) | `fasterrag_cache_events_total` | TBD-until-measured |
| **DLQ depth** | dead-lettered documents currently unresolved | `fasterrag_dlq_depth` | TBD-until-measured (alarm threshold) |

Notes:

- Degraded-mode responses (D4) **count as available** for the availability SLI but are tracked separately via `fasterrag_degraded_responses_total`; a separate degraded-time budget will be set at baselining.
- Retrieval latency deliberately **excludes LLM generation** so the SLI measures what fasterRag controls; generation latency is tracked but provider-dominated.
- All SLIs are computed from the same metrics catalogue the dashboard and Grafana consume ([observability.md](observability.md)) — one source of truth.

## 2. How targets get set (S11 baselining procedure)

1. Run the full load + soak + chaos suites on the documented reference hardware ([performance.md](performance.md) methodology).
2. Record baselines in the [benchmark ledger](benchmarks.md) (claim, method, dataset, hardware, date, numbers, commit).
3. Set each SLO target from the measured baseline with explicit headroom (e.g. target = baseline p95 × 1.2), recorded here with a link to its ledger entry.
4. Replace every `TBD-until-measured` in this file in the same change; [todo.md](todo.md) tick required.

## 3. Error budget policy

- Each SLO implies an error budget (e.g. 99.5% availability → 0.5% budget per 30 days).
- **When the budget is burning** (projected exhaustion within the window at current burn rate): **feature work freezes and reliability tasks take priority in [todo.md](todo.md)** — reliability tasks are filed immediately and scheduled ahead of all feature tasks until the burn rate is back under control.
- Budget accounting reviews happen at each release and whenever an alert fires; outcomes are recorded as todo tasks, not prose reports.
- Chaos-suite regressions (a previously-passing scenario failing) are treated as budget-burning events regardless of production impact.
