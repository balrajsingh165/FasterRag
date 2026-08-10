# reliability.md — Reliability Doctrine & Patterns Catalogue

**A claim without a measurement is a bug.**

That sentence governs this repository: every "fastest," "most reliable," or "best" anywhere in the docs must trace to a [benchmark ledger](benchmarks.md) entry or a passing test, or it must be rewritten as a goal. This file specifies the engineering discipline that makes reliability provable rather than asserted.

## 1. Error taxonomy (typed exception hierarchy)

```text
FasterRagError                      # root; carries code, trace_id, retryable
├── ConfigError                     # invalid config / missing env var (fail-fast at startup)
├── IngestionError
│   ├── ParseError                  # document unparseable → DLQ reason PARSE_FAILED
│   ├── ChunkError                  # chunker invariant violated
│   └── EmbedError                  # embedding stage failure
├── RetrievalError                  # dense/BM25 leg, fusion, or rerank failure
├── GenerationError                 # LLM generation failure
├── ProviderError                   # external provider transport/API error; retryable: bool
├── CacheError                      # cache backend failure (system degrades to cache-off)
└── ProvisioningError               # doctor/provisioner failure; carries fix-it hint
```

Rules:

- Every exception carries `code` (stable, identical to the API problem `code` — table in [api-reference.md](api-reference.md)), `trace_id`, and `retryable`.
- **No bare `except`. No silently swallowed exceptions. Anywhere. Ever.** Every caught exception is either handled with a logged correlation/trace id or rethrown.
- The API surfaces errors as RFC 9457 `application/problem+json` with the stable `code`; the Python package raises the same typed classes (`fasterrag.errors`); the CLI maps them to exit codes.
- Retries act only on `retryable: true` errors; `retryable` is set at the adapter boundary where the semantics are known (e.g. HTTP 429/503 → retryable; 401 → not).

## 2. Resilience patterns (mandatory on every external interaction)

| Pattern | Rule | Config |
|---|---|---|
| **Timeouts** | Explicit, configured timeout on every network/provider call — no unbounded await exists in the codebase. **Status: wired into every adapter.** The suspected violation this row used to carry as open (TASK-0153) was closed as a **mis-read, not a defect**: the apparent hang was the observer's own shell budget, and a single query returned in 10.78 s on the same deployment. `reliability.timeouts.llm_ms` is applied by the adapter at construction. | `reliability.timeouts.*` |
| **Retries** | Exponential backoff + jitter, only on errors flagged `retryable`; attempts bounded. | `reliability.retries.*` |
| **Circuit breakers** | One breaker per provider, owned by the service that calls it: **LLM** (`GenerationService`), **embeddings** (`EmbeddingPool`), and **vector DB** (`RetrievalService`). Each is built once per service instance, never per request — a breaker rebuilt per call counts to one forever and can never open. Open after `failure_threshold` consecutive failures; half-open probes after `reset_timeout_ms`; **state exported as the `fasterrag_circuit_state` metric**. Only **retryable** failures count: a 401 fails identically forever, so counting it would open the breaker, half-open it after the timeout, fail again, and re-open — presenting a permanent misconfiguration as an intermittent outage. **Status: wired for embeddings and the vector DB** — into the embedding pool's retry loop, checked *inside* it so a breaker that opens mid-batch stops the remaining attempts rather than spending the whole retry budget on a provider already known dead, and into the Qdrant and pgvector adapters. **The `llm` breaker is constructed at startup and consulted by nothing** (TASK-0245), so the generation path currently has retries and timeouts but no breaker. All three are built eagerly so the metric reports `closed` from startup rather than springing into existence on the first failure. | `reliability.circuit_breaker.*` |
| **Bulkheads** | Ingestion and query paths use separate worker pools and bounded queues, so an ingestion storm can never starve live queries. Structural — not configurable off. | pool sizes under `workers.*` |
| **Idempotency keys** | All mutating API endpoints accept `Idempotency-Key`; replays return the original result and perform no duplicate work. **Status: specified, not yet implemented on the shipped routers.** | — |
| **Backpressure** | Bounded queues everywhere; overflow returns `429` with `Retry-After` instead of accepting unbounded work. | `workers.queue_depth` |
| **Degradation ladder** | Component loss maps to an explicit degraded mode, never a silent quality drop (D4). **Status: `hybrid_only` (reranker loss) and `extractive` (LLM loss) are implemented and chaos-verified; the `cache_only` rung (vector-DB loss) is not built yet (TASK-0159) — today that failure surfaces as a retryable `RETRIEVAL_FAILED` problem, exactly as the chaos log in [failure-modes.md](failure-modes.md) records.** | `reliability.degradation_ladder` |

## 3. Data safety

- **Atomic file writes** (write-temp-then-rename) for the ingestion journal, `index.lock`, and the index manifest — a crash can never leave a half-written control file.
- The index manifest and alias swap update **transactionally** (D2): the alias points at exactly one complete, eval-validated collection at all times.
- Idempotent, deterministic chunk IDs make index writes replay-safe (exactly-once effects, D3).
- Backups and the restore drill per [disaster-recovery.md](disaster-recovery.md); the drill must actually be executed and ticked in [todo.md](todo.md).

## 4. Testing pyramid

Specified in full in [testing-strategy.md](testing-strategy.md).

**Built and running:** unit (≥ 85% coverage on `core/`, `adapters/`, `workers/`) · property-based chunker invariants (Hypothesis) · golden-file parser tests · testcontainers integration (Qdrant docker + remote modes, and pgvector) · the shared adapter contract suite · the retrieval eval harness · the chaos suite (D12).

**Specified but not built** (TASK-0243): load (k6/Locust, hardware-annotated) · 24 h soak (no memory/fd growth) · mutation testing (mutmut) on chunk-boundary and retrieval-scoring logic. Mutation checking is currently done by hand, slice by slice, rather than by a tool.

## 5. Static discipline

- Strict typing: mypy (or pyright) strict mode, zero errors.
- Ruff for lint + format.
- Pre-commit hooks enforce lint, format, typing, and the commit-message rule (single line, no trailers).
- Any `# type: ignore` requires an adjacent `# CRITICAL:` justification (consistent with the comment policy — docstrings only, plus `# CRITICAL:` and `# TODO:`/`# BLOCKED:`).

## 6. Security & supply chain

Specified in [security.md](security.md): pinned hash-locked dependencies · secret scanning in CI · API keys with scopes + rate limiting on by default · non-root containers · input size limits on all endpoints · SBOM at each tagged release.

## 7. Observability for reliability

- **RED metrics** (rate, errors, duration) per endpoint.
- Per-stage spans (parse, chunk, embed, retrieve, rerank, generate) correlated by trace id — the id appears in logs, problem responses, and traces.
- Exported gauges: queue depth, DLQ depth, circuit-breaker state, cache hit rate.
- `/healthz` (liveness) is distinct from `/readyz` (dependencies actually checked).
- Full catalogue in [observability.md](observability.md).

## 8. Release reliability

- Risky features (auto-provisioning, autopilot, dashboard) ship behind config flags defaulting to `false`.
- Feature branches only; a tag at every slice boundary (`v0.x.0-sN`); written revert playbook in [deployment.md](deployment.md).
- Error-budget policy in [slo.md](slo.md): when the budget is burning, feature work freezes and reliability tasks take priority in [todo.md](todo.md).
- FMEA in [failure-modes.md](failure-modes.md): every anticipated failure names its detection signal, mitigation, recovery, and the test that proves it.
