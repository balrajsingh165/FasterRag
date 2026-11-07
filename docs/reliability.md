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
- *How a handler satisfies "with a logged correlation/trace id":* by logging at all. The JSON formatter in `observability/logging.py` reads the `fasterrag_trace_id` context variable and stamps `trace_id` onto **every** record, so a plain `_logger.warning("…", extra={…})` already carries the correlation id and a handler never needs to thread one in by hand. The rule is therefore about whether the handler says anything, not about how it says it — which is exactly what `scripts/check_error_taxonomy.py` (§5) measures.
- The API surfaces errors as RFC 9457 `application/problem+json` with the stable `code`; the Python package raises the same typed classes (`fasterrag.errors`); the CLI maps them to exit codes.
- Retries act only on `retryable: true` errors; `retryable` is set at the adapter boundary where the semantics are known (e.g. HTTP 429/503 → retryable; 401 → not).

## 2. Resilience patterns (mandatory on every external interaction)

| Pattern | Rule | Config |
|---|---|---|
| **Timeouts** | Explicit, configured timeout on every network/provider call — no unbounded await exists in the codebase. **Status: wired into every adapter.** The suspected violation this row used to carry as open (TASK-0153) was closed as a **mis-read, not a defect**: the apparent hang was the observer's own shell budget, and a single query returned in 10.78 s on the same deployment. `reliability.timeouts.llm_ms` is applied by the adapter at construction. | `reliability.timeouts.*` |
| **Retries** | Exponential backoff + jitter, only on errors flagged `retryable`; attempts bounded. | `reliability.retries.*` |
| **Circuit breakers** | One breaker per provider, owned by the service that calls it: **LLM** (`GenerationService`), **embeddings** (`EmbeddingPool`), and **vector DB** (`RetrievalService`). Each is built once per service instance, never per request — a breaker rebuilt per call counts to one forever and can never open. Open after `failure_threshold` consecutive failures; half-open probes after `reset_timeout_ms`; **state exported as the `fasterrag_circuit_state` metric**. Only **retryable** failures count: a 401 fails identically forever, so counting it would open the breaker, half-open it after the timeout, fail again, and re-open — presenting a permanent misconfiguration as an intermittent outage. **Status: wired for embeddings and the vector DB** — into the embedding pool's retry loop, checked *inside* it so a breaker that opens mid-batch stops the remaining attempts rather than spending the whole retry budget on a provider already known dead, and into the Qdrant and pgvector adapters. **The `llm` breaker is constructed at startup and consulted by nothing** (TASK-0245), so the generation path currently has retries and timeouts but no breaker. All three are built eagerly so the metric reports `closed` from startup rather than springing into existence on the first failure. | `reliability.circuit_breaker.*` |
| **Bulkheads** | Ingestion and query paths use separate worker pools and bounded queues, so an ingestion storm can never starve live queries. Structural — not configurable off. | pool sizes under `workers.*` |
| **Idempotency keys** | All mutating API endpoints accept `Idempotency-Key`; replays return the original result and perform no duplicate work. **Status: shipped 2026-11-08 (TASK-0318).** `api/idempotency.py` is ASGI middleware over the `IdempotencyStore` in `services/idempotency.py`, so a mutating router is covered the moment it is mounted rather than when somebody remembers a dependency. Scope is `(tenant, key)` over a fingerprint of method, path, query and body; a replay returns the stored response with `Idempotent-Replayed: true`; a changed request under a live key is `422 IDEMPOTENCY_KEY_REUSED`; a concurrent duplicate loses an `O_EXCL` race and is refused `409 IDEMPOTENCY_IN_PROGRESS` rather than executing. Only 2xx is stored. `/v1/query`, `/v1/retrieve`, `/v1/estimate` and `/v1/replay` are deliberately out of scope — they are `POST` for the body, not for an effect. **Records are files under the journal root, so replay protection spans every worker on one host but not several hosts without shared storage** ([api-reference.md](api-reference.md) §Idempotency). The `idempotency_key` body field on `POST /v1/ingest` still works and is deprecated for 1.0. | — |
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

**Specified but not built** (TASK-0243): load (k6/Locust, hardware-annotated) and 24 h soak (no memory/fd growth) — both still waiting on the reference hardware TASK-0084 needs. **Mutation testing landed 2026-11-04** as `scripts/mutation_run.py`, built on CPython's own `ast` rather than mutmut (which refuses to run on Windows) or cosmic-ray (whose parser rejects valid code in this tree); it runs nightly, never on the push path, and reports survivors rather than failing the build, because the set contains equivalent mutants. Its first measured run covered four of six targets — 134 mutants, 71 killed, 63 survivors (TASK-0300) — but **that run used the runner before TASK-0303 ✅ fixed its isolation**, when tests were re-read live from the working tree on every mutant while those same tests were being edited, so the split is indicative rather than exact and wants re-deriving. The gaps it pointed at are real; the counts are not evidence. `breaker` is re-verified on the fixed runner at 32 of 34 killed. Mutation checking by hand continues alongside it: the runner finds gaps, a reviewer still proves each new test red before it is trusted.

## 5. Static discipline

- Strict typing: mypy (or pyright) strict mode, zero errors.
- Ruff for lint + format.
- Pre-commit hooks enforce lint, format, typing, and the commit-message rule (single line, no trailers).
- Any `# type: ignore` requires an adjacent `# CRITICAL:` justification (consistent with the comment policy — docstrings only, plus `# CRITICAL:` and `# TODO:`/`# BLOCKED:`).
- **The taxonomy rules above are enforced by `scripts/check_error_taxonomy.py`** (landed 2026-11-07), an `ast` gate whose handler rules read `src/fasterrag` only — `scripts/` is scanned for `# type: ignore` and nothing else, so a swallow inside a gate script is invisible to the gate (TASK-0338). **Status: the gate and its tests ship and its ratchet is empty as of 2026-11-08 — the six violations it shipped with were fixed, not annotated (TASK-0317). Wiring it into the CI quality job alongside the other `check_*` steps is still open under the same task, so until that lands it is run by hand (`python scripts/check_error_taxonomy.py`).** Until it existed nothing checked §1 at all: `ruff` catches a bare `except:` (E722) and nothing else, which left the commoner shape — a handler whose whole body is `pass`, `continue`, or `return None` — unchecked for the life of the repository. Three of its rules fail the build and three only warn, and the split is measured rather than asserted:

| Rule | Severity | What it fires on | Measured on `main`, 2026-11-08 |
|---|---|---|---|
| `swallow` | **error** | A handler whose entire body is `pass`/`continue`/`break`/`return`/`return None`/`...`. Exempt: `StopIteration`/`StopAsyncIteration`, and a `pass` arm on a `try`/`except`/`else` (a probe, not a catch). | 0 sites. 4 on 2026-11-07, all ratcheted; all four fixed 2026-11-08 (TASK-0317) |
| `unjustified-ignore` | **error** | `# type: ignore` with no `# CRITICAL:` on the same line or in the comment block above it. Read from comment *tokens*, so a mention inside a string never counts. | 0 sites. 2 on 2026-11-07; both removed 2026-11-08 by fixing the types, not by annotating them (TASK-0317) |
| `swallow-fallback` | **error** | A handler whose whole body is `return <expr>` for a substantive `expr`, catching `Exception`/`BaseException`, where `expr` does not mention the caught exception — the `swallow` shape with a fallback value in place of `pass`. **Scoped to a broad catch by measurement, not by taste**: of 34 lone-fallback returns in the tree, 14 return a value built from the exception and 20 do not; of those 20, 13 are predicates or probes where the exception *is* the answer (`_is_ip_address`, `supports_pooling`, `port_is_free`, the `csv.Sniffer` fallback) — 65% false positives at the wider width. At broad-catch width it was 0 false positives of 1. It also suppresses `broad-except` on the same handler, so a blocking verdict never also sits in a warning bucket. | 0 sites. 1 on 2026-11-08 (`tokenizer.py:81`), fixed the same day (TASK-0333) |
| `swallow-documented` | warning | The same trivial `return None`, but where the function's return annotation already admits `None`. Warns rather than fails because the absence is published — and stays visible because an annotation cannot tell "missing" from "corrupt". | 7 sites |
| `broad-except` | warning | `except Exception`/`except BaseException` that rethrows nothing. A broad catch that translates to a typed error is the adapter boundary working as specified, so 15 of the 22 broad catches in the tree are correctly silent. | 7 sites |
| `untyped-raise` | warning | A **builtin** exception raised from a *public* function under `services/` or `adapters/` with no enclosing `try` to translate it. Warns because it resolves names lexically: a factory call or a local subclass is invisible to it by design. | 1 site |

  The gate carries a **ratchet**, not an allowlist: each `(rule, file)` entry records how many known violations that file holds, so the file may shrink but never grow. Raising a number to make a new violation green is the one thing the gate forbids. The map is **empty today**, which is the only state in which the gate means what it says; `test_the_real_tree_carries_no_error_severity_finding_at_all` is what keeps emptiness from meaning *unmeasured*. It also carries a vacuity guard (trap T12): a run that matches no module, or parses no `except` handler, fails rather than reporting a clean tree.

  What it cannot do: it cannot tell a deliberate swallow from a careless one — the remedy for a deliberate one is a one-line log, not an exemption — and it says nothing about whether the problem+json contract holds at runtime. That contract is held by `api/problems.py`, whose catch-all `Exception` handler renders any unclassified failure as `INTERNAL` with a problem body and a trace id, and is covered by the API tests rather than by this gate.

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
