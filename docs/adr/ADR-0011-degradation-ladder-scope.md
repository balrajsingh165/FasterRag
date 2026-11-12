# ADR-0011: The Degradation Ladder Is the Rungs That Ship — `cache_only` Is Deferred

- Status: accepted
- Date: 2026-11-13
- Deciders: fasterRag maintainers

## Context and Problem Statement

D4 ([differentiators.md](../differentiators.md)) specified a three-rung degradation ladder: reranker down → `hybrid_only`, vector database down → `cache_only` (answer from the semantic cache), LLM down → `extractive`. Every degraded response carries `degraded: true` and its `mode`.

Two of the rungs shipped and are chaos-verified: `hybrid_only` (`services/querying.py::fused_shortlist`, TASK-0289, TASK-0449) and `extractive` (`services/generation.py::extractive_answer`, TASK-0438). The circuit breaker the ladder pairs with shipped too (`core/breaker.py`, TASK-0148). `cache_only` did not: since TASK-0394 it was declared in `services/degradation.py`'s `_PLANNED_RUNGS`, and a vector-database outage returns a typed, retryable `503 RETRIEVAL_FAILED`. Documents carried a status annotation for it wherever the ladder was described, and the first release was gated on deciding whether to build it (TASK-0159) or narrow D4 to what ships (TASK-0165).

## Decision Drivers

- **What `cache_only` could add over the code that already runs.** `GenerationService` looks the semantic cache up *before* retrieval, at `cache.similarity_threshold`, and returns a hit without touching the vector database at all. A query that reaches retrieval has therefore already missed the cache at the configured threshold. A `cache_only` rung at that threshold has nothing left to serve; one that relaxes the threshold on failure serves an answer to a *different* question, which is a quality drop with a label on it — the thing D4 exists to prevent.
- **Who would get it.** `cache.semantic` defaults to `false`, and the default `memory` backend is per process, so a default install would never reach the rung, and a multi-replica deployment would reach it only on the replica that happened to cache a near match.
- **What failing does today.** A vector-database outage is a typed, retryable `RETRIEVAL_FAILED` problem with a remedy, the breaker opens so the outage is not hammered, and recovery is automatic (failure-modes rows 12, 14, 16). That is honest and actionable; it is not a silent quality drop.
- **The provable-claims policy.** A ladder documented with a rung the code refuses to serve is a standing claim the tree contradicts, held in place only by annotations.

## Considered Options

1. **Narrow D4 to the rungs that ship** — `hybrid_only` and `extractive` — and defer `cache_only` explicitly.
2. **Build `cache_only` as specified** (TASK-0159): on a retrieval failure, consult the semantic cache and serve `degraded: true, mode: cache_only`.

## Decision Outcome

**Chosen: option 1**, on the maintainer's decision of 2026-11-13.

The ladder is two rungs below `full`:

| Component lost | `mode` | What is served |
|---|---|---|
| Reranker (fails, times out, or cannot load) | `hybrid_only` | The fused dense + BM25 ranking, unreranked |
| LLM provider (and its fallback) before the first token | `extractive` | The retrieved passages, with citations, and no generated text |
| Vector database | — | Not a rung: a typed, retryable `503 RETRIEVAL_FAILED` problem; adapters that consult the circuit breaker (Qdrant, pgvector) stop calling a database it has seen fail repeatedly |

`cache_only` was considered and **deferred**: it stays out of the ladder, out of the wire contract for `mode`, and out of the documentation except as this deferral. In code, `DegradationMode.CACHE_ONLY` and the `_PLANNED_RUNGS` entry in `services/degradation.py` are kept, so `serving()` still refuses it and `scripts/check_unbuilt_claims.py` can still decide any sentence that says it is not built.

### Consequences

- Good: every document describes exactly the rungs `_BUILTIN_RUNGS` registers; the specified-not-built annotations for `cache_only` in reliability.md, api-reference.md, glossary.md, failure-modes.md, differentiators.md, data-model.md, observability.md, quickstart.md, testing-strategy.md, troubleshooting.md and verification.md are replaced by the narrowed ladder and a link here.
- Good: no new code path that answers from the cache with less evidence than a normal hit, and no new interaction between the cache, tenancy and the breaker to test.
- Bad: when the vector database is down, a client gets an error rather than a degraded answer, and must retry. D4's original promise of an answer for every single-component loss is narrowed to the reranker and the LLM.
- Neutral: the `mode` values a response can carry are `full`, `hybrid_only` and `extractive`; no response has ever carried `cache_only`, so no client sees a change.
- Follow-up: revisit if the semantic cache is turned on by default, or gains a cross-replica backend that most deployments run. A rung that ships must move from `_PLANNED_RUNGS` to `_BUILTIN_RUNGS` in the same commit, and this ADR is then superseded.
