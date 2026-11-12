# ADR-0010: Ship Cross-Encoder Reranking Off by Default

- Status: accepted (amends the default chosen in [ADR-0004](ADR-0004-hybrid-search-plus-reranking.md))
- Date: 2026-11-13
- Deciders: fasterRag maintainers

## Context and Problem Statement

[ADR-0004](ADR-0004-hybrid-search-plus-reranking.md) made cross-encoder reranking part of the default retrieval stack, and the shipped `config.yaml` set `retrieval.rerank: true` with `retrieval.reranker_model: BAAI/bge-reranker-v2-m3`. TASK-0174 made the first-run wait visible but left the default itself open as TASK-0175: should a fresh install rerank?

BENCH-0003 (2026-11-03, `scripts/measure_rerank.py`) priced the trade for the first time. It is **not citable** — non-isolated hardware, CPU-only — and is used here as indicative input to a decision, not as a published claim:

- **Cost.** The shipped model is 2.14 GB on disk, a one-time download measured at 357 s. Loading it from a populated cache takes 8.3–9.3 s, about a second more than the 87.56 MB `cross-encoder/ms-marco-MiniLM-L-6-v2`. The recurring cost is per query: 17–25× the small model's rerank latency, which at the shipped `retrieval.rerank_top_n: 100` is **55.7–86.9 s per query** on CPU, against 4.3 s for the same query path with `rerank: false`.
- **Benefit.** Not measurable on the committed golden sets. `handbook` is saturated (every configuration scores 1.0 on every metric); on `policies` the paired MRR gain is +0.0938 ± 0.0504 over 16 queries, 1.86 standard errors, inside the two-sigma bar; and no measured difference separates the 2.2 GB model from the 90 MB one. A golden set large enough to resolve the gain is TASK-0297, which no hardware substitutes for.

The decision is therefore a cost-only decision: the cost is measured, and the benefit cannot be measured until a corpus that does not yet exist is built.

## Decision Drivers

- A first query that downloads 2.14 GB and then takes about a minute per query on CPU is every new user's first impression, and reads as a hang.
- The provable-claims policy: a default justified by an expected quality gain that our own harness cannot resolve is a default justified by an unmeasured claim.
- Reversibility: reranking is one setting (`retrieval.rerank: true`) plus the `rerank` extra; nothing about the index or the collection depends on it.
- Risky or costly behaviour ships behind a flag defaulting to `false` (CLAUDE.md coding standards).

## Considered Options

1. **`retrieval.rerank: false` by default**, with the existing `BAAI/bge-reranker-v2-m3` kept as the model used when a user opts in.
2. A smaller cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on by default, the large one as a documented upgrade.
3. Keep `rerank: true` with the large model and document the first-run and per-query cost.

## Decision Outcome

**Chosen: option 1**, on the maintainer's decision of 2026-11-13.

Option 3 keeps a measured cost of tens of seconds per query on CPU for a benefit nobody can show. Option 2 cuts that cost by an order of magnitude, but still adds a model download and per-query latency to every fresh install on the strength of a gain BENCH-0003 could not resolve, and it would change which model a `rerank: true` deployment gets without a measurement saying the small model is good enough. Option 1 is the only one that ships no unmeasured trade: retrieval is hybrid dense + BM25 fused with RRF out of the box, and reranking is a one-line opt-in that a user can measure on their own corpus (`fasterrag autopilot run` before and after, [cookbook.md](../cookbook.md) R8).

`retrieval.reranker_model` keeps `BAAI/bge-reranker-v2-m3` as its default, so a configuration that sets `rerank: true` today behaves exactly as before.

### Consequences

- Good: a fresh install answers its first query without a 2.14 GB download, and without the `rerank` extra installed.
- Good: the canonical `config.yaml` and the packaged local profile (`fasterrag/data/config.local.yaml`, which already set `rerank: false`) now agree on reranking.
- Good: the `eval-gate` CI job and the committed `handbook` and `policies` baselines were already recorded and pinned at `rerank=false`, so neither baseline moves. The handbook baseline's stated reasons for not matching the shipped defaults shrink from three to two (`token_counter` and `top_k`).
- Bad: a deployment that relied on the implicit default and never set `retrieval.rerank` stops reranking on upgrade. Because `retrieval.rerank` is a comparability field, `retrieval_config_hash` changes for such a deployment, so its `index.lock` reports drift and a recorded eval baseline taken with reranking on reports `NOT_COMPARABLE`. Setting `retrieval.rerank: true` explicitly restores the previous behaviour and hash.
- Bad: the quality effect of reranking — expected, from the literature, to be the strongest lever in the stack — is no longer what a default install gets. That is the trade, taken knowingly; it is revisited when TASK-0297 produces a golden set that can measure the gain.
- Neutral: the `hybrid_only` degradation rung (D4) only engages when reranking is on, since there is no reranker to lose otherwise.
- Follow-up: when TASK-0297 lands, re-run `scripts/measure_rerank.py` against it and re-open this decision if the gain is resolvable and worth its cost.
