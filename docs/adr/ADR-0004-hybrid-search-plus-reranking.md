# ADR-0004: Hybrid Search + RRF (k=60) + Cross-Encoder Reranking as the Default Retrieval Stack

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

Retrieval accuracy is the highest-stakes pain point (#2): a RAG system that retrieves the wrong chunks cannot be saved downstream. What retrieval stack ships as the default?

## Decision Drivers

- Dense-only search misses exact identifiers, rare terms, and out-of-vocabulary tokens; sparse-only misses paraphrase/semantics.
- Fusion must be robust without per-corpus score calibration.
- The literature and industry evidence base should carry the defaults, pending our own ledger measurements.
- Latency budget must stay compatible with sub-second retrieval goals.

## Considered Options

1. **Hybrid dense + BM25, fused with Reciprocal Rank Fusion (k=60), then cross-encoder reranking of the top candidates**
2. Dense-only ANN search
3. Hybrid with learned/weighted score fusion
4. Hybrid + RRF without reranking

## Decision Outcome

**Chosen: option 1.** Both legs run in parallel with pushed-down metadata filters; fusion is rank-based RRF with **k=60**, the value recommended in the original 2009 SIGIR paper by Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher (*"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"*), shown robust across TREC/LETOR benchmarks — rank-based fusion needs no score normalization across heterogeneous legs. A cross-encoder then reranks the fused top 100–1000 candidates before truncation to top-K: reranking adds ~100–300 ms per query and is the single biggest quality lever available; combined with contextual enrichment, Anthropic's September 2024 Contextual Retrieval results report failed retrievals dropping 5.7% → 1.9% (−67%).

### Consequences

- Good: best-known-practice retrieval quality out of the box; every stage is config-toggleable (`retrieval.hybrid`, `retrieval.rerank`) for latency-sensitive deployments.
- Good: rank-based fusion is backend-agnostic — it works identically across all vector DB adapters.
- Bad: BM25 index maintenance alongside the vector index (storage + ingest cost); accepted.
- Bad: reranker adds latency and a local model dependency; mitigated by the degradation ladder (`hybrid_only` mode, D4) and config toggle.
- Follow-up: our own eval-harness measurements land in the [benchmark ledger](../benchmarks.md); defaults are re-examined if our data disagrees with the literature.
