# ADR-0007: The BM25 Leg Lives in the Vector Database as Sparse Vectors

- Status: accepted
- Date: 2026-07-30
- Deciders: fasterRag maintainers

## Context and Problem Statement

Hybrid retrieval needs a sparse BM25 leg beside the dense leg ([ADR-0004](ADR-0004-hybrid-search-plus-reranking.md)). [architecture.md](../architecture.md) §6 requires that both legs run over **the same corpus** and that **both receive pushed-down metadata filters**. Where should that BM25 index physically live?

## Decision Drivers

- Both legs must see identical documents and identical filters, or hybrid retrieval silently returns inconsistent candidate sets.
- BM25 scoring needs IDF, which is a statistic over the **whole live corpus** — it changes with every ingest and delete.
- The corpus target is hundreds of millions of chunks, so the sparse index must be on disk, shardable, and replicable.
- Zero-downtime reindexing (D2) swaps a collection alias atomically. Anything outside that collection is not covered by the swap.
- [integrations.md](../integrations.md) already states that the pgvector adapter's BM25 leg uses PostgreSQL full-text, so the sparse leg is understood to be backend-specific.

## Considered Options

1. **Sparse vectors inside the vector database collection**, one collection holding both legs
2. A separate BM25 index that fasterRag owns and persists itself
3. An in-process library index (for example `rank_bm25`) rebuilt at startup

## Decision Outcome

**Chosen: option 1.** The `VectorDBAdapter` contract is extended additively with an optional sparse vector on `CollectionSpec`, `Point`, and `SearchQuery`. Qdrant stores BM25 term frequencies as a named sparse vector and applies IDF server-side via `Modifier.IDF`, so the client supplies term frequencies and the backend supplies the corpus statistic that only it can know.

Option 3 was rejected outright: an in-memory index cannot hold hundreds of millions of chunks, and rebuilding it at startup makes start-up time a function of corpus size.

Option 2 was rejected because every property the requirements ask for would have to be rebuilt: its own metadata filtering, its own persistence and crash recovery, its own sharding, its own blue/green alias swap, and its own IDF maintenance on every ingest and delete. That is a second database to operate, and the two indexes would drift apart precisely when a job fails halfway.

### Consequences

- Good: both legs share one collection, so metadata filters, sharding, replication, tenancy, and the D2 alias swap apply to both for free and cannot drift.
- Good: IDF is always current, because the backend recomputes it as the corpus changes.
- Good: a backend with native hybrid fusion can serve both legs in one round trip.
- Bad: the documented adapter interface grew. The additions are optional, so an adapter that does not implement sparse retrieval keeps working and reports the capability as absent — but a provider wanting hybrid support now has more to implement, and the contract suite tests it.
- Bad: the sparse leg's quality now depends on each backend's sparse implementation rather than on one implementation fasterRag controls. The shared contract suite is what keeps that honest.
- Constraint: term-frequency encoding stays in fasterRag (`fasterrag.core.retrieval.bm25`) so the same tokenization and saturation apply across backends; only IDF and storage are delegated.
