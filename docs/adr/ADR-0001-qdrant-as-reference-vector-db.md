# ADR-0001: Qdrant as the Reference Vector Database

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

fasterRag must support any vector database through adapters, but one backend has to be the reference implementation: the default in config, the one exercised hardest in tests, the one the system can auto-provision, and the template every other adapter is measured against. Which backend should that be?

## Decision Drivers

- Self-hostable with a simple single-container story (auto-provisioning target).
- Strong performance at very large scale (hundreds of millions of vectors), with sharding/replication.
- First-class payload/metadata filtering pushed down into search (pain point #7).
- Open source, permissive license, active development, good client ergonomics.
- Clean API-key security story for remote deployments.

## Considered Options

1. **Qdrant**
2. Milvus
3. Weaviate
4. pgvector
5. Chroma

## Decision Outcome

**Chosen: Qdrant.** Single lightweight container (auto-provisionable with a named volume), rich payload filtering, collection aliases (which D2 zero-downtime reindexing builds on), snapshots (disaster recovery), sharding + replication for scale, `QDRANT__SERVICE__API_KEY` auth, and a well-documented client whose defaults we adopt verbatim (REST 6333, gRPC 6334, `prefer_grpc=False`; both ports must be exposed — Qdrant Discussion #2195).

### Consequences

- Good: one blessed path that doctor, provisioning, DR drills, and benchmarks all target; alias-based blue/green reindexing comes almost free.
- Good: the Qdrant adapter becomes the executable specification for the adapter contract suite.
- Bad: reference-first development risks Qdrant-shaped assumptions leaking into core — mitigated by the hard adapter boundary ([ADR-0002](ADR-0002-adapter-factory-pluggability.md)) and by running the contract suite against all adapters in CI.
- Neutral: Milvus/Weaviate/Pinecone/pgvector/Chroma remain fully supported via the same interface; none is disadvantaged at the API level.
