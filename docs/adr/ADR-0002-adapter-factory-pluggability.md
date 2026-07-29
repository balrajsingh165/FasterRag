# ADR-0002: Adapter/Factory Pattern for All Vendor Pluggability

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

"Any vector database, any embedding model, any LLM provider — selected purely through configuration" is a core promise. How do we structure vendor integrations so that swapping a backend is a config edit, never a refactor, and so third parties can add providers without forking?

## Decision Drivers

- One-line config swap (`vector_db.provider: qdrant → pgvector`) with zero application-code changes.
- Vendor lock-in is a catalogued pain point (#17); anti-lock-in is a differentiator (D11).
- Testability: "any vector DB" must be a tested promise, not a hope.
- Extensibility for the Python package: third-party providers without forking.

## Considered Options

1. **Abstract adapter base classes + config-driven factory + entry-point plugin registration**
2. Direct vendor SDK usage guarded by `if provider == ...` branches in core
3. A generic ORM-style abstraction layer over one primary DB

## Decision Outcome

**Chosen: option 1.** Abstract `VectorDBAdapter` (methods: `create_collection`, `upsert`, `search`, `update`, `delete`, `health`), plus equivalent `EmbeddingAdapter` and `LLMAdapter` bases. A factory reads `*.provider` from config and instantiates the concrete adapter. Third parties register adapters via `fasterrag.vectordb` / `fasterrag.embeddings` / `fasterrag.llm` entry points. Vendor types never escape `adapters/`. Every implementation must pass the shared adapter contract test suite.

### Consequences

- Good: backend swap = config change; core depends only on interfaces; the contract suite makes conformance objective.
- Good: entry-point registration turns "supports everything out there" into an extension mechanism rather than an ever-growing in-repo burden.
- Bad: lowest-common-denominator risk — vendor-unique features need explicit interface extensions; accepted, reviewed case-by-case.
- Bad: N adapters × contract suite = real CI cost; accepted as the price of a tested promise.
