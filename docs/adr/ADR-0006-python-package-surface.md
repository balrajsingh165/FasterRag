# ADR-0006: Ship fasterRag as an Importable Python Package (`pip install fasterrag`)

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

fasterRag began as a backend service controlled via REST + CLI. Users adopting it as a *framework* want to `import fasterrag` and use the pipeline — or individual pieces of it — inside their own applications without running an HTTP service. What is the packaging and API-surface decision?

## Decision Drivers

- Framework adoption: piecemeal use (just chunkers, just fusion, just evals) lowers the barrier to entry.
- One behavior everywhere: the library must not become a second implementation that drifts from the service.
- Extensibility: third-party providers should plug in without forking (supports the "everything out there" ambition).
- Stability: a published package needs an explicit compatibility contract.

## Considered Options

1. **One package: the service is a thin shell over an importable engine; public API = engine facade (`FasterRag`) + standalone components + typed errors + entry-point plugin contract**
2. Service-only (users wrap the REST API)
3. Two codebases (SDK client + server)

## Decision Outcome

**Chosen: option 1.** `pip install fasterrag` installs the engine; `fasterrag serve` and the REST API are thin layers over the same service layer the library exposes. The public surface is specified in [python-api.md](../python-api.md): `FasterRag` facade (async + sync), standalone components (`fasterrag.parsing/chunking/retrieval/rerank/evals`), `fasterrag.errors` (same taxonomy and `code`s as the API), and entry-point groups for third-party adapters. SemVer 2.0.0 governs the exported surface; extras (`fasterrag[openai]`, `[milvus]`, `[all]`) keep the core install lean.

### Consequences

- Good: library, CLI, and API cannot drift — they are the same engine; errors carry identical `code`s across all three surfaces.
- Good: piecemeal adoption and a supported plugin path for the long tail of providers.
- Bad: the public API becomes a compatibility contract — breaking changes now require major versions and migration notes; accepted deliberately.
- Bad: in-process worker pools in library mode complicate resource guidance; mitigated by documented deployment modes ([deployment.md](../deployment.md)).
- Note: this extends [ADR-0005](ADR-0005-api-cli-only-control-plane.md)'s control-plane definition to three programmatic surfaces; the no-control-GUI rule is unchanged.
