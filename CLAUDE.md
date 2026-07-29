# CLAUDE.md

Always-loaded instructions for Claude Code working in the fasterRag repository. Keep this file dense; detail lives in the linked docs. **All documentation lives in `docs/` — this file is the only instruction/doc file kept at repository root (plus README.md).**

## Project overview

fasterRag is a FastAPI-based, backend-only, one-stop Retrieval-Augmented Generation (RAG) solution engineered for very, very large datasets. This repository ships the first beta of an all-in-one RAG system whose goal is to be the fastest, most efficient, and most optimized RAG project available (a goal — see the provable-claims policy below). Speed comes from multi-worker parallel processing across ingestion, chunking, embedding, and indexing; maximum chunking quality via a configurable chunking pipeline; and aggressive caching, batching, streaming, and async I/O. Everything is pluggable — any vector database, any embedding model, any LLM provider — selected purely through configuration. The system is operated ONLY through a REST API and a terminal/CLI; a separate, optional, self-hosted observability dashboard exists purely for inspection (Langfuse-like), never for control.

**Current repository state: documentation only.** No implementation code exists and none may be written until a Gate C task in [docs/todo.md](docs/todo.md) is explicitly authorized by the maintainer. Documentation work only.

## Tech stack (approved)

- Python 3.12+
- FastAPI (async endpoints for all I/O-bound operations)
- Pydantic v2 + pydantic-settings (YAML config source + `.env` secrets)
- Uvicorn (dev) / Gunicorn with Uvicorn workers (prod)
- Async task/worker system (queue-decoupled CPU pool + embedding pool)
- Qdrant as the reference vector DB (adapters for Milvus, Weaviate, Pinecone, pgvector, Chroma)
- OpenTelemetry for instrumentation

### DO-NOT-USE

- No control-plane GUI framework of any kind (no Streamlit/Gradio/NiceGUI controlling the RAG; the dashboard is observability-only).
- No secrets in YAML — `config.yaml` never contains credentials; secrets live in `.env` only, referenced by env-var name.
- No ORM-style coupling to a single vector DB — all vendor access goes through the `VectorDBAdapter` interface and factory.

## Commands (documented intentions — the code does not exist yet)

- Build: `pip install -e ".[dev]"`
- Lint: `ruff check .` and `ruff format --check .`
- Typecheck: `mypy src/` (strict; zero errors)
- Test: `pytest` (unit) / `pytest -m integration` (integration) / `pytest -m eval` (retrieval eval harness)
- Run dev API: `uvicorn fasterrag.api.main:app --reload`
- Run workers: `python -m fasterrag.workers` (spawns CPU + embedding pools per `config.yaml`)

## Coding standards

- **Docstrings only.** NO inline comments and NO explanatory comments in code. Comments are permitted in EXACTLY two cases:
  - Super-critical flag: `# CRITICAL: <why this must not change>`
  - Blocker/pending marker: `# TODO: <what remains>` or `# BLOCKED: <blocker + ticket/date>`
- Every module, class, and public function gets a docstring. Docstrings explain what and why; code explains how.
- Full type hints everywhere; `mypy --strict` must pass. Any `# type: ignore` requires an adjacent `# CRITICAL:` justification.
- **Typed error taxonomy is mandatory.** No bare `except`, no silently swallowed exceptions, anywhere, ever. Every caught exception is either handled with a logged correlation/trace id or rethrown. Hierarchy in [docs/reliability.md](docs/reliability.md).
- API errors are RFC 9457 `application/problem+json` with a stable machine-readable `code` field — never generic 500s without a problem body.
- Async-first: all I/O paths are `async`; CPU-bound work goes to worker pools, never the event loop.
- Routers contain zero business logic; services orchestrate; adapters isolate vendors. See [docs/structure.md](docs/structure.md).
- Risky features (auto-provisioning, autopilot, dashboard) ship behind config flags defaulting to `false`.

## Git standards

- **Single-line commit messages only.** No multi-line bodies. No trailers of any kind.
- **Absolutely NO Claude/AI attribution in commits** — no `Co-Authored-By: Claude`, no `Generated with Claude Code`, no AI signatures.
- Feature-branch workflow. **Never commit directly to `main`** (merges to `main` happen only when the maintainer instructs).
- Commit frequently — small, coherent, revertable commits. Tag slice boundaries (`v0.x.0-sN`) during the build phase.

## Incremental shipping discipline

- Ship small OR large features continuously, committing in the middle of work, so the repo can always be reverted (`git revert` / `git checkout <sha>`) if a change goes bad.
- Every commit must leave documentation (and later, code) in a coherent state.
- No big-bang changes: no increment may exceed a reviewable size.

## Provable-claims policy (permanent)

- **A claim without a measurement is a bug.** Every "fastest", "most reliable", "best" in any doc or commit must trace to an entry in the benchmark ledger ([docs/benchmarks.md](docs/benchmarks.md)) — claim, method, dataset, hardware, date, numbers, commit hash — or be rewritten as a goal ("targets sub-second p95 retrieval").
- "Fastest" may only be claimed relative to a named baseline we measured ourselves, with the harness committed to this repo.
- Periodically grep docs for superlatives and verify each is ledger-linked; unlinked ones are bugs — file them in [docs/todo.md](docs/todo.md).

## Task tracking

- [docs/todo.md](docs/todo.md) is the ONLY task file. Never create any other todo/task/tracking file. When completing a task, tick it and append `— ✅ YYYY-MM-DD`; ticked entries are append-only and frozen.

## Folder boundaries (sensitive — change only with explicit task authorization)

- `config/` loaders and schema — validation contract for the whole system; fail-fast behavior must not be weakened.
- `adapters/` — vendor isolation boundary; never leak vendor types outside an adapter.
- Provisioning code (Langfuse/Grafana/Qdrant auto-setup) — must remain config-driven with zero application-code changes at toggle time, and gated by `fasterrag doctor`.
- `.env`, `.env.*` — never read, print, or commit secret values.

## Pointers (all under docs/)

- [scope.md](docs/scope.md) — vision, goals, non-goals, pain-point catalogue
- [structure.md](docs/structure.md) — repository layout and layer responsibilities
- [flow.md](docs/flow.md) — end-to-end Mermaid flows
- [architecture.md](docs/architecture.md) — components, workers, adapters, scaling, fault tolerance
- [python-api.md](docs/python-api.md) — importable package surface (`pip install fasterrag`), standalone components, plugin contract
- [differentiators.md](docs/differentiators.md) — the twelve flagship capabilities (uniqueness contract)
- [reliability.md](docs/reliability.md) — reliability doctrine, error taxonomy, resilience patterns
- [failure-modes.md](docs/failure-modes.md) — FMEA table
- [slo.md](docs/slo.md) — SLIs, SLO targets, error budget policy
- [disaster-recovery.md](docs/disaster-recovery.md) — backups, restore drill, RPO/RTO
- [config-reference.md](docs/config-reference.md) — full `config.yaml` schema
- [api-reference.md](docs/api-reference.md) — REST endpoints + RFC 9457 error model
- [cli-reference.md](docs/cli-reference.md) — terminal commands
- [observability.md](docs/observability.md) — dashboard, metrics, Langfuse/Grafana provisioning
- [deployment.md](docs/deployment.md) — self-hosting, Docker, sizing, revert playbook
- [security.md](docs/security.md) — secrets, auth, multi-tenancy, supply chain
- [testing-strategy.md](docs/testing-strategy.md) — testing pyramid, eval harness, CI gates
- [performance.md](docs/performance.md) — performance goals and measurement methodology
- [benchmarks.md](docs/benchmarks.md) — the benchmark ledger (backs every claim)
- [integrations.md](docs/integrations.md) — supported providers and how config enables them
- [glossary.md](docs/glossary.md) — canonical terminology; use these meanings exactly
- [references.md](docs/references.md) — external evidence sources backing every sourced claim
- [archive-format.md](docs/archive-format.md) — portability archive specification (D11)
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — contributor rules
- [CHANGELOG.md](docs/CHANGELOG.md) — Keep a Changelog + SemVer
- [docs/adr/](docs/adr/) — Architecture Decision Records (MADR)

## DO NOT

- Do NOT write implementation code (documentation phase; Gate C requires explicit authorization).
- Do NOT create more than one todo file.
- Do NOT add Claude/AI attribution or trailers to commits.
- Do NOT use multi-line commit messages.
- Do NOT add inline/explanatory comments except the two allowed cases (`# CRITICAL:`; `# TODO:` / `# BLOCKED:`).
- Do NOT build any GUI for controlling the RAG (dashboard is observability-only).
- Do NOT store secrets in `config.yaml` (secrets live in `.env` only).
- Do NOT make application-code changes as part of enabling an integration toggle (provisioning is config-only).
- Do NOT publish an unmeasured performance or uniqueness claim anywhere.
- Do NOT let Autopilot auto-apply configuration changes (suggest-only, human approves).
- Do NOT swallow exceptions, use bare `except`, or return generic 500s without a problem+json body.
- Do NOT skip, merge, or reorder gates (A: audit → B: doc hardening → C: build slices).
