# CLAUDE.md

Always-loaded instructions for Claude Code working in the fasterRag repository. Keep this file dense; detail lives in the linked docs.

## Project overview

fasterRag is a FastAPI-based, backend-only, one-stop Retrieval-Augmented Generation (RAG) solution engineered for very, very large datasets. This repository ships the first beta of an all-in-one RAG system that aims to be the fastest, most efficient, and most optimized RAG project available. Speed comes from multi-worker parallel processing across ingestion, chunking, embedding, and indexing; maximum chunking quality via a configurable best-in-class chunking pipeline; and aggressive caching, batching, streaming, and async I/O. Everything is pluggable — any vector database, any embedding model, any LLM provider — selected purely through configuration. The system is operated ONLY through a REST API and a terminal/CLI; a separate, optional, self-hosted observability dashboard exists purely for inspection (Langfuse-like), never for control.

**Current repository state: documentation only.** No implementation code exists yet. Do not create implementation code unless a task in [todo.md](todo.md) explicitly authorizes the build phase.

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
- Typecheck: `mypy src/`
- Test: `pytest` (unit) / `pytest -m integration` (integration) / `pytest -m eval` (retrieval eval harness)
- Run dev API: `uvicorn fasterrag.api.main:app --reload`
- Run workers: `python -m fasterrag.workers` (spawns CPU + embedding pools per `config.yaml`)

## Coding standards

- **Docstrings only.** NO inline comments and NO explanatory comments in code. Comments are permitted in EXACTLY two cases:
  - Super-critical flag: `# CRITICAL: <why this must not change>`
  - Blocker/pending marker: `# TODO: <what remains>` or `# BLOCKED: <blocker + ticket/date>`
- Every module, class, and public function gets a docstring. Docstrings explain what and why; code explains how.
- Full type hints everywhere; `mypy --strict` must pass.
- Async-first: all I/O paths are `async`; CPU-bound work goes to worker pools, never the event loop.
- Routers contain zero business logic; services orchestrate; adapters isolate vendors. See [structure.md](structure.md).

## Git standards

- **Single-line commit messages only.** No multi-line bodies. No trailers of any kind.
- **Absolutely NO Claude/AI attribution in commits** — no `Co-Authored-By: Claude`, no `Generated with Claude Code`, no AI signatures.
- Feature-branch workflow. **Never commit directly to `main`.**
- Commit frequently — small, coherent, revertable commits.

## Incremental shipping discipline

- Ship small OR large features continuously, committing in the middle of work, so the repo can always be reverted (`git revert` / `git checkout <sha>`) if a change goes bad.
- Every commit must leave documentation (and later, code) in a coherent state.
- Prefer many small PRs from feature branches over one large merge.

## Task tracking

- [todo.md](todo.md) is the ONLY task file. Never create any other todo/task/tracking file. When completing a task, tick it and append `— ✅ YYYY-MM-DD`; ticked entries are append-only.

## Folder boundaries (sensitive — change only with explicit task authorization)

- `config/` loaders and schema — validation contract for the whole system; fail-fast behavior must not be weakened.
- `adapters/` — vendor isolation boundary; never leak vendor types outside an adapter.
- Provisioning code (Langfuse/Grafana/Qdrant auto-setup) — must remain config-driven with zero application-code changes at toggle time.
- `.env`, `.env.*` — never read, print, or commit secret values.

## Pointers

- [scope.md](scope.md) — vision, goals, non-goals, pain-point catalogue
- [structure.md](structure.md) — repository layout and layer responsibilities
- [flow.md](flow.md) — end-to-end Mermaid flows
- [architecture.md](architecture.md) — components, workers, adapters, scaling, fault tolerance
- [config-reference.md](config-reference.md) — full `config.yaml` schema
- [api-reference.md](api-reference.md) — REST endpoints
- [cli-reference.md](cli-reference.md) — terminal commands
- [observability.md](observability.md) — dashboard, metrics, Langfuse/Grafana provisioning
- [deployment.md](deployment.md) — self-hosting, Docker, sizing
- [security.md](security.md) — secrets, auth, multi-tenancy
- [testing-strategy.md](testing-strategy.md) — tests, eval harness, CI gates
- [performance.md](performance.md) — benchmark targets and methodology
- [integrations.md](integrations.md) — supported providers and how config enables them
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor rules
- [CHANGELOG.md](CHANGELOG.md) — Keep a Changelog + SemVer
- [docs/adr/](docs/adr/) — Architecture Decision Records (MADR)

## DO NOT

- Do NOT write implementation code (documentation phase).
- Do NOT create more than one todo file.
- Do NOT add Claude/AI attribution or trailers to commits.
- Do NOT use multi-line commit messages.
- Do NOT add inline/explanatory comments except the two allowed cases (`# CRITICAL:`; `# TODO:` / `# BLOCKED:`).
- Do NOT build any GUI for controlling the RAG (dashboard is observability-only).
- Do NOT store secrets in `config.yaml` (secrets live in `.env` only).
- Do NOT make application-code changes as part of enabling an integration toggle (provisioning is config-only).
