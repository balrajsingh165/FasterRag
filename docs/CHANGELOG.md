# Changelog

All notable changes to fasterRag are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). Entries are reverse-chronological; dates are `YYYY-MM-DD`. Sections used per release: Added / Changed / Deprecated / Removed / Fixed / Security.

## [Unreleased]

### Added

- Complete beta documentation set under `docs/` (2026-07-29): scope, structure, flow, architecture, config reference, API reference (RFC 9457 error model), CLI reference, Python package surface (`python-api.md`), observability (Langfuse/Grafana auto-provisioning), deployment (incl. revert playbook), security (incl. supply chain), testing strategy (full pyramid incl. chaos suite), performance methodology, benchmark ledger, integrations matrix.
- Differentiation and reliability layer (2026-07-29): `differentiators.md` (twelve flagship capabilities D1–D12 with acceptance tests and proof metrics), `reliability.md` (error taxonomy, resilience patterns), `failure-modes.md` (37-row FMEA), `slo.md` (SLIs, TBD-until-measured targets, error budget policy), `disaster-recovery.md` (backup inventory, restore drill, RPO/RTO).
- Architecture Decision Records ADR-0001–ADR-0006 (MADR style).
- `CLAUDE.md` instruction file, single universal task file `docs/todo.md`, detailed `README.md`.
- Adoption guides and build-de-risking specs (2026-07-29): `docs/quickstart.md` (CLI/Python/REST walkthrough), `docs/cookbook.md` (nine composable configuration recipes), `docs/migration-guide.md` (concept mapping from other RAG frameworks, with explicit trade-offs), `docs/troubleshooting.md` (symptom → cause → fix), `docs/data-model.md` (canonical entity schemas, ID scheme, cross-entity invariants), and `docs/prompts.md` (the four LLM call-site contracts P1–P4, with versioning and override rules).
- Repository hardening (2026-07-29): `.gitignore`, `.env.example`, canonical default `config.yaml` (byte-consistent with the config-reference example), GitHub PR/issue templates enforcing the contribution rules, private vulnerability-reporting policy (`.github/SECURITY.md`), `docs/glossary.md` (canonical terminology), `docs/references.md` (external evidence bibliography R1–R14), `docs/archive-format.md` (D11 portability archive spec v1.0.0), and the golden-set JSONL schema in `docs/testing-strategy.md`.
- Build slice S2 — Qdrant adapter and doctor v1 (2026-07-30): vendor-neutral `VectorDBAdapter` contract with its request and result types and a vendor-neutral metadata-filter grammar (`fasterrag.adapters.vectordb.base`); the provider factory, including third-party registration through the `fasterrag.vectordb` entry point, with built-in names taking precedence (`factory`); the Qdrant reference adapter covering the docker, external, and remote modes, both the 6333 and 6334 paths, `prefer_grpc`, API-key auth, deterministic point-id mapping, filter push-down, and translation of every vendor failure into the typed taxonomy (`qdrant`); system-managed Docker provisioning that converges rather than reinstalls, publishes both ports, uses a named volume, and hands the server key to the container by variable name rather than on the command line (`services.provisioning`); `doctor` preflight diagnostics covering Python, disk, memory, GPU, Docker, secrets, per-port reachability, and backend health, each failure carrying a concrete fix (`services.doctor`); and the shared adapter contract suite run against a real Qdrant in every mode, wired into CI.
- Build slice S1 — skeleton (2026-07-29): the first implementation code. Packaging and tooling (`pyproject.toml` with ruff, `mypy --strict`, pytest and coverage config; `.pre-commit-config.yaml`; `.github/workflows/ci.yml` running the blocking gates; `scripts/check_commit_message.py` enforcing the single-line no-trailer commit rule). Typed error taxonomy (`fasterrag.errors`) with the stable error-code table and its RFC 9457 transport metadata. Structured JSON logging with OpenTelemetry-shaped correlation ids (`fasterrag.observability.logging`). Configuration schema and fail-fast loader (`fasterrag.config`) enforcing every key, bound, and cross-field rule in `docs/config-reference.md`, including the presence of referenced environment variables. API application factory with correlation middleware, problem-document exception handlers, and the `/healthz` and `/readyz` endpoints.

### Changed

- Documentation relocated into `docs/` with only `CLAUDE.md` and `README.md` at repository root (2026-07-29).
- `docs/api-reference.md` (2026-07-29): added the `NOT_READY` error code and specified the `/readyz` response shape, including the `dependencies[]` extension member — the endpoint's documented 503 problem body previously had no code to carry.
- `docs/structure.md` (2026-07-29): recorded the files slice S1 added and marked the paths that are still pending.

### Deprecated

- _(none)_

### Removed

- _(none)_

### Fixed

- _(none)_

### Security

- Documented secrets policy (`.env`-only, env-var-name references), supply-chain requirements (hash-locked deps, secret scanning, non-root containers, SBOM per release).
- Configuration validation never echoes offending input values, and validation failures are raised without chaining the underlying `ValidationError`, so a credential mistakenly pasted into `config.yaml` cannot reach logs or tracebacks (2026-07-29).
- Interactive API documentation (`/docs`, `/redoc`) is not served: it is a web interface capable of driving the RAG, and the control plane is programmatic only (ADR-0005). The machine-readable `/openapi.json` schema is still published (2026-07-29).
- The Qdrant server API key is exported into the provisioning subprocess environment and passed through with `-e NAME`, so the value never appears in `argv`, a process listing, or a log line. Vector-database authentication failures name the environment variable and are never retried (2026-07-30).

---

Release history begins when the first beta (`0.1.0-beta.1`) ships from the build phase; each build slice updates **Unreleased** in the same change that lands it.
