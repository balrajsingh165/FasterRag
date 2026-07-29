# Changelog

All notable changes to fasterRag are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). Entries are reverse-chronological; dates are `YYYY-MM-DD`. Sections used per release: Added / Changed / Deprecated / Removed / Fixed / Security.

## [Unreleased]

### Added

- Complete beta documentation set under `docs/` (2026-07-29): scope, structure, flow, architecture, config reference, API reference (RFC 9457 error model), CLI reference, Python package surface (`python-api.md`), observability (Langfuse/Grafana auto-provisioning), deployment (incl. revert playbook), security (incl. supply chain), testing strategy (full pyramid incl. chaos suite), performance methodology, benchmark ledger, integrations matrix.
- Differentiation and reliability layer (2026-07-29): `differentiators.md` (twelve flagship capabilities D1–D12 with acceptance tests and proof metrics), `reliability.md` (error taxonomy, resilience patterns), `failure-modes.md` (37-row FMEA), `slo.md` (SLIs, TBD-until-measured targets, error budget policy), `disaster-recovery.md` (backup inventory, restore drill, RPO/RTO).
- Architecture Decision Records ADR-0001–ADR-0006 (MADR style).
- `CLAUDE.md` instruction file, single universal task file `docs/todo.md`, detailed `README.md`.

### Changed

- Documentation relocated into `docs/` with only `CLAUDE.md` and `README.md` at repository root (2026-07-29).

### Deprecated

- _(none)_

### Removed

- _(none)_

### Fixed

- _(none)_

### Security

- Documented secrets policy (`.env`-only, env-var-name references), supply-chain requirements (hash-locked deps, secret scanning, non-root containers, SBOM per release).

---

Release history begins when the first beta (`0.1.0-beta.1`) ships from the build phase; each build slice updates **Unreleased** in the same change that lands it.
