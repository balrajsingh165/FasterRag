# ADR-0003: config.yaml + .env Split (Behavior vs Secrets)

- Status: accepted
- Date: 2026-07-29
- Deciders: fasterRag maintainers

## Context and Problem Statement

The framework is config-driven: one file must describe all behavior, including flipping entire subsystems on (`langfuse: true`). Where do credentials live, and how is configuration loaded and validated?

## Decision Drivers

- Secrets must never land in version control; `config.yaml` should be safely committable.
- 12-factor separation of config and credentials.
- Fail-fast: a misconfigured process must never serve.
- One loading mechanism for API, CLI, and the importable Python package.

## Considered Options

1. **`config.yaml` for all behavior + `.env` for secrets only, loaded via pydantic-settings (YAML source + env source), secrets referenced by env-var name (`api_key_env: OPENAI_API_KEY`)**
2. Single YAML containing secrets
3. Environment variables for everything
4. Dynamic config service / control API

## Decision Outcome

**Chosen: option 1.** pydantic-settings composes a YAML settings source (behavior) with env/`.env` (secrets). The schema validates everything at startup and fails fast with a clear error naming the offending key; missing referenced env vars are fatal and named without logging values. This decouples application logic from the config source and enforces 12-factor separation. Every integration toggle defaults to `false`.

### Consequences

- Good: `config.yaml` is committable and diffable — config rollback is git revert; secrets rotate without touching config.
- Good: one validated `Settings` object serves API, CLI, and library identically; the whole config surface is documentable in one reference ([config-reference.md](../config-reference.md)).
- Bad: two files to manage instead of one; mitigated by `.env.example` and doctor's missing-var checks with fix-it hints.
- Bad: no runtime mutation of config by design (restart or provision command required) — accepted; it keeps the control plane auditable.
