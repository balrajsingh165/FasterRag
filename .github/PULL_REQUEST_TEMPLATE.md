# Pull Request

## What & why

<!-- One or two sentences. Link the TASK-XXXX entries from docs/todo.md this PR advances. -->

Tasks: TASK-

## Enforcement checklist (all mandatory — reviewers reject on any unchecked box)

- [ ] **Commits**: every commit message is a single line; no bodies, no trailers, no AI attribution of any kind.
- [ ] **Branch**: this PR comes from a feature branch (never work directly on `main`).
- [ ] **One todo file**: `docs/todo.md` is the only task file touched; completed tasks are ticked with `— ✅ YYYY-MM-DD`; no previously-ticked entry was edited.
- [ ] **Provable claims**: no new unmeasured superlative or performance claim anywhere; new claims link to a `docs/benchmarks.md` ledger entry or are phrased as goals.
- [ ] **Docs in the same PR**: every behavior/spec change updates the affected docs here, not later.
- [ ] **Comment policy** (code PRs): docstrings only; inline comments limited to `# CRITICAL:` and `# TODO:` / `# BLOCKED:`.
- [ ] **Error policy** (code PRs): no bare `except`; typed taxonomy used; API errors are RFC 9457 problem+json with a stable `code`.
- [ ] **Secrets**: no credentials in `config.yaml`, code, tests, or fixtures; new secrets are documented in `.env.example` + `docs/security.md`.
- [ ] **Quality gates** (code PRs): ruff + mypy strict clean; unit/integration tests green; coverage ≥ 85% on touched `core/`/`adapters/`/`workers/`; eval regression gate green if retrieval paths changed.
- [ ] **Scope**: reviewable increment (no big-bang); risky features are behind config flags defaulting to `false`.

## How it was verified

<!-- Tests run, docs cross-checked, or the ledger entry produced. "Not verified" is a reason to not merge. -->
