# CONTRIBUTING.md — Contributor Rules

These rules are non-negotiable and enforced by review, pre-commit hooks, and CI. They restate the permanent constraints from [CLAUDE.md](../CLAUDE.md); on any conflict, CLAUDE.md wins.

## 0. Local setup

```bash
pip install -e ".[dev]"        # tooling: ruff, mypy, pytest, pre-commit
pre-commit install             # installs both the pre-commit and commit-msg hooks
```

One `install` covers both stages: `.pre-commit-config.yaml` sets `default_install_hook_types`, so the commit-message rule is wired without a second command.

The hooks run ruff, `ruff format`, `mypy --strict`, and the **fast tests only** — `pytest -m "not integration and not eval"`. Integration needs a Docker daemon and eval downloads model weights; gating every commit on either would make committing impossible on a laptop with Docker stopped. CI runs those as separate jobs ([testing-strategy.md](testing-strategy.md) §2), so a clean pre-commit run means the fast gates agree, not that the whole pyramid passed.

`pre-commit run --all-files` runs every hook over the whole tree — worth doing after a rebase or a dependency bump.

Optional extras, needed only for what they name: `.[huggingface]` and `.[rerank]` for local models, `.[openai]` / `.[anthropic]` / `.[cohere]` / `.[ollama]` for hosted providers, `.[ocr]` for scanned PDFs.

## 1. Commit rules

- **Conventional Commits, single line.** The subject is `<type>(<optional scope>): <description>`, at most 100 characters, with no body. Use `<type>!:` for a breaking change.
- **Allowed types**: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`.
- The type prefix makes history machine-readable — release notes and SemVer bumps derive from it instead of being hand-curated — while the single-line rule keeps a log skimmable.
- **No trailers of any kind.** Absolutely no AI attribution: no `Co-Authored-By: Claude`, no `Generated with Claude Code`, no AI signatures.
- Write imperative descriptions: `feat(cache): invalidate semantic entries on corpus change`.
- Enforced by `scripts/check_commit_message.py`, which runs as a `commit-msg` hook locally and over the pull-request range in CI.
- **Trunk-based development: all work lands on `main`.** No feature branches (maintainer instruction, 2026-07-30). Because there is no branch to hide behind, every commit must leave `main` green: lint, types, and tests pass before you commit.
- Commit frequently — small, coherent, revertable commits; the repo must always be revertible via `git revert`.

## 2. Comment rules (code)

- **Docstrings only.** Every module, class, and public function gets a docstring explaining what and why.
- NO inline comments and NO explanatory comments, with EXACTLY two permitted exceptions:
  - `# CRITICAL: <why this must not change>` — a super-critical requirement flagged directly in code.
  - `# TODO: <what remains>` / `# BLOCKED: <blocker + ticket/date>` — explicit pending/blocker markers.
- Any `# type: ignore` requires an adjacent `# CRITICAL:` justification.

## 3. One-todo-file rule

- [docs/todo.md](todo.md) is the ONLY task file in this repository. Never create any other todo/task/tracking file (no `TASKS.md`, no `ROADMAP-todo.md`, no per-module todos).
- Entry format: `- [ ] TASK-0001: <description>`; on completion: `- [x] TASK-0001: <description> — ✅ YYYY-MM-DD`.
- Ticked entries are **frozen** — append-only, never edited afterwards. Task IDs are sequential and never reused.

## 4. Incremental-shipping rule

- Ship small OR large features continuously with frequent commits — including mid-work — so the repo can always be reverted if a change goes bad.
- No big-bang changes: no increment may exceed a reviewable size.
- Build-phase slices end with a git tag (`v0.x.0-sN`); the revert playbook lives in [deployment.md](deployment.md).

## 5. Provable-claims rule

- **A claim without a measurement is a bug.** Any performance/superiority statement in docs, code, or release notes must link to a [benchmark ledger](benchmarks.md) entry (claim, method, dataset, hardware, date, numbers, commit hash) — otherwise phrase it as a goal.
- "Fastest" only ever against a named baseline we measured ourselves with a harness committed to this repo.

## 6. Reliability rules

- Typed error taxonomy required ([reliability.md](reliability.md)); **no bare `except`, no silently swallowed exceptions**; API errors are RFC 9457 problem+json with a stable `code`.
- Every external call has an explicit timeout; retries only on `retryable` errors; risky features ship behind config flags defaulting to `false`.
- Config drives all behavior (`config.yaml`); secrets live ONLY in `.env`, referenced by env-var name. Every integration toggle defaults to `false`.
- Control plane = REST API + CLI + Python library. The dashboard is observability-only — a PR adding any control capability to it will be rejected.

## 7. Quality gates for every PR

Lint + format clean (ruff) · `mypy --strict` zero errors · unit + integration tests green · coverage ≥ 85% on touched `core/`/`adapters/`/`workers/` packages · eval regression gate green when retrieval paths change · secret scan clean · docs updated in the same PR as any behavior change · [todo.md](todo.md) updated and ticked with dates.

## 8. Adding a provider adapter

Implement the base class, register the entry point ([python-api.md](python-api.md) §Extending), and pass the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). Adapters that don't pass the contract suite are not merged.

## 9. ADRs

Architecturally significant decisions get a new MADR-style record in [docs/adr/](adr/), sequentially numbered (`ADR-XXXX-*.md`). ADRs are never deleted; superseded ADRs get status `Superseded by ADR-XXXX`.
