# CONTRIBUTING.md — Contributor Rules

These rules are non-negotiable and enforced by review, pre-commit hooks, and CI. They restate the permanent constraints from [CLAUDE.md](../CLAUDE.md); on any conflict, CLAUDE.md wins.

## 0. Local setup

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # tooling: ruff, mypy, pytest, pre-commit
pre-commit install             # installs both the pre-commit and commit-msg hooks
```

One `install` covers both stages: `.pre-commit-config.yaml` sets `default_install_hook_types`, so the commit-message rule is wired without a second command.

**Commit with the environment active.** The `mypy` and `pytest` hooks are `language: system` — they deliberately run the project's own interpreter rather than a copy pre-commit builds, because `mypy --strict` has to see the installed package and its stubs. Committing from a shell where `.venv` is not on `PATH` fails them with `Executable not found` or a wall of `ModuleNotFoundError`; activate the environment rather than reaching for `--no-verify`.

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

- **Docstrings only, and short.** Every module, class, and public function gets a docstring explaining what and why. A function, method or class docstring is **at most four lines**, and one is usually enough. Module docstrings are exempt: they are the file's documentation and what [internals.md](internals.md) points readers at.
- **The reasoning goes in [code-notes.md](code-notes.md), not in the function header.** Why a thing is the way it is — the trap it avoids, the alternative rejected, the incident it came from — is worth keeping and worth reading, but a reader scanning a module should see the code, not three paragraphs before each function. `scripts/trim_docstrings.py` moves overflow there, addressed by `module::qualname`. It rewrites via the AST and refuses any file whose code structure changes, because an earlier regex pass over docstrings split sentences mid-line and had to be reverted.
- NO inline comments and NO explanatory comments, with EXACTLY two permitted exceptions:
  - `# CRITICAL: <why this must not change>` — a super-critical requirement flagged directly in code.
  - `# TODO: <what remains>` / `# BLOCKED: <blocker + ticket/date>` — explicit pending/blocker markers.
- A marker may open a **block**: `#` lines immediately following a `# CRITICAL:`/`# TODO:`/`# BLOCKED:` line continue that marker and count as part of it (this codifies the existing practice in `src/`). A marker sharing a line with code opens nothing — only a standalone one introduces a block.
- `scripts/check_comments.py` enforces all of the above over `src/` and `scripts/` in CI (not pre-commit), reading tokens so a `#` inside a string literal never matches. Tool directives — `# noqa`, `# type: ignore`, `# pragma:`, `# fmt:`, `# ruff:`, `# mypy:`, `# isort:` — need no marker: they address a tool rather than a reader and have nowhere else to live. A shebang and a PEP 263 encoding line are not comments for this purpose. `tests/` is out of scope, deliberately: the comments there label fixtures and scenarios rather than explaining production behaviour.
- Any `# type: ignore` requires an adjacent `# CRITICAL:` justification. This is checked under `src/` and `scripts/` only; the ~141 in `tests/` are out of scope by the same reasoning (see TASK-0319).
- A `# TODO:`/`# BLOCKED:` marker may only cite an **open** ledger entry. The marker says work remains and a ticked entry says it is finished, so the pair is always wrong one way or the other; `scripts/check_task_ids.py` fails on it. A `# CRITICAL:` block or a docstring may cite a completed task freely — that is history, correctly attributed.

## 3. One-todo-file rule

- [docs/todo.md](todo.md) is the ONLY task file in this repository. Never create any other todo/task/tracking file (no `TASKS.md`, no `ROADMAP-todo.md`, no per-module todos).
- Entry format: `- [ ] TASK-0001: <description>`; on completion: `- [x] TASK-0001: <description> — ✅ YYYY-MM-DD`.
- Ticked entries are **frozen** — append-only, never edited afterwards. Task IDs are sequential and never reused.
- Every `TASK-####` written in a comment or docstring under `src/`, `tests/` or `scripts/` must resolve to an entry in todo.md. A dangling id is unreadable: whoever greps for it cannot tell whether the task was renumbered, never filed, or imagined. `scripts/check_task_ids.py` enforces this in CI (not pre-commit).

## 4. Incremental-shipping rule

- Ship small OR large features continuously with frequent commits — including mid-work — so the repo can always be reverted if a change goes bad.
- No big-bang changes: no increment may exceed a reviewable size.
- Build-phase slices end with a git tag (`v0.x.0-sN`); the revert playbook lives in [deployment.md](deployment.md).

## 4a. Verifying a suspected defect

**Check the instrument before filing the bug.** Two entries in [todo.md](todo.md) were opened against fasterRag and closed as measurement errors — TASK-0153 (a "hang" that was a shell timeout summing three bounded `curl -m 120` calls) and TASK-0177 (a "parser encoding bug" that was `json.load(sys.stdin)` decoding UTF-8 with the Windows console codepage). Both cost more to investigate than the check would have.

Windows traps that produce convincing false positives:

- **`sys.stdin` and `sys.stdout` use the console codepage, not UTF-8.** Piping JSON into `python -c` mangles every non-ASCII byte before your code sees it. Read the bytes and `.decode("utf-8")` explicitly, or use `urllib` instead of a pipe.
- **A terminal that cannot render a codepoint prints `?`.** That is a font and codepage artifact, not data. Compare `ord(c)` or `repr()`, never the rendered glyph.
- **A shell timeout is not an application hang.** Time the single operation before concluding anything about the code.

Narrow to the smallest layer that still reproduces — parse alone, adapter round-trip alone — before filing. A defect that survives that is real; one that does not was never in the code.

## 5. Provable-claims rule

- **A claim without a measurement is a bug.** Any performance/superiority statement in docs, code, or release notes must link to a [benchmark ledger](benchmarks.md) entry (claim, method, dataset, hardware, date, numbers, commit hash) — otherwise phrase it as a goal.
- "Fastest" only ever against a named baseline we measured ourselves with a harness committed to this repo.

## 6. Reliability rules

- Typed error taxonomy required ([reliability.md](reliability.md)); **no bare `except`, no silently swallowed exceptions**; API errors are RFC 9457 problem+json with a stable `code`.
- Every external call has an explicit timeout; retries only on `retryable` errors; risky features ship behind config flags defaulting to `false`.
- Config drives all behavior (`config.yaml`); secrets live ONLY in `.env`, referenced by env-var name. Every integration toggle defaults to `false`.
- Control plane = REST API + CLI + Python library. The dashboard is observability-only — a PR adding any control capability to it will be rejected.

## 7. Quality gates for every PR

Lint + format clean (ruff) · `mypy --strict` zero errors · unit + integration tests green · coverage ≥ 85% on touched `core/`/`adapters/`/`workers/` packages · eval regression gate green when retrieval paths change · secret scan clean · every `TASK-` id cited in code resolves (`scripts/check_task_ids.py`) · docs updated in the same PR as any behavior change · [todo.md](todo.md) updated and ticked with dates.

The supply-chain audit is defined **once**, as the composite action `.github/actions/dependency-audit`, which both `ci.yml` and `security.yml` call. An accepted advisory ignore is added there and nowhere else: when the same list lived in both workflows, four ignores went into one copy and the scheduled run was red every night until somebody noticed.

## 8. Adding a provider adapter

Implement the base class, register the entry point ([python-api.md](python-api.md) §Extending), and pass the shared adapter contract suite ([testing-strategy.md](testing-strategy.md) §1.5). Adapters that don't pass the contract suite are not merged.

## 9. ADRs

Architecturally significant decisions get a new MADR-style record in [docs/adr/](adr/), sequentially numbered (`ADR-XXXX-*.md`). ADRs are never deleted; superseded ADRs get status `Superseded by ADR-XXXX`.
