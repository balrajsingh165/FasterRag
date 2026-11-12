# verification.md — Acceptance Verification Procedure

How to prove, on any machine, that a checkout (or a rebuild per the [rebuild-playbook](rebuild-playbook.md)) actually is the system the documentation describes. Run the tiers in order, starting at tier −1; each assumes the previous passed. Expected outputs marked *(2026-08-10)* are dated observations — counts grow as the ledger moves; a **deviation in kind** (a failing gate, a missing capability) is a finding, a deviation in count is not. The dated observations below were recorded before tier −1 existed and therefore carry a date but no tree state; **every observation added from 2026-11-11 carries the provenance line**, and one that does not is incomplete rather than merely terse.

> **This is a procedure, not a tracker.** The one task file is [todo.md](todo.md); §7 below is a read-only cross-reference of open verification work, citing existing `TASK-` ids only — same contract as [blockers.md](blockers.md). Any deviation this procedure finds is filed **in todo.md**, then optionally reflected here.

## Tier −1 — State the tree, before any number is recorded

**A number without its tree state is not a measurement.** Every count this procedure produces — tests passed, files typed, coverage percent, seconds elapsed — is evidence about *one tree at one commit*, and printed without saying which tree it is an anecdote. Any run whose numbers are written into a document, a ledger entry, a task note, or a report **must paste the provenance line beside them**:

```bash
pytest -q -m "not integration and not eval"
```

prints, beside the counts and regardless of `-q`:

```
measured: 882a07e | working tree: clean | D:\Github\FasterRag
```

and the unsuppressed header form, which is the one to paste when there is room for four lines:

```
checkout: D:\Github\FasterRag
commit: 882a07e9... (full sha)
working tree: clean
fasterrag imported from: D:\Github\FasterRag\src\fasterrag
```

Rules, in the order they bite:

1. **A recorded number without that line is incomplete**, and is to be treated as unverified until the run is repeated. This holds for every dated observation in this document and for every `Provenance:` field in [benchmarks.md](benchmarks.md) (ledger rule 7).
2. **`working tree:` must read `clean`** for any number that will be compared against another number — a count, a timing, a coverage figure, a release-readiness derivation. The instrument says so in capitals (`A NUMBER MEASURED HERE IS NOT REPRODUCIBLE`) because a dirty tree silently invalidates the commit beside it. A dirty-tree number may still be recorded; it may not be compared, cited, or used to close a task.
3. **`fasterrag imported from:` must sit inside the checkout above it.** When it does not, the run measured different code than the commit names — a second worktree, a stale `site-packages`, an editable install pointing elsewhere. This has happened in this repository.

**Why this is a tier and not a footnote.** TASK-0322 spent an entire session establishing whether a discrepancy between two test runs was a defect or two different trees. The question was unanswerable from either log, because neither log said what it had measured; the session's whole cost was reconstruction. Both logs would have answered it in seconds had they carried these four lines. The instrument in `tests/conftest.py` exists because of that session, and has twice since caught a measurement that was about to be recorded as if it were reproducible. The rule is written here with its cause attached so that a later reader can see what it is paying for, rather than deleting a line that looks like noise.

## Tier 0 — Environment preflight

```bash
python --version                # 3.12+
pip install -e ".[dev]" && pre-commit install
fasterrag config validate       # exit 0 on the canonical config
fasterrag doctor                # every failed check must print a concrete fix
```

Doctor failures about Docker are acceptable for tiers 0–2 (nothing below tier 3 needs it) — every other check must pass.

## Tier 1 — Static gates

| Command | Expected |
|---|---|
| `ruff check .` and `ruff format --check .` | clean *(2026-08-10: 0 errors, 246+ files formatted)* |
| `mypy` | zero errors, strict, over `src`+`tests`+`scripts` *(308 files)* |
| `uv lock --check` | lockfile current |
| `python scripts/check_commit_message.py --range origin/main~20..origin/main` | all pass |
| `python scripts/check_doc_truth.py` · `check_blockers.py` · `check_openapi_drift.py` · `check_claims.py` · `stamp_release.py --check` | all silent/zero |
| `python scripts/check_package.py` | `package ok: ...` — builds the wheel, installs it into a clean venv, and imports it as a consumer would *(2026-10-29: 95 s on a Windows dev machine, Python 3.14)*. Since TASK-0444 it then adds `[chroma,huggingface]` and runs the local profile end to end (ingest a Markdown file, answer a `retrieve`, no service, no key); `--skip-local-profile` stops after the core checks |
| `pytest -q tests/unit/docs tests/unit/scripts` | green — the reference-parity, ledger-integrity, and blockers-numbering gates |

## Tier 2 — Unit, property, golden (no Docker, no network, no credentials)

```bash
pytest -q -m "not integration and not eval"
```

Expected: green *(2026-08-10: 2,060 passed, 6 skipped, 103 s on a dev laptop)*. This tier includes the Hypothesis chunker invariants, RRF/BM25 property tests, golden-file parser snapshots, the pinned route→scope table, the metrics declared-never-written pin, and the seam-level chaos scenarios. Then the coverage gate as CI runs it:

```bash
pytest -m "not integration and not eval" --cov=src/fasterrag/core --cov=src/fasterrag/adapters --cov=src/fasterrag/workers --cov-fail-under=85
```

Expected: pass. *(Known open: measured 83% on a Windows dev machine 2026-08-10 — TASK-0252; until reconciled, treat a sub-85 local result on Windows as "reconciliation pending", and a sub-85 result on Linux as a hard failure.)*

## Tier 3 — Integration (Docker required)

```bash
pytest -q -m integration
```

Covers: the shared vector-DB contract suite against real Qdrant (all three modes, auth on/off, both transports) **and** real pgvector; ingest→query round trips; journal resume; DLQ routing; redis cache backend; real-container-stop and real-`ENOSPC` chaos variants; the archive round trip over REST. Also run the model tier once (downloads weights): `pytest -q -m eval` — real embedder/reranker checks plus the handbook retrieval baseline (recall@5 1.0 / MRR 1.0 on the committed fixture).

## Tier 4 — Capability walkthrough (live stack, one terminal)

Prereq: `fasterrag provision qdrant` after doctor passes. Run in order; each row's observable is the acceptance.

| # | Capability | Do | Must observe |
|---|---|---|---|
| 1 | D3 ingestion | `fasterrag ingest ./docs --recursive` twice | First run indexes; second reports 100% deduplicated, zero new vectors |
| 2 | D3 sources | ingest one `url` and one `inline` source | Both index; document ids stable across re-runs |
| 3 | Retrieval + D4 | `fasterrag query "..." --show-chunks --show-timings` | Chunks carry both leg ranks + rrf score; timings split per stage; `mode: full` |
| 4 | D5 grounding | enable `generation.grounded_or_refuse`, ask something unanswerable | Structured `insufficient_evidence`, no fabricated answer |
| 5 | D1 lockfile | `fasterrag index lock verify`; change `chunking.chunk_size`; verify again | Exit 0, then exit 1 naming `config_hash` and `chunk_size` with both values |
| 6 | D2 reindex | `fasterrag index reembed <col> --dataset tests/eval/datasets/handbook`, then `index rollback` | Swap under load without failed queries; rollback restores the previous answer + citation id |
| 7 | D8 replay | `fasterrag traces list` → `replay --trace <id> --config candidate.yaml` | Identical config ⇒ identical retrieval set; a changed key ⇒ structured chunk/rank diff naming the key |
| 8 | D9 estimator | `fasterrag estimate ./docs --all-providers` | Token counts + priced providers, each price carrying source+date; with `cost.estimator: false` every estimate surface refuses |
| 9 | D11 archive | `export --out a.fragx` → `import a.fragx --target-collection copy` | Round trip; re-export identical in every data member (`manifest.json` carries `created_at`, and since TASK-0329 `checksums.sha256` carries its digest, so those two differ); tampered checksum refuses before any write |
| 10 | D6 autopilot | `fasterrag autopilot run --budget-minutes 5` | Suggested fragment + measured deltas; `config.yaml` byte-identical after |
| 11 | Security | enable `security.auth` + `multi_tenancy`; call with wrong scope/tenant | 401/403 problems with stable codes; tenant B cannot read tenant A's collections, traces, or cache |
| 12 | Degradation | stop Qdrant mid-session; query | Typed retryable failure now (`cache_only` rung is TASK-0159); breaker metric transitions; recovery on restart |
| 13 | Backup/DR | `fasterrag backup` → delete collection → `restore` | §4 shortcut restores; same query, same citation id (clean-host drill remains TASK-0085) |
| 14 | REST parity | repeat 1/3/9 over `curl` incl. SSE | Same behavior; SSE order `meta→token…→citations→usage→done`; missing `done` = incomplete |
| 15 | Facade parity | repeat 1/3 via `FasterRag` and `fasterrag.sync` | Same results, same error codes |
| 16 | D9 governor | set `cost.per_query_token_budget: 1`, then query | 402 `BUDGET_EXCEEDED` problem, non-retryable, refused *before* the provider is called — never absorbed into a degraded `extractive` answer |
| 17 | `--watch` | `fasterrag ingest ./docs --recursive --watch`, then the same with `--json` | Job id printed before the work starts, then appended `progress` lines as the tally moves, final counts unchanged from a run without the flag; under `--json` a refusal on stderr and an identical JSON document |

## Tier 5 — Observability & provisioning

`observability.dashboard: true` → read-only UI on :8080, auth + tenant-scoped, zero mutating routes. `fasterrag provision langfuse` → six containers, `http://<host>:3000` healthy, bootstrapped keys authenticate, traces export (207-partial counts as failure). `fasterrag provision grafana` → provisioned datasource returns real series; panels move under ingest/query load. `observability.otel: true` → spans **and** metric catalogue arrive at the collector with the propagated trace id. *(REST-side provisioning of langfuse/grafana shipped 2026-08-11, TASK-0251 ✅ — verify over REST or CLI.)*

## Tier 6 — Packaging

`python -m build` → wheel + sdist; `twine check dist/*` passes; wheel installs into a clean venv; `fasterrag config init && fasterrag doctor` runs from it; the packaged `config.yaml` is byte-identical to the repo's.

## 7. Open-verification cross-reference (read-only; ids live in todo.md)

**Cannot pass yet, by design** — measurement program blocked on isolated hardware: citable benchmarks + SLO targets (TASK-0084), chaos recovery times (TASK-0136), clean-host DR drill + RPO/RTO (TASK-0085). **Known gaps this procedure will flag until closed**: coverage reconciliation (TASK-0252), REST provisioning parity (TASK-0251), `cache_only` rung (TASK-0159) and LLM-breaker consultation (TASK-0245), D7 gate in CI (TASK-0244), disk-full typing (TASK-0234), the 24 h soak run and every load measurement, both needing TASK-0084's hardware — the layers themselves are built (TASK-0243), nightly regression comparison (TASK-0246), live-provider eval checks (TASK-0205), D11 cross-backend acceptance + facade wrap (TASK-0079). The cost governor (TASK-0242) and `ingest`/`index reembed --watch` (TASK-0196) are **no longer gaps** — both shipped, and Tier 4 rows 16 and 17 were added to exercise them rather than leave them unverified. **Release-decision items** are listed in [release.md](release.md), not here.
