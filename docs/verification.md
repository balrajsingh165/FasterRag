# verification.md — Acceptance Verification Procedure

How to prove, on any machine, that a checkout (or a rebuild per the [rebuild-playbook](rebuild-playbook.md)) actually is the system the documentation describes. Run the tiers in order; each assumes the previous passed. Expected outputs marked *(2026-08-10)* are dated observations — counts grow as the ledger moves; a **deviation in kind** (a failing gate, a missing capability) is a finding, a deviation in count is not.

> **This is a procedure, not a tracker.** The one task file is [todo.md](todo.md); §7 below is a read-only cross-reference of open verification work, citing existing `TASK-` ids only — same contract as [blockers.md](blockers.md). Any deviation this procedure finds is filed **in todo.md**, then optionally reflected here.

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
| `python scripts/check_doc_truth.py` · `check_blockers.py` · `check_openapi_drift.py` | all silent/zero |
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
| 9 | D11 archive | `export --out a.fragx` → `import a.fragx --target-collection copy` | Round trip; re-export byte-identical; tampered checksum refuses before any write |
| 10 | D6 autopilot | `fasterrag autopilot run --budget-minutes 5` | Suggested fragment + measured deltas; `config.yaml` byte-identical after |
| 11 | Security | enable `security.auth` + `multi_tenancy`; call with wrong scope/tenant | 401/403 problems with stable codes; tenant B cannot read tenant A's collections, traces, or cache |
| 12 | Degradation | stop Qdrant mid-session; query | Typed retryable failure now (`cache_only` rung is TASK-0159); breaker metric transitions; recovery on restart |
| 13 | Backup/DR | `fasterrag backup` → delete collection → `restore` | §4 shortcut restores; same query, same citation id (clean-host drill remains TASK-0085) |
| 14 | REST parity | repeat 1/3/9 over `curl` incl. SSE | Same behavior; SSE order `meta→token…→citations→usage→done`; missing `done` = incomplete |
| 15 | Facade parity | repeat 1/3 via `FasterRag` and `fasterrag.sync` | Same results, same error codes |

## Tier 5 — Observability & provisioning

`observability.dashboard: true` → read-only UI on :8080, auth + tenant-scoped, zero mutating routes. `fasterrag provision langfuse` → six containers, `http://<host>:3000` healthy, bootstrapped keys authenticate, traces export (207-partial counts as failure). `fasterrag provision grafana` → provisioned datasource returns real series; panels move under ingest/query load. `observability.otel: true` → spans **and** metric catalogue arrive at the collector with the propagated trace id. *(REST-side provisioning of langfuse/grafana is TASK-0251 — verify via CLI until it lands.)*

## Tier 6 — Packaging

`python -m build` → wheel + sdist; `twine check dist/*` passes; wheel installs into a clean venv; `fasterrag config init && fasterrag doctor` runs from it; the packaged `config.yaml` is byte-identical to the repo's.

## 7. Open-verification cross-reference (read-only; ids live in todo.md)

**Cannot pass yet, by design** — measurement program blocked on isolated hardware: citable benchmarks + SLO targets (TASK-0084), chaos recovery times (TASK-0136), clean-host DR drill + RPO/RTO (TASK-0085). **Known gaps this procedure will flag until closed**: coverage reconciliation (TASK-0252), REST provisioning parity (TASK-0251), `cache_only` rung (TASK-0159) and LLM-breaker consultation (TASK-0245), cost governor build-or-cut (TASK-0242), D7 gate in CI (TASK-0244), disk-full typing (TASK-0234), missing chaos/reindex scenarios (TASK-0241), load/soak/mutation layers (TASK-0243), nightly regression comparison (TASK-0246), live-provider eval checks (TASK-0205), D11 cross-backend acceptance + facade wrap (TASK-0079), `--watch`/`--fix` (TASK-0196/0197). **Release-decision items** are listed in [release.md](release.md), not here.
