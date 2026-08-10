# rebuild-playbook.md — Reproducing fasterRag From Its Documentation

The test this document exists to pass: **an engineer with this `docs/` tree and no access to `src/` should be able to rebuild the system and arrive at the same behavior, in the same order, without rediscovering the traps we hit.** [structure.md](structure.md) says what goes where, [internals.md](internals.md) says how the finished code is organized; this file says **in what order to build it and what will go wrong**. Every trap below is real — each cost time once and is now encoded in code, config, tests, or gates.

## 0. Prerequisites

Python 3.12+ · git · Docker Desktop/Engine (integration + provisioning; the unit tier runs without it) · ~10 GB disk for models if using local embedding/rerank defaults (see the B6 note in [blockers.md](blockers.md)) · Windows or Linux (both are CI targets; two shipped defects were Windows-only, so do not develop blind to either). Tooling is pinned by `pyproject.toml` + `uv.lock`: `pip install -e ".[dev]"`, then `pre-commit install` (one command wires both hook stages).

## 1. Process before code (Gates A → B → C)

Reproduce the *process*, not just the artifact — the process is why the artifact is trustworthy:

1. **Rules first**: CLAUDE.md (comment policy, single-line Conventional Commits, one task file, provable-claims), the ledger (`docs/todo.md`), and the enforcement scripts (`scripts/check_commit_message.py`, later `check_doc_truth.py`, `check_blockers.py`, `check_openapi_drift.py`). A rule without a gate decays; every rule here got a gate.
2. **Contract docs before implementation** (Gate B): config-reference, api-reference, data-model, prompts, differentiators, reliability, failure-modes. The build then satisfies a reviewed spec instead of improvising one — several defects below were caught precisely because a doc predicted different behavior.
3. **Build in slices** (Gate C), each landing with its tests and doc updates in the same commit series, trunk-based, every commit green.

## 2. Build order (S1–S14) with the load-bearing dependencies

The order is not arbitrary; each row names what breaks if you reorder.

| Slice | Build | Why this position |
|---|---|---|
| S1 skeleton | config loader (fail-fast + cross-field rules + accepted-but-unenforced guard), error taxonomy + problem funnel, logging with trace ids, `/healthz`/`/readyz`, app factory | Everything downstream raises through the taxonomy and reads validated settings; retrofitting either is a rewrite |
| S2 Qdrant + doctor | adapter base **with the factory and shared contract suite on day one**, Qdrant impl (all three modes, both ports), docker provisioning (named volume), doctor | The contract suite must predate the second adapter or "any vector DB" is a hope; doctor must predate any other provisioner |
| S3 ingestion core | parsers (+golden files), chunkers (+Hypothesis invariants **before** more strategies), identity, journal/dedup/DLQ, both pools + bounded queue, indexer, estimator | Identity before indexer (T1); sparse-vector design decided before the indexer lands (ADR-0007) or the collection layout is wrong (T13) |
| S4 retrieval | parallel legs, RRF(k=60), filter push-down | Needs S3's sparse vectors |
| S5 rerank + eval v1 | cross-encoder (degrading, not failing), metrics harness, golden-set format, regression service | The eval harness must exist before S10's gated swap and S12 |
| S6 generation | context assembly, LLM adapters, SSE service, P1 citations, grounded-or-refuse + P3 | P3 as a **separate call** from P1 (prompts.md) — merging them lets the grader self-justify |
| S7 caches | embedding + semantic (tenant-scoped keys from the start), stats | Invalidation hooks into S3's settle path |
| S8 CLI + security | full CLI over the same services; auth middleware with scopes, rate limit, tenancy; archive read/write | Security as middleware, not per-route decoration — the pinned route→scope table test only works against one enforcement point |
| S9 traces + replay | trace store, four spans on one clock, metrics catalogue (declared at import), replay with structured diff | The dashboard and Langfuse export both read this store; build the store first (that ordering *is* the S13-before-S14 rule) |
| S10 reindex + lockfile | blue/green with atomic alias swap (three gate states: pass/fail/**could-not-run**), retention, rollback; lockfile + drift | Aliases go in the adapter contract — they cannot be emulated above it |
| S11 chaos + benchmark | scripted chaos at honest seams (later: real container stop, real `ENOSPC`), benchmark suite with ledger emitter | Chaos proves S1–S10's failure claims; do not accept a scenario that asserts `(TypedError, OSError)` — it passes either way (T13) |
| S12 autopilot | suggest-only search over query-time params, byte-identical config assertion | Needs S5's harness and golden sets |
| S13 Langfuse/Grafana | provisioning (doctor-gated, secrets written once), OTLP + Langfuse export | Export before dashboard: the trace pipeline must be proven by a consumer you didn't write |
| S14 dashboard | read-only, authenticated, tenant-scoped, zero mutating routes (asserted) | Last, by design |

Cross-cutting passes, in the order they happened and should happen again: facade + sync + entry points (after S8, once the services are stable); packaging (wheel force-include of the canonical config, T16); pgvector as the *second* contract proof; D11 live round-trip; the claims audit (sweep every doc for statements the code cannot back — budget a full pass, ours corrected 88); a release-readiness review ([release.md](release.md)).

## 3. The trap registry

Symptoms → cause → where the fix is now encoded. If rebuilding, read this list **before** each slice.

| # | Trap | Encoded fix |
|---|---|---|
| T1 | Staging/temp paths used as document identity — dedup never fires again, corpus grows every run | `DocumentTask.source` vs `.location`; CRITICAL block on `readable`; test "same inline payload is the same document twice" |
| T2 | `inline:` URI prefix leaked into payload decoding — first 7 chars of every inline doc eaten | `_decode_inline` decodes the raw value; scheme used only for URI minting |
| T3 | Spawned pool children re-import the entrypoint on Windows/macOS — pool dies mysteriously | `__main__` guard documented as CRITICAL in python-api quickstart; `CHUNK_FAILED` names the cause |
| T4 | Editable-install `.pth` makes a worktree's tests run `main`'s source — commits go green untested | `pythonpath = ["src", …]` first in pyproject (TASK-0248) |
| T5 | Only port 6333 exposed; client attempts gRPC on 6334 and fails remotely | doctor per-port check; docs everywhere; FMEA row 15 |
| T6 | Qdrant bind mounts lose data on Windows/WSL | loader rejects bind-mount paths on win32; named volume default |
| T7 | Sparse vectors cannot be added to an existing dense-only collection; layouts are not interchangeable | ADR-0007 decided *before* the indexer; adapter refuses the mismatch (TASK-0119 documents the reindex path) |
| T8 | Windows path-case (`d:\` vs `D:\`) mints two document ids — looked like recall 0.0 | identity folds case platform-aware (TASK-0141→fixed); Windows CI leg exists because of this class |
| T9 | UTF-8 BOM rejected by the config loader | stripped in the loader; Windows leg again |
| T10 | Alias swap as delete-then-create leaves a window where the name resolves to nothing | single atomic alias op in the adapter contract, atomicity in the contract suite |
| T11 | Regenerating Langfuse `SALT`/`ENCRYPTION_KEY`/`NEXTAUTH_SECRET` invalidates every credential; `LANGFUSE_INIT_*` values must not be quoted in compose | provisioner writes once, never regenerates; unquoted templates; R9/R10 in references.md |
| T12 | An eval gate that *couldn't run* reported as a pass | three-state gate (`gate_ran: false`); blocked = exit 5, distinct from crash |
| T13 | Chaos assertions accepting the untyped error too — the suite can't see the promise break | TASK-0234's paired test was written to go red when the fix landed, and did (TASK-0234 ✅); rule: never `(Typed, OSError)`, and assert the chained cause so the errno survives translation |
| T14 | Faked httpx clients: patching the module attribute patches the global (self-recursion), and async clients require async byte streams | capture-before-patch helper + async-generator bodies in `test_sources.py` |
| T15 | Benchmark numbers from a busy laptop looked citable — variance exceeded the effect | ledger rule 5 (isolation) + non-citable entries as the worked example |
| T16 | Relocating `config.yaml` in the sdist breaks the wheel's force-include (wheel builds *from* the sdist) | CRITICAL comment in pyproject; packaged template = repo file, byte-identical |
| T17 | Tree-wide coverage number hides regressions in the packages that matter | gate scoped to `core/`+`adapters/`+`workers/`; scope note in testing-strategy §2 (platform sensitivity: TASK-0252) |
| T18 | Handlers/flags accepted and silently ignored (API fields, CLI flags, config keys) | wired or refused: `_reject_unenforced_settings`, the "wire the flags" and "wire the request fields" fixes; recipe rule in internals §5 |
| T19 | An accurate `Content-Length` is still a header the peer controls | size enforced twice: declared and while streaming |
| T20 | Model-version drift is invisible if the lockfile records only the model name | `embedding_model_version` in lock + chunk; drift names both values |
| T21 | Two surfaces, one shipped: CLI provisions three tools, REST one | TASK-0251; recipe rule "wire both surfaces or file the asymmetry same-day" |
| T22 | 2.2 GB default reranker: first query looks hung for minutes | load made visible (TASK-0174); default under decision (B6) — decide *before* first release |

## 4. When are you done?

A rebuild is complete when [verification.md](verification.md) passes tier for tier — same gates, same suites, same live walkthroughs — and the deltas it cannot yet check are exactly the open `TASK-` ids it cites, no more. That is also this repository's own definition of done.
