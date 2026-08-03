# blockers.md — What Is Waiting On A Human

**This is a read-only view, not a second task file.** Every entry is a `TASK-` id that already exists in [todo.md](todo.md), which remains the only place tasks are created, tracked, or ticked. Nothing is recorded here that is not recorded there; this file exists because a blocker buried on line 250 of a 290-line ledger is a blocker nobody sees.

## How to read the numbering

Blockers are grouped by **root cause**, not by id. A root gets a whole number; everything it holds up gets a decimal under it:

```
B3      TASK-0165   the decision nobody has made
B3.1    TASK-0148   cannot start until B3 is answered
B3.2    TASK-0159   cannot start until B3 is answered
```

**Answering the root either resolves the children outright or opens the path to them.** That is the whole point of the grouping: `B3` is one conversation that unblocks three pieces of work, and it is worth more attention than three separate line items would suggest.

A child is never *also* a root. If something is blocked by two roots it is listed under the one that must move first, with the other named inline.

Rules: add nothing without a `TASK-` id; delete an entry when its blocker clears; mark a completed task cited as history with ✅ so it reads as context rather than as outstanding. `scripts/check_blockers.py` enforces the first and third in CI.

---

## B1 — The license is a default, not a decision (TASK-0164)

**Costs the most to leave.** `pyproject.toml` declares `GPL-3.0-or-later`. Nothing records that as a choice.

The stated adoption thesis is people embedding fasterRag inside their own products, and GPL's copyleft is what a commercial legal review blocks on. Relicensing later needs every contributor's agreement, and anyone who took the GPL grant keeps it. **A PyPI version number can never be reused**, even after a release is deleted — so the first upload fixes the license permanently for that version.

**Needed:** an ADR saying GPL-3.0-or-later or a permissive license, and why. Either answer is fine; the undocumented default is not.

| | Task | Blocked because |
|---|---|---|
| **B1.1** | TASK-0087 | PyPI publish. Also needs **B2** and **B7**; distribution mechanics themselves are verified (build, `twine check`, clean-venv install all pass). |

---

## B2 — Beta version number (TASK-0020)

`0.1.0.dev0`, classified `Development Status :: 2 - Pre-Alpha`. Needs stamping in `CHANGELOG.md` before release. Same permanence problem as B1: the number is consumed forever on first upload.

Feeds **B1.1**.

---

## B3 — Degradation-ladder scope (TASK-0165)

`docs/reliability.md` specifies a circuit breaker and a `cache_only` rung. Neither exists; only the breaker's *configuration* does.

**Decide:** build both as specified, or formally narrow D4 to the two rungs that ship today.

| | Task | Blocked because |
|---|---|---|
| **B3.1** | TASK-0148 | The circuit breaker. `fasterrag_circuit_state` is the one catalogue metric nothing writes, so the dashboard's circuit panel is empty by design and a test pins it as the single permitted exemption. That exemption should not become permanent by inertia. |
| **B3.2** | TASK-0159 | The `cache_only` rung — consult the semantic cache when retrieval raises, instead of returning a bare `RETRIEVAL_FAILED`. |
| **B3.3** | — | Four documents (`reliability.md`, `api-reference.md`, `glossary.md`, `failure-modes.md`) carry *specified-not-built* annotations that stay until B3 is answered either way. |

---

## B4 — No isolated reference hardware (TASK-0084)

Load, soak, and chaos runs need a documented, isolated machine. `BENCH-0001` and `BENCH-0002` are committed and **explicitly marked not citable**: taken on a developer laptop with Docker, an IDE, and co-tenant load, failing benchmark-ledger rule 5. Run-to-run variance on one commit was 5.5 s vs 8.7 s.

The nightly CI job produces numbers but its own header states they are not citable either — a shared GitHub runner has neighbours. It is a regression signal, not evidence.

| | Task | Blocked because |
|---|---|---|
| **B4.1** | TASK-0136 | Per-scenario chaos recovery times, which D12's proof metric asks for. |
| **B4.2** | — | Every `TBD-until-measured` in `slo.md`. |
| **B4.3** | — | Every performance claim the provable-claims policy would otherwise permit. |

---

## B5 — Disaster-recovery drill, clean-host half (TASK-0085)

**Partially executed 2026-08-01.** Steps 4, 5, and 7 ran for real against a live Qdrant: a collection was snapshotted, deleted, restored, and verified to answer with an identical citation. Steps 1–3 (clean host, `.env` recreation, `fasterrag doctor`) and step 6 (`index lock verify`) did not, because the restore was onto the same running host.

| | Task | Blocked because |
|---|---|---|
| **B5.1** | — | RPO and RTO stay `TBD`. They are measurements, not estimates to fill in. |

---

## B6 — The shipped reranker is 2.2 GB (TASK-0175)

Default config is `retrieval.rerank: true` with `BAAI/bge-reranker-v2-m3`. **Measured: did not finish loading within 400 s** on a developer laptop; with `rerank: false` the same query path returns in 4.3 s. A `pip install` user's first query downloads 2.2 GB and appears to hang.

TASK-0174 ✅ made the wait *visible* — the loader now warns, names the cost, and names the two ways out — but it did not decide whether the wait should exist.

**The tension is real:** cross-encoder reranking is the single biggest quality lever in the stack. Turning it off by default trades the framework's best feature for a faster first run.

**Options:** a small cross-encoder by default with the large one as a documented upgrade; `rerank: false` by default; or keep it and treat the first-run cost as documented behaviour.

| | Task | Blocked because |
|---|---|---|
| **B6.1** | — | The quickstart's credibility. Every documented first-run walkthrough currently understates how long it takes. |

---

## B7 — Supply chain has no lockfile (TASK-0158)

**Partially done.** A `supply-chain` CI job runs gitleaks over full history and pip-audit as advisory-only.

| | Task | Blocked because |
|---|---|---|
| **B7.1** | — | pip-audit stays advisory. Made blocking without a lockfile, it fails on any CVE in the resolved tree — including ones with no fixed version — so every push would break on something nobody can act on. |
| **B7.2** | — | The SBOM, which **B1.1** needs. |

---

## B8 — The eval fixture cannot detect small regressions (TASK-0142)

The handbook baseline catches large regressions — verified to catch an index missing four of six documents (recall 1.0 → 0.3333, gate blocked, exit 5). But at six documents and k=5 a retriever returns five of six, so a *subtle* ranking regression passes.

| | Task | Blocked because |
|---|---|---|
| **B8.1** | TASK-0144 | D6's acceptance test — "on a fixture corpus with a known-better config, autopilot's suggestion matches or beats it" — is only verified by unit test. On the handbook corpus every candidate scores 1.0, so there is no improvement to find and Autopilot correctly suggests nothing. |

**This is the one root on this page that needs no decision and no hardware.** It is ordinary work: a larger fixture with topically overlapping documents.

---

## B9 — `url` and `inline` ingest sources (TASK-0129)

`docs/api-reference.md` documents all three source types; only `path` is accepted, and the other two are rejected with `VALIDATION_FAILED` naming what is supported.

**Not just plumbing.** `DocumentTask` uses one field as both the identity input and the read location, and a document id derives from it. Staging a URL to a temp file would make the id derive from that path, so re-ingesting the same URL would mint a new id every time and silently defeat deduplication.

**Needed:** agreement to split `DocumentTask` into a canonical `source` (identity) and a resolved `location` (bytes), which changes a dataclass crossing the process-pool boundary.

---

## B10 — Built, working, unratified

Not blocked — *unreviewed*. Each resolved an ambiguity no document covered, and each should be confirmed or corrected before it hardens into precedent by default.

| | Task | What was decided unilaterally |
|---|---|---|
| **B10.1** | TASK-0123 | Grounded-or-refuse while streaming. A token cannot be unsaid once sent, so with the flag on, generation buffers and grades before emitting any `token` event — trading time-to-first-token for the guarantee. |
| **B10.2** | TASK-0126 | `VectorDBAdapter` gained `list_collections()` and `drop_collection()` with a vendor-neutral `CollectionInfo`. Every future adapter must implement them. |
| **B10.3** | TASK-0133 | `VectorDBAdapter` gained `set_alias`, `alias_target`, `delete_alias`. Aliases are the primitive blue/green reindexing rests on; not every vendor has them. |
| **B10.4** | TASK-0131 | `fasterrag traces list\|show` was added beyond `cli-reference.md`, because a 32-hex trace id is unreachable without a way to list recent ones. Now documented; confirm the addition. |
| **B10.5** | TASK-0115 | `sentence-transformers` ships as the `huggingface` extra rather than core, because it pulls a multi-gigabyte stack. Confirm the trade. |

---

## B11 — Maintainer actions outside the repository

| | Task | Action |
|---|---|---|
| **B11.1** | TASK-0098 | Enable branch protection on `main` — require PRs, block direct pushes. |
| **B11.2** | TASK-0111 | Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2`. |
