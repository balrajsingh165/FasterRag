# blockers.md — What Is Waiting On A Human

**This is a read-only view, not a second task file.** Every entry is a `TASK-` id that already exists in [todo.md](todo.md), which remains the only place tasks are created, tracked, or ticked. Nothing is recorded here that is not recorded there; this file exists because a blocker buried on line 250 of a 290-line ledger is a blocker nobody sees.

## How to read the numbering

Blockers are grouped by **root cause**, not by id. A root gets a whole number; everything it holds up gets a decimal under it:

```
B9      <root-task>    the decision nobody has made
B9.1    <child-task>   cannot start until B9 is answered
B9.2    <child-task>   cannot start until B9 is answered
```

Deliberately not real ids. The gate below reads every `TASK-` reference in this file and
checks it against todo.md, so an illustration built from live ids fails the moment one of
them is ticked — the gate would then be reporting on the example rather than on a blocker.

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
| **B1.1** | TASK-0087 | PyPI publish. Also needs **B2**; distribution mechanics themselves are verified (build, `twine check`, clean-venv install all pass). |

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
| **B3.1** | TASK-0159 | The `cache_only` rung — consult the semantic cache when retrieval raises, instead of returning a bare `RETRIEVAL_FAILED`. |
| **B3.2** | — | Four documents (`reliability.md`, `api-reference.md`, `glossary.md`, `failure-modes.md`) carry *specified-not-built* annotations that stay until B3 is answered either way. |

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

**The tension is real:** cross-encoder reranking is expected to be the strongest retrieval-quality lever in the stack — expected, not measured (TASK-0084). Turning it off by default trades that expected quality for a faster first run, and the trade cannot be priced until the gain is a number.

**Options:** a small cross-encoder by default with the large one as a documented upgrade; `rerank: false` by default; or keep it and treat the first-run cost as documented behaviour.

| | Task | Blocked because |
|---|---|---|
| **B6.1** | — | The quickstart's credibility. Every documented first-run walkthrough currently understates how long it takes. |

---

## B7 — Verified against unit tests, never against the real backend (TASK-0079)

**The daemon is back** (Docker Desktop 4.85.0 / engine 29.6.2, 2026-08-09), so nothing here is blocked any longer — each row is simply still unrun. Kept as a group because the reason they are all unverified is the same one, and because the first thing run against the real thing (TASK-0190 ✅ — the container image) failed on its first build, which is the argument for working through the rest of the list rather than assuming it.

Nothing below is *broken*; each is **built and unit-tested but never run against the real thing**, and that distinction is the point of listing them.

**B7.1 (the Langfuse trace export) closed on 2026-08-09.** Running it against a real server was worth doing: Langfuse accepted the batch, but the exercise also showed the exporter could not have told anyone if it had not. See the ledger for the evidence.

| | Task | What cannot be confirmed |
|---|---|---|
| **B7.2** | TASK-0079 | `iterate_points` has never walked a live collection, and no archive has been exported from or imported into a real Qdrant. The D11 round-trip acceptance test (Qdrant → Qdrant vector copy, Qdrant → pgvector re-embed) cannot run at all. |
| **B7.3** | — | The Grafana and Langfuse provisioners, the `/readyz` failure path, and every end-to-end query check that was passing earlier in the session. |

**What is needed:** nothing but the time to run them. Everything is scripted and ready.

---

## B8 — Built, working, unratified

Not blocked — *unreviewed*. Each resolved an ambiguity no document covered, and each should be confirmed or corrected before it hardens into precedent by default.

| | Task | What was decided unilaterally |
|---|---|---|
| **B8.1** | TASK-0123 | Grounded-or-refuse while streaming. A token cannot be unsaid once sent, so with the flag on, generation buffers and grades before emitting any `token` event — trading time-to-first-token for the guarantee. |
| **B8.2** | TASK-0126 | `VectorDBAdapter` gained `list_collections()` and `drop_collection()` with a vendor-neutral `CollectionInfo`. Every future adapter must implement them. |
| **B8.3** | TASK-0133 | `VectorDBAdapter` gained `set_alias`, `alias_target`, `delete_alias`. Aliases are the primitive blue/green reindexing rests on; not every vendor has them. |
| **B8.4** | TASK-0131 | `fasterrag traces list\|show` was added beyond `cli-reference.md`, because a 32-hex trace id is unreachable without a way to list recent ones. Now documented; confirm the addition. |
| **B8.5** | TASK-0115 | `sentence-transformers` ships as the `huggingface` extra rather than core, because it pulls a multi-gigabyte stack. Confirm the trade. |

---

## B9 — Maintainer actions outside the repository

| | Task | Action |
|---|---|---|
| **B9.1** | TASK-0098 | Enable branch protection on `main` — require PRs, block direct pushes. |
| **B9.2** | TASK-0111 | Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2`. |
