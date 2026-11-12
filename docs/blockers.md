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

**Answering the root either resolves the children outright or opens the path to them.** That is the whole point of the grouping: a root is one conversation that unblocks several pieces of work, and it is worth more attention than the same number of separate line items would suggest.

A child is never *also* a root. If something is blocked by two roots it is listed under the one that must move first, with the other named inline.

Rules: add nothing without a `TASK-` id; delete an entry when its blocker clears; mark a completed task cited as history with ✅ so it reads as context rather than as outstanding. `scripts/check_blockers.py` enforces the first and third in CI.

---

## B4 — No isolated reference hardware (TASK-0084)

Load, soak, and chaos runs need a documented, isolated machine. `BENCH-0001`, `BENCH-0002` and `BENCH-0003` are committed and **explicitly marked not citable**: all taken on a developer workstation under co-tenant load, failing benchmark-ledger rule 5 — BENCH-0003 additionally for a reason this hardware would not cure, its quality deltas being smaller than their own standard error. Run-to-run variance on one commit was 5.5 s vs 8.7 s.

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
| **B8.4** | TASK-0115 | `sentence-transformers` ships as the `huggingface` extra rather than core, because it pulls a multi-gigabyte stack. Confirm the trade. |

---

## B9 — Maintainer actions outside the repository

| | Task | Action |
|---|---|---|
| **B9.1** | TASK-0098 | Enable branch protection on `main` — require PRs, block direct pushes. |
| **B9.2** | TASK-0111 | Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2`. |
