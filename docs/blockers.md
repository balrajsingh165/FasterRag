# blockers.md — What Is Waiting On A Human

**This is a read-only view, not a second task file.** Every entry here is a `TASK-` id that already exists in [todo.md](todo.md), which remains the only place tasks are created, tracked, or ticked. Nothing is recorded here that is not recorded there; this file exists because a blocker buried on line 250 of a 290-line ledger is a blocker nobody sees.

Add nothing here without a `TASK-` id. When a blocker clears, tick it in `todo.md` and delete the entry here.

Ordered by what it costs to leave unresolved, not by id.

---

## 1. Irreversible — decide before the first publish

### TASK-0164 — the license is a default, not a decision

`pyproject.toml` declares `GPL-3.0-or-later`. Nothing records that as a choice.

**Why it blocks:** the stated adoption thesis is people embedding fasterRag inside their own products, and GPL's copyleft is what a commercial legal review blocks on. Relicensing later needs every contributor's agreement, and anyone who took the GPL grant keeps it. **A PyPI version number can never be reused**, even after a release is deleted — so the first upload fixes both the license and the version permanently.

**Blocks:** TASK-0087 (PyPI publish), and with it the whole distribution story.

**What is needed:** an ADR saying GPL-3.0-or-later or a permissive license, and why. Either answer is fine. The undocumented default is not.

### TASK-0020 — beta version number

`0.1.0.dev0`, classified `Development Status :: 2 - Pre-Alpha`. Needs stamping to `0.1.0-beta.1` (or whatever is chosen) in `CHANGELOG.md` before release. Same permanence problem as above.

---

## 2. Product decisions — they change what users experience

### TASK-0175 — the shipped reranker is 2.2 GB

Default config is `retrieval.rerank: true` with `BAAI/bge-reranker-v2-m3`. **Measured: did not finish loading within 400 s** on a developer laptop; with `rerank: false` the same query path returns in 4.3 s.

A `pip install` user's first query downloads 2.2 GB and then appears to hang with no output. TASK-0174 ✅ made the wait *visible* — the loader now warns, names the cost, and names the two ways out — but it did not decide whether the wait should exist.

**The tension is real:** cross-encoder reranking is the single biggest quality lever in the stack. Turning it off by default trades the framework's best feature for a faster first run.

**Options:** ship a small cross-encoder by default and document the large one as an upgrade; ship `rerank: false` and document turning it on; or keep it and treat the first-run cost as documented behaviour.

### TASK-0115 — `sentence-transformers` placement

Ships as the `huggingface` extra rather than core, because it pulls a multi-gigabyte deep-learning stack. Confirm that is the intended trade.

---

## 3. Architecture decisions — they block implementation work

### TASK-0165 — degradation-ladder scope (ADR-0008)

`docs/reliability.md` specifies a circuit breaker and a `cache_only` rung. Neither exists; only the breaker's *configuration* does.

**Blocks two implementations outright:**
- **TASK-0148** — `fasterrag_circuit_state` is the one catalogue metric nothing writes, so the dashboard's circuit panel is empty by design. A test pins it as the single permitted exemption, and that exemption should not become permanent by inertia.
- **TASK-0159** — the `cache_only` rung.

**Decide:** build both as specified, or formally narrow D4 to the two rungs that ship today. Four documents (`reliability.md`, `api-reference.md`, `glossary.md`, `failure-modes.md`) carry *specified-not-built* annotations until this is answered.

### TASK-0129 — `url` and `inline` ingest sources

`docs/api-reference.md` documents all three source types; only `path` is accepted, and the other two are rejected with `VALIDATION_FAILED` naming what is supported.

**Not just plumbing.** `DocumentTask` uses one field as both the identity input and the read location, and a document id derives from it. Staging a URL to a temp file would make the id derive from the temp path, so re-ingesting the same URL would mint a new id every time and silently defeat deduplication.

**What is needed:** agreement to split `DocumentTask` into a canonical `source` (identity) and a resolved `location` (where bytes are read from), which changes a dataclass crossing the process-pool boundary. Deliberately not attempted in the 2026-08-03 sprint for that reason.

---

## 4. Blocked on hardware nobody has yet

### TASK-0084 — reference-hardware measurements

Load, soak, and chaos runs on an isolated, documented machine.

`BENCH-0001` and `BENCH-0002` are committed and **explicitly marked not citable**: taken on a developer laptop with Docker, an IDE, and co-tenant load, failing benchmark-ledger rule 5's isolation requirement. Run-to-run variance on one commit was 5.5 s vs 8.7 s.

**Blocks:** every `TBD-until-measured` in `slo.md`, and every performance claim the provable-claims policy would otherwise permit.

**Downstream:** TASK-0136 (per-scenario chaos recovery times, which D12's proof metric asks for).

### TASK-0085 — disaster-recovery drill, clean-host half

**Partially executed 2026-08-01.** Steps 4, 5, and 7 ran for real against a live Qdrant: a collection was snapshotted, deleted, restored, and verified to answer with an identical citation. Steps 1–3 (clean host, `.env` recreation, `fasterrag doctor`) and step 6 (`index lock verify`) did not, because the restore was onto the same running host.

**RPO and RTO stay `TBD` until the clean-host half runs.** They are not estimates to fill in.

---

## 5. Awaiting maintainer review — built, working, needs sign-off

These are not blocked; they are *unratified*. Each extended a contract or resolved an ambiguity no document covered, and each should be confirmed or corrected before it hardens into precedent.

| Task | What was decided unilaterally |
|---|---|
| TASK-0123 | Grounded-or-refuse while streaming. No doc specified it, and a token cannot be unsaid once sent — so with the flag on, generation buffers and grades before emitting any `token` event, trading time-to-first-token for the guarantee. |
| TASK-0126 | `VectorDBAdapter` gained `list_collections()` and `drop_collection()` with a vendor-neutral `CollectionInfo`. Every future adapter must implement them. |
| TASK-0133 | `VectorDBAdapter` gained `set_alias`, `alias_target`, `delete_alias`. Aliases are the primitive blue/green reindexing rests on; not every vendor has them. |
| TASK-0131 | `fasterrag traces list|show` was added beyond `cli-reference.md`, because a 32-hex trace id is unreachable without a way to list recent ones. Now documented; confirm the addition. |

---

## 6. Maintainer actions outside the repository

| Task | Action |
|---|---|
| TASK-0098 | Enable branch protection on `main` in GitHub settings — require PRs, block direct pushes. |
| TASK-0111 | Tag the landed slice boundaries `v0.1.0-s1` and `v0.2.0-s2`. |

---

## 7. Blocked on other work, not on a person

Listed so they are not mistaken for available work.

| Task | Waiting on |
|---|---|
| TASK-0144 | TASK-0142. D6's acceptance test needs a fixture with an improvement to find; on the handbook corpus every candidate scores 1.0, so Autopilot correctly suggests nothing. |
| TASK-0136 | TASK-0084 (isolated hardware). |
| TASK-0148, TASK-0159 | TASK-0165 (the scope ADR above). |
| TASK-0087 | TASK-0164, TASK-0020, TASK-0158. |
