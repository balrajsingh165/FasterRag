# engineering-audit.md — Independent Engineering Audit of 2026-11-11

> **Who this is for.** A maintainer deciding what to work on, a contributor (human or AI) who wants the reasoning behind a task before starting it, and anyone evaluating whether to depend on fasterRag who would rather read the weaknesses from the project than discover them.
>
> **What this is.** The analysis behind the score: how the audit was done, what it found in each of fourteen categories, what was fixed the same day, what is still open, and what each further step is expected to be worth. It is a dated record. The living numbers are in [engineering-scorecard.md](engineering-scorecard.md) §3, the work is in [todo.md](todo.md), and the step-by-step instructions for each piece of work are in [task-briefs.md](task-briefs.md).
>
> **Status of the numbers.** Every score is one auditor's judgement against the written rubric in [engineering-scorecard.md](engineering-scorecard.md) §2. None is a measurement and none may be quoted as a performance claim; [benchmarks.md](benchmarks.md) is the only place those live. Projected scores in §8 are estimates and are labelled as such.
>
> **Not published.** Excluded from the documentation site for the reason [todo.md](todo.md) is: it describes unfixed weaknesses in enough detail to act on.

---

## 1. Verdict

fasterRag is a carefully structured, heavily tested RAG backend whose core path — ingest a corpus, retrieve with hybrid search and reranking, answer with citations — works, and whose documentation is unusually candid about what does not. It is **not yet a system a stranger can install and rely on**. On the day of the audit it scored **47/100**; after the defects fixed that day it scores **52/100**.

The gap to the target of 90 is not spread evenly. Three things account for most of it:

1. **Nothing is measured at scale.** There is no citable throughput, latency or retrieval-quality number, so the project's central ambition is a goal and not a result (TASK-0454).
2. **Several documented surfaces are absent or unreachable**: a worker process that does no work, plugins the configuration schema refuses, an embedding cache nothing calls, no way to delete a document, no release (TASK-0440, TASK-0439, TASK-0442, TASK-0443, TASK-0445).
3. **Retrieval and generation stop at the baseline every comparable project already ships**: English-only sparse search, single-turn answers, no relevance floor, no query transformation (TASK-0447, TASK-0448, TASK-0458).

What the project has that the comparison set mostly lacks is the habit this audit relied on: claims are gated, weaknesses are written down, and a fix arrives with the test that proves it. That habit is the reason the path in §8 is credible.

## 2. Method

| Step | What was done |
|---|---|
| Reconnaissance | Every package under `src/fasterrag` read at module level; the ingest path and the query path read line by line from entry point to vendor call. |
| Reachability | For each documented capability, a call site was searched for from the REST routers, the CLI commands and the Python facade. Code no entry point reaches was scored as absent. |
| Execution | The unit tier was run locally. Integration, eval, load, soak and chaos tiers were read, and their CI results inspected, but not all were run locally. |
| Verification of findings | Each finding names the file and function it was read from. Findings read from code and never executed are marked **NOT VERIFIED** in [todo.md](todo.md) (three of them: TASK-0441, TASK-0452 and one half of TASK-0447). |
| Comparison | Ten projects were researched from their public documentation and repositories on 2026-11-11: LlamaIndex, LangChain, Haystack, RAGFlow, txtai, LightRAG, GraphRAG, R2R, Dify and Onyx. None was installed or benchmarked, so the comparison is of *offered capabilities*, never of speed or quality. |
| Scoring | Fourteen categories, each 0-100 on the band table in the scorecard, weighted and summed. Five rules applied without exception: implemented beats documented, reachable beats present, proven beats plausible, measured beats asserted, and a defect that silently loses or leaks data caps its category at 59. |

**Limits of this audit.** One auditor, one pass, one day. No load was applied, no security testing beyond code reading was performed, and no real provider was called. A category score can be wrong by several points in either direction; the ordering of the categories is more reliable than any single number.

## 3. Scores

Snapshot at commit `d67fdf6` (2026-11-12). This table is frozen; the one that moves is [engineering-scorecard.md](engineering-scorecard.md) §3.

| # | Category | Weight | At audit | After same-day fixes | Band now | Contribution to total |
|---|---|---|---|---|---|---|
| 1 | Architecture and design | 10% | 52 | 58 | Basic | 5.80 |
| 2 | Core RAG functionality | 12% | 58 | 62 | Functional but limited | 7.44 |
| 3 | Document ingestion and processing | 8% | 52 | 60 | Functional but limited | 4.80 |
| 4 | Retrieval and search quality | 10% | 45 | 45 | Basic | 4.50 |
| 5 | Embedding and vector infrastructure | 7% | 58 | 60 | Functional but limited | 4.20 |
| 6 | Advanced RAG capabilities | 7% | 15 | 15 | Fundamentally missing | 1.05 |
| 7 | Extensibility and modularity | 8% | 50 | 51 | Basic | 4.08 |
| 8 | Performance and scalability | 8% | 18 | 24 | Very immature | 1.92 |
| 9 | Reliability and error handling | 7% | 42 | 58 | Basic | 4.06 |
| 10 | Testing and quality assurance | 7% | 66 | 68 | Functional but limited | 4.76 |
| 11 | Security | 5% | 55 | 64 | Functional but limited | 3.20 |
| 12 | Developer experience and APIs | 5% | 58 | 62 | Functional but limited | 3.10 |
| 13 | Documentation | 3% | 55 | 62 | Functional but limited | 1.86 |
| 14 | Packaging and release readiness | 3% | 35 | 35 | Very immature | 1.05 |
| | **Weighted total** | 100% | **47** | **52** | | **51.82** |

Where the missing 48 points are, largest first (weight × distance from 100):

| Category | Points still on the table |
|---|---|
| Performance and scalability | 6.08 |
| Advanced RAG capabilities | 5.95 |
| Retrieval and search quality | 5.50 |
| Core RAG functionality | 4.56 |
| Architecture and design | 4.20 |
| Extensibility and modularity | 3.92 |
| Document ingestion and processing | 3.20 |
| Reliability and error handling | 2.94 |
| Embedding and vector infrastructure | 2.80 |
| Testing and quality assurance | 2.24 |
| Packaging and release readiness | 1.95 |
| Developer experience and APIs | 1.90 |
| Security | 1.80 |
| Documentation | 1.14 |

The three categories at the top hold 17.5 of the 48 missing points, and all three wait on the same instrument: an eval set and a benchmark harness that can tell better from worse (TASK-0454). That is why the order of work in the scorecard puts measurement before any retrieval improvement.

## 4. What is already strong

These are the properties the audit found to be true in the code and proven by tests. They are the foundation every task below builds on, and a change that weakens one of them is a regression whatever else it adds.

- **Layering that holds where it matters.** Routers contain no business logic; REST, CLI and the Python facade call one service layer; vendor SDK types do not leave `adapters/`.
- **One typed error taxonomy**, rendered as RFC 9457 `application/problem+json` with a stable `code` on every API error path. There is no bare `except` in `src/`, and a gate keeps it that way.
- **Idempotent ingestion as a contract**: content-hash deduplication, idempotency keys, deterministic chunk identity, and — since the audit — a ledger that records a document as indexed only when the vector database holds every chunk of it.
- **A vector-adapter contract suite** that every registered adapter must pass, and that has already caught real adapter defects.
- **Gates on prose.** Nine scripts and a set of documentation tests fail the build when a document claims something the tree contradicts, cites a task that does not exist, or asserts superiority without a ledger entry.
- **Candour.** The documents mark what is unbuilt, the benchmark ledger marks its own entries non-citable, and the task ledger records defects in enough detail to reproduce them.
- **Operational groundwork** that most comparable projects leave to the user: a cost governor with per-tenant budgets, query replay under a candidate configuration, a retrieval regression gate in CI, a degradation ladder, circuit breakers, a dead-letter queue.

## 5. Findings by category

Each block gives the score, what earns it, what holds it down, and the tasks that close the gap. The bar for 90 in each category is in [engineering-scorecard.md](engineering-scorecard.md) §4.

### 5.1 Architecture and design — 52 → 58

**Earns the score.** Clean adapter boundary, one service layer behind three surfaces, typed errors end to end. Since the audit: circuit breakers and the reranker model are process-wide resources reached through `shared_*` functions.

**Holds it down.** Import cycles between `workers` and `services` and between `core` and `adapters`; `services/evaluation.py` and `facade.py` import builders from `api/`; query services are constructed on every request; the documented worker process, `app.workers` and `python -m fasterrag.workers` do no work; a GitHub Actions reader for this repository ships inside the wheel.

**Closes the gap.** TASK-0456, TASK-0440 with TASK-0130, TASK-0438, TASK-0457, TASK-0437, TASK-0466.

### 5.2 Core RAG functionality — 58 → 62

**Earns the score.** Ingest, hybrid retrieve, rerank, generate with citations, streaming and non-streaming, a faithfulness check, an extractive fallback. Since the audit: streamed queries store a trace, and the semantic cache can no longer serve an answer across collections or filters.

**Holds it down.** `answer()` and `stream()` are two copies that have diverged; there is no way to delete or update a document; generation is single-turn with a fixed prompt; the LLM call is never retried; two provider adapters look written against SDK shapes that current releases no longer return.

**Closes the gap.** TASK-0438, TASK-0443, TASK-0449, TASK-0458, TASK-0441, TASK-0467, TASK-0468, TASK-0469.

### 5.3 Document ingestion and processing — 52 → 60

**Earns the score.** Seven text formats, five chunking strategies, contextual enrichment, deduplication, a dead-letter queue. Since the audit: parsing is concurrent up to the pool size, one bad document costs one document, and the journal follows the index write.

**Holds it down.** The semantic chunking strategy cannot run in the pipeline; HTML, URL and inline-source parsing have correctness defects; no office formats beyond DOCX; an interrupted job is never resumed; a parser that hangs stalls its job for ever; an empty file is recorded as indexed.

**Closes the gap.** TASK-0470, TASK-0473, TASK-0431, TASK-0459, TASK-0436, TASK-0471, TASK-0472, TASK-0462.

### 5.4 Retrieval and search quality — 45 → 45

**Earns the score.** Dense plus sparse with reciprocal rank fusion and a cross-encoder reranker, metadata filters, a regression gate.

**Holds it down.** The sparse leg tokenises ASCII only and stems English only, so it is silently skipped on other scripts; no relevance threshold, no diversity, no query transformation; the committed eval sets are a handful of small documents and one of the two baselines is saturated, so no change to retrieval can currently be shown to help or hurt.

**Closes the gap.** TASK-0454 first — nothing else here is provable without it — then TASK-0447, TASK-0448, TASK-0481, TASK-0463, TASK-0480.

### 5.5 Embedding and vector infrastructure — 58 → 60

**Earns the score.** Five registered vector adapters behind one interface and one contract suite; four embedding providers; batching with retry and a breaker.

**Holds it down.** The embedding cache is implemented and wired to nothing; the four newer adapters repeat over a thousand lines and have drifted on score convention, error code and list-valued filters; two of the five run against a real server only on a developer machine; a suspected under-return on filtered pgvector queries has not been executed.

**Closes the gap.** TASK-0442, TASK-0451, TASK-0452, TASK-0412, TASK-0352, TASK-0464, TASK-0049, TASK-0477, TASK-0478, TASK-0479.

### 5.6 Advanced RAG capabilities — 15 → 15

**Earns the score.** A faithfulness gate that can refuse, query replay, a retrieval regression gate.

**Holds it down.** No query rewriting, multi-query, HyDE, routing, corrective retrieval, graph retrieval or agentic loop.

**Closes the gap.** This category is 7% of the total, and at 15 it alone removes six points from the ceiling. The audit's guidance stands — nothing here ships before it can be measured — but the category cannot stay at 15 if the total is to reach 90. TASK-0465 records the decision; TASK-0482 to TASK-0484 are the first measured steps, each gated on TASK-0454.

### 5.7 Extensibility and modularity — 50 → 51

**Earns the score.** Provider selection by configuration; three adapter factories with entry-point resolution.

**Holds it down.** The configuration schema types the provider fields as closed literals, so an installed plugin can never be selected; chunkers, parsers and rerankers have no extension point at all; there is no plugin-author guide and no sample plugin.

**Closes the gap.** TASK-0439, TASK-0486.

### 5.8 Performance and scalability — 18 → 24

**Earns the score.** Async I/O throughout, a bounded queue between parsing and embedding, batching. Since the audit: parsing uses the whole pool, and the default reranker is loaded once per process.

**Holds it down.** One process, no queue-backed worker, journal lookups that scan every job file, blocking I/O on the event loop in several places, and no citable number of any kind — the largest corpus ever measured is 24 chunks.

**Closes the gap.** TASK-0454, TASK-0130, TASK-0437, TASK-0442, TASK-0407 to TASK-0410, TASK-0466, TASK-0487, TASK-0488.

### 5.9 Reliability and error handling — 42 → 58

**Earns the score.** Since the audit: "indexed" means in the index; a job always reaches a terminal status; a refused write no longer ends a worker; breakers are shared and can open; every journal write of a job is serialised.

**Holds it down.** A job interrupted by a process exit stays `running` for ever; shutdown drains nothing; journal writes are not `fsync`ed and have a window with no live record; the half-open breaker admits every waiter at once; neither Redis client has a timeout; the LLM call has no retry, no fallback and no deadline.

**Closes the gap.** TASK-0436, TASK-0437, TASK-0449, TASK-0450, TASK-0453, TASK-0473, TASK-0474, TASK-0475, TASK-0476.

### 5.10 Testing and quality assurance — 66 → 68

**Earns the score.** A large suite across unit, integration, contract, eval, load, soak and chaos tiers; Linux and Windows lanes; a must-run gate that fails the build when a required test is skipped everywhere.

**Holds it down.** No test runs the production `ProcessPoolExecutor`, which is why serial parsing survived fourteen slices; two vector adapters skip their contract suite in CI, and two LLM adapters have no call-path test of any kind; the eval sets cannot discriminate and nothing evaluates the generated answer; the nightly mutation run still carries 63 survivors from its first pass; CI tests one Python version while the package claims two.

**Closes the gap.** TASK-0455, TASK-0412, TASK-0441, TASK-0454, TASK-0481, TASK-0489, TASK-0490, TASK-0300.

### 5.11 Security — 55 → 64

**Earns the score.** Path and archive confinement, SSRF-guarded fetching, constant-time key comparison, rate limiting, secrets kept out of YAML. Since the audit: loopback bind by default with an exposure warning, `/v1/estimate` confined, the semantic cache scoped.

**Holds it down.** Tenant scoping of the ingest and query paths is opt-in and incomplete; retrieved text is interpolated raw into the prompt and into the grader; a key with no tenant may act for any tenant; keys are plaintext and cannot be rotated without a restart; administrative actions leave no audit trail; three supply-chain statements in security.md had nothing behind them.

**Closes the gap.** TASK-0339 and TASK-0381 (a maintainer decision is pending — read both first), TASK-0446, TASK-0460, TASK-0326, TASK-0491 to TASK-0496.

### 5.12 Developer experience and APIs — 58 → 62

**Earns the score.** Three coherent surfaces, a `doctor` command, problem-detail errors. Since the audit: the README and quickstart examples run as written.

**Holds it down.** The first run needs a vector database, two environment variables and two extras; the facade cannot query as a tenant; a caller error surfaces as a 500-class code; the blocking facade refuses to start in a notebook; there are no runnable examples.

**Closes the gap.** TASK-0444, TASK-0461, TASK-0497, TASK-0498, TASK-0500.

### 5.13 Documentation — 55 → 62

**Earns the score.** Broad, cross-linked and gated; status notes on unbuilt surfaces.

**Holds it down.** Several documents still describe process surfaces that do not exist; no document's code snippets are executed by a test.

**Closes the gap.** TASK-0440, TASK-0499, TASK-0500, TASK-0501.

### 5.14 Packaging and release readiness — 35 → 35

**Earns the score.** A lockfile, a packaging check, a container image, a supply-chain job.

**Holds it down.** No tag, no release, no publish workflow, and a mandatory dependency set that includes a web server and a PDF stack.

**Closes the gap.** TASK-0445 with TASK-0087, TASK-0444, TASK-0502, TASK-0496, TASK-0503.

## 6. What was fixed on the day

Eight tasks, each with a test that fails without the fix. Full entries are in [todo.md](todo.md) under "Independent engineering audit follow-ups".

| Task | Defect closed | Category moved |
|---|---|---|
| TASK-0427 | A document was journalled as indexed, and its dedup hash written, before any chunk was embedded | Reliability, ingestion |
| TASK-0428 | Parsing was serial behind a process pool; an untyped parser error aborted the job | Performance, ingestion |
| TASK-0429 | A refused vector write ended the embedding worker and could hang the job | Reliability |
| TASK-0430 | The reranker model reloaded on every REST query; no breaker could open on the REST path | Performance, architecture |
| TASK-0432 | Open bind with auth off by default; `/v1/estimate` unconfined; non-ASCII bearer token crashed auth | Security |
| TASK-0433 | The semantic cache served answers across collections and filters | Security, core RAG |
| TASK-0434 | Streamed queries stored no trace | Core RAG |
| TASK-0435 | The README and quickstart Python examples failed as written | Developer experience |

Two follow-up commits were needed before CI was green, both for defects the fixes themselves introduced: a test header type the CI interpreter's `httpx` rejects, and a real race in which a checkpoint from the parse side and one from the embedding side rewrote one job record through the same temporary file. The second was found only by the integration lane against a real backend — the argument for working rule 2 in the scorecard, in one incident.

## 7. What is pending

Counts by priority. The entries, with evidence and acceptance criteria, are in [todo.md](todo.md); the implementation steps are in [task-briefs.md](task-briefs.md).

| Priority | Meaning | Tasks |
|---|---|---|
| P0 | Blocks a credible public alpha | TASK-0431, TASK-0436 to TASK-0445 |
| P1 | Blocks a beta | TASK-0446 to TASK-0461 |
| P2 | Competitiveness; each needs a measurement before it becomes a default | TASK-0462 to TASK-0465 |
| Scenario coverage | Missing behaviours and missing tests found by walking each category's bar for 90 on 2026-11-12 | TASK-0466 to TASK-0504 |
| Earlier, still open, and on the path | Tenant scoping, real-server CI for two adapters, the worker, event-loop blocking, release, export/import, Pinecone | TASK-0339, TASK-0381, TASK-0412, TASK-0352, TASK-0130, TASK-0407 to TASK-0410, TASK-0087, TASK-0079, TASK-0049 |

**Waiting on a maintainer decision, not on code:** tenant scoping of query and ingest (TASK-0339, TASK-0381), the worker-or-remove decision (TASK-0440 with TASK-0130), the adapter-contract review (TASK-0451 with TASK-0182), and the advanced-retrieval decision (TASK-0465).

## 8. What each step is expected to be worth

**These are estimates, not commitments and not measurements.** They exist so that whoever chooses the next task can see what it buys. Re-score from evidence when a stage closes (working rule 11).

| Category | Weight | Now | After P0 | After P1 | After P2 and scenario tasks |
|---|---|---|---|---|---|
| Architecture and design | 10% | 58 | 66 | 80 | 88 |
| Core RAG functionality | 12% | 62 | 72 | 82 | 90 |
| Document ingestion and processing | 8% | 60 | 70 | 78 | 88 |
| Retrieval and search quality | 10% | 45 | 45 | 68 | 85 |
| Embedding and vector infrastructure | 7% | 60 | 66 | 78 | 88 |
| Advanced RAG capabilities | 7% | 15 | 15 | 20 | 55 |
| Extensibility and modularity | 8% | 51 | 72 | 76 | 90 |
| Performance and scalability | 8% | 24 | 30 | 62 | 85 |
| Reliability and error handling | 7% | 58 | 72 | 84 | 90 |
| Testing and quality assurance | 7% | 68 | 72 | 82 | 90 |
| Security | 5% | 64 | 64 | 80 | 90 |
| Developer experience and APIs | 5% | 62 | 76 | 80 | 90 |
| Documentation | 3% | 62 | 70 | 76 | 90 |
| Packaging and release readiness | 3% | 35 | 70 | 72 | 90 |
| **Weighted total** | | **52** | **about 60** | **about 73** | **about 86** |

Three things this table says that are easy to miss:

- **P0 is worth about eight points and P1 about thirteen.** P1 is worth more because it contains the measurement task, which is what lets retrieval, performance and testing move at all.
- **Every task currently written down reaches about 86, not 90.** The remaining four points are in advanced retrieval: with that category at 55, every other category would have to average 93 for the total to reach 90. Either advanced retrieval reaches roughly 75 on measured evidence, or 90 is not reached. That is a decision for the maintainer (TASK-0465), and it should be made with the eval set in hand.
- **Retrieval, performance and advanced RAG do not move in the P0 column at all.** Anyone optimising for the score alone would skip P0. Do not: P0 is what makes the system safe to hand to a stranger, and a measured number about a system that loses jobs on restart is not worth having.

## 9. Position against the comparison set

A comparison of offered capabilities only. No speed or quality comparison is made or implied.

| Capability | State across the ten projects researched | fasterRag |
|---|---|---|
| Hybrid dense plus sparse retrieval with rank fusion | Offered by nearly all | Yes, English only on the sparse leg (TASK-0447) |
| Reranking | Offered by nearly all | Yes, local cross-encoder only (TASK-0458) |
| Citations and token streaming | Offered by nearly all | Yes |
| Pluggable model and store providers | Offered by nearly all | Yes for built-ins; third-party plugins blocked (TASK-0439) |
| Broad document formats through a conversion library | Common | Seven text formats (TASK-0462) |
| Multi-turn conversation | Common | Absent (TASK-0458) |
| Document update and delete | Common | Absent (TASK-0443) |
| Query transformation, graph or agentic retrieval | What several lead with | Absent (TASK-0448, TASK-0465) |
| Start with no external service | Common | Absent (TASK-0444) |
| Published releases | All | None (TASK-0445) |
| Idempotent ingestion as a stated contract | Rare | Yes |
| Per-tenant cost budgets | Rare | Yes |
| Replay of a past query under a candidate configuration | Rare | Yes |
| Retrieval regression gate that runs in CI | Rare | Yes, on eval sets too small to discriminate (TASK-0454) |
| Portable cross-store archive | Rare | Read side and writer landed; open (TASK-0079) |
| Configuration that refuses bad input at load | Rare | Yes |
| Documents gated against the code | Not found in any | Yes |

The reading the audit takes from this: the lower half of the table is where fasterRag is ahead and where the incumbents are weakest. The upper half is commodity, and the rows marked absent there are the minimum a user arriving from any of the ten will expect. Close the upper half to be considered at all; finish and prove the lower half to be chosen.

## 10. Release readiness

Two different scores bear on a release, and they must not be confused.

- **The project's own gate** is the release-confidence score in [release-readiness.md](release-readiness.md), which [release.md](release.md) §1 requires to be at least 85 for a beta and 90 for 1.0, re-derived within seven days of the tag, with every release-gating item closed and CI green on the tag commit. The last recorded value is 62, from 2026-08-10, and is stale. That score measures whether the documents, the gates and the code agree; it is the authority for the release decision.
- **This audit's score** measures engineering maturity against the rubric. It does not replace the gate; it says what the gate cannot — how the system compares with what a user will expect of it.

| Stage | Audit score at which it becomes defensible | Today | What gates it |
|---|---|---|---|
| Private use by the maintainer, single process, trusted network | — | Usable | — |
| First public pre-release, labelled alpha (`0.1.0-alpha.1`) | About 60 | 52 — not ready | Every P0 task: thirteen in all (TASK-0431, TASK-0436 to TASK-0445, TASK-0470, TASK-0473). The ones a first user would hit within an hour are an interrupted job that stays `running` (TASK-0436), no way to delete a document (TASK-0443), a first run that needs a vector database and three environment variables (TASK-0444), and plugins that cannot be selected (TASK-0439). |
| The beta the release documents have decided on (`0.1.0-beta.1`) | About 75 | Not ready | Every P1 task, tenant scoping decided and built (TASK-0339, TASK-0381), at least one citable measurement (TASK-0454), and the project's own gate at 85. |
| 1.0 | 90 | Not ready | The rubric at 90, re-derived by someone other than the author of the changes; the project's own gate at 90; the measurement programme complete. |

**The recommendation.** Do not publish anything at 52. Publish an **alpha** when P0 is closed — at about 60 — because an early, honestly labelled release is how a project earns the users and the bug reports the remaining categories need, and every P0 item is something a stranger meets on the first day. Keep the **beta** label for about 75, which is where the release documents' own threshold of 85 becomes reachable. Note that the release documents currently plan to skip the alpha and go straight to `0.1.0-beta.1`: publishing an alpha first is a change to that plan and is the maintainer's decision. A version number can never be reused on PyPI, so the label chosen for the first upload is permanent.

## 11. How to use this document

1. Read [engineering-scorecard.md](engineering-scorecard.md) for the rubric, the bar for 90 per category, and the eleven working rules.
2. Pick the first open task in the order of work (scorecard §6).
3. Read its entry in [todo.md](todo.md) for the evidence and the acceptance criterion.
4. Read its brief in [task-briefs.md](task-briefs.md) for the files, the steps, the tests to write and the traps.
5. Do the work, tick the task, update the scorecard row only if a defect is closed *and* proven.

This document is not updated as tasks close. It is replaced by the next audit.
