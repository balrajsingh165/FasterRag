# release-readiness.md — Release Confidence Assessment

**Assessment date: 2026-08-10 · Release assessed: the first public tagged beta (`v0.1.0-beta.1` + PyPI publication, TASK-0020/TASK-0087) · Confidence: 62/100 today, ~84/100 after the release-gating checklist below.**

This is a point-in-time review, the successor to the first formal audit (2026-08-02, TASK-0169). It re-derives the state of the system independently — gates re-run, code inspected, docs cross-checked against implementation — rather than trusting the ledger, and then reconciles against the ledger. Method: full fast suite + lint + strict typing executed on the working tree; coverage measured on the gated packages; targeted verification of every capability the README claims shipped; doc-vs-code sweeps for surfaces the automated gates cannot see. Numbers below marked *(dev machine)* follow the changelog convention: dated observations, not benchmark-ledger entries.

## 1. What the score means

"Confidence in the release" is answered for a specific question: **if `v0.1.0-beta.1` were tagged and published today, how confident are we it would be correct, honest, and safe for adopters?** The 62 is held down almost entirely by *unmade decisions and unverified claims*, not by construction quality — which is why the post-checklist number (84) is close, and why nothing on the checklist requires hardware, new subsystems, or research.

| Dimension | Weight | Score | One-line basis |
|---|---|---|---|
| Engineering completeness vs the beta spec | 25% | 85 | All 14 slices landed; 10 of 12 differentiators fully real; gaps are D9's governor half, D4's third rung, 4 of 6 adapters |
| Verification depth | 20% | 88 | 2,060 fast tests green *(dev machine, 103 s)*; strict mypy over 308 files; property tests on chunkers/RRF/BM25; contract suite proven on two backends; real-container and real-disk chaos |
| Truthfulness of claims | 15% | 95 | The 2026-08-09 claims audit corrected 88 unbacked statements; ledger discipline is demonstrably enforced by CI gates (doc-truth, OpenAPI drift, blockers view, duplicate ids) |
| Operational readiness | 15% | 60 | Docker image + compose ✓, dashboard ✓, OTLP ✓, backups with retention ✓; but clean-host DR drill unexecuted, SLOs all TBD, rate limiter is per-replica, no load/soak evidence |
| Performance evidence | 10% | 35 | Zero citable benchmark entries; the project's central "fastest" ambition is currently unmeasurable — honestly labelled everywhere, but absent |
| Release mechanics | 15% | 45 | Build + `twine check` + clean-venv install verified; `uv.lock` gated in CI; but the license is an undecided default (irreversible once on PyPI), no version stamped, no tags exist, branch protection off |

Weighted: **≈ 68 raw**, adjusted to **62** for two cross-cutting discoveries this review made (§4): a REST/CLI provisioning asymmetry the doc gates cannot see, and the coverage gate measuring below its documented threshold on this machine — both small, both exactly the class of drift the project promises not to have.

## 2. What is verifiably done (independent inventory, not ledger transcription)

Every item below was re-verified this session by running it, reading its implementation, or confirming its committed test — the ledger agreed in all cases, which is itself a finding: **the ledger can be trusted.**

- **The full pipeline**: parse (PDF incl. OCR trigger + configurable thresholds, HTML, MD, DOCX, tabular; BOM-safe) → five chunkers with Hypothesis-verified invariants and real-tokenizer counting → contextual enrichment (non-fatal failure path) → stateful embedding pool with backoff retries → idempotent indexer with deterministic ids → hybrid dense+BM25 (property-tested fusion) → cross-encoder rerank → token-budgeted context assembly (empty-text-safe) → SSE generation with span citations, truncation reporting, and grounded-or-refuse.
- **The three control surfaces** with one service layer: REST (RFC 9457 problems, per-route scope table pinned by test), CLI (flags now wired end-to-end, directory expansion, `--set` overrides, `config show`/`init`), and the `FasterRag` facade (async + sync, lazy import, entry-point plugins with built-in-name precedence).
- **Reliability machinery**: typed taxonomy with no bare excepts; timeouts + jittered backoff; bulkheaded pools; circuit breakers on the embedding and vector-DB paths (LLM breaker constructed but unconsulted — TASK-0245); `hybrid_only` and `extractive` degradation rungs chaos-verified; checkpointed journal with real-`ENOSPC`-tested disk-full behavior (with one known typing gap, TASK-0234); DLQ with reason codes; identity forgery refusals (source-URI and tenant-id).
- **Security**: API-key auth with scopes as ASGI middleware, per-key rate limiting (per-replica — TASK-0216), tenant isolation enforced and adversarially tested (replay lookups tenant-scoped, cross-tenant collection addressing refused), body-size limits actually read, secrets never logged.
- **Operations**: Dockerfile + compose profiles that build and run; system-managed Qdrant; Langfuse v3 and Grafana provisioning verified live (via CLI — see §4.1); read-only dashboard, authenticated and tenant-scoped; OTLP export of traces *and* the metric catalogue with trace-id preservation; timestamped backup sets with retention pruning; blue/green reindex with eval-gated atomic alias swap and rollback.
- **Portability (D11)**: archive writer/reader with checksum, traversal, and referential-integrity verification before any write; byte-identical re-exports; REST + CLI surfaces; a live Qdrant round trip verified; **pgvector adapter shipped and passing the shared contract suite** — the "any vector DB" promise now has two genuinely different proofs.
- **Measurement scaffolding**: eval harness (recall@k/MRR/nDCG + faithfulness), committed golden set + baseline, regression-gate service, benchmark suite with ledger emitter, nightly workflow, cost estimation with per-entry price provenance across 48 models.
- **Process integrity**: 186 single-line Conventional Commits, zero attribution; `uv.lock` gated; gitleaks + pip-audit + Windows leg + doc-truth + OpenAPI-drift + doctor-smoke gates in CI; one todo file with enforced id uniqueness; blockers view with enforced numbering.

## 3. What is still missing (grouped by what it blocks)

### 3.1 Release-gating — must close before tagging `v0.1.0-beta.1`

| Item | Why it gates | Owner |
|---|---|---|
| **License decision (TASK-0164, B1)** | The only irreversible item: a PyPI version permanently fixes its license. Currently an undecided default (GPL-3.0-or-later) that contradicts nothing but records no reasoning | Maintainer |
| **Version stamp (TASK-0020, B2)** | The release *is* this decision | Maintainer |
| **Identity-encoding decision (TASK-0210)** | Changing the id scheme after the first tag forces every adopter through a reindex; "harmless now, impossible after" is exactly a release gate | Maintainer |
| **Cost governor: build or cut (TASK-0242)** | An unbuilt spend cap is the most dangerous kind of unbuilt feature; five docs describe enforcement that does not exist. Cutting the surface is a legitimate close | Engineering |
| **D7 gate into CI (TASK-0244)** | The regression gate exists but is not in the path of a push — the one gate whose absence lets quality regress silently | Engineering |
| **REST provisioning parity (TASK-0251, §4.1)** | api-reference promises three tools over REST; the router serves one | Engineering |
| **Coverage-gate reconciliation (TASK-0252, §4.2)** | The documented blocking gate may not be holding | Engineering |
| **LLM breaker: consult or stop constructing (TASK-0245)** | A metric reporting a state no code transitions is a false observability signal | Engineering |
| **Disk-full typing (TASK-0234 ✅)** | Closed 2026-08-10. One translation at one boundary, with the chained `OSError` preserved so errno 28 survives; the paired chaos assertion was inverted in the same commit | Engineering |
| **Reranker default (TASK-0175, B6)** | A 2.2 GB model with a 400 s first-load *(dev machine)* as the default is every adopter's first impression | Maintainer |
| **Ladder-scope ADR (TASK-0165, B3)** | Ships either `cache_only` (TASK-0159) or an honest narrowing; releasing with "specified-not-built" annotations is acceptable, releasing without the decision recorded is not | Maintainer |
| **SBOM at tag (remainder of TASK-0087)** | Documented supply-chain promise attached to the first tag | Engineering |

### 3.2 Should ship in the beta window but does not gate the tag

Accumulated maintainer reviews (TASK-0123, 0126, 0131, 0133, 0182, 0191 — the adapter surface has grown four times under review-later; ratifying them is one sitting) · rate limiter across replicas or an slo.md boundary note (TASK-0216) · queue-backend decision (TASK-0130) · `--watch`/`--fix` (TASK-0196/0197) · D11 cross-backend acceptance + facade wrap (TASK-0079) · missing chaos scenarios (TASK-0241) · live-provider eval checks (TASK-0205) · nightly regression comparison (TASK-0246) · R3 citation (TASK-0247) · pgvector pooling/timeouts (TASK-0239/0240).

### 3.3 Post-release program (honest to ship without, dishonest to forget)

The measurement program (TASK-0084/0085/0136, B4/B5): citable benchmarks, SLO targets, RPO/RTO, chaos recovery times — all need the isolated hardware that remains unrented. Load/soak/mutation layers (TASK-0243). Remaining four adapters (TASK-0049). Autopilot index-time search (TASK-0145).

## 4. What this review found that the ledger did not

**4.1 — REST/CLI provisioning asymmetry (filed as TASK-0251).** `api/admin.py` holds `_PROVISIONABLE = {"qdrant"}` and a stale `# TODO:` citing TASK-0043/0044 as unshipped — both shipped 2026-08-02, and `fasterrag provision langfuse|grafana` works. `POST /v1/admin/provision/langfuse` returns `NOT_FOUND` while [api-reference.md](api-reference.md) documents it as supported. Neither the OpenAPI-drift gate (the route exists; the constraint is value-level) nor the claims audit (which swept docs, not in-code TODOs) could see it. Lesson recorded with the task: stale in-code `# TODO:` markers citing ticked tasks are themselves detectable — a small gate could cross-check marker citations against the ledger's ticked ids.

**4.2 — The coverage gate may not be holding (filed as TASK-0252).** Gated packages (`core/`, `adapters/`, `workers/`) measured **83% branch coverage** this session *(dev machine, Windows)* against the **85** CI gate; [testing-strategy.md](testing-strategy.md) still cited the 87.11% figure from 2026-08-02 (corrected as part of this review). Three possibilities, none good to leave unresolved: the Linux CI run measures ≥85 and the platforms diverge by ≥2 points (then the gate is platform-dependent), CI is currently red (then main is not green), or the gate step's scope drifted. ~5,900 statements were added to the gated packages since the gate was set; growth outpacing coverage is precisely what the gate exists to catch — it may be doing its job right now, unobserved from this machine.

**4.3 — Confirmations worth stating.** TASK-0153's closure (the "10-minute hang" was a shell artifact, not a defect) checks out against the wired timeouts; the security slice genuinely enforces what the schema accepts (the accepted-but-unenforced guard now lists only the cost budgets — matching TASK-0242 exactly); README/quickstart/python-api status tables match the code precisely everywhere else this review probed. After 4.1, the sweep found no second asymmetry.

## 5. Risks and edge cases going into a beta

1. **First-run experience**: default config downloads a 2.2 GB reranker (B6) and multi-GB sentence-transformers — an adopter's first `fasterrag query` may look hung. Mitigation exists (TASK-0174's visibility); the default remains undecided.
2. **Windows-vs-Linux behavior**: two shipped defects were Windows-only; the Windows CI leg covers the fast suite only — integration behavior on Windows outside CI's sight.
3. **Single-replica assumptions**: the rate limiter (TASK-0216) and in-process ingestion (TASK-0130) both quietly assume one replica; the docs say so, but a beta adopter scaling replicas gets N× the configured limit and journal-resume-only restart semantics.
4. **Performance expectations**: the name is "fasterRag" and the ledger cannot yet back it. The framing everywhere is goals-not-claims — but the gap between the name and the ledger is a reputational risk the first measured release closes.
5. **GPL default** (if kept): materially narrows the embed-in-products funnel the docs court. Deciding is the gate; either answer is defensible.

## 6. Recommendations, with reasoning

1. **Run the release-gating checklist (§3.1) as the next unit of work — it is days, not weeks.** Ten of twelve items are decisions or single-boundary fixes; none needs hardware. The two engineering items with real surface (cost governor, D7-in-CI) both have "narrow honestly" fallbacks that keep the tag honest.
2. **Cut D9 to the estimator for the beta** (TASK-0242's narrowing path) rather than building the governor under release pressure — a spend cap built hastily is worse than an absent one clearly labelled; rebuild it as the first post-beta slice.
3. **Batch the six maintainer reviews into one sitting.** The adapter contract has grown four members under review-later; ratifying (or reverting) them before the first tag is the last cheap moment — after the tag they are public API.
4. **Rent the baseline hardware within the beta window, not before the tag.** A beta labelled "no citable numbers yet" is honest and the docs already say it everywhere it matters; blocking the tag on B4 gains little over shipping and measuring in-window.
5. **Keep this document current at each release decision**: re-run the method, restate the score. A release-confidence number that is not re-derived decays into the kind of unbacked claim the ledger exists to prevent.

## 7. Reconciliation with the ledger

This review adds TASK-0251 and TASK-0252 (its two independent findings), records itself as TASK-0253, and ticks nothing it did not verify. Existing open tasks were left untouched; where §3 groups them, the grouping is this document's judgment, not a ledger change. The blockers view (B1–B9) matches the open set faithfully — the numbering gate is doing its job.
