# benchmarks.md — The Benchmark Ledger

**Doctrine: a claim without a measurement is a bug.** This file is the single source of truth that backs every performance or superiority claim anywhere in fasterRag's documentation, README, release notes, or marketing. If a claim is not linked to a ledger entry here, the claim is a bug — file it in [todo.md](todo.md) and rewrite the claim as a goal.

## Ledger rules

1. Every entry records: **claim · method · dataset · hardware · date · numbers · commit hash**. All seven fields mandatory.
2. Entries are **append-only** — corrections are new entries superseding old ones (`supersedes: BENCH-XXXX`), never edits.
3. "Fastest"/"faster than X" may only cite an entry that measured **X itself** with the same harness, dataset, and hardware, with the harness committed to this repo.
4. Unmeasured statements in docs must be phrased as goals ("targets sub-second p95 retrieval"), never as facts.
5. Entries are produced by `fasterrag benchmark --ledger` (hardware fingerprint + commit hash auto-captured) and reviewed like code.
6. Docs are periodically grepped for superlatives ("fastest", "best", "most"); each hit must link here or it is filed as a bug.

## Entry template

```markdown
### BENCH-0001 — <short claim>
- Claim: <exact sentence the docs are allowed to state>
- Method: <suite + flags, warm/cold, repetitions, percentile rules>
- Dataset: <name@version, doc/token counts>
- Hardware: <CPU, RAM, GPU, storage, OS>
- Date: YYYY-MM-DD
- Numbers: <the measurements, all percentiles>
- Commit: <sha>
- Supersedes: <BENCH-XXXX | none>
```

## Ledger

**No entry below may currently be cited as a claim.** Both were measured on a developer laptop running Docker, an IDE, and other co-tenant load, which fails ledger rule 5's isolation requirement (`docs/performance.md` §Methodology: "no co-tenant load; background jobs disabled"). They are recorded because a measurement taken and caveated is worth more than none, and because they establish that the harness produces complete entries — not because they license a claim. The SLO targets in [slo.md](slo.md) therefore remain TBD-until-measured, and [performance.md](performance.md) still contains goals only. Superseding entries from isolated reference hardware land with TASK-0084.

### BENCH-0001 — parse-and-chunk throughput for a 20-page PDF, on non-isolated hardware
- Claim: **Not citable.** Isolation requirement unmet; recorded as an indicative datapoint only.
- Method: `fasterrag benchmark --suite ingest --ledger`; 3 warmed repetitions of 5 iterations, median reported, cold start measured separately, nearest-rank percentiles. Parse and chunk only — embedding and indexing excluded, because their cost belongs to the provider and the backend rather than to fasterRag, and including them would measure a network round trip.
- Dataset: `startup-ideas.pdf`, 1 document, 20 pages, 21 chunks, 16,029 tokens
- Hardware: Intel64 Family 6 Model 183 (16 cores), 39.7 GB RAM, NVIDIA GeForce RTX 3050 6 GB Laptop GPU, 244.1 GB storage, Windows 11, Python 3.13.12 — **with co-tenant load (Docker, Qdrant, IDE)**
- Date: 2026-08-01
- Numbers: warmed median of 3 repetitions — p50 8681.85 ms, p95 8928.43 ms, p99 8928.43 ms, throughput 0.119/s; cold start 3798.71 ms. A separate invocation minutes earlier on the same tree gave p50 5463.77 ms, and cold start was *faster* than the warmed median in both — run-to-run variance exceeds the effect being measured, which is itself evidence the isolation requirement is not optional.
- Commit: 9e7bbc1ff06241c322c3475002e8498f3d899114
- Supersedes: none

### BENCH-0002 — end-to-end non-cached query latency, on non-isolated hardware
- Claim: **Not citable.** Isolation requirement unmet; recorded as an indicative datapoint only.
- Method: `fasterrag benchmark --suite query --ledger`; 3 warmed repetitions of 5 iterations, median reported, cold start measured separately, nearest-rank percentiles. The semantic cache is disabled for the run: the suite repeats one question, so a live cache would report hit latency under the name "query latency".
- Dataset: `apitest` collection, 2 chunks; single fixed question
- Hardware: as BENCH-0001 — **with co-tenant load**
- Date: 2026-08-01
- Numbers: warmed median of 3 repetitions — p50 997.85 ms, p95 1166.77 ms, p99 1166.77 ms, throughput 0.986/s; cold start 16,133.29 ms. Latency is dominated by the OpenAI round trip, which fasterRag does not control; the cold start is the local embedding model loading, and the 16× gap between it and the warmed p50 is why the methodology reports the two separately.
- Commit: 9e7bbc1ff06241c322c3475002e8498f3d899114
- Supersedes: none
