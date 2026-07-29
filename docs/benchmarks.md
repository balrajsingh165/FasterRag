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

_No entries yet._ The first entries land at build-phase slice S11 (chaos/load/soak + baselines on documented reference hardware — see [todo.md](todo.md)). Until then, **no measured performance claims exist anywhere in this repository**, and [performance.md](performance.md) intentionally contains goals only, marked TBD-until-measured.
