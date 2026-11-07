# benchmarks.md — The Benchmark Ledger

**Doctrine: a claim without a measurement is a bug.** This file is the single source of truth that backs every performance or superiority claim anywhere in fasterRag's documentation, README, release notes, or marketing. If a claim is not linked to a ledger entry here, the claim is a bug — file it in [todo.md](todo.md) and rewrite the claim as a goal.

## Ledger rules

1. Every entry records: **claim · method · dataset · hardware · date · numbers · commit hash**. All seven fields mandatory.
2. Entries are **append-only** — corrections are new entries superseding old ones (`supersedes: BENCH-XXXX`), never edits.
3. "Fastest"/"faster than X" may only cite an entry that measured **X itself** with the same harness, dataset, and hardware, with the harness committed to this repo.
4. Unmeasured statements in docs must be phrased as goals ("targets sub-second p95 retrieval"), never as facts.
5. Entries are produced by `fasterrag benchmark --ledger` (hardware fingerprint + commit hash auto-captured) and reviewed like code.
6. Docs are periodically grepped for superlatives ("fastest", "best", "most"); each hit must link here or it is filed as a bug.

### The nightly comparison is not a source of entries

`.github/workflows/nightly.yml` compares each night's `fasterrag benchmark --json` with the previous night's using `scripts/compare_nightly.py`, and fails when a tracked metric moves more than 10% in the worse direction ([performance.md](performance.md) §3.7). **This changes nothing about what qualifies for this ledger.** Those runs happen on a shared GitHub runner and fail the same isolation requirement that makes BENCH-0001 and BENCH-0002 non-citable, and a comparison is a change detector rather than a measurement: it reports whether two non-citable numbers moved relative to each other, which is a reason to go and measure, not a measurement.

**What it compares is the ingest suite, and only that** (2026-11-08, TASK-0323). `--suite ingest` is parse-and-chunk; the query suite needs an LLM credential a hosted runner does not hold, and is not run. So the nightly watches chunking throughput for night-over-night movement and watches nothing else — no retrieval, no embedding, no indexing, no generation. A reader looking here for evidence that query latency has not regressed will not find it, and the nightly is not where it will appear. [performance.md](performance.md) §3.7 carries the credential decision this is waiting on.

The job therefore never writes to this file. Appending an entry stays a human act performed against a run on documented, isolated hardware (TASK-0084), and the one mechanism that produces entries is still `fasterrag benchmark --ledger` under rule 5.

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

**No entry below may currently be cited as a claim.** All three were measured on a developer workstation running an IDE and other co-tenant load, which fails ledger rule 5's isolation requirement (`docs/performance.md` §Methodology: "no co-tenant load; background jobs disabled"). They are recorded because a measurement taken and caveated is worth more than none, and because they establish that the harnesses produce complete entries — not because they license a claim. The SLO targets in [slo.md](slo.md) therefore remain TBD-until-measured, and [performance.md](performance.md) still contains goals only. Superseding entries from isolated reference hardware land with TASK-0084.

BENCH-0003 carries a **second, independent** reason it may not be cited, which isolated hardware would not cure: its quality numbers are smaller than their own standard error on a 16-query golden set, and two of its three comparisons separate the candidate models by nothing at all. An entry can be non-citable because of where it ran *and* because of what it found, and conflating the two would let a bigger machine look like an answer to a question that is actually about dataset size.

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

### BENCH-0003 — what cross-encoder reranking costs, and what the committed golden sets can say it buys, on non-isolated hardware
- Claim: **Not citable.** Isolation requirement unmet (ledger rule 5), and the quality half is additionally *inconclusive by its own statistics* — see Numbers. Recorded as an indicative datapoint and as input to TASK-0175, which it does not resolve. **No documentation claim may be built on it**, in particular none of the form "reranking improves retrieval quality by X" or "the large reranker is better than the small one".
- Method: `python scripts/measure_rerank.py --repetitions 3 --probe-candidates 100 --json <path>`, plus a second invocation adding `--dataset tests/eval/datasets/policies --chunk-size 128 --repetitions 1`. Three configurations per dataset — `retrieval.rerank: false`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, and the shipped `BAAI/bge-reranker-v2-m3` — scored with the repository's own eval harness (`fasterrag.core.evals.evaluate`: recall@k, MRR, nDCG@k, adversarial records excluded from the averages), at the dataset's committed `k=5`. Quality is deterministic here, and the three repetitions agreed exactly; the repetitions therefore buy latency samples, not quality variance. Quality deltas are **paired per query** against the rerank-off run and reported with the standard error of that per-query difference; a delta is called resolvable only when it exceeds **two** standard errors. Model size counts only the files a sentence-transformers load reads — alternate ONNX/Flax serializations and README screenshots are excluded, and the largest weight file is reported separately. Cold load is a first load from an already-populated hugging-face cache **in a process that has already imported torch and transformers**, and includes one single-pair forward pass; the download is measured separately and excluded. Latency is wall-clock around `CrossEncoderReranker.rerank`, CPU-only (`torch 2.13.0+cpu` — the machine's GPU is present but unused). **Substitution:** the vector database is replaced by an exact in-process brute-force store so the run needs no container; the parser, chunker, embedder, BM25 encoder, RRF fusion, reranker, and `build_retrieval` wiring are the shipped ones. Every configuration is scored against the same store, so the comparison is unaffected, but the absolute numbers are not a reproduction of a Qdrant deployment.
- Dataset: `tests/eval/datasets/handbook` (6 documents → 6 chunks at the shipped 768-token chunk size; 15 golden records, 12 scored, 3 adversarial) and `tests/eval/datasets/policies` (12 documents → 12 chunks at 768 tokens, 24 at 128 tokens; 19 golden records, 16 scored, 3 adversarial). Both golden sets are human-authored with document-level ground truth.
- Hardware: Intel64 Family 6 Model 183 (16 cores), 39.7 GB RAM, NVIDIA GeForce RTX 3050 6 GB Laptop GPU (**present but unused — CPU-only torch build**), 242.8 GB storage, Windows 11, Python 3.14.7 — **with co-tenant load (IDE, concurrent agent sessions editing this same tree)**
- Date: 2026-11-03
- Numbers:
  - **Cost, and the correction it forces.** `BAAI/bge-reranker-v2-m3` occupies **2.14 GB** on disk (2.12 GB of it one `model.safetensors`); `cross-encoder/ms-marco-MiniLM-L-6-v2` occupies **87.56 MB** (86.66 MB of weights). Cold load from a warm cache was **8.3–9.3 s** for the 2.2 GB model and **7.3–7.7 s** for the 90 MB one — so the marginal load cost of the large model is about **1 s**, not minutes. The **357 s** this entry measured for the large model is its *download* (2.14 GB over a domestic connection, once). The "over 400 s to load" recorded in TASK-0174 and quoted by TASK-0175 therefore describes a first run that downloads, not a load: on a populated cache the load is a few seconds. Rerank latency, CPU-only: over the real eval shortlists, **4.15 s** p50 at 6 candidates and **12.88 s** p50 at 12 candidates for the large model, against **0.22 s** and **0.65 s** for the small one. At the shipped `retrieval.rerank_top_n: 100`, the standalone probe gives the large model **55.7 s** p50 (handbook chunks) and **86.9 s** p50 (policies chunks, 768-token), against **2.26 s** and **5.18 s** for the small one — a **17–25× gap**, with the absolute figures dominated by total tokens reranked rather than by candidate count alone (the same 100-candidate probe over 128-token chunks costs 29.8 s and 3.1 s respectively). These are CPU numbers and a GPU would change them by orders of magnitude; nothing here measures one. **The latency figures are order-of-magnitude, not precise**: repeating the 6-candidate small-model measurement afterwards on an otherwise idle machine gave **53.0 ms** p50 against the **220.1 ms** recorded above — the same code, the same data, a 4× swing from co-tenant load alone. The 17–25× ratio between the two models survives that swing comfortably; the absolute milliseconds do not.
  - **Quality, and why it settles nothing.** `handbook` is **saturated**: all three configurations score recall@5 = MRR = nDCG@5 = 1.0000, so the dataset carries no information about reranking at all. On `policies` at the shipped chunk size, rerank-off scores recall 1.0000 / MRR 0.9062 / nDCG 0.9308 and **both** rerankers score **1.0000 / 1.0000 / 1.0000** — identical to each other. The paired gain is **MRR +0.0938 ± 0.0504** and **nDCG +0.0692 ± 0.0372** over 16 queries, i.e. **1.86 standard errors — inside the two-sigma band, so not resolvable.** On the deliberately harder 128-token variant of `policies` (24 chunks) the small model scores MRR 0.9688 / nDCG 1.0000 and the large model **0.9583 / 0.9957**, i.e. the 2.2 GB model is *marginally behind* the 90 MB one; the two gains over rerank-off are +0.0625 ± 0.0427 (1.46σ) and +0.0521 ± 0.0449 (1.16σ), both again inside the band. recall@5 is 1.0000 in every configuration measured, because a 5-result window over a 6-to-24-chunk corpus returns most of it.
  - **What that means, stated as narrowly as the data allows.** Reranking moved MRR and nDCG in the right direction in every non-saturated case, and never moved recall. The move is consistently smaller than the noise of a 16-query golden set, and **no measured difference separates the 2.2 GB default from the 90 MB alternative** — on two of three comparisons they are identical, and on the third the small one is ahead. These corpora cannot distinguish them: a 12-to-24-chunk corpus is not a retrieval problem a cross-encoder is needed for. Pricing the quality side of TASK-0175 needs a golden set one or two orders of magnitude larger over a corpus big enough that recall@5 is not already 1.0 — not a re-run of this harness.
- Commit: 24ccc56350cedecccac4fc7e1f7f83fec4dadaa1 (working tree dirty: concurrent unrelated work on `chroma` settings and baseline fingerprints was present; the paths this measurement reads — `core/evals`, `core/rerank`, `core/retrieval`, `services/querying`, and both datasets' `golden.jsonl` and `corpus/` — were unmodified)
- Supersedes: none
