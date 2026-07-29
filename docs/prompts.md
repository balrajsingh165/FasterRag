# prompts.md — LLM Prompt Contracts

fasterRag calls an LLM in exactly **four** places. Each call has a defined purpose, input contract, output contract, and failure behavior. Prompt wording is quality-critical and provider-independent: templates ship with the package, are versioned, and are overridable — but the **contracts** below are fixed, because the pipeline parses the outputs.

> Templates below are the specified defaults, shown as illustrative snippets. Prompt-template changes are retrieval/quality-affecting and therefore run the regression gate (D7) like any other quality change.

| # | Call site | When | Config gate | Failure behavior |
|---|---|---|---|---|
| P1 | **Answer generation** | Every non-cached query | always (the core path) | `GenerationError` → degradation ladder `extractive` mode (D4) |
| P2 | **Contextual enrichment** | Per chunk, at ingestion | `chunking.contextual_enrichment: true` | Chunk indexed **without** prefix + `parse_flags` note; never dead-letters the document |
| P3 | **Faithfulness scoring** | Per answer | `generation.grounded_or_refuse: true` (or when faithfulness is being recorded) | Score `null`; answer returned un-gated with `faithfulness: null` |
| P4 | **Golden-set generation** | Autopilot / eval setup | `autopilot.enabled: true` or explicit CLI | Aborts the generation run with a typed error; never writes a partial set |

Prompt-template versioning: each template carries a `template_version`; it is part of the `config_hash` in `index.lock` for P2 (enrichment changes chunk text, so it is drift) and part of the trace `config_snapshot` for P1/P3 (so replay is meaningful — D8).

---

## P1 — Answer generation

**Purpose.** Answer the user's question **only** from the assembled context, with span-level citations.

**Input contract.** System prompt + a user turn containing: the assembled context (top-K chunks, each with a stable reference marker and its `source`/`page`), then the question. Context is assembled within the token budget by [context assembly](architecture.md); chunks arrive in final rank order.

**Output contract.** Free-form answer text with inline reference markers of the form `[^c_9f2]` matching the chunk ids supplied in context. The citation resolver maps markers → `Citation` objects ([data-model.md](data-model.md)); a marker that doesn't resolve to a supplied chunk is dropped and logged (invariant 2: citations can never reference unretrieved content).

```text
SYSTEM
You answer strictly from the provided context. If the context does not contain
enough information to answer, say so plainly instead of guessing — a partial or
absent answer is correct behavior, a fabricated one is not.

Cite the specific chunk that supports each factual claim using its marker, e.g. [^c_9f2].
Every factual sentence needs a marker. Do not cite chunks you did not use.
Do not use outside knowledge, even if you are confident it is correct.
Quote exact figures, dates, and identifiers rather than paraphrasing them.

USER
<context>
[^c_9f2] (source: contracts/vendor-2024.pdf, page 12)
Either party may terminate this agreement with thirty (30) days written notice...

[^c_a01] (source: contracts/vendor-2024.pdf, page 13)
Termination for cause requires...
</context>

Question: What does the vendor agreement say about termination?
```

**Streaming.** Emitted token-by-token; citations are resolved and sent as one `citations` SSE event once retrieval concludes ([api-reference.md](api-reference.md)).

**Prompt caching.** The system prompt and any stable instruction block are placed first so provider prompt caching can hit across queries; volatile content (context, question) goes last.

---

## P2 — Contextual enrichment

**Purpose.** Generate the short document-level context prepended to a chunk before embedding and BM25 indexing — the Contextual Retrieval technique (−49% failed retrievals, −67% with reranking; [references.md](references.md) R1).

**Input contract.** The **whole parent document** plus the single chunk to situate. The parent document is sent as a cacheable prefix so the provider's prompt cache absorbs it across all chunks of that document — this is what makes per-chunk enrichment affordable.

**Output contract.** Plain text only, **no preamble, no markers**, target `chunking.context_tokens` (default 75, valid 25–150). The output is stored as `context_prefix` and prepended to `original_text` to form `text` ([data-model.md](data-model.md)).

```text
SYSTEM
You write a short situating context for a chunk of a larger document, so the chunk
can be retrieved and understood on its own.

Output the context only — no preamble, no explanation, no quotation marks.
Target 50-100 tokens. State what section this is from, what entity or subject it
concerns, and any referent a pronoun or shorthand in the chunk depends on.
Do not summarize the chunk itself and do not add information absent from the document.

USER
<document>
{{ full parent document }}
</document>

<chunk>
{{ chunk text }}
</chunk>
```

**Cost control.** One call per chunk, mitigated by (a) parent-document prompt caching, (b) the short output target, and (c) tiered routing — enrichment may use a cheaper model than P1 via `embeddings.tiering`-style class routing. `fasterrag estimate` accounts for enrichment calls when the toggle is on (D9).

**Failure behavior.** A failed enrichment call is **non-fatal**: the chunk is indexed without a prefix and flagged, because a slightly worse chunk beats a dead-lettered document.

---

## P3 — Faithfulness scoring

**Purpose.** Produce the 0–1 score that gates grounded-or-refuse (D5) and feeds the `fasterrag_faithfulness` metric.

**Input contract.** The assembled context, the question, and the generated answer. Runs **after** generation, as a separate call — deliberately not folded into P1, so the grader never sees its own generation instructions and cannot self-justify.

**Output contract.** Strict JSON, schema-constrained where the provider supports it:

```json
{"score": 0.93,
 "unsupported_claims": ["The agreement was signed in 2019."],
 "reasoning": "All claims except the signing date are directly supported by [^c_9f2]."}
```

| Field | Type | Meaning |
|---|---|---|
| `score` | float 0–1 | Fraction of the answer's factual claims supported by the supplied context |
| `unsupported_claims` | list[str] | Claims not traceable to context — surfaced in traces for debugging |
| `reasoning` | str | Short justification; stored in the trace, never returned to end users by default |

```text
SYSTEM
You grade whether an answer is supported by its context. You are not the author of
the answer and you do not improve it — you only judge support.

Break the answer into factual claims. For each, decide whether the context directly
supports it. Score = supported claims / total claims. Opinions, hedges, and explicit
statements of uncertainty are not claims and are excluded from the count.
An answer that correctly says the context is insufficient scores 1.0.

Respond with JSON only: {"score": <0-1>, "unsupported_claims": [...], "reasoning": "..."}
```

**Gating.** `score < generation.faithfulness_threshold` → the answer is withheld and an `INSUFFICIENT_EVIDENCE` response is returned with `best_candidates`. A `null` score (grader failed) never withholds an answer — failing closed on a grader outage would turn a monitoring problem into an availability problem.

**Model choice.** May use a different (typically cheaper/faster) model than P1; both are recorded in the trace.

---

## P4 — Golden-set generation

**Purpose.** Build the golden Q&A set Autopilot (D6) and the regression gate (D7) measure against, from the user's own corpus.

**Input contract.** A sampled chunk (plus its document context) per generated record; sampling is stratified across documents and metadata classes so the set isn't dominated by one source.

**Output contract.** JSON matching the golden-set record schema in [testing-strategy.md](testing-strategy.md) §1.6, written as JSONL with `source: "autopilot"`.

```text
SYSTEM
You write evaluation questions for a retrieval system, from a passage of a real corpus.

Write questions a real user of this corpus would ask, whose answer is contained in
the passage. Use the corpus's own terminology. Make the question answerable without
seeing the passage — no "according to the text" or "in this section".
Avoid questions answerable from general knowledge alone.

Respond with JSON only:
{"query": "...", "answer_reference": "...", "unanswerable": false}
```

**Adversarial records.** A fraction of the set is generated as deliberately unanswerable questions (`unanswerable: true`, empty `relevant_chunk_ids`, `answer_reference: null`) — these are what test grounded-or-refuse (D5) and prevent tuning that simply makes the system answer more confidently.

**Provenance.** Generated records carry `source: "autopilot"`. Promotion to `source: "human"` requires human review — an ungoverned generated set would let the system grade its own homework, and Autopilot's suggestions are only as trustworthy as the set behind them.

---

## Overriding templates

Templates are package resources with a documented override path; overriding is a **quality-affecting change**:

- P2 overrides alter chunk text → change the `config_hash` → register as index drift (D1) and require a reindex to take effect on existing chunks.
- P1/P3 overrides do not change the index but do change answers → the regression gate should be run before adopting one.
- Overrides are recorded in the trace `config_snapshot`, so replay (D8) compares like with like.

## Invariants (asserted by tests)

1. P1 output markers that don't correspond to a supplied chunk never reach a `Citation`.
2. P2 output is never allowed to exceed the `context_tokens` bound after truncation, and a failed P2 never fails the document.
3. P3 is a separate call from P1 in every code path — no configuration merges them.
4. A `null` faithfulness score never withholds an answer.
5. P4 never writes a golden set containing zero adversarial records when adversarial generation is enabled.
