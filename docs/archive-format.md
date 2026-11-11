# archive-format.md — Portability Archive Specification (D11)

The portable archive is the contract behind `fasterrag export` / `import` and `POST /v1/admin/export` / `import` ([differentiators.md](differentiators.md) D11). It is vendor-neutral by construction: nothing in it depends on which vector DB produced it. This spec is versioned; `import` validates `format_version` and refuses unknown majors.

## Container

A single `tar.gz` archive (extension `.fragx`) with this layout:

```text
archive.fragx
├── manifest.json          # REQUIRED — archive self-description (schema below)
├── checksums.sha256       # REQUIRED — SHA-256 of every other file, manifest included; import verifies before reading
├── documents.jsonl        # REQUIRED — one source document record per line
├── chunks.jsonl           # REQUIRED — one chunk record per line
├── vectors.jsonl          # OPTIONAL — present only when exported with --include-vectors
├── index.lock             # REQUIRED — the lockfile of the exported collection (D1)
└── golden-set.jsonl       # OPTIONAL — the collection's golden set, if one exists
```

## `manifest.json` schema

| Field | Type | Required | Meaning |
|---|---|---|---|
| `format_version` | str (SemVer) | ✅ | Archive format version. This document specifies `2.1.0`, which carries the keyword leg; an archive with no keyword leg declares `1.1.0` instead (see [Versioning](#versioning)). |
| `created_at` | str (ISO 8601) | ✅ | Export timestamp. |
| `fasterrag_version` | str | ✅ | Version that produced the archive. |
| `collection.name` | str | ✅ | Source collection name **as the exporting caller named it**, never the backend's tenant-scoped form (see [Tenancy](#tenancy)). |
| `collection.distance` | str | ✅ | `cosine` \| `dot` \| `euclid`. |
| `embedding.provider` / `embedding.model` / `embedding.model_version` | str | ✅ | Exactly what embedded the vectors. `model` and `model_version` mirror `index.lock`, and **import refuses a manifest that disagrees with it** ([Import semantics](#import-semantics) §9); `provider` has no counterpart there and is not cross-checked. |
| `embedding.dimensions` | int | ✅ | Vector dimensionality. |
| `chunking.strategy` / `chunking.chunk_size` / `chunking.overlap` / `chunking.contextual_enrichment` | mixed | ✅ | Chunking settings the chunks were produced with. |
| `counts.documents` / `counts.chunks` / `counts.vectors` | int | ✅ | Row counts; import verifies against actual line counts. |
| `includes_vectors` | bool | ✅ | Whether `vectors.jsonl` is present. |
| `includes_sparse` | bool | — | Present and `true` only when the vector rows carry a `sparse` keyword leg. Its presence is what makes the archive a `2.x` one; import refuses an archive whose rows and manifest disagree. |
| `source_provider` | str | ✅ | Informational only (e.g. `qdrant`) — import must not branch on it. |
| `tenant` | str\|null | — | Tenant the export was taken as, when multi-tenancy is on. Provenance only: import does not compare it with the importing tenant (see [Tenancy](#tenancy)). |

## Record schemas (JSONL, one object per line, UTF-8)

**`documents.jsonl`**

| Field | Type | Required |
|---|---|---|
| `document_id` | str | ✅ |
| `source_uri` | str | ✅ |
| `content_hash` | str (SHA-256) | ✅ |
| `version` | int | ✅ |
| `metadata` | object | ✅ (may be `{}`) |
| `parse_flags` | list[str] | — (e.g. `low_text_yield`) |

**`chunks.jsonl`**

| Field | Type | Required |
|---|---|---|
| `chunk_id` | str | ✅ (deterministic — same rules as the indexer) |
| `document_id` | str | ✅ (must exist in `documents.jsonl`) |
| `text` | str | ✅ |
| `span` | `{start: int, end: int}` | ✅ (character offsets into the parsed document) |
| `page` | int\|null | — |
| `metadata` | object | ✅ |
| `context_prefix` | str\|null | — (the contextual-enrichment prefix, when enabled) |

**`vectors.jsonl`** (only with `--include-vectors`)

| Field | Type | Required |
|---|---|---|
| `chunk_id` | str | ✅ |
| `vector` | list[float] | ✅ (length == `embedding.dimensions`) |
| `sparse` | `{indices: list[int], values: list[float]}` | — (the BM25 keyword leg, present only when the source collection held one) |

A `sparse` entry is the stored keyword leg exactly as the backend held it: term indices from
`core.retrieval.bm25.term_index` paired with the saturated term frequencies computed at
ingest, the two lists equal in length and the indices **strictly ascending**. Ascending order
is the writer's guarantee, not the backend's, so two exports of an unchanged collection stay
byte-identical whatever order a vendor's walk returns. A duplicate index is refused on import:
a backend keeps whichever value it sees last, so the same postings would score differently on
each one.

IDF is deliberately absent, as it is everywhere else in the system ([ADR-0007](adr/ADR-0007-bm25-as-backend-sparse-vectors.md)):
it is a statistic of the live corpus that each backend derives per query. What the archive
carries is the half that `k1` and `b` are already baked into, which is exactly the half a
different deployment cannot re-derive identically.

## Versioning

| Declared `format_version` | Written when | Readable by |
|---|---|---|
| `1.0.0` | the archive carries no keyword leg | every build; **no longer written, and no longer read** (see the minor below) |
| `2.0.0` | any vector row carries `sparse` | builds from TASK-0302 (2026-11-04) onward; **no longer written, and no longer read** |
| `1.1.0` | no keyword leg, and `checksums.sha256` covers `manifest.json` | every build |
| `2.1.0` | any vector row carries `sparse`, and `checksums.sha256` covers `manifest.json` | builds from TASK-0302 onward |

The version states what a reader must understand to read the archive *faithfully*, not which
build wrote it. A pre-TASK-0302 reader validates only the major and ignores unknown fields, so
handed a `sparse` field it would import a collection whose keyword ranking silently differs
from the source's — the one failure mode worse than a refusal, because nothing reports it. The
major bump is what converts that into a clean refusal. An archive that gained no new data
declares `1.0.0` and is not stranded over a feature it does not use.

Readers from TASK-0302 onward accept majors `1` and `2`. A `1.x` archive therefore still
imports; see [Import semantics](#import-semantics) §8 for what it does about the missing leg.

**Converter path (2.x → 1.x)**, for a deployment that must feed an older reader: drop the
`sparse` key from every row of `vectors.jsonl`, set `format_version` to `1.1.0`, remove
`includes_sparse` from `manifest.json`, and recompute `checksums.sha256`. The result is a
valid `1.1.0` archive whose keyword leg is recomputed from chunk text on import. No converter
is needed in the other direction: a `1.x` archive is read as-is.

### The `x.1` minor: the manifest is checksummed (TASK-0329)

Until this minor, `checksums.sha256` listed the data members only and the manifest was
written *after* it, so the one member no reader verified was the one that declares the
embedding space a vector copy is checked against. An archive whose manifest had been
rewritten to another `model`/`model_version` and repacked, with no checksummed member
touched, passed verification and then passed the vector-copy guard. Nothing else catches
it: counts, widths and referential integrity all compare the manifest against the rows, and
`model`/`model_version` have no row to disagree with.

The minor is decided in both directions, and the two directions are not symmetric:

- **An older reader handed a `x.1` archive** loses nothing. `checksums.sha256` has been read
  as a *list* by every reader that ever shipped, not as a fixed set of names, so an older
  reader verifies the manifest line too. Nothing is dropped and nothing is silently
  misread, which is the test this format applies to a major bump — so this is a minor.
- **A `x.1` reader handed a `x.0` archive refuses it.** The manifest is unverified there,
  and that is exactly the defect. The requirement is **unconditional**: a reader may not
  decide whether to demand the manifest's digest by reading a field *of the manifest*,
  because an attacker who strips the line also lowers the version beside it. So the rule is
  "`manifest.json` appears in `checksums.sha256`, or the archive is refused", and the minor
  is a statement to operators rather than the thing enforcing it.

**The softer rule was considered and rejected.** An `x.0` archive could have been accepted
for the re-embed path only — that path derives its own vectors, so an unverified
`embedding.model` decides nothing — and refused only for a vector copy. It keeps every
existing archive importable, and it was turned down because it makes "is this manifest
trusted?" a property that varies per call site: the next guard added over a manifest field
would have to re-derive the answer, and the one that forgot would be this defect again. A
reader either verified the manifest or it did not. The converter below is one appended line
and no data change, which is what makes the strict rule cheap enough to hold.

**Converter path (x.0 → x.1)**: append `<sha256 of manifest.json>  manifest.json` to
`checksums.sha256` and raise the manifest's minor to `1`. No data member changes. The
converter is the honest statement of what the check is worth: **checksums are integrity,
not authenticity.** They catch a corrupted or partially rewritten archive, and they do not
stop someone who repacks the whole thing — that person can run the converter too. The
second rule below raises the cost (the forgery must now also edit `index.lock`), and only
signing the archive would settle it. The format does not sign yet.

One consequence is deliberate: `checksums.sha256` now varies between two exports of an
unchanged collection, because the manifest carries `created_at`. Every other member stays
byte-identical.

## Import semantics

1. **Verify first, and stream while verifying**: the reader copies every member out of the tar into a private staging directory in fixed-size blocks, hashing each one as it lands, and then checks checksums — **including `manifest.json`'s, and refusing any archive that does not list it** — manifest counts, referential integrity (every chunk's `document_id` resolves; every vector's `chunk_id` resolves), and vector width (every row in `vectors.jsonl` is `embedding.dimensions` long, since that is the width the target collection is created at) by reading those staged files back a line at a time. Any failure → `VALIDATION_FAILED` problem, nothing written to the target and the staging directory removed. Streaming does not weaken the ordering: the bytes that were hashed are the same bytes on disk that every verification pass and the writer then read, and all of it still happens before the first upsert.
2. **Vector copy path** (no `--reembed`): allowed **iff** `includes_vectors` and the target collection's model, model version, and dimensions all match the manifest — otherwise import refuses and names every mismatch at once (`CONFLICT`). The target's identity is read from its own `index.lock` when it already has one, because that is what actually built it; otherwise from `embeddings.model` and `embeddings.dimensions`, because that is what will query it next. A target with no lockfile declares no model version, and a value nobody declares is not compared.
3. **Re-embed path** (`--reembed`): ignores `vectors.jsonl`; chunks flow through the normal embedding pool under the current config; the manifest's chunking settings are preserved as chunk metadata.
4. **Provenance is rejoined**: `source_uri`, `content_hash`, and the document's `metadata` live in `documents.jsonl`, not on the chunk row. Import joins them back onto every chunk payload — a restored chunk without a `source_uri` cannot be cited, and one without a `content_hash` cannot be deduplicated.
5. **Idempotent**: chunk ids are deterministic, so importing the same archive twice upserts over itself rather than duplicating. This is also the recovery path for a partial import.
6. **Nothing half-written survives**: if a write fails partway, a collection the import created is dropped and the original error is raised; a collection that already existed is left alone — dropping it would destroy rows the archive never carried — and the refusal names the mixed state and says that re-running completes it. A process killed outright cannot roll anything back; re-running the import is what repairs it.
7. **The keyword leg is transplanted, not re-derived**: when the archive carries `sparse`
   postings and the target has `retrieval.hybrid` on, they are written through unchanged, so
   the restored collection ranks keyword queries exactly as the source did and the weights
   keep the `k1`/`b` the corpus was indexed with. With `retrieval.hybrid` off the target has
   no keyword leg to store them in and they are dropped, because the collection is created
   dense-only.
8. **An archive with no keyword leg still gets one**: a `1.0.0` archive, or any row without
   `sparse`, falls back to recomputing BM25 from the chunk text under the target's
   `retrieval.bm25_k1`/`bm25_b`. That re-derives the weights rather than transplanting them,
   which is the pre-TASK-0302 behaviour, and it is kept because a collection with no keyword
   leg at all answers hybrid queries on one leg and reports nothing. `--reembed` recomputes
   for the same reason it re-embeds: that path rebuilds the corpus under the target's
   configuration, and the BM25 parameters are part of it.
9. **The manifest must agree with `index.lock`**: `embedding.model`, `embedding.model_version`,
   `embedding.dimensions`, `chunking.strategy`, `chunking.chunk_size` and `chunking.overlap`
   are compared against the archived lockfile, which the exporter built them from, and any
   disagreement is a `VALIDATION_FAILED` naming all of them. A field the lock leaves empty
   is not compared, and an archive from a deployment that keeps no lockfiles carries `{}`
   and is not cross-checked at all. This is what gives the embedding identity a row to
   disagree with, the way the counts and the vector widths already have one (TASK-0329).
10. **Lockfile**: the imported collection writes a fresh `index.lock` (model, dimensions, and the archive's per-document content hashes); the archive's own lock is retained beside it as `<collection>.source.lock.json` for provenance.

## Tenancy

`collection.name` is the name the **exporting caller** used, never the backend's scoped
form. Under `security.multi_tenancy` a request for `docs` from tenant `acme` addresses the
backend collection `acme__docs`, and recording that made import scope an already-scoped
name into `acme__acme__docs` and answer with `acme__docs` — a backend-prefixed name in a
200 body, which no tenant ever chose (TASK-0340). The exporter unscopes before building the
manifest; for a single-tenant deployment, and for the CLI, this is a no-op.

`index.lock` is exempt, deliberately: it is a verbatim copy of the source collection's
lockfile and its `collection` field therefore still reads `acme__docs`. Rewriting it would
make it something other than the copy it is advertised as, and nothing reads it as a target
name — import writes a fresh lock for the target and keeps this one beside it as
`<collection>.source.lock.json`. The prefix travels inside the archive; it no longer
travels into a collection name or a response body.

`tenant` keeps the provenance the name no longer carries. Import does **not** compare it
with the importing tenant, and that is a deliberate choice rather than an omission: an
archive is a corpus, not a grant, so one exported by `acme` imports cleanly under `globex`
into `globex__docs` and is owned by `globex` from then on. The authorization boundary is
who may call import and which directories `security.allowed_archive_roots` permits, not a
string inside a file the caller supplied. A deployment that wants archives pinned to one
tenant must not hand them to another operator.

**Known gap**: import stamps no tenant onto the restored chunk payloads. A chunk's `tenant`
field travels in the chunk row's `metadata` and is restored verbatim, so an archive from
`acme` imported by `globex` lands rows tagged `acme` in `globex__docs` — invisible to
`globex` under `security.tenant_scoped_retrieval`, and invisible to `acme`, whose queries
address another collection. An archive from an untenanted deployment carries no tag at all
and is equally invisible. Measured 2026-11-11, not inferred. `fasterrag index tenants`
([security.md](security.md)) is the repair until import stamps the tag itself.

## Resource bounds on import

A collection is precisely the thing that does not fit in memory, so neither side of the archive may assume it does: export streams rows through a staging directory, and import (TASK-0293) streams them back.

- **Memory is bounded by row counts, not by archive size.** Peak is a small multiple of the reader's block size while members are staged, plus two things that scale with rows rather than bytes: the document rows (`documents.jsonl`), which the payload join needs per chunk and which the fresh `index.lock` already requires one hash of each anyway; and, **only** for an archive whose vectors are not in chunk order, one file offset per vector row.
- **Vector order decides which read path is taken.** The exporter writes `vectors.jsonl` in the same order as `chunks.jsonl`, so the two are read in lockstep and one vector row is held at a time. The format does not *require* that order — an archive whose vectors are shuffled is still valid, is still verified before anything is written, and is read through the offset index instead, which costs memory per vector row.
- **Disk is the trade.** Staging needs free temporary space for the uncompressed archive. The directory is removed when the reader is closed, when it is garbage collected, and on any verification failure.
- **The decompression-bomb ceilings are unchanged** (`docs/security.md` §5.2): 1 GiB per member, refused on the tar header, and 4 GiB per archive as a running budget charged against bytes actually read. The budget is spent block by block, so an archive that breaches it is refused on the block that crosses the line rather than after a large read.

## Known gaps

- **`golden-set.jsonl` is specified but never written.** The exporter emits no golden set, and the reader neither expects nor reads one. It stays in the container listing as an optional member the format reserves. Tracked by TASK-0079.
- **Checksums are not signatures.** Every member including the manifest is digested, which
  catches corruption and a partial rewrite; it does not catch an attacker who repacks the
  archive and recomputes the list, because nothing in the file is bound to a key. The
  manifest/`index.lock` cross-check raises the cost of such a forgery — two members must now
  agree — and does not close it. Signing the archive would; it is unbuilt and unscheduled,
  and no claim anywhere should say an archive is authenticated. Tracked by TASK-0079.
- **The keyword leg is proven against a fake adapter only.** `vectors.jsonl` carries `sparse` as of TASK-0302 (2026-11-04), end to end: the adapters return it (TASK-0291), the writer records it, and import transplants it. What has actually run is the unit suite over a fake adapter plus the format-spec pins. The live assertion — that a restored collection's keyword *ranking and scores* match the source's — is `tests/integration/test_archive_round_trip.py::test_the_keyword_ranking_and_scores_survive_the_round_trip`, which skips without a Docker daemon (B7.2). Tracked by TASK-0079.

## Compatibility rules

- Minor format additions are additive-only (new optional fields, or a new line in `checksums.sha256`); readers ignore unknown fields, and read the checksums member as a list rather than as a fixed set of names — which is what let the `x.1` minor stay a minor.
- A major `format_version` bump requires an explicit migration note in [CHANGELOG.md](CHANGELOG.md) and a converter path. The `2.0.0` bump (TASK-0302) and the `x.1` minor (TASK-0329) have their converter paths under [Versioning](#versioning) and their migration notes in [CHANGELOG.md](CHANGELOG.md).
- A reader accepts every major it can read faithfully, not only the one it writes: this build reads `1` and `2`.
- **A reader may refuse an older *minor* when the minor is what makes the archive verifiable.** This build refuses `1.0.0` and `2.0.0` because their manifests are unchecksummed, which is the whole of TASK-0329; the converter is one appended line. That is the only case, and anything else refused on a minor is a bug.
- **A verification rule may not depend on a field of the member it verifies.** The manifest cannot be asked whether it must be checksummed, nor can `format_version` decide whether `format_version` is trusted. Rules about the manifest are unconditional, or they are decoration. Dropping an older major would strand archives, and accepting a newer one would import data the reader cannot interpret.
- The acceptance test for this spec is the D11 round-trip test ([differentiators.md](differentiators.md)): Qdrant → pgvector (`--reembed`) and Qdrant → Qdrant (vector copy), with zero chunk/metadata loss and eval metrics within regression-gate tolerance.
