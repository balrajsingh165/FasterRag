# archive-format.md — Portability Archive Specification (D11)

The portable archive is the contract behind `fasterrag export` / `import` and `POST /v1/admin/export` / `import` ([differentiators.md](differentiators.md) D11). It is vendor-neutral by construction: nothing in it depends on which vector DB produced it. This spec is versioned; `import` validates `format_version` and refuses unknown majors.

## Container

A single `tar.gz` archive (extension `.fragx`) with this layout:

```text
archive.fragx
├── manifest.json          # REQUIRED — archive self-description (schema below)
├── checksums.sha256       # REQUIRED — SHA-256 of every other file; import verifies before reading
├── documents.jsonl        # REQUIRED — one source document record per line
├── chunks.jsonl           # REQUIRED — one chunk record per line
├── vectors.jsonl          # OPTIONAL — present only when exported with --include-vectors
├── index.lock             # REQUIRED — the lockfile of the exported collection (D1)
└── golden-set.jsonl       # OPTIONAL — the collection's golden set, if one exists
```

## `manifest.json` schema

| Field | Type | Required | Meaning |
|---|---|---|---|
| `format_version` | str (SemVer) | ✅ | Archive format version; this document specifies `1.0.0`. |
| `created_at` | str (ISO 8601) | ✅ | Export timestamp. |
| `fasterrag_version` | str | ✅ | Version that produced the archive. |
| `collection.name` | str | ✅ | Source collection name. |
| `collection.distance` | str | ✅ | `cosine` \| `dot` \| `euclid`. |
| `embedding.provider` / `embedding.model` / `embedding.model_version` | str | ✅ | Exactly what embedded the vectors (mirrors `index.lock`). |
| `embedding.dimensions` | int | ✅ | Vector dimensionality. |
| `chunking.strategy` / `chunking.chunk_size` / `chunking.overlap` / `chunking.contextual_enrichment` | mixed | ✅ | Chunking settings the chunks were produced with. |
| `counts.documents` / `counts.chunks` / `counts.vectors` | int | ✅ | Row counts; import verifies against actual line counts. |
| `includes_vectors` | bool | ✅ | Whether `vectors.jsonl` is present. |
| `source_provider` | str | ✅ | Informational only (e.g. `qdrant`) — import must not branch on it. |
| `tenant` | str\|null | — | Tenant scope of the export, when multi-tenancy is on. |

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

## Import semantics

1. **Verify first**: checksums, manifest counts, referential integrity (every chunk's `document_id` resolves; every vector's `chunk_id` resolves), and vector width (every row in `vectors.jsonl` is `embedding.dimensions` long, since that is the width the target collection is created at). Any failure → `VALIDATION_FAILED` problem, nothing written.
2. **Vector copy path** (no `--reembed`): allowed **iff** `includes_vectors` and the target collection's model, model version, and dimensions all match the manifest — otherwise import refuses and names every mismatch at once (`CONFLICT`). The target's identity is read from its own `index.lock` when it already has one, because that is what actually built it; otherwise from `embeddings.model` and `embeddings.dimensions`, because that is what will query it next. A target with no lockfile declares no model version, and a value nobody declares is not compared.
3. **Re-embed path** (`--reembed`): ignores `vectors.jsonl`; chunks flow through the normal embedding pool under the current config; the manifest's chunking settings are preserved as chunk metadata.
4. **Provenance is rejoined**: `source_uri`, `content_hash`, and the document's `metadata` live in `documents.jsonl`, not on the chunk row. Import joins them back onto every chunk payload — a restored chunk without a `source_uri` cannot be cited, and one without a `content_hash` cannot be deduplicated.
5. **Idempotent**: chunk ids are deterministic, so importing the same archive twice upserts over itself rather than duplicating. This is also the recovery path for a partial import.
6. **Nothing half-written survives**: if a write fails partway, a collection the import created is dropped and the original error is raised; a collection that already existed is left alone — dropping it would destroy rows the archive never carried — and the refusal names the mixed state and says that re-running completes it. A process killed outright cannot roll anything back; re-running the import is what repairs it.
7. **Lockfile**: the imported collection writes a fresh `index.lock` (model, dimensions, and the archive's per-document content hashes); the archive's own lock is retained beside it as `<collection>.source.lock.json` for provenance.

## Known gaps

- **`golden-set.jsonl` is specified but never written.** The exporter emits no golden set, and the reader neither expects nor reads one. It stays in the container listing as an optional member the format reserves. Tracked by TASK-0079.
- **Sparse vectors are not carried.** `vectors.jsonl` holds the dense vector only; the vendor-neutral read side (`VectorDBAdapter.iterate_points`) does not return a point's sparse vector at all. Import recomputes BM25 sparse vectors from chunk text under the target's `retrieval.bm25_k1`/`bm25_b` when `retrieval.hybrid` is on, which re-derives them under the target's parameters rather than transplanting the source's. Tracked by TASK-0079.

## Compatibility rules

- Minor format additions are additive-only (new optional fields); readers ignore unknown fields.
- A major `format_version` bump requires an explicit migration note in [CHANGELOG.md](CHANGELOG.md) and a converter path.
- The acceptance test for this spec is the D11 round-trip test ([differentiators.md](differentiators.md)): Qdrant → pgvector (`--reembed`) and Qdrant → Qdrant (vector copy), with zero chunk/metadata loss and eval metrics within regression-gate tolerance.
