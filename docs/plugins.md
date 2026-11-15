# Writing a plugin

A plugin adds a vector database, an embedding provider, an LLM provider, a chunking strategy or
a file-format parser to fasterRag without forking it. It is an ordinary Python package that
subclasses one of the adapter base classes (or implements the chunker or parser shape) and
registers the subclass under an entry-point group; once it is installed, one line of
`config.yaml` selects it. This page walks through the whole job using the sample package in the
repository, `examples/plugin-vectordb/`, which CI installs with pip and runs against the shared
contract suite on every push.

The reference for the extension surface — every abstract member, the credential rules, the
`provider` attribute — is [python-api.md](python-api.md#extending-the-framework-plugin-contract).
This page is the procedure.

## 1. Pick the group and the base class

| You are adding | Entry-point group | Subclass | Base module |
|---|---|---|---|
| A vector database | `fasterrag.vectordb` | `VectorDBAdapter` | `fasterrag.adapters.vectordb.base` |
| An embedding provider | `fasterrag.embeddings` | `EmbeddingAdapter` | `fasterrag.adapters.embeddings.base` |
| An LLM provider | `fasterrag.llm` | `LLMAdapter` | `fasterrag.adapters.llm.base` |

| A chunking strategy | `fasterrag.chunkers` | — (implements `Chunker`) | `fasterrag.chunking` |
| A file-format parser | `fasterrag.parsers` | — (implements `ParserPlugin`) | `fasterrag.parsing` |

Chunkers and parsers are plain classes rather than subclasses; [Chunkers and
parsers](#chunkers-and-parsers) below gives their shape. Rerankers have no entry-point group yet
(TASK-0513).

## 2. Implement the adapter

Subclass the base class and implement every abstract member; the constructor receives the
validated `Settings`. Use only the types in the base module and the typed errors in
`fasterrag.errors` — a vendor's own exception must never escape the adapter. The sample's
`src/fasterrag_memory_vectordb/adapter.py` is a complete vector database in about 300 lines:
dense scoring for all three distances, the shared filter grammar (checked with
`validate_filter`), the BM25 keyword leg with IDF taken from the live collection, aliases and
snapshots.

Declare the adapter API you built against, as a string literal:

```python
from typing import ClassVar

from fasterrag.adapters.vectordb.base import VectorDBAdapter


class MemoryVectorDB(VectorDBAdapter):
    adapter_api_version: ClassVar[str] = "1.0"
```

Each base module exports `ADAPTER_API_VERSION` (`MAJOR.MINOR`, `"1.0"` for all three today).
The factory compares your declaration with it when it loads your entry point, and refuses the
plugin with a `ConfigError` (`CONFIG_INVALID`) that names both versions when:

- the major versions differ — the interface changed incompatibly;
- your minor version is newer — you may call members this installation does not have;
- the attribute is missing, is not a string, or is not `MAJOR.MINOR`.

Write the literal; do not assign `ADAPTER_API_VERSION` to it. An assignment reads whatever
version is installed when your module is imported, so it always matches and the check proves
nothing. Embedding and LLM adapters also set the class attribute `provider` to the name they
register (see [python-api.md](python-api.md#extending-the-framework-plugin-contract)).

## 3. Register it

In your package's `pyproject.toml`:

```toml
[project.entry-points."fasterrag.vectordb"]
memory = "fasterrag_memory_vectordb:MemoryVectorDB"
```

The name on the left is what `config.yaml` selects. A built-in name (`qdrant`, `chroma`, …)
cannot be taken over: built-ins always win, so a package cannot silently intercept a deployment.

## 4. Pass the contract suite

The shared contract suite, `tests/contract/vectordb.py` in the fasterRag repository, is what
every built-in vector database passes, and a plugin is conformant only when it passes it too.
Subclass `VectorDBContract` and supply three fixtures: `adapter`, `collection` (a fresh, empty
collection with `DIMENSIONS` dimensions) and `hybrid_collection` (the same with
`sparse=True`). The sample's `tests/test_contract.py` builds the adapter through the factory,
by name, so a pass proves the entry point and the version check as well as the behaviour:

```bash
pip install ./examples/plugin-vectordb
python -m pytest examples/plugin-vectordb/tests
```

The suite ships in the source distribution and the repository, not in the wheel, so run it
from a fasterRag checkout (or an unpacked sdist) with your package installed. For an embedding or
LLM provider, follow the recorded-response pattern in `tests/unit/adapters/`.

## 5. Select it

```yaml
vector_db:
  provider: memory
  api_key_env: null   # always written out for a plugin: a variable name, or null
```

Validation checks the name against installed entry points without importing your code, so a
typo fails at load and lists every name that would have worked. If your plugin needs a
credential, read it through the section's `api_key_env` and call `require_key` /
`require_llm_key` from the base module when the adapter is built.

## Chunkers and parsers

The same steps apply — implement, declare the version, register, select — with these shapes.

**A chunker** is a class constructed as `cls(settings, counter=counter)` — the validated
`Settings` and the pipeline's `TokenCounter` — whose `split(document)` returns a list of
`TextChunk` in document order, with `start`/`end` offsets into `document.text` and
`strategy` set to the registered name. It declares `adapter_api_version` against
`fasterrag.chunking.CHUNKER_API_VERSION` (`"1.0"`):

```toml
[project.entry-points."fasterrag.chunkers"]
sentences = "my_pkg.chunker:SentenceChunker"
```

```yaml
chunking:
  strategy: sentences
```

`chunking.strategy` is validated like a provider name: a built-in name (`fixed`, `recursive`,
`semantic`, `layout`, `late`) always wins, and an unknown name stops the load listing the
built-in and installed chunkers. A chunker reads `chunking.chunk_size` and `chunking.overlap`
from `settings` if it wants them; it gets no keys of its own. The strategy name is part of the
chunker hash that chunk ids are derived from, so if your chunker's boundaries change between
releases, register the new behaviour under a new name or tell operators to reindex.

**A parser** is a class constructed with no arguments whose
`parse(data, *, filename, options)` returns a `ParsedDocument`, built with
`fasterrag.parsing.DocumentBuilder` so the block offsets stay consistent. It declares
`adapter_api_version` against `fasterrag.parsing.PARSER_API_VERSION` (`"1.0"`). The entry-point
name is the parser's name, not a suffix:

```toml
[project.entry-points."fasterrag.parsers"]
acme-rtf = "my_pkg.parser:RtfParser"
```

```yaml
ingestion:
  parsers:
    .rtf: acme-rtf
```

**Installing a parser package enables nothing by itself.** A suffix is read by a plugin only
when `ingestion.parsers` maps it to that plugin's name. A built-in parser is chosen by suffix
with no configuration, so a package that claimed suffixes on installation could silently
change how an existing corpus is parsed — for every deployment that happened to have it in
the environment, and differently on two machines with different packages. Keeping the choice
in `config.yaml` makes it reviewable and the same everywhere the file is. It also makes the
replacement of a built-in parser explicit: map `.pdf` to your plugin and it reads PDFs, while
an operator who did not write that line keeps the built-in one. Keying the entry point by name
rather than by suffix lets two installed packages offer a parser for the same suffix, with the
configuration choosing between them.

Both kinds are loaded and their versions checked in the process that accepts the job, before
any document is handed to the parse workers, so a broken or incompatible plugin refuses the
job with `CONFIG_INVALID` instead of dead-lettering every document. The workers then load the
plugin again themselves; it must be importable there, which an installed package always is.
`tests/unit/test_pipeline_plugins.py` proves the path end to end with a fake distribution.

## When the interface changes

The names the base modules export, the abstract members of each base class, the group names
and the chunker and parser API versions are held by `tests/unit/test_public_surface.py`.
Removing a name, or changing what a plugin must
implement without moving `ADAPTER_API_VERSION`, fails fasterRag's own build. A major bump means
existing plugins are refused until their authors re-test and re-declare; a minor bump adds
members with defaults, and plugins built for an older minor version keep loading.
