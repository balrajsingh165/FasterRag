# Writing a plugin

A plugin adds a vector database, an embedding provider or an LLM provider to fasterRag without
forking it. It is an ordinary Python package that subclasses one of the adapter base classes and
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

Chunkers, parsers and rerankers have no entry-point group yet (TASK-0513).

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

## When the interface changes

The names the base modules export, the abstract members of each base class and the group names
are held by `tests/unit/test_public_surface.py`. Removing a name, or changing what a plugin must
implement without moving `ADAPTER_API_VERSION`, fails fasterRag's own build. A major bump means
existing plugins are refused until their authors re-test and re-declare; a minor bump adds
members with defaults, and plugins built for an older minor version keep loading.
