"""A third-party chunker and parser, named in ``config.yaml``, shape what is indexed (TASK-0513).

Until TASK-0513 ``chunking.strategy`` was a closed ``Literal`` and ``create_chunker`` an
if-chain, and the parsers were a private dict keyed by suffix: neither could be extended
without a fork. Nothing inside fasterRag is replaced here. A fake distribution registers the
classes in ``tests/support/acme_pipeline_plugin.py`` under ``fasterrag.chunkers`` and
``fasterrag.parsers``; the configuration is a file read by the real loader; the ingest runs
the production pipeline, so the plugins are loaded again inside the parse workers.

The parser policy is part of what is proven: an installed parser package enables nothing on
its own. A suffix is read by a plugin only when ``ingestion.parsers`` maps it to that
plugin's name, so installing a package cannot change how an existing corpus is parsed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.config.loader import load_settings
from fasterrag.config.plugins import BUILTIN_CHUNKING_STRATEGIES
from fasterrag.config.schema import Settings
from fasterrag.core.chunking import CHUNKER_API_VERSION, create_chunker
from fasterrag.core.parsing import PARSER_API_VERSION, parse_bytes
from fasterrag.errors import ConfigError, IngestionError, ParseError
from fasterrag.facade import FasterRag
from fasterrag.services.sources import resolve_sources
from fasterrag.workers.cpu_pool import require_runnable_chunker
from tests.support.plugins import install_distribution

PROVIDERS = "tests.support.acme_plugin"
PIPELINE = "tests.support.acme_pipeline_plugin"

RECORDS = b"warranty: thirty days\ncolour: turbo red\nweight: four kilograms\n"

CONFIG = """\
vector_db:
  provider: acme
  api_key_env: null
embeddings:
  provider: acme
  model: acme-embed
llm:
  provider: acme
  model: acme-chat
  api_key_env: null
retrieval:
  rerank: false
chunking:
  strategy: acme-blocks
ingestion:
  parsers:
    .acme: acme-records
"""


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install one distribution registering providers, chunkers and parsers."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {
            "fasterrag.vectordb": {"acme": f"{PROVIDERS}:AcmeVectorDB"},
            "fasterrag.embeddings": {"acme": f"{PROVIDERS}:AcmeEmbedder"},
            "fasterrag.llm": {"acme": f"{PROVIDERS}:AcmeLLM"},
            "fasterrag.chunkers": {
                "acme-blocks": f"{PIPELINE}:AcmeBlockChunker",
                "acme-future": f"{PIPELINE}:FutureChunker",
                "acme-nothing": f"{PIPELINE}:NotAChunker",
            },
            "fasterrag.parsers": {
                "acme-records": f"{PIPELINE}:AcmeRecordParser",
                "acme-unversioned": f"{PIPELINE}:UnversionedParser",
            },
        },
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: None) -> Path:
    """Work from an empty deployment directory holding no ``.env``."""
    root = tmp_path / "deployment"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("FASTERRAG_API_KEY", "unit-test-control-plane-key")
    return root


def write_config(root: Path, text: str = CONFIG) -> Path:
    """Write a ``config.yaml`` into the deployment directory."""
    path = root / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def settings(chunking: dict[str, object] | None = None, **ingestion: object) -> Settings:
    """Validate settings with the given chunking and ingestion sections."""
    return Settings.model_validate({"chunking": chunking or {}, "ingestion": ingestion})


async def test_a_chunker_and_parser_named_in_config_yaml_ingest_and_answer_a_query(
    workspace: Path,
) -> None:
    """The defect: neither a chunking strategy nor a file format could come from a plugin."""
    document = workspace / "widget.acme"
    document.write_bytes(RECORDS)

    async with FasterRag.from_config(write_config(workspace)) as rag:
        job = await rag.ingest([str(document)])
        assert job.status == "completed"
        chunks = await rag.retrieve("how long is the warranty?", top_k=3)
        answer = await rag.query("how long is the warranty?")

    texts = sorted(chunk.text for chunk in chunks)
    assert texts == [
        "The colour is turbo red.",
        "The warranty is thirty days.",
        "The weight is four kilograms.",
    ]
    assert answer.answer is not None
    assert answer.citations


def test_an_installed_parser_reads_nothing_until_configuration_names_it(installed: None) -> None:
    """Installing a package must not change how an existing corpus is parsed."""
    with pytest.raises(ParseError, match=r"no parser is registered for '\.acme'"):
        parse_bytes(RECORDS, filename="widget.acme")

    document = parse_bytes(RECORDS, filename="widget.acme", parsers={".acme": "acme-records"})

    assert document.parser == "acme-records"
    assert document.text.startswith("The warranty is thirty days.")


def test_a_configured_parser_replaces_a_built_in_one_for_its_suffix(installed: None) -> None:
    """Naming a suffix is an explicit operator choice, so a built-in format may be taken over."""
    document = parse_bytes(RECORDS, filename="widget.txt", parsers={".txt": "acme-records"})

    assert document.parser == "acme-records"
    assert parse_bytes(RECORDS, filename="widget.txt").parser == "plaintext"


def test_a_plugin_chunker_is_built_by_the_chunker_factory(installed: None) -> None:
    """``create_chunker`` was an if-chain over five names."""
    chunker = create_chunker(settings({"strategy": "acme-blocks"}))
    document = parse_bytes(RECORDS, filename="widget.acme", parsers={".acme": "acme-records"})

    assert type(chunker).__name__ == "AcmeBlockChunker"
    assert chunker.split(document)[0].text == "The warranty is thirty days."


def test_an_unknown_strategy_fails_at_load_listing_what_is_installed(workspace: Path) -> None:
    """Fail-fast did not weaken: a typo still stops the load, and the error says what would work."""
    path = write_config(workspace, CONFIG.replace("strategy: acme-blocks", "strategy: acme-blox"))

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None, require_env=False)

    detail = caught.value.detail
    assert "chunking.strategy: 'acme-blox' is not an installed chunker" in detail
    assert "built-in chunkers: fixed, late, layout, recursive, semantic" in detail
    assert "'fasterrag.chunkers' entry-point group: acme-blocks, acme-future, acme-nothing" in (
        detail
    )


def test_a_parser_that_is_not_installed_fails_at_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same file on a machine without the package is refused before anything is built."""
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, "ingestion:\n  parsers:\n    .acme: acme-records\n")

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None, require_env=False)

    assert "'acme-records', which no installed package registers" in caught.value.detail
    assert "'fasterrag.parsers' entry-point group" in caught.value.detail


@pytest.mark.parametrize("suffix", ["acme", ".ACME", ".", ".a b"])
def test_a_parser_key_that_is_not_a_lower_case_suffix_is_refused(
    installed: None, suffix: str
) -> None:
    """A key that could never match a file's suffix would enable nothing, silently."""
    with pytest.raises(ValueError, match="lower-case file suffix"):
        settings(parsers={suffix: "acme-records"})


def test_a_chunker_built_for_another_api_is_refused_naming_both_versions(
    installed: None,
) -> None:
    """A plugin chunker is held to its declared interface version, like an adapter."""
    with pytest.raises(ConfigError) as caught:
        create_chunker(settings({"strategy": "acme-future"}))

    assert "fasterrag.chunkers entry point 'acme-future'" in caught.value.detail
    assert "chunker API 2.0" in caught.value.detail
    assert f"chunker API {CHUNKER_API_VERSION}" in caught.value.detail


def test_an_entry_point_that_is_not_a_chunker_is_refused(installed: None) -> None:
    """A class without ``split`` would fail at the first document instead."""
    with pytest.raises(ConfigError, match="is not a chunker class"):
        create_chunker(settings({"strategy": "acme-nothing"}))


def test_a_parser_with_no_api_version_is_refused(installed: None) -> None:
    """The parser API is versioned the same way as the adapter API."""
    with pytest.raises(ConfigError) as caught:
        parse_bytes(RECORDS, filename="widget.acme", parsers={".acme": "acme-unversioned"})

    assert "declares no adapter_api_version string" in caught.value.detail
    assert "the parser API this fasterrag provides" in caught.value.detail
    assert repr(PARSER_API_VERSION) in caught.value.detail


@pytest.mark.parametrize(
    ("chunking", "ingestion"),
    [
        ({"strategy": "acme-future"}, {}),
        ({}, {"parsers": {".acme": "acme-unversioned"}}),
    ],
)
def test_a_broken_plugin_stops_the_job_before_any_document_is_handed_out(
    installed: None, chunking: dict[str, object], ingestion: dict[str, object]
) -> None:
    """Refused only inside the workers, every document of the job would be dead-lettered."""
    with pytest.raises(ConfigError):
        require_runnable_chunker(settings(chunking, **ingestion))


async def test_an_inline_source_may_name_a_configured_suffix(installed: None) -> None:
    """The inline source check knew only the built-in suffixes."""
    configured = settings(parsers={".acme": "acme-records"})
    entry = {"type": "inline", "value": RECORDS.decode(), "filename": "widget.acme"}

    resolved = await resolve_sources([entry], configured)
    try:
        assert resolved.sources[0].endswith(".acme")
    finally:
        resolved.cleanup()

    with pytest.raises(IngestionError, match="no suffix a parser reads"):
        await resolve_sources([entry], settings())


@pytest.mark.parametrize("strategy", sorted(BUILTIN_CHUNKING_STRATEGIES - {"semantic"}))
def test_every_built_in_strategy_name_is_one_the_factory_builds_itself(strategy: str) -> None:
    """``config/plugins.py`` repeats the built-in names; a mismatch would look for a plugin."""
    assert create_chunker(settings({"strategy": strategy})).strategy == strategy


def test_the_semantic_strategy_is_built_in_too() -> None:
    """It needs an embedder, so it is refused for that reason, not as an unknown plugin."""
    with pytest.raises(ConfigError, match="needs an embedding model"):
        create_chunker(settings({"strategy": "semantic"}))
