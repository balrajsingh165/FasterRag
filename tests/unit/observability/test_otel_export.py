"""OTLP export of query traces.

The mapping is asserted without a collector, because it is where this integration is most
likely to be wrong and it is invisible until someone opens a trace viewer.
"""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("opentelemetry.sdk", reason="OTLP export ships in the optional 'otel' extra")

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Span, Trace
from fasterrag.observability.otel_export import SERVICE_NAME, OtelExporter, build_spans, span_id_for
from fasterrag.services.traces import TraceStore, create_exporters, create_otel_exporter

TRACE_ID = "169a6d2cb76270357a33642877487b02"


def trace(**overrides: Any) -> Trace:
    fields: dict[str, Any] = {
        "trace_id": TRACE_ID,
        "query": "what is the UK meal allowance",
        "collection": "policies",
        "tenant": "acme",
        "retrieved": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "result": {
            "mode": "full",
            "degraded": False,
            "faithfulness": 0.91,
            "citations": [{"n": 1}],
            "usage": {"prompt_tokens": 386, "completion_tokens": 51},
        },
        "spans": [
            Span("retrieval", 0.0, 120.5, {"top_k": 10}),
            Span("reranker", 120.5, 410.0, {"model": "bge-reranker-v2-m3"}),
            Span("context-assembly", 410.0, 415.0, {}),
            Span("generation", 415.0, 1890.0, {"model": "gpt-4o-mini"}),
        ],
        "created_at": "2026-08-05T10:00:00+00:00",
    }
    fields.update(overrides)
    return Trace(**fields)


def test_a_root_span_wraps_every_stage() -> None:
    spans = build_spans(trace())

    assert [span.name for span in spans] == [
        "query",
        "retrieval",
        "reranker",
        "context-assembly",
        "generation",
    ]


def test_the_fasterrag_trace_id_is_preserved() -> None:
    """The id in a problem+json body must be the id pasted into a trace viewer's search."""
    spans = build_spans(trace())

    assert format(spans[0].context.trace_id, "032x") == TRACE_ID


def test_every_stage_shares_that_trace_id() -> None:
    spans = build_spans(trace())

    assert {format(span.context.trace_id, "032x") for span in spans} == {TRACE_ID}


def test_stages_are_parented_to_the_root() -> None:
    """Unparented stages render as separate traces, one per stage."""
    spans = build_spans(trace())
    root = spans[0]

    assert all(span.parent is not None for span in spans[1:])
    assert {span.parent.span_id for span in spans[1:]} == {root.context.span_id}


def test_span_ids_are_distinct() -> None:
    spans = build_spans(trace())

    assert len({span.context.span_id for span in spans}) == len(spans)


def test_span_ids_are_stable_across_exports() -> None:
    """A retried export must be one trace in the backend, not two overlapping copies."""
    assert [span.context.span_id for span in build_spans(trace())] == [
        span.context.span_id for span in build_spans(trace())
    ]


def test_a_span_id_is_never_zero() -> None:
    """Zero is not a legal span id, and a silently invalid span is worse than a nudged one."""
    assert span_id_for(TRACE_ID, "retrieval") != 0


def test_the_root_covers_the_whole_query() -> None:
    spans = build_spans(trace())
    root = spans[0]

    assert (root.end_time - root.start_time) / 1_000_000 == pytest.approx(1890.0)


def test_stage_durations_survive_the_mapping() -> None:
    generation = next(span for span in build_spans(trace()) if span.name == "generation")

    assert (generation.end_time - generation.start_time) / 1_000_000 == pytest.approx(1475.0)


def test_spans_are_placed_at_the_query_wall_clock() -> None:
    """Raw offsets would put every trace at the epoch, stacked on top of one another."""
    root = build_spans(trace())[0]

    assert root.start_time > 1_700_000_000 * 1_000_000_000


def test_an_unusable_timestamp_does_not_lose_the_trace() -> None:
    """A trace in the wrong place on a timeline is recoverable; a dropped one is not."""
    spans = build_spans(trace(created_at="not a timestamp"))

    assert len(spans) == 5
    assert spans[0].start_time > 0


def test_the_root_carries_what_the_query_did() -> None:
    attributes = dict(build_spans(trace())[0].attributes or {})

    assert attributes["fasterrag.query"] == "what is the UK meal allowance"
    assert attributes["fasterrag.collection"] == "policies"
    assert attributes["fasterrag.chunks_retrieved"] == 2
    assert attributes["fasterrag.prompt_tokens"] == 386


def test_stage_attributes_are_carried() -> None:
    reranker = next(span for span in build_spans(trace()) if span.name == "reranker")

    assert dict(reranker.attributes or {})["model"] == "bge-reranker-v2-m3"


def test_absent_attributes_are_omitted_rather_than_sent_as_null() -> None:
    """OTLP has no null; sending one drops the attribute with a warning nobody reads."""
    attributes = dict(build_spans(trace(tenant=None, result={}))[0].attributes or {})

    assert "fasterrag.tenant" not in attributes
    assert None not in attributes.values()


def test_an_unsupported_attribute_type_is_stringified() -> None:
    """The SDK silently drops a dict, so the attribute would simply vanish."""
    spans = build_spans(trace(spans=[Span("retrieval", 0.0, 1.0, {"filters": {"a": 1}})]))
    retrieval = next(span for span in spans if span.name == "retrieval")

    assert "a" in str(dict(retrieval.attributes or {})["filters"])


def test_the_service_is_named() -> None:
    """An unnamed service groups fasterRag's spans under 'unknown_service' in every viewer."""
    root = build_spans(trace())[0]

    assert dict(root.resource.attributes)["service.name"] == SERVICE_NAME


def test_a_trace_with_no_stages_still_exports() -> None:
    spans = build_spans(trace(spans=[]))

    assert len(spans) == 1


def test_it_encodes_as_real_otlp_protobuf() -> None:
    """The mapping being right in Python says nothing about it surviving the wire."""
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    encoded = encode_spans(build_spans(trace()))

    assert len(encoded.SerializeToString()) > 0
    assert len(encoded.resource_spans[0].scope_spans[0].spans) == 5


class Receiver(BaseHTTPRequestHandler):
    """Accepts one OTLP export, recording the body."""

    bodies: ClassVar[list[bytes]] = []

    def do_POST(self) -> None:
        """Accept the export."""
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        Receiver.bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Stay quiet."""


async def test_a_trace_reaches_a_listening_endpoint() -> None:
    """The wire path, not just the mapping: a real POST a collector would accept."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        exporter = OtelExporter(f"http://127.0.0.1:{server.server_address[1]}/v1/traces")
        accepted = await exporter.export(trace())
        await exporter.close()
    finally:
        server.shutdown()

    assert accepted is True
    request = ExportTraceServiceRequest()
    request.ParseFromString(Receiver.bodies[0])
    wire = request.resource_spans[0].scope_spans[0].spans
    assert wire[0].trace_id.hex() == TRACE_ID


async def test_an_unreachable_collector_costs_the_record_and_nothing_else() -> None:
    """The query has already been answered; a dead collector must not surface anywhere."""
    exporter = OtelExporter("http://127.0.0.1:1/v1/traces", timeout=1)

    assert await exporter.export(trace()) is False

    await exporter.close()


def test_the_toggle_off_builds_no_exporter() -> None:
    assert create_otel_exporter(Settings.model_validate({})) is None


def test_the_toggle_on_builds_one() -> None:
    settings = Settings.model_validate(
        {"observability": {"otel": True, "otel_endpoint": "http://localhost:4318/v1/traces"}}
    )

    assert create_otel_exporter(settings) is not None


def test_both_exporters_can_run_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """Langfuse and OTLP answer different questions; a deployment may want both."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    settings = Settings.model_validate(
        {
            "observability": {
                "otel": True,
                "otel_endpoint": "http://localhost:4318/v1/traces",
                "langfuse": True,
            }
        }
    )

    assert len(create_exporters(settings)) == 2


async def test_the_store_ships_to_every_exporter(tmp_path: Path) -> None:
    class Recording:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def export(self, exported: Trace) -> bool:
            self.seen.append(exported.trace_id)
            return True

        async def close(self) -> None:
            return None

    first, second = Recording(), Recording()
    store = TraceStore(tmp_path, exporters=[first, second])

    store.store(trace())
    await store.drain()

    assert first.seen == [TRACE_ID]
    assert second.seen == [TRACE_ID]
