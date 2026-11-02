"""Unit tests for the rerank measurement script's pure logic.

Model inference is deliberately out of scope: these cover argument handling, the on-disk
accounting, the paired statistics, and the tables, because those are the parts that can be
silently wrong and still print a plausible number.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_rerank import (
    DEFAULT_DATASETS,
    EMBEDDING_MODEL,
    METRICS,
    OFF,
    Configuration,
    Footprint,
    Latency,
    Measurement,
    build_settings,
    comparison_table,
    configurations,
    delta_table,
    footprint,
    human_bytes,
    loadable_files,
    paired_delta,
    parse_args,
    per_query,
)

from fasterrag.core.evals import EvalReport, QueryScore


def score(
    identifier: str,
    *,
    recall: float,
    rank: float,
    ndcg: float,
    adversarial: bool = False,
) -> QueryScore:
    return QueryScore(
        id=identifier,
        retrieved=(),
        hits=(),
        total_relevant=1,
        recall=recall,
        reciprocal_rank=rank,
        ndcg=ndcg,
        adversarial=adversarial,
    )


def report(*scores: QueryScore, k: int = 5) -> EvalReport:
    measurable = [item for item in scores if not item.adversarial]
    count = len(measurable) or 1
    return EvalReport(
        k=k,
        scored=len(measurable),
        adversarial=len(scores) - len(measurable),
        recall_at_k=sum(item.recall for item in measurable) / count,
        mrr=sum(item.reciprocal_rank for item in measurable) / count,
        ndcg_at_k=sum(item.ndcg for item in measurable) / count,
        per_query=list(scores),
    )


def measurement(
    dataset: str,
    name: str,
    *scores: QueryScore,
    disk: Footprint | None = None,
    latency: Latency | None = None,
) -> Measurement:
    return Measurement(
        dataset=dataset,
        configuration=Configuration(
            name=name, rerank=name != OFF, model="" if name == OFF else name
        ),
        report=report(*scores),
        footprint=disk,
        rerank_latency=latency,
    )


class TestConfigurations:
    def test_reranking_off_is_always_the_first_configuration(self) -> None:
        """The baseline must come first: every delta is measured against it."""
        built = configurations(["a/model", "b/model"])

        assert [item.name for item in built] == [OFF, "a/model", "b/model"]
        assert built[0].rerank is False
        assert built[0].model == ""

    def test_each_rerank_configuration_carries_its_model(self) -> None:
        built = configurations(["a/model"])

        assert built[1].rerank is True
        assert built[1].model == "a/model"

    def test_a_repeated_model_is_measured_once(self) -> None:
        assert [item.name for item in configurations(["a", "a", "b"])] == [OFF, "a", "b"]

    def test_an_empty_model_name_is_dropped(self) -> None:
        assert [item.name for item in configurations(["", "a"])] == [OFF, "a"]

    def test_no_models_leaves_only_the_baseline(self) -> None:
        assert [item.name for item in configurations([])] == [OFF]


class TestHumanBytes:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1024, "1.00 KB"),
            (90_870_598, "86.66 MB"),
            (2_271_071_852, "2.12 GB"),
        ],
    )
    def test_the_largest_unit_that_keeps_the_number_above_one_is_used(
        self, count: int, expected: str
    ) -> None:
        assert human_bytes(count) == expected


class TestLoadableFiles:
    def test_alternate_serializations_are_not_charged_to_the_model(self) -> None:
        """ms-marco-MiniLM ships five ONNX builds of the same 90 MB of weights."""
        kept = loadable_files(
            ["config.json", "model.safetensors", "onnx/model.onnx", "onnx/model_O1.onnx"]
        )

        assert kept == ["config.json", "model.safetensors"]

    def test_readme_screenshots_are_not_charged_to_the_model(self) -> None:
        kept = loadable_files(["config.json", "README.md", "assets/chart.png"])

        assert kept == ["config.json"]

    def test_a_duplicate_weight_format_is_dropped_when_safetensors_is_present(self) -> None:
        kept = loadable_files(["model.safetensors", "pytorch_model.bin", "flax_model.msgpack"])

        assert kept == ["model.safetensors"]

    def test_the_only_weight_file_is_kept_whatever_its_format(self) -> None:
        """Dropping it would report a model with no weights as weighing nothing."""
        assert loadable_files(["config.json", "pytorch_model.bin"]) == [
            "config.json",
            "pytorch_model.bin",
        ]

    def test_a_dotfile_is_excluded_by_name_because_it_has_no_suffix(self) -> None:
        """`Path('.gitattributes').suffix` is empty, so a suffix rule never catches it."""
        assert loadable_files(["config.json", ".gitattributes"]) == ["config.json"]

    def test_the_result_is_sorted(self) -> None:
        assert loadable_files(["z.json", "a.json"]) == ["a.json", "z.json"]


class TestFootprint:
    def test_only_the_loaded_files_are_counted_and_the_weights_are_reported_apart(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "onnx").mkdir()
        (tmp_path / "model.safetensors").write_bytes(b"w" * 400)
        (tmp_path / "adapter.safetensors").write_bytes(b"a" * 100)
        (tmp_path / "config.json").write_bytes(b"c" * 50)
        (tmp_path / "README.md").write_bytes(b"r" * 9999)
        (tmp_path / "onnx" / "model.onnx").write_bytes(b"o" * 8888)

        measured = footprint("some/model", tmp_path)

        assert measured.model == "some/model"
        assert measured.counted_bytes == 550
        assert measured.weight_bytes == 400
        assert measured.files == ("adapter.safetensors", "config.json", "model.safetensors")

    def test_a_model_with_no_weight_file_reports_zero_weight_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_bytes(b"c" * 50)

        assert footprint("some/model", tmp_path).weight_bytes == 0


class TestPerQuery:
    def test_adversarial_records_are_excluded(self) -> None:
        """They have no ground truth, so a delta over them would be a delta over nothing."""
        scored = report(
            score("q1", recall=1.0, rank=1.0, ndcg=1.0),
            score("q2", recall=0.0, rank=0.0, ndcg=0.0, adversarial=True),
        )

        assert per_query(scored, "recall") == {"q1": 1.0}

    def test_each_harness_metric_is_readable(self) -> None:
        scored = report(score("q1", recall=0.5, rank=0.25, ndcg=0.125))

        assert per_query(scored, "recall") == {"q1": 0.5}
        assert per_query(scored, "reciprocal_rank") == {"q1": 0.25}
        assert per_query(scored, "ndcg") == {"q1": 0.125}

    def test_an_unknown_metric_is_refused_rather_than_invented(self) -> None:
        with pytest.raises(ValueError, match="not one of the harness metrics"):
            per_query(report(score("q1", recall=1.0, rank=1.0, ndcg=1.0)), "precision")

    def test_the_known_metrics_are_exactly_the_harness_fields(self) -> None:
        scored = score("q1", recall=1.0, rank=1.0, ndcg=1.0)

        assert all(hasattr(scored, metric) for metric in METRICS)


class TestPairedDelta:
    def test_the_mean_is_the_mean_of_the_per_query_differences(self) -> None:
        delta = paired_delta("ndcg", {"a": 0.0, "b": 0.5}, {"a": 0.4, "b": 0.5})

        assert delta.mean == pytest.approx(0.2)
        assert delta.n == 2

    def test_the_standard_error_is_the_deviation_divided_by_the_root_of_the_count(self) -> None:
        """It is the error of the *mean*; reporting the raw deviation overstates the noise."""
        delta = paired_delta("ndcg", {"a": 0.0, "b": 0.5}, {"a": 0.4, "b": 0.5})

        assert delta.standard_error == pytest.approx(0.2)

    def test_the_standard_error_shrinks_as_queries_agree(self) -> None:
        agreeing = paired_delta("ndcg", {"a": 0.0, "b": 0.0}, {"a": 0.2, "b": 0.2})
        disagreeing = paired_delta("ndcg", {"a": 0.0, "b": 0.0}, {"a": 0.6, "b": -0.2})

        assert agreeing.standard_error == pytest.approx(0.0)
        assert disagreeing.standard_error > agreeing.standard_error

    def test_only_queries_present_in_both_runs_are_paired(self) -> None:
        delta = paired_delta("recall", {"a": 0.0, "b": 0.0}, {"a": 1.0, "c": 1.0})

        assert delta.n == 1

    def test_no_shared_queries_yields_an_empty_delta_rather_than_a_crash(self) -> None:
        delta = paired_delta("recall", {"a": 0.0}, {"b": 1.0})

        assert delta.n == 0
        assert delta.mean == 0.0
        assert delta.resolvable is False

    def test_a_single_pair_is_never_called_larger_than_noise(self) -> None:
        """One query cannot estimate its own variance, so it cannot resolve anything."""
        delta = paired_delta("mrr", {"a": 0.0}, {"a": 1.0})

        assert delta.n == 1
        assert delta.resolvable is False

    def test_a_move_inside_two_standard_errors_is_not_resolvable(self) -> None:
        delta = paired_delta(
            "mrr", {"a": 0.0, "b": 0.0, "c": 0.0}, {"a": 0.5, "b": -0.4, "c": 0.05}
        )

        assert delta.mean > 0
        assert delta.resolvable is False

    def test_a_move_between_one_and_two_standard_errors_is_still_not_resolvable(self) -> None:
        """The band the measured rerank gain actually lands in: suggestive, not resolved."""
        delta = paired_delta("mrr", {"a": 0.0, "b": 0.0, "c": 0.0}, {"a": 0.3, "b": 0.2, "c": 0.0})

        assert delta.standard_error < abs(delta.mean) < 2 * delta.standard_error
        assert delta.resolvable is False

    def test_a_move_outside_two_standard_errors_is_resolvable(self) -> None:
        delta = paired_delta(
            "mrr", {"a": 0.0, "b": 0.0, "c": 0.0}, {"a": 0.50, "b": 0.52, "c": 0.48}
        )

        assert delta.resolvable is True

    def test_a_consistent_decline_is_resolvable_too(self) -> None:
        """A reranker that reliably makes things worse is as much a finding as one that helps."""
        delta = paired_delta(
            "mrr", {"a": 0.5, "b": 0.5, "c": 0.5}, {"a": 0.0, "b": 0.02, "c": 0.01}
        )

        assert delta.mean < 0
        assert delta.resolvable is True

    def test_the_description_carries_the_sign_and_the_error(self) -> None:
        delta = paired_delta("ndcg", {"a": 0.0, "b": 0.0}, {"a": 0.2, "b": 0.2})

        assert delta.describe() == "+0.2000 +/- 0.0000"


class TestLatency:
    def test_percentiles_and_mean_come_from_the_samples(self) -> None:
        latency = Latency(samples=(10.0, 20.0, 30.0, 40.0), candidates=7)

        assert latency.p50 == 20.0
        assert latency.p95 == 40.0
        assert latency.mean == pytest.approx(25.0)
        assert latency.as_dict()["candidates"] == 7

    def test_no_samples_reports_zero_rather_than_raising(self) -> None:
        assert Latency().mean == 0.0
        assert Latency().p50 == 0.0


class TestComparisonTable:
    def test_a_configuration_without_a_reranker_reports_no_model_cost(self) -> None:
        table = comparison_table(
            [measurement("ds", OFF, score("q1", recall=1.0, rank=1.0, ndcg=1.0))]
        )
        row = table.splitlines()[-1]

        assert row.count("n/a") == 4

    def test_quality_size_load_and_latency_all_reach_the_row(self) -> None:
        table = comparison_table(
            [
                measurement(
                    "ds",
                    "small/model",
                    score("q1", recall=1.0, rank=0.5, ndcg=0.75),
                    disk=Footprint(model="small/model", counted_bytes=1024, load_seconds=7.25),
                    latency=Latency(samples=(100.0, 200.0, 300.0, 400.0)),
                )
            ]
        )
        row = table.splitlines()[-1]

        assert "1.0000" in row
        assert "0.5000" in row
        assert "0.7500" in row
        assert "1.00 KB" in row
        assert "7.2 s" in row
        assert "200.0 ms" in row
        assert "400.0 ms" in row

    def test_every_measurement_gets_exactly_one_row_under_one_header(self) -> None:
        table = comparison_table(
            [
                measurement("ds", OFF, score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
                measurement("ds", "small/model", score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
            ]
        )

        assert len(table.splitlines()) == 4


class TestDeltaTable:
    def test_every_metric_of_every_variant_is_compared_against_rerank_off(self) -> None:
        table = delta_table(
            [
                measurement("ds", OFF, score("q1", recall=1.0, rank=0.5, ndcg=0.5)),
                measurement("ds", "small/model", score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
            ]
        )
        rows = table.splitlines()[2:]

        assert len(rows) == len(METRICS)
        assert all("small/model" in row for row in rows)
        assert not any(f"| {OFF} |" in row for row in rows)

    def test_a_move_inside_the_noise_is_labelled_as_such(self) -> None:
        table = delta_table(
            [
                measurement(
                    "ds",
                    OFF,
                    score("q1", recall=1.0, rank=0.0, ndcg=0.0),
                    score("q2", recall=1.0, rank=1.0, ndcg=1.0),
                ),
                measurement(
                    "ds",
                    "small/model",
                    score("q1", recall=1.0, rank=1.0, ndcg=1.0),
                    score("q2", recall=1.0, rank=0.2, ndcg=0.2),
                ),
            ]
        )
        mrr = next(row for row in table.splitlines() if "| MRR |" in row)

        assert mrr.rstrip().endswith("| no |")

    def test_a_move_outside_the_noise_is_labelled_as_such(self) -> None:
        table = delta_table(
            [
                measurement(
                    "ds",
                    OFF,
                    score("q1", recall=1.0, rank=0.5, ndcg=0.5),
                    score("q2", recall=1.0, rank=0.5, ndcg=0.5),
                ),
                measurement(
                    "ds",
                    "small/model",
                    score("q1", recall=1.0, rank=1.0, ndcg=1.0),
                    score("q2", recall=1.0, rank=1.0, ndcg=1.0),
                ),
            ]
        )
        mrr = next(row for row in table.splitlines() if "| MRR |" in row)

        assert mrr.rstrip().endswith("| yes |")

    def test_datasets_are_compared_against_their_own_baseline(self) -> None:
        table = delta_table(
            [
                measurement("one", OFF, score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
                measurement("one", "small/model", score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
                measurement("two", OFF, score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
                measurement("two", "small/model", score("q1", recall=1.0, rank=1.0, ndcg=1.0)),
            ]
        )

        assert sum(row.startswith("| one |") for row in table.splitlines()) == len(METRICS)
        assert sum(row.startswith("| two |") for row in table.splitlines()) == len(METRICS)

    def test_a_dataset_with_no_baseline_produces_no_rows_rather_than_a_wrong_one(self) -> None:
        """Without rerank off there is nothing to subtract, and inventing a zero would lie."""
        table = delta_table(
            [measurement("ds", "small/model", score("q1", recall=1.0, rank=1.0, ndcg=1.0))]
        )

        assert len(table.splitlines()) == 2


class TestBuildSettings:
    def test_rerank_off_leaves_the_shipped_model_name_untouched(self) -> None:
        settings = build_settings(Configuration(name=OFF, rerank=False), EMBEDDING_MODEL)

        assert settings.retrieval.rerank is False
        assert settings.embeddings.model == EMBEDDING_MODEL

    def test_a_rerank_configuration_selects_its_own_cross_encoder(self) -> None:
        settings = build_settings(
            Configuration(name="small/model", rerank=True, model="small/model"), EMBEDDING_MODEL
        )

        assert settings.retrieval.rerank is True
        assert settings.retrieval.reranker_model == "small/model"

    def test_the_chunk_size_override_reaches_the_chunker_with_a_legal_overlap(self) -> None:
        settings = build_settings(Configuration(name=OFF, rerank=False), EMBEDDING_MODEL, 128)

        assert settings.chunking.chunk_size == 128
        assert settings.chunking.overlap < settings.chunking.chunk_size

    def test_the_shipped_chunk_size_is_used_when_nothing_overrides_it(self) -> None:
        default = build_settings(Configuration(name=OFF, rerank=False), EMBEDDING_MODEL)
        overridden = build_settings(Configuration(name=OFF, rerank=False), EMBEDDING_MODEL, 128)

        assert default.chunking.chunk_size != overridden.chunking.chunk_size


class TestParseArgs:
    def test_the_defaults_measure_both_committed_datasets_and_both_models(self) -> None:
        args = parse_args([])

        assert args.datasets is None
        assert args.models is None
        assert args.allow_download is False
        assert DEFAULT_DATASETS

    def test_datasets_and_models_are_repeatable(self) -> None:
        args = parse_args(["--dataset", "a", "--dataset", "b", "--model", "m"])

        assert args.datasets == ["a", "b"]
        assert args.models == ["m"]

    def test_weights_are_never_fetched_unless_asked_for(self) -> None:
        """The shipped default is 2.2 GB; a script that pulls it silently is the TASK-0174 bug."""
        assert parse_args([]).allow_download is False
        assert parse_args(["--allow-download"]).allow_download is True

    def test_the_probe_width_defaults_to_the_shipped_rerank_top_n(self) -> None:
        assert parse_args([]).probe_candidates == 100

    @pytest.mark.parametrize("flag", ["--repetitions", "--probe-candidates"])
    def test_a_count_below_one_is_refused(self, flag: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([flag, "0"])

    def test_the_chunk_size_override_is_absent_by_default(self) -> None:
        assert parse_args([]).chunk_size is None
        assert parse_args(["--chunk-size", "128"]).chunk_size == 128


class TestMeasurementSerialization:
    def test_the_json_form_carries_every_number_the_decision_turns_on(self) -> None:
        serialized = measurement(
            "ds",
            "small/model",
            score("q1", recall=1.0, rank=1.0, ndcg=1.0),
            disk=Footprint(model="small/model", counted_bytes=1024, weight_bytes=900),
            latency=Latency(samples=(5.0,), candidates=3),
        ).as_dict()

        assert serialized["quality"]["recall_at_k"] == 1.0
        assert serialized["footprint"]["counted_bytes"] == 1024
        assert serialized["rerank_latency"]["p50_ms"] == 5.0
        assert serialized["configuration"] == "small/model"

    def test_a_configuration_without_a_model_serializes_without_inventing_one(self) -> None:
        serialized = measurement("ds", OFF, score("q1", recall=1.0, rank=1.0, ndcg=1.0)).as_dict()

        assert serialized["footprint"] is None
        assert serialized["rerank_latency"] is None
