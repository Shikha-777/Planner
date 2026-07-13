from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_capability_benchmark_suite.py"
SPEC = importlib.util.spec_from_file_location("run_capability_benchmark_suite", SCRIPT)
assert SPEC and SPEC.loader
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)

PREPARE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_capability_benchmark_slices.py"
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_capability_benchmark_slices", PREPARE_SCRIPT
)
assert PREPARE_SPEC and PREPARE_SPEC.loader
prepare = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(prepare)

TOOL_PREPARE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_tool_action_benchmark_slices.py"
)
TOOL_PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_tool_action_benchmark_slices", TOOL_PREPARE_SCRIPT
)
assert TOOL_PREPARE_SPEC and TOOL_PREPARE_SPEC.loader
tool_prepare = importlib.util.module_from_spec(TOOL_PREPARE_SPEC)
TOOL_PREPARE_SPEC.loader.exec_module(tool_prepare)


def test_run_suite_writes_per_benchmark_and_aggregate_outputs(tmp_path):
    config = {
        "planner_mode": "rules_first",
        "gold_root": str(tmp_path / "gold"),
        "output_root": str(tmp_path / "results"),
        "benchmarks": [
            {
                "name": "adapter_smoke",
                "input": "data/benchmarks/capability_benchmark_adapter_smoke.jsonl",
                "dataset": "auto",
            }
        ],
    }

    summary = suite.run_suite(config)

    assert summary["benchmark_count"] == 1
    assert summary["case_count"] == 10
    assert summary["aggregate_metrics"]["capability_plan_accept_rate"] == 1.0

    benchmark = summary["benchmarks"][0]
    assert benchmark["adapted_cases"] == 10
    assert benchmark["metrics"]["input_audit_accuracy"] == 1.0

    for path in [
        summary["outputs"]["aggregate_metrics"],
        summary["outputs"]["aggregate_review_csv"],
        summary["outputs"]["suite_summary"],
        benchmark["outputs"]["gold"],
        benchmark["outputs"]["predictions"],
        benchmark["outputs"]["metrics"],
        benchmark["outputs"]["review_csv"],
    ]:
        assert Path(path).exists()


def test_prepare_configs_can_use_labelled_suite_paths(tmp_path):
    input_path = tmp_path / "source" / "clinc.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"id":"x","utterance":"weather tomorrow","intent":"weather"}\n')

    config_paths = prepare.build_configs(
        tmp_path / "slices",
        tmp_path / "configs",
        {"clinc_intent": input_path},
        suite_label="Scaled Public",
    )

    all_config = tmp_path / "configs" / "capability_benchmark_suite_all_scaled_public.json"
    assert all_config in config_paths
    text = all_config.read_text(encoding="utf-8")
    assert "benchmark_suite_all_scaled_public" in text
    assert "results/capability_benchmark_benchmark_suite_all_scaled_public" in text


def test_limited_by_label_honors_offset_per_label():
    rows = [
        {"label": "a", "value": "a1"},
        {"label": "b", "value": "b1"},
        {"label": "a", "value": "a2"},
        {"label": "b", "value": "b2"},
        {"label": "a", "value": "a3"},
        {"label": "b", "value": "b3"},
    ]

    selected = prepare.limited_by_label(
        rows,
        "label",
        ["a", "b"],
        limit_per_label=1,
        offset_per_label=1,
    )

    assert [row["value"] for row in selected] == ["a2", "b2"]


def test_local_controls_can_use_holdout_variant():
    rows, metadata = prepare.local_missing_controls("holdout_v2")

    assert metadata["variant"] == "holdout_v2"
    assert [row["id"] for row in rows] == [
        "local_missing_bio_002",
        "local_missing_revenue_plot_002",
        "local_draft_note_002",
    ]


def test_tool_action_expected_shapes_are_explicit():
    tool_exp = tool_prepare.tool_action_expected()
    control_exp = tool_prepare.no_tool_control_expected()

    assert tool_exp["run2"]["must_include_external_action_type"] == ["other"]
    assert tool_exp["run3"]["must_include_capability"] == [
        "select_and_execute_external_action"
    ]
    assert control_exp["run2"]["must_not_include_external_action_type"] == ["other"]
    assert control_exp["run3"]["must_not_include_capability"] == [
        "select_and_execute_external_action"
    ]
