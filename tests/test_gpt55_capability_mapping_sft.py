from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_gpt55_capability_mapping_sft.py"
SPEC = importlib.util.spec_from_file_location("generate_gpt55_capability_mapping_sft", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generate_gpt55_capability_mapping_sft_dataset(tmp_path: Path):
    summary = module.generate_dataset(
        count=160,
        out_dir=tmp_path,
        validation_fraction=0.2,
        seed=55,
    )

    assert summary["total_records"] == 160
    assert summary["train_records"] == 128
    assert summary["validation_records"] == 32
    assert summary["family_count"] >= 30
    assert summary["by_tool_decision"]["call"] > 0
    assert summary["by_tool_decision"]["ask_user"] > 0
    assert "web_search" in summary["by_external_action_type"]
    assert "external_tool_call" in summary["by_external_action_type"]
    assert "image_generation" in summary["by_external_action_type"]

    train_rows = read_jsonl(tmp_path / "train.sft.jsonl")
    validation_rows = read_jsonl(tmp_path / "validation.sft.jsonl")
    all_rows = read_jsonl(tmp_path / "all.records.jsonl")

    assert len(train_rows) == 128
    assert len(validation_rows) == 32
    assert len(all_rows) == 160

    sample = train_rows[0]
    assert [message["role"] for message in sample["messages"]] == ["system", "user", "assistant"]
    assistant_target = json.loads(sample["messages"][-1]["content"])
    assert assistant_target == sample["target"]
    assert {"final_user_intent", "input_audit", "route", "capability_plan", "evidence"} <= set(sample["target"])

    for row in all_rows:
        caps = row["target"]["capability_plan"]["ordered_capabilities"]
        cap_ids = {cap["id"] for cap in caps}
        for cap in caps:
            assert cap["capability_name"]
            assert cap["outputs"]
            assert cap["done_when"]
            assert set(cap["depends_on"]) <= cap_ids

    tool_rows = [row for row in all_rows if row["available_tools"]]
    assert tool_rows
    for row in tool_rows:
        assert "tool_binding" in row["target"]
        selected = row["target"]["tool_binding"]["ordered_tool_names"]
        assert row["target"]["tool_binding"]["call_count"] == len(selected)

    by_family = {row["family"]: row for row in all_rows}
    hard_families = {
        "tool_false_no_call_hard",
        "tool_parameter_not_missing",
        "tool_semantic_near_miss_hard",
        "tool_overcall_related_hard",
        "tool_keyword_no_call_hard",
    }
    assert hard_families <= set(by_family)

    for family in {
        "tool_false_no_call_hard",
        "tool_parameter_not_missing",
        "tool_semantic_near_miss_hard",
        "tool_overcall_related_hard",
    }:
        rows = [row for row in all_rows if row["family"] == family]
        assert rows
        for row in rows:
            assert row["target"]["route"]["tool_decision"] == "call"
            assert row["target"]["input_audit"]["missing_inputs"] == []
            assert row["target"]["tool_binding"]["call_count"] >= 1

    for row in [row for row in all_rows if row["family"] == "tool_parameter_not_missing"]:
        assert "provided_parameter_evidence" in row["target"]["evidence"]

    for row in [row for row in all_rows if row["family"] == "tool_semantic_near_miss_hard"]:
        assert row["target"]["tool_binding"]["call_count"] == 1

    for row in [row for row in all_rows if row["family"] == "tool_overcall_related_hard"]:
        assert row["target"]["tool_binding"]["call_count"] == 1

    for row in [row for row in all_rows if row["family"] == "tool_keyword_no_call_hard"]:
        assert row["target"]["route"]["tool_decision"] == "no_tool"
        assert row["target"]["tool_binding"]["call_count"] == 0

    for row in [row for row in all_rows if row["family"] == "tool_parallel_multiple"]:
        assert row["target"]["tool_binding"]["call_count"] == 4
