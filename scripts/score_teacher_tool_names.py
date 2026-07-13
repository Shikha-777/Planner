#!/usr/bin/env python3
"""Score a teacher planner's ordered BFCL tool-name predictions.

The teacher file should contain JSONL rows with:
  id, predicted_tool_names

The gold/sample file is a local tool-binding fixture row with expected_tool_binding.
This scorer intentionally ignores arguments so it can compare planner routing only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from taskdecomp.tool_binding import build_tool_binding_plan


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_names(row: dict[str, Any]) -> list[str]:
    calls = (row.get("expected_tool_binding") or {}).get("calls") or []
    return [str(call.get("tool_name") or "") for call in calls if isinstance(call, dict) and call.get("tool_name")]


def load_gold_rows(sample_rows: list[dict[str, Any]], gold_dir: Path) -> dict[str, dict[str, Any]]:
    if all(isinstance(row.get("expected_tool_binding"), dict) for row in sample_rows):
        return {str(row.get("id")): row for row in sample_rows}
    gold_by_id: dict[str, dict[str, Any]] = {}
    wanted_ids = {str(row.get("id")) for row in sample_rows}
    for path in sorted(gold_dir.glob("bfcl_*.jsonl")):
        if path.name == "gpt55_teacher_sample.jsonl":
            continue
        for row in read_jsonl(path):
            rid = str(row.get("id"))
            if rid in wanted_ids and isinstance(row.get("expected_tool_binding"), dict):
                gold_by_id[rid] = row
    return gold_by_id


def as_calls(names: list[str]) -> list[dict[str, Any]]:
    return [{"tool_name": name, "arguments": {}} for name in names if name]


def rules_names(row: dict[str, Any]) -> list[str]:
    plan = build_tool_binding_plan(str(row.get("request") or ""), row.get("tools") or [])
    return [
        str(call.get("tool_name") or "")
        for call in plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]


def score_one(expected: list[str], predicted: list[str]) -> dict[str, bool]:
    return {
        "tool_set_ok": set(expected) == set(predicted),
        "tool_multiset_ok": Counter(expected) == Counter(predicted),
        "count_ok": len(expected) == len(predicted),
        "ordered_ok": expected == predicted,
    }


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {
            f"{prefix}_total": 0,
            f"{prefix}_tool_set_accuracy": 0,
            f"{prefix}_tool_multiset_accuracy": 0,
            f"{prefix}_count_accuracy": 0,
            f"{prefix}_ordered_sequence_accuracy": 0,
        }
    return {
        f"{prefix}_total": total,
        f"{prefix}_tool_set_accuracy": sum(row["tool_set_ok"] for row in rows) / total,
        f"{prefix}_tool_multiset_accuracy": sum(row["tool_multiset_ok"] for row in rows) / total,
        f"{prefix}_count_accuracy": sum(row["count_ok"] for row in rows) / total,
        f"{prefix}_ordered_sequence_accuracy": sum(row["ordered_ok"] for row in rows) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=Path("data/benchmarks/tool_binding"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    args = parser.parse_args()

    sample_rows = read_jsonl(args.sample)
    teacher_by_id = {str(row.get("id")): row for row in read_jsonl(args.teacher_predictions)}
    gold_by_id = load_gold_rows(sample_rows, args.gold_dir)

    review_rows = []
    teacher_cases = []
    rules_cases = []
    by_category: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in sample_rows:
        rid = str(row.get("id"))
        category = str(row.get("category") or "")
        expected = expected_names(gold_by_id.get(rid, row))
        teacher_row = teacher_by_id.get(rid, {})
        teacher_pred = [str(name) for name in (teacher_row.get("predicted_tool_names") or [])]
        rules_pred = rules_names(row)
        teacher_score = score_one(expected, teacher_pred)
        rules_score = score_one(expected, rules_pred)
        teacher_case = {"id": rid, "category": category, **teacher_score}
        rules_case = {"id": rid, "category": category, **rules_score}
        teacher_cases.append(teacher_case)
        rules_cases.append(rules_case)
        by_category.setdefault(category, {"teacher": [], "rules": []})
        by_category[category]["teacher"].append(teacher_case)
        by_category[category]["rules"].append(rules_case)
        if not teacher_score["ordered_ok"] or not rules_score["ordered_ok"]:
            review_rows.append(
                {
                    "id": rid,
                    "category": category,
                    "expected": json.dumps(expected, ensure_ascii=False),
                    "teacher_predicted": json.dumps(teacher_pred, ensure_ascii=False),
                    "rules_predicted": json.dumps(rules_pred, ensure_ascii=False),
                    "teacher_ordered_ok": teacher_score["ordered_ok"],
                    "rules_ordered_ok": rules_score["ordered_ok"],
                    "teacher_rationale": teacher_row.get("rationale_short", ""),
                    "request": str(row.get("request") or "")[:1000],
                }
            )

    payload = {
        "sample": str(args.sample),
        "teacher_predictions": str(args.teacher_predictions),
        "gold_dir": str(args.gold_dir),
        "aggregate": {
            **summarize(teacher_cases, "teacher"),
            **summarize(rules_cases, "rules"),
            "missing_gold_count": sum(1 for row in sample_rows if str(row.get("id")) not in gold_by_id),
        },
        "by_category": {
            category: {
                **summarize(groups["teacher"], "teacher"),
                **summarize(groups["rules"], "rules"),
            }
            for category, groups in sorted(by_category.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "expected",
        "teacher_predicted",
        "rules_predicted",
        "teacher_ordered_ok",
        "rules_ordered_ok",
        "teacher_rationale",
        "request",
    ]
    with args.review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
