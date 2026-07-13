#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SAMPLE = Path("data/benchmarks/tool_binding/gpt55_teacher_sample.jsonl")
DEFAULT_PREDICTIONS = Path("results/gpt55_teacher_bfcl_sample_predictions.jsonl")
DEFAULT_METRICS = Path("results/gpt55_teacher_bfcl_sample.metrics.json")
DEFAULT_REVIEW = Path("results/gpt55_teacher_bfcl_sample.review.csv")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tool_names(row: dict[str, Any]) -> list[str]:
    binding = row.get("expected_tool_binding") if isinstance(row.get("expected_tool_binding"), dict) else {}
    calls = binding.get("calls") if isinstance(binding, dict) else []
    return [
        str(call.get("tool_name"))
        for call in calls
        if isinstance(call, dict) and call.get("tool_name")
    ]


def predicted_tool_names(row: dict[str, Any]) -> list[str]:
    names = row.get("predicted_tool_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names if str(name)]


def load_gold_by_id(gold_dir: Path) -> dict[str, dict[str, Any]]:
    gold_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(gold_dir.glob("bfcl_*.jsonl")):
        if path.name.startswith("bfcl_exec_"):
            continue
        for row in read_jsonl(path):
            row_id = row.get("id")
            if row_id and isinstance(row.get("expected_tool_binding"), dict):
                gold_by_id[str(row_id)] = row
    return gold_by_id


def score_case(sample_row: dict[str, Any], prediction: dict[str, Any], gold: dict[str, Any] | None) -> dict[str, Any]:
    expected = expected_tool_names(gold or {})
    predicted = predicted_tool_names(prediction)
    expected_counter = Counter(expected)
    predicted_counter = Counter(predicted)
    expected_set = set(expected)
    predicted_set = set(predicted)
    return {
        "id": sample_row.get("id"),
        "category": sample_row.get("category"),
        "expected": expected,
        "predicted": predicted,
        "missing_gold": gold is None,
        "missing_prediction": not prediction,
        "ordered_ok": predicted == expected,
        "multiset_ok": predicted_counter == expected_counter,
        "set_ok": predicted_set == expected_set,
        "count_ok": len(predicted) == len(expected),
        "top1_ok": bool(expected) and bool(predicted) and predicted[0] == expected[0],
        "expected_call_count": len(expected),
        "predicted_call_count": len(predicted),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    if not total:
        return {
            "case_count": 0,
            "ordered_sequence_accuracy": 0.0,
            "unordered_multiset_accuracy": 0.0,
            "tool_set_accuracy": 0.0,
            "call_count_accuracy": 0.0,
            "top1_accuracy_when_expected_call": 0.0,
            "missing_gold_count": 0,
            "missing_prediction_count": 0,
        }

    expected_call_cases = [case for case in cases if case["expected_call_count"] > 0]
    return {
        "case_count": total,
        "ordered_sequence_accuracy": round(sum(case["ordered_ok"] for case in cases) / total, 4),
        "unordered_multiset_accuracy": round(sum(case["multiset_ok"] for case in cases) / total, 4),
        "tool_set_accuracy": round(sum(case["set_ok"] for case in cases) / total, 4),
        "call_count_accuracy": round(sum(case["count_ok"] for case in cases) / total, 4),
        "top1_accuracy_when_expected_call": round(
            sum(case["top1_ok"] for case in expected_call_cases) / len(expected_call_cases),
            4,
        )
        if expected_call_cases
        else 0.0,
        "missing_gold_count": sum(case["missing_gold"] for case in cases),
        "missing_prediction_count": sum(case["missing_prediction"] for case in cases),
    }


def write_review_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "expected",
        "predicted",
        "ordered_ok",
        "multiset_ok",
        "set_ok",
        "count_ok",
        "missing_gold",
        "missing_prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["expected"] = json.dumps(row["expected"], ensure_ascii=False)
            row["predicted"] = json.dumps(row["predicted"], ensure_ascii=False)
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Score GPT-5.5 teacher BFCL tool-binding sample predictions.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold-dir", type=Path, default=Path("data/benchmarks/tool_binding"))
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()

    sample_rows = read_jsonl(args.sample)
    prediction_rows = {str(row.get("id")): row for row in read_jsonl(args.predictions)}
    gold_rows = load_gold_by_id(args.gold_dir)

    cases = [
        score_case(row, prediction_rows.get(str(row.get("id")), {}), gold_rows.get(str(row.get("id"))))
        for row in sample_rows
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("category"))].append(case)

    metrics = {
        "sample": str(args.sample),
        "predictions": str(args.predictions),
        "gold_dir": str(args.gold_dir),
        "prediction_sha256": file_sha256(args.predictions),
        "note": "Gold expected_tool_binding rows are loaded only by this scorer, after predictions are written.",
        "aggregate": summarize(cases),
        "by_category": {category: summarize(rows) for category, rows in sorted(grouped.items())},
    }
    write_json(args.output, metrics)
    write_review_csv(args.review_csv, cases)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
