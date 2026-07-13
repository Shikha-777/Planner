#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


WEIGHTED_KEYS = [
    "tool_set_accuracy",
    "tool_multiset_accuracy",
    "count_accuracy",
    "ordered_sequence_accuracy",
    "unordered_sequence_accuracy",
    "top1_tool_hit_rate_when_called",
]

SUM_KEYS = ["failure_count", "no_call_accuracy_count"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_metric(rows: list[dict[str, Any]], key: str) -> float:
    total = sum(int(row.get("total") or 0) for row in rows)
    if not total:
        return 0.0
    return sum(float(row.get(key) or 0.0) * int(row.get("total") or 0) for row in rows) / total


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("total") or 0) for row in rows)
    aggregate: dict[str, Any] = {"total": total}
    for key in WEIGHTED_KEYS:
        aggregate[key] = weighted_metric(rows, key)
    for key in SUM_KEYS:
        aggregate[key] = sum(int(row.get(key) or 0) for row in rows)
    return aggregate


def merge_review_csv(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output.open("w", encoding="utf-8", newline="") as out_handle:
        writer = None
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                if writer is None:
                    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames or [])
                    writer.writeheader()
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)
        if not wrote_header:
            out_handle.write("")


def merge_jsonl(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out_handle:
        for path in paths:
            with path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if line.strip():
                        out_handle.write(line if line.endswith("\n") else f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", required=True)
    parser.add_argument("--review-glob", required=True)
    parser.add_argument("--plans-glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--plans-jsonl")
    args = parser.parse_args()

    metric_paths = sorted(Path(path) for path in glob.glob(args.metrics_glob))
    if not metric_paths:
        raise SystemExit(f"no metric files matched {args.metrics_glob}")

    payloads = [read_json(path) for path in metric_paths]
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("by_category") or []:
            category_rows[str(row.get("category") or "unknown")].append(row)

    by_category = []
    for category in sorted(category_rows):
        row = aggregate_rows(category_rows[category])
        row["category"] = category
        by_category.append(row)

    aggregate = aggregate_rows(by_category)
    merged = {
        "aggregate": aggregate,
        "by_category": by_category,
        "shard_count": len(metric_paths),
        "shards": [str(path) for path in metric_paths],
        "planner_mode": payloads[0].get("planner_mode"),
        "ensemble_variants": payloads[0].get("ensemble_variants"),
        "model": payloads[0].get("model"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    review_paths = sorted(Path(path) for path in glob.glob(args.review_glob))
    merge_review_csv(review_paths, Path(args.review_csv))

    if args.plans_glob and args.plans_jsonl:
        plan_paths = sorted(Path(path) for path in glob.glob(args.plans_glob))
        merge_jsonl(plan_paths, Path(args.plans_jsonl))

    print(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
