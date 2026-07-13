#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


OWNER_BUCKETS = [
    "adapter_mislabeled",
    "planner_routing_or_capability",
    "scorer_too_strict_or_loose",
    "needs_manual_review",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def suggested_owner(row: dict[str, str]) -> str:
    notes = row.get("notes", "").lower()
    failure = row.get("main_failure_type", "")
    if not failure:
        return ""
    if "unexpectedly included" in notes and "expected" in notes:
        return "needs_manual_review"
    if failure in {"invalid_json", "bad_dependency", "used_tool_name", "used_agent_name", "vague_capability"}:
        return "planner_routing_or_capability"
    if failure in {"wrong_final_intent", "missed_missing_input", "input_audit_error", "missed_current_info", "wrong_external_action", "under_decomposed", "over_decomposed"}:
        return "needs_manual_review"
    return "needs_manual_review"


def build_review(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures = [row for row in rows if row.get("main_failure_type")]
    review_rows = []
    failure_counts = Counter()
    owner_counts = Counter()
    for row in failures:
        owner = suggested_owner(row)
        failure_counts[row.get("main_failure_type", "")] += 1
        owner_counts[owner] += 1
        review_rows.append(
            {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "main_failure_type": row.get("main_failure_type", ""),
                "suggested_owner": owner,
                "final_owner": "",
                "request": row.get("request", ""),
                "notes": row.get("notes", ""),
            }
        )
    summary = {
        "case_count": len(rows),
        "failure_count": len(failures),
        "failure_counts": dict(sorted(failure_counts.items())),
        "suggested_owner_counts": dict(sorted(owner_counts.items())),
        "owner_buckets": OWNER_BUCKETS,
    }
    return review_rows, summary


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "main_failure_type",
        "suggested_owner",
        "final_owner",
        "request",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a manual-review scaffold for benchmark planner failures. "
            "The final_owner column should be filled with adapter_mislabeled, "
            "planner_routing_or_capability, or scorer_too_strict_or_loose."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.review_csv)
    review_rows, summary = build_review(rows)
    write_review_csv(args.out_csv, review_rows)
    write_json(args.out_json, {"summary": summary, "failures": review_rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
