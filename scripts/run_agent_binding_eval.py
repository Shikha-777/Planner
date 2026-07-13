#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from taskdecomp.agent_binding import build_agent_binding_plan
from taskdecomp.agent_binding_eval import (
    read_jsonl,
    score_predictions,
    write_json,
    write_jsonl,
    write_review_csv,
)


DEFAULT_INPUT = Path("data/benchmarks/agent_binding/synthetic_100.jsonl")
DEFAULT_OUTPUT_ROOT = Path("results/agent_binding")


def run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for index, row in enumerate(rows):
        payload = {
            "request": row.get("request") or "",
            "capability_plan": row.get("gold_capability_plan") or row.get("capability_plan") or {},
            "agents": row.get("agents") if isinstance(row.get("agents"), list) else None,
        }
        plan = build_agent_binding_plan(payload)
        predictions.append(
            {
                "id": row.get("id"),
                "row_index": index,
                "category": row.get("category") or (row.get("metadata") or {}).get("category"),
                "agent_binding_plan": plan,
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic agent-binding eval.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    name = args.name or args.input.stem
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = run_rows(rows)
    scored = score_predictions(rows, predictions)

    gold_path = output_root / f"{name}.gold.jsonl"
    predictions_path = output_root / f"{name}.predictions.jsonl"
    metrics_path = output_root / f"{name}.metrics.json"
    review_path = output_root / f"{name}.review.csv"
    write_jsonl(gold_path, rows)
    write_jsonl(predictions_path, predictions)
    write_json(metrics_path, scored)
    write_review_csv(review_path, scored["cases"])

    print(
        json.dumps(
            {
                "input": str(args.input),
                "case_count": len(rows),
                "metrics": scored["metrics"],
                "outputs": {
                    "gold": str(gold_path),
                    "predictions": str(predictions_path),
                    "metrics": str(metrics_path),
                    "review_csv": str(review_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
