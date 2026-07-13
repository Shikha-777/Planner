#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from taskdecomp.tool_binding import build_tool_binding_plan
from taskdecomp.tool_binding_eval import (
    read_jsonl,
    score_predictions,
    write_json,
    write_jsonl,
    write_review_csv,
)


DEFAULT_OUTPUT_ROOT = Path("results/bfcl_tool_binding")


def load_gold(input_path: Path) -> list[dict[str, Any]]:
    if input_path.name == "manifest.json":
        manifest = json.loads(input_path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for file_path in (manifest.get("files") or {}).values():
            rows.extend(read_jsonl(Path(file_path)))
        return rows
    return read_jsonl(input_path)


def run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for index, row in enumerate(rows):
        request = str(row.get("request") or "")
        tools = row.get("tools") if isinstance(row.get("tools"), list) else []
        plan = build_tool_binding_plan(request, tools)
        predictions.append(
            {
                "id": row.get("id"),
                "row_index": index,
                "category": row.get("category"),
                "tool_binding_plan": plan,
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFCL-style tool-binding eval.")
    parser.add_argument("--input", type=Path, default=Path("data/benchmarks/tool_binding/manifest.json"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    rows = load_gold(args.input)
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

    summary = {
        "input": str(args.input),
        "case_count": len(rows),
        "metrics": scored["metrics"],
        "outputs": {
            "gold": str(gold_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "review_csv": str(review_path),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
