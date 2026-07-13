#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapt_capability_benchmark import adapt_records, canonical_dataset, read_records, slug
from run_gptoss_capability_plan import compact_json, row_to_request, run_capability_plan
from taskdecomp.capability_eval import score_predictions, write_csv, write_json


DEFAULT_OUTPUT_ROOT = Path("results/capability_benchmark_suite")
DEFAULT_GOLD_ROOT = Path("data/capability_planning/benchmark_suite")


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def suite_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        config = read_config(args.config)
    else:
        if not args.input:
            raise SystemExit("--input is required when --config is not provided")
        config = {
            "benchmarks": [
                {
                    "name": args.name or Path(args.input).stem,
                    "input": str(args.input),
                    "dataset": args.dataset,
                    "limit": args.limit,
                    "source_wrapper": args.source_wrapper,
                }
            ]
        }

    config.setdefault("planner_mode", args.planner_mode)
    config.setdefault("model", args.model)
    config.setdefault("max_new_tokens", args.max_new_tokens)
    config.setdefault("output_root", str(args.output_root))
    config.setdefault("gold_root", str(args.gold_root))
    return config


def run_suite(config: dict[str, Any]) -> dict[str, Any]:
    planner_mode = str(config.get("planner_mode") or "rules_first")
    if planner_mode != "rules_first":
        raise ValueError("run_capability_benchmark_suite currently supports planner_mode='rules_first'")

    model = str(config.get("model") or "openai/gpt-oss-20b")
    max_new_tokens = int(config.get("max_new_tokens") or 1200)
    output_root = Path(config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    gold_root = Path(config.get("gold_root") or DEFAULT_GOLD_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    gold_root.mkdir(parents=True, exist_ok=True)

    benchmarks = config.get("benchmarks") or []
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("Suite config must include a non-empty benchmarks list")

    benchmark_results = []
    combined_gold: list[dict[str, Any]] = []
    combined_predictions: list[dict[str, Any]] = []

    for index, spec in enumerate(benchmarks, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"Benchmark spec {index} must be an object")
        result = run_benchmark(
            spec,
            output_root=output_root,
            gold_root=gold_root,
            planner_mode=planner_mode,
            model=model,
            max_new_tokens=max_new_tokens,
        )
        benchmark_results.append(result["summary"])
        combined_gold.extend(result["gold_rows"])
        combined_predictions.extend(result["prediction_rows"])

    combined_result = score_predictions(combined_gold, combined_predictions)
    combined_metrics_path = output_root / "aggregate.metrics.json"
    combined_review_path = output_root / "aggregate.review.csv"
    suite_summary_path = output_root / "suite_summary.json"
    write_json(combined_metrics_path, combined_result)
    write_csv(combined_review_path, combined_result["cases"])

    summary = {
        "planner_mode": planner_mode,
        "model": model,
        "benchmark_count": len(benchmark_results),
        "case_count": combined_result["metrics"]["case_count"],
        "aggregate_metrics": combined_result["metrics"],
        "benchmarks": benchmark_results,
        "outputs": {
            "aggregate_metrics": str(combined_metrics_path),
            "aggregate_review_csv": str(combined_review_path),
            "suite_summary": str(suite_summary_path),
            "gold_root": str(gold_root),
            "output_root": str(output_root),
        },
    }
    write_json(suite_summary_path, summary)
    return summary


def run_benchmark(
    spec: dict[str, Any],
    *,
    output_root: Path,
    gold_root: Path,
    planner_mode: str,
    model: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_value = spec.get("input")
    if not input_value:
        raise ValueError("Benchmark spec is missing input")
    input_path = Path(str(input_value))
    dataset = str(spec.get("dataset") or "auto")
    source_wrapper = str(spec.get("source_wrapper") or "auto")
    limit = spec.get("limit")
    name = str(spec.get("name") or input_path.stem)
    safe_name = slug(name)

    raw_records = read_records(input_path)
    if limit is not None:
        raw_records = raw_records[: int(limit)]
    gold_rows = adapt_records(
        raw_records,
        dataset=dataset,
        id_prefix=safe_name,
        source_wrapper=source_wrapper,
    )
    gold_path = gold_root / f"{safe_name}.gold.jsonl"
    predictions_path = output_root / f"{safe_name}.predictions.jsonl"
    metrics_path = output_root / f"{safe_name}.metrics.json"
    review_path = output_root / f"{safe_name}.review.csv"

    write_jsonl(gold_path, gold_rows)
    prediction_rows = run_planner_rows(
        gold_rows,
        planner_mode=planner_mode,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    write_jsonl(predictions_path, prediction_rows)

    scored = score_predictions(gold_rows, prediction_rows)
    write_json(metrics_path, scored)
    write_csv(review_path, scored["cases"])
    detected_datasets = sorted(
        {
            str(row.get("source_dataset"))
            for row in gold_rows
            if row.get("source_dataset")
        }
    )
    detected_profiles = sorted(
        {
            str(row.get("benchmark_profile"))
            for row in gold_rows
            if row.get("benchmark_profile")
        }
    )

    summary = {
        "name": name,
        "dataset": canonical_dataset(dataset),
        "detected_datasets": detected_datasets,
        "detected_profiles": detected_profiles,
        "input": str(input_path),
        "raw_records": len(raw_records),
        "adapted_cases": len(gold_rows),
        "metrics": scored["metrics"],
        "outputs": {
            "gold": str(gold_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "review_csv": str(review_path),
        },
    }
    return {"summary": summary, "gold_rows": gold_rows, "prediction_rows": prediction_rows}


def run_planner_rows(
    rows: list[dict[str, Any]],
    *,
    planner_mode: str,
    model: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prediction_rows = []
    for index, row in enumerate(rows):
        try:
            task, context, attachments_metadata = row_to_request(row)
            result = run_capability_plan(
                None,
                None,
                task,
                context,
                attachments_metadata,
                max_new_tokens,
                planner_mode,
                False,
            )
            result.update(
                {
                    "id": row.get("id"),
                    "row_index": row.get("row_index", index),
                    "label": row.get("label"),
                    "category": row.get("category"),
                    "source_dataset": row.get("source_dataset"),
                    "benchmark_profile": row.get("benchmark_profile"),
                    "model": model,
                    "planner_mode": planner_mode,
                }
            )
        except Exception as exc:
            result = {
                "id": row.get("id"),
                "row_index": row.get("row_index", index),
                "category": row.get("category"),
                "source_dataset": row.get("source_dataset"),
                "benchmark_profile": row.get("benchmark_profile"),
                "model": model,
                "planner_mode": planner_mode,
                "error": f"{type(exc).__name__}: {exc}",
                "passes": {},
                "validation": {
                    "valid": False,
                    "violations": [
                        {
                            "type": "runner_error",
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "minimal_repairs": [],
                },
            }
        prediction_rows.append(result)
    return prediction_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark-shaped capability-planning evals end to end: "
            "raw benchmark rows -> adapter -> planner -> scorer -> aggregate metrics."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--source-wrapper",
        choices=["auto", "summarize", "classify", "extract", "rewrite", "translate", "compare", "format", "explain"],
        default="auto",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--planner-mode", choices=["rules_first"], default="rules_first")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()

    summary = run_suite(suite_from_args(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
