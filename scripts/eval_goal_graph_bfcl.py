#!/usr/bin/env python3
"""Evaluate BFCL tool-name routing through the goal-graph runtime.

This runner scores the compiled graph calls against BFCL possible answers. It
does not execute benchmark functions; it measures tool name, count, and order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
COMPUTE2_DIR = ROOT_DIR / "compute2"
for path in (SRC_DIR, SCRIPT_DIR, COMPUTE2_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# compute2.bfcl_compare_eval imports grammar_pipeline for other workflows. This
# scorer only needs prompt/tool normalization, so a stub keeps local runs light.
if "grammar_pipeline" not in sys.modules:
    grammar_pipeline = types.ModuleType("grammar_pipeline")
    grammar_pipeline.run_grammar_pipeline = lambda *args, **kwargs: None
    sys.modules["grammar_pipeline"] = grammar_pipeline

from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
from goal_graph_eval_common import benchmark_compile_tools, plan_and_compile_goal_graph
from run_gptoss_capability_plan import generate_text, load_model
from taskdecomp.tool_binding import _model_binding_schema_default, _properties, _required


DEFAULT_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_user_turn(text: str) -> str:
    role_pattern = re.compile(r"(?im)^(system|user|assistant|tool):\s*")
    matches = list(role_pattern.finditer(text))
    if not matches:
        return re.sub(r"^user:\s*", "", text.strip(), flags=re.I)

    last_user: str | None = None
    for index, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if role == "user" and segment:
            last_user = segment
    return last_user if last_user is not None else text[matches[-1].end() :].strip()


def prompt_text(record: dict[str, Any]) -> str:
    return latest_user_turn(extract_prompt(record).strip())


def expected_names(answer_row: dict[str, Any]) -> list[str]:
    names = []
    for call in answer_row.get("ground_truth") or []:
        if isinstance(call, dict) and call:
            names.append(str(next(iter(call.keys()))))
    return names


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row.get("id")): row for row in read_jsonl(path)}


def multiset_equal(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def bfcl_result_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    compiled = [
        call
        for call in result.get("calls") or []
        if isinstance(call, dict) and (call.get("tool_name") or call.get("name"))
    ]
    if compiled:
        return compiled
    plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    return [
        call
        for call in plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name") and not call.get("missing_arguments")
    ]


def official_normalized_calls(
    result: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Convert compiled goal-graph calls to BFCL scorer-compatible calls."""
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools or []}
    normalized = []
    for call in bfcl_result_calls(result):
        if not isinstance(call, dict):
            continue
        name = call.get("tool_name") or call.get("name")
        if not name:
            continue
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        arguments = drop_ungrounded_optional_defaults(arguments, tools_by_name.get(str(name)), prompt)
        normalized.append({"name": str(name), "arguments": arguments})
    return normalized


def drop_ungrounded_optional_defaults(
    arguments: dict[str, Any],
    tool: dict[str, Any] | None,
    prompt: str,
) -> dict[str, Any]:
    if not tool:
        return arguments
    properties = _properties(tool)
    required = set(_required(tool))
    cleaned: dict[str, Any] = {}
    for name, value in arguments.items():
        spec = properties.get(str(name))
        if spec is not None and str(name) not in required:
            default = _model_binding_schema_default(spec)
            if default is not None and _same_scalar_value(value, default) and not value_grounded_in_prompt(prompt, value):
                continue
        cleaned[name] = value
    return cleaned


def _same_scalar_value(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left).strip().lower() == str(right).strip().lower()
    return left == right


def value_grounded_in_prompt(prompt: str, value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        lowered = prompt.lower()
        if text.lower() in lowered:
            return True
        tokens = [token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) > 2]
        return bool(tokens) and all(re.search(rf"\b{re.escape(token)}\b", lowered) for token in tokens)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(re.search(rf"(?<![\d.]){re.escape(str(value))}(?!\d)(?!\.\d)", prompt))
    if isinstance(value, list):
        return all(value_grounded_in_prompt(prompt, item) for item in value)
    return False


def evaluate_file(
    input_path: Path,
    answer_path: Path,
    offset: int,
    limit: int,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    repair_attempts: int,
    plan_handle: Any | None,
    official_handle: Any | None,
    allow_side_effects: bool,
    use_binder_fallback: bool,
    planner_mode: str,
    model_name: str,
) -> dict[str, Any]:
    records = load_records(str(input_path))
    if offset:
        records = records[offset:]
    if limit:
        records = records[:limit]
    answers = load_answers(answer_path)

    review_rows = []
    counts = Counter()
    total = 0
    expected_call_total = 0
    category = input_path.stem.replace("BFCL_v4_", "")

    for index, record in enumerate(records):
        rid = record_id(record, offset + index)
        answer = answers.get(rid, {"ground_truth": []})
        expected = expected_names(answer)
        if expected:
            expected_call_total += 1
        prompt = prompt_text(record)
        tools = benchmark_compile_tools(normalize_tools(record))
        result = plan_and_compile_goal_graph(
            model,
            tokenizer,
            generate_text,
            prompt,
            tools,
            max_new_tokens=max_new_tokens,
            repair_attempts=repair_attempts,
            allow_side_effects=allow_side_effects,
            use_binder_fallback=use_binder_fallback,
            planner_mode=planner_mode,
        )
        result_calls = bfcl_result_calls(result)
        predicted = [
            str(call.get("tool_name") or "")
            for call in result_calls
            if isinstance(call, dict) and call.get("tool_name")
        ]
        total += 1

        if plan_handle is not None:
            plan_handle.write(
                json.dumps(
                    {
                        "id": rid,
                        "category": category,
                        "prompt": prompt,
                        "expected": expected,
                        "predicted": predicted,
                        "goal_graph_result": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            plan_handle.flush()

        if official_handle is not None:
            official_handle.write(
                json.dumps(
                    {
                        "id": rid,
                        "baseline": {
                            "normalized_calls": official_normalized_calls(result, tools, prompt),
                            "model": model_name,
                            "error": result.get("parse_error") or "",
                            "verification_ok": bool(result.get("verification_ok")),
                            "diagnostic_codes": result.get("diagnostic_codes") or [],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            official_handle.flush()

        tool_set_ok = set(predicted) == set(expected)
        tool_multiset_ok = multiset_equal(predicted, expected)
        count_ok = len(predicted) == len(expected)
        ordered_ok = predicted == expected
        unordered_ok = tool_multiset_ok
        top1_tool_ok = bool(expected) and bool(predicted) and predicted[0] in set(expected)
        no_call_ok = not predicted and not expected

        counts["tool_set_ok"] += int(tool_set_ok)
        counts["tool_multiset_ok"] += int(tool_multiset_ok)
        counts["count_ok"] += int(count_ok)
        counts["ordered_ok"] += int(ordered_ok)
        counts["unordered_ok"] += int(unordered_ok)
        counts["top1_tool_ok"] += int(top1_tool_ok)
        counts["no_call_ok"] += int(no_call_ok)
        counts["parse_error"] += int(bool(result.get("parse_error")))
        counts["verification_error"] += int(not result.get("verification_ok"))
        counts["no_prediction"] += int(not predicted)

        if not ordered_ok:
            review_rows.append(
                {
                    "id": rid,
                    "prompt": prompt,
                    "expected": json.dumps(expected, ensure_ascii=False),
                    "predicted": json.dumps(predicted, ensure_ascii=False),
                    "tool_set_ok": tool_set_ok,
                    "tool_multiset_ok": tool_multiset_ok,
                    "count_ok": count_ok,
                    "ordered_ok": ordered_ok,
                    "verification_ok": bool(result.get("verification_ok")),
                    "parse_error": result.get("parse_error") or "",
                    "diagnostic_codes": json.dumps(result.get("diagnostic_codes") or []),
                    "diagnostics": json.dumps(result.get("diagnostics") or [], ensure_ascii=False),
                    "capability_count": result.get("capability_count", 0),
                    "compiled_calls": json.dumps(result_calls, ensure_ascii=False),
                }
            )

    metrics = {
        "category": category,
        "input": str(input_path),
        "answers": str(answer_path),
        "total": total,
        "tool_set_accuracy": counts["tool_set_ok"] / total if total else 0.0,
        "tool_multiset_accuracy": counts["tool_multiset_ok"] / total if total else 0.0,
        "count_accuracy": counts["count_ok"] / total if total else 0.0,
        "ordered_sequence_accuracy": counts["ordered_ok"] / total if total else 0.0,
        "unordered_sequence_accuracy": counts["unordered_ok"] / total if total else 0.0,
        "top1_tool_hit_rate_when_called": counts["top1_tool_ok"] / max(1, expected_call_total),
        "no_call_accuracy_count": counts["no_call_ok"],
        "parse_error_rate": counts["parse_error"] / total if total else 0.0,
        "verification_error_rate": counts["verification_error"] / total if total else 0.0,
        "no_prediction_rate": counts["no_prediction"] / total if total else 0.0,
        "failure_count": len(review_rows),
    }
    return {"metrics": metrics, "review_rows": review_rows}


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "category",
        "id",
        "prompt",
        "expected",
        "predicted",
        "tool_set_ok",
        "tool_multiset_ok",
        "count_ok",
        "ordered_ok",
        "verification_ok",
        "parse_error",
        "diagnostic_codes",
        "diagnostics",
        "capability_count",
        "compiled_calls",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--plans-jsonl")
    parser.add_argument(
        "--official-predictions-jsonl",
        help=(
            "Optional BFCL official-scorer prediction JSONL. Rows use the "
            "baseline.normalized_calls lane expected by score_bfcl_official_ast.py."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--planner-mode", choices=["stepwise", "one_shot"], default="stepwise")
    parser.add_argument(
        "--strict-read-only",
        action="store_true",
        help="Keep runtime side-effect gates active during offline benchmark scoring.",
    )
    parser.add_argument(
        "--no-binder-fallback",
        action="store_true",
        help="Disable frozen-pipeline binder fallback in one_shot planner mode.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    allow_side_effects = not args.strict_read_only
    use_binder_fallback = not args.no_binder_fallback

    data_dir = Path(args.data_dir)
    all_metrics = []
    all_review_rows = []
    plan_path = Path(args.plans_jsonl) if args.plans_jsonl else None
    official_path = Path(args.official_predictions_jsonl) if args.official_predictions_jsonl else None
    plan_handle = None
    official_handle = None
    try:
        if plan_path is not None:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_handle = plan_path.open("w", encoding="utf-8")
        if official_path is not None:
            official_path.parent.mkdir(parents=True, exist_ok=True)
            official_handle = official_path.open("w", encoding="utf-8")
        for category in args.categories:
            input_path = data_dir / f"BFCL_v4_{category}.json"
            answer_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
            result = evaluate_file(
                input_path,
                answer_path,
                args.offset,
                args.limit,
                model,
                tokenizer,
                args.max_new_tokens,
                args.repair_attempts,
                plan_handle,
                official_handle,
                allow_side_effects,
                use_binder_fallback,
                args.planner_mode,
                args.model,
            )
            all_metrics.append(result["metrics"])
            for row in result["review_rows"]:
                row["category"] = category
                all_review_rows.append(row)
    finally:
        if plan_handle is not None:
            plan_handle.close()
        if official_handle is not None:
            official_handle.close()

    total = sum(item["total"] for item in all_metrics)
    aggregate = {
        "total": total,
        "tool_set_accuracy": sum(item["tool_set_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "tool_multiset_accuracy": sum(item["tool_multiset_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "count_accuracy": sum(item["count_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "ordered_sequence_accuracy": sum(item["ordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "unordered_sequence_accuracy": sum(item["unordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "parse_error_rate": sum(item["parse_error_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "verification_error_rate": sum(item["verification_error_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "no_prediction_rate": sum(item["no_prediction_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
    }
    payload = {
        "pipeline": "goal_graph_runtime",
        "aggregate": aggregate,
        "by_category": all_metrics,
        "offset": args.offset,
        "limit": args.limit,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "repair_attempts": args.repair_attempts,
        "planner_mode": args.planner_mode,
        "benchmark_compile_allows_side_effect_names": allow_side_effects,
        "stepwise_binding_enabled": args.planner_mode == "stepwise",
        "binder_fallback_enabled": use_binder_fallback if args.planner_mode == "one_shot" else False,
        "official_predictions_jsonl": str(official_path) if official_path else "",
    }
    write_json(Path(args.output), payload)
    write_review_csv(Path(args.review_csv), all_review_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
