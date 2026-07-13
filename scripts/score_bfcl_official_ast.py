#!/usr/bin/env python3
"""Score compare-run JSONL with BFCL's official AST checker.

This is intended for BFCL single-turn categories. For AST categories such as
simple_python, multiple, parallel, and parallel_multiple, it imports BFCL's
official ast_checker. For relevance/irrelevance, it mirrors BFCL's separate
presence/absence-of-function-call check on the already-normalized calls.
"""

import argparse
import json
import sys
import types
from pathlib import Path

# compute2.bfcl_compare_eval imports grammar_pipeline for other workflows. The
# official AST scorer only needs record/tool normalization, so keep scoring
# independent of that optional pipeline module.
if "grammar_pipeline" not in sys.modules:
    grammar_pipeline = types.ModuleType("grammar_pipeline")
    grammar_pipeline.run_grammar_pipeline = lambda *args, **kwargs: None
    sys.modules["grammar_pipeline"] = grammar_pipeline

from bfcl_compare_eval import load_records, normalize_tools, record_id


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_answers(path):
    if not path:
        return {}
    answers = {}
    for row in load_jsonl(path):
        rid = str(row.get("id") or row.get("question_id"))
        answers[rid] = row.get("ground_truth", [])
    return answers


def load_tools_by_id(path, function_doc_dir=""):
    records = load_records(path)
    tools_by_id = {}
    for index, record in enumerate(records):
        rid = record_id(record, index)
        tools_by_id[rid] = normalize_tools(record, function_doc_dir or None)
    return tools_by_id


def category_from_id(record_id_value):
    return str(record_id_value).rsplit("_", 1)[0]


def is_relevance_or_irrelevance(category):
    return "relevance" in category or "irrelevance" in category


def language_for_category(category):
    from bfcl_eval.constants.enums import Language

    if category.endswith("_java"):
        return Language.JAVA
    if category.endswith("_javascript"):
        return Language.JAVASCRIPT
    return Language.PYTHON


def to_official_model_output(calls):
    return [{call["name"]: call.get("arguments") or {}} for call in calls]


def check_lane(ast_checker, tools, calls, expected, language, category, checker_model_name):
    output = to_official_model_output(calls)
    result = ast_checker(
        tools,
        output,
        expected,
        language,
        category,
        checker_model_name,
    )
    return {
        "valid": bool(result.get("valid")),
        "error": result.get("error", []),
        "error_type": result.get("error_type", ""),
        "model_output": output,
    }


def check_relevance_lane(calls, category):
    has_call = bool(calls)
    if "irrelevance" in category:
        valid = not has_call
        error_type = "" if valid else "irrelevance_error:decoder_success"
        error = [] if valid else ["Function call present when BFCL expects no function call."]
    else:
        valid = has_call
        error_type = "" if valid else "relevance_error:decoder_failed"
        error = [] if valid else ["No function call present when BFCL expects a function call."]
    return {
        "valid": valid,
        "error": error,
        "error_type": error_type,
        "model_output": to_official_model_output(calls),
    }


def summarize(rows, lane):
    total = len(rows)
    valid = sum(1 for row in rows if row[lane]["valid"])
    return {
        "exact_accuracy": round(valid / total, 3) if total else 0.0,
        "exact_count": valid,
        "total": total,
    }


def available_lanes(row):
    return [
        lane
        for lane in (
            "baseline",
            "react",
            "pipeline",
            "selective_pipeline",
            "grammar_pipeline",
            "cardinality_pipeline",
        )
        if lane in row
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="bfcl_compare_*.jsonl file")
    parser.add_argument("--input", required=True, help="Official BFCL question JSON/JSONL file")
    parser.add_argument(
        "--answers",
        default="",
        help="Official BFCL possible_answer JSON/JSONL file. Not needed for relevance/irrelevance.",
    )
    parser.add_argument("--function-doc-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--checker-model-name",
        default="gorilla-openfunctions-v2",
        help=(
            "Existing BFCL model config name used only for function-name conversion. "
            "The default keeps dotted function names unchanged."
        ),
    )
    parser.add_argument(
        "--convert-dots-to-underscores",
        action="store_true",
        help=(
            "Mirror BFCL model configs whose checker replaces dotted function names "
            "with underscores, e.g. foo.bar -> foo_bar. Leave off for our normalized "
            "compare outputs, which keep original BFCL tool names."
        ),
    )
    args = parser.parse_args()

    # BFCL's ast_checker imports MODEL_CONFIG_MAPPING only to decide whether
    # function names with dots should be converted to underscores. Importing the
    # full BFCL model registry pulls in many API-provider dependencies that are
    # irrelevant for offline scoring, so provide the one field ast_checker uses.
    model_config_stub = types.ModuleType("bfcl_eval.constants.model_config")
    model_config_stub.MODEL_CONFIG_MAPPING = {
        args.checker_model_name.replace("_", "/"): types.SimpleNamespace(
            underscore_to_dot=args.convert_dots_to_underscores
        )
    }
    sys.modules["bfcl_eval.constants.model_config"] = model_config_stub

    rows = []
    missing = []
    answers = load_answers(args.answers)
    tools_by_id = load_tools_by_id(args.input, args.function_doc_dir)
    prediction_rows = load_jsonl(args.predictions)
    needs_ast_checker = any(
        not is_relevance_or_irrelevance(category_from_id(row["id"]))
        for row in prediction_rows
    )
    ast_checker = None
    if needs_ast_checker:
        try:
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        except ImportError as exc:
            raise SystemExit(
                "Could not import BFCL ast_checker. Run with PYTHONPATH pointing to "
                "external/gorilla/berkeley-function-call-leaderboard, or install bfcl-eval."
            ) from exc

    for row in prediction_rows:
        rid = str(row["id"])
        category = category_from_id(rid)
        relevance_mode = is_relevance_or_irrelevance(category)
        if (not relevance_mode and rid not in answers) or rid not in tools_by_id:
            missing.append(rid)
            continue

        expected = [] if relevance_mode else answers[rid]
        tools = tools_by_id[rid]

        lanes = available_lanes(row)
        if relevance_mode:
            scored = {
                "id": rid,
                "category": category,
                "expected": expected,
            }
            for lane in lanes:
                scored[lane] = check_relevance_lane(
                    row[lane]["normalized_calls"],
                    category,
                )
        else:
            language = language_for_category(category)
            scored = {
                "id": rid,
                "category": category,
                "expected": expected,
            }
            for lane in lanes:
                scored[lane] = check_lane(
                    ast_checker,
                    tools,
                    row[lane]["normalized_calls"],
                    expected,
                    language,
                    category,
                    args.checker_model_name,
                )
        rows.append(scored)

    all_lanes = available_lanes(rows[0]) if rows else []
    both = baseline_only = pipeline_only = neither = 0
    for row in rows:
        if "baseline" in row and "pipeline" in row:
            baseline_valid = row["baseline"]["valid"]
            pipeline_valid = row["pipeline"]["valid"]
            if baseline_valid and pipeline_valid:
                both += 1
            elif baseline_valid and not pipeline_valid:
                baseline_only += 1
            elif pipeline_valid and not baseline_valid:
                pipeline_only += 1
            else:
                neither += 1

    report = {
        "predictions": args.predictions,
        "input": args.input,
        "answers": args.answers,
        "checker_model_name": args.checker_model_name,
        "scored": len(rows),
        "missing_ids": missing,
        "summary": {lane: summarize(rows, lane) for lane in all_lanes},
        "breakdown": {
            "both_exact": both,
            "baseline_only": baseline_only,
            "pipeline_only": pipeline_only,
            "neither": neither,
        },
        "rows": rows,
    }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["breakdown"], indent=2))
    if missing:
        print("missing ids:", missing[:20])


if __name__ == "__main__":
    main()
