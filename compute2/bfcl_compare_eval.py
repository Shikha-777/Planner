#!/usr/bin/env python3
import argparse
import ast
import json
import os
import random
import re
from pathlib import Path

try:
    from smoke_eval import (
        run_baseline,
        run_cardinality_pipeline,
        run_pipeline,
        run_react_baseline,
        route_tool_call_candidates,
        score_calls,
        summarize,
        summarize_runtime_diagnostics,
    )
    from grammar_pipeline import run_grammar_pipeline
except ImportError:
    from .smoke_eval import (
        run_baseline,
        run_cardinality_pipeline,
        run_pipeline,
        run_react_baseline,
        route_tool_call_candidates,
        score_calls,
        summarize,
        summarize_runtime_diagnostics,
    )
    from .grammar_pipeline import run_grammar_pipeline


TOOL_KEYS = (
    "tools",
    "tool",
    "functions",
    "function",
    "function_list",
    "function_docs",
    "available_tools",
)

PROMPT_KEYS = ("prompt", "query", "instruction", "user_prompt", "question")

EXPECTED_KEYS = (
    "expected_calls",
    "ground_truth",
    "answer",
    "answers",
    "target",
    "label",
)


def load_records(path):
    raw = Path(path).read_text()
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    try:
        value = json.loads(raw)
    except ValueError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "records", "examples"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    raise ValueError(f"Unsupported BFCL input shape in {path}")


def maybe_json(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return value
    return value


def as_list(value):
    value = maybe_json(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_tool(tool):
    tool = maybe_json(tool)
    if isinstance(tool, list):
        return [normalize_tool(item) for item in tool]
    if not isinstance(tool, dict):
        raise ValueError(f"Unsupported tool entry: {tool!r}")

    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = dict(tool["function"])
    elif isinstance(tool.get("function"), dict):
        fn = dict(tool["function"])
    else:
        fn = dict(tool)

    name = fn.get("name") or fn.get("function_name")
    if not name:
        raise ValueError(f"Function schema is missing a name: {tool}")

    return {
        "name": name,
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters") or fn.get("parameter") or {"type": "object", "properties": {}},
    }


def flatten_tools(raw_tools):
    tools = []
    for item in as_list(raw_tools):
        normalized = normalize_tool(item)
        if isinstance(normalized, list):
            tools.extend(normalized)
        else:
            tools.append(normalized)
    return tools


def load_function_doc(path):
    records = load_records(str(path))
    tools = []
    for record in records:
        try:
            tools.extend(flatten_tools(record))
        except ValueError:
            for key in TOOL_KEYS:
                if key in record:
                    tools.extend(flatten_tools(record[key]))
    return tools


def normalize_tools(record, function_doc_dir=None):
    tools = []
    for key in TOOL_KEYS:
        if key in record:
            tools.extend(flatten_tools(record[key]))

    if function_doc_dir:
        doc_refs = []
        for key in ("function_doc", "function_doc_file", "function_doc_path", "involved_classes"):
            doc_refs.extend(as_list(record.get(key)))
        for ref in doc_refs:
            if not isinstance(ref, str):
                continue
            candidate = Path(function_doc_dir) / ref
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".json")
            if candidate.exists():
                tools.extend(load_function_doc(candidate))

    unique = {}
    for tool in tools:
        unique[tool["name"]] = tool
    return list(unique.values())


def message_text(message):
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content") or message.get("message") or message.get("text")
        if isinstance(content, str):
            role = message.get("role")
            return f"{role}: {content}" if role else content
    if isinstance(message, list):
        return "\n".join(filter(None, [message_text(item) for item in message]))
    return ""


def extract_prompt(record):
    if isinstance(record.get("messages"), list):
        return message_text(record["messages"])
    for key in PROMPT_KEYS:
        if key in record:
            text = message_text(record[key])
            if text:
                return text
    raise ValueError(f"Could not find prompt/question in record {record.get('id', '<no id>')}")


def parse_arguments_text(text):
    args = {}
    text = text.strip()
    if not text:
        return args
    for part in re.split(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)", text):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        try:
            args[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            args[key] = value
    return args


def normalize_expected_call(value):
    value = maybe_json(value)
    if isinstance(value, dict):
        if "function" in value and isinstance(value["function"], dict):
            fn = value["function"]
            args = fn.get("arguments", {})
            return {"name": fn.get("name"), "arguments": maybe_json(args)}
        name = value.get("name") or value.get("function_name")
        args = value.get("arguments") or value.get("args") or value.get("parameters") or {}
        if name:
            return {"name": name, "arguments": maybe_json(args)}
    if isinstance(value, str):
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", value)
        if match:
            return {"name": match.group(1), "arguments": parse_arguments_text(match.group(2))}
    return None


def extract_expected_calls(record):
    for key in EXPECTED_KEYS:
        if key not in record:
            continue
        value = maybe_json(record[key])
        calls = []
        for item in as_list(value):
            if isinstance(item, list):
                for nested in item:
                    call = normalize_expected_call(nested)
                    if call:
                        calls.append(call)
            else:
                call = normalize_expected_call(item)
                if call:
                    calls.append(call)
        if calls:
            return calls
    return []


def record_id(record, index):
    return str(record.get("id") or record.get("question_id") or record.get("idx") or index)


def lane_order(
    order,
    index,
    include_react=False,
    include_grammar_pipeline=False,
    include_cardinality_pipeline=False,
):
    lanes = ["baseline", "react", "pipeline"] if include_react else ["baseline", "pipeline"]
    if order == "baseline-first":
        ordered = lanes
    elif order == "pipeline-first":
        ordered = (
            ["pipeline", "react", "baseline"]
            if include_react
            else ["pipeline", "baseline"]
        )
    elif order == "react-first" and include_react:
        ordered = ["react", "baseline", "pipeline"]
    else:
        rotations = [lanes[i:] + lanes[:i] for i in range(len(lanes))]
        ordered = rotations[index % len(rotations)]
    if include_grammar_pipeline:
        ordered = ordered + ["grammar_pipeline"]
    if include_cardinality_pipeline:
        ordered = ordered + ["cardinality_pipeline"]
    return ordered


def planned_tool_names(plan, available_names):
    names = []
    for line in plan.splitlines():
        match = re.search(r"\btool=([^;\s]+)", line)
        if not match:
            continue
        name = match.group(1).strip()
        if name in available_names and name not in names:
            names.append(name)
    return names


def called_tool_names(calls):
    names = []
    for call in calls:
        name = call.get("name")
        if name and name not in names:
            names.append(name)
    return names


def coverage_diagnostics(planned, calls):
    called = called_tool_names(calls)
    return {
        "called_tools": called,
        "missing_planned_tools": [name for name in planned if name not in called],
        "extra_unplanned_tools": [name for name in called if name not in planned],
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline vs decomposition on BFCL-style records.")
    parser.add_argument("--input", default=os.environ.get("BFCL_INPUT", "benchmarks/sample_bfcl_cases.jsonl"))
    parser.add_argument("--function-doc-dir", default=os.environ.get("BFCL_FUNCTION_DOC_DIR", ""))
    parser.add_argument("--output", default=os.environ.get("BFCL_COMPARE_OUTPUT", "benchmark_results/bfcl_compare_results.jsonl"))
    parser.add_argument("--summary-output", default=os.environ.get("BFCL_COMPARE_SUMMARY", ""))
    parser.add_argument("--endpoint", default=os.environ.get("QWEN_ENDPOINT", "http://127.0.0.1:11434/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", os.environ.get("MODEL", "qwen2.5:14b")))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("QWEN_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("QWEN_TIMEOUT", "900")))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("BFCL_LIMIT", "0")))
    parser.add_argument("--offset", type=int, default=int(os.environ.get("BFCL_OFFSET", "0")))
    parser.add_argument("--sample", type=int, default=int(os.environ.get("BFCL_SAMPLE", "0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("BFCL_SEED", "13")))
    parser.add_argument("--pipeline-mode", choices=["per-subtask", "context"], default=os.environ.get("PIPELINE_MODE", "per-subtask"))
    parser.add_argument("--order", choices=["alternate", "baseline-first", "pipeline-first", "react-first"], default=os.environ.get("LANE_ORDER", "alternate"))
    parser.add_argument(
        "--include-react",
        action="store_true",
        default=os.environ.get("INCLUDE_REACT", "0") in {"1", "true", "TRUE", "yes", "YES"},
        help="Also run a ReAct-style no-execution single-pass baseline.",
    )
    parser.add_argument(
        "--include-grammar-pipeline",
        action="store_true",
        default=os.environ.get("INCLUDE_GRAMMAR_PIPELINE", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Add the separate Ollama structured-output pipeline lane.",
    )
    parser.add_argument(
        "--include-cardinality-pipeline",
        action="store_true",
        default=os.environ.get("INCLUDE_CARDINALITY_PIPELINE", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Add a separate schema-aware cardinality-repair pipeline lane.",
    )
    parser.add_argument(
        "--include-selective-pipeline",
        action="store_true",
        default=os.environ.get("INCLUDE_SELECTIVE_PIPELINE", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Add a post-generation router that chooses between direct and "
            "decomposed candidates using schema, grounding, and plan signals."
        ),
    )
    parser.add_argument(
        "--selective-router-margin",
        type=float,
        default=float(os.environ.get("SELECTIVE_ROUTER_MARGIN", "0.45")),
        help=(
            "Minimum score advantage required before the router selects the "
            "decomposition candidate. Ties fall back to direct."
        ),
    )
    parser.add_argument(
        "--selective-abstention-policy",
        choices=["strict", "calibrated", "calibrated_trust_no_intent"],
        default=os.environ.get("SELECTIVE_ABSTENTION_POLICY", "strict"),
        help=(
            "How the selective router handles empty pipeline/no_intent outputs. "
            "strict always trusts them; calibrated lets a schema-valid, "
            "action-like direct call override uncertain abstention; "
            "calibrated_trust_no_intent keeps calibrated behavior except for "
            "explicit planner no-intent verdicts."
        ),
    )
    parser.add_argument(
        "--selective-include-react",
        action="store_true",
        default=os.environ.get("SELECTIVE_INCLUDE_REACT", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Allow the selective router to choose ReAct when that lane is present.",
    )
    parser.add_argument(
        "--selective-react-policy",
        choices=["score_margin", "empty_only"],
        default=os.environ.get("SELECTIVE_REACT_POLICY", "score_margin"),
        help=(
            "When ReAct is enabled for the selective router, score_margin lets "
            "it win by router score; empty_only only lets it rescue cases where "
            "direct and pipeline both emitted no calls."
        ),
    )
    parser.add_argument(
        "--grammar-only",
        action="store_true",
        default=os.environ.get("GRAMMAR_ONLY", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Run only the grammar pipeline lane; useful for fast grammar on/off tests.",
    )
    parser.add_argument(
        "--grammar-constraint",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("GRAMMAR_CONSTRAINT", "1")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Enable or disable structured decoding for the grammar pipeline lane.",
    )
    parser.add_argument(
        "--grammar-logprobs",
        action="store_true",
        default=os.environ.get("GRAMMAR_LOGPROBS", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--grammar-probe-logprobs",
        action="store_true",
        default=os.environ.get("GRAMMAR_PROBE_LOGPROBS", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help="Run an extra one-token unconstrained probe for a masked-mass lower bound.",
    )
    parser.add_argument(
        "--grammar-top-logprobs",
        type=int,
        default=int(os.environ.get("GRAMMAR_TOP_LOGPROBS", "20")),
    )
    parser.add_argument(
        "--drop-ungrounded-optional-args",
        action="store_true",
        default=os.environ.get("DROP_UNGROUNDED_OPTIONAL_ARGS", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "In pipeline verification, remove optional scalar arguments whose "
            "values are not grounded in the original prompt. Baseline/ReAct are "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--strict-value-copy",
        action="store_true",
        default=os.environ.get("STRICT_VALUE_COPY", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Add stricter exact-span value-copy instructions to pipeline "
            "selectors. Baseline/ReAct are unchanged."
        ),
    )
    parser.add_argument(
        "--value-copy-fewshot",
        action="store_true",
        default=os.environ.get("VALUE_COPY_FEWSHOT", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Add few-shot selector examples for exact value copying and simple "
            "canonicalization. Baseline/ReAct are unchanged."
        ),
    )
    parser.add_argument(
        "--irrelevance-guard",
        action="store_true",
        default=os.environ.get("IRRELEVANCE_GUARD", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Run a final pipeline-only abstention verifier that may override "
            "candidate calls to [] when the request should not execute any "
            "tool. Baseline/ReAct are unchanged."
        ),
    )
    parser.add_argument(
        "--irrelevance-prefilter",
        action="store_true",
        default=os.environ.get("IRRELEVANCE_PREFILTER", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Run a pipeline-only no-call classifier after planning but before "
            "selector calls. If it says abstain, emit [] directly. "
            "Baseline/ReAct are unchanged."
        ),
    )
    parser.add_argument(
        "--span-inventory",
        action="store_true",
        default=os.environ.get("SPAN_INVENTORY", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
        help=(
            "Add a deterministic, advisory inventory of prompt-grounded spans "
            "to pipeline selectors. Baseline/ReAct are unchanged."
        ),
    )
    parser.add_argument(
        "--multseq-k",
        type=int,
        default=int(os.environ.get("MULTSEQ_K", "1")),
        help=(
            "Number of decomposition plans to generate (MultSeq). "
            "1 = off (default). 3 is recommended for live categories."
        ),
    )
    parser.add_argument(
        "--multseq-temperature",
        type=float,
        default=float(os.environ.get("MULTSEQ_TEMPERATURE", "0.5")),
        help="Sampling temperature for MultSeq planner calls (diversity). Default 0.5.",
    )
    parser.add_argument(
        "--multseq-min-votes",
        type=int,
        default=int(os.environ.get("MULTSEQ_MIN_VOTES", "2")),
        help=(
            "Minimum number of plans that must contain a subtask for it to be kept. "
            "Default 2 (majority of 3). Set to 1 for union (full MultSeq)."
        ),
    )
    parser.add_argument(
        "--multseq-strategy",
        choices=["vote", "medoid", "score_filter", "saf"],
        default=os.environ.get("MULTSEQ_STRATEGY", "vote"),
        help=(
            "How to use sampled plans: vote merges recurring subtasks; medoid "
            "selects the single plan with highest cross-plan agreement; "
            "score_filter/saf selects a sampled plan with consensus, grounding, "
            "and fragmentation penalties."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_records(args.input)
    total_records = len(records)
    if args.offset:
        records = records[args.offset :]
    if args.sample:
        rng = random.Random(args.seed)
        sample_size = min(args.sample, len(records))
        records = rng.sample(records, sample_size)
    if args.limit:
        records = records[: args.limit]

    if args.dry_run:
        preview = []
        for index, record in enumerate(records[:3]):
            tools = normalize_tools(record, args.function_doc_dir or None)
            preview.append(
                {
                    "id": record_id(record, index),
                    "prompt_preview": extract_prompt(record)[:160],
                    "tool_count": len(tools),
                    "tool_names": [tool["name"] for tool in tools[:8]],
                    "expected_count": len(extract_expected_calls(record)),
                }
            )
        print(
            json.dumps(
                {
                    "input": args.input,
                    "total_records": total_records,
                    "offset": args.offset,
                    "sample": args.sample,
                    "seed": args.seed,
                    "records": len(records),
                    "model": args.model,
                    "pipeline_mode": args.pipeline_mode,
                    "include_react": args.include_react,
                    "include_grammar_pipeline": args.include_grammar_pipeline,
                    "include_cardinality_pipeline": args.include_cardinality_pipeline,
                    "grammar_only": args.grammar_only,
                    "grammar_constraint": args.grammar_constraint,
                    "drop_ungrounded_optional_args": args.drop_ungrounded_optional_args,
                    "strict_value_copy": args.strict_value_copy,
                    "value_copy_fewshot": args.value_copy_fewshot,
                    "irrelevance_guard": args.irrelevance_guard,
                    "irrelevance_prefilter": args.irrelevance_prefilter,
                    "span_inventory": args.span_inventory,
                    "preview": preview,
                },
                indent=2,
            )
        )
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output_path.open("w") as handle:
        for index, record in enumerate(records):
            rid = record_id(record, index)
            print(f"running {rid}...", flush=True)
            task = {
                "id": rid,
                "prompt": extract_prompt(record),
                "tools": normalize_tools(record, args.function_doc_dir or None),
                "expected_calls": extract_expected_calls(record),
            }
            if not task["tools"]:
                raise ValueError(f"Record {rid} has no tools")

            baseline = None
            react = None
            pipeline = None
            grammar_pipeline = None
            cardinality_pipeline = None
            selective_pipeline = None
            order_index = index if args.sample else index + args.offset
            if args.grammar_only:
                order = ["grammar_pipeline"]
            else:
                order = lane_order(
                    args.order,
                    order_index,
                    include_react=args.include_react,
                    include_grammar_pipeline=args.include_grammar_pipeline,
                    include_cardinality_pipeline=args.include_cardinality_pipeline,
                )
            for lane in order:
                if lane == "baseline":
                    baseline = run_baseline(task, args)
                elif lane == "react":
                    react = run_react_baseline(task, args)
                elif lane == "grammar_pipeline":
                    grammar_pipeline = run_grammar_pipeline(task, args)
                elif lane == "cardinality_pipeline":
                    cardinality_pipeline = run_cardinality_pipeline(task, args)
                else:
                    pipeline = run_pipeline(task, args)
            if not args.grammar_only:
                assert baseline is not None and pipeline is not None
                if args.include_selective_pipeline:
                    selective_pipeline = route_tool_call_candidates(
                        task,
                        baseline,
                        pipeline,
                        react=react,
                        margin=args.selective_router_margin,
                        abstention_policy=args.selective_abstention_policy,
                        include_react=args.selective_include_react,
                        react_policy=args.selective_react_policy,
                    )
            assert grammar_pipeline is not None if args.grammar_only else True
            plan_source = grammar_pipeline if args.grammar_only else pipeline
            planned_tools = planned_tool_names(
                plan_source["decomposition"],
                [tool["name"] for tool in task["tools"]],
            )

            row = {
                "id": rid,
                "prompt": task["prompt"],
                "tool_names": [tool["name"] for tool in task["tools"]],
                "planned_tools": planned_tools,
                "expected_calls": task["expected_calls"],
            }
            if baseline is not None:
                row["baseline"] = baseline
                row["baseline_coverage"] = coverage_diagnostics(
                    planned_tools,
                    baseline["normalized_calls"],
                )
            if pipeline is not None:
                row["pipeline"] = pipeline
                row["pipeline_coverage"] = coverage_diagnostics(
                    planned_tools,
                    pipeline["normalized_calls"],
                )
            if selective_pipeline is not None:
                row["selective_pipeline"] = selective_pipeline
                row["selective_pipeline_coverage"] = coverage_diagnostics(
                    planned_tools,
                    selective_pipeline["normalized_calls"],
                )
            if args.include_react:
                assert react is not None
                row["react"] = react
                row["react_coverage"] = coverage_diagnostics(planned_tools, react["normalized_calls"])
            if args.include_grammar_pipeline or args.grammar_only:
                assert grammar_pipeline is not None
                row["grammar_pipeline"] = grammar_pipeline
                row["grammar_pipeline_coverage"] = coverage_diagnostics(
                    planned_tools,
                    grammar_pipeline["normalized_calls"],
                )
            if args.include_cardinality_pipeline:
                assert cardinality_pipeline is not None
                row["cardinality_pipeline"] = cardinality_pipeline
                row["cardinality_pipeline_coverage"] = coverage_diagnostics(
                    planned_tools,
                    cardinality_pipeline["normalized_calls"],
                )
            results.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            call_parts = []
            token_parts = []
            if baseline is not None:
                call_parts.append(
                    f"baseline calls={len(baseline['normalized_calls'])}"
                )
                token_parts.append(
                    "baseline="
                    f"{baseline['usage']['max_request_prompt_tokens']}"
                )
            if pipeline is not None:
                call_parts.append(
                    f"pipeline calls={len(pipeline['normalized_calls'])}"
                )
                token_parts.append(
                    "pipeline="
                    f"{pipeline['usage']['max_request_prompt_tokens']}"
                )
            if react is not None:
                call_parts.append(f"react calls={len(react['normalized_calls'])}")
                token_parts.append(
                    f"react={react['usage']['max_request_prompt_tokens']}"
                )
            if grammar_pipeline is not None:
                call_parts.append(
                    "grammar_pipeline calls="
                    f"{len(grammar_pipeline['normalized_calls'])}"
                )
                token_parts.append(
                    "grammar_pipeline="
                    f"{grammar_pipeline['usage']['max_request_prompt_tokens']}"
                )
            if cardinality_pipeline is not None:
                call_parts.append(
                    "cardinality_pipeline calls="
                    f"{len(cardinality_pipeline['normalized_calls'])}"
                )
                token_parts.append(
                    "cardinality_pipeline="
                    f"{cardinality_pipeline['usage']['max_request_prompt_tokens']}"
                )
            if selective_pipeline is not None:
                call_parts.append(
                    "selective_pipeline calls="
                    f"{len(selective_pipeline['normalized_calls'])}"
                )
                token_parts.append(
                    "selective_pipeline="
                    f"{selective_pipeline['usage']['max_request_prompt_tokens']}"
                )
            grammar_text = ""
            if grammar_pipeline is not None:
                diagnostics = grammar_pipeline["grammar_diagnostics"]
                grammar_text = (
                    f"; grammar calls={len(grammar_pipeline['normalized_calls'])}"
                    f", constraint={grammar_pipeline['grammar_enabled']}"
                    f", parse_failures={diagnostics['parse_failure_count']}"
                    f", hallucinated_tools={diagnostics['hallucinated_tool_count']}"
                )
            print(
                "  " + "; ".join(call_parts),
                f"{grammar_text}; max prompt tokens {', '.join(token_parts)}; "
                f"verdict={plan_source.get('verdict')}; "
                f"dropped={plan_source.get('verification', {}).get('dropped_count', 0)}; "
                f"merged={plan_source.get('verification', {}).get('merged_count', 0)}",
                flush=True,
            )

    scored = [row for row in results if row["expected_calls"]]
    context_length = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "0") or 0)
    summary = {
        "model": args.model,
        "input": args.input,
        "total_records": total_records,
        "offset": args.offset,
        "sample": args.sample,
        "seed": args.seed,
        "pipeline_mode": args.pipeline_mode,
        "include_react": args.include_react,
        "include_grammar_pipeline": args.include_grammar_pipeline,
        "include_cardinality_pipeline": args.include_cardinality_pipeline,
        "include_selective_pipeline": args.include_selective_pipeline,
        "selective_router_margin": args.selective_router_margin,
        "grammar_only": args.grammar_only,
        "grammar_constraint": args.grammar_constraint,
        "drop_ungrounded_optional_args": args.drop_ungrounded_optional_args,
        "strict_value_copy": args.strict_value_copy,
        "value_copy_fewshot": args.value_copy_fewshot,
        "irrelevance_guard": args.irrelevance_guard,
        "irrelevance_prefilter": args.irrelevance_prefilter,
        "span_inventory": args.span_inventory,
        "context_length": context_length,
        "total": len(results),
        "scored": len(scored),
        "summary": summarize(scored) if scored else None,
        "diagnostics": summarize_runtime_diagnostics(
            results,
            context_length=context_length or None,
        ),
        "output": str(output_path),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
