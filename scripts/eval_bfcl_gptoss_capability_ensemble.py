#!/usr/bin/env python3
"""Evaluate BFCL tool routing with GPT-OSS capability-plan ensemble context.

This measures only tool names, count, and order. The pipeline is:

1. Run the GPT-OSS capability planner, usually in ensemble mode.
2. Feed the resolved capability plan into the local tool binder.
3. Score predicted tool-name sequences against BFCL possible answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from run_gptoss_capability_plan import (
    DEFAULT_ENSEMBLE_VARIANTS,
    ensemble_requires_model,
    load_model,
    parse_ensemble_variants,
    run_capability_plan,
    run_semantic_slot_frame,
)
from taskdecomp.tool_binding import build_tool_binding_plan


STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "help",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "provide",
    "show",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "want",
    "what",
    "with",
    "you",
}


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


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[_\.\-/]+", " ", text)


def toks(text: Any) -> list[str]:
    raw = split_identifier(str(text or "")).lower()
    return [tok for tok in re.findall(r"[a-z0-9]{2,}", raw) if tok not in STOPWORDS]


def tool_schema_text(tool: dict[str, Any]) -> str:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    parts = [str(tool.get("name") or ""), str(tool.get("description") or "")]
    for key, spec in props.items():
        parts.append(str(key))
        if isinstance(spec, dict):
            parts.append(str(spec.get("description") or ""))
            enum = spec.get("enum")
            if isinstance(enum, list):
                parts.extend(str(item) for item in enum)
    return " ".join(parts)


def bm25_scores(prompt: str, tools: list[dict[str, Any]]) -> list[tuple[str, float]]:
    query = toks(prompt)
    if not query:
        return [(str(tool.get("name") or ""), 0.0) for tool in tools]
    docs = [toks(tool_schema_text(tool)) for tool in tools]
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))
    avg_len = sum(len(doc) for doc in docs) / max(1, len(docs))
    k1 = 1.4
    b = 0.75
    scored = []
    for tool, doc in zip(tools, docs):
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / max(1.0, avg_len))
            score += idf * tf[term] * (k1 + 1) / denom
        name_bonus = len(set(query) & set(toks(tool.get("name") or ""))) * 0.75
        scored.append((str(tool.get("name") or ""), score + name_bonus))
    return sorted(scored, key=lambda item: item[1], reverse=True)


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


def predict_names(
    model: Any,
    tokenizer: Any,
    prompt: str,
    tools: list[dict[str, Any]],
    planner_mode: str,
    ensemble_variants: list[str],
    max_new_tokens: int,
    rules_first_missing_input_slot_filler: bool,
    use_semantic_slot_frame: bool,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    capability_plan = run_capability_plan(
        model,
        tokenizer,
        prompt,
        "",
        [],
        max_new_tokens,
        planner_mode,
        rules_first_missing_input_slot_filler,
        "baseline",
        ensemble_variants,
    )
    if use_semantic_slot_frame and model is not None and tokenizer is not None:
        capability_plan = {
            **capability_plan,
            "semantic_input_frame": run_semantic_slot_frame(
                model,
                tokenizer,
                prompt,
                "",
                [],
                capability_plan,
                max_new_tokens,
            ),
        }
    plan = build_tool_binding_plan(prompt, tools, capability_plan=capability_plan)
    predicted = [
        str(call.get("tool_name") or "")
        for call in plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]
    ranked = bm25_scores(prompt, tools)
    best_name, best_score = ranked[0] if ranked else ("", 0.0)
    diagnostics = {
        "task_route": {
            "route": plan.get("tool_decision"),
            "operation": "tool_binding_with_capability_ensemble",
            "source_requirement": "tool_schema",
        },
        "missing_inputs": plan.get("missing_inputs") or [],
        "best_tool": best_name,
        "best_score": round(best_score, 4),
        "ranked_tools": [{"name": name, "score": round(score, 4)} for name, score in ranked[:5]],
        "predicted_count": len(predicted),
        "tool_binding_plan": plan,
    }
    return predicted, diagnostics, capability_plan


def evaluate_file(
    input_path: Path,
    answer_path: Path,
    offset: int,
    limit: int,
    model: Any,
    tokenizer: Any,
    planner_mode: str,
    ensemble_variants: list[str],
    max_new_tokens: int,
    rules_first_missing_input_slot_filler: bool,
    use_semantic_slot_frame: bool,
    plan_handle: Any | None,
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
    for index, record in enumerate(records):
        rid = record_id(record, offset + index)
        answer = answers.get(rid, {"ground_truth": []})
        expected = expected_names(answer)
        if expected:
            expected_call_total += 1
        prompt = prompt_text(record)
        tools = normalize_tools(record)
        predicted, diagnostics, capability_plan = predict_names(
            model,
            tokenizer,
            prompt,
            tools,
            planner_mode,
            ensemble_variants,
            max_new_tokens,
            rules_first_missing_input_slot_filler,
            use_semantic_slot_frame,
        )
        total += 1

        if plan_handle is not None:
            plan_handle.write(
                json.dumps(
                    {
                        "id": rid,
                        "category": input_path.stem.replace("BFCL_v4_", ""),
                        "prompt": prompt,
                        "expected": expected,
                        "predicted": predicted,
                        "capability_plan": capability_plan,
                        "tool_binding_plan": diagnostics["tool_binding_plan"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            plan_handle.flush()

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
                    "best_tool": diagnostics["best_tool"],
                    "best_score": diagnostics["best_score"],
                    "route": diagnostics["task_route"].get("route"),
                    "operation": diagnostics["task_route"].get("operation"),
                    "source_requirement": diagnostics["task_route"].get("source_requirement"),
                    "missing_inputs": json.dumps(diagnostics["missing_inputs"], ensure_ascii=False),
                    "ranked_tools": json.dumps(diagnostics["ranked_tools"], ensure_ascii=False),
                }
            )

    metrics = {
        "category": input_path.stem.replace("BFCL_v4_", ""),
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
        "best_tool",
        "best_score",
        "route",
        "operation",
        "source_requirement",
        "missing_inputs",
        "ranked_tools",
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
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"],
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--plans-jsonl")
    parser.add_argument("--planner-mode", choices=["multi_pass", "chunked", "one_shot", "rules_first", "ensemble"], default="ensemble")
    parser.add_argument("--ensemble-variants", default=",".join(DEFAULT_ENSEMBLE_VARIANTS))
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--rules-first-missing-input-slot-filler", action="store_true")
    parser.add_argument(
        "--semantic-slot-frame",
        action="store_true",
        help="Run one extra GPT-OSS semantic input-frame pass and let the Python binder verify/use it.",
    )
    args = parser.parse_args()

    ensemble_variants = parse_ensemble_variants(args.ensemble_variants)
    needs_model = True
    if args.planner_mode == "rules_first" and not args.rules_first_missing_input_slot_filler:
        needs_model = False
    if args.planner_mode == "ensemble" and not ensemble_requires_model(
        ensemble_variants,
        args.rules_first_missing_input_slot_filler,
    ):
        needs_model = False
    if args.semantic_slot_frame:
        needs_model = True
    model, tokenizer = load_model(args.model) if needs_model else (None, None)

    data_dir = Path(args.data_dir)
    all_metrics = []
    all_review_rows = []
    plan_path = Path(args.plans_jsonl) if args.plans_jsonl else None
    plan_handle = None
    try:
        if plan_path is not None:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_handle = plan_path.open("w", encoding="utf-8")
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
                args.planner_mode,
                ensemble_variants,
                args.max_new_tokens,
                args.rules_first_missing_input_slot_filler,
                args.semantic_slot_frame,
                plan_handle,
            )
            all_metrics.append(result["metrics"])
            for row in result["review_rows"]:
                row["category"] = category
                all_review_rows.append(row)
    finally:
        if plan_handle is not None:
            plan_handle.close()

    total = sum(item["total"] for item in all_metrics)
    aggregate = {
        "total": total,
        "tool_set_accuracy": sum(item["tool_set_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "tool_multiset_accuracy": sum(item["tool_multiset_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "count_accuracy": sum(item["count_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "ordered_sequence_accuracy": sum(item["ordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "unordered_sequence_accuracy": sum(item["unordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
    }
    payload = {
        "aggregate": aggregate,
        "by_category": all_metrics,
        "planner_mode": args.planner_mode,
        "ensemble_variants": ensemble_variants,
        "offset": args.offset,
        "limit": args.limit,
        "model": args.model if needs_model else None,
        "semantic_slot_frame": bool(args.semantic_slot_frame),
    }
    write_json(Path(args.output), payload)
    write_review_csv(Path(args.review_csv), all_review_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
