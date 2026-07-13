#!/usr/bin/env python3
"""Run the GPT-OSS + ToolACE ensemble on API-Bank level-1/2 API-call rows."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import eval_apibank_testdata_ensemble as ens
import eval_apibank_toolace_official as base


def chat_context(chat_history: list[dict[str, Any]]) -> str:
    return "\n".join(base.role_line(item) for item in chat_history)


def parse_plan(raw_text: str) -> Any:
    for obj in ens.iter_json_objects(raw_text):
        if isinstance(obj, dict):
            return obj
    return raw_text


def sanitize_call(call: dict[str, Any] | None) -> dict[str, Any] | None:
    if not call:
        return None
    args = call.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("arguments"), dict) and isinstance(args.get("name"), str):
        return {"name": args["name"], "arguments": args["arguments"]}
    return call


def plan_with_gptoss(
    bank: ens.GPTOSSAdapterBank,
    tools: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    api_descriptions: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are an API-Bank planning module. Decide the single next API call "
        "needed from the conversation. Return compact JSON only."
    )
    user = {
        "conversation_history": chat_context(chat_history),
        "available_tools": ens.compact_tools(tools),
        "api_descriptions": api_descriptions,
        "rules": [
            "Call exactly one next API.",
            "Use visible API responses for tokens and previous outputs.",
            "If the needed API description is not available and ToolSearcher is available, call ToolSearcher.",
            "This benchmark year is 2023; never use the current system date.",
            "Do not answer the user directly.",
        ],
        "return_schema": {
            "next_subgoal": "short string",
            "preferred_api_names": ["ApiName"],
            "candidate_call": {"name": "ApiName", "arguments": {}},
        },
    }
    result = bank.generate(
        bank.planner_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": ens.short_json(user)}],
        tools=None,
        max_new_tokens=args.max_plan_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result["parsed"] = parse_plan(result["raw_text"])
    result["source"] = "planner"
    return result


def execute_with_gptoss(
    bank: ens.GPTOSSAdapterBank,
    tools: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are an API-Bank tool-calling agent. Choose exactly one next API request. "
        "Return only one machine-readable tool call. This benchmark year is 2023; never use the current system date."
    )
    user = (
        "Conversation history:\n"
        + chat_context(chat_history)
        + "\n\nAvailable tools:\n"
        + json.dumps(ens.compact_tools(tools), ensure_ascii=False)
        + "\n\nPrivate plan:\n"
        + ens.short_json(plan.get("parsed") or plan.get("raw_text", ""))
        + "\n\nReturn one API call now."
    )
    result = bank.generate(
        bank.executor_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=ens.openai_tools(tools),
        max_new_tokens=args.max_action_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    call = sanitize_call(ens.parse_candidate_call(result["raw_text"]))
    result["call"] = call
    result["pred"] = ens.call_to_text(call, result["raw_text"])
    result["source"] = "executor"
    return result


def recover_with_gptoss(
    bank: ens.GPTOSSAdapterBank,
    tools: list[dict[str, Any]],
    chat_history: list[dict[str, Any]],
    plan: dict[str, Any],
    executor: dict[str, Any],
    issues: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are a recovery module for API-Bank tool calls. Repair the candidate "
        "if it is invalid or semantically off. Return JSON only with keys name and arguments. "
        "This benchmark year is 2023; never use the current system date."
    )
    user = {
        "conversation_history": chat_context(chat_history),
        "available_tools": ens.compact_tools(tools),
        "private_plan": plan.get("parsed") or plan.get("raw_text", ""),
        "executor_candidate": {
            "pred": executor.get("pred"),
            "raw_text": executor.get("raw_text"),
            "parsed_call": executor.get("call"),
            "schema_issues": issues,
        },
        "return_schema": {"name": "ApiName", "arguments": {"arg": "value"}},
    }
    result = bank.generate(
        bank.recovery_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": ens.short_json(user)}],
        tools=None,
        max_new_tokens=args.max_recovery_tokens,
        temperature=0.0,
        top_p=args.top_p,
    )
    call = sanitize_call(ens.parse_candidate_call(result["raw_text"]))
    result["call"] = call
    result["pred"] = ens.call_to_text(call, result["raw_text"])
    result["source"] = "recovery"
    return result


def run_gptoss_phase(rows: list[tuple[str, int, Any, Any]], tool_search_enabled: bool, args: argparse.Namespace) -> list[dict[str, Any]]:
    bank = ens.GPTOSSAdapterBank(args)
    records: list[dict[str, Any]] = []
    try:
        for idx, (file_name, sample_id, evaluator, sample_obj) in enumerate(rows, start=1):
            _, chat_history = evaluator.get_model_input(sample_id)
            api_descriptions = base.api_descriptions_for_history(evaluator, chat_history, tool_search_enabled)
            tools = base.toolace_tools(api_descriptions)
            plan = plan_with_gptoss(bank, tools, chat_history, api_descriptions, args)
            executor = execute_with_gptoss(bank, tools, chat_history, plan, args)
            issues = ens.schema_issues(executor.get("call"), tools)
            recovery = None
            if args.recovery_mode == "always" or (args.recovery_mode == "invalid" and issues):
                recovery = recover_with_gptoss(bank, tools, chat_history, plan, executor, issues, args)
            records.append(
                {
                    "row": (file_name, sample_id, evaluator, sample_obj),
                    "tools": tools,
                    "chat_history": chat_history,
                    "api_descriptions": api_descriptions,
                    "plan": plan,
                    "executor": executor,
                    "recovery": recovery,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "gptoss",
                        "n": idx,
                        "total": len(rows),
                        "file": file_name,
                        "id": sample_id,
                        "executor": executor.get("pred"),
                        "recovery": recovery.get("pred") if recovery else None,
                        "issues": issues,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        bank.unload()
    return records


def run_toolace_phase(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.toolace_mode == "off":
        return
    tool_args = argparse.Namespace(
        base=args.toolace_base,
        precision=args.toolace_precision,
        device_map=args.toolace_device_map,
        load_4bit=args.toolace_load_4bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.toolace_max_new_tokens,
        temperature=args.toolace_temperature,
        top_p=args.top_p,
    )
    print(f"[lv12-ensemble] loading ToolACE candidate model: {args.toolace_base}", flush=True)
    model, tokenizer = base.load_model(tool_args)
    try:
        for idx, record in enumerate(records, start=1):
            try:
                result = base.generate_one(model, tokenizer, record["api_descriptions"], record["chat_history"], tool_args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"raw_text": "", "pred": "", "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            call = sanitize_call(ens.parse_candidate_call(result.get("pred") or result.get("raw_text", "")))
            record["toolace"] = {
                "source": "toolace",
                "pred": ens.call_to_text(call, result.get("raw_text", "")),
                "raw_text": result.get("raw_text", ""),
                "call": call,
                "latency_ms": result.get("latency_ms", 0),
                "generated_tokens": result.get("generated_tokens", 0),
                "error": error,
            }
            print(
                json.dumps(
                    {
                        "phase": "toolace",
                        "n": idx,
                        "total": len(records),
                        "pred": record["toolace"]["pred"],
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        import torch

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def summarize_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    return ens.summarize_candidate(candidate)


def score_lv12_candidate(candidate: dict[str, Any], tools: list[dict[str, Any]]) -> tuple[float, list[str]]:
    score, issues = ens.score_candidate(candidate, tools, preferred=set())
    source_bias = {"executor": 0.0, "recovery": 0.4, "toolace": 1.2}
    # Remove the level-3 source prior and apply a level-1/2 prior. ToolACE is
    # the most stable direct caller on API-Bank lv1/lv2; GPT-OSS candidates are
    # still useful as fallbacks when ToolACE is invalid.
    old_bias = {"chronology": 6.0, "executor": 0.55, "recovery": 0.45, "toolace": 0.25}
    source = str(candidate.get("source"))
    score = score - old_bias.get(source, 0.0) + source_bias.get(source, 0.0)
    return score, issues


def choose_lv12_candidate(candidates: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for candidate in candidates:
        score, issues = score_lv12_candidate(candidate, tools)
        candidate["selector_score"] = round(score, 3)
        candidate["selector_issues"] = issues
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else {"source": "none", "pred": "", "raw_text": "", "call": None}


def build_predictions(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    preds: list[dict[str, Any]] = []
    pred_map: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        file_name, sample_id, _evaluator, sample_obj = record["row"]
        candidates = [record["executor"]]
        if record.get("recovery"):
            candidates.append(record["recovery"])
        if record.get("toolace"):
            candidates.append(record["toolace"])
        preferred = ens.preferred_names(record["plan"], record.get("recovery"), record["tools"])
        chosen = choose_lv12_candidate(candidates, record["tools"])
        pred = {
            "file": file_name,
            "id": sample_id,
            "pred": chosen.get("pred", ""),
            "raw_text": chosen.get("raw_text", ""),
            "expected_output": f"[{sample_obj.ground_truth['api_name']}(...)]",
            "expected": sample_obj.ground_truth,
            "latency_ms": round(sum(float((cand or {}).get("latency_ms") or 0) for cand in candidates), 3),
            "generated_tokens": int(sum(int((cand or {}).get("generated_tokens") or 0) for cand in candidates)),
            "error": chosen.get("error", ""),
            "ensemble": {
                "chosen_source": chosen.get("source"),
                "preferred_api_names": sorted(preferred),
                "plan_raw_text": record["plan"].get("raw_text"),
                "plan_parsed": record["plan"].get("parsed"),
                "candidates": [summarize_candidate(candidate) for candidate in candidates],
            },
        }
        preds.append(pred)
        pred_map[(file_name, sample_id)] = pred
    return preds, pred_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--executor-adapter", default=ens.DEFAULT_EXECUTOR)
    parser.add_argument("--planner-adapter", default=ens.DEFAULT_PLANNER)
    parser.add_argument("--recovery-adapter", default=ens.DEFAULT_RECOVERY)
    parser.add_argument("--gptoss-device-map", default="auto")
    parser.add_argument("--no-mxfp4-dequant", action="store_true")
    parser.add_argument("--max-plan-tokens", type=int, default=220)
    parser.add_argument("--max-action-tokens", type=int, default=160)
    parser.add_argument("--max-recovery-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--recovery-mode", choices=("off", "invalid", "always"), default="always")
    parser.add_argument("--toolace-mode", choices=("off", "invalid", "always"), default="always")
    parser.add_argument("--toolace-base", default=ens.DEFAULT_TOOLACE)
    parser.add_argument("--toolace-max-new-tokens", type=int, default=160)
    parser.add_argument("--toolace-temperature", type=float, default=0.0)
    parser.add_argument("--toolace-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--toolace-device-map", default="auto")
    parser.add_argument("--toolace-load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    base.install_optional_dependency_stubs()
    from evaluator_by_json import Evaluator, Sample

    data_dir = api_bank_root / args.data_dir
    tool_search_enabled = not data_dir.name.endswith("given-desc")
    rows = base.iter_samples(data_dir, Sample, Evaluator, limit=args.limit, sample=args.sample, seed=args.seed)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "data_dir": args.data_dir,
                "executor_adapter": args.executor_adapter,
                "planner_adapter": args.planner_adapter,
                "recovery_adapter": args.recovery_adapter,
                "toolace_mode": args.toolace_mode,
                "tool_search_enabled": tool_search_enabled,
            },
            indent=2,
        ),
        flush=True,
    )

    records = run_gptoss_phase(rows, tool_search_enabled, args)
    run_toolace_phase(records, args)
    preds, pred_map = build_predictions(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, pred in enumerate(preds, start=1):
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "phase": "final",
                        "n": idx,
                        "total": len(preds),
                        "source": pred["ensemble"]["chosen_source"],
                        "pred": pred["pred"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    base.args = argparse.Namespace(max_error_details=args.max_error_details)
    paper_ability = (
        "Retrieve+Call (ToolSearcher; API-Bank lv1-lv2-samples/level-2-toolsearcher)"
        if tool_search_enabled
        else "Call (known API descriptions; API-Bank lv1-lv2-samples/level-1-given-desc)"
    )
    score = base.score_predictions(rows, pred_map, paper_ability=paper_ability)
    score_path = Path(args.score_output)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: score[key]
                for key in (
                    "paper_metric",
                    "paper_ability",
                    "response_rouge_l",
                    "total_api_calls",
                    "correct_api_calls",
                    "accuracy",
                    "by_filename_level",
                    "errors",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
