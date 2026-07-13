#!/usr/bin/env python3
"""Run ToolACE on API-Bank HF test-data API-call rows."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import eval_apibank_toolace_official as base_eval


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("name") or obj.get("apiCode")):
            objects.append(obj)
    return objects


def extract_python_literal_objects(text: str) -> list[dict[str, Any]]:
    objects = []
    for marker in ("->{", "Output: {", "Observation: {"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            brace = text.find("{", idx)
            depth = 0
            quote = ""
            escaped = False
            end = None
            for pos in range(brace, len(text)):
                ch = text[pos]
                if quote:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == quote:
                        quote = ""
                    continue
                if ch in ("'", '"'):
                    quote = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
            if end is None:
                break
            try:
                obj = ast.literal_eval(text[brace:end])
            except Exception:
                start = end
                continue
            if isinstance(obj, dict):
                objects.append(obj)
            start = end
    return objects


def candidate_tool_infos(row: dict[str, Any]) -> list[dict[str, Any]]:
    infos = extract_json_objects(row.get("instruction", "") + "\n" + row.get("input", ""))
    for obj in extract_python_literal_objects(row.get("input", "")):
        if obj.get("name") or obj.get("apiCode"):
            infos.append(obj)
        output = obj.get("output")
        if isinstance(output, dict) and (output.get("name") or output.get("apiCode")):
            infos.append(output)
        elif isinstance(output, list):
            infos.extend(item for item in output if isinstance(item, dict) and (item.get("name") or item.get("apiCode")))
    return infos


def tools_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    seen = set()
    tools = []
    for obj in candidate_tool_infos(row):
        name = obj.get("name") or obj.get("apiCode")
        if not name or name in seen:
            continue
        seen.add(name)
        tools.append(
            {
                "name": name,
                "description": obj.get("description", ""),
                "arguments": obj.get("input_parameters") or obj.get("parameters", {}),
            }
        )
    if not tools:
        tools.append({"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.", "arguments": {"keywords": {"type": "str", "description": "The keyword to search for."}}})
    return tools


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    return base_eval.load_model(args)


def generate_api(model: Any, tokenizer: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    user_prompt = row.get("instruction", "") + "\n" + row.get("input", "")
    messages = [
        {"role": "system", "content": base_eval.TOOLACE_SYSTEM_PROMPT.format(functions=json.dumps(tools_from_row(row), ensure_ascii=False))},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    start = time.time()
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(inputs, **kwargs)
    new_tokens = generated[0][input_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    call = base_eval.parse_first_call(raw_text)
    return {
        "raw_text": raw_text,
        "pred": base_eval.call_to_apibank_text(call, raw_text),
        "latency_ms": round((time.time() - start) * 1000, 3),
        "generated_tokens": int(new_tokens.numel()),
    }


def expected_call(text: str) -> str:
    match = re.search(r"\[[\s\S]*\]", text or "")
    return match.group(0) if match else ""


def score_level3(preds: list[dict[str, Any]], gt_path: Path, api_bank_root: Path, max_details: int) -> dict[str, Any]:
    base_eval.install_optional_dependency_stubs()
    from api_call_extraction import get_api_call, parse_api_call
    from lv3_evaluator import split_by_uppercase
    from tool_manager import ToolManager

    gts = json.loads(gt_path.read_text(encoding="utf-8"))
    tool_manager = ToolManager("./lv3_apis")
    def deterministic_tool_search(keywords: str) -> dict[str, Any]:
        compact = "".join(ch for ch in keywords if ch.isalnum()).lower()
        for api in tool_manager.apis:
            name = api["name"]
            if compact == "".join(ch for ch in name if ch.isalnum()).lower():
                info = {k: v for k, v in api.items() if k not in {"class", "init_database"}}
                return {"api_name": "ToolSearcher", "input": {"keywords": keywords}, "output": info, "exception": None}
        return {"api_name": "ToolSearcher", "input": {"keywords": keywords}, "output": None, "exception": "No exact API-name match"}

    correct = 0
    sample_errors = set()
    errors = {"no_api_call": 0, "parse_error": 0, "api_execution_error": 0, "incorrect_result": 0}
    details = []
    for idx, pred in enumerate(preds):
        sample_id = pred["sample_id"]
        api_id = pred["api_id"]
        gt = gts[sample_id]["apis"][api_id]
        gt_api_name = gt["api_name"]
        api_call = get_api_call(pred["pred"])
        if not api_call:
            errors["no_api_call"] += 1
            sample_errors.add(sample_id)
            details.append({"idx": idx, "error": "no_api_call", "pred": pred.get("pred"), "expected": gt})
            continue
        try:
            pred_api_name, pred_param_dict = parse_api_call(api_call)
        except Exception as exc:
            errors["parse_error"] += 1
            sample_errors.add(sample_id)
            details.append({"idx": idx, "error": f"parse_error: {exc}", "pred": pred.get("pred"), "expected": gt})
            continue
        try:
            if pred_api_name == "ToolSearcher":
                pred_param_dict["keywords"] = split_by_uppercase(pred_param_dict["keywords"])
                pred_result = deterministic_tool_search(pred_param_dict["keywords"])
            else:
                pred_result = tool_manager.api_call(pred_api_name, **pred_param_dict)
        except Exception as exc:
            errors["api_execution_error"] += 1
            sample_errors.add(sample_id)
            details.append({"idx": idx, "error": f"{type(exc).__name__}: {exc}", "pred": pred.get("pred"), "expected": gt})
            continue
        gt_api = tool_manager.init_tool(gt_api_name)
        try:
            is_correct = gt_api.check_api_call_correctness(pred_result, copy.deepcopy(gt["output"]))
        except Exception:
            is_correct = False
        if is_correct:
            correct += 1
        else:
            errors["incorrect_result"] += 1
            sample_errors.add(sample_id)
            details.append({"idx": idx, "error": "incorrect_result", "pred": pred.get("pred"), "result": pred_result, "expected": gt})
    total = len(preds)
    sample_total = len(gts)
    return {
        "paper_metric": "Plan+Retrieve+Call API correctness, using API-Bank lv3_evaluator execution semantics.",
        "paper_ability": "Plan+Retrieve+Call",
        "total_api_calls": total,
        "correct_api_calls": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "sample_total": sample_total,
        "sample_correct": sample_total - len(sample_errors),
        "sample_accuracy": round((sample_total - len(sample_errors)) / sample_total, 6) if sample_total else 0.0,
        "errors": errors,
        "details": details[:max_details],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--level3-gt", default="")
    parser.add_argument("--base", default="Team-ACE/ToolACE-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    base_eval.install_optional_dependency_stubs()
    rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args)

    preds = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            try:
                result = generate_api(model, tokenizer, row, args)
                error = ""
            except Exception as exc:
                result = {"raw_text": "", "pred": "", "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            pred = {
                **{k: row[k] for k in ("file", "id", "sample_id", "api_id") if k in row},
                "pred": result["pred"],
                "raw_text": result["raw_text"],
                "expected_output": row.get("expected_output") or row.get("output", ""),
                "expected_call": expected_call(row.get("expected_output") or row.get("output", "")),
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
                "error": error,
            }
            preds.append(pred)
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"n": idx, "total": len(rows), "pred": pred["pred"], "error": error}, ensure_ascii=False), flush=True)

    if args.level3_gt:
        score = score_level3(preds, Path(args.level3_gt), api_bank_root, args.max_error_details)
        Path(args.score_output).write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: score[k] for k in ("paper_metric", "paper_ability", "total_api_calls", "correct_api_calls", "accuracy", "sample_total", "sample_correct", "sample_accuracy", "errors")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
