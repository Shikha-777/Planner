#!/usr/bin/env python3
"""Run API-Bank level-1/2 with the current ToolACE+xLAM+APIGen+TaskBench ensemble."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import eval_apibank_testdata_ensemble as ens
import eval_apibank_toolace_official as base
import eval_bfcl_hf_adapter as hf
from eval_bfcl_ensemble_select import canonical_source, first_json, infer_chosen_source


DEFAULT_PROJ = "/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss"
DEFAULT_GPTOSS_BASE = "openai/gpt-oss-20b"
DEFAULT_APIGEN = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024"
DEFAULT_TASKBENCH = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora"
DEFAULT_TOOLACE = "Team-ACE/ToolACE-8B"
DEFAULT_XLAM = "Salesforce/Llama-xLAM-2-8b-fc-r"


API_BANK_SYSTEM_PROMPT = (
    "You are an API-Bank tool-calling agent. Choose exactly one next API request "
    "from the available tools. Return only one machine-readable tool call. Use "
    "exact API names and parameter names. Use values grounded in the conversation "
    "and previous API responses. This benchmark year is 2023; never use the current "
    "system date."
)


def chat_context(chat_history: list[dict[str, Any]]) -> str:
    return "\n".join(base.role_line(item) for item in chat_history)


def build_prompt(api_descriptions: str, chat_history: list[dict[str, Any]]) -> str:
    return (
        base.API_BANK_USER_PROMPT.format(history=chat_context(chat_history))
        + "\n\nAvailable API descriptions for reference:\n"
        + api_descriptions
    )


def sanitize_call(call: dict[str, Any] | None) -> dict[str, Any] | None:
    if not call:
        return None
    args = call.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("arguments"), dict) and isinstance(args.get("name"), str):
        return {"name": args["name"], "arguments": args["arguments"]}
    return call


def normalize_call(call: dict[str, Any] | None, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    return ens.normalize_call_to_schema(sanitize_call(call), tools)


def parse_one_call(text: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    calls = hf.parse_tool_calls(text)
    if calls:
        return normalize_call(calls[0], tools)
    return normalize_call(ens.parse_candidate_call(text), tools)


def make_candidate(
    source: str,
    result: dict[str, Any],
    tools: list[dict[str, Any]],
    error: str = "",
) -> dict[str, Any]:
    raw_text = str(result.get("raw_text") or result.get("pred") or "")
    call = parse_one_call(str(result.get("pred") or raw_text), tools) or parse_one_call(raw_text, tools)
    pred = ens.call_to_text(call, raw_text)
    issues = ens.schema_issues(call, tools)
    return {
        "source": source,
        "pred": pred,
        "raw_text": raw_text,
        "call": call,
        "calls": [call] if call else [],
        "issues": issues,
        "latency_ms": result.get("latency_ms", 0),
        "generated_tokens": result.get("generated_tokens", 0),
        "error": error or result.get("error", ""),
    }


def source_bias(source: str) -> float:
    table = {
        "xlam": 8.0,
        "toolace": 6.0,
        "gptoss_apigen": 5.0,
        "taskbench": 3.0,
        "verifier_repair": 9.0,
    }
    return table.get(source, 0.0)


def static_score(candidate: dict[str, Any]) -> float:
    score = source_bias(str(candidate.get("source")))
    if candidate.get("call"):
        score += 20.0
    score -= 50.0 * len(candidate.get("issues") or [])
    return score


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": candidate.get("source"),
        "pred": candidate.get("pred"),
        "call": candidate.get("call"),
        "schema_issues": candidate.get("issues"),
        "selector_score": candidate.get("selector_score"),
        "error": candidate.get("error", ""),
    }


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    base.install_optional_dependency_stubs()
    from evaluator_by_json import Evaluator, Sample

    data_dir = api_bank_root / args.data_dir
    rows = base.iter_samples(data_dir, Sample, Evaluator, limit=args.limit, sample=args.sample, seed=args.seed)
    tool_search_enabled = not data_dir.name.endswith("given-desc")
    records = []
    for file_name, sample_id, evaluator, sample_obj in rows:
        _, chat_history = evaluator.get_model_input(sample_id)
        api_descriptions = base.api_descriptions_for_history(evaluator, chat_history, tool_search_enabled)
        tools = base.toolace_tools(api_descriptions)
        records.append(
            {
                "file_name": file_name,
                "sample_id": sample_id,
                "evaluator": evaluator,
                "sample_obj": sample_obj,
                "chat_history": chat_history,
                "api_descriptions": api_descriptions,
                "tools": tools,
                "prompt": build_prompt(api_descriptions, chat_history),
            }
        )
    return records


def run_toolace(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.toolace_mode == "off":
        return
    tool_args = argparse.Namespace(
        base=args.toolace_model,
        precision=args.precision if args.precision in {"bf16", "fp16"} else "bf16",
        device_map=args.device_map,
        load_4bit=args.load_4bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[apibank-current] loading ToolACE candidate: {args.toolace_model}", flush=True)
    model, tokenizer = base.load_model(tool_args)
    try:
        for idx, record in enumerate(records, start=1):
            try:
                result = base.generate_one(model, tokenizer, record["api_descriptions"], record["chat_history"], tool_args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"raw_text": "", "pred": "", "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            record.setdefault("candidates", []).append(make_candidate("toolace", result, record["tools"], error))
            print(json.dumps({"phase": "toolace", "n": idx, "total": len(records), "error": error}, ensure_ascii=False), flush=True)
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def hf_args_for(args: argparse.Namespace, model_name: str, adapter: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        base=model_name,
        adapter=adapter,
        tokenizer=adapter or "",
        trust_remote_code=args.trust_remote_code or bool(adapter),
        precision=args.precision,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        openai_tool_schema=False,
        system_prompt=API_BANK_SYSTEM_PROMPT,
    )


def run_hf_candidate(records: list[dict[str, Any]], source: str, model_name: str, adapter: str, args: argparse.Namespace) -> None:
    hf_args = hf_args_for(args, model_name, adapter)
    print(f"[apibank-current] loading {source}: base={model_name} adapter={adapter or '<none>'}", flush=True)
    model, tokenizer = hf.load_model_and_tokenizer(hf_args)
    try:
        for idx, record in enumerate(records, start=1):
            try:
                result = hf.generate_one(model, tokenizer, record["prompt"], ens.openai_tools(record["tools"]), hf_args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"raw_text": "", "normalized_calls": [], "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            if result.get("normalized_calls"):
                result["raw_text"] = json.dumps(result["normalized_calls"][0], ensure_ascii=False)
            record.setdefault("candidates", []).append(make_candidate(source, result, record["tools"], error))
            print(json.dumps({"phase": source, "n": idx, "total": len(records), "error": error}, ensure_ascii=False), flush=True)
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def normalize_repair_call(value: Any, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    value = ens.maybe_json(value)
    if isinstance(value, list):
        value = value[0] if value else None
    call = ens.normalize_one_call(value)
    return normalize_call(call, tools)


class APIBankVerifier:
    def __init__(self, model_name: str, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=hf.resolve_dtype(args.precision, torch),
            device_map=args.device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = args.verifier_max_new_tokens

    def choose(self, record: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str]:
        candidates = record.get("candidates") or []
        allowed_sources = [str(item["source"]) for item in candidates]
        compact = [
            {
                "source": item["source"],
                "call": item.get("call"),
                "schema_issues": item.get("issues") or [],
            }
            for item in candidates
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON-only verifier for completed API-Bank tool-call candidates. "
                    "Do not solve the whole task. Audit the candidates and choose exactly one "
                    "source from allowed_sources, or null if none is usable. Reject candidates "
                    "with schema_issues unless every candidate is invalid. If multiple valid "
                    "candidates are equivalent, choose the first valid source in this priority "
                    "order: xlam, toolace, gptoss_apigen, taskbench. Return exactly one compact "
                    "JSON object and no prose. Required shape: "
                    "{\"chosen_source\":\"<allowed source or null>\",\"repair_call\":null,\"reason\":\"short\"}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "conversation_and_request": record["prompt"],
                        "allowed_sources": allowed_sources,
                        "available_tools_for_validation_only": ens.compact_tools(record["tools"]),
                        "candidate_call_options": compact,
                        "answer_format": {
                            "chosen_source": allowed_sources + [None],
                            "repair_call": None,
                            "reason": "short string",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        json_prefix = '{"chosen_source":'
        inputs = self.tokenizer(rendered_prompt + json_prefix, return_tensors="pt")
        inputs = {key: value.to(next(self.model.parameters()).device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = json_prefix + self.tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        parsed = first_json(text)
        chosen = parsed.get("chosen_source") if isinstance(parsed, dict) else None
        repair = parsed.get("repair_call") if isinstance(parsed, dict) else None
        chosen = canonical_source(chosen, allowed_sources) or infer_chosen_source(text, allowed_sources)
        repair_call = normalize_repair_call(repair, record["tools"]) if repair else None
        return chosen, repair_call, text

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    verifier = None
    if args.verifier_model and args.verifier_model != "none":
        print(f"[apibank-current] loading verifier: {args.verifier_model}", flush=True)
        verifier = APIBankVerifier(args.verifier_model, args)
    try:
        for idx, record in enumerate(records, start=1):
            chosen_source = None
            verifier_raw = ""
            repair_call = None
            if verifier is not None:
                try:
                    chosen_source, repair_call, verifier_raw = verifier.choose(record)
                except Exception as exc:  # noqa: BLE001
                    verifier_raw = f"{type(exc).__name__}: {exc}"
            if repair_call and not ens.schema_issues(repair_call, record["tools"]):
                chosen = make_candidate(
                    "verifier_repair",
                    {"raw_text": verifier_raw, "latency_ms": 0, "generated_tokens": 0},
                    record["tools"],
                )
                chosen["call"] = repair_call
                chosen["calls"] = [repair_call]
                chosen["pred"] = ens.call_to_text(repair_call, verifier_raw)
                chosen["issues"] = []
            else:
                chosen = None
                if chosen_source:
                    for candidate in record.get("candidates") or []:
                        if candidate.get("source") == chosen_source and not candidate.get("issues"):
                            chosen = candidate
                            break
                if chosen is None:
                    ranked = sorted(record.get("candidates") or [], key=static_score, reverse=True)
                    chosen = ranked[0] if ranked else make_candidate("none", {"raw_text": ""}, record["tools"])
            for candidate in record.get("candidates") or []:
                candidate["selector_score"] = round(static_score(candidate), 3)
            record["chosen"] = chosen
            record["verifier"] = {"chosen_source": chosen_source, "raw_text": verifier_raw}
            print(
                json.dumps(
                    {
                        "phase": "select",
                        "n": idx,
                        "total": len(records),
                        "chosen": chosen.get("source"),
                        "verifier_chosen": chosen_source,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if verifier is not None:
            verifier.unload()


def write_and_score(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    preds: list[dict[str, Any]] = []
    pred_map: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    for record in records:
        file_name = record["file_name"]
        sample_id = record["sample_id"]
        sample_obj = record["sample_obj"]
        chosen = record.get("chosen") or {}
        candidates = record.get("candidates") or []
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
                "implementation": "current_toolace_xlam_apigen_taskbench_xlam_verifier",
                "chosen_source": chosen.get("source"),
                "verifier": record.get("verifier"),
                "candidates": [compact_candidate(candidate) for candidate in candidates],
            },
        }
        preds.append(pred)
        pred_map[(file_name, sample_id)] = pred
        rows.append((file_name, sample_id, record["evaluator"], sample_obj))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pred in preds:
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")

    base.args = argparse.Namespace(max_error_details=args.max_error_details)
    paper_ability = (
        "Retrieve+Call (ToolSearcher; API-Bank lv1-lv2-samples/level-2-toolsearcher)"
        if "toolsearcher" in args.data_dir
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--toolace-mode", choices=("off", "always"), default="always")
    parser.add_argument("--toolace-model", default=DEFAULT_TOOLACE)
    parser.add_argument("--xlam-model", default=DEFAULT_XLAM)
    parser.add_argument("--verifier-model", default=DEFAULT_XLAM)
    parser.add_argument("--gptoss-base", default=DEFAULT_GPTOSS_BASE)
    parser.add_argument("--apigen-adapter", default=DEFAULT_APIGEN)
    parser.add_argument("--taskbench-adapter", default=DEFAULT_TASKBENCH)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    start = time.time()
    records = load_rows(args)
    if args.sample:
        records = random.Random(args.seed).sample(records, min(args.sample, len(records)))
    print(
        json.dumps(
            {
                "rows": len(records),
                "data_dir": args.data_dir,
                "implementation": "current_toolace_xlam_apigen_taskbench_xlam_verifier",
                "toolace_model": args.toolace_model,
                "xlam_model": args.xlam_model,
                "verifier_model": args.verifier_model,
                "apigen_adapter": args.apigen_adapter,
                "taskbench_adapter": args.taskbench_adapter,
            },
            indent=2,
        ),
        flush=True,
    )
    run_toolace(records, args)
    run_hf_candidate(records, "xlam", args.xlam_model, "", args)
    run_hf_candidate(records, "gptoss_apigen", args.gptoss_base, args.apigen_adapter, args)
    run_hf_candidate(records, "taskbench", args.gptoss_base, args.taskbench_adapter, args)
    select_records(records, args)
    write_and_score(records, args)
    print(json.dumps({"wall_seconds": round(time.time() - start, 3)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
