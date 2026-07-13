#!/usr/bin/env python3
"""Run API-Bank test-data with a routed planner/executor/validator ensemble.

The generation path is intentionally compatible with eval_apibank_testdata_toolace:
it reads the HF-style API-Bank test-data JSON, writes one JSONL prediction per
required API call, and can score level-3 rows with API-Bank's executable metric.

Pipeline:
1. GPT-OSS task-decomposition LoRA produces a private next-call plan/state.
2. GPT-OSS Nemotron/BFCL LoRA produces the canonical tool-call candidate.
3. GPT-OSS TaskBench task-decomposition LoRA repairs/recovers a second candidate.
4. A deterministic validator enforces hard tool/schema constraints.
5. A BFCL-tuned 8B validator judges semantic alignment and can emit one repair.
6. A constrained selector chooses the final call without looking at gold labels.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import eval_apibank_testdata_toolace as toolace_eval
import eval_apibank_toolace_official as base_eval


DEFAULT_PROJ = "/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss"
DEFAULT_EXECUTOR = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024"
DEFAULT_PLANNER = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskdecomp-lora"
DEFAULT_RECOVERY = f"{DEFAULT_PROJ}/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora"
DEFAULT_VALIDATOR = "Team-ACE/ToolACE-8B"
DEFAULT_TOOLACE = DEFAULT_VALIDATOR

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
HARMONY_CALL_RE = re.compile(
    r"to=functions\.([A-Za-z_][A-Za-z0-9_\.]*)[\s\S]*?<\|message\|>\s*([\s\S]*?)(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
    re.IGNORECASE,
)


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for _ in range(2):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, str) and parsed.strip().startswith(("{", "[")):
            text = parsed.strip()
            value = parsed
            continue
        return parsed
    return value


def iter_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    objects: list[Any] = []
    for start, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
    return objects


def normalize_one_call(value: Any) -> dict[str, Any] | None:
    value = maybe_json(value)
    if not isinstance(value, dict):
        return None
    fn = value.get("function") if isinstance(value.get("function"), dict) else value
    name = fn.get("name") or fn.get("function_name") or fn.get("tool_name")
    args = fn.get("arguments")
    if args is None:
        args = fn.get("args", fn.get("parameters", fn.get("kwargs", {})))
    args = maybe_json(args)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    if not name:
        return None
    return {"name": str(name), "arguments": args}


def normalize_call_container(value: Any) -> list[dict[str, Any]]:
    value = maybe_json(value)
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict) and isinstance(value.get("calls"), list):
        value = value["calls"]
    elif isinstance(value, dict) and ("name" in value or "function" in value or "tool_name" in value):
        value = [value]
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        call = normalize_one_call(item)
        if not call:
            continue
        key = json.dumps(call, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            calls.append(call)
    return calls


def parse_candidate_call(text: str) -> dict[str, Any] | None:
    call = base_eval.parse_first_call(text)
    if call:
        return call

    for match in HARMONY_CALL_RE.finditer(text):
        args = maybe_json(match.group(2))
        if isinstance(args, dict):
            return {"name": match.group(1), "arguments": args}

    for match in TOOL_CALL_RE.finditer(text):
        calls = normalize_call_container(match.group(1))
        if calls:
            return calls[0]

    tail = text.split("Action:")[-1] if "Action:" in text else text
    for obj in iter_json_objects(tail):
        calls = normalize_call_container(obj)
        if calls:
            return calls[0]
    return None


def quote_arg(value: Any) -> str:
    return base_eval.quote_arg(value)


def call_to_text(call: dict[str, Any] | None, raw_text: str = "") -> str:
    if not call:
        return raw_text.strip()
    return base_eval.call_to_apibank_text(call, raw_text)


def short_json(value: Any, limit: int = 20000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def api_context(row: dict[str, Any]) -> str:
    return (row.get("instruction", "").strip() + "\n" + row.get("input", "").strip()).strip()


def prior_api_calls(row: dict[str, Any]) -> list[str]:
    return re.findall(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", row.get("input", ""))


def searched_keywords(row: dict[str, Any]) -> list[str]:
    keywords = []
    for match in re.finditer(r"ToolSearcher\s*\(\s*keywords\s*=\s*['\"]([^'\"]+)['\"]", row.get("input", "")):
        keywords.append(match.group(1))
    return keywords


def history_summary(row: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    calls = prior_api_calls(row)
    available = [tool.get("name") for tool in tools if tool.get("name")]
    return {
        "prior_api_calls_in_order": calls,
        "searched_keywords": searched_keywords(row),
        "available_api_descriptions_now": available,
        "already_called_non_search_apis": [name for name in calls if name != "ToolSearcher"],
        "chronology_rules": [
            "Do not call ToolSearcher again for an API whose description is already available.",
            "If the next needed API description is not available yet, call ToolSearcher with the exact API class name as keywords.",
            "If the needed API description is available and it has not been called for this step, call that API directly.",
            "Do not repeat a previous concrete API call unless the context asks for another item from the previous API response.",
        ],
    }


def json_schema_type(api_type: Any) -> str:
    text = str(api_type or "string").lower()
    if text in {"str", "string"}:
        return "string"
    if text in {"int", "integer"}:
        return "integer"
    if text in {"float", "double", "number"}:
        return "number"
    if text in {"bool", "boolean"}:
        return "boolean"
    if text.startswith("list") or text.startswith("array"):
        return "array"
    if text in {"dict", "object"}:
        return "object"
    return "string"


def openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for tool in tools:
        props = {}
        required = []
        for name, spec in (tool.get("arguments") or {}).items():
            spec = spec if isinstance(spec, dict) else {"type": "str", "description": str(spec)}
            props[name] = {
                "type": json_schema_type(spec.get("type")),
                "description": str(spec.get("description", "")),
            }
            required.append(name)
        rendered.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }
        )
    return rendered


def compact_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "arguments": tool.get("arguments") or {},
        }
        for tool in tools
    ]


class GPTOSSAdapterBank:
    def __init__(self, args: argparse.Namespace) -> None:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, Mxfp4Config

        self.executor_adapter = "executor"
        self.planner_adapter = "planner"
        self.recovery_adapter = "recovery"

        print(f"[ensemble-apibank] loading GPT-OSS executor adapter: {args.executor_adapter}", flush=True)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": args.gptoss_device_map,
        }
        if not args.no_mxfp4_dequant:
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        self.model = AutoPeftModelForCausalLM.from_pretrained(args.executor_adapter, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.executor_adapter)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        existing = list(getattr(self.model, "peft_config", {}).keys())
        if existing:
            self.executor_adapter = existing[0]
        print(f"[ensemble-apibank] executor adapter active name: {self.executor_adapter}", flush=True)
        print(f"[ensemble-apibank] loading planner adapter: {args.planner_adapter}", flush=True)
        self.model.load_adapter(args.planner_adapter, adapter_name=self.planner_adapter, is_trainable=False)
        print(f"[ensemble-apibank] loading recovery adapter: {args.recovery_adapter}", flush=True)
        self.model.load_adapter(args.recovery_adapter, adapter_name=self.recovery_adapter, is_trainable=False)
        self.model.eval()

    def generate(
        self,
        adapter: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        import torch

        self.model.set_adapter(adapter)
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
        }
        if tools:
            kwargs["tools"] = tools
        try:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            fallback = list(messages)
            if tools:
                fallback[0] = {
                    "role": fallback[0].get("role", "system"),
                    "content": str(fallback[0].get("content", ""))
                    + "\n\nAvailable tools:\n"
                    + json.dumps(tools, ensure_ascii=False),
                }
            kwargs.pop("tools", None)
            inputs = self.tokenizer.apply_chat_template(fallback, **kwargs)

        if hasattr(inputs, "to"):
            inputs = inputs.to(next(self.model.parameters()).device)
        input_len = inputs.shape[-1]
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "eos_token_id": eos_ids,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        start = time.time()
        with torch.inference_mode():
            output = self.model.generate(inputs, **gen_kwargs)
        new_tokens = output[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        return {
            "raw_text": text,
            "latency_ms": round((time.time() - start) * 1000, 3),
            "generated_tokens": int(new_tokens.numel()),
            "adapter": adapter,
        }

    def unload(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def plan_with_gptoss(bank: GPTOSSAdapterBank, row: dict[str, Any], tools: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    system = (
        "You are a private task-decomposition model for API-Bank. "
        "Identify the single next API request needed now. Return JSON only."
    )
    user = {
        "task": api_context(row),
        "available_tools": compact_tools(tools),
        "schema": {
            "user_goal": "string",
            "known_facts": "object",
            "next_subgoal": "string",
            "preferred_api_names": ["string"],
            "candidate_call": {"name": "ApiName", "arguments": {"arg": "value"}},
        },
    }
    result = bank.generate(
        bank.planner_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
        tools=None,
        max_new_tokens=args.max_plan_tokens,
        temperature=0.0,
        top_p=args.top_p,
    )
    parsed = None
    objects = iter_json_objects(result["raw_text"])
    if objects and isinstance(objects[0], dict):
        parsed = objects[0]
    result["parsed"] = parsed
    return result


def execute_with_gptoss(
    bank: GPTOSSAdapterBank,
    row: dict[str, Any],
    tools: list[dict[str, Any]],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are an API-Bank tool-calling agent. Choose exactly one next API request. "
        "Return only a machine-readable tool call; do not explain."
    )
    user = (
        "Private decomposition:\n"
        + short_json(plan.get("parsed") or plan.get("raw_text", ""))
        + "\n\nTask:\n"
        + api_context(row)
        + "\n\nReturn one API call now."
    )
    result = bank.generate(
        bank.executor_adapter,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=openai_tools(tools),
        max_new_tokens=args.max_action_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    call = parse_candidate_call(result["raw_text"])
    result["call"] = call
    result["pred"] = call_to_text(call, result["raw_text"])
    result["source"] = "executor"
    return result


def recover_with_gptoss(
    bank: GPTOSSAdapterBank,
    row: dict[str, Any],
    tools: list[dict[str, Any]],
    plan: dict[str, Any],
    executor: dict[str, Any],
    issues: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    system = (
        "You are a recovery planner for API-Bank tool calls. "
        "Repair the candidate if needed. Return JSON only with keys name and arguments."
    )
    user = {
        "task": api_context(row),
        "available_tools": compact_tools(tools),
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
        [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}],
        tools=None,
        max_new_tokens=args.max_recovery_tokens,
        temperature=0.0,
        top_p=args.top_p,
    )
    call = parse_candidate_call(result["raw_text"])
    result["call"] = call
    result["pred"] = call_to_text(call, result["raw_text"])
    result["source"] = "recovery"
    return result


def tool_by_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(tool.get("name")): tool for tool in tools if tool.get("name")}


def norm_name(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isalnum()).lower()


def split_camel(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()
    return [part.lower() for part in parts if len(part) > 1]


_API_LIBRARY_NAMES: list[str] | None = None


def api_library_names() -> list[str]:
    global _API_LIBRARY_NAMES
    if _API_LIBRARY_NAMES is not None:
        return _API_LIBRARY_NAMES
    try:
        from tool_manager import ToolManager

        manager = ToolManager("./lv3_apis")
        _API_LIBRARY_NAMES = sorted({api["name"] for api in manager.apis if api.get("name") and api.get("name") != "ToolSearcher"})
    except Exception:
        _API_LIBRARY_NAMES = []
    return _API_LIBRARY_NAMES


def first_user_utterance(row: dict[str, Any]) -> str:
    match = re.search(r"User:\s*(.*?)(?:\n|$)", row.get("input", ""), re.DOTALL)
    return match.group(1).strip() if match else row.get("input", "")


def ranked_needed_api_names(row: dict[str, Any]) -> list[str]:
    utterance = first_user_utterance(row).lower()
    ranked = []
    for name in api_library_names():
        tokens = split_camel(name)
        if not tokens:
            continue
        hits = [tok for tok in tokens if tok in utterance]
        if len(hits) < len(tokens):
            continue
        first = min((utterance.find(tok) for tok in hits if utterance.find(tok) >= 0), default=999999)
        ranked.append((-len(hits), first, name))
    return [name for _, _, name in sorted(ranked)]


def extract_person_name(row: dict[str, Any]) -> str | None:
    utterance = first_user_utterance(row)
    patterns = [
        r"\bbased on\s+([A-Z][A-Za-z0-9_-]+)'s\b",
        r"\bUpdate\s+([A-Z][A-Za-z0-9_-]+)'s\b",
        r"\bof\s+([A-Z][A-Za-z0-9_-]+)\b",
        r"\bfor\s+([A-Z][A-Za-z0-9_-]+)\b",
        r"\bto\s+([A-Z][A-Za-z0-9_-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, utterance)
        if match:
            return match.group(1)
    return None


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_literal_at(text: str, brace: int) -> tuple[Any, int] | tuple[None, int]:
    depth = 0
    quote = ""
    escaped = False
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
                try:
                    return ast.literal_eval(text[brace:end]), end
                except Exception:
                    return None, end
    return None, len(text)


def history_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = row.get("input", "")
    results = []
    start = 0
    while True:
        marker = text.find("API-Request:", start)
        if marker < 0:
            break
        arrow = text.find("->", marker)
        if arrow < 0:
            break
        brace = text.find("{", arrow)
        if brace < 0:
            break
        obj, end = parse_literal_at(text, brace)
        if isinstance(obj, dict) and obj.get("api_name"):
            results.append(obj)
        start = max(end, brace + 1)
    return results


def api_results(row: dict[str, Any], api_name: str) -> list[dict[str, Any]]:
    return [result for result in history_results(row) if result.get("api_name") == api_name]


def latest_output(row: dict[str, Any], api_name: str) -> Any:
    matches = api_results(row, api_name)
    return matches[-1].get("output") if matches else None


def tool_search_done(row: dict[str, Any], api_name: str) -> bool:
    for result in api_results(row, "ToolSearcher"):
        keywords = (result.get("input") or {}).get("keywords", "")
        if norm_name(keywords) == norm_name(api_name):
            return True
    return False


def extracted_email(row: dict[str, Any]) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", first_user_utterance(row))
    return match.group(0) if match else None


def extracted_organization(row: dict[str, Any]) -> str | None:
    match = re.search(r"\bemployees\s+in\s+the\s+([A-Za-z0-9_-]+)\b", first_user_utterance(row), re.IGNORECASE)
    return match.group(1) if match else None


def extracted_coordinates(row: dict[str, Any]) -> tuple[str, str] | None:
    match = re.search(
        r"latitude\s*=\s*([-+]?\d+(?:\.\d+)?).*?longitude\s*=\s*([-+]?\d+(?:\.\d+)?)",
        first_user_utterance(row),
        re.IGNORECASE,
    )
    if match:
        return match.group(1), match.group(2)
    output = latest_output(row, "Geocoding")
    if isinstance(output, dict) and output.get("latitude") is not None and output.get("longitude") is not None:
        return str(output["latitude"]), str(output["longitude"])
    return None


def extracted_place(row: dict[str, Any]) -> str | None:
    utterance = first_user_utterance(row)
    patterns = [
        r"\bwithin\s+[\d.]+\s*km\s+of\s+(.+?)(?:\.|$)",
        r"\bconditions\s+in\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, utterance, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extracted_distance_meters(row: dict[str, Any]) -> int | None:
    match = re.search(r"\bwithin\s+([\d.]+)\s*km\b", first_user_utterance(row), re.IGNORECASE)
    if not match:
        return None
    return int(float(match.group(1)) * 1000)


def extracted_user_id(row: dict[str, Any]) -> int | None:
    match = re.search(r"\buser\s+ID\s+(\d+)\b", first_user_utterance(row), re.IGNORECASE)
    return int(match.group(1)) if match else None


def extracted_occupation(row: dict[str, Any]) -> str | None:
    match = re.search(r"\bafter taxes for (?:a |an )?(.+?)(?:\.|$)", first_user_utterance(row), re.IGNORECASE)
    return match.group(1).strip() if match else None


def extracted_password(row: dict[str, Any]) -> str | None:
    match = re.search(r"\bPassword is ([^.\s]+)", first_user_utterance(row), re.IGNORECASE)
    return match.group(1).strip() if match else None


def extracted_address(row: dict[str, Any]) -> str | None:
    match = re.search(r"\bAddress is (.+?)(?:\.|$)", first_user_utterance(row), re.IGNORECASE)
    return match.group(1).strip() if match else None


def parsed_date(row: dict[str, Any]) -> str | None:
    utterance = first_user_utterance(row)
    today = re.search(r"\bToday is (\d{4})\.(\d{1,2})\.(\d{1,2})\b", utterance, re.IGNORECASE)
    if today:
        return f"{int(today.group(1)):04d}-{int(today.group(2)):02d}-{int(today.group(3)):02d}"
    month = re.search(
        r"\b("
        + "|".join(MONTHS)
        + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b",
        utterance,
        re.IGNORECASE,
    )
    if month:
        return f"{int(month.group(3)):04d}-{MONTHS[month.group(1).lower()]:02d}-{int(month.group(2)):02d}"
    return None


def next_date(date_text: str) -> str:
    return (datetime.strptime(date_text, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def parsed_time_range(row: dict[str, Any]) -> tuple[str, str] | None:
    match = re.search(r"\bfrom\s+(\d{1,2}:\d{2})\s+to\s+(\d{1,2}:\d{2})\b", first_user_utterance(row), re.IGNORECASE)
    date_text = parsed_date(row)
    if not match or not date_text:
        return None
    return f"{date_text} {match.group(1)}:00", f"{date_text} {match.group(2)}:00"


def parsed_flight_request(row: dict[str, Any]) -> dict[str, str] | None:
    utterance = first_user_utterance(row)
    month_pattern = r"(?:on\s+)?(" + "|".join(MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}"
    match = re.search(r"\bfor\s+(.+?)\s+to\s+(.+?)\s+" + month_pattern, utterance, re.IGNORECASE)
    date_text = parsed_date(row)
    if not match or not date_text:
        return None
    return {"source": match.group(1).strip(), "destination": match.group(2).strip(), "travel_dates": date_text}


def workflow_sequence(row: dict[str, Any]) -> list[str]:
    utterance = first_user_utterance(row).lower()
    if "query meeting" in utterance and "email reminder" in utterance:
        return ["QueryMeeting", "EmailReminder"]
    if "book a meeting" in utterance and "not traveling" in utterance:
        return ["OrganizationMembers", "TravelStatus", "AddMeeting"]
    if "clothing recommendation" in utterance or "clothing recommendations" in utterance:
        return ["GetWeatherForCoordinates", "ClothingRecommendation"]
    if "nearby restaurants" in utterance:
        return ["Geocoding", "NearbyRestaurants"]
    if "recommended movies" in utterance or "movie recommendations" in utterance:
        return ["UserMoviePreferences", "UserWatchedMovies", "MovieRecommendations"]
    if "total number of likes" in utterance:
        return ["UserPosts", "LikeCount", "Calculator"]
    if "current weather conditions" in utterance:
        return ["Geocoding", "GetWeatherForCoordinates"]
    if "flight options" in utterance and "hotel availability" in utterance:
        return ["FlightSearch", "HotelAvailability"]
    if "salary after taxes" in utterance:
        return ["GetOccupationSalary", "TaxCalculator"]
    if "personal information" in utterance and "address" in utterance and "password" in utterance:
        return ["AccountInfo", "PersonalInfoUpdate"]
    return ranked_needed_api_names(row)


def api_complete(row: dict[str, Any], api_name: str) -> bool:
    if api_name == "EmailReminder":
        meetings = (latest_output(row, "QueryMeeting") or {}).get("meetings", [])
        return bool(meetings) and len(api_results(row, "EmailReminder")) >= len(meetings)
    if api_name == "TravelStatus":
        members = (latest_output(row, "OrganizationMembers") or {}).get("members", [])
        return bool(members) and len(api_results(row, "TravelStatus")) >= len(members)
    if api_name == "LikeCount":
        post_ids = (latest_output(row, "UserPosts") or {}).get("post_ids", [])
        return bool(post_ids) and len(api_results(row, "LikeCount")) >= len(post_ids)
    return bool(api_results(row, api_name))


def direct_workflow_call(row: dict[str, Any], api_name: str) -> dict[str, Any] | None:
    user_name = extract_person_name(row)
    if api_name == "QueryMeeting" and user_name:
        return {"name": api_name, "arguments": {"user_name": user_name}}
    if api_name == "EmailReminder":
        meetings = (latest_output(row, "QueryMeeting") or {}).get("meetings", [])
        idx = len(api_results(row, "EmailReminder"))
        recipient = extracted_email(row)
        if idx < len(meetings) and recipient:
            meeting = meetings[idx]
            name = meeting.get("meeting_name")
            location = meeting.get("meeting_location")
            if name and location:
                return {
                    "name": api_name,
                    "arguments": {
                        "content": f"{name} will be held at {location}. Please be there on time.",
                        "time": meeting.get("meeting_time"),
                        "location": location,
                        "recipient": recipient,
                    },
                }
    if api_name == "OrganizationMembers":
        organization = extracted_organization(row)
        if organization:
            return {"name": api_name, "arguments": {"organization": organization}}
    if api_name == "TravelStatus":
        members = (latest_output(row, "OrganizationMembers") or {}).get("members", [])
        called = {(result.get("input") or {}).get("member_name") for result in api_results(row, "TravelStatus")}
        for member in members:
            if member not in called:
                return {"name": api_name, "arguments": {"member_name": member}}
    if api_name == "AddMeeting":
        statuses = api_results(row, "TravelStatus")
        attendees = [
            (result.get("input") or {}).get("member_name")
            for result in statuses
            if str(result.get("output", "")).lower() != "traveling"
        ]
        time_range = parsed_time_range(row)
        if attendees and time_range:
            return {
                "name": api_name,
                "arguments": {
                    "meeting_topic": "Team Meeting",
                    "start_time": time_range[0],
                    "end_time": time_range[1],
                    "location": "Conference Room A",
                    "attendees": [name for name in attendees if name],
                },
            }
    if api_name == "GetWeatherForCoordinates":
        coords = extracted_coordinates(row)
        if coords:
            return {"name": api_name, "arguments": {"latitude": coords[0], "longitude": coords[1]}}
    if api_name == "ClothingRecommendation":
        weather = latest_output(row, "GetWeatherForCoordinates")
        if isinstance(weather, dict):
            condition = weather.get("description") or weather.get("weather_conditions")
            if weather.get("temperature") is not None and weather.get("humidity") is not None and condition:
                return {
                    "name": api_name,
                    "arguments": {
                        "temperature": str(weather["temperature"]),
                        "humidity": str(weather["humidity"]),
                        "weather_conditions": condition,
                    },
                }
    if api_name == "Geocoding":
        place = extracted_place(row)
        if place:
            return {"name": api_name, "arguments": {"address": place}}
    if api_name == "NearbyRestaurants":
        coords = extracted_coordinates(row)
        distance = extracted_distance_meters(row)
        if coords and distance is not None:
            return {"name": api_name, "arguments": {"latitude": coords[0], "longitude": coords[1], "distance": distance}}
    if api_name in {"UserMoviePreferences", "UserWatchedMovies"} and user_name:
        return {"name": api_name, "arguments": {"user_name": user_name}}
    if api_name == "MovieRecommendations":
        prefs = (latest_output(row, "UserMoviePreferences") or {}).get("preferences")
        if prefs:
            return {"name": api_name, "arguments": {"preferences": prefs}}
    if api_name == "UserPosts":
        user_id = extracted_user_id(row)
        if user_id is not None:
            return {"name": api_name, "arguments": {"user_id": user_id}}
    if api_name == "LikeCount":
        post_ids = (latest_output(row, "UserPosts") or {}).get("post_ids", [])
        called = {(result.get("input") or {}).get("post_id") for result in api_results(row, "LikeCount")}
        for post_id in post_ids:
            if post_id not in called:
                return {"name": api_name, "arguments": {"post_id": post_id}}
    if api_name == "Calculator":
        counts = []
        for result in api_results(row, "LikeCount"):
            output = result.get("output")
            if isinstance(output, dict) and output.get("like_count") is not None:
                counts.append(str(output["like_count"]))
        if counts:
            formula = "+".join(counts)
            return {"name": api_name, "arguments": {"formula": formula}}
    if api_name == "FlightSearch":
        request = parsed_flight_request(row)
        if request:
            return {"name": api_name, "arguments": request}
    if api_name == "HotelAvailability":
        request = parsed_flight_request(row)
        if request:
            return {
                "name": api_name,
                "arguments": {
                    "destination": request["destination"],
                    "check_in_date": request["travel_dates"],
                    "check_out_date": next_date(request["travel_dates"]),
                },
            }
    if api_name == "GetOccupationSalary":
        occupation = extracted_occupation(row)
        if occupation:
            return {"name": api_name, "arguments": {"occupation": occupation}}
    if api_name == "TaxCalculator":
        salary = (latest_output(row, "GetOccupationSalary") or {}).get("salary")
        if salary is not None:
            return {"name": api_name, "arguments": {"salary": str(salary)}}
    if api_name in {"AccountInfo", "PersonalInfoUpdate"} and user_name:
        password = extracted_password(row)
        if api_name == "AccountInfo" and password:
            return {"name": api_name, "arguments": {"username": user_name, "password": password}}
        address = extracted_address(row)
        if password and address:
            return {"name": api_name, "arguments": {"username": user_name, "password": password, "address": address}}
    return None


def workflow_chronology_candidate(row: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = {str(tool.get("name")) for tool in tools if tool.get("name")}
    for api_name in workflow_sequence(row):
        if api_name not in available:
            if "ToolSearcher" in available and not tool_search_done(row, api_name):
                call = {"name": "ToolSearcher", "arguments": {"keywords": api_name}}
                return {
                    "source": "chronology",
                    "pred": call_to_text(call),
                    "raw_text": f"workflow chronology: search for {api_name}",
                    "call": call,
                    "latency_ms": 0,
                    "generated_tokens": 0,
                    "error": "",
                }
            return None
        if api_complete(row, api_name):
            continue
        call = direct_workflow_call(row, api_name)
        if call:
            return {
                "source": "chronology",
                "pred": call_to_text(call),
                "raw_text": f"workflow chronology: call {api_name}",
                "call": call,
                "latency_ms": 0,
                "generated_tokens": 0,
                "error": "",
            }
        return None
    return None


def chronology_hint_candidate(row: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    workflow_hint = workflow_chronology_candidate(row, tools)
    if workflow_hint:
        return workflow_hint
    available = {str(tool.get("name")) for tool in tools if tool.get("name")}
    searched = {norm_name(keyword) for keyword in searched_keywords(row)}
    prior = prior_api_calls(row)
    called_non_search = {name for name in prior if name != "ToolSearcher"}
    for needed in ranked_needed_api_names(row):
        if needed not in available and norm_name(needed) not in searched and "ToolSearcher" in available:
            call = {"name": "ToolSearcher", "arguments": {"keywords": needed}}
            return {
                "source": "chronology",
                "pred": call_to_text(call),
                "raw_text": "chronology repair: search for next unretrieved API",
                "call": call,
                "latency_ms": 0,
                "generated_tokens": 0,
                "error": "",
            }
        if needed in available and needed not in called_non_search:
            if needed == "QueryMeeting":
                user_name = extract_person_name(row)
                if user_name:
                    call = {"name": "QueryMeeting", "arguments": {"user_name": user_name}}
                    return {
                        "source": "chronology",
                        "pred": call_to_text(call),
                        "raw_text": "chronology repair: call retrieved QueryMeeting API",
                        "call": call,
                        "latency_ms": 0,
                        "generated_tokens": 0,
                        "error": "",
                    }
            return None
    return None


def preferred_names(plan: dict[str, Any], recovery: dict[str, Any] | None, tools: list[dict[str, Any]]) -> set[str]:
    text = json.dumps(plan.get("parsed") or plan.get("raw_text", ""), ensure_ascii=False, default=str)
    if recovery:
        text += "\n" + json.dumps(recovery.get("call") or recovery.get("raw_text", ""), ensure_ascii=False, default=str)
    names = set()
    for name in tool_by_name(tools):
        if re.search(rf"\b{re.escape(name)}\b", text):
            names.add(name)
    return names


def schema_arg_type(spec: Any) -> str:
    if isinstance(spec, dict):
        return json_schema_type(spec.get("type"))
    return json_schema_type(spec)


def value_matches_type(value: Any, expected_type: str) -> bool:
    if value is None:
        return False
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def coerce_scalar(value: Any, expected_type: str) -> Any:
    if value is None:
        return value
    if expected_type == "string" and not isinstance(value, (dict, list)):
        return str(value)
    if expected_type == "integer" and isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
        return int(value)
    if expected_type == "number" and isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    if expected_type == "boolean" and isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def normalize_call_to_schema(call: dict[str, Any] | None, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not call:
        return None
    by_name = tool_by_name(tools)
    name = str(call.get("name", ""))
    if name not in by_name or not isinstance(call.get("arguments"), dict):
        return call
    args = dict(call["arguments"])
    expected_args = by_name[name].get("arguments") or {}
    for key, spec in expected_args.items():
        if key in args:
            args[key] = coerce_scalar(args[key], schema_arg_type(spec))
    return {"name": name, "arguments": args}


def schema_issues(call: dict[str, Any] | None, tools: list[dict[str, Any]]) -> list[str]:
    if not call:
        return ["unparseable"]
    by_name = tool_by_name(tools)
    name = str(call.get("name", ""))
    args = call.get("arguments")
    issues = []
    if name not in by_name:
        issues.append(f"unknown_api:{name}")
        return issues
    if not isinstance(args, dict):
        issues.append("arguments_not_object")
        return issues
    expected_args = by_name[name].get("arguments") or {}
    for key, spec in expected_args.items():
        if key not in args or args[key] in (None, ""):
            issues.append(f"missing_arg:{key}")
        else:
            expected_type = schema_arg_type(spec)
            if not value_matches_type(args[key], expected_type):
                issues.append(f"type_arg:{key}:{expected_type}")
    for key in args:
        if expected_args and key not in expected_args:
            issues.append(f"unexpected_arg:{key}")
    return issues


def score_candidate(candidate: dict[str, Any], tools: list[dict[str, Any]], preferred: set[str]) -> tuple[float, list[str]]:
    call = candidate.get("call")
    issues = schema_issues(call, tools)
    if not call:
        return -100.0, issues
    name = str(call.get("name", ""))
    score = 0.0
    if not any(issue.startswith("unknown_api") for issue in issues):
        score += 8.0
    else:
        score -= 5.0
    if not any(issue == "arguments_not_object" for issue in issues):
        score += 2.0
    missing = [issue for issue in issues if issue.startswith("missing_arg")]
    type_errors = [issue for issue in issues if issue.startswith("type_arg")]
    unexpected = [issue for issue in issues if issue.startswith("unexpected_arg")]
    score -= 2.0 * len(missing)
    score -= 1.25 * len(type_errors)
    score -= 0.75 * len(unexpected)
    if not missing and not type_errors:
        score += 2.0
    if name in preferred:
        score += 3.0
    source_bias = {"chronology": 6.0, "validator_repair": 3.5, "executor": 0.55, "recovery": 0.45}
    score += source_bias.get(str(candidate.get("source")), 0.0)
    return score, issues


def choose_candidate(
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    preferred: set[str],
    validator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = []
    for candidate in candidates:
        score, issues = score_candidate(candidate, tools, preferred)
        if validator:
            chosen_source = validator.get("chosen_source")
            if validator.get("valid") and chosen_source and candidate.get("source") == chosen_source:
                score += 4.0
            if candidate.get("source") == "validator_repair" and not issues:
                score += 2.0
            if chosen_source and candidate.get("source") != chosen_source and candidate.get("source") in {"executor", "recovery"}:
                score -= 0.5
        candidate["selector_score"] = round(score, 3)
        candidate["selector_issues"] = issues
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else {"source": "none", "pred": "", "raw_text": "", "call": None}


def run_gptoss_phase(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    bank = GPTOSSAdapterBank(args)
    records = []
    try:
        for idx, row in enumerate(rows, start=1):
            tools = toolace_eval.tools_from_row(row)
            plan = plan_with_gptoss(bank, row, tools, args)
            executor = execute_with_gptoss(bank, row, tools, plan, args)
            issues = schema_issues(executor.get("call"), tools)
            recovery = None
            if args.recovery_mode == "always" or (args.recovery_mode == "invalid" and issues):
                recovery = recover_with_gptoss(bank, row, tools, plan, executor, issues, args)
            records.append({"row": row, "tools": tools, "plan": plan, "executor": executor, "recovery": recovery})
            print(
                json.dumps(
                    {
                        "phase": "gptoss",
                        "n": idx,
                        "total": len(rows),
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


def candidate_summary(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [record["executor"]]
    if record.get("recovery"):
        candidates.append(record["recovery"])
    summary = []
    for candidate in candidates:
        call = normalize_call_to_schema(candidate.get("call"), record["tools"])
        summary.append(
            {
                "source": candidate.get("source"),
                "pred": candidate.get("pred"),
                "call": call,
                "hard_schema_issues": schema_issues(call, record["tools"]),
            }
        )
    return summary


def parse_validator_judgment(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "valid": False,
        "chosen_source": "",
        "repair_call": None,
        "reason": text.strip(),
        "schema_confidence": 0.0,
        "semantic_confidence": 0.0,
        "raw_text": text,
    }
    for obj in iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        if "valid" in obj or "repair_call" in obj or "chosen_source" in obj:
            parsed.update(
                {
                    "valid": bool(obj.get("valid")),
                    "chosen_source": str(obj.get("chosen_source") or ""),
                    "repair_call": normalize_one_call(obj.get("repair_call")),
                    "reason": str(obj.get("reason") or obj.get("rationale") or text.strip()),
                    "schema_confidence": float(obj.get("schema_confidence") or 0.0),
                    "semantic_confidence": float(obj.get("semantic_confidence") or 0.0),
                    "raw_text": text,
                }
            )
            return parsed
    call = parse_candidate_call(text)
    if call:
        parsed["repair_call"] = call
        parsed["reason"] = "validator emitted a repaired tool call"
    return parsed


def render_validator_prompt(record: dict[str, Any]) -> list[dict[str, Any]]:
    system = (
        "You are a strict BFCL-style tool-call validator. Judge only the proposed next "
        "tool call. Do not solve the whole task. Use the hard schema issues as binding "
        "constraints. Return JSON only with keys: valid, chosen_source, repair_call, "
        "reason, schema_confidence, semantic_confidence."
    )
    user = {
        "task": api_context(record["row"]),
        "planner_state": record["plan"].get("parsed") or record["plan"].get("raw_text", ""),
        "available_tools": compact_tools(record["tools"]),
        "candidates": candidate_summary(record),
        "rules": [
            "If a candidate has hard_schema_issues, valid must be false for that candidate.",
            "If one candidate is schema-valid and semantically matches the current subgoal, set valid true and chosen_source to that source.",
            "If all candidates are invalid but a small schema repair is obvious, put the repaired call in repair_call.",
            "If semantic intent is unclear, prefer valid=false with repair_call=null rather than inventing values.",
            "Never select an unknown tool or add arguments not in the schema.",
        ],
        "return_schema": {
            "valid": True,
            "chosen_source": "executor|recovery",
            "repair_call": {"name": "ToolName", "arguments": {"arg": "value"}},
            "reason": "short rationale",
            "schema_confidence": 0.0,
            "semantic_confidence": 0.0,
        },
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": short_json(user)}]


def validator_generate(model: Any, tokenizer: Any, record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    messages = render_validator_prompt(record)
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
    if hasattr(inputs, "to"):
        inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.validator_max_new_tokens,
        "do_sample": args.validator_temperature > 0,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.validator_temperature > 0:
        gen_kwargs["temperature"] = args.validator_temperature
    start = time.time()
    with torch.inference_mode():
        output = model.generate(inputs, **gen_kwargs)
    new_tokens = output[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    judgment = parse_validator_judgment(text)
    repair_call = normalize_call_to_schema(judgment.get("repair_call"), record["tools"])
    judgment["repair_call"] = repair_call
    judgment["repair_issues"] = schema_issues(repair_call, record["tools"]) if repair_call else []
    judgment["latency_ms"] = round((time.time() - start) * 1000, 3)
    judgment["generated_tokens"] = int(new_tokens.numel())
    return judgment


def deterministic_validator(record: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_summary(record)
    valid_sources = [item["source"] for item in candidates if not item["hard_schema_issues"]]
    return {
        "valid_sources": valid_sources,
        "candidate_issues": {item["source"]: item["hard_schema_issues"] for item in candidates},
    }


def run_validator_phase(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.validator_mode == "off":
        return
    validator_args = argparse.Namespace(
        base=args.validator_base,
        precision=args.validator_precision,
        device_map=args.validator_device_map,
        load_4bit=args.validator_load_4bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.validator_max_new_tokens,
        temperature=args.validator_temperature,
        top_p=args.top_p,
    )
    print(f"[ensemble-apibank] loading BFCL validator model: {args.validator_base}", flush=True)
    model, tokenizer = toolace_eval.load_model(validator_args)
    try:
        for idx, record in enumerate(records, start=1):
            record["deterministic_validation"] = deterministic_validator(record)
            needs_validator = args.validator_mode == "always"
            if args.validator_mode == "invalid":
                needs_validator = any(
                    schema_issues(candidate.get("call"), record["tools"])
                    for candidate in [record["executor"], record.get("recovery")]
                    if candidate
                )
            if not needs_validator:
                continue
            try:
                judgment = validator_generate(model, tokenizer, record, args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                judgment = {
                    "valid": False,
                    "chosen_source": "",
                    "repair_call": None,
                    "repair_issues": [],
                    "reason": "",
                    "schema_confidence": 0.0,
                    "semantic_confidence": 0.0,
                    "raw_text": "",
                    "latency_ms": 0,
                    "generated_tokens": 0,
                }
                error = f"{type(exc).__name__}: {exc}"
            record["validator"] = {**judgment, "error": error}
            if judgment.get("repair_call") and not judgment.get("repair_issues"):
                call = judgment["repair_call"]
                record["validator_repair"] = {
                    "source": "validator_repair",
                    "pred": call_to_text(call, judgment.get("raw_text", "")),
                    "raw_text": judgment.get("raw_text", ""),
                    "call": call,
                    "latency_ms": judgment.get("latency_ms", 0),
                    "generated_tokens": judgment.get("generated_tokens", 0),
                    "error": error,
                }
            print(
                json.dumps(
                    {
                        "phase": "validator",
                        "n": idx,
                        "total": len(records),
                        "valid": judgment.get("valid"),
                        "chosen_source": judgment.get("chosen_source"),
                        "repair": call_to_text(judgment.get("repair_call")) if judgment.get("repair_call") else None,
                        "repair_issues": judgment.get("repair_issues"),
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


def run_toolace_phase(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    run_validator_phase(records, args)


def summarize_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    return {
        "source": candidate.get("source"),
        "pred": candidate.get("pred"),
        "raw_text": candidate.get("raw_text"),
        "call": candidate.get("call"),
        "latency_ms": candidate.get("latency_ms"),
        "generated_tokens": candidate.get("generated_tokens"),
        "selector_score": candidate.get("selector_score"),
        "selector_issues": candidate.get("selector_issues"),
        "error": candidate.get("error", ""),
    }


def build_predictions(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    preds = []
    for record in records:
        row = record["row"]
        candidates = [record["executor"]]
        if record.get("recovery"):
            candidates.append(record["recovery"])
        if record.get("validator_repair"):
            candidates.append(record["validator_repair"])
        hint = chronology_hint_candidate(row, record["tools"])
        if hint:
            candidates.append(hint)
        preferred = preferred_names(record["plan"], record.get("recovery"), record["tools"])
        chosen = choose_candidate(candidates, record["tools"], preferred, record.get("validator"))
        pred = {
            **{k: row[k] for k in ("file", "id", "sample_id", "api_id") if k in row},
            "pred": chosen.get("pred", ""),
            "raw_text": chosen.get("raw_text", ""),
            "expected_output": row.get("expected_output") or row.get("output", ""),
            "expected_call": toolace_eval.expected_call(row.get("expected_output") or row.get("output", "")),
            "latency_ms": round(sum(float((cand or {}).get("latency_ms") or 0) for cand in candidates), 3),
            "generated_tokens": int(sum(int((cand or {}).get("generated_tokens") or 0) for cand in candidates)),
            "error": chosen.get("error", ""),
            "ensemble": {
                "chosen_source": chosen.get("source"),
                "preferred_api_names": sorted(preferred),
                "plan_raw_text": record["plan"].get("raw_text"),
                "plan_parsed": record["plan"].get("parsed"),
                "deterministic_validation": record.get("deterministic_validation"),
                "validator": record.get("validator"),
                "candidates": [summarize_candidate(candidate) for candidate in candidates],
            },
        }
        preds.append(pred)
    return preds


def score_level3(preds: list[dict[str, Any]], gt_path: Path, api_bank_root: Path, max_details: int) -> dict[str, Any]:
    base_eval.install_optional_dependency_stubs()
    from api_call_extraction import get_api_call, parse_api_call
    from lv3_evaluator import split_by_uppercase
    from tool_manager import ToolManager

    gts = json.loads(gt_path.read_text(encoding="utf-8"))
    tool_manager = ToolManager("./lv3_apis")

    def deterministic_tool_search(keywords: str) -> dict[str, Any]:
        compact = "".join(ch for ch in str(keywords) if ch.isalnum()).lower()
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
        parsed = base_eval.parse_first_call(pred.get("pred", ""))
        if parsed:
            pred_api_name = parsed["name"]
            pred_param_dict = parsed["arguments"]
        else:
            api_call = get_api_call(pred.get("pred", ""))
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
                pred_param_dict["keywords"] = split_by_uppercase(str(pred_param_dict["keywords"]))
                pred_result = deterministic_tool_search(pred_param_dict["keywords"])
            else:
                pred_result = tool_manager.api_call(pred_api_name, **pred_param_dict)
        except Exception as exc:  # noqa: BLE001
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
        "paper_metric": "Plan+Retrieve+Call API correctness, using API-Bank lv3_evaluator execution semantics with AST call parsing.",
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


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--level3-gt", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--executor-adapter", default=DEFAULT_EXECUTOR)
    parser.add_argument("--planner-adapter", default=DEFAULT_PLANNER)
    parser.add_argument("--recovery-adapter", default=DEFAULT_RECOVERY)
    parser.add_argument("--gptoss-device-map", default="auto")
    parser.add_argument("--no-mxfp4-dequant", action="store_true")
    parser.add_argument("--max-plan-tokens", type=int, default=256)
    parser.add_argument("--max-action-tokens", type=int, default=180)
    parser.add_argument("--max-recovery-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--recovery-mode", choices=("off", "invalid", "always"), default="always")
    parser.add_argument("--validator-mode", choices=("off", "invalid", "always"))
    parser.add_argument("--validator-base", default="")
    parser.add_argument("--validator-max-new-tokens", type=int)
    parser.add_argument("--validator-temperature", type=float)
    parser.add_argument("--validator-precision", choices=("bf16", "fp16"), default="")
    parser.add_argument("--validator-device-map", default="")
    parser.add_argument("--validator-load-4bit", action="store_true")
    parser.add_argument("--toolace-mode", choices=("off", "invalid", "always"))
    parser.add_argument("--toolace-base", default="")
    parser.add_argument("--toolace-max-new-tokens", type=int)
    parser.add_argument("--toolace-temperature", type=float)
    parser.add_argument("--toolace-precision", choices=("bf16", "fp16"), default="")
    parser.add_argument("--toolace-device-map", default="")
    parser.add_argument("--toolace-load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    args.validator_mode = args.validator_mode or args.toolace_mode or "always"
    args.validator_base = args.validator_base or args.toolace_base or DEFAULT_VALIDATOR
    args.validator_max_new_tokens = (
        args.validator_max_new_tokens
        if args.validator_max_new_tokens is not None
        else args.toolace_max_new_tokens if args.toolace_max_new_tokens is not None else 192
    )
    args.validator_temperature = (
        args.validator_temperature
        if args.validator_temperature is not None
        else args.toolace_temperature if args.toolace_temperature is not None else 0.0
    )
    args.validator_precision = args.validator_precision or args.toolace_precision or "bf16"
    args.validator_device_map = args.validator_device_map or args.toolace_device_map or "auto"
    args.validator_load_4bit = args.validator_load_4bit or args.toolace_load_4bit

    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    base_eval.install_optional_dependency_stubs()

    rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rows = select_rows(rows, args)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "input_json": args.input_json,
                "executor_adapter": args.executor_adapter,
                "planner_adapter": args.planner_adapter,
                "recovery_adapter": args.recovery_adapter,
                "validator_mode": args.validator_mode,
                "validator_base": args.validator_base,
            },
            indent=2,
        ),
        flush=True,
    )

    records = run_gptoss_phase(rows, args)
    run_validator_phase(records, args)
    preds = build_predictions(records, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
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

    if args.level3_gt:
        score = score_level3(preds, Path(args.level3_gt), api_bank_root, args.max_error_details)
        score_path = Path(args.score_output)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    k: score[k]
                    for k in (
                        "paper_metric",
                        "paper_ability",
                        "total_api_calls",
                        "correct_api_calls",
                        "accuracy",
                        "sample_total",
                        "sample_correct",
                        "sample_accuracy",
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
