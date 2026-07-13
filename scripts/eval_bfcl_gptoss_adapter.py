#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, Mxfp4Config


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
CHANNEL_MESSAGE_RE = re.compile(
    r"<\|channel\|>(?P<channel>[^<]*?)<\|message\|>(?P<message>.*?)(?:<\|call\|>|<\|return\|>|<\|end\|>)",
    re.DOTALL,
)


def add_import_roots(workflow: str, bfcl_root: str) -> None:
    for path in (Path(workflow) / "scripts", Path(bfcl_root)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_one_call(value: Any) -> dict[str, Any] | None:
    value = maybe_json(value)
    if not isinstance(value, dict):
        return None
    fn = value["function"] if isinstance(value.get("function"), dict) else value
    name = fn.get("name") or fn.get("function_name")
    args = fn.get("arguments")
    if args is None:
        args = fn.get("args", fn.get("parameters", {}))
    args = maybe_json(args)
    if not isinstance(args, dict):
        args = {"value": args}
    if not name:
        return None
    return {"name": str(name), "arguments": args}


def normalize_call_container(value: Any) -> list[dict[str, Any]]:
    from smoke_eval import dedupe_tool_calls

    value = maybe_json(value)
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict) and ("name" in value or "function" in value):
        value = [value]
    if not isinstance(value, list):
        return []
    return dedupe_tool_calls([call for item in value if (call := normalize_one_call(item))])


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    from smoke_eval import dedupe_tool_calls

    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        calls.extend(normalize_call_container(match.group(1)))
    if calls:
        return dedupe_tool_calls(calls)

    for match in CHANNEL_MESSAGE_RE.finditer(text):
        channel = match.group("channel").strip()
        message = match.group("message").strip()
        if "functions" in channel or message.startswith(("{", "[")):
            calls.extend(normalize_call_container(message))
    if calls:
        return dedupe_tool_calls(calls)

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return normalize_call_container(stripped)


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(records))
    if args.offset:
        indexed = indexed[args.offset :]
    if args.sample:
        rng = random.Random(args.seed)
        indexed = rng.sample(indexed, min(args.sample, len(indexed)))
    if args.limit:
        indexed = indexed[: args.limit]
    return indexed


def load_gptoss_adapter(adapter: str) -> tuple[Any, Any]:
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=Mxfp4Config(dequantize=True),
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def render_inputs(tokenizer: Any, prompt: str, tools: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    from smoke_eval import TOOL_SYSTEM_PROMPT, openai_tools

    messages = [
        {"role": "system", "content": args.system_prompt or TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered_tools = openai_tools(tools)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=rendered_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except Exception:
        tool_text = json.dumps(rendered_tools, ensure_ascii=False)
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    (args.system_prompt or TOOL_SYSTEM_PROMPT)
                    + "\n\nAvailable tools are encoded as JSON:\n"
                    + tool_text
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            fallback_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )


def generate_one(model: Any, tokenizer: Any, prompt: str, tools: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    inputs = render_inputs(tokenizer, prompt, tools, args)
    if hasattr(inputs, "to"):
        inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.append(im_end_id)

    start = time.time()
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(inputs, **kwargs)
    new_tokens = generated[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return {
        "latency_ms": round((time.time() - start) * 1000, 3),
        "normalized_calls": parse_tool_calls(text),
        "raw_text": text,
        "generated_tokens": int(new_tokens.numel()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--bfcl-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--function-doc-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    add_import_roots(args.workflow, args.bfcl_root)
    from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id

    records = load_records(args.input)
    selected = select_records(records, args)
    model, tokenizer = load_gptoss_adapter(args.adapter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in selected:
            rid = record_id(record, index)
            prompt = extract_prompt(record)
            tools = normalize_tools(record, args.function_doc_dir or None)
            try:
                result = generate_one(model, tokenizer, prompt, tools, args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"latency_ms": 0, "normalized_calls": [], "raw_text": "", "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "id": rid,
                "baseline": {
                    **result,
                    "model": "openai/gpt-oss-20b",
                    "adapter": args.adapter,
                    "error": error,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"id": rid, "calls": len(result["normalized_calls"]), "error": error}), flush=True)

    print(json.dumps({"input": args.input, "output": args.output, "rows": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
