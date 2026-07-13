#!/usr/bin/env python3
"""Run BFCL single-turn inference with a local HF model or PEFT adapter.

The output is compatible with score_bfcl_official_ast.py: every row contains a
`baseline.normalized_calls` lane. The lane name is intentionally reused so the
existing scorer does not need to learn about model-loading details.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

try:
    from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
    from smoke_eval import TOOL_SYSTEM_PROMPT, dedupe_tool_calls, openai_tools
except ImportError:
    from .bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
    from .smoke_eval import TOOL_SYSTEM_PROMPT, dedupe_tool_calls, openai_tools


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

    if isinstance(value.get("function"), dict):
        fn = value["function"]
    else:
        fn = value

    name = fn.get("name") or fn.get("function_name")
    arguments = fn.get("arguments")
    if arguments is None:
        arguments = fn.get("args", fn.get("parameters", {}))
    arguments = maybe_json(arguments)
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    if not name:
        return None
    return {"name": str(name), "arguments": arguments}


def normalize_call_container(value: Any) -> list[dict[str, Any]]:
    value = maybe_json(value)
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    elif isinstance(value, dict) and "name" in value:
        value = [value]
    elif isinstance(value, dict) and "function" in value:
        value = [value]

    if not isinstance(value, list):
        return []

    calls = []
    for item in value:
        call = normalize_one_call(item)
        if call:
            calls.append(call)
    return dedupe_tool_calls(calls)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        calls.extend(normalize_call_container(match.group(1)))
    for match in HARMONY_CALL_RE.finditer(text):
        args = maybe_json(match.group(2))
        if isinstance(args, dict):
            calls.append({"name": match.group(1), "arguments": args})
    if calls:
        return dedupe_tool_calls(calls)

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    calls = normalize_call_container(stripped)
    if calls:
        return calls
    for obj in iter_json_objects(stripped):
        calls = normalize_call_container(obj)
        if calls:
            return calls
    return []


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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.tokenizer or args.adapter or args.base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.precision, torch),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


def resolve_dtype(value: str, torch: Any) -> Any:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def render_inputs(tokenizer: Any, prompt: str, tools: list[dict[str, Any]], args: argparse.Namespace) -> Any:
    messages = [
        {"role": "system", "content": args.system_prompt or TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered_tools = openai_tools(tools) if args.openai_tool_schema else tools
    return tokenizer.apply_chat_template(
        messages,
        tools=rendered_tools,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )


def generate_one(model: Any, tokenizer: Any, prompt: str, tools: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    inputs = render_inputs(tokenizer, prompt, tools, args)
    if hasattr(inputs, "to"):
        inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_token_id.append(im_end_id)

    start = time.time()
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "eos_token_id": eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(inputs, **generate_kwargs)
    latency_ms = round((time.time() - start) * 1000, 3)
    new_tokens = generated[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    calls = parse_tool_calls(text)
    return {
        "latency_ms": latency_ms,
        "normalized_calls": calls,
        "raw_text": text,
        "generated_tokens": int(new_tokens.numel()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="BFCL question JSON/JSONL")
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--base", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter", default="", help="Optional PEFT adapter directory")
    parser.add_argument("--tokenizer", default="", help="Optional tokenizer/template source")
    parser.add_argument("--function-doc-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--raw-tool-schema",
        dest="openai_tool_schema",
        action="store_false",
        help="Pass normalized BFCL function dicts directly instead of OpenAI tool wrappers.",
    )
    parser.set_defaults(openai_tool_schema=True)
    parser.add_argument("--system-prompt", default="")
    args = parser.parse_args()

    records = load_records(args.input)
    selected = select_records(records, args)
    model, tokenizer = load_model_and_tokenizer(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in selected:
            rid = record_id(record, index)
            prompt = extract_prompt(record)
            tools = normalize_tools(record, args.function_doc_dir or None)
            try:
                result = generate_one(model, tokenizer, prompt, tools, args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {
                    "latency_ms": 0,
                    "normalized_calls": [],
                    "raw_text": "",
                    "generated_tokens": 0,
                }
                error = f"{type(exc).__name__}: {exc}"

            row = {
                "id": rid,
                "baseline": {
                    **result,
                    "model": args.base,
                    "adapter": args.adapter,
                    "error": error,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(
                json.dumps(
                    {
                        "id": rid,
                        "calls": len(result["normalized_calls"]),
                        "latency_ms": result["latency_ms"],
                        "error": error,
                    }
                ),
                flush=True,
            )

    print(json.dumps({"input": args.input, "output": args.output, "rows": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
