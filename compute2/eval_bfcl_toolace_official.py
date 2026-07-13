#!/usr/bin/env python3
"""Run BFCL inference with the official ToolACE prompt/output format."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
from smoke_eval import dedupe_tool_calls


TOOLACE_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.
{functions}
"""


def py_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [py_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [py_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {py_value(k): py_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = py_value(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null":
            return None
        return node.id
    return ast.unparse(node) if hasattr(ast, "unparse") else None


def func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = func_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def call_from_ast(node: ast.AST) -> dict[str, Any] | None:
    if not isinstance(node, ast.Call):
        return None
    name = func_name(node.func)
    if not name:
        return None
    args = {kw.arg: py_value(kw.value) for kw in node.keywords if kw.arg}
    for idx, value in enumerate(node.args):
        args[f"arg{idx}"] = py_value(value)
    return {"name": name, "arguments": args}


def parse_calls(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    match = re.search(r"\[[\s\S]*\]", stripped)
    if match:
        stripped = match.group(0)
    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError:
        return []
    root = tree.body
    nodes = root.elts if isinstance(root, ast.List) else [root]
    calls = [call for node in nodes if (call := call_from_ast(node))]
    return dedupe_tool_calls(calls)


def toolace_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        converted.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "arguments": tool.get("parameters", {}),
            }
        )
    return converted


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


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "torch_dtype": resolve_dtype(args.precision, torch),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    model.eval()
    return model, tokenizer


def generate_one(model: Any, tokenizer: Any, prompt: str, tools: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    messages = [
        {"role": "system", "content": TOOLACE_SYSTEM_PROMPT.format(functions=json.dumps(toolace_tools(tools), ensure_ascii=False))},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    input_len = inputs.shape[-1]
    start = time.time()
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        generated = model.generate(inputs, **generate_kwargs)
    new_tokens = generated[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "latency_ms": round((time.time() - start) * 1000, 3),
        "normalized_calls": parse_calls(text),
        "raw_text": text,
        "generated_tokens": int(new_tokens.numel()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="Team-ACE/ToolACE-8B")
    parser.add_argument("--function-doc-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    records = load_records(args.input)
    selected = select_records(records, args)
    model, tokenizer = load_model(args)

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
            except Exception as exc:
                result = {"latency_ms": 0, "normalized_calls": [], "raw_text": "", "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            row = {"id": rid, "baseline": {**result, "model": args.base, "adapter": "", "error": error}}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"id": rid, "calls": len(result["normalized_calls"]), "error": error}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
