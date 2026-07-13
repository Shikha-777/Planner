#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


SYSTEM_PROMPT = """You are a task decomposition model.
Return only compact valid JSON. Do not use markdown.
Use this schema:
{"decision":"decompose|no_decomposition","rationale":"short reason","subtasks":[{"id":"s1","text":"verb-led action"}],"dependencies":[{"before":"s1","after":"s2"}]}"""


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "no JSON object found"
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return parsed if isinstance(parsed, dict) else None, "parsed value was not an object"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/gpt-oss-20b-trace-decomp-lora-v1")
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_path = Path(args.model)
    loader = AutoPeftModelForCausalLM if (model_path / "adapter_config.json").exists() else AutoModelForCausalLM
    model = loader.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        quantization_config=Mxfp4Config(dequantize=True),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    payload = {"task": args.task, "context": args.context}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Decompose this task only if it needs multiple meaningful actions. "
                "Use at most 5 subtasks.\nInput:\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    parsed, parse_error = extract_json(raw_text)
    result = {
        "model": args.model,
        "task": args.task,
        "context": args.context,
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": None if parsed is not None else parse_error,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
