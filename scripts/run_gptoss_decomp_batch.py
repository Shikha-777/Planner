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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "no JSON object found"
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "parsed value was not an object"
    return parsed, None


def build_messages(task: str, context: str) -> list[dict[str, str]]:
    payload = {"task": task, "context": context}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Decompose this task only if it needs multiple meaningful actions. "
                "Use at most 8 subtasks.\nInput:\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
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

    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            task = str(row["task"])
            context = str(row.get("context") or "")
            messages = build_messages(task, context)
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
                "id": row.get("id"),
                "label": row.get("label"),
                "model": args.model,
                "task": task,
                "context": context,
                "raw_text": raw_text,
                "parsed": parsed,
                "parse_error": parse_error,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"id": result["id"], "parse_error": parse_error, "parsed": parsed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
