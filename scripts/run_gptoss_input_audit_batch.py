#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


SYSTEM_PROMPT = """You inspect original user requests extracted from a dataset row.
Use only the text inside ORIGINAL_USER_MESSAGES. Do not answer the original user's task.
Do not say the user wants an audit, JSON analysis, or conversation inspection; that is the wrapper task, not the original request.
Bad final_user_want examples: "Audit this conversation input", "Return the JSON audit", "Inspect the dataset row".
Good final_user_want examples: "choose the correct answer to a multiple-choice legal question", "draft a professional email", "explain why an event mattered".
Identify:
1. the final thing the original user wants the assistant to produce or do,
2. the concrete inputs needed to satisfy that original request,
3. whether each needed input is already available in ORIGINAL_USER_MESSAGES.
Return only compact valid JSON with this schema:
{"final_user_want":"...","inputs_needed":[{"input":"...","available":true,"evidence":"..."}],"all_inputs_available":true,"missing_inputs":["..."]}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_last_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    end = text.rfind("}")
    if end < 0:
        return None, "no closing brace found"
    for match in reversed(list(re.finditer(r"{", text[: end + 1]))):
        snippet = text[match.start() : end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
    return None, "no valid JSON object found"


def prompt_for(row: dict[str, Any]) -> list[dict[str, str]]:
    user_messages = row.get("user_messages") or []
    original = "\n\n".join(
        f"TURN {message.get('turn')}: {message.get('content')}" for message in user_messages
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"DATASET_ROW_ID: {row.get('id')}\n"
                "ORIGINAL_USER_MESSAGES_START\n"
                f"{original}\n"
                "ORIGINAL_USER_MESSAGES_END"
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=700)
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
            messages = prompt_for(row)
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
            parsed, parse_error = extract_last_json(raw_text)
            result = {
                "id": row.get("id"),
                "row_index": row.get("row_index"),
                "domain": row.get("domain"),
                "source": row.get("source"),
                "generator": row.get("generator"),
                "messages_length": row.get("messages_length"),
                "user_messages": row.get("user_messages"),
                "raw_text": raw_text,
                "parsed": parsed,
                "parse_error": parse_error,
            }
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(json.dumps({"id": result["id"], "parse_error": parse_error, "parsed": parsed}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
