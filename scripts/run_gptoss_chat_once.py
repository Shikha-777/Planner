#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
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
    result = {
        "model": args.model,
        "system": args.system,
        "prompt": args.prompt,
        "raw_text": raw_text,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
