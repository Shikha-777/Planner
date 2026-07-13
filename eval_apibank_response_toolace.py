#!/usr/bin/env python3
"""Run ToolACE on API-Bank response rows and score ROUGE-L."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import eval_apibank_toolace_official as base_eval


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref or not hyp:
        return 0.0
    lcs = lcs_len(ref, hyp)
    recall = lcs / len(ref)
    precision = lcs / len(hyp)
    return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)


def clean_response(text: str) -> str:
    text = text.strip()
    for prefix in ("AI:", "Response:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    return base_eval.load_model(args)


def generate_response(model: Any, tokenizer: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    prompt = row.get("instruction", "") + "\n" + row.get("input", "")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Generate only the next assistant response. Do not generate API calls unless the requested output is an API call."},
        {"role": "user", "content": prompt},
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
    return {"pred": clean_response(raw_text), "raw_text": raw_text, "latency_ms": round((time.time() - start) * 1000, 3), "generated_tokens": int(new_tokens.numel())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--ability", required=True)
    parser.add_argument("--base", default="Team-ACE/ToolACE-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args)
    preds = []
    scores = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            try:
                result = generate_response(model, tokenizer, row, args)
                error = ""
            except Exception as exc:
                result = {"pred": "", "raw_text": "", "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            expected = clean_response(row.get("expected_output") or row.get("output", ""))
            score = rouge_l_f(expected, result["pred"])
            scores.append(score)
            pred = {
                **{k: row[k] for k in ("file", "id", "sample_id", "api_id") if k in row},
                "pred": result["pred"],
                "raw_text": result["raw_text"],
                "expected_output": expected,
                "rouge_l": round(score, 6),
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
                "error": error,
            }
            preds.append(pred)
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"n": idx, "total": len(rows), "rouge_l": pred["rouge_l"], "error": error}, ensure_ascii=False), flush=True)
    summary = {
        "paper_metric": "Response quality ROUGE-L F1 over API-Bank response rows.",
        "paper_ability": args.ability,
        "total_responses": len(scores),
        "rouge_l": round(sum(scores) / len(scores), 6) if scores else 0.0,
    }
    Path(args.score_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
