from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, pipeline

from .graph import transitive_reduction_like
from .prompts import SYSTEM_PROMPT, user_prompt


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        from json_repair import repair_json

        repaired = repair_json(snippet, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError(f"Repaired model output was not a JSON object: {text[:500]}")
        return repaired


def normalize_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    decision = pred.get("decision")
    if decision not in {"decompose", "no_decomposition"}:
        decision = "decompose" if pred.get("subtasks") else "no_decomposition"

    subtasks = pred.get("subtasks") or []
    norm_subtasks = []
    seen_ids = set()
    for i, item in enumerate(subtasks):
        if isinstance(item, str):
            sid, text = f"s{i + 1}", item
        else:
            sid, text = str(item.get("id") or f"s{i + 1}"), str(item.get("text") or "")
        text = text.strip()
        if not text or sid in seen_ids:
            continue
        seen_ids.add(sid)
        norm_subtasks.append({"id": sid, "text": text})

    ids = [s["id"] for s in norm_subtasks]
    deps = []
    for dep in pred.get("dependencies") or []:
        if not isinstance(dep, dict):
            continue
        before = str(dep.get("before") or "").strip()
        after = str(dep.get("after") or "").strip()
        deps.append((before, after))
    deps = transitive_reduction_like(ids, deps)

    if decision == "no_decomposition":
        norm_subtasks = []
        deps = []

    return {
        "decision": decision,
        "rationale": str(pred.get("rationale") or "").strip(),
        "subtasks": norm_subtasks,
        "dependencies": [{"before": a, "after": b} for a, b in deps],
    }


class DecompositionPipeline:
    def __init__(self, model: str, max_new_tokens: int = 1200) -> None:
        self.max_new_tokens = max_new_tokens
        if "gpt-oss" in model.lower():
            loaded_model = AutoPeftModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                device_map="auto",
                quantization_config=Mxfp4Config(dequantize=True),
            )
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipe = pipeline(
            "text-generation",
            model=loaded_model,
            tokenizer=tokenizer,
        )

    def __call__(self, task: str, context: str = "") -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(task, context)},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        if isinstance(text, list):
            text = text[-1].get("content", "")
        return normalize_prediction(extract_json(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--task", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pred = DecompositionPipeline(args.model)(args.task, args.context)
    text = json.dumps(pred, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
