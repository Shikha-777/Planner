#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from taskdecomp.pipeline import DecompositionPipeline


def fallback_prediction(error: Exception) -> dict[str, Any]:
    return {
        "decision": "no_decomposition",
        "rationale": f"prediction_error: {type(error).__name__}",
        "subtasks": [],
        "dependencies": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/predictions.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()

    pipe = DecompositionPipeline(args.model, max_new_tokens=args.max_new_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as src, args.output.open("w") as dst:
        for i, line in enumerate(src):
            if args.limit is not None and i >= args.limit:
                break
            row = json.loads(line)
            try:
                pred = pipe(row["task"], row.get("context", ""))
            except Exception as exc:
                pred = fallback_prediction(exc)
                pred["error"] = str(exc)[:1000]
            dst.write(json.dumps(pred, ensure_ascii=False) + "\n")
            dst.flush()


if __name__ == "__main__":
    main()
