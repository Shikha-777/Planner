#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from taskdecomp.agent_binding_benchmark import build_synthetic_agent_binding_cases


DEFAULT_OUTPUT = Path("data/benchmarks/agent_binding/synthetic_100.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic agent-binding benchmark.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = build_synthetic_agent_binding_cases(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
