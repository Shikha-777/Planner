#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nvidia/Nemotron-Cascade-2-SFT-Data")
    parser.add_argument("--config", default="chat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/nemotron_input_audit/messages_sample.jsonl"))
    parser.add_argument("--max-records", type=int, default=25)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if index < args.skip:
                continue
            messages = row.get("messages") or []
            user_messages = [
                {"turn": turn, "content": str(message.get("content") or "")}
                for turn, message in enumerate(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if not user_messages:
                continue
            out_row = {
                "id": f"{args.config}:{args.split}:{index}",
                "row_index": index,
                "domain": row.get("domain"),
                "source": row.get("source"),
                "generator": row.get("generator"),
                "messages_length": len(messages),
                "user_messages": user_messages,
                "user_text": "\n\n".join(
                    f"User turn {item['turn']}: {item['content']}" for item in user_messages
                ),
            }
            handle.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if args.max_records and written >= args.max_records:
                break
    print(json.dumps({"out": str(args.out), "written": written, "skip": args.skip, "max_records": args.max_records}, indent=2))


if __name__ == "__main__":
    main()
