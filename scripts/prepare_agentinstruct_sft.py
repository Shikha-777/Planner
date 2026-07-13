#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import get_dataset_split_names, load_dataset


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def split_for_id(record_id: str, validation_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "validation" if bucket < validation_ratio else "train"


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_conversation(
    raw_messages: Any,
    loss_policy: str,
) -> tuple[list[dict[str, Any]], list[bool]] | None:
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict[str, Any]] = []
    loss_mask: list[bool] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = ROLE_MAP.get(str(item.get("from") or item.get("role") or "").lower())
        content = item.get("value", item.get("content"))
        if role not in {"system", "user", "assistant"} or content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        messages.append({"role": role, "content": text})
        if role == "assistant":
            loss_value = item.get("loss")
            if loss_policy == "all-assistant":
                loss_mask.append(True)
            elif loss_value is None:
                loss_mask.append(True)
            else:
                loss_mask.append(bool(loss_value))
        else:
            loss_mask.append(False)

    if len(messages) < 2 or not any(message["role"] == "assistant" for message in messages):
        return None
    if not any(loss_mask[index] for index, message in enumerate(messages) if message["role"] == "assistant"):
        if loss_policy == "true-only":
            return None
        loss_mask = [message["role"] == "assistant" for message in messages]
    return messages, loss_mask


def convert_row(row: dict[str, Any], split: str, index: int, loss_policy: str) -> dict[str, Any] | None:
    normalized = normalize_conversation(row.get("conversations"), loss_policy)
    if normalized is None:
        return None
    messages, loss_mask = normalized
    source_id = str(row.get("id") or f"{split}_{index}")
    return {
        "id": f"agentinstruct:{split}:{source_id}",
        "messages": messages,
        "loss_mask": loss_mask,
        "source_repo": "zai-org/AgentInstruct",
        "source_split": split,
        "source_index": index,
        "source_id": source_id,
    }


def iter_split(dataset_name: str, split: str, streaming: bool):
    yield from load_dataset(dataset_name, split=split, streaming=streaming)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="zai-org/AgentInstruct")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("data/agentinstruct_sft"))
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--loss-policy",
        choices=["true-only", "all-assistant"],
        default="true-only",
        help="true-only respects AgentInstruct's assistant loss flags.",
    )
    args = parser.parse_args()

    splits = args.splits or get_dataset_split_names(args.dataset)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "dataset": args.dataset,
        "splits": splits,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "loss_policy": args.loss_policy,
        "max_records": args.max_records,
        "counts": {"seen": 0, "written": 0, "train": 0, "validation": 0, "skipped": 0},
        "by_split": {},
    }

    for split in splits:
        split_stats = stats["by_split"].setdefault(
            split,
            {"seen": 0, "written": 0, "train": 0, "validation": 0, "skipped": 0},
        )
        for index, row in enumerate(iter_split(args.dataset, split, args.streaming)):
            if args.max_records and stats["counts"]["seen"] >= args.max_records:
                break
            stats["counts"]["seen"] += 1
            split_stats["seen"] += 1
            converted = convert_row(row, split, index, args.loss_policy)
            if converted is None:
                stats["counts"]["skipped"] += 1
                split_stats["skipped"] += 1
                continue
            out_split = split_for_id(converted["id"], args.validation_ratio, args.seed)
            if out_split == "validation":
                validation_rows.append(converted)
            else:
                train_rows.append(converted)
            stats["counts"]["written"] += 1
            stats["counts"][out_split] += 1
            split_stats["written"] += 1
            split_stats[out_split] += 1
        if args.max_records and stats["counts"]["seen"] >= args.max_records:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, args.out_dir / "train.jsonl")
    write_jsonl(validation_rows, args.out_dir / "validation.jsonl")
    (args.out_dir / "summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
