#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from taskdecomp.prompts import training_messages


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: Any) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def loads_maybe_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return clean_text(message.get("content"))
    return ""


def tool_name_to_text(name: str) -> str:
    name = re.sub(r"[_\\-]+", " ", name).strip()
    return f"Use {name}."


def summarize_arguments(arguments: Any, max_items: int = 4) -> str:
    arguments = loads_maybe_json(arguments, {})
    if not isinstance(arguments, dict):
        return ""
    pieces = []
    for key, value in list(arguments.items())[:max_items]:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)[:80]
        pieces.append(f"{key}={value}")
    return "; ".join(pieces)


def extract_tool_steps(messages: list[dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = clean_text(function.get("name"))
            if not name:
                continue
            args = summarize_arguments(function.get("arguments"))
            text = tool_name_to_text(name)
            if args:
                text = f"{text[:-1]} with {args}."
            key = f"{name}|{args}"
            if key in seen:
                continue
            seen.add(key)
            steps.append(text)
    return steps


def target_from_steps(steps: list[str]) -> dict[str, Any]:
    if len(steps) < 2:
        return {
            "decision": "no_decomposition",
            "rationale": "The request can be handled directly or with a single action.",
            "subtasks": [],
            "dependencies": [],
        }
    ids = [f"s{i + 1}" for i in range(len(steps))]
    return {
        "decision": "decompose",
        "rationale": "The request requires multiple ordered tool-mediated actions.",
        "subtasks": [{"id": sid, "text": step} for sid, step in zip(ids, steps)],
        "dependencies": [{"before": ids[i], "after": ids[i + 1]} for i in range(len(ids) - 1)],
    }


def row_to_sft(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    messages = loads_maybe_json(row.get("messages"), [])
    if not isinstance(messages, list):
        return None
    task = first_user_message(messages)
    if not task:
        return None
    steps = extract_tool_steps(messages)
    target = target_from_steps(steps)
    tool_names = []
    tools = loads_maybe_json(row.get("tools"), [])
    for tool in tools if isinstance(tools, list) else []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            tool_names.append(str(function["name"]))
    context = "; ".join(
        part
        for part in [
            f"Dataset: {source}",
            f"Domain: {row.get('domain')}" if row.get("domain") else "",
            f"Available tools: {', '.join(tool_names[:20])}" if tool_names else "",
        ]
        if part
    )
    return {
        "task": task,
        "context": context,
        "decision": target["decision"],
        "messages": training_messages(task, context, target),
        "target": target,
    }


def iter_hf_jsonl(dataset: str, split: str) -> Iterable[tuple[dict[str, Any], int]]:
    path = hf_hub_download(repo_id=dataset, repo_type="dataset", filename=f"data/{split}.jsonl")
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield {"_malformed": True}, line_no
                continue
            if isinstance(row, dict):
                yield row, line_no


def collect_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    per_split_limit = max(1, args.max_records // len(args.splits)) if args.max_records else None
    for split in args.splits:
        kept = 0
        malformed = 0
        skipped = 0
        seen = 0
        for row, _line_no in iter_hf_jsonl(args.dataset, split):
            seen += 1
            if row.get("_malformed"):
                malformed += 1
                continue
            sft_row = row_to_sft(row, source=f"{args.dataset}:{split}")
            if sft_row is None:
                skipped += 1
                continue
            rows.append(sft_row)
            kept += 1
            if per_split_limit is not None and kept >= per_split_limit:
                break
        stats[f"{split}_seen"] = seen
        stats[f"{split}_kept"] = kept
        stats[f"{split}_skipped"] = skipped
        stats[f"{split}_malformed"] = malformed
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nvidia/Nemotron-SFT-Agentic-v2")
    parser.add_argument("--splits", nargs="+", default=["tool_calling", "interactive_agent"])
    parser.add_argument("--out-dir", type=Path, default=Path("data/nemotron_only"))
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--validation-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows, stats = collect_rows(args)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_n = max(1, int(len(rows) * args.validation_ratio)) if rows else 0
    validation = rows[:val_n]
    train = rows[val_n:]

    write_jsonl(train, args.out_dir / "train.sft.jsonl")
    write_jsonl(validation, args.out_dir / "validation.sft.jsonl")
    summary = {
        "dataset": args.dataset,
        "splits": args.splits,
        "train": len(train),
        "validation": len(validation),
        "decisions": {
            decision: sum(1 for row in rows if row["decision"] == decision)
            for decision in sorted({row["decision"] for row in rows})
        },
        "source_stats": stats,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
