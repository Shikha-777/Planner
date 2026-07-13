#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download


DATASET_FILES = {
    "nvidia/Nemotron-SFT-Agentic-v2": {
        "interactive_agent": "data/interactive_agent.jsonl",
        "tool_calling": "data/tool_calling.jsonl",
        "search": "data/search.jsonl",
    },
    "nvidia/Nemotron-Agentic-v1": {
        "interactive_agent": "data/interactive_agent.jsonl",
        "tool_calling": "data/tool_calling.jsonl",
    },
}


def iter_hf_jsonl(repo: str, filename: str) -> Iterable[tuple[dict[str, Any] | None, int]]:
    path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield None, line_no
                continue
            yield row if isinstance(row, dict) else None, line_no


def clean_message(message: Any, keep_reasoning: bool = False) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool", "function"}:
        return None
    cleaned: dict[str, Any] = {"role": "tool" if role == "function" else role, "content": message.get("content") or ""}
    for key in ("name", "tool_call_id"):
        if message.get(key):
            cleaned[key] = message[key]
    for key in ("tool_calls", "function_call"):
        value = message.get(key)
        if value not in (None, [], {}):
            cleaned[key] = value
    if keep_reasoning and message.get("reasoning_content"):
        cleaned["reasoning_content"] = message["reasoning_content"]
    return cleaned


def loads_maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def clean_tool(tool: Any) -> dict[str, Any] | None:
    tool = loads_maybe_json(tool)
    if not isinstance(tool, dict):
        return None
    cleaned = dict(tool)
    function = cleaned.get("function")
    if isinstance(function, dict):
        function = dict(function)
        if "parameters" in function:
            function["parameters"] = loads_maybe_json(function["parameters"])
        cleaned["function"] = function
    return cleaned


def stable_id(repo: str, subset: str, index: int, row: dict[str, Any]) -> str:
    for key in ("uuid", "id"):
        if row.get(key):
            return str(row[key])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("uuid", "id", "alt_id"):
            if metadata.get(key):
                return str(metadata[key])
    digest = hashlib.sha1(
        json.dumps(row.get("messages") or [], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"{repo}:{subset}:{index}:{digest}"


def normalize_record(
    repo: str,
    subset: str,
    source_path: str,
    index: int,
    row: dict[str, Any],
    keep_reasoning: bool,
) -> dict[str, Any] | None:
    messages = [
        message
        for message in (clean_message(item, keep_reasoning=keep_reasoning) for item in row.get("messages") or [])
        if message is not None
    ]
    if len(messages) < 2 or not any(item.get("role") == "assistant" for item in messages):
        return None
    tools = [tool for tool in (clean_tool(item) for item in row.get("tools") or []) if tool is not None]
    return {
        "id": stable_id(repo, subset, index, row),
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": row.get("parallel_tool_calls"),
        "source_repo": repo,
        "source_subset": subset,
        "source_path": source_path,
        "source_index": index,
        "domain": row.get("domain"),
        "used_in": row.get("used_in"),
        "license": row.get("license"),
    }


def split_for_id(record_id: str, val_fraction: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "validation" if bucket < val_fraction else "train"


def parse_subset(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("subset must be REPO:SUBSET")
    repo, subset = value.split(":", 1)
    if repo not in DATASET_FILES or subset not in DATASET_FILES[repo]:
        raise argparse.ArgumentTypeError(f"unsupported subset {value!r}")
    return repo, subset


def write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        action="append",
        type=parse_subset,
        default=[],
        help="Dataset subset as REPO:SUBSET. May be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/nemotron_agentic_sft"))
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--keep-reasoning", action="store_true")
    args = parser.parse_args()

    subsets = args.subset or [
        ("nvidia/Nemotron-SFT-Agentic-v2", "tool_calling"),
        ("nvidia/Nemotron-SFT-Agentic-v2", "interactive_agent"),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "validation.jsonl"
    summary_path = args.out_dir / "summary.json"

    stats: dict[str, Any] = {
        "selected_subsets": [f"{repo}:{subset}" for repo, subset in subsets],
        "max_records": args.max_records,
        "validation_ratio": args.validation_ratio,
        "counts": {"seen": 0, "written": 0, "train": 0, "validation": 0, "skipped": 0, "malformed": 0},
        "by_subset": {},
    }

    buffer: list[dict[str, Any]] = []
    per_subset_limit = max(1, args.max_records // len(subsets)) if args.max_records else 0
    for repo, subset in subsets:
        source_path = DATASET_FILES[repo][subset]
        subset_stats = stats["by_subset"].setdefault(
            subset, {"seen": 0, "written": 0, "skipped": 0, "malformed": 0}
        )
        for row, line_no in iter_hf_jsonl(repo, source_path):
            stats["counts"]["seen"] += 1
            subset_stats["seen"] += 1
            if row is None:
                stats["counts"]["malformed"] += 1
                subset_stats["malformed"] += 1
                continue
            record = normalize_record(repo, subset, source_path, line_no, row, args.keep_reasoning)
            if record is None:
                stats["counts"]["skipped"] += 1
                subset_stats["skipped"] += 1
                continue
            buffer.append(record)
            subset_stats["written"] += 1
            if per_subset_limit and subset_stats["written"] >= per_subset_limit:
                break
        if args.max_records and len(buffer) >= args.max_records:
            break

    random.Random(args.seed).shuffle(buffer)
    with train_path.open("w", encoding="utf-8") as train_fh, val_path.open("w", encoding="utf-8") as val_fh:
        for record in buffer:
            split = split_for_id(record["id"], args.validation_ratio, args.seed)
            write_record(val_fh if split == "validation" else train_fh, record)
            stats["counts"]["written"] += 1
            stats["counts"][split] += 1

    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"train": str(train_path), "validation": str(val_path), **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
