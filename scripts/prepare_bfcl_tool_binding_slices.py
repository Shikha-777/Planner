#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_capability_benchmark_slices import first_user_content, write_json, write_jsonl
from taskdecomp.tool_binding import normalize_expected_ground_truth, parse_call_string


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
DEFAULT_OUTPUT_ROOT = Path("data/benchmarks/tool_binding")

POSSIBLE_ANSWER_FILES = {
    "bfcl_simple": ("BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json"),
    "bfcl_multiple": ("BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"),
    "bfcl_parallel": ("BFCL_v3_parallel.json", "possible_answer/BFCL_v3_parallel.json"),
    "bfcl_parallel_multiple": (
        "BFCL_v3_parallel_multiple.json",
        "possible_answer/BFCL_v3_parallel_multiple.json",
    ),
}

EXEC_FILES = {
    "bfcl_exec_simple": "BFCL_v3_exec_simple.json",
    "bfcl_exec_multiple": "BFCL_v3_exec_multiple.json",
    "bfcl_exec_parallel": "BFCL_v3_exec_parallel.json",
    "bfcl_exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
}

IRRELEVANCE_FILE = "BFCL_v3_irrelevance.json"


def prepare_possible_answer_file(
    name: str,
    data_file: str,
    answer_file: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    data_path = hf_hub_download(BFCL_REPO, data_file, repo_type="dataset")
    answer_path = hf_hub_download(BFCL_REPO, answer_file, repo_type="dataset")
    answer_by_id = {
        item["id"]: item
        for item in _read_jsonl(answer_path)
        if isinstance(item, dict) and item.get("id")
    }
    rows = []
    selected = 0
    for raw_index, item in enumerate(_read_jsonl(data_path)):
        if raw_index < offset:
            continue
        if selected >= limit:
            break
        row_id = str(item.get("id") or f"{name}_{raw_index + 1:04d}")
        answer = answer_by_id.get(row_id)
        if not answer:
            continue
        calls = normalize_expected_ground_truth(answer.get("ground_truth"))
        rows.append(
            _row(
                f"{name}_{raw_index + 1:04d}",
                name,
                item,
                calls,
                source_file=data_file,
                source_record_id=row_id,
            )
        )
        selected += 1
    return rows, {
        "source": BFCL_REPO,
        "data_file": data_file,
        "answer_file": answer_file,
        "offset": offset,
        "rows": len(rows),
    }


def prepare_exec_file(
    name: str,
    data_file: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    data_path = hf_hub_download(BFCL_REPO, data_file, repo_type="dataset")
    rows = []
    selected = 0
    for raw_index, item in enumerate(_read_jsonl(data_path)):
        if raw_index < offset:
            continue
        if selected >= limit:
            break
        calls = [
            parsed
            for value in item.get("ground_truth") or []
            if isinstance(value, str)
            if (parsed := parse_call_string(value))
        ]
        rows.append(
            _row(
                f"{name}_{raw_index + 1:04d}",
                name,
                item,
                calls,
                source_file=data_file,
                source_record_id=str(item.get("id") or raw_index),
            )
        )
        selected += 1
    return rows, {
        "source": BFCL_REPO,
        "data_file": data_file,
        "offset": offset,
        "rows": len(rows),
    }


def prepare_irrelevance(
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    data_path = hf_hub_download(BFCL_REPO, IRRELEVANCE_FILE, repo_type="dataset")
    rows = []
    selected = 0
    for raw_index, item in enumerate(_read_jsonl(data_path)):
        if raw_index < offset:
            continue
        if selected >= limit:
            break
        rows.append(
            _row(
                f"bfcl_irrelevance_{raw_index + 1:04d}",
                "bfcl_irrelevance",
                item,
                [],
                source_file=IRRELEVANCE_FILE,
                source_record_id=str(item.get("id") or raw_index),
            )
        )
        selected += 1
    return rows, {
        "source": BFCL_REPO,
        "data_file": IRRELEVANCE_FILE,
        "offset": offset,
        "rows": len(rows),
    }


def _row(
    row_id: str,
    category: str,
    item: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    source_file: str,
    source_record_id: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "request": first_user_content(item.get("question")),
        "tools": item.get("function") or [],
        "expected_tool_binding": {
            "tool_decision": "call" if calls else "no_tool",
            "calls": calls,
        },
        "source_dataset": "bfcl",
        "source_file": source_file,
        "source_record_id": source_record_id,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_step(
    name: str,
    path: Path,
    prepare_fn: Any,
    manifest: list[dict[str, Any]],
) -> bool:
    try:
        rows, metadata = prepare_fn()
        write_jsonl(path, rows)
        manifest.append(
            {
                "name": name,
                "status": "prepared",
                "path": str(path),
                "row_count": len(rows),
                "metadata": metadata,
            }
        )
        return bool(rows)
    except Exception as exc:
        manifest.append(
            {
                "name": name,
                "status": "skipped",
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BFCL tool-binding gold slices.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-per-file", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--irrelevance-limit", type=int, default=25)
    parser.add_argument("--include-exec", action="store_true")
    args = parser.parse_args()

    manifest: list[dict[str, Any]] = []
    prepared: dict[str, str] = {}
    for name, (data_file, answer_file) in POSSIBLE_ANSWER_FILES.items():
        path = args.output_root / f"{name}.jsonl"
        if run_step(
            name,
            path,
            lambda name=name, data_file=data_file, answer_file=answer_file: prepare_possible_answer_file(
                name,
                data_file,
                answer_file,
                limit=args.limit_per_file,
                offset=args.offset,
            ),
            manifest,
        ):
            prepared[name] = str(path)

    if args.include_exec:
        for name, data_file in EXEC_FILES.items():
            path = args.output_root / f"{name}.jsonl"
            if run_step(
                name,
                path,
                lambda name=name, data_file=data_file: prepare_exec_file(
                    name,
                    data_file,
                    limit=args.limit_per_file,
                    offset=args.offset,
                ),
                manifest,
            ):
                prepared[name] = str(path)

    path = args.output_root / "bfcl_irrelevance.jsonl"
    if run_step(
        "bfcl_irrelevance",
        path,
        lambda: prepare_irrelevance(limit=args.irrelevance_limit, offset=args.offset),
        manifest,
    ):
        prepared["bfcl_irrelevance"] = str(path)

    manifest_path = args.output_root / "manifest.json"
    write_json(manifest_path, {"prepared": manifest, "files": prepared})
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "prepared_count": sum(1 for item in manifest if item["status"] == "prepared"),
                "skipped_count": sum(1 for item in manifest if item["status"] == "skipped"),
                "files": prepared,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
