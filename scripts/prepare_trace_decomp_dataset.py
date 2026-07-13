#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from taskdecomp.trajectory_dataset import (
    HF_SOURCES,
    iter_hf_records,
    record_to_annotation_row,
    record_to_sft_row,
    split_for_id,
)


def write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a trace-derived decomposition dataset from public agent/web trajectories."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/trace_decomp"))
    parser.add_argument("--sources", nargs="+", choices=sorted(HF_SOURCES), default=sorted(HF_SOURCES))
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=100,
        help="Maximum rows to read from each source. Use 0 for no explicit cap.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-trace-events", type=int, default=24)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record source errors in summary.json instead of failing the whole build.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_records": args.out_dir / "train.records.jsonl",
        "validation_records": args.out_dir / "validation.records.jsonl",
        "train_sft": args.out_dir / "train.sft.jsonl",
        "validation_sft": args.out_dir / "validation.sft.jsonl",
        "annotation_queue": args.out_dir / "annotation_queue.jsonl",
    }
    stats: dict[str, Any] = {
        "schema": "trace_decomp_v1",
        "out_dir": str(args.out_dir.resolve()),
        "sources": args.sources,
        "max_records_per_source": args.max_records_per_source,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "max_trace_events": args.max_trace_events,
        "max_text_chars": args.max_text_chars,
        "counts": {
            "records": 0,
            "train": 0,
            "validation": 0,
            "annotation_queue": 0,
            "success": 0,
            "failure": 0,
            "unknown": 0,
        },
        "by_source": {},
        "errors": [],
        "files": {name: str(path) for name, path in paths.items()},
    }

    with ExitStack() as stack:
        handles = {name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in paths.items()}
        for source in args.sources:
            source_stats = stats["by_source"].setdefault(
                source,
                {"records": 0, "train": 0, "validation": 0, "annotation_queue": 0},
            )
            try:
                records = iter_hf_records(
                    source=source,
                    max_records=args.max_records_per_source,
                    max_trace_events=args.max_trace_events,
                    max_text_chars=args.max_text_chars,
                    streaming=args.streaming,
                )
                for record in records:
                    split = split_for_id(record["id"], args.validation_ratio, args.seed)
                    write_row(handles[f"{split}_records"], record)
                    write_row(handles[f"{split}_sft"], record_to_sft_row(record))
                    status = record.get("verification", {}).get("status", "unknown")
                    stats["counts"][status if status in {"success", "failure"} else "unknown"] += 1
                    if record.get("failure_analysis", {}).get("needs_annotation"):
                        write_row(handles["annotation_queue"], record_to_annotation_row(record))
                        stats["counts"]["annotation_queue"] += 1
                        source_stats["annotation_queue"] += 1
                    stats["counts"]["records"] += 1
                    stats["counts"][split] += 1
                    source_stats["records"] += 1
                    source_stats[split] += 1
            except Exception as exc:  # noqa: BLE001 - source availability varies across machines.
                error = {"source": source, "type": type(exc).__name__, "message": str(exc)}
                stats["errors"].append(error)
                if not args.continue_on_error:
                    raise

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
