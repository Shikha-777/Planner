#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_source(spec: str) -> tuple[Path, int]:
    if ":" not in spec:
        return Path(spec), 0
    path, raw_limit = spec.rsplit(":", 1)
    return Path(path), int(raw_limit)


def sample_rows(path: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[: min(limit, len(rows))]
    for row in rows:
        row.setdefault("source_file", str(path))
    return rows


def build_split(specs: list[str], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for spec in specs:
        path, limit = parse_source(spec)
        selected = sample_rows(path, limit, rng)
        rows.extend(selected)
        manifest.append({"path": str(path), "limit": limit, "selected": len(selected)})
    rng.shuffle(rows)
    return rows, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mix message-style SFT JSONL files deterministically.")
    parser.add_argument("--train-source", action="append", default=[], help="Path or path:limit")
    parser.add_argument("--validation-source", action="append", default=[], help="Path or path:limit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if not args.train_source:
        raise SystemExit("At least one --train-source is required.")
    if not args.validation_source:
        raise SystemExit("At least one --validation-source is required.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_manifest = build_split(args.train_source, args.seed)
    validation_rows, validation_manifest = build_split(args.validation_source, args.seed + 1)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "validation.jsonl", validation_rows)
    summary = {
        "seed": args.seed,
        "train": {"rows": len(train_rows), "sources": train_manifest},
        "validation": {"rows": len(validation_rows), "sources": validation_manifest},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
