#!/usr/bin/env python3
"""Prepare a small executable Live API-Bench/VAKRA capability-1 subset.

The public VAKRA data release contains the executable SLOT/SEL-style BI API
task data and SQLite databases. This script downloads selected domains and
regenerates the MCP universe mapping from released gold trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


SELECTION_ONLY_TOOLS = {
    "select_data_equal_to",
    "select_data_not_equal_to",
    "select_data_greater_than",
    "select_data_less_than",
    "select_data_greater_than_equal_to",
    "select_data_less_than_equal_to",
    "select_data_contains",
    "select_data_like",
    "sort_data_ascending",
    "sort_data_descending",
    "compute_data_min",
    "compute_data_max",
    "compute_data_sum",
    "compute_data_mean",
    "compute_data_count",
    "compute_data_std",
    "compute_data_argmin",
    "compute_data_argmax",
    "truncate",
    "transform_data_to_substring",
    "transform_data_to_absolute_value",
    "transform_data_to_datetime_part",
}

SLOT_FILLING_ONLY_TOOLS = {
    "filter_data",
    "retrieve_data",
    "sort_data",
    "aggregate_data",
    "transform_data",
}


def infer_server_type(tool_calls: list[dict[str, Any]]) -> str:
    for call in tool_calls:
        name = call.get("name", "")
        if name in SELECTION_ONLY_TOOLS or name.startswith("get_"):
            return "selection"
        if name in SLOT_FILLING_ONLY_TOOLS:
            return "slot_filling"
    return "slot_filling"


def build_mapping(vakra_root: Path, split: str, domains: list[str]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    output_dir = vakra_root / "data" / split / "capability_1_bi_apis" / "output"
    for domain in domains:
        path = output_dir / f"{domain}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing released gold output: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            turns = rec.get("output") or []
            if not turns:
                continue
            turn = turns[0]
            seq = turn.get("sequence") or {}
            tool_calls = seq.get("tool_call") or []
            init_call = next((c for c in tool_calls if c.get("name") == "initialize_active_data"), None)
            if not init_call:
                continue
            uuid = str(rec["uuid"])
            mapping[uuid] = {
                "domain": str(rec.get("domain") or domain),
                "server_type": infer_server_type(tool_calls),
                "init_args": init_call.get("arguments") or {},
                "query": str(turn.get("query") or ""),
            }
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vakra-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="ibm-research/VAKRA")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    args.vakra_root.mkdir(parents=True, exist_ok=True)
    data_root = args.vakra_root / "data"
    patterns: list[str] = []
    for domain in args.domains:
        patterns.extend(
            [
                f"databases/{domain}/**",
                f"{args.split}/capability_1_bi_apis/input/{domain}.json",
                f"{args.split}/capability_1_bi_apis/output/{domain}.json",
            ]
        )

    if not args.skip_download:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(data_root),
            allow_patterns=patterns,
        )

    mapping = build_mapping(args.vakra_root, args.split, args.domains)
    mapping_path = args.vakra_root / "environment" / "configs" / "mcp_tool_universe_id_mapping.yaml"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(yaml.safe_dump(mapping, sort_keys=True, allow_unicode=True), encoding="utf-8")

    counts: dict[str, int] = {}
    for entry in mapping.values():
        key = f"{entry['domain']}:{entry['server_type']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"mapping": str(mapping_path), "universes": len(mapping), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
