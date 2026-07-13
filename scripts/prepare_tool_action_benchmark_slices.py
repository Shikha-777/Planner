#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_capability_holdout import case, expected, inp
from prepare_capability_benchmark_slices import (
    first_user_content,
    prepare_swe_bench,
    prepare_webarena,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path("data/benchmarks/tool_action_slices")
DEFAULT_CONFIG_DIR = Path("configs")

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_TOOL_FILES = {
    "bfcl_simple_tool": "BFCL_v3_simple.json",
    "bfcl_multiple_tool": "BFCL_v3_multiple.json",
    "bfcl_parallel_tool": "BFCL_v3_parallel.json",
    "bfcl_rest_tool": "BFCL_v3_rest.json",
}
BFCL_IRRELEVANCE_FILE = "BFCL_v3_irrelevance.json"
VAKRA_TASK_FILES = [
    Path("external/vakra/environment/bpo/data/tasks.json"),
    Path("external/vakra/environment/bpo/data/tasks_edge_cases.json"),
]


def tool_action_expected() -> dict[str, Any]:
    return expected(
        inputs=[inp("tool action request", True, "none")],
        current=False,
        include_actions=["other"],
        include_caps=["select_and_execute_external_action"],
    )


def no_tool_control_expected() -> dict[str, Any]:
    return expected(
        exclude_actions=["other"],
        exclude_caps=["select_and_execute_external_action"],
    )


def wrap_tool_request(content: str) -> str:
    return (
        "Available external tool/API documentation is provided by the benchmark. "
        f"Plan the external tool/API action needed for this request: {content.strip()}"
    )


def prepare_bfcl_tool_file(
    name: str,
    repo_file: str,
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BFCL_REPO, repo_file, repo_type="dataset")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        seen = 0
        for raw_index, line in enumerate(handle):
            if not line.strip():
                continue
            if raw_index < offset:
                continue
            if seen >= limit:
                break
            item = json.loads(line)
            content = first_user_content(item.get("question")).strip()
            if not content:
                continue
            row = case(
                f"{name}_{raw_index + 1:04d}",
                "tool_external",
                wrap_tool_request(content),
                tool_action_expected(),
            )
            row.update(
                {
                    "source_dataset": "bfcl_tool",
                    "benchmark_profile": "tool_external",
                    "source_file": repo_file,
                    "source_record_id": item.get("id"),
                    "tool_doc_count": len(item.get("function") or []),
                }
            )
            rows.append(row)
            seen += 1
    return rows, {
        "source": BFCL_REPO,
        "file": repo_file,
        "offset": offset,
        "rows": len(rows),
    }


def prepare_bfcl_irrelevance(
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BFCL_REPO, BFCL_IRRELEVANCE_FILE, repo_type="dataset")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        seen = 0
        for raw_index, line in enumerate(handle):
            if not line.strip():
                continue
            if raw_index < offset:
                continue
            if seen >= limit:
                break
            item = json.loads(line)
            content = first_user_content(item.get("question")).strip()
            if not content:
                continue
            row = case(
                f"bfcl_irrelevance_no_tool_{raw_index + 1:04d}",
                "no_tool_control",
                content,
                no_tool_control_expected(),
            )
            row.update(
                {
                    "source_dataset": "bfcl_irrelevance",
                    "benchmark_profile": "no_tool_control",
                    "source_file": BFCL_IRRELEVANCE_FILE,
                    "source_record_id": item.get("id"),
                }
            )
            rows.append(row)
            seen += 1
    return rows, {
        "source": BFCL_REPO,
        "file": BFCL_IRRELEVANCE_FILE,
        "offset": offset,
        "rows": len(rows),
    }


def prepare_vakra_tool_actions(
    *,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in VAKRA_TASK_FILES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for group in data:
            for test_case in group.get("test_cases", []):
                calls = ((test_case.get("expected_output") or {}).get("tool_calls") or [])
                if calls:
                    candidates.append((path, test_case))

    rows: list[dict[str, Any]] = []
    for selected_index, (path, test_case) in enumerate(candidates[offset : offset + limit], start=offset):
        intent = str(test_case.get("intent") or "").strip()
        if not intent:
            continue
        row = case(
            f"vakra_bpo_tool_{selected_index + 1:04d}",
            "tool_external",
            wrap_tool_request(intent),
            tool_action_expected(),
        )
        row.update(
            {
                "source_dataset": "vakra",
                "benchmark_profile": "tool_external",
                "source_file": str(path),
                "source_record_id": test_case.get("name"),
                "difficulty": test_case.get("difficulty"),
            }
        )
        rows.append(row)
    return rows, {
        "source": "external/vakra",
        "files": [str(path) for path in VAKRA_TASK_FILES if path.exists()],
        "offset": offset,
        "candidate_rows": len(candidates),
        "rows": len(rows),
    }


def suite_entry(name: str, path: Path, dataset: str = "auto") -> dict[str, Any]:
    return {"name": name, "input": str(path), "dataset": dataset}


def suite_suffix(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return normalized or "tool_action"


def run_step(
    name: str,
    path: Path,
    prepare: Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> bool:
    try:
        rows, metadata = prepare()
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


def build_config(
    config_dir: Path,
    prepared: dict[str, tuple[Path, str]],
    *,
    suite_label: str,
) -> Path:
    suffix = suite_suffix(suite_label)
    config_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        suite_entry(name, path, dataset)
        for name, (path, dataset) in prepared.items()
    ]
    config = {
        "planner_mode": "rules_first",
        "model": "openai/gpt-oss-20b",
        "max_new_tokens": 1200,
        "gold_root": f"data/capability_planning/benchmark_suite_{suffix}",
        "output_root": f"results/capability_benchmark_benchmark_suite_{suffix}",
        "benchmarks": entries,
    }
    path = config_dir / f"capability_benchmark_suite_{suffix}.json"
    write_json(path, config)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare harder tool/action benchmark slices for capability planning."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--suite-label", default="tool_action_scaled")
    parser.add_argument("--bfcl-limit-per-file", type=int, default=25)
    parser.add_argument("--bfcl-offset", type=int, default=0)
    parser.add_argument("--bfcl-irrelevance-limit", type=int, default=25)
    parser.add_argument("--bfcl-irrelevance-offset", type=int, default=0)
    parser.add_argument("--vakra-limit", type=int, default=25)
    parser.add_argument("--vakra-offset", type=int, default=0)
    parser.add_argument("--swe-limit", type=int, default=50)
    parser.add_argument("--swe-offset", type=int, default=0)
    parser.add_argument("--webarena-limit", type=int, default=50)
    parser.add_argument("--webarena-offset", type=int, default=0)
    args = parser.parse_args()

    tool_dir = args.output_root / "tool_external"
    control_dir = args.output_root / "controls"
    code_web_dir = args.output_root / "code_web"
    manifest: list[dict[str, Any]] = []
    prepared: dict[str, tuple[Path, str]] = {}

    for name, repo_file in BFCL_TOOL_FILES.items():
        path = tool_dir / f"{name}.gold.jsonl"
        if run_step(
            name,
            path,
            lambda name=name, repo_file=repo_file: prepare_bfcl_tool_file(
                name,
                repo_file,
                limit=args.bfcl_limit_per_file,
                offset=args.bfcl_offset,
            ),
            manifest,
        ):
            prepared[name] = (path, "auto")

    path = control_dir / "bfcl_irrelevance_no_tool.gold.jsonl"
    if run_step(
        "bfcl_irrelevance_no_tool",
        path,
        lambda: prepare_bfcl_irrelevance(
            limit=args.bfcl_irrelevance_limit,
            offset=args.bfcl_irrelevance_offset,
        ),
        manifest,
    ):
        prepared["bfcl_irrelevance_no_tool"] = (path, "auto")

    path = tool_dir / "vakra_bpo_tool.gold.jsonl"
    if run_step(
        "vakra_bpo_tool",
        path,
        lambda: prepare_vakra_tool_actions(
            limit=args.vakra_limit,
            offset=args.vakra_offset,
        ),
        manifest,
    ):
        prepared["vakra_bpo_tool"] = (path, "auto")

    path = code_web_dir / "swe_bench_lite_real.jsonl"
    if run_step(
        "swe_bench_lite",
        path,
        lambda: prepare_swe_bench(args.swe_limit, args.swe_offset),
        manifest,
    ):
        prepared["swe_bench_lite"] = (path, "swe_bench")

    path = code_web_dir / "webarena_mini_real_wrapped.jsonl"
    if run_step(
        "webarena_mini",
        path,
        lambda: prepare_webarena(args.webarena_limit, args.webarena_offset),
        manifest,
    ):
        prepared["webarena_mini"] = (path, "webarena")

    config_path = build_config(
        args.config_dir,
        prepared,
        suite_label=args.suite_label,
    )
    manifest_path = args.output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "suite_label": args.suite_label,
            "prepared": manifest,
            "config": str(config_path),
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "prepared_count": sum(1 for item in manifest if item["status"] == "prepared"),
                "skipped_count": sum(1 for item in manifest if item["status"] == "skipped"),
                "config": str(config_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
