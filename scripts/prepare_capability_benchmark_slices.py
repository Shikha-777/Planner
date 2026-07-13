#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Callable


DEFAULT_OUTPUT_ROOT = Path("data/benchmarks/public_slices")
DEFAULT_CONFIG_DIR = Path("configs")

SUPERNI_TASKS = {
    "superni_summarize": {
        "repo_file": "test/task645_summarization_test.jsonl",
        "dataset": "superni",
        "source_wrapper": "summarize",
    },
    "superni_classify": {
        "repo_file": "test/task1344_glue_entailment_classification_test.jsonl",
        "dataset": "superni",
        "source_wrapper": "classify",
    },
    "superni_rewrite": {
        "repo_file": "test/task121_zest_text_modification_test.jsonl",
        "dataset": "superni",
        "source_wrapper": "rewrite",
    },
    "superni_extract": {
        "repo_file": "test/task1540_parsed_pdfs_summarization_test.jsonl",
        "dataset": "superni",
        "source_wrapper": "extract",
    },
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def limited_by_label(
    rows: list[dict[str, Any]],
    label_key: str,
    wanted: list[str],
    limit_per_label: int,
    offset_per_label: int = 0,
) -> list[dict[str, Any]]:
    counts = {label: 0 for label in wanted}
    seen = {label: 0 for label in wanted}
    selected: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get(label_key))
        if label not in counts:
            continue
        seen[label] += 1
        if seen[label] <= offset_per_label or counts[label] >= limit_per_label:
            continue
        selected.append(row)
        counts[label] += 1
        if all(count >= limit_per_label for count in counts.values()):
            break
    return selected


def prepare_clinc(
    limit_per_intent: int,
    offset_per_intent: int = 0,
    split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("clinc/clinc_oos", "small", split=split)
    labels = dataset.features["intent"].names
    wanted = ["weather", "exchange_rate", "definition", "translate", "oos"]
    wanted_ids = {label: labels.index(label) for label in wanted if label in labels}
    counts = {label: 0 for label in wanted_ids}
    seen = {label: 0 for label in wanted_ids}
    rows = []
    for item in dataset:
        label = labels[int(item["intent"])]
        if label not in wanted_ids:
            continue
        seen[label] += 1
        if seen[label] <= offset_per_intent or counts[label] >= limit_per_intent:
            continue
        rows.append(
            {
                "id": f"clinc_{label}_{seen[label]:03d}",
                "utterance": item["text"],
                "intent": label,
                "source_dataset": "clinc",
            }
        )
        counts[label] += 1
        if all(count >= limit_per_intent for count in counts.values()):
            break
    return rows, {
        "source": "clinc/clinc_oos",
        "config": "small",
        "split": split,
        "offset_per_intent": offset_per_intent,
        "counts": counts,
    }


def prepare_massive(
    limit_per_label: int,
    offset_per_label: int = 0,
    split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    repo_file = f"{split}/en.json.gz"
    path = hf_hub_download("mteb/amazon_massive_intent", repo_file, repo_type="dataset")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    wanted = ["weather_query", "qa_stock", "qa_definition", "email_sendemail"]
    selected = limited_by_label(rows, "label_text", wanted, limit_per_label, offset_per_label)
    out = [
        {
            "id": f"massive_{row['label_text']}_{index + 1:03d}",
            "utt": row["text"],
            "intent": row["label_text"],
            "source_dataset": "massive",
        }
        for index, row in enumerate(selected)
    ]
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["label_text"]] = counts.get(row["label_text"], 0) + 1
    return out, {
        "source": "mteb/amazon_massive_intent",
        "file": repo_file,
        "offset_per_label": offset_per_label,
        "counts": counts,
    }


def prepare_superni(
    repo_file: str,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Muennighoff/natural-instructions", repo_file, repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as handle:
        selected_index = 0
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if not line.strip():
                continue
            if selected_index >= limit:
                break
            item = json.loads(line)
            rows.append(
                {
                    "id": item.get("id") or f"{Path(repo_file).stem}_{index + 1:03d}",
                    "definition": item.get("definition"),
                    "input": item.get("inputs"),
                    "target": item.get("targets"),
                    "task_name": item.get("task_name") or Path(repo_file).stem,
                    "source_dataset": "superni",
                }
            )
            selected_index += 1
    return rows, {
        "source": "Muennighoff/natural-instructions",
        "file": repo_file,
        "offset": offset,
        "rows": len(rows),
    }


def prepare_bfcl_sql(limit: int, offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "BFCL_v3_sql.json",
        repo_type="dataset",
    )
    rows = []
    with open(path, encoding="utf-8") as handle:
        selected_index = 0
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if selected_index >= limit:
                break
            item = json.loads(line)
            content = first_user_content(item.get("question"))
            rows.append(
                {
                    "id": item.get("id") or f"bfcl_sql_{index + 1:03d}",
                    "question": f"Write the SQL query needed to answer this request: {content}",
                    "category": "sql",
                    "source_dataset": "bfcl",
                    "function": item.get("function"),
                }
            )
            selected_index += 1
    return rows, {
        "source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "file": "BFCL_v3_sql.json",
        "offset": offset,
        "rows": len(rows),
    }


def prepare_swe_bench(limit: int, offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    split = f"test[{offset}:{offset + limit}]" if offset else f"test[:{limit}]"
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    rows = []
    for item in dataset:
        rows.append(
            {
                "id": item["instance_id"],
                "problem_statement": f"Fix this repository issue:\n\n{item['problem_statement']}",
                "repo": item["repo"],
                "source_dataset": "swe_bench",
            }
        )
    return rows, {"source": "princeton-nlp/SWE-bench_Lite", "split": split, "rows": len(rows)}


def prepare_webarena(limit: int, offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    split = f"test[{offset}:{offset + limit}]" if offset else f"test[:{limit}]"
    dataset = load_dataset("WPRM/mini_benchmark_webarena", split=split)
    rows = []
    for item in dataset:
        start_url = item.get("start_url") or item.get("current_url") or ""
        intent = item.get("intent") or item.get("task") or ""
        rows.append(
            {
                "id": f"webarena_{item.get('task_id', len(rows))}_{item.get('step_id', 0)}",
                "task": f"Open {start_url} and complete this web task: {intent}",
                "intent": intent,
                "start_url": start_url,
                "source_dataset": "webarena",
            }
        )
    return rows, {"source": "WPRM/mini_benchmark_webarena", "split": split, "rows": len(rows)}


def local_missing_controls(variant: str = "default") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = {
        "default": [
            ("local_missing_resume_001", "Can you improve my resume summary?", "missing_source"),
            ("local_missing_chart_001", "Graph monthly signups by acquisition channel.", "missing_source"),
            (
                "local_draft_email_001",
                "Draft a short email saying onboarding is complete and support coverage starts Monday.",
                "self_contained_writing",
            ),
        ],
        "holdout_v2": [
            ("local_missing_bio_002", "Can you polish my LinkedIn bio?", "missing_source"),
            ("local_missing_revenue_plot_002", "Plot revenue by region.", "missing_source"),
            (
                "local_draft_note_002",
                "Write a brief note that the deploy finished and rollback is not needed.",
                "self_contained_writing",
            ),
        ],
    }
    selected = variants.get(variant, variants["default"])
    rows = [
        {
            "id": row_id,
            "utterance": utterance,
            "intent": intent,
            "source_dataset": "local_controls",
        }
        for row_id, utterance, intent in selected
    ]
    return rows, {"source": "local_controls", "variant": variant, "rows": len(rows)}


def first_user_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or value.get("text") or "")
    if isinstance(value, list):
        for item in value:
            content = first_user_content(item)
            if content:
                return content
    return ""


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
        return True
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


def suite_entry(name: str, path: Path, dataset: str, source_wrapper: str = "auto") -> dict[str, Any]:
    return {
        "name": name,
        "input": str(path),
        "dataset": dataset,
        "source_wrapper": source_wrapper,
    }


def suite_suffix(label: str | None) -> str:
    if not label:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"_{normalized}" if normalized else ""


def build_configs(
    output_root: Path,
    config_dir: Path,
    prepared: dict[str, Path],
    *,
    suite_label: str | None = None,
) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    source_intent_entries = []
    tool_code_entries = []
    suffix = suite_suffix(suite_label)

    for name in ["clinc_intent", "massive_intent", "local_intent_controls"]:
        if name in prepared:
            dataset = "auto" if name == "local_intent_controls" else name.split("_")[0]
            source_intent_entries.append(suite_entry(name, prepared[name], dataset))

    for name, spec in SUPERNI_TASKS.items():
        if name in prepared:
            source_intent_entries.append(
                suite_entry(name, prepared[name], spec["dataset"], spec["source_wrapper"])
            )

    if "bfcl_sql" in prepared:
        tool_code_entries.append(suite_entry("bfcl_sql", prepared["bfcl_sql"], "bfcl"))
    if "swe_bench_lite" in prepared:
        tool_code_entries.append(suite_entry("swe_bench_lite", prepared["swe_bench_lite"], "swe_bench"))
    if "webarena_mini" in prepared:
        tool_code_entries.append(suite_entry("webarena_mini", prepared["webarena_mini"], "webarena"))

    configs = [
        (
            f"capability_benchmark_suite_source_intent{suffix}.json",
            f"benchmark_suite_source_intent{suffix}",
            source_intent_entries,
        ),
        (
            f"capability_benchmark_suite_tool_code{suffix}.json",
            f"benchmark_suite_tool_code{suffix}",
            tool_code_entries,
        ),
        (
            f"capability_benchmark_suite_all{suffix}.json",
            f"benchmark_suite_all{suffix}",
            source_intent_entries + tool_code_entries,
        ),
    ]
    paths = []
    for filename, suite_name, entries in configs:
        config = {
            "planner_mode": "rules_first",
            "model": "openai/gpt-oss-20b",
            "max_new_tokens": 1200,
            "gold_root": f"data/capability_planning/{suite_name}",
            "output_root": f"results/capability_benchmark_{suite_name}",
            "benchmarks": entries,
        }
        path = config_dir / filename
        write_json(path, config)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare bounded public benchmark slices for capability-planning suite runs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--clinc-limit-per-intent", type=int, default=3)
    parser.add_argument("--clinc-offset-per-intent", type=int, default=0)
    parser.add_argument("--clinc-split", default="test")
    parser.add_argument("--massive-limit-per-label", type=int, default=3)
    parser.add_argument("--massive-offset-per-label", type=int, default=0)
    parser.add_argument("--massive-split", default="test")
    parser.add_argument("--superni-limit-per-task", type=int, default=3)
    parser.add_argument("--superni-offset-per-task", type=int, default=0)
    parser.add_argument("--bfcl-limit", type=int, default=4)
    parser.add_argument("--bfcl-offset", type=int, default=0)
    parser.add_argument("--swe-limit", type=int, default=4)
    parser.add_argument("--swe-offset", type=int, default=0)
    parser.add_argument("--webarena-limit", type=int, default=4)
    parser.add_argument("--webarena-offset", type=int, default=0)
    parser.add_argument(
        "--local-controls-variant",
        choices=["default", "holdout_v2"],
        default="default",
    )
    parser.add_argument(
        "--suite-label",
        help="Optional suffix for generated config, gold, and result directories.",
    )
    args = parser.parse_args()

    source_dir = args.output_root / "source_intent"
    tool_dir = args.output_root / "tool_code"
    manifest: list[dict[str, Any]] = []
    prepared: dict[str, Path] = {}

    steps: list[tuple[str, Path, Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]]]] = [
        (
            "clinc_intent",
            source_dir / "clinc_intent_real.jsonl",
            lambda: prepare_clinc(
                args.clinc_limit_per_intent,
                args.clinc_offset_per_intent,
                args.clinc_split,
            ),
        ),
        (
            "massive_intent",
            source_dir / "massive_intent_en_real.jsonl",
            lambda: prepare_massive(
                args.massive_limit_per_label,
                args.massive_offset_per_label,
                args.massive_split,
            ),
        ),
        (
            "local_intent_controls",
            source_dir / "local_intent_controls.jsonl",
            lambda: local_missing_controls(args.local_controls_variant),
        ),
    ]
    for name, spec in SUPERNI_TASKS.items():
        steps.append(
            (
                name,
                source_dir / f"{name}_real.jsonl",
                lambda spec=spec: prepare_superni(
                    spec["repo_file"],
                    args.superni_limit_per_task,
                    args.superni_offset_per_task,
                ),
            )
        )
    steps.extend(
        [
            (
                "bfcl_sql",
                tool_dir / "bfcl_sql_real_wrapped.jsonl",
                lambda: prepare_bfcl_sql(args.bfcl_limit, args.bfcl_offset),
            ),
            (
                "swe_bench_lite",
                tool_dir / "swe_bench_lite_real.jsonl",
                lambda: prepare_swe_bench(args.swe_limit, args.swe_offset),
            ),
            (
                "webarena_mini",
                tool_dir / "webarena_mini_real_wrapped.jsonl",
                lambda: prepare_webarena(args.webarena_limit, args.webarena_offset),
            ),
        ]
    )

    for name, path, prepare in steps:
        if run_step(name, path, prepare, manifest):
            prepared[name] = path

    config_paths = build_configs(
        args.output_root,
        args.config_dir,
        prepared,
        suite_label=args.suite_label,
    )
    manifest_path = args.output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "suite_label": args.suite_label,
            "offsets": {
                "clinc_split": args.clinc_split,
                "clinc_offset_per_intent": args.clinc_offset_per_intent,
                "massive_split": args.massive_split,
                "massive_offset_per_label": args.massive_offset_per_label,
                "superni_offset_per_task": args.superni_offset_per_task,
                "bfcl_offset": args.bfcl_offset,
                "swe_offset": args.swe_offset,
                "webarena_offset": args.webarena_offset,
                "local_controls_variant": args.local_controls_variant,
            },
            "prepared": manifest,
            "configs": [str(path) for path in config_paths],
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "prepared_count": sum(1 for item in manifest if item["status"] == "prepared"),
                "skipped_count": sum(1 for item in manifest if item["status"] == "skipped"),
                "configs": [str(path) for path in config_paths],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
