#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from taskdecomp.data import (
    build_sft_rows,
    download_tasklama,
    make_atomic_negatives,
    read_taskbench,
    read_tasklama_jsonl,
    split_examples,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--negative-ratio", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-taskbench", action="store_true")
    parser.add_argument(
        "--taskbench-configs",
        nargs="+",
        default=["dailylifeapis", "huggingface", "multimedia"],
    )
    parser.add_argument("--taskbench-max-examples", type=int)
    parser.add_argument("--taskbench-validation-ratio", type=float, default=0.05)
    parser.add_argument("--taskbench-single-as-negative", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tasklama_dir = download_tasklama(args.raw_dir)
    splits = {
        "train": read_tasklama_jsonl(tasklama_dir / "train.jsonl"),
        "validation": read_tasklama_jsonl(tasklama_dir / "validation.jsonl"),
        "test": read_tasklama_jsonl(tasklama_dir / "test.jsonl"),
    }

    sft_splits = {}
    for split, examples in splits.items():
        neg_n = int(len(examples) * args.negative_ratio) if split != "test" else 0
        sft_splits[split] = examples + make_atomic_negatives(neg_n, args.seed)

    if args.include_taskbench:
        taskbench_examples = read_taskbench(
            configs=args.taskbench_configs,
            max_examples=args.taskbench_max_examples,
            single_as_negative=args.taskbench_single_as_negative,
        )
        taskbench_train, taskbench_validation = split_examples(
            taskbench_examples,
            validation_ratio=args.taskbench_validation_ratio,
            seed=args.seed,
        )
        sft_splits["train"].extend(taskbench_train)
        sft_splits["validation"].extend(taskbench_validation)
        write_jsonl(build_sft_rows(taskbench_train), args.out_dir / "taskbench.train.sft.jsonl")
        write_jsonl(build_sft_rows(taskbench_validation), args.out_dir / "taskbench.validation.sft.jsonl")
        print(
            "Added TaskBench examples: "
            f"{len(taskbench_train)} train, {len(taskbench_validation)} validation "
            f"from configs={','.join(args.taskbench_configs)}"
        )

    for split, examples in splits.items():
        write_jsonl(build_sft_rows(sft_splits[split]), args.out_dir / f"{split}.sft.jsonl")
        write_jsonl(
            [{"task": ex.task, "context": ex.context, "target": ex.__dict__} for ex in examples],
            args.out_dir / f"{split}.eval.jsonl",
        )

    print(f"Wrote processed files to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
