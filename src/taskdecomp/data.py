from __future__ import annotations

import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .prompts import training_messages
from .schemas import TaskExample

TASKLAMA_URL = "https://storage.googleapis.com/gresearch/tasklama/tasklama.zip"
TASKBENCH_CONFIGS = ("dailylifeapis", "huggingface", "multimedia")

ATOMIC_NEGATIVES = [
    "What is the capital of France?",
    "Translate hello into Spanish.",
    "Summarize this sentence in one line.",
    "Is water wet?",
    "Set a timer for ten minutes.",
    "Convert 12 inches to feet.",
    "Find the definition of photosynthesis.",
    "Choose a random number between 1 and 10.",
    "Tell me today's date.",
    "Classify this review as positive or negative.",
    "Extract the email address from this text.",
    "Rename this file to notes.txt.",
]


def download_tasklama(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "tasklama.zip"
    out_dir = raw_dir / "tasklama"
    if not zip_path.exists():
        urllib.request.urlretrieve(TASKLAMA_URL, zip_path)
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    return out_dir


def read_tasklama_jsonl(path: Path) -> list[TaskExample]:
    examples: list[TaskExample] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            context = "; ".join(a["assumption"] for a in row.get("assumptions", []))
            steps = [s["step"] for s in row.get("substeps", [])]
            deps = [(d["subtask1"], d["subtask2"]) for d in row.get("dependencies", [])]
            examples.append(
                TaskExample(
                    task=row["task"],
                    context=context,
                    decision="decompose",
                    subtasks=steps,
                    dependencies=deps,
                )
            )
    return examples


def _loads_maybe_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _clean_taskbench_step(step: str) -> str:
    text = str(step).strip()
    text = re.sub(r"^step\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^initial step\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def taskbench_row_to_example(row: dict[str, Any], single_as_negative: bool = True) -> TaskExample | None:
    instruction = str(row.get("instruction") or "").strip()
    if not instruction:
        return None

    n_tools = int(row.get("n_tools") or 0)
    row_type = str(row.get("type") or "")
    if single_as_negative and (n_tools <= 1 or row_type == "single"):
        return TaskExample(task=instruction, decision="no_decomposition")

    raw_steps = _loads_maybe_json(row.get("tool_steps"), [])
    steps = [_clean_taskbench_step(step) for step in raw_steps if str(step).strip()]
    if len(steps) < 2:
        return TaskExample(task=instruction, decision="no_decomposition")

    raw_nodes = _loads_maybe_json(row.get("tool_nodes"), [])
    node_to_step: dict[str, str] = {}
    if isinstance(raw_nodes, list):
        for i, node in enumerate(raw_nodes[: len(steps)]):
            if not isinstance(node, dict):
                continue
            task_name = str(node.get("task") or "").strip()
            if task_name and task_name not in node_to_step:
                node_to_step[task_name] = steps[i]

    raw_links = _loads_maybe_json(row.get("tool_links"), [])
    dependencies = []
    if isinstance(raw_links, list):
        for link in raw_links:
            if not isinstance(link, dict):
                continue
            source = str(link.get("source") or "").strip()
            target = str(link.get("target") or "").strip()
            before = node_to_step.get(source)
            after = node_to_step.get(target)
            if before and after and before != after:
                dependencies.append((before, after))

    context = f"TaskBench type: {row_type}; number of tools: {n_tools}"
    return TaskExample(
        task=instruction,
        context=context,
        decision="decompose",
        subtasks=steps,
        dependencies=dependencies,
    )


def read_taskbench(
    configs: Iterable[str] = TASKBENCH_CONFIGS,
    max_examples: int | None = None,
    single_as_negative: bool = True,
) -> list[TaskExample]:
    from datasets import load_dataset

    examples: list[TaskExample] = []
    for config in configs:
        dataset = load_dataset("microsoft/Taskbench", config, split="test")
        for row in dataset:
            example = taskbench_row_to_example(dict(row), single_as_negative=single_as_negative)
            if example is None:
                continue
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


def split_examples(
    examples: list[TaskExample],
    validation_ratio: float = 0.05,
    seed: int = 13,
) -> tuple[list[TaskExample], list[TaskExample]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    validation_n = max(1, int(len(shuffled) * validation_ratio)) if shuffled else 0
    return shuffled[validation_n:], shuffled[:validation_n]


def example_to_target(example: TaskExample) -> dict:
    if example.decision == "no_decomposition":
        return {
            "decision": "no_decomposition",
            "rationale": "The request is atomic or can be handled directly.",
            "subtasks": [],
            "dependencies": [],
        }

    ids = {step: f"s{i + 1}" for i, step in enumerate(example.subtasks)}
    return {
        "decision": "decompose",
        "rationale": "The request requires multiple ordered actions.",
        "subtasks": [{"id": ids[step], "text": step} for step in example.subtasks],
        "dependencies": [
            {"before": ids[a], "after": ids[b]}
            for a, b in example.dependencies
            if a in ids and b in ids and a != b
        ],
    }


def make_atomic_negatives(n: int, seed: int = 13) -> list[TaskExample]:
    rng = random.Random(seed)
    tasks = [rng.choice(ATOMIC_NEGATIVES) for _ in range(n)]
    return [TaskExample(task=t, decision="no_decomposition") for t in tasks]


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft_rows(examples: list[TaskExample]) -> list[dict]:
    rows = []
    for ex in examples:
        rows.append(
            {
                "task": ex.task,
                "context": ex.context,
                "decision": ex.decision,
                "messages": training_messages(ex.task, ex.context, example_to_target(ex)),
                "target": example_to_target(ex),
            }
        )
    return rows
