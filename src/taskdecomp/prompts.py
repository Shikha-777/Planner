from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a task decomposition model.
Return only valid JSON.
First decide whether decomposition is useful. Decomposition is useful only when the
user goal naturally requires multiple meaningful actions, ordering, coordination, or
intermediate deliverables. If the goal is atomic, a single lookup, a preference
question, a yes/no question, or already a concrete action, return no_decomposition.
When decomposition is useful, produce concise actionable subtasks and temporal
dependencies as a DAG."""


def user_prompt(task: str, context: str | None = None) -> str:
    payload = {"task": task, "context": context or ""}
    return (
        "Decompose this task when useful. Use this exact JSON schema:\n"
        "{"
        '"decision":"decompose|no_decomposition",'
        '"rationale":"short reason",'
        '"subtasks":[{"id":"s1","text":"verb-led action"}],'
        '"dependencies":[{"before":"s1","after":"s2"}]'
        "}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def training_messages(task: str, context: str, target_json: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(task, context)},
        {"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)},
    ]
