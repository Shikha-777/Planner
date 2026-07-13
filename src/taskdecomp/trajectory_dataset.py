from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "trace_decomp_v1"

HF_SOURCES: dict[str, dict[str, str]] = {
    "swe-agent": {"dataset": "nebius/SWE-agent-trajectories", "split": "train"},
    "openhands": {"dataset": "nebius/SWE-rebench-openhands-trajectories", "split": "train"},
    "mind2web": {"dataset": "osunlp/Mind2Web", "split": "train"},
}


def clean_text(value: Any, max_chars: int = 1200) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def stable_hash(value: Any, prefix: str = "") -> str:
    digest = hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    hashed = digest.hexdigest()[:16]
    return f"{prefix}{hashed}" if prefix else hashed


def split_for_id(record_id: str, validation_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "validation" if bucket < validation_ratio else "train"


def chain_dependencies(subtask_ids: list[str]) -> list[dict[str, str]]:
    return [{"before": before, "after": after} for before, after in zip(subtask_ids, subtask_ids[1:])]


def target_from_steps(
    steps: list[str],
    rationale: str,
    success_criteria: list[str] | None = None,
) -> dict[str, Any]:
    subtasks = [
        {
            "id": f"s{index + 1}",
            "text": clean_text(step, 320),
            "success_criteria": clean_text(
                success_criteria[index] if success_criteria and index < len(success_criteria) else "",
                320,
            ),
        }
        for index, step in enumerate(steps)
        if clean_text(step, 320)
    ]
    ids = [subtask["id"] for subtask in subtasks]
    return {
        "decision": "decompose" if len(subtasks) > 1 else "no_decomposition",
        "rationale": rationale,
        "subtasks": subtasks,
        "dependencies": chain_dependencies(ids) if len(ids) > 1 else [],
    }


def trace_training_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    target = {
        "decision": record["target"]["decision"],
        "failure_analysis": record["failure_analysis"],
        "corrected_decomposition": {
            "rationale": record["target"]["rationale"],
            "subtasks": record["target"]["subtasks"],
            "dependencies": record["target"]["dependencies"],
        },
        "verification": record["verification"],
    }
    user_payload = {
        "task": record["task"],
        "domain": record.get("domain", ""),
        "context": record.get("context", ""),
        "trace_summary": record.get("trace_summary", []),
        "verification": record.get("verification", {}),
    }
    return [
        {
            "role": "system",
            "content": (
                "You repair or produce task decompositions from execution traces. "
                "Return only valid JSON. Include a failure_analysis object, a corrected_decomposition "
                "with concise subtasks and dependencies, and verification signals."
            ),
        },
        {
            "role": "user",
            "content": "Create the best decomposition for this trace:\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2),
        },
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def extract_issue_text(text: str, max_chars: int = 1000) -> str:
    patterns = [
        r"<issue_description>\s*(.*?)\s*</issue_description>",
        r"ISSUE:\s*(.*?)(?:\n\nWe need|$)",
        r"Consider the following issue description:\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_text(match.group(1), max_chars)
    return clean_text(text, max_chars)


def first_user_text(trajectory: Iterable[dict[str, Any]]) -> str:
    for message in trajectory:
        role = str(message.get("role") or "").lower()
        if role in {"user", "human"}:
            return str(message.get("content") or message.get("text") or "")
    return ""


def extract_changed_files(patch: str, limit: int = 6) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch or "", flags=re.MULTILINE):
        candidate = match.group(2).strip()
        if candidate and candidate not in files:
            files.append(candidate)
        if len(files) >= limit:
            break
    return files


def code_block_summary(text: str, max_chars: int = 420) -> str:
    blocks = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, flags=re.DOTALL)
    if blocks:
        block = "\n".join(line for line in blocks[0].splitlines() if line.strip())
        return clean_text(f"Proposed command/action: {block}", max_chars)
    first_sentence = re.split(r"(?<=[.!?])\s+", clean_text(text, max_chars), maxsplit=1)[0]
    return clean_text(first_sentence, max_chars)


def observation_summary(text: str, max_chars: int = 420) -> str:
    cleaned = clean_text(text, max_chars)
    lower = cleaned.lower()
    if "traceback" in lower:
        return clean_text(f"Observation: traceback/error reported. {cleaned}", max_chars)
    if "failed" in lower or "error" in lower:
        return clean_text(f"Observation: failure signal. {cleaned}", max_chars)
    if "passed" in lower or "applied patch" in lower:
        return clean_text(f"Observation: success signal. {cleaned}", max_chars)
    return clean_text(f"Observation: {cleaned}", max_chars)


def summarize_code_trajectory(
    trajectory: list[dict[str, Any]],
    max_events: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, message in enumerate(trajectory):
        role = str(message.get("role") or "").lower()
        if role == "system":
            continue
        text = str(message.get("content") or message.get("text") or "")
        if not text.strip():
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                text = f"Tool call: {json.dumps(tool_calls, ensure_ascii=False)[:max_chars]}"
            else:
                continue
        if role in {"ai", "assistant"}:
            kind = "agent_action"
            summary = code_block_summary(text, max_chars)
        elif role in {"tool", "function"}:
            kind = "tool_observation"
            summary = observation_summary(text, max_chars)
        else:
            kind = "environment_observation" if events else "user_request"
            summary = extract_issue_text(text, max_chars) if not events else observation_summary(text, max_chars)
        events.append({"index": index, "role": "assistant" if role == "ai" else role, "kind": kind, "text": summary})
        if len(events) >= max_events:
            break
    return events


def coding_steps(repo: str | None, changed_files: list[str]) -> tuple[list[str], list[str]]:
    repo_hint = f" in {repo}" if repo else ""
    files_hint = ", ".join(changed_files[:4]) if changed_files else "the files implicated by the trace"
    steps = [
        f"Inspect the issue, repository layout, and reproduction context{repo_hint}.",
        f"Locate the code paths and tests connected to {files_hint}.",
        "Define the behavioral contract the fix must satisfy, including edge cases from the issue.",
        f"Implement the minimal code and test changes in {files_hint}.",
        "Run targeted verification and inspect any failing logs.",
        "Review the final patch against the issue and submit only after the verifier criteria are met.",
    ]
    criteria = [
        "The issue is restated with enough context to guide implementation.",
        "Relevant files, APIs, and existing tests are identified.",
        "Expected behavior and regression risk are explicit.",
        "The patch is scoped and includes test or reproduction coverage when possible.",
        "Verification commands produce passing or diagnostically useful output.",
        "The final patch matches the issue and avoids unrelated churn.",
    ]
    return steps, criteria


def verification_status(success: bool | None, signals: dict[str, Any]) -> dict[str, Any]:
    if success is True:
        status = "success"
    elif success is False:
        status = "failure"
    else:
        status = "unknown"
    return {"status": status, "signals": signals}


def failure_reason_for_code(source: str, row: dict[str, Any], success: bool) -> str:
    if success:
        return "The observed trajectory is treated as a successful decomposition because the source verifier marked it resolved."
    if source == "openhands":
        gen_tests_correct = row.get("gen_tests_correct")
        pred_passes_gen_tests = row.get("pred_passes_gen_tests")
        if pred_passes_gen_tests and not row.get("resolved"):
            return (
                "The patch passed generated tests but did not resolve the benchmark instance, "
                "so the plan likely missed hidden requirements or overfit local checks."
            )
        if gen_tests_correct == 0:
            return "Generated tests were not correct and the benchmark verifier marked the issue unresolved."
    logs = clean_text(row.get("eval_logs", ""), 900)
    lower = logs.lower()
    if "failed" in lower:
        return "The verifier logs contain failing checks after patch application."
    if "error" in lower or "traceback" in lower:
        return "The verifier logs contain an error or traceback after patch application."
    return "The source verifier marked the trajectory unresolved; a human or judge model should annotate the specific planning failure."


def make_code_record(
    source: str,
    row: dict[str, Any],
    index: int,
    max_trace_events: int,
    max_text_chars: int,
) -> dict[str, Any]:
    trajectory = row.get("trajectory") or []
    user_text = first_user_text(trajectory)
    task = extract_issue_text(user_text, 1200) or clean_text(row.get("instance_id"), 300)
    patch = str(row.get("model_patch") or row.get("generated_patch") or "")
    changed_files = extract_changed_files(patch)
    repo = row.get("repo")
    if source == "swe-agent":
        success = bool(row.get("target"))
        source_id = str(row.get("instance_id") or index)
        signals = {
            "instance_id": source_id,
            "model_name": row.get("model_name"),
            "exit_status": row.get("exit_status"),
            "changed_files": changed_files,
            "eval_log_excerpt": clean_text(row.get("eval_logs", ""), 600),
        }
    else:
        success = bool(row.get("resolved"))
        source_id = str(row.get("trajectory_id") or row.get("instance_id") or index)
        signals = {
            "instance_id": row.get("instance_id"),
            "repo": repo,
            "exit_status": row.get("exit_status"),
            "resolved": row.get("resolved"),
            "gen_tests_correct": row.get("gen_tests_correct"),
            "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
            "changed_files": changed_files,
        }

    steps, criteria = coding_steps(str(repo) if repo else None, changed_files)
    target = target_from_steps(
        steps,
        "Coding trajectories decompose best into issue understanding, localization, contract definition, implementation, verification, and final review.",
        criteria,
    )
    record_id = f"{source}:{source_id}"
    status = verification_status(success, signals)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "source": source,
        "source_id": source_id,
        "domain": "software_engineering",
        "task": task,
        "context": clean_text(f"Repository: {repo or 'unknown'}; changed files: {', '.join(changed_files) or 'unknown'}", 600),
        "trace_summary": summarize_code_trajectory(trajectory, max_trace_events, max_text_chars),
        "verification": status,
        "failure_analysis": {
            "observed_status": status["status"],
            "needs_annotation": not success,
            "reason": failure_reason_for_code(source, row, success),
        },
        "target": target,
    }


def clean_web_action(action_repr: str) -> str:
    text = clean_text(action_repr, 360)
    match = re.match(r"^\[(?P<element>[^\]]+)\]\s*(?P<label>.*?)\s*->\s*(?P<op>[A-Z_]+)(?::\s*(?P<value>.*))?$", text)
    if not match:
        return f"Perform web action: {text}"
    element = clean_text(match.group("element"), 80) or "element"
    label = clean_text(match.group("label"), 120) or element
    operation = match.group("op")
    value = clean_text(match.group("value"), 160)
    if operation == "CLICK":
        return f"Click {label}."
    if operation == "TYPE":
        return f"Type {value} into {label}."
    if operation == "SELECT":
        return f"Set {label} to {value}."
    return f"Use {label} with operation {operation}{(': ' + value) if value else ''}."


def make_mind2web_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    source_id = str(row.get("annotation_id") or index)
    raw_actions = [str(item) for item in row.get("action_reprs") or [] if str(item).strip()]
    steps = [clean_web_action(action) for action in raw_actions]
    criteria = ["The browser reaches the same state as the demonstrated action." for _ in steps]
    target = target_from_steps(
        steps,
        "Mind2Web provides human-demonstrated web workflows, so the demonstrated actions are used as the corrected decomposition.",
        criteria,
    )
    trace_summary = [
        {"index": idx, "role": "demonstrator", "kind": "web_action", "text": clean_text(action, 420)}
        for idx, action in enumerate(raw_actions)
    ]
    verification = verification_status(
        True,
        {
            "annotation_id": source_id,
            "website": row.get("website"),
            "domain": row.get("domain"),
            "subdomain": row.get("subdomain"),
            "source_kind": "human_demonstration",
        },
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"mind2web:{source_id}",
        "source": "mind2web",
        "source_id": source_id,
        "domain": clean_text(f"{row.get('domain') or ''}/{row.get('subdomain') or ''}".strip("/"), 200),
        "task": clean_text(row.get("confirmed_task"), 1000),
        "context": clean_text(f"Website: {row.get('website') or 'unknown'}", 400),
        "trace_summary": trace_summary,
        "verification": verification,
        "failure_analysis": {
            "observed_status": "success",
            "needs_annotation": False,
            "reason": "The workflow is a human-demonstrated successful web-action sequence.",
        },
        "target": target,
    }


def record_to_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "source": record["source"],
        "domain": record.get("domain", ""),
        "task": record["task"],
        "context": record.get("context", ""),
        "verification_status": record.get("verification", {}).get("status"),
        "messages": trace_training_messages(record),
        "target": record["target"],
        "failure_analysis": record["failure_analysis"],
    }


def record_to_annotation_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "source": record["source"],
        "task": record["task"],
        "context": record.get("context", ""),
        "trace_summary": record.get("trace_summary", []),
        "verification": record.get("verification", {}),
        "failure_analysis_candidate": record.get("failure_analysis", {}),
        "annotation_schema": {
            "bad_decomposition": "Describe the specific decomposition or planning mistake.",
            "failure_reason": "Explain why the verifier failed, using trace evidence.",
            "corrected_decomposition": {
                "subtasks": [{"id": "s1", "text": "verb-led corrected subtask", "success_criteria": "checkable criterion"}],
                "dependencies": [{"before": "s1", "after": "s2"}],
            },
        },
    }


def iter_hf_records(
    source: str,
    max_records: int,
    max_trace_events: int,
    max_text_chars: int,
    split: str | None = None,
    streaming: bool = True,
) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    if source not in HF_SOURCES:
        raise ValueError(f"Unsupported source {source!r}. Choose from: {', '.join(sorted(HF_SOURCES))}")
    info = HF_SOURCES[source]
    dataset_split = split or info["split"]
    dataset = load_dataset(info["dataset"], split=dataset_split, streaming=streaming)
    for index, row in enumerate(dataset):
        if max_records and index >= max_records:
            break
        raw = dict(row)
        if source == "mind2web":
            yield make_mind2web_record(raw, index)
        else:
            yield make_code_record(source, raw, index, max_trace_events, max_text_chars)
