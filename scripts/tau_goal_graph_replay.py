#!/usr/bin/env python
"""Replay logged tau planner states without a user simulator or environment step."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_replay_cases(debug_path: Path, steps: set[int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = debug_path.parent / "goal_graph_replay_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing replay manifest beside {debug_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tools = manifest.get("goal_graph_tools")
    if not isinstance(tools, list):
        raise ValueError("replay manifest does not contain goal_graph_tools")

    cases: list[dict[str, Any]] = []
    for line in debug_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = row.get("step")
        if not isinstance(step, int) or (steps is not None and step not in steps):
            continue
        result = row.get("goal_graph_result")
        if not isinstance(result, dict):
            continue
        request = result.get("planning_request")
        binding_request = result.get("binding_request")
        if not isinstance(request, str) or not isinstance(binding_request, str):
            continue
        cases.append(
            {
                "step": step,
                "planning_request": request,
                "binding_request": binding_request,
                "execution_history": result.get("stateful_execution_history") or [],
                "stateful_goal_ledger": result.get("stateful_goal_ledger_input") or {},
                "original_action": row.get("final_action"),
            }
        )
    return tools, cases


def replay_case(
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    case: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    from goal_graph_eval_common import plan_and_compile_goal_graph
    from run_gptoss_capability_plan import generate_text

    result = plan_and_compile_goal_graph(
        model,
        tokenizer,
        generate_text,
        case["planning_request"],
        tools,
        max_new_tokens=max_new_tokens,
        repair_attempts=1,
        allow_side_effects=True,
        use_binder_fallback=True,
        planner_mode="stepwise",
        stateful=True,
        binding_request=case["binding_request"],
        execution_history=case["execution_history"],
        stateful_goal_ledger=case["stateful_goal_ledger"],
        stateful_goal_ledger_required=True,
        stateful_semantic_only=True,
        stateful_semantic_review=True,
    )
    plan = result.get("tool_binding_plan") or {}
    return {
        "source_step": case["step"],
        "original_action": case["original_action"],
        "tool_decision": plan.get("tool_decision"),
        "calls": plan.get("calls") or [],
        "verification_ok": result.get("verification_ok"),
        "parse_error": result.get("parse_error"),
        "diagnostic_codes": result.get("diagnostic_codes") or [],
        "review": result.get("stateful_semantic_review") or {},
    }


def parse_steps(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def write_replay_results(
    model: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    max_new_tokens: int,
    output: Path,
) -> None:
    """Write each replay immediately so long later cases do not hide earlier ones."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            started_at = time.monotonic()
            replay = replay_case(model, tokenizer, tools, case, max_new_tokens)
            replay["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            handle.write(json.dumps(replay, ensure_ascii=False) + "\n")
            handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-jsonl", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--steps", help="Comma-separated source steps to replay.")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tools, cases = load_replay_cases(args.debug_jsonl, parse_steps(args.steps))
    if not cases:
        raise SystemExit("no replayable cases found; enable TAU_GOAL_GRAPH_REPLAY_TRACE in a fresh trace")

    from run_gptoss_capability_plan import load_model

    model, tokenizer = load_model(args.model)
    write_replay_results(model, tokenizer, tools, cases, args.max_new_tokens, args.output)


if __name__ == "__main__":
    main()
