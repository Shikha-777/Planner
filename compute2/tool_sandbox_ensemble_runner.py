#!/usr/bin/env python3
"""Run Apple ToolSandbox scenarios with the local GPT-OSS ensemble.

This runner keeps the first ToolSandbox integration deliberately small:
single-turn scenarios can be evaluated without an external GPT user simulator by
ending the conversation after the agent gives a final user-facing response.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import traceback
import uuid
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from tau_ensemble_agent import (
    DEFAULT_EXECUTOR,
    DEFAULT_PLANNER,
    DEFAULT_RECOVERY,
    GPTOSSAdapterBank,
    parse_action_text,
    short_json,
)
from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.scenarios import named_scenarios


def resolve_selected_scenarios(names: list[str], preferred_tool_backend: ToolBackend) -> dict[str, Any]:
    scenarios = named_scenarios(preferred_tool_backend=preferred_tool_backend)
    selected = {name: scenarios[name] for name in names if name in scenarios}
    missing = sorted(set(names) - set(selected))
    if missing:
        raise KeyError(f"Unknown ToolSandbox scenarios: {missing}")
    return selected


def category_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for category in result["categories"]:
            grouped[str(category)]["similarity"].append(float(result["similarity"]))
            grouped[str(category)]["turn_count"].append(float(result["turn_count"]))
        grouped["ALL_CATEGORIES"]["similarity"].append(float(result["similarity"]))
        grouped["ALL_CATEGORIES"]["turn_count"].append(float(result["turn_count"]))
    return {
        category: {metric: sum(values) / len(values) for metric, values in metrics.items()}
        for category, metrics in grouped.items()
    }


def write_summary(results: list[dict[str, Any]], output_directory: Path) -> None:
    payload = {
        "per_scenario_results": [
            {**result, "categories": [str(category) for category in result["categories"]]}
            for result in results
        ],
        "category_aggregated_results": category_summary(results),
    }
    with (output_directory / "result_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


class AutoEndUser(BaseRole):
    """Deterministic user for single-turn ToolSandbox smoke tests."""

    role_type: RoleType = RoleType.USER
    model_name = "auto_end_user"

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return
        self.add_messages(
            [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content="print(repr(end_conversation()))",
                )
            ]
        )


class ToolSandboxEnsembleAgent(BaseRole):
    """ToolSandbox role backed by the same GPT-OSS adapter bank used on TauBench."""

    role_type: RoleType = RoleType.AGENT
    model_name = "local-gptoss-ensemble"

    def __init__(self) -> None:
        self.bank = GPTOSSAdapterBank(
            os.environ.get("TAU_ENSEMBLE_EXECUTOR_ADAPTER", DEFAULT_EXECUTOR),
            os.environ.get("TAU_ENSEMBLE_PLANNER_ADAPTER", DEFAULT_PLANNER),
            os.environ.get("TAU_ENSEMBLE_RECOVERY_ADAPTER", DEFAULT_RECOVERY),
        )
        self.max_action_tokens = int(os.environ.get("TOOLSANDBOX_ACTION_TOKENS", "700"))
        self.max_plan_tokens = int(os.environ.get("TOOLSANDBOX_PLAN_TOKENS", "500"))

    def _transcript(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for message in messages[-18:]:
            if message.sender == RoleType.SYSTEM and message.recipient == RoleType.AGENT:
                continue
            sender = str(message.sender).split(".")[-1].lower()
            recipient = str(message.recipient).split(".")[-1].lower()
            content = str(message.content)
            if len(content) > 4000:
                content = content[:4000] + "...<truncated>"
            lines.append(f"{sender}->{recipient}: {content}")
        return "\n".join(lines)

    def _tool_docs(self, tools: dict[str, Callable[..., Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "doc": getattr(tool, "__doc__", "") or "",
                "signature": str(getattr(tool, "__annotations__", {})),
            }
            for name, tool in tools.items()
        ]

    def _agent_messages(
        self,
        messages: list[Message],
        openai_tools: list[dict[str, Any]],
        available_tools: dict[str, Callable[..., Any]],
        plan: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, str]]:
        system = (
            "You are a tool-using phone assistant inside Apple ToolSandbox. "
            "Choose exactly one next action. If a tool is needed, emit JSON "
            "{\"name\": tool_name, \"arguments\": {...}} using one available tool. "
            "If the task is complete or you need to speak to the user, emit JSON "
            "{\"name\":\"respond\",\"arguments\":{\"content\":\"...\"}}. "
            "Do not invent tools, contacts, IDs, or argument values."
        )
        user = {
            "planner_state": plan or {},
            "available_tools": self._tool_docs(available_tools),
            "openai_tool_schema": openai_tools,
            "transcript": self._transcript(messages),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": short_json(user, limit=30000)},
        ]

    def _plan(self, messages: list[Message], available_tools: dict[str, Callable[..., Any]]) -> dict[str, Any]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the ToolSandbox state and recommend the next single "
                    "tool call or response. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": short_json(
                    {
                        "available_tools": self._tool_docs(available_tools),
                        "transcript": self._transcript(messages),
                        "schema": {
                            "known_facts": "object",
                            "next_action": {"name": "tool or respond", "arguments": "object"},
                        },
                    },
                    limit=26000,
                ),
            },
        ]
        result = self.bank.generate(
            self.bank.planner_adapter,
            prompt,
            max_new_tokens=self.max_plan_tokens,
            temperature=0.0,
        )
        try:
            parsed = json.loads(result.text.split("<|", 1)[0].strip())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {"raw": result.text[:1000]}

    def _coerce_tool_name(self, name: str, available_tool_names: set[str]) -> str:
        if name in available_tool_names:
            return name
        match = get_close_matches(name, list(available_tool_names), n=1, cutoff=0.84)
        return match[0] if match else name

    def _add_tool_call(self, name: str, arguments: dict[str, Any], available_tool_names: set[str]) -> None:
        current_context = get_current_context()
        execution_facing_name = current_context.get_execution_facing_tool_name(name)
        tool_id = f"call_{uuid.uuid4().hex[:12]}"
        tool_call = ChatCompletionMessageToolCall(
            id=tool_id,
            type="function",
            function=Function(name=name, arguments=json.dumps(arguments)),
        )
        self.add_messages(
            [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=openai_tool_call_to_python_code(
                        tool_call,
                        available_tool_names,
                        execution_facing_tool_name=execution_facing_name,
                    ),
                    openai_tool_call_id=tool_id,
                    openai_function_name=name,
                )
            ]
        )

    def respond(self, ending_index: Optional[int] = None) -> None:
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return

        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools)
        openai_tools = convert_to_openai_tools(available_tools)
        plan = self._plan(messages, available_tools)
        result = self.bank.generate(
            self.bank.executor_adapter,
            self._agent_messages(messages, openai_tools, available_tools, plan=plan),
            tools=openai_tools,
            max_new_tokens=self.max_action_tokens,
            temperature=0.0,
        )
        action = parse_action_text(result.text)
        name = self._coerce_tool_name(action.name, available_tool_names)
        kwargs = action.kwargs if isinstance(action.kwargs, dict) else {}

        if name in available_tool_names:
            self._add_tool_call(name, kwargs, available_tool_names)
            return

        content = (
            kwargs.get("content")
            or kwargs.get("message")
            or kwargs.get("response")
            or result.text.strip()
            or "Done."
        )
        self.add_messages(
            [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=str(content),
                )
            ]
        )


def run_one(name: str, scenario: Any, output_directory: Path, agent: ToolSandboxEnsembleAgent) -> dict[str, Any]:
    roles = {
        RoleType.USER: AutoEndUser(),
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        RoleType.AGENT: agent,
    }
    try:
        result = scenario.play_and_evaluate(
            roles=roles,
            output_directory=output_directory,
            scenario_name=name,
        )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": None,
            "exception_type": None,
            "milestone_similarity": result.evaluation_result.milestone_similarity,
            "minefield_similarity": result.evaluation_result.minefield_similarity,
            "similarity": result.evaluation_result.similarity,
            "turn_count": result.evaluation_result.turn_count,
            "milestone_mapping": result.evaluation_result.milestone_mapping,
            "minefield_mapping": result.evaluation_result.minefield_mapping,
        }
    except Exception as exc:
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": traceback.format_exc(),
            "exception_type": type(exc).__name__,
            "milestone_similarity": 0,
            "minefield_similarity": 0,
            "similarity": 0,
            "turn_count": scenario.max_messages,
            "milestone_mapping": {},
            "minefield_mapping": {},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["wifi_off"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preferred-tool-backend", default="DEFAULT", choices=[str(t) for t in ToolBackend])
    args = parser.parse_args()

    random.seed(42)
    pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
    pl.Config.set_tbl_formatting("ASCII_FULL")
    name_to_scenario = resolve_selected_scenarios(
        names=args.scenarios,
        preferred_tool_backend=ToolBackend(args.preferred_tool_backend),
    )
    run_dir = (
        args.output_dir
        / f"agent_local-gptoss-ensemble_user_auto-end_{dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Storing outputs to '{run_dir}'.")
    agent = ToolSandboxEnsembleAgent()
    results = [run_one(name, scenario, run_dir, agent) for name, scenario in name_to_scenario.items()]
    write_summary(results, run_dir)
    print(json.dumps({"output_dir": str(run_dir), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
