#!/usr/bin/env python3
"""Replay tau-bench checkpoints through the current goal-graph guard layer.

This is a fast offline iteration tool.  It does not load GPT-OSS and it does
not execute tau-bench tools.  Instead, it replays saved trajectories and asks
the current tau goal-graph policy/repair layer what it would do at each
assistant action.  Use this to debug failed tau trajectories before spending a
queued H100 run on validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SCHEMAS: dict[str, dict[str, Any]] = {
    "find_user_id_by_email": {
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
    "find_user_id_by_name_zip": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["first_name", "last_name", "zip"],
    },
    "get_order_details": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "get_product_details": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    "get_user_details": {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
    "cancel_pending_order": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "reason": {
                "type": "string",
                "enum": ["no longer needed", "ordered by mistake"],
            },
        },
        "required": ["order_id", "reason"],
    },
    "exchange_delivered_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "return_delivered_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "payment_method_id"],
    },
    "modify_pending_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "modify_pending_order_address": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "address1": {"type": "string"},
            "address2": {"type": "string", "description": "Optional second line, may be blank."},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "modify_pending_order_payment": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "payment_method_id"],
    },
    "modify_user_address": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "address1": {"type": "string"},
            "address2": {"type": "string", "description": "Optional second line, may be blank."},
            "city": {"type": "string"},
            "state": {"type": "string"},
            "country": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "get_reservation_details": {
        "type": "object",
        "properties": {"reservation_id": {"type": "string"}},
        "required": ["reservation_id"],
    },
    "cancel_reservation": {
        "type": "object",
        "properties": {"reservation_id": {"type": "string"}},
        "required": ["reservation_id"],
    },
    "search_direct_flight": {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["origin", "destination", "date"],
    },
    "search_onestop_flight": {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["origin", "destination", "date"],
    },
    "update_reservation_flights": {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string"},
            "cabin": {"type": "string", "enum": ["basic_economy", "economy", "business"]},
            "flights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"flight_number": {"type": "string"}, "date": {"type": "string"}},
                    "required": ["flight_number", "date"],
                },
            },
            "payment_id": {"type": "string"},
        },
        "required": ["reservation_id", "cabin", "flights", "payment_id"],
    },
    "update_reservation_baggages": {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string"},
            "total_baggages": {"type": "integer"},
            "nonfree_baggages": {"type": "integer"},
            "payment_id": {"type": "string"},
        },
        "required": ["reservation_id", "total_baggages", "nonfree_baggages", "payment_id"],
    },
    "update_reservation_passengers": {
        "type": "object",
        "properties": {
            "reservation_id": {"type": "string"},
            "passengers": {"type": "array"},
        },
        "required": ["reservation_id", "passengers"],
    },
}

RETAIL_TOOL_NAMES = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}

AIRLINE_TOOL_NAMES = {
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "think",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}


def install_model_stubs() -> None:
    tau_bench = types.ModuleType("tau_bench")
    tau_bench_agents = types.ModuleType("tau_bench.agents")
    tau_bench_agents_base = types.ModuleType("tau_bench.agents.base")
    tau_bench_envs = types.ModuleType("tau_bench.envs")
    tau_bench_envs_base = types.ModuleType("tau_bench.envs.base")
    tau_bench_types = types.ModuleType("tau_bench.types")

    class StubAgent:
        pass

    class StubAction:
        def __init__(self, name: str, kwargs: dict[str, Any]) -> None:
            self.name = name
            self.kwargs = kwargs

        def model_dump(self) -> dict[str, Any]:
            return {"name": self.name, "kwargs": self.kwargs}

    class StubSolveResult:
        def __init__(
            self,
            reward: float,
            messages: list[dict[str, Any]],
            info: dict[str, Any],
            total_cost: float | None = None,
        ) -> None:
            self.reward = reward
            self.messages = messages
            self.info = info
            self.total_cost = total_cost

        def model_dump(self) -> dict[str, Any]:
            return {
                "reward": self.reward,
                "messages": self.messages,
                "info": self.info,
                "total_cost": self.total_cost,
            }

    tau_bench_agents_base.Agent = StubAgent
    tau_bench_envs_base.Env = Any
    tau_bench_types.Action = StubAction
    tau_bench_types.SolveResult = StubSolveResult
    tau_bench_types.RESPOND_ACTION_FIELD_NAME = "content"
    tau_bench_types.RESPOND_ACTION_NAME = "respond"
    sys.modules["tau_bench"] = tau_bench
    sys.modules["tau_bench.agents"] = tau_bench_agents
    sys.modules["tau_bench.agents.base"] = tau_bench_agents_base
    sys.modules["tau_bench.envs"] = tau_bench_envs
    sys.modules["tau_bench.envs.base"] = tau_bench_envs_base
    sys.modules["tau_bench.types"] = tau_bench_types

    goal_graph_eval_common = types.ModuleType("goal_graph_eval_common")
    goal_graph_eval_common.plan_and_compile_goal_graph = lambda *args, **kwargs: {
        "verification_ok": True,
        "calls": [],
    }
    sys.modules["goal_graph_eval_common"] = goal_graph_eval_common

    run_gptoss = types.ModuleType("run_gptoss_capability_plan")
    run_gptoss.generate_text = lambda *args, **kwargs: ""
    run_gptoss.load_model = lambda *args, **kwargs: (None, None)
    sys.modules["run_gptoss_capability_plan"] = run_gptoss


def load_agent_module() -> Any:
    install_model_stubs()
    for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "compute2"):
        sys.path.insert(0, str(path))
    module_name = "tau_goal_graph_agent_replay"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "compute2" / "tau_goal_graph_agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compute2/tau_goal_graph_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a tau_run checkpoint list")
    return [row for row in data if isinstance(row, dict)]


def expected_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    task = (row.get("info") or {}).get("task")
    if not isinstance(task, dict):
        return []
    actions = task.get("actions")
    return [action for action in actions if isinstance(action, dict)] if isinstance(actions, list) else []


def expected_match(action: Any, expected: list[dict[str, Any]]) -> bool:
    dumped = action.model_dump() if hasattr(action, "model_dump") else {"name": action.name, "kwargs": action.kwargs}
    for want in expected:
        if dumped == want:
            return True
    return False


def infer_domain(rows: list[dict[str, Any]]) -> str:
    haystack_parts: list[str] = []
    for row in rows:
        for message in row.get("traj") or []:
            if isinstance(message, dict):
                haystack_parts.append(str(message.get("content") or ""))
                action = message.get("action") if isinstance(message.get("action"), dict) else {}
                haystack_parts.append(str(action.get("name") or ""))
                haystack_parts.append(str(message.get("name") or ""))
        for want in expected_actions(row):
            haystack_parts.append(str(want.get("name") or ""))
    haystack = "\n".join(haystack_parts).lower()
    if "airline agent policy" in haystack or "reservation" in haystack or "flight" in haystack:
        return "airline"
    return "retail"


def collect_available_names(rows: list[dict[str, Any]]) -> set[str]:
    domain = infer_domain(rows)
    names = set(AIRLINE_TOOL_NAMES if domain == "airline" else RETAIL_TOOL_NAMES)
    names.update({"respond", "think", "transfer_to_human_agents"})
    for row in rows:
        for want in expected_actions(row):
            if want.get("name"):
                names.add(str(want["name"]))
        for message in row.get("traj") or []:
            action = message.get("action") if isinstance(message, dict) else None
            if isinstance(action, dict) and action.get("name"):
                names.add(str(action["name"]))
            if isinstance(message, dict) and message.get("name"):
                names.add(str(message["name"]))
    return names


def action_dump(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump"):
        dumped = action.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {"name": getattr(action, "name", ""), "kwargs": getattr(action, "kwargs", {})}


def short(value: Any, limit: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def replay_row(
    module: Any,
    row: dict[str, Any],
    available_names: set[str],
    schemas: dict[str, dict[str, Any]],
    include_clean: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    traj = row.get("traj") if isinstance(row.get("traj"), list) else []
    want = expected_actions(row)
    for index, message in enumerate(traj):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        action = module.action_from_dict(message.get("action"))
        if action is None:
            continue
        guarded, info = module.tau_policy_guard_action(action, traj[:index], available_names, schemas)
        old_dump = action_dump(action)
        new_dump = action_dump(guarded)
        changed = old_dump != new_dump
        if not include_clean and not changed and not info.get("used"):
            continue
        events.append(
            {
                "message_index": index,
                "old_action": old_dump,
                "new_action": new_dump,
                "changed": changed,
                "guard_reason": info.get("reason") if isinstance(info, dict) else None,
                "matches_expected": expected_match(guarded, want),
            }
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a tau_run checkpoint JSON file.")
    parser.add_argument("--task-id", type=int, action="append", help="Only replay this task id. Can be repeated.")
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Replay successful tasks too. By default only failed tasks are replayed unless --task-id is set.",
    )
    parser.add_argument("--include-clean", action="store_true", help="Also show unchanged assistant actions.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    rows = load_checkpoint(args.checkpoint)
    if args.task_id is not None:
        wanted = set(args.task_id)
        rows = [row for row in rows if int(row.get("task_id", -1)) in wanted]
    elif not args.all_tasks:
        rows = [row for row in rows if float(row.get("reward") or 0.0) < 1.0 - 1e-6]
    module = load_agent_module()
    available_names = collect_available_names(rows)

    report: list[dict[str, Any]] = []
    for row in rows:
        events = replay_row(module, row, available_names, DEFAULT_SCHEMAS, args.include_clean)
        report.append(
            {
                "task_id": row.get("task_id"),
                "reward": row.get("reward"),
                "expected_actions": expected_actions(row),
                "events": events,
            }
        )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    print(f"checkpoint={args.checkpoint}")
    scope = "selected" if args.task_id is not None else ("all" if args.all_tasks else "failed")
    print(f"tasks={len(report)} scope={scope} available_tools={len(available_names)}")
    for row in report:
        print(f"\ntask={row['task_id']} reward={row['reward']} expected={short(row['expected_actions'])}")
        if not row["events"]:
            print("  no guard changes")
            continue
        for event in row["events"]:
            marker = "MATCH" if event["matches_expected"] else "    "
            print(
                f"  [{marker}] msg={event['message_index']} reason={event['guard_reason']} "
                f"old={short(event['old_action'], 240)} -> new={short(event['new_action'], 300)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
