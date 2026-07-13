"""Direct goal-graph runtime agent for tau-bench.

This is intentionally separate from ``tau_ensemble_agent``.  It uses the same
goal-graph stepwise planner/compiler used by BFCL/API-Bank and converts the
first verified compiled call into tau-bench's single-step ``Action`` API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tau_bench.agents.base import Agent
    from tau_bench.envs.base import Env
    from tau_bench.types import (
        Action,
        RESPOND_ACTION_FIELD_NAME,
        RESPOND_ACTION_NAME,
        SolveResult,
    )
except ImportError:
    @dataclass
    class Action:
        name: str
        kwargs: Dict[str, Any]

        def model_dump(self) -> Dict[str, Any]:
            return {"name": self.name, "kwargs": self.kwargs}

    @dataclass
    class SolveResult:
        reward: float
        messages: List[Dict[str, Any]]
        info: Dict[str, Any]
        total_cost: Optional[float] = None

        def model_dump(self) -> Dict[str, Any]:
            return {
                "reward": self.reward,
                "messages": self.messages,
                "info": self.info,
                "total_cost": self.total_cost,
            }

    class Agent:
        pass

    Env = Any
    RESPOND_ACTION_FIELD_NAME = "content"
    RESPOND_ACTION_NAME = "respond"

from goal_graph_eval_common import plan_and_compile_goal_graph
from taskdecomp.episode_state import EpisodeState
from taskdecomp.goal_graph_runtime import GoalGraphRuntime
from run_gptoss_capability_plan import generate_text, load_model
from taskdecomp.capability_planning import extract_json_object
from taskdecomp.tool_binding import _filter_schema_value_incompatible_calls


DEFAULT_MODEL = "openai/gpt-oss-20b"
RETAIL_AUTH_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
RETAIL_ORDER_DETAIL_TOOL = "get_order_details"
RESERVATION_DETAIL_TOOL = "get_reservation_details"
USER_DETAIL_TOOL = "get_user_details"
MUTATING_ACTION_PREFIXES = ("cancel_", "modify_", "return_", "exchange_", "update_", "book_", "send_")
MUTATING_ACTION_NAMES = {
    "book_reservation",
    "cancel_pending_order",
    "cancel_reservation",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}
ONE_SHOT_ITEM_MUTATIONS = {"exchange_delivered_order_items", "modify_pending_order_items"}
SINGLE_LIST_ITEM_MUTATIONS = {"return_delivered_order_items"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def short_json(value: Any, limit: int = 24000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def model_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    dict_dump = getattr(value, "dict", None)
    if callable(dict_dump):
        dumped = dict_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def action_to_dict(action: Action) -> Dict[str, Any]:
    if hasattr(action, "model_dump"):
        dumped = action.model_dump()
        return dumped if isinstance(dumped, dict) else {"name": action.name, "kwargs": action.kwargs}
    if hasattr(action, "dict"):
        dumped = action.dict()
        return dumped if isinstance(dumped, dict) else {"name": action.name, "kwargs": action.kwargs}
    return {"name": action.name, "kwargs": action.kwargs}


def transcript_for_goal_graph(messages: List[Dict[str, Any]], max_messages: int = 24) -> str:
    lines: list[str] = []
    for message in messages[-max_messages:]:
        role = str(message.get("role") or "unknown")
        if role == "system":
            continue
        if role == "assistant" and isinstance(message.get("action"), dict):
            lines.append("assistant_action: " + short_json(message["action"], 4000))
            continue
        if role == "tool":
            name = str(message.get("name") or "tool")
            lines.append(f"tool:{name}: {str(message.get('content', ''))[:4000]}")
            continue
        content = str(message.get("content") or "")
        if len(content) > 4000:
            content = content[:4000] + "...<truncated>"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def transcript_for_binding(
    messages: List[Dict[str, Any]],
    max_messages: int = 24,
    allowed_observation_fields: set[str] | None = None,
) -> str:
    """Render user text and raw successful observations without action history."""
    lines: list[str] = []
    for message in messages[-max_messages:]:
        role = str(message.get("role") or "unknown")
        if role == "system":
            continue
        if role == "user":
            content = str(message.get("content") or "")
            if len(content) > 4000:
                content = content[:4000] + "...<truncated>"
            lines.append(f"user: {content}")
            continue
        if role == "tool":
            content = str(message.get("content", ""))
            if tool_observation_failed(content):
                continue
            try:
                observation = json.loads(content)
            except json.JSONDecodeError:
                observation = content[:4000]
            lines.append("tool_observation: " + short_json(observation, 4000))
    return "\n".join(lines)


def bounded_stateful_transcript(
    messages: List[Dict[str, Any]],
    renderer: Any,
    *,
    max_messages: int,
    max_chars: int,
) -> str:
    """Preserve the initial goal and recent state within a fixed inference budget."""
    if max_chars <= 0:
        return ""
    recent_messages = messages[-max_messages:] if max_messages > 0 else []
    recent = str(renderer(recent_messages, max_messages=max_messages))
    initial_user = next(
        (message for message in messages if str(message.get("role") or "") == "user"),
        None,
    )
    if initial_user is not None and initial_user not in recent_messages:
        initial = str(renderer([initial_user], max_messages=1))
        rendered = initial + "\n...<earlier state compacted>...\n" + recent
    else:
        rendered = recent
    if len(rendered) <= max_chars:
        return rendered
    marker = "\n...<context compacted>...\n"
    if max_chars <= len(marker) + 2:
        return rendered[-max_chars:]
    head_chars = max(1, max_chars // 3)
    tail_chars = max(1, max_chars - head_chars - len(marker))
    return rendered[:head_chars] + marker + rendered[-tail_chars:]


def execution_history_from_messages(messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
    """Return observed tool calls while allowing new user input to unblock retries."""
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    pending: Action | None = None
    for message in messages:
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if action is not None:
            name = str(action.get("name") or "").strip()
            kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
            pending = Action(name=name, kwargs=kwargs) if name and name != RESPOND_ACTION_NAME else None
            continue
        if str(message.get("role") or "").lower() == "user":
            completed = {
                signature: item
                for signature, item in completed.items()
                if item.get("outcome") == "success"
            }
            continue
        if pending is None or str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != pending.name:
            continue
        signature = action_signature(pending)
        observation: Any = None
        content = message.get("content")
        if isinstance(content, str):
            try:
                observation = json.loads(content)
            except json.JSONDecodeError:
                observation = None
        completed[signature] = {
            "tool_name": pending.name,
            "arguments": dict(pending.kwargs),
            "outcome": "failure" if tool_observation_failed(message.get("content")) else "success",
        }
        if observation is not None:
            completed[signature]["observation"] = observation
        pending = None
    return list(completed.values())


def tau_tools_to_goal_graph_tools(tools_info: List[Dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for raw in tools_info:
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        parameters = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        tools.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": parameters,
                # tau-bench is an executable simulator. Keep inferred
                # retrieve/mutate/communicate kinds and risks, but let the
                # environment and customer transcript serve as the benchmark
                # authorization boundary instead of requiring an extra
                # goal-graph confirmation field for every state-changing API.
                "requires_confirmation": False,
                "requires_unique_target": False,
                "effects": [],
            }
        )
    tools.append(
        {
            "name": RESPOND_ACTION_NAME,
            "description": "Send a customer-facing message when clarification or a final response is needed.",
            # In tau-bench, respond is the simulator's normal next-message
            # action, not an external communication API. Keep it executable
            # without the mutation/communication confirmation gate.
            "kind": "retrieve",
            "risk": "read_only",
            "requires_confirmation": False,
            "requires_unique_target": False,
            "effects": [],
            "parameters": {
                "type": "object",
                "properties": {
                    RESPOND_ACTION_FIELD_NAME: {
                        "type": "string",
                        "description": "The concise message to send to the customer.",
                    }
                },
                "required": [RESPOND_ACTION_FIELD_NAME],
            },
        }
    )
    return tools


def tau_tool_function(raw: Dict[str, Any]) -> Dict[str, Any]:
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    return fn if isinstance(fn, dict) else {}


def normalize_action_name(name: str, available: set[str]) -> str:
    aliases = {
        "transfer_to_human": "transfer_to_human_agents",
        "transfer_to_human_agent": "transfer_to_human_agents",
    }
    candidate = aliases.get(name, name)
    if candidate in available or candidate == RESPOND_ACTION_NAME:
        return candidate
    return name


def fallback_response(text: str = "Could you clarify what you would like me to do next?") -> Action:
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: text})


ACTION_INTENT_PATTERNS = {
    "book": r"\bbook(?:s|ed|ing)?\b",
    "cancel": r"\bcancell?(?:s|ed|ing|ation)?\b",
    "exchange": r"\b(?:exchange(?:s|d|ing)?|swap(?:s|ped|ping)?|replace(?:s|d|ment|ing)?)\b",
    "modify": r"\b(?:modif(?:y|ies|ied|ying)|change(?:s|d|ing)?|update(?:s|d|ing)?|switch(?:es|ed|ing)?)\b",
    "return": r"\b(?:return(?:s|ed|ing)?|refund(?:s|ed|ing)?|(?:send|mail|ship)(?:ing)?\s+back)\b",
    "send": r"\bsend(?:s|ing)?\b",
    "update": r"\b(?:update(?:s|d|ing)?|change(?:s|d|ing)?|switch(?:es|ed|ing)?)\b",
}


def mentions_action_intent(text: str, *intents: str) -> bool:
    if not text:
        return False
    return any(
        re.search(pattern, text, flags=re.I)
        for intent in intents
        for pattern in [ACTION_INTENT_PATTERNS.get(intent, "")]
        if pattern
    )


def action_from_goal_graph_result(
    result: dict[str, Any],
    available_tool_names: set[str],
    goal_graph_tools: list[dict[str, Any]] | None = None,
    execution_history: list[dict[str, Any]] | None = None,
) -> Action:
    call_skeleton = result.get("call_skeleton_output") if isinstance(result.get("call_skeleton_output"), dict) else {}
    parsed_skeleton = call_skeleton.get("parsed") if isinstance(call_skeleton.get("parsed"), dict) else {}
    ordered_calls = parsed_skeleton.get("ordered_calls") if isinstance(parsed_skeleton.get("ordered_calls"), list) else []
    calls = [call for call in ordered_calls if isinstance(call, dict)]
    if not calls:
        calls = [call for call in result.get("calls") or [] if isinstance(call, dict)]
    completed = {
        (
            str(item.get("tool_name") or item.get("name") or ""),
            json.dumps(
                item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        for item in execution_history or []
        if isinstance(item, dict) and str(item.get("outcome") or "").lower() in {
            "success",
            "succeeded",
            "ok",
            "completed",
            "failure",
            "failed",
            "error",
        }
    }
    for call in calls:
        name = str(call.get("tool_name") or call.get("name") or "").strip()
        name = normalize_action_name(name, available_tool_names)
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        signature = (name, json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str))
        if name != RESPOND_ACTION_NAME and signature in completed:
            continue
        if goal_graph_tools and name != RESPOND_ACTION_NAME:
            kept, _dropped = _filter_schema_value_incompatible_calls(
                goal_graph_tools,
                [{"tool_name": name, "arguments": arguments}],
            )
            if not kept:
                continue
        if name == RESPOND_ACTION_NAME:
            content = (
                arguments.get(RESPOND_ACTION_FIELD_NAME)
                or arguments.get("message")
                or arguments.get("response")
                or arguments.get("text")
                or ""
            )
            if not isinstance(content, str) or not content.strip():
                content = "Could you clarify what you would like me to do next?"
            return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content.strip()})
        if name not in available_tool_names:
            continue
        return Action(name=name, kwargs=arguments)
    response_message = _result_semantic_response_message(result)
    if response_message:
        return fallback_response(response_message)
    clarification_message = _result_semantic_clarification_message(result)
    if clarification_message:
        return fallback_response(clarification_message)
    missing_inputs = _result_missing_inputs(result)
    if missing_inputs:
        if len(missing_inputs) == 1:
            return fallback_response(f"Please provide the {missing_inputs[0].replace('_', ' ')}.")
        return fallback_response(
            "Please provide the missing details: "
            + ", ".join(item.replace("_", " ") for item in missing_inputs)
            + "."
        )
    semantic_ask = _result_semantic_ask_user(result)
    if semantic_ask:
        return fallback_response(f"Please confirm or clarify before I proceed: {semantic_ask}")
    return fallback_response()


def _result_missing_inputs(result: dict[str, Any]) -> list[str]:
    binding_plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    graph = result.get("graph") if isinstance(result.get("graph"), dict) else {}
    values = binding_plan.get("missing_inputs") or graph.get("clarification_reasons") or []
    return [str(value).strip() for value in values if str(value).strip()]


def _result_semantic_ask_user(result: dict[str, Any]) -> str:
    binding_plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan.get("capability_plan"), dict) else {}
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan.get("semantic_input_frame"), dict) else {}
    if str(frame.get("tool_decision") or "").strip().lower() not in {"ask_user", "respond"}:
        return ""
    request = str(frame.get("canonical_request") or "").strip()
    return request[:800]


def _result_semantic_clarification_message(result: dict[str, Any]) -> str:
    """Return the binder's optional user-facing clarification verbatim enough to act on.

    The semantic model selects among alternative schemas.  The adapter must not
    collapse that choice into a single field name, or it can turn an intended
    "email or name and ZIP" request into an arbitrary request for email.
    """
    binding_plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan.get("capability_plan"), dict) else {}
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan.get("semantic_input_frame"), dict) else {}
    decision = str(frame.get("tool_decision") or "").strip().lower()
    message = frame.get("clarification_message")
    if decision not in {"ask_user", "respond"} or not isinstance(message, str):
        return ""
    return message.strip()[:800]


def _result_semantic_response_message(result: dict[str, Any]) -> str:
    """Return an explicitly planned final response without inventing adapter text."""
    binding_plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan.get("capability_plan"), dict) else {}
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan.get("semantic_input_frame"), dict) else {}
    if str(frame.get("tool_decision") or "").strip().lower() not in {"ask_user", "respond", "no_tool"}:
        return ""
    message = frame.get("response_message")
    return message.strip()[:800] if isinstance(message, str) else ""


def all_conversation_text(messages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            continue
        if isinstance(message.get("action"), dict):
            parts.append(short_json(message["action"], 2000))
        content = message.get("content")
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def user_conversation_text(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if str(message.get("role") or "").lower() == "user"
    )


def latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() == "user":
            return str(message.get("content") or "")
    return ""


def tool_success_seen(messages: List[Dict[str, Any]], names: set[str]) -> bool:
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") not in names:
            continue
        content = str(message.get("content") or "")
        lowered = content.lower()
        if '"error"' in lowered or "not found" in lowered or "could not" in lowered:
            continue
        return True
    return False


def parse_tool_json(content: Any) -> Any:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def iter_json_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(iter_json_values(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(iter_json_values(item, key))
    return found


def tool_output_contains(messages: List[Dict[str, Any]], tool_name: str, key: str, value: str) -> bool:
    if not value:
        return False
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != tool_name:
            continue
        content = str(message.get("content") or "")
        if "error" in content.lower():
            continue
        parsed = parse_tool_json(content)
        if parsed is not None:
            if any(str(found) == value for found in iter_json_values(parsed, key)):
                return True
        elif value in content:
            return True
    return False


def order_details_seen(messages: List[Dict[str, Any]], order_id: str) -> bool:
    if not order_id:
        return False
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != RETAIL_ORDER_DETAIL_TOOL:
            continue
        content = str(message.get("content") or "")
        if order_id in content and '"status"' in content and '"error"' not in content.lower():
            return True
    return False


def seen_order_detail_ids(messages: List[Dict[str, Any]]) -> list[str]:
    order_ids: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != RETAIL_ORDER_DETAIL_TOOL:
            continue
        parsed = parse_tool_json(message.get("content"))
        if not isinstance(parsed, dict):
            continue
        order_id = str(parsed.get("order_id") or "")
        if order_id and order_id not in order_ids:
            order_ids.append(order_id)
    return order_ids


def reservation_details_seen(messages: List[Dict[str, Any]], reservation_id: str) -> bool:
    return tool_output_contains(messages, RESERVATION_DETAIL_TOOL, "reservation_id", reservation_id)


def reservation_detail_payloads(messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != RESERVATION_DETAIL_TOOL:
            continue
        parsed = parse_tool_json(message.get("content"))
        if isinstance(parsed, dict) and parsed.get("reservation_id"):
            payloads.append(parsed)
    return payloads


def user_details_seen(messages: List[Dict[str, Any]], user_id: str) -> bool:
    if tool_output_contains(messages, USER_DETAIL_TOOL, "user_id", user_id):
        return True
    return action_was_called(messages, USER_DETAIL_TOOL, "user_id", user_id) and tool_success_seen(messages, {USER_DETAIL_TOOL})


def action_was_called(messages: List[Dict[str, Any]], name: str, key: str = "", value: Any = None) -> bool:
    for message in messages:
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if not action or action.get("name") != name:
            continue
        if not key:
            return True
        kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
        if str(kwargs.get(key) or "") == str(value):
            return True
    return False


def action_kwargs_was_called(messages: List[Dict[str, Any]], name: str, kwargs: Dict[str, Any]) -> bool:
    for message in messages:
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if not action or action.get("name") != name:
            continue
        seen_kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
        if all(str(seen_kwargs.get(key) or "") == str(value) for key, value in kwargs.items()):
            return True
    return False


def action_error_was_seen(messages: List[Dict[str, Any]], name: str, key: str = "", value: Any = None) -> bool:
    pending = False
    for message in messages:
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if action and action.get("name") == name:
            if key:
                kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
                pending = str(kwargs.get(key) or "") == str(value)
            else:
                pending = True
            continue
        if not pending:
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != name:
            continue
        content = str(message.get("content") or "").lower()
        if "error" in content or "not found" in content:
            return True
        pending = False
    return False


def action_kwargs_error_was_seen(messages: List[Dict[str, Any]], name: str, kwargs: Dict[str, Any]) -> bool:
    pending = False
    for message in messages:
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if action and action.get("name") == name:
            seen_kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
            pending = all(str(seen_kwargs.get(key) or "") == str(value) for key, value in kwargs.items())
            continue
        if not pending:
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != name:
            continue
        content = str(message.get("content") or "").lower()
        if "error" in content or "not found" in content:
            return True
        pending = False
    return False


def tool_observation_failed(content: Any) -> bool:
    lowered = str(content or "").lower()
    return "error" in lowered or "not found" in lowered or "could not" in lowered


def actions_share_target(left: Action, right: Action) -> bool:
    if left.name != right.name:
        return False
    left_key, left_value = primary_action_key(left)
    right_key, right_value = primary_action_key(right)
    if left_key and right_key:
        return left_key == right_key and str(left_value) == str(right_value)
    if left_key:
        return str(right.kwargs.get(left_key) or "") == str(left_value)
    if right_key:
        return str(left.kwargs.get(right_key) or "") == str(right_value)
    return json.dumps(left.kwargs, sort_keys=True, default=str) == json.dumps(
        right.kwargs, sort_keys=True, default=str
    )


def completed_item_mutation_satisfies_deferred(successful: Action, proposed: Action) -> bool:
    item_mutations = ONE_SHOT_ITEM_MUTATIONS | SINGLE_LIST_ITEM_MUTATIONS
    if successful.name not in item_mutations or proposed.name not in item_mutations:
        return False
    if str(successful.kwargs.get("order_id") or "") != str(proposed.kwargs.get("order_id") or ""):
        return False
    successful_items = {str(item_id) for item_id in successful.kwargs.get("item_ids") or []}
    proposed_items = {str(item_id) for item_id in proposed.kwargs.get("item_ids") or []}
    return bool(successful_items and proposed_items and proposed_items.issubset(successful_items))


def mutating_action_success_was_seen(messages: List[Dict[str, Any]], action: Action) -> bool:
    pending_match = False
    for message in messages:
        prior_action = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
        if prior_action is not None:
            pending_match = actions_share_target(prior_action, action)
            continue
        if not pending_match:
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != action.name:
            continue
        if not tool_observation_failed(message.get("content")):
            return True
        pending_match = False
    return False


def any_mutating_action_success_was_seen(messages: List[Dict[str, Any]]) -> bool:
    pending_action: Action | None = None
    for message in messages:
        prior_action = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
        if prior_action is not None:
            pending_action = prior_action if mutating_action(prior_action.name) else None
            continue
        if pending_action is None:
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != pending_action.name:
            continue
        if not tool_observation_failed(message.get("content")):
            return True
        pending_action = None
    return False


def latest_successful_mutating_action(messages: List[Dict[str, Any]]) -> Action | None:
    pending_action: Action | None = None
    latest_success: Action | None = None
    for message in messages:
        prior_action = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
        if prior_action is not None:
            pending_action = prior_action if mutating_action(prior_action.name) else None
            continue
        if pending_action is None:
            continue
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != pending_action.name:
            continue
        if not tool_observation_failed(message.get("content")):
            latest_success = pending_action
        pending_action = None
    return latest_success


def latest_user_mentions_different_primary_target(messages: List[Dict[str, Any]], action: Action) -> bool:
    key, value = primary_action_key(action)
    if not key or value is None:
        return False
    latest = latest_user_text(messages)
    if not latest:
        return False
    target = str(value)
    if key == "order_id":
        explicit_targets = order_ids_in_text(latest)
    elif key == "reservation_id":
        explicit_targets = re.findall(r"\b[A-Z0-9]{6}\b", latest) if re.search(r"\b(reservation|booking|flight)\b", latest, re.I) else []
    else:
        explicit_targets = re.findall(rf"\b{re.escape(target)}\b", latest)
    return any(str(explicit) != target for explicit in explicit_targets)


def completed_mutation_response(messages: List[Dict[str, Any]], *, already: bool = False) -> str:
    action = latest_successful_mutating_action(messages)
    prefix = "The requested action has already been completed." if already else "The requested action has been completed."
    if action is None:
        return prefix
    if action.name == "return_delivered_order_items":
        return (
            "The return/refund request has already been completed and submitted."
            if already
            else "The return/refund request has been completed and submitted."
        ) + " You will receive an email with instructions for returning the item(s), and the refund will be issued to the selected payment method."
    if action.name == "exchange_delivered_order_items":
        return (
            "The exchange request has already been completed and submitted."
            if already
            else "The exchange request has been completed and submitted."
        ) + " You will receive an email with instructions for returning the original item(s); there is no need to place a new order."
    if action.name == "cancel_pending_order":
        return (
            "The order cancellation has already been completed."
            if already
            else "The order cancellation has been completed."
        ) + " Any refund will be issued to the original payment method according to the order policy."
    if action.name.startswith("modify_pending_order"):
        return "The pending order modification has already been completed." if already else "The pending order modification has been completed."
    if action.name == "modify_user_address":
        return "The address update has already been completed." if already else "The address update has been completed."
    if action.name == "cancel_reservation":
        return "The reservation cancellation has already been completed." if already else "The reservation cancellation has been completed."
    return prefix


def last_message_is_successful_mutating_tool(messages: List[Dict[str, Any]]) -> bool:
    if not messages:
        return False
    message = messages[-1]
    if str(message.get("role") or "").lower() != "tool":
        return False
    if not mutating_action(str(message.get("name") or "")):
        return False
    return not tool_observation_failed(message.get("content"))


def extract_email(messages: List[Dict[str, Any]]) -> str:
    emails = extract_emails(messages)
    return emails[0] if emails else ""


def extract_emails(messages: List[Dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    emails: list[str] = []
    for email in re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", all_conversation_text(messages)):
        lowered = email.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        emails.append(lowered)
    return emails


def extract_order_id(messages: List[Dict[str, Any]], action: Action | None = None) -> str:
    if action is not None:
        order_id = action.kwargs.get("order_id")
        if isinstance(order_id, str) and order_id.strip():
            return order_id.strip()
    match = re.search(r"#W\d{6,}", all_conversation_text(messages))
    return match.group(0) if match else ""


def order_ids_in_text(text: str) -> list[str]:
    seen: list[str] = []
    for order_id in re.findall(r"#W\d{6,}", text):
        if order_id not in seen:
            seen.append(order_id)
    return seen


def text_segment_for_order(text: str, order_id: str) -> str:
    if not text or not order_id:
        return ""
    matches = list(re.finditer(r"#W\d{6,}", text))
    for index, match in enumerate(matches):
        if match.group(0) != order_id:
            continue
        start = 0
        if index > 0:
            start = match.start()
            sentence_start = max(text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start()), text.rfind("\n", 0, match.start()))
            if sentence_start >= 0:
                start = sentence_start + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end].strip(" .;\n\t")
    return ""


def latest_unseen_order_id(messages: List[Dict[str, Any]]) -> str:
    seen_orders = seen_order_detail_ids(messages)
    for order_id in reversed(order_ids_in_text(latest_user_text(messages))):
        if not order_details_seen(messages, order_id):
            repaired = closest_known_order_id(order_id, seen_orders)
            if repaired and order_details_seen(messages, repaired):
                continue
            return order_id
    return ""


def current_order_id(messages: List[Dict[str, Any]], action: Action | None = None) -> str:
    if action is not None:
        order_id = action.kwargs.get("order_id")
        if isinstance(order_id, str) and order_id.strip():
            return order_id.strip()
    latest_ids = order_ids_in_text(latest_user_text(messages))
    for order_id in reversed(latest_ids):
        if order_details_seen(messages, order_id):
            return order_id
        repaired = closest_known_order_id(order_id, seen_order_detail_ids(messages))
        if repaired and order_details_seen(messages, repaired):
            return repaired
    if latest_ids:
        return latest_ids[-1]
    latest_order = latest_order_payload(messages)
    if latest_order.get("order_id"):
        return str(latest_order["order_id"])
    return extract_order_id(messages)


def extract_reservation_id(messages: List[Dict[str, Any]], action: Action | None = None) -> str:
    if action is not None:
        reservation_id = action.kwargs.get("reservation_id")
        if isinstance(reservation_id, str) and reservation_id.strip():
            return reservation_id.strip()
    text = all_conversation_text(messages)
    labeled = re.findall(
        r"\b(?:reservation|booking|confirmation)(?:\s+(?:id|number|code))?\s*(?:is|:|#)?\s*([A-Z0-9]{6})\b",
        text,
        flags=re.I,
    )
    if labeled:
        return labeled[-1].upper()
    if re.search(r"\b(?:reservation|booking|cancel|flight)\b", text, flags=re.I):
        candidates = re.findall(r"\b[A-Z0-9]{6}\b", text)
        for candidate in reversed(candidates):
            if not re.match(r"^[A-Z]{3}\d{3}$", candidate):
                return candidate
    return ""


def extract_json_value_from_tools(messages: List[Dict[str, Any]], key: str) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "tool":
            continue
        content = str(message.get("content") or "")
        if "error" in content.lower():
            continue
        parsed = parse_tool_json(content)
        if parsed is None:
            continue
        values = [value for value in iter_json_values(parsed, key) if isinstance(value, str) and value.strip()]
        if values:
            return values[-1].strip()
    return ""


def extract_authenticated_user_id(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") not in RETAIL_AUTH_TOOLS:
            continue
        content = str(message.get("content") or "").strip()
        if not content or "error" in content.lower():
            continue
        parsed = parse_tool_json(content)
        if isinstance(parsed, str):
            content = parsed
        match = re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]*_\d+\b", content)
        if match:
            return match.group(0)
    return ""


def extract_user_id(messages: List[Dict[str, Any]], action: Action | None = None) -> str:
    if action is not None:
        user_id = action.kwargs.get("user_id")
        if isinstance(user_id, str) and user_id.strip():
            return user_id.strip()
    auth_user_id = extract_authenticated_user_id(messages)
    if auth_user_id:
        return auth_user_id
    from_tools = extract_json_value_from_tools(messages, "user_id")
    if from_tools:
        return from_tools
    match = re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]*_\d+\b", all_conversation_text(messages))
    return match.group(0) if match else ""


def user_order_ids(messages: List[Dict[str, Any]]) -> list[str]:
    order_ids: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != USER_DETAIL_TOOL:
            continue
        parsed = parse_tool_json(message.get("content"))
        if not isinstance(parsed, dict):
            continue
        orders = parsed.get("orders")
        if not isinstance(orders, list):
            continue
        for order_id in orders:
            text = str(order_id)
            if text.strip() and text not in order_ids:
                order_ids.append(text)
    return order_ids


def next_unseen_user_order_id(messages: List[Dict[str, Any]]) -> str:
    for order_id in user_order_ids(messages):
        if not order_details_seen(messages, order_id):
            return order_id
    return ""


def requested_order_ids(messages: List[Dict[str, Any]]) -> list[str]:
    order_ids: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "user":
            continue
        for order_id in order_ids_in_text(str(message.get("content") or "")):
            if order_id not in order_ids:
                order_ids.append(order_id)
    return order_ids


def next_unseen_requested_order_id(messages: List[Dict[str, Any]]) -> str:
    for order_id in requested_order_ids(messages):
        if not order_details_seen(messages, order_id):
            return order_id
    return ""


def normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def edit_distance(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        row_min = current[0]
        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def closest_known_order_id(candidate: str, known_orders: list[str]) -> str:
    if not candidate or not known_orders:
        return ""
    if candidate in known_orders:
        return candidate
    candidate_key = normalized_identifier(candidate)
    scored: list[tuple[int, str]] = []
    for order_id in known_orders:
        order_key = normalized_identifier(order_id)
        if not candidate_key or not order_key:
            continue
        score = edit_distance(candidate_key, order_key, limit=2)
        if score <= 2:
            scored.append((score, order_id))
    scored.sort()
    if len(scored) == 1 or (len(scored) > 1 and scored[0][0] < scored[1][0]):
        return scored[0][1]
    return ""


def repair_order_id_from_user_orders(action: Action, messages: List[Dict[str, Any]]) -> tuple[Action, dict[str, Any]]:
    order_id = action.kwargs.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        return action, {}
    known_orders = user_order_ids(messages)
    for seen_order_id in seen_order_detail_ids(messages):
        if seen_order_id not in known_orders:
            known_orders.append(seen_order_id)
    repaired = closest_known_order_id(order_id.strip(), known_orders)
    if not repaired or repaired == order_id:
        return action, {}
    kwargs = dict(action.kwargs)
    kwargs["order_id"] = repaired
    return Action(name=action.name, kwargs=kwargs), {"order_id": {"from": order_id, "to": repaired}}


def extract_name_zip(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    text = user_conversation_text(messages)
    zip_matches = list(re.finditer(r"\b\d{5}(?:-\d{4})?\b", text))
    if not zip_matches:
        return {}
    invalid_name_terms = {"really", "very", "busy", "sorry", "sure", "ready", "frustrated", "urgent"}
    name_patterns = [
        r"(?i:(?:my full name(?: is)?|my name(?: is)?|full name(?: is)?|this is))\s+([A-Z][A-Za-z' -]+?)\s*(?:,|and|\.|\n|$)",
        r"(?i:\bname(?:\s+is|[:\s]+))\s+([A-Z][A-Za-z' -]+?)\s*(?:,|and|\.|\n|$)",
    ]
    for zip_match in reversed(zip_matches):
        windows = [
            text[max(0, zip_match.start() - 300) : zip_match.start()],
            text[zip_match.end() : min(len(text), zip_match.end() + 300)],
        ]
        for window in windows:
            first_matches = list(
                re.finditer(
                    r"(?i:\bfirst\s+name(?:\s+is\s+|[:\s]+)([A-Z][A-Za-z' -]+?))\s*(?=,|and\b|\blast\s+name\b|\bzip\b|\.|\n|$)",
                    window,
                )
            )
            last_matches = list(
                re.finditer(
                    r"(?i:\blast\s+name(?:\s+is\s+|[:\s]+)([A-Z][A-Za-z' -]+?))\s*(?=,|and\b|\bzip\b|\.|\n|$)",
                    window,
                )
            )
            if first_matches and last_matches:
                first_name = first_matches[-1].group(1).strip()
                last_name = last_matches[-1].group(1).strip()
                if (
                    first_name
                    and last_name
                    and first_name.lower() not in invalid_name_terms
                    and last_name.lower() not in invalid_name_terms
                ):
                    return {"first_name": first_name, "last_name": last_name, "zip": zip_match.group(0)}
            for pattern in name_patterns:
                matches = list(re.finditer(pattern, window))
                if not matches:
                    continue
                name = matches[-1].group(1).strip()
                pieces = [part for part in re.split(r"\s+", name) if part]
                if len(pieces) >= 2 and not any(piece.lower() in invalid_name_terms for piece in pieces):
                    return {"first_name": pieces[0], "last_name": pieces[-1], "zip": zip_match.group(0)}
    return {}


def extract_partial_name_zip(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    text = user_conversation_text(messages)
    partial: dict[str, str] = {}
    zip_matches = list(re.finditer(r"\b\d{5}(?:-\d{4})?\b", text))
    if zip_matches:
        partial["zip"] = zip_matches[-1].group(0)

    first_matches = list(
        re.finditer(
            r"(?i:\bfirst\s+name(?:\s+is\s+|[:\s]+)([A-Z][A-Za-z' -]+?))\s*(?=,|and\b|\blast\s+name\b|\bzip\b|\.|\n|$)",
            text,
        )
    )
    last_matches = list(
        re.finditer(
            r"(?i:\blast\s+name(?:\s+is\s+|[:\s]+)([A-Z][A-Za-z' -]+?))\s*(?=,|and\b|\bfirst\s+name\b|\bzip\b|\.|\n|$)",
            text,
        )
    )
    if first_matches:
        partial["first_name"] = first_matches[-1].group(1).strip()
    if last_matches:
        partial["last_name"] = last_matches[-1].group(1).strip()

    if "first_name" not in partial or "last_name" not in partial:
        name_matches = list(
            re.finditer(
                r"(?i:(?:my full name(?: is)?|my name(?: is)?|full name(?: is)?|this is|name(?:\s+is|[:\s]+)))\s+([A-Z][A-Za-z' -]+?)\s*(?:,|and|\.|\n|$)",
                text,
            )
        )
        if name_matches:
            name = name_matches[-1].group(1).strip()
            pieces = [part for part in re.split(r"\s+", name) if part]
            if len(pieces) >= 2:
                partial.setdefault("first_name", pieces[0])
                partial.setdefault("last_name", pieces[-1])
            elif len(pieces) == 1:
                partial.setdefault("first_name", pieces[0])

    invalid_name_terms = {"really", "very", "busy", "sorry", "sure", "ready", "frustrated", "urgent"}
    for key in ("first_name", "last_name"):
        if partial.get(key, "").lower() in invalid_name_terms:
            partial.pop(key, None)
    return partial


def mutating_action(name: str) -> bool:
    return name in MUTATING_ACTION_NAMES or name.startswith(MUTATING_ACTION_PREFIXES)


def primary_action_key(action: Action) -> tuple[str, Any] | tuple[str, None]:
    for key in ("order_id", "reservation_id", "user_id"):
        value = action.kwargs.get(key)
        if value:
            return key, value
    return "", None


def tool_schema_required(tool_schemas: Dict[str, Dict[str, Any]] | None, name: str) -> list[str]:
    if not tool_schemas:
        return []
    schema = tool_schemas.get(name) or {}
    required = schema.get("required")
    return [str(item) for item in required] if isinstance(required, list) else []


def tool_schema_property(tool_schemas: Dict[str, Dict[str, Any]] | None, name: str, arg: str) -> Dict[str, Any]:
    if not tool_schemas:
        return {}
    schema = tool_schemas.get(name) or {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    prop = properties.get(arg)
    return prop if isinstance(prop, dict) else {}


def enum_values(tool_schemas: Dict[str, Dict[str, Any]] | None, name: str, arg: str) -> list[str]:
    values = tool_schema_property(tool_schemas, name, arg).get("enum")
    return [str(value) for value in values] if isinstance(values, list) else []


def placeholder_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()
    return normalized in {"", "ask", "task", "respond", "clarify", "none", "n/a", "na", "unknown"}


def missing_required_arguments(action: Action, tool_schemas: Dict[str, Dict[str, Any]] | None) -> list[str]:
    missing: list[str] = []
    for arg in tool_schema_required(tool_schemas, action.name):
        value = action.kwargs.get(arg)
        if value == "":
            description = str(tool_schema_property(tool_schemas, action.name, arg).get("description") or "").lower()
            if "''" in description or "empty" in description or "blank" in description:
                continue
        if value is None or value == "" or value == [] or value == {} or placeholder_text(value):
            missing.append(arg)
    return missing


def grounded_value_for_argument(
    arg: str,
    action: Action,
    messages: List[Dict[str, Any]],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> Any:
    if arg == "order_id":
        return extract_order_id(messages, action)
    if arg == "reservation_id":
        return extract_reservation_id(messages, action)
    if arg == "user_id":
        return extract_user_id(messages, action)
    if arg == "email":
        return extract_email(messages)
    if arg in {"first_name", "last_name", "zip"}:
        return extract_name_zip(messages).get(arg, "")
    values = enum_values(tool_schemas, action.name, arg)
    if arg == "reason":
        reason = infer_reason_from_text(messages, values)
        if reason:
            return reason
    text = user_conversation_text(messages).lower()
    for value in values:
        if value.lower() in text:
            return value
    return ""


def repair_action_arguments(
    action: Action,
    messages: List[Dict[str, Any]],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> tuple[Action, dict[str, Any]]:
    if not tool_schemas or action.name not in tool_schemas:
        return action, {}
    repaired = dict(action.kwargs)
    repairs: dict[str, Any] = {}
    for arg in missing_required_arguments(action, tool_schemas):
        value = grounded_value_for_argument(arg, action, messages, tool_schemas)
        if value not in ("", None, [], {}):
            repaired[arg] = value
            repairs[arg] = value
    if not repairs:
        return action, {}
    return Action(name=action.name, kwargs=repaired), {"filled": repairs}


def repair_reason_argument_from_transcript(
    action: Action,
    messages: List[Dict[str, Any]],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> tuple[Action, dict[str, Any]]:
    values = enum_values(tool_schemas, action.name, "reason")
    if not values:
        return action, {}
    inferred = infer_reason_from_text(messages, values)
    if not inferred:
        return action, {}
    current = action.kwargs.get("reason")
    if str(current or "") == inferred:
        return action, {}
    repaired = dict(action.kwargs)
    repaired["reason"] = inferred
    return Action(name=action.name, kwargs=repaired), {"reason": {"from": current, "to": inferred}}


def missing_argument_response(action: Action, missing: list[str], tool_schemas: Dict[str, Dict[str, Any]] | None) -> str:
    if len(missing) == 1:
        arg = missing[0]
        values = enum_values(tool_schemas, action.name, arg)
        if values:
            return f"Please provide the {arg.replace('_', ' ')}: " + " or ".join(values) + "."
        return f"Please provide the {arg.replace('_', ' ')}."
    pretty = ", ".join(arg.replace("_", " ") for arg in missing)
    return f"Please provide the missing details: {pretty}."


def action_from_dict(value: Any) -> Action | None:
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or value.get("tool_name") or "").strip()
    kwargs = value.get("kwargs") if isinstance(value.get("kwargs"), dict) else value.get("arguments")
    if not name or not isinstance(kwargs, dict):
        return None
    return Action(name=name, kwargs=kwargs)


def value_grounded_for_policy_repair(
    value: Any,
    arg: str,
    action_name: str,
    messages: List[Dict[str, Any]],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> bool:
    if value in (None, "", [], {}):
        return True
    if arg == RESPOND_ACTION_FIELD_NAME and action_name == RESPOND_ACTION_NAME:
        return True
    allowed_enums = set(enum_values(tool_schemas, action_name, arg))
    if isinstance(value, str):
        if value in allowed_enums:
            return True
        return value in all_conversation_text(messages)
    if isinstance(value, (int, float, bool)):
        return str(value) in all_conversation_text(messages)
    if isinstance(value, list):
        return all(value_grounded_for_policy_repair(item, arg, action_name, messages, tool_schemas) for item in value)
    if isinstance(value, dict):
        return all(
            value_grounded_for_policy_repair(item, key, action_name, messages, tool_schemas)
            for key, item in value.items()
        )
    return False


def value_supported_by_evidence(
    value: Any,
    arg: str,
    action_name: str,
    evidence_text: str,
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> bool:
    if value in (None, "", [], {}):
        return True
    if arg == RESPOND_ACTION_FIELD_NAME and action_name == RESPOND_ACTION_NAME:
        return True
    if isinstance(value, str) and value in set(enum_values(tool_schemas, action_name, arg)):
        return True
    if isinstance(value, str):
        return value in evidence_text
    if isinstance(value, (int, float, bool)):
        return str(value) in evidence_text
    if isinstance(value, list):
        return all(value_supported_by_evidence(item, arg, action_name, evidence_text, tool_schemas) for item in value)
    if isinstance(value, dict):
        return all(
            value_supported_by_evidence(item, key, action_name, evidence_text, tool_schemas)
            for key, item in value.items()
        )
    return False


def policy_repair_action_from_raw(
    raw_text: str,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> tuple[Action | None, dict[str, Any]]:
    parsed, parse_error = extract_json_object(raw_text)
    info: dict[str, Any] = {"raw_text": raw_text, "parse_error": parse_error}
    if not isinstance(parsed, dict):
        info["accepted"] = False
        info["reason"] = "parse_error"
        return None, info
    action_payload = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    action = action_from_dict(action_payload)
    if action is None:
        info["accepted"] = False
        info["reason"] = "missing_action"
        info["parsed"] = parsed
        return None, info
    action = Action(name=normalize_action_name(action.name, available_tool_names), kwargs=action.kwargs)
    if action.name != RESPOND_ACTION_NAME and action.name not in available_tool_names:
        info["accepted"] = False
        info["reason"] = "unknown_tool"
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    if non_actionable_action(action):
        info["accepted"] = False
        info["reason"] = "non_actionable_action"
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    schema = (tool_schemas or {}).get(action.name) or {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if properties:
        filtered = {
            key: value
            for key, value in action.kwargs.items()
            if key in properties or (action.name == RESPOND_ACTION_NAME and key == RESPOND_ACTION_FIELD_NAME)
        }
    else:
        filtered = dict(action.kwargs)
    action = Action(name=action.name, kwargs=filtered)

    # A repair is a semantic proposal, not an authority to invent state.  Keep
    # its evidence machine-checkable so this layer can be shared by stateful
    # tool environments beyond tau-bench.
    evidence = parsed.get("evidence")
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or not all(isinstance(span, str) and span.strip() for span in evidence):
        info["accepted"] = False
        info["reason"] = "invalid_evidence"
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    ungrounded = [
        key
        for key, value in action.kwargs.items()
        if not value_grounded_for_policy_repair(value, key, action.name, messages, tool_schemas)
    ]
    if ungrounded:
        info["accepted"] = False
        info["reason"] = "ungrounded_arguments"
        info["ungrounded_arguments"] = ungrounded
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    # The model sees role-labelled transcript lines, while some argument
    # grounders consume only raw message bodies.  Evidence must be validated
    # against both representations so a faithful quote such as ``user: ...``
    # is not rejected merely because of the transport format.
    transcript = all_conversation_text(messages) + "\n" + transcript_for_goal_graph(messages, max_messages=len(messages))
    invalid_evidence = [span for span in evidence if span not in transcript]
    if invalid_evidence:
        info["accepted"] = False
        info["reason"] = "ungrounded_evidence"
        info["ungrounded_evidence"] = invalid_evidence
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    if action.name != RESPOND_ACTION_NAME and not evidence:
        info["accepted"] = False
        info["reason"] = "missing_evidence"
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    evidence_text = "\n".join(evidence)
    unsupported_by_evidence = [
        key
        for key, value in action.kwargs.items()
        if not value_supported_by_evidence(value, key, action.name, evidence_text, tool_schemas)
    ]
    if unsupported_by_evidence:
        info["accepted"] = False
        info["reason"] = "arguments_not_supported_by_evidence"
        info["unsupported_arguments"] = unsupported_by_evidence
        info["parsed"] = parsed
        info["proposed_action"] = action_to_dict(action)
        return None, info
    info["accepted"] = True
    info["reason"] = "accepted"
    info["evidence"] = evidence
    info["parsed"] = parsed
    info["proposed_action"] = action_to_dict(action)
    return action, info


def is_escalation_action(name: str) -> bool:
    """Recognize the generic handoff class without depending on a domain API."""
    return bool(re.search(r"(?:transfer|escalat|handoff|hand_off|human|supervisor)", name, re.I))


def is_probably_state_changing_action(name: str) -> bool:
    """Conservative verb-level classification used only for duplicate protection."""
    return bool(
        re.match(
            r"(?:add|book|cancel|create|delete|exchange|issue|modify|remove|return|send|set|update|write)_",
            name,
            re.I,
        )
    )


def action_signature(action: Action) -> tuple[str, str]:
    return action.name, json.dumps(action.kwargs, sort_keys=True, separators=(",", ":"), default=str)


def action_success_was_seen_generic(messages: List[Dict[str, Any]], action: Action) -> bool:
    """Match a tool result to the exact preceding call, regardless of entity names."""
    expected = action_signature(action)
    pending: tuple[str, str] | None = None
    for message in messages:
        previous = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
        if previous is not None:
            pending = action_signature(previous)
            continue
        if pending != expected or str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != action.name:
            continue
        if not tool_observation_failed(message.get("content")):
            return True
        pending = None
    return False


def action_matches_verified_goal_graph_call(action: Action, goal_graph_result: Dict[str, Any] | None) -> bool:
    """Recognize a call already schema- and evidence-checked by the shared runtime."""
    if not isinstance(goal_graph_result, dict) or not goal_graph_result.get("verification_ok"):
        return False
    for call in goal_graph_result.get("calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or call.get("name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if name == action.name and arguments == action.kwargs:
            return True
    return False


def generic_action_guard(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None = None,
    goal_graph_result: Dict[str, Any] | None = None,
) -> tuple[Action, dict[str, Any]]:
    """Schema and transcript verifier for any single-step tool environment.

    This deliberately knows nothing about accounts, orders, reservations, or
    benchmark policies.  Policy interpretation belongs to the model; this
    layer merely prevents malformed, ungrounded, or duplicate write actions.
    """
    name = normalize_action_name(action.name, available_tool_names)
    if name != RESPOND_ACTION_NAME and name not in available_tool_names:
        return fallback_response("I do not have an available tool for that action."), {
            "used": True,
            "reason": "unknown_tool",
            "proposed_action": action_to_dict(action),
        }

    schema = (tool_schemas or {}).get(name) or {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if properties:
        kwargs = {key: value for key, value in action.kwargs.items() if key in properties}
    else:
        kwargs = dict(action.kwargs)
    normalized = Action(name=name, kwargs=kwargs)
    verified_goal_graph_call = action_matches_verified_goal_graph_call(normalized, goal_graph_result)

    missing = missing_required_arguments(normalized, tool_schemas)
    if missing:
        return fallback_response(missing_argument_response(normalized, missing, tool_schemas)), {
            "used": True,
            "reason": "missing_required_arguments",
            "missing_arguments": missing,
            "proposed_action": action_to_dict(normalized),
        }

    ungrounded = [] if verified_goal_graph_call else [
        key
        for key, value in normalized.kwargs.items()
        if not value_grounded_for_policy_repair(value, key, normalized.name, messages, tool_schemas)
    ]
    if ungrounded:
        return fallback_response(
            "Please provide the information needed to complete that action: "
            + ", ".join(key.replace("_", " ") for key in ungrounded)
            + "."
        ), {
            "used": True,
            "reason": "ungrounded_arguments",
            "ungrounded_arguments": ungrounded,
            "proposed_action": action_to_dict(normalized),
        }

    if action_success_was_seen_generic(messages, normalized):
        return fallback_response("That action has already completed and produced an observation."), {
            "used": True,
            "reason": "duplicate_successful_action",
            "proposed_action": action_to_dict(normalized),
        }
    return normalized, {"used": False, "verified_goal_graph_call": verified_goal_graph_call}


def user_confirmed_after_agent_request_generic(messages: List[Dict[str, Any]]) -> bool:
    """Detect confirmation turns without assuming the domain or action type."""
    latest = latest_user_text(messages)
    if not re.search(r"\b(?:yes|confirm(?:ed)?|proceed|go ahead|sounds good|that'?s correct)\b", latest, re.I):
        return False
    for message in reversed(messages[:-1]):
        action = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
        if action is None or action.name != RESPOND_ACTION_NAME:
            continue
        content = str(action.kwargs.get(RESPOND_ACTION_FIELD_NAME) or "")
        return bool(re.search(r"\b(?:confirm|confirmation|proceed|reply with yes)\b", content, re.I))
    return False


def normalized_response_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def response_repeats_after_new_user_turn(action: Action, messages: List[Dict[str, Any]]) -> bool:
    """Catch no-progress clarification loops without interpreting the domain."""
    if action.name != RESPOND_ACTION_NAME:
        return False
    proposed = normalized_response_text(action.kwargs.get(RESPOND_ACTION_FIELD_NAME))
    if not proposed:
        return False
    for index in range(len(messages) - 1, -1, -1):
        prior = action_from_dict(messages[index].get("action")) if isinstance(messages[index].get("action"), dict) else None
        if prior is None or prior.name != RESPOND_ACTION_NAME:
            continue
        prior_text = normalized_response_text(prior.kwargs.get(RESPOND_ACTION_FIELD_NAME))
        if not prior_text or prior_text != proposed:
            return False
        return any(str(message.get("role") or "").lower() == "user" for message in messages[index + 1 :])
    return False


def response_follows_tool_observation(action: Action, messages: List[Dict[str, Any]]) -> bool:
    """A response after fresh state is a useful point for one semantic check."""
    return bool(
        action.name == RESPOND_ACTION_NAME
        and messages
        and str(messages[-1].get("role") or "").lower() == "tool"
    )


def semantic_repair_needed_generic(action: Action, messages: List[Dict[str, Any]]) -> bool:
    """Identify cases requiring policy reasoning rather than rule-based routing."""
    return (
        non_actionable_response(action)
        or is_escalation_action(action.name)
        or action_success_was_seen_generic(messages, action)
        or (action.name == RESPOND_ACTION_NAME and user_confirmed_after_agent_request_generic(messages))
        or response_repeats_after_new_user_turn(action, messages)
        or response_follows_tool_observation(action, messages)
    )


def latest_deferred_mutating_action(messages: List[Dict[str, Any]]) -> Action | None:
    successful = latest_successful_mutating_action(messages)
    for message in reversed(messages):
        guard = message.get("tau_policy_guard") if isinstance(message.get("tau_policy_guard"), dict) else None
        goal_graph = message.get("goal_graph") if isinstance(message.get("goal_graph"), dict) else {}
        if guard is None and isinstance(goal_graph, dict):
            guard = goal_graph.get("tau_policy_guard") if isinstance(goal_graph.get("tau_policy_guard"), dict) else None
        if not guard:
            continue
        proposed = action_from_dict(guard.get("proposed_action"))
        if proposed is None or not mutating_action(proposed.name):
            continue
        if successful is not None and completed_item_mutation_satisfies_deferred(successful, proposed):
            continue
        key, value = primary_action_key(proposed)
        if key and action_was_called(messages, proposed.name, key, value):
            continue
        if not key and action_was_called(messages, proposed.name):
            continue
        return proposed
    return None


def user_confirmed_after_confirmation_request(messages: List[Dict[str, Any]]) -> bool:
    last_confirm_prompt = -1
    for index, message in enumerate(messages):
        action = message.get("action") if isinstance(message.get("action"), dict) else None
        if action and action.get("name") == RESPOND_ACTION_NAME:
            content = str((action.get("kwargs") or {}).get(RESPOND_ACTION_FIELD_NAME) or "")
            if re.search(r"\b(confirm|confirmation|proceed|yes)\b", content, re.I):
                last_confirm_prompt = index
    if last_confirm_prompt < 0:
        return False
    for message in messages[last_confirm_prompt + 1 :]:
        if str(message.get("role") or "").lower() != "user":
            continue
        if re.search(r"\b(yes|confirmed|confirm|please proceed|go ahead|sounds good)\b", str(message.get("content") or ""), re.I):
                return True
    return False


def latest_user_explicitly_confirms_action(messages: List[Dict[str, Any]], action: Action) -> bool:
    latest = latest_user_text(messages).lower()
    if not latest:
        return False
    if not re.search(r"\b(yes|confirm|confirmed|proceed|go ahead|do it|sounds good)\b", latest):
        return False
    action_words = {
        "cancel": ("cancel",),
        "exchange": ("exchange", "exchanging", "exchanged", "swap", "swapping", "replace", "replacing"),
        "return": ("return",),
        "modify": ("modify", "change", "update"),
        "update": ("update", "change"),
        "book": ("book",),
        "send": ("send",),
    }
    lowered_name = action.name.lower()
    for prefix, words in action_words.items():
        if lowered_name.startswith(prefix) and (
            mentions_action_intent(latest, prefix)
            or any(re.search(rf"\b{re.escape(word)}\b", latest) for word in words)
        ):
            return True
    return False


def user_requested_action(messages: List[Dict[str, Any]], action: Action) -> bool:
    text = user_conversation_text(messages).lower()
    if not text:
        return False
    action_words = {
        "cancel": ("cancel", "cancellation", "refund", "full refund"),
        "exchange": ("exchange", "swap", "replace"),
        "return": ("return", "refund", "full refund"),
        "modify": ("modify", "change", "update", "switch", "exchange", "swap", "replace"),
        "update": ("update", "change", "switch"),
        "book": ("book",),
        "send": ("send",),
    }
    lowered_name = action.name.lower()
    if not any(
        lowered_name.startswith(prefix)
        and (
            mentions_action_intent(text, prefix)
            or any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)
        )
        for prefix, words in action_words.items()
    ):
        return False
    key, value = primary_action_key(action)
    if key and value and str(value).lower() not in text and not tool_output_contains(messages, RETAIL_ORDER_DETAIL_TOOL, key, str(value)):
        return False
    latest = latest_user_text(messages).lower()
    if action.name.startswith("cancel") and re.search(r"\b(don't|do not|not)\s+(?:need|want)?\s*(?:a\s+)?cancell?ation\b", latest):
        return False
    for message in messages:
        if str(message.get("role") or "").lower() != "user":
            continue
        content = str(message.get("content") or "").lower()
        if not content or not any(
            lowered_name.startswith(prefix)
            and (
                mentions_action_intent(content, prefix)
                or any(re.search(rf"\b{re.escape(word)}\b", content) for word in words)
            )
            for prefix, words in action_words.items()
        ):
            continue
        asks_capability = bool(re.search(r"\b(?:can|could|would)\s+i\b", content))
        direct_language = bool(
            re.search(
                r"\b(please|i want|i need|i would like|i'd like|can you|could you|would you|"
                r"go ahead|proceed|do it|make sure)\b",
                content,
            )
            or re.search(r"^\s*(cancel|return|exchange|modify|change|update|refund)\b", content)
        )
        if direct_language and not (asks_capability and not re.search(r"\b(please|i want|i need|i would like|i'd like)\b", content)):
            return True
    return False


def required_value_grounded_for_direct_mutation(
    value: Any,
    arg: str,
    action: Action,
    messages: List[Dict[str, Any]],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> bool:
    if value in (None, "", [], {}) or placeholder_text(value):
        return False
    if arg == "reason":
        inferred = infer_reason_from_text(messages, enum_values(tool_schemas, action.name, arg))
        return bool(inferred and str(value) == inferred)
    haystack = all_conversation_text(messages)
    if isinstance(value, str):
        return value in haystack
    if isinstance(value, (int, float, bool)):
        return str(value) in haystack
    if isinstance(value, list):
        return all(
            required_value_grounded_for_direct_mutation(item, arg, action, messages, tool_schemas)
            for item in value
        )
    if isinstance(value, dict):
        return all(
            required_value_grounded_for_direct_mutation(item, key, action, messages, tool_schemas)
            for key, item in value.items()
        )
    return False


def direct_mutation_request_is_grounded(
    messages: List[Dict[str, Any]],
    action: Action,
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> bool:
    if not mutating_action(action.name):
        return False
    if not user_requested_action(messages, action):
        return False
    required = tool_schema_required(tool_schemas, action.name)
    if not required:
        return all(
            required_value_grounded_for_direct_mutation(value, key, action, messages, tool_schemas)
            for key, value in action.kwargs.items()
        )
    for arg in required:
        if not required_value_grounded_for_direct_mutation(action.kwargs.get(arg), arg, action, messages, tool_schemas):
            return False
    return True


def retail_account_task_text_seen(messages: List[Dict[str, Any]]) -> bool:
    text = user_conversation_text(messages).lower()
    if not text:
        return False
    return bool(
        mentions_action_intent(text, "cancel", "exchange", "return", "modify", "update")
        or re.search(
            r"\b(order|orders|purchase|purchased|refund|payment|address|profile|account|item|items|product|products)\b",
            text,
        )
    )


def retail_auth_request_action(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if not (RETAIL_AUTH_TOOLS & available_tool_names):
        return None
    if tool_success_seen(messages, RETAIL_AUTH_TOOLS):
        return None
    if not retail_account_task_text_seen(messages):
        return None
    emails = extract_emails(messages)
    if any(
        "find_user_id_by_email" in available_tool_names
        and not action_was_called(messages, "find_user_id_by_email", "email", email)
        for email in emails
    ):
        return None
    name_zip = extract_name_zip(messages)
    if (
        name_zip
        and "find_user_id_by_name_zip" in available_tool_names
        and not action_kwargs_was_called(messages, "find_user_id_by_name_zip", name_zip)
    ):
        return None
    partial_name_zip = extract_partial_name_zip(messages)
    if partial_name_zip and not name_zip and "find_user_id_by_name_zip" in available_tool_names:
        missing = [field for field in ("first_name", "last_name", "zip") if not partial_name_zip.get(field)]
        if missing:
            labels = {
                "first_name": "first name",
                "last_name": "last name",
                "zip": "ZIP code",
            }
            if len(missing) == 1:
                missing_text = f"your {labels[missing[0]]}"
            else:
                missing_text = " and ".join(f"your {labels[field]}" for field in missing)
            return fallback_response(
                f"Please provide {missing_text} so I can authenticate you with your name and ZIP code."
            )
    if emails or name_zip:
        return fallback_response(
            "I could not find an account with the information provided. Please provide another email address, or your first name, last name, and ZIP code so I can locate your account."
        )
    return fallback_response(
        "Please provide the email address on the account, or your first name, last name, and ZIP code so I can authenticate you."
    )


def mutation_confirmation_detail(action: Action, messages: List[Dict[str, Any]]) -> str:
    detail = action.name.replace("_", " ")
    order_id = extract_order_id(messages, action)
    reservation_id = extract_reservation_id(messages, action)
    if order_id:
        detail += f" for order {order_id}"
        item_ids = action.kwargs.get("item_ids")
        order = latest_order_payload(messages, order_id)
        items = order.get("items") if isinstance(order.get("items"), list) else []
        names_by_id = {
            str(item.get("item_id")): str(item.get("name") or item.get("product_name") or "")
            for item in items
            if isinstance(item, dict) and item.get("item_id")
        }
        if isinstance(item_ids, list):
            names = [names_by_id.get(str(item_id), str(item_id)) for item_id in item_ids]
            names = [name for name in names if name]
            if names:
                detail += " involving " + ", ".join(names)
    elif reservation_id:
        detail += f" for reservation {reservation_id}"
    return detail


def airline_cancellation_reason_seen(messages: List[Dict[str, Any]]) -> bool:
    text = user_conversation_text(messages).lower()
    return bool(
        re.search(
            r"\b(change of plan|changed my plan|change plans|airline cancelled|airline canceled|flight cancelled|flight canceled|weather|health|medical|sick|ill|other reason|another reason)\b",
            text,
        )
    )


def latest_user_still_selecting_item_options(messages: List[Dict[str, Any]]) -> bool:
    latest = latest_user_text(messages).lower()
    if not latest:
        return False
    if refund_payment_method_question_seen(messages):
        return False
    asks_for_options = re.search(r"\b(available|which|what color|what size|suggest|recommend)\b", latest)
    uncertainty = re.search(
        r"\b(not sure|unsure|undecided|haven't decided|have not decided|which one|what color|what size)\b",
        latest,
    )
    if final_item_choice_text(latest):
        if uncertainty:
            return True
        open_option_request = re.search(
            r"\b(what|which|options?|suggest|recommend|tell me more|more details|more information)\b",
            latest,
        )
        execution_request = re.search(
            r"\b(go ahead|proceed|move forward|send me|process|exchange|swap|replace|return|modify|that'?s all|that is all)\b",
            latest,
        )
        if not open_option_request or execution_request:
            return False
    if final_item_choice_text(latest) and not asks_for_options and not uncertainty:
        return False
    if re.search(r"\boptions?\b", latest) and not final_item_choice_text(latest):
        asks_for_options = asks_for_options or re.search(r"\boptions?\b", latest)
    return bool(asks_for_options or uncertainty)


def latest_user_requests_deferred_mutation_execution(messages: List[Dict[str, Any]]) -> bool:
    latest = latest_user_text(messages).lower()
    if not latest:
        return False
    action_request = re.search(
        r"\b(go ahead|proceed|move forward|send me|process|exchange|swap|replace|return|modify|that'?s all|that is all)\b",
        latest,
    )
    if not (final_item_choice_text(latest) or action_request):
        return False
    if latest_user_still_selecting_item_options(messages):
        return False
    return True


def refund_payment_method_question_seen(messages: List[Dict[str, Any]]) -> bool:
    latest = latest_user_text(messages).lower()
    if not latest:
        return False
    if not re.search(r"\b(refund|payment|pay(?:ment)?|method|methods|gift card|original payment)\b", latest):
        return False
    question_word = r"(?:what|which|where|how|available|options?|details)"
    payment_term = r"(?:refund|payment|pay(?:ment)?|methods?|gift card|original payment)"
    method_question = bool(
        re.search(rf"\b{question_word}\b[^.?!]{{0,100}}\b{payment_term}\b", latest)
        or re.search(rf"\b{payment_term}\b[^.?!]{{0,100}}\b{question_word}\b", latest)
        or re.search(r"\b(?:can|could)\b[^.?!]{0,60}\b(?:refund|payment)\b[^.?!]{0,60}\b(?:gift card|original payment|paypal|credit card)\b", latest)
    )
    if not method_question:
        return False
    explicit_methods_question = bool(
        re.search(r"\b(?:what|which)\b[^.?!]{0,80}\b(?:refund|payment)\b[^.?!]{0,80}\b(?:methods|options|available)\b", latest)
        or re.search(r"\b(?:refund|payment)\b[^.?!]{0,80}\b(?:methods|options)\b[^.?!]{0,80}\b(?:available|can|could|which|what)\b", latest)
    )
    if re.search(r"\b(exchange|swap|replace|update|change|changed|switch|modify)\b", latest) and not explicit_methods_question:
        return False
    return True


def refund_payment_method_choice_finalized(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> bool:
    latest = latest_user_text(messages).lower()
    if not latest:
        return False
    if re.search(r"\boriginal payment(?: method)?\b", latest):
        return True
    preferred = preferred_payment_method_id(messages, order, latest, include_all_user_text=False)
    if not preferred:
        return False
    return bool(
        re.search(
            r"\b(i'?ll|i will|i'?d prefer|i would prefer|prefer|go with|use|choose|chose|decide(?:d)?|"
            r"opt for|put it on|send it to|refund (?:it )?(?:to|on)|let'?s do|do it with|"
            r"process (?:my )?refund|finali[sz]e|that works|sounds good)\b",
            latest,
            re.I,
        )
    )


def eligible_refund_payment_method_ids(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> list[str]:
    methods: list[str] = []
    original = original_order_payment_method_id(order)
    if original:
        methods.append(original)
    for method_id, source in sorted(payment_methods_from_user_details(messages).items()):
        if source == "gift_card" and method_id not in methods:
            methods.append(method_id)
    return methods


def refund_payment_method_already_specified(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> bool:
    text = user_conversation_text(messages).lower()
    if re.search(r"\boriginal payment(?: method)?\b", text):
        return True
    preferred = preferred_payment_method_id(messages, order)
    return bool(preferred)


def refund_payment_method_selection_open(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> bool:
    if refund_payment_method_choice_finalized(messages, order):
        return False
    if refund_payment_method_question_seen(messages):
        return True
    text = user_conversation_text(messages).lower()
    if not re.search(r"\b(refund|payment|method|methods|gift card|paypal|credit card|original payment)\b", text):
        return False
    uncertainty = r"(?:not sure|unsure|undecided|haven'?t decided|have not decided|which|what kind|what type|options?|available|more information|leaning)"
    payment = r"(?:refund|payment|methods?|gift card|paypal|credit card|original payment)"
    return bool(
        re.search(rf"\b{uncertainty}\b[^.?!]{{0,140}}\b{payment}\b", text)
        or re.search(rf"\b{payment}\b[^.?!]{{0,140}}\b{uncertainty}\b", text)
    )


def return_refund_method_question(
    action: Action,
    messages: List[Dict[str, Any]],
) -> tuple[Action, dict[str, Any]] | None:
    if action.name != "return_delivered_order_items":
        return None
    order = latest_order_payload(messages, extract_order_id(messages, action))
    if not order:
        return None
    if not refund_payment_method_selection_open(messages, order):
        eligible_methods = eligible_refund_payment_method_ids(messages, order)
        if len(eligible_methods) <= 1 or refund_payment_method_already_specified(messages, order):
            return None
    response = refund_payment_method_options_response(messages, order) or fallback_response(
        "Which refund payment method should I use for this return?"
    )
    return response, {"eligible_payment_method_ids": eligible_refund_payment_method_ids(messages, order)}


def non_actionable_response(action: Action) -> bool:
    if action.name != RESPOND_ACTION_NAME:
        return False
    content = str(action.kwargs.get(RESPOND_ACTION_FIELD_NAME) or "").strip()
    if placeholder_text(content):
        return True
    content = content.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", content).strip()
    return normalized in {
        "could you clarify what you would like me to do next",
        "please clarify what you would like me to do next",
    }


def non_actionable_action(action: Action) -> bool:
    if non_actionable_response(action):
        return True
    if action.name == "transfer_to_human_agents":
        summary = action.kwargs.get("summary")
        if placeholder_text(summary):
            return True
    return False


def latest_user_confirmed_mutation_prompt(messages: List[Dict[str, Any]]) -> bool:
    latest = latest_user_text(messages)
    if not re.search(r"\b(yes|confirm|confirmed|proceed|go ahead|sounds good|that'?s correct)\b", latest, re.I):
        return False
    for message in reversed(messages):
        if str(message.get("role") or "").lower() == "user":
            continue
        action_payload = message.get("action") if isinstance(message.get("action"), dict) else None
        if not action_payload or action_payload.get("name") != RESPOND_ACTION_NAME:
            continue
        content = str((action_payload.get("kwargs") or {}).get(RESPOND_ACTION_FIELD_NAME) or "")
        if not re.search(r"\b(confirm|confirmation|proceed|reply with yes)\b", content, re.I):
            continue
        if mentions_action_intent(content, "cancel", "modify", "update", "return", "exchange", "book", "send"):
            return True
    return False


def confirmation_followup_response_needs_action(action: Action, messages: List[Dict[str, Any]]) -> bool:
    return action.name == RESPOND_ACTION_NAME and latest_user_confirmed_mutation_prompt(messages)


def latest_user_reconfirms_completed_mutation(messages: List[Dict[str, Any]]) -> bool:
    successful = latest_successful_mutating_action(messages)
    if successful is None:
        return False
    if latest_deferred_mutating_action(messages) is not None:
        return False
    if latest_user_mentions_different_primary_target(messages, successful):
        return False
    if last_message_is_successful_mutating_tool(messages):
        return False
    latest = latest_user_text(messages)
    if not latest:
        return False
    if not mentions_action_intent(latest, "cancel", "exchange", "return", "modify", "update", "book", "send"):
        return False
    if re.search(r"\b(yes|confirm|confirmed|proceed|go ahead|do it|sounds good|let'?s confirm|can you proceed)\b", latest, re.I):
        return True
    if re.search(
        r"\b(what|when|how|where|next steps?|mail|mailing|ship|shipping|refund|processed|process|instructions?)\b",
        latest,
        re.I,
    ):
        return True
    return bool(re.search(r"\b(i'?d like|i would like|could i|can i|please|thanks|thank you)\b", latest, re.I))


def completed_item_mutation_option_followup_response(messages: List[Dict[str, Any]]) -> Action | None:
    if latest_deferred_mutating_action(messages) is not None:
        return None
    action = latest_successful_mutating_action(messages)
    if action is None or action.name not in ONE_SHOT_ITEM_MUTATIONS:
        return None
    latest = latest_user_text(messages)
    if not latest:
        return None
    if not re.search(
        r"\b(options?|available|which|what color|what size|suggest|recommend|decide|between|would .+ work|confirm .+ available)\b",
        latest,
        re.I,
    ):
        return None
    return fallback_response(completed_mutation_response(messages, already=True))


def reason_evidence_text(messages: List[Dict[str, Any]]) -> str:
    after_reason_prompt: list[str] = []
    collecting = False
    for message in messages:
        action_payload = message.get("action") if isinstance(message.get("action"), dict) else None
        if action_payload and action_payload.get("name") == RESPOND_ACTION_NAME:
            content = str((action_payload.get("kwargs") or {}).get(RESPOND_ACTION_FIELD_NAME) or "")
            if re.search(r"\b(reason|why)\b", content, re.I):
                after_reason_prompt = []
                collecting = True
                continue
        if collecting and str(message.get("role") or "").lower() == "user":
            content = str(message.get("content") or "").strip()
            if content:
                after_reason_prompt.append(content)
    if after_reason_prompt:
        return "\n".join(after_reason_prompt)
    latest = latest_user_text(messages)
    if latest:
        return latest + "\n" + user_conversation_text(messages)
    return user_conversation_text(messages)


def infer_reason_from_text(messages: List[Dict[str, Any]], values: list[str]) -> str:
    if not values:
        return ""
    text = reason_evidence_text(messages).lower()
    if "no longer needed" in values and re.search(
        r"\b(no longer|don'?t need|do not need|not needed|don'?t actually need|do not actually need)\b",
        text,
    ):
        return "no longer needed"
    for value in values:
        if value.lower() in text:
            return value
    phrase_map = {
        "ordered by mistake": ("mistake", "accident", "wrong"),
        "no longer needed": (
            "no longer",
            "don't need",
            "do not need",
            "not needed",
            "full refund",
            "found the same item",
            "found it elsewhere",
            "found elsewhere",
            "half the price",
            "price difference",
            "better price",
            "cheaper store",
            "cheaper elsewhere",
            "buy from another",
            "buy it from another",
            "purchase from another",
            "purchase from the cheaper",
            "too expensive",
            "changed my mind",
            "won't be around",
            "will not be around",
            "won't be able to receive",
            "will not be able to receive",
            "can't receive",
            "cannot receive",
            "not receive",
            "traveling",
            "travelling",
            "out of town",
            "away when",
            "away by the time",
            "future date",
        ),
    }
    for value in values:
        if any(phrase in text for phrase in phrase_map.get(value, ())):
            return value
    return ""


def latest_tool_json(messages: List[Dict[str, Any]], tool_name: str) -> Any:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != tool_name:
            continue
        content = str(message.get("content") or "")
        if "error" in content.lower():
            continue
        parsed = parse_tool_json(content)
        if parsed is not None:
            return parsed
    return None


def product_detail_payloads(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != "get_product_details":
            continue
        parsed = parse_tool_json(message.get("content"))
        if isinstance(parsed, dict) and parsed.get("product_id"):
            payloads[str(parsed["product_id"])] = parsed
    return payloads


def latest_order_payload(messages: List[Dict[str, Any]], order_id: str = "") -> Dict[str, Any]:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "tool":
            continue
        tool_name = str(message.get("name") or "")
        parsed = parse_tool_json(message.get("content"))
        if not isinstance(parsed, dict):
            continue
        if not parsed.get("order_id") or (tool_name != RETAIL_ORDER_DETAIL_TOOL and "items" not in parsed):
            continue
        if order_id and str(parsed.get("order_id") or "") != order_id:
            continue
        return parsed
    return {}


def order_items_by_item_id(messages: List[Dict[str, Any]], order_id: str = "") -> Dict[str, Dict[str, Any]]:
    order = latest_order_payload(messages, order_id)
    items = order.get("items") if isinstance(order.get("items"), list) else []
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("item_id"):
            indexed[str(item["item_id"])] = item
    return indexed


def text_match_key(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("-", " ")
    text = re.sub(r"\bfeet\b", "ft", text)
    text = re.sub(r"\bfoot\b", "ft", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def option_value_mentioned(text: str, value: Any) -> bool:
    key = text_match_key(value)
    if not key:
        return False
    normalized = f" {text_match_key(text)} "
    return f" {key} " in normalized


def positive_option_value_mentioned(text: str, value: Any) -> bool:
    key = text_match_key(value)
    if not key:
        return False
    normalized = text_match_key(text)
    patterns = [
        rf"\b(?:want|prefer|like|choose|select|get|use|pick|decided on|go with|proceed with|replace with|exchange for|swap for|switch to|looking for)\s+(?:(?:a|an|the)\s+)?{re.escape(key)}\b",
        rf"\b(?:would\s+prefer|preferably|ideally|favor|favour|leaning\s+towards?|leaning\s+to)\s+(?:(?:a|an|the)\s+)?{re.escape(key)}\b",
        rf"\b(?:prefer|would\s+prefer|like|would\s+like)\b.{{0,50}}\b{re.escape(key)}\b",
        rf"\bfor\s+(?:(?:a|an|the)\s+)?(?:[a-z0-9]+\s+){{0,3}}{re.escape(key)}\b",
        rf"\b(?:in|with|to)\s+(?:(?:a|an|the)\s+)?{re.escape(key)}\b(?!\s+(?:and|or|options?|colou?rs?|sizes?|materials?|heights?)\b)",
        rf"\b{re.escape(key)}\s+(?:if|when)\s+(?:it\s+is\s+|its\s+|available\s+)?available\b",
        rf"\b{re.escape(key)}\s+(?:color|option|model|one|seems|sounds|would work|would be ideal|would be preferred|would be great|is ideal|is preferred|is fine|is ok|is okay)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def negative_option_value_mentioned(text: str, value: Any) -> bool:
    key = text_match_key(value)
    if not key:
        return False
    normalized = text_match_key(text)
    patterns = [
        rf"\b(?:not|no|avoid|without|except|besides|other than)\s+(?:(?:a|an|the)\s+)?{re.escape(key)}\b",
        rf"\b(?:anything|something|another|one)\s+(?:but|besides|except|other than)\s+{re.escape(key)}\b",
        rf"\b{re.escape(key)}\s+(?:won t work|wouldn t work|not preferred|not ok|not okay|is not preferred)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def preserve_option_key_mentioned(text: str, option_key: Any) -> bool:
    key = text_match_key(option_key)
    if not key:
        return False
    normalized = text_match_key(text)
    patterns = [
        rf"\b(?:not|don t|do not|doesn t|does not|no)\b.{{0,50}}\b(?:change|changing|different)\b.{{0,30}}\b{re.escape(key)}\b",
        rf"\b(?:not|don t|do not|doesn t|does not|no)\b.{{0,50}}\b{re.escape(key)}\b.{{0,30}}\b(?:change|changing|different)\b",
        rf"\b(?:same|current|original|existing)\b.{{0,30}}\b{re.escape(key)}\b",
        rf"\b(?:keep|preserve|leave)\b.{{0,30}}\b{re.escape(key)}\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def source_option_value_mentioned(text: str, value: Any) -> bool:
    key = text_match_key(value)
    if not key:
        return False
    normalized = text_match_key(text)
    return bool(re.search(rf"\bfrom\b.{{0,80}}\b{re.escape(key)}\b.{{0,120}}\bto\b", normalized))


def item_continuation_chunk(chunk: str, target_terms: set[str] | None = None) -> bool:
    normalized = text_match_key(chunk)
    if target_terms:
        tokens = normalized.split()
        offset = 0
        if tokens[:1] == ["for"]:
            offset = 1
        if offset < len(tokens) and tokens[offset] in {"a", "an", "the", "my", "this", "that"}:
            first = tokens[offset + 1] if offset + 1 < len(tokens) else ""
            phrase = tokens[offset + 1 : offset + 5]
            generic_refs = {
                "cheaper",
                "color",
                "height",
                "item",
                "items",
                "material",
                "model",
                "new",
                "one",
                "ones",
                "option",
                "options",
                "same",
                "size",
                "style",
                "type",
                "version",
            }
            if (
                first
                and first not in target_terms
                and first not in generic_refs
                and not any(token in {"model", "one", "ones", "option", "options", "version"} for token in phrase)
            ):
                return False
    return bool(
        re.search(
            r"\b(it|its|that|this|one|ones|them|those|these|instead|specifically|preferably|ideally|"
            r"available|option|options|either|whichever|whatever|fine|ok|okay|cheaper|cheapest|"
            r"lowest|least expensive|price|not|avoid|without|except|besides|other than)\b",
            normalized,
        )
    )


def other_order_item_terms(old_item: Dict[str, Any], messages: List[Dict[str, Any]]) -> set[str]:
    order = latest_order_payload(messages)
    items = order.get("items") if isinstance(order.get("items"), list) else []
    old_item_id = str(old_item.get("item_id") or "")
    terms: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if old_item_id and str(item.get("item_id") or "") == old_item_id:
            continue
        terms.update(item_mention_terms(item))
        for phrase in item_name_phrases(item):
            terms.update(term for term in phrase.split() if len(term) >= 4)
    return terms


def chunk_mentions_terms(chunk: str, terms: set[str]) -> bool:
    normalized = f" {text_match_key(chunk)} "
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)


def product_option_value_keys(product: Dict[str, Any]) -> set[str]:
    values: set[str] = set()
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    for variant in variants.values():
        if not isinstance(variant, dict):
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        for value in options.values():
            key = text_match_key(value)
            if key:
                values.add(key)
    return values


def choice_chunk_mentions_product_option(chunk: str, product: Dict[str, Any]) -> bool:
    normalized = f" {text_match_key(chunk)} "
    if not (
        final_item_choice_text(chunk)
        or re.search(r"\b(prefer|leaning|decided|choose|select|go with|proceed|would work|works|okay|ok)\b", normalized)
    ):
        return False
    return any(f" {value} " in normalized for value in product_option_value_keys(product))


def item_context_chunks(text: str) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(
            r"(?<=[.!?])\s+|[;\n]+|,\s*(?:and|but|also)\s+|"
            r"\band\s+(?=(?:a|an|the|my|this|that)\s+)|"
            r"\band\s+(?=(?:exchange(?:s|d|ing)?|swap(?:s|ped|ping)?|replace(?:s|d|ment|ing)?|modif(?:y|ies|ied|ying)|change(?:s|d|ing)?)\b)|"
            r"\balso[, ]+\s*",
            text,
            flags=re.I,
        )
        if chunk.strip()
    ]


def followup_choice_chunks_for_product(
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
    other_terms: set[str],
) -> list[str]:
    product_terms = {
        term
        for term in re.findall(r"[A-Za-z0-9]+", str(product.get("name") or "").lower())
        if len(term) > 2
    }
    option_prompt_seen = False
    selected: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            action = action_from_dict(message.get("action")) if isinstance(message.get("action"), dict) else None
            content = str((action.kwargs if action else {}).get(RESPOND_ACTION_FIELD_NAME) or message.get("content") or "")
            normalized = text_match_key(content)
            if (
                action is None
                or action.name == RESPOND_ACTION_NAME
            ) and "available" in normalized and any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in product_terms):
                option_prompt_seen = True
            continue
        if role != "user" or not option_prompt_seen:
            continue
        product_context_active = False
        for chunk in item_context_chunks(str(message.get("content") or "")):
            mentions_product = any(re.search(rf"\b{re.escape(term)}\b", chunk, flags=re.I) for term in product_terms)
            mentions_other = bool(other_terms and chunk_mentions_terms(chunk, other_terms))
            if mentions_other and not mentions_product:
                product_context_active = False
                continue
            if choice_chunk_mentions_product_option(chunk, product) and (mentions_product or product_context_active):
                selected.append(chunk)
                product_context_active = True
                continue
            if mentions_product:
                product_context_active = True
                continue
            if not item_continuation_chunk(chunk, product_terms):
                product_context_active = False
    return selected


def item_context_text(old_item: Dict[str, Any], product: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    text = user_conversation_text(messages)
    names = " ".join(str(value or "") for value in (old_item.get("name"), product.get("name")))
    terms = {term for term in re.findall(r"[A-Za-z0-9]+", names.lower()) if len(term) > 2}
    other_terms = other_order_item_terms(old_item, messages)
    chunks = item_context_chunks(text)
    selected_indices = [
        index
        for index, chunk in enumerate(chunks)
        if any(re.search(rf"\b{re.escape(term)}\b", chunk, flags=re.I) for term in terms)
    ]
    included_indices = set(selected_indices)
    for index in selected_indices:
        next_index = index + 1
        if (
            next_index < len(chunks)
            and item_continuation_chunk(chunks[next_index], terms)
            and not (other_terms and chunk_mentions_terms(chunks[next_index], other_terms))
        ):
            included_indices.add(next_index)
    selected = [chunk for index, chunk in enumerate(chunks) if index in included_indices]
    selected.extend(chunk for chunk in followup_choice_chunks_for_product(product, messages, other_terms) if chunk not in selected)
    return " ".join(selected) if selected else text


def price_preference_seen(messages: List[Dict[str, Any]], text: str = "") -> bool:
    source = text or user_conversation_text(messages)
    return bool(re.search(r"\b(cheapest|cheaper|lowest price|least expensive|lower price)\b", source, re.I))


def option_values_indifferent(text: str, values: set[str]) -> bool:
    mentioned = sorted({text_match_key(value) for value in values if value and option_value_mentioned(text, value)})
    if len(mentioned) < 2:
        return False
    normalized = text_match_key(text)
    indifference = r"(?:fine|ok|okay|either|whichever|whatever|don t care|do not care|doesn t matter|does not matter|no preference|don t mind|do not mind)"
    for index, left in enumerate(mentioned):
        for right in mentioned[index + 1 :]:
            either_order = (
                rf"\b{re.escape(left)}\b.{{0,20}}\b(?:or|and)\b.{{0,20}}\b{re.escape(right)}\b"
                rf"|\b{re.escape(right)}\b.{{0,20}}\b(?:or|and)\b.{{0,20}}\b{re.escape(left)}\b"
            )
            if re.search(rf"(?:{either_order}).{{0,80}}\b{indifference}\b", normalized):
                return True
            if re.search(rf"\b{indifference}\b.{{0,80}}(?:{either_order})", normalized):
                return True
    return False


def final_item_choice_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(decided|go ahead|proceed|can we proceed|set on|i'?m set|i would like|i'd like|let'?s|final choice|that should cover)\b",
            text,
            re.I,
        )
    )


IDENTIFIER_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "item",
    "items",
    "like",
    "need",
    "one",
    "order",
    "please",
    "product",
    "card",
    "that",
    "the",
    "this",
    "want",
    "with",
    "yes",
    "no",
    "true",
    "false",
    "full",
}


def item_mention_terms(item: Dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    name = str(item.get("name") or "")
    for term in re.findall(r"[a-z0-9]+", text_match_key(name)):
        if len(term) >= 4 and term not in IDENTIFIER_STOPWORDS:
            terms.add(term)
    options = item.get("options") if isinstance(item.get("options"), dict) else {}
    for value in options.values():
        normalized = text_match_key(value)
        if not normalized:
            continue
        if len(normalized) >= 3 and normalized not in IDENTIFIER_STOPWORDS:
            terms.add(normalized)
        for term in re.findall(r"[a-z0-9]+", normalized):
            if len(term) >= 4 and term not in IDENTIFIER_STOPWORDS:
                terms.add(term)
    return terms


def item_name_phrases(item: Dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for value in (item.get("name"), item.get("product_name")):
        key = text_match_key(value)
        if key and len(key.split()) >= 2 and key not in phrases:
            phrases.append(key)
    return phrases


def item_mention_position(
    item: Dict[str, Any],
    text: str,
    term_counts: Dict[str, int] | None = None,
) -> int | None:
    normalized = f" {text_match_key(text)} "
    positions: list[int] = []
    for phrase in item_name_phrases(item):
        pos = normalized.find(f" {phrase} ")
        if pos >= 0:
            positions.append(pos)
    for term in item_mention_terms(item):
        if term_counts is not None and term_counts.get(term, 0) > 1:
            continue
        needle = f" {term} "
        pos = normalized.find(needle)
        if pos >= 0:
            positions.append(pos)
    return min(positions) if positions else None


def item_change_intent_text(text: str) -> bool:
    return mentions_action_intent(text, "exchange", "return", "modify", "update")


def text_mentions_item_phrase(text: str, items: list[Any]) -> bool:
    normalized = f" {text_match_key(text)} "
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(f" {phrase} " in normalized for phrase in item_name_phrases(item)):
            return True
    return False


def item_actionable_text(text: str, items: list[Any]) -> bool:
    return item_change_intent_text(text) or text_mentions_item_phrase(text, items)


def additive_item_followup_text(text: str) -> bool:
    return bool(re.search(r"\b(also|too|as well|same order|add|include)\b", text, re.I))


def followup_messages_for_order(user_messages: list[str], start_index: int, order_id: str) -> list[str]:
    selected: list[str] = []
    for message in user_messages[start_index:]:
        explicit_order_ids = order_ids_in_text(message)
        if explicit_order_ids and order_id not in explicit_order_ids:
            continue
        selected.append(message)
    return selected


def item_request_text(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> str:
    user_messages = [
        str(message.get("content") or "")
        for message in messages
        if str(message.get("role") or "").lower() == "user" and str(message.get("content") or "").strip()
    ]
    if not user_messages:
        return ""
    latest = user_messages[-1]
    order_id = str(order.get("order_id") or "")
    items = order.get("items") if isinstance(order.get("items"), list) else []
    if order_id and order_id in latest and item_actionable_text(latest, items):
        scoped = text_segment_for_order(latest, order_id)
        return scoped or latest
    if text_mentions_item_phrase(latest, items):
        latest_rejects_item = any(
            isinstance(item, dict) and latest_user_rejects_item_topic(messages, item, item)
            for item in items
        )
        if not latest_rejects_item:
            if additive_item_followup_text(latest):
                for index in range(len(user_messages) - 2, -1, -1):
                    if item_actionable_text(user_messages[index], items):
                        return " ".join(user_messages[index:])
                    if order_id and order_id in user_messages[index]:
                        return " ".join(user_messages[index:])
            return latest

    if order_id:
        for index in range(len(user_messages) - 1, -1, -1):
            if order_id not in user_messages[index] or not item_actionable_text(user_messages[index], items):
                continue
            scoped = text_segment_for_order(user_messages[index], order_id)
            followups = followup_messages_for_order(user_messages, index + 1, order_id)
            if scoped:
                return " ".join([scoped, *followups])
            return " ".join([user_messages[index], *followups])
        for index in range(len(user_messages) - 1, -1, -1):
            if order_id in user_messages[index]:
                scoped = text_segment_for_order(user_messages[index], order_id)
                followups = followup_messages_for_order(user_messages, index + 1, order_id)
                if scoped:
                    return " ".join([scoped, *followups])
                return " ".join([user_messages[index], *followups])
    return " ".join(user_messages)


def requested_order_items(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> list[Dict[str, Any]]:
    text = item_request_text(messages, order)
    items = order.get("items") if isinstance(order.get("items"), list) else []
    term_counts: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for term in item_mention_terms(item):
            term_counts[term] = term_counts.get(term, 0) + 1
    positioned: list[tuple[int, Dict[str, Any]]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if latest_user_rejects_item_topic(messages, item, item):
            continue
        position = item_mention_position(item, text, term_counts)
        if position is not None:
            positioned.append((position * 100 + index, item))
    positioned.sort(key=lambda pair: pair[0])
    return [item for _position, item in positioned]


def order_payment_method_id(order: Dict[str, Any]) -> str:
    history = order.get("payment_history") if isinstance(order.get("payment_history"), list) else []
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        payment_method_id = entry.get("payment_method_id")
        if payment_method_id and str(entry.get("transaction_type") or "").lower() == "payment":
            return str(payment_method_id)
    for entry in reversed(history):
        if isinstance(entry, dict) and entry.get("payment_method_id"):
            return str(entry["payment_method_id"])
    return ""


def original_order_payment_method_id(order: Dict[str, Any]) -> str:
    history = order.get("payment_history") if isinstance(order.get("payment_history"), list) else []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        payment_method_id = entry.get("payment_method_id")
        if payment_method_id and str(entry.get("transaction_type") or "").lower() == "payment":
            return str(payment_method_id)
    return order_payment_method_id(order)


PAYMENT_ID_RE = re.compile(r"\b(?:credit_card|paypal|gift_card)_\d+\b", re.I)
PAYMENT_SOURCE_TERMS = {
    "paypal": ("paypal",),
    "credit_card": ("credit card", "credit-card", "visa", "mastercard", "amex", "card"),
    "gift_card": ("gift card", "gift-card"),
}


def payment_source_from_id(payment_method_id: str) -> str:
    value = str(payment_method_id or "").lower()
    for source in PAYMENT_SOURCE_TERMS:
        if value.startswith(source):
            return source
    return ""


def payment_methods_from_user_details(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if str(message.get("name") or "") != USER_DETAIL_TOOL:
            continue
        parsed = parse_tool_json(message.get("content"))
        if not isinstance(parsed, dict):
            continue
        raw_methods = parsed.get("payment_methods")
        if not isinstance(raw_methods, dict):
            continue
        for method_key, raw_method in raw_methods.items():
            method_id = str(method_key)
            source = payment_source_from_id(method_id)
            if isinstance(raw_method, dict):
                method_id = str(raw_method.get("id") or method_id)
                source = str(raw_method.get("source") or source)
            if method_id:
                methods[method_id] = source or payment_source_from_id(method_id)
    return methods


def payment_source_mentions(text: str) -> list[tuple[int, str]]:
    lowered = str(text or "").lower()
    mentions: list[tuple[int, str]] = []
    for source, terms in PAYMENT_SOURCE_TERMS.items():
        for term in terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", lowered):
                if source == "credit_card" and term == "card":
                    before = lowered[max(0, match.start() - 8) : match.start()]
                    if re.search(r"\bgift[-\s]*$", before):
                        continue
                mentions.append((match.start(), source))
    mentions.sort()
    return mentions


def payment_method_for_source(
    source: str,
    messages: List[Dict[str, Any]],
    order: Dict[str, Any] | None = None,
) -> str:
    if not source:
        return ""
    methods = payment_methods_from_user_details(messages)
    matching_user_methods = [
        method_id
        for method_id, method_source in methods.items()
        if method_source == source or payment_source_from_id(method_id) == source
    ]
    if len(matching_user_methods) == 1:
        return matching_user_methods[0]
    history = (order or {}).get("payment_history") if isinstance((order or {}).get("payment_history"), list) else []
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        method_id = str(entry.get("payment_method_id") or "")
        if payment_source_from_id(method_id) == source:
            return method_id
    if matching_user_methods:
        return sorted(matching_user_methods)[0]
    return ""


def preferred_payment_method_id(
    messages: List[Dict[str, Any]],
    order: Dict[str, Any] | None = None,
    text: str = "",
    *,
    include_all_user_text: bool = True,
) -> str:
    sources = [text or latest_user_text(messages)]
    all_user_text = user_conversation_text(messages)
    if include_all_user_text and all_user_text and all_user_text not in sources:
        sources.append(all_user_text)
    for source_text in sources:
        exact_matches = list(PAYMENT_ID_RE.finditer(source_text))
        source_mentions = payment_source_mentions(source_text)
        if source_mentions:
            last_exact_pos = exact_matches[-1].start() if exact_matches else -1
            for source_pos, source in reversed(source_mentions):
                if source_pos <= last_exact_pos:
                    continue
                source_method = payment_method_for_source(source, messages, order)
                if source_method:
                    return source_method
        if exact_matches:
            return exact_matches[-1].group(0)
        if source_mentions:
            source_method = payment_method_for_source(source_mentions[-1][1], messages, order)
            if source_method:
                return source_method
    return ""


def payment_source_preferred_over_later_source(text: str) -> str:
    lowered = str(text or "").lower()
    mentions = payment_source_mentions(lowered)
    for index in range(len(mentions) - 1):
        left_pos, left_source = mentions[index]
        right_pos, _right_source = mentions[index + 1]
        if right_pos - left_pos > 100:
            continue
        between = lowered[left_pos:right_pos]
        if re.search(r"\b(instead of|rather than|over)\b", between):
            return left_source
    for index, (pos, source) in enumerate(mentions):
        before = lowered[max(0, pos - 40) : pos]
        if not re.search(r"\b(not|avoid|don t use|do not use|instead of|rather than)\b", before):
            continue
        for _next_pos, next_source in mentions[index + 1 :]:
            return next_source
        return ""
    return ""


def order_payment_update_source(text: str) -> str:
    lowered = str(text or "").lower()
    mentions = payment_source_mentions(lowered)
    for index in range(len(mentions) - 1, -1, -1):
        pos, source = mentions[index]
        previous_source_pos = mentions[index - 1][0] if index > 0 else 0
        before = lowered[max(previous_source_pos, pos - 160) : pos]
        if re.search(
            r"\b(change|update|switch|modify|set)\b.{0,120}\b(payment|paying|method|pay)\b",
            before,
        ):
            return source
        if re.search(
            r"\b(payment|paying|method)\b.{0,100}\b(to|with|via|by|use|using|on)\W*$",
            before,
        ):
            return source
        if "order" in before and re.search(
            r"\b(change|update|switch|modify|set)\b.{0,100}\b(to|with|via|by|use|using|on)\W*$",
            before,
        ):
            return source
    return ""


def payment_adjustment_source(text: str) -> str:
    lowered = str(text or "").lower()
    preferred = payment_source_preferred_over_later_source(lowered)
    if preferred:
        return preferred
    for pos, source in reversed(payment_source_mentions(lowered)):
        window = lowered[max(0, pos - 100) : pos + 100]
        if re.search(
            r"\b(pay|paid|payment|refund|refunded|refunds|charge|charged|credit|credited|difference)\b",
            window,
        ):
            return source
    return ""


def order_payment_update_method_id(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> str:
    text = item_request_text(messages, order)
    source = order_payment_update_source(text)
    if source:
        return payment_method_for_source(source, messages, order)
    return ""


def item_mutation_payment_method_id(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> str:
    text = item_request_text(messages, order)
    source = payment_adjustment_source(text)
    if source:
        method_id = payment_method_for_source(source, messages, order)
        if method_id:
            return method_id
    return preferred_payment_method_id(messages, order, text, include_all_user_text=not bool(text)) or order_payment_method_id(order)


def return_payment_method_id(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> str:
    original = original_order_payment_method_id(order)
    preferred = preferred_payment_method_id(messages, order)
    if not preferred:
        return original
    if preferred == original or payment_source_from_id(preferred) == "gift_card":
        return preferred
    if payment_source_from_id(preferred) == payment_source_from_id(original):
        return original
    return original


def refund_payment_method_options_response(messages: List[Dict[str, Any]], order: Dict[str, Any]) -> Action | None:
    original = original_order_payment_method_id(order)
    methods = payment_methods_from_user_details(messages)
    gift_cards = sorted(method_id for method_id, source in methods.items() if source == "gift_card")
    parts: list[str] = []
    if original:
        parts.append(f"the original payment method ({original})")
    if gift_cards:
        parts.append("existing gift card " + ", ".join(gift_cards))
    if not parts:
        return None
    content = "For a return refund, the refund can go to " + " or ".join(parts) + "."
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: content})


def refund_payment_method_action(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if not refund_payment_method_question_seen(messages):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        if order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
            return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id})
        return None
    if refund_payment_method_choice_finalized(messages, order):
        return None
    user_id = extract_user_id(messages)
    if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
        return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id})

    return refund_payment_method_options_response(messages, order)


def pending_payment_update_before_item_mutation(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if action.name != "modify_pending_order_items" or "modify_pending_order_payment" not in available_tool_names:
        return None
    order_id = extract_order_id(messages, action)
    order = latest_order_payload(messages, order_id)
    if not order or not str(order.get("status") or "").lower().startswith("pending"):
        return None
    if latest_user_mentions_different_primary_target(messages, action):
        desired_payment_method_id = str(action.kwargs.get("payment_method_id") or "")
    else:
        desired_payment_method_id = order_payment_update_method_id(messages, order)
    if not desired_payment_method_id:
        return None
    if desired_payment_method_id == order_payment_method_id(order):
        return None
    if action_error_was_seen(messages, "modify_pending_order_payment", "order_id", str(order.get("order_id") or order_id)):
        return None
    return Action(
        name="modify_pending_order_payment",
        kwargs={"order_id": str(order.get("order_id") or order_id), "payment_method_id": desired_payment_method_id},
    )


def pending_item_mutation_payment_method_question(action: Action, messages: List[Dict[str, Any]]) -> Action | None:
    if action.name != "modify_pending_order_items":
        return None
    order_id = extract_order_id(messages, action)
    order = latest_order_payload(messages, order_id)
    if not order or not str(order.get("status") or "").lower().startswith("pending"):
        return None
    if preferred_payment_method_id(messages, order, item_request_text(messages, order)):
        return None
    current = order_payment_method_id(order)
    if payment_source_from_id(current) != "gift_card":
        return None
    methods = payment_methods_from_user_details(messages)
    alternatives = [
        (method_id, source)
        for method_id, source in sorted(methods.items())
        if method_id != current and source != "gift_card"
    ]
    if not alternatives:
        return None
    option_text = ", ".join(f"{source.replace('_', ' ')} ({method_id})" for method_id, source in alternatives)
    if current:
        option_text += f", or gift card ({current})"
    return fallback_response(
        "Which payment or refund method should I use for this item change: "
        + option_text
        + "?"
    )


def repair_item_mutation_payment_method(
    action: Action,
    messages: List[Dict[str, Any]],
) -> tuple[Action, dict[str, Any]]:
    if action.name not in ONE_SHOT_ITEM_MUTATIONS and action.name not in SINGLE_LIST_ITEM_MUTATIONS:
        return action, {}
    order_id = extract_order_id(messages, action)
    order = latest_order_payload(messages, order_id)
    if not order:
        return action, {}
    if action.kwargs.get("payment_method_id") and latest_user_mentions_different_primary_target(messages, action):
        return action, {}
    desired = return_payment_method_id(messages, order) if action.name == "return_delivered_order_items" else item_mutation_payment_method_id(messages, order)
    if not desired or action.kwargs.get("payment_method_id") == desired:
        return action, {}
    repaired = dict(action.kwargs)
    previous = repaired.get("payment_method_id")
    repaired["payment_method_id"] = desired
    return Action(name=action.name, kwargs=repaired), {"payment_method_id": {"from": previous, "to": desired}}


def desired_option_constraints(
    old_item: Dict[str, Any],
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, set[str]]:
    text = item_context_text(old_item, product, messages)
    old_options = old_item.get("options") if isinstance(old_item.get("options"), dict) else {}
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    option_values: Dict[str, Dict[str, str]] = {}
    for variant in variants.values():
        if not isinstance(variant, dict):
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        for key, value in options.items():
            option_key = str(key)
            option_values.setdefault(option_key, {})[text_match_key(value)] = str(value)

    constraints: Dict[str, set[str]] = {}
    for option_key, values_by_key in option_values.items():
        old_value = str(old_options.get(option_key) or "")
        if old_value and preserve_option_key_mentioned(text, option_key):
            constraints[option_key] = {text_match_key(old_value)}
            continue
        mentioned: list[str] = []
        for value in values_by_key.values():
            if option_value_mentioned(text, value):
                mentioned.append(value)
        if not mentioned:
            continue
        positive_desired: list[str] = []
        neutral_desired: list[str] = []
        nonnegative_mentioned: list[str] = []
        for value in mentioned:
            if negative_option_value_mentioned(text, value):
                continue
            nonnegative_mentioned.append(value)
            is_positive = positive_option_value_mentioned(text, value) and not source_option_value_mentioned(text, value)
            if old_value and text_match_key(value) == text_match_key(old_value):
                if is_positive:
                    positive_desired.append(value)
                continue
            if is_positive:
                positive_desired.append(value)
            else:
                neutral_desired.append(value)
        if option_values_indifferent(text, {text_match_key(value) for value in nonnegative_mentioned}):
            desired = [
                value
                for value in nonnegative_mentioned
                if not source_option_value_mentioned(text, value)
            ]
        else:
            desired = positive_desired if positive_desired else neutral_desired
        if not desired and old_value and positive_option_value_mentioned(text, old_value):
            if not source_option_value_mentioned(text, old_value):
                desired.append(old_value)
        if desired:
            constraints[option_key] = {text_match_key(value) for value in desired}
    return constraints


def choose_variant_for_item(
    old_item: Dict[str, Any],
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> str:
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    constraints = desired_option_constraints(old_item, product, messages)
    if not constraints:
        return ""

    candidates: list[tuple[float, int, float, str]] = []
    old_item_id = str(old_item.get("item_id") or "")
    old_options = old_item.get("options") if isinstance(old_item.get("options"), dict) else {}
    context = item_context_text(old_item, product, messages)
    indifferent_option_keys = {
        option_key
        for option_key, allowed_values in constraints.items()
        if len(allowed_values) > 1 and option_values_indifferent(context, allowed_values)
    }
    prefer_price = price_preference_seen(messages, context) or bool(indifferent_option_keys)
    for variant_id, variant in variants.items():
        if not isinstance(variant, dict):
            continue
        item_id = str(variant.get("item_id") or variant_id)
        if item_id == old_item_id:
            continue
        if variant.get("available") is False:
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        if any(negative_option_value_mentioned(context, value) for value in options.values()):
            continue
        matched = 0
        rejected = False
        for option_key, allowed_values in constraints.items():
            if text_match_key(options.get(option_key)) not in allowed_values:
                rejected = True
                break
            matched += 1
        if rejected:
            continue
        preserved = 0
        for option_key, old_value in old_options.items():
            if option_key in constraints:
                continue
            if option_key in indifferent_option_keys:
                continue
            if old_value and text_match_key(options.get(option_key)) == text_match_key(old_value):
                preserved += 1
        price = variant.get("price")
        numeric_price = float(price) if isinstance(price, (int, float)) else float("inf")
        primary = numeric_price if prefer_price else -matched
        secondary = -matched if prefer_price else -preserved
        tertiary = -preserved if prefer_price else numeric_price
        candidates.append((primary, secondary, tertiary, item_id))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][3]


def product_option_keys(product: Dict[str, Any]) -> list[str]:
    keys: list[str] = []
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    for variant in variants.values():
        if not isinstance(variant, dict):
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        for key in options:
            text = str(key)
            if text not in keys:
                keys.append(text)
    return keys


def queried_option_keys(product: Dict[str, Any], text: str) -> list[str]:
    normalized = text_match_key(text)
    keys = product_option_keys(product)
    selected: list[str] = []
    for key in keys:
        key_text = text_match_key(key)
        key_terms = [term for term in key_text.split() if len(term) > 2]
        if key_text and f" {key_text} " in f" {normalized} ":
            selected.append(key)
            continue
        if any(re.search(rf"\b{re.escape(term)}s?\b", normalized) for term in key_terms):
            selected.append(key)
    if selected:
        return selected
    if re.search(r"\b(options?|available|which|what)\b", normalized):
        return keys
    return []


def explicitly_queried_option_keys(product: Dict[str, Any], text: str) -> list[str]:
    normalized = text_match_key(text)
    selected: list[str] = []
    for key in product_option_keys(product):
        key_text = text_match_key(key)
        key_terms = [term for term in key_text.split() if len(term) > 2]
        if key_text and f" {key_text} " in f" {normalized} ":
            selected.append(key)
            continue
        if any(re.search(rf"\b{re.escape(term)}s?\b", normalized) for term in key_terms):
            selected.append(key)
    return selected


def available_option_values_for_item(
    old_item: Dict[str, Any],
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
    option_key: str,
) -> list[str]:
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    constraints = desired_option_constraints(old_item, product, messages)
    constraints.pop(option_key, None)
    values: list[str] = []
    for variant in variants.values():
        if not isinstance(variant, dict) or variant.get("available") is False:
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        rejected = False
        for key, allowed_values in constraints.items():
            if text_match_key(options.get(key)) not in allowed_values:
                rejected = True
                break
        if rejected:
            continue
        value = str(options.get(option_key) or "")
        if negative_option_value_mentioned(latest_user_text(messages), value):
            continue
        if value and value not in values:
            values.append(value)
    return values


def candidate_variants_for_item(
    old_item: Dict[str, Any],
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    variants = product.get("variants") if isinstance(product.get("variants"), dict) else {}
    constraints = desired_option_constraints(old_item, product, messages)
    context = item_context_text(old_item, product, messages)
    old_item_id = str(old_item.get("item_id") or "")
    candidates: list[Dict[str, Any]] = []
    for variant_id, variant in variants.items():
        if not isinstance(variant, dict):
            continue
        item_id = str(variant.get("item_id") or variant_id)
        if item_id == old_item_id:
            continue
        if variant.get("available") is False:
            continue
        options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
        if any(negative_option_value_mentioned(context, value) for value in options.values()):
            continue
        if any(text_match_key(options.get(key)) not in allowed for key, allowed in constraints.items()):
            continue
        candidates.append(variant)
    return candidates


def option_values_look_measured(values: set[str]) -> bool:
    nonempty = [value for value in values if value]
    if not nonempty:
        return False
    measured = [value for value in nonempty if re.search(r"\d", value)]
    return len(measured) >= max(1, len(nonempty) // 2)


def ambiguous_unconstrained_option_keys(
    old_item: Dict[str, Any],
    product: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> list[str]:
    context = item_context_text(old_item, product, messages)
    if price_preference_seen(messages, context):
        return []
    constraints = desired_option_constraints(old_item, product, messages)
    candidates = candidate_variants_for_item(old_item, product, messages)
    if len(candidates) <= 1:
        return []
    old_options = old_item.get("options") if isinstance(old_item.get("options"), dict) else {}
    ambiguous: list[str] = []
    for option_key in product_option_keys(product):
        if option_key in constraints:
            continue
        values = {
            text_match_key((variant.get("options") if isinstance(variant.get("options"), dict) else {}).get(option_key))
            for variant in candidates
        }
        values.discard("")
        old_value = old_options.get(option_key)
        if old_value and option_values_look_measured(values):
            continue
        if len(values) > 1:
            ambiguous.append(option_key)
    return ambiguous


def unresolved_item_option_response(action: Action, messages: List[Dict[str, Any]]) -> Action | None:
    if action.name not in ONE_SHOT_ITEM_MUTATIONS:
        return None
    item_ids = action.kwargs.get("item_ids")
    if not isinstance(item_ids, list) or not item_ids:
        return None
    new_item_ids = action.kwargs.get("new_item_ids")
    if (
        isinstance(new_item_ids, list)
        and len(new_item_ids) == len(item_ids)
        and all(str(item_id or "").strip() for item_id in new_item_ids)
        and latest_user_mentions_different_primary_target(messages, action)
    ):
        return None
    order_id = extract_order_id(messages, action)
    order_items = order_items_by_item_id(messages, order_id)
    products = product_detail_payloads(messages)
    snippets: list[str] = []
    for raw_item_id in item_ids:
        old_item = order_items.get(str(raw_item_id))
        if not old_item:
            continue
        product = products.get(str(old_item.get("product_id") or ""))
        if not product:
            continue
        if latest_user_rejects_item_topic(messages, old_item, product):
            continue
        context = item_context_text(old_item, product, messages)
        constraints = desired_option_constraints(old_item, product, messages)
        option_keys = explicitly_queried_option_keys(product, context)
        for option_key in ambiguous_unconstrained_option_keys(old_item, product, messages):
            if option_key not in option_keys:
                option_keys.append(option_key)
        for option_key in option_keys:
            if option_key in constraints:
                continue
            values = available_option_values_for_item(old_item, product, messages, option_key)
            if len(values) <= 1:
                continue
            product_name = str(product.get("name") or old_item.get("name") or "item")
            values_text = ", ".join(sorted(values, key=text_match_key))
            snippets.append(f"For {product_name}, available {option_key} options are: {values_text}.")
    if not snippets:
        return None
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: " ".join(snippets)})


def price_question_seen(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if not re.search(r"\b(price|prices|cost|costs|affordable|cheaper|cheapest|less expensive|more expensive|how much)\b", lowered):
        return False
    return bool(
        "?" in text
        or re.search(r"\b(which|what|how much|tell me|show me|check|compare|comparison|between)\b", lowered)
        or re.search(r"\b(price|prices|cost|costs)\b", lowered)
    )


def variant_price(variant: Dict[str, Any]) -> float:
    price = variant.get("price")
    if isinstance(price, (int, float)):
        return float(price)
    try:
        return float(str(price))
    except (TypeError, ValueError):
        return float("inf")


def variant_option_summary(variant: Dict[str, Any]) -> str:
    options = variant.get("options") if isinstance(variant.get("options"), dict) else {}
    parts = [str(value) for _key, value in options.items() if str(value or "").strip()]
    return " ".join(parts) if parts else str(variant.get("item_id") or "option")


def catalog_price_response(messages: List[Dict[str, Any]]) -> Action | None:
    latest = latest_user_text(messages)
    if not price_question_seen(latest):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        return None
    products = product_detail_payloads(messages)
    if not products:
        return None
    target_items = requested_order_items(messages, order)
    if not target_items:
        return None
    snippets: list[str] = []
    for item in target_items:
        product_id = str(item.get("product_id") or "")
        product = products.get(product_id)
        if not product:
            continue
        context = item_context_text(item, product, messages)
        if not price_question_seen(context):
            continue
        priced_candidates: list[tuple[float, str, Dict[str, Any]]] = []
        for variant in candidate_variants_for_item(item, product, messages):
            price = variant_price(variant)
            if price == float("inf"):
                continue
            item_id = str(variant.get("item_id") or "")
            priced_candidates.append((price, item_id, variant))
        if not priced_candidates:
            continue
        priced_candidates.sort(key=lambda entry: (entry[0], entry[1]))
        product_name = str(product.get("name") or item.get("name") or "item")
        cheapest_price, _cheapest_id, cheapest = priced_candidates[0]
        comparisons = []
        for price, _item_id, variant in priced_candidates[:4]:
            comparisons.append(f"{variant_option_summary(variant)} is ${price:.2f}")
        snippets.append(
            f"For {product_name}, {'; '.join(comparisons)}. "
            f"The cheapest matching available option is {variant_option_summary(cheapest)} at ${cheapest_price:.2f}."
        )
    if not snippets:
        return None
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: " ".join(snippets)})


def catalog_option_response(messages: List[Dict[str, Any]]) -> Action | None:
    latest = latest_user_text(messages)
    if refund_payment_method_question_seen(messages):
        return None
    if final_item_choice_text(latest) and not re.search(
        r"\b(what color|which color|what size|which size|available|availability|do you have|color options?|size options?|material options?|height options?)\b",
        latest,
        re.I,
    ):
        return None
    if not re.search(
        r"\b(?:what|which|available|availability|do you have|more information|different colors|"
        r"color options?|size options?|material options?|height options?)\b",
        latest,
        re.I,
    ):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        return None
    products = product_detail_payloads(messages)
    if not products:
        return None
    target_items = requested_order_items(messages, order)
    if not target_items:
        return None
    snippets: list[str] = []
    for item in target_items:
        product_id = str(item.get("product_id") or "")
        product = products.get(product_id)
        if not product:
            continue
        option_keys = queried_option_keys(product, latest)
        for option_key in option_keys:
            values = available_option_values_for_item(item, product, messages, option_key)
            if not values:
                continue
            product_name = str(product.get("name") or item.get("name") or "item")
            values_text = ", ".join(sorted(values, key=text_match_key))
            snippets.append(f"For {product_name}, available {option_key} options are: {values_text}.")
    if not snippets:
        return None
    return Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: " ".join(snippets)})


def catalog_option_lookup_action(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if "get_product_details" not in available_tool_names:
        return None
    latest = latest_user_text(messages)
    if refund_payment_method_question_seen(messages):
        return None
    if not re.search(
        r"\b(?:what|which|available|availability|do you have|more information|different colors|"
        r"color options?|size options?|material options?|height options?)\b",
        latest,
        re.I,
    ):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        return None
    target_items = requested_order_items(messages, order)
    if not target_items:
        return None
    products = product_detail_payloads(messages)
    for item in target_items:
        product_id = str(item.get("product_id") or "")
        if product_id and product_id not in products:
            return Action(name="get_product_details", kwargs={"product_id": product_id})
    return None


def missing_product_detail_for_item_mutation(
    action: Action,
    messages: List[Dict[str, Any]],
) -> str:
    if action.name not in ONE_SHOT_ITEM_MUTATIONS:
        return ""
    item_ids = action.kwargs.get("item_ids")
    if not isinstance(item_ids, list):
        return ""
    order_id = extract_order_id(messages, action)
    order_items = order_items_by_item_id(messages, order_id)
    products = product_detail_payloads(messages)
    for raw_item_id in item_ids:
        item = order_items.get(str(raw_item_id))
        if not item:
            continue
        product_id = str(item.get("product_id") or "")
        if product_id and product_id not in products:
            return product_id
    return ""


def latest_user_rejects_item_topic(messages: List[Dict[str, Any]], item: Dict[str, Any], product: Dict[str, Any]) -> bool:
    latest = latest_user_text(messages)
    if not latest:
        return False
    if not re.search(
        r"\b(?:don'?t|do not)\s+(?:need|want|care|make|change|order|look|check|know)\b|"
        r"\bno need\b|\b(?:ignore|skip)\b|\b(?:put|set)\b.{0,40}\baside\b|"
        r"\bnot\s+(?:working on|looking at|interested in)\b",
        latest,
        re.I,
    ):
        return False
    normalized = text_match_key(latest)
    names = [str(product.get("name") or ""), str(item.get("name") or item.get("product_name") or "")]
    phrases = [text_match_key(name) for name in names if text_match_key(name)]
    if any(phrase and f" {phrase} " in f" {normalized} " for phrase in phrases):
        return True
    terms = {
        term
        for phrase in phrases
        for term in phrase.split()
        if len(term) >= 4 and term not in IDENTIFIER_STOPWORDS
    }
    return bool(terms and any(re.search(rf"\b{re.escape(term)}s?\b", normalized) for term in terms))


def repair_item_mutation_variants(
    action: Action,
    messages: List[Dict[str, Any]],
) -> tuple[Action, dict[str, Any]]:
    if action.name not in ONE_SHOT_ITEM_MUTATIONS:
        return action, {}
    item_ids = action.kwargs.get("item_ids")
    if not isinstance(item_ids, list) or not item_ids:
        return action, {}
    order_id = extract_order_id(messages, action)
    order_items = order_items_by_item_id(messages, order_id)
    products = product_detail_payloads(messages)
    if not order_items or not products:
        return action, {}

    selected: list[str] = []
    replacements: dict[str, str] = {}
    for raw_item_id in item_ids:
        old_item_id = str(raw_item_id)
        old_item = order_items.get(old_item_id)
        if not old_item:
            return action, {}
        product_id = str(old_item.get("product_id") or "")
        product = products.get(product_id)
        if not product:
            return action, {}
        new_item_id = choose_variant_for_item(old_item, product, messages)
        if not new_item_id:
            return action, {}
        selected.append(new_item_id)
        replacements[old_item_id] = new_item_id

    current = action.kwargs.get("new_item_ids")
    if isinstance(current, list) and [str(item) for item in current] == selected:
        return action, {"resolved_new_item_ids": replacements}
    repaired = dict(action.kwargs)
    repaired["new_item_ids"] = selected
    return Action(name=action.name, kwargs=repaired), {"new_item_ids": replacements}


def complete_one_shot_item_batch(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> tuple[Action, dict[str, Any]] | None:
    if action.name not in ONE_SHOT_ITEM_MUTATIONS:
        return None
    item_ids = action.kwargs.get("item_ids")
    if not isinstance(item_ids, list):
        return None
    order_id = extract_order_id(messages, action)
    order = latest_order_payload(messages, order_id)
    if not order:
        return None
    if latest_user_mentions_different_primary_target(messages, action):
        return None
    requested_items = requested_order_items(messages, order)
    requested_item_ids = [str(item.get("item_id")) for item in requested_items if item.get("item_id")]
    if not requested_item_ids:
        return None
    current_item_ids = [str(item_id) for item_id in item_ids]
    if set(requested_item_ids).issubset(set(current_item_ids)) and set(requested_item_ids) != set(current_item_ids):
        completed_kwargs = dict(action.kwargs)
        completed_kwargs["item_ids"] = requested_item_ids
        new_item_ids = action.kwargs.get("new_item_ids")
        if isinstance(new_item_ids, list) and len(new_item_ids) == len(current_item_ids):
            replacements = {old_id: str(new_id) for old_id, new_id in zip(current_item_ids, new_item_ids)}
            completed_kwargs["new_item_ids"] = [
                replacements[item_id] for item_id in requested_item_ids if replacements.get(item_id)
            ]
        completed = Action(name=action.name, kwargs=completed_kwargs)
        completed, item_repair = repair_item_mutation_variants(completed, messages)
        return completed, {
            "used": True,
            "reason": "prune_unrequested_one_shot_item_batch",
            "proposed_action": action_to_dict(action),
            "repaired_action": action_to_dict(completed),
            "requested_item_ids": requested_item_ids,
            **({"item_variant_repair": item_repair} if item_repair else {}),
        }
    if len(requested_item_ids) <= 1:
        return None
    if set(requested_item_ids).issubset(set(current_item_ids)):
        return None

    products = product_detail_payloads(messages)
    if "get_product_details" in available_tool_names:
        for item in requested_items:
            item_id = str(item.get("item_id") or "")
            if item_id in current_item_ids:
                continue
            product_id = str(item.get("product_id") or "")
            if product_id and product_id not in products:
                return Action(name="get_product_details", kwargs={"product_id": product_id}), {
                    "used": True,
                    "reason": "lookup_product_before_one_shot_batch_completion",
                    "proposed_action": action_to_dict(action),
                    "missing_item_id": item_id,
                    "product_id": product_id,
                    "requested_item_ids": requested_item_ids,
                }

    completed_kwargs = dict(action.kwargs)
    completed_kwargs["item_ids"] = requested_item_ids
    if not completed_kwargs.get("payment_method_id"):
        payment_method_id = item_mutation_payment_method_id(messages, order)
        if payment_method_id:
            completed_kwargs["payment_method_id"] = payment_method_id
    completed = Action(name=action.name, kwargs=completed_kwargs)
    completed, item_repair = repair_item_mutation_variants(completed, messages)
    return completed, {
        "used": True,
        "reason": "complete_one_shot_item_batch",
        "proposed_action": action_to_dict(action),
        "repaired_action": action_to_dict(completed),
        "requested_item_ids": requested_item_ids,
        **({"item_variant_repair": item_repair} if item_repair else {}),
    }


def repair_single_list_item_mutation_scope(
    action: Action,
    messages: List[Dict[str, Any]],
) -> tuple[Action, dict[str, Any]]:
    if action.name not in SINGLE_LIST_ITEM_MUTATIONS:
        return action, {}
    item_ids = action.kwargs.get("item_ids")
    if not isinstance(item_ids, list):
        return action, {}
    order_id = extract_order_id(messages, action)
    order = latest_order_payload(messages, order_id)
    if not order:
        return action, {}
    request_text = item_request_text(messages, order)
    if re.search(r"\b(all|everything|entire order|all items|all of (?:the )?items)\b", request_text, re.I):
        return action, {}
    requested_items = requested_order_items(messages, order)
    requested_item_ids = [str(item.get("item_id")) for item in requested_items if item.get("item_id")]
    if not requested_item_ids:
        return action, {}
    current_item_ids = [str(item_id) for item_id in item_ids]
    if current_item_ids == requested_item_ids:
        return action, {}
    if set(current_item_ids) == set(requested_item_ids):
        return action, {}

    repaired_kwargs = dict(action.kwargs)
    repaired_kwargs["item_ids"] = requested_item_ids
    repaired = Action(name=action.name, kwargs=repaired_kwargs)
    return repaired, {
        "item_ids": {"from": current_item_ids, "to": requested_item_ids},
        "requested_item_ids": requested_item_ids,
    }


def propose_item_return_from_observations(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if "return_delivered_order_items" not in available_tool_names:
        return None
    text = user_conversation_text(messages).lower()
    if not mentions_action_intent(text, "return"):
        return None
    if mentions_action_intent(text, "exchange", "modify"):
        return None
    if latest_user_still_selecting_item_options(messages):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id}) if order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names else None
    if str(order.get("status") or "").lower() != "delivered":
        return None
    target_items = requested_order_items(messages, order)
    items = order.get("items") if isinstance(order.get("items"), list) else []
    if not target_items and re.search(r"\b(all|everything|entire order)\b", text):
        target_items = [item for item in items if isinstance(item, dict)]
    if not target_items:
        return None
    if (
        any(source == "gift_card" for _position, source in payment_source_mentions(latest_user_text(messages)))
        and not payment_method_for_source("gift_card", messages, order)
    ):
        user_id = extract_user_id(messages)
        if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
            return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id})
        return None
    payment_method_id = return_payment_method_id(messages, order)
    if not payment_method_id:
        return None
    return Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": str(order.get("order_id") or order_id),
            "item_ids": [str(item.get("item_id")) for item in target_items if item.get("item_id")],
            "payment_method_id": payment_method_id,
        },
    )


def product_issue_text_seen(messages: List[Dict[str, Any]]) -> bool:
    text = user_conversation_text(messages).lower()
    if re.search(
        r"\b("
        r"defective|broken|damaged|faulty|doesn'?t work|does not work|"
        r"won'?t work|will not work|can'?t use|cannot use|not connecting|won'?t connect|will not connect|connect properly|"
        r"chip(?:ped|ping)?|crack(?:ed|ing)?|defect(?:ive)?|pair(?:ing)?|sync(?:ing)?"
        r")\b",
        text,
    ):
        return True

    generic_issue = re.finditer(r"\b(trouble|issue|problem|problems)\b", text)
    financial_context = re.compile(
        r"\b(refund|payment|pay(?:ment)?|price|cost|charge|discount|coupon|match|cancel|"
        r"cancellation|cheaper|expensive|store|account)\b"
    )
    product_context = re.compile(
        r"\b(with|using|on|for)\s+(?:my|the|this|that|a|an)\s+[a-z0-9][a-z0-9 -]{1,60}"
    )
    for match in generic_issue:
        window = text[max(0, match.start() - 80) : min(len(text), match.end() + 100)]
        if financial_context.search(window):
            continue
        if product_context.search(window):
            return True
    return False


def propose_product_issue_return_from_observations(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if "return_delivered_order_items" not in available_tool_names:
        return None
    if not product_issue_text_seen(messages):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        user_id = extract_user_id(messages)
        if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
            return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id})
        next_order_id = next_unseen_user_order_id(messages)
        if next_order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
            return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": next_order_id})
        return None
    if str(order.get("status") or "").lower() != "delivered":
        next_order_id = next_unseen_user_order_id(messages)
        if next_order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
            return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": next_order_id})
        return None
    items = order.get("items") if isinstance(order.get("items"), list) else []
    target_items = requested_order_items(messages, order)
    if not target_items and len(items) == 1 and isinstance(items[0], dict):
        target_items = [items[0]]
    if not target_items:
        next_order_id = next_unseen_user_order_id(messages)
        if next_order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
            return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": next_order_id})
        return None
    payment_method_id = return_payment_method_id(messages, order)
    if not payment_method_id:
        return None
    return Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": str(order.get("order_id") or order_id),
            "item_ids": [str(item.get("item_id")) for item in target_items if item.get("item_id")],
            "payment_method_id": payment_method_id,
        },
    )


def pending_refund_or_cancel_text_seen(messages: List[Dict[str, Any]], text: str = "") -> bool:
    text = (text or user_conversation_text(messages)).lower()
    return bool(
        mentions_action_intent(text, "cancel")
        or re.search(
            r"\b(full refund|refund|cancel(?:lation)?|price match|match (?:the )?price|half the price|"
            r"cheaper store|cheaper elsewhere|buy (?:it )?from another|purchase from another|too expensive)\b",
            text,
        )
    )


def propose_pending_refund_cancel_from_observations(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> Action | None:
    if "cancel_pending_order" not in available_tool_names:
        return None
    if not pending_refund_or_cancel_text_seen(messages):
        return None
    if product_issue_text_seen(messages):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order or not str(order.get("status") or "").lower().startswith("pending"):
        return None
    scoped_text = item_request_text(messages, order).lower() or latest_user_text(messages).lower()
    if not pending_refund_or_cancel_text_seen(messages, scoped_text):
        return None
    if (
        re.search(r"\brefund\b", scoped_text)
        and not mentions_action_intent(scoped_text, "cancel")
        and mentions_action_intent(scoped_text, "modify", "update", "exchange")
    ):
        return None
    kwargs: dict[str, Any] = {"order_id": str(order.get("order_id") or order_id)}
    reason = infer_reason_from_text(messages, enum_values(tool_schemas, "cancel_pending_order", "reason"))
    if reason:
        kwargs["reason"] = reason
    return Action(name="cancel_pending_order", kwargs=kwargs)


def propose_item_exchange_from_observations(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    text = user_conversation_text(messages).lower()
    latest = latest_user_text(messages).lower()
    if not mentions_action_intent(text, "exchange", "modify"):
        return None
    if latest and mentions_action_intent(latest, "return") and not mentions_action_intent(latest, "exchange"):
        return None
    order_id = current_order_id(messages)
    order = latest_order_payload(messages, order_id)
    if not order:
        return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id}) if order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names else None
    status = str(order.get("status") or "").lower()
    if status == "delivered":
        mutation_name = "exchange_delivered_order_items"
    elif status.startswith("pending"):
        mutation_name = "modify_pending_order_items"
    else:
        return None
    if mutation_name not in available_tool_names:
        return None
    target_items = requested_order_items(messages, order)
    if not target_items:
        if not order_ids_in_text(user_conversation_text(messages)):
            next_order_id = next_unseen_user_order_id(messages)
            if next_order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
                return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": next_order_id})
        return None
    products = product_detail_payloads(messages)
    if "get_product_details" in available_tool_names:
        for item in target_items:
            product_id = str(item.get("product_id") or "")
            if product_id and product_id not in products:
                return Action(name="get_product_details", kwargs={"product_id": product_id})
    payment_method_id = item_mutation_payment_method_id(messages, order)
    if not payment_method_id:
        return None
    candidate = Action(
        name=mutation_name,
        kwargs={
            "order_id": str(order.get("order_id") or order_id),
            "item_ids": [str(item.get("item_id")) for item in target_items if item.get("item_id")],
            "payment_method_id": payment_method_id,
        },
    )
    repaired, item_repair = repair_item_mutation_variants(candidate, messages)
    if item_repair:
        return repaired
    return candidate


def redirect_return_action_to_exchange_intent(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if action.name != "return_delivered_order_items":
        return None
    text = user_conversation_text(messages).lower()
    if not mentions_action_intent(text, "exchange", "modify"):
        return None
    if not ({"exchange_delivered_order_items", "modify_pending_order_items"} & available_tool_names):
        return None
    return propose_item_exchange_from_observations(messages, available_tool_names)


def propose_airline_cancellation_fallback_from_observations(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> Action | None:
    if action.name != "transfer_to_human_agents" or "cancel_reservation" not in available_tool_names:
        return None
    text = all_conversation_text(messages).lower()
    summary = str(action.kwargs.get("summary") or "").lower()
    if not mentions_action_intent(user_conversation_text(messages), "modify", "update"):
        return None
    if not re.search(r"\b(flight|reservation|trip|booking)\b", text):
        return None
    if not re.search(r"\b(?:cannot|can't|not possible|unavailable|unable)\b.{0,80}\b(?:modif|change|update|switch)", summary):
        return None
    for reservation in reversed(reservation_detail_payloads(messages)):
        reservation_id = str(reservation.get("reservation_id") or "")
        if not reservation_id:
            continue
        if str(reservation.get("cabin") or "").lower() == "basic_economy":
            return Action(name="cancel_reservation", kwargs={"reservation_id": reservation_id})
    return None


def propose_action_from_transcript(
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None,
) -> Action | None:
    latest = latest_user_text(messages).lower()
    if not latest:
        return None
    conversation = user_conversation_text(messages).lower()
    order_id = current_order_id(messages)
    reservation_id = extract_reservation_id(messages)
    user_id = extract_user_id(messages)

    candidates: list[Action] = []
    if not tool_success_seen(messages, RETAIL_AUTH_TOOLS):
        for email in extract_emails(messages):
            if (
                "find_user_id_by_email" in available_tool_names
                and not action_was_called(messages, "find_user_id_by_email", "email", email)
            ):
                candidates.append(Action(name="find_user_id_by_email", kwargs={"email": email}))
                break
        name_zip = extract_name_zip(messages)
        if (
            name_zip
            and "find_user_id_by_name_zip" in available_tool_names
            and not action_kwargs_was_called(messages, "find_user_id_by_name_zip", name_zip)
        ):
            candidates.append(Action(name="find_user_id_by_name_zip", kwargs=name_zip))
    if (
        not order_id
        and user_id
        and USER_DETAIL_TOOL in available_tool_names
        and not user_details_seen(messages, user_id)
        and (
            re.search(r"\b(order|orders|recent|purchase)\b", latest)
            or mentions_action_intent(latest, "exchange", "return", "cancel", "modify", "update")
            or (
                retail_account_task_text_seen(messages)
                and (
                    re.search(r"\b(order|orders|recent|purchase)\b", conversation)
                    or mentions_action_intent(conversation, "exchange", "return", "cancel", "modify", "update")
                )
            )
        )
    ):
        candidates.append(Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id}))
    latest_unseen_order = latest_unseen_order_id(messages)
    if latest_unseen_order and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
        candidates.append(Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": latest_unseen_order}))
    if mentions_action_intent(latest, "cancel") and "cancel_pending_order" in available_tool_names and order_id:
        kwargs: dict[str, Any] = {"order_id": order_id}
        reason = infer_reason_from_text(messages, enum_values(tool_schemas, "cancel_pending_order", "reason"))
        if reason:
            kwargs["reason"] = reason
        candidates.append(Action(name="cancel_pending_order", kwargs=kwargs))
    if mentions_action_intent(latest, "cancel") and "cancel_reservation" in available_tool_names and reservation_id:
        candidates.append(Action(name="cancel_reservation", kwargs={"reservation_id": reservation_id}))
    if order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names and re.search(r"\b(order|status|details?|where|track)\b", latest):
        candidates.append(Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id}))
    if (
        reservation_id
        and RESERVATION_DETAIL_TOOL in available_tool_names
        and re.search(r"\b(reservation|booking|flight|status|details?)\b", latest)
    ):
        candidates.append(Action(name=RESERVATION_DETAIL_TOOL, kwargs={"reservation_id": reservation_id}))

    for candidate in candidates:
        repaired, _ = repair_action_arguments(candidate, messages, tool_schemas)
        key, value = primary_action_key(repaired)
        if key and action_was_called(messages, repaired.name, key, value):
            continue
        if repaired.name == RETAIL_ORDER_DETAIL_TOOL and key == "order_id" and order_details_seen(messages, str(value)):
            continue
        if not key and action_was_called(messages, repaired.name):
            continue
        return repaired
    issue_return = propose_product_issue_return_from_observations(messages, available_tool_names)
    if issue_return is not None:
        return issue_return
    pending_refund_cancel = propose_pending_refund_cancel_from_observations(
        messages,
        available_tool_names,
        tool_schemas,
    )
    if pending_refund_cancel is not None:
        return pending_refund_cancel
    item_return = propose_item_return_from_observations(messages, available_tool_names)
    if item_return is not None:
        return item_return
    item_exchange = propose_item_exchange_from_observations(messages, available_tool_names)
    if item_exchange is not None:
        return item_exchange
    return None


def retail_auth_guard(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> tuple[Action, dict[str, Any]] | None:
    if not (RETAIL_AUTH_TOOLS & available_tool_names):
        return None
    if action.name in RETAIL_AUTH_TOOLS or tool_success_seen(messages, RETAIL_AUTH_TOOLS):
        return None
    for email in extract_emails(messages):
        if (
            "find_user_id_by_email" in available_tool_names
            and not action_was_called(messages, "find_user_id_by_email", "email", email)
        ):
            return Action(name="find_user_id_by_email", kwargs={"email": email}), {
                "used": True,
                "reason": "authenticate_before_action",
                "proposed_action": action_to_dict(action),
            }
    name_zip = extract_name_zip(messages)
    if (
        name_zip
        and "find_user_id_by_name_zip" in available_tool_names
        and not action_kwargs_was_called(messages, "find_user_id_by_name_zip", name_zip)
    ):
        return Action(name="find_user_id_by_name_zip", kwargs=name_zip), {
            "used": True,
            "reason": "authenticate_before_action",
            "proposed_action": action_to_dict(action),
        }
    return fallback_response(
        "Before I can help with account or order information, please provide another email address, or your first name, last name, and zip code."
    ), {
        "used": True,
        "reason": "request_authentication_before_action",
        "proposed_action": action_to_dict(action),
    }


def retry_after_failed_auth_action(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
) -> tuple[Action, dict[str, Any]] | None:
    if action.name not in RETAIL_AUTH_TOOLS:
        return None
    if not action_kwargs_error_was_seen(messages, action.name, action.kwargs):
        return None
    for email in extract_emails(messages):
        candidate = {"email": email}
        if (
            "find_user_id_by_email" in available_tool_names
            and not action_kwargs_was_called(messages, "find_user_id_by_email", candidate)
        ):
            return Action(name="find_user_id_by_email", kwargs=candidate), {
                "used": True,
                "reason": "retry_next_auth_email_after_lookup_failure",
                "proposed_action": action_to_dict(action),
            }
    name_zip = extract_name_zip(messages)
    if (
        name_zip
        and "find_user_id_by_name_zip" in available_tool_names
        and not action_kwargs_was_called(messages, "find_user_id_by_name_zip", name_zip)
    ):
        return Action(name="find_user_id_by_name_zip", kwargs=name_zip), {
            "used": True,
            "reason": "retry_name_zip_after_auth_lookup_failure",
            "proposed_action": action_to_dict(action),
        }
    return fallback_response(
        "I could not find an account with the information provided. Please provide another email address, or your first name, last name, and ZIP code so I can locate your account."
    ), {
        "used": True,
        "reason": "request_new_auth_after_lookup_failures",
        "proposed_action": action_to_dict(action),
    }


def tau_policy_guard_action(
    action: Action,
    messages: List[Dict[str, Any]],
    available_tool_names: set[str],
    tool_schemas: Dict[str, Dict[str, Any]] | None = None,
) -> tuple[Action, dict[str, Any]]:
    if action.name == RESPOND_ACTION_NAME:
        if latest_user_reconfirms_completed_mutation(messages):
            return fallback_response(completed_mutation_response(messages, already=True)), {
                "used": True,
                "reason": "respond_after_prior_successful_mutation",
                "original_action": action_to_dict(action),
            }
        completed_option_followup = completed_item_mutation_option_followup_response(messages)
        if completed_option_followup is not None:
            return completed_option_followup, {
                "used": True,
                "reason": "avoid_option_exploration_after_completed_item_mutation",
                "original_action": action_to_dict(action),
            }
        auth_request = retail_auth_request_action(messages, available_tool_names)
        if auth_request is not None:
            return auth_request, {
                "used": True,
                "reason": "request_retail_authentication_before_response",
                "original_action": action_to_dict(action),
            }
        if not refund_payment_method_question_seen(messages) and latest_deferred_mutating_action(messages) is None:
            pending_refund_cancel = propose_pending_refund_cancel_from_observations(
                messages,
                available_tool_names,
                tool_schemas,
            )
            if pending_refund_cancel is not None:
                guarded, guard = tau_policy_guard_action(
                    pending_refund_cancel,
                    messages,
                    available_tool_names,
                    tool_schemas,
                )
                original_reason = str(guard.get("reason") or "execute_pending_refund_cancel")
                if guarded.name != action.name or guarded.kwargs != action.kwargs:
                    return guarded, {
                        "used": True,
                        "reason": f"pending_refund_cancel_{original_reason}",
                        "proposed_action": action_to_dict(pending_refund_cancel),
                        "original_action": action_to_dict(action),
                    }
        refund_payment = refund_payment_method_action(messages, available_tool_names)
        if refund_payment is not None:
            if refund_payment.name == RESPOND_ACTION_NAME:
                return refund_payment, {
                    "used": True,
                    "reason": "refund_payment_answer_refund_payment_methods",
                    "proposed_action": action_to_dict(refund_payment),
                    "original_action": action_to_dict(action),
                }
            guarded, guard = tau_policy_guard_action(refund_payment, messages, available_tool_names, tool_schemas)
            original_reason = str(guard.get("reason") or "answer_refund_payment_methods")
            if guarded.name != action.name or guarded.kwargs != action.kwargs:
                return guarded, {
                    "used": True,
                    "reason": f"refund_payment_{original_reason}",
                    "proposed_action": action_to_dict(refund_payment),
                    "original_action": action_to_dict(action),
                }
        issue_return = propose_product_issue_return_from_observations(messages, available_tool_names)
        if issue_return is not None:
            guarded, guard = tau_policy_guard_action(issue_return, messages, available_tool_names, tool_schemas)
            original_reason = str(guard.get("reason") or "execute_product_issue_return")
            if guarded.name != action.name or guarded.kwargs != action.kwargs:
                return guarded, {
                    "used": True,
                    "reason": f"product_issue_{original_reason}",
                    "proposed_action": action_to_dict(issue_return),
                    "original_action": action_to_dict(action),
                }
        deferred = latest_deferred_mutating_action(messages)
        if deferred is not None and latest_user_requests_deferred_mutation_execution(messages):
            guarded, guard = tau_policy_guard_action(deferred, messages, available_tool_names, tool_schemas)
            original_reason = str(guard.get("reason") or "execute_deferred_mutation")
            if guarded.name != action.name or guarded.kwargs != action.kwargs:
                return guarded, {
                    "used": True,
                    "reason": f"resume_deferred_{original_reason}",
                    "proposed_action": action_to_dict(deferred),
                    "original_action": action_to_dict(action),
                }
        option_lookup = catalog_option_lookup_action(messages, available_tool_names)
        if option_lookup is not None:
            return option_lookup, {
                "used": True,
                "reason": "lookup_product_before_item_option_answer",
                "original_action": action_to_dict(action),
            }
        price_response = catalog_price_response(messages)
        if price_response is not None:
            return price_response, {
                "used": True,
                "reason": "answer_item_price_question",
                "original_action": action_to_dict(action),
            }
        option_response = catalog_option_response(messages)
        if option_response is not None:
            return option_response, {
                "used": True,
                "reason": "answer_item_option_question",
                "original_action": action_to_dict(action),
            }
        deferred = latest_deferred_mutating_action(messages)
        if deferred is not None:
            guarded, guard = tau_policy_guard_action(deferred, messages, available_tool_names, tool_schemas)
            original_reason = str(guard.get("reason") or "execute_deferred_mutation")
            if guarded.name != action.name or guarded.kwargs != action.kwargs:
                return guarded, {
                    "used": True,
                    "reason": f"resume_deferred_{original_reason}",
                    "proposed_action": action_to_dict(deferred),
                    "original_action": action_to_dict(action),
                }
        proposed = propose_action_from_transcript(messages, available_tool_names, tool_schemas)
        if proposed is not None:
            guarded, guard = tau_policy_guard_action(proposed, messages, available_tool_names, tool_schemas)
            original_reason = str(guard.get("reason") or "execute_proposed_action")
            if guarded.name != action.name or guarded.kwargs != action.kwargs:
                return guarded, {
                    "used": True,
                    "reason": f"transcript_proposal_{original_reason}",
                    "proposed_action": action_to_dict(proposed),
                    "original_action": action_to_dict(action),
                }
        return action, {"used": False}

    action, repair = repair_action_arguments(action, messages, tool_schemas)
    action, reason_repair = repair_reason_argument_from_transcript(action, messages, tool_schemas)
    if reason_repair:
        repair = {**repair, "semantic": reason_repair}

    auth_retry = retry_after_failed_auth_action(action, messages, available_tool_names)
    if auth_retry is not None:
        guarded, info = auth_retry
        if repair:
            info["argument_repair"] = repair
        return guarded, info

    auth_guard = retail_auth_guard(action, messages, available_tool_names)
    if auth_guard is not None:
        guarded, info = auth_guard
        if repair:
            info["argument_repair"] = repair
        return guarded, info

    if action.name == "transfer_to_human_agents":
        issue_return = propose_product_issue_return_from_observations(messages, available_tool_names)
        if issue_return is not None:
            guarded, guard = tau_policy_guard_action(issue_return, messages, available_tool_names, tool_schemas)
            if issue_return.name == USER_DETAIL_TOOL:
                reason = "lookup_user_for_product_issue"
            elif issue_return.name == RETAIL_ORDER_DETAIL_TOOL:
                reason = "lookup_order_for_product_issue"
            else:
                reason = guard.get("reason") or "product_issue_return"
            return guarded, {
                "used": True,
                "reason": f"avoid_transfer_{reason}",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
            }
        user_id = extract_user_id(messages, action)
        if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
            return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id}), {
                "used": True,
                "reason": "avoid_transfer_lookup_user_for_product_issue",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
            }
        known_orders = user_order_ids(messages)
        if product_issue_text_seen(messages) and len(known_orders) == 1 and RETAIL_ORDER_DETAIL_TOOL in available_tool_names:
            order_id_for_issue = known_orders[0]
            if not order_details_seen(messages, order_id_for_issue):
                return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id_for_issue}), {
                    "used": True,
                    "reason": "avoid_transfer_lookup_order_for_product_issue",
                    "proposed_action": action_to_dict(action),
                    **({"argument_repair": repair} if repair else {}),
                }
        airline_cancel = propose_airline_cancellation_fallback_from_observations(
            action,
            messages,
            available_tool_names,
        )
        if airline_cancel is not None:
            guarded, guard = tau_policy_guard_action(airline_cancel, messages, available_tool_names, tool_schemas)
            return guarded, {
                "used": True,
                "reason": f"avoid_transfer_airline_cancellation_fallback_{guard.get('reason') or 'execute_cancel_fallback'}",
                "proposed_action": action_to_dict(action),
                "replacement_action": action_to_dict(airline_cancel),
                **({"argument_repair": repair} if repair else {}),
            }
        proposed = propose_action_from_transcript(messages, available_tool_names, tool_schemas)
        if proposed is not None:
            guarded, guard = tau_policy_guard_action(proposed, messages, available_tool_names, tool_schemas)
            return guarded, {
                "used": True,
                "reason": f"avoid_transfer_transcript_proposal_{guard.get('reason') or 'execute_proposed_action'}",
                "proposed_action": action_to_dict(action),
                "replacement_action": action_to_dict(proposed),
                **({"argument_repair": repair} if repair else {}),
            }

    order_id_repair: dict[str, Any] = {}
    if action.kwargs.get("order_id"):
        action, order_id_repair = repair_order_id_from_user_orders(action, messages)

    if mutating_action(action.name) and mutating_action_success_was_seen(messages, action):
        return fallback_response(completed_mutation_response(messages, already=True)), {
            "used": True,
            "reason": "avoid_duplicate_mutation_after_success",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    if action.name == RETAIL_ORDER_DETAIL_TOOL:
        order_id = extract_order_id(messages, action)
        if order_id and order_details_seen(messages, order_id):
            item_return = propose_item_return_from_observations(messages, available_tool_names)
            if item_return is not None:
                guarded, guard = tau_policy_guard_action(item_return, messages, available_tool_names, tool_schemas)
                return guarded, {
                    "used": True,
                    "reason": f"avoid_duplicate_order_lookup_{guard.get('reason') or 'advance_to_return_action'}",
                    "proposed_action": action_to_dict(action),
                    **({"argument_repair": repair} if repair else {}),
                    **({"order_id_repair": order_id_repair} if order_id_repair else {}),
                }
            item_exchange = propose_item_exchange_from_observations(messages, available_tool_names)
            if item_exchange is not None:
                guarded, guard = tau_policy_guard_action(item_exchange, messages, available_tool_names, tool_schemas)
                return guarded, {
                    "used": True,
                    "reason": f"avoid_duplicate_order_lookup_{guard.get('reason') or 'advance_to_item_action'}",
                    "proposed_action": action_to_dict(action),
                    **({"argument_repair": repair} if repair else {}),
                    **({"order_id_repair": order_id_repair} if order_id_repair else {}),
                }
            return fallback_response("I already have the order details. What would you like me to do next?"), {
                "used": True,
                "reason": "avoid_duplicate_order_lookup",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
                **({"order_id_repair": order_id_repair} if order_id_repair else {}),
            }
        if action_error_was_seen(messages, RETAIL_ORDER_DETAIL_TOOL, "order_id", order_id):
            repaired_action, retry_repair = repair_order_id_from_user_orders(action, messages)
            if retry_repair:
                return repaired_action, {
                    "used": True,
                    "reason": "repair_failed_order_id_from_user_orders",
                    "proposed_action": action_to_dict(action),
                    "order_id_repair": retry_repair,
                    **({"argument_repair": repair} if repair else {}),
                }
            user_id = extract_user_id(messages, action)
            if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
                return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id}), {
                    "used": True,
                    "reason": "lookup_user_orders_after_failed_order_lookup",
                    "proposed_action": action_to_dict(action),
                    **({"argument_repair": repair} if repair else {}),
                }
            return fallback_response("I could not find that order. Please confirm the full order id."), {
                "used": True,
                "reason": "request_order_id_after_failed_order_lookup",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
            }

    if not mutating_action(action.name):
        missing = missing_required_arguments(action, tool_schemas)
        if missing:
            return fallback_response(missing_argument_response(action, missing, tool_schemas)), {
                "used": True,
                "reason": "request_missing_required_arguments",
                "missing_arguments": missing,
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
                **({"order_id_repair": order_id_repair} if order_id_repair else {}),
            }
        if order_id_repair:
            return action, {
                "used": True,
                "reason": "repair_order_id_from_user_orders",
                "proposed_action": action_to_dict(action),
                "order_id_repair": order_id_repair,
                **({"argument_repair": repair} if repair else {}),
            }
        if repair:
            return action, {
                "used": True,
                "reason": "execute_argument_repair",
                "proposed_action": action_to_dict(action),
                "argument_repair": repair,
            }
        return action, {"used": False}

    order_id = extract_order_id(messages, action)
    if order_id and RETAIL_ORDER_DETAIL_TOOL in available_tool_names and not order_details_seen(messages, order_id):
        return Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": order_id}), {
            "used": True,
            "reason": "lookup_order_before_mutation",
            "proposed_action": action_to_dict(action),
        }

    reservation_id = extract_reservation_id(messages, action)
    if (
        reservation_id
        and RESERVATION_DETAIL_TOOL in available_tool_names
        and not reservation_details_seen(messages, reservation_id)
    ):
        return Action(name=RESERVATION_DETAIL_TOOL, kwargs={"reservation_id": reservation_id}), {
            "used": True,
            "reason": "lookup_reservation_before_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
        }

    user_id = extract_user_id(messages, action)
    if user_id and USER_DETAIL_TOOL in available_tool_names and not user_details_seen(messages, user_id):
        return Action(name=USER_DETAIL_TOOL, kwargs={"user_id": user_id}), {
            "used": True,
            "reason": "lookup_user_before_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
        }

    refund_method_question = return_refund_method_question(action, messages)
    if refund_method_question is not None:
        refund_method_response, refund_method_info = refund_method_question
        return refund_method_response, {
            "used": True,
            "reason": "request_refund_payment_method_before_return_mutation",
            "proposed_action": action_to_dict(action),
            **refund_method_info,
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    payment_update = pending_payment_update_before_item_mutation(action, messages, available_tool_names)
    if payment_update is not None:
        return payment_update, {
            "used": True,
            "reason": "update_payment_before_pending_item_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    payment_question = pending_item_mutation_payment_method_question(action, messages)
    if payment_question is not None:
        return payment_question, {
            "used": True,
            "reason": "request_payment_method_before_pending_item_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    exchange_redirect = redirect_return_action_to_exchange_intent(action, messages, available_tool_names)
    if exchange_redirect is not None:
        guarded, guard = tau_policy_guard_action(exchange_redirect, messages, available_tool_names, tool_schemas)
        return guarded, {
            "used": True,
            "reason": f"redirect_return_to_exchange_intent_{guard.get('reason') or 'execute_exchange_intent'}",
            "proposed_action": action_to_dict(action),
            "exchange_action": action_to_dict(exchange_redirect),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    item_batch_repair: dict[str, Any] = {}
    if action.name in ONE_SHOT_ITEM_MUTATIONS:
        batch_completion = complete_one_shot_item_batch(action, messages, available_tool_names)
        if batch_completion is not None:
            completed_action, batch_info = batch_completion
            if completed_action.name == "get_product_details":
                return completed_action, {
                    **batch_info,
                    **({"argument_repair": repair} if repair else {}),
                    **({"order_id_repair": order_id_repair} if order_id_repair else {}),
                }
            action = completed_action
            item_batch_repair = batch_info

    item_payment_repair: dict[str, Any] = {}
    if action.name in ONE_SHOT_ITEM_MUTATIONS or action.name in SINGLE_LIST_ITEM_MUTATIONS:
        action, item_payment_repair = repair_item_mutation_payment_method(action, messages)

    item_scope_repair: dict[str, Any] = {}
    if action.name in SINGLE_LIST_ITEM_MUTATIONS:
        action, item_scope_repair = repair_single_list_item_mutation_scope(action, messages)

    if action.name in ONE_SHOT_ITEM_MUTATIONS and "get_product_details" in available_tool_names:
        missing_product_id = missing_product_detail_for_item_mutation(action, messages)
        if missing_product_id:
            return Action(name="get_product_details", kwargs={"product_id": missing_product_id}), {
                "used": True,
                "reason": "lookup_product_before_item_mutation",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
                **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
                **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
            }

    original_item_action = action
    item_variant_repair: dict[str, Any] = {}
    if action.name in ONE_SHOT_ITEM_MUTATIONS:
        action, item_variant_repair = repair_item_mutation_variants(action, messages)

    if action.name == "cancel_reservation" and not airline_cancellation_reason_seen(messages):
        return fallback_response(
            "Please provide the reason for cancelling the reservation: change of plan, airline cancelled flight, or another reason."
        ), {
            "used": True,
            "reason": "request_airline_cancellation_reason",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
        }

    missing = missing_required_arguments(action, tool_schemas)
    if missing:
        return fallback_response(missing_argument_response(action, missing, tool_schemas)), {
            "used": True,
            "reason": "request_missing_required_arguments",
            "missing_arguments": missing,
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
            **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
            **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
            **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
            **({"item_variant_repair": item_variant_repair} if item_variant_repair else {}),
        }

    if action.name in ONE_SHOT_ITEM_MUTATIONS:
        option_response = unresolved_item_option_response(action, messages)
        if option_response is not None:
            return option_response, {
                "used": True,
                "reason": "answer_unresolved_item_option_question",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
                **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
                **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
                **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
                **({"item_variant_repair": item_variant_repair} if item_variant_repair else {}),
            }

    if action.name in ONE_SHOT_ITEM_MUTATIONS and latest_user_still_selecting_item_options(messages):
        option_response = catalog_option_response(messages)
        if option_response is not None:
            return option_response, {
                "used": True,
                "reason": "answer_item_option_question",
                "proposed_action": action_to_dict(action),
                **({"argument_repair": repair} if repair else {}),
                **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
                **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
                **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
                **({"item_variant_repair": item_variant_repair} if item_variant_repair else {}),
            }
        return fallback_response(
            "Before I make this one-time item change, please confirm the exact final items you want to exchange or modify."
        ), {
            "used": True,
            "reason": "request_final_item_choices_before_one_shot_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
            **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
            **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
            **({"item_variant_repair": item_variant_repair} if item_variant_repair else {}),
        }

    if (
        not user_confirmed_after_confirmation_request(messages)
        and not latest_user_explicitly_confirms_action(messages, action)
        and not direct_mutation_request_is_grounded(messages, action, tool_schemas)
    ):
        detail = mutation_confirmation_detail(action, messages)
        return fallback_response(
            f"I can {detail}. Please confirm with yes if you want me to proceed."
        ), {
            "used": True,
            "reason": "confirm_before_mutation",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
            **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
            **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
            **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
            **({"item_variant_repair": item_variant_repair} if item_variant_repair else {}),
        }

    if item_batch_repair or item_payment_repair or item_scope_repair or item_variant_repair:
        return action, {
            "used": True,
            "reason": item_batch_repair.get("reason")
            or ("repair_item_mutation_payment_method" if item_payment_repair else "")
            or ("repair_single_list_item_mutation_scope" if item_scope_repair.get("item_ids") else "")
            or (
                "repair_item_variant_selection"
                if "new_item_ids" in item_variant_repair
                else "resolve_item_variant_selection"
            ),
            "proposed_action": action_to_dict(original_item_action),
            "repaired_action": action_to_dict(action),
            **({"item_batch_repair": item_batch_repair} if item_batch_repair else {}),
            **({"item_payment_repair": item_payment_repair} if item_payment_repair else {}),
            **({"item_scope_repair": item_scope_repair} if item_scope_repair else {}),
            "item_variant_repair": item_variant_repair,
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    if repair or order_id_repair:
        return action, {
            "used": True,
            "reason": "execute_argument_repair" if repair else "repair_order_id_from_user_orders",
            "proposed_action": action_to_dict(action),
            **({"argument_repair": repair} if repair else {}),
            **({"order_id_repair": order_id_repair} if order_id_repair else {}),
        }

    return action, {"used": False}


class GoalGraphAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.tools_info = tools_info
        self.goal_graph_tools = tau_tools_to_goal_graph_tools(tools_info)
        self._capability_registry = GoalGraphRuntime(self.goal_graph_tools).registry
        self.available_tool_names = {
            str(tau_tool_function(tool).get("name") or "")
            for tool in tools_info
            if isinstance(tool, dict)
        }
        self.available_tool_names.discard("")
        self.tool_schemas = {
            str(tau_tool_function(tool).get("name") or ""): tau_tool_function(tool).get("parameters")
            for tool in tools_info
            if isinstance(tool, dict) and isinstance(tau_tool_function(tool).get("parameters"), dict)
        }
        self.tool_schemas = {name: schema for name, schema in self.tool_schemas.items() if name}
        self.wiki = wiki
        self.model_name = os.environ.get("TAU_GOAL_GRAPH_MODEL", model if model != "local-goal-graph" else DEFAULT_MODEL)
        self.max_new_tokens = env_int("TAU_GOAL_GRAPH_MAX_NEW_TOKENS", 900)
        self.repair_attempts = env_int("TAU_GOAL_GRAPH_REPAIR_ATTEMPTS", 1)
        self.max_steps = env_int("TAU_GOAL_GRAPH_MAX_STEPS", 30)
        self.transcript_context_messages = env_int("TAU_GOAL_GRAPH_TRANSCRIPT_CONTEXT_MESSAGES", 12)
        self.planning_context_chars = env_int("TAU_GOAL_GRAPH_PLANNING_CONTEXT_CHARS", 9000)
        self.binding_context_chars = env_int("TAU_GOAL_GRAPH_BINDING_CONTEXT_CHARS", 8000)
        self.policy_context_chars = env_int("TAU_GOAL_GRAPH_POLICY_CONTEXT_CHARS", 3000)
        self.planner_mode = os.environ.get("TAU_GOAL_GRAPH_PLANNER_MODE", "stepwise")
        self.allow_side_effects = env_bool("TAU_GOAL_GRAPH_ALLOW_SIDE_EFFECTS", True)
        self.preflight_enabled = env_bool("TAU_GOAL_GRAPH_PREFLIGHT", True)
        # Match the BFCL/API-Bank runtime: GPT-OSS produces a semantic frame,
        # then the shared deterministic binder and goal-graph verifier compile
        # one call.  Extra policy-repair generations are an optional ablation,
        # not part of the default cross-benchmark pipeline.
        self.policy_repair_enabled = env_bool("TAU_GOAL_GRAPH_POLICY_REPAIR", False)
        self.policy_repair_max_new_tokens = env_int("TAU_GOAL_GRAPH_POLICY_REPAIR_MAX_NEW_TOKENS", 900)
        # The old adapter remains useful for regression comparisons, but the
        # production path is the schema/evidence verifier below.  Keeping the
        # switch explicit prevents benchmark-domain rules from silently
        # becoming part of the general goal-graph runtime.
        self.legacy_tau_heuristics = env_bool("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", False)
        self._episode_state = EpisodeState()
        self._pending_mutation_packets: dict[tuple[str, str], Any] = {}
        self.debug_dir = Path(os.environ.get("TAU_GOAL_GRAPH_DEBUG_DIR", "")) if os.environ.get("TAU_GOAL_GRAPH_DEBUG_DIR") else None
        self.replay_trace_enabled = env_bool("TAU_GOAL_GRAPH_REPLAY_TRACE", True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.write_replay_manifest()
        print(f"[tau-goal-graph] loading model={self.model_name}", flush=True)
        self.model_obj, self.tokenizer = load_model(self.model_name)
        print("[tau-goal-graph] model ready", flush=True)

    def build_request(self, messages: List[Dict[str, Any]], previous_source: str) -> str:
        # The shared binder expects natural-language task context.  A JSON
        # envelope makes schema keys look like user-requested entities.
        state_context = (
            "\n\nRuntime-owned episode state (current typed facts and obligations, not raw evidence):\n"
            + short_json(self._episode_state.planner_projection(max_facts=36), 4200)
        )
        prefix = (
            "Current task: choose exactly one next action in the tool environment. "
            f"Use {RESPOND_ACTION_NAME} only for missing information or a final answer.\n\n"
            f"Policy:\n{self.wiki[:self.policy_context_chars]}\n\n"
            f"Previous observation source: {previous_source}\n\n"
            "Transcript:\n"
        )
        transcript_budget = max(1000, self.planning_context_chars - len(prefix) - len(state_context))
        transcript = bounded_stateful_transcript(
            messages,
            transcript_for_goal_graph,
            max_messages=self.transcript_context_messages,
            max_chars=transcript_budget,
        )
        return prefix + transcript + state_context

    def build_binding_request(self, messages: List[Dict[str, Any]], previous_source: str) -> str:
        # Keep deterministic slot binding grounded in the live state only.
        # Policy text is useful to the semantic model, but it is not evidence
        # that concrete IDs or argument values are present in the environment.
        state_context = short_json(self._episode_state.planner_projection(max_facts=32), 3600)
        prefix = "Binding evidence from user text and environment values:\n"
        prefix += "Runtime-owned active state with source IDs:\n" + state_context + "\n\nTranscript evidence:\n"
        transcript = bounded_stateful_transcript(
            messages,
            transcript_for_binding,
            max_messages=self.transcript_context_messages,
            max_chars=max(1000, self.binding_context_chars - len(prefix)),
        )
        return prefix + transcript

    def build_policy_repair_request(
        self,
        messages: List[Dict[str, Any]],
        previous_source: str,
        rejected_action: Action,
    ) -> str:
        tools = []
        for raw in self.tools_info:
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "description": str(fn.get("description") or ""),
                    "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {},
                }
            )
        tools.append(
            {
                "name": RESPOND_ACTION_NAME,
                "description": "Send one customer-facing message.",
                "parameters": {
                    "type": "object",
                    "properties": {RESPOND_ACTION_FIELD_NAME: {"type": "string"}},
                    "required": [RESPOND_ACTION_FIELD_NAME],
                },
            }
        )
        payload = {
            "task": (
                "The previous next-step proposal was non-actionable, an escalation, or appears to have "
                "missed a required post-confirmation step. Infer exactly one next policy-valid action from "
                "the policy, transcript, available tools, and observed tool outputs. Use a lookup tool when "
                "more state is needed. Use a state-changing tool only when its arguments are grounded in the "
                "transcript or tool outputs. Use respond only for missing information, confirmation, or a "
                "final answer. Do not escalate or hand off unless the visible policy explicitly requires it "
                "and no available tool can make further progress."
            ),
            "policy": self.wiki,
            "previous_observation_source": previous_source,
            "rejected_action": action_to_dict(rejected_action),
            "transcript": transcript_for_goal_graph(messages, max_messages=30),
            "available_tools": tools,
            "output_schema": {
                "action": {
                    "name": "available tool name or respond",
                    "kwargs": "object using the selected tool schema",
                },
                "reason": "brief policy/status reasoning",
                "evidence": ["one or two exact substrings, at most 120 characters each, supporting the action and arguments"],
            },
            "output_rule": "Return only one compact JSON object. Do not include analysis, markdown, or prose before JSON.",
        }
        return short_json(payload, 36000)

    def policy_repair_action(
        self,
        messages: List[Dict[str, Any]],
        previous_source: str,
        rejected_action: Action,
    ) -> tuple[Action | None, dict[str, Any]]:
        request = self.build_policy_repair_request(messages, previous_source, rejected_action)
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You repair invalid one-step tool decisions. Choose exactly one executable next action. "
                    "Every evidence entry must be copied exactly from the supplied transcript and be at most 120 characters. "
                    "Return strict compact JSON only, beginning with { and ending with }."
                ),
            },
            {"role": "user", "content": request},
        ]
        raw_text = generate_text(self.model_obj, self.tokenizer, repair_messages, self.policy_repair_max_new_tokens)
        action, info = policy_repair_action_from_raw(
            raw_text,
            messages,
            self.available_tool_names,
            self.tool_schemas,
        )
        if action is None and info.get("reason") in {
            "parse_error",
            "invalid_evidence",
            "ungrounded_evidence",
            "missing_evidence",
            "arguments_not_supported_by_evidence",
        }:
            format_messages = [
                {
                    "role": "system",
                    "content": (
                        "Convert the prior decision into one compact JSON object only. Do not include analysis. "
                        "Use the requested action schema. Cite one or two exact evidence substrings of at most 120 characters."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original decision request:\n{request}\n\n"
                        f"Previous invalid output:\n{raw_text}\n\n"
                        "Return only the corrected JSON object."
                    ),
                },
            ]
            retry_raw_text = generate_text(
                self.model_obj,
                self.tokenizer,
                format_messages,
                min(self.policy_repair_max_new_tokens, 500),
            )
            retry_action, retry_info = policy_repair_action_from_raw(
                retry_raw_text,
                messages,
                self.available_tool_names,
                self.tool_schemas,
            )
            info["format_retry"] = retry_info
            if retry_action is not None:
                action = retry_action
                info = retry_info
                info["format_retry_used"] = True
        info["request"] = request
        return action, info

    def guard_action(
        self,
        action: Action,
        messages: List[Dict[str, Any]],
        goal_graph_result: Dict[str, Any] | None = None,
    ) -> tuple[Action, dict[str, Any]]:
        if self.legacy_tau_heuristics:
            return tau_policy_guard_action(
                action,
                messages,
                self.available_tool_names,
                self.tool_schemas,
            )
        return generic_action_guard(
            action,
            messages,
            self.available_tool_names,
            self.tool_schemas,
            goal_graph_result,
        )

    def preflight_action(self, messages: List[Dict[str, Any]]) -> tuple[Action | None, dict[str, Any]]:
        if not self.preflight_enabled or not self.legacy_tau_heuristics:
            return None, {}
        if last_message_is_successful_mutating_tool(messages) and latest_deferred_mutating_action(messages) is None:
            proposed = propose_action_from_transcript(messages, self.available_tool_names, self.tool_schemas)
            if proposed is not None and (
                not mutating_action(proposed.name) or not mutating_action_success_was_seen(messages, proposed)
            ):
                guarded_action, guard = self.guard_action(proposed, messages)
                if not (
                    guarded_action.name == RESPOND_ACTION_NAME
                    and re.search(r"\b(completed|already)\b", str(guarded_action.kwargs.get(RESPOND_ACTION_FIELD_NAME) or ""), re.I)
                ):
                    return guarded_action, {
                        "verification_ok": True,
                        "calls": [{"tool_name": guarded_action.name, "arguments": guarded_action.kwargs}],
                        "planner_skipped": True,
                        "tau_policy_guard": {
                            "used": True,
                            "reason": f"continue_after_success_{guard.get('reason') or 'transcript_proposal'}",
                            "proposed_action": action_to_dict(proposed),
                            **({"nested_guard": guard} if guard else {}),
                        },
                    }
            action = fallback_response(completed_mutation_response(messages))
            next_order_id = next_unseen_requested_order_id(messages)
            if next_order_id and RETAIL_ORDER_DETAIL_TOOL in self.available_tool_names:
                action = Action(name=RETAIL_ORDER_DETAIL_TOOL, kwargs={"order_id": next_order_id})
                return action, {
                    "verification_ok": True,
                    "calls": [{"tool_name": action.name, "arguments": action.kwargs}],
                    "planner_skipped": True,
                    "tau_policy_guard": {
                        "used": True,
                        "reason": "continue_after_success_lookup_next_requested_order",
                    },
                }
            return action, {
                "verification_ok": True,
                "calls": [{"tool_name": action.name, "arguments": action.kwargs}],
                "planner_skipped": True,
                "tau_policy_guard": {"used": True, "reason": "respond_after_successful_mutation"},
            }
        if latest_user_confirmed_mutation_prompt(messages) and latest_deferred_mutating_action(messages) is None:
            return None, {}
        seed = fallback_response("task")
        guarded_action, guard = self.guard_action(seed, messages)
        if not guard.get("used"):
            return None, {}
        if guarded_action.name == seed.name and guarded_action.kwargs == seed.kwargs:
            return None, {}
        result = {
            "verification_ok": True,
            "calls": [{"tool_name": guarded_action.name, "arguments": guarded_action.kwargs}],
            "planner_skipped": True,
            "tau_policy_guard": guard,
        }
        return guarded_action, result

    def runtime_legal_transition_action(self) -> tuple[Action | None, dict[str, Any], float]:
        """Execute or select among runtime-proven read-only resolution transitions.

        This is deliberately narrow: it only bypasses model planning when the
        typed episode state has a legal lookup menu. Writes and ordinary
        semantic tool selection remain on the shared model/binder path.
        """
        transitions = [
            transition
            for transition in self._episode_state.legal_resolution_transitions()
            if str(transition.get("tool_name") or "") in self.available_tool_names
        ]
        if not transitions:
            return None, {}, 0.0
        selection: dict[str, Any] = {"model_call": False, "valid": True}
        transition: dict[str, Any] | None = None
        if len(transitions) == 1:
            transition = transitions[0]
        else:
            selection_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one minified JSON object with transition_id and evidence_ids. "
                        "Choose one supplied legal transition only. Do not create tools, arguments, values, "
                        "or additional fields. The first character must be '{'."
                    ),
                },
                {
                    "role": "user",
                    "content": short_json(
                        {
                            "runtime_state": self._episode_state.planner_projection(max_facts=24),
                            "legal_transitions": transitions,
                        },
                        12000,
                    ),
                },
            ]
            started_at = time.time()
            raw_text = generate_text(
                self.model_obj,
                self.tokenizer,
                selection_messages,
                min(max(self.max_new_tokens // 3, 240), 600),
            )
            latency_ms = round((time.time() - started_at) * 1000, 3)
            parsed, parse_error = extract_json_object(raw_text)
            transition_id = str(parsed.get("transition_id") or "") if isinstance(parsed, dict) else ""
            transition = next(
                (candidate for candidate in transitions if candidate.get("transition_id") == transition_id),
                None,
            )
            selection = {
                "model_call": True,
                "raw_text": raw_text,
                "parse_error": parse_error,
                "selected_transition_id": transition_id,
                "valid": transition is not None,
            }
            if transition is None:
                return None, {"runtime_legal_transition_selection": selection}, latency_ms
        if transition is None:
            return None, {}, 0.0
        arguments = transition.get("arguments") if isinstance(transition.get("arguments"), dict) else {}
        action = Action(name=str(transition["tool_name"]), kwargs=arguments)
        return action, {
            "verification_ok": True,
            "calls": [{"tool_name": action.name, "arguments": dict(action.kwargs)}],
            "planner_skipped": True,
            "runtime_legal_transition": transition,
            "runtime_legal_transition_selection": selection,
            "stateful_goal_ledger": self._episode_state.goal_ledger(),
        }, 0.0 if not selection["model_call"] else latency_ms

    def admit_collection_resolution(self, result: dict[str, Any]) -> dict[str, Any]:
        """Persist the shared runtime's bounded collection-search route.

        The generic compiler already proves that this is one read-only route
        over a source-audited collection. Here it becomes explicit state so
        subsequent candidates are scheduled by the runtime rather than being
        reconstructed from transcript text.
        """
        details = result.get("stateful_collection_disambiguation")
        if not isinstance(details, dict) or not details.get("used"):
            return {"used": False}
        tool_name = str(details.get("tool_name") or "").strip()
        input_name = str(details.get("input_name") or "").strip()
        raw_candidates = details.get("candidate_ids")
        candidates = [str(value).strip() for value in raw_candidates if str(value).strip()] if isinstance(raw_candidates, list) else []
        if not tool_name or not input_name or len(dict.fromkeys(candidates)) < 2:
            return {"used": False, "reason": "incomplete_collection_route"}
        key = short_json({"tool": tool_name, "argument": input_name, "candidates": candidates}, 1000)
        resolution_id = "resolution_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        resolution = self._episode_state.open_resolution(
            resolution_id,
            input_name.removesuffix("_id") or "record",
            candidates,
            lookup_tool_name=tool_name,
            lookup_argument_name=input_name,
            lookup_budget=len(candidates),
        )
        attempted = details.get("attempted_ids")
        attempted_ids = [str(value).strip() for value in attempted if str(value).strip()] if isinstance(attempted, list) else []
        for candidate_id in attempted_ids:
            if candidate_id in resolution.candidate_ids:
                self._episode_state.record_resolution_inspection(resolution.resolution_id, candidate_id)
        return {"used": True, "resolution_id": resolution.resolution_id, "state": resolution.to_dict()}

    def gate_mutation_action(self, action: Action, result: dict[str, Any]) -> Action:
        """Prepare exactly one effect packet for a capability-classified write.

        The shared capability registry, not a benchmark tool-name list, decides
        whether an action is mutating. A packet is committed only after the
        environment returns a successful observation in ``solve``.
        """
        capability = self._capability_registry.get(action.name)
        if capability is None or capability.kind != "mutate":
            return action
        confirmation_id: str | None = None
        if capability.requires_confirmation:
            confirmation = self._episode_state.confirmation_for_action(action.name, action.kwargs)
            if confirmation is None:
                pending = self._episode_state.confirmation_for_action(
                    action.name,
                    action.kwargs,
                    statuses={"pending"},
                )
                if pending is None:
                    target_ids = [
                        str(value)
                        for name, value in action.kwargs.items()
                        if re.search(r"(?:^|_)(?:id|identifier)$", str(name), re.I)
                        and isinstance(value, (str, int, float))
                        and str(value).strip()
                    ]
                    if not target_ids:
                        target_ids = ["arguments:" + hashlib.sha256(short_json(action.kwargs, 4000).encode("utf-8")).hexdigest()[:16]]
                    pending = self._episode_state.prepare_confirmation(
                        goal_ids=[],
                        operation=action.name,
                        target_ids=target_ids,
                        arguments=action.kwargs,
                        human_summary=f"{action.name.replace('_', ' ')} for {', '.join(target_ids)}",
                    )
                result["mutation_gate"] = {
                    "allowed": False,
                    "reason": "scoped_confirmation_required",
                    "confirmation": pending.to_dict(),
                }
                return fallback_response(f"Please confirm: {pending.human_summary}.")
            confirmation_id = confirmation.confirmation_id
        try:
            packet = self._episode_state.build_mutation_packet(
                goal_ids=[],
                tool_name=action.name,
                arguments=action.kwargs,
                confirmation_id=confirmation_id,
            )
        except ValueError as exc:
            result["mutation_gate"] = {"allowed": False, "reason": str(exc)}
            return fallback_response("The requested update has already been completed.")
        self._pending_mutation_packets[action_signature(action)] = packet
        result["mutation_gate"] = {"allowed": True, "packet": packet.to_dict()}
        return action

    def record_action_effect(self, action: Action, *, source_event_id: str, success: bool) -> None:
        packet = self._pending_mutation_packets.pop(action_signature(action), None)
        if packet is not None and success:
            self._episode_state.record_mutation_commit(packet, source_event_id=source_event_id)

    def apply_policy_repair_if_needed(
        self,
        action: Action,
        result: dict[str, Any],
        messages: List[Dict[str, Any]],
        previous_source: str,
        *,
        force: bool = False,
        rejected_action: Action | None = None,
    ) -> Action:
        repair_key = "tau_policy_repair" if self.legacy_tau_heuristics else "semantic_repair"
        if force and repair_key in result:
            repair_key += "_retry"
        already_repaired = repair_key in result
        if self.legacy_tau_heuristics:
            missed_confirmed_mutation = (
                not already_repaired
                and latest_user_confirmed_mutation_prompt(messages)
                and not mutating_action(action.name)
            )
            needs_repair = (
                non_actionable_action(action)
                or (not already_repaired and confirmation_followup_response_needs_action(action, messages))
                or missed_confirmed_mutation
            )
        else:
            needs_repair = force or (not already_repaired and semantic_repair_needed_generic(action, messages))
        if not self.policy_repair_enabled or not needs_repair:
            return action
        rejected_action = rejected_action or action
        repaired_action, policy_repair = self.policy_repair_action(messages, previous_source, rejected_action)
        if force:
            policy_repair["trigger"] = "schema_or_grounding_rejection"
        result[repair_key] = policy_repair
        if repaired_action is None:
            return action
        if self.legacy_tau_heuristics:
            repaired_action, repair = repair_action_arguments(repaired_action, messages, self.tool_schemas)
            if repair:
                policy_repair["argument_repair"] = repair
        guarded_action, repair_guard = self.guard_action(repaired_action, messages)
        if repair_guard.get("used"):
            policy_repair["action_verifier"] = repair_guard
            action = guarded_action
        else:
            action = repaired_action
        result[repair_key] = policy_repair
        return action

    def plan_action(self, messages: List[Dict[str, Any]], previous_source: str) -> tuple[Action, dict[str, Any], float]:
        legal_action, legal_result, legal_latency_ms = self.runtime_legal_transition_action()
        if legal_action is not None:
            return legal_action, legal_result, legal_latency_ms
        legal_selection_fallback = legal_result.get("runtime_legal_transition_selection") if legal_result else None
        preflight_action, preflight_result = self.preflight_action(messages)
        if preflight_action is not None:
            return preflight_action, preflight_result, 0.0

        request = self.build_request(messages, previous_source)
        binding_request = self.build_binding_request(messages, previous_source)
        execution_history = self._episode_state.execution_history()
        input_goal_ledger = self._episode_state.goal_ledger()
        start = time.time()
        result = plan_and_compile_goal_graph(
            self.model_obj,
            self.tokenizer,
            generate_text,
            request,
            self.goal_graph_tools,
            max_new_tokens=self.max_new_tokens,
            repair_attempts=self.repair_attempts,
            allow_side_effects=self.allow_side_effects,
            use_binder_fallback=True,
            planner_mode=self.planner_mode,
            stateful=True,
            binding_request=binding_request,
            execution_history=execution_history,
            stateful_goal_ledger=input_goal_ledger,
            stateful_goal_ledger_required=False,
            stateful_semantic_only=True,
            stateful_semantic_review=True,
        )
        if self.replay_trace_enabled:
            result["planning_request"] = request
            result["stateful_goal_ledger_input"] = input_goal_ledger
            result["episode_state"] = self._episode_state.to_dict(include_events=True)
        requested_fact_delta = result.get("stateful_requested_fact_delta")
        requested_fact_application = self._episode_state.apply_requested_fact_delta(
            requested_fact_delta,
            source_event_id=self._episode_state.latest_user_event_id(),
        )
        result["stateful_requested_fact_delta_application"] = requested_fact_application
        confirmation_application = self._episode_state.apply_confirmation_delta(
            result.get("stateful_confirmation_delta"),
            source_event_id=self._episode_state.latest_user_event_id(),
        )
        result["stateful_confirmation_delta_application"] = confirmation_application
        goal_delta = result.get("stateful_goal_delta")
        goal_delta_application = self._episode_state.apply_goal_delta(
            goal_delta,
            source_event_id=self._episode_state.latest_user_event_id(),
        )
        result["stateful_goal_delta_application"] = goal_delta_application
        result["stateful_goal_ledger"] = self._episode_state.goal_ledger()
        result["runtime_collection_resolution"] = self.admit_collection_resolution(result)
        if isinstance(legal_selection_fallback, dict):
            result["runtime_legal_transition_selection"] = legal_selection_fallback
        latency_ms = round((time.time() - start) * 1000, 3)
        action = action_from_goal_graph_result(
            result,
            self.available_tool_names,
            self.goal_graph_tools,
            execution_history,
        )
        if self.legacy_tau_heuristics:
            repaired_action, argument_repair = repair_action_arguments(action, messages, self.tool_schemas)
            if argument_repair:
                result["tau_argument_repair"] = argument_repair
                action = repaired_action
        action = self.apply_policy_repair_if_needed(action, result, messages, previous_source)
        guarded_action, guard = self.guard_action(action, messages, result)
        if guard.get("used"):
            result["tau_policy_guard" if self.legacy_tau_heuristics else "action_verifier"] = guard
            rejected_action = action
            action = guarded_action
            if (
                not self.legacy_tau_heuristics
                and str(guard.get("reason") or "")
                in {
                    "unknown_tool",
                    "missing_required_arguments",
                    "ungrounded_arguments",
                    "duplicate_successful_action",
                }
            ):
                action = self.apply_policy_repair_if_needed(
                    action,
                    result,
                    messages,
                    previous_source,
                    force=True,
                    rejected_action=rejected_action,
                )
        action = self.apply_policy_repair_if_needed(action, result, messages, previous_source)
        action = self.gate_mutation_action(action, result)
        return action, result, latency_ms

    def write_debug(self, task_index: Optional[int], row: Dict[str, Any]) -> None:
        if self.debug_dir is None:
            return
        suffix = task_index if task_index is not None else "unknown"
        try:
            with (self.debug_dir / f"task_{suffix}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            print(f"[tau-goal-graph] debug write failed: {type(exc).__name__}: {exc}", flush=True)

    def write_replay_manifest(self) -> None:
        if self.debug_dir is None:
            return
        payload = {
            "format": "tau_goal_graph_replay_v1",
            "goal_graph_tools": self.goal_graph_tools,
            "planner": {
                "mode": self.planner_mode,
                "max_new_tokens": self.max_new_tokens,
                "stateful": True,
                "semantic_only": True,
                "semantic_review": True,
            },
        }
        try:
            (self.debug_dir / "goal_graph_replay_manifest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[tau-goal-graph] replay manifest write failed: {type(exc).__name__}: {exc}", flush=True)

    def solve(self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        max_steps = min(self.max_steps, max_num_steps)
        reset = env.reset(task_index=task_index)
        info = model_to_dict(reset.info)
        reward = 0.0
        previous_source = "user"
        self._episode_state = EpisodeState()
        self._pending_mutation_packets = {}
        self._episode_state.record_user_message(reset.observation)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": reset.observation},
        ]
        diagnostics: list[dict[str, Any]] = []

        for step in range(max_steps):
            print(f"[tau-goal-graph] step {step} plan start", flush=True)
            action, result, latency_ms = self.plan_action(messages, previous_source)
            print(
                f"[tau-goal-graph] step {step} action={action.name} "
                f"verification_ok={bool(result.get('verification_ok'))}",
                flush=True,
            )
            try:
                response = env.step(action)
            except Exception as exc:  # noqa: BLE001
                if action.name != RESPOND_ACTION_NAME or not any_mutating_action_success_was_seen(messages):
                    raise
                action_dict = action_to_dict(action)
                error_text = f"{type(exc).__name__}: {exc}"
                diag = {
                    "step": step,
                    "goal_graph_result": result,
                    "final_action": action_dict,
                    "observation": "",
                    "done": True,
                    "latency_ms": latency_ms,
                    "source": "post_success_response_error",
                    "step_error": error_text,
                }
                diagnostics.append(diag)
                self.write_debug(task_index, diag)
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action_dict, ensure_ascii=False),
                        "action": action_dict,
                        "goal_graph": {
                            "verification_ok": bool(result.get("verification_ok")),
                            "diagnostic_codes": result.get("diagnostic_codes") or [],
                            "tau_policy_guard": result.get("tau_policy_guard"),
                        },
                    }
                )
                info["post_success_response_error"] = error_text
                try:
                    reward_res = env.calculate_reward()
                    reward = reward_res.reward
                    info = {**info, **model_to_dict(reward_res.info)}
                except Exception as reward_exc:  # noqa: BLE001
                    info["reward_error"] = f"{type(reward_exc).__name__}: {reward_exc}"
                break
            reward = response.reward
            info = {**info, **model_to_dict(response.info)}
            previous_source = getattr(response.info, "source", None) or "unknown"
            action_dict = action_to_dict(action)

            diag = {
                "step": step,
                "goal_graph_result": result,
                "final_action": action_dict,
                "observation": str(response.observation)[:4000],
                "done": response.done,
                "latency_ms": latency_ms,
                "source": previous_source,
            }
            diagnostics.append(diag)
            self.write_debug(task_index, diag)

            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(action_dict, ensure_ascii=False),
                    "action": action_dict,
                    "goal_graph": {
                        "verification_ok": bool(result.get("verification_ok")),
                        "diagnostic_codes": result.get("diagnostic_codes") or [],
                        "tau_policy_guard": result.get("tau_policy_guard"),
                    },
                }
            )
            if action.name != RESPOND_ACTION_NAME:
                messages.append({"role": "tool", "name": action.name, "content": response.observation})
                tool_event = self._episode_state.record_tool_result(
                    action.name,
                    action.kwargs,
                    response.observation,
                    success=not tool_observation_failed(response.observation),
                )
                self.record_action_effect(
                    action,
                    source_event_id=tool_event.event_id,
                    success=not tool_observation_failed(response.observation),
                )
                inspected_resolutions = self._episode_state.record_resolution_lookup(action.name, action.kwargs)
                if inspected_resolutions:
                    diag["runtime_resolution_lookups"] = inspected_resolutions
            else:
                messages.append({"role": "user", "content": response.observation})
                self._episode_state.record_user_message(response.observation)

            if response.done:
                # A simulator may stop immediately after replying to a clarification.
                # Its final user turn is still new state: give the shared planner one
                # chance to execute a verified transition, without starting another
                # user-simulation exchange.
                if action.name == RESPOND_ACTION_NAME and str(response.observation).strip():
                    final_action, final_result, final_latency_ms = self.plan_action(messages, "user")
                    final_action_dict = action_to_dict(final_action)
                    diag["terminal_user_finalization"] = {
                        "planned_action": final_action_dict,
                        "executed": final_action.name != RESPOND_ACTION_NAME,
                    }
                    if final_action.name != RESPOND_ACTION_NAME:
                        final_response = env.step(final_action)
                        reward = final_response.reward
                        info = {**info, **model_to_dict(final_response.info)}
                        final_source = getattr(final_response.info, "source", None) or final_action.name
                        final_diag = {
                            "step": step,
                            "goal_graph_result": final_result,
                            "final_action": final_action_dict,
                            "observation": str(final_response.observation)[:4000],
                            "done": final_response.done,
                            "latency_ms": final_latency_ms,
                            "source": final_source,
                            "terminal_user_finalization": True,
                        }
                        diagnostics.append(final_diag)
                        self.write_debug(task_index, final_diag)
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(final_action_dict, ensure_ascii=False),
                                "action": final_action_dict,
                                "goal_graph": {
                                    "verification_ok": bool(final_result.get("verification_ok")),
                                    "diagnostic_codes": final_result.get("diagnostic_codes") or [],
                                    "tau_policy_guard": final_result.get("tau_policy_guard"),
                                },
                            }
                        )
                        messages.append(
                            {"role": "tool", "name": final_action.name, "content": final_response.observation}
                        )
                        tool_event = self._episode_state.record_tool_result(
                            final_action.name,
                            final_action.kwargs,
                            final_response.observation,
                            success=not tool_observation_failed(final_response.observation),
                        )
                        self.record_action_effect(
                            final_action,
                            source_event_id=tool_event.event_id,
                            success=not tool_observation_failed(final_response.observation),
                        )
                        inspected_resolutions = self._episode_state.record_resolution_lookup(
                            final_action.name,
                            final_action.kwargs,
                        )
                        if inspected_resolutions:
                            final_diag["runtime_resolution_lookups"] = inspected_resolutions
                break

        if info.get("reward_info") is None:
            try:
                reward_res = env.calculate_reward()
                reward = reward_res.reward
                info = {**info, **model_to_dict(reward_res.info)}
            except Exception as exc:  # noqa: BLE001
                info["reward_error"] = f"{type(exc).__name__}: {exc}"
        info["goal_graph_diagnostics"] = diagnostics
        info["agent"] = "goal_graph_runtime"
        return SolveResult(reward=reward, messages=messages, info=info, total_cost=0.0)
