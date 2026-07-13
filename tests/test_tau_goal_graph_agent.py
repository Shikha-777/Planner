from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tau_goal_graph_module(monkeypatch, plan_result=None, generated_texts=None):
    calls = []
    generated = list(generated_texts or [])
    if generated_texts is not None:
        monkeypatch.setenv("TAU_GOAL_GRAPH_POLICY_REPAIR", "1")

    def plan_and_compile_goal_graph(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return plan_result or {"verification_ok": True, "calls": []}

    def generate_text(*args, **kwargs):
        return generated.pop(0) if generated else ""

    monkeypatch.setitem(
        sys.modules,
        "goal_graph_eval_common",
        types.SimpleNamespace(plan_and_compile_goal_graph=plan_and_compile_goal_graph),
    )
    monkeypatch.setitem(
        sys.modules,
        "run_gptoss_capability_plan",
        types.SimpleNamespace(
            generate_text=generate_text,
            load_model=lambda model: (f"model:{model}", "tokenizer"),
        ),
    )

    module_name = "tau_goal_graph_agent_for_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "compute2" / "tau_goal_graph_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module._planner_calls = calls
    return module


def test_tau_tool_conversion_adds_respond_without_marking_everything_read_only(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    tools = module.tau_tools_to_goal_graph_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "update_order",
                    "description": "Update an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ]
    )

    assert [tool["name"] for tool in tools] == ["update_order", "respond"]
    assert tools[0]["parameters"]["required"] == ["order_id"]
    assert tools[0]["requires_confirmation"] is False
    assert tools[0]["requires_unique_target"] is False
    assert "risk" not in tools[0]
    assert tools[1]["parameters"]["required"] == ["content"]
    assert tools[1]["kind"] == "retrieve"
    assert tools[1]["risk"] == "read_only"
    assert tools[1]["requires_confirmation"] is False
    assert tools[1]["requires_unique_target"] is False


def test_action_from_goal_graph_result_maps_tools_and_response(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    tool_action = module.action_from_goal_graph_result(
        {"calls": [{"tool_name": "get_order", "arguments": {"order_id": "O-1"}}]},
        {"get_order"},
    )
    assert tool_action.name == "get_order"
    assert tool_action.kwargs == {"order_id": "O-1"}

    response_action = module.action_from_goal_graph_result(
        {"calls": [{"tool_name": "respond", "arguments": {"message": "Done."}}]},
        {"get_order"},
    )
    assert response_action.name == "respond"
    assert response_action.kwargs == {"content": "Done."}

    unknown_action = module.action_from_goal_graph_result(
        {"calls": [{"tool_name": "delete_everything", "arguments": {}}]},
        {"get_order"},
    )
    assert unknown_action.name == "respond"
    assert "clarify" in unknown_action.kwargs["content"].lower()


def test_action_from_goal_graph_result_skips_completed_stateful_replay(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action = module.action_from_goal_graph_result(
        {
            "calls": [
                {"tool_name": "lookup_order", "arguments": {"order_id": "O-1"}},
                {"tool_name": "lookup_user", "arguments": {"user_id": "U-1"}},
            ]
        },
        {"lookup_order", "lookup_user"},
        execution_history=[
            {"tool_name": "lookup_order", "arguments": {"order_id": "O-1"}, "outcome": "success"}
        ],
    )

    assert action.name == "lookup_user"
    assert action.kwargs == {"user_id": "U-1"}


def test_action_from_goal_graph_result_renders_semantic_ask_user(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action = module.action_from_goal_graph_result(
        {
            "calls": [],
            "tool_binding_plan": {
                "tool_decision": "ask_user",
                "missing_inputs": [],
                "capability_plan": {
                    "semantic_input_frame": {
                        "tool_decision": "ask_user",
                        "canonical_request": "Confirm the requested account change before proceeding.",
                    }
                },
            },
        },
        {"update_account"},
    )

    assert action.name == "respond"
    assert "confirm" in action.kwargs["content"].lower()
    assert "account change" in action.kwargs["content"].lower()


def test_action_from_goal_graph_result_preserves_semantic_alternative_clarification(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action = module.action_from_goal_graph_result(
        {
            "calls": [],
            "tool_binding_plan": {
                "tool_decision": "ask_user",
                "missing_inputs": ["identity evidence"],
                "capability_plan": {
                    "semantic_input_frame": {
                        "tool_decision": "ask_user",
                        "missing_inputs": ["identity evidence"],
                        "clarification_message": "Please provide either your email or your full name and ZIP code.",
                    }
                },
            },
        },
        {"update_account"},
    )

    assert action.name == "respond"
    assert action.kwargs == {
        "content": "Please provide either your email or your full name and ZIP code."
    }


def test_action_from_goal_graph_result_renders_semantic_final_response(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action = module.action_from_goal_graph_result(
        {
            "calls": [],
            "tool_binding_plan": {
                "capability_plan": {
                    "semantic_input_frame": {
                        "tool_decision": "no_tool",
                        "response_message": "The requested operation is complete.",
                    }
                }
            },
        },
        set(),
    )

    assert action.name == "respond"
    assert action.kwargs["content"] == "The requested operation is complete."


def test_execution_history_clears_failed_calls_after_new_user_turn(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    history = module.execution_history_from_messages(
        [
            {"role": "assistant", "action": {"name": "lookup_a", "kwargs": {"id": "A-1"}}},
            {"role": "tool", "name": "lookup_a", "content": '{"id":"A-1"}'},
            {"role": "assistant", "action": {"name": "lookup_b", "kwargs": {"id": "B-1"}}},
            {"role": "tool", "name": "lookup_b", "content": "Error: not found"},
            {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Done."}}},
            {"role": "user", "content": "Thanks."},
        ]
    )

    assert history == [
        {
            "tool_name": "lookup_a",
            "arguments": {"id": "A-1"},
            "outcome": "success",
            "observation": {"id": "A-1"},
        }
    ]


def test_execution_history_keeps_failed_call_until_new_user_turn(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    history = module.execution_history_from_messages(
        [
            {"role": "assistant", "action": {"name": "calculate", "kwargs": {"expression": "PM"}}},
            {"role": "tool", "name": "calculate", "content": "Error: invalid characters in expression"},
        ]
    )

    assert history == [{"tool_name": "calculate", "arguments": {"expression": "PM"}, "outcome": "failure"}]


def test_action_from_goal_graph_result_prefers_final_call_skeleton(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action = module.action_from_goal_graph_result(
        {
            "calls": [{"tool_name": "respond", "arguments": {"content": "process this quickly"}}],
            "call_skeleton_output": {
                "parsed": {
                    "ordered_calls": [
                        {
                            "tool_name": "respond",
                            "arguments": {"content": "Please provide your name and ZIP code."},
                        }
                    ]
                }
            },
        },
        {"find_user_id_by_email"},
    )

    assert action.name == "respond"
    assert action.kwargs == {"content": "Please provide your name and ZIP code."}


def test_goal_graph_agent_uses_stepwise_planner_directly(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "get_order", "arguments": {"order_id": "O-1"}}],
        },
    )

    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Get an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, latency_ms = agent.plan_action(
        [{"role": "user", "content": "Please check order O-1."}],
        previous_source="user",
    )

    assert action.name == "get_order"
    assert action.kwargs == {"order_id": "O-1"}
    assert result["verification_ok"] is True
    assert latency_ms >= 0
    planner_call = module._planner_calls[-1]
    assert planner_call["args"][3].startswith("Current task:")
    assert planner_call["kwargs"]["binding_request"].startswith("Binding evidence")
    assert "Policy text" not in planner_call["kwargs"]["binding_request"]
    assert planner_call["kwargs"]["planner_mode"] == "stepwise"
    assert planner_call["kwargs"]["allow_side_effects"] is True
    assert planner_call["kwargs"]["use_binder_fallback"] is True
    assert planner_call["kwargs"]["stateful"] is True
    assert planner_call["kwargs"]["stateful_semantic_only"] is True
    assert planner_call["kwargs"]["stateful_semantic_review"] is True
    assert planner_call["kwargs"]["execution_history"] == []
    assert planner_call["kwargs"]["stateful_goal_ledger"] == {}
    assert planner_call["kwargs"]["stateful_goal_ledger_required"] is False


def test_goal_graph_agent_applies_additive_goal_delta_to_runtime_owned_ledger(monkeypatch):
    goal_delta = {
        "add": [
            {
                "goal_id": "goal_lookup",
                "kind": "retrieve",
                "objective": "Retrieve record O-1.",
                "evidence_ids": ["event_1"],
            }
        ]
    }
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={"verification_ok": True, "calls": [], "stateful_goal_delta": goal_delta},
    )
    agent = module.GoalGraphAgent(
        tools_info=[],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )

    agent.plan_action(
        [{"role": "user", "content": "Please update record O-1."}],
        previous_source="user",
    )
    request = agent.build_request(
        [{"role": "user", "content": "Please continue."}],
        previous_source="user",
    )

    assert agent._episode_state.goal_ledger()["goals"][0]["id"] == "goal_lookup"
    assert "Runtime-owned episode state" in request
    assert '"goal_lookup"' in request


def test_goal_graph_agent_executes_a_single_runtime_legal_resolution_transition(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "get_record",
                    "description": "Get a record.",
                    "parameters": {
                        "type": "object",
                        "properties": {"record_id": {"type": "string"}},
                        "required": ["record_id"],
                    },
                },
            }
        ],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )
    agent._episode_state.open_resolution(
        "resolve_record_1",
        "record",
        ["R-1", "R-2"],
        lookup_tool_name="get_record",
        lookup_argument_name="record_id",
    )

    action, result, latency_ms = agent.plan_action([], previous_source="tool")

    assert action.name == "get_record"
    assert action.kwargs == {"record_id": "R-1"}
    assert latency_ms == 0.0
    assert result["planner_skipped"] is True
    assert result["runtime_legal_transition"]["purpose"] == "identify_target"
    assert module._planner_calls == []


def test_goal_graph_agent_model_selects_only_from_multiple_legal_transitions(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        generated_texts=['{"transition_id":"resolve_b.lookup.B-1","evidence_ids":[]}'],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "get_a",
                    "parameters": {
                        "type": "object",
                        "properties": {"a_id": {"type": "string"}},
                        "required": ["a_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_b",
                    "parameters": {
                        "type": "object",
                        "properties": {"b_id": {"type": "string"}},
                        "required": ["b_id"],
                    },
                },
            },
        ],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )
    agent._episode_state.open_resolution(
        "resolve_a",
        "record_a",
        ["A-1", "A-2"],
        lookup_tool_name="get_a",
        lookup_argument_name="a_id",
    )
    agent._episode_state.open_resolution(
        "resolve_b",
        "record_b",
        ["B-1", "B-2"],
        lookup_tool_name="get_b",
        lookup_argument_name="b_id",
    )

    action, result, _latency_ms = agent.plan_action([], previous_source="tool")

    assert action.name == "get_b"
    assert action.kwargs == {"b_id": "B-1"}
    assert result["runtime_legal_transition_selection"]["model_call"] is True
    assert result["runtime_legal_transition_selection"]["valid"] is True
    assert module._planner_calls == []


def test_binding_transcript_preserves_raw_observation_without_action_history(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    text = module.transcript_for_binding(
        [
            {"role": "user", "content": "Please check order #W5442520."},
            {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
            {
                "role": "tool",
                "name": "get_order_details",
                "content": '{"order_id":"#W5442520","status":"pending","payment_method_id":"paypal_1"}',
            },
        ]
    )

    assert "#W5442520" in text
    assert "pending" in text
    assert "paypal_1" in text
    assert "get_order_details" not in text
    assert "payment_method_id" in text
    assert "kwargs" not in text
    assert "assistant_action_values" not in text
    assert "tool_observation_values" not in text


def test_binding_transcript_preserves_structured_observation_values(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    text = module.transcript_for_binding(
        [
            {"role": "assistant", "action": {"name": "wrong_previous_action", "kwargs": {"user_id": "wrong-user"}}},
            {
                "role": "tool",
                "name": "get_record",
                "content": (
                    '{"order_id":"O-1","status":"pending","passengers":[1988,423],'
                    '"payment_method_id":"pay-1","created_at":"2024-05-17"}'
                ),
            },
        ],
    )

    assert '"order_id": "O-1"' in text
    assert '"passengers": [1988, 423]' in text
    assert '"created_at": "2024-05-17"' in text
    assert "wrong-user" not in text
    assert "get_record" not in text


def test_binding_transcript_keeps_collection_shape_without_name_mapping(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    text = module.transcript_for_binding(
        [{"role": "tool", "content": '{"reservations":["R-1","R-2"],"amount":123}'}],
    )

    assert '"reservations": ["R-1", "R-2"]' in text
    assert '"amount": 123' in text


def test_bounded_stateful_transcript_preserves_goal_and_recent_state(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [{"role": "user", "content": "Initial goal: update order O-1."}]
    messages.extend(
        {"role": "tool", "name": "lookup", "content": f"intermediate-state-{index}-" + "x" * 80}
        for index in range(8)
    )
    messages.append({"role": "user", "content": "Latest instruction: use the confirmed address."})

    rendered = module.bounded_stateful_transcript(
        messages,
        module.transcript_for_goal_graph,
        max_messages=3,
        max_chars=220,
    )

    assert len(rendered) <= 220
    assert "Initial goal" in rendered
    assert "Latest instruction" in rendered


def test_goal_graph_agent_passes_raw_observation_to_shared_binder(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "name": "update_record",
                "description": "Update a record.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            }
        ],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )

    request = agent.build_binding_request(
        [
            {"role": "user", "content": "Update my record."},
            {"role": "tool", "name": "get_record", "content": '{"record_id":"R-1","counter":999}'},
        ],
        previous_source="get_record",
    )

    assert '"record_id": "R-1"' in request
    assert '"counter": 999' in request


def test_goal_graph_agent_accepts_raw_tau_tool_schemas(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "search_direct_flight", "arguments": {"origin": "DFW", "destination": "EWR"}}],
        },
    )

    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "name": "search_direct_flight",
                "description": "Search direct flights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            }
        ],
        wiki="Policy text",
        model="local-goal-graph",
        provider="openai",
    )

    action, _result, _latency_ms = agent.plan_action(
        [{"role": "user", "content": "Move my flight from DFW to EWR."}],
        previous_source="user",
    )

    assert action.name == "search_direct_flight"
    assert action.kwargs == {"origin": "DFW", "destination": "EWR"}


def test_policy_repair_rejects_ungrounded_model_arguments(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action, info = module.policy_repair_action_from_raw(
        '{"action":{"name":"get_order_details","kwargs":{"order_id":"#W9999999"}}}',
        [{"role": "user", "content": "Please check my order."}],
        {"get_order_details"},
        {"get_order_details": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    )

    assert action is None
    assert info["accepted"] is False
    assert info["reason"] == "ungrounded_arguments"
    assert info["ungrounded_arguments"] == ["order_id"]


def test_goal_graph_agent_uses_gpt_policy_repair_for_non_actionable_response(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "task"}}],
        },
        generated_texts=[
            (
                '{"action":{"name":"cancel_pending_order",'
                '"kwargs":{"order_id":"#W3942868","reason":"no longer needed"}},'
                '"reason":"The observed order is pending and the user wants a refund rather than keeping it.",'
                '"evidence":["#W3942868","\"status\":\"pending\"","refund"]}'
            )
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "cancel_pending_order",
                    "description": "Cancel a pending order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_user_id_by_email",
                    "description": "Find a user.",
                    "parameters": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "description": "Get an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            },
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "user", "content": "My email is harper@example.com. I want a full refund for order #W3942868."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "harper@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "harper_moore_3210"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W3942868"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W3942868","status":"pending"}'},
    ]

    action, result, _latency_ms = agent.plan_action(messages, previous_source="user")

    assert action.name == "cancel_pending_order"
    assert action.kwargs == {"order_id": "#W3942868", "reason": "no longer needed"}
    assert result["semantic_repair"]["accepted"] is True


def test_goal_graph_agent_repairs_response_after_mutation_confirmation(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "I will make that change."}}],
        },
        generated_texts=[
            '{"action":{"name":"modify_pending_order_payment","kwargs":{"order_id":"#W5442520"}},"reason":"The user confirmed the pending order payment change.","evidence":["#W5442520"]}'
        ],
    )
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "modify_pending_order_payment",
                "description": "Modify payment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "payment_method_id": {"type": "string"},
                    },
                    "required": ["order_id", "payment_method_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get user details.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        },
    ]
    agent = module.GoalGraphAgent(
        tools_info=tools_info,
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "user", "content": "Please change order #W5442520 to PayPal."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W5442520","status":"pending","user_id":"olivia_ito_3591"}'},
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "I will change the payment method to PayPal. Please confirm by replying yes."},
            },
        },
        {"role": "user", "content": "Yes, that's correct."},
    ]

    action, result, _latency_ms = agent.plan_action(messages, previous_source="user")

    assert action.name == "get_user_details"
    assert action.kwargs == {"user_id": "olivia_ito_3591"}
    assert result["tau_policy_repair"]["accepted"] is True
    assert result["tau_policy_repair"]["action_verifier"]["reason"] == "lookup_user_before_mutation"


def test_goal_graph_agent_repairs_lookup_after_mutation_confirmation(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "get_order_details", "arguments": {"order_id": "#W7941031"}}],
        },
        generated_texts=[
            '{"action":{"name":"modify_pending_order_payment","kwargs":{"order_id":"#W5442520","payment_method_id":"paypal_8049766"}},"reason":"The user confirmed the previous payment change before adding a new request.","evidence":["#W5442520","paypal_8049766"]}'
        ],
    )
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "modify_pending_order_payment",
                "description": "Modify payment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "payment_method_id": {"type": "string"},
                    },
                    "required": ["order_id", "payment_method_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get user.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        },
    ]
    agent = module.GoalGraphAgent(
        tools_info=tools_info,
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "user", "content": "Please change order #W5442520 to PayPal paypal_8049766."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W5442520","status":"pending","user_id":"olivia_ito_3591"}'},
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "I will change the payment method to PayPal paypal_8049766. Please confirm by replying yes."},
            },
        },
        {"role": "user", "content": "Yes, that's correct. Also update order #W7941031."},
    ]

    action, result, _latency_ms = agent.plan_action(messages, previous_source="user")

    assert action.name == "get_user_details"
    assert action.kwargs == {"user_id": "olivia_ito_3591"}
    assert result["tau_policy_repair"]["accepted"] is True
    assert result["tau_policy_repair"]["action_verifier"]["reason"] == "lookup_user_before_mutation"


def test_tau_policy_guard_repairs_placeholder_required_arguments(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="find_user_id_by_name_zip",
        kwargs={"first_name": "task", "last_name": "task", "zip": "task"},
    )
    schemas = {
        "find_user_id_by_name_zip": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "zip": {"type": "string"},
            },
            "required": ["first_name", "last_name", "zip"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        proposed,
        [{"role": "user", "content": "My name is Olivia Ito and my zip code is 80218."}],
        {"find_user_id_by_name_zip"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "execute_argument_repair"
    assert guarded.name == "find_user_id_by_name_zip"
    assert guarded.kwargs == {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}


def test_goal_graph_agent_uses_gpt_policy_repair_for_placeholder_transfer(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_PREFLIGHT", "0")
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "transfer_to_human_agents", "arguments": {"summary": "task"}}],
        },
        generated_texts=[
            '{"action":{"name":"get_order_details","kwargs":{"order_id":"#W5442520"}},"reason":"Need current order state.","evidence":["#W5442520"]}'
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "transfer_to_human_agents",
                    "description": "Transfer.",
                    "parameters": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_details",
                    "description": "Get an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            },
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, _latency_ms = agent.plan_action(
        [{"role": "user", "content": "Please update order #W5442520."}],
        previous_source="user",
    )

    assert action.name == "get_order_details"
    assert action.kwargs == {"order_id": "#W5442520"}
    assert result["semantic_repair"]["accepted"] is True


def test_generic_semantic_repair_replaces_handoff_with_grounded_tool(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "handoff_to_specialist", "arguments": {"summary": "duplicate"}}],
        },
        generated_texts=[
            (
                '{"action":{"name":"close_case","kwargs":{"case_ref":"C-17","resolution":"duplicate"}},'
                '"reason":"The case is explicitly marked duplicate.",'
                '"evidence":["C-17","duplicate"]}'
            )
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "handoff_to_specialist",
                    "description": "Hand off a case.",
                    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "close_case",
                    "description": "Close a case.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_ref": {"type": "string"},
                            "resolution": {"type": "string", "enum": ["duplicate", "resolved"]},
                        },
                        "required": ["case_ref", "resolution"],
                    },
                },
            },
        ],
        wiki="Cases marked duplicate may be closed without a handoff.",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, _latency_ms = agent.plan_action(
        [{"role": "user", "content": "Please close case C-17; it is a duplicate."}],
        previous_source="user",
    )

    assert agent.legacy_tau_heuristics is False
    assert action.name == "close_case"
    assert action.kwargs == {"case_ref": "C-17", "resolution": "duplicate"}
    assert result["semantic_repair"]["accepted"] is True


def test_generic_semantic_repair_breaks_repeated_clarification_loop(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "Please provide the case reference."}}],
        },
        generated_texts=[
            (
                '{"action":{"name":"get_case","kwargs":{"case_ref":"C-17"}},'
                '"reason":"The customer supplied the requested reference.",'
                '"evidence":["case reference","C-17"]}'
            )
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "get_case",
                    "description": "Retrieve one case.",
                    "parameters": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]},
                },
            }
        ],
        wiki="Use a case reference to retrieve the case.",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide the case reference."}}},
        {"role": "user", "content": "The case reference is C-17."},
    ]

    action, result, _latency_ms = agent.plan_action(messages, previous_source="user")

    assert action.name == "get_case"
    assert action.kwargs == {"case_ref": "C-17"}
    assert result["semantic_repair"]["accepted"] is True


def test_generic_semantic_repair_retries_schema_rejection_with_another_tool(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "archive_case", "arguments": {"case_ref": "C-17"}}],
        },
        generated_texts=[
            (
                '{"action":{"name":"get_case","kwargs":{"case_ref":"C-17"}},'
                '"reason":"The archive call lacks a required reason, so retrieve the case first.",'
                '"evidence":["C-17"]}'
            )
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "archive_case",
                    "description": "Archive one case.",
                    "parameters": {
                        "type": "object",
                        "properties": {"case_ref": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["case_ref", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_case",
                    "description": "Retrieve one case.",
                    "parameters": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]},
                },
            },
        ],
        wiki="Retrieve a case before archiving it.",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, _latency_ms = agent.plan_action(
        [{"role": "user", "content": "Archive case C-17."}],
        previous_source="user",
    )

    assert action.name == "get_case"
    assert action.kwargs == {"case_ref": "C-17"}
    assert result["action_verifier"]["reason"] == "missing_required_arguments"
    assert result["semantic_repair"]["trigger"] == "schema_or_grounding_rejection"
    assert result["semantic_repair"]["accepted"] is True


def test_generic_semantic_repair_reviews_response_after_tool_observation(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "Please provide the case reference."}}],
        },
        generated_texts=[
            (
                '{"action":{"name":"resolve_case","kwargs":{"case_ref":"C-17","outcome":"resolved"}},'
                '"reason":"The tool observation already identifies the resolved case.",'
                '"evidence":["C-17","resolved"]}'
            )
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "resolve_case",
                    "description": "Resolve one case.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_ref": {"type": "string"},
                            "outcome": {"type": "string", "enum": ["resolved", "rejected"]},
                        },
                        "required": ["case_ref", "outcome"],
                    },
                },
            }
        ],
        wiki="A case returned with status resolved can be finalized.",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "user", "content": "Please resolve case C-17."},
        {"role": "assistant", "action": {"name": "get_case", "kwargs": {"case_ref": "C-17"}}},
        {"role": "tool", "name": "get_case", "content": '{"case_ref":"C-17","status":"resolved"}'},
    ]

    action, result, _latency_ms = agent.plan_action(messages, previous_source="get_case")

    assert action.name == "resolve_case"
    assert action.kwargs == {"case_ref": "C-17", "outcome": "resolved"}
    assert result["semantic_repair"]["accepted"] is True


def test_policy_repair_rejects_evidence_not_present_in_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action, info = module.policy_repair_action_from_raw(
        '{"action":{"name":"close_case","kwargs":{"case_ref":"C-17"}},"evidence":["C-99"]}',
        [{"role": "user", "content": "Please close case C-17."}],
        {"close_case"},
        {"close_case": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]}},
    )

    assert action is None
    assert info["reason"] == "ungrounded_evidence"


def test_policy_repair_requires_evidence_for_non_enum_arguments(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    action, info = module.policy_repair_action_from_raw(
        '{"action":{"name":"close_case","kwargs":{"case_ref":"C-17"}},"evidence":["duplicate"]}',
        [{"role": "user", "content": "Please close case C-17; it is a duplicate."}],
        {"close_case"},
        {"close_case": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]}},
    )

    assert action is None
    assert info["reason"] == "arguments_not_supported_by_evidence"
    assert info["unsupported_arguments"] == ["case_ref"]


def test_policy_repair_accepts_role_labelled_model_visible_evidence(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    message = "Please retrieve case C-17."

    action, info = module.policy_repair_action_from_raw(
        '{"action":{"name":"get_case","kwargs":{"case_ref":"C-17"}},"evidence":["user: Please retrieve case C-17."]}',
        [{"role": "user", "content": message}],
        {"get_case"},
        {"get_case": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]}},
    )

    assert action is not None
    assert action.name == "get_case"
    assert info["accepted"] is True


def test_generic_guard_accepts_exact_call_already_verified_by_goal_graph(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(name="search_records", kwargs={"date": "2024-05-20"})
    schema = {
        "search_records": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        }
    }
    result = {
        "verification_ok": True,
        "calls": [{"tool_name": "search_records", "arguments": {"date": "2024-05-20"}}],
    }

    guarded, info = module.generic_action_guard(
        action,
        [{"role": "user", "content": "Please search records from May 20th."}],
        {"search_records"},
        schema,
        result,
    )

    assert guarded == action
    assert info["used"] is False
    assert info["verified_goal_graph_call"] is True


def test_goal_graph_agent_retries_malformed_semantic_repair_as_compact_json(monkeypatch):
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={"verification_ok": True, "calls": [{"tool_name": "respond", "arguments": {"content": "task"}}]},
        generated_texts=[
            "analysis The next action should retrieve case C-17, but this output is not JSON.",
            '{"action":{"name":"get_case","kwargs":{"case_ref":"C-17"}},"evidence":["C-17"]}',
        ],
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "get_case",
                    "description": "Retrieve one case.",
                    "parameters": {"type": "object", "properties": {"case_ref": {"type": "string"}}, "required": ["case_ref"]},
                },
            }
        ],
        wiki="Retrieve a case when its reference is supplied.",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, _latency_ms = agent.plan_action(
        [{"role": "user", "content": "Please retrieve case C-17."}],
        previous_source="user",
    )

    assert action.name == "get_case"
    assert result["semantic_repair"]["format_retry_used"] is True


def test_tau_policy_guard_requires_auth_before_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W1234567", "reason": "no longer needed"},
    )

    guarded, info = module.tau_policy_guard_action(
        proposed,
        [{"role": "user", "content": "My email is a@example.com. Cancel order #W1234567."}],
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "authenticate_before_action"
    assert guarded.name == "find_user_id_by_email"
    assert guarded.kwargs == {"email": "a@example.com"}


def test_tau_policy_guard_requires_retail_auth_before_read_tool(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "#W1234567"})

    guarded, info = module.tau_policy_guard_action(
        proposed,
        [{"role": "user", "content": "My email is a@example.com. What is the status of order #W1234567?"}],
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "authenticate_before_action"
    assert guarded.name == "find_user_id_by_email"
    assert guarded.kwargs == {"email": "a@example.com"}


def test_tau_policy_guard_requests_auth_before_order_details_response(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="respond",
        kwargs={"content": "Please provide your order id, item ids, and payment method."},
    )

    guarded, info = module.tau_policy_guard_action(
        proposed,
        [{"role": "user", "content": "I need to return a skateboard and get a refund."}],
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_retail_authentication_before_response"
    assert guarded.name == "respond"
    assert "email" in guarded.kwargs["content"].lower()
    assert "zip" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_requests_name_zip_after_failed_email_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="respond", kwargs={"content": "Please provide your order id."})
    messages = [
        {"role": "user", "content": "I need to return a skateboard. My email is missing@example.com."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "missing@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "Error: user not found"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_retail_authentication_before_response"
    assert guarded.name == "respond"
    assert "first name" in guarded.kwargs["content"].lower()
    assert "zip" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_tries_next_grounded_email_after_failed_email_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": (
                "I need to cancel order #W9933266. It might be under raj89@example.com, "
                "rajlee@example.com, or lee42@example.com."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "raj89@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "Error: user not found"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] in {"transcript_proposal_execute_proposed_action", "transcript_proposal_authenticate_before_action"}
    assert guarded.name == "find_user_id_by_email"
    assert guarded.kwargs == {"email": "rajlee@example.com"}


def test_tau_policy_guard_does_not_repeat_failed_name_zip_auth_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "#W9933266"})
    messages = [
        {"role": "user", "content": "Please cancel order #W9933266. My full name is Raj Lee and zip code is 61379."},
        {
            "role": "assistant",
            "action": {
                "name": "find_user_id_by_name_zip",
                "kwargs": {"first_name": "Raj", "last_name": "Lee", "zip": "61379"},
            },
        },
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "Error: user not found"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_authentication_before_action"
    assert guarded.name == "respond"
    assert "another email" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_uses_full_name_zip_for_auth_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="respond", kwargs={"content": "Please provide your order id."})
    messages = [
        {"role": "user", "content": "I need to return a skateboard."},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide your name and ZIP."}}},
        {"role": "user", "content": "My full name is Mohamed Khan and my zip code is 60651."},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "find_user_id_by_name_zip"
    assert guarded.kwargs == {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}


def test_tau_policy_guard_uses_zip_then_name_for_auth_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="respond", kwargs={"content": "Please provide your order id."})
    messages = [
        {"role": "user", "content": "I need to return a skateboard."},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide your name and ZIP."}}},
        {"role": "user", "content": "My zip code is 60651. Oh, and my name is Mohamed Khan."},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "find_user_id_by_name_zip"
    assert guarded.kwargs == {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}


def test_extract_name_zip_ignores_busy_phrase_before_real_name(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    assert module.extract_name_zip(
        [
            {
                "role": "user",
                "content": (
                    "I bought a skateboard and need a full refund. My zip code is 60651 "
                    "and my name is Mohamed Khan. I'm really busy and need this quick."
                ),
            }
        ]
    ) == {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}


def test_extract_name_zip_handles_labeled_first_last_zip(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    assert module.extract_name_zip(
        [
            {
                "role": "user",
                "content": "My first name is Mohamed, last name Khan, and my zip code is 60651.",
            }
        ]
    ) == {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}


def test_extract_name_zip_does_not_invent_name_from_zip_only(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    assert module.extract_name_zip(
        [{"role": "user", "content": "Sure, my zip code is 60651. That should be enough to identify me."}]
    ) == {}


def test_extract_partial_name_zip_keeps_grounded_partial_fields(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    assert module.extract_partial_name_zip(
        [{"role": "user", "content": "My zip code is 60651 and my first name is Mohamed."}]
    ) == {"zip": "60651", "first_name": "Mohamed"}


def test_tau_policy_guard_requests_missing_last_name_for_partial_auth(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "My zip code is 60651 and my first name is Mohamed. Can we proceed with the refund?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_retail_authentication_before_response"
    assert guarded.name == "respond"
    content = guarded.kwargs["content"].lower()
    assert "last name" in content
    assert "first name" not in content


def test_tau_policy_guard_repairs_user_id_from_bare_auth_output(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_user_address",
        kwargs={
            "address1": "1 Main St",
            "address2": "",
            "city": "Boston",
            "state": "MA",
            "country": "USA",
            "zip": "02110",
        },
    )
    schemas = {
        "modify_user_address": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "address1": {"type": "string"},
                "address2": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "country": {"type": "string"},
                "zip": {"type": "string"},
            },
            "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
        }
    }
    messages = [
        {"role": "user", "content": "My email is a@example.com. Please change my address to 1 Main St, Boston MA 02110."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "sara_doe_496"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_user_details", "modify_user_address"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_user_before_mutation"
    assert info["argument_repair"]["filled"]["user_id"] == "sara_doe_496"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "sara_doe_496"}


def test_missing_required_arguments_allows_schema_documented_blank_string(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="modify_user_address",
        kwargs={
            "user_id": "sara_doe_496",
            "address1": "1 Main St",
            "address2": "",
            "city": "Boston",
            "state": "MA",
            "country": "USA",
            "zip": "02110",
        },
    )
    schemas = {
        "modify_user_address": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "address1": {"type": "string"},
                "address2": {"type": "string", "description": "The second line of the address, such as 'Apt 1' or ''."},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "country": {"type": "string"},
                "zip": {"type": "string"},
            },
            "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
        }
    }

    assert module.missing_required_arguments(action, schemas) == []


def test_tau_policy_guard_requires_order_lookup_after_auth(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W1234567", "reason": "no longer needed"},
    )
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_order_before_mutation"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W1234567"}


def test_tau_policy_guard_requires_confirmation_after_order_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W1234567", "reason": "no longer needed"},
    )
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "confirm_before_mutation"
    assert guarded.name == "respond"
    assert "confirm" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_allows_confirmed_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W1234567", "reason": "no longer needed"},
    )
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {
                    "content": "I can modify pending order items for order #W5442520 involving Patio Umbrella. Please confirm with yes."
                },
            },
        },
        {"role": "user", "content": "yes"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is False
    assert guarded is proposed


def test_tau_policy_guard_resumes_deferred_mutation_for_confirmation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Please provide auth info."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "request_authentication_before_mutation",
                    "proposed_action": {
                        "name": "cancel_pending_order",
                        "kwargs": {"order_id": "#W1234567", "reason": "no longer needed"},
                    },
                }
            },
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "resume_deferred_confirm_before_mutation"
    assert guarded.name == "respond"
    assert "confirm" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_executes_deferred_mutation_after_confirmation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Please provide auth info."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "request_authentication_before_mutation",
                    "proposed_action": {
                        "name": "cancel_pending_order",
                        "kwargs": {"order_id": "#W1234567", "reason": "no longer needed"},
                    },
                }
            },
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {
                    "content": "I can modify pending order items for order #W5442520 involving Patio Umbrella. Please confirm with yes."
                },
            },
        },
        {"role": "user", "content": "yes, please proceed"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "resume_deferred_execute_deferred_mutation"
    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W1234567", "reason": "no longer needed"}


def test_tau_policy_guard_blocks_duplicate_mutation_after_success(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W1234567", "reason": "no longer needed"},
    )
    messages = [
        {"role": "user", "content": "My email is a@example.com. Cancel order #W1234567 because I no longer need it."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}}},
        {"role": "user", "content": "Yes, please proceed."},
        {
            "role": "assistant",
            "action": {
                "name": "cancel_pending_order",
                "kwargs": {"order_id": "#W1234567", "reason": "no longer needed"},
            },
        },
        {
            "role": "tool",
            "name": "cancel_pending_order",
            "content": '{"order_id":"#W1234567","status":"cancelled","cancel_reason":"no longer needed"}',
        },
        {"role": "user", "content": "Please make sure the order is canceled completely."},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "avoid_duplicate_mutation_after_success"
    assert guarded.name == "respond"
    assert "already been completed" in guarded.kwargs["content"]


def test_tau_policy_guard_blocks_transcript_resurrection_after_success(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})
    messages = [
        {"role": "user", "content": "I want to exchange one item from order #W7273336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],'
                '"items":[{"name":"Gaming Mouse","product_id":"5713490933",'
                '"item_id":"8214883393","options":{"color":"black","sensor type":"laser"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443",'
                '"options":{"color":"white","sensor type":"optical"},"available":true,"price":137.22}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}}},
        {"role": "user", "content": "Yes, exchange the black laser mouse for a white optical one."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393"],
                    "new_item_ids": ["2880340443"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {
            "role": "tool",
            "name": "exchange_delivered_order_items",
            "content": '{"order_id":"#W7273336","status":"exchange requested"}',
        },
        {"role": "user", "content": "Please exchange the black laser mouse for the white optical one."},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "respond_after_prior_successful_mutation"
    assert guarded.name == "respond"
    assert "already been completed" in guarded.kwargs["content"]


def test_tau_policy_guard_requires_reservation_lookup_before_airline_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_reservation", kwargs={"reservation_id": "ZFA04Y"})

    guarded, info = module.tau_policy_guard_action(
        proposed,
        [{"role": "user", "content": "Please cancel reservation ZFA04Y."}],
        {"get_reservation_details", "cancel_reservation"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_reservation_before_mutation"
    assert guarded.name == "get_reservation_details"
    assert guarded.kwargs == {"reservation_id": "ZFA04Y"}


def test_tau_policy_guard_requires_user_lookup_after_reservation_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_reservation", kwargs={"reservation_id": "ZFA04Y"})
    messages = [
        {"role": "user", "content": "Please cancel reservation ZFA04Y."},
        {"role": "assistant", "action": {"name": "get_reservation_details", "kwargs": {"reservation_id": "ZFA04Y"}}},
        {"role": "tool", "name": "get_reservation_details", "content": '{"reservation_id":"ZFA04Y","user_id":"sara_doe_496"}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_reservation_details", "get_user_details", "cancel_reservation"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_user_before_mutation"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "sara_doe_496"}


def test_tau_policy_guard_requests_airline_cancel_reason_after_lookups(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_reservation", kwargs={"reservation_id": "ZFA04Y"})
    messages = [
        {"role": "user", "content": "Please cancel reservation ZFA04Y."},
        {"role": "assistant", "action": {"name": "get_reservation_details", "kwargs": {"reservation_id": "ZFA04Y"}}},
        {"role": "tool", "name": "get_reservation_details", "content": '{"reservation_id":"ZFA04Y","user_id":"sara_doe_496"}'},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"reservations":["ZFA04Y"]}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_reservation_details", "get_user_details", "cancel_reservation"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_airline_cancellation_reason"
    assert guarded.name == "respond"
    assert "reason" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_confirms_airline_cancel_after_reason(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_reservation", kwargs={"reservation_id": "ZFA04Y"})
    messages = [
        {"role": "user", "content": "Please cancel reservation ZFA04Y because of a change of plan."},
        {"role": "assistant", "action": {"name": "get_reservation_details", "kwargs": {"reservation_id": "ZFA04Y"}}},
        {"role": "tool", "name": "get_reservation_details", "content": '{"reservation_id":"ZFA04Y","user_id":"sara_doe_496"}'},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"reservations":["ZFA04Y"]}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_reservation_details", "get_user_details", "cancel_reservation"},
    )

    assert info["used"] is False
    assert guarded.name == "cancel_reservation"
    assert guarded.kwargs == {"reservation_id": "ZFA04Y"}


def test_tau_policy_guard_resumes_airline_cancel_after_reason_answer(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})
    messages = [
        {"role": "user", "content": "Please cancel reservation ZFA04Y."},
        {"role": "assistant", "action": {"name": "get_reservation_details", "kwargs": {"reservation_id": "ZFA04Y"}}},
        {"role": "tool", "name": "get_reservation_details", "content": '{"reservation_id":"ZFA04Y","user_id":"sara_doe_496"}'},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "sara_doe_496"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"reservations":["ZFA04Y"]}'},
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Please provide the reason for cancelling."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "request_airline_cancellation_reason",
                    "proposed_action": {"name": "cancel_reservation", "kwargs": {"reservation_id": "ZFA04Y"}},
                }
            },
        },
        {"role": "user", "content": "change of plan"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_reservation_details", "get_user_details", "cancel_reservation"},
    )

    assert info["used"] is True
    assert info["reason"] == "resume_deferred_execute_deferred_mutation"
    assert guarded.name == "cancel_reservation"
    assert guarded.kwargs == {"reservation_id": "ZFA04Y"}


def test_tau_policy_guard_avoids_airline_transfer_with_cancel_fallback(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    transfer = module.Action(
        name="transfer_to_human_agents",
        kwargs={
            "summary": (
                "User wants to change return flight from Texas to Newark, but reservation "
                "is basic economy which cannot be modified per policy."
            )
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "I'm trying to change my return flight from Texas back to Newark. "
                "My current flight leaves at 3 PM but I'd like to switch to a later one today."
            ),
        },
        {"role": "user", "content": "My user ID is olivia_gonzalez_2305, but I do not remember the reservation ID."},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_gonzalez_2305"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"user_id":"olivia_gonzalez_2305","reservations":["Z7GOZK"]}',
        },
        {"role": "assistant", "action": {"name": "get_reservation_details", "kwargs": {"reservation_id": "Z7GOZK"}}},
        {
            "role": "tool",
            "name": "get_reservation_details",
            "content": (
                '{"reservation_id":"Z7GOZK","user_id":"olivia_gonzalez_2305",'
                '"origin":"EWR","destination":"EWR","cabin":"basic_economy",'
                '"insurance":"yes","flights":[{"origin":"EWR","destination":"IAH",'
                '"flight_number":"HAT188","date":"2024-05-28"},{"origin":"IAH",'
                '"destination":"EWR","flight_number":"HAT207","date":"2024-05-28"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        transfer,
        messages,
        {"get_user_details", "get_reservation_details", "cancel_reservation", "transfer_to_human_agents"},
    )

    assert info["used"] is True
    assert info["reason"] == "avoid_transfer_airline_cancellation_fallback_request_airline_cancellation_reason"
    assert guarded.name == "respond"
    assert "reason" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_blocks_one_shot_item_exchange_while_user_requests_options(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["8896479688"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {"role": "user", "content": "Exchange order #W7273336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "omar@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "omar_lopez_3107"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W7273336","status":"delivered","user_id":"omar_lopez_3107"}'},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "omar_lopez_3107"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"payment_methods":{"paypal_1530316":{"id":"paypal_1530316"}}}'},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}}},
        {
            "role": "user",
            "content": "Yes, please proceed, but can you tell me what color options are available? I am not sure yet.",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "get_user_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_final_item_choices_before_one_shot_mutation"
    assert guarded.name == "respond"
    assert "final items" in guarded.kwargs["content"]


def test_tau_policy_guard_proposes_cancel_reservation_from_grounded_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})

    guarded, info = module.tau_policy_guard_action(
        fallback,
        [{"role": "user", "content": "Please cancel reservation ZFA04Y."}],
        {"get_reservation_details", "cancel_reservation"},
        {"cancel_reservation": {"type": "object", "properties": {"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_lookup_reservation_before_mutation"
    assert guarded.name == "get_reservation_details"
    assert guarded.kwargs == {"reservation_id": "ZFA04Y"}


def test_tau_policy_guard_proposes_order_lookup_from_grounded_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})

    guarded, info = module.tau_policy_guard_action(
        fallback,
        [{"role": "user", "content": "My email is a@example.com. What is the status of order #W1234567?"}],
        {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details"},
        {"get_order_details": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "find_user_id_by_email"
    assert guarded.kwargs == {"email": "a@example.com"}


def test_tau_policy_guard_proposes_name_zip_auth_from_grounded_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})

    guarded, info = module.tau_policy_guard_action(
        fallback,
        [{"role": "user", "content": "Sure, my name is Olivia Ito and my zip code is 80218."}],
        {"find_user_id_by_name_zip", "get_order_details"},
        {
            "find_user_id_by_name_zip": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["first_name", "last_name", "zip"],
            }
        },
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "find_user_id_by_name_zip"
    assert guarded.kwargs == {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}


def test_goal_graph_agent_preflights_grounded_auth_without_planner(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "fallback"}}],
        },
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "find_user_id_by_name_zip",
                    "description": "Find user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "first_name": {"type": "string"},
                            "last_name": {"type": "string"},
                            "zip": {"type": "string"},
                        },
                        "required": ["first_name", "last_name", "zip"],
                    },
                },
            }
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )

    action, result, latency_ms = agent.plan_action(
        [{"role": "user", "content": "My name is Olivia Ito and zip is 80218."}],
        previous_source="user",
    )

    assert action.name == "find_user_id_by_name_zip"
    assert action.kwargs == {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}
    assert result["planner_skipped"] is True
    assert result["tau_policy_guard"]["reason"] == "transcript_proposal_execute_proposed_action"
    assert latency_ms == 0.0
    assert module._planner_calls == []


def test_goal_graph_agent_preflights_success_response_after_mutation(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(
        monkeypatch,
        plan_result={
            "verification_ok": True,
            "calls": [{"tool_name": "respond", "arguments": {"content": "fallback"}}],
        },
    )
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "exchange_delivered_order_items",
                    "description": "Exchange items.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {"role": "user", "content": "Please exchange order #W7273336."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393"],
                    "new_item_ids": ["2880340443"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {"role": "tool", "name": "exchange_delivered_order_items", "content": '{"order_id":"#W7273336","status":"exchange requested"}'},
    ]

    action, result, latency_ms = agent.plan_action(messages, previous_source="exchange_delivered_order_items")

    assert action.name == "respond"
    assert "completed" in action.kwargs["content"]
    assert result["planner_skipped"] is True
    assert result["tau_policy_guard"]["reason"] == "respond_after_successful_mutation"
    assert latency_ms == 0.0
    assert module._planner_calls == []


def test_goal_graph_agent_preflight_resumes_deferred_action_before_success_response(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(monkeypatch)
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "modify_pending_order_items",
                    "description": "Modify items.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "item_ids": {"type": "array"},
                            "new_item_ids": {"type": "array"},
                            "payment_method_id": {"type": "string"},
                        },
                        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_pending_order_payment",
                    "description": "Modify payment.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {
            "role": "user",
            "content": "Update order #W5442520 to PayPal and exchange the Patio Umbrella.",
        },
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Working on it."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "update_payment_before_pending_item_mutation",
                    "proposed_action": {
                        "name": "modify_pending_order_items",
                        "kwargs": {
                            "order_id": "#W5442520",
                            "item_ids": ["3111466194"],
                            "new_item_ids": ["2001307871"],
                            "payment_method_id": "paypal_8049766",
                        },
                    },
                }
            },
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194"}]}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "modify_pending_order_payment",
                "kwargs": {"order_id": "#W5442520", "payment_method_id": "paypal_8049766"},
            },
        },
        {
            "role": "tool",
            "name": "modify_pending_order_payment",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194"}]}'
            ),
        },
    ]

    action, result, latency_ms = agent.plan_action(messages, previous_source="modify_pending_order_payment")

    assert action.name == "respond"
    assert result["planner_skipped"] is True
    assert result["tau_policy_guard"]["reason"] == "resume_deferred_confirm_before_mutation"
    assert "confirm" in action.kwargs["content"].lower()
    assert latency_ms == 0.0
    assert module._planner_calls == []


def test_goal_graph_agent_preflight_continues_after_success_when_actions_remain(monkeypatch):
    monkeypatch.setenv("TAU_GOAL_GRAPH_LEGACY_HEURISTICS", "1")
    module = load_tau_goal_graph_module(monkeypatch)
    agent = module.GoalGraphAgent(
        tools_info=[
            {
                "type": "function",
                "function": {
                    "name": "modify_pending_order_items",
                    "description": "Modify items.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "item_ids": {"type": "array"},
                            "new_item_ids": {"type": "array"},
                            "payment_method_id": {"type": "string"},
                        },
                        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_pending_order_payment",
                    "description": "Modify payment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}, "payment_method_id": {"type": "string"}},
                        "required": ["order_id", "payment_method_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_details",
                    "description": "Get product.",
                    "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
                },
            },
        ],
        wiki="Retail policy",
        model="local-goal-graph",
        provider="openai",
    )
    messages = [
        {
            "role": "user",
            "content": (
                "For #W5442520, change payment to PayPal and exchange the Patio Umbrella "
                "from red polyester manual tilt to blue sunbrella auto tilt."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_product_details", "kwargs": {"product_id": "9743693396"}}},
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Patio Umbrella","product_id":"9743693396","variants":{'
                '"2001307871":{"item_id":"2001307871","available":true,'
                '"options":{"size":"6 ft","color":"blue","material":"sunbrella","tilt mechanism":"auto tilt"}}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes to proceed with these updates."}}},
        {"role": "user", "content": "Yes, please proceed."},
        {"role": "assistant", "action": {"name": "modify_pending_order_payment", "kwargs": {"order_id": "#W5442520", "payment_method_id": "paypal_8049766"}}},
        {
            "role": "tool",
            "name": "modify_pending_order_payment",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
    ]

    action, result, latency_ms = agent.plan_action(messages, previous_source="modify_pending_order_payment")

    assert action.name == "modify_pending_order_items"
    assert action.kwargs["order_id"] == "#W5442520"
    assert action.kwargs["item_ids"] == ["3111466194"]
    assert action.kwargs["new_item_ids"] == ["2001307871"]
    assert action.kwargs["payment_method_id"] == "paypal_8049766"
    assert result["planner_skipped"] is True
    assert result["tau_policy_guard"]["reason"].startswith("continue_after_success_")
    assert latency_ms == 0.0


def test_tau_policy_guard_executes_deferred_action_after_other_mutation_success(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "Update order #W5442520 to PayPal and exchange the Patio Umbrella.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194"}]}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "modify_pending_order_payment",
                "kwargs": {"order_id": "#W5442520", "payment_method_id": "paypal_8049766"},
            },
        },
        {
            "role": "tool",
            "name": "modify_pending_order_payment",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194"}]}'
            ),
        },
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "confirm_before_mutation",
                    "proposed_action": {
                        "name": "modify_pending_order_items",
                        "kwargs": {
                            "order_id": "#W5442520",
                            "item_ids": ["3111466194"],
                            "new_item_ids": ["2001307871"],
                            "payment_method_id": "paypal_8049766",
                        },
                    },
                }
            },
        },
        {"role": "user", "content": "Yes, please go ahead and make the changes."},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {
            "find_user_id_by_name_zip",
            "get_order_details",
            "modify_pending_order_payment",
            "modify_pending_order_items",
        },
    )

    assert info["used"] is True
    assert info["reason"] == "resume_deferred_execute_deferred_mutation"
    assert guarded.name == "modify_pending_order_items"
    assert guarded.kwargs == {
        "order_id": "#W5442520",
        "item_ids": ["3111466194"],
        "new_item_ids": ["2001307871"],
        "payment_method_id": "paypal_8049766",
    }


def test_deferred_item_mutation_does_not_absorb_later_order_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["3111466194"],
            "new_item_ids": ["2001307871"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {"role": "user", "content": "Please update order #W5442520 to PayPal and exchange the Patio Umbrella."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":['
                '{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194","options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}},'
                '{"name":"Hiking Boots","product_id":"7363354090","item_id":"2648909398","options":{"size":"8","material":"leather","waterproof":"yes"}}'
                ']}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_product_details", "kwargs": {"product_id": "9743693396"}}},
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Patio Umbrella","product_id":"9743693396","variants":{'
                '"2001307871":{"item_id":"2001307871","available":true,"options":{"size":"6 ft","color":"blue","material":"sunbrella","tilt mechanism":"auto tilt"}},'
                '"9879255677":{"item_id":"9879255677","available":true,"options":{"size":"6 ft","color":"green","material":"olefin","tilt mechanism":"auto tilt"}}'
                '}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {
                    "content": "I can modify pending order items for order #W5442520 involving Patio Umbrella. Please confirm with yes."
                },
            },
        },
        {
            "role": "user",
            "content": (
                "Yes, please go ahead with the patio umbrella. Also, could you do the same for "
                "order #W7941031 and use my credit card for that one?"
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "modify_pending_order_payment", "modify_pending_order_items"},
    )

    assert info["used"] is False
    assert guarded.name == "modify_pending_order_items"
    assert guarded.kwargs == {
        "order_id": "#W5442520",
        "item_ids": ["3111466194"],
        "new_item_ids": ["2001307871"],
        "payment_method_id": "paypal_8049766",
    }


def test_tau_policy_guard_fetches_new_order_from_latest_turn(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Please confirm."})
    messages = [
        {"role": "user", "content": "Please change order #W5442520."},
        {
            "role": "assistant",
            "action": {
                "name": "find_user_id_by_name_zip",
                "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"},
            },
        },
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W5442520","status":"pending"}'},
        {"role": "user", "content": "Yes, and can you also do the same for order #W7941031?"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_name_zip", "get_order_details", "modify_pending_order_payment"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W7941031"}


def test_latest_unseen_order_ignores_typo_for_already_seen_order(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {"role": "user", "content": "Please check order #W7273336."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W7273336","status":"delivered"}'},
        {"role": "user", "content": "Yes, that order #W727336 is the one."},
    ]

    assert module.latest_unseen_order_id(messages) == ""


def test_tau_policy_guard_fetches_user_orders_when_order_id_unknown(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "find once I log in to my account"})
    messages = [
        {"role": "user", "content": "I want to exchange a couple items from a recent order."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "omar@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "omar_lopez_3107"},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Could you please provide the order ID?"}}},
        {
            "role": "user",
            "content": "I don't have the exact order ID. Can we proceed from my recent orders?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_email", "get_user_details", "get_order_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "omar_lopez_3107"}


def test_tau_policy_guard_fetches_user_details_after_auth_from_prior_order_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "I need help cancelling one of my recent orders."},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide your email."}}},
        {"role": "user", "content": "My email is harper@example.com."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "harper@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "harper_moore_3210"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"find_user_id_by_email", "get_user_details", "get_order_details", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "harper_moore_3210"}


def test_tau_policy_guard_retries_next_email_after_auth_lookup_failure(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="find_user_id_by_email", kwargs={"email": "raj89@example.com"})
    messages = [
        {
            "role": "user",
            "content": (
                "Can we use any of these emails: raj89@example.com, rajlee@example.com, "
                "lee42@example.com, or raj.lee6137@example.com to cancel the order?"
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "raj89@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "Error: user not found"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "find_user_id_by_name_zip"},
    )

    assert info["used"] is True
    assert info["reason"] == "retry_next_auth_email_after_lookup_failure"
    assert guarded.name == "find_user_id_by_email"
    assert guarded.kwargs == {"email": "rajlee@example.com"}


def test_tau_policy_guard_proposes_reservation_lookup_from_grounded_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "Could you clarify?"})

    guarded, info = module.tau_policy_guard_action(
        fallback,
        [{"role": "user", "content": "Can you check the details for reservation ZFA04Y?"}],
        {"get_reservation_details"},
        {"get_reservation_details": {"type": "object", "properties": {"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_reservation_details"
    assert guarded.kwargs == {"reservation_id": "ZFA04Y"}


def test_tau_policy_guard_requests_missing_required_enum_after_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_pending_order", kwargs={"order_id": "#W1234567"})
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "request_missing_required_arguments"
    assert info["missing_arguments"] == ["reason"]
    assert guarded.name == "respond"
    assert "no longer needed" in guarded.kwargs["content"]


def test_tau_policy_guard_maps_unavailable_to_receive_cancel_reason(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_pending_order", kwargs={"order_id": "#W1234567"})
    messages = [
        {"role": "user", "content": "Cancel order #W1234567 because I will be traveling and won't be able to receive it."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "execute_argument_repair"
    assert info["proposed_action"]["kwargs"]["reason"] == "no longer needed"
    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W1234567", "reason": "no longer needed"}


def test_tau_policy_guard_repairs_enum_from_user_answer_not_assistant_prompt(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="cancel_pending_order", kwargs={"order_id": "#W1234567"})
    messages = [
        {"role": "user", "content": "Cancel order #W1234567."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "a@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": '{"user_id":"u1"}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W1234567"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W1234567","status":"pending"}'},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide the reason: no longer needed or ordered by mistake."}}},
        {"role": "user", "content": "ordered by mistake"},
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "cancel_pending_order"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "execute_argument_repair"
    assert info["proposed_action"]["kwargs"]["reason"] == "ordered by mistake"
    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W1234567", "reason": "ordered by mistake"}


def test_tau_policy_guard_looks_up_product_before_item_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["8896479688"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {"role": "user", "content": "Please exchange the black laser mouse for a white optical one."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": '{"order_id":"#W7273336","status":"delivered","items":[{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser"}}]}',
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_product_before_item_mutation"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "5713490933"}


def test_tau_policy_guard_gets_user_orders_after_failed_order_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "#W727336"})
    messages = [
        {"role": "user", "content": "My email is omar.lopez1868@example.com. I need order #W727336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "omar.lopez1868@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "omar_lopez_3107"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W727336"}}},
        {"role": "tool", "name": "get_order_details", "content": "Error: order not found"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "get_user_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_user_orders_after_failed_order_lookup"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "omar_lopez_3107"}


def test_tau_policy_guard_repairs_near_miss_order_id_from_user_orders(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "#W727336"})
    messages = [
        {"role": "user", "content": "My email is omar.lopez1868@example.com. I need order #W727336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "omar.lopez1868@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "omar_lopez_3107"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W727336"}}},
        {"role": "tool", "name": "get_order_details", "content": "Error: order not found"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "omar_lopez_3107"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"user_id":"omar_lopez_3107","orders":["#W7273336","#W7073860","#W1764038"]}',
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "get_user_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_order_id_from_user_orders"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W7273336"}
    assert info["order_id_repair"] == {"order_id": {"from": "#W727336", "to": "#W7273336"}}


def test_tau_policy_guard_repairs_non_benchmark_order_identifier_shape(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "ORD-19A4"})
    messages = [
        {"role": "user", "content": "Please check order ORD-19A4."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "ORD-19A4"}}},
        {"role": "tool", "name": "get_order_details", "content": "Error: order not found"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "user_123"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"user_id":"user_123","orders":["ORD-1984","INV-7777"]}',
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_email", "get_order_details", "get_user_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_order_id_from_user_orders"
    assert guarded.kwargs == {"order_id": "ORD-1984"}


def test_tau_policy_guard_repairs_item_variants_from_observed_catalog(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393", "8018699955"],
            "new_item_ids": ["8896479688", "8895454203"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange my black laser mouse for a white optical one; wired or wireless is fine, "
                "whichever is cheaper. Also replace the 4-feet metal bookshelf with a 5-feet glass bookshelf "
                "in brown color."
            ),
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}}},
        {"role": "user", "content": "Yes, please proceed."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22},'
                '"8214883393":{"item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"},"available":true,"price":150.58}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01},'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65},'
                '"8018699955":{"item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"},"available":true,"price":467.86}'
                '}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_item_variant_selection"
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs["new_item_ids"] == ["2880340443", "4894369688"]
    assert info["item_variant_repair"]["new_item_ids"] == {
        "8214883393": "2880340443",
        "8018699955": "4894369688",
    }


def test_tau_policy_guard_fetches_missing_product_before_one_shot_subset(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["2880340443"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one and also exchange "
                "the 4-feet metal bookshelf for a 5-feet glass bookshelf."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_product_before_one_shot_batch_completion"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "8600330539"}


def test_tau_policy_guard_completes_one_shot_item_subset(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["2880340443"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one and also exchange "
                "the 4-feet metal bookshelf for a 5-feet glass bookshelf."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8479046075":{"item_id":"8479046075","options":{"material":"wood","color":"white","height":"5 ft"},"available":true,"price":451.01},'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65}'
                '}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "complete_one_shot_item_batch"
    assert info["item_batch_repair"]["reason"] == "complete_one_shot_item_batch"
    assert info["proposed_action"]["kwargs"]["item_ids"] == ["8214883393", "8018699955"]
    assert info["proposed_action"]["kwargs"]["new_item_ids"] == ["2880340443", "8895454203"]
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs["item_ids"] == ["8214883393", "8018699955"]
    assert guarded.kwargs["new_item_ids"] == ["2880340443", "8895454203"]


def test_tau_policy_guard_blocks_one_shot_when_options_still_requested_with_catalog(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393", "8018699955"],
            "new_item_ids": ["2880340443", "8895454203"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one and the 4-feet metal "
                "bookshelf for a 5-feet glass bookshelf. What color options are available for the "
                "bookshelf? I'm not sure which one would be best."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65},'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01}'
                '}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "answer_unresolved_item_option_question"
    assert guarded.name == "respond"
    assert "Bookshelf" in guarded.kwargs["content"]
    assert "color options" in guarded.kwargs["content"]
    assert "brown" in guarded.kwargs["content"]
    assert "white" in guarded.kwargs["content"]


def test_variant_selector_keeps_pronoun_continuation_constraints(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "8479046075": {
                "item_id": "8479046075",
                "options": {"material": "wood", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 451.01,
            },
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "One is the black laser gaming mouse. And also, I ordered a 4 feet metal bookshelf, "
                "but it's too short for where I want to place it in my room. Can I exchange it for "
                "a taller one made of glass? Specifically, I'm looking at a 5-feet tall bookshelf."
            ),
        }
    ]

    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_respects_soft_option_preferences(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the 4-feet metal bookshelf for a 5-feet glass bookshelf. "
                "Any available color is okay, preferably brown."
            ),
        }
    ]

    assert module.positive_option_value_mentioned(messages[0]["content"], "brown") is True
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_respects_something_like_preference(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "4900661478": {
                "item_id": "4900661478",
                "options": {"material": "glass", "color": "black", "height": "5 ft"},
                "available": True,
                "price": 480.00,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "I want a taller 5-feet glass bookshelf and would prefer something like brown if it is an option.",
        }
    ]

    assert module.positive_option_value_mentioned(messages[0]["content"], "brown") is True
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_preserves_unspecified_existing_options(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "4900661478": {
                "item_id": "4900661478",
                "options": {"material": "glass", "color": "black", "height": "5 ft"},
                "available": True,
                "price": 480.00,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Please exchange the metal bookshelf for a 5-feet glass bookshelf.",
        }
    ]

    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_surfaces_unconstrained_option_before_one_shot_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Gaming Mouse",
        "product_id": "5713490933",
        "item_id": "8214883393",
        "options": {"color": "black", "sensor type": "laser", "connectivity": "wireless"},
    }
    product = {
        "name": "Gaming Mouse",
        "product_id": "5713490933",
        "variants": {
            "8896479688": {
                "item_id": "8896479688",
                "options": {"color": "white", "sensor type": "optical", "connectivity": "wireless"},
                "available": True,
                "price": 143.15,
            },
            "2880340443": {
                "item_id": "2880340443",
                "options": {"color": "white", "sensor type": "optical", "connectivity": "wired"},
                "available": True,
                "price": 137.22,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Please exchange my black laser mouse for a white optical mouse instead.",
        }
    ]

    assert module.ambiguous_unconstrained_option_keys(old_item, product, messages) == ["connectivity"]


def test_unresolved_item_option_response_surfaces_open_ended_variant_choice(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": "Please exchange my black laser mouse for a white optical mouse instead.",
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393",'
                '"options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                "]}"
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                "}}"
            ),
        },
    ]

    response = module.unresolved_item_option_response(action, messages)

    assert response is not None
    assert response.name == module.RESPOND_ACTION_NAME
    content = response.kwargs[module.RESPOND_ACTION_FIELD_NAME]
    assert "connectivity options" in content
    assert "wired" in content
    assert "wireless" in content


def test_tau_policy_guard_answers_grounded_variant_price_question(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    assert module.price_question_seen("Which one is cheaper between the wired and wireless mouse?") is True
    assert module.price_question_seen("I'll go with the white optical wired mouse since it's the cheapest option.") is False
    messages = [
        {
            "role": "user",
            "content": "Please exchange my black laser mouse for a white optical mouse instead.",
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393",'
                '"options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                "]}"
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                "}}"
            ),
        },
        {
            "role": "user",
            "content": "Which one is cheaper between the wired and wireless white optical mouse?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "answer_item_price_question"
    assert guarded.name == module.RESPOND_ACTION_NAME
    content = guarded.kwargs[module.RESPOND_ACTION_FIELD_NAME]
    assert "wired" in content
    assert "$137.22" in content
    assert "cheapest" in content.lower()


def test_variant_selector_uses_item_local_final_choice_over_other_item_preferences(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
            "4900661478": {
                "item_id": "4900661478",
                "options": {"material": "glass", "color": "black", "height": "5 ft"},
                "available": True,
                "price": 463.04,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "I received a black laser gaming mouse and a metal bookshelf. The mouse color doesn't "
                "match my setup, so I want to get a white optical mouse instead, and the 4-feet tall "
                "metal bookshelf is too short. I want something taller made of glass, ideally 5 feet "
                "in height. What colors are available for the glass bookshelf?"
            ),
        },
        {
            "role": "user",
            "content": (
                "Great, I would like the white optical mouse. For the bookshelf, since brown is "
                "available, let's go ahead with a brown glass bookshelf."
            ),
        },
        {
            "role": "user",
            "content": (
                "For the mouse, wired or wireless is fine as long as it is cheaper. For the bookshelf, "
                "please switch it to a glass model that is 5 feet tall and brown in color."
            ),
        },
    ]

    assert module.price_preference_seen(messages, module.item_context_text(old_item, product, messages)) is False
    assert module.desired_option_constraints(old_item, product, messages)["color"] == {"brown"}
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_does_not_treat_option_list_query_as_color_choice(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
            "4900661478": {
                "item_id": "4900661478",
                "options": {"material": "glass", "color": "black", "height": "5 ft"},
                "available": True,
                "price": 463.04,
            },
        },
    }
    option_query = (
        "For the bookshelf, can you confirm if there are any glass material bookshelves "
        "in black and brown options?"
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the 4-feet metal bookshelf for a 5-feet glass bookshelf. "
                "What color options are available?"
            ),
        },
        {"role": "user", "content": option_query},
        {
            "role": "user",
            "content": (
                "It seems like the exact setup I want is available now. Could you confirm if there's "
                "a 5-foot tall glass bookshelf in brown? If so, let's go ahead with that."
            ),
        },
    ]

    assert module.positive_option_value_mentioned(option_query, "black") is False
    assert module.desired_option_constraints(old_item, product, messages)["color"] == {"brown"}
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_excludes_other_order_item_continuation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "user",
            "content": (
                "I received a black laser gaming mouse and a 4-feet metal bookshelf, but I'd like "
                "to exchange the mouse for a white optical one, whichever is cheaper. Also, can you "
                "help me find a taller glass bookshelf? Ideally, I want something 5-feet tall."
            ),
        },
    ]

    context = module.item_context_text(old_item, product, messages)
    assert "whichever is cheaper" not in context
    assert module.price_preference_seen(messages, context) is False
    assert module.desired_option_constraints(old_item, product, messages).get("color") is None
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_tau_policy_guard_asks_unresolved_item_choice_before_one_shot_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393", "8018699955"],
            "new_item_ids": ["8896479688", "8895454203"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange my black laser mouse for a white optical one and swap the bookshelf "
                "for a 5-feet glass bookshelf. What color options are available?"
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65},'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01},'
                '"4900661478":{"item_id":"4900661478","options":{"material":"glass","color":"black","height":"5 ft"},"available":true,"price":463.04}'
                '}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "For Bookshelf, available color options are: black, brown, white."},
            },
        },
        {
            "role": "user",
            "content": (
                "That's great to know! I'm leaning towards the brown option. "
                "Could we go ahead with exchanging my items as requested then?"
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "answer_unresolved_item_option_question"
    assert guarded.name == module.RESPOND_ACTION_NAME
    assert "Gaming Mouse" in guarded.kwargs[module.RESPOND_ACTION_FIELD_NAME]
    assert "connectivity options" in guarded.kwargs[module.RESPOND_ACTION_FIELD_NAME]


def test_tau_policy_guard_resolves_deferred_item_choice_after_price_preference(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393", "8018699955"],
            "new_item_ids": ["8896479688", "8895454203"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange my black laser mouse for a white optical one and swap the bookshelf "
                "for a 5-feet glass bookshelf. What color options are available?"
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65},'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01},'
                '"4900661478":{"item_id":"4900661478","options":{"material":"glass","color":"black","height":"5 ft"},"available":true,"price":463.04}'
                '}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "For Bookshelf, available color options are: black, brown, white."},
            },
        },
        {
            "role": "user",
            "content": (
                "I've decided on the brown glass bookshelf. For the mouse, wired or wireless is fine; "
                "please use whichever white optical option is cheaper."
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_item_variant_selection"
    assert info["item_variant_repair"]["new_item_ids"] == {
        "8214883393": "2880340443",
        "8018699955": "4894369688",
    }
    assert info["repaired_action"]["kwargs"]["new_item_ids"] == ["2880340443", "4894369688"]
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs["new_item_ids"] == ["2880340443", "4894369688"]


def test_variant_selector_scopes_option_only_followup_to_recent_product(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the gaming mouse for a white optical one, whichever is cheaper. "
                "Also exchange the bookshelf for a 5 ft glass one; I need to pick a color."
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22},'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15}}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65},'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01}}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "For Bookshelf, available color options are: brown, white."},
            },
        },
        {
            "role": "user",
            "content": (
                "That works for the mouse. Can you proceed with the white optical wired one? "
                "And what about the bookshelf in brown glass? How does that sound?"
            ),
        },
    ]
    order_items = module.order_items_by_item_id(messages, "#W7273336")
    products = module.product_detail_payloads(messages)

    assert module.choose_variant_for_item(order_items["8214883393"], products["5713490933"], messages) == "2880340443"
    assert module.choose_variant_for_item(order_items["8018699955"], products["8600330539"], messages) == "4894369688"


def test_variant_selector_uses_price_when_user_is_indifferent_between_option_values(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Gaming Mouse",
        "product_id": "5713490933",
        "item_id": "8214883393",
        "options": {"color": "black", "sensor type": "laser", "connectivity": "wireless"},
    }
    product = {
        "name": "Gaming Mouse",
        "product_id": "5713490933",
        "variants": {
            "8896479688": {
                "item_id": "8896479688",
                "options": {"color": "white", "sensor type": "optical", "connectivity": "wireless"},
                "available": True,
                "price": 143.15,
            },
            "2880340443": {
                "item_id": "2880340443",
                "options": {"color": "white", "sensor type": "optical", "connectivity": "wired"},
                "available": True,
                "price": 137.22,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange my black laser mouse for a white optical mouse. "
                "Either wired or wireless is fine as long as it is white."
            ),
        }
    ]
    context = module.item_context_text(old_item, product, messages)

    assert module.option_values_indifferent(context, {"wired", "wireless"}) is True
    assert module.price_preference_seen(messages, context) is False
    assert module.choose_variant_for_item(old_item, product, messages) == "2880340443"


def test_tau_policy_guard_resumes_deferred_exchange_before_reanswering_options(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"user_id":"omar_lopez_3107","payment_methods":{"paypal_1530316":{"source":"paypal","id":"paypal_1530316"}}}',
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","user_id":"omar_lopez_3107",'
                '"payment_history":[{"payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01},'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65}}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "For Bookshelf, available color options are: brown, white."},
            },
            "tau_policy_guard": {
                "proposed_action": {
                    "name": "exchange_delivered_order_items",
                    "kwargs": {
                        "order_id": "#W7273336",
                        "item_ids": ["8214883393", "8018699955"],
                        "new_item_ids": ["8896479688", "4894369688"],
                        "payment_method_id": "paypal_1530316",
                    },
                }
            },
        },
        {
            "role": "user",
            "content": (
                "Thanks for confirming. I'd like to exchange the gaming mouse for a white optical one, "
                "either wired or wireless, whichever is cheaper. And could you please send me a "
                "5-feet tall glass bookshelf in brown? If that's not available, white would be okay too."
            ),
        },
    ]

    assert module.latest_user_requests_deferred_mutation_execution(messages) is True
    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_user_details", "get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "resume_deferred_repair_item_variant_selection"
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs["new_item_ids"] == ["2880340443", "4894369688"]


def test_requested_order_items_does_not_leak_terms_from_other_orders(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    order = {
        "order_id": "#W5442520",
        "items": [
            {
                "name": "Patio Umbrella",
                "item_id": "3111466194",
                "product_id": "9743693396",
                "options": {"size": "7 ft", "color": "red", "material": "polyester", "tilt mechanism": "manual tilt"},
            },
            {
                "name": "Hiking Boots",
                "item_id": "2648909398",
                "product_id": "7363354090",
                "options": {"size": "8", "material": "leather", "waterproof": "yes"},
            },
        ],
    }
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W5442520, change payment to PayPal and exchange Patio Umbrella "
                "from 7 ft red polyester manual tilt to 6 ft blue Sunbrella auto tilt."
            ),
        },
        {
            "role": "user",
            "content": (
                "For order #W7941031, exchange Wristwatch from a leather strap and white dial "
                "to silicone straps and blue dials."
            ),
        },
        {"role": "user", "content": "Can we proceed with these changes now?"},
    ]

    items = module.requested_order_items(messages, order)

    assert [item["item_id"] for item in items] == ["3111466194"]


def test_complete_one_shot_item_batch_prunes_unrequested_items(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["3111466194", "2648909398"],
            "new_item_ids": ["2001307871", "3812493782"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W5442520, exchange Patio Umbrella from 7 ft red polyester manual tilt "
                "to 6 ft blue Sunbrella auto tilt."
            ),
        },
        {"role": "user", "content": "For order #W7941031, exchange Wristwatch from leather strap to silicone strap."},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": json.dumps(
                {
                    "order_id": "#W5442520",
                    "status": "pending",
                    "items": [
                        {
                            "name": "Patio Umbrella",
                            "product_id": "9743693396",
                            "item_id": "3111466194",
                            "options": {
                                "size": "7 ft",
                                "color": "red",
                                "material": "polyester",
                                "tilt mechanism": "manual tilt",
                            },
                        },
                        {
                            "name": "Hiking Boots",
                            "product_id": "7363354090",
                            "item_id": "2648909398",
                            "options": {"size": "8", "material": "leather", "waterproof": "yes"},
                        },
                    ],
                }
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": json.dumps(
                {
                    "name": "Patio Umbrella",
                    "product_id": "9743693396",
                    "variants": {
                        "2001307871": {
                            "item_id": "2001307871",
                            "available": True,
                            "options": {
                                "size": "6 ft",
                                "color": "blue",
                                "material": "sunbrella",
                                "tilt mechanism": "auto tilt",
                            },
                        }
                    },
                }
            ),
        },
        {"role": "user", "content": "Can we proceed with these changes now?"},
    ]

    repaired, info = module.complete_one_shot_item_batch(action, messages, {"get_product_details"})

    assert info["reason"] == "prune_unrequested_one_shot_item_batch"
    assert repaired.kwargs["item_ids"] == ["3111466194"]
    assert repaired.kwargs["new_item_ids"] == ["2001307871"]


def test_unresolved_item_option_response_skips_rejected_item_topic(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["2648909398"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {"role": "user", "content": "Please update my order."},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": json.dumps(
                {
                    "order_id": "#W5442520",
                    "status": "pending",
                    "items": [
                        {
                            "name": "Hiking Boots",
                            "product_id": "7363354090",
                            "item_id": "2648909398",
                            "options": {"size": "8", "material": "leather", "waterproof": "yes"},
                        }
                    ],
                }
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": json.dumps(
                {
                    "name": "Hiking Boots",
                    "product_id": "7363354090",
                    "variants": {
                        "8106223139": {
                            "item_id": "8106223139",
                            "available": True,
                            "options": {"size": "9", "material": "leather", "waterproof": "yes"},
                        },
                        "4582956489": {
                            "item_id": "4582956489",
                            "available": True,
                            "options": {"size": "12", "material": "synthetic", "waterproof": "no"},
                        },
                    },
                }
            ),
        },
        {"role": "user", "content": "I don't need hiking boot information right now. Focus on the order updates."},
    ]

    assert module.unresolved_item_option_response(action, messages) is None


def test_variant_selector_avoids_negative_option_preferences(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "4900661478": {
                "item_id": "4900661478",
                "options": {"material": "glass", "color": "black", "height": "5 ft"},
                "available": True,
                "price": 480.00,
            },
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Please exchange the bookshelf for a 5 ft glass one, but not black.",
        }
    ]

    assert module.negative_option_value_mentioned(messages[0]["content"], "black") is True
    assert module.choose_variant_for_item(old_item, product, messages) == "8895454203"


def test_variant_selector_treats_availability_fallback_as_positive_preference(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "item_id": "8018699955",
        "options": {"material": "metal", "color": "brown", "height": "4 ft"},
    }
    product = {
        "name": "Bookshelf",
        "product_id": "8600330539",
        "variants": {
            "8895454203": {
                "item_id": "8895454203",
                "options": {"material": "glass", "color": "white", "height": "5 ft"},
                "available": True,
                "price": 504.65,
            },
            "4894369688": {
                "item_id": "4894369688",
                "options": {"material": "glass", "color": "brown", "height": "5 ft"},
                "available": True,
                "price": 537.01,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Please find a 5-feet tall glass bookshelf. Brown would be ideal if it's available; "
                "if brown isn't available, white might be okay as well."
            ),
        }
    ]

    assert module.positive_option_value_mentioned(messages[0]["content"], "brown") is True
    assert module.negative_option_value_mentioned(messages[0]["content"], "brown") is False
    assert module.choose_variant_for_item(old_item, product, messages) == "4894369688"


def test_variant_selector_treats_from_value_as_source_not_target(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Digital Camera",
        "product_id": "8940227892",
        "item_id": "5996159312",
        "options": {"resolution": "24MP", "zoom": "3x", "storage": "SD card"},
    }
    product = {
        "name": "Digital Camera",
        "product_id": "8940227892",
        "variants": {
            "6384525445": {
                "item_id": "6384525445",
                "options": {"resolution": "30MP", "zoom": "5x", "storage": "CF card"},
                "available": True,
                "price": 2929.62,
            },
            "7255224608": {
                "item_id": "7255224608",
                "options": {"resolution": "30MP", "zoom": "3x", "storage": "CF card"},
                "available": True,
                "price": 2922.97,
            },
        },
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Exchange my Digital Camera from 24MP resolution with 3x zoom and SD card storage "
                "to a 30MP model with 5x zoom and CF card storage."
            ),
        }
    ]

    assert module.source_option_value_mentioned(messages[0]["content"], "3x") is True
    assert module.choose_variant_for_item(old_item, product, messages) == "6384525445"


def test_variant_selector_preserves_unspecified_existing_option_without_prompt(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Espresso Machine",
        "product_id": "4354588079",
        "item_id": "9884666842",
        "options": {"pressure": "19 bar", "capacity": "1L", "type": "manual"},
    }
    product = {
        "name": "Espresso Machine",
        "product_id": "4354588079",
        "variants": {
            "3815173328": {
                "item_id": "3815173328",
                "options": {"pressure": "9 bar", "capacity": "1.5L", "type": "capsule"},
                "available": True,
                "price": 2908.42,
            },
            "7806008610": {
                "item_id": "7806008610",
                "options": {"pressure": "9 bar", "capacity": "1L", "type": "capsule"},
                "available": True,
                "price": 2742.67,
            },
        },
    }
    messages = [{"role": "user", "content": "Please switch my espresso machine to a 9 bar capsule model."}]

    assert module.ambiguous_unconstrained_option_keys(old_item, product, messages) == []
    assert module.choose_variant_for_item(old_item, product, messages) == "7806008610"


def test_variant_selector_preserves_option_when_user_rejects_changing_that_dimension(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    old_item = {
        "name": "Espresso Machine",
        "product_id": "4354588079",
        "item_id": "9884666842",
        "options": {"pressure": "19 bar", "capacity": "1L", "type": "manual"},
    }
    product = {
        "name": "Espresso Machine",
        "product_id": "4354588079",
        "variants": {
            "3815173328": {
                "item_id": "3815173328",
                "options": {"pressure": "9 bar", "capacity": "1.5L", "type": "capsule"},
                "available": True,
                "price": 2908.42,
            },
            "7806008610": {
                "item_id": "7806008610",
                "options": {"pressure": "9 bar", "capacity": "1L", "type": "capsule"},
                "available": True,
                "price": 2742.67,
            },
        },
    }
    messages = [
        {"role": "user", "content": "Please switch my espresso machine to a 9 bar capsule model."},
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "For Espresso Machine, available capacity options are: 1.5L, 1L, 2L."}},
        },
        {"role": "user", "content": "I'm not looking for a different capacity, just a 9 bar capsule model."},
    ]

    assert module.desired_option_constraints(old_item, product, messages)["capacity"] == {"1l"}
    assert module.ambiguous_unconstrained_option_keys(old_item, product, messages) == []
    assert module.choose_variant_for_item(old_item, product, messages) == "7806008610"


def test_catalog_option_response_ignores_final_option_choice(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {"role": "user", "content": "Please exchange the bookshelf for a 5 ft glass one."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955",'
                '"options":{"material":"metal","color":"brown","height":"4 ft"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true}}}'
            ),
        },
        {"role": "user", "content": "The brown glass bookshelf is perfect. Let's go with that option."},
    ]

    assert module.catalog_option_response(messages) is None


def test_catalog_option_response_ignores_final_choice_with_generic_what(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {"role": "user", "content": "Please exchange the bookshelf for a 5 ft glass one and tell me color options."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955",'
                '"options":{"material":"metal","color":"brown","height":"4 ft"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true},'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true}}}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Based on what you've provided, I've decided on getting the 5-feet tall "
                "glass bookshelf in brown. Can we proceed with the exchange?"
            ),
        },
    ]

    assert module.catalog_option_response(messages) is None


def test_tau_policy_guard_answers_unresolved_item_option_before_one_shot_mutation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8018699955"],
            "new_item_ids": ["8895454203"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W7273336, exchange my metal bookshelf for a 5 ft glass one. "
                "What color options are available for the glass bookshelf?"
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true},'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true},'
                '"4900661478":{"item_id":"4900661478","options":{"material":"glass","color":"black","height":"5 ft"},"available":true}'
                '}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "answer_unresolved_item_option_question"
    assert guarded.name == "respond"
    assert "brown" in guarded.kwargs["content"]
    assert "white" in guarded.kwargs["content"]


def test_tau_policy_guard_fetches_missing_product_before_partial_option_answer(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one and the metal bookshelf "
                "for a 5-feet glass bookshelf. What color options are available for the glass bookshelf?"
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true}}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "lookup_product_before_item_option_answer"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "8600330539"}


def test_tau_policy_guard_proposes_item_exchange_product_lookup_from_observations(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "I want to exchange two items from order #W7273336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one, whichever is cheaper, "
                "and exchange the 4-feet metal bookshelf for a brown 5-feet glass bookshelf."
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "5713490933"}


def test_tau_policy_guard_scans_other_orders_when_requested_exchange_item_not_in_current_order(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "I need to exchange my recent espresso machine order for a 9 bar capsule model.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Omar", "last_name": "Silva", "zip": "92107"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "omar_silva_7446"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "omar_silva_7446"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"user_id":"omar_silva_7446","orders":["#W9728773","#W9673784","#W1216601"]}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9728773"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W9728773","status":"pending",'
                '"items":[{"name":"Jigsaw Puzzle","product_id":"1808611083","item_id":"7127170374"}],'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"gift_card_5540683"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W9673784"}


def test_tau_policy_guard_proposes_item_exchange_for_plural_exchange_intent(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": (
                "I'd like to make exchanges for a couple items in order #W7273336: "
                "the black laser mouse for a white optical one, and the metal bookshelf "
                "for a 5-feet tall glass version."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "5713490933"}


def test_tau_policy_guard_avoids_repaired_duplicate_order_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(name="get_order_details", kwargs={"order_id": "#W727336"})
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W727336, I'd like to make exchanges: the black laser mouse "
                "for a white optical one, and the metal bookshelf for a glass one."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"].startswith("avoid_duplicate_order_lookup")
    assert info["order_id_repair"]["order_id"] == {"from": "#W727336", "to": "#W7273336"}
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "5713490933"}


def test_tau_policy_guard_proposes_final_item_exchange_from_observations(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "I want to exchange two items from order #W7273336."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "person@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "user_123"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered","payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955","options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393","options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Please exchange the black laser mouse for a white optical one, whichever is cheaper, "
                "and exchange the 4-feet metal bookshelf for a brown 5-feet glass bookshelf."
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"8896479688":{"item_id":"8896479688","options":{"color":"white","sensor type":"optical","connectivity":"wireless"},"available":true,"price":143.15},'
                '"2880340443":{"item_id":"2880340443","options":{"color":"white","sensor type":"optical","connectivity":"wired"},"available":true,"price":137.22}'
                '}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Bookshelf","product_id":"8600330539","variants":{'
                '"4894369688":{"item_id":"4894369688","options":{"material":"glass","color":"brown","height":"5 ft"},"available":true,"price":537.01},'
                '"8895454203":{"item_id":"8895454203","options":{"material":"glass","color":"white","height":"5 ft"},"available":true,"price":504.65}'
                '}}'
            ),
        },
        {"role": "user", "content": "Yes, proceed with exchanging both items accordingly."},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_resolve_item_variant_selection"
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs == {
        "order_id": "#W7273336",
        "item_ids": ["8214883393", "8018699955"],
        "payment_method_id": "paypal_1530316",
        "new_item_ids": ["2880340443", "4894369688"],
    }


def test_tau_policy_guard_responds_after_completed_mutation_reconfirmed(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please exchange the mouse and bookshelf."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393", "8018699955"],
                    "new_item_ids": ["2880340443", "4894369688"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {
            "role": "tool",
            "name": "exchange_delivered_order_items",
            "content": '{"order_id":"#W7273336","status":"exchange requested"}',
        },
        {"role": "user", "content": "Great, can you proceed with these exchanges?"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "respond_after_prior_successful_mutation"
    assert guarded.name == "respond"
    assert "completed" in guarded.kwargs["content"]


def test_tau_policy_guard_responds_after_completed_mutation_followup_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please exchange the mouse and bookshelf."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393", "8018699955"],
                    "new_item_ids": ["2880340443", "4894369688"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {
            "role": "tool",
            "name": "exchange_delivered_order_items",
            "content": '{"order_id":"#W7273336","status":"exchange requested"}',
        },
        {"role": "user", "content": "Thanks, I would like to exchange the mouse and bookshelf as discussed."},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "respond_after_prior_successful_mutation"
    assert guarded.name == "respond"
    assert "completed" in guarded.kwargs["content"]


def test_tau_policy_guard_does_not_treat_new_order_request_as_completed_followup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please update order #W5442520."},
        {
            "role": "assistant",
            "action": {
                "name": "modify_pending_order_items",
                "kwargs": {
                    "order_id": "#W5442520",
                    "item_ids": ["3111466194"],
                    "new_item_ids": ["2001307871"],
                    "payment_method_id": "paypal_8049766",
                },
            },
        },
        {
            "role": "tool",
            "name": "modify_pending_order_items",
            "content": '{"order_id":"#W5442520","status":"pending"}',
        },
        {"role": "user", "content": "Great. Now can you change order #W7941031 too?"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W7941031"}


def test_preflight_after_success_advances_to_next_requested_order(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    agent = module.GoalGraphAgent.__new__(module.GoalGraphAgent)
    agent.preflight_enabled = True
    agent.legacy_tau_heuristics = True
    agent.available_tool_names = {"get_order_details", "modify_pending_order_items"}
    agent.tool_schemas = {}
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W5442520, exchange the Patio Umbrella. "
                "For order #W7941031, exchange the Wristwatch."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W5442520","status":"pending"}'},
        {
            "role": "assistant",
            "action": {
                "name": "modify_pending_order_items",
                "kwargs": {
                    "order_id": "#W5442520",
                    "item_ids": ["3111466194"],
                    "new_item_ids": ["2001307871"],
                    "payment_method_id": "paypal_8049766",
                },
            },
        },
        {"role": "tool", "name": "modify_pending_order_items", "content": '{"order_id":"#W5442520","status":"pending"}'},
    ]

    action, result = agent.preflight_action(messages)

    assert action.name == "get_order_details"
    assert action.kwargs == {"order_id": "#W7941031"}
    assert result["tau_policy_guard"]["reason"] in {
        "continue_after_success_lookup_next_requested_order",
        "continue_after_success_transcript_proposal",
    }


def test_tau_policy_guard_does_not_reopen_item_options_after_completed_exchange(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please exchange the mouse and bookshelf."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393", "8018699955"],
                    "new_item_ids": ["2880340443", "4894369688"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {
            "role": "tool",
            "name": "exchange_delivered_order_items",
            "content": '{"order_id":"#W7273336","status":"exchange requested"}',
        },
        {
            "role": "user",
            "content": "Could you also check what color options are available for the 5-feet glass bookshelf?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "avoid_option_exploration_after_completed_item_mutation"
    assert guarded.name == "respond"
    assert "already" in guarded.kwargs["content"].lower()
    assert "completed" in guarded.kwargs["content"].lower()


def test_deferred_return_proposal_is_satisfied_by_completed_exchange(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {"role": "user", "content": "Please exchange the mouse and bookshelf."},
        {
            "role": "assistant",
            "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}},
            "goal_graph": {
                "tau_policy_guard": {
                    "used": True,
                    "reason": "redirect_return_to_exchange_intent_confirm_before_mutation",
                    "proposed_action": {
                        "name": "return_delivered_order_items",
                        "kwargs": {
                            "order_id": "#W7273336",
                            "item_ids": ["8214883393", "8018699955"],
                            "payment_method_id": "paypal_1530316",
                        },
                    },
                }
            },
        },
        {"role": "user", "content": "Yes, proceed."},
        {
            "role": "assistant",
            "action": {
                "name": "exchange_delivered_order_items",
                "kwargs": {
                    "order_id": "#W7273336",
                    "item_ids": ["8214883393", "8018699955"],
                    "new_item_ids": ["2880340443", "4894369688"],
                    "payment_method_id": "paypal_1530316",
                },
            },
        },
        {
            "role": "tool",
            "name": "exchange_delivered_order_items",
            "content": '{"order_id":"#W7273336","status":"exchange requested"}',
        },
    ]

    assert module.latest_deferred_mutating_action(messages) is None


def test_tau_policy_guard_answers_return_logistics_after_completed_mutation_before_catalog_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please return the Skateboard and Desk Lamp from order #W3814930."},
        {
            "role": "assistant",
            "action": {
                "name": "return_delivered_order_items",
                "kwargs": {
                    "order_id": "#W3814930",
                    "item_ids": ["5324287154", "5489837791"],
                    "payment_method_id": "credit_card_9753331",
                },
            },
        },
        {
            "role": "tool",
            "name": "return_delivered_order_items",
            "content": '{"order_id":"#W3814930","status":"return requested"}',
        },
        {
            "role": "user",
            "content": "What are the next steps for mailing back the Skateboard and Desk Lamp, and when will the refund be processed?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_product_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "respond_after_prior_successful_mutation"
    assert guarded.name == "respond"
    assert "completed" in guarded.kwargs["content"]


def test_tau_policy_guard_confirms_before_broad_return_and_scopes_to_requested_item(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W4887592",
            "item_ids": ["2343503231", "9385662952", "4447749792"],
            "payment_method_id": "paypal_1249653",
        },
    )
    messages = [
        {
            "role": "user",
            "content": "I want to return the Skateboard from order #W4887592 for a refund.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "mohamed_khan_3010"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "mohamed_khan_3010"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"user_id":"mohamed_khan_3010","payment_methods":{"paypal_1249653":{"source":"paypal","id":"paypal_1249653"}}}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W4887592"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W4887592","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1249653"}],'
                '"items":['
                '{"name":"Skateboard","product_id":"1968349452","item_id":"2343503231"},'
                '{"name":"Desk Lamp","product_id":"6817146515","item_id":"4447749792"},'
                '{"name":"Fleece Jacket","product_id":"8560156827","item_id":"9385662952"}'
                ']}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_user_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_single_list_item_mutation_scope"
    assert info["item_scope_repair"]["item_ids"] == {
        "from": ["2343503231", "9385662952", "4447749792"],
        "to": ["2343503231"],
    }
    assert info["proposed_action"]["kwargs"]["item_ids"] == ["2343503231"]
    assert guarded.name == "return_delivered_order_items"
    assert guarded.kwargs["item_ids"] == ["2343503231"]


def test_requested_return_items_ignores_full_refund_as_jacket_option(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "I bought a skateboard recently for around $200 but found it cheaper elsewhere. "
                "Can I return it and get a full refund? Also, I want to return a desk lamp "
                "from the same order."
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W4887592","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1249653"}],'
                '"items":['
                '{"name":"Skateboard","product_id":"1968349452","item_id":"2343503231",'
                '"options":{"deck material":"maple","length":"34 inch","design":"graphic"}},'
                '{"name":"Desk Lamp","product_id":"6817146515","item_id":"4447749792",'
                '"options":{"color":"white","brightness":"medium","power source":"AC adapter"}},'
                '{"name":"Fleece Jacket","product_id":"8560156827","item_id":"9385662952",'
                '"options":{"size":"L","color":"black","zipper":"full"}}'
                ']}'
            ),
        },
    ]
    order = module.latest_order_payload(messages, "#W4887592")

    assert [(item["name"], item["item_id"]) for item in module.requested_order_items(messages, order)] == [
        ("Skateboard", "2343503231"),
        ("Desk Lamp", "4447749792"),
    ]


def test_tau_policy_guard_redirects_return_action_when_exchange_intent_is_present(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393", "8018699955"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Please exchange the black laser gaming mouse for a white optical one and return "
                "the 4-feet metal bookshelf in favor of a taller 5-feet glass bookshelf."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],'
                '"items":['
                '{"name":"Bookshelf","product_id":"8600330539","item_id":"8018699955",'
                '"options":{"material":"metal","color":"brown","height":"4 ft"}},'
                '{"name":"Gaming Mouse","product_id":"5713490933","item_id":"8214883393",'
                '"options":{"color":"black","sensor type":"laser","connectivity":"wireless"}}'
                ']}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"].startswith("redirect_return_to_exchange_intent_")
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "5713490933"}


def test_tau_policy_guard_adds_item_from_additive_return_confirmation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W4887592",
            "item_ids": ["2343503231"],
            "payment_method_id": "paypal_1249653",
        },
    )
    messages = [
        {
            "role": "user",
            "content": "I want to return the Skateboard from order #W4887592 for a refund.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Mohamed", "last_name": "Khan", "zip": "60651"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "mohamed_khan_3010"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "mohamed_khan_3010"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"user_id":"mohamed_khan_3010","payment_methods":{"paypal_1249653":{"source":"paypal","id":"paypal_1249653"}}}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W4887592"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W4887592","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1249653"}],'
                '"items":['
                '{"name":"Skateboard","product_id":"1968349452","item_id":"2343503231"},'
                '{"name":"Desk Lamp","product_id":"6817146515","item_id":"4447749792"},'
                '{"name":"Fleece Jacket","product_id":"8560156827","item_id":"9385662952"}'
                ']}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "I can return the Skateboard from order #W4887592. Please confirm with yes."},
            },
            "goal_graph": {
                "tau_policy_guard": {
                    "proposed_action": {
                        "name": "return_delivered_order_items",
                        "kwargs": {
                            "order_id": "#W4887592",
                            "item_ids": ["2343503231"],
                            "payment_method_id": "paypal_1249653",
                        },
                    }
                }
            },
        },
        {"role": "user", "content": "Yes, and also return the Desk Lamp from the same order."},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_user_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_single_list_item_mutation_scope"
    assert guarded.name == "return_delivered_order_items"
    assert guarded.kwargs == {
        "order_id": "#W4887592",
        "item_ids": ["2343503231", "4447749792"],
        "payment_method_id": "paypal_1249653",
    }


def test_tau_policy_guard_requires_confirmation_for_direct_grounded_exchange_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["2880340443"],
            "payment_method_id": "paypal_1530316",
        },
    )
    messages = [
        {
            "role": "user",
            "content": "Please exchange the black laser mouse in order #W7273336 for a white optical mouse.",
        },
        {
            "role": "assistant",
            "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7273336"}},
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7273336","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1530316"}],'
                '"items":[{"name":"Gaming Mouse","product_id":"5713490933",'
                '"item_id":"8214883393","options":{"color":"black","sensor type":"laser"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Gaming Mouse","product_id":"5713490933","variants":{'
                '"2880340443":{"item_id":"2880340443",'
                '"options":{"color":"white","sensor type":"optical"},"available":true,"price":137.22}}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "exchange_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "resolve_item_variant_selection"
    assert info["proposed_action"] == proposed.model_dump()
    assert guarded.name == "exchange_delivered_order_items"
    assert guarded.kwargs == proposed.kwargs


def test_tau_policy_guard_proposes_pending_item_modify_from_latest_order(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please update payment for order #W5442520 to PayPal."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396",'
                '"item_id":"3111466194","options":{"size":"7 ft","color":"red"}}]}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Yes, please go ahead with those changes for order #W5442520. Once that's done, "
                "could you also update the payment method for order #W7941031 to PayPal under "
                "paypal_8049766 and exchange the Wristwatch leather strap white dial for one "
                "with a silicone strap and blue dial? However, I prefer to pay or get refunded "
                "by credit card for this last request."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7941031"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7941031","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Wristwatch","product_id":"6066914160",'
                '"item_id":"1355937109","options":{"strap material":"leather","dial color":"white"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Wristwatch","product_id":"6066914160","variants":{'
                '"2226219750":{"item_id":"2226219750","options":{"strap material":"silicone","dial color":"white"},"available":true,"price":2009.03},'
                '"8886009523":{"item_id":"8886009523","options":{"strap material":"silicone","dial color":"blue"},"available":true,"price":1944.02}}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "modify_pending_order_items", "get_user_details"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_resolve_item_variant_selection"
    assert guarded.name == "modify_pending_order_items"
    assert guarded.kwargs == {
        "order_id": "#W7941031",
        "item_ids": ["1355937109"],
        "payment_method_id": "credit_card_9753331",
        "new_item_ids": ["8886009523"],
    }


def test_tau_policy_guard_updates_payment_before_pending_item_modify(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["3111466194"],
            "new_item_ids": ["2001307871"],
            "payment_method_id": "credit_card_9753331",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "After updating the payment method for #W5442520 to PayPal, exchange the Patio Umbrella "
                "from 7 ft red polyester manual tilt to 6 ft blue sunbrella auto tilt."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending","user_id":"olivia_ito_3591",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Patio Umbrella","product_id":"9743693396","variants":{'
                '"2001307871":{"item_id":"2001307871","options":{"size":"6 ft","color":"blue","material":"sunbrella","tilt mechanism":"auto tilt"},"available":true,"price":302.63}}}'
            ),
        },
        {"role": "user", "content": "Thanks, my zip is 80218."},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "get_user_details", "modify_pending_order_payment", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "update_payment_before_pending_item_mutation"
    assert guarded.name == "modify_pending_order_payment"
    assert guarded.kwargs == {"order_id": "#W5442520", "payment_method_id": "paypal_8049766"}


def test_item_mutation_payment_method_is_scoped_to_current_order_text(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    order = {
        "order_id": "#W5442520",
        "status": "pending",
        "payment_history": [{"transaction_type": "payment", "payment_method_id": "credit_card_9753331"}],
        "items": [{"name": "Patio Umbrella", "item_id": "3111466194"}],
    }
    messages = [
        {
            "role": "user",
            "content": (
                "For #W5442520, change payment to PayPal and exchange the Patio Umbrella. "
                "For #W7941031, use my credit card for payment or refund."
            ),
        },
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
    ]

    assert module.item_mutation_payment_method_id(messages, order) == "paypal_8049766"


def test_pending_refund_alternative_does_not_become_cancel_for_payment_update(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W7941031, can the payment method be changed to my credit card "
                "instead of PayPal? If it is complicated, a refund would also work."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7941031"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7941031","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Wristwatch","item_id":"1355937109"}]}'
            ),
        },
    ]

    assert module.propose_pending_refund_cancel_from_observations(
        messages,
        {"cancel_pending_order"},
        {"cancel_pending_order": {"type": "object", "properties": {"reason": {"enum": ["no longer needed", "ordered by mistake"]}}}},
    ) is None


def test_refund_mention_from_other_order_does_not_cancel_current_modify_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "For #W7941031, a refund would also work if needed. "
                "For #W3657213, change payment to credit_card_9753331 and exchange the Digital Camera "
                "from 24MP 3x SD card to 30MP 5x CF card."
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W3657213"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W3657213","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"gift_card_7794233"}],'
                '"items":[{"name":"Digital Camera","product_id":"8940227892","item_id":"5996159312",'
                '"options":{"resolution":"24MP","zoom":"3x","storage":"SD card"}}]}'
            ),
        },
    ]

    assert module.propose_pending_refund_cancel_from_observations(
        messages,
        {"cancel_pending_order"},
        {"cancel_pending_order": {"type": "object", "properties": {"reason": {"enum": ["no longer needed", "ordered by mistake"]}}}},
    ) is None


def test_tau_policy_guard_repairs_stale_item_payment_after_pending_payment_update(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["3111466194"],
            "new_item_ids": ["2001307871"],
            "payment_method_id": "credit_card_9753331",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "After updating the payment method for #W5442520 to PayPal, exchange the Patio Umbrella "
                "from 7 ft red polyester manual tilt to 6 ft blue sunbrella auto tilt."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "modify_pending_order_payment", "kwargs": {"order_id": "#W5442520", "payment_method_id": "paypal_8049766"}}},
        {
            "role": "tool",
            "name": "modify_pending_order_payment",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":['
                '{"transaction_type":"payment","payment_method_id":"credit_card_9753331"},'
                '{"transaction_type":"payment","payment_method_id":"paypal_8049766"},'
                '{"transaction_type":"refund","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Patio Umbrella","product_id":"9743693396","variants":{'
                '"2001307871":{"item_id":"2001307871","options":{"size":"6 ft","color":"blue","material":"sunbrella","tilt mechanism":"auto tilt"},"available":true,"price":302.63}}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "get_user_details", "modify_pending_order_payment", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "confirm_before_mutation"
    assert guarded.name == "respond"
    assert info["proposed_action"]["kwargs"]["payment_method_id"] == "paypal_8049766"
    assert info["item_payment_repair"]["payment_method_id"] == {
        "from": "credit_card_9753331",
        "to": "paypal_8049766",
    }


def test_tau_policy_guard_asks_payment_method_before_gift_card_pending_item_modify(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W9673784",
            "item_ids": ["9884666842"],
            "new_item_ids": ["7806008610"],
            "payment_method_id": "gift_card_5540683",
        },
    )
    messages = [
        {"role": "user", "content": "Please exchange my espresso machine for a 9 bar capsule model."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Omar", "last_name": "Silva", "zip": "92107"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "omar_silva_7446"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "omar_silva_7446"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"omar_silva_7446","payment_methods":{'
                '"paypal_2192303":{"source":"paypal","id":"paypal_2192303"},'
                '"gift_card_5540683":{"source":"gift_card","id":"gift_card_5540683"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9673784"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W9673784","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"gift_card_5540683"}],'
                '"items":[{"name":"Espresso Machine","product_id":"4354588079","item_id":"9884666842",'
                '"options":{"pressure":"19 bar","capacity":"1L","type":"manual"}}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_user_details", "modify_pending_order_items", "modify_pending_order_payment"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_payment_method_before_pending_item_mutation"
    assert guarded.name == "respond"
    assert "paypal_2192303" in guarded.kwargs["content"]
    assert "gift_card_5540683" in guarded.kwargs["content"]


def test_tau_policy_guard_repairs_pending_item_payment_preference_without_payment_update(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W9673784",
            "item_ids": ["9884666842"],
            "new_item_ids": ["7806008610"],
            "payment_method_id": "gift_card_5540683",
        },
    )
    messages = [
        {"role": "user", "content": "Please exchange my espresso machine for a 9 bar capsule model."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Omar", "last_name": "Silva", "zip": "92107"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "omar_silva_7446"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "omar_silva_7446"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"omar_silva_7446","payment_methods":{'
                '"paypal_2192303":{"source":"paypal","id":"paypal_2192303"},'
                '"gift_card_5540683":{"source":"gift_card","id":"gift_card_5540683"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9673784"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W9673784","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"gift_card_5540683"}],'
                '"items":[{"name":"Espresso Machine","product_id":"4354588079","item_id":"9884666842",'
                '"options":{"pressure":"19 bar","capacity":"1L","type":"manual"}}]}'
            ),
        },
        {
            "role": "user",
            "content": "I prefer to use PayPal if possible.",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_user_details", "modify_pending_order_items", "modify_pending_order_payment"},
    )

    assert info["used"] is True
    assert info["reason"] == "confirm_before_mutation"
    assert guarded.name == "respond"
    assert info["proposed_action"]["name"] == "modify_pending_order_items"
    assert info["proposed_action"]["kwargs"]["payment_method_id"] == "paypal_2192303"
    assert info["item_payment_repair"]["payment_method_id"] == {
        "from": "gift_card_5540683",
        "to": "paypal_2192303",
    }


def test_tau_policy_guard_does_not_retry_failed_payment_update(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="modify_pending_order_items",
        kwargs={
            "order_id": "#W5442520",
            "item_ids": ["3111466194"],
            "new_item_ids": ["2001307871"],
            "payment_method_id": "credit_card_9753331",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "After updating the payment method for #W5442520 to PayPal, exchange the Patio Umbrella "
                "from 7 ft red polyester manual tilt to 6 ft blue sunbrella auto tilt."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5442520","status":"pending",'
                '"payment_history":['
                '{"transaction_type":"payment","payment_method_id":"credit_card_9753331"},'
                '{"transaction_type":"payment","payment_method_id":"paypal_8049766"},'
                '{"transaction_type":"refund","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Patio Umbrella","product_id":"9743693396","item_id":"3111466194",'
                '"options":{"size":"7 ft","color":"red","material":"polyester","tilt mechanism":"manual tilt"}}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "modify_pending_order_payment", "kwargs": {"order_id": "#W5442520", "payment_method_id": "credit_card_9753331"}}},
        {"role": "tool", "name": "modify_pending_order_payment", "content": "Error: there should be exactly one payment for a pending order"},
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Patio Umbrella","product_id":"9743693396","variants":{'
                '"2001307871":{"item_id":"2001307871","options":{"size":"6 ft","color":"blue","material":"sunbrella","tilt mechanism":"auto tilt"},"available":true,"price":302.63}}}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_order_details", "get_product_details", "get_user_details", "modify_pending_order_payment", "modify_pending_order_items"},
    )

    assert guarded.name == "respond"
    assert info["reason"] == "confirm_before_mutation"
    assert info["proposed_action"]["kwargs"]["payment_method_id"] == "paypal_8049766"
    assert info["item_payment_repair"]["payment_method_id"] == {
        "from": "credit_card_9753331",
        "to": "paypal_8049766",
    }


def test_payment_update_and_item_adjustment_payment_are_separate_slots(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "Change the payment method for order #W7941031 to PayPal. Also exchange the wristwatch "
                "from leather strap with white dial to silicone strap with blue dial, but I prefer to "
                "pay or get refunded by credit card instead of PayPal."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},'
                '"credit_card_9753331":{"source":"credit_card","id":"credit_card_9753331"}}}'
            ),
        },
    ]
    order = {
        "order_id": "#W7941031",
        "status": "pending",
        "payment_history": [{"transaction_type": "payment", "payment_method_id": "credit_card_9753331"}],
    }

    assert module.order_payment_update_method_id(messages, order) == "paypal_8049766"
    assert module.item_mutation_payment_method_id(messages, order) == "credit_card_9753331"


def test_payment_adjustment_ignores_far_earlier_payment_update_source(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    text = (
        "Could you update the payment method for order #W7941031 to PayPal account paypal_8049766? "
        "I want to exchange my Wristwatch from leather strap and white dial to silicone strap with blue dial. "
        "However, instead of using PayPal, I would like to pay or refund via credit card."
    )

    assert module.order_payment_update_source(text) == "paypal"
    assert module.payment_adjustment_source(text) == "credit_card"


def test_requested_items_ignore_boolean_option_values(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    order = {
        "order_id": "#W5442520",
        "items": [
            {
                "name": "Patio Umbrella",
                "product_id": "9743693396",
                "item_id": "3111466194",
                "options": {"size": "7 ft", "color": "red", "tilt mechanism": "manual tilt"},
            },
            {
                "name": "Hiking Boots",
                "product_id": "7363354090",
                "item_id": "2648909398",
                "options": {"size": "8", "material": "leather", "waterproof": "yes"},
            },
        ],
    }
    messages = [
        {
            "role": "user",
            "content": "Yes, please swap the Patio Umbrella from 7 ft red manual tilt to 6 ft blue auto tilt.",
        }
    ]

    assert [item["item_id"] for item in module.requested_order_items(messages, order)] == ["3111466194"]


def test_requested_items_scope_to_current_order_and_ignore_ambiguous_terms(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    order = {
        "order_id": "#W3657213",
        "items": [
            {
                "name": "Action Camera",
                "product_id": "3377618313",
                "item_id": "6700049080",
                "options": {"resolution": "4K", "waterproof": "yes", "color": "black"},
            },
            {
                "name": "Digital Camera",
                "product_id": "8940227892",
                "item_id": "5996159312",
                "options": {"resolution": "24MP", "zoom": "3x", "storage": "SD card"},
            },
            {
                "name": "Cycling Helmet",
                "product_id": "7765186836",
                "item_id": "5886093635",
                "options": {"size": "S", "color": "blue", "ventilation": "low"},
            },
        ],
    }
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W7941031, exchange the blue wristwatch and green backpack. "
                "Use my credit card."
            ),
        },
        {
            "role": "user",
            "content": (
                "Lastly, for order #W3657213, change payment to credit card and exchange "
                "the digital camera from 24MP resolution with 3x zoom and SD card storage "
                "to a 30MP resolution with 5x zoom and CF card storage."
            ),
        },
    ]

    assert [item["item_id"] for item in module.requested_order_items(messages, order)] == ["5996159312"]


def test_requested_items_use_prior_item_request_when_latest_turn_only_asks_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    order = {
        "order_id": "#W7273336",
        "items": [
            {
                "name": "Bookshelf",
                "product_id": "8600330539",
                "item_id": "8018699955",
                "options": {"material": "metal", "color": "brown", "height": "4 ft"},
            },
            {
                "name": "Gaming Mouse",
                "product_id": "5713490933",
                "item_id": "8214883393",
                "options": {"color": "black", "sensor type": "laser", "connectivity": "wireless"},
            },
        ],
    }
    messages = [
        {
            "role": "user",
            "content": (
                "For order #W7273336, I need to exchange the black laser gaming mouse for "
                "a white optical one and switch the metal bookshelf to a taller glass one."
            ),
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide auth."}}},
        {"role": "user", "content": "My email is omar@example.com. Could you look up order #W7273336 details?"},
    ]

    assert [item["item_id"] for item in module.requested_order_items(messages, order)] == [
        "8214883393",
        "8018699955",
    ]


def test_tau_policy_guard_proposes_delivered_item_return_from_observations(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "Please return the cleaner and headphones from delivered order #W2378156 to the original payment method.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "yusuf_rossi_19122"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W2378156"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W2378156","status":"delivered","user_id":"yusuf_rossi_19122",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_1234567"}],'
                '"items":['
                '{"name":"Vacuum Cleaner","product_id":"p_cleaner","item_id":"111","options":{"color":"grey"}},'
                '{"name":"Headphones","product_id":"p_headphones","item_id":"222","options":{"color":"black"}},'
                '{"name":"T-Shirt","product_id":"p_tshirt","item_id":"333","options":{"size":"m"}}'
                ']}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "yusuf_rossi_19122"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"user_id":"yusuf_rossi_19122","payment_methods":{"credit_card_1234567":{"source":"credit_card","id":"credit_card_1234567"}}}'},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_user_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert info["proposed_action"]["kwargs"] == {
        "order_id": "#W2378156",
        "item_ids": ["111", "222"],
        "payment_method_id": "credit_card_1234567",
    }
    assert guarded.name == "return_delivered_order_items"
    assert guarded.kwargs == {
        "order_id": "#W2378156",
        "item_ids": ["111", "222"],
        "payment_method_id": "credit_card_1234567",
    }


def test_tau_policy_guard_waits_for_refund_method_choice_before_return(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W5866402",
            "item_ids": ["9727387530", "6242772310"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Return all items from my order, but I am not sure which payment method to use for the refund. "
                "Can you tell me the refund payment options?"
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"gift_card_7794233":{"source":"gift_card","id":"gift_card_7794233"},'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"}}}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5866402"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5866402","status":"delivered","user_id":"olivia_ito_3591",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Sneakers","item_id":"9727387530"},{"name":"Espresso Machine","item_id":"6242772310"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"find_user_id_by_name_zip", "get_user_details", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_refund_payment_method_before_return_mutation"
    assert guarded.name == "respond"
    assert "gift_card_7794233" in guarded.kwargs["content"]


def test_tau_policy_guard_asks_refund_method_when_multiple_destinations_exist(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W5866402",
            "item_ids": ["9727387530", "6242772310"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {"role": "user", "content": "Please return all items from my order."},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"gift_card_7794233":{"source":"gift_card","id":"gift_card_7794233"},'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"}}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5866402","status":"delivered","user_id":"olivia_ito_3591",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Sneakers","item_id":"9727387530"},{"name":"Espresso Machine","item_id":"6242772310"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "request_refund_payment_method_before_return_mutation"
    assert info["eligible_payment_method_ids"] == ["paypal_8049766", "gift_card_7794233"]
    assert guarded.name == "respond"


def test_tau_policy_guard_uses_final_gift_card_choice_for_return(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    proposed = module.Action(
        name="return_delivered_order_items",
        kwargs={
            "order_id": "#W5866402",
            "item_ids": ["9727387530", "6242772310"],
            "payment_method_id": "paypal_8049766",
        },
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Return all items from my order, but I am not sure which payment method to use for the refund. "
                "Can you tell me the refund payment options?"
            ),
        },
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": (
                '{"user_id":"olivia_ito_3591","payment_methods":{'
                '"gift_card_7794233":{"source":"gift_card","id":"gift_card_7794233"},'
                '"paypal_8049766":{"source":"paypal","id":"paypal_8049766"}}}'
            ),
        },
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5866402","status":"delivered","user_id":"olivia_ito_3591",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Sneakers","item_id":"9727387530"},{"name":"Espresso Machine","item_id":"6242772310"}]}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {
                    "content": (
                        "For a return refund, the refund can go to the original payment method "
                        "(paypal_8049766) or existing gift card gift_card_7794233."
                    )
                },
            },
        },
        {"role": "user", "content": "I’d prefer to receive my refund as a new gift card. Can we proceed with that?"},
    ]

    guarded, info = module.tau_policy_guard_action(
        proposed,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "repair_item_mutation_payment_method"
    assert guarded.name == "return_delivered_order_items"
    assert guarded.kwargs["payment_method_id"] == "gift_card_7794233"


def test_tau_policy_guard_answers_return_followup_after_completed_return(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "These earbuds are faulty. Can I get a refund?"},
        {
            "role": "assistant",
            "action": {
                "name": "return_delivered_order_items",
                "kwargs": {
                    "order_id": "#W3508684",
                    "item_ids": ["3694871183"],
                    "payment_method_id": "paypal_1575973",
                },
            },
        },
        {
            "role": "tool",
            "name": "return_delivered_order_items",
            "content": '{"order_id":"#W3508684","status":"return requested"}',
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "The requested action has been completed."}}},
        {"role": "user", "content": "Can I please get the full refund and skip the return process?"},
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"return_delivered_order_items", "get_order_details", "get_user_details"},
    )

    content = guarded.kwargs["content"].lower()
    assert info["used"] is True
    assert info["reason"] == "respond_after_prior_successful_mutation"
    assert guarded.name == "respond"
    assert "already" in content
    assert "email" in content
    assert "returning" in content
    assert "refund" in content


def test_infer_cancel_reason_from_cheaper_elsewhere(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "I found the same item for half the price at another store, so I want a full refund "
                "and I will purchase from the cheaper store instead."
            ),
        }
    ]

    assert module.infer_reason_from_text(messages, ["ordered by mistake", "no longer needed"]) == "no longer needed"


def test_infer_cancel_reason_prefers_no_longer_needed_when_user_mentions_both(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": "I ordered by mistake, but I don't actually need any of the items now.",
        }
    ]

    assert module.infer_reason_from_text(messages, ["ordered by mistake", "no longer needed"]) == "no longer needed"


def test_tau_policy_guard_repairs_direct_cancel_reason_from_transcript(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="cancel_pending_order",
        kwargs={"order_id": "#W9933266", "reason": "ordered by mistake"},
    )
    messages = [
        {"role": "user", "content": "Please cancel order #W9933266."},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9933266"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W9933266","status":"pending"}'},
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "I can cancel pending order #W9933266. Please confirm with yes to proceed."},
            },
        },
        {"role": "user", "content": "Yes, please proceed."},
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please provide the reason."}}},
        {
            "role": "user",
            "content": "I ordered by mistake, but I don't actually need any of the items now.",
        },
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["ordered by mistake", "no longer needed"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        action,
        messages,
        {"cancel_pending_order", "get_order_details"},
        schemas,
    )

    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W9933266", "reason": "no longer needed"}
    assert info["used"] is True
    assert info["reason"] == "execute_argument_repair"
    assert info["argument_repair"]["semantic"]["reason"]["to"] == "no longer needed"


def test_tau_policy_guard_routes_pending_refund_to_cancel_before_product_options(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": (
                "I found the same tea kettle elsewhere at half the price. I want a full refund "
                "unless you can match the lower price."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "harper@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "harper_moore_3210"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W3942868"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W3942868","status":"pending","user_id":"harper_moore_3210",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_7665260"}],'
                '"items":[{"name":"Tea Kettle","product_id":"9832717871","item_id":"6454334990"}]}'
            ),
        },
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "cancel_pending_order"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "pending_refund_cancel_execute_pending_refund_cancel"
    assert info["proposed_action"]["name"] == "cancel_pending_order"
    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W3942868", "reason": "no longer needed"}


def test_tau_policy_guard_keeps_pending_refund_cancel_after_price_answer(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": (
                "Can you look into this issue with the refund? I want to buy it from the cheaper store "
                "unless you can match their price."
            ),
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "harper@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "harper_moore_3210"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "harper_moore_3210"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"orders":["#W3942868"],"payment_methods":{"credit_card_7665260":{"source":"credit_card","id":"credit_card_7665260"}}}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W3942868"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W3942868","status":"pending","user_id":"harper_moore_3210",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_7665260"}],'
                '"items":[{"name":"Tea Kettle","product_id":"9832717871","item_id":"6454334990","price":98.82}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "get_product_details", "kwargs": {"product_id": "9832717871"}}},
        {
            "role": "tool",
            "name": "get_product_details",
            "content": (
                '{"name":"Tea Kettle","product_id":"9832717871","variants":{'
                '"4238115171":{"item_id":"4238115171","options":{"material":"stainless steel"},"available":true,"price":91.78},'
                '"6454334990":{"item_id":"6454334990","options":{"material":"glass"},"available":false,"price":98.82}}}'
            ),
        },
        {
            "role": "assistant",
            "action": {
                "name": "respond",
                "kwargs": {"content": "The cheapest available tea kettle option is $91.78."},
            },
        },
        {
            "role": "user",
            "content": (
                "I found the same tea kettle for just $45. Please match that price or issue a full refund; "
                "otherwise I'll buy it from another store."
            ),
        },
    ]
    schemas = {
        "cancel_pending_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]},
            },
            "required": ["order_id", "reason"],
        }
    }

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "cancel_pending_order"},
        schemas,
    )

    assert info["used"] is True
    assert info["reason"] == "pending_refund_cancel_execute_pending_refund_cancel"
    assert guarded.name == "cancel_pending_order"
    assert guarded.kwargs == {"order_id": "#W3942868", "reason": "no longer needed"}


def test_tau_policy_guard_looks_up_user_before_product_issue_transfer(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    transfer = module.Action(
        name="transfer_to_human_agents",
        kwargs={"summary": "The earbuds will not connect to my iPhone."},
    )
    messages = [
        {"role": "user", "content": "My earbuds will not connect to my iPhone. Can you help?"},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "fatima@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "fatima_smith_4908"},
    ]

    guarded, info = module.tau_policy_guard_action(
        transfer,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items", "transfer_to_human_agents"},
    )

    assert info["used"] is True
    assert info["reason"] == "avoid_transfer_lookup_user_for_product_issue"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "fatima_smith_4908"}


def test_product_issue_text_does_not_treat_computer_setup_as_damage(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": "The black mouse does not match my computer setup, so I want to exchange it for a white one.",
        }
    ]

    assert module.product_issue_text_seen(messages) is False


def test_product_issue_text_does_not_treat_refund_issue_as_damage(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "Can you look into this issue with the refund? I found the same item for half the price "
                "at another store."
            ),
        }
    ]

    assert module.product_issue_text_seen(messages) is False


def test_tau_policy_guard_avoids_transfer_when_transcript_has_next_order_lookup(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    transfer = module.Action(name="transfer_to_human_agents", kwargs={"summary": "task"})
    messages = [
        {
            "role": "user",
            "content": "For order #W5442520, please exchange the patio umbrella and change payment to PayPal.",
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5442520"}}},
        {"role": "tool", "name": "get_order_details", "content": '{"order_id":"#W5442520","status":"pending"}'},
        {
            "role": "user",
            "content": (
                "That is right. Also for order #W3657213, switch the payment to credit_card_9753331 "
                "and exchange the digital camera."
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        transfer,
        messages,
        {"get_order_details", "transfer_to_human_agents"},
    )

    assert info["used"] is True
    assert info["reason"] == "avoid_transfer_transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W3657213"}


def test_tau_policy_guard_routes_product_issue_to_return_before_transfer(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    transfer = module.Action(
        name="transfer_to_human_agents",
        kwargs={"summary": "The earbuds will not connect to my iPhone."},
    )
    messages = [
        {"role": "user", "content": "The wireless earbuds I ordered will not connect to my iPhone."},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "fatima@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "fatima_smith_4908"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "fatima_smith_4908"}}},
        {"role": "tool", "name": "get_user_details", "content": '{"orders":["#W3508684"],"payment_methods":{"paypal_1575973":{"source":"paypal","id":"paypal_1575973"}}}'},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W3508684"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W3508684","status":"delivered","user_id":"fatima_smith_4908",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_1575973"}],'
                '"items":[{"name":"Wireless Earbuds","product_id":"p_earbuds","item_id":"3694871183","options":{"color":"white"}}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        transfer,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items", "transfer_to_human_agents"},
    )

    assert info["used"] is True
    assert info["reason"].startswith("avoid_transfer_")
    assert guarded.name == "respond"
    assert "confirm" in guarded.kwargs["content"].lower()


def test_tau_policy_guard_scans_user_orders_for_product_issue_item(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "The skateboard I received as a gift chipped after one use. Can I get a refund?"},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "liam@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "liam_li_5260"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "liam_li_5260"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"orders":["#W9653558","#W8512927"],"payment_methods":{"credit_card_7933535":{"source":"credit_card","id":"credit_card_7933535"}}}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9653558"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W9653558","status":"pending",'
                '"items":[{"name":"Wireless Earbuds","item_id":"3694871183"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert info["reason"] == "product_issue_execute_product_issue_return"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W8512927"}


def test_tau_policy_guard_returns_damaged_item_after_scanned_order(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "The skateboard I received as a gift chipped after one use. Can I get a refund?"},
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "liam@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "liam_li_5260"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "liam_li_5260"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"orders":["#W9653558","#W8512927"],"payment_methods":{"credit_card_7933535":{"source":"credit_card","id":"credit_card_7933535"}}}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W9653558"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": '{"order_id":"#W9653558","status":"pending","items":[{"name":"Wireless Earbuds","item_id":"3694871183"}]}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W8512927"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W8512927","status":"delivered","user_id":"liam_li_5260",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_7933535"}],'
                '"items":[{"name":"Skateboard","product_id":"p_skateboard","item_id":"5120532699"}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_user_details", "get_order_details", "return_delivered_order_items", "cancel_pending_order"},
    )

    assert info["used"] is True
    assert guarded.name == "respond"
    assert "confirm" in guarded.kwargs["content"].lower()
    assert info["proposed_action"]["kwargs"] == {
        "order_id": "#W8512927",
        "item_ids": ["5120532699"],
        "payment_method_id": "credit_card_7933535",
    }


def test_tau_policy_guard_fetches_user_details_before_gift_card_return(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "Please return the headphones from delivered order #W2378156 and refund me with a gift card.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "yusuf_rossi_19122"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W2378156"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W2378156","status":"delivered","user_id":"yusuf_rossi_19122",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_1234567"}],'
                '"items":[{"name":"Headphones","product_id":"p_headphones","item_id":"222","options":{"color":"black"}}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_user_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_user_details"
    assert guarded.kwargs == {"user_id": "yusuf_rossi_19122"}


def test_repair_item_mutation_payment_method_uses_gift_card_for_returns(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    action = module.Action(
        name="return_delivered_order_items",
        kwargs={"order_id": "#W5866402", "item_ids": ["9727387530"], "payment_method_id": "paypal_8049766"},
    )
    messages = [
        {"role": "user", "content": "Please return the sneakers and refund me with a gift card."},
        {
            "role": "assistant",
            "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}},
        },
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"payment_methods":{"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},"gift_card_7794233":{"source":"gift_card","id":"gift_card_7794233","balance":42}}}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5866402"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5866402","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Sneakers","item_id":"9727387530"}]}'
            ),
        },
    ]

    repaired, repair = module.repair_item_mutation_payment_method(action, messages)

    assert repaired.kwargs["payment_method_id"] == "gift_card_7794233"
    assert repair["payment_method_id"] == {"from": "paypal_8049766", "to": "gift_card_7794233"}


def test_tau_policy_guard_answers_refund_payment_method_question_before_catalog(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please return the sneakers from order #W5866402."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_user_details", "kwargs": {"user_id": "olivia_ito_3591"}}},
        {
            "role": "tool",
            "name": "get_user_details",
            "content": '{"payment_methods":{"paypal_8049766":{"source":"paypal","id":"paypal_8049766"},"gift_card_7794233":{"source":"gift_card","id":"gift_card_7794233","balance":42}}}',
        },
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W5866402"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W5866402","status":"delivered",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"paypal_8049766"}],'
                '"items":[{"name":"Sneakers","product_id":"p_sneakers","item_id":"9727387530"}]}'
            ),
        },
        {"role": "assistant", "action": {"name": "respond", "kwargs": {"content": "Please confirm with yes."}}},
        {
            "role": "user",
            "content": "Yes, please start the refund. What payment methods are available for refunds?",
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_user_details", "get_product_details", "return_delivered_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "refund_payment_answer_refund_payment_methods"
    assert guarded.name == "respond"
    assert "gift_card_7794233" in guarded.kwargs["content"]
    assert "paypal_8049766" in guarded.kwargs["content"]


def test_refund_payment_helper_does_not_hijack_exchange_payment_request(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {"role": "user", "content": "Please update order #W5442520."},
        {"role": "assistant", "action": {"name": "find_user_id_by_name_zip", "kwargs": {"first_name": "Olivia", "last_name": "Ito", "zip": "80218"}}},
        {"role": "tool", "name": "find_user_id_by_name_zip", "content": "olivia_ito_3591"},
        {
            "role": "user",
            "content": (
                "What about the next one, for order #W7941031? Could you update the payment "
                "to PayPal and exchange the wristwatch, but instead of getting a refund I want "
                "to use my credit card."
            ),
        },
    ]

    assert module.refund_payment_method_question_seen(messages) is False

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_user_details", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_order_details"
    assert guarded.kwargs == {"order_id": "#W7941031"}


def test_refund_payment_helper_does_not_hijack_payment_change_question(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    messages = [
        {
            "role": "user",
            "content": (
                "What about the payment method for order #W7941031? Can that be changed "
                "to my credit card instead of PayPal? A refund would also work."
            ),
        }
    ]

    assert module.refund_payment_method_question_seen(messages) is False


def test_tau_policy_guard_fetches_pending_product_before_item_modify(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)
    fallback = module.Action(name="respond", kwargs={"content": "task"})
    messages = [
        {
            "role": "user",
            "content": "Please exchange the Wristwatch in order #W7941031 for one with a silicone strap and blue dial.",
        },
        {"role": "assistant", "action": {"name": "find_user_id_by_email", "kwargs": {"email": "olivia@example.com"}}},
        {"role": "tool", "name": "find_user_id_by_email", "content": "olivia_ito_3591"},
        {"role": "assistant", "action": {"name": "get_order_details", "kwargs": {"order_id": "#W7941031"}}},
        {
            "role": "tool",
            "name": "get_order_details",
            "content": (
                '{"order_id":"#W7941031","status":"pending",'
                '"payment_history":[{"transaction_type":"payment","payment_method_id":"credit_card_9753331"}],'
                '"items":[{"name":"Wristwatch","product_id":"6066914160",'
                '"item_id":"1355937109","options":{"strap material":"leather","dial color":"white"}}]}'
            ),
        },
    ]

    guarded, info = module.tau_policy_guard_action(
        fallback,
        messages,
        {"get_order_details", "get_product_details", "modify_pending_order_items"},
    )

    assert info["used"] is True
    assert info["reason"] == "transcript_proposal_execute_proposed_action"
    assert guarded.name == "get_product_details"
    assert guarded.kwargs == {"product_id": "6066914160"}


def test_solve_scores_after_post_success_response_timeout(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    class Info(dict):
        def __getattr__(self, key):
            return self.get(key)

    class Response:
        def __init__(self, observation="", reward=0.0, info=None, done=False):
            self.observation = observation
            self.reward = reward
            self.info = info or Info()
            self.done = done

    class FakeEnv:
        def __init__(self):
            self.actions = []

        def reset(self, task_index=None):
            return Response(
                observation="Please exchange item 8214883393 in order #W7273336.",
                info=Info(source="user"),
            )

        def step(self, action):
            self.actions.append(action)
            if action.name == "exchange_delivered_order_items":
                return Response(
                    observation='{"order_id":"#W7273336","status":"exchange requested"}',
                    info=Info(source=action.name),
                )
            raise TimeoutError("user simulator timed out")

        def calculate_reward(self):
            return Response(reward=1.0, info=Info(reward_info={"reward": 1.0}))

    agent = module.GoalGraphAgent(tools_info=[], wiki="Retail policy", model="local", provider="openai")
    planned = [
        module.Action(
            name="exchange_delivered_order_items",
            kwargs={
                "order_id": "#W7273336",
                "item_ids": ["8214883393"],
                "new_item_ids": ["2880340443"],
                "payment_method_id": "paypal_1530316",
            },
        ),
        module.Action(name="respond", kwargs={"content": "Done."}),
    ]

    def plan_action(messages, previous_source):
        return planned.pop(0), {"verification_ok": True}, 0

    agent.plan_action = plan_action

    result = agent.solve(FakeEnv(), max_num_steps=2)

    assert result.reward == 1.0
    assert "TimeoutError" in result.info["post_success_response_error"]
    assert result.info["goal_graph_diagnostics"][-1]["source"] == "post_success_response_error"


def test_solve_executes_one_verified_transition_after_terminal_confirmation(monkeypatch):
    module = load_tau_goal_graph_module(monkeypatch)

    class Info(dict):
        def __getattr__(self, key):
            return self.get(key)

    class Response:
        def __init__(self, observation="", reward=0.0, info=None, done=False):
            self.observation = observation
            self.reward = reward
            self.info = info or Info()
            self.done = done

    class FakeEnv:
        def __init__(self):
            self.actions = []

        def reset(self, task_index=None):
            return Response(
                observation="Please exchange item 8214883393 in order #W7273336.",
                info=Info(source="user"),
            )

        def step(self, action):
            self.actions.append(action)
            if action.name == "respond":
                return Response(
                    observation="Yes, please go ahead with that exchange.\n\n###STOP###",
                    info=Info(source="user"),
                    done=True,
                )
            if action.name == "exchange_delivered_order_items":
                return Response(
                    observation='{"order_id":"#W7273336","status":"exchange requested"}',
                    reward=1.0,
                    info=Info(source=action.name),
                    done=True,
                )
            raise AssertionError(action)

        def calculate_reward(self):
            return Response(reward=1.0, info=Info(reward_info={"reward": 1.0}))

    agent = module.GoalGraphAgent(tools_info=[], wiki="Retail policy", model="local", provider="openai")
    deferred = module.Action(
        name="exchange_delivered_order_items",
        kwargs={
            "order_id": "#W7273336",
            "item_ids": ["8214883393"],
            "new_item_ids": ["2880340443"],
            "payment_method_id": "paypal_1530316",
        },
    )
    planned = [
        (
            module.Action(name="respond", kwargs={"content": "I can exchange it. Please confirm with yes."}),
            {
                "verification_ok": True,
                "tau_policy_guard": {
                    "used": True,
                    "reason": "confirm_before_mutation",
                    "proposed_action": deferred.model_dump(),
                },
            },
        ),
        (deferred, {"verification_ok": True}),
    ]

    def plan_action(messages, previous_source):
        action, result = planned.pop(0)
        return action, result, 0

    agent.plan_action = plan_action
    env = FakeEnv()

    result = agent.solve(env, max_num_steps=3)

    assert [action.name for action in env.actions] == ["respond", "exchange_delivered_order_items"]
    assert result.reward == 1.0
    finalization = result.info["goal_graph_diagnostics"][0]["terminal_user_finalization"]
    assert finalization["executed"] is True
    assert finalization["planned_action"] == deferred.model_dump()
    assert result.info["goal_graph_diagnostics"][-1]["terminal_user_finalization"] is True
