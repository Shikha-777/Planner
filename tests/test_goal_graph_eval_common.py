import json

from scripts.goal_graph_eval_common import (
    benchmark_compile_tools,
    _binding_plan_from_call_skeleton,
    build_tool_binding_frame_messages,
    extract_call_skeleton_json_object,
    extract_goal_graph_json_object,
    extract_semantic_frame_json_object,
    _evidence_in_request,
    _graph_from_binding_plan,
    plan_and_compile_goal_graph,
    _prefer_raw_binding_plan,
    _prefer_skeleton_binding_plan,
    _should_generate_call_skeleton,
    _stateful_nonterminal_recovery_plan,
    _stateful_readonly_progress_plan,
    _stateful_active_user_turn,
    _stateful_missing_input_evidence_adjudication,
    _stateful_next_collection_readonly_plan,
    _drop_successfully_replayed_calls,
    _generate_semantic_frame,
    _limit_stateful_plan_to_one_call,
    _merge_stateful_goal_ledgers,
    _normalize_stateful_goal_ledger,
    _drop_stateful_unverified_fallback_calls,
    _stateful_plan_defers_to_user_or_terminal,
    _stateful_progress_repair_plan,
    _stateful_repair_needs_binding_correction,
    _stateful_verified_observation_facts,
    _review_stateful_candidate_calls,
    _review_stateful_terminal_decision,
)
from scripts.eval_goal_graph_bfcl import bfcl_result_calls
from taskdecomp.goal_graph_runtime import GoalGraphRuntime
from taskdecomp.tool_binding import _filter_schema_value_incompatible_calls, build_tool_binding_plan


def test_benchmark_compile_tools_marks_offline_side_effect_names_read_only():
    original = [{"name": "CreateEvent", "description": "Create an event."}]

    prepared = benchmark_compile_tools(original)

    assert prepared[0]["risk"] == "read_only"
    assert prepared[0]["requires_confirmation"] is False
    assert prepared[0]["requires_unique_target"] is False
    assert "risk" not in original[0]


def test_stateful_semantic_prompt_preserves_record_roles_and_fallback_constraints():
    messages = build_tool_binding_frame_messages(
        "Complete this under my profile, preferring option A but allowing option B.",
        [],
        execution_history=[
            {
                "tool_name": "get_profile",
                "arguments": {"user_id": "U-1"},
                "outcome": "success",
                "observation": {"name": "Ada", "related_items": [{"name": "Other"}]},
            }
        ],
        stateful=True,
        stateful_goal_ledger_required=True,
    )

    prompt = messages[0]["content"]
    assert "Preserve semantic roles across observed records" in prompt
    assert "matching observed record" in prompt
    assert "fallback alternatives have been exhausted" in messages[1]["content"]


def test_stateful_semantic_prompt_expands_collection_goals_after_discovery():
    messages = build_tool_binding_frame_messages(
        "Apply the requested update to every discovered record.",
        [],
        stateful=True,
        stateful_goal_ledger_required=True,
    )

    assert "one concrete goal per matching entity" in messages[1]["content"]
    assert "never completes the collection" in messages[1]["content"]


def test_stateful_semantic_prompt_requires_record_disambiguation_before_mutation():
    messages = build_tool_binding_frame_messages(
        "Update the matching reservation.",
        [],
        stateful=True,
        stateful_goal_ledger_required=True,
    )

    assert "do not select the first one arbitrarily" in messages[0]["content"]
    assert "before an observation verifies that match" in messages[0]["content"]


def test_plan_and_compile_goal_graph_accepts_valid_model_json():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return """
        {
          "goal": "Get weather in Boston",
          "nodes": [
            {
              "id": "n1",
              "kind": "retrieve",
              "capability": "weather.get",
              "inputs": {
                "location": {
                  "value": "Boston",
                  "source": "query",
                  "evidence": "Boston"
                }
              },
              "risk": "read_only"
            }
          ]
        }
        """

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "What is the weather in Boston?",
        [
            {
                "name": "weather.get",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        max_new_tokens=100,
        planner_mode="one_shot",
    )

    assert result["verification_ok"]
    assert result["calls"][0]["tool_name"] == "weather.get"
    assert result["calls"][0]["arguments"] == {"location": "Boston"}


def test_stepwise_stateful_binding_request_defers_terminal_response():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return """
        {
          "tool_decision": "ask_user",
          "canonical_request": "Need account email before doing anything.",
          "slots_observed": [],
          "tool_bindings": [],
          "missing_inputs": ["email"]
        }
        """

    tools = [
        {
            "name": "get_order_details",
            "description": "Retrieve order details by order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
        {
            "name": "respond",
            "description": "Send a customer-facing message when clarification or a final response is needed.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    ]

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Policy: ask for email first.\nTranscript:\nuser: Please check order W3942868.",
        tools,
        max_new_tokens=100,
        planner_mode="stepwise",
        stateful=True,
        binding_request="Transcript:\nuser: Please check order W3942868.",
    )

    assert result["verification_ok"]
    assert result["binding_request_separate"] is True
    assert result["calls"][0]["tool_name"] == "get_order_details"
    assert result["calls"][0]["arguments"] == {"order_id": "W3942868"}

    recovery = _stateful_nonterminal_recovery_plan(
        "Transcript:\nuser: Please check order W3942868.",
        tools,
        {
            "tool_decision": "call",
            "calls": [{"tool_name": "respond", "arguments": {"content": "available"}}],
        },
        GoalGraphRuntime(tools),
    )
    assert recovery is not None
    assert recovery["calls"][0]["tool_name"] == "get_order_details"
    assert recovery["stateful_terminal_recovery"]["used"] is True

    mutation_first = _stateful_readonly_progress_plan(
        "Transcript:\nuser: Please check order W3942868.",
        {
            "tool_decision": "call",
            "calls": [
                {"tool_name": "cancel_order", "arguments": {"order_id": "W3942868"}},
                {"tool_name": "get_order_details", "arguments": {"order_id": "W3942868"}},
            ],
        },
        GoalGraphRuntime(
            [
                *tools,
                {
                    "name": "cancel_order",
                    "description": "Cancel an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            ]
        ),
        tools,
    )
    assert mutation_first is not None
    assert mutation_first["calls"][0]["tool_name"] == "get_order_details"
    assert mutation_first["stateful_readonly_progress"]["used"] is True

    retry_after_drop = _stateful_nonterminal_recovery_plan(
        "Transcript:\nuser: update payment for order #W5442520 to PayPal account paypal_8049766",
        [
            *tools,
            {
                "name": "modify_pending_order_payment",
                "description": "Modify payment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "order id"},
                        "payment_method_id": {"type": "string", "description": "payment method id"},
                    },
                    "required": ["order_id", "payment_method_id"],
                },
            },
        ],
        {
            "tool_decision": "no_tool",
            "calls": [],
            "dropped_incompatible_calls": [
                {
                    "tool_name": "modify_pending_order_payment",
                    "arguments": {"order_id": "W5442520", "payment_method_id": "W5442520"},
                    "reason": "value role 'order' is incompatible with identifier slot 'payment_method_id'",
                }
            ],
        },
        GoalGraphRuntime(
            [
                *tools,
                {
                    "name": "modify_pending_order_payment",
                    "description": "Modify payment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "order id"},
                            "payment_method_id": {"type": "string", "description": "payment method id"},
                        },
                        "required": ["order_id", "payment_method_id"],
                    },
                },
            ]
        ),
    )
    assert retry_after_drop is not None
    assert retry_after_drop["calls"][0]["tool_name"] == "get_order_details"
    assert retry_after_drop["calls"][0]["arguments"] == {"order_id": "#W5442520"}
    assert retry_after_drop["stateful_readonly_retry"]["used"] is True


def test_stepwise_stateful_preserves_semantic_confirmation_request():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"ask_user","canonical_request":"Confirm the proposed change before proceeding.",'
            '"slots_observed":[],"tool_bindings":[],"missing_inputs":[]}'
        )

    tools = [
        {
            "name": "get_order_details",
            "description": "Retrieve order details by order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
        {
            "name": "respond",
            "description": "Send a customer-facing message.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    ]

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Transcript:\nuser: Change order O-1.",
        tools,
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        binding_request="Binding evidence:\nuser: Change order O-1.",
    )

    assert result["calls"] == []
    assert result["tool_binding_plan"]["tool_decision"] == "no_tool"
    assert result["steps"][1]["semantic_ask_user_veto"] is True
    assert result["tool_binding_recovery_plan"] == {}


def test_stateful_progress_filter_keeps_new_calls_and_drops_successful_replays():
    plan, info = _drop_successfully_replayed_calls(
        {
            "tool_decision": "call",
            "calls": [
                {"id": "call_1", "tool_name": "lookup_order", "arguments": {"order_id": "O-1"}},
                {"id": "call_2", "tool_name": "lookup_user", "arguments": {"user_id": "U-1"}},
            ],
        },
        [{"tool_name": "lookup_order", "arguments": {"order_id": "O-1"}, "outcome": "success"}],
    )

    assert info["used"] is True
    assert info["dropped_calls"][0]["tool_name"] == "lookup_order"
    assert plan["calls"] == [{"id": "call_1", "tool_name": "lookup_user", "arguments": {"user_id": "U-1"}}]


def test_stateful_progress_filter_drops_replayed_failed_call():
    plan, info = _drop_successfully_replayed_calls(
        {"tool_decision": "call", "calls": [{"tool_name": "calculate", "arguments": {"expression": "PM"}}]},
        [{"tool_name": "calculate", "arguments": {"expression": "PM"}, "outcome": "failure"}],
    )

    assert info["used"] is True
    assert plan["calls"] == []
    assert plan["tool_decision"] == "no_tool"


def test_semantic_frame_receives_structured_stateful_observation():
    messages = build_tool_binding_frame_messages(
        "Update record O-1.",
        [],
        execution_history=[
            {
                "tool_name": "get_record",
                "arguments": {"order_id": "O-1"},
                "outcome": "success",
                "observation": {"order_id": "O-1", "status": "pending"},
            }
        ],
    )

    payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])
    assert payload["execution_history"] == [
        {"tool_name": "get_record", "arguments": {"order_id": "O-1"}, "outcome": "success"}
    ]
    assert {"tool_name": "get_record", "path": "status", "value": "pending"} in payload["verified_observation_facts"]
    assert "trusted, compact, source-attributed" in messages[0]["content"]


def test_stateful_semantic_prompt_bounds_observation_facts_without_raw_duplication():
    history = [
        {
            "tool_name": f"lookup_{index}",
            "arguments": {"id": str(index)},
            "outcome": "success",
            "observation": {f"field_{item}": f"value-{index}-{item}" for item in range(40)},
        }
        for index in range(2)
    ]

    messages = build_tool_binding_frame_messages("Use the discovered records.", [], execution_history=history)
    payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])

    assert len(payload["verified_observation_facts"]) == 32
    assert all("observation" not in call for call in payload["execution_history"])


def test_stateful_observation_facts_preserve_source_and_nested_fields():
    history = [
        {
            "tool_name": "get_profile",
            "outcome": "success",
            "observation": {"name": {"first": "Mia"}, "dob": "1990-04-05"},
        }
    ]

    facts = _stateful_verified_observation_facts(history)
    messages = build_tool_binding_frame_messages("Complete the booking.", [], execution_history=history)

    assert {"tool_name": "get_profile", "path": "dob", "value": "1990-04-05"} in facts
    assert '"verified_observation_facts"' in messages[-1]["content"]
    assert '"path":"dob","value":"1990-04-05"' in messages[-1]["content"]
    assert "declining to restate a fact" in messages[0]["content"]


def test_stateful_observation_facts_prioritize_direct_fields_under_budget():
    history = [
        {
            "tool_name": "get_profile",
            "outcome": "success",
            "observation": {
                "preferences": {f"nested_{index}": f"value-{index}" for index in range(20)},
                "membership": "gold",
            },
        }
    ]

    facts = _stateful_verified_observation_facts(history, per_observation_max=3)

    assert {"tool_name": "get_profile", "path": "membership", "value": "gold"} in facts
    assert len(facts) == 3


def test_stateful_binding_accepts_model_value_from_audited_observation_fact():
    plan = build_tool_binding_plan(
        "user: Continue with the confirmed profile.",
        [
            {
                "name": "get_profile",
                "description": "Retrieve a profile by user identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "canonical_request": "Retrieve the confirmed user profile.",
                "tool_bindings": [
                    {
                        "tool_name": "get_profile",
                        "call_count": 1,
                        "arguments": {"user_id": "U-1"},
                        "evidence_spans": {"user_id": "U-1"},
                    }
                ],
            }
        },
        allow_model_binding_prefix=True,
        verified_evidence=[{"tool_name": "get_profile", "path": "user_id", "value": "U-1"}],
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["arguments"] == {"user_id": "U-1"}


def test_stateful_missing_input_adjudication_requires_quoted_observation_fact():
    outcome = _stateful_missing_input_evidence_adjudication(
        None,
        None,
        lambda *_args: (
            '{"available_missing_inputs":[{"missing_input":"passenger_dob",'
            '"evidence_span":"1990-04-05"}]}'
        ),
        {
            "tool_decision": "ask_user",
            "missing_inputs": ["passenger_dob"],
        },
        [
            {
                "tool_name": "get_profile",
                "outcome": "success",
                "observation": {"date_of_birth": "1990-04-05"},
            }
        ],
        100,
    )

    assert outcome["available_missing_inputs"] == [
        {
            "missing_input": "passenger_dob",
            "evidence_span": "1990-04-05",
            "source_fact": {
                "tool_name": "get_profile",
                "path": "date_of_birth",
                "value": "1990-04-05",
            },
        }
    ]


def test_stateful_missing_input_adjudication_keeps_collection_identifiers_ambiguous():
    outcome = _stateful_missing_input_evidence_adjudication(
        None,
        None,
        lambda *_args: (
            '{"available_missing_inputs":[{"missing_input":"reservation_id",'
            '"evidence_span":"R-1"}]}'
        ),
        {
            "tool_decision": "ask_user",
            "missing_inputs": ["reservation_id"],
        },
        [
            {
                "tool_name": "get_profile",
                "outcome": "success",
                "observation": {"reservations": ["R-1", "R-2"]},
            }
        ],
        100,
    )

    assert outcome["available_missing_inputs"] == []
    assert outcome["ambiguous_missing_inputs"] == [
        {
            "missing_input": "reservation_id",
            "candidate_facts": [
                {"tool_name": "get_profile", "path": "reservations[0]", "value": "R-1"},
                {"tool_name": "get_profile", "path": "reservations[1]", "value": "R-2"},
            ],
        }
    ]


def test_stateful_active_user_turn_prefers_latest_labelled_turn():
    request = "user: First request.\nassistant_action: {}\nuser: Yes, proceed with the confirmed action."

    assert _stateful_active_user_turn(request) == "Yes, proceed with the confirmed action."


def test_stateful_repair_retries_any_terminal_alias_after_rejection():
    assert _stateful_repair_needs_binding_correction(
        {"tool_decision": "respond"},
        {"tool_decision": "respond", "calls": []},
        semantic_only=True,
        no_action_was_rejected=True,
    ) is True


def test_stateful_progress_prompt_prefers_grounded_read_resolution_over_asking_for_opaque_ids():
    messages = build_tool_binding_frame_messages(
        "Update the selected resource.",
        [],
        require_stateful_progress=True,
    )

    assert "opaque identifier" in messages[-1]["content"]
    assert "read-only tool can resolve it" in messages[-1]["content"]


def test_stateful_prompt_preserves_the_order_of_conditional_alternatives():
    messages = build_tool_binding_frame_messages(
        "Try option A; if it is unavailable, use option B.",
        [],
        stateful=True,
    )

    prompt = messages[-1]["content"]
    assert "alternatives and fallbacks as conditional" in prompt
    assert "preserve their stated priority" in prompt
    assert "observation rules it out" in prompt


def test_stateful_prompt_reuses_fields_from_successful_observations():
    messages = build_tool_binding_frame_messages(
        "Complete the requested operation.",
        [],
        stateful=True,
    )

    prompt = messages[0]["content"]
    assert "successful observation contains a field" in prompt
    assert "asking the user to repeat it" in prompt


def test_semantic_binding_prompt_asks_for_schema_supported_alternatives_without_committing():
    messages = build_tool_binding_frame_messages(
        "Update my account.",
        [],
    )

    prompt = messages[-1]["content"]
    assert "multiple schemas can establish the same prerequisite" in prompt
    assert "do not arbitrarily commit to one path" in prompt
    assert "clarification_message" in prompt


def test_semantic_binding_prompt_exposes_stateful_goal_ledger_as_continuity_context():
    ledger = {
        "goals": [
            {
                "id": "goal_auth",
                "objective": "Verify the requested account.",
                "status": "pending",
                "depends_on": [],
            }
        ],
        "next_goal_id": "goal_auth",
    }
    messages = build_tool_binding_frame_messages(
        "Update the requested record.",
        [],
        stateful_goal_ledger=ledger,
        stateful=True,
        stateful_goal_ledger_required=True,
    )
    payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])

    assert payload["stateful_goal_ledger"] == ledger
    assert "semantic continuity context" in messages[-1]["content"]


def test_stateful_semantic_binding_prompt_initializes_an_empty_goal_ledger():
    messages = build_tool_binding_frame_messages(
        "Update the requested record.",
        [],
        stateful=True,
        stateful_goal_ledger_required=True,
    )
    payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])

    assert payload["stateful_goal_ledger"] == {}
    assert "Include goal_ledger" in messages[-1]["content"]


def test_stateful_semantic_binding_prompt_requires_goal_ledger_when_requested():
    messages = build_tool_binding_frame_messages(
        "Update the requested record.",
        [],
        stateful=True,
        stateful_goal_ledger_required=True,
    )

    assert "goal_ledger is REQUIRED" in messages[-1]["content"]
    assert "required goal_ledger" in messages[-1]["content"]
    assert "every active concrete user-requested outcome" in messages[-1]["content"]
    assert "never use meta-goals" in messages[-1]["content"]


def test_runtime_owned_stateful_prompt_requests_only_additive_goal_delta():
    messages = build_tool_binding_frame_messages(
        "Cancel the matching reservation.",
        [],
        stateful=True,
        stateful_goal_ledger_required=False,
    )

    prompt = messages[-1]["content"]

    assert "The runtime owns the goal ledger" in prompt
    assert "optional goal_delta object" in prompt
    assert "requested_fact_delta" in prompt
    assert "confirmation_delta" in prompt
    assert "Do not emit goal_ledger" in prompt
    assert "goal_ledger is REQUIRED" not in prompt


def test_stateful_goal_ledger_normalizes_malformed_dependency_shape():
    ledger = _normalize_stateful_goal_ledger(
        {
            "goals": [
                {
                    "id": "goal_1",
                    "objective": "Retrieve record 101.",
                    "status": "pending",
                    "depends_on": 0,
                }
            ],
            "next_goal_id": "goal_1",
        }
    )

    assert ledger["goals"][0]["depends_on"] == []


def test_stateful_goal_ledger_preserves_unresolved_goal_order_and_appends_new_goals():
    previous = {
        "goals": [
            {"id": "goal_a", "objective": "Complete first request.", "status": "pending", "depends_on": []},
            {"id": "goal_b", "objective": "Complete second request.", "status": "pending", "depends_on": []},
        ],
        "next_goal_id": "goal_a",
    }
    proposed = {
        "goals": [
            {"id": "goal_b", "objective": "Changed text.", "status": "pending", "depends_on": []},
            {"id": "goal_a", "objective": "Changed text.", "status": "pending", "depends_on": []},
            {"id": "goal_c", "objective": "Later request.", "status": "pending", "depends_on": ["goal_b"]},
        ],
        "next_goal_id": "goal_b",
    }

    merged = _merge_stateful_goal_ledgers(previous, proposed)

    assert [goal["id"] for goal in merged["goals"]] == ["goal_a", "goal_b", "goal_c"]
    assert merged["next_goal_id"] == "goal_a"


def test_stateful_goal_ledger_advances_only_after_active_goal_is_completed():
    previous = {
        "goals": [
            {"id": "goal_a", "objective": "Complete first request.", "status": "pending", "depends_on": []},
            {"id": "goal_b", "objective": "Complete second request.", "status": "pending", "depends_on": []},
        ],
        "next_goal_id": "goal_a",
    }
    proposed = {
        "goals": [
            {"id": "goal_a", "objective": "Complete first request.", "status": "completed", "depends_on": []},
            {"id": "goal_b", "objective": "Complete second request.", "status": "pending", "depends_on": []},
        ],
        "next_goal_id": "goal_b",
    }

    merged = _merge_stateful_goal_ledgers(previous, proposed)

    assert merged["next_goal_id"] == "goal_b"


def test_stepwise_stateful_result_preserves_validated_semantic_goal_ledger():
    ledger = {
        "goals": [
            {
                "id": "goal_lookup",
                "objective": "Retrieve record 101.",
                "status": "pending",
                "depends_on": [],
            }
        ],
        "next_goal_id": "goal_lookup",
    }
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: json.dumps(
            {
                "tool_decision": "no_tool",
                "canonical_request": "No tool is needed.",
                "slots_observed": [],
                "call_groups": [],
                "tool_bindings": [],
                "missing_inputs": [],
                "goal_ledger": ledger,
            }
        ),
        "No tool is needed.",
        [],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
        stateful_goal_ledger=ledger,
    )

    assert result["stateful_goal_ledger"] == ledger


def test_stateful_semantic_frame_uses_compact_single_transition_token_budget():
    token_limits: list[int] = []

    _generate_semantic_frame(
        None,
        None,
        lambda _model, _tokenizer, _messages, max_tokens: token_limits.append(max_tokens) or '{"tool_decision":"no_tool","missing_inputs":[]}',
        "Retrieve record 101.",
        [],
        900,
        stateful=True,
    )

    assert token_limits == [1400]


def test_stateful_semantic_frame_adaptively_retries_incomplete_json():
    token_limits: list[int] = []
    outputs = iter(
        [
            '{"tool_decision":"call",',
            '{"tool_decision":"no_tool","canonical_request":"Done.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}',
        ]
    )

    result = _generate_semantic_frame(
        None,
        None,
        lambda _model, _tokenizer, _messages, max_tokens: token_limits.append(max_tokens) or next(outputs),
        "Retrieve record 101.",
        [],
        900,
        stateful=True,
    )

    assert token_limits == [1400, 2800]
    assert result["parsed"]["tool_decision"] == "no_tool"
    assert result["adaptive_retry"]["used"] is True


def test_stateful_semantic_frame_recovers_prose_as_json_after_retries():
    token_limits: list[int] = []
    outputs = iter(
        [
            "The next action should ask the user for an account ID.",
            "The next action should ask the user for an account ID.",
            json.dumps(
                {
                    "tool_decision": "ask_user",
                    "canonical_request": "Retrieve the requested record.",
                    "slots_observed": [],
                    "call_groups": [],
                    "tool_bindings": [],
                    "missing_inputs": ["account_id"],
                    "clarification_message": "Please provide the account ID.",
                }
            ),
        ]
    )

    result = _generate_semantic_frame(
        None,
        None,
        lambda _model, _tokenizer, _messages, max_tokens: token_limits.append(max_tokens) or next(outputs),
        "Retrieve the requested record.",
        [],
        900,
        stateful=True,
    )

    assert token_limits == [1400, 2800, 1000]
    assert result["parsed"]["missing_inputs"] == ["account_id"]
    assert result["format_recovery"]["used"] is True


def test_stateful_semantic_frame_retries_when_required_goal_ledger_is_missing():
    token_limits: list[int] = []
    ledger = {
        "goals": [{"id": "goal_1", "objective": "Retrieve record 101.", "status": "pending", "depends_on": []}],
        "next_goal_id": "goal_1",
    }
    outputs = iter(
        [
            '{"tool_decision":"no_tool","canonical_request":"Wait.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}',
            json.dumps(
                {
                    "tool_decision": "no_tool",
                    "canonical_request": "Wait.",
                    "slots_observed": [],
                    "call_groups": [],
                    "tool_bindings": [],
                    "missing_inputs": [],
                    "goal_ledger": ledger,
                }
            ),
        ]
    )

    result = _generate_semantic_frame(
        None,
        None,
        lambda _model, _tokenizer, _messages, max_tokens: token_limits.append(max_tokens) or next(outputs),
        "Retrieve record 101.",
        [],
        900,
        stateful=True,
        stateful_goal_ledger_required=True,
    )

    assert token_limits == [1400, 2800]
    assert result["parsed"]["goal_ledger"] == ledger
    assert result["adaptive_retry"]["initial_parse_error"] == "stateful semantic frame omitted the required goal_ledger"


def test_stateful_semantic_frame_preserves_existing_ledger_when_model_omits_echo():
    ledger = {
        "goals": [{"id": "goal_1", "objective": "Retrieve record 101.", "status": "pending", "depends_on": []}],
        "next_goal_id": "goal_1",
    }

    result = _generate_semantic_frame(
        None,
        None,
        lambda *_args: (
            '{"tool_decision":"call","canonical_request":"Retrieve record 101.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
        ),
        "Retrieve record 101.",
        [],
        900,
        stateful=True,
        stateful_goal_ledger=ledger,
        stateful_goal_ledger_required=True,
    )

    assert result["parsed"]["goal_ledger"] == ledger
    assert result["adaptive_retry"]["attempted"] is False
    assert result["goal_ledger_recovery"]["used"] is True


def test_stateful_candidate_reviewer_allows_the_supplied_candidate_only():
    def fake_generate(_model, _tokenizer, messages, _max_new_tokens):
        payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])
        assert payload["candidate_calls"] == [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
        return json.dumps(
            {
                "verdict": "allow",
                "reason": "The call retrieves the requested record.",
                "candidate_calls": payload["candidate_calls"],
            }
        )

    review = _review_stateful_candidate_calls(
        None,
        None,
        fake_generate,
        "Retrieve record R-1.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            }
        ],
        [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}],
        [],
        100,
    )

    assert review["attempted"] is True
    assert review["allowed"] is True


def test_stateful_candidate_reviewer_allows_bounded_readonly_disambiguation_prompt():
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]

    def fake_generate(_model, _tokenizer, messages, _max_new_tokens):
        assert "bounded disambiguation step" in messages[0]["content"]
        return json.dumps(
            {
                "verdict": "allow",
                "reason": "The read can identify the matching record.",
                "candidate_calls": candidate_calls,
            }
        )

    review = _review_stateful_candidate_calls(
        None,
        None,
        fake_generate,
        "Find the matching record.",
        [],
        candidate_calls,
        [
            {
                "tool_name": "get_profile",
                "outcome": "success",
                "observation": {"records": ["R-1", "R-2"]},
            }
        ],
        100,
    )

    assert review["allowed"] is True


def test_stateful_candidate_reviewer_preserves_verified_bounded_readonly_disambiguation():
    tools = [
        {
            "name": "get_record",
            "description": "Retrieve details for one record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        }
    ]
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
    review = _review_stateful_candidate_calls(
        None,
        None,
        lambda *_args: json.dumps(
            {
                "verdict": "reject",
                "reason": "The user did not name R-1.",
                "checks": {"target_uniquely_resolved": "false"},
                "failed_check": "target_uniquely_resolved",
                "evidence_ids": ["obs_profile:/records"],
                "candidate_calls": candidate_calls,
            }
        ),
        "Find the matching record.",
        tools,
        candidate_calls,
        [
            {
                "tool_name": "get_profile",
                "outcome": "success",
                "observation": {"records": ["R-1", "R-2"]},
            }
        ],
        100,
        runtime=GoalGraphRuntime(tools),
    )

    assert review["allowed"] is True
    assert review["overridden"] is True


def test_stateful_continues_one_started_readonly_collection_search():
    tools = [
        {
            "name": "get_record",
            "description": "Retrieve details for one record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        }
    ]
    plan = _stateful_next_collection_readonly_plan(
        "Find the matching record among R-1, R-2, and R-3.",
        GoalGraphRuntime(tools),
        [
            {"tool_name": "get_profile", "outcome": "success", "observation": {"records": ["R-1", "R-2", "R-3"]}},
            {
                "tool_name": "get_record",
                "arguments": {"record_id": "R-1"},
                "outcome": "success",
                "observation": {"record_id": "R-1", "matches": False},
            },
        ],
    )

    assert plan is not None
    assert plan["calls"] == [
        {
            "id": "call_1",
            "tool_name": "get_record",
            "arguments": {"record_id": "R-2"},
            "argument_evidence": {"record_id": "R-2"},
            "depends_on": [],
            "missing_arguments": [],
        }
    ]
    assert plan["stateful_collection_disambiguation"]["used"] is True


def test_stateful_candidate_reviewer_rejects_an_explicit_rejection():
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
    review = _review_stateful_candidate_calls(
        None,
        None,
        lambda *_args: json.dumps(
            {
                "verdict": "reject",
                "reason": "test",
                "checks": {"facts_are_current": "false"},
                "failed_check": "facts_are_current",
                "evidence_ids": ["obs_1:/status"],
                "candidate_calls": candidate_calls,
            }
        ),
        "Retrieve record R-1.",
        [],
        candidate_calls,
        [],
        100,
    )

    assert review["attempted"] is True
    assert review["allowed"] is False
    assert review["failed_check"] == "facts_are_current"


def test_stateful_candidate_reviewer_abstains_from_an_unstructured_rejection():
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
    review = _review_stateful_candidate_calls(
        None,
        None,
        lambda *_args: json.dumps({"verdict": "reject", "reason": "test", "candidate_calls": candidate_calls}),
        "Retrieve record R-1.",
        [],
        candidate_calls,
        [],
        100,
    )

    assert review["allowed"] is True
    assert review["abstained"] is True


def test_stateful_candidate_reviewer_abstains_when_it_echoes_a_different_call():
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
    review = _review_stateful_candidate_calls(
        None,
        None,
        lambda *_args: json.dumps(
            {
                "verdict": "reject",
                "reason": "Wrong record.",
                "candidate_calls": [{"tool_name": "get_record", "arguments": {"record_id": "R-2"}}],
            }
        ),
        "Retrieve record R-1.",
        [],
        candidate_calls,
        [],
        100,
    )

    assert review["allowed"] is True
    assert review["abstained"] is True


def test_stateful_terminal_reviewer_rejects_empty_clarification_when_lookup_can_progress():
    def fake_generate(_model, _tokenizer, messages, _max_new_tokens):
        payload = json.loads(messages[-1]["content"].split("Input:", 1)[1])
        assert payload["candidate_decision"] == "no_tool"
        assert payload["candidate_missing_inputs"] == []
        assert payload["candidate_response_message"] == ""
        return '{"verdict":"reject","reason":"A grounded lookup can advance the request."}'

    review = _review_stateful_terminal_decision(
        None,
        None,
        fake_generate,
        "Update record R-1.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            }
        ],
        {"tool_decision": "no_tool", "missing_inputs": []},
        [{"tool_name": "authenticate", "arguments": {}, "outcome": "success", "observation": {"user_id": "U-1"}}],
        100,
    )

    assert review["attempted"] is True
    assert review["allowed"] is False


def test_stateful_terminal_reviewer_recovers_malformed_json_once():
    outputs = iter(
        [
            "The no-action decision should be rejected.",
            '{"verdict":"reject","reason":"A grounded lookup can advance the request."}',
        ]
    )
    review = _review_stateful_terminal_decision(
        None,
        None,
        lambda *_args: next(outputs),
        "Retrieve record R-1.",
        [],
        {"tool_decision": "no_tool", "missing_inputs": []},
        [],
        100,
    )

    assert review["allowed"] is False
    assert review["format_recovery"]["used"] is True


def test_stateful_terminal_reviewer_recovers_json_with_invalid_verdict_label_once():
    outputs = iter(
        [
            '{"verdict":"ask_user","reason":"The user must confirm the action."}',
            '{"verdict":"allow","reason":"The candidate already asks for confirmation."}',
        ]
    )
    review = _review_stateful_terminal_decision(
        None,
        None,
        lambda *_args: next(outputs),
        "Confirm the proposed action.",
        [],
        {"tool_decision": "ask_user", "missing_inputs": ["confirmation"]},
        [],
        100,
    )

    assert review["allowed"] is True
    assert review["format_recovery"]["used"] is True


def test_stateful_terminal_reviewer_accepts_only_a_matching_terminal_decision_alias():
    outputs = iter(
        [
            '{"verdict":"ask_user","reason":"Confirmation is still required."}',
            '{"verdict":"ask_user","reason":"Confirmation is still required."}',
        ]
    )
    review = _review_stateful_terminal_decision(
        None,
        None,
        lambda *_args: next(outputs),
        "Confirm the proposed action.",
        [],
        {"tool_decision": "ask_user", "missing_inputs": ["confirmation"]},
        [],
        100,
    )

    assert review["allowed"] is True
    assert review["verdict_alias_normalized"] is True


def test_stepwise_stateful_progress_repair_corrects_empty_clarification_to_lookup():
    outputs = iter(
        [
            '{"tool_decision":"no_tool","canonical_request":"Update record 101.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}',
            '{"tool_decision":"ask_user","canonical_request":"Update record 101.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}',
            '{"verdict":"reject","reason":"A grounded record lookup can advance the task."}',
            '{"tool_decision":"call","canonical_request":"Retrieve record 101.","slots_observed":[],"call_groups":[{"expected_call_count":1}],"tool_bindings":[{"tool_name":"get_record","call_count":1,"arguments":{"record_id":"101"},"evidence_spans":{"record_id":"101"}}],"missing_inputs":[]}',
            '{"verdict":"allow","reason":"The grounded lookup is the immediate next step."}',
        ]
    )

    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: next(outputs),
        "Current task: update record 101.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record by identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
        stateful_semantic_review=True,
        binding_request="Binding evidence:\nuser: update record 101.",
        execution_history=[
            {"tool_name": "authenticate", "arguments": {}, "outcome": "success", "observation": {"user_id": "U-1"}}
        ],
    )

    assert result["calls"][0]["tool_name"] == "get_record"
    assert result["calls"][0]["arguments"] == {"record_id": "101"}
    assert result["stateful_semantic_review"]["allowed"] is True
    assert result["stateful_progress_repair"]["binding_correction"]["used"] is True


def test_stateful_candidate_repair_preserves_a_semantic_response_after_rejection():
    candidate_calls = [{"tool_name": "get_record", "arguments": {"record_id": "R-1"}}]
    outputs = iter(
        [
            '{"tool_decision":"call","canonical_request":"Retrieve R-1.","slots_observed":[],"call_groups":[{"expected_call_count":1}],"tool_bindings":[{"tool_name":"get_record","call_count":1,"arguments":{"record_id":"R-1"},"evidence_spans":{"record_id":"R-1"}}],"missing_inputs":[]}',
            json.dumps(
                {
                    "verdict": "reject",
                    "reason": "Respond with the result instead.",
                    "checks": {"effect_not_already_completed": "false"},
                    "failed_check": "effect_not_already_completed",
                    "evidence_ids": ["obs_1:/record_id"],
                    "candidate_calls": candidate_calls,
                }
            ),
            '{"tool_decision":"no_tool","canonical_request":"Report the retrieved result.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[],"response_message":"The requested record is available."}',
            '{"verdict":"allow","reason":"The repaired response is a complete customer-facing answer."}',
        ]
    )
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: next(outputs),
        "Retrieve record R-1.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record by identifier.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]},
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
        stateful_semantic_review=True,
        binding_request="Binding evidence:\nuser: retrieve record R-1.",
    )

    assert result["calls"] == []
    frame = result["tool_binding_plan"]["capability_plan"]["semantic_input_frame"]
    assert frame["response_message"] == "The requested record is available."
    assert result["stateful_semantic_review"]["repair_terminal_review"]["allowed"] is True


def test_stepwise_stateful_terminal_review_runs_before_any_tool_observation():
    outputs = iter(
        [
            '{"tool_decision":"no_tool","canonical_request":"Retrieve record 101.","slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}',
            '{"verdict":"reject","reason":"A grounded lookup can advance the request."}',
            '{"tool_decision":"call","canonical_request":"Retrieve record 101.","slots_observed":[],"call_groups":[{"expected_call_count":1}],"tool_bindings":[{"tool_name":"get_record","call_count":1,"arguments":{"record_id":"101"},"evidence_spans":{"record_id":"101"}}],"missing_inputs":[]}',
            '{"verdict":"allow","reason":"The lookup is a grounded immediate transition."}',
        ]
    )

    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: next(outputs),
        "Current task: retrieve record 101.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record by identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
        stateful_semantic_review=True,
        binding_request="Binding evidence:\nuser: retrieve record 101.",
    )

    assert result["calls"][0]["tool_name"] == "get_record"
    assert result["stateful_semantic_review"]["terminal_review"]["allowed"] is False


def test_stateful_single_call_execution_preserves_model_order_without_routing():
    plan, info = _limit_stateful_plan_to_one_call(
        {
            "tool_decision": "call",
            "calls": [
                {"id": "call_4", "tool_name": "lookup_first", "arguments": {"id": "A"}},
                {"id": "call_5", "tool_name": "lookup_second", "arguments": {"id": "B"}},
            ],
        }
    )

    assert plan["calls"] == [{"id": "call_1", "tool_name": "lookup_first", "arguments": {"id": "A"}}]
    assert info["used"] is True
    assert info["selected_call"]["tool_name"] == "lookup_first"
    assert info["deferred_calls"] == [{"tool_name": "lookup_second", "arguments": {"id": "B"}}]


def test_stateful_semantic_only_mode_does_not_fall_back_to_heuristic_tool_routing():
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: '{"tool_decision":"call","canonical_request":"Retrieve R-1.","tool_bindings":[],"missing_inputs":[]}',
        "Retrieve record R-1.",
        [
            {
                "name": "get_record",
                "description": "Retrieve a record.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
    )

    assert result["calls"] == []
    assert result["tool_binding_plan"]["tool_decision"] == "no_tool"
    assert result["stateful_semantic_only"] is True


def test_stateful_semantic_only_adjudicates_nonliteral_schema_enum_with_provenance():
    responses = iter(
        [
            json.dumps(
                {
                    "tool_decision": "call",
                    "canonical_request": "Retrieve options for one outbound journey.",
                    "slots_observed": [],
                    "call_groups": [{"expected_call_count": 1}],
                    "tool_bindings": [
                        {
                            "tool_name": "lookup_trip_options",
                            "call_count": 1,
                            "argument_groups": [{"arguments": {"trip_kind": "single"}}],
                            "evidence_spans": {"trip_kind": "single"},
                        }
                    ],
                    "missing_inputs": [],
                }
            ),
            json.dumps(
                {
                    "approved": [
                        {
                            "tool_name": "lookup_trip_options",
                            "group_index": 0,
                            "argument_name": "trip_kind",
                            "value": "single",
                            "evidence_span": "one outbound journey",
                        }
                    ]
                }
            ),
        ]
    )
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: next(responses),
        "Look up travel options for one outbound journey on Monday.",
        [
            {
                "name": "lookup_trip_options",
                "description": "Retrieve options for a single or return trip.",
                "parameters": {
                    "type": "object",
                    "properties": {"trip_kind": {"type": "string", "enum": ["single", "return"]}},
                    "required": ["trip_kind"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
    )

    assert result["calls"][0]["tool_name"] == "lookup_trip_options"
    assert result["calls"][0]["arguments"] == {"trip_kind": "single"}
    assert result["semantic_enum_grounding"]["used"] is True
    assert result["semantic_enum_grounding"]["approved"][0]["evidence_span"] == "one outbound journey"


def test_stateful_semantic_enum_adjudication_cannot_create_non_enum_argument_values():
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: json.dumps(
            {
                "tool_decision": "call",
                "canonical_request": "Retrieve options.",
                "slots_observed": [],
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "lookup_trip_options",
                        "call_count": 1,
                        "argument_groups": [{"arguments": {"trip_kind": "single", "user_id": "invented"}}],
                        "evidence_spans": {"trip_kind": "single", "user_id": "invented"},
                    }
                ],
                "missing_inputs": [],
            }
        ),
        "Look up travel options for one outbound journey.",
        [
            {
                "name": "lookup_trip_options",
                "description": "Retrieve options for a single or return trip.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trip_kind": {"type": "string", "enum": ["single", "return"]},
                        "user_id": {"type": "string"},
                    },
                    "required": ["trip_kind", "user_id"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
    )

    assert result["calls"] == []
    assert result["semantic_enum_grounding"]["candidate_count"] == 1
    assert result["semantic_enum_grounding"]["used"] is False


def test_stateful_semantic_only_recovers_unique_schema_grounded_read_only_transition_after_malformed_output():
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: "analysis without a JSON frame",
        "Transcript:\nuser: My user ID is mia_li_3668.",
        [
            {
                "name": "get_user_details",
                "description": "Retrieve user details by user id.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
            {
                "name": "search_flights",
                "description": "Search flights by origin and destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            },
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
    )

    assert result["verification_ok"] is True
    assert result["calls"][0]["tool_name"] == "get_user_details"
    assert result["calls"][0]["arguments"] == {"user_id": "mia_li_3668"}
    assert result["stateful_schema_recovery"]["used"] is True


def test_stateful_semantic_only_advances_unique_read_only_transition_before_unnecessary_clarification():
    result = plan_and_compile_goal_graph(
        None,
        None,
        lambda *_args: (
            '{"tool_decision":"ask_user","canonical_request":"Need passenger details.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":["passenger_details"]}'
        ),
        "Transcript:\nuser: My user ID is mia_li_3668.",
        [
            {
                "name": "get_user_details",
                "description": "Retrieve user details by user id.",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                    "required": ["user_id"],
                },
            },
            {
                "name": "search_flights",
                "description": "Search flights by origin and destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            },
        ],
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        stateful_semantic_only=True,
    )

    assert result["verification_ok"] is True
    assert result["calls"][0]["tool_name"] == "get_user_details"
    assert result["calls"][0]["arguments"] == {"user_id": "mia_li_3668"}
    assert result["stateful_schema_recovery"]["used"] is True


def test_semantic_respond_decision_is_treated_as_a_request_for_user_input():
    plan = build_tool_binding_plan(
        "Update record R-1.",
        [
            {
                "name": "update_record",
                "description": "Update a record.",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "respond",
                "canonical_request": "Please provide the account email.",
                "tool_bindings": [],
                "missing_inputs": ["email"],
            }
        },
    )

    assert plan["tool_decision"] == "ask_user"




def test_stateful_raw_safety_filter_keeps_grounded_lookup_only():
    tools = [
        {
            "name": "get_record",
            "description": "Retrieve a record by identifier.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
        {"name": "list_everything", "description": "List all records.", "parameters": {"type": "object"}},
        {
            "name": "transfer_to_human",
            "description": "Escalate to a human.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    ]
    plan, info = _drop_stateful_unverified_fallback_calls(
        {
            "tool_decision": "call",
            "calls": [
                {"tool_name": "list_everything", "arguments": {}},
                {"tool_name": "transfer_to_human", "arguments": {"summary": "record R-1"}},
                {"tool_name": "get_record", "arguments": {"record_id": "R-1"}},
            ],
        },
        tools,
        "Please retrieve record R-1.",
    )

    assert info["used"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["get_record"]


def test_model_binding_allows_grounded_state_gathering_before_mutation():
    tools = [
        {
            "name": "get_product_details",
            "description": "Retrieve details for one product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        }
    ]
    plan = build_tool_binding_plan(
        "Exchange product 6066914160 for a different configuration.",
        tools,
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "canonical_request": "Retrieve product details before exchanging it.",
                "tool_bindings": [
                    {
                        "tool_name": "get_product_details",
                        "call_count": 1,
                        "argument_groups": [{"product_id": "6066914160"}],
                        "evidence_spans": {"product_id": "6066914160"},
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["tool_name"] == "get_product_details"


def test_stateful_terminal_repair_replans_to_grounded_lookup():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"call","canonical_request":"Retrieve product details.",'
            '"slots_observed":[],"call_groups":[{"expected_call_count":1}],'
            '"tool_bindings":[{"tool_name":"get_product_details","call_count":1,'
            '"argument_groups":[{"product_id":"6066914160"}],'
            '"evidence_spans":{"product_id":"6066914160"}}],"missing_inputs":[]}'
        )

    tools = [
        {
            "name": "get_product_details",
            "description": "Retrieve details for one product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        }
    ]
    repaired, info = _stateful_progress_repair_plan(
        None,
        None,
        fake_generate,
        "Transcript:\nuser: Exchange product 6066914160.",
        "Binding evidence:\nuser: Exchange product 6066914160.",
        tools,
        [{"tool_name": "get_order", "arguments": {"order_id": "O-1"}, "outcome": "success"}],
        100,
    )

    assert info["used"] is True
    assert repaired is not None
    assert repaired["calls"][0]["tool_name"] == "get_product_details"
    assert _stateful_plan_defers_to_user_or_terminal({"tool_decision": "ask_user", "calls": []}, tools)


def test_stateful_progress_repair_corrects_schema_rejected_semantic_binding_once():
    outputs = iter(
        [
            (
                '{"tool_decision":"call","canonical_request":"Retrieve product details.",'
                '"slots_observed":[],"call_groups":[{"expected_call_count":1}],'
                '"tool_bindings":[{"tool_name":"get_product_details","call_count":1,'
                '"argument_groups":[{}]}],"missing_inputs":[]}'
            ),
            (
                '{"tool_decision":"call","canonical_request":"Retrieve product details.",'
                '"slots_observed":[],"call_groups":[{"expected_call_count":1}],'
                '"tool_bindings":[{"tool_name":"get_product_details","call_count":1,'
                '"argument_groups":[{"product_id":"6066914160"}],'
                '"evidence_spans":{"product_id":"6066914160"}}],"missing_inputs":[]}'
            ),
        ]
    )
    tools = [
        {
            "name": "get_product_details",
            "description": "Retrieve details for one product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        }
    ]

    repaired, info = _stateful_progress_repair_plan(
        None,
        None,
        lambda *_args: next(outputs),
        "Transcript:\nuser: Exchange product 6066914160.",
        "Binding evidence:\nuser: Exchange product 6066914160.",
        tools,
        [],
        100,
        semantic_only=True,
        reviewer_feedback="A grounded lookup is needed.",
    )

    assert repaired is not None
    assert repaired["calls"][0]["tool_name"] == "get_product_details"
    assert info["binding_correction"]["attempted"] is True
    assert info["binding_correction"]["used"] is True


def test_stateful_terminal_repair_retries_a_rejected_no_action_as_grounded_question():
    outputs = iter(
        [
            (
                '{"tool_decision":"no_tool","canonical_request":"Complete the request.",'
                '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[],'
                '"response_message":"I cannot proceed."}'
            ),
            (
                '{"tool_decision":"ask_user","canonical_request":"Complete the request.",'
                '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":["confirmation"],'
                '"clarification_message":"Please confirm the action details."}'
            ),
        ]
    )

    repaired, info = _stateful_progress_repair_plan(
        None,
        None,
        lambda *_args: next(outputs),
        "Current task: complete the requested operation.",
        "User: Please complete the requested operation after I confirm it.",
        [
            {
                "name": "submit_operation",
                "description": "Complete the requested operation after confirmation.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        [],
        100,
        semantic_only=True,
        reviewer_feedback="The prior no-action decision was rejected because confirmation is required.",
        no_action_was_rejected=True,
    )

    assert repaired is not None
    assert repaired["tool_decision"] == "ask_user"
    assert repaired["missing_inputs"] == ["confirmation"]
    assert info["binding_correction"]["used"] is True


def test_stepwise_stateful_progress_repair_uses_new_model_bound_action():
    outputs = iter(
        [
            (
                '{"tool_decision":"call","canonical_request":"Calculate factorial of 5.",'
                '"slots_observed":[],"call_groups":[{"intent":"factorial","expected_call_count":1}],'
                '"tool_bindings":[{"tool_name":"math.factorial","intent":"factorial","call_count":1,'
                '"argument_groups":[{"arguments":{"number":5},"evidence_spans":{"number":"5"}}]}],'
                '"missing_inputs":[]}'
            ),
            (
                '{"tool_decision":"call","canonical_request":"Square 6.",'
                '"slots_observed":[],"call_groups":[{"intent":"square","expected_call_count":1}],'
                '"tool_bindings":[{"tool_name":"math.square","intent":"square","call_count":1,'
                '"argument_groups":[{"arguments":{"number":6},"evidence_spans":{"number":"6"}}]}],'
                '"missing_inputs":[]}'
            ),
        ]
    )

    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return next(outputs)

    tools = [
        {
            "name": "math.factorial",
            "description": "Calculate the factorial of a number.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        },
        {
            "name": "math.square",
            "description": "Square a number.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        },
        {
            "name": "respond",
            "description": "Send a customer-facing message.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    ]

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Calculate factorial of 5 and square 6.",
        tools,
        max_new_tokens=100,
        repair_attempts=0,
        stateful=True,
        execution_history=[
            {"tool_name": "math.factorial", "arguments": {"number": 5}, "outcome": "success"}
        ],
    )

    assert result["calls"][0]["tool_name"] == "math.square"
    assert result["calls"][0]["arguments"] == {"number": 6}
    assert result["stateful_progress_filter"]["used"] is True
    assert result["stateful_progress_repair"]["used"] is True


def test_tool_binding_drops_identifier_role_mismatches():
    tools = [
        {
            "name": "get_user_details",
            "description": "Get user details.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
        {
            "name": "get_order_details",
            "description": "Get order details.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    ]

    kept, dropped = _filter_schema_value_incompatible_calls(
        tools,
        [
            {"tool_name": "get_user_details", "arguments": {"user_id": "#W5442520"}},
            {"tool_name": "get_order_details", "arguments": {"order_id": "#W5442520"}},
        ],
    )

    assert [call["tool_name"] for call in kept] == ["get_order_details"]
    assert dropped[0]["reason"].startswith("value role 'order'")

    kept_wrapped, dropped_wrapped = _filter_schema_value_incompatible_calls(
        [{"type": "function", "function": tools[0]}],
        [{"tool_name": "get_user_details", "arguments": {"user_id": "#W5442520"}}],
    )
    assert kept_wrapped == []
    assert dropped_wrapped[0]["reason"].startswith("value role 'order'")


def test_identifier_role_uses_parameter_name_over_related_entity_description():
    tools = [
        {
            "name": "modify_payment",
            "description": "Change the payment method on an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "payment_method_id": {
                        "type": "string",
                        "description": "The payment method ID to use for the order.",
                    },
                },
                "required": ["order_id", "payment_method_id"],
            },
        }
    ]

    kept, dropped = _filter_schema_value_incompatible_calls(
        tools,
        [
            {
                "tool_name": "modify_payment",
                "arguments": {"order_id": "#W5442520", "payment_method_id": "#W5442520"},
            }
        ],
    )

    assert kept == []
    assert dropped[0]["reason"] == "value role 'order' is incompatible with identifier slot 'payment_method_id'"


def test_tool_binding_drops_copied_identifier_values_across_roles():
    tools = [
        {
            "name": "update_reservation_flights",
            "description": "Update reservation flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "string"},
                    "payment_id": {"type": "string"},
                },
                "required": ["reservation_id", "payment_id"],
            },
        }
    ]

    kept, dropped = _filter_schema_value_incompatible_calls(
        tools,
        [
            {
                "tool_name": "update_reservation_flights",
                "arguments": {"reservation_id": "K67C4W", "payment_id": "K67C4W"},
            }
        ],
    )

    assert kept == []
    assert "incompatible identifier slots" in dropped[0]["reason"]


def test_tool_binding_does_not_treat_dates_as_identifier_slots():
    tools = [
        {
            "name": "search_direct_flight",
            "description": "Search direct flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "The flight date."},
                },
                "required": ["origin", "destination", "date"],
            },
        }
    ]

    kept, dropped = _filter_schema_value_incompatible_calls(
        tools,
        [
            {
                "tool_name": "search_direct_flight",
                "arguments": {"origin": "DFW", "destination": "EWR", "date": "today"},
            }
        ],
    )

    assert dropped == []
    assert kept[0]["arguments"]["date"] == "today"


def test_fallback_graph_marks_ungrounded_schema_defaults_as_policy_defaults():
    request = (
        "clone the repo git@github.com:zelarhq/nodejs-welcome.git, analyze it, "
        "create dockerfile and kubernetes yamls, then commit / push to repo"
    )
    tools = benchmark_compile_tools(
        [
            {
                "name": "push_git_changes_to_github",
                "description": "Commit and push git changes to GitHub.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_name": {"type": "string"},
                        "commit_message": {"type": "string"},
                        "branch_name": {"type": "string", "default": "main"},
                        "force_push": {"type": "boolean", "default": False},
                    },
                    "required": ["directory_name", "commit_message"],
                },
            }
        ]
    )
    runtime = GoalGraphRuntime(tools)
    graph = _graph_from_binding_plan(
        runtime,
        request,
        {
            "calls": [
                {
                    "tool_name": "push_git_changes_to_github",
                    "arguments": {
                        "directory_name": "nodejs-welcome",
                        "commit_message": "Update changes",
                        "branch_name": "main",
                        "force_push": False,
                    },
                    "argument_evidence": {
                        "directory_name": "nodejs-welcome",
                        "commit_message": "commit / push to repo",
                        "branch_name": "main",
                        "force_push": "false",
                    },
                    "missing_arguments": [],
                }
            ]
        },
    )

    inputs = graph["nodes"][0]["inputs"]
    assert inputs["branch_name"]["source"] == "policy_default"
    assert inputs["branch_name"]["status"] == "defaulted"
    assert inputs["force_push"]["source"] == "policy_default"
    assert inputs["force_push"]["status"] == "defaulted"
    assert runtime.compile(graph, request, allow_side_effects=True).verification.ok


def test_extract_goal_graph_prefers_outer_graph_over_nested_objects():
    text = (
        'analysis... assistantfinal{"goal":"Compute area","nodes":[{"id":"n1",'
        '"kind":"retrieve","capability":"calculate_area","inputs":{"base":'
        '{"value":6,"source":"query","evidence":"6cm","status":"resolved"},'
        '"height":{"value":10,"source":"query","evidence":"10cm","status":"resolved"}},'
        '"outputs":["area"],"depends_on":[],"must_be_unique":false,"risk":"read_only",'
        '"authorized":false,"policy_evidence":[],"expected_effect":{}}],'
        '"clarification_needed":false,"clarification_reasons":[]}'
    )

    parsed, error = extract_goal_graph_json_object(text)

    assert error is None
    assert parsed["goal"] == "Compute area"
    assert parsed["nodes"][0]["capability"] == "calculate_area"


def test_extract_semantic_frame_repairs_missing_tool_bindings_array_bracket():
    text = (
        'analysis... assistantfinal{"tool_decision":"call","canonical_request":"Check alarm.",'
        '"slots_observed":[],"call_groups":[{"name":"QueryAlarm","expected_call_count":1}],'
        '"tool_bindings":[{"tool_name":"QueryAlarm","call_count":1,"argument_groups":[{'
        '"arguments":{"token":"tok","time":"2023-03-20 06:30:00"},'
        '"evidence_spans":{"token":"{\'token\': \'tok\'}","time":"March 20th, 2023 at 6:30AM"}}]},'
        '"missing_inputs":[]}'
    )

    parsed, error = extract_semantic_frame_json_object(text)

    assert error is None
    assert parsed["tool_bindings"][0]["tool_name"] == "QueryAlarm"
    assert parsed["tool_bindings"][0]["argument_groups"][0]["arguments"]["token"] == "tok"


def test_extract_goal_graph_wraps_single_node_object():
    text = '{"id":"n1","kind":"retrieve","capability":"weather.get","inputs":{}}'

    parsed, error = extract_goal_graph_json_object(text)

    assert error is None
    assert parsed["nodes"][0]["capability"] == "weather.get"


def test_binder_fallback_synthesizes_graph_when_model_omits_graph_json():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return '{"number":{"value":5,"source":"query","evidence":"5","status":"resolved"}}'

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Calculate the factorial of 5 using math functions.",
        [
            {
                "name": "math.factorial",
                "description": "Return the factorial of a number.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
        use_binder_fallback=True,
        planner_mode="one_shot",
    )

    assert result["verification_ok"]
    assert result["binder_fallback"]["used"]
    assert result["calls"][0]["tool_name"] == "math.factorial"


def test_bfcl_result_calls_falls_back_to_verified_binding_plan():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "calls": [
                {"tool_name": "repo.clone", "arguments": {"url": "https://example.test/repo.git"}},
                {"tool_name": "repo.push", "arguments": {"image": "service"}, "missing_arguments": []},
                {"tool_name": "repo.skip", "missing_arguments": ["target"]},
            ]
        },
    }

    assert [call["tool_name"] for call in bfcl_result_calls(result)] == ["repo.clone", "repo.push"]


def test_extract_semantic_frame_prefers_outer_frame():
    text = (
        'analysis... {"canonical_request":"factorial five","slots_observed":'
        '[{"role":"number","value":5,"evidence_span":"5"}],"call_groups":'
        '[{"intent":"calculate factorial","expected_call_count":1}],'
        '"missing_inputs":[]}'
    )

    parsed, error = extract_semantic_frame_json_object(text)

    assert error is None
    assert parsed["canonical_request"] == "factorial five"
    assert parsed["slots_observed"][0]["role"] == "number"


def test_extract_semantic_frame_repairs_missing_argument_groups_bracket():
    text = (
        'assistantfinal{"canonical_request":"Retrieve protein sequences.",'
        '"slots_observed":[],"call_groups":[{"name":"get_protein_sequence",'
        '"expected_call_count":2}],"tool_bindings":[{"tool_name":"get_protein_sequence",'
        '"call_count":2,"argument_groups":[{"arguments":{"gene":"BRCA1","species":"Homo sapiens"},'
        '"evidence_spans":{"gene":"BRCA1","species":"Homo sapiens"}},'
        '{"arguments":{"gene":"BRCA2","species":"Pan troglodytes"},'
        '"evidence_spans":{"gene":"BRCA2","species":"Pan troglodytes"}}}],"missing_inputs":[]}'
    )

    parsed, error = extract_semantic_frame_json_object(text)

    assert error is None
    assert parsed["tool_bindings"][0]["call_count"] == 2
    assert len(parsed["tool_bindings"][0]["argument_groups"]) == 2


def test_stepwise_pipeline_uses_one_model_call_for_semantic_frame_then_binder():
    calls = []

    def fake_generate(_model, _tokenizer, messages, _max_new_tokens):
        calls.append(messages)
        return (
            '{"canonical_request":"Calculate factorial of 5.",'
            '"slots_observed":[{"role":"number","value":5,"value_type":"number",'
            '"evidence_span":"5","status":"explicit","confidence":1.0}],'
            '"call_groups":[{"intent":"calculate factorial","unit_of_work":"factorial",'
            '"requested_entities":["5"],"expected_call_count":1,"result_count":null,'
            '"can_use_batch_tool_if_available":true}],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Calculate the factorial of 5 using math functions.",
        [
            {
                "name": "math.factorial",
                "description": "Return the factorial of a number.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["planner_mode"] == "stepwise"
    assert len(calls) == 1
    assert result["steps"][0]["step"] == "semantic_frame"
    assert result["steps"][1]["step"] == "tool_binding"
    assert result["verification_ok"]
    assert result["calls"][0]["tool_name"] == "math.factorial"


def test_stepwise_pipeline_can_use_verified_call_skeleton_for_complex_order():
    calls = []

    def fake_generate(_model, _tokenizer, messages, _max_new_tokens):
        calls.append(messages)
        if len(calls) == 1:
            return (
                '{"canonical_request":"Do alpha, after that beta.",'
                '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
            )
        return (
            '{"ordered_calls":[{"tool_name":"alpha.do","evidence_span":"alpha"},'
            '{"tool_name":"beta.do","evidence_span":"beta"}],"confidence":0.9}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Do alpha, after that beta.",
        benchmark_compile_tools(
            [
                {
                    "name": "alpha.do",
                    "description": "Handle alpha.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": "beta.do",
                    "description": "Handle beta.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            ]
        ),
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert len(calls) == 2
    assert result["verification_ok"]
    assert result["call_skeleton_output"]["used"] is True
    assert [call["tool_name"] for call in result["calls"]] == ["alpha.do", "beta.do"]


def test_extract_call_skeleton_repairs_missing_final_braces():
    text = (
        'analysis... {"ordered_calls":[{"tool_name":"alpha.do",'
        '"evidence_span":"alpha"},{"tool_name":"beta.do",'
        '"evidence_span":"beta"}],"confidence":0.9'
    )

    parsed, error = extract_call_skeleton_json_object(text)

    assert error is None
    assert [item["tool_name"] for item in parsed["ordered_calls"]] == ["alpha.do", "beta.do"]


def test_extract_call_skeleton_ignores_echoed_schema_placeholder():
    text = (
        'analysis Schema:{"ordered_calls":[{"tool_name":"available tool name",'
        '"evidence_span":"exact request span supporting this call",'
        '"arguments":{"schema_arg":"grounded value"},'
        '"evidence_spans":{"schema_arg":"exact evidence span"}}],"confidence":0.0}'
        ' assistantfinal{"ordered_calls":[{"tool_name":"alpha.do",'
        '"evidence_span":"alpha"}],"confidence":0.9}'
    )

    parsed, error = extract_call_skeleton_json_object(text)

    assert error is None
    assert parsed["ordered_calls"] == [{"tool_name": "alpha.do", "evidence_span": "alpha"}]


def test_call_skeleton_expands_repeated_weather_dates_and_normalizes_order():
    request = (
        '"Can you tell me what the weather was like in New York City on 2020-12-25 '
        "and 2021-01-01, and also provide the historical weather data for the "
        "geographical coordinates (40.7128, -74.0060) on 2021-01-15? Additionally, "
        'can you forecast the weather for the same coordinates for the next 10 days?"'
    )
    tools = benchmark_compile_tools(
        [
            {
                "name": "weather.get_forecast_by_coordinates",
                "description": "Get the weather forecast for a specific geographical coordinates.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "coordinates": {
                            "type": "tuple",
                            "items": {"type": "float"},
                            "description": "The geographical coordinates for which to retrieve the weather.",
                        },
                        "days_ahead": {
                            "type": "integer",
                            "description": "Number of days to forecast from current date.",
                        },
                    },
                    "required": ["coordinates"],
                },
            },
            {
                "name": "weather.get_by_coordinates_date",
                "description": "Retrieves the historical weather data based on coordinates and date.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "coordinates": {
                            "type": "tuple",
                            "items": {"type": "float"},
                            "description": "The geographical coordinates for which to retrieve the weather.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date for which to retrieve the historical weather data in the format YYYY-MM-DD.",
                        },
                    },
                    "required": ["coordinates", "date"],
                },
            },
            {
                "name": "weather.get_by_city_date",
                "description": "Retrieves the historical weather data based on city and date.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city for which to retrieve the weather.",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date for which to retrieve the historical weather data in the format YYYY-MM-DD.",
                        },
                    },
                    "required": ["city", "date"],
                },
            },
        ]
    )
    parsed = {
        "ordered_calls": [
            {
                "tool_name": "weather.get_by_city_date",
                "evidence_span": "New York City on 2020-12-25",
                "arguments": {"city": "New York City", "date": "2020-12-25"},
                "evidence_spans": {"city": "New York City", "date": "2020-12-25"},
            },
            {
                "tool_name": "weather.get_by_city_date",
                "evidence_span": "New York City on 2021-01-01",
                "arguments": {"city": "New York City", "date": "2021-01-01"},
                "evidence_spans": {"city": "New York City", "date": "2021-01-01"},
            },
            {
                "tool_name": "weather.get_by_coordinates_date",
                "evidence_span": "geographical coordinates (40.7128, -74.0060) on 2021-01-15",
                "arguments": {"coordinates": [40.7128, -74.0060], "date": "2021-01-15"},
                "evidence_spans": {"coordinates": "(40.7128, -74.0060)", "date": "2021-01-15"},
            },
            {
                "tool_name": "weather.get_forecast_by_coordinates",
                "evidence_span": "forecast the weather for the same coordinates for the next 10 days",
                "arguments": {"coordinates": [40.7128, -74.0060], "days_ahead": 10},
                "evidence_spans": {"coordinates": "(40.7128, -74.0060)", "days_ahead": "10 days"},
            },
        ],
        "confidence": 0.9,
    }

    skeleton_plan = _binding_plan_from_call_skeleton(request, tools, parsed, {"calls": []})

    assert skeleton_plan is not None
    assert [call["tool_name"] for call in skeleton_plan["calls"]] == [
        "weather.get_by_coordinates_date",
        "weather.get_by_city_date",
        "weather.get_by_city_date",
        "weather.get_forecast_by_coordinates",
    ]
    assert skeleton_plan["calls"][0]["arguments"] == {
        "coordinates": [40.7128, -74.006],
        "date": "2021-01-15",
    }
    assert skeleton_plan["calls"][2]["arguments"]["date"] == "2021-01-01"


def test_call_skeleton_can_recover_action_request_with_no_deterministic_calls():
    tools = [{"name": "alpha.do"}, {"name": "beta.do"}]

    assert _should_generate_call_skeleton("Find a 3 bedroom villa in San Diego.", tools, {"calls": []})
    assert not _should_generate_call_skeleton("hello", tools, {"calls": []})
    assert not _should_generate_call_skeleton("Do not call or use any tool.", tools, {"calls": []})


def test_call_skeleton_can_repair_duplicate_required_slot_audit_with_verified_arguments():
    request = "Calculate the final speed of an object dropped from 100 m without air resistance."
    tools = benchmark_compile_tools(
        [
            {
                "name": "calculate_final_speed",
                "description": "Calculate final speed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "initial_velocity": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["initial_velocity", "height"],
                },
            }
        ]
    )
    base_plan = build_tool_binding_plan(request, tools)

    assert not base_plan["calls"]
    assert _should_generate_call_skeleton(request, tools, base_plan)

    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "calculate_final_speed",
                    "evidence_span": "dropped from 100 m",
                    "arguments": {"initial_velocity": 0, "height": 100},
                    "evidence_spans": {"initial_velocity": "dropped", "height": "100 m"},
                }
            ],
            "confidence": 0.9,
        },
        base_plan,
    )

    assert skeleton_plan is not None
    assert [call["tool_name"] for call in skeleton_plan["calls"]] == ["calculate_final_speed"]
    assert skeleton_plan["calls"][0]["arguments"] == {"initial_velocity": 0, "height": 100}


def test_call_skeleton_runs_for_single_command_executor_with_independent_actions():
    tools = [
        {
            "name": "cmd_controller.execute",
            "description": "Execute a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command to execute."}},
                "required": ["command"],
            },
        }
    ]
    base_plan = {
        "model_tool_binding": {"used": True, "accepted": True},
        "calls": [
            {
                "tool_name": "cmd_controller.execute",
                "arguments": {"command": "dir C:\\ && echo > C:\\testing.txt"},
            }
        ],
    }
    skeleton_plan = {
        "calls": [
            {"tool_name": "cmd_controller.execute", "arguments": {"command": "list file in c drive"}},
            {"tool_name": "cmd_controller.execute", "arguments": {"command": "make file called testing.txt"}},
        ]
    }

    assert _should_generate_call_skeleton(
        "list file in c drive and make file called testing.txt",
        tools,
        base_plan,
    )
    assert _prefer_skeleton_binding_plan(
        "list file in c drive and make file called testing.txt",
        base_plan,
        skeleton_plan,
    )


def test_call_skeleton_accepts_verified_command_executor_steps():
    request = "list file in c drive and make file called testing.txt"
    tools = benchmark_compile_tools(
        [
            {
                "name": "cmd_controller.execute",
                "description": "Execute a shell command on the user computer.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command line instruction to execute.",
                        }
                    },
                    "required": ["command"],
                },
            }
        ]
    )

    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "cmd_controller.execute",
                    "evidence_span": "list file in c drive",
                    "arguments": {"command": "dir C:\\"},
                },
                {
                    "tool_name": "cmd_controller.execute",
                    "evidence_span": "make file called testing.txt",
                    "arguments": {"command": "echo. > C:\\testing.txt"},
                },
            ]
        },
        {"calls": []},
    )

    assert skeleton_plan is not None
    assert [call["arguments"]["command"] for call in skeleton_plan["calls"]] == [
        "dir C:\\",
        "echo. > C:\\testing.txt",
    ]


def test_call_skeleton_runs_for_non_english_single_control_tool_no_call():
    tools = [
        {
            "name": "ControlAppliance.execute",
            "description": "Control or execute actions for appliances and devices.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ]

    assert _should_generate_call_skeleton(
        "거실, 에어컨, 실행하고, 침실, 공기청정기, 중지해줘.",
        tools,
        {"calls": []},
    )
    assert _should_generate_call_skeleton(
        "거실, 에어컨, 실행하고, 침실, 공기청정기, 중지해줘.",
        tools,
        {"tool_decision": "ask_user", "missing_inputs": ["command"], "calls": []},
    )


def test_call_skeleton_accepts_verified_semantic_synonym_route():
    request = "Could you add 1 and 2 together and also tell me the current time in 'Europe/Berlin' including the date?"
    tools = benchmark_compile_tools(
        [
            {
                "name": "sum",
                "description": "Sums two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "getCurrentTime",
                "description": "Get the current time for a timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                        "include_date": {"type": "boolean"},
                    },
                    "required": ["timezone", "include_date"],
                },
            },
        ]
    )

    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "sum",
                    "evidence_span": "add 1 and 2 together",
                    "arguments": {"a": 1, "b": 2},
                    "evidence_spans": {"a": "1", "b": "2"},
                },
                {
                    "tool_name": "getCurrentTime",
                    "evidence_span": "current time in 'Europe/Berlin' including the date",
                    "arguments": {"timezone": "Europe/Berlin", "include_date": True},
                    "evidence_spans": {
                        "timezone": "'Europe/Berlin'",
                        "include_date": "including the date",
                    },
                },
            ]
        },
        {"calls": [{"tool_name": "getCurrentTime", "arguments": {"timezone": "Europe/Berlin"}}]},
    )

    assert skeleton_plan is not None
    assert [call["tool_name"] for call in skeleton_plan["calls"]] == ["sum", "getCurrentTime"]


def test_call_skeleton_repairs_unverified_derived_numeric_literals_to_grounded_pairs():
    request = (
        "Can you calculate the highest common factor of the pair of numbers (45, 60) and then use that "
        "result to find the highest common factor with another pair of numbers (90, 120)? Please also "
        "find the highest common factor of the pair (36, 48) and then find the highest common factor "
        "of that result with the pair (72, 96)."
    )
    tools = benchmark_compile_tools(
        [
            {
                "name": "math.hcf",
                "description": "Calculate the highest common factor of two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number1": {"type": "integer"},
                        "number2": {"type": "integer"},
                    },
                    "required": ["number1", "number2"],
                },
            }
        ]
    )

    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {"tool_name": "math.hcf", "evidence_span": "(45, 60)", "arguments": {"number1": 45, "number2": 60}},
                {"tool_name": "math.hcf", "evidence_span": "(90, 120)", "arguments": {"number1": 15, "number2": 90}},
                {"tool_name": "math.hcf", "evidence_span": "(90, 120)", "arguments": {"number1": 15, "number2": 120}},
                {"tool_name": "math.hcf", "evidence_span": "(36, 48)", "arguments": {"number1": 36, "number2": 48}},
                {"tool_name": "math.hcf", "evidence_span": "(72, 96)", "arguments": {"number1": 12, "number2": 72}},
                {"tool_name": "math.hcf", "evidence_span": "(72, 96)", "arguments": {"number1": 12, "number2": 96}},
            ]
        },
        {"calls": []},
    )

    assert skeleton_plan is not None
    assert [call["arguments"] for call in skeleton_plan["calls"]] == [
        {"number1": 45, "number2": 60},
        {"number1": 90, "number2": 120},
        {"number1": 36, "number2": 48},
        {"number1": 72, "number2": 96},
    ]
    assert not _evidence_in_request("12", request)
    assert not _evidence_in_request("15", request)


def test_call_skeleton_does_not_resurrect_verified_no_tool_hard_conflict():
    request = "How much gas is generated from heating a 2 m³ closed chamber with air at 25°C to 100°C?"
    tools = benchmark_compile_tools(
        [
            {
                "name": "thermodynamics.calc_gas_pressure",
                "description": "Calculate gas pressure in a closed chamber due to heating.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "volume": {"type": "number"},
                        "initial_temperature": {"type": "number"},
                        "final_temperature": {"type": "number"},
                    },
                    "required": ["volume", "initial_temperature", "final_temperature"],
                },
            }
        ]
    )
    base_plan = build_tool_binding_plan(
        request,
        tools,
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "tool_bindings": [
                    {
                        "tool_name": "thermodynamics.calc_gas_pressure",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "volume": 2,
                                    "initial_temperature": 25,
                                    "final_temperature": 100,
                                },
                                "evidence_spans": {
                                    "volume": "2 m³",
                                    "initial_temperature": "25°C",
                                    "final_temperature": "100°C",
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert base_plan["tool_decision"] == "no_tool"
    assert not _should_generate_call_skeleton(request, tools, base_plan)
    assert (
        _binding_plan_from_call_skeleton(
            request,
            tools,
            {
                "ordered_calls": [
                    {
                        "tool_name": "thermodynamics.calc_gas_pressure",
                        "evidence_span": "2 m³ closed chamber with air at 25°C to 100°C",
                        "arguments": {"volume": 2, "initial_temperature": 25, "final_temperature": 100},
                        "evidence_spans": {
                            "volume": "2 m³",
                            "initial_temperature": "25°C",
                            "final_temperature": "100°C",
                        },
                    }
                ]
            },
            base_plan,
        )
        is None
    )


def test_call_skeleton_can_recover_single_tool_repeated_entities():
    tools = [
        {
            "name": "math.gcd",
            "description": "Calculate the greatest common divisor of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                "required": ["num1", "num2"],
            },
        }
    ]
    request = "John chose 36 and 48, while Mary chose 60 and 96. Find the GCD for each player."
    base_plan = {"calls": [{"tool_name": "math.gcd", "arguments": {"num1": 36, "num2": 48}}]}

    assert _should_generate_call_skeleton(request, tools, base_plan)
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        benchmark_compile_tools(tools),
        {
            "ordered_calls": [
                {
                    "tool_name": "math.gcd",
                    "evidence_span": "John chose 36 and 48",
                    "arguments": {"num1": 36, "num2": 48},
                    "evidence_spans": {"num1": "36", "num2": "48"},
                },
                {
                    "tool_name": "math.gcd",
                    "evidence_span": "Mary chose 60 and 96",
                    "arguments": {"num1": 60, "num2": 96},
                    "evidence_spans": {"num1": "60", "num2": "96"},
                },
            ],
            "confidence": 0.92,
        },
        base_plan,
    )

    assert skeleton_plan is not None
    assert [call["arguments"] for call in skeleton_plan["calls"]] == [
        {"num1": 36, "num2": 48},
        {"num1": 60, "num2": 96},
    ]


def test_call_skeleton_runs_for_single_tool_result_dependent_chains():
    tools = [
        {
            "name": "math.hcf",
            "description": "Calculate the highest common factor of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"number1": {"type": "integer"}, "number2": {"type": "integer"}},
                "required": ["number1", "number2"],
            },
        }
    ]
    request = (
        "Calculate the highest common factor of (45, 60) and then use that result to find "
        "the highest common factor with another pair (90, 120). Also find the highest common "
        "factor of (36, 48) and then find the highest common factor of that result with (72, 96)."
    )
    base_plan = {
        "calls": [
            {"tool_name": "math.hcf", "arguments": {"number1": 45, "number2": 60}},
            {"tool_name": "math.hcf", "arguments": {"number1": 90, "number2": 120}},
        ]
    }

    assert _should_generate_call_skeleton(request, tools, base_plan)


def test_call_skeleton_can_recover_repeated_required_array_batches():
    tools = [
        {
            "name": "create_histogram",
            "description": "Create a histogram for a numeric data set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "integer"}},
                    "bins": {"type": "integer"},
                },
                "required": ["data", "bins"],
            },
        }
    ]
    request = (
        "Create two histograms: the first data set is [1, 2, 3] and the second data set is [4, 5, 6], "
        "with 5 bins each."
    )
    base_plan = {
        "calls": [
            {
                "tool_name": "create_histogram",
                "arguments": {"data": [1, 2, 3], "bins": 5},
            }
        ]
    }

    assert _should_generate_call_skeleton(request, tools, base_plan)
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        benchmark_compile_tools(tools),
        {
            "ordered_calls": [
                {
                    "tool_name": "create_histogram",
                    "evidence_span": "first data set is [1, 2, 3]",
                    "arguments": {"data": [1, 2, 3], "bins": 5},
                    "evidence_spans": {"data": "[1, 2, 3]", "bins": "5 bins"},
                },
                {
                    "tool_name": "create_histogram",
                    "evidence_span": "second data set is [4, 5, 6]",
                    "arguments": {"data": [4, 5, 6], "bins": 5},
                    "evidence_spans": {"data": "[4, 5, 6]", "bins": "5 bins"},
                },
            ],
            "confidence": 0.91,
        },
        base_plan,
    )

    assert skeleton_plan is not None
    assert [call["arguments"] for call in skeleton_plan["calls"]] == [
        {"data": [1, 2, 3], "bins": 5},
        {"data": [4, 5, 6], "bins": 5},
    ]


def test_call_skeleton_runs_for_identical_duplicate_optional_array_plan():
    tools = [
        {
            "name": "museum.exhibition_detail",
            "description": "Provides details of an exhibition, including cost per visit for age groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exhibition_name": {"type": "string"},
                    "museum_name": {"type": "string"},
                    "visitor_type": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["exhibition_name", "museum_name"],
            },
        }
    ]
    request = (
        "Give me the detail of the exhibition named 'Wonder of Nature' in the Louvre museum, and "
        "'Age of Reptiles' in the British Museum. Plus their cost per visit for children and adult."
    )
    base_plan = {
        "calls": [
            {
                "tool_name": "museum.exhibition_detail",
                "arguments": {
                    "exhibition_name": "Wonder of Nature",
                    "museum_name": "British Museum",
                    "visitor_type": ["adult"],
                },
            },
            {
                "tool_name": "museum.exhibition_detail",
                "arguments": {
                    "exhibition_name": "Wonder of Nature",
                    "museum_name": "British Museum",
                    "visitor_type": ["adult"],
                },
            },
        ]
    }

    assert _should_generate_call_skeleton(request, tools, base_plan)


def test_call_skeleton_dedupes_identical_verified_calls():
    tools = [
        {
            "name": "geometry.volume",
            "description": "Calculate volume for one cone.",
            "parameters": {
                "type": "object",
                "properties": {"radius": {"type": "integer"}, "height": {"type": "integer"}},
                "required": ["radius", "height"],
            },
        }
    ]
    request = "Create two identical cones, each with radius 10 and height 30. Calculate the volume of each cone."
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        benchmark_compile_tools(tools),
        {
            "ordered_calls": [
                {
                    "tool_name": "geometry.volume",
                    "evidence_span": "each with radius 10 and height 30",
                    "arguments": {"radius": 10, "height": 30},
                    "evidence_spans": {"radius": "radius 10", "height": "height 30"},
                },
                {
                    "tool_name": "geometry.volume",
                    "evidence_span": "each with radius 10 and height 30",
                    "arguments": {"radius": 10, "height": 30},
                    "evidence_spans": {"radius": "radius 10", "height": "height 30"},
                },
            ],
            "confidence": 0.86,
        },
        {"calls": [{"tool_name": "geometry.volume"}]},
    )

    assert skeleton_plan is not None
    assert len(skeleton_plan["calls"]) == 1
    assert skeleton_plan["calls"][0]["arguments"] == {"radius": 10, "height": 30}


def test_call_skeleton_can_reorder_repeated_mixed_tool_plan():
    tools = [{"name": "alpha.do"}, {"name": "beta.do"}]
    base_plan = {
        "calls": [
            {"tool_name": "alpha.do"},
            {"tool_name": "alpha.do"},
            {"tool_name": "beta.do"},
            {"tool_name": "beta.do"},
        ]
    }
    skeleton_plan = {
        "calls": [
            {"tool_name": "alpha.do"},
            {"tool_name": "beta.do"},
            {"tool_name": "alpha.do"},
            {"tool_name": "beta.do"},
        ]
    }

    assert _should_generate_call_skeleton("Find cases and judges for both companies.", tools, base_plan)
    assert not _prefer_skeleton_binding_plan("Find cases and judges for both companies.", base_plan, skeleton_plan)
    assert _prefer_skeleton_binding_plan(
        "Find cases and judges for both companies, respectively.",
        base_plan,
        skeleton_plan,
    )


def test_call_skeleton_can_replace_same_tool_sequence_with_better_arguments():
    request = (
        "First, secure a room for 2 adults and 1 child at the Sheraton Hotel in New York "
        "with check-in on May 1, 2022, and check-out on May 5, 2022. Then, reserve a "
        "room for 1 adult and 2 children at the Marriott in Los Angeles, checking in on "
        "June 1, 2022, and checking out on June 10, 2022."
    )
    tools = benchmark_compile_tools(
        [
            {
                "name": "hotel_booking_book",
                "description": "Book a hotel room.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "adults": {"type": "integer"},
                        "children": {"type": "integer"},
                        "hotel_name": {"type": "string"},
                        "check_in": {"type": "string"},
                        "check_out": {"type": "string"},
                    },
                    "required": ["location", "adults", "children", "hotel_name", "check_in", "check_out"],
                },
            }
        ]
    )
    base_plan = {
        "calls": [
            {
                "tool_name": "hotel_booking_book",
                "arguments": {"location": "New York, NY", "hotel_name": "Sheraton Hotel"},
            },
            {
                "tool_name": "hotel_booking_book",
                "arguments": {"location": "New York, NY", "hotel_name": "Marriott"},
            },
        ]
    }
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "hotel_booking_book",
                    "evidence_span": (
                        "2 adults and 1 child at the Sheraton Hotel in New York with check-in "
                        "on May 1, 2022, and check-out on May 5, 2022"
                    ),
                    "arguments": {
                        "location": "New York, NY",
                        "adults": 2,
                        "children": 1,
                        "hotel_name": "Sheraton Hotel",
                        "check_in": "2022-05-01",
                        "check_out": "2022-05-05",
                    },
                },
                {
                    "tool_name": "hotel_booking_book",
                    "evidence_span": (
                        "1 adult and 2 children at the Marriott in Los Angeles, checking in "
                        "on June 1, 2022, and checking out on June 10, 2022"
                    ),
                    "arguments": {
                        "location": "Los Angeles, CA",
                        "adults": 1,
                        "children": 2,
                        "hotel_name": "Marriott",
                        "check_in": "2022-06-01",
                        "check_out": "2022-06-10",
                    },
                },
            ],
            "confidence": 1.0,
        },
        base_plan,
    )

    assert skeleton_plan is not None
    assert skeleton_plan["calls"][1]["arguments"]["location"] == "Los Angeles, CA"
    assert _prefer_skeleton_binding_plan(request, base_plan, skeleton_plan)


def test_call_skeleton_does_not_shrink_same_tool_repeated_plan():
    base_plan = {
        "calls": [
            {"tool_name": "log_food"},
            {"tool_name": "log_food"},
            {"tool_name": "log_food"},
            {"tool_name": "log_food"},
        ]
    }
    smaller_plan = {"calls": [{"tool_name": "log_food"}, {"tool_name": "log_food"}]}

    assert not _prefer_skeleton_binding_plan(
        "I had 8 pieces of mango and a chai tea. Earlier I had two slices of pizza and a coffee.",
        base_plan,
        smaller_plan,
    )


def test_call_skeleton_does_not_expand_accepted_model_binding_with_extra_tool():
    accepted_model_plan = {
        "model_tool_binding": {"used": True, "accepted": True},
        "calls": [
            {"tool_name": "get_relevant_classes"},
            {"tool_name": "get_signature"},
            {"tool_name": "get_signature"},
        ],
    }
    skeleton_plan = {
        "calls": [
            {"tool_name": "get_relevant_classes"},
            {"tool_name": "get_class_info"},
            {"tool_name": "get_signature"},
            {"tool_name": "get_signature"},
        ]
    }

    assert not _prefer_skeleton_binding_plan(
        "Find relevant classes and provide the signatures of two methods.",
        accepted_model_plan,
        skeleton_plan,
    )

    deterministic_plan = {
        "calls": [
            {"tool_name": "get_relevant_classes"},
            {"tool_name": "get_signature"},
            {"tool_name": "get_signature"},
        ],
    }

    assert not _prefer_skeleton_binding_plan(
        "Find relevant classes that might be related to CellResult and provide the signatures of setCellValue and getCellValue.",
        deterministic_plan,
        skeleton_plan,
    )


def test_call_skeleton_prunes_unrequested_helper_info_call():
    tools = benchmark_compile_tools(
        [
            {
                "name": "get_relevant_classes",
                "description": "Find relevant Java classes in a repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_string": {"type": "string"},
                        "include_subdirectories": {"type": "boolean", "default": False},
                    },
                    "required": ["search_string"],
                },
            },
            {
                "name": "get_class_info",
                "description": "Return class metadata and details.",
                "parameters": {
                    "type": "object",
                    "properties": {"class_name": {"type": "string"}},
                    "required": ["class_name"],
                },
            },
            {
                "name": "get_signature",
                "description": "Return a method signature for a class.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "class_name": {"type": "string"},
                        "method_name": {"type": "string"},
                    },
                    "required": ["class_name", "method_name"],
                },
            },
        ]
    )
    request = (
        "Find relevant classes that might be related to 'CellResult' in the repository including "
        "subdirectories. Also provide the signatures of 'setCellValue' and 'getCellValue' methods "
        "in the 'AbstractCellHandler' class."
    )
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "get_relevant_classes",
                    "arguments": {"search_string": "CellResult", "include_subdirectories": True},
                    "evidence_spans": {"search_string": "CellResult"},
                },
                {
                    "tool_name": "get_class_info",
                    "arguments": {"class_name": "CellResult"},
                    "evidence_spans": {"class_name": "CellResult"},
                },
                {
                    "tool_name": "get_signature",
                    "arguments": {"class_name": "AbstractCellHandler", "method_name": "setCellValue"},
                    "evidence_spans": {
                        "class_name": "AbstractCellHandler",
                        "method_name": "setCellValue",
                    },
                },
                {
                    "tool_name": "get_signature",
                    "arguments": {"class_name": "AbstractCellHandler", "method_name": "getCellValue"},
                    "evidence_spans": {
                        "class_name": "AbstractCellHandler",
                        "method_name": "getCellValue",
                    },
                },
            ]
        },
        {"calls": []},
    )

    assert skeleton_plan is not None
    assert [call["tool_name"] for call in skeleton_plan["calls"]] == [
        "get_relevant_classes",
        "get_signature",
        "get_signature",
    ]


def test_call_skeleton_can_recover_expression_function_argument():
    tools = benchmark_compile_tools(
        [
            {
                "name": "math_gcd",
                "description": "Find the greatest common divisor.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "estimate_derivative",
                "description": "Estimate derivative of a function at x.",
                "parameters": {
                    "type": "object",
                    "properties": {"function": {"type": "string"}, "x": {"type": "number"}},
                    "required": ["function", "x"],
                },
            },
        ]
    )
    request = "Find the highest common factor for 36 and 48, and then tell me how steep the curve of the function f(x) = x^2 is at x = 5?"
    skeleton_plan = _binding_plan_from_call_skeleton(
        request,
        tools,
        {
            "ordered_calls": [
                {
                    "tool_name": "math_gcd",
                    "arguments": {"a": 36, "b": 48},
                    "evidence_spans": {"a": "36", "b": "48"},
                },
                {
                    "tool_name": "estimate_derivative",
                    "evidence_span": "f(x) = x^2 at x = 5",
                    "arguments": {"function": "x**2", "x": 5},
                    "evidence_spans": {"function": "f(x) = x^2", "x": "x = 5"},
                },
            ],
            "confidence": 0.9,
        },
        {"calls": [{"tool_name": "math_gcd", "arguments": {"a": 36, "b": 48}}]},
    )

    assert skeleton_plan is not None
    assert [call["tool_name"] for call in skeleton_plan["calls"]] == ["math_gcd", "estimate_derivative"]
    assert skeleton_plan["calls"][1]["arguments"] == {"function": "x**2", "x": 5.0}


def test_tool_binding_frame_prompt_exposes_available_tool_schemas():
    messages = build_tool_binding_frame_messages(
        "Calculate the factorial of 5.",
        [
            {
                "name": "math.factorial",
                "description": "Return the factorial of a number.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
    )

    prompt = messages[-1]["content"]
    assert "tool_bindings" in prompt
    assert "math.factorial" in prompt
    assert "number" in prompt


def test_stepwise_pipeline_uses_verified_model_tool_binding():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"canonical_request":"Calculate factorial of 5.",'
            '"slots_observed":[{"role":"number","value":5,"value_type":"number",'
            '"evidence_span":"5","status":"explicit","confidence":1.0}],'
            '"call_groups":[{"intent":"calculate factorial","unit_of_work":"factorial",'
            '"requested_entities":["5"],"expected_call_count":1,"result_count":null,'
            '"can_use_batch_tool_if_available":true}],'
            '"tool_bindings":[{"tool_name":"math.factorial","intent":"calculate factorial",'
            '"call_count":1,"argument_groups":[{"arguments":{"number":5},'
            '"evidence_spans":{"number":"5"}}],"confidence":0.95}],'
            '"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Calculate the factorial of 5 using math functions.",
        [
            {
                "name": "math.factorial",
                "description": "Return the factorial of a number.",
                "parameters": {
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["tool_binding_plan"]["model_tool_binding"]["used"] is True
    assert result["calls"][0]["tool_name"] == "math.factorial"
    assert result["graph"]["nodes"][0]["inputs"]["number"]["evidence"] == "5"


def test_stepwise_pipeline_keeps_accepted_model_binding_over_raw_superset():
    model_plan = {
        "model_tool_binding": {"used": True, "accepted": True},
        "calls": [{"tool_name": "history.battle_details", "missing_arguments": []}],
    }
    raw_plan = {
        "calls": [
            {"tool_name": "history.battle_details", "missing_arguments": []},
            {"tool_name": "history.leader_info", "missing_arguments": []},
            {"tool_name": "history.war_details", "missing_arguments": []},
        ]
    }

    assert not _prefer_raw_binding_plan(
        "Who were the participants and location of the Battle of Stalingrad?",
        model_plan,
        raw_plan,
    )


def test_stepwise_pipeline_compiles_database_column_delete_graph():
    request = (
        "I need to delete some columns from my employees database on personal_data table. "
        "I want to remove their email addresses and social security numbers to respect privacy."
    )

    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"canonical_request":"Delete email addresses and social security numbers columns '
            'from personal_data table in employees database.",'
            '"slots_observed":[{"role":"database_name","value":"employees","value_type":"text",'
            '"evidence_span":"employees database","status":"explicit","confidence":1.0},'
            '{"role":"table_name","value":"personal_data","value_type":"text",'
            '"evidence_span":"personal_data table","status":"explicit","confidence":1.0},'
            '{"role":"columns_to_remove","value":["email addresses","social security numbers"],'
            '"value_type":"list","evidence_span":"email addresses and social security numbers",'
            '"status":"explicit","confidence":1.0}],'
            '"call_groups":[{"intent":"delete columns","unit_of_work":"delete columns from table",'
            '"requested_entities":["columns_to_remove"],"expected_call_count":1,'
            '"result_count":null,"can_use_batch_tool_if_available":true}],'
            '"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        request,
        benchmark_compile_tools(
            [
                {
                    "name": "database.modify_columns",
                    "description": "This function allows deletion or addition of columns in a database",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "db_name": {"type": "string", "description": "The name of the database to modify."},
                            "table": {"type": "string", "description": "The name of the table to modify."},
                            "operation": {
                                "type": "string",
                                "description": "The operation to carry out on the table. Can be 'delete' or 'add'.",
                            },
                            "columns": {
                                "type": "array",
                                "description": "List of the columns to add or delete from the table.",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["db_name", "table", "operation", "columns"],
                    },
                }
            ]
        ),
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"], result["diagnostics"]
    assert result["calls"][0]["tool_name"] == "database.modify_columns"
    assert result["calls"][0]["arguments"] == {
        "db_name": "employees",
        "table": "personal_data",
        "operation": "delete",
        "columns": ["email", "social_security_number"],
    }
    assert result["graph"]["nodes"][0]["inputs"]["columns"]["evidence"] == request


def test_stepwise_pipeline_recovers_when_semantic_frame_vetoes_raw_query_binding():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"canonical_request":"Acknowledge the user.",'
            '"slots_observed":[],"call_groups":[],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.",
        [
            {
                "name": "QueryHealthData",
                "description": "This API queries the recorded health data in database of a given user and time span.",
                "input_parameters": {
                    "user_id": {"type": "str", "description": "The user id of the given user. Cases are ignored."},
                    "start_time": {"type": "str", "description": "The start time of the time span. Format: %Y-%m-%d %H:%M:%S"},
                    "end_time": {"type": "str", "description": "The end time of the time span. Format: %Y-%m-%d %H:%M:%S"},
                },
            },
            {
                "name": "CancelRegistration",
                "description": "This API cancels the registration of a patient given appointment ID.",
                "input_parameters": {
                    "appointment_id": {"type": "str", "description": "The ID of appointment."},
                },
            },
            {
                "name": "ModifyRegistration",
                "description": "This API modifies the registration of a patient given appointment ID.",
                "input_parameters": {
                    "appointment_id": {"type": "str", "description": "The ID of appointment."},
                    "new_appointment_date": {"type": "str", "description": "The new appointment date. Format: %Y-%m-%d."},
                    "new_appointment_doctor": {"type": "str", "description": "The new appointment doctor."},
                },
            },
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["steps"][1]["raw_query_recovery_used"] is True
    assert result["calls"][0]["tool_name"] == "QueryHealthData"


def test_stepwise_pipeline_honors_explicit_semantic_no_tool_decision():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"no_tool","canonical_request":"Find the freezing point of water at 10 kPa.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "What is the freezing point of water at a pressure of 10 kPa?",
        [
            {
                "name": "thermodynamics.calculate_boiling_point",
                "description": "Calculate the boiling point of a substance at a given pressure.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "substance": {"type": "string"},
                        "pressure": {"type": "number"},
                    },
                    "required": ["substance", "pressure"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["calls"] == []
    assert result["steps"][1]["semantic_no_tool_veto"] is True
    assert result["steps"][1]["raw_query_recovery_used"] is False


def test_stepwise_pipeline_recovers_when_semantic_no_tool_is_really_missing_inputs():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"no_tool","canonical_request":"Query health data for user J46801 from March 5th to March 12th",'
            '"slots_observed":[{"user_id":"J46801","date_range":"March 5th to March 12th"}],'
            '"call_groups":[],"tool_bindings":[],"missing_inputs":["start_time","end_time"]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.\nAi: Got it.",
        [
            {
                "name": "QueryHealthData",
                "description": "This API queries the recorded health data in database of a given user and time span.",
                "input_parameters": {
                    "user_id": {"type": "str", "description": "The user id of the given user. Cases are ignored."},
                    "start_time": {"type": "str", "description": "The start time of the time span. Format: %Y-%m-%d %H:%M:%S"},
                    "end_time": {"type": "str", "description": "The end time of the time span. Format: %Y-%m-%d %H:%M:%S"},
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["steps"][1]["semantic_no_tool_veto"] is False
    assert [call["tool_name"] for call in result["calls"]] == ["QueryHealthData"]


def test_stepwise_pipeline_honors_explicit_semantic_ask_user_decision():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"ask_user","canonical_request":"Find the roots of bx + c = 0.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":["b","c"]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "What is the roots of linear equation bx + c = 0?",
        [
            {
                "name": "find_roots",
                "description": "Find the roots of a quadratic equation ax^2 + bx + c = 0.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "c": {"type": "number"},
                    },
                    "required": ["a", "b", "c"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["calls"] == []
    assert result["tool_binding_plan"]["tool_decision"] == "ask_user"
    assert result["steps"][1]["semantic_ask_user_veto"] is True
    assert result["steps"][1]["raw_query_recovery_used"] is False
    assert result["steps"][3]["model_call"] is False


def test_stepwise_pipeline_overrides_semantic_ask_user_with_complete_raw_plan():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"ask_user","canonical_request":"Authenticate the user.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],'
            '"missing_inputs":["username","password"]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Earlier user: Can you check my account balance?\n"
        "User: My username is user1 and my password is user1pass.",
        [
            {
                "name": "GetUserToken",
                "description": "Get a user token using username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username."},
                    "password": {"type": "str", "description": "The password."},
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["steps"][1]["semantic_ask_user_veto"] is True
    assert result["steps"][1]["semantic_terminal_veto_active"] is False
    assert result["steps"][1]["raw_query_recovery_used"] is True
    assert [call["tool_name"] for call in result["calls"]] == ["GetUserToken"]


def test_stepwise_pipeline_overrides_semantic_no_tool_only_in_dialogue_context():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"no_tool","canonical_request":"Credentials were provided.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Earlier user: What is my current balance?\n"
        "User: My username is user1 and my password is user1pass.",
        [
            {
                "name": "GetUserToken",
                "description": "Get a user token using username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username."},
                    "password": {"type": "str", "description": "The password."},
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["steps"][1]["semantic_no_tool_veto"] is True
    assert result["steps"][1]["semantic_terminal_veto_active"] is False
    assert [call["tool_name"] for call in result["calls"]] == ["GetUserToken"]


def test_stepwise_pipeline_keeps_static_semantic_no_tool_over_raw_match():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"no_tool","canonical_request":"Do not use the species tool.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        'Identify the genetic code sequence "ATCG".',
        [
            {
                "name": "identify_species",
                "description": "Identifies the species of an organism based on its genetic code sequence.",
                "parameters": {
                    "type": "dict",
                    "properties": {"sequence": {"type": "string"}},
                    "required": ["sequence"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["calls"] == []
    assert result["steps"][1]["semantic_no_tool_veto"] is True
    assert result["steps"][1]["semantic_terminal_veto_active"] is True


def test_stepwise_pipeline_prefers_raw_recovery_for_partial_model_binding_with_missing_inputs():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"ask_user","canonical_request":"Get current weather for Boston, MA and San Francisco",'
            '"slots_observed":[{"location":"Boston, MA","evidence":"Boston, MA"},'
            '{"location":"San Francisco","evidence":"San Francisco"}],'
            '"call_groups":[{"name":"get_current_weather","expected_call_count":1}],'
            '"tool_bindings":[{"tool_name":"get_current_weather","call_count":1,'
            '"argument_groups":[{"arguments":{"location":"Boston, MA"},'
            '"evidence_spans":{"location":"Boston, MA"}}]}],'
            '"missing_inputs":["location for San Francisco"]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Could you tell me the current weather conditions for Boston, MA and also for San Francisco?",
        [
            {
                "name": "get_current_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "City and region."},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"], "default": "fahrenheit"},
                    },
                    "required": ["location"],
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["steps"][1]["raw_query_recovery_used"] is True
    assert [call["tool_name"] for call in result["calls"]] == ["get_current_weather", "get_current_weather"]


def test_stepwise_pipeline_exempts_catalog_search_from_semantic_no_tool_veto():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"tool_decision":"no_tool","canonical_request":"Find a stock-price API.",'
            '"slots_observed":[],"call_groups":[],"tool_bindings":[],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Can you please tell me the stock price of Microsoft on 4th February 2022?",
        [
            {
                "name": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "input_parameters": {
                    "keywords": {"type": "str", "description": "The keyword to search for."}
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert [call["tool_name"] for call in result["calls"]] == ["ToolSearcher"]
    assert result["steps"][1]["semantic_no_tool_veto"] is False


def test_stepwise_pipeline_binds_semantic_medical_condition_to_symptom_slot():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"canonical_request":"Help me find out about shortness of breath.",'
            '"slots_observed":[{"role":"medical_condition","value":"shortness of breath",'
            '"value_type":"text","evidence_span":"shortness of breath","status":"explicit",'
            '"confidence":1.0}],"call_groups":[{"intent":"search medical knowledge",'
            '"unit_of_work":"search for information about shortness of breath",'
            '"requested_entities":[],"expected_call_count":1,"result_count":null,'
            '"can_use_batch_tool_if_available":true}],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "User: Can you help me find out about shortness of breath?\n"
        "Ai: Sure, I can call the EmergencyKnowledge API to search for information about shortness of breath.",
        [
            {
                "name": "EmergencyKnowledge",
                "description": "This API searches emergency medical knowledge for a symptom.",
                "input_parameters": {
                    "symptom": {"type": "str", "description": "The symptom to search for."},
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"]
    assert result["calls"][0]["tool_name"] == "EmergencyKnowledge"
    assert result["calls"][0]["arguments"] == {"symptom": "shortness of breath"}


def test_stepwise_pipeline_keeps_model_binding_with_schema_default_outside_enum():
    def fake_generate(_model, _tokenizer, _messages, _max_new_tokens):
        return (
            '{"canonical_request":"Provide news reports for Paris and Letterkenny.",'
            '"slots_observed":[],"call_groups":[{"intent":"get news report",'
            '"unit_of_work":"news report for Paris","expected_call_count":1},'
            '{"intent":"get news report","unit_of_work":"news report for Letterkenny",'
            '"expected_call_count":1}],'
            '"tool_bindings":[{"tool_name":"get_news_report","call_count":1,'
            '"argument_groups":[{"arguments":{"location":"Paris, France",'
            '"category":"General","language":"en"},"evidence_spans":{'
            '"location":"Paris, France","language":"in English"}}]},'
            '{"tool_name":"get_news_report","call_count":1,"argument_groups":[{'
            '"arguments":{"location":"Letterkenny, Ireland","category":"Technology",'
            '"language":"en"},"evidence_spans":{"location":"Letterkenny, Ireland",'
            '"category":"technology","language":"in English"}}]}],"missing_inputs":[]}'
        )

    result = plan_and_compile_goal_graph(
        None,
        None,
        fake_generate,
        "Could you provide me with the latest news report for Paris, France, in English and also for "
        "Letterkenny, Ireland, focusing on technology news again in English?",
        [
            {
                "name": "get_news_report",
                "description": "Retrieve the latest news based on a specified location.",
                "parameters": {
                    "type": "object",
                    "required": ["location"],
                    "properties": {
                        "location": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["Politics", "Technology", "Sports"],
                            "default": "General",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["en", "es", "fr"],
                            "default": "en",
                        },
                    },
                },
            }
        ],
        max_new_tokens=100,
        repair_attempts=0,
    )

    assert result["verification_ok"], result["diagnostics"]
    assert result["tool_binding_plan"]["model_tool_binding"]["used"] is True
    assert [call["tool_name"] for call in result["calls"]] == ["get_news_report", "get_news_report"]
    assert result["calls"][0]["arguments"]["category"] == "General"
    assert result["steps"][1]["raw_query_recovery_used"] is False
