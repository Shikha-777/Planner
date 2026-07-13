from taskdecomp.goal_graph_runtime import (
    GoalGraphRuntime,
    build_capability_registry,
    build_goal_graph_planner_messages,
    compile_goal_graph,
    compiled_calls_to_dicts,
    parse_goal_graph,
    verify_goal_graph,
)


def test_capability_registry_normalizes_tool_schema():
    registry = build_capability_registry(
        [
            {
                "name": "weather.get",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "default": "celsius"},
                    },
                    "required": ["location"],
                },
            }
        ]
    )

    capability = registry["weather.get"]
    assert capability.tool_name == "weather.get"
    assert capability.kind == "retrieve"
    assert capability.risk == "read_only"
    assert list(capability.required_inputs) == ["location"]
    assert capability.optional_inputs["unit"].default == "celsius"


def test_capability_registry_extracts_description_defaults_and_prose_enums():
    registry = build_capability_registry(
        [
            {
                "name": "religion.retrieve",
                "description": "Retrieve religion information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "religion_name": {"type": "string"},
                        "detail_level": {
                            "type": "string",
                            "description": "Level of detail, either 'summary' or 'full'. Default is 'summary'.",
                        },
                    },
                    "required": ["religion_name", "detail_level"],
                },
            }
        ]
    )

    detail_level = registry["religion.retrieve"].required_inputs["detail_level"]
    assert detail_level.default == "summary"
    assert detail_level.allowed_values == ("summary", "full")


def test_goal_graph_compiles_grounded_retrieve_node():
    registry = build_capability_registry(
        [
            {
                "name": "weather.get",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "default": "celsius"},
                    },
                    "required": ["location"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
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
                            "evidence": "Boston",
                        }
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(graph, registry, "What is the weather in Boston?")

    assert verification.ok
    assert compiled_calls_to_dicts(calls) == [
        {
            "id": "call_1",
            "graph_node_id": "n1",
            "tool_name": "weather.get",
            "arguments": {"location": "Boston", "unit": "celsius"},
            "depends_on": [],
        }
    ]


def test_query_grounding_accepts_value_when_evidence_is_not_contiguous():
    registry = build_capability_registry(
        [
            {
                "name": "calculate_area",
                "description": "Calculate triangle area.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Calculate triangle area.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "calculate_area",
                    "inputs": {
                        "base": {"value": 6, "source": "query", "evidence": "base length 6cm"},
                        "height": {"value": 10, "source": "query", "evidence": "height length 10cm"},
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "Calculate the area of a right-angled triangle given the lengths of its base and height as 6cm and 10cm.",
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"] == {"base": 6, "height": 10}


def test_literal_resolve_node_can_feed_query_value_to_tool_node():
    registry = build_capability_registry(
        [
            {
                "name": "calculate_derivative",
                "description": "Calculate a derivative.",
                "parameters": {
                    "type": "object",
                    "properties": {"function": {"type": "string"}},
                    "required": ["function"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Calculate derivative.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "resolve",
                    "capability": "resolve",
                    "inputs": {
                        "function": {
                            "value": "3x^2 + 2x - 1",
                            "source": "query",
                            "evidence": "3x^2 + 2x - 1",
                        }
                    },
                    "outputs": ["function"],
                    "risk": "read_only",
                },
                {
                    "id": "n2",
                    "kind": "retrieve",
                    "capability": "calculate_derivative",
                    "depends_on": ["n1"],
                    "inputs": {
                        "function": {
                            "value": "$n1.function",
                            "source": "node_output",
                            "evidence": "$n1.function",
                        }
                    },
                    "risk": "read_only",
                },
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "Calculate the derivative of the function 3x^2 + 2x - 1.",
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls) == [
        {
            "id": "call_1",
            "graph_node_id": "n2",
            "tool_name": "calculate_derivative",
            "arguments": {"function": "3x^2 + 2x - 1"},
            "depends_on": [],
        }
    ]


def test_goal_graph_compiler_fills_required_schema_default():
    registry = build_capability_registry(
        [
            {
                "name": "poker_probability.full_house",
                "description": "Calculate full house probability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "deck_size": {"type": "integer", "description": "The size of the deck. Default is 52."},
                        "hand_size": {"type": "integer", "description": "The size of the hand. Default is 5."},
                    },
                    "required": ["deck_size", "hand_size"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Calculate full house probability.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "poker_probability.full_house",
                    "inputs": {},
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "What is the probability of getting a full house in poker?",
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"] == {"deck_size": 52, "hand_size": 5}


def test_goal_graph_runtime_ignores_second_phase_inputs_for_first_phase_status():
    registry = build_capability_registry(
        [
            {
                "name": "ForgotPassword",
                "kind": "communicate",
                "risk": "read_only",
                "description": "Reset a forgotten password.",
                "input_parameters": {
                    "status": {"type": "str", "description": "Forgot Password or Verification Code."},
                    "username": {"type": "str", "description": "Only needed for the first call."},
                    "email": {"type": "str", "description": "Only needed for the first call."},
                    "verification_code": {"type": "int", "description": "Only needed for the second call."},
                    "new_password": {"type": "str", "description": "Only needed for the second call."},
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Reset password.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "communicate",
                    "capability": "ForgotPassword",
                    "inputs": {
                        "status": {"value": "Forgot Password", "source": "query", "evidence": "forgot my password"},
                        "username": {"value": "foo", "source": "query", "evidence": "foo"},
                        "email": {"value": "foo@example.com", "source": "query", "evidence": "foo@example.com"},
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "I forgot my password. My username is foo and my email is foo@example.com.",
        allow_side_effects=True,
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"] == {
        "status": "Forgot Password",
        "username": "foo",
        "email": "foo@example.com",
    }


def test_goal_graph_runtime_ignores_first_phase_inputs_for_second_phase_status():
    registry = build_capability_registry(
        [
            {
                "name": "ForgotPassword",
                "kind": "communicate",
                "risk": "read_only",
                "description": "Reset a forgotten password.",
                "input_parameters": {
                    "status": {"type": "str", "description": "Forgot Password or Verification Code."},
                    "username": {"type": "str", "description": "Only needed for the first call."},
                    "email": {"type": "str", "description": "Only needed for the first call."},
                    "verification_code": {"type": "int", "description": "Only needed for the second call."},
                    "new_password": {"type": "str", "description": "Only needed for the second call."},
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Finish password reset.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "communicate",
                    "capability": "ForgotPassword",
                    "inputs": {
                        "status": {"value": "Verification Code", "source": "query", "evidence": "verification code"},
                        "verification_code": {"value": 970420, "source": "query", "evidence": "970420"},
                        "new_password": {"value": "newpassword", "source": "query", "evidence": "newpassword"},
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "The verification code is 970420. My new password is newpassword.",
        allow_side_effects=True,
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"] == {
        "status": "Verification Code",
        "verification_code": 970420,
        "new_password": "newpassword",
    }


def test_goal_graph_verifier_rejects_values_outside_schema_allowed_values():
    registry = build_capability_registry(
        [
            {
                "name": "religion.retrieve",
                "description": "Retrieve religion information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "religion_name": {"type": "string"},
                        "detail_level": {
                            "type": "string",
                            "description": "Level of detail, either 'summary' or 'full'.",
                        },
                    },
                    "required": ["religion_name", "detail_level"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Retrieve full Buddhism history.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "religion.retrieve",
                    "risk": "read_only",
                    "inputs": {
                        "religion_name": {
                            "value": "Buddhism",
                            "source": "query",
                            "evidence": "Buddhism",
                        },
                        "detail_level": {
                            "value": "detailed",
                            "source": "query",
                            "evidence": "full",
                        },
                    },
                }
            ],
        }
    )

    result = verify_goal_graph(graph, registry, "Retrieve the full history of Buddhism.")

    assert not result.ok
    assert "value_not_allowed" in {diagnostic.code for diagnostic in result.diagnostics}


def test_goal_graph_verifier_allows_schema_default_outside_enum():
    registry = build_capability_registry(
        [
            {
                "name": "get_news_report",
                "description": "Retrieve news.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["Politics", "Technology", "Sports"],
                            "default": "General",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Get news for Paris.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "get_news_report",
                    "risk": "read_only",
                    "inputs": {
                        "location": {"value": "Paris", "source": "query", "evidence": "Paris"},
                        "category": {
                            "value": "General",
                            "source": "policy_default",
                            "status": "defaulted",
                        },
                    },
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(graph, registry, "Get news for Paris.")

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"]["category"] == "General"


def test_goal_graph_verifier_allows_required_generated_id_schema_default():
    registry = build_capability_registry(
        [
            {
                "name": "lawsuit.judge",
                "description": "Fetch the judge handling a lawsuit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "lawsuit_id": {
                            "type": "integer",
                            "description": "Generated lawsuit identifier. Default is 123.",
                            "default": 123,
                        },
                    },
                    "required": ["company_name", "lawsuit_id"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Find the judge handling a lawsuit for Tesla.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "lawsuit.judge",
                    "risk": "read_only",
                    "inputs": {
                        "company_name": {"value": "Tesla", "source": "query", "evidence": "Tesla"},
                        "lawsuit_id": {
                            "value": 123,
                            "source": "policy_default",
                            "status": "defaulted",
                        },
                    },
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(graph, registry, "Find the judge handling a lawsuit for Tesla.")

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"] == {
        "company_name": "Tesla",
        "lawsuit_id": 123,
    }


def test_goal_graph_verifier_allows_grounded_required_string_missing_from_incomplete_enum():
    registry = build_capability_registry(
        [
            {
                "name": "ControlAppliance.execute",
                "description": "Control a home appliance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["거실, 에어컨, 실행"],
                        }
                    },
                    "required": ["command"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Stop bedroom purifier.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "ControlAppliance.execute",
                    "risk": "read_only",
                    "inputs": {
                        "command": {
                            "value": "침실, 공기청정기, 중지",
                            "source": "query",
                            "evidence": "침실, 공기청정기, 중지",
                        }
                    },
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "거실, 에어컨, 실행하고, 침실, 공기청정기, 중지해줘.",
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls)[0]["arguments"]["command"] == "침실, 공기청정기, 중지"


def test_goal_graph_rejects_ungrounded_query_argument():
    registry = build_capability_registry(
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
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Get weather in Boston",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "weather.get",
                    "inputs": {
                        "location": {
                            "value": "Seattle",
                            "source": "query",
                            "evidence": "Seattle",
                        }
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(graph, registry, "What is the weather in Boston?")

    assert calls == []
    assert not verification.ok
    assert {diagnostic.code for diagnostic in verification.diagnostics} == {"evidence_not_in_query"}


def test_goal_graph_resolves_dependency_outputs_before_compile():
    registry = build_capability_registry(
        [
            {
                "name": "calendar.search_events",
                "description": "Search calendar events.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}, "time": {"type": "string"}},
                    "required": ["date", "time"],
                },
            },
            {
                "name": "calendar.get_event",
                "capability": "calendar.get_event",
                "kind": "retrieve",
                "description": "Retrieve one event by event_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"event_id": {"type": "string"}},
                    "required": ["event_id"],
                },
            },
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Find today's 5:30 meeting and retrieve its details.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "search",
                    "capability": "calendar.search_events",
                    "inputs": {
                        "date": {"value": "today", "source": "query", "evidence": "today"},
                        "time": {"value": "5:30", "source": "query", "evidence": "5:30"},
                    },
                    "outputs": ["event_id"],
                    "risk": "read_only",
                },
                {
                    "id": "n2",
                    "kind": "retrieve",
                    "capability": "calendar.get_event",
                    "depends_on": ["n1"],
                    "inputs": {
                        "event_id": {
                            "value": "$n1.event_id",
                            "source": "node_output",
                        }
                    },
                    "risk": "read_only",
                },
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "Find today's 5:30 meeting.",
        observations={"n1": {"event_id": "evt_123"}},
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls) == [
        {
            "id": "call_1",
            "graph_node_id": "n1",
            "tool_name": "calendar.search_events",
            "arguments": {"date": "today", "time": "5:30"},
            "depends_on": [],
        },
        {
            "id": "call_2",
            "graph_node_id": "n2",
            "tool_name": "calendar.get_event",
            "arguments": {"event_id": "evt_123"},
            "depends_on": ["call_1"],
        },
    ]


def test_goal_graph_rejects_reference_without_dependency():
    registry = build_capability_registry(
        [
            {
                "name": "calendar.search_events",
                "description": "Search calendar events.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                    "required": ["date"],
                },
            },
            {
                "name": "calendar.get_event",
                "capability": "calendar.get_event",
                "kind": "retrieve",
                "description": "Retrieve one event by event_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"event_id": {"type": "string"}},
                    "required": ["event_id"],
                },
            },
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Find today's meeting.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "search",
                    "capability": "calendar.search_events",
                    "inputs": {"date": {"value": "today", "source": "query", "evidence": "today"}},
                    "outputs": ["event_id"],
                    "risk": "read_only",
                },
                {
                    "id": "n2",
                    "kind": "retrieve",
                    "capability": "calendar.get_event",
                    "inputs": {"event_id": {"value": "$n1.event_id", "source": "node_output"}},
                    "risk": "read_only",
                },
            ],
        }
    )

    result = verify_goal_graph(graph, registry, "Find today's meeting.")

    assert not result.ok
    assert "missing_dependency_for_reference" in {diagnostic.code for diagnostic in result.diagnostics}


def test_goal_graph_rejects_unknown_inputs_and_underspecified_values():
    registry = build_capability_registry(
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
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Get weather.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "weather.get",
                    "inputs": {
                        "location": {"value": None, "status": "underspecified"},
                        "units": {"value": "celsius", "source": "query", "evidence": "celsius"},
                    },
                    "risk": "read_only",
                }
            ],
        }
    )

    result = verify_goal_graph(graph, registry, "Get weather in celsius.")

    assert not result.ok
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert {"unknown_input", "unresolved_input", "empty_input"} <= codes


def test_goal_graph_side_effects_require_explicit_gates():
    registry = build_capability_registry(
        [
            {
                "name": "calendar.cancel_event",
                "description": "Cancel a calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {"event_id": {"type": "string"}},
                    "required": ["event_id"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Cancel the event.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "mutate",
                    "capability": "calendar.cancel_event",
                    "inputs": {
                        "event_id": {
                            "value": "evt_123",
                            "source": "context",
                        }
                    },
                    "risk": "destructive_side_effect",
                }
            ],
        }
    )

    result = verify_goal_graph(graph, registry, "Cancel the event.", context={"event_id": "evt_123"})

    assert not result.ok
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert {"side_effects_not_allowed", "mutation_target_not_unique", "side_effect_not_authorized"} <= codes


def test_goal_graph_compiles_authorized_side_effect_when_enabled():
    registry = build_capability_registry(
        [
            {
                "name": "calendar.cancel_event",
                "description": "Cancel a calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {"event_id": {"type": "string"}},
                    "required": ["event_id"],
                },
            }
        ]
    )
    graph = parse_goal_graph(
        {
            "goal": "Cancel the event.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "mutate",
                    "capability": "calendar.cancel_event",
                    "inputs": {
                        "event_id": {
                            "value": "evt_123",
                            "source": "context",
                        }
                    },
                    "risk": "destructive_side_effect",
                    "must_be_unique": True,
                    "authorized": True,
                }
            ],
        }
    )

    calls, verification = compile_goal_graph(
        graph,
        registry,
        "Cancel the event.",
        context={"event_id": "evt_123"},
        allow_side_effects=True,
    )

    assert verification.ok
    assert compiled_calls_to_dicts(calls) == [
        {
            "id": "call_1",
            "graph_node_id": "n1",
            "tool_name": "calendar.cancel_event",
            "arguments": {"event_id": "evt_123"},
            "depends_on": [],
        }
    ]


def test_goal_graph_control_nodes_do_not_compile_to_calls():
    registry = build_capability_registry([])
    graph = parse_goal_graph(
        {
            "goal": "Ask for missing email body.",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "ask_user",
                    "description": "Ask user for email body.",
                    "inputs": {},
                }
            ],
            "clarification_needed": True,
            "clarification_reasons": ["Email body is underspecified."],
        }
    )

    calls, verification = compile_goal_graph(graph, registry, "Email Carlos.")

    assert verification.ok
    assert calls == []


def test_goal_graph_planner_prompt_lists_capabilities_not_final_calls():
    registry = build_capability_registry(
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
        ]
    )

    messages = build_goal_graph_planner_messages(
        "What is the weather in Boston?",
        registry,
        policies=["Read-only weather lookup is allowed."],
        failure_lessons=["Top N is usually a limit, not repeated calls."],
    )

    assert messages[0]["role"] == "system"
    assert "not a direct tool caller" in messages[0]["content"]
    assert "weather.get" in messages[1]["content"]
    assert "Read-only weather lookup is allowed." in messages[1]["content"]
    assert "Top N is usually a limit" in messages[1]["content"]
    assert "Every concrete value from the query must become a graph input" in messages[1]["content"]


def test_goal_graph_planner_prompt_exposes_defaults_and_allowed_values():
    registry = build_capability_registry(
        [
            {
                "name": "religion.retrieve",
                "description": "Retrieve religion information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "religion_name": {"type": "string"},
                        "detail_level": {
                            "type": "string",
                            "description": "Level of detail, either 'summary' or 'full'. Default is 'summary'.",
                        },
                    },
                    "required": ["religion_name", "detail_level"],
                },
            }
        ]
    )

    messages = build_goal_graph_planner_messages("Retrieve the full history of Buddhism.", registry)

    assert "detail_level(default='summary', allowed=summary|full)" in messages[1]["content"]
    assert "Schema defaults are policy/schema-grounded values" in messages[1]["content"]


def test_goal_graph_runtime_facade_compiles_from_dict():
    runtime = GoalGraphRuntime(
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
        ]
    )

    output = runtime.compile(
        {
            "goal": "Get weather in Boston",
            "nodes": [
                {
                    "id": "n1",
                    "kind": "retrieve",
                    "capability": "weather.get",
                    "risk": "read_only",
                    "inputs": {
                        "location": {
                            "value": "Boston",
                            "source": "query",
                            "evidence": "Boston",
                        }
                    },
                }
            ],
        },
        "What is the weather in Boston?",
    )

    assert output.verification.ok
    assert compiled_calls_to_dicts(output.calls)[0]["tool_name"] == "weather.get"
