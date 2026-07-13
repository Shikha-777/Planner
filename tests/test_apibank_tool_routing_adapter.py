from __future__ import annotations

import json

from scripts.eval_apibank_capability_tool_routing import (
    _level3_next_step_tools,
    build_api_catalog,
    build_tool_search_index,
    expected_name,
    latest_dialogue_context,
    prompt_text,
    read_rows,
    resolve_api_bank_level_paths,
    tools_from_row,
)
from scripts.eval_goal_graph_apibank_routing import predicted_api_tool_names


def test_apibank_adapter_ignores_nested_parameter_name_dicts():
    row = {
        "instruction": """
        {"name": "QueryScene", "description": "Query a smart-home scene.",
         "input_parameters": {"name": {"type": "str", "description": "The name of the scene."}}}
        """,
        "input": "User: What devices are included in Morning Routine?",
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["QueryScene"]


def test_apibank_expected_name_reads_level3_output_field():
    row = {"output": "API-Request: [ToolSearcher(keywords='QueryMeeting')]"}

    assert expected_name(row) == "ToolSearcher"


def test_apibank_goal_graph_prediction_uses_auth_route_for_missing_token():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "ask_user",
            "candidate_tool_audits": [
                {
                    "tool_name": "AddAgenda",
                    "semantic_fit": "exact",
                    "missing_slots": ["token"],
                },
                {
                    "tool_name": "GetUserToken",
                    "semantic_fit": "rejected",
                    "missing_slots": ["username", "password"],
                },
            ],
        },
    }
    tools = [
        {"name": "AddAgenda", "description": "Add an agenda item."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]

    assert predicted_api_tool_names(result, tools) == ["GetUserToken"]


def test_apibank_goal_graph_prediction_uses_auth_route_for_missing_token_even_with_action_prompt():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "ask_user",
            "candidate_tool_audits": [
                {
                    "tool_name": "AddAgenda",
                    "semantic_fit": "exact",
                    "required_slots": ["token", "content", "time"],
                    "missing_slots": ["token"],
                },
                {
                    "tool_name": "GetUserToken",
                    "semantic_fit": "rejected",
                    "required_slots": ["username", "password"],
                    "missing_slots": ["username", "password"],
                },
            ],
        },
    }
    tools = [
        {"name": "AddAgenda", "description": "Add an agenda item."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = 'User: Can you add "Meeting with clients" on December 15, 2023?'

    assert predicted_api_tool_names(result, tools, prompt) == ["GetUserToken"]


def test_apibank_goal_graph_prediction_uses_protected_route_after_token_available():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "ask_user",
            "candidate_tool_audits": [
                {
                    "tool_name": "DeleteAccount",
                    "semantic_fit": "exact",
                    "required_slots": ["token"],
                    "missing_slots": ["token"],
                },
                {
                    "tool_name": "GetUserToken",
                    "semantic_fit": "rejected",
                    "required_slots": ["username", "password"],
                    "missing_slots": ["username", "password"],
                },
            ],
        },
    }
    tools = [
        {"name": "DeleteAccount", "description": "Delete an account using a token."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Can you please help me delete my account?\n"
        "User: My username is foo and my password is bar.\n"
        "Ai: Great, I was able to retrieve your token.\n"
        "Latest prior API result: {'token': 'tok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["DeleteAccount"]


def test_apibank_goal_graph_prediction_recovers_auth_name_from_audits():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "ask_user",
            "candidate_tool_audits": [
                {
                    "tool_name": "AddAgenda",
                    "semantic_fit": "exact",
                    "missing_slots": ["token"],
                    "required_slots": ["token", "content", "time", "location"],
                },
                {
                    "tool_name": "GetUserToken",
                    "semantic_fit": "rejected",
                    "missing_slots": ["username", "password"],
                    "required_slots": ["username", "password"],
                },
            ],
        },
    }
    tools = [{"name": "AddAgenda", "description": "Add an agenda item."}]

    assert predicted_api_tool_names(result, tools) == ["GetUserToken"]


def test_apibank_goal_graph_prediction_uses_auth_route_for_rejected_protected_action_missing_only_token():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "no_tool",
            "candidate_tool_audits": [
                {
                    "tool_name": "AddAgenda",
                    "score": 16.8,
                    "semantic_fit": "rejected",
                    "required_slots": ["token", "content", "time", "location"],
                    "missing_slots": ["token"],
                    "planned_calls": [
                        {
                            "arguments": {
                                "content": "Meeting with John",
                                "time": "9am",
                                "location": "coffee shop on Main Street",
                            },
                            "missing_arguments": ["token"],
                        }
                    ],
                }
            ],
        },
    }
    tools = [
        {"name": "AddAgenda", "description": "Add an agenda item."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]

    assert predicted_api_tool_names(result, tools) == ["GetUserToken"]


def test_apibank_goal_graph_prediction_uses_raw_route_after_terminal_empty_plan():
    result = {
        "calls": [],
        "tool_binding_plan": {
            "tool_decision": "no_tool",
            "calls": [],
            "candidate_tool_audits": [],
        },
    }
    tools = [
        {
            "name": "GetUserToken",
            "description": "Get a user token using username and password.",
            "input_parameters": {
                "username": {"type": "str", "description": "The username."},
                "password": {"type": "str", "description": "The password."},
            },
        }
    ]
    prompt = (
        "Earlier user: What is my current balance?\n"
        "User: Okay. My username is user1 and my password is user1pass."
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["GetUserToken"]


def test_apibank_goal_graph_prediction_prefers_protected_action_when_token_available():
    result = {
        "calls": [
            {"tool_name": "GetUserToken", "arguments": {"username": "user1", "password": "user1pass"}},
            {"tool_name": "GetUserToken", "arguments": {"username": "user1", "password": "user1pass"}},
        ],
        "tool_binding_plan": {
            "tool_decision": "call",
            "candidate_tool_audits": [
                {
                    "tool_name": "GetUserToken",
                    "semantic_fit": "exact",
                    "required_slots": ["username", "password"],
                    "missing_slots": [],
                    "planned_calls": [{"arguments": {"username": "user1", "password": "user1pass"}}],
                },
                {
                    "tool_name": "AddReminder",
                    "semantic_fit": "rejected",
                    "required_slots": ["token", "content", "time"],
                    "missing_slots": [],
                    "slot_bindings": {
                        "token": "n9m8k7j6h5g4f3d2s1a0",
                        "content": "Call mom",
                        "time": "2022-05-10 at 10 AM",
                    },
                    "planned_calls": [
                        {
                            "arguments": {
                                "token": "n9m8k7j6h5g4f3d2s1a0",
                                "content": "Call mom",
                                "time": "2022-05-10 at 10 AM",
                            },
                            "missing_arguments": [],
                        }
                    ],
                },
            ],
        },
    }
    tools = [
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
        {
            "name": "AddReminder",
            "description": "Add a reminder for the authenticated user.",
            "parameters": {"required": ["token", "content", "time"]},
        },
    ]

    assert predicted_api_tool_names(result, tools) == ["AddReminder"]


def test_apibank_goal_graph_prediction_prefers_protected_action_over_context_helper():
    result = {
        "calls": [{"tool_name": "GetToday", "arguments": {}}],
        "tool_binding_plan": {
            "tool_decision": "call",
            "candidate_tool_audits": [
                {
                    "tool_name": "ModifyAlarm",
                    "semantic_fit": "exact",
                    "required_slots": ["token", "from_time", "to_time"],
                    "missing_slots": [],
                    "slot_bindings": {
                        "token": "o8i7u6y5t4r3e2w1q0",
                        "from_time": "2023-03-20 06:30:00",
                        "to_time": "2023-03-20 07:00:00",
                    },
                    "planned_calls": [
                        {
                            "arguments": {
                                "token": "o8i7u6y5t4r3e2w1q0",
                                "from_time": "2023-03-20 06:30:00",
                                "to_time": "2023-03-20 07:00:00",
                            },
                            "missing_arguments": [],
                        }
                    ],
                }
            ],
        },
    }
    tools = [
        {"name": "GetToday", "description": "Return today's date.", "parameters": {"required": []}},
        {"name": "ModifyAlarm", "description": "Modify an alarm.", "parameters": {"required": ["token"]}},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]

    assert predicted_api_tool_names(result, tools) == ["ModifyAlarm"]


def test_apibank_goal_graph_prediction_uses_verified_plan_when_compile_empty():
    result = {
        "calls": [],
        "verification_ok": False,
        "tool_binding_plan": {
            "tool_decision": "call",
            "calls": [
                {
                    "tool_name": "AppointmentRegistration",
                    "arguments": {"doctor_name": "Dr. Smith"},
                    "missing_arguments": [],
                }
            ],
        },
    }

    assert predicted_api_tool_names(result, []) == ["AppointmentRegistration"]


def test_apibank_goal_graph_prediction_collapses_duplicate_catalog_search_routes():
    result = {
        "calls": [
            {"tool_name": "ToolSearcher", "arguments": {"keywords": "query meeting"}},
            {"tool_name": "ToolSearcher", "arguments": {"keywords": "query meeting"}},
        ],
        "tool_binding_plan": {},
    }

    assert predicted_api_tool_names(result, [{"name": "ToolSearcher"}]) == ["ToolSearcher"]


def test_apibank_prediction_prefers_route_grounded_in_latest_user_turn():
    result = {
        "calls": [
            {
                "tool_name": "AppointmentRegistration",
                "arguments": {"patient_name": "John Smith", "doctor_name": "Dr. Johnson", "date": "2023-08-15"},
            },
            {"tool_name": "EmergencyKnowledge", "arguments": {"symptom": "fever"}},
        ],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "AppointmentRegistration", "description": "Register a doctor appointment."},
        {"name": "EmergencyKnowledge", "description": "Provide emergency medical knowledge for symptoms."},
    ]
    prompt = (
        "Earlier user: My name is John Smith.\n"
        "Earlier user: I want to see Dr. Johnson.\n"
        "User: I have a fever, can you tell me what it might be?\n"
        "Latest prior API result: 54752427"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["EmergencyKnowledge"]


def test_apibank_prediction_prefers_latest_delete_over_stale_add():
    result = {
        "calls": [
            {"tool_name": "DeleteAgenda", "arguments": {"token": "tok", "content": "meeting"}},
            {"tool_name": "AddAgenda", "arguments": {"token": "tok", "content": "meeting"}},
        ],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "AddAgenda", "description": "Add an agenda item."},
        {"name": "DeleteAgenda", "description": "Delete an agenda item."},
    ]
    prompt = "Earlier user: Add an agenda item.\nUser: Yes, delete the same agenda item."

    assert predicted_api_tool_names(result, tools, prompt) == ["DeleteAgenda"]


def test_apibank_prediction_prefers_auth_when_latest_turn_requests_token():
    result = {
        "calls": [
            {"tool_name": "ModifyMeeting", "arguments": {"token": "oldtok"}},
            {"tool_name": "ModifyAlarm", "arguments": {"token": "oldtok"}},
        ],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "ModifyMeeting", "description": "Modify a meeting."},
        {"name": "ModifyAlarm", "description": "Modify an alarm."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Modify a meeting.\n"
        "User: My username is JaneSmith.\n"
        "Ai: Alright, I'll need to get your token for authentication.\n"
        "Latest prior API result: {'token': 'oldtok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["GetUserToken"]


def test_apibank_prediction_prefers_compiled_route_when_latest_token_result_available():
    result = {
        "calls": [{"tool_name": "QueryAlarm", "arguments": {"token": "oldtok", "time": "6:30AM"}}],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "QueryAlarm", "description": "Query alarm information."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Hi, can you help me check my alarm for March 20th, 2023 at 6:30AM?\n"
        "User: My username is JaneSmith and my password is password.\n"
        "Ai: Okay, you are authenticated now. Checking your alarm information.\n"
        "Latest prior API result: {'token': 'oldtok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["QueryAlarm"]


def test_apibank_prediction_prefers_compiled_route_after_token_obtained_message():
    result = {
        "calls": [{"tool_name": "QueryMeeting", "arguments": {"token": "tok", "meeting_topic": "Sales Meeting"}}],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "QueryMeeting", "description": "Query meeting information."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Check my Sales Meeting.\n"
        "User: My username is JohnDoe and my password is pass123.\n"
        "Ai: Thank you. Now, I need to get the user token for authentication.\n"
        "Ai: Alright, I got the token. Now, I'll check your meeting.\n"
        "Latest prior API result: {'token': 'tok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["QueryMeeting"]


def test_apibank_prediction_prefers_compiled_route_after_retrieved_token_message():
    result = {
        "calls": [{"tool_name": "DeleteAccount", "arguments": {"token": "tok"}}],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "DeleteAccount", "description": "Delete an account using a token."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Can you help me delete my account?\n"
        "User: Yes, my username is foo and password is bar.\n"
        "Ai: Thank you. I need to retrieve your token from our server.\n"
        "Ai: Great, I have retrieved your token.\n"
        "Latest prior API result: {'token': 'tok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["DeleteAccount"]


def test_apibank_prediction_prefers_compiled_route_after_able_to_retrieve_token_message():
    result = {
        "calls": [{"tool_name": "DeleteAccount", "arguments": {"token": "tok"}}],
        "tool_binding_plan": {},
    }
    tools = [
        {"name": "DeleteAccount", "description": "Delete an account using a token."},
        {"name": "GetUserToken", "description": "Get a user token by username and password."},
    ]
    prompt = (
        "Earlier user: Can you help me delete my account?\n"
        "User: My username is foo and my password is bar.\n"
        "Ai: Great, I was able to retrieve your token.\n"
        "Latest prior API result: {'token': 'tok'}"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["DeleteAccount"]


def test_apibank_prediction_recovers_followup_route_from_prior_result_schema():
    result = {"calls": [], "tool_binding_plan": {"tool_decision": "no_tool", "calls": []}}
    tools = [
        {
            "name": "SymptomSearch",
            "description": "Searches for a given symptom.",
            "output_parameters": {
                "results": {
                    "type": "list",
                    "description": 'Records like [{"name": disease name, "description": disease details}]',
                }
            },
        },
        {
            "name": "AppointmentRegistration",
            "description": "Registers an appointment.",
            "output_parameters": {"appointment_id": {"type": "str"}},
        },
    ]
    prompt = (
        "Earlier user: Fatigue.\n"
        "User: Yes, please tell me more about Chronic fatigue syndrome.\n"
        "Latest prior API result: [{'name': 'Chronic fatigue syndrome', 'description': 'details'}]"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["SymptomSearch"]


def test_apibank_prediction_uses_prior_result_schema_to_disambiguate_followup_tools():
    result = {"calls": [], "tool_binding_plan": {"tool_decision": "no_tool", "calls": []}}
    tools = [
        {
            "name": "EmergencyKnowledge",
            "description": "Search emergency knowledge for a symptom.",
            "output_parameters": {
                "results": {
                    "type": "list",
                    "description": 'Records like [{"name": disease name, "aid": first-aid method}]',
                }
            },
        },
        {
            "name": "SymptomSearch",
            "description": "Searches for a given symptom.",
            "output_parameters": {
                "results": {
                    "type": "list",
                    "description": 'Records like [{"name": disease name, "description": disease details}]',
                }
            },
        },
    ]
    prompt = (
        "AI: Would you like me to provide more information on each of these?\n"
        "User: Yes please.\n"
        "Latest prior API result: [{'name': 'Anemia', 'aid': 'Treatment may involve iron supplements'}]"
    )

    assert predicted_api_tool_names(result, tools, prompt) == ["EmergencyKnowledge"]


def test_apibank_read_rows_expands_raw_level3_trace(tmp_path):
    path = tmp_path / "level-3.json"
    path.write_text(
        json.dumps(
            [
                {
                    "requirement": "Query meeting of John and send email reminder to john@example.com.",
                    "response": "",
                    "apis": [
                        {
                            "api_name": "ToolSearcher",
                            "input": {"keywords": "QueryMeeting"},
                            "output": {
                                "api_name": "ToolSearcher",
                                "output": {
                                    "name": "QueryMeeting",
                                    "description": "Query meetings.",
                                    "input_parameters": {"user_name": {"type": "str"}},
                                },
                            },
                        },
                        {
                            "api_name": "QueryMeeting",
                            "input": {"user_name": "John"},
                            "output": {
                                "api_name": "QueryMeeting",
                                "output": {"meetings": [{"meeting_name": "A"}]},
                            },
                        },
                        {
                            "api_name": "ToolSearcher",
                            "input": {"keywords": "EmailReminder"},
                            "output": {
                                "api_name": "ToolSearcher",
                                "output": {
                                    "name": "EmailReminder",
                                    "description": "Send email reminders.",
                                    "input_parameters": {"recipient": {"type": "str"}},
                                },
                            },
                        },
                        {
                            "api_name": "EmailReminder",
                            "input": {"recipient": "john@example.com"},
                            "output": {"api_name": "EmailReminder", "output": "success"},
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = read_rows(path)

    assert [expected_name(row) for row in rows] == [
        "ToolSearcher",
        "QueryMeeting",
        "ToolSearcher",
        "EmailReminder",
    ]
    assert [tool["name"] for tool in tools_from_row(rows[0])] == ["ToolSearcher"]
    assert [tool["name"] for tool in tools_from_row(rows[1])] == ["QueryMeeting"]
    assert [tool["name"] for tool in tools_from_row(rows[2])] == ["ToolSearcher"]
    assert [tool["name"] for tool in tools_from_row(rows[3])] == ["EmailReminder"]


def test_apibank_read_rows_expands_jsonl_api_trace(tmp_path):
    path = tmp_path / "AddAgenda-level-3-1.jsonl"
    events = [
        {"role": "User", "text": "Add an agenda item for tomorrow."},
        {"role": "AI", "text": "I will search for a tool."},
        {"role": "API", "api_name": "ToolSearcher", "param_dict": {"keywords": "add agenda"}, "result": {"output": []}},
        {"role": "AI", "text": "Please provide credentials."},
        {"role": "User", "text": "username is u and password is p."},
        {"role": "API", "api_name": "GetUserToken", "param_dict": {"username": "u", "password": "p"}, "result": {"output": {"token": "tok"}}},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    rows = read_rows(path)

    assert [expected_name(row) for row in rows] == ["ToolSearcher", "GetUserToken"]
    assert "User: Add an agenda item" in rows[0]["input"]
    assert "API-Request: [ToolSearcher" in rows[1]["input"]


def test_apibank_resolves_sample_directory_as_level_paths(tmp_path):
    root = tmp_path / "api-bank"
    sample_dir = root / "lv1-lv2-samples" / "level-2-toolsearcher"
    sample_dir.mkdir(parents=True)
    (sample_dir / "b.jsonl").write_text("{}", encoding="utf-8")
    (sample_dir / "a.jsonl").write_text("{}", encoding="utf-8")
    (sample_dir / "ignore.txt").write_text("x", encoding="utf-8")

    resolved = resolve_api_bank_level_paths(root, ["level-2-toolsearcher"])

    assert [path.name for _level, path in resolved] == ["a.jsonl", "b.jsonl"]


def test_apibank_adapter_reads_level3_apicode_tool_schema():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}},
            }
        )
        + " User: Query meeting of John."
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["ToolSearcher"]


def test_apibank_adapter_narrows_level3_toolsearcher_result_to_discovered_tool():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}},
            }
        )
        + """
        User: Query meeting of John and send email reminder to john@example.com.
        API-Request: [ToolSearcher(keywords='QueryMeeting')]->{'name': 'QueryMeeting', 'description': 'Query meetings.', 'input_parameters': {'user_name': {'type': 'str'}}}
        """
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["QueryMeeting"]


def test_apibank_level3_first_step_exposes_only_toolsearcher():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + " User: Query meeting of John and send email reminder to john@example.com."
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["ToolSearcher"]


def test_apibank_level3_after_toolsearcher_exposes_discovered_tool():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query meeting of John and send email reminder to john@example.com.
        API-Request: [ToolSearcher(keywords='QueryMeeting')]->{'api_name': 'ToolSearcher', 'output': {'name': 'QueryMeeting', 'description': 'Query meetings.', 'input_parameters': {'user_name': {'type': 'str'}}}}
        """
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["QueryMeeting"]


def test_apibank_level3_auth_followup_overrides_bad_toolsearcher_discovery():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Can you help me book a meeting?
        API-Request: [ToolSearcher(keywords='book meeting')]->{'name': 'BookHotel', 'description': 'This API orders a hotel room.', 'input_parameters': {'hotel_name': {'type': 'str'}}}
        AI: Do you have an account with us?
        User: Yes, my username is JohnDoe and my password is pass123.
        AI: Okay, let me get your token.
        """
    }
    catalog = {
        "GetUserToken": {
            "name": "GetUserToken",
            "description": "Get the user token.",
            "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
        },
        "AddMeeting": {
            "name": "AddMeeting",
            "description": "Add a meeting.",
            "input_parameters": {"token": {"type": "str"}, "meeting_topic": {"type": "str"}},
        },
    }
    tool_search_index = {
        "book meeting": [
            catalog["GetUserToken"],
            catalog["AddMeeting"],
            {
                "name": "BookHotel",
                "description": "This API orders a hotel room.",
                "input_parameters": {"hotel_name": {"type": "str"}},
            },
        ]
    }

    tools = tools_from_row(row, api_catalog=catalog, tool_search_index=tool_search_index)

    assert [tool["name"] for tool in tools] == ["GetUserToken"]


def test_apibank_level3_after_data_lookup_exposes_toolsearcher_for_next_discovery():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query meeting of John and send email reminder to john@example.com.
        API-Request: [ToolSearcher(keywords='QueryMeeting')]->{'api_name': 'ToolSearcher', 'output': {'name': 'QueryMeeting', 'description': 'Query meetings.', 'input_parameters': {'user_name': {'type': 'str'}}}}
        API-Request: [QueryMeeting(user_name='John')]->{'api_name': 'QueryMeeting', 'output': {'meetings': [{'meeting_name': 'A'}]}}
        """
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["ToolSearcher"]


def test_apibank_level3_data_lookup_does_not_skip_next_toolsearcher_when_catalog_has_concrete():
    raw_text = (
        json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query organization members and then check their travel status.
        API-Request: [ToolSearcher(keywords='OrganizationMembers')]->{'api_name': 'ToolSearcher', 'output': {'name': 'OrganizationMembers', 'description': 'API to retrieve the list of members in the organization.', 'input_parameters': {'organization': {'type': 'str'}}, 'output_parameters': {'members': {'type': 'list'}}}}
        API-Request: [OrganizationMembers(organization='Alibaba')]->{'api_name': 'OrganizationMembers', 'input': {'organization': 'Alibaba'}, 'output': {'members': ['John', 'Mary']}}
        """
    )
    tools = [
        {
            "name": "ToolSearcher",
            "description": "Searches for relevant tools in library based on the keywords.",
            "input_parameters": {"keywords": {"type": "str"}},
        },
        {
            "name": "OrganizationMembers",
            "description": "API to retrieve the list of members in the organization.",
            "input_parameters": {"organization": {"type": "str"}},
            "output_parameters": {"members": {"type": "list"}},
        },
        {
            "name": "TravelStatus",
            "description": "API for retrieving the current travel status of each member.",
            "input_parameters": {"member_name": {"type": "str"}},
            "output_parameters": {"status": {"type": "str"}},
        },
    ]

    assert [tool["name"] for tool in _level3_next_step_tools(raw_text, tools)] == ["ToolSearcher"]


def test_apibank_level3_repeats_concrete_tool_until_source_values_are_consumed():
    raw_text = (
        json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query organization members and then check their travel status.
        API-Request: [ToolSearcher(keywords='OrganizationMembers')]->{'api_name': 'ToolSearcher', 'output': {'name': 'OrganizationMembers', 'description': 'API to retrieve the list of members in the organization.', 'input_parameters': {'organization': {'type': 'str'}}, 'output_parameters': {'members': {'type': 'list'}}}}
        API-Request: [OrganizationMembers(organization='Alibaba')]->{'api_name': 'OrganizationMembers', 'input': {'organization': 'Alibaba'}, 'output': {'members': ['John', 'Mary']}}
        API-Request: [ToolSearcher(keywords='TravelStatus')]->{'api_name': 'ToolSearcher', 'output': {'name': 'TravelStatus', 'description': 'API for retrieving the current travel status of each member.', 'input_parameters': {'member_name': {'type': 'str'}}, 'output_parameters': {'status': {'type': 'str'}}}}
        API-Request: [TravelStatus(member_name='John')]->{'api_name': 'TravelStatus', 'input': {'member_name': 'John'}, 'output': 'Traveling'}
        """
    )
    tools = [
        {
            "name": "ToolSearcher",
            "description": "Searches for relevant tools in library based on the keywords.",
            "input_parameters": {"keywords": {"type": "str"}},
        },
        {
            "name": "TravelStatus",
            "description": "API for retrieving the current travel status of each member.",
            "input_parameters": {"member_name": {"type": "str"}},
            "output_parameters": {"status": {"type": "str"}},
        },
    ]

    assert [tool["name"] for tool in _level3_next_step_tools(raw_text, tools)] == ["TravelStatus"]


def test_apibank_level3_post_id_lookup_returns_to_toolsearcher_after_lookup():
    raw_text = (
        json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Find post IDs for this user and then count likes for each post.
        API-Request: [ToolSearcher(keywords='UserPosts')]->{'api_name': 'ToolSearcher', 'output': {'name': 'UserPosts', 'description': 'API to retrieve the post IDs for a specific user.', 'input_parameters': {'user_id': {'type': 'int'}}, 'output_parameters': {'post_ids': {'type': 'list'}}}}
        API-Request: [UserPosts(user_id=2)]->{'api_name': 'UserPosts', 'input': {'user_id': 2}, 'output': {'post_ids': [4, 5, 6]}}
        """
    )
    tools = [
        {
            "name": "ToolSearcher",
            "description": "Searches for relevant tools in library based on the keywords.",
            "input_parameters": {"keywords": {"type": "str"}},
        },
        {
            "name": "UserPosts",
            "description": "API to retrieve the post IDs for a specific user.",
            "input_parameters": {"user_id": {"type": "int"}},
            "output_parameters": {"post_ids": {"type": "list"}},
        },
        {
            "name": "LikeCount",
            "description": "API to retrieve the number of likes for the post.",
            "input_parameters": {"post_id": {"type": "int"}},
            "output_parameters": {"like_count": {"type": "int"}},
        },
    ]

    assert [tool["name"] for tool in _level3_next_step_tools(raw_text, tools)] == ["ToolSearcher"]


def test_apibank_level3_input_api_box_ignores_instruction_examples():
    row = {
        "instruction": """
        Example: [ToolSearcher(keywords='calculator')]->{'name': 'Calculator', 'description': 'Calculate arithmetic.', 'input_parameters': {'formula': {'type': 'str'}}}
        """,
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query meeting of John and send email reminder to john@example.com.
        API-Request: [ToolSearcher(keywords='QueryMeeting')]->{'api_name': 'ToolSearcher', 'output': {'name': 'QueryMeeting', 'description': 'Query meetings.', 'input_parameters': {'user_name': {'type': 'str'}}}}
        API-Request: [QueryMeeting(user_name='John')]->{'api_name': 'QueryMeeting', 'output': {'meetings': [{'meeting_name': 'A'}]}}
        """
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["ToolSearcher"]


def test_apibank_level3_repeats_action_tool_after_action_success():
    row = {
        "input": json.dumps(
            {
                "apiCode": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "parameters": {"keywords": {"type": "str"}},
            }
        )
        + """
        User: Query meeting of John and send email reminder to john@example.com.
        API-Request: [ToolSearcher(keywords='QueryMeeting')]->{'api_name': 'ToolSearcher', 'output': {'name': 'QueryMeeting', 'description': 'Query meetings.', 'input_parameters': {'user_name': {'type': 'str'}}}}
        API-Request: [QueryMeeting(user_name='John')]->{'api_name': 'QueryMeeting', 'output': {'meetings': [{'meeting_name': 'A'}, {'meeting_name': 'B'}]}}
        API-Request: [ToolSearcher(keywords='EmailReminder')]->{'api_name': 'ToolSearcher', 'output': {'name': 'EmailReminder', 'description': 'Send email reminders.', 'input_parameters': {'recipient': {'type': 'str'}, 'content': {'type': 'str'}}}}
        API-Request: [EmailReminder(recipient='john@example.com', content='A')]->{'api_name': 'EmailReminder', 'output': 'success'}
        """
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["EmailReminder"]


def test_apibank_adapter_reads_retrieved_api_descriptions_from_transcript():
    row = {
        "instruction": """
        {"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.",
         "input_parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}}
        """,
        "input": """
        User: Can you please tell me the stock price of Microsoft on 4th February 2022?
        API-Request: [ToolSearcher(keywords='Query Stock')]->{'name': 'QueryStock', 'description': 'This API queries the stock price of a given stock.', 'input_parameters': {'stock_code': {'type': 'str', 'description': 'The stock code of the given stock.'}, 'date': {'type': 'str', 'description': 'The date of the stock price.'}}}
        """,
    }

    tools = tools_from_row(row)

    assert [tool["name"] for tool in tools] == ["QueryStock"]


def test_apibank_adapter_builds_catalog_from_nested_schema_lists(tmp_path):
    root = tmp_path / "api-bank"
    test_data = root / "test-data"
    test_data.mkdir(parents=True)
    rows = [
        {
            "instruction": json.dumps(
                {
                    "apis": [
                        {
                            "name": "QueryReminder",
                            "description": "Query reminder items for a user.",
                            "input_parameters": {"token": {"type": "str"}, "content": {"type": "str"}},
                        }
                    ]
                }
            ),
            "input": "",
            "expected_output": "API-Request: [GetUserToken(username='user1', password='user1pass')]",
        }
    ]
    (test_data / "level-2-api.json").write_text(json.dumps(rows), encoding="utf-8")

    catalog = build_api_catalog(root)

    assert list(catalog) == ["QueryReminder"]


def test_apibank_adapter_resolves_named_toolsearcher_discoveries_from_catalog():
    row = {
        "instruction": """
        {"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.",
         "input_parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}}
        """,
        "input": """
        User: Can you query my reminders about dentist appointments?
        API-Request: [ToolSearcher(keywords='query reminder')]->Based on my search, I can use the QueryReminder API. I need a token from the GetUserToken API first.
        AI: Please provide your username and password.
        User: user2 user2pass
        Generate API Request:
        """,
    }
    catalog = {
        "QueryReminder": {
            "name": "QueryReminder",
            "description": "Query reminder items for a user.",
            "input_parameters": {
                "token": {"type": "str", "description": "The user token."},
                "content": {"type": "str", "description": "The reminder content."},
            },
        },
        "GetUserToken": {
            "name": "GetUserToken",
            "description": "Get a user token from username and password.",
            "input_parameters": {
                "username": {"type": "str", "description": "The username."},
                "password": {"type": "str", "description": "The password."},
            },
        },
    }

    tools = tools_from_row(row, api_catalog=catalog)

    assert [tool["name"] for tool in tools] == ["QueryReminder", "GetUserToken"]


def test_apibank_adapter_builds_tool_search_index_from_hidden_trace(tmp_path):
    root = tmp_path / "api-bank"
    sample_dir = root / "lv1-lv2-samples" / "level-2-toolsearcher"
    sample_dir.mkdir(parents=True)
    api_catalog = {
        "GetUserToken": {
            "name": "GetUserToken",
            "description": "Get a user token.",
            "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
        },
        "QueryReminder": {
            "name": "QueryReminder",
            "description": "Query reminder items.",
            "input_parameters": {"token": {"type": "str"}, "content": {"type": "str"}, "time": {"type": "str"}},
        },
    }
    events = [
        {"role": "User", "text": "Can you help me set a reminder?"},
        {"role": "AI", "text": "Sure. [ToolSearcher(keywords='set reminder')]"},
        {"role": "API", "api_name": "GetUserToken", "param_dict": {"username": "u", "password": "p"}},
        {
            "role": "API",
            "api_name": "QueryReminder",
            "param_dict": {"token": "tok", "content": "Book flight tickets"},
        },
    ]
    (sample_dir / "QueryReminder-level-3-2.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )

    index = build_tool_search_index(root, api_catalog=api_catalog)

    assert [tool["name"] for tool in index["set reminder"]] == ["GetUserToken", "QueryReminder"]


def test_apibank_adapter_uses_tool_search_index_for_hidden_discovery():
    row = {
        "instruction": """
        {"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.",
         "input_parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}}
        """,
        "input": """
        User: Can you help me set a reminder to book my flight tickets on March 22 at 8 AM?
        AI: Sure, let me check first. [ToolSearcher(keywords='set reminder')]
        API-Request: [GetUserToken(username='user2', password='user2pass')]->{'token': 'o9i8u7y6t5r4e3w2q1'}
        Generate API Request:
        """,
    }
    catalog = {
        "GetUserToken": {
            "name": "GetUserToken",
            "description": "Get a user token.",
            "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
        },
        "QueryReminder": {
            "name": "QueryReminder",
            "description": "Query reminder items.",
            "input_parameters": {"token": {"type": "str"}, "content": {"type": "str"}, "time": {"type": "str"}},
        },
    }
    tool_search_index = {"set reminder": [catalog["GetUserToken"], catalog["QueryReminder"]]}

    tools = tools_from_row(row, api_catalog=catalog, tool_search_index=tool_search_index)

    assert [tool["name"] for tool in tools] == ["GetUserToken", "QueryReminder"]


def test_apibank_adapter_uses_catalog_text_match_when_only_toolsearcher_is_visible():
    row = {
        "instruction": """
        {"name": "ToolSearcher", "description": "Searches for relevant tools in library based on the keywords.",
         "input_parameters": {"keywords": {"type": "str", "description": "The keyword to search for."}}}
        """,
        "input": """
        User: Can you check my account balance?
        API-Request: [GetUserToken(username='foo', password='bar')]->{'token': 'z9x8c7v6b5n4m3q2w1'}
        AI: Now that you are authenticated, let me check your account balance.
        Generate API Request:
        """,
    }
    catalog = {
        "GetUserToken": {
            "name": "GetUserToken",
            "description": "Get a user token.",
            "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
        },
        "QueryBalance": {
            "name": "QueryBalance",
            "description": "This API queries the balance of a given user.",
            "input_parameters": {"token": {"type": "str", "description": "The token of the user."}},
        },
    }

    tools = tools_from_row(row, api_catalog=catalog)

    assert [tool["name"] for tool in tools] == ["GetUserToken", "QueryBalance"]


def test_apibank_adapter_removes_prior_api_call_prefixes_from_prompt():
    row = {
        "input": "User: Need stock price. API-Request: [ToolSearcher(keywords='Query Stock')]->{'name': 'QueryStock'} Generate API Request:"
    }

    cleaned = prompt_text(row)

    assert "API-Request" not in cleaned
    assert "ToolSearcher(" not in cleaned
    assert "Prior API result" in cleaned


def test_apibank_latest_context_keeps_previous_user_slots_for_confirmation():
    text = (
        "User: Appointment ID is 34567890, new date is March 26th, doctor is Dr. Lee.\n"
        "AI: Do you want me to modify it?\n"
        "User: Yes.\n"
        "Prior API result: success"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user: Appointment ID is 34567890" in context
    assert "User: Yes." in context
    assert "Latest prior API result: success" in context


def test_apibank_latest_context_keeps_assistant_offer_for_yes_please_confirmation():
    text = (
        "User: Can you tell me the stock price for AAPL on January 3rd, 2023?\n"
        "AI: The stock price of AAPL on January 3rd, 2023 is not available. "
        "However, I can tell you the stock price on January 3rd, 2022 if you want?\n"
        "User: Yes, please.\n"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user: Can you tell me the stock price for AAPL on January 3rd, 2023?" in context
    assert "Assistant: The stock price of AAPL on January 3rd, 2023 is not available." in context
    assert "January 3rd, 2022" in context
    assert "User: Yes, please." in context


def test_apibank_latest_context_keeps_previous_user_id_for_followup_values():
    text = (
        "User: My user ID is J46801.\n"
        "AI: What should I record?\n"
        "User: The time is 2023-10-13 09:30:00 and my heart rate is 80.\n"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user: My user ID is J46801." in context
    assert "User: The time is 2023-10-13 09:30:00" in context


def test_apibank_latest_context_omits_previous_user_when_current_has_identity():
    text = (
        "User: Appointment ID is 34567890, new date is March 26th.\n"
        "AI: Done.\n"
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.\n"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user:" not in context
    assert "Appointment ID is 34567890" not in context
    assert "User: My user ID is J46801" in context


def test_apibank_latest_context_omits_previous_user_for_short_value_turn():
    text = (
        "User: Can you help me find possible diseases related to fatigue?\n"
        "AI: Which symptom?\n"
        "User: Fatigue.\n"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user:" not in context
    assert context == "User: Fatigue."


def test_apibank_latest_context_omits_following_assistant_prose_for_short_value_turn():
    text = (
        "User: Fatigue.\n"
        "AI: I understand. Here are some possible diseases related to fatigue: "
        "Chronic fatigue syndrome, Anemia, and Depression. Would you like me to provide more information?\n"
    )

    context = latest_dialogue_context(text)

    assert context == "User: Fatigue."


def test_apibank_latest_context_keeps_previous_request_for_labeled_slot_fill():
    text = (
        "User: Yes, please tell me more about Chronic fatigue syndrome.\n"
        "AI: Chronic fatigue syndrome is a debilitating condition.\n"
        "User: Can you help me book an appointment with Dr. John on 2023-10-15?\n"
        "AI: Please provide your full name to register the appointment.\n"
        "User: My name is Emily Smith.\n"
    )

    context = latest_dialogue_context(text)

    assert "Earlier user: Can you help me book an appointment with Dr. John on 2023-10-15?" in context
    assert "Earlier user: Yes, please tell me more" not in context
    assert "User: My name is Emily Smith." in context


def test_apibank_latest_context_drops_prior_result_before_selected_window():
    text = (
        "User: Can you also check my health data for the past week?\n"
        "AI: I need your user ID and dates.\n"
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.\n"
        "API-Request: [QueryHealthData(user_id='J46801')]->[{'time': '2023-03-11 14:20:00'}]\n"
        "AI: Here are your health data.\n"
        "User: Actually, can you cancel the appointment instead of modifying it?\n"
        "AI: Please provide the appointment ID.\n"
        "User: The appointment ID is 90123456, the patient name is Olivia Davis, "
        "the date is October 10th, and the doctor name is Dr. Smith.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: Actually, can you cancel the appointment instead of modifying it?" in context
    assert "User: The appointment ID is 90123456" in context
    assert "Latest prior API result" not in context
    assert "2023-03-11" not in context


def test_apibank_latest_context_keeps_complete_token_prior_result():
    text = (
        "User: Can you delete my account?\n"
        "AI: I need to authenticate you.\n"
        "User: My email is foo and my password is bar.\n"
        "API-Request: [GetUserToken(username='foo', password='bar')]->{'token': 'z9x8c7v6'}\n"
        "AI: I got the token.\n"
        "User: Yes, delete it.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Latest prior API result: {'token': 'z9x8c7v6'}" in context
    assert context.endswith("Latest prior API result: {'token': 'z9x8c7v6'}")


def test_apibank_latest_context_drops_completed_task_before_auth_followup():
    text = (
        "User: Can you tell me the stock price of NFLX on February 17, 2022?\n"
        "AI: Sure, I can help you with that.\n"
        "API-Request: [QueryStock(stock_code='NFLX', date='2022-02-17')]->605.7\n"
        "AI: The stock price was 605.7.\n"
        "User: What is my current balance?\n"
        "AI: To protect your account, please provide your username and password.\n"
        "User: Okay. My username is user1 and my password is user1pass.\n"
        "API-Request: [GetUserToken(username='user1', password='user1pass')]->{'token': 'n9m8'}\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: What is my current balance?" in context
    assert "stock price of NFLX" not in context
    assert "User: Okay. My username is user1" in context
    assert "Latest prior API result: {'token': 'n9m8'}" in context


def test_apibank_latest_context_keeps_inline_equation_self_contained():
    text = (
        "User: Can you tell me about the historical events of April 21st?\n"
        "AI: Sure, for which year do you want the historical events?\n"
        "User: 1903\n"
        "API-Request: [QueryHistoryToday(date='04-21')]->['event']\n"
        "AI: The New York Yankees played their first game.\n"
        "User: Can you solve this equation for me: (2+3)*5\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert context == "User: Can you solve this equation for me: (2+3)*5"


def test_apibank_latest_context_keeps_final_token_prior_result():
    text = (
        'User: Add "Meeting with John" to my agenda tomorrow at 9am.\n'
        "AI: Sure, I will add that to your agenda.\n"
        "Prior API result:{'token': 'a9s8d7f6g5h4j3k2l1'}"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "User: Add" in context
    assert "Latest prior API result: {'token': 'a9s8d7f6g5h4j3k2l1'}" in context


def test_apibank_latest_context_keeps_token_for_authenticated_slot_followup():
    text = (
        "User: I am looking for an alarm clock API.\n"
        "AI: It requires authentication. Please provide username and password.\n"
        "User: My username is user4 and my password is user4pass.\n"
        "API-Request: [GetUserToken(username='user4', password='user4pass')]->{'token': 'q9w8e7r6'}\n"
        "AI: Now that I have the token, can you please tell me the time for which you want to query alarm?\n"
        "User: I want to check alarms for March 25th, 2023 at 2:10 pm.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "User: I want to check alarms for March 25th" in context
    assert "Latest prior API result: {'token': 'q9w8e7r6'}" in context


def test_apibank_latest_context_keeps_prior_token_for_authenticated_action_slot_fill():
    text = (
        "User: Can you help me modify a meeting reservation?\n"
        "AI: Please provide credentials.\n"
        "User: My username is JaneSmith and my password is password.\n"
        "API-Request: [GetUserToken(username='JaneSmith', password='password')]->{'token': 'o8i7u6'}\n"
        "AI: Now, please provide the meeting information.\n"
        "User: The meeting information is meeting topic is Team Building Activity.\n"
        "API-Request: [ModifyMeeting(token='o8i7u6')]->success\n"
        "AI: The meeting has been successfully modified.\n"
        "User: Can you help me modify an alarm?\n"
        "AI: Sure, please provide me the time for which alarm needs to be modified.\n"
        "User: The alarm time is on March 20, 2023 at 6:30 AM.\n"
        "AI: Now, please provide me with the modified alarm time.\n"
        "User: The modified alarm time is on March 20, 2023 at 7:00 AM.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "User: The modified alarm time is on March 20, 2023 at 7:00 AM." in context
    assert "Latest prior API result: {'token': 'o8i7u6'}" in context


def test_apibank_latest_context_prefers_token_over_recent_success_for_authenticated_slot_fill():
    text = (
        "User: Can you help me modify an alarm for user3 at 2023-03-24 09:00:00?\n"
        "AI: Please provide credentials.\n"
        "User: The credentials are username(user3) and password(user3pass).\n"
        "API-Request: [GetUserToken(username='user3', password='user3pass')]->{'token': 'p9o8i7u6'}\n"
        "AI: User authenticated. Now, please specify the new time for the alarm.\n"
        "User: The new time is 2023-03-24 10:00:00.\n"
        "API-Request: [ModifyAlarm(token='p9o8i7u6')]->success\n"
        "AI: Alarm modified successfully.\n"
        "User: Can you please add an agenda for user3 on 2023-03-24 14:00:00?\n"
        "AI: Sure. Now, please provide me with the agenda content and location.\n"
        'User: The content is "Lunch with friends" and location is "Restaurant X".'
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert 'User: The content is "Lunch with friends" and location is "Restaurant X".' in context
    assert "Latest prior API result: {'token': 'p9o8i7u6'}" in context
    assert "Latest prior API result: success" not in context


def test_apibank_latest_context_drops_tool_catalog_prior_result():
    text = (
        "User: Can you help me book a meeting?\n"
        "AI: Let me search.\n"
        "API-Request: [ToolSearcher(keywords='book meeting')]->"
        "{'name': 'BookHotel', 'description': 'This API orders a hotel room.', "
        "'input_parameters': {'hotel_name': {'type': 'str'}}}\n"
        "AI: Do you have an account with us?\n"
        "User: Yes, my username is JohnDoe and my password is pass123.\n"
        "AI: Okay, let me get your token.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "User: Yes, my username is JohnDoe" in context
    assert "Latest prior API result" not in context
    assert "BookHotel" not in context


def test_apibank_latest_context_keeps_anaphora_context():
    text = (
        "User: Can you help me find some information about a rash?\n"
        "AI: Sure. What specifically do you want to know?\n"
        "User: I have a rash and I want to know what it could be.\n"
        "AI: I found Contact dermatitis and Eczema.\n"
        "User: I'm not sure. Can you tell me more about each of them?\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: I have a rash" in context
    assert "User: I'm not sure. Can you tell me more about each of them?" in context


def test_apibank_latest_context_keeps_task_for_short_value_turn():
    text = (
        "User: Can you check my registration?\n"
        "AI: What name should I search for?\n"
        "User: John Doe\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: Can you check my registration?" in context
    assert "User: John Doe" in context


def test_apibank_latest_context_keeps_recent_slots_for_use_value_turn():
    text = (
        "User: I want to open a bank account.\n"
        "AI: What account identifier would you like to use?\n"
        "User: My account identifier is user4.\n"
        "AI: What password would you like to use?\n"
        "User: My password is user4pass.\n"
        "AI: What should I use as your name on the account?\n"
        "User: Use John.\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: I want to open a bank account." in context
    assert "Earlier user: My account identifier is user4." in context
    assert "Earlier user: My password is user4pass." in context
    assert "User: Use John." in context


def test_apibank_latest_context_keeps_request_for_unlabeled_credentials():
    text = (
        "User: Can you add a reminder for a meeting on 2022-05-06 at 2 PM?\n"
        "AI: For which username should I add the reminder?\n"
        "User: user1\n"
        "AI: Please tell me your username and password.\n"
        "User: user1 user1pass\n"
    )

    context = latest_dialogue_context(prompt_text({"input": text}))

    assert "Earlier user: Can you add a reminder for a meeting on 2022-05-06 at 2 PM?" in context
    assert "Earlier user: user1" in context
    assert "User: user1 user1pass" in context
