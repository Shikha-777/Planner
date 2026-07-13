from __future__ import annotations

from taskdecomp.tool_binding import (
    build_task_frame,
    build_tool_binding_plan,
    extract_numbers,
    infer_argument_value,
    normalize_expected_ground_truth,
    parse_call_string,
    _collapse_identical_repeated_calls,
    _intent_clauses,
    _location_units,
    _order_independent_calls_for_benchmark,
)
from taskdecomp.tool_binding_eval import score_predictions


def test_parse_bfcl_call_string_with_fraction():
    call = parse_call_string("calc_binomial_probability(n=20, k=5, p=1/6)")

    assert call == {
        "tool_name": "calc_binomial_probability",
        "arguments": {"n": 20, "k": 5, "p": 1 / 6},
    }


def test_normalize_possible_answer_ground_truth():
    calls = normalize_expected_ground_truth(
        [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    )

    assert calls == [
        {
            "tool_name": "calculate_triangle_area",
            "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
        }
    ]


def test_intent_clauses_keep_next_time_window_together():
    clauses = _intent_clauses(
        "Provide the stock forecast for Apple for the next 30 days using ARIMA, "
        "and then provide the weather forecast for Boston for the next 7 days."
    )

    assert clauses == [
        "Provide the stock forecast for Apple for the next 30 days using ARIMA",
        "provide the weather forecast for Boston for the next 7 days",
    ]
    assert _intent_clauses(
        "Tell me the stock prediction for Apple Inc. for the next 30 days, "
        "then provide the forecast for Microsoft Corp. for the next 45 days."
    ) == [
        "Tell me the stock prediction for Apple Inc. for the next 30 days",
        "provide the forecast for Microsoft Corp. for the next 45 days",
    ]
    assert _intent_clauses(
        "Get temperature and humidity forecast for Boston and precipitation forecast for Rome."
    ) == [
        "Get temperature and humidity forecast for Boston",
        "precipitation forecast for Rome",
    ]


def test_tool_binding_extracts_stock_forecast_company_not_article():
    value = infer_argument_value(
        "Tell me what the stock price prediction for Apple Inc. is for the next 30 days using the ARIMA model.",
        "company",
        {"type": "string", "description": "The company to forecast."},
    )
    second_model = infer_argument_value(
        "Tell me what the stock price prediction for Apple Inc. is for the next 30 days using the ARIMA model, "
        "then forecast Microsoft Corp. for the next 45 days using the LSTM model.",
        "model",
        {"type": "string", "description": "The forecast model to use."},
        call_index=1,
        call_count=2,
    )

    assert value == "Apple Inc"
    assert second_model == "LSTM"


def _registration_health_tools():
    return [
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
    ]


def test_tool_binding_simple_triangle_area():
    plan = build_tool_binding_plan(
        "Find the area of a triangle with a base of 10 units and height of 5 units.",
        [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate the area of a triangle given its base and height.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer", "description": "The base of the triangle."},
                        "height": {"type": "integer", "description": "The height of the triangle."},
                        "unit": {"type": "string", "description": "The unit of measure."},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["tool_name"] == "calculate_triangle_area"
    assert plan["calls"][0]["arguments"]["base"] == 10
    assert plan["calls"][0]["arguments"]["height"] == 5


def test_tool_binding_rejects_irrelevant_tool():
    plan = build_tool_binding_plan(
        "Calculate the area of a triangle given the base is 10 meters and height is 5 meters.",
        [
            {
                "name": "determine_body_mass_index",
                "description": "Calculate body mass index given weight and height.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "weight": {"type": "float", "description": "Weight in kilograms."},
                        "height": {"type": "float", "description": "Height in meters."},
                    },
                    "required": ["weight", "height"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_routes_when_args_are_present_but_named_generically():
    plan = build_tool_binding_plan(
        "Calculate the hypotenuse of a right triangle given the lengths of the other two sides as 4 and 5.",
        [
            {
                "name": "math.hypot",
                "description": "Calculate the hypotenuse of a right triangle.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "x": {"type": "integer", "description": "First side length."},
                        "y": {"type": "integer", "description": "Second side length."},
                    },
                    "required": ["x", "y"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["math.hypot"]


def test_tool_binding_rejects_semantically_incompatible_tool():
    plan = build_tool_binding_plan(
        "What is the roots of linear equation bx + c = 0?",
        [
            {
                "name": "find_roots",
                "description": "Find the roots of a quadratic equation ax^2 + bx + c = 0.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "float"},
                        "b": {"type": "float"},
                        "c": {"type": "float"},
                    },
                    "required": ["a", "b", "c"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_preserves_raw_hard_conflict_with_semantic_frame():
    plan = build_tool_binding_plan(
        "What is the roots of linear equation bx + c = 0?",
        [
            {
                "name": "find_roots",
                "description": "Find the roots of a quadratic equation ax^2 + bx + c = 0.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "float"},
                        "b": {"type": "float"},
                        "c": {"type": "float"},
                    },
                    "required": ["a", "b", "c"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "call_groups": [
                    {
                        "intent": "find roots",
                        "unit_of_work": "one requested operation/entity/group",
                        "requested_entities": ["roots"],
                        "expected_call_count": 1,
                        "can_use_batch_tool_if_available": True,
                    }
                ],
                "slots_observed": [
                    {
                        "role": "equation",
                        "value": "bx + c = 0",
                        "value_type": "text",
                        "evidence_span": "bx + c = 0",
                        "status": "explicit",
                        "confidence": 0.0,
                    }
                ],
            }
        },
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_rejects_shared_numeric_arg_without_matching_intent():
    plan = build_tool_binding_plan(
        "Solve the quadratic equation with coefficients a = 1, b = 2, and c = 3.",
        [
            {
                "name": "math.sum",
                "description": "Compute the sum of all numbers in a list.",
                "parameters": {
                    "type": "dict",
                    "properties": {"numbers": {"type": "array", "description": "Numbers to add."}},
                    "required": ["numbers"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_rejects_prime_tool_for_plain_closest_integer_request():
    plan = build_tool_binding_plan(
        "What is the closest integer to 30?",
        [
            {
                "name": "get_closest_prime",
                "description": "Get the closest prime number to a given number.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "number": {"type": "integer"},
                        "skip": {"type": "integer"},
                    },
                    "required": ["number", "skip"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "call_groups": [
                    {
                        "intent": "find_closest_integer",
                        "unit_of_work": "one requested operation/entity/group",
                        "requested_entities": ["integer"],
                        "expected_call_count": 1,
                        "can_use_batch_tool_if_available": True,
                    }
                ],
                "slots_observed": [
                    {
                        "role": "target_number",
                        "value": "30",
                        "value_type": "number",
                        "evidence_span": "30",
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
            }
        },
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_carries_game_context_into_guide_calls():
    plan = build_tool_binding_plan(
        (
            "In game Battle Reign, change the armor level to 5 and find me a game guide "
            "for how to win in snowy weather conditions. Also find me any strategy guides "
            "available for game Shadow Fall."
        ),
        [
            {
                "name": "BattleReignGameAPI.update_player_equipment",
                "description": "Modify the player's equipment level for specified attributes",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "attribute": {"type": "string", "description": "The attribute of the equipment to modify."},
                        "level": {"type": "integer", "description": "The level to modify the attribute to."},
                        "playerID": {"type": "integer", "description": "Player ID of the player. Default to 123", "default": 123},
                    },
                    "required": ["attribute", "level"],
                },
            },
            {
                "name": "GameGuideAPI.search_guide",
                "description": "Search for game guides given specific conditions and preferences",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "game": {"type": "string", "description": "Name of the game."},
                        "condition": {
                            "type": "string",
                            "description": "Specific game conditions. (eg: 'snowy weather', 'hard mode').",
                            "default": "",
                        },
                        "type": {
                            "type": "string",
                            "description": "Specific type of guide. (eg: 'strategy', 'walkthrough')",
                            "default": "",
                        },
                    },
                    "required": ["game"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "BattleReignGameAPI.update_player_equipment",
        "GameGuideAPI.search_guide",
        "GameGuideAPI.search_guide",
    ]
    assert plan["calls"][1]["arguments"]["game"] == "Battle Reign"
    assert plan["calls"][1]["arguments"]["condition"] == "snowy weather"
    assert plan["calls"][2]["arguments"]["game"] == "Shadow Fall"
    assert plan["calls"][2]["arguments"]["type"] == "strategy"


def test_tool_binding_carries_recipe_context_into_detail_calls():
    plan = build_tool_binding_plan(
        (
            "I want a homemade healthy spaghetti recipe that is gluten free, how long will it "
            "take to prepare and cook, and what nutritional information could it provide me."
        ),
        [
            {
                "name": "recipe_prep_time",
                "description": "Calculate the estimated preparation and cooking time for a specified recipe.",
                "parameters": {
                    "type": "dict",
                    "properties": {"recipe": {"type": "string", "description": "Name of the recipe to calculate time for."}},
                    "required": ["recipe"],
                },
            },
            {
                "name": "recipe_nutrition_info",
                "description": "Provide detailed nutritional information for a specified recipe.",
                "parameters": {
                    "type": "dict",
                    "properties": {"recipe": {"type": "string", "description": "Name of the recipe to fetch nutrition info for."}},
                    "required": ["recipe"],
                },
            },
            {
                "name": "recipe_search",
                "description": "Search for a recipe based on a particular ingredient or dietary requirement.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "ingredient": {"type": "string", "description": "The ingredient that you want to have in the recipe."},
                        "dietary_requirements": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["gluten_free", "dairy_free", "vegetarian", "vegan"]},
                            "description": "Dietary requirements in the recipe.",
                        },
                        "isHomemade": {
                            "type": "boolean",
                            "description": "If true, returns homemade recipe; otherwise, return not homemade recipe.",
                        },
                    },
                    "required": ["ingredient", "dietary_requirements", "isHomemade"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "recipe_search",
        "recipe_prep_time",
        "recipe_nutrition_info",
    ]
    assert plan["calls"][0]["arguments"]["ingredient"] == "spaghetti"
    assert plan["calls"][1]["arguments"]["recipe"] == "homemade healthy spaghetti"
    assert plan["calls"][2]["arguments"]["recipe"] == "homemade healthy spaghetti"


def test_tool_binding_rejects_tool_with_wrong_domain_object():
    plan = build_tool_binding_plan(
        "Find the area of a rectangle with length 7 and breadth 3. Also calculate the area of a circle with radius 5.",
        [
            {
                "name": "volume_cylinder.calculate",
                "description": "Calculate the volume of a cylinder given the radius and the height.",
                "parameters": {
                    "type": "dict",
                    "properties": {"radius": {"type": "float"}, "height": {"type": "float"}},
                    "required": ["radius", "height"],
                },
            },
            {
                "name": "area_rectangle.calculate",
                "description": "Calculate the area of a rectangle given the length and breadth.",
                "parameters": {
                    "type": "dict",
                    "properties": {"length": {"type": "float"}, "breadth": {"type": "float"}},
                    "required": ["length", "breadth"],
                },
            },
            {
                "name": "area_circle.calculate",
                "description": "Calculate the area of a circle given the radius.",
                "parameters": {
                    "type": "dict",
                    "properties": {"radius": {"type": "float"}},
                    "required": ["radius"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "area_rectangle.calculate",
        "area_circle.calculate",
    ]


def test_tool_binding_prefers_meeting_tool_over_hotel_for_book_meeting():
    plan = build_tool_binding_plan(
        'I got your token. Now let me book the meeting for the topic "Project Update" at 3pm.',
        [
            {
                "name": "BookHotel",
                "description": "This API orders a hotel room.",
                "input_parameters": {},
            },
            {
                "name": "AddMeeting",
                "description": "This API allows users to make a reservation for a meeting.",
                "input_parameters": {},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["AddMeeting"]


def test_tool_binding_prefers_query_meeting_for_check_reservation():
    plan = build_tool_binding_plan(
        "Now I will check the meeting reservation for Marketing Campaign Planning.",
        [
            {
                "name": "AddMeeting",
                "description": "This API allows users to make a reservation for a meeting.",
                "input_parameters": {},
            },
            {
                "name": "QueryMeeting",
                "description": "This API queries a meeting reservation.",
                "input_parameters": {},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryMeeting"]


def test_tool_binding_prefers_delete_meeting_for_cancel_reservation():
    plan = build_tool_binding_plan(
        "Please cancel my reservation for the Board Meeting.",
        [
            {
                "name": "ModifyMeeting",
                "description": "This API allows users to modify a reservation for a meeting.",
                "input_parameters": {},
            },
            {
                "name": "DeleteMeeting",
                "description": "Delete user's reservation for a meeting.",
                "input_parameters": {},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["DeleteMeeting"]


def test_tool_binding_keeps_matching_delete_action_with_auth_context():
    plan = build_tool_binding_plan(
        "Earlier user: Can you help me delete my reservation for the Board Meeting on April 5th at 3:00 PM?\n"
        "User: My username is admin and the password is adminpass.\n"
        "Ai: Before proceeding, I need to get your token for authentication.\n"
        "Ai: Alright, I have received the token.\n"
        "Latest prior API result: {'token': 'm9n8b7v6c5x4z3a2s1'}",
        [
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
            },
            {
                "name": "DeleteMeeting",
                "description": "This API allows users to delete a reservation for a meeting.",
                "input_parameters": {"token": {"type": "str"}, "meeting_topic": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["DeleteMeeting"]


def test_tool_binding_prefers_auth_when_credentials_are_supplied_without_token():
    plan = build_tool_binding_plan(
        "Earlier user: Can you help me book a meeting on March 15th at 3pm?\n"
        "User: Yes, my username is JohnDoe and my password is pass123.\n"
        "Ai: Okay, let me get your token.",
        [
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
            },
            {
                "name": "AddMeeting",
                "description": "This API allows users to make a reservation for a meeting.",
                "input_parameters": {"meeting_topic": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]


def test_tool_binding_maps_person_context_to_patient_name_slot():
    plan = build_tool_binding_plan(
        "Earlier user: Can you please check if I have an appointment on April 12th?\n"
        "User: John Doe",
        [
            {
                "name": "QueryRegistration",
                "description": "This API queries appointment registration.",
                "input_parameters": {
                    "patient_name": {"type": "str"},
                    "date": {"type": "str"},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Check appointment on April 12th for John Doe",
                "slots_observed": [
                    {"role": "date", "value": "2024-04-12", "evidence_span": "April 12th", "confidence": 1.0},
                    {"role": "person", "value": "John Doe", "evidence_span": "John Doe", "confidence": 1.0},
                ],
                "missing_inputs": [],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryRegistration"]
    assert plan["calls"][0]["arguments"]["patient_name"] == "John Doe"


def test_tool_binding_extracts_meeting_attendees_by_name():
    plan = build_tool_binding_plan(
        "Please modify my reservation for Product Development Meeting with attendees Robert Lee, Anna Zhang, and Tony Wang.\n"
        "Latest prior API result: {'token': 'q9w8e7r6t5y4u3i2o1'}",
        [
            {
                "name": "ModifyMeeting",
                "description": "This API allows users to modify a reservation for a meeting.",
                "input_parameters": {
                    "token": {"type": "str"},
                    "meeting_topic": {"type": "str"},
                    "attendees": {"type": "list(str)"},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ModifyMeeting"]
    assert plan["calls"][0]["arguments"]["attendees"] == ["Robert Lee", "Anna Zhang", "Tony Wang"]


def test_tool_binding_binds_symptom_from_context():
    plan = build_tool_binding_plan(
        "Earlier user: Can you help me find some information about a rash?\n"
        "Earlier user: I have a rash and I want to know what it could be.\n"
        "User: I'm not sure. Can you tell me more about each of them?",
        [
            {
                "name": "SymptomSearch",
                "description": "This API searches for a given symptom.",
                "input_parameters": {"symptom": {"type": "str"}},
            },
        ],
    )

    assert plan["calls"][0]["tool_name"] == "SymptomSearch"
    assert plan["calls"][0]["arguments"]["symptom"] == "rash"


def test_tool_binding_binds_symptom_from_information_on_phrase():
    plan = build_tool_binding_plan(
        "User: Can you help me find some information on a rash I've been experiencing?",
        [
            {
                "name": "SymptomSearch",
                "description": "This API searches for a given symptom.",
                "input_parameters": {"symptom": {"type": "str"}},
            },
        ],
    )

    assert plan["calls"][0]["tool_name"] == "SymptomSearch"
    assert plan["calls"][0]["arguments"]["symptom"] == "rash"


def test_tool_binding_rejects_query_tool_for_schedule_meeting_when_add_tool_exists():
    plan = build_tool_binding_plan(
        "Earlier user: Can you help me schedule a meeting with John for next Monday at 2 pm in the meeting room?\n"
        "Latest prior API result: {'token': 'a9s8d7f6g5h4j3k2l1'}",
        [
            {
                "name": "QueryAgenda",
                "description": "The API for getting a schedule item includes parameters for token, content, time, and location.",
                "input_parameters": {
                    "token": {"type": "str"},
                    "content": {"type": "str"},
                    "time": {"type": "str"},
                    "location": {"type": "str"},
                },
            },
            {
                "name": "AddMeeting",
                "description": "This API allows users to make a reservation for a meeting.",
                "input_parameters": {"token": {"type": "str"}, "meeting_topic": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["AddMeeting"]


def test_tool_binding_rejects_add_tool_for_query_meeting_when_query_tool_exists():
    plan = build_tool_binding_plan(
        "Earlier user: My username is testuser and password is testpass.\n"
        "Now I will check the meeting reservation for Marketing Campaign Planning.\n"
        "Latest prior API result: {'token': 'p9o8i7u6y5t4k3e2w1q'}",
        [
            {
                "name": "AddMeeting",
                "description": "This API allows users to make a reservation for a meeting.",
                "input_parameters": {"token": {"type": "str"}, "meeting_topic": {"type": "str"}},
            },
            {
                "name": "QueryMeeting",
                "description": "This API queries a meeting reservation.",
                "input_parameters": {"user_name": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryMeeting"]


def test_tool_binding_does_not_treat_have_set_as_create_action():
    plan = build_tool_binding_plan(
        "Earlier user: Can you check if I have set any alarms tomorrow morning?\n"
        "Latest prior API result: {'token': 'q9w8e7r6t5y4u3i2o1'}",
        [
            {
                "name": "QueryAlarm",
                "description": "This API checks whether the user has alarms.",
                "input_parameters": {"token": {"type": "str"}, "time": {"type": "str"}},
            },
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryAlarm"]
    assert plan["calls"][0]["arguments"]["time"] == "tomorrow morning"


def test_tool_binding_skips_auth_tool_when_token_exists_and_concrete_tool_matches():
    plan = build_tool_binding_plan(
        "Earlier user: My username is testuser and password is testpass.\n"
        "Now I will check the meeting reservation for Marketing Campaign Planning.\n"
        "Latest prior API result: {'token': 'p9o8i7u6y5t4k3e2w1q'}",
        [
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
            },
            {
                "name": "QueryMeeting",
                "description": "This API queries a meeting reservation.",
                "input_parameters": {"user_name": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryMeeting"]


def test_tool_binding_trusts_modify_tool_name_over_bad_delete_description():
    plan = build_tool_binding_plan(
        "Let me modify your reservation for Product Development Meeting.\n"
        "Latest prior API result: {'token': 'q9w8e7r6t5y4u3i2o1'}",
        [
            {
                "name": "ModifyMeeting",
                "description": "This API allows users to modify a reservation for a meeting. Function: Delete user's reservation for a meeting.",
                "input_parameters": {"token": {"type": "str"}, "meeting_topic": {"type": "str"}},
            },
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {"username": {"type": "str"}, "password": {"type": "str"}},
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ModifyMeeting"]


def test_tool_binding_repeats_same_tool_for_repeat_requests():
    plan = build_tool_binding_plan(
        "Calculate the induced electromagnetic force for a magnetic field of 5 Tesla, area of 2 square meters and change in time of 4 seconds, then repeat with a change in time of 10 seconds.",
        [
            {
                "name": "calculate_em_force",
                "description": "Calculate induced electromagnetic force.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "b_field": {"type": "float", "description": "magnetic field"},
                        "area": {"type": "float", "description": "area"},
                        "d_time": {"type": "float", "description": "change in time"},
                    },
                    "required": ["b_field", "area", "d_time"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_em_force", "calculate_em_force"]


def test_tool_binding_repeats_same_tool_for_coordinated_artists():
    plan = build_tool_binding_plan(
        "Play songs from the artists Taylor Swift and Maroon 5, with a play time of 20 minutes and 15 minutes respectively, on Spotify.",
        [
            {
                "name": "spotify.play",
                "description": "Play specific tracks from a given artist for a specific time duration.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "artist": {"type": "string", "description": "The artist whose songs you want to play."},
                        "duration": {"type": "integer", "description": "Duration in minutes."},
                    },
                    "required": ["artist", "duration"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["spotify.play", "spotify.play"]


def test_tool_binding_repeats_same_tool_for_multiple_enum_values():
    plan = build_tool_binding_plan(
        "Calculate the resistance of a wire with a length of 5m and cross sectional area 0.01m² with resistivity of copper and aluminum.",
        [
            {
                "name": "calculate_resistance",
                "description": "Calculate the resistance of a wire using resistivity, length, and cross-sectional area.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "length": {"type": "integer", "description": "The length of the wire in meters."},
                        "area": {"type": "float", "description": "The cross-sectional area of the wire."},
                        "resistivity": {
                            "type": "string",
                            "description": "Resistivity of the material. Allowed values: 'copper', 'aluminum'.",
                        },
                    },
                    "required": ["length", "area"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_resistance",
        "calculate_resistance",
    ]


def test_tool_binding_selects_multiple_tools_in_prompt_order():
    plan = build_tool_binding_plan(
        "Find the sum of all the multiples of 3 and 5 between 1 and 1000. Also find the product of the first five prime numbers.",
        [
            {
                "name": "math_toolkit.product_of_primes",
                "description": "Find product of first prime numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            },
            {
                "name": "math_toolkit.sum_of_multiples",
                "description": "Find sum of multiples in a range.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "lower_limit": {"type": "integer"},
                        "upper_limit": {"type": "integer"},
                        "multiples": {"type": "array"},
                    },
                    "required": ["lower_limit", "upper_limit", "multiples"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "math_toolkit.sum_of_multiples",
        "math_toolkit.product_of_primes",
    ]


def test_tool_binding_expands_common_math_aliases_for_multiple_tools():
    plan = build_tool_binding_plan(
        "Calculate the Greatest Common Divisor (GCD) of 96 and 128, and the least common multiple (LCM) of 15 and 25.",
        [
            {
                "name": "primeFactors",
                "description": "Find all prime factors of an integer.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num": {"type": "integer"}},
                    "required": ["num"],
                },
            },
            {
                "name": "lcm",
                "description": "Calculate the least common multiple of two integers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                    "required": ["num1", "num2"],
                },
            },
            {
                "name": "gcd",
                "description": "Calculate the greatest common divisor of two integers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                    "required": ["num1", "num2"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["gcd", "lcm"]


def test_tool_binding_scopes_multi_intent_numbers_by_tool():
    semantic_frame = {
        "canonical_request": "Calculate GCD of 96 and 128, and LCM of 15 and 25.",
        "slots_observed": [
            {"role": "gcd_numbers", "value": [96, 128], "evidence_span": "96 and 128"},
            {"role": "lcm_numbers", "value": [15, 25], "evidence_span": "15 and 25"},
        ],
        "call_groups": [
            {"intent": "calculate GCD", "unit_of_work": "GCD of 96 and 128", "expected_call_count": 1},
            {"intent": "calculate LCM", "unit_of_work": "LCM of 15 and 25", "expected_call_count": 1},
        ],
    }
    plan = build_tool_binding_plan(
        "Calculate the Greatest Common Divisor (GCD) of 96 and 128, and the least common multiple (LCM) of 15 and 25.",
        [
            {
                "name": "primeFactors",
                "description": "Find all prime factors of an integer.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num": {"type": "integer", "description": "The integer."}},
                    "required": ["num"],
                },
            },
            {
                "name": "lcm",
                "description": "Calculate the least common multiple of two integers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                    "required": ["num1", "num2"],
                },
            },
            {
                "name": "gcd",
                "description": "Calculate the greatest common divisor of two integers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"num1": {"type": "integer"}, "num2": {"type": "integer"}},
                    "required": ["num1", "num2"],
                },
            },
        ],
        capability_plan={"semantic_input_frame": semantic_frame},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["gcd", "lcm"]
    assert [(call["arguments"]["num1"], call["arguments"]["num2"]) for call in plan["calls"]] == [(96, 128), (15, 25)]


def test_tool_binding_scopes_shared_numeric_slots_for_distinct_physics_tools():
    plan = build_tool_binding_plan(
        "Calculate the magnetic field produced by a wire carrying a current of 4 amps with a distance of 2 m from the wire. "
        "And find the voltage difference of a region in the direction of the electric field that is 3 m apart, assuming the electric field is 5 N/C.",
        [
            {
                "name": "calculate_voltage_difference",
                "description": "Calculate voltage difference from electric field and distance.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "electric_field": {"type": "float", "description": "The electric field in newtons per coulomb."},
                        "distance": {"type": "float", "description": "The distance in meters."},
                    },
                    "required": ["electric_field", "distance"],
                },
            },
            {
                "name": "calculate_magnetic_field",
                "description": "Calculate the magnetic field produced by a current in a wire.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "current": {"type": "float", "description": "The current in amperes."},
                        "distance": {"type": "float", "description": "The distance from the wire in meters."},
                    },
                    "required": ["current", "distance"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_magnetic_field",
        "calculate_voltage_difference",
    ]


def test_tool_binding_scopes_money_amounts_for_conversion_and_deposit():
    plan = build_tool_binding_plan(
        "I need to convert 10 dollars to Euros and make a 10 dollar deposit in my local bank account with account number - 987654.",
        [
            {
                "name": "banking_service",
                "description": "Deposit money to a bank account.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "account_id": {"type": "string", "description": "Target account to make deposit to."},
                        "amount": {"type": "float", "description": "Amount to deposit."},
                    },
                    "required": ["account_id", "amount"],
                },
            },
            {
                "name": "currency_conversion",
                "description": "Convert currency amounts.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "amount": {"type": "float", "description": "Amount to convert."},
                        "from_currency": {"type": "string", "description": "Source currency."},
                        "to_currency": {"type": "string", "description": "Target currency."},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["currency_conversion", "banking_service"]


def test_tool_binding_orders_tools_by_specific_name_tokens():
    plan = build_tool_binding_plan(
        "Book a flight from Seattle to Boston with American Airlines and book a hotel in Boston for 4 nights.",
        [
            {
                "name": "hotel_book",
                "description": "Book a hotel in a location.",
                "parameters": {
                    "type": "dict",
                    "properties": {"location": {"type": "string"}, "nights": {"type": "integer"}},
                    "required": ["location", "nights"],
                },
            },
            {
                "name": "flight_book",
                "description": "Book a flight from an origin to a destination.",
                "parameters": {
                    "type": "dict",
                    "properties": {"_from": {"type": "string"}, "to": {"type": "string"}, "airlines": {"type": "string"}},
                    "required": ["_from", "to", "airlines"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["flight_book", "hotel_book"]


def test_tool_binding_rejects_symbolic_numeric_placeholders():
    plan = build_tool_binding_plan(
        "How far will a car travel in time 't' when launched with velocity 'v' at an angle 'theta'?",
        [
            {
                "name": "calculate_projectile_range",
                "description": "Calculate projectile range from initial velocity and angle.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "initial_velocity": {"type": "float"},
                        "angle": {"type": "float"},
                    },
                    "required": ["initial_velocity", "angle"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_abstains_on_meta_no_query():
    plan = build_tool_binding_plan(
        "The user did not provide a query",
        [
            {
                "name": "ControlAppliance.execute",
                "description": "Execute an appliance command.",
                "parameters": {
                    "type": "dict",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_accepts_explicit_function_name_request():
    plan = build_tool_binding_plan(
        "Can you use the update_user_info function to update the email for user ID 12345 to maya@example.com?",
        [
            {
                "name": "update_user_info",
                "description": "Update fields for a user.",
                "parameters": {
                    "type": "dict",
                    "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["user_id", "email"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["update_user_info"]


def test_tool_binding_asks_for_missing_required_slot_before_calling():
    plan = build_tool_binding_plan(
        "Can you use the update_user_info function to update the email for user ID 12345?",
        [
            {
                "name": "update_user_info",
                "description": "Update fields for a user.",
                "parameters": {
                    "type": "dict",
                    "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}},
                    "required": ["user_id", "email"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "ask_user"
    assert plan["calls"] == []
    assert plan["missing_inputs"] == ["email"]
    assert plan["candidate_tool_audits"][0]["missing_slots"] == ["email"]


def test_tool_binding_normalizes_apibank_argument_schema():
    plan = build_tool_binding_plan(
        "Search for available hotels in Boston.",
        [
            {
                "name": "HotelSearch",
                "description": "Search for available hotels by city.",
                "arguments": {"city": {"type": "str", "description": "Destination city."}},
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["HotelSearch"]


def test_extract_numbers_ignores_unit_exponents():
    assert extract_numbers("temperature is 298K and volume is 10 m^3") == [298, 10]


def test_task_frame_exposes_neutral_intent_and_atomic_units():
    frame = build_task_frame("What is the current weather in Lisbon and Shanghai?")

    assert "current_weather" in frame["intent_tags"]
    assert frame["parallelizable"] is True
    assert frame["unit_count"] == 2


def test_tool_binding_multilingual_weather_recall():
    plan = build_tool_binding_plan(
        "我想知道上海目前的天气状况，可以帮我查询吗？顺便使用摄氏度来显示温度。",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["get_current_weather"]


def test_tool_binding_parallel_weather_uses_one_call_per_location():
    plan = build_tool_binding_plan(
        "Could you tell me the current temperature in Boston, MA and San Francisco, CA?",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "get_current_weather",
        "get_current_weather",
    ]


def test_tool_binding_single_city_region_weather_stays_one_call():
    plan = build_tool_binding_plan(
        "Im at Riga, Latvia, Can you tell me the current temperature?",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_current_weather"]


def test_tool_binding_batch_capable_weather_list_stays_one_call():
    plan = build_tool_binding_plan(
        "Could you tell me the current temperature in Boston, MA and San Francisco, CA?",
        [
            {
                "name": "get_current_weather_batch",
                "description": "Get current weather for a list of cities.",
                "parameters": {
                    "type": "dict",
                    "properties": {"cities": {"type": "array", "items": {"type": "string"}}},
                    "required": ["cities"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_current_weather_batch"]


def test_tool_binding_requirement_frame_audits_slots_and_cardinality():
    plan = build_tool_binding_plan(
        "Find Dune showtimes at the Tivoli and the Alamo Drafthouse for tomorrow.",
        [
            {
                "name": "find_movie_showtimes",
                "description": "Find movie showtimes at one theater for a movie and date.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "movie": {"type": "string"},
                        "theater": {"type": "string"},
                        "date": {"type": "string"},
                    },
                    "required": ["movie", "theater", "date"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "find_movie_showtimes",
        "find_movie_showtimes",
    ]
    audit = plan["candidate_tool_audits"][0]
    assert audit["missing_slots"] == []
    assert audit["slot_availability"]["theater"] == "available_from_query_repeated_entity"
    assert audit["call_policy"]["call_count"] == 2
    assert audit["requirement_frame"]["required_slots_available"] is True
    assert audit["requirement_frame"]["expected_call_count_if_single_entity_tool"] == 2


def test_tool_binding_recovers_single_tool_with_complete_required_slots():
    plan = build_tool_binding_plan(
        "I want to know the rise and fall of Christianity in Egypt and Turkey from 100 A.D to 1500 A.D.",
        [
            {
                "name": "religion_history.track",
                "description": "Track the historical development of a specific religion in a specific area within a specific time frame.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "region": {"type": "string"},
                        "religion": {"type": "string"},
                        "start_year": {"type": "integer"},
                        "end_year": {"type": "integer"},
                    },
                    "required": ["region", "religion", "start_year", "end_year"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == [
        "religion_history.track",
        "religion_history.track",
    ]
    assert [call["arguments"]["region"] for call in plan["calls"]] == ["Egypt", "Turkey"]
    assert [call["arguments"]["start_year"] for call in plan["calls"]] == [100, 100]
    assert [call["arguments"]["end_year"] for call in plan["calls"]] == [1500, 1500]
    assert plan["candidate_tool_audits"][0]["recovery_reason"] == "single_tool_complete_required_slots"


def test_tool_binding_capability_prepass_softens_unbound_available_slots():
    plan = build_tool_binding_plan(
        "How to assess the population growth in deer and their impact on woodland in Washington state over the past decade?",
        [
            {
                "name": "wildlife_population.assess_growth",
                "description": "Assess wildlife population growth and ecological impact for a species in an ecosystem over a duration.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "species": {"type": "string"},
                        "duration": {"type": "string"},
                        "ecosystem": {"type": "string"},
                    },
                    "required": ["species", "duration", "ecosystem"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["capability_plan"]["available"] is True
    audit = plan["candidate_tool_audits"][0]
    assert audit["missing_slots"] == []
    assert audit["slot_availability"]["species"] == "available_from_query"
    assert audit["slot_availability"]["duration"] == "available_from_query"
    assert audit["slot_availability"]["ecosystem"] == "available_from_query"
    assert set(plan["calls"][0]["arguments"]) >= {"species", "duration", "ecosystem"}


def test_tool_binding_population_growth_prefers_duration_backed_growth_tool():
    plan = build_tool_binding_plan(
        "How to assess the population growth in deer and their impact on woodland in Washington state over the past decade?",
        [
            {
                "name": "wildlife_population.assess_growth",
                "description": "Assesses the population growth of a specific species in a specified location over a period.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "species": {"type": "string", "description": "The species for which the growth is to be calculated."},
                        "location": {"type": "string", "description": "The area where the species is present."},
                        "duration": {"type": "integer", "description": "The time period for which the population growth should be calculated in years."},
                    },
                    "required": ["species", "location", "duration"],
                },
            },
            {
                "name": "ecological_impact.analyze",
                "description": "Analyzes the impact of a species on a particular ecosystem.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "species": {"type": "string", "description": "The species whose impact is to be calculated."},
                        "ecosystem": {"type": "string", "description": "The ecosystem being affected."},
                        "location": {"type": "string", "description": "The area where the impact is analyzed."},
                        "timeframe": {"type": "integer", "description": "The time period for which the impact analysis should be carried out in years.", "default": 5},
                    },
                    "required": ["species", "ecosystem", "location"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["wildlife_population.assess_growth"]
    assert plan["calls"][0]["arguments"]["duration"] == 10


def test_tool_binding_database_delete_columns_binds_operation():
    plan = build_tool_binding_plan(
        "I need to delete some columns from my employees database on personal_data table. "
        "I want to remove their email addresses and social security numbers to respect privacy.",
        [
            {
                "name": "database.modify_columns",
                "description": "This function allows deletion or addition of columns in a database",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "db_name": {"type": "string", "description": "The name of the database to modify."},
                        "table": {"type": "string", "description": "The name of the table to modify."},
                        "operation": {"type": "string", "description": "The operation to carry out on the table. Can be 'delete' or 'add'."},
                        "columns": {"type": "array", "description": "List of the columns to add or delete from the table.", "items": {"type": "string"}},
                    },
                    "required": ["db_name", "table", "operation", "columns"],
                },
            },
            {
                "name": "database.create_backup",
                "description": "This function creates a backup of the database before modification",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "db_name": {"type": "string"},
                        "backup_location": {"type": "string"},
                    },
                    "required": ["db_name", "backup_location"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["database.modify_columns"]
    assert plan["calls"][0]["arguments"]["db_name"] == "employees"
    assert plan["calls"][0]["arguments"]["table"] == "personal_data"
    assert plan["calls"][0]["arguments"]["operation"] == "delete"
    assert plan["calls"][0]["arguments"]["columns"] == ["email", "social_security_number"]


def test_tool_binding_parallel_movie_time_pairs_make_separate_calls():
    plan = build_tool_binding_plan(
        "Find two movie theatres near San Diego with availability for Tenet at 5 pm and No Time To Die at 7:30 pm.",
        [
            {
                "name": "find_movie_showing",
                "description": "Find local movie theatres and their schedule for a specific movie",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Diego, CA"},
                        "movie": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["Tenet", "No Time To Die"]},
                            "description": "Preferred movie to watch.",
                        },
                        "time": {"type": "array", "items": {"type": "string", "description": "Show time for each movie"}},
                    },
                    "required": ["location", "movie", "time"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["find_movie_showing", "find_movie_showing"]
    assert [call["arguments"]["location"] for call in plan["calls"]] == ["San Diego", "San Diego"]
    assert [call["arguments"]["movie"] for call in plan["calls"]] == [["Tenet"], ["No Time To Die"]]
    assert [call["arguments"]["time"] for call in plan["calls"]] == [["5 pm"], ["7:30 pm"]]


def test_tool_binding_parallel_numeric_location_pairs_make_separate_calls():
    plan = build_tool_binding_plan(
        "Predict house price for a house of size 3000 sq ft. in location New York and 4000 sq ft. in Los Angeles using Machine Learning Model.",
        [
            {
                "name": "ml.predict_house_price",
                "description": "Predict house price using Machine Learning model given the house size and location",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "Location of the house"},
                        "size": {"type": "integer", "description": "Size of the house in square feet"},
                    },
                    "required": ["location", "size"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ml.predict_house_price", "ml.predict_house_price"]
    assert [call["arguments"]["location"] for call in plan["calls"]] == ["New York", "Los Angeles"]
    assert [call["arguments"]["size"] for call in plan["calls"]] == [3000, 4000]


def test_tool_binding_parallel_capital_gain_state_scenarios_make_separate_calls():
    plan = build_tool_binding_plan(
        "What will be the capital gains tax for a short term capital gains of $15000, "
        "long term gains of $25000 in the state of California and $20000 short term, "
        "$50000 long term in Florida?",
        [
            {
                "name": "calculate_capital_gains_tax",
                "description": "Calculate the capital gains tax for a given gains type and amount",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "short_term_gain": {"type": "integer", "description": "The short term capital gain amount."},
                        "long_term_gain": {"type": "integer", "description": "The long term capital gain amount."},
                        "state": {"type": "string", "description": "The state where the income is generated.", "default": "federal"},
                    },
                    "required": ["short_term_gain", "long_term_gain"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_capital_gains_tax", "calculate_capital_gains_tax"]
    assert [call["arguments"]["short_term_gain"] for call in plan["calls"]] == [15000, 20000]
    assert [call["arguments"]["long_term_gain"] for call in plan["calls"]] == [25000, 50000]
    assert [call["arguments"]["state"] for call in plan["calls"]] == ["California", "Florida"]


def test_tool_binding_parallel_stock_companies_make_separate_calls_with_enum_data_points():
    plan = build_tool_binding_plan(
        "Get the latest closing prices and volumes for Apple Inc., Google LLC., and Microsoft Corporation in the New York Stock Exchange",
        [
            {
                "name": "get_stock_data",
                "description": "Retrieve the most recent trading day's closing price and volume for a specified stock.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "symbol": {"type": "string", "description": "The stock symbol of the company."},
                        "data_points": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["price", "volume"]},
                            "description": "The type of data you want to retrieve for the stock.",
                        },
                    },
                    "required": ["symbol", "data_points"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_stock_data", "get_stock_data", "get_stock_data"]
    assert [call["arguments"]["symbol"] for call in plan["calls"]] == ["AAPL", "GOOG", "MSFT"]
    assert [call["arguments"]["data_points"] for call in plan["calls"]] == [["price", "volume"]] * 3


def test_tool_binding_rejects_wrong_shape_tool_for_circle_request():
    plan = build_tool_binding_plan(
        "Find the area and perimeter of a circle with a radius of 5 and also find the circumference of a circle with diameter of 10.",
        [
            {
                "name": "circle.calculate_circumference",
                "description": "Calculate the circumference of a circle based on the diameter.",
                "parameters": {"type": "dict", "properties": {"diameter": {"type": "integer"}}, "required": ["diameter"]},
            },
            {
                "name": "circle.calculate_area",
                "description": "Calculate the area of a circle based on the radius.",
                "parameters": {"type": "dict", "properties": {"radius": {"type": "integer"}}, "required": ["radius"]},
            },
            {
                "name": "rectangle.calculate_perimeter",
                "description": "Calculate the perimeter of a rectangle based on the length and breadth.",
                "parameters": {
                    "type": "dict",
                    "properties": {"length": {"type": "integer"}, "breadth": {"type": "integer"}},
                    "required": ["length", "breadth"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["circle.calculate_area", "circle.calculate_circumference"]


def test_tool_binding_rectangle_property_list_makes_separate_calls():
    request = "What are the length and the width of a rectangle which has a perimeter of 14 and area of 15."
    tools = [
            {
                "name": "get_rectangle_property",
                "description": "Get specific property of the rectangle (like length, width) based on perimeter and area.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "perimeter": {"type": "integer", "description": "Perimeter of the rectangle."},
                        "area": {"type": "integer", "description": "Area of the rectangle."},
                        "property": {"type": "string", "description": "Specific property required. It can be length, width or diagonal."},
                    },
                    "required": ["perimeter", "area", "property"],
                },
            }
        ]
    plan = build_tool_binding_plan(request, tools)

    assert [call["tool_name"] for call in plan["calls"]] == ["get_rectangle_property", "get_rectangle_property"]
    assert [call["arguments"]["property"] for call in plan["calls"]] == ["length", "width"]

    semantic_capped_plan = build_tool_binding_plan(
        request,
        tools,
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": request,
                "slots_observed": [
                    {"role": "perimeter", "value": 14, "value_type": "number", "evidence_span": "perimeter of 14"},
                    {"role": "area", "value": 15, "value_type": "number", "evidence_span": "area of 15"},
                ],
                "call_groups": [
                    {
                        "intent": "calculate rectangle dimensions",
                        "requested_entities": ["length", "width"],
                        "expected_call_count": 1,
                    }
                ],
                "missing_inputs": ["length", "width"],
            }
        },
    )
    assert [call["tool_name"] for call in semantic_capped_plan["calls"]] == [
        "get_rectangle_property",
        "get_rectangle_property",
    ]


def test_tool_binding_rejects_force_tool_for_time_required_request():
    plan = build_tool_binding_plan(
        "Calculate the time required for a car moving at 50 m/s to travel a distance of 600 m. "
        "Also calculate the time required for a bullet moving at 400 m/s to cover a distance of 1000 m.",
        [
            {
                "name": "physics.calculate_force",
                "description": "Calculate the force required to move an object of a particular mass at a particular acceleration.",
                "parameters": {
                    "type": "dict",
                    "properties": {"mass": {"type": "integer"}, "acceleration": {"type": "integer"}},
                    "required": ["mass", "acceleration"],
                },
            },
            {
                "name": "kinematics.calculate_time",
                "description": "Calculate time required for an object to travel a particular distance at a particular velocity.",
                "parameters": {
                    "type": "dict",
                    "properties": {"velocity": {"type": "integer"}, "distance": {"type": "integer"}},
                    "required": ["velocity", "distance"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["kinematics.calculate_time", "kinematics.calculate_time"]


def test_tool_binding_prefers_musical_ticket_over_concert_ticket():
    plan = build_tool_binding_plan(
        "Buy me a ticket to the Mamma Mia musical for next Friday, June 30th 2023, also get me a train ticket from New York to Chicago for the same day.",
        [
            {"name": "train_ticket.buy", "description": "Buy a train ticket for a specific date and route.", "parameters": {"type": "dict", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}},
            {"name": "musical_ticket.buy", "description": "Buy a ticket for a musical", "parameters": {"type": "dict", "properties": {"show": {"type": "string"}, "date": {"type": "string"}}, "required": ["show", "date"]}},
            {"name": "concert_ticket.buy", "description": "Buy a concert ticket", "parameters": {"type": "dict", "properties": {"artist": {"type": "string"}, "date": {"type": "string"}}, "required": ["artist", "date"]}},
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["musical_ticket.buy", "train_ticket.buy"]


def test_tool_binding_event_name_semantic_frame_matches_show_slot():
    prompt = "Buy me a ticket to the Mamma Mia musical for next Friday, June 30th 2023, also get me a train ticket from New York to Chicago for the same day."
    plan = build_tool_binding_plan(
        prompt,
        [
            {"name": "train_ticket.buy", "description": "Buy a train ticket for a specific date and route.", "parameters": {"type": "dict", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}},
            {"name": "musical_ticket.buy", "description": "Buy a ticket for a musical", "parameters": {"type": "dict", "properties": {"show": {"type": "string"}, "date": {"type": "string"}}, "required": ["show", "date"]}},
            {"name": "concert_ticket.buy", "description": "Buy a concert ticket", "parameters": {"type": "dict", "properties": {"artist": {"type": "string"}, "date": {"type": "string"}}, "required": ["artist", "date"]}},
        ],
        capability_plan={
            "semantic_input_frame": {
                "call_groups": [
                    {
                        "intent": "book event ticket",
                        "unit_of_work": "ticket booking",
                        "requested_entities": ["Mamma Mia musical ticket"],
                        "expected_call_count": 1,
                        "can_use_batch_tool_if_available": True,
                    },
                    {
                        "intent": "book train ticket",
                        "unit_of_work": "train ticket booking",
                        "requested_entities": ["train ticket"],
                        "expected_call_count": 1,
                        "can_use_batch_tool_if_available": True,
                    },
                ],
                "slots_observed": [
                    {
                        "role": "event_name",
                        "value": "Mamma Mia musical",
                        "value_type": "text",
                        "evidence_span": "Mamma Mia musical",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "event_date",
                        "value": "2023-06-30",
                        "value_type": "date",
                        "evidence_span": "next Friday, June 30th 2023",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "train_origin",
                        "value": "New York",
                        "value_type": "location",
                        "evidence_span": "New York",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "train_destination",
                        "value": "Chicago",
                        "value_type": "location",
                        "evidence_span": "Chicago",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "train_date",
                        "value": "2023-06-30",
                        "value_type": "date",
                        "evidence_span": "same day",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["musical_ticket.buy", "train_ticket.buy"]


def test_tool_binding_repeats_energy_for_multiple_substances():
    plan = build_tool_binding_plan(
        "Calculate the energy required to heat 100 grams of water from 25 degrees Celsius to 100 degrees Celsius in joules, "
        "and also calculate the energy required to heat the same mass of Aluminium under same conditions in joules",
        [
            {
                "name": "energy_calculator.calculate",
                "description": "Calculate the energy needed to heat a substance from an initial to a final temperature.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "substance": {"type": "string"},
                        "mass": {"type": "float"},
                        "initial_temperature": {"type": "float"},
                        "final_temperature": {"type": "float"},
                    },
                    "required": ["substance", "mass", "initial_temperature", "final_temperature"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["energy_calculator.calculate", "energy_calculator.calculate"]
    assert [call["arguments"]["substance"] for call in plan["calls"]] == ["water", "aluminium"]


def test_tool_binding_orders_history_before_projection_by_prompt_order():
    plan = build_tool_binding_plan(
        "Give me the population size of tigers in Bangladesh and India for the last 5 years. "
        "Also provide the projected population size of tigers in Nepal and Malaysia for the next 10 years.",
        [
            {"name": "animal_population.get_history", "description": "Retrieve historical population size of a specific species in a given country.", "parameters": {"type": "dict", "properties": {"country": {"type": "string"}, "species": {"type": "string"}, "years": {"type": "integer"}}, "required": ["country", "species", "years"]}},
            {"name": "animal_population.get_projection", "description": "Predict the future population size of a specific species in a given country.", "parameters": {"type": "dict", "properties": {"country": {"type": "string"}, "species": {"type": "string"}, "years": {"type": "integer"}}, "required": ["country", "species", "years"]}},
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "animal_population.get_history",
        "animal_population.get_history",
        "animal_population.get_projection",
        "animal_population.get_projection",
    ]


def test_tool_binding_restaurant_pairs_and_flight_all_selected():
    plan = build_tool_binding_plan(
        "Find a Chinese restaurant near me in New York and suggest a high-rated of 4 Italian restaurant in Los Angeles. "
        "Then find a cheapest flight for round-trip from New York to Los Angeles",
        [
            {"name": "restaurant.search", "description": "Find a restaurant in a specified location based on the cuisine and ratings.", "parameters": {"type": "dict", "properties": {"location": {"type": "string"}, "cuisine": {"type": "string"}, "rating": {"type": "float"}}, "required": ["location", "cuisine"]}},
            {"name": "flight.search", "description": "Find flights between two cities.", "parameters": {"type": "dict", "properties": {"_from": {"type": "string"}, "to": {"type": "string"}, "type": {"type": "string"}}, "required": ["_from", "to", "type"]}},
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["restaurant.search", "restaurant.search", "flight.search"]
    assert [call["arguments"]["cuisine"] for call in plan["calls"][:2]] == ["Chinese", "Italian"]


def test_ordering_interleaves_repeated_tools_when_entities_are_shared_by_occurrence():
    calls = [
        {"id": "call_1", "tool_name": "lawsuit.fetch_details", "arguments": {"company_name": "Pacific Gas and Electric"}},
        {"id": "call_2", "tool_name": "lawsuit.fetch_details", "arguments": {"company_name": "Tesla Inc."}},
        {
            "id": "call_3",
            "tool_name": "lawsuit.judge",
            "arguments": {"company_name": "Pacific Gas and Electric", "lawsuit_id": 123},
        },
        {"id": "call_4", "tool_name": "lawsuit.judge", "arguments": {"company_name": "Tesla Inc.", "lawsuit_id": 123}},
    ]

    ordered = _order_independent_calls_for_benchmark(
        "Find how many cases and the judge handling a specific lawsuit for Pacific Gas and Electric and Tesla Inc.",
        [{"name": "lawsuit.fetch_details"}, {"name": "lawsuit.judge"}],
        calls,
    )

    assert [call["tool_name"] for call in ordered] == [
        "lawsuit.fetch_details",
        "lawsuit.judge",
        "lawsuit.fetch_details",
        "lawsuit.judge",
    ]
    assert [call["arguments"]["company_name"] for call in ordered] == [
        "Pacific Gas and Electric",
        "Pacific Gas and Electric",
        "Tesla Inc.",
        "Tesla Inc.",
    ]


def test_ordering_keeps_repeated_tools_grouped_when_entities_are_not_shared():
    calls = [
        {"id": "call_1", "tool_name": "animal_population.get_history", "arguments": {"country": "Bangladesh", "species": "tigers"}},
        {"id": "call_2", "tool_name": "animal_population.get_history", "arguments": {"country": "India", "species": "tigers"}},
        {"id": "call_3", "tool_name": "animal_population.get_projection", "arguments": {"country": "Nepal", "species": "tigers"}},
        {"id": "call_4", "tool_name": "animal_population.get_projection", "arguments": {"country": "Malaysia", "species": "tigers"}},
    ]

    ordered = _order_independent_calls_for_benchmark(
        "Give me tiger population history for Bangladesh and India, and projections for Nepal and Malaysia.",
        [{"name": "animal_population.get_history"}, {"name": "animal_population.get_projection"}],
        calls,
    )

    assert ordered == calls


def test_ordering_does_not_treat_profit_as_model_fit_dependency():
    calls = [
        {"id": "call_1", "tool_name": "financial_ratio.net_profit_margin", "arguments": {}},
        {"id": "call_2", "tool_name": "financial_ratio.debt_ratio", "arguments": {}},
    ]

    ordered = _order_independent_calls_for_benchmark(
        "Calculate net profit margin and debt ratio.",
        [{"name": "financial_ratio.net_profit_margin"}, {"name": "financial_ratio.debt_ratio"}],
        calls,
    )

    assert [call["tool_name"] for call in ordered] == [
        "financial_ratio.net_profit_margin",
        "financial_ratio.debt_ratio",
    ]


def test_tool_binding_allows_hydration_with_steps_request():
    plan = build_tool_binding_plan(
        "How many steps do I need to walk in order to lose 500 calories and how much water do I need to intake today if I exercise for 2 hours?",
        [
            {"name": "steps_calorie_calculation", "description": "Calculate how many steps you need to walk to burn a specified amount of calories.", "parameters": {"type": "dict", "properties": {"calorie": {"type": "float"}}, "required": ["calorie"]}},
            {"name": "hydration_calculator", "description": "Calculate the amount of water to drink in a day given the hours of exercise.", "parameters": {"type": "dict", "properties": {"exercise_time": {"type": "float", "description": "The number of hours of exercise."}}, "required": ["exercise_time"]}},
            {"name": "payment_calculation", "description": "Calculate total payment from item prices and quantities.", "parameters": {"type": "dict", "properties": {"items": {"type": "array"}, "quantities": {"type": "array"}}, "required": ["items", "quantities"]}},
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["steps_calorie_calculation", "hydration_calculator"]
    assert plan["calls"][1]["arguments"]["exercise_time"] == 2.0


def test_tool_binding_query_evidence_rescues_bad_capability_missing_slot():
    plan = build_tool_binding_plan(
        "Generate a random forest model with 100 trees and a depth of 5 on the provided data my_data.",
        [
            {
                "name": "random_forest.train",
                "description": "Train a random forest model on provided data.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "data": {"type": "any", "description": "training data"},
                        "n_estimators": {"type": "integer", "description": "number of trees"},
                        "max_depth": {"type": "integer", "description": "maximum tree depth"},
                    },
                    "required": ["data", "n_estimators", "max_depth"],
                },
            }
        ],
        capability_plan={
            "intent_input_audit": {
                "final_user_want": "train random forest model",
                "inputs": [],
                "missing_inputs": ["data"],
            }
        },
    )

    assert plan["tool_decision"] == "call"
    assert plan["missing_inputs"] == []
    assert plan["calls"][0]["tool_name"] == "random_forest.train"
    assert plan["calls"][0]["arguments"]["data"] == "my_data"


def test_tool_binding_uses_exact_single_schema_for_science_lookup():
    plan = build_tool_binding_plan(
        "What is the genotype frequency of AA genotype in a population, given that allele frequency of A is 0.3?",
        [
            {
                "name": "calculate_genotype_frequency",
                "description": "Calculate the frequency of homozygous dominant genotype based on the allele frequency.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "allele_frequency": {"type": "float", "description": "Dominant allele frequency."},
                        "genotype": {
                            "type": "string",
                            "description": "The genotype whose frequency is needed.",
                            "enum": ["AA", "Aa", "aa"],
                        },
                    },
                    "required": ["allele_frequency", "genotype"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_genotype_frequency"]


def test_tool_binding_keeps_single_calculation_as_one_call():
    plan = build_tool_binding_plan(
        "Calculate the heat capacity at constant pressure for air, given its temperature is 298K and volume is 10 m^3.",
        [
            {
                "name": "calc_heat_capacity",
                "description": "Calculate the heat capacity at constant pressure of air using its temperature and volume.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "temp": {"type": "integer", "description": "The temperature of the gas in Kelvin."},
                        "volume": {"type": "integer", "description": "The volume of the gas in m^3."},
                    },
                    "required": ["temp", "volume"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calc_heat_capacity"]


def test_tool_binding_route_between_two_cities_is_one_call():
    plan = build_tool_binding_plan(
        "Find the shortest distance between two cities, New York and Los Angeles, through the train and you can transfer.",
        [
            {
                "name": "city_distance.find_shortest",
                "description": "Calculates the shortest distance between two cities via available public transportation.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "start_city": {"type": "string", "description": "The city you are starting from."},
                        "end_city": {"type": "string", "description": "The city you are heading to."},
                    },
                    "required": ["start_city", "end_city"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["city_distance.find_shortest"]


def test_tool_binding_semantic_slot_binds_sentiment_text_and_language():
    plan = build_tool_binding_plan(
        "Analyze the sentiment of a customer review 'I love the food here! It's always fresh and delicious.'.",
        [
            {
                "name": "sentiment_analysis",
                "description": "Analyze sentiment of text in a given language.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze."},
                        "language": {"type": "string", "description": "Language of the input text."},
                    },
                    "required": ["text", "language"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["text"].startswith("I love the food")
    assert plan["calls"][0]["arguments"]["language"] == "en"


def test_tool_binding_semantic_slot_binds_file_path_as_data_source():
    plan = build_tool_binding_plan(
        "Analyze my fMRI data in ~/data/myfMRI.nii from a multi-band sequence, that is smoothed at 6mm with an isotropic voxel size of 2mm.",
        [
            {
                "name": "fMRI.analyze",
                "description": "Analyze fMRI data from an input data source.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "data_source": {"type": "string", "description": "The source fMRI dataset."},
                        "sequence": {"type": "string", "description": "Acquisition sequence."},
                    },
                    "required": ["data_source"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["data_source"] == "~/data/myfMRI.nii"


def test_tool_binding_semantic_slot_binds_zodiac_sign_pair():
    plan = build_tool_binding_plan(
        "Find the compatibility score in percentage of Aries with Gemini.",
        [
            {
                "name": "get_zodiac_compatibility",
                "description": "Get compatibility score for two zodiac signs.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "sign1": {"type": "string"},
                        "sign2": {"type": "string"},
                    },
                    "required": ["sign1", "sign2"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["sign1"] == "Aries"
    assert plan["calls"][0]["arguments"]["sign2"] == "Gemini"


def test_tool_binding_semantic_slot_binds_place_pair():
    plan = build_tool_binding_plan(
        "What's the time difference between San Francisco and Sydney?",
        [
            {
                "name": "get_time_difference",
                "description": "Get the time difference between two places.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "place1": {"type": "string"},
                        "place2": {"type": "string"},
                    },
                    "required": ["place1", "place2"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["place1"] == "San Francisco"
    assert plan["calls"][0]["arguments"]["place2"] == "Sydney"


def test_tool_binding_semantic_slot_binds_art_edit_values():
    plan = build_tool_binding_plan(
        "Change my painting's medium to oil and change size to 12x18 with red dominant color.",
        [
            {
                "name": "modify_painting",
                "description": "Modify a painting medium, size, and dominant color.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "medium": {"type": "string"},
                        "size": {"type": "string"},
                        "dominant_color": {"type": "string"},
                    },
                    "required": ["medium", "size"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["medium"] == "oil"
    assert plan["calls"][0]["arguments"]["size"] == "12x18"
    assert plan["calls"][0]["arguments"]["dominant_color"] == "red"


def test_tool_binding_top_n_result_count_is_not_call_count():
    plan = build_tool_binding_plan(
        "Get me the top 10 landmark cases in constitutional law in China.",
        [
            {
                "name": "get_top_cases",
                "description": "Get top legal cases for a field of law in a country.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "field_of_law": {"type": "string"},
                        "country": {"type": "string"},
                        "limit": {"type": "integer", "description": "Number of results to return."},
                    },
                    "required": ["field_of_law", "country"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_top_cases"]
    assert plan["calls"][0]["arguments"]["limit"] == 10


def test_tool_binding_uses_grounded_semantic_frame_for_currency_slots():
    plan = build_tool_binding_plan(
        "Get the exchange rate from British pounds to Japanese yen with the fee 0.02 included.",
        [
            {
                "name": "get_exchange_rate_with_fee",
                "description": "Get an exchange rate including a fee.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base_currency": {"type": "string"},
                        "target_currency": {"type": "string"},
                        "fee": {"type": "float"},
                    },
                    "required": ["base_currency", "target_currency"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "get GBP to JPY exchange rate with fee",
                "slots_observed": [
                    {
                        "role": "base_currency",
                        "value": "GBP",
                        "value_type": "currency",
                        "evidence_span": "British pounds",
                        "status": "semantic",
                        "confidence": 0.95,
                    },
                    {
                        "role": "target_currency",
                        "value": "JPY",
                        "value_type": "currency",
                        "evidence_span": "Japanese yen",
                        "status": "semantic",
                        "confidence": 0.95,
                    },
                ],
                "call_groups": [{"intent": "get exchange rate", "unit_of_work": "one currency pair", "expected_call_count": 1}],
            }
        },
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["base_currency"] == "GBP"
    assert plan["calls"][0]["arguments"]["target_currency"] == "JPY"


def test_tool_binding_keeps_model_grounded_compact_codes_for_non_currency_schema_slots():
    plan = build_tool_binding_plan(
        "Find a direct flight from New York to Seattle on May 20th.",
        [
            {
                "name": "search_direct_flight",
                "description": "Search direct flights using origin and destination airport codes.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure airport code."},
                        "destination": {"type": "string", "description": "Arrival airport code."},
                        "date": {"type": "string", "description": "Travel date."},
                    },
                    "required": ["origin", "destination", "date"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "canonical_request": "Search direct flights.",
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "search_direct_flight",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}}
                        ],
                        "evidence_spans": {
                            "origin": "New York",
                            "destination": "Seattle",
                            "date": "May 20th",
                        },
                    }
                ],
            }
        },
        allow_model_binding_prefix=True,
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"] == {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}


def test_tool_binding_preserves_evidence_verified_date_over_unrelated_conversation_date():
    plan = build_tool_binding_plan(
        "My date of birth is 1990-04-05. Search a direct flight from JFK to SEA on May 20th, 2024.",
        [
            {
                "name": "search_direct_flight",
                "description": "Search direct flights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string", "description": "Flight departure date."},
                    },
                    "required": ["origin", "destination", "date"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "search_direct_flight",
                        "call_count": 1,
                        "arguments": {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"},
                        "evidence_spans": {"origin": "JFK", "destination": "SEA", "date": "May 20th, 2024"},
                    }
                ],
            }
        },
        allow_model_binding_prefix=True,
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["arguments"]["date"] == "2024-05-20"


def test_tool_binding_uses_semantic_canonical_intent_for_terse_stateful_confirmation():
    plan = build_tool_binding_plan(
        "Please complete this for user_123.",
        [
            {
                "name": "book_reservation",
                "description": "Book a reservation for a user.",
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
                "canonical_request": "Book reservation for user_123.",
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "book_reservation",
                        "call_count": 1,
                        "arguments": {"user_id": "user_123"},
                        "evidence_spans": {"user_id": "user_123"},
                    }
                ],
            }
        },
        allow_model_binding_prefix=True,
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["tool_name"] == "book_reservation"


def test_tool_binding_accepts_sentence_terminated_structured_identifier_evidence():
    plan = build_tool_binding_plan(
        "My user ID is mia_li_3668.",
        [
            {
                "name": "get_user_details",
                "description": "Retrieve details for a user.",
                "parameters": {
                    "type": "dict",
                    "properties": {"user_id": {"type": "string", "description": "The ID of the user."}},
                    "required": ["user_id"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "canonical_request": "Retrieve the user details.",
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "get_user_details",
                        "call_count": 1,
                        "argument_groups": [{"arguments": {"user_id": "mia_li_3668"}}],
                        "evidence_spans": {"user_id": "mia_li_3668"},
                    }
                ],
            }
        },
        allow_model_binding_prefix=True,
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"] == {"user_id": "mia_li_3668"}


def test_tool_binding_rejects_unsupported_currency_code_despite_evidence_span():
    plan = build_tool_binding_plan(
        "Convert British pounds to Japanese yen.",
        [
            {
                "name": "convert_currency",
                "description": "Convert one currency into another.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base_currency": {"type": "string"},
                        "target_currency": {"type": "string"},
                    },
                    "required": ["base_currency", "target_currency"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "call",
                "canonical_request": "Convert currency.",
                "call_groups": [{"expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "convert_currency",
                        "call_count": 1,
                        "argument_groups": [{"arguments": {"base_currency": "USD", "target_currency": "JPY"}}],
                        "evidence_spans": {"base_currency": "British pounds", "target_currency": "Japanese yen"},
                    }
                ],
            }
        },
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"] == {"base_currency": "GBP", "target_currency": "JPY"}
    assert plan["model_tool_binding"]["accepted"] is False


def test_tool_binding_semantic_frame_avoids_shared_domain_slot_confusion():
    plan = build_tool_binding_plan(
        "User: The appointment ID is 34567890 and the new date is March 26th with Dr. Lee.",
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Modify appointment 34567890 to March 26th with Dr. Lee.",
                "slots_observed": [
                    {
                        "role": "appointment_id",
                        "value": "34567890",
                        "value_type": "number",
                        "evidence_span": "34567890",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "new_date",
                        "value": "March 26th",
                        "value_type": "date",
                        "evidence_span": "March 26th",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "doctor_name",
                        "value": "Dr. Lee",
                        "value_type": "text",
                        "evidence_span": "Dr. Lee",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"intent": "modify appointment", "unit_of_work": "appointment modification", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ModifyRegistration"]
    assert plan["calls"][0]["arguments"] == {
        "appointment_id": "34567890",
        "new_appointment_date": "March 26th",
        "new_appointment_doctor": "Dr. Lee",
    }


def test_tool_binding_semantic_frame_does_not_count_each_update_slot_as_call():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Can you please modify my appointment scheduled for March 25th "
            "with Dr. Kim to March 26th with Dr. Lee?\n"
            "User: The appointment ID is 34567890 and the new date is March 26th with Dr. Lee."
        ),
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Modify appointment 34567890 to March 26th with Dr. Lee.",
                "slots_observed": [
                    {
                        "role": "appointment_id",
                        "value": "34567890",
                        "value_type": "identifier",
                        "evidence_span": "34567890",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "new_appointment_date",
                        "value": "March 26th",
                        "value_type": "date",
                        "evidence_span": "March 26th",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "new_appointment_doctor",
                        "value": "Dr. Lee",
                        "value_type": "person",
                        "evidence_span": "Dr. Lee",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [],
                "tool_bindings": [],
                "missing_inputs": [],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ModifyRegistration"]
    assert plan["calls"][0]["arguments"] == {
        "appointment_id": "34567890",
        "new_appointment_date": "March 26th",
        "new_appointment_doctor": "Dr. Lee",
    }


def test_tool_binding_semantic_slot_coverage_rescues_weak_lexical_route():
    plan = build_tool_binding_plan(
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.",
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Check appointments for user J46801 from March 5th to March 12th.",
                "slots_observed": [
                    {
                        "role": "user_id",
                        "value": "J46801",
                        "value_type": "text",
                        "evidence_span": "J46801",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "start_date",
                        "value": "2024-03-05",
                        "value_type": "date",
                        "evidence_span": "March 5th",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "end_date",
                        "value": "2024-03-12",
                        "value_type": "date",
                        "evidence_span": "March 12th",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"intent": "check schedule", "unit_of_work": "time-range lookup", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryHealthData"]
    assert plan["calls"][0]["arguments"] == {
        "user_id": "J46801",
        "start_time": "2024-03-05",
        "end_time": "2024-03-12",
    }


def test_tool_binding_accepts_key_value_semantic_slots():
    plan = build_tool_binding_plan(
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.",
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Check appointments for user J46801 from March 5th to March 12th.",
                "slots_observed": [
                    {
                        "user_id": "J46801",
                        "start_time": "2024-03-05",
                        "end_time": "2024-03-12",
                        "evidence_spans": {
                            "user_id": "J46801",
                            "start_time": "March 5th",
                            "end_time": "March 12th",
                        },
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "check schedule", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryHealthData"]
    assert plan["calls"][0]["arguments"] == {
        "user_id": "J46801",
        "start_time": "2024-03-05",
        "end_time": "2024-03-12",
    }


def test_tool_binding_semantic_range_slot_fills_start_and_end_boundaries():
    plan = build_tool_binding_plan(
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.",
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Check appointments for user J46801 from March 5th to March 12th.",
                "slots_observed": [
                    {
                        "user_id": "J46801",
                        "date_range": "March 5th to March 12th",
                        "evidence_spans": {
                            "user_id": "J46801",
                            "date_range": "March 5th to March 12th",
                        },
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "check schedule", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryHealthData"]
    assert plan["calls"][0]["arguments"] == {
        "user_id": "J46801",
        "start_time": "March 5th",
        "end_time": "March 12th",
    }


def test_tool_binding_accepts_binding_level_evidence_for_argument_groups():
    request = (
        '"What is the function of the molecule ATP in the mitochondria and does it have a specific '
        'function within this organelle? Also, can you tell me the function of the molecule DNA in '
        'the nucleus and whether it has a specific function within the nucleus?"'
    )
    tool = {
        "name": "cell_biology.function_lookup",
        "description": "Looks up biological function.",
        "parameters": {
            "type": "dict",
            "properties": {
                "molecule": {"type": "string", "description": "The molecule of interest."},
                "organelle": {"type": "string", "description": "The organelle of interest."},
                "specific_function": {
                    "type": "boolean",
                    "description": "Whether to look up a specific function within the organelle.",
                },
            },
            "required": ["molecule", "organelle", "specific_function"],
        },
    }

    plan = build_tool_binding_plan(
        request,
        [tool],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Find ATP in mitochondria and DNA in nucleus.",
                "slots_observed": [],
                "call_groups": [
                    {"operation": "ATP in mitochondria", "expected_call_count": 1},
                    {"operation": "DNA in nucleus", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "cell_biology.function_lookup",
                        "call_count": 2,
                        "argument_groups": [
                            {"molecule": "ATP", "organelle": "mitochondria", "specific_function": True},
                            {"molecule": "DNA", "organelle": "nucleus", "specific_function": True},
                        ],
                        "evidence_spans": {
                            "molecule": ["ATP", "DNA"],
                            "organelle": ["mitochondria", "nucleus"],
                            "specific_function": [
                                "specific function within this organelle",
                                "specific function within the nucleus",
                            ],
                        },
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["arguments"] for call in plan["calls"]] == [
        {"molecule": "ATP", "organelle": "mitochondria", "specific_function": True},
        {"molecule": "DNA", "organelle": "nucleus", "specific_function": True},
    ]


def test_tool_binding_reuses_single_shared_evidence_for_repeated_model_argument_groups():
    request = "What's cost of 2 and 4 gb ram machine on aws ec2 with one CPU?"
    tool = {
        "name": "get_aws_pricing",
        "description": "Get AWS EC2 pricing.",
        "parameters": {
            "type": "dict",
            "properties": {
                "memory": {"type": "integer", "description": "Memory in GB."},
                "cpu": {"type": "string", "description": "CPU count.", "enum": ["single", "dual", "quad"]},
            },
            "required": ["memory", "cpu"],
        },
    }

    plan = build_tool_binding_plan(
        request,
        [tool],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get pricing for two instance sizes with one CPU.",
                "call_groups": [{"tool_name": "get_aws_pricing", "expected_call_count": 2}],
                "tool_bindings": [
                    {
                        "tool_name": "get_aws_pricing",
                        "call_count": 2,
                        "argument_groups": [
                            {"memory": 2, "cpu": "single"},
                            {"memory": 4, "cpu": "single"},
                        ],
                        "evidence_spans": {"memory": ["2", "4"], "cpu": ["one CPU"]},
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["arguments"] for call in plan["calls"]] == [
        {"memory": 2, "cpu": "single"},
        {"memory": 4, "cpu": "single"},
    ]


def test_tool_binding_rejects_model_binding_that_undercounts_grounded_cross_product():
    request = "What are the boiling and melting points of water and iron at sea levels of 0 meters and 1000 meters respectively?"
    tool = {
        "name": "get_boiling_melting_points",
        "description": "Get boiling and melting points for a substance at a sea level.",
        "parameters": {
            "type": "dict",
            "properties": {
                "substance": {"type": "string", "description": "The name of the substance."},
                "sea_level": {"type": "integer", "description": "The sea level in meters."},
            },
            "required": ["substance", "sea_level"],
        },
    }

    plan = build_tool_binding_plan(
        request,
        [tool],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve boiling and melting points for water and iron.",
                "slots_observed": [],
                "call_groups": [{"operation": "get points", "expected_call_count": 2}],
                "tool_bindings": [
                    {
                        "tool_name": "get_boiling_melting_points",
                        "call_count": 2,
                        "argument_groups": [
                            {"substance": "water", "sea_level": 0},
                            {"substance": "iron", "sea_level": 1000},
                        ],
                        "evidence_spans": [
                            {"substance": "water", "sea_level": "0 meters"},
                            {"substance": "iron", "sea_level": "1000 meters"},
                        ],
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert [call["arguments"] for call in plan["calls"]] == [
        {"substance": "water", "sea_level": 0},
        {"substance": "water", "sea_level": 1000},
        {"substance": "iron", "sea_level": 0},
        {"substance": "iron", "sea_level": 1000},
    ]


def test_tool_binding_allows_grounded_model_binding_prefix_for_stateful_execution():
    request = (
        "What are the boiling and melting points of water and iron at sea levels of "
        "0 meters and 1000 meters respectively?"
    )
    tool = {
        "name": "get_boiling_melting_points",
        "description": "Get boiling and melting points for a substance at a sea level.",
        "parameters": {
            "type": "dict",
            "properties": {
                "substance": {"type": "string", "description": "The name of the substance."},
                "sea_level": {"type": "integer", "description": "The sea level in meters."},
            },
            "required": ["substance", "sea_level"],
        },
    }

    plan = build_tool_binding_plan(
        request,
        [tool],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve a first boiling and melting point result.",
                "call_groups": [{"operation": "get points", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "get_boiling_melting_points",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"substance": "water", "sea_level": 0},
                                "evidence_spans": {"substance": "water", "sea_level": "0 meters"},
                            }
                        ],
                    }
                ],
                "missing_inputs": [],
            }
        },
        allow_model_binding_prefix=True,
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["model_tool_binding"]["prefix_accepted"] is True
    assert [call["arguments"] for call in plan["calls"]] == [{"substance": "water", "sea_level": 0}]


def test_tool_binding_crosses_quoted_primary_values_with_requested_enum_values():
    request = (
        'Can you analyze the DNA sequence "AGCTTAGCTA" and "AGCTTAGGCTA" using the reference '
        'sequence "AGCTTAGCTA" to identify any potential \'insertion\' mutations, then repeat '
        "the same analysis for 'deletion' and 'substitution' mutations?"
    )
    tool = {
        "name": "analyze_dna_sequence",
        "description": "Analyze one sequence against a reference for a mutation type.",
        "parameters": {
            "type": "dict",
            "properties": {
                "sequence": {"type": "string", "description": "The DNA sequence to be analyzed."},
                "reference_sequence": {"type": "string", "description": "The reference DNA sequence."},
                "mutation_type": {
                    "type": "string",
                    "enum": ["insertion", "deletion", "substitution"],
                    "description": "Type of the mutation to be looked for in the sequence.",
                    "default": "insertion",
                },
            },
            "required": ["sequence", "reference_sequence"],
        },
    }

    plan = build_tool_binding_plan(request, [tool])

    assert [call["tool_name"] for call in plan["calls"]] == ["analyze_dna_sequence"] * 6
    assert [call["arguments"]["sequence"] for call in plan["calls"]] == [
        "AGCTTAGCTA",
        "AGCTTAGCTA",
        "AGCTTAGCTA",
        "AGCTTAGGCTA",
        "AGCTTAGGCTA",
        "AGCTTAGGCTA",
    ]
    assert [call["arguments"]["mutation_type"] for call in plan["calls"]] == [
        "insertion",
        "deletion",
        "substitution",
        "insertion",
        "deletion",
        "substitution",
    ]


def test_tool_binding_generic_check_date_range_routes_to_query_tool():
    plan = build_tool_binding_plan(
        "User: My user ID is J46801 and I want to check from March 5th to March 12th.",
        _registration_health_tools(),
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryHealthData"]
    assert plan["calls"][0]["arguments"] == {
        "user_id": "J46801",
        "start_time": "March 5th",
        "end_time": "March 12th",
    }


def test_tool_binding_semantic_medical_condition_binds_symptom_slot():
    plan = build_tool_binding_plan(
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
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Help me find out about shortness of breath.",
                "slots_observed": [
                    {
                        "role": "medical_condition",
                        "value": "shortness of breath",
                        "value_type": "text",
                        "evidence_span": "shortness of breath",
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "search medical knowledge", "expected_call_count": 1}],
                "missing_inputs": [],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["EmergencyKnowledge"]
    assert plan["calls"][0]["arguments"] == {"symptom": "shortness of breath"}


def test_tool_binding_semantic_single_next_action_ignores_prior_assistant_each_phrase():
    request = (
        "Earlier user: I'm feeling really tired lately, can you give me some information on what might be causing it?\n"
        "Earlier user: Fatigue.\n"
        "Assistant: I have found some possible causes for your fatigue. It can be due to "
        "Chronic fatigue syndrome, Anemia or Depression. Would you like me to provide you more "
        "information on each of these?\n"
        "User: Yes please.\n"
        "Latest prior API result: [{'name': 'Chronic fatigue syndrome'}, {'name': 'Anemia'}, {'name': 'Depression'}]"
    )

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "EmergencyKnowledge",
                "description": "This API searches for a given symptom for emergency knowledge.",
                "input_parameters": {
                    "symptom": {"type": "str", "description": "The symptom to search."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Provide more information about the confirmed fatigue cause.",
                "call_groups": [{"intent": "search medical knowledge", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["EmergencyKnowledge"]


def test_tool_binding_semantic_calculation_expression_binds_formula_slot():
    plan = build_tool_binding_plan(
        "User: Can you calculate (5+6)*3 for me?",
        [
            {
                "name": "Calculator",
                "description": "This API calculates the value of a given formula.",
                "input_parameters": {
                    "formula": {"type": "str", "description": "The formula to calculate."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate (5+6)*3",
                "slots_observed": [
                    {
                        "role": "calculation_expression",
                        "value": "(5+6)*3",
                        "value_type": "text",
                        "evidence_span": "calculate (5+6)*3",
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "calculate", "unit_of_work": "calculation", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["Calculator"]
    assert plan["calls"][0]["arguments"] == {"formula": "(5+6)*3"}


def test_tool_binding_binds_verbal_square_to_formula_slot():
    plan = build_tool_binding_plan(
        "User: Can you help me calculate the square of 8?",
        [
            {
                "name": "Calculator",
                "description": "This API provides basic arithmetic operations.",
                "input_parameters": {
                    "formula": {"type": "str", "description": "The formula that needs to be calculated."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["Calculator"]
    assert plan["calls"][0]["arguments"] == {"formula": "8*8"}


def test_tool_binding_binds_bare_arithmetic_expression_to_formula_slot():
    plan = build_tool_binding_plan(
        "User: Can you help me calculate 25*3+7/3?",
        [
            {
                "name": "Calculator",
                "description": "This API provides basic arithmetic operations.",
                "input_parameters": {
                    "formula": {"type": "str", "description": "The formula that needs to be calculated."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["Calculator"]
    assert plan["calls"][0]["arguments"] == {"formula": "25*3+7/3"}


def test_tool_binding_semantic_health_record_array_excludes_metadata_numbers():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Can you record my blood pressure and heart rate?\n"
            "Earlier user: My user ID is 12345.\n"
            "User: Sure, the time is 2023-10-13 09:30:00, "
            "my blood pressure is 120/80, and my heart rate is 80."
        ),
        [
            {
                "name": "RecordHealthData",
                "description": "This API records the health data of a user.",
                "input_parameters": {
                    "user_id": {"type": "str", "description": "The ID of user."},
                    "time": {"type": "str", "description": "The time of health data. Format: %Y-%m-%d %H:%M:%S"},
                    "health_data": {
                        "type": "list",
                        "description": "The health data, with the format like [{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]",
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Record blood pressure 120/80 and heart rate 80 for user 12345 at 2023-10-13 09:30:00.",
                "slots_observed": [
                    {
                        "role": "user_id",
                        "value": "12345",
                        "value_type": "number",
                        "evidence_span": "My user ID is 12345.",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "timestamp",
                        "value": "2023-10-13 09:30:00",
                        "value_type": "date",
                        "evidence_span": "the time is 2023-10-13 09:30:00",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "blood_pressure",
                        "value": "120/80",
                        "value_type": "text",
                        "evidence_span": "my blood pressure is 120/80",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "heart_rate",
                        "value": "80",
                        "value_type": "number",
                        "evidence_span": "my heart rate is 80",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"intent": "record_vitals", "unit_of_work": "record vitals", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["RecordHealthData"]
    assert plan["calls"][0]["arguments"] == {
        "user_id": "12345",
        "time": "2023-10-13 09:30:00",
        "health_data": ["120/80", "80"],
    }


def test_tool_binding_record_health_data_uses_labeled_values_before_raw_numbers():
    plan = build_tool_binding_plan(
        (
            "Earlier user: No, thank you. Can you also record my health data?\n"
            "User: My user ID is 12345, the time is 2023-08-06 09:00:00 "
            "and my health data is blood_pressure 120/80 and heart_rate 80."
        ),
        [
            {
                "name": "RecordHealthData",
                "description": "This API records the health data of a user.",
                "input_parameters": {
                    "user_id": {"type": "str", "description": "The ID of user."},
                    "time": {"type": "str", "description": "The time of health data."},
                    "health_data": {
                        "type": "list",
                        "description": "The health data, with the format like [{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]",
                    },
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["RecordHealthData"]
    assert plan["calls"][0]["arguments"]["health_data"] == ["120/80", "80"]


def test_tool_binding_semantic_patient_name_does_not_fill_doctor_name():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Can you help me book an appointment with Dr. John on 2023-10-15?\n"
            "User: My name is Emily Smith."
        ),
        [
            {
                "name": "AppointmentRegistration",
                "description": "This API registers an appointment of hospital.",
                "input_parameters": {
                    "patient_name": {"type": "str", "description": "The name of patient."},
                    "date": {"type": "str", "description": "The date of appointment. Format be like %Y-%m-%d"},
                    "doctor_name": {"type": "str", "description": "The name of appointed doctor."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Book an appointment for Emily Smith with Dr. John on 2023-10-15.",
                "slots_observed": [
                    {
                        "role": "patient_name",
                        "value": "Emily Smith",
                        "value_type": "person",
                        "evidence_span": "My name is Emily Smith",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "doctor_name",
                        "value": "Dr. John",
                        "value_type": "person",
                        "evidence_span": "Dr. John",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "date",
                        "value": "2023-10-15",
                        "value_type": "date",
                        "evidence_span": "2023-10-15",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"intent": "book_appointment", "unit_of_work": "appointment", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["AppointmentRegistration"]
    assert plan["calls"][0]["arguments"] == {
        "patient_name": "Emily Smith",
        "date": "2023-10-15",
        "doctor_name": "Dr. John",
    }


def test_tool_binding_short_title_binds_single_name_slot():
    plan = build_tool_binding_plan(
        "User: Morning Routine",
        [
            {
                "name": "QueryScene",
                "description": "This API queries a scene by name.",
                "input_parameters": {
                    "name": {"type": "str", "description": "The name of the scene."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryScene"]
    assert plan["calls"][0]["arguments"] == {"name": "Morning Routine"}


def test_tool_binding_short_title_survives_bad_semantic_paraphrase():
    plan = build_tool_binding_plan(
        "User: Morning Routine",
        [
            {
                "name": "QueryScene",
                "description": "This API queries a scene of smart home system, given the scene name.",
                "input_parameters": {
                    "name": {"type": "str", "description": "The name of the scene."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "User requests a morning routine.",
                "slots_observed": [
                    {
                        "role": "routine_name",
                        "value": "Morning Routine",
                        "value_type": "text",
                        "evidence_span": "Morning Routine",
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "plan routine", "unit_of_work": "morning routine plan", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryScene"]
    assert plan["calls"][0]["arguments"] == {"name": "Morning Routine"}


def test_tool_binding_short_symptom_value_binds_symptom_slot():
    plan = build_tool_binding_plan(
        "User: Fatigue.",
        [
            {
                "name": "SymptomSearch",
                "description": "Search diseases related to a symptom.",
                "input_parameters": {
                    "symptom": {"type": "str", "description": "The symptom to search for."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["SymptomSearch"]
    assert plan["calls"][0]["arguments"] == {"symptom": "Fatigue"}


def test_tool_binding_email_label_can_bind_username_slot():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Hi, can you help me delete my account?\n"
            "User: My email is foo and my password is bar.\n"
            "AI: Thanks, I will authenticate you."
        ),
        [
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]
    assert plan["calls"][0]["arguments"] == {"username": "foo", "password": "bar"}


def test_tool_binding_auth_token_prefers_labeled_username_over_email_and_calls_once():
    request = (
        "Earlier user: Hi, can you help me check my alarm for March 20th, 2023 at 6:30AM?\n"
        "User: My email is janesmith@example.com, username is JaneSmith, and password is password."
    )

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]
    assert plan["calls"][0]["arguments"] == {"username": "JaneSmith", "password": "password"}


def test_tool_binding_collapses_model_auth_token_duplicates_for_single_account():
    request = (
        "Earlier user: Hi, can you help me check my alarm for March 20th, 2023 at 6:30AM?\n"
        "User: My email is janesmith@example.com, username is JaneSmith, and password is password."
    )

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Authenticate the user for the protected alarm query.",
                "call_groups": [{"intent": "authenticate user", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "GetUserToken",
                        "intent": "authenticate user",
                        "call_count": 2,
                        "argument_groups": [
                            {
                                "arguments": {"username": "JaneSmith", "password": "password"},
                                "evidence_spans": {"username": "username is JaneSmith", "password": "password"},
                            },
                            {
                                "arguments": {"username": "janesmith@example.com", "password": "password"},
                                "evidence_spans": {"username": "janesmith@example.com", "password": "password"},
                            },
                        ],
                    }
                ],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]
    assert plan["calls"][0]["arguments"] == {"username": "JaneSmith", "password": "password"}


def test_tool_binding_prior_api_token_binds_token_slot():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Delete my account.\n"
            "Latest prior API result: [{'token': 'z9x8c7v6b5n4m3q2w1'}].\n"
            "User: Yes, delete it."
        ),
        [
            {
                "name": "DeleteAccount",
                "description": "This API deletes a user account.",
                "input_parameters": {
                    "token": {"type": "str", "description": "The access token of the user."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["DeleteAccount"]
    assert plan["calls"][0]["arguments"] == {"token": "z9x8c7v6b5n4m3q2w1"}


def test_tool_binding_token_slot_does_not_use_unrelated_quoted_text():
    plan = build_tool_binding_plan(
        'User: Add an agenda item called "Meeting with clients" for tomorrow.',
        [
            {
                "name": "AddAgenda",
                "description": "This API adds an agenda item.",
                "input_parameters": {
                    "token": {"type": "str", "description": "The access token of the user."},
                    "content": {"type": "str", "description": "The agenda content."},
                },
            }
        ],
    )

    assert plan["tool_decision"] == "ask_user"
    assert plan["calls"] == []
    assert plan["missing_inputs"] == ["token"]


def test_tool_binding_password_slot_does_not_use_unrelated_quoted_text():
    plan = build_tool_binding_plan(
        'User: Can you add "Meeting with clients" on December 15, 2023 at 2:00 PM?',
        [
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            }
        ],
    )

    audit = plan["candidate_tool_audits"][0]
    assert plan["tool_decision"] == "ask_user"
    assert audit["slot_availability"]["password"] == "missing"
    assert audit["missing_slots"] == ["username", "password"]


def test_tool_binding_uses_auth_tool_when_requested_protected_tool_needs_token():
    plan = build_tool_binding_plan(
        (
            "Earlier user: My account identifier is user4.\n"
            "Earlier user: My password is user4pass.\n"
            "User: How much money do I have in my account?"
        ),
        [
            {
                "name": "OpenBankAccount",
                "description": "This API opens a bank account.",
                "input_parameters": {
                    "account": {"type": "str", "description": "The account identifier."},
                    "password": {"type": "str", "description": "The account password."},
                    "name": {"type": "str", "description": "The user name."},
                },
            },
            {
                "name": "QueryBalance",
                "description": "This API queries account balance.",
                "input_parameters": {
                    "token": {"type": "str", "description": "The access token."},
                },
            },
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username."},
                    "password": {"type": "str", "description": "The password."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]
    assert plan["calls"][0]["arguments"] == {"username": "user4", "password": "user4pass"}


def test_tool_binding_use_value_binds_name_slot():
    plan = build_tool_binding_plan(
        (
            "Earlier user: I want to open a bank account.\n"
            "Earlier user: My account identifier is user4.\n"
            "Earlier user: My password is user4pass.\n"
            "User: Use John."
        ),
        [
            {
                "name": "OpenBankAccount",
                "description": "This API opens a bank account.",
                "input_parameters": {
                    "account": {"type": "str", "description": "The account identifier."},
                    "password": {"type": "str", "description": "The account password."},
                    "name": {"type": "str", "description": "The user name."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["OpenBankAccount"]
    assert plan["calls"][0]["arguments"] == {"account": "user4", "password": "user4pass", "name": "John"}


def test_tool_binding_account_creation_credentials_do_not_force_auth_tool():
    plan = build_tool_binding_plan(
        (
            "Earlier user: I want to open a bank account.\n"
            "Earlier user: My account identifier is user4.\n"
            "Earlier user: My password is user4pass.\n"
            "User: Use John."
        ),
        [
            {
                "name": "OpenBankAccount",
                "description": "This is an API for opening a bank account for a user, given the account, password and name.",
                "input_parameters": {
                    "account": {"type": "str", "description": "The account for the user."},
                    "password": {"type": "str", "description": "The password."},
                    "name": {"type": "str", "description": "account holder name."},
                },
            },
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["OpenBankAccount"]


def test_tool_binding_register_user_credentials_do_not_force_auth_tool():
    plan = build_tool_binding_plan(
        (
            "Earlier user: I want to register for an account.\n"
            "User: My username is foo, my password is bar, and my email is foo@example.com.\n"
            "Ai: Thank you. Let me call the RegisterUser API to register your account."
        ),
        [
            {
                "name": "RegisterUser",
                "description": "The API for registering a account, given the username, password and email.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                    "email": {"type": "str", "description": "The email of the user."},
                },
            },
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["RegisterUser"]


def test_tool_binding_unlabeled_credential_pair_binds_token_tool():
    plan = build_tool_binding_plan(
        "Earlier user: Can you add a reminder for a meeting on 2022-05-06 at 2 PM?\nUser: user1 user1pass",
        [
            {
                "name": "GetUserToken",
                "description": "Get the user token by username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username of the user."},
                    "password": {"type": "str", "description": "The password of the user."},
                },
            },
            {
                "name": "AddReminder",
                "description": "This API adds a reminder.",
                "input_parameters": {
                    "token": {"type": "str", "description": "User token."},
                    "content": {"type": "str", "description": "Reminder content."},
                    "time": {"type": "str", "description": "Reminder time."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["GetUserToken"]
    assert plan["calls"][0]["arguments"] == {"username": "user1", "password": "user1pass"}


def test_tool_binding_token_present_prefers_protected_tool_over_stale_setup():
    plan = build_tool_binding_plan(
        (
            "Earlier user: My account identifier is user4.\n"
            "Earlier user: My password is user4pass.\n"
            "Earlier user: Use John.\n"
            "User: How much money do I have in my account?\n"
            "Latest prior API result: {'token': 'q9w8e7r6t5y4u3i2o1'}"
        ),
        [
            {
                "name": "OpenBankAccount",
                "description": "This API opens a bank account.",
                "input_parameters": {
                    "account": {"type": "str", "description": "The account identifier."},
                    "password": {"type": "str", "description": "The account password."},
                    "name": {"type": "str", "description": "The user name."},
                },
            },
            {
                "name": "QueryBalance",
                "description": "This API queries account balance.",
                "input_parameters": {
                    "token": {"type": "str", "description": "The access token."},
                },
            },
            {
                "name": "GetUserToken",
                "description": "This API gets a user token from username and password.",
                "input_parameters": {
                    "username": {"type": "str", "description": "The username."},
                    "password": {"type": "str", "description": "The password."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryBalance"]
    assert plan["calls"][0]["arguments"] == {"token": "q9w8e7r6t5y4u3i2o1"}


def test_tool_binding_forgot_password_first_phase_uses_conditional_fields():
    plan = build_tool_binding_plan(
        (
            "Earlier user: I forgot my password, can you help me reset it?\n"
            "User: My username is foo and my email is foo@example.com."
        ),
        [
            {
                "name": "ForgotPassword",
                "description": "Need call twice, first with 'Forgot Password' status to get the verification code, then call again with 'Verification Code' status to change the password.",
                "input_parameters": {
                    "status": {"type": "str", "description": "'Forgot Password' for first call, after get the verification code, call again with 'Verification Code' to change the password."},
                    "username": {"type": "str", "description": "The username of the user. Only needed for the first call."},
                    "email": {"type": "str", "description": "The email of the user. Only needed for the first call."},
                    "verification_code": {"type": "int", "description": "The verification code sent to the user. Only needed for the second call."},
                    "new_password": {"type": "str", "description": "The new password of the user. Only needed for the second call."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ForgotPassword"]
    assert plan["calls"][0]["arguments"] == {
        "status": "Forgot Password",
        "username": "foo",
        "email": "foo@example.com",
    }


def test_tool_binding_forgot_password_second_phase_uses_conditional_fields():
    plan = build_tool_binding_plan(
        (
            "Earlier user: The verification code is 970420.\n"
            "User: My new password is newpassword."
        ),
        [
            {
                "name": "ForgotPassword",
                "description": "Need call twice, first with 'Forgot Password' status to get the verification code, then call again with 'Verification Code' status to change the password.",
                "input_parameters": {
                    "status": {"type": "str", "description": "'Forgot Password' for first call, after get the verification code, call again with 'Verification Code' to change the password."},
                    "username": {"type": "str", "description": "The username of the user. Only needed for the first call."},
                    "email": {"type": "str", "description": "The email of the user. Only needed for the first call."},
                    "verification_code": {"type": "int", "description": "The verification code sent to the user. Only needed for the second call."},
                    "new_password": {"type": "str", "description": "The new password of the user. Only needed for the second call."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ForgotPassword"]
    assert plan["calls"][0]["arguments"] == {
        "status": "Verification Code",
        "verification_code": 970420,
        "new_password": "newpassword",
    }


def test_tool_binding_semantic_disease_name_can_satisfy_symptom_slot():
    plan = build_tool_binding_plan(
        "User: Yes, please tell me more about Chronic fatigue syndrome.",
        [
            {
                "name": "SymptomSearch",
                "description": "Search diseases related to a symptom.",
                "input_parameters": {
                    "symptom": {"type": "str", "description": "The symptom or disease name."},
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "User requests more information about Chronic fatigue syndrome",
                "slots_observed": [
                    {
                        "role": "disease_name",
                        "value": "Chronic fatigue syndrome",
                        "value_type": "text",
                        "evidence_span": "Chronic fatigue syndrome",
                        "status": "explicit",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [{"intent": "provide information", "unit_of_work": "disease info", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["SymptomSearch"]
    assert plan["calls"][0]["arguments"] == {"symptom": "Chronic fatigue syndrome"}


def test_tool_binding_routes_from_current_semantic_request_not_prior_api_result():
    plan = build_tool_binding_plan(
        (
            "User: The appointment ID is 90123456, the patient name is Olivia Davis, "
            "the date is October 10th, and the doctor name is Dr. Smith. "
            "Latest prior API result: [{'time': '2023-03-11 14:20:00', 'blood_pressure': [140, 90]}]. "
            "AI: Here are your health data for that period. User: Actually, cancel that appointment."
        ),
        _registration_health_tools(),
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Cancel appointment with ID 90123456.",
                "slots_observed": [
                    {
                        "role": "appointment_id",
                        "value": "90123456",
                        "value_type": "number",
                        "evidence_span": "90123456",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "appointment_date",
                        "value": "October 10th",
                        "value_type": "date",
                        "evidence_span": "October 10th",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "doctor_name",
                        "value": "Dr. Smith",
                        "value_type": "text",
                        "evidence_span": "Dr. Smith",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"intent": "cancel appointment", "unit_of_work": "one appointment", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["CancelRegistration"]
    assert plan["calls"][0]["arguments"] == {"appointment_id": "90123456"}


def test_tool_binding_modify_registration_confirmation_uses_target_doctor_and_id():
    plan = build_tool_binding_plan(
        (
            "Earlier user: Thank you for your help. Can you modify my appointment "
            "with Dr. Smith on October 10th to October 15th with Dr. Johnson?\n"
            "Earlier user: It's 90123456.\n"
            "User: Yes."
        ),
        [
            {
                "name": "EmergencyKnowledge",
                "description": "This API searches for a given symptom for emergency knowledge.",
                "input_parameters": {
                    "symptom": {"type": "str", "description": "The symptom to search."},
                },
            },
            {
                "name": "RecordHealthData",
                "description": "This API records the health data of a user.",
                "input_parameters": {
                    "user_id": {"type": "str", "description": "The ID of user."},
                    "time": {"type": "str", "description": "The time of health data."},
                    "health_data": {"type": "list", "description": "The health data."},
                },
            },
            {
                "name": "ModifyRegistration",
                "description": "This API modifies the registration of a patient given appointment ID.",
                "input_parameters": {
                    "appointment_id": {"type": "str", "description": "The ID of appointment."},
                    "new_appointment_date": {"type": "str", "description": "The new appointment date."},
                    "new_appointment_doctor": {"type": "str", "description": "The new appointment doctor."},
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["ModifyRegistration"]
    assert plan["calls"][0]["arguments"] == {
        "appointment_id": "90123456",
        "new_appointment_date": "October 15th",
        "new_appointment_doctor": "Dr. Johnson",
    }


def test_tool_binding_rejects_ungrounded_semantic_frame_fact_and_uses_raw_text():
    plan = build_tool_binding_plan(
        "Convert 150 Euros to Canadian dollars.",
        [
            {
                "name": "currency_conversion.convert",
                "description": "Convert from one currency to another.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["from_currency", "to_currency"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "convert currency",
                "slots_observed": [
                    {
                        "role": "from_currency",
                        "value": "USD",
                        "value_type": "currency",
                        "evidence_span": "US dollars",
                        "status": "semantic",
                        "confidence": 0.95,
                    }
                ],
                "call_groups": [{"intent": "convert currency", "unit_of_work": "one currency pair", "expected_call_count": 1}],
            }
        },
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["from_currency"] == "EUR"
    assert plan["calls"][0]["arguments"]["to_currency"] == "CAD"


def test_tool_binding_semantic_frame_caps_single_hotel_booking_count():
    plan = build_tool_binding_plan(
        "Book a luxury room in Hotel Paradise, Las Vegas, with a city view for 3 days starting from May 12, 2022.",
        [
            {
                "name": "book_hotel",
                "description": "Book a hotel room.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "hotel_name": {"type": "string"},
                        "location": {"type": "string"},
                        "room_type": {"type": "string"},
                        "start_date": {"type": "string"},
                        "stay_duration": {"type": "integer"},
                    },
                    "required": ["hotel_name", "location", "room_type", "start_date", "stay_duration"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "book one luxury room at Hotel Paradise in Las Vegas",
                "slots_observed": [
                    {"role": "hotel_name", "value": "Hotel Paradise", "value_type": "organization", "evidence_span": "Hotel Paradise", "status": "explicit", "confidence": 0.95},
                    {"role": "location", "value": "Las Vegas", "value_type": "location", "evidence_span": "Las Vegas", "status": "explicit", "confidence": 0.95},
                    {"role": "room_type", "value": "luxury room", "value_type": "category", "evidence_span": "luxury room", "status": "explicit", "confidence": 0.9},
                    {"role": "start_date", "value": "May 12, 2022", "value_type": "date", "evidence_span": "May 12, 2022", "status": "explicit", "confidence": 0.95},
                    {"role": "stay_duration", "value": 3, "value_type": "number", "evidence_span": "3 days", "status": "explicit", "confidence": 0.9},
                ],
                "call_groups": [{"intent": "book hotel room", "unit_of_work": "one hotel booking", "expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["book_hotel"]


def test_tool_binding_counts_booking_actions_not_guest_and_date_fragments():
    plan = build_tool_binding_plan(
        "Book a room for 2 adults and a child at the Sheraton Hotel in New York "
        "with check-in on May 1, 2022 and check-out on May 5, 2022. Also, "
        "Book a room for 1 adult and 2 children at the Marriott in Los Angeles "
        "with check-in on June 1, 2022 and check-out on June 10, 2022.",
        [
            {
                "name": "hotel_booking.book",
                "description": "Book a hotel room at the specified location for the specified number of adults and children.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "hotel_name": {"type": "string", "description": "The name of the hotel."},
                        "location": {"type": "string", "description": "The city where the hotel is located."},
                        "check_in": {"type": "string", "description": "The check-in date in the format yyyy-mm-dd."},
                        "check_out": {"type": "string", "description": "The check-out date in the format yyyy-mm-dd."},
                        "adults": {"type": "integer", "description": "The number of adults for the booking."},
                        "children": {"type": "integer", "description": "The number of children for the booking."},
                    },
                    "required": ["hotel_name", "location", "check_in", "check_out", "adults", "children"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["hotel_booking.book", "hotel_booking.book"]


def test_tool_binding_repeats_for_scoped_first_second_setups():
    plan = build_tool_binding_plan(
        "A circular loop is used in both setups. "
        "The first setup has radius 0.5 meters and current 10 amperes. "
        "The second setup has radius 1 meter and current 15 amperes. "
        "Calculate the magnetic field for both setups.",
        [
            {
                "name": "calculate_magnetic_field",
                "description": "Calculate magnetic field for a circular loop.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "current": {"type": "integer", "description": "The current through the loop."},
                        "radius": {"type": "float", "description": "The radius of the loop."},
                    },
                    "required": ["current", "radius"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_magnetic_field",
        "calculate_magnetic_field",
    ]


def test_tool_binding_repeats_for_scoped_conditions_count():
    plan = build_tool_binding_plan(
        "I have a container with volume 2.5 m^3 at 300 Kelvin. "
        "I will repeat the experiment at a higher temperature of 350 Kelvin "
        "and then at a lower volume of 1.5 m^3. "
        "Calculate heat capacity for these three different conditions.",
        [
            {
                "name": "calc_heat_capacity",
                "description": "Calculate heat capacity at constant pressure.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "temp": {"type": "integer", "description": "The temperature of the gas in Kelvin."},
                        "volume": {"type": "float", "description": "The volume of the gas in m^3."},
                    },
                    "required": ["temp", "volume"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calc_heat_capacity",
        "calc_heat_capacity",
        "calc_heat_capacity",
    ]


def test_tool_binding_two_sample_ttest_keeps_labeled_arrays_separate():
    plan = build_tool_binding_plan(
        "Run a two sample T-test to compare the average of Group A [3, 4, 5, 6, 4] and Group B [7, 8, 9, 8, 7] assuming equal variance.",
        [
            {
                "name": "run_two_sample_ttest",
                "description": "Run a two-sample t-test comparing two groups.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "group1": {"type": "array", "items": {"type": "float"}},
                        "group2": {"type": "array", "items": {"type": "float"}},
                        "equal_variance": {"type": "boolean"},
                    },
                    "required": ["group1", "group2"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "call"
    assert plan["calls"][0]["arguments"]["group1"] == [3, 4, 5, 6, 4]
    assert plan["calls"][0]["arguments"]["group2"] == [7, 8, 9, 8, 7]
    assert plan["calls"][0]["arguments"]["equal_variance"] is True


def test_tool_binding_rejects_food_order_as_ride_hailing():
    plan = build_tool_binding_plan(
        "I want to order five burgers and six chicken wings from Uber Eats.",
        [
            {
                "name": "uber.ride",
                "description": "Book an Uber ride from a pickup to a dropoff location.",
                "parameters": {
                    "type": "dict",
                    "properties": {"pickup": {"type": "string"}, "dropoff": {"type": "string"}},
                    "required": ["pickup", "dropoff"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_food_change_is_one_workflow_call():
    plan = build_tool_binding_plan(
        "I would like to switch my order from pizza to a burger.",
        [
            {
                "name": "change_food",
                "description": "Change an existing food order from one dish to another.",
                "parameters": {
                    "type": "dict",
                    "properties": {"old_item": {"type": "string"}, "new_item": {"type": "string"}},
                    "required": ["old_item", "new_item"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["change_food"]


def test_tool_binding_logs_each_food_item_with_default_piece_portions():
    plan = build_tool_binding_plan(
        "For breakfast I had a 12 ounce iced coffee and a banana.\n\n"
        "For lunch I had a quesadilla\n\n"
        "Breakfast four ounces of asparagus, two eggs, one piece of gluten free bread.",
        [
            {
                "name": "log_food",
                "description": "Logs a food item with details about the portion size and the meal it is associated with.",
                "parameters": {
                    "type": "dict",
                    "required": ["food_name", "portion_amount", "meal_name"],
                    "properties": {
                        "food_name": {"type": "string", "description": "The name of the food to log."},
                        "portion_amount": {
                            "type": "float",
                            "description": "The amount of the food item that was consumed, in specified units.",
                        },
                        "portion_unit": {
                            "type": "string",
                            "description": "The unit of measure for the portion amount.",
                            "enum": ["grams", "ounces", "pieces", "cups", "tablespoons"],
                            "default": "grams",
                        },
                        "meal_name": {
                            "type": "string",
                            "description": "The name of the meal with which the food item is associated.",
                        },
                    },
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["log_food"] * 6
    assert [call["arguments"]["food_name"] for call in plan["calls"]] == [
        "iced coffee",
        "banana",
        "quesadilla",
        "asparagus",
        "eggs",
        "gluten free bread",
    ]
    assert [call["arguments"]["portion_amount"] for call in plan["calls"]] == [12.0, 1.0, 1.0, 4.0, 2.0, 1.0]
    assert [call["arguments"]["portion_unit"] for call in plan["calls"]] == [
        "ounces",
        "pieces",
        "pieces",
        "ounces",
        "pieces",
        "pieces",
    ]
    assert [call["arguments"]["meal_name"] for call in plan["calls"]] == [
        "breakfast",
        "breakfast",
        "lunch",
        "breakfast",
        "breakfast",
        "breakfast",
    ]


def test_tool_binding_logs_unlabeled_food_and_drinks_with_schema_units():
    plan = build_tool_binding_plan(
        "I had 8 pieces of frozen mango and a chai tea.\n\n"
        "Earlier I had two slices of pepperoni pizza and a coffee",
        [
            {
                "name": "log_food",
                "description": "Logs a food item with a given portion size to track dietary intake.",
                "parameters": {
                    "type": "dict",
                    "required": ["food_name", "portion_amount", "portion_unit"],
                    "properties": {
                        "food_name": {"type": "string", "description": "The name of the food to log."},
                        "portion_amount": {"type": "float", "description": "The amount consumed."},
                        "portion_unit": {
                            "type": "string",
                            "description": "The unit of measure for the portion amount.",
                            "enum": ["cup", "grams", "slice", "piece", "tablespoon"],
                        },
                        "meal_type": {
                            "type": "string",
                            "enum": ["breakfast", "lunch", "dinner", "snack"],
                            "default": "snack",
                        },
                        "log_date": {
                            "type": "string",
                            "description": "The date and time when the food was consumed.",
                            "default": None,
                        },
                    },
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["log_food"] * 4
    assert [call["arguments"]["food_name"] for call in plan["calls"]] == [
        "frozen mango",
        "chai tea",
        "pepperoni pizza",
        "coffee",
    ]
    assert [call["arguments"]["portion_amount"] for call in plan["calls"]] == [8.0, 1.0, 2.0, 1.0]
    assert [call["arguments"]["portion_unit"] for call in plan["calls"]] == ["piece", "cup", "slice", "cup"]
    assert all("log_date" not in call["arguments"] for call in plan["calls"])


def test_tool_binding_rejects_browser_screenshot_as_http_request():
    plan = build_tool_binding_plan(
        "I need to take a screenshot of the current website shown in Google Chrome. How can I do this with Python?",
        [
            {
                "name": "requests.get",
                "description": "Perform an HTTP GET request for a URL.",
                "parameters": {
                    "type": "dict",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_rejects_card_probability_as_coin_tool():
    plan = build_tool_binding_plan(
        "What is the probability of drawing a face card from a standard deck?",
        [
            {
                "name": "probability.coin_toss_heads",
                "description": "Calculate the probability of getting heads in repeated coin tosses.",
                "parameters": {
                    "type": "dict",
                    "properties": {"tosses": {"type": "integer"}},
                    "required": ["tosses"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_platform_level_tool_counts_platforms_not_items():
    plan = build_tool_binding_plan(
        "Find the list of TV shows and their ratings on Netflix for 'Friends', and Hulu for 'The Office' and 'Stranger Things'.",
        [
            {
                "name": "streaming_services.shows_list_and_ratings",
                "description": "Get show ratings on a specific streaming service.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "streaming_service": {"type": "string"},
                        "show_list": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["streaming_service", "show_list"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "streaming_services.shows_list_and_ratings",
        "streaming_services.shows_list_and_ratings",
    ]


def test_tool_binding_uses_lone_retrieval_tool_for_actionable_request():
    plan = build_tool_binding_plan(
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
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["ToolSearcher"]
    assert "stock" in plan["calls"][0]["arguments"]["keywords"]


def test_tool_binding_prefers_retrieved_concrete_api_over_search_tool():
    plan = build_tool_binding_plan(
        "User: Can you please tell me the stock price of Microsoft on 4th February 2022? "
        "Prior API result: {'name': 'QueryStock', 'description': 'This API queries the stock price of a given stock.'}",
        [
            {
                "name": "ToolSearcher",
                "description": "Searches for relevant tools in library based on the keywords.",
                "input_parameters": {
                    "keywords": {"type": "str", "description": "The keyword to search for."}
                },
            },
            {
                "name": "QueryStock",
                "description": "This API queries the stock price of a given stock.",
                "input_parameters": {
                    "stock_code": {"type": "str", "description": "The stock code of the given stock."},
                    "date": {"type": "str", "description": "The date of the stock price."},
                },
            },
        ],
    )

    assert plan["calls"]
    assert plan["calls"][0]["tool_name"] == "QueryStock"


def test_tool_binding_stock_code_slot_uses_company_from_stock_request():
    plan = build_tool_binding_plan(
        "User: Can you tell me the stock price of Amazon on March 12th, 2022?",
        [
            {
                "name": "QueryStock",
                "description": "This API queries the stock price of a given stock.",
                "input_parameters": {
                    "stock_code": {"type": "str", "description": "The stock code of the given stock."},
                    "date": {"type": "str", "description": "The date of the stock price."},
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["QueryStock"]
    assert plan["calls"][0]["arguments"]["stock_code"] == "Amazon"


def test_tool_binding_counts_multiple_numeric_array_payload_groups():
    plan = build_tool_binding_plan(
        "Calculate the standard deviation for each dataset: the first dataset has 1, 2, 3, "
        "the second dataset has 4, 5, 6, and the third dataset has 7, 8, 9.",
        [
            {
                "name": "calculate_standard_deviation",
                "description": "Calculates the standard deviation of a list of numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"numbers": {"type": "array", "items": {"type": "float"}}},
                    "required": ["numbers"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_standard_deviation",
        "calculate_standard_deviation",
        "calculate_standard_deviation",
    ]


def test_tool_binding_keeps_single_numeric_array_payload_as_one_call():
    plan = build_tool_binding_plan(
        "Sort this list of numbers in descending order: 45, 23, 67, 89, 12, 34, 56, 78.",
        [
            {
                "name": "sort_array",
                "description": "Sort an array of numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["numbers"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["sort_array"]


def test_tool_binding_counts_repeated_numeric_slot_values():
    plan = build_tool_binding_plan(
        "Calculate the monthly mortgage payment for a loan amount of $400,000, "
        "with an annual interest rate of 4% and a loan term of 15, 20 and 30 years.",
        [
            {
                "name": "calculate_mortgage_payment",
                "description": "Calculate monthly mortgage payment.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "loan_amount": {"type": "integer"},
                        "interest_rate": {"type": "float"},
                        "loan_term": {"type": "integer", "description": "The loan term in years."},
                    },
                    "required": ["loan_amount", "interest_rate", "loan_term"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_mortgage_payment",
        "calculate_mortgage_payment",
        "calculate_mortgage_payment",
    ]


def test_tool_binding_explicit_numeric_list_overrides_semantic_single_call_cap():
    plan = build_tool_binding_plan(
        "Find the factorial of 5,10 and 15.",
        [
            {
                "name": "math.factorial",
                "description": "Calculate the factorial of a given positive integer.",
                "parameters": {
                    "type": "dict",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "math.factorial",
        "math.factorial",
        "math.factorial",
    ]


def test_tool_binding_explicit_repeated_term_overrides_semantic_single_call_cap():
    plan = build_tool_binding_plan(
        "Calculate the monthly mortgage payment for a loan amount of $400,000, "
        "with an annual interest rate of 4% and a loan term of 15, 20 and 30 years.",
        [
            {
                "name": "calculate_mortgage_payment",
                "description": "Calculate monthly mortgage payment.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "loan_amount": {"type": "integer"},
                        "interest_rate": {"type": "float"},
                        "loan_term": {"type": "integer", "description": "The loan term in years."},
                    },
                    "required": ["loan_amount", "interest_rate", "loan_term"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_mortgage_payment",
        "calculate_mortgage_payment",
        "calculate_mortgage_payment",
    ]


def test_tool_binding_multi_intent_conflicts_do_not_drop_requested_tool():
    plan = build_tool_binding_plan(
        "Calculate the area under the curve from x=1 to x=5 for f(x)=x^2. "
        "And find the derivative at x=3.",
        [
            {
                "name": "integral",
                "description": "Calculate the definite integral of a function over an interval.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "function": {"type": "string"},
                        "a": {"type": "float"},
                        "b": {"type": "float"},
                    },
                    "required": ["function", "a", "b"],
                },
            },
            {
                "name": "derivative",
                "description": "Find the derivative of a function at a point.",
                "parameters": {
                    "type": "dict",
                    "properties": {"function": {"type": "string"}, "x": {"type": "float"}},
                    "required": ["function", "x"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["integral", "derivative"]


def test_tool_binding_semantic_frame_topic_binds_domain_name_and_prose_enum():
    plan = build_tool_binding_plan(
        "Retrieve the full historyof Buddhism",
        [
            {
                "name": "retrieve_religion_info",
                "description": "Retrieve the history and main beliefs of a religion.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "religion_name": {"type": "string", "description": "The name of the religion."},
                        "detail_level": {
                            "type": "string",
                            "description": "Level of detail for the returned information, either 'summary' or 'full'.",
                            "default": "summary",
                        },
                    },
                    "required": ["religion_name", "detail_level"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve the full history of Buddhism",
                "slots_observed": [
                    {
                        "role": "topic",
                        "value": "Buddhism",
                        "value_type": "text",
                        "evidence_span": "Buddhism",
                        "status": "semantic",
                        "confidence": 1.0,
                    }
                ],
                "call_groups": [
                    {
                        "intent": "search",
                        "unit_of_work": "search for full history of Buddhism",
                        "requested_entities": ["search_query"],
                        "expected_call_count": 1,
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["retrieve_religion_info"]
    assert plan["calls"][0]["arguments"]["religion_name"] == "Buddhism"
    assert plan["calls"][0]["arguments"]["detail_level"] == "full"


def test_tool_binding_promotes_soft_event_evidence_to_argument():
    plan = build_tool_binding_plan(
        "When did the Treaty of Tordesillas take place? Put it in the format of YYYY.",
        [
            {
                "name": "european_history.get_event_date",
                "description": "Retrieve the date of a specific event in European history.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "event_name": {"type": "string", "description": "The name of the event."},
                        "format": {"type": "string", "description": "Optional format of the returned date. Default is 'MM-DD-YYYY'."},
                    },
                    "required": ["event_name"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["european_history.get_event_date"]
    assert plan["calls"][0]["arguments"]["event_name"] == "Tordesillas"


def test_tool_binding_uses_description_defaults_for_required_slots():
    plan = build_tool_binding_plan(
        "What is the probability of getting a full house in poker?",
        [
            {
                "name": "poker_probability.full_house",
                "description": "Calculate the probability of getting a full house in a poker game.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "deck_size": {"type": "integer", "description": "The size of the deck. Default is 52."},
                        "hand_size": {"type": "integer", "description": "The size of the hand. Default is 5."},
                    },
                    "required": ["deck_size", "hand_size"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["poker_probability.full_house"]
    assert plan["calls"][0]["arguments"]["deck_size"] == 52
    assert plan["calls"][0]["arguments"]["hand_size"] == 5


def test_tool_binding_extracts_conversion_units_and_currency_names():
    unit_plan = build_tool_binding_plan(
        "How many ounces in 2 pounds of butter?",
        [
            {
                "name": "cooking_conversion.convert",
                "description": "Convert cooking measurements from one unit to another.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "quantity": {"type": "integer"},
                        "from_unit": {"type": "string"},
                        "to_unit": {"type": "string"},
                        "item": {"type": "string"},
                    },
                    "required": ["quantity", "from_unit", "to_unit", "item"],
                },
            }
        ],
    )
    currency_plan = build_tool_binding_plan(
        "How much will 20000 Japanese Yen be in United States Dollar?",
        [
            {
                "name": "convert_currency",
                "description": "Converts an amount from a particular currency to another currency.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base_currency": {"type": "string"},
                        "target_currency": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["base_currency", "target_currency", "amount"],
                },
            }
        ],
    )

    assert unit_plan["calls"][0]["arguments"] == {
        "quantity": 2,
        "from_unit": "pounds",
        "to_unit": "ounces",
        "item": "butter",
    }
    assert currency_plan["calls"][0]["arguments"] == {
        "base_currency": "JPY",
        "target_currency": "USD",
        "amount": 20000,
    }


def test_tool_binding_uses_grounded_slots_to_override_lexical_near_miss():
    structure_plan = build_tool_binding_plan(
        "What is the structural dynamic analysis of the building with building Id B1004 for 2nd, 3rd and 4th floors?",
        [
            {
                "name": "analyze_structure",
                "description": "Analyze a structure of a building based on its Id and floor numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "building_id": {"type": "string"},
                        "floors": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["building_id", "floors"],
                },
            }
        ],
    )
    schedule_plan = build_tool_binding_plan(
        "What are the next five matches for Manchester United and who are they playing against in the English Premier League?",
        [
            {
                "name": "sports.match_schedule",
                "description": "Retrieve the match schedule for a specific sports team.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "team_name": {"type": "string"},
                        "num_matches": {"type": "integer", "description": "The number of upcoming matches you want to get."},
                    },
                    "required": ["team_name", "num_matches"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in structure_plan["calls"]] == ["analyze_structure"]
    assert [call["tool_name"] for call in schedule_plan["calls"]] == ["sports.match_schedule"]


def test_tool_binding_repeats_loan_eligibility_by_financial_institution():
    plan = build_tool_binding_plan(
        "Check loan eligibility for HSBC for an amount of 500000 with annual income 100000 and Wells Fargo for an amount of 700000 with annual income 120000.",
        [
            {
                "name": "loan_eligibility_check",
                "description": "Check for eligibility for a loan given income and loan amount.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "financial_institution": {"type": "string", "description": "The name of the financial institution e.g. HSBC"},
                        "loan_amount": {"type": "integer", "description": "The loan amount that is requested"},
                        "annual_income": {"type": "integer", "description": "Annual income of the applicant"},
                    },
                    "required": ["financial_institution", "loan_amount", "annual_income"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["loan_eligibility_check", "loan_eligibility_check"]
    assert [call["arguments"]["financial_institution"] for call in plan["calls"]] == ["HSBC", "Wells Fargo"]
    assert [call["arguments"]["loan_amount"] for call in plan["calls"]] == [500000, 700000]
    assert [call["arguments"]["annual_income"] for call in plan["calls"]] == [100000, 120000]


def test_tool_binding_repeats_circle_circumference_for_explicit_circle_count():
    plan = build_tool_binding_plan(
        "Calculate the circumference for four circles: first circle radius 5 cm, second circle radius 10 cm, third circle radius 15 cm, and fourth circle radius 20 cm.",
        [
            {
                "name": "calculate_circumference",
                "description": "Calculate circumference of a circle.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "radius": {"type": "integer", "description": "The radius of the circle"},
                        "unit": {"type": "string", "description": "The unit of measure"},
                    },
                    "required": ["radius"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_circumference"] * 4
    assert [call["arguments"]["radius"] for call in plan["calls"]] == [5, 10, 15, 20]


def test_tool_binding_repeats_prime_factors_for_oxford_comma_number_list():
    plan = build_tool_binding_plan(
        "Find the prime factors of 45, 100, and 150.",
        [
            {
                "name": "number_analysis.prime_factors",
                "description": "Find the prime factors of a number.",
                "parameters": {
                    "type": "dict",
                    "properties": {"number": {"type": "integer", "description": "The number to factor"}},
                    "required": ["number"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["number_analysis.prime_factors"] * 3
    assert [call["arguments"]["number"] for call in plan["calls"]] == [45, 100, 150]


def test_tool_binding_keeps_enum_array_features_in_single_museum_call():
    plan = build_tool_binding_plan(
        "Get museum info for the Louvre in Paris with timings, exhibitions, and accessibility.",
        [
            {
                "name": "museum_info",
                "description": "Retrieve museum information.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "museum": {"type": "string", "description": "The museum name"},
                        "city": {"type": "string", "description": "The city"},
                        "features": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["timings", "exhibitions", "accessibility", "events", "history"]},
                            "description": "Requested museum information features",
                        },
                    },
                    "required": ["museum", "city"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 3}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["museum_info"]
    assert set(plan["calls"][0]["arguments"]["features"]) == {"timings", "exhibitions", "accessibility"}


def test_tool_binding_museum_city_pairs_do_not_overcount_venue_names():
    plan = build_tool_binding_plan(
        "Find opening hours and ticket prices for adults and children for the National Museum in Washington D.C. and the Louvre Museum in Paris.",
        [
            {
                "name": "museum_info.get_info",
                "description": "Retrieve specific details about museums, such as opening hours and ticket prices.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "City where the museum is located."},
                        "details": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["Opening hours", "Adult tickets", "Child tickets"]},
                            "description": "List of details to retrieve about the museum.",
                        },
                    },
                    "required": ["location", "details"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["museum_info.get_info"] * 2
    assert [call["arguments"]["location"] for call in plan["calls"]] == ["Washington D.C.", "Paris"]
    assert all(
        set(call["arguments"]["details"]) == {"Opening hours", "Adult tickets", "Child tickets"}
        for call in plan["calls"]
    )


def test_tool_binding_repeats_gcd_for_named_number_pairs():
    plan = build_tool_binding_plan(
        "John chose the numbers 36 and 48, while Mary chose the numbers 60 and 96. Calculate the greatest common divisor for each pair.",
        [
            {
                "name": "math.gcd",
                "description": "Calculate the greatest common divisor of two numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "num1": {"type": "integer", "description": "First number"},
                        "num2": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["num1", "num2"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["math.gcd", "math.gcd"]
    assert [(call["arguments"]["num1"], call["arguments"]["num2"]) for call in plan["calls"]] == [(36, 48), (60, 96)]


def test_tool_binding_repeats_same_tool_for_relevant_intent_clauses():
    plan = build_tool_binding_plan(
        "Find all law cases where Charles Dickens is a party and it happened in Boston. "
        "Also, get cases where University of California was a party and happened in Los Angeles.",
        [
            {
                "name": "legal_case.find_parties",
                "description": "Find legal cases by party and city.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "party_name": {"type": "string", "description": "The name of the party involved in the case"},
                        "city": {"type": "string", "description": "The city where the case was heard"},
                    },
                    "required": ["party_name", "city"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "legal_case.find_parties",
        "legal_case.find_parties",
    ]


def test_tool_binding_repeats_for_optional_season_selector_values():
    plan = build_tool_binding_plan(
        "Check if the player with id 3142 in team RocketLeague has achieved top scorer status in seasons 2017, 2018 and 2019.",
        [
            {
                "name": "player_status.check",
                "description": "Check a player's status in a team for a particular season.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "team": {"type": "string", "description": "The team where the player plays."},
                        "player_id": {"type": "integer", "description": "The id of the player."},
                        "season": {"type": "integer", "description": "The season for which player's status need to be checked. Optional."},
                    },
                    "required": ["team", "player_id"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "player_status.check",
        "player_status.check",
        "player_status.check",
    ]
    assert [call["arguments"]["season"] for call in plan["calls"]] == [2017, 2018, 2019]


def test_tool_binding_counts_repeated_semantic_scalar_not_enum_array_payload():
    plan = build_tool_binding_plan(
        "What are the RGB and HEX color values for navy, purple and maroon?",
        [
            {
                "name": "color_converter.get_color_info",
                "description": "Retrieve RGB values and hexadecimal codes of a specific color.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "color_name": {"type": "string", "description": "The name of the color."},
                        "conversion_type": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["RGB", "HEX"]},
                            "description": "The conversion type for the color.",
                        },
                    },
                    "required": ["color_name"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "slots_observed": [
                    {
                        "role": "color_names",
                        "value": ["navy", "purple", "maroon"],
                        "value_type": "text",
                        "evidence_span": "navy, purple and maroon",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                    {
                        "role": "color_value_format",
                        "value": ["RGB", "HEX"],
                        "value_type": "text",
                        "evidence_span": "RGB and HEX",
                        "status": "explicit",
                        "confidence": 1.0,
                    },
                ],
                "call_groups": [{"expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["color_converter.get_color_info"] * 3
    assert [call["arguments"]["color_name"] for call in plan["calls"]] == ["navy", "purple", "maroon"]


def test_tool_binding_semantic_cardinality_uses_arg_name_not_loose_description():
    plan = build_tool_binding_plan(
        "Calculate the induced electromagnetic force for a magnetic field of 5 Tesla, area of 2 square meters and change in time of 4 seconds, then repeat with a change in time of 10 seconds.",
        [
            {
                "name": "calculate_em_force",
                "description": "Calculate induced electromagnetic force.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "b_field": {"type": "integer", "description": "The magnetic field in Tesla."},
                        "area": {"type": "integer", "description": "The change in area of magnetic field in square meters."},
                        "d_time": {"type": "integer", "description": "The change in time in seconds."},
                    },
                    "required": ["b_field", "area", "d_time"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "slots_observed": [
                    {"role": "magnetic_field", "value": "5", "value_type": "number", "evidence_span": "5", "status": "explicit", "confidence": 1.0},
                    {"role": "area", "value": "2", "value_type": "number", "evidence_span": "2", "status": "explicit", "confidence": 1.0},
                    {"role": "change_in_time", "value": "4", "value_type": "number", "evidence_span": "4 seconds", "status": "explicit", "confidence": 1.0},
                    {"role": "change_in_time", "value": "10", "value_type": "number", "evidence_span": "10 seconds", "status": "explicit", "confidence": 1.0},
                ],
                "call_groups": [{"expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_em_force", "calculate_em_force"]
    assert [call["arguments"]["d_time"] for call in plan["calls"]] == [4, 10]


def test_tool_binding_repeats_unit_labeled_numeric_durations_without_treating_next_as_result_count():
    plan = build_tool_binding_plan(
        "Calculate alimony for the next 10 years and 20 years.",
        [
            {
                "name": "calculate_alimony",
                "description": "Calculate alimony payment duration.",
                "parameters": {
                    "type": "dict",
                    "properties": {"years": {"type": "integer", "description": "Duration in years."}},
                    "required": ["years"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_alimony", "calculate_alimony"]
    assert [call["arguments"]["years"] for call in plan["calls"]] == [10, 20]


def test_tool_binding_repeats_route_pairs_and_binds_each_pair_by_call_index():
    plan = build_tool_binding_plan(
        "Calculate driving distance from Austin to Dallas and then from Houston to San Antonio.",
        [
            {
                "name": "driving_distance",
                "description": "Calculate driving distance for a route.",
                "parameters": {
                    "type": "dict",
                    "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
                    "required": ["origin", "destination"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["driving_distance", "driving_distance"]
    assert [(call["arguments"]["origin"], call["arguments"]["destination"]) for call in plan["calls"]] == [
        ("Austin", "Dallas"),
        ("Houston", "San Antonio"),
    ]


def test_tool_binding_repeats_parenthesized_numeric_tuples():
    plan = build_tool_binding_plan(
        "Calculate hypotenuse for (3, 4), (6, 8), and (9, 12).",
        [
            {
                "name": "math.hypot",
                "description": "Calculate hypotenuse.",
                "parameters": {
                    "type": "dict",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            }
        ],
        capability_plan={"semantic_input_frame": {"call_groups": [{"expected_call_count": 1}]}},
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["math.hypot"] * 3
    assert [(call["arguments"]["x"], call["arguments"]["y"]) for call in plan["calls"]] == [(3, 4), (6, 8), (9, 12)]


def test_tool_binding_numbered_semantic_roles_match_schema_slot_for_cardinality():
    plan = build_tool_binding_plan(
        "Get event information for the Magna Carta and the Renaissance.",
        [
            {
                "name": "event_info",
                "description": "Get information about an event.",
                "parameters": {
                    "type": "dict",
                    "properties": {"event": {"type": "string"}},
                    "required": ["event"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "slots_observed": [
                    {"role": "topic_1", "value": "Magna Carta", "value_type": "text", "evidence_span": "Magna Carta", "status": "explicit", "confidence": 1.0},
                    {"role": "topic_2", "value": "Renaissance", "value_type": "text", "evidence_span": "Renaissance", "status": "explicit", "confidence": 1.0},
                ],
                "call_groups": [{"expected_call_count": 1}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["event_info", "event_info"]
    assert [call["arguments"]["event"] for call in plan["calls"]] == ["Magna Carta", "Renaissance"]


def test_tool_binding_accepts_grounded_model_tool_bindings():
    plan = build_tool_binding_plan(
        "Calculate hypotenuse for (3, 4) and (6, 8).",
        [
            {
                "name": "math.hypot",
                "description": "Calculate hypotenuse.",
                "parameters": {
                    "type": "dict",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate two hypotenuses.",
                "call_groups": [{"intent": "calculate hypotenuse", "expected_call_count": 2}],
                "tool_bindings": [
                    {
                        "tool_name": "math.hypot",
                        "intent": "calculate hypotenuse",
                        "call_count": 2,
                        "argument_groups": [
                            {"arguments": {"x": 3, "y": 4}, "evidence_spans": {"x": "3", "y": "4"}},
                            {"arguments": {"x": 6, "y": 8}, "evidence_spans": {"x": "6", "y": "8"}},
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["used"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["math.hypot", "math.hypot"]
    assert [(call["arguments"]["x"], call["arguments"]["y"]) for call in plan["calls"]] == [(3, 4), (6, 8)]
    assert plan["calls"][0]["argument_evidence"] == {"x": "3", "y": "4"}


def test_tool_binding_accepts_evidence_backed_inferred_required_numeric():
    plan = build_tool_binding_plan(
        "Calculate the final velocity of an object, knowing that it started from rest, accelerated at a rate of 9.8 m/s^2 for a duration of 5 seconds.",
        [
            {
                "name": "calculate_final_velocity",
                "description": "Calculate final velocity.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "initial_velocity": {"type": "integer"},
                        "acceleration": {"type": "number"},
                        "time": {"type": "integer"},
                    },
                    "required": ["initial_velocity", "acceleration", "time"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate final velocity from rest over 5 seconds.",
                "call_groups": [{"intent": "calculate final velocity", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "calculate_final_velocity",
                        "call_count": 1,
                        "arguments": {"initial_velocity": 0, "acceleration": 9.8, "time": 5},
                        "evidence_spans": {
                            "initial_velocity": "started from rest",
                            "acceleration": "9.8 m/s^2",
                            "time": "5 seconds",
                        },
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_final_velocity"]
    assert plan["calls"][0]["arguments"] == {"initial_velocity": 0, "acceleration": 9.8, "time": 5}


def test_tool_binding_merges_single_empty_argument_group_with_top_level_arguments():
    plan = build_tool_binding_plan(
        "Get the average temperature in Austin for the next 3 days in Celsius.",
        [
            {
                "name": "average_temperature",
                "description": "Get the average temperature.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer"},
                        "temp_unit": {"type": "string"},
                    },
                    "required": ["location", "days"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve average temperature for Austin over the next 3 days in Celsius.",
                "call_groups": [{"intent": "average_temperature", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "average_temperature",
                        "call_count": 1,
                        "argument_groups": [{}],
                        "arguments": {"location": "Austin", "days": 3, "temp_unit": "Celsius"},
                        "evidence_spans": {
                            "location": "Austin",
                            "days": "next 3 days",
                            "temp_unit": "Celsius",
                        },
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["average_temperature"]
    assert plan["calls"][0]["arguments"] == {"location": "Austin", "days": 3, "temp_unit": "Celsius"}


def test_tool_binding_ignores_ungrounded_optional_model_argument():
    plan = build_tool_binding_plan(
        "Find the highest grossing bank in the U.S for year 2020.",
        [
            {
                "name": "highest_grossing_banks",
                "description": "Find high grossing banks.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "country": {"type": "string"},
                        "year": {"type": "integer"},
                        "top_n": {"type": "integer"},
                    },
                    "required": ["country", "year"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Find the highest grossing bank in the U.S for 2020.",
                "call_groups": [{"intent": "highest grossing bank", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "highest_grossing_banks",
                        "call_count": 1,
                        "arguments": {"country": "U.S", "year": 2020, "top_n": 1},
                        "evidence_spans": {"country": "U.S", "year": "2020"},
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["highest_grossing_banks"]
    assert plan["calls"][0]["arguments"] == {"country": "U.S", "year": 2020}


def test_tool_binding_rejects_inferred_identifier_numeric():
    plan = build_tool_binding_plan(
        "Delete the reminder about groceries.",
        [
            {
                "name": "DeleteReminder",
                "description": "Delete a reminder by id.",
                "parameters": {
                    "type": "dict",
                    "properties": {"reminder_id": {"type": "integer"}},
                    "required": ["reminder_id"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Delete the grocery reminder.",
                "call_groups": [{"intent": "delete reminder", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "DeleteReminder",
                        "call_count": 1,
                        "arguments": {"reminder_id": 1},
                        "evidence_spans": {"reminder_id": "groceries"},
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert plan["calls"] == []


def test_tool_binding_model_binding_allows_query_word_for_strong_identity_tool():
    plan = build_tool_binding_plan(
        "Get me the predictions of the evolutionary rate for Homo Sapiens for next 50 years using Darwin model",
        [
            {
                "name": "prediction.evolution",
                "description": "Predict the evolutionary rate for a specific species for a given timeframe.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "species": {"type": "string"},
                        "years": {"type": "integer"},
                        "model": {"type": "string", "default": "Darwin"},
                    },
                    "required": ["species", "years"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Predict evolutionary rate for Homo Sapiens for 50 years using Darwin model",
                "call_groups": [{"tool_name": "prediction.evolution", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "prediction.evolution",
                        "call_count": 1,
                        "arguments": {"species": "Homo Sapiens", "years": 50, "model": "Darwin"},
                        "evidence_spans": {
                            "species": "Homo Sapiens",
                            "years": "50",
                            "model": "Darwin",
                        },
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["prediction.evolution"]


def test_tool_binding_model_binding_allows_retail_request_that_mentions_water():
    plan = build_tool_binding_plan(
        "I want to buy apples, rice, and 12 pack of bottled water from a Walmart near San Jose. Show me the product information and stock availability.",
        [
            {
                "name": "walmart.purchase",
                "description": "Retrieve information of items from Walmart including stock availability.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "loc": {"type": "string"},
                        "product_list": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["loc", "product_list"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve product information and stock availability for apples, rice, and bottled water from Walmart near San Jose.",
                "call_groups": [{"tool_name": "walmart.purchase", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "walmart.purchase",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "loc": "San Jose",
                                    "product_list": ["apples", "rice", "12 pack of bottled water"],
                                },
                                "evidence_spans": {
                                    "loc": "San Jose",
                                    "product_list": "apples, rice, and 12 pack of bottled water",
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["walmart.purchase"]


def test_tool_binding_nutrient_words_route_to_nutrition_tool():
    plan = build_tool_binding_plan(
        "Check the amount of protein, calories and carbs in an avocado from Walmart.",
        [
            {
                "name": "grocery_info.nutritional_info",
                "description": "Retrieve nutritional information for a given food item from a particular store",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "store": {"type": "string"},
                        "food": {"type": "string"},
                        "information": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["Protein", "Calories", "Carbohydrates", "Fat", "Fiber"],
                            },
                        },
                    },
                    "required": ["store", "food", "information"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve protein, calories, and carbohydrates for an avocado from Walmart.",
                "call_groups": [{"operation": "grocery_info.nutritional_info", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "grocery_info.nutritional_info",
                        "call_count": 1,
                        "arguments": {
                            "store": "Walmart",
                            "food": "avocado",
                            "information": ["Protein", "Calories", "Carbohydrates"],
                        },
                        "evidence_spans": {
                            "store": "Walmart",
                            "food": "avocado",
                            "information": "protein, calories and carbs",
                        },
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["grocery_info.nutritional_info"]


def test_tool_binding_honors_explicit_semantic_no_tool_decision():
    plan = build_tool_binding_plan(
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
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "no_tool",
                "canonical_request": "Find the freezing point of water at 10 kPa.",
                "slots_observed": [],
                "call_groups": [],
                "tool_bindings": [],
                "missing_inputs": [],
            }
        },
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []
    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["model_tool_binding"]["decision"] == "no_tool"


def test_tool_binding_honors_explicit_semantic_ask_user_decision():
    plan = build_tool_binding_plan(
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
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "ask_user",
                "canonical_request": "Find the roots of the linear equation bx + c = 0.",
                "slots_observed": [],
                "call_groups": [],
                "tool_bindings": [],
                "missing_inputs": ["b", "c"],
            }
        },
    )

    assert plan["tool_decision"] == "ask_user"
    assert plan["calls"] == []
    assert plan["missing_inputs"] == ["b", "c"]
    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["model_tool_binding"]["decision"] == "ask_user"


def test_tool_binding_keeps_catalog_search_when_semantic_no_tool_has_only_search_tool():
    plan = build_tool_binding_plan(
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
        capability_plan={
            "semantic_input_frame": {
                "tool_decision": "no_tool",
                "canonical_request": "Find a stock-price API for Microsoft on 4th February 2022.",
                "slots_observed": [],
                "call_groups": [],
                "tool_bindings": [],
                "missing_inputs": [],
            }
        },
    )

    assert plan["tool_decision"] == "call"
    assert [call["tool_name"] for call in plan["calls"]] == ["ToolSearcher"]
    assert plan["model_tool_binding"]["accepted"] is False


def test_tool_binding_rejects_duplicate_required_slot_fill_during_recovery():
    plan = build_tool_binding_plan(
        "Find a picnic spot in Miami.",
        [
            {
                "name": "local_fauna",
                "description": "Find local fauna species by location and species type.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string"},
                        "species_type": {"type": "string"},
                    },
                    "required": ["location", "species_type"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["candidate_tool_audits"][0]["slot_binding_warnings"] == ["duplicate_required_slot_values"]
    assert plan["candidate_tool_audits"][0].get("recovery_reason") is None


def test_tool_binding_rejects_duplicate_string_fill_for_distinct_required_roles():
    plan = build_tool_binding_plan(
        "Who was the artist behind the famous painting 'The Scream'?",
        [
            {
                "name": "artwork_search",
                "description": "Find details about an artwork given its name.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "artwork_name": {"type": "string", "description": "The name of the artwork."},
                        "museum_location": {
                            "type": "string",
                            "description": "The location of the museum, e.g., Paris, France.",
                        },
                        "specific_details": {
                            "type": "string",
                            "description": "Specific details wanted such as 'artist', 'year', etc.",
                            "default": "all details",
                        },
                    },
                    "required": ["artwork_name", "museum_location"],
                },
            }
        ],
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []
    assert plan["candidate_tool_audits"][0]["slot_binding_warnings"] == ["duplicate_required_slot_values"]


def test_tool_binding_rejects_partial_model_binding_and_recovers_data_dependency_chain():
    request = (
        "Use the data from dataset.csv file and fit a linear regression model to predict future sales "
        "by setting x=data['sales'] and y=data['future_sales']. Additionally, calculate and return the residuals."
    )

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "linear_regression_fit",
                "description": "Fit a linear regression model to data.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "x": {"type": "array", "items": {"type": "float"}, "description": "Array of the predictor variable."},
                        "y": {"type": "array", "items": {"type": "float"}, "description": "Array of the dependent variable."},
                        "return_residuals": {
                            "type": "boolean",
                            "description": "Flag indicating whether to return the residuals. Optional.",
                            "default": "false",
                        },
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "data_loading",
                "description": "Load data from a csv file into a data structure.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "file_path": {"type": "string", "description": "The path to the file to load."},
                        "delimiter": {"type": "string", "description": "The character used to separate values in the file. Optional.", "default": ","},
                    },
                    "required": ["file_path"],
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Load dataset.csv, fit a linear regression model, and return residuals.",
                "call_groups": [
                    {"intent": "load data", "expected_call_count": 1},
                    {"intent": "fit linear regression", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "data_loading",
                        "intent": "load csv file",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"file_path": "dataset.csv"}, "evidence_spans": {"file_path": "dataset.csv"}}
                        ],
                    }
                ],
                "missing_inputs": ["x", "y"],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert [call["tool_name"] for call in plan["calls"]] == ["data_loading", "linear_regression_fit"]
    assert plan["calls"][1]["arguments"] == {
        "x": "data['sales']",
        "y": "data['future_sales']",
        "return_residuals": True,
    }


def test_tool_binding_preserves_independent_model_binding_order_without_dependency():
    request = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "financial_ratios.interest_coverage",
                "description": "Calculate interest coverage ratio.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company_name": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["company_name", "years"],
                },
            },
            {
                "name": "sales_growth.calculate",
                "description": "Calculate sales growth rate.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["company", "years"],
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Retrieve sales growth and interest coverage for XYZ over three years.",
                "call_groups": [
                    {"intent": "calculate sales growth", "expected_call_count": 1},
                    {"intent": "calculate interest coverage", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "sales_growth.calculate",
                        "intent": "calculate sales growth",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"company": "XYZ", "years": 3}, "evidence_spans": {"company": "company XYZ", "years": "last 3 years"}}
                        ],
                    },
                    {
                        "tool_name": "financial_ratios.interest_coverage",
                        "intent": "calculate interest coverage",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"company_name": "XYZ", "years": 3}, "evidence_spans": {"company_name": "company XYZ", "years": "last 3 years"}}
                        ],
                    },
                ],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "sales_growth.calculate",
        "financial_ratios.interest_coverage",
    ]


def test_tool_binding_groups_clause_scoped_repeated_operations_in_prompt_order():
    request = (
        "What was the average life expectancy in the USA in the year 1900 and 1950? "
        "Additionally, what was the Gross Domestic Product (GDP) of the USA in these years?"
    )

    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "us_history.gdp",
                "description": "Retrieves the Gross Domestic Product of the USA for a specific year.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year for which to retrieve GDP data.",
                        }
                    },
                    "required": ["year"],
                },
            },
            {
                "name": "us_history.life_expectancy",
                "description": "Retrieves the average life expectancy of the USA for a specific year.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "year": {
                            "type": "integer",
                            "description": "The year for which to retrieve life expectancy.",
                        }
                    },
                    "required": ["year"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "us_history.life_expectancy",
        "us_history.life_expectancy",
        "us_history.gdp",
        "us_history.gdp",
    ]


def test_tool_binding_collapses_identical_single_workflow_step_calls():
    calls = [
        {
            "id": "call_1",
            "tool_name": "ForgotPassword",
            "arguments": {"status": "Forgot Password", "username": "foo", "email": "foo@example.com"},
            "depends_on": [],
            "missing_arguments": [],
        },
        {
            "id": "call_2",
            "tool_name": "ForgotPassword",
            "arguments": {"status": "Forgot Password", "username": "foo", "email": "foo@example.com"},
            "depends_on": [],
            "missing_arguments": [],
        },
    ]

    collapsed = _collapse_identical_repeated_calls("I forgot my password. My username is foo.", calls)
    repeated = _collapse_identical_repeated_calls("Repeat the same call again for username foo.", calls)

    assert [call["tool_name"] for call in collapsed] == ["ForgotPassword"]
    assert len(repeated) == 2


def test_tool_binding_collapses_identical_read_only_parallel_calls():
    calls = [
        {
            "id": "call_1",
            "tool_name": "geometry.calculate_cone_volume",
            "arguments": {"radius": 10, "height": 30, "round_off": 2},
            "depends_on": [],
            "missing_arguments": [],
        },
        {
            "id": "call_2",
            "tool_name": "geometry.calculate_cone_volume",
            "arguments": {"radius": 10, "height": 30, "round_off": 2},
            "depends_on": [],
            "missing_arguments": [],
        },
        {
            "id": "call_3",
            "tool_name": "physics.calculate_cone_mass",
            "arguments": {"radius": 10, "height": 30, "density": 5.2},
            "depends_on": [],
            "missing_arguments": [],
        },
        {
            "id": "call_4",
            "tool_name": "physics.calculate_cone_mass",
            "arguments": {"radius": 10, "height": 30, "density": 7.8},
            "depends_on": [],
            "missing_arguments": [],
        },
    ]
    tools = {
        "geometry.calculate_cone_volume": {
            "name": "geometry.calculate_cone_volume",
            "description": "Calculate the volume of a cone.",
        },
        "physics.calculate_cone_mass": {
            "name": "physics.calculate_cone_mass",
            "description": "Calculate the mass of a cone.",
        },
    }

    collapsed = _collapse_identical_repeated_calls(
        "Create two identical cones. Calculate the volume of each cone and the mass using two densities.",
        calls,
        tools,
    )

    assert [call["tool_name"] for call in collapsed] == [
        "geometry.calculate_cone_volume",
        "physics.calculate_cone_mass",
        "physics.calculate_cone_mass",
    ]


def test_tool_binding_preserves_identical_side_effect_parallel_calls():
    calls = [
        {
            "id": "call_1",
            "tool_name": "concert.book_ticket",
            "arguments": {"artist": "Taylor Swift", "location": "New York"},
            "depends_on": [],
            "missing_arguments": [],
        },
        {
            "id": "call_2",
            "tool_name": "concert.book_ticket",
            "arguments": {"artist": "Taylor Swift", "location": "New York"},
            "depends_on": [],
            "missing_arguments": [],
        },
    ]
    tools = {
        "concert.book_ticket": {
            "name": "concert.book_ticket",
            "description": "Book a concert ticket.",
        }
    }

    collapsed = _collapse_identical_repeated_calls(
        "Book two Taylor Swift tickets in New York.",
        calls,
        tools,
    )

    assert len(collapsed) == 2


def test_tool_binding_preserves_verified_model_single_structured_payload():
    request = (
        "My user ID is 123, time is 2023-09-01 10:15:00, and my health data is "
        "[{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]."
    )
    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "RecordHealthData",
                "description": "Record health data for a user.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "user_id": {"type": "string"},
                        "time": {"type": "string"},
                        "health_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["user_id", "time", "health_data"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Record one health data payload.",
                "call_groups": [{"intent": "record health data", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "RecordHealthData",
                        "intent": "record health data",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "user_id": "123",
                                    "time": "2023-09-01 10:15:00",
                                    "health_data": [
                                        {"name": "blood_pressure", "value": "120/80"},
                                        {"name": "heart_rate", "value": "80"},
                                    ],
                                },
                                "evidence_spans": {
                                    "user_id": "123",
                                    "time": "2023-09-01 10:15:00",
                                    "health_data": "[{'name': 'blood_pressure', 'value': '120/80'}, {'name': 'heart_rate', 'value': '80'}]",
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["RecordHealthData"]
    assert plan["calls"][0]["arguments"]["time"] == "2023-09-01 10:15:00"
    assert plan["calls"][0]["arguments"]["health_data"] == [
        {"name": "blood_pressure", "value": "120/80"},
        {"name": "heart_rate", "value": "80"},
    ]


def test_tool_binding_does_not_expand_verified_single_model_binding_with_multiple_slots():
    plan = build_tool_binding_plan(
        "How much revenue would company XYZ generate if we increase the sales units of product A by 10% while keeping the price the same?",
        [
            {
                "name": "corporate_finance.revenue_forecast",
                "description": "Forecast revenue for a company and product.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company": {"type": "string", "description": "Company name."},
                        "product": {"type": "string", "description": "Product name."},
                        "sales_units_increase_percentage": {
                            "type": "integer",
                            "description": "Sales units increase percentage.",
                        },
                    },
                    "required": ["company", "product"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Forecast revenue for company XYZ product A with a 10% sales-unit increase.",
                "call_groups": [{"intent": "calculate revenue", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "corporate_finance.revenue_forecast",
                        "intent": "Estimate revenue with increased sales units",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "company": "XYZ",
                                    "product": "A",
                                    "sales_units_increase_percentage": 10,
                                },
                                "evidence_spans": {
                                    "company": "company XYZ",
                                    "product": "product A",
                                    "sales_units_increase_percentage": "10%",
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["corporate_finance.revenue_forecast"]
    assert plan["calls"][0]["arguments"] == {
        "company": "XYZ",
        "product": "A",
        "sales_units_increase_percentage": 10,
    }


def test_tool_binding_does_not_invent_optional_enum_unit_from_slot_name():
    request = "¿Podrías decirme las condiciones actuales del clima en Cancún, QR, Playa del Carmen, QR y Tulum, QR?"
    plan = build_tool_binding_plan(
        request,
        [
            {
                "name": "get_current_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g. Boston, MA."},
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit.",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get current weather conditions for three locations.",
                "slots_observed": [
                    {"role": "location", "value": "Cancún, QR", "value_type": "location", "evidence_span": "Cancún, QR"},
                    {
                        "role": "location",
                        "value": "Playa del Carmen, QR",
                        "value_type": "location",
                        "evidence_span": "Playa del Carmen, QR",
                    },
                    {"role": "location", "value": "Tulum, QR", "value_type": "location", "evidence_span": "Tulum, QR"},
                ],
                "call_groups": [{"intent": "get weather", "expected_call_count": 3}],
            }
        },
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "get_current_weather",
        "get_current_weather",
        "get_current_weather",
    ]
    assert [call["arguments"]["location"] for call in plan["calls"]] == [
        "Cancún, QR",
        "Playa del Carmen, QR",
        "Tulum, QR",
    ]
    assert all("unit" not in call["arguments"] for call in plan["calls"])


def test_tool_binding_rejects_ungrounded_model_binding_and_falls_back():
    plan = build_tool_binding_plan(
        "What is the weather in Boston?",
        [
            {
                "name": "weather.get",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string", "description": "City to query."}},
                    "required": ["city"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get weather in Boston.",
                "call_groups": [{"intent": "get weather", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "weather.get",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"city": "Paris"}, "evidence_spans": {"city": "Paris"}}
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert any(item["code"] == "ungrounded_argument" for item in plan["model_tool_binding"]["diagnostics"])
    assert [call["tool_name"] for call in plan["calls"]] == ["weather.get"]
    assert plan["calls"][0]["arguments"]["city"] == "Boston"


def test_tool_binding_rejects_model_binding_for_semantically_wrong_single_tool():
    plan = build_tool_binding_plan(
        "Calculate the area of a triangle given the base is 10 meters and height is 5 meters.",
        [
            {
                "name": "determine_body_mass_index",
                "description": "Calculate body mass index given weight and height.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "weight": {"type": "float", "description": "Weight in kilograms."},
                        "height": {"type": "float", "description": "Height in meters."},
                    },
                    "required": ["weight", "height"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate triangle area from base 10 and height 5.",
                "call_groups": [{"intent": "calculate triangle area", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "determine_body_mass_index",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"weight": 10, "height": 5}, "evidence_spans": {"weight": "10", "height": "5"}}
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert any(item["code"] == "unsupported_tool_route" for item in plan["model_tool_binding"]["diagnostics"])
    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []


def test_tool_binding_accepts_grounded_generic_command_executor_steps():
    plan = build_tool_binding_plan(
        "list file in c drive and make file called testing.txt",
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
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "List files on the C drive and create testing.txt.",
                "call_groups": [
                    {"intent": "list files", "expected_call_count": 1},
                    {"intent": "create file", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "cmd_controller.execute",
                        "intent": "list files on C drive",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"command": "dir C:\\"},
                                "evidence_spans": {"command": "list file in c drive"},
                            }
                        ],
                    },
                    {
                        "tool_name": "cmd_controller.execute",
                        "intent": "create testing.txt",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"command": "echo. > C:\\testing.txt"},
                                "evidence_spans": {"command": "make file called testing.txt"},
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == [
        "cmd_controller.execute",
        "cmd_controller.execute",
    ]
    assert [call["arguments"]["command"] for call in plan["calls"]] == [
        "dir C:\\",
        "echo. > C:\\testing.txt",
    ]


def test_tool_binding_accepts_nested_object_model_binding_with_schema_aliases():
    plan = build_tool_binding_plan(
        "I'd like to change my food order to a Caesar salad without anchovies, and for the drink, "
        "can you update my order 123 to a large hot coffee with regular sweetness and almond milk, please?",
        [
            {
                "name": "ChaFod",
                "description": "Changes the food item based on the customer's request.",
                "parameters": {
                    "type": "dict",
                    "required": ["foodItem"],
                    "properties": {
                        "foodItem": {"type": "string", "description": "The food item to be modified."},
                        "newIngredients": {"type": "string", "default": ""},
                        "removeIngredients": {"type": "string", "default": ""},
                        "specialInstructions": {"type": "string", "default": ""},
                    },
                },
            },
            {
                "name": "ChaDri.change_drink",
                "description": "Modifies the existing drink order.",
                "parameters": {
                    "type": "dict",
                    "required": ["drink_id", "new_preferences"],
                    "properties": {
                        "drink_id": {"type": "string", "description": "The unique identifier of the drink."},
                        "new_preferences": {
                            "type": "dict",
                            "description": "The updated preferences for the drink order.",
                            "properties": {
                                "size": {"type": "string", "enum": ["small", "medium", "large"], "default": "medium"},
                                "temperature": {"type": "string", "enum": ["cold", "warm", "hot"], "default": "cold"},
                                "sweetness_level": {
                                    "type": "string",
                                    "enum": ["none", "light", "regular", "extra"],
                                    "default": "regular",
                                },
                                "milk_type": {
                                    "type": "string",
                                    "enum": ["regular", "soy", "almond", "coconut"],
                                    "default": "regular",
                                },
                            },
                        },
                    },
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Change the food and drink order.",
                "call_groups": [
                    {"intent": "change food", "expected_call_count": 1},
                    {"intent": "change drink", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "ChaFod",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"foodItem": "Caesar salad", "removeIngredients": "anchovies"},
                                "evidence_spans": {"foodItem": "Caesar salad", "removeIngredients": "anchovies"},
                            }
                        ],
                    },
                    {
                        "tool_name": "ChaDri.change_drink",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "drink_id": "123",
                                    "new_preferences": {
                                        "size": "large",
                                        "temperature": "hot",
                                        "sweetness": "regular",
                                        "milk": "almond milk",
                                    },
                                },
                                "evidence_spans": {"drink_id": "123"},
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["ChaFod", "ChaDri.change_drink"]
    assert plan["calls"][1]["arguments"]["new_preferences"] == {
        "size": "large",
        "temperature": "hot",
        "sweetness_level": "regular",
        "milk_type": "almond",
    }


def test_tool_binding_accepts_non_english_command_executor_model_binding():
    plan = build_tool_binding_plan(
        "거실, 에어컨, 실행하고, 침실, 공기청정기, 중지해줘.",
        [
            {
                "name": "ControlAppliance.execute",
                "description": "Control a home appliance. The command must be specified as a string in Korean.",
                "parameters": {
                    "type": "dict",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Korean command with room, appliance, and operation.",
                            "enum": ["거실, 에어컨, 실행", "다용도실, 통돌이, 중지", "침실, 공기청정기, 중지"],
                        }
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Run the living-room air conditioner and stop the bedroom air purifier.",
                "call_groups": [
                    {"intent": "control appliance", "expected_call_count": 1},
                    {"intent": "control appliance", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "ControlAppliance.execute",
                        "call_count": 2,
                        "argument_groups": [
                            {
                                "arguments": {"command": "거실, 에어컨, 실행"},
                                "evidence_spans": {"command": "거실, 에어컨, 실행"},
                            },
                            {
                                "arguments": {"command": "침실, 공기청정기, 중지"},
                                "evidence_spans": {"command": "침실, 공기청정기, 중지"},
                            },
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["arguments"]["command"] for call in plan["calls"]] == [
        "거실, 에어컨, 실행",
        "침실, 공기청정기, 중지",
    ]


def test_tool_binding_accepts_grounded_command_executor_value_missing_from_enum():
    plan = build_tool_binding_plan(
        "거실, 에어컨, 실행하고, 침실, 공기청정기, 중지해줘.",
        [
            {
                "name": "ControlAppliance.execute",
                "description": "Control a home appliance by executing a Korean command string.",
                "parameters": {
                    "type": "dict",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Korean command with room, appliance, and operation.",
                            "enum": ["거실, 에어컨, 실행"],
                        }
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Run the living-room air conditioner and stop the bedroom air purifier.",
                "call_groups": [
                    {"intent": "control appliance", "expected_call_count": 1},
                    {"intent": "control appliance", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "ControlAppliance.execute",
                        "call_count": 2,
                        "argument_groups": [
                            {
                                "arguments": {"command": "거실, 에어컨, 실행"},
                                "evidence_spans": {"command": "거실, 에어컨, 실행"},
                            },
                            {
                                "arguments": {"command": "침실, 공기청정기, 중지"},
                                "evidence_spans": {"command": "침실, 공기청정기, 중지"},
                            },
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["arguments"]["command"] for call in plan["calls"]] == [
        "거실, 에어컨, 실행",
        "침실, 공기청정기, 중지",
    ]


def test_tool_binding_accepts_schema_default_value_not_listed_in_enum():
    plan = build_tool_binding_plan(
        "Could you provide the latest news report for Paris, France, in English?",
        [
            {
                "name": "get_news_report",
                "description": "Fetch the latest news report for a location, category, and language.",
                "parameters": {
                    "type": "dict",
                    "required": ["location", "language"],
                    "properties": {
                        "location": {"type": "string", "description": "Location for the news report."},
                        "language": {"type": "string", "enum": ["English", "French", "German"]},
                        "category": {
                            "type": "string",
                            "description": "News category.",
                            "enum": ["Politics", "Technology", "Sports"],
                            "default": "General",
                        },
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get the general news report for Paris in English.",
                "call_groups": [{"intent": "get news report", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "get_news_report",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "location": "Paris, France",
                                    "language": "English",
                                    "category": "General",
                                },
                                "evidence_spans": {"location": "Paris, France", "language": "English"},
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["arguments"]["category"] == "General"


def test_tool_binding_accepts_include_boolean_grounded_by_argument_context():
    plan = build_tool_binding_plan(
        "Find the relevant Java classes for PaymentService, including subdirectories.",
        [
            {
                "name": "get_relevant_classes",
                "description": "Find relevant Java classes in a repository.",
                "parameters": {
                    "type": "dict",
                    "required": ["search_string"],
                    "properties": {
                        "search_string": {"type": "string", "description": "Class or symbol to search for."},
                        "include_subdirectories": {
                            "type": "boolean",
                            "description": "Whether to include subdirectories in the search.",
                            "default": False,
                        },
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Find PaymentService classes including subdirectories.",
                "call_groups": [{"intent": "find relevant classes", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "get_relevant_classes",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "search_string": "PaymentService",
                                    "include_subdirectories": True,
                                },
                                "evidence_spans": {"search_string": "PaymentService"},
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["arguments"]["include_subdirectories"] is True


def test_model_binding_action_conflict_allows_shared_query_action_in_tool_name():
    plan = build_tool_binding_plan(
        "Can you tell me the weather in Seoul, South Korea using Celsius units? "
        "Also, turn on the air conditioner in the living room.",
        [
            {
                "name": "OpenWeatherMap.get_current_weather",
                "description": "Fetches the current weather information for a specified location.",
                "parameters": {
                    "type": "dict",
                    "required": ["location"],
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location for current weather.",
                            "enum": ["Seoul, South Korea"],
                        },
                        "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
                    },
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get weather for Seoul in Celsius.",
                "call_groups": [{"intent": "get weather", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "OpenWeatherMap.get_current_weather",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"location": "Seoul, South Korea", "units": "metric"},
                                "evidence_spans": {"location": "Seoul, South Korea", "units": "Celsius units"},
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert plan["calls"][0]["tool_name"] == "OpenWeatherMap.get_current_weather"


def test_model_binding_accepts_turn_by_degrees_as_rotate_action():
    plan = build_tool_binding_plan(
        "turn it by 20 degree and flip it horizontally",
        [
            {
                "name": "rotateImageAction",
                "description": "Rotate an image by the requested number of degrees.",
                "parameters": {
                    "type": "dict",
                    "properties": {"degrees": {"type": "integer"}},
                    "required": ["degrees"],
                },
            },
            {
                "name": "flipImageAction",
                "description": "Flip or mirror an image horizontally or vertically.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "flip_direction": {
                            "type": "string",
                            "enum": ["horizontal", "vertical"],
                        }
                    },
                    "required": ["flip_direction"],
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Rotate image by 20 degrees and flip it horizontally.",
                "call_groups": [
                    {"intent": "rotate image", "expected_call_count": 1},
                    {"intent": "flip image", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "rotateImageAction",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"degrees": 20}, "evidence_spans": {"degrees": "20 degree"}}
                        ],
                    },
                    {
                        "tool_name": "flipImageAction",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"flip_direction": "horizontal"},
                                "evidence_spans": {"flip_direction": "horizontally"},
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["rotateImageAction", "flipImageAction"]


def test_model_binding_accepts_steep_curve_as_derivative_action():
    plan = build_tool_binding_plan(
        "Find the highest common factor for 36 and 48, and then tell me how steep the curve of the function f(x) = x^2 is at x = 5?",
        [
            {
                "name": "math_gcd",
                "description": "Find the greatest common divisor of two numbers.",
                "parameters": {
                    "type": "dict",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "estimate_derivative",
                "description": "Estimate the derivative or slope of a function at a point.",
                "parameters": {
                    "type": "dict",
                    "properties": {"function": {"type": "string"}, "x": {"type": "number"}},
                    "required": ["function", "x"],
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate GCD and estimate derivative of x^2 at x=5.",
                "call_groups": [
                    {"intent": "calculate GCD", "expected_call_count": 1},
                    {"intent": "estimate derivative", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "math_gcd",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"a": 36, "b": 48}, "evidence_spans": {"a": "36", "b": "48"}}
                        ],
                    },
                    {
                        "tool_name": "estimate_derivative",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"function": "x**2", "x": 5},
                                "evidence_spans": {"function": "f(x) = x^2", "x": "x = 5"},
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["math_gcd", "estimate_derivative"]


def test_tool_binding_expands_scalar_entity_enum_cross_product():
    plan = build_tool_binding_plan(
        "I need all pending and active mandates of users parath and bhanu",
        [
            {
                "name": "user.mandates",
                "description": (
                    "Fetches the mandates associated with a user based on the provided user name "
                    "and the status of the mandates."
                ),
                "parameters": {
                    "type": "dict",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The full name of the partner's client for whom to fetch mandates.",
                        },
                        "status": {
                            "type": "string",
                            "description": "The status of the mandates to be fetched.",
                            "enum": ["active", "pending", "inactive"],
                            "default": "active",
                        },
                    },
                },
            },
            {
                "name": "partner.mandates",
                "description": "Fetches the mandates associated with a partner based on the specified status.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "pending", "inactive"],
                            "default": "all",
                        }
                    },
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["user.mandates"] * 4
    assert [call["arguments"] for call in plan["calls"]] == [
        {"name": "parath", "status": "active"},
        {"name": "parath", "status": "pending"},
        {"name": "bhanu", "status": "active"},
        {"name": "bhanu", "status": "pending"},
    ]


def test_location_units_do_not_treat_language_as_location_between_places():
    locations = _location_units(
        "Could you provide me with the latest news report for Paris, France, in English and also for "
        "Letterkenny, Ireland, focusing on technology news again in English?"
    )

    assert locations == ["Paris, France", "Letterkenny, Ireland"]


def test_model_binding_expands_repeated_array_slot_cross_product():
    plan = build_tool_binding_plan(
        "Find details of lawsuits with case numbers '67813', '71249' filed in the New York District court "
        "for type 'Civil' and 'Criminal' cases.",
        [
            {
                "name": "court_case.find",
                "description": "Locate details of court cases based on specific parameters like case number and case type.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city and court where the lawsuit is filed."},
                        "case_number": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The unique case numbers of the lawsuits.",
                        },
                        "case_type": {
                            "type": "string",
                            "enum": ["Civil", "Criminal"],
                            "description": "Type of the court case.",
                            "default": "Civil",
                        },
                    },
                    "required": ["location", "case_number"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Find court cases for two case numbers and two case types.",
                "call_groups": [
                    {"intent": "find lawsuit details", "expected_call_count": 1},
                    {"intent": "find lawsuit details", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "court_case.find",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "location": "New York District court",
                                    "case_number": ["67813", "71249"],
                                    "case_type": "Civil",
                                },
                                "evidence_spans": {
                                    "location": "New York District court",
                                    "case_number": "67813', '71249",
                                    "case_type": "Civil",
                                },
                            }
                        ],
                    },
                    {
                        "tool_name": "court_case.find",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "location": "New York District court",
                                    "case_number": ["67813", "71249"],
                                    "case_type": "Criminal",
                                },
                                "evidence_spans": {
                                    "location": "New York District court",
                                    "case_number": "67813', '71249",
                                    "case_type": "Criminal",
                                },
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["court_case.find"] * 4
    assert [call["arguments"]["case_number"] for call in plan["calls"]] == [
        ["67813"],
        ["71249"],
        ["67813"],
        ["71249"],
    ]


def test_model_binding_accepts_direct_argument_group_objects():
    plan = build_tool_binding_plan(
        "Create two histograms: the first data set is [1, 2, 3] and the second data set is [4, 5, 6], with 5 bins each.",
        [
            {
                "name": "create_histogram",
                "description": "Create a histogram for a numeric data set.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "data": {"type": "array", "items": {"type": "integer"}},
                        "bins": {"type": "integer"},
                    },
                    "required": ["data", "bins"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Create two histograms with 5 bins.",
                "call_groups": [{"name": "create_histogram", "expected_call_count": 2}],
                "tool_bindings": [
                    {
                        "tool_name": "create_histogram",
                        "call_count": 2,
                        "argument_groups": [
                            {
                                "data": [1, 2, 3],
                                "bins": 5,
                                "evidence_spans": {"data": "[1, 2, 3]", "bins": "5 bins each"},
                            },
                            {
                                "data": [4, 5, 6],
                                "bins": 5,
                                "evidence_spans": {"data": "[4, 5, 6]", "bins": "5 bins each"},
                            },
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["arguments"]["data"] for call in plan["calls"]] == [[1, 2, 3], [4, 5, 6]]


def test_model_binding_expands_single_call_with_paired_array_slots():
    plan = build_tool_binding_plan(
        "Find two movie theatres near San Diego with availability for Tenet at 5 pm and "
        "No Time To Die at 7:30 pm.",
        [
            {
                "name": "find_movie_showing",
                "description": "Find local movie theatres and their schedule for a specific movie.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Diego, CA"},
                        "movie": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["Tenet", "No Time To Die"]},
                            "description": "Preferred movie to watch.",
                        },
                        "time": {
                            "type": "array",
                            "items": {"type": "string", "description": "Show time for each movie"},
                        },
                    },
                    "required": ["location", "movie", "time"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Find movie theatres for two movie/time pairs.",
                "call_groups": [{"intent": "find movie theatres", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "find_movie_showing",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {
                                    "location": "San Diego, CA",
                                    "movie": ["Tenet", "No Time To Die"],
                                    "time": ["5 pm", "7:30 pm"],
                                },
                                "evidence_spans": {
                                    "location": "San Diego",
                                    "movie": "Tenet at 5 pm and No Time To Die",
                                    "time": "5 pm and No Time To Die at 7:30 pm",
                                },
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == ["find_movie_showing", "find_movie_showing"]
    assert [(call["arguments"]["movie"], call["arguments"]["time"]) for call in plan["calls"]] == [
        (["Tenet"], ["5 pm"]),
        (["No Time To Die"], ["7:30 pm"]),
    ]


def test_model_binding_keeps_multi_tool_calls_when_per_tool_counts_agree():
    prompt = (
        "Give me the population size of tigers in Bangladesh and India for the last 5 years. "
        "Also provide the projected population size of tigers in Nepal and Malaysia for the next 10 years."
    )
    plan = build_tool_binding_plan(
        prompt,
        [
            {
                "name": "animal_population.get_history",
                "description": "Get historical animal population size by country and species.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "country": {"type": "string"},
                        "species": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["country", "species", "years"],
                },
            },
            {
                "name": "animal_population.get_projection",
                "description": "Get projected animal population size by country and species.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "country": {"type": "string"},
                        "species": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["country", "species", "years"],
                },
            },
            {
                "name": "crop_yield.get_history",
                "description": "Get historical crop yield by country and crop.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "country": {"type": "string"},
                        "crop": {"type": "string"},
                        "years": {"type": "integer"},
                    },
                    "required": ["country", "crop", "years"],
                },
            },
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get tiger population history and projections for four country/time groups.",
                "call_groups": [
                    {"intent": "get tiger population history", "expected_call_count": 1},
                    {"intent": "get tiger population history", "expected_call_count": 1},
                    {"intent": "get tiger population projection", "expected_call_count": 1},
                ],
                "tool_bindings": [
                    {
                        "tool_name": "animal_population.get_history",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"country": "Bangladesh", "species": "tiger", "years": 5},
                                "evidence_spans": {
                                    "country": "Bangladesh",
                                    "species": "tigers",
                                    "years": "last 5 years",
                                },
                            }
                        ],
                    },
                    {
                        "tool_name": "animal_population.get_history",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"country": "India", "species": "tiger", "years": 5},
                                "evidence_spans": {
                                    "country": "India",
                                    "species": "tigers",
                                    "years": "last 5 years",
                                },
                            }
                        ],
                    },
                    {
                        "tool_name": "animal_population.get_projection",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"country": "Nepa", "species": "tiger", "years": 10},
                                "evidence_spans": {
                                    "country": "Nepa",
                                    "species": "tigers",
                                    "years": "next 10 years",
                                },
                            }
                        ],
                    },
                    {
                        "tool_name": "animal_population.get_projection",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"country": "Malaysia", "species": "tiger", "years": 10},
                                "evidence_spans": {
                                    "country": "Malaysia",
                                    "species": "tigers",
                                    "years": "next 10 years",
                                },
                            }
                        ],
                    },
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is True
    assert [call["tool_name"] for call in plan["calls"]] == [
        "animal_population.get_history",
        "animal_population.get_history",
        "animal_population.get_projection",
        "animal_population.get_projection",
    ]
    assert plan["calls"][2]["arguments"]["country"] == "Nepal"


def test_tool_binding_rejects_model_binding_call_count_disagreement():
    plan = build_tool_binding_plan(
        "Could you tell me the current temperature in Boston, MA and San Francisco, CA?",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for one city.",
                "parameters": {
                    "type": "dict",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Get weather for Boston and San Francisco.",
                "call_groups": [{"intent": "get current weather", "expected_call_count": 2}],
                "tool_bindings": [
                    {
                        "tool_name": "get_current_weather",
                        "call_count": 1,
                        "argument_groups": [
                            {"arguments": {"city": "Boston, MA"}, "evidence_spans": {"city": "Boston, MA"}}
                        ],
                    }
                ],
            }
        },
    )

    assert plan["model_tool_binding"]["accepted"] is False
    assert any(
        item["code"] == "call_count_disagrees_with_semantic_frame"
        for item in plan["model_tool_binding"]["diagnostics"]
    )
    assert [call["tool_name"] for call in plan["calls"]] == [
        "get_current_weather",
        "get_current_weather",
    ]


def test_tool_binding_repeats_population_calls_for_mixed_city_state_country_list():
    plan = build_tool_binding_plan(
        "Fetch the population of New York City, NY, and Los Angeles, CA from US Census Database, "
        "and also get the population data for Alaska state and USA",
        [
            {
                "name": "database_us_census.get_population",
                "description": "Fetch population data from US Census database.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "area": {"type": "string", "description": "Name of the city, state, or country."},
                        "type": {"type": "string", "description": "Specify whether the area is city/state/country."},
                        "year": {"type": "integer", "default": 2000},
                    },
                    "required": ["area", "type"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["database_us_census.get_population"] * 4


def test_tool_binding_repeats_for_color_names_despite_format_array_slot():
    plan = build_tool_binding_plan(
        "What are the RGB and HEX color values for navy, purple and maroon?",
        [
            {
                "name": "color_converter.get_color_info",
                "description": "Retrieve RGB values and hexadecimal codes of a specific color.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "color_name": {"type": "string", "description": "The name of the color."},
                        "conversion_type": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["RGB", "HEX"]},
                            "description": "The conversion type for the color.",
                        },
                    },
                    "required": ["color_name", "conversion_type"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["color_converter.get_color_info"] * 3


def test_tool_binding_cross_products_genres_and_locations_for_concert_search():
    plan = build_tool_binding_plan(
        "Can you find me any upcoming rock and jazz concerts for the next month in "
        "San Francisco, California and New York, New York?",
        [
            {
                "name": "concert_finder",
                "description": "Locate upcoming concerts based on music genre in specified city and state.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state to find concerts."},
                        "music_genre": {"type": "string", "description": "Music genre of the concerts."},
                        "time_period": {"type": "integer", "default": 30},
                    },
                    "required": ["location", "music_genre"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["concert_finder"] * 4


def test_tool_binding_allows_repeated_what_is_clauses_for_multiple_tools():
    plan = build_tool_binding_plan(
        "What is the capital city of Australia, what is the current population of Canada, "
        "and what is the largest city in Brazil?",
        [
            {
                "name": "country_info.largest_city",
                "description": "Fetch the largest city of a specified country.",
                "parameters": {
                    "type": "dict",
                    "properties": {"country": {"type": "string", "description": "Name of the country."}},
                    "required": ["country"],
                },
            },
            {
                "name": "country_info.population",
                "description": "Fetch the current population of a specified country.",
                "parameters": {
                    "type": "dict",
                    "properties": {"country": {"type": "string", "description": "Name of the country."}},
                    "required": ["country"],
                },
            },
            {
                "name": "country_info.capital",
                "description": "Fetch the capital city of a specified country.",
                "parameters": {
                    "type": "dict",
                    "properties": {"country": {"type": "string", "description": "Name of the country."}},
                    "required": ["country"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "country_info.capital",
        "country_info.population",
        "country_info.largest_city",
    ]


def test_tool_binding_splits_numbered_function_list():
    prompt = (
        "John received grades {'Math': 85, 'English': 90, 'Science': 88, 'History': 92, 'Art': 89}. "
        "1) Calculate the average grade using the calculate_average function. "
        "2) Calculate the standard deviation using the calculate_standard_deviation function. "
        "3) Identify the subject in which John scored the highest using the highest_grade function."
    )
    plan = build_tool_binding_plan(
        prompt,
        [
            {
                "name": "highest_grade",
                "description": "Identify the subject with the highest grade.",
                "parameters": {
                    "type": "dict",
                    "properties": {"gradeDict": {"type": "object"}},
                    "required": ["gradeDict"],
                },
            },
            {
                "name": "calculate_average",
                "description": "Calculate average grade.",
                "parameters": {
                    "type": "dict",
                    "properties": {"gradeDict": {"type": "object"}},
                    "required": ["gradeDict"],
                },
            },
            {
                "name": "calculate_standard_deviation",
                "description": "Calculate standard deviation of grades.",
                "parameters": {
                    "type": "dict",
                    "properties": {"gradeDict": {"type": "object"}},
                    "required": ["gradeDict"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "calculate_average",
        "calculate_standard_deviation",
        "highest_grade",
    ]


def test_tool_binding_binds_real_estate_property_budget_and_location():
    plan = build_tool_binding_plan(
        "Find a 3 bedroom villa for sale within $300,000 to $400,000 budget in San Diego.",
        [
            {
                "name": "realestate.find_properties",
                "description": "Find properties based on location, budget, and specifications",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "City and state where the property is located."},
                        "propertyType": {"type": "string", "description": "Type of property such as villa, condo, apartment."},
                        "bedrooms": {"type": "integer", "description": "Number of bedrooms required in the property."},
                        "budget": {
                            "type": "dict",
                            "properties": {
                                "min": {"type": "float", "description": "Minimum budget limit."},
                                "max": {"type": "float", "description": "Maximum budget limit."},
                            },
                        },
                    },
                    "required": ["location", "propertyType", "bedrooms", "budget"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["realestate.find_properties"]
    assert plan["calls"][0]["arguments"] == {
        "location": "San Diego",
        "propertyType": "villa",
        "bedrooms": 3,
        "budget": {"min": 300000, "max": 400000},
    }


def test_tool_binding_binds_next_month_to_month_slot():
    plan = build_tool_binding_plan(
        "Predict the house prices for the next month in New York.",
        [
            {
                "name": "house_price_forecast",
                "description": "Predict the house prices for a specific location and time frame.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city for the prediction."},
                        "months": {"type": "integer", "description": "Number of future months for the prediction."},
                    },
                    "required": ["location", "months"],
                },
            },
            {
                "name": "stock_market_forecast",
                "description": "Predict stock prices for a company and time frame.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company": {"type": "string", "description": "The company."},
                        "days": {"type": "integer", "description": "Number of future days."},
                    },
                    "required": ["company", "days"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["house_price_forecast"]
    assert plan["calls"][0]["arguments"] == {"location": "New York", "months": 1}


def test_tool_binding_repeats_location_criteria_requests():
    plan = build_tool_binding_plan(
        "Find a supermarket in New York City that opens 24 hours and another one in San Diego that offers home delivery.",
        [
            {
                "name": "grocery_store.find_by_criteria",
                "description": "Find grocery stores based on location, hours of operation, or services.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The city where you want to find a grocery store."},
                        "criteria": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["24 hours", "Home Delivery", "In-store Pickup"]},
                        },
                    },
                    "required": ["location", "criteria"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["grocery_store.find_by_criteria"] * 2
    assert [call["arguments"]["location"] for call in plan["calls"]] == ["New York City", "San Diego"]


def test_tool_binding_repeats_currency_conversion_scenarios():
    plan = build_tool_binding_plan(
        "How much will it cost in dollars if I transfer 15000 Euro to dollars? "
        "and how much if I convert 200 pounds to dollars?",
        [
            {
                "name": "get_conversion_cost",
                "description": "Convert a value from one currency to another including conversion charges.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "amount": {"type": "integer", "description": "The amount of money to be converted."},
                        "from_currency": {"type": "string", "description": "The current currency of the amount."},
                        "to_currency": {"type": "string", "description": "The target currency."},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_conversion_cost"] * 2
    assert [call["arguments"] for call in plan["calls"]] == [
        {"amount": 15000, "from_currency": "EUR", "to_currency": "USD"},
        {"amount": 200, "from_currency": "GBP", "to_currency": "USD"},
    ]


def test_tool_binding_preserves_same_tool_tuple_repeats_despite_explicit_mentions():
    plan = build_tool_binding_plan(
        "Calculate the Euclidean norm from the origin to the point (3, 4) using math.hypot, "
        "then calculate the norm to (6, 8) using the same function. Also calculate the norm to (9, 12, 15) using math.hypot.",
        [
            {
                "name": "math.hypot",
                "description": "Calculate the Euclidean norm.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "x": {"type": "integer", "description": "The x-coordinate value."},
                        "y": {"type": "integer", "description": "The y-coordinate value."},
                        "z": {"type": "integer", "description": "Optional. Default is 0.", "default": 0},
                    },
                    "required": ["x", "y"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["math.hypot"] * 3
    assert [call["arguments"]["x"] for call in plan["calls"]] == [3, 6, 9]


def test_tool_binding_counts_derivative_actions_not_referential_mentions():
    plan = build_tool_binding_plan(
        "Could you calculate the derivative of the polynomial function '3x^3 - 2x^2 + 5x - 7' "
        "and then evaluate this derivative at x=4? After that, could you also calculate the "
        "derivative of the resulting function and evaluate it at x=2?",
        [
            {
                "name": "calculate_derivative",
                "description": "Calculate the derivative of a polynomial function.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "function": {"type": "string", "description": "The polynomial function."},
                        "x_value": {
                            "type": "integer",
                            "description": (
                                "The x-value at which the derivative is calculated. Optional, if not given, "
                                "the function will return a function of the derivative instead of a specific value. default is 0."
                            ),
                        },
                    },
                    "required": ["function"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_derivative", "calculate_derivative"]


def test_tool_binding_repeats_quadratic_coefficient_groups():
    plan = build_tool_binding_plan(
        "Find the roots of two quadratic equations. The first equation is 3x^2 + 4x + 2 = 0. "
        "The second equation is 5x^2 - 7x + 3 = 0.",
        [
            {
                "name": "algebra.quadratic_roots",
                "description": "Find the roots of a quadratic equation ax^2 + bx + c = 0.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "integer", "description": "Coefficient of x^2."},
                        "b": {"type": "integer", "description": "Coefficient of x."},
                        "c": {"type": "integer", "description": "Constant term."},
                    },
                    "required": ["a", "b", "c"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["algebra.quadratic_roots"] * 2
    assert [call["arguments"] for call in plan["calls"]] == [
        {"a": 3, "b": 4, "c": 2},
        {"a": 5, "b": -7, "c": 3},
    ]


def test_tool_binding_preserves_company_names_with_internal_and():
    plan = build_tool_binding_plan(
        "Find how many cases and the judge handling a specific lawsuit for Pacific Gas and Electric and Tesla Inc.",
        [
            {
                "name": "lawsuit.fetch_details",
                "description": "Fetch the details of a lawsuit for a specific company.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company_name": {"type": "string", "description": "The company involved in the lawsuit."}
                    },
                    "required": ["company_name"],
                },
            },
            {
                "name": "lawsuit.judge",
                "description": "Fetch the judge handling a lawsuit for a specific company.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "company_name": {"type": "string", "description": "The company involved in the lawsuit."},
                        "lawsuit_id": {"type": "integer", "description": "Default to 123", "default": 123},
                    },
                    "required": ["company_name"],
                },
            },
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == [
        "lawsuit.fetch_details",
        "lawsuit.judge",
        "lawsuit.fetch_details",
        "lawsuit.judge",
    ]
    assert [call["arguments"]["company_name"] for call in plan["calls"]] == [
        "Pacific Gas and Electric",
        "Pacific Gas and Electric",
        "Tesla Inc",
        "Tesla Inc",
    ]


def test_tool_binding_repairs_spanish_comma_paired_weather_locations():
    plan = build_tool_binding_plan(
        "Dame el clima actual en Cancun, QR, Playa del Carmen, QR y Tulum, QR.",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit.",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        capability_plan={},
    )

    assert [call["arguments"]["location"] for call in plan["calls"]] == [
        "Cancun, QR",
        "Playa del Carmen, QR",
        "Tulum, QR",
    ]


def test_tool_binding_normalizes_cjk_weather_locations_when_schema_requires_country():
    plan = build_tool_binding_plan(
        "查询中国广州市、北京市和上海现在的天气。",
        [
            {
                "name": "get_current_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, e.g. Paris, France.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit.",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        capability_plan={},
    )

    assert [call["arguments"]["location"] for call in plan["calls"]] == [
        "Guangzhou, China",
        "Beijing, China",
        "Shanghai, China",
    ]


def test_tool_binding_expands_repeated_hotel_reservations_by_scenario():
    plan = build_tool_binding_plan(
        "Book the Hilton Hotel in New York for 2 adults and a child checking in March 10 and checking out March 12. "
        "Then book Marriott in Los Angeles with 1 adult checking in March 14 and checking out March 15. "
        "This year is 2023.",
        [
            {
                "name": "book_hotel",
                "description": "Book a hotel reservation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hotel_name": {"type": "string", "description": "Hotel name"},
                        "location": {"type": "string", "description": "City, state short form"},
                        "check_in": {"type": "string", "description": "Check in date in YYYY-MM-DD"},
                        "check_out": {"type": "string", "description": "Check out date in YYYY-MM-DD"},
                        "adults": {"type": "integer", "description": "number of adults"},
                        "children": {"type": "integer", "description": "number of children"},
                    },
                    "required": ["hotel_name", "location", "check_in", "check_out", "adults"],
                },
            }
        ],
        capability_plan={},
    )

    assert [call["arguments"] for call in plan["calls"]] == [
        {
            "hotel_name": "Hilton Hotel",
            "location": "New York, NY",
            "check_in": "2023-03-10",
            "check_out": "2023-03-12",
            "adults": 2,
            "children": 1,
        },
        {
            "hotel_name": "Marriott",
            "location": "Los Angeles, CA",
            "check_in": "2023-03-14",
            "check_out": "2023-03-15",
            "adults": 1,
            "children": 0,
        },
    ]


def test_tool_binding_does_not_treat_numeric_operands_as_locations():
    plan = build_tool_binding_plan(
        "What is the greatest common divisor for 36 and 48?",
        [
            {
                "name": "math_gcd",
                "description": "Compute greatest common divisor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "larger number"},
                        "b": {"type": "integer", "description": "second number"},
                    },
                    "required": ["a", "b"],
                },
            }
        ],
        capability_plan={},
    )

    assert plan["calls"][0]["arguments"] == {"a": 48, "b": 36}


def test_tool_binding_cross_products_entities_with_numeric_condition_slot():
    plan = build_tool_binding_plan(
        "What are the boiling and melting points of water and iron at sea levels of 0 meters and 1000 meters respectively?",
        [
            {
                "name": "get_boiling_melting_points",
                "description": "Retrieve the boiling point and melting point of a substance based on its name and the sea level.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "substance": {"type": "string", "description": "The name of the substance."},
                        "sea_level": {"type": "integer", "description": "The sea level in meters."},
                    },
                    "required": ["substance", "sea_level"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["get_boiling_melting_points"] * 4
    assert {tuple(call["arguments"].items()) for call in plan["calls"]} == {
        tuple({"substance": "water", "sea_level": 0}.items()),
        tuple({"substance": "water", "sea_level": 1000}.items()),
        tuple({"substance": "iron", "sea_level": 0}.items()),
        tuple({"substance": "iron", "sea_level": 1000}.items()),
    }


def test_tool_binding_expands_positive_and_negative_boolean_counterfactual():
    plan = build_tool_binding_plan(
        "The process starts at an initial temperature of 300 Kelvin and ends at a final temperature of 350 Kelvin. "
        "The heat capacity of the system is 4.18 J/K. The process is isothermal. "
        "Can you calculate the entropy change for this process? What if the process is not isothermal?",
        [
            {
                "name": "calculate_entropy_change",
                "description": "Calculate the entropy change for an isothermal and reversible process.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "initial_temp": {"type": "integer", "description": "The initial temperature in Kelvin."},
                        "final_temp": {"type": "integer", "description": "The final temperature in Kelvin."},
                        "heat_capacity": {"type": "float", "description": "The heat capacity in J/K."},
                        "isothermal": {"type": "boolean", "description": "Whether the process is isothermal."},
                    },
                    "required": ["initial_temp", "final_temp", "heat_capacity"],
                },
            }
        ],
    )

    assert [call["tool_name"] for call in plan["calls"]] == ["calculate_entropy_change", "calculate_entropy_change"]
    assert [call["arguments"]["isothermal"] for call in plan["calls"]] == [True, False]
    assert all(call["arguments"]["initial_temp"] == 300 for call in plan["calls"])


def test_tool_binding_eval_scores_exact_match():
    gold = [
        {
            "id": "case_1",
            "expected_tool_binding": {
                "tool_decision": "call",
                "calls": [
                    {
                        "tool_name": "calculate_triangle_area",
                        "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
                    }
                ],
            },
        }
    ]
    predictions = [
        {
            "id": "case_1",
            "tool_binding_plan": {
                "tool_decision": "call",
                "calls": [
                    {
                        "tool_name": "calculate_triangle_area",
                        "arguments": {"base": 10, "height": 5},
                    }
                ],
            },
        }
    ]

    scored = score_predictions(gold, predictions)

    assert scored["metrics"]["exact_call_match_rate"] == 1.0


def test_tool_binding_rejects_ungrounded_numeric_model_binding_and_weak_duplicate_fallback():
    plan = build_tool_binding_plan(
        "What will be the energy needed to increase the temperature of 3 kg of water by 4 degrees Celsius?",
        [
            {
                "name": "calculate_heat",
                "description": "Calculate heat energy from mass, specific heat, and temperature change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mass": {"type": "number"},
                        "specific_heat": {"type": "number"},
                        "change_in_temp": {"type": "number"},
                    },
                    "required": ["mass", "specific_heat", "change_in_temp"],
                },
            }
        ],
        capability_plan={
            "semantic_input_frame": {
                "canonical_request": "Calculate heat for water.",
                "call_groups": [{"intent": "calculate heat", "expected_call_count": 1}],
                "tool_bindings": [
                    {
                        "tool_name": "calculate_heat",
                        "call_count": 1,
                        "argument_groups": [
                            {
                                "arguments": {"mass": 3, "specific_heat": 4.184, "change_in_temp": 4},
                                "evidence_spans": {
                                    "mass": "3 kg",
                                    "specific_heat": "water",
                                    "change_in_temp": "4 degrees Celsius",
                                },
                            }
                        ],
                    }
                ],
                "missing_inputs": [],
            }
        },
    )

    assert plan["tool_decision"] == "no_tool"
    assert plan["calls"] == []
    assert plan["model_tool_binding"]["accepted"] is False
    assert any(item["code"] == "ungrounded_argument" for item in plan["model_tool_binding"]["diagnostics"])
