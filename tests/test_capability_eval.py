from __future__ import annotations

from taskdecomp.capability_eval import score_case, score_predictions


def prediction(
    run1,
    run2,
    run3,
    run4,
    run5,
    validation=None,
):
    return {
        "passes": {
            "intent_input_audit": {"parsed": run1, "parse_error": None},
            "transformation_externality_audit": {"parsed": run2, "parse_error": None},
            "capability_requirements": {"parsed": run3, "parse_error": None},
            "capability_normalization": {"parsed": run4, "parse_error": None},
            "capability_ordering": {"parsed": run5, "parse_error": None},
        },
        "validation": validation or {"valid": True, "violations": [], "minimal_repairs": []},
    }


def test_score_case_accepts_property_level_good_plan():
    gold = {
        "id": "missing_essay_001",
        "category": "missing_input",
        "request": "Can you make my essay better?",
        "expected": {
            "run1": {
                "final_intent_keywords": ["revise", "essay"],
                "inputs": [{"keyword": "essay", "available": False, "format": "unknown"}],
                "missing_input_keywords": ["essay"],
            },
            "run2": {
                "needs_current_or_external_info": False,
                "must_include_external_action_type": ["user_input"],
                "must_not_include_external_action_type": ["file_reading", "web_search"],
            },
            "run3": {
                "must_include_capability": ["request_missing_input"],
                "must_include_external_action_type": ["user_input"],
            },
            "graph": {},
        },
    }
    pred = prediction(
        {
            "final_user_want": "revise and improve the user's essay",
            "inputs": [
                {
                    "name": "essay text",
                    "needed_for": "revision",
                    "available": False,
                    "format": "unknown",
                    "evidence": "No essay text was provided.",
                }
            ],
            "missing_inputs": ["essay text"],
        },
        {
            "starting_state": "essay text is missing",
            "desired_state": "essay can be revised",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [
                {"action_type": "user_input", "needed": True, "reason": "essay is missing"}
            ],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "capability_description": "Ask the user for the missing essay text.",
                    "requires_external_action": True,
                    "external_action_type": "user_input",
                    "inputs": ["missing essay text"],
                    "outputs": ["essay text"],
                    "done_when": "the essay text has been requested",
                }
            ]
        },
        {
            "normalized_capabilities": [
                {
                    "id": "cap_1",
                    "original_name": "request_missing_input",
                    "normalized_name": "request_missing_input",
                    "meaning_changed": False,
                    "external_action_type": "user_input",
                }
            ],
            "merged_capabilities": [],
        },
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "depends_on": [],
                    "inputs": ["missing essay text"],
                    "outputs": ["essay text"],
                    "done_when": "the essay text has been requested",
                }
            ]
        },
    )

    score = score_case(gold, pred)

    assert score["run1_ok"]
    assert score["run2_ok"]
    assert score["run3_ok"]
    assert score["graph_ok"]
    assert score["auto_plan_acceptable"]


def test_score_case_labels_missed_current_info():
    gold = {
        "id": "current_001",
        "category": "current_facts",
        "request": "latest model?",
        "expected": {
            "run1": {"final_intent_keywords": ["latest", "model"]},
            "run2": {
                "needs_current_or_external_info": True,
                "must_include_external_action_type": ["web_search"],
            },
            "run3": {
                "must_include_capability": ["retrieve_current_information"],
                "must_include_external_action_type": ["web_search"],
            },
            "graph": {},
        },
    }
    pred = prediction(
        {"final_user_want": "answer latest model", "inputs": [], "missing_inputs": []},
        {
            "starting_state": "question is available",
            "desired_state": "answer is available",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [{"action_type": "none", "needed": False, "reason": "none"}],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "answer_question",
                    "capability_description": "Answer from existing knowledge.",
                    "requires_external_action": False,
                    "external_action_type": "none",
                    "inputs": ["question"],
                    "outputs": ["answer"],
                    "done_when": "answer is drafted",
                }
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "answer_question",
                    "depends_on": [],
                    "inputs": ["question"],
                    "outputs": ["answer"],
                    "done_when": "answer is drafted",
                }
            ]
        },
    )

    score = score_case(gold, pred)

    assert not score["run2_ok"]
    assert not score["run3_ok"]
    assert score["main_failure_type"] == "missed_current_info"


def test_score_predictions_summarizes_rates():
    gold = [
        {
            "id": "one",
            "category": "simple",
            "request": "Draft text",
            "expected": {
                "run1": {"final_intent_keywords": ["draft"]},
                "run2": {"must_include_external_action_type": ["none"]},
                "run3": {"must_include_capability": ["draft_text"]},
                "graph": {},
            },
        }
    ]
    pred = prediction(
        {"final_user_want": "draft text", "inputs": [], "missing_inputs": []},
        {
            "starting_state": "request is available",
            "desired_state": "draft is available",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [{"action_type": "none", "needed": False, "reason": "none"}],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "draft_text",
                    "capability_description": "Draft text from requirements.",
                    "requires_external_action": False,
                    "external_action_type": "none",
                    "inputs": ["requirements"],
                    "outputs": ["draft"],
                    "done_when": "draft is ready",
                }
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "draft_text",
                    "depends_on": [],
                    "inputs": ["requirements"],
                    "outputs": ["draft"],
                    "done_when": "draft is ready",
                }
            ]
        },
    )
    pred["id"] = "one"

    result = score_predictions(gold, [pred])

    assert result["metrics"]["case_count"] == 1
    assert result["metrics"]["input_audit_accuracy"] == 1.0
    assert result["metrics"]["auto_capability_plan_accept_rate"] == 1.0


def test_score_case_accepts_semantic_capability_aliases():
    gold = {
        "id": "essay_revision_pasted_001",
        "category": "pasted_essay_revision",
        "request": "Please make this essay clearer",
        "expected": {
            "run1": {
                "final_intent_keywords": ["revise", "essay"],
                "inputs": [{"keyword": "essay", "available": True, "format": "pasted_text"}],
            },
            "run2": {"must_include_external_action_type": ["none"]},
            "run3": {"must_include_capability": ["revise_text_for_clarity"]},
            "graph": {},
        },
    }
    pred = prediction(
        {
            "final_user_want": "Produce a clearer and more polished essay.",
            "inputs": [
                {
                    "name": "essay_text",
                    "needed_for": "polishing",
                    "available": True,
                    "format": "pasted_text",
                    "evidence": "essay text is provided",
                }
            ],
            "missing_inputs": [],
        },
        {
            "starting_state": "essay is available",
            "desired_state": "polished essay is available",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [{"action_type": "none", "needed": False, "reason": "none"}],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "polish_and_clarify_essay",
                    "capability_description": "Improve the essay's clarity.",
                    "requires_external_action": False,
                    "external_action_type": "none",
                    "inputs": ["essay"],
                    "outputs": ["polished essay"],
                    "done_when": "the essay is clearer",
                }
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "polish_and_clarify_essay",
                    "depends_on": [],
                    "inputs": ["essay"],
                    "outputs": ["polished essay"],
                    "done_when": "the essay is clearer",
                }
            ]
        },
    )

    score = score_case(gold, pred)

    assert score["run1_ok"]
    assert score["run3_ok"]


def test_score_case_accepts_make_better_as_revise_intent():
    gold = {
        "id": "missing_essay_001",
        "category": "missing_input",
        "request": "Can you make my essay better?",
        "expected": {
            "run1": {
                "final_intent_keywords": ["revise", "essay"],
                "inputs": [{"keyword": "essay", "available": False, "format": "unknown"}],
                "missing_input_keywords": ["essay"],
            },
            "run2": {"must_include_external_action_type": ["user_input"]},
            "run3": {"must_include_capability": ["request_missing_input"]},
            "graph": {},
        },
    }
    pred = prediction(
        {
            "final_user_want": "Make my essay better.",
            "inputs": [
                {
                    "name": "essay_text",
                    "needed_for": "essay_text",
                    "available": False,
                    "format": "unknown",
                    "evidence": "missing user input",
                }
            ],
            "missing_inputs": ["essay_text"],
        },
        {
            "starting_state": "essay text is missing",
            "desired_state": "improved essay",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [
                {"action_type": "user_input", "needed": True, "reason": "essay missing"}
            ],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "capability_description": "Ask for essay text.",
                    "requires_external_action": True,
                    "external_action_type": "user_input",
                    "inputs": [],
                    "outputs": ["essay_text"],
                    "done_when": "essay text is requested",
                }
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "depends_on": [],
                    "inputs": [],
                    "outputs": ["essay_text"],
                    "done_when": "essay text is requested",
                }
            ]
        },
    )

    score = score_case(gold, pred)

    assert score["run1_ok"]
    assert score["auto_plan_acceptable"]


def test_score_case_splits_combined_run2_action_types():
    gold = {
        "id": "merge_pdfs_001",
        "category": "file_read_write",
        "request": "Merge PDFs",
        "expected": {
            "run1": {"final_intent_keywords": ["merge"]},
            "run2": {
                "must_include_external_action_type": ["file_reading", "file_writing"]
            },
            "run3": {
                "must_include_capability": ["combine_files"],
                "must_include_external_action_type": ["code_execution"],
            },
            "graph": {},
        },
    }
    pred = prediction(
        {"final_user_want": "Merge PDFs", "inputs": [], "missing_inputs": []},
        {
            "starting_state": "PDFs are available",
            "desired_state": "combined PDF is available",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [
                {
                    "action_type": "file_reading | file_writing | code_execution",
                    "needed": True,
                    "reason": "merge PDF files",
                }
            ],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "merge_pdfs",
                    "capability_description": "Combine the PDF files.",
                    "requires_external_action": True,
                    "external_action_type": "code_execution",
                    "inputs": ["PDFs"],
                    "outputs": ["combined PDF"],
                    "done_when": "combined PDF exists",
                }
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "merge_pdfs",
                    "depends_on": [],
                    "inputs": ["PDFs"],
                    "outputs": ["combined PDF"],
                    "done_when": "combined PDF exists",
                }
            ]
        },
    )

    score = score_case(gold, pred)

    assert score["run2_ok"]
    assert score["run3_ok"]


def test_score_case_accepts_calculate_real_world_height_as_measure_step():
    gold = {
        "id": "unsupported_measure_image_001",
        "category": "unsupported_missing_scale",
        "request": "Measure the real-world height of the lamp in this image.",
        "expected": {
            "run1": {
                "final_intent_keywords": ["measure", "height"],
                "inputs": [
                    {"keyword": "image", "available": True, "format": "image"},
                    {"keyword": "scale", "available": False, "format": "unknown"},
                ],
                "missing_input_keywords": ["scale"],
            },
            "run2": {
                "needs_current_or_external_info": False,
                "must_include_external_action_type": ["image_understanding", "user_input"],
            },
            "run3": {
                "must_include_capability": ["request_missing_input"],
                "must_include_external_action_type": ["image_understanding", "user_input"],
            },
            "graph": {
                "must_precede": [{"before": "request_missing_input", "after": "measure"}]
            },
        },
    }
    pred = prediction(
        {
            "final_user_want": "Measure the real-world height of the lamp.",
            "inputs": [
                {
                    "name": "image",
                    "needed_for": "measure lamp height",
                    "available": True,
                    "format": "image",
                    "evidence": "lamp.jpg",
                },
                {
                    "name": "scale or reference measurement",
                    "needed_for": "measure lamp height",
                    "available": False,
                    "format": "unknown",
                    "evidence": "missing scale",
                },
            ],
            "missing_inputs": ["scale or reference measurement"],
        },
        {
            "starting_state": "image with unknown scale",
            "desired_state": "lamp height measurement",
            "transformations_needed": [],
            "needs_current_or_external_info": False,
            "external_actions": [
                {"action_type": "user_input", "needed": True, "reason": "Need scale"},
                {
                    "action_type": "image_understanding",
                    "needed": True,
                    "reason": "Interpret image",
                },
            ],
        },
        {
            "capabilities_needed": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "capability_description": "Ask for scale.",
                    "requires_external_action": True,
                    "external_action_type": "user_input",
                    "inputs": [],
                    "outputs": ["reference_scale"],
                    "done_when": "reference scale is provided",
                },
                {
                    "id": "cap_2",
                    "capability_name": "calculate_real_world_height",
                    "capability_description": "Compute lamp height from image and scale.",
                    "requires_external_action": False,
                    "external_action_type": "none",
                    "inputs": ["image", "reference_scale"],
                    "outputs": ["lamp_height"],
                    "done_when": "lamp height is computed",
                },
                {
                    "id": "cap_3",
                    "capability_name": "interpret_image_content",
                    "capability_description": "Interpret the image.",
                    "requires_external_action": True,
                    "external_action_type": "image_understanding",
                    "inputs": ["image"],
                    "outputs": ["visual facts"],
                    "done_when": "image is interpreted",
                },
            ]
        },
        {"normalized_capabilities": [], "merged_capabilities": []},
        {
            "ordered_capabilities": [
                {
                    "id": "cap_1",
                    "capability_name": "request_missing_input",
                    "depends_on": [],
                    "inputs": [],
                    "outputs": ["reference_scale"],
                    "done_when": "reference scale is provided",
                },
                {
                    "id": "cap_2",
                    "capability_name": "calculate_real_world_height",
                    "depends_on": ["cap_1"],
                    "inputs": ["image", "reference_scale"],
                    "outputs": ["lamp_height"],
                    "done_when": "lamp height is computed",
                },
                {
                    "id": "cap_3",
                    "capability_name": "interpret_image_content",
                    "depends_on": ["cap_1"],
                    "inputs": ["image"],
                    "outputs": ["visual facts"],
                    "done_when": "image is interpreted",
                },
            ]
        },
    )

    score = score_case(gold, pred)

    assert score["graph_ok"]
    assert score["auto_plan_acceptable"]
