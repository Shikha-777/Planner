from __future__ import annotations

from copy import deepcopy

from taskdecomp.capability_planning import (
    apply_missing_input_slot_filler,
    build_evidence_chunks,
    build_missing_input_slot_filler_messages,
    build_messages_for_pass,
    build_one_shot_capability_plan_messages,
    build_rules_first_capability_plan,
    classify_task_family,
    classify_task_route,
    merge_split_intent_input_audit,
    repair_capability_requirements,
    repair_capability_normalization,
    repair_capability_ordering,
    repair_intent_input_audit,
    repair_transformation_audit,
    should_run_missing_input_slot_filler,
    validate_capability_plan,
)


def base_plan():
    intent = {
        "final_user_want": "revise a provided paragraph for clarity",
        "inputs": [
            {
                "name": "paragraph",
                "needed_for": "text revision",
                "available": True,
                "format": "pasted_text",
                "evidence": "The paragraph is in the request.",
            }
        ],
        "missing_inputs": [],
    }
    transform = {
        "starting_state": "draft paragraph is available",
        "desired_state": "clearer paragraph is produced",
        "transformations_needed": [
            {
                "transformation": "revise wording for clarity",
                "input_state": "draft paragraph",
                "output_state": "clear paragraph",
                "reason": "the user asked for a revision",
            }
        ],
        "needs_current_or_external_info": False,
        "external_actions": [{"action_type": "none", "needed": False, "reason": "no outside data"}],
    }
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "revise_text_for_clarity",
                "capability_description": "Transform provided text into clearer wording.",
                "input_state": "draft text is available",
                "output_state": "revised text is available",
                "requires_external_action": False,
                "external_action_type": "none",
                "inputs": ["draft text"],
                "outputs": ["revised text"],
                "done_when": "the paragraph is revised for clarity",
            }
        ]
    }
    normalization = {
        "normalized_capabilities": [
            {
                "id": "cap_1",
                "original_name": "revise_text_for_clarity",
                "normalized_name": "revise_text_for_clarity",
                "meaning_changed": False,
                "external_action_type": "none",
            }
        ],
        "merged_capabilities": [],
    }
    ordering = {
        "ordered_capabilities": [
            {
                "id": "cap_1",
                "capability_name": "revise_text_for_clarity",
                "depends_on": [],
                "inputs": ["draft text"],
                "outputs": ["revised text"],
                "done_when": "the paragraph is revised for clarity",
            }
        ]
    }
    return intent, transform, requirements, normalization, ordering


def validate(parts):
    return validate_capability_plan(*parts)


def violation_types(result):
    return {item["type"] for item in result["violations"]}


def test_valid_capability_plan_passes():
    result = validate(base_plan())
    assert result["valid"]
    assert result["violations"] == []
    assert result["minimal_repairs"] == []


def test_build_evidence_chunks_separates_instruction_and_pasted_text():
    chunks = build_evidence_chunks(
        "Please make this essay clearer:\n\nSchool lunches should be healthier.",
        attachments_metadata=[],
    )

    by_type = {chunk["type"]: chunk for chunk in chunks}
    assert by_type["instruction"]["text"] == "Please make this essay clearer:"
    assert by_type["pasted_text"]["format"] == "pasted_text"
    assert "School lunches" in by_type["pasted_text"]["text"]


def test_build_evidence_chunks_preserves_attachment_formats():
    chunks = build_evidence_chunks(
        "Summarize the attached memo.",
        attachments_metadata=[
            {"name": "policy_memo.pdf", "format": "pdf", "available": True}
        ],
    )

    attachment = next(chunk for chunk in chunks if chunk["type"] == "attachment")
    assert attachment["format"] == "pdf"
    assert attachment["available"] is True


def test_merge_split_intent_input_audit_keeps_expected_schema():
    merged = merge_split_intent_input_audit(
        {"final_user_want": "revise the essay"},
        {"required_inputs": [{"name": "essay text", "needed_for": "revision"}]},
        {
            "inputs": [
                {
                    "name": "essay text",
                    "needed_for": "revision",
                    "available": True,
                    "format": "pasted_text",
                    "source_chunk_ids": ["pasted_text_1"],
                    "evidence": "pasted text chunk",
                }
            ],
            "missing_inputs": [],
        },
    )

    assert merged["final_user_want"] == "revise the essay"
    assert merged["inputs"][0]["format"] == "pasted_text"
    assert merged["missing_inputs"] == []


def test_repair_intent_labels_pasted_essay_input():
    intent = {
        "final_user_want": "make this essay clearer",
        "inputs": [
            {
                "name": "pasted_text",
                "needed_for": "modify_text",
                "available": True,
                "format": "pasted_text",
                "evidence": "School lunches should be healthier.",
            }
        ],
        "missing_inputs": [],
    }

    repaired, repairs = repair_intent_input_audit(
        "Please make this essay clearer:\n\nSchool lunches should be healthier.",
        [],
        intent,
    )

    assert repaired["inputs"][0]["name"] == "essay text"
    assert any(repair["action"] == "edit_input_name" for repair in repairs)


def test_repair_intent_resolves_attachment_marker_to_file_name():
    intent = {
        "final_user_want": "fix parser bug and tests",
        "inputs": [
            {
                "name": "parser_file_path",
                "needed_for": "inspect parser",
                "available": True,
                "format": "file_path",
                "evidence": "attachment_1",
            }
        ],
        "missing_inputs": [],
    }

    repaired, _ = repair_intent_input_audit(
        "In src/parser.py, fix the bug.",
        [{"name": "src/parser.py", "format": "file_path", "available": True}],
        intent,
    )

    assert repaired["inputs"][0]["name"] == "src/parser.py"
    assert repaired["inputs"][0]["evidence"] == "src/parser.py"


def test_repair_intent_adds_unavailable_entries_for_missing_inputs():
    intent = {
        "final_user_want": "make essay better",
        "inputs": [],
        "missing_inputs": [{"name": "essay_text", "format": "unknown"}],
    }

    repaired, repairs = repair_intent_input_audit(
        "Can you make my essay better?",
        [],
        intent,
    )

    assert repaired["inputs"][0]["available"] is False
    assert repaired["inputs"][0]["format"] == "unknown"
    assert "essay" in repaired["inputs"][0]["name"]
    assert any(repair["action"] == "add_missing_input_entry" for repair in repairs)


def test_one_shot_prompt_requests_complete_compatible_plan():
    messages = build_one_shot_capability_plan_messages(
        "Summarize the attached memo.",
        attachments_metadata=[{"name": "memo.pdf", "format": "pdf", "available": True}],
    )
    text = "\n".join(message["content"] for message in messages)

    assert "intent_input_audit" in text
    assert "transformation_externality_audit" in text
    assert "capability_requirements" in text
    assert "capability_ordering" in text
    assert "exactly one external_action_type" in text
    assert "Start with { and end with }" in text
    assert "short lowercase snake_case" in text
    assert "request_missing_input" in text


def test_dependency_must_reference_valid_capability_id():
    parts = list(base_plan())
    ordering = deepcopy(parts[4])
    ordering["ordered_capabilities"][0]["depends_on"] = ["cap_missing"]
    parts[4] = ordering

    result = validate(parts)

    assert not result["valid"]
    assert "unknown_dependency" in violation_types(result)


def test_dependency_cycle_is_rejected():
    parts = list(base_plan())
    requirements = deepcopy(parts[2])
    requirements["capabilities_needed"].append(
        {
            **requirements["capabilities_needed"][0],
            "id": "cap_2",
            "capability_name": "validate_output_against_requirements",
        }
    )
    normalization = deepcopy(parts[3])
    normalization["normalized_capabilities"].append(
        {
            "id": "cap_2",
            "original_name": "validate_output_against_requirements",
            "normalized_name": "validate_output_against_requirements",
            "meaning_changed": False,
            "external_action_type": "none",
        }
    )
    ordering = {
        "ordered_capabilities": [
            {
                "id": "cap_1",
                "capability_name": "revise_text_for_clarity",
                "depends_on": ["cap_2"],
                "inputs": ["draft text"],
                "outputs": ["revised text"],
                "done_when": "the text is revised",
            },
            {
                "id": "cap_2",
                "capability_name": "validate_output_against_requirements",
                "depends_on": ["cap_1"],
                "inputs": ["revised text"],
                "outputs": ["validated text"],
                "done_when": "the revision is checked",
            },
        ]
    }
    parts[2] = requirements
    parts[3] = normalization
    parts[4] = ordering

    result = validate(parts)

    assert not result["valid"]
    assert "dependency_cycle" in violation_types(result)


def test_missing_input_requires_request_missing_input_capability():
    parts = list(base_plan())
    intent = deepcopy(parts[0])
    intent["inputs"][0]["available"] = False
    intent["missing_inputs"] = ["paragraph text"]
    parts[0] = intent

    result = validate(parts)

    assert not result["valid"]
    assert "missing_input_without_request_capability" in violation_types(result)
    assert any(
        repair["patch"].get("capability_name") == "request_missing_input"
        for repair in result["minimal_repairs"]
    )


def test_pasted_text_does_not_need_file_extraction():
    parts = list(base_plan())
    requirements = deepcopy(parts[2])
    requirements["capabilities_needed"][0].update(
        {
            "capability_name": "extract_information_from_attached_document",
            "requires_external_action": True,
            "external_action_type": "file_reading",
        }
    )
    parts[2] = requirements

    result = validate(parts)

    assert not result["valid"]
    assert "unneeded_file_extraction_for_pasted_text" in violation_types(result)


def test_pdf_text_needs_extraction_capability():
    parts = list(base_plan())
    intent = deepcopy(parts[0])
    intent["final_user_want"] = "summarize the attached PDF"
    intent["inputs"][0].update(
        {
            "name": "attached PDF",
            "needed_for": "summarize document contents",
            "format": "pdf",
            "evidence": "A PDF is attached.",
        }
    )
    parts[0] = intent

    result = validate(parts)

    assert not result["valid"]
    assert "missing_pdf_text_extraction" in violation_types(result)


def test_current_information_requires_retrieval_capability():
    parts = list(base_plan())
    transform = deepcopy(parts[1])
    transform["needs_current_or_external_info"] = True
    transform["external_actions"] = [
        {"action_type": "web_search", "needed": True, "reason": "current price is needed"}
    ]
    parts[1] = transform

    result = validate(parts)

    assert not result["valid"]
    assert "missing_current_information_capability" in violation_types(result)


def test_non_web_external_action_must_be_explicit_in_capability():
    parts = list(base_plan())
    transform = deepcopy(parts[1])
    transform["external_actions"] = [
        {"action_type": "calculation", "needed": True, "reason": "numeric total is needed"}
    ]
    parts[1] = transform

    result = validate(parts)

    assert not result["valid"]
    assert "missing_capability_external_action_type" in violation_types(result)


def test_tool_or_agent_capability_name_is_rejected():
    parts = list(base_plan())
    requirements = deepcopy(parts[2])
    requirements["capabilities_needed"][0]["capability_name"] = "research_agent"
    parts[2] = requirements

    result = validate(parts)

    assert not result["valid"]
    assert "tool_or_worker_capability" in violation_types(result)


def test_capability_requirement_prompt_forbids_tool_selection():
    previous = {
        "intent_input_audit": base_plan()[0],
        "transformation_externality_audit": base_plan()[1],
    }

    messages = build_messages_for_pass(
        "capability_requirements",
        user_request="Summarize this article",
        previous=previous,
    )
    prompt_text = "\n".join(message["content"] for message in messages)

    assert "Do not choose concrete tools" in prompt_text
    assert "external_action_type" in prompt_text
    assert "state transformation" in prompt_text


def test_repair_adds_missing_input_capability_without_mutating_original():
    intent, transform, requirements, _, _ = base_plan()
    intent = deepcopy(intent)
    intent["inputs"][0]["available"] = False
    intent["missing_inputs"] = ["paragraph text"]
    requirements = {"capabilities_needed": []}

    repaired, repairs = repair_capability_requirements(intent, transform, requirements)

    caps = repaired["capabilities_needed"]
    assert requirements["capabilities_needed"] == []
    assert any(cap["capability_name"] == "request_missing_input" for cap in caps)
    assert any(repair["action"] == "add_capability" for repair in repairs)


def test_repair_splits_combined_external_action_and_adds_missing_actions():
    intent, transform, requirements, _, _ = base_plan()
    transform = deepcopy(transform)
    transform["external_actions"] = [
        {
            "action_type": "file_reading | file_writing | code_execution",
            "needed": True,
            "reason": "read, edit, and test project files",
        }
    ]
    intent["inputs"][0].update({"format": "file_path", "name": "src/parser.py"})
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "repair_project_code",
                "capability_description": "Fix the project code.",
                "requires_external_action": True,
                "external_action_type": "file_reading | file_writing | code_execution",
                "inputs": ["project files"],
                "outputs": ["updated project"],
                "done_when": "the project is fixed",
            }
        ]
    }

    repaired, _ = repair_capability_requirements(intent, transform, requirements)
    actions = {
        cap["external_action_type"]
        for cap in repaired["capabilities_needed"]
        if isinstance(cap, dict)
    }

    assert {"file_reading", "file_writing", "code_execution"} <= actions


def test_repair_does_not_expand_schema_placeholder_action_enum():
    intent, transform, requirements, _, _ = base_plan()
    transform = deepcopy(transform)
    transform["needs_current_or_external_info"] = True
    transform["external_actions"] = [
        {"action_type": "web_search", "needed": True, "reason": "latest facts are needed"}
    ]
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "...",
                "capability_description": "...",
                "requires_external_action": True,
                "external_action_type": (
                    "none | file_reading | file_writing | web_search | fact_checking | "
                    "calculation | code_execution | image_understanding | image_generation | "
                    "user_input | other"
                ),
                "inputs": [],
                "outputs": [],
                "done_when": "...",
            }
        ]
    }

    repaired, _ = repair_capability_requirements(intent, transform, requirements)
    actions = {
        cap["external_action_type"]
        for cap in repaired["capabilities_needed"]
        if isinstance(cap, dict)
    }

    assert actions == {"web_search", "none"}


def test_repair_adds_current_information_capability():
    intent, transform, requirements, _, _ = base_plan()
    transform = deepcopy(transform)
    transform["needs_current_or_external_info"] = True
    transform["external_actions"] = [
        {"action_type": "web_search", "needed": True, "reason": "latest facts are needed"}
    ]
    requirements = {"capabilities_needed": []}

    repaired, _ = repair_capability_requirements(intent, transform, requirements)

    assert any(
        cap["capability_name"] == "retrieve_current_information"
        for cap in repaired["capabilities_needed"]
    )


def test_repair_intent_marks_current_fact_query_available():
    audit = {
        "final_user_want": "identify the latest OpenAI API model and best use",
        "inputs": [
            {
                "name": "OpenAI model query",
                "needed_for": "answer current model question",
                "available": False,
                "format": "none",
                "evidence": "latest OpenAI model",
            }
        ],
        "missing_inputs": ["OpenAI model query"],
    }

    repaired, repairs = repair_intent_input_audit(
        "What is the latest OpenAI model available in the API?",
        [],
        audit,
    )

    assert repaired["inputs"][0]["available"] is True
    assert repaired["inputs"][0]["format"] == "none"
    assert repaired["missing_inputs"] == []
    assert any(repair["action"] == "edit_input_availability" for repair in repairs)


def test_repair_intent_adds_pasted_text_when_split_pass_omits_inputs():
    repaired, _ = repair_intent_input_audit(
        "Please make this essay clearer:\n\nSchool lunches should be healthier.",
        [],
        {
            "final_user_want": "polish and clarify the essay",
            "inputs": [],
            "missing_inputs": [],
        },
    )

    assert repaired["inputs"][0]["name"] == "essay text"
    assert repaired["inputs"][0]["format"] == "pasted_text"


def test_repair_intent_preserves_compute_numeric_for_calculation():
    repaired, _ = repair_intent_input_audit(
        "What is 1 + 2 + log base 3 of 90?",
        [],
        {
            "final_user_want": "Calculate 1 + 2 + log base 3 of 90.",
            "inputs": [],
            "missing_inputs": [],
        },
    )

    assert "compute" in repaired["final_user_want"]
    assert "numeric" in repaired["final_user_want"]


def test_repair_intent_adds_missing_scale_for_image_measurement():
    audit = {
        "final_user_want": "measure the real-world height of the lamp",
        "inputs": [
            {
                "name": "lamp.jpg",
                "needed_for": "measurement",
                "available": True,
                "format": "image",
                "evidence": "attached image",
            }
        ],
        "missing_inputs": [],
    }

    repaired, _ = repair_intent_input_audit(
        "Measure the real-world height of the lamp in this image.",
        [{"name": "lamp.jpg", "format": "image", "available": True}],
        audit,
    )

    assert any(
        item.get("available") is False and "scale" in item.get("name", "")
        for item in repaired["inputs"]
    )


def test_repair_transformation_adds_code_execution_for_test_tasks():
    intent, transform, _, _, _ = base_plan()
    intent = deepcopy(intent)
    intent["final_user_want"] = "fix parser bug and update tests"
    intent["inputs"][0]["format"] = "file_path"
    transform = deepcopy(transform)
    transform["external_actions"] = [
        {"action_type": "file_reading", "needed": True, "reason": "read code"},
        {"action_type": "file_writing", "needed": True, "reason": "write code"},
    ]

    repaired, _ = repair_transformation_audit(
        "In src/parser.py, fix the bug and update or add tests for it.",
        intent,
        transform,
    )
    actions = {action["action_type"] for action in repaired["external_actions"]}

    assert {"file_reading", "file_writing", "code_execution"} <= actions


def test_repair_transformation_does_not_add_file_actions_for_generic_function():
    intent, transform, _, _, _ = base_plan()
    intent = deepcopy(intent)
    intent["final_user_want"] = "write a Python function"
    intent["inputs"][0]["format"] = "none"
    transform = deepcopy(transform)
    transform["external_actions"] = [
        {"action_type": "file_reading", "needed": True, "reason": "bad inference"}
    ]

    repaired, _ = repair_transformation_audit(
        "Write a Python function max_window_sum(nums, k, m).",
        intent,
        transform,
    )
    actions = {action["action_type"] for action in repaired["external_actions"]}

    assert "file_reading" not in actions


def test_repair_transformation_removes_current_info_for_image_measurement():
    intent = {
        "final_user_want": "measure the real-world height of the lamp",
        "inputs": [
            {
                "name": "lamp.jpg",
                "needed_for": "measurement",
                "available": True,
                "format": "image",
                "evidence": "attached image",
            },
            {
                "name": "scale",
                "needed_for": "measurement",
                "available": False,
                "format": "unknown",
                "evidence": "scale is required",
            },
        ],
        "missing_inputs": ["scale"],
    }
    transform = {
        "starting_state": "image is available",
        "desired_state": "height is measured",
        "transformations_needed": [],
        "needs_current_or_external_info": True,
        "external_actions": [
            {"action_type": "web_search", "needed": True, "reason": "bad inference"}
        ],
    }

    repaired, _ = repair_transformation_audit(
        "Measure the real-world height of the lamp in this image.",
        intent,
        transform,
    )
    actions = {action["action_type"] for action in repaired["external_actions"]}

    assert repaired["needs_current_or_external_info"] is False
    assert "web_search" not in actions
    assert {"image_understanding", "calculation", "user_input"} <= actions


def test_repair_requirements_adds_answer_after_current_retrieval():
    intent, transform, requirements, _, _ = base_plan()
    transform = deepcopy(transform)
    transform["needs_current_or_external_info"] = True
    transform["external_actions"] = [
        {"action_type": "web_search", "needed": True, "reason": "latest facts"}
    ]
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "retrieve_current_information",
                "capability_description": "Retrieve current facts.",
                "requires_external_action": True,
                "external_action_type": "web_search",
                "inputs": ["query"],
                "outputs": ["facts"],
                "done_when": "facts are available",
            }
        ]
    }

    repaired, _ = repair_capability_requirements(intent, transform, requirements)
    names = {cap["capability_name"] for cap in repaired["capabilities_needed"]}

    assert "answer_with_current_information" in names


def test_missing_input_does_not_trigger_current_information_capabilities():
    intent = {
        "final_user_want": "revise essay",
        "inputs": [
            {
                "name": "essay_text",
                "needed_for": "revision",
                "available": False,
                "format": "unknown",
                "evidence": "missing user input",
            }
        ],
        "missing_inputs": ["essay_text"],
    }
    transform = {
        "starting_state": "unknown",
        "desired_state": "revised essay",
        "transformations_needed": [],
        "needs_current_or_external_info": True,
        "external_actions": [
            {"action_type": "user_input", "needed": True, "reason": "Need essay text"}
        ],
    }

    repaired_transform, _ = repair_transformation_audit(
        "Can you make my essay better?",
        intent,
        transform,
    )
    requirements, _ = repair_capability_requirements(
        intent,
        repaired_transform,
        {"capabilities_needed": []},
    )
    actions = {cap["external_action_type"] for cap in requirements["capabilities_needed"]}
    names = {cap["capability_name"] for cap in requirements["capabilities_needed"]}

    assert repaired_transform["needs_current_or_external_info"] is False
    assert "web_search" not in actions
    assert "request_missing_input" in names


def test_missing_input_capability_name_is_canonicalized():
    intent = {
        "final_user_want": "revise essay",
        "inputs": [
            {
                "name": "essay_text",
                "needed_for": "revision",
                "available": False,
                "format": "unknown",
                "evidence": "missing user input",
            }
        ],
        "missing_inputs": ["essay_text"],
    }
    transform = {
        "external_actions": [
            {"action_type": "user_input", "needed": True, "reason": "Need essay text"}
        ]
    }
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "Request Missing Input",
                "capability_description": "Ask for missing essay text.",
                "requires_external_action": False,
                "external_action_type": "user_input",
                "inputs": [],
                "outputs": ["essay_text"],
                "done_when": "essay_text is provided",
            }
        ]
    }

    repaired, repairs = repair_capability_requirements(intent, transform, requirements)

    assert repaired["capabilities_needed"][0]["capability_name"] == "request_missing_input"
    assert repaired["capabilities_needed"][0]["requires_external_action"] is True
    assert any(repair["action"] == "rename_capability" for repair in repairs)


def test_repair_requirements_drops_unneeded_file_reading_capability():
    intent, transform, requirements, _, _ = base_plan()
    intent = deepcopy(intent)
    intent["inputs"][0]["format"] = "none"
    transform = deepcopy(transform)
    transform["external_actions"] = [
        {"action_type": "web_search", "needed": True, "reason": "latest facts"}
    ]
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "read_external_source",
                "capability_description": "Incorrectly read a file.",
                "requires_external_action": True,
                "external_action_type": "file_reading",
                "inputs": ["query"],
                "outputs": ["facts"],
                "done_when": "facts are available",
            }
        ]
    }

    repaired, _ = repair_capability_requirements(intent, transform, requirements)
    actions = {
        cap["external_action_type"]
        for cap in repaired["capabilities_needed"]
        if isinstance(cap, dict)
    }

    assert "file_reading" not in actions
    assert "web_search" in actions


def test_repair_requirements_renames_code_file_reading_as_inspection():
    intent, transform, requirements, _, _ = base_plan()
    intent = deepcopy(intent)
    intent["final_user_want"] = "fix parser bug and tests"
    intent["inputs"][0]["format"] = "file_path"
    transform = deepcopy(transform)
    transform["external_actions"] = [
        {"action_type": "file_reading", "needed": True, "reason": "read code"}
    ]
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "extract_file_content",
                "capability_description": "Read code file content.",
                "requires_external_action": True,
                "external_action_type": "file_reading",
                "inputs": ["src/parser.py"],
                "outputs": ["file content"],
                "done_when": "file content is available",
            }
        ]
    }

    repaired, _ = repair_capability_requirements(intent, transform, requirements)

    assert repaired["capabilities_needed"][0]["capability_name"] == "inspect_existing_code"


def test_repair_normalization_converts_human_names_to_snake_case():
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "Essay Improvement",
                "external_action_type": "none",
            }
        ]
    }
    normalization = {
        "normalized_capabilities": [
            {
                "id": "cap_1",
                "original_name": "Essay Improvement",
                "normalized_name": "Essay Improvement",
                "meaning_changed": False,
                "external_action_type": "none",
            }
        ],
        "merged_capabilities": [],
    }

    repaired, repairs = repair_capability_normalization(requirements, normalization)

    assert repaired["normalized_capabilities"][0]["normalized_name"] == "essay_improvement"
    assert any(repair["action"] == "normalize_capability_name" for repair in repairs)


def test_repair_ordering_restores_missing_capabilities_and_dependencies():
    requirements = {
        "capabilities_needed": [
            {
                "id": "cap_1",
                "capability_name": "extract_information_from_attached_document",
                "capability_description": "Extract text from a PDF.",
                "requires_external_action": True,
                "external_action_type": "file_reading",
                "inputs": ["attached PDF"],
                "outputs": ["extracted text"],
                "done_when": "the PDF text is available",
            },
            {
                "id": "cap_2",
                "capability_name": "summarize_document",
                "capability_description": "Summarize the extracted document.",
                "requires_external_action": False,
                "external_action_type": "none",
                "inputs": ["extracted text"],
                "outputs": ["summary"],
                "done_when": "the summary is ready",
            },
        ]
    }
    ordering = {
        "ordered_capabilities": [
            {
                "id": "cap_2",
                "capability_name": "summarize_document",
                "depends_on": [],
                "inputs": ["extracted text"],
                "outputs": ["summary"],
                "done_when": "the summary is ready",
            }
        ]
    }

    repaired, repairs = repair_capability_ordering(requirements, ordering)
    caps = {cap["id"]: cap for cap in repaired["ordered_capabilities"]}

    assert "cap_1" in caps
    assert "cap_1" in caps["cap_2"]["depends_on"]
    assert any(repair["action"] == "add_ordered_capability" for repair in repairs)


def test_rules_first_missing_attachment_requests_input_without_file_reading():
    plan = build_rules_first_capability_plan(
        "Summarize the paper I attached and explain the methods section."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any(
        item["available"] is False and "paper" in item["name"]
        for item in intent["inputs"]
    )
    actions = {item["action_type"] for item in transform["external_actions"]}
    assert "user_input" in actions
    assert "file_reading" not in actions
    assert any(
        cap["capability_name"] == "request_missing_input"
        for cap in requirements["capabilities_needed"]
    )
    assert plan["validation"]["valid"]


def test_rules_first_release_notes_uses_pasted_notes_and_file_write():
    plan = build_rules_first_capability_plan(
        "Create docs/release_notes.md from these notes: fixed login timeout, "
        "added CSV export, improved dashboard loading.",
        attachments_metadata=[
            {"name": "docs/release_notes.md", "format": "file_path", "available": True}
        ],
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    formats = {item["format"] for item in intent["inputs"]}
    assert {"pasted_text", "file_path"} <= formats
    actions = {item["action_type"] for item in transform["external_actions"]}
    assert actions == {"file_writing"}
    assert any(cap["capability_name"] == "write_file" for cap in requirements["capabilities_needed"])
    assert plan["validation"]["valid"]


def test_rules_first_inputs_keep_evidence_chunk_ids_and_spans():
    plan = build_rules_first_capability_plan(
        "Create docs/release_notes.md from these notes: fixed login timeout, "
        "added CSV export, improved dashboard loading.",
        attachments_metadata=[
            {"name": "docs/release_notes.md", "format": "file_path", "available": True}
        ],
    )
    inputs = plan["passes"]["intent_input_audit"]["parsed"]["inputs"]
    by_format = {item["format"]: item for item in inputs}

    assert by_format["file_path"]["source_chunk_ids"] == ["attachment_1"]
    assert by_format["file_path"]["evidence_span"] == "docs/release_notes.md"
    assert by_format["pasted_text"]["source_chunk_ids"] == ["pasted_text_1"]
    assert "fixed login timeout" in by_format["pasted_text"]["evidence_span"]


def test_rules_first_task_family_reports_matched_rules():
    family = classify_task_family(
        "What is Apple's latest stock price and market cap?",
        [],
    )

    assert family["families"] == ["current_fact"]
    assert family["signals"]["matched_rules"][0]["rule"] == "current_or_latest_request"


def test_rules_first_pasted_error_log_is_not_code_file_edit():
    plan = build_rules_first_capability_plan(
        "Explain this error log and suggest the likely fix:\n\n"
        "ImportError: cannot import name 'Client' from 'api.client' while loading tests/test_client.py"
    )
    family = plan["task_family"]["families"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]

    assert "pasted_text_analysis" in family
    assert "code_edit" not in family
    assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
    assert plan["validation"]["valid"]


def test_rules_first_inline_json_dates_are_not_log_calculation():
    request = (
        'From this JSON, list active users sorted by last_login descending: '
        '[{"name":"Mina","active":true,"last_login":"2026-06-01"},'
        '{"name":"Rae","active":true,"last_login":"2026-06-04"}]'
    )
    plan = build_rules_first_capability_plan(request)
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert "compute numeric" not in intent["final_user_want"].lower()
    assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
    assert any(
        cap["capability_name"] == "analyze_provided_dataset"
        and cap["external_action_type"] == "none"
        for cap in requirements["capabilities_needed"]
    )
    assert plan["validation"]["valid"]


def test_rules_first_classifies_spreadsheet_as_structured_analysis():
    family = classify_task_family(
        "In the attached workbook, calculate year-over-year growth by product.",
        [{"name": "product_growth.xlsx", "format": "attached_file", "available": True}],
    )

    assert "structured_data_analysis" in family["families"]


def test_rules_first_required_input_gate_marks_source_missing():
    plan = build_rules_first_capability_plan("Can you improve my resume summary?")
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any(item["available"] is False and "resume" in item["name"] for item in intent["inputs"])
    assert any("resume" in str(item).lower() for item in intent["missing_inputs"])
    assert {item["action_type"] for item in transform["external_actions"]} == {"user_input"}
    assert [cap["capability_name"] for cap in requirements["capabilities_needed"]] == [
        "request_missing_input"
    ]


def test_rules_first_inline_csv_routes_to_structured_calculation():
    plan = build_rules_first_capability_plan(
        "From this CSV, compute total revenue by plan:\n"
        "plan,revenue\n"
        "Basic,120\n"
        "Pro,300\n"
        "Basic,80"
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any(item["format"] == "structured_data" for item in intent["inputs"])
    assert all(item["format"] != "pasted_text" for item in intent["inputs"])
    assert {item["action_type"] for item in transform["external_actions"]} == {"calculation"}
    assert any(
        cap["capability_name"] == "analyze_provided_dataset"
        and cap["external_action_type"] == "calculation"
        for cap in requirements["capabilities_needed"]
    )


def test_rules_first_sql_routes_to_generate_code():
    plan = build_rules_first_capability_plan(
        "Write a SQL query that returns the top 10 customers by total spend "
        "from orders(customer_id, amount)."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any(item["available"] is True and item["format"] == "none" for item in intent["inputs"])
    assert any(cap["capability_name"] == "generate_code" for cap in requirements["capabilities_needed"])


def test_rules_first_two_url_compare_routes_retrieve_then_compare():
    plan = build_rules_first_capability_plan(
        "Compare the pricing pages at https://example.com/basic and https://example.com/pro."
    )
    requirements = plan["passes"]["capability_requirements"]["parsed"]
    ordering = plan["passes"]["capability_ordering"]["parsed"]
    names = [cap["capability_name"] for cap in requirements["capabilities_needed"]]
    caps_by_name = {cap["capability_name"]: cap for cap in ordering["ordered_capabilities"]}

    assert "retrieve_external_information" in names
    assert "compare_texts" in names
    assert caps_by_name["retrieve_external_information"]["depends_on"] == []
    assert "cap_1" in caps_by_name["compare_texts"]["depends_on"]


def test_rules_first_repo_fixme_routes_to_search_files():
    plan = build_rules_first_capability_plan(
        "Find all FIXME comments in this repository and write a short cleanup report.",
        attachments_metadata=[
            {"name": "repository files", "format": "file_path", "available": True}
        ],
    )
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any(
        cap["capability_name"] == "search_provided_files"
        for cap in requirements["capabilities_needed"]
    )


def test_task_route_classifies_core_ontology_routes():
    cases = [
        (
            "Can you polish my LinkedIn bio?",
            [],
            {
                "route": "missing_user_source",
                "input_status": "missing",
                "input_format": "no_source_available",
                "operation": "rewrite",
                "source_requirement": "requires_user_source",
            },
        ),
        (
            "From this TSV, calculate total signups by channel:\nchannel\tsignups\nPaid\t80\nOrganic\t75",
            [],
            {
                "route": "structured_data_calculation",
                "input_status": "available",
                "input_format": "structured_data",
                "operation": "calculate",
            },
        ),
        (
            "Write a Postgres query that returns monthly revenue from orders(created_at, total).",
            [],
            {
                "route": "code_generation",
                "input_status": "available",
                "input_format": "request_spec",
                "operation": "generate_code",
            },
        ),
        (
            "Compare these three pages: https://example.com/a https://example.com/b https://example.com/c",
            [],
            {
                "route": "external_multi_source_comparison",
                "input_format": "url_list",
                "operation": "compare",
                "source_requirement": "requires_external_retrieval",
            },
        ),
        (
            "Scan this codebase for deprecated API calls and summarize the files affected.",
            [{"name": "repository files", "format": "file_path", "available": True}],
            {
                "route": "repo_search",
                "input_format": "repo_files",
                "operation": "repo_search",
                "source_requirement": "requires_file_access",
            },
        ),
    ]

    for request, attachments, expected in cases:
        plan = build_rules_first_capability_plan(request, attachments_metadata=attachments)
        route = plan["task_route"]
        direct_route = classify_task_route(
            request,
            plan["passes"]["intent_input_audit"]["parsed"],
            plan["task_family"],
            plan["evidence_chunks"],
        )

        for key, value in expected.items():
            assert route[key] == value
            assert getattr(direct_route, key) == value


def test_rules_first_near_missing_source_variants_are_gated():
    for request, keyword in [
        ("Can you polish my LinkedIn bio?", "bio"),
        ("Revise my grant proposal intro so it is clearer.", "proposal"),
        ("Plot revenue by region for Q2.", "revenue"),
        ("Summarize this blog post in five bullets.", "blog post"),
    ]:
        plan = build_rules_first_capability_plan(request)
        intent = plan["passes"]["intent_input_audit"]["parsed"]
        requirements = plan["passes"]["capability_requirements"]["parsed"]

        assert any(keyword in str(item).lower() for item in intent["missing_inputs"])
        assert [cap["capability_name"] for cap in requirements["capabilities_needed"]] == [
            "request_missing_input"
        ]


def test_rules_first_operation_hints_do_not_match_substrings():
    plan = build_rules_first_capability_plan(
        "Make this LinkedIn bio tighter: Operations leader with experience helping "
        "cross-functional teams improve process, reporting, and vendor management."
    )
    family = plan["task_family"]
    intent = plan["passes"]["intent_input_audit"]["parsed"]

    assert family["signals"]["operation"] == "rewrite"
    assert "missing_input" not in family["families"]
    assert any(item["format"] == "pasted_text" for item in intent["inputs"])


def test_rules_first_pipe_and_loose_csv_route_to_structured_data():
    for request in [
        "From this table, calculate total hours by project:\nproject|hours\nApollo|4\nBeacon|6\nApollo|3",
        "Here is CSV data. Count orders by status.\nstatus,orders\nOpen,12\nClosed,18\nOpen,5",
    ]:
        plan = build_rules_first_capability_plan(request)
        intent = plan["passes"]["intent_input_audit"]["parsed"]
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]

        assert any(item["format"] == "structured_data" for item in intent["inputs"])
        assert {item["action_type"] for item in transform["external_actions"]} == {"calculation"}


def test_task_frame_segments_delimited_sources_structurally():
    cases = [
        (
            "Categorize these comments as praise, bug, or question:\n"
            "1. Love the new dashboard.\n2. Export fails on Safari.\n3. Where is billing history?",
            "classify_provided_text",
        ),
        (
            "Rewrite this policy blurb to be clearer: Users may initiate cancellation "
            "after the current renewal term has elapsed.",
            "revise_text_for_clarity",
        ),
        (
            "Summarize this memo in two bullets: The launch moves to August. "
            "Security review is complete.",
            "summarize_document",
        ),
        (
            'Translate "hola" into English.',
            "translate_text",
        ),
    ]

    for request, expected_capability in cases:
        plan = build_rules_first_capability_plan(request)
        intent = plan["passes"]["intent_input_audit"]["parsed"]
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        requirements = plan["passes"]["capability_requirements"]["parsed"]

        assert any(item["format"] == "pasted_text" for item in intent["inputs"])
        assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
        assert any(
            cap["capability_name"] == expected_capability
            for cap in requirements["capabilities_needed"]
        )


def test_task_frame_routes_only_from_instruction_not_source_content():
    cases = [
        (
            "Compare these CTAs and pick the more direct one.\n\n"
            "A: Start simplifying approvals today.\n\n"
            "B: SELECT your plan now.",
            "compare_texts",
        ),
        (
            "Categorize these comments as praise, bug, or question:\n"
            "1. Where is billing history?\n2. SELECT your plan now.\n3. Start today.",
            "classify_provided_text",
        ),
    ]

    for request, expected_capability in cases:
        plan = build_rules_first_capability_plan(request)
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        requirements = plan["passes"]["capability_requirements"]["parsed"]
        route = plan["task_frame"]

        assert route["requires_external_info"] is False
        assert route["route"] not in {"current_information_answer", "code_generation"}
        assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
        assert any(
            cap["capability_name"] == expected_capability
            for cap in requirements["capabilities_needed"]
        )


def test_url_inside_source_span_is_embedded_not_retrieved():
    plan = build_rules_first_capability_plan(
        "Rewrite this sentence for clarity: Visit https://example.com/current/select "
        "today to choose a package."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]
    route = plan["task_frame"]
    url_chunks = [chunk for chunk in plan["evidence_chunks"] if chunk["type"] == "url"]

    assert any(item["format"] == "pasted_text" for item in intent["inputs"])
    assert all(item["format"] != "url" for item in intent["inputs"])
    assert url_chunks == []
    assert route["route"] == "pasted_text_rewrite"
    assert route["requires_external_info"] is False
    assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
    assert any(
        cap["capability_name"] == "revise_text_for_clarity"
        for cap in requirements["capabilities_needed"]
    )


def test_url_path_tokens_are_masked_before_operation_and_current_detection():
    plan = build_rules_first_capability_plan(
        "Compare https://example.com/select and https://example.com/current."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]
    route = plan["task_frame"]

    assert sum(1 for item in intent["inputs"] if item["format"] == "url") == 2
    assert route["route"] == "external_multi_source_comparison"
    assert route["operation"] == "compare"
    assert {item["action_type"] for item in transform["external_actions"]} == {"web_search"}
    assert "generate_code" not in [
        cap["capability_name"] for cap in requirements["capabilities_needed"]
    ]
    assert "answer_with_current_information" not in [
        cap["capability_name"] for cap in requirements["capabilities_needed"]
    ]


def test_current_as_plain_adjective_does_not_trigger_live_info():
    cases = [
        (
            "Create a banner graphic for a webinar about current hiring trends.",
            "image_generation",
            {"image_generation"},
        ),
        (
            "Draft a short email saying the current rollout is complete.",
            "text_generation",
            {"none"},
        ),
    ]

    for request, expected_route, expected_actions in cases:
        plan = build_rules_first_capability_plan(request)
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        route = plan["task_frame"]

        assert route["route"] == expected_route
        assert route["requires_external_info"] is False
        assert {item["action_type"] for item in transform["external_actions"]} == expected_actions


def test_generic_comments_classification_without_source_requests_missing_input():
    plan = build_rules_first_capability_plan("Classify my launch comments by sentiment.")
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert any("comments" in str(item).lower() for item in intent["missing_inputs"])
    assert {item["action_type"] for item in transform["external_actions"]} == {"user_input"}
    assert [cap["capability_name"] for cap in requirements["capabilities_needed"]] == [
        "request_missing_input"
    ]


def test_explain_has_source_and_concept_variants():
    source_plan = build_rules_first_capability_plan(
        "Explain this customer message:\n\nCan I SELECT a new plan today?"
    )
    source_transform = source_plan["passes"]["transformation_externality_audit"]["parsed"]
    source_caps = source_plan["passes"]["capability_requirements"]["parsed"][
        "capabilities_needed"
    ]

    assert source_plan["task_frame"]["route"] == "pasted_text_diagnosis"
    assert {item["action_type"] for item in source_transform["external_actions"]} == {"none"}
    assert [cap["capability_name"] for cap in source_caps] == ["analyze_provided_text"]

    missing_plan = build_rules_first_capability_plan("Explain why this code fails.")
    missing_caps = missing_plan["passes"]["capability_requirements"]["parsed"][
        "capabilities_needed"
    ]
    assert missing_plan["task_frame"]["route"] == "missing_user_source"
    assert [cap["capability_name"] for cap in missing_caps] == ["request_missing_input"]

    concept_plan = build_rules_first_capability_plan("Explain recursion in simple terms.")
    concept_caps = concept_plan["passes"]["capability_requirements"]["parsed"][
        "capabilities_needed"
    ]
    assert concept_plan["task_frame"]["route"] == "concept_explanation"
    assert [cap["capability_name"] for cap in concept_caps] == ["provide_explanation"]


def test_current_rates_plural_triggers_current_info():
    plan = build_rules_first_capability_plan("What are current 30-year mortgage rates in the US?")
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert plan["task_frame"]["route"] == "current_information_answer"
    assert {item["action_type"] for item in transform["external_actions"]} == {"web_search"}
    assert any(cap["capability_name"] == "retrieve_current_information" for cap in caps)


def test_public_intent_queries_route_to_specific_capabilities():
    cases = [
        ("how would they say butter in zambia", "self_contained_translation", "translate_text", {"none"}),
        ("how does one say wonderful in german", "self_contained_translation", "translate_text", {"none"}),
        ("what is the right way to say excuse me in spanish", "self_contained_translation", "translate_text", {"none"}),
        ("what is the definition of forensic", "concept_explanation", "provide_explanation", {"none"}),
        ("look up the definition of blunder", "concept_explanation", "provide_explanation", {"none"}),
        ("what's the week's forecast", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("please find today's most read stories from the new york times", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what were the top stories this week", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what is happening with brexit right now", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what is the current temperature outside", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("is it raining", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("is it currently snowing", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("is it going to be windy tomorrow", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("should i bring an umbrella tomorrow", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("how cold will it get each night this week", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what is the exchange rate between us dollars and japanese yen", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("convert 200 us dollars to british pounds", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what is terranova stock going for", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("were the stocks rising or declining", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("what is the current share value for m. s. n. b. c.", "current_information_answer", "retrieve_current_information", {"web_search"}),
        ("reply to dad write i will be late", "text_generation", "draft_text", {"none"}),
        ("drop a message to jan write i will be late", "text_generation", "draft_text", {"none"}),
        ("send this email to reply to the latest emails", "missing_user_source", "request_missing_input", {"user_input"}),
        ("how many prime numbers are there between 0 and 100", "numeric_calculation", "compute_numeric_result", {"calculation"}),
    ]

    for request, expected_route, expected_capability, expected_actions in cases:
        plan = build_rules_first_capability_plan(request)
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

        assert plan["task_frame"]["route"] == expected_route
        assert {item["action_type"] for item in transform["external_actions"]} == expected_actions
        assert any(cap["capability_name"] == expected_capability for cap in caps)


def test_general_self_contained_question_uses_explanation_capability():
    for request in [
        "what is a chair",
        "where does the power steering fluid go",
        "how do i compute the median of a set of numbers",
    ]:
        plan = build_rules_first_capability_plan(request)
        caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

        assert plan["task_frame"]["route"] in {"concept_explanation", "general_request"}
        assert [cap["capability_name"] for cap in caps] == ["provide_explanation"]


def test_personal_recent_activity_is_not_public_current_info():
    plan = build_rules_first_capability_plan("show me recent activity in my backyard")
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert plan["task_frame"]["route"] == "general_request"
    assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
    assert [cap["capability_name"] for cap in caps] == ["provide_explanation"]


def test_extract_from_url_is_retrieval_not_missing_source():
    plan = build_rules_first_capability_plan(
        "Extract migration steps from https://example.com/releases/v3."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert any(item["format"] == "url" for item in intent["inputs"])
    assert plan["task_frame"]["route"] == "external_source_summary"
    assert {item["action_type"] for item in transform["external_actions"]} == {"web_search"}
    assert any(cap["capability_name"] == "retrieve_external_information" for cap in caps)


def test_url_inside_explicit_source_text_is_not_retrieval():
    plan = build_rules_first_capability_plan(
        "Summarize this text: Visit https://example.com/current/select today to choose a package."
    )
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert any(item["format"] == "pasted_text" for item in intent["inputs"])
    assert not any(item["format"] == "url" for item in intent["inputs"])
    assert plan["task_frame"]["route"] == "pasted_text_summary"
    assert {item["action_type"] for item in transform["external_actions"]} == {"none"}
    assert [cap["capability_name"] for cap in caps] == ["summarize_document"]


def test_bare_url_after_article_instruction_still_requires_retrieval():
    plan = build_rules_first_capability_plan("Summarize this article: https://example.com/current")
    intent = plan["passes"]["intent_input_audit"]["parsed"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert any(item["format"] == "url" for item in intent["inputs"])
    assert plan["task_frame"]["route"] == "external_source_summary"
    assert {item["action_type"] for item in transform["external_actions"]} == {"web_search"}
    assert any(cap["capability_name"] == "retrieve_external_information" for cap in caps)


def test_repo_search_does_not_trigger_current_info_from_search_word():
    for request in [
        "Search for references to ENABLE_BILLING_V2 in this repository.",
        "Search the codebase for console.log calls and make a cleanup list.",
    ]:
        plan = build_rules_first_capability_plan(
            request,
            attachments_metadata=[
                {"name": "repository files", "format": "file_path", "available": True}
            ],
        )
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

        assert plan["task_frame"]["route"] == "repo_search"
        assert plan["task_frame"]["requires_external_info"] is False
        assert {item["action_type"] for item in transform["external_actions"]} == {"file_reading"}
        assert any(cap["capability_name"] == "search_provided_files" for cap in caps)


def test_attached_structured_data_ranking_requires_reading_and_calculation():
    plan = build_rules_first_capability_plan(
        "Analyze the attached CSV of monthly sales and tell me the top three regions by revenue.",
        attachments_metadata=[
            {"name": "monthly_sales.csv", "format": "structured_data", "available": True}
        ],
    )
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    caps = plan["passes"]["capability_requirements"]["parsed"]["capabilities_needed"]

    assert {item["action_type"] for item in transform["external_actions"]} == {
        "file_reading",
        "calculation",
    }
    assert any(
        cap["capability_name"] == "analyze_provided_dataset"
        and cap["external_action_type"] == "calculation"
        for cap in caps
    )


def test_source_consuming_operations_without_source_request_missing_input():
    for request, keyword in [
        ("Turn my customer interview notes into themes.", "notes"),
        ("Cluster my research notes into three themes.", "notes"),
        ("Extract objections from the sales calls.", "call"),
        ("Extract risks from my project plan.", "project plan"),
    ]:
        plan = build_rules_first_capability_plan(request)
        intent = plan["passes"]["intent_input_audit"]["parsed"]
        transform = plan["passes"]["transformation_externality_audit"]["parsed"]
        requirements = plan["passes"]["capability_requirements"]["parsed"]

        assert any(
            item["available"] is False and keyword in str(item).lower()
            for item in intent["inputs"]
        )
        assert {item["action_type"] for item in transform["external_actions"]} == {"user_input"}
        assert [cap["capability_name"] for cap in requirements["capabilities_needed"]] == [
            "request_missing_input"
        ]


def test_missing_input_slot_filler_triggers_only_for_source_dependent_empty_inputs():
    empty_resume_intent = {
        "final_user_want": "improve my resume summary",
        "inputs": [],
        "missing_inputs": [],
    }
    current_fact_intent = {
        "final_user_want": "answer latest React version",
        "inputs": [],
        "missing_inputs": [],
    }
    writing_intent = {
        "final_user_want": "draft a birthday email",
        "inputs": [
            {
                "name": "writing requirements",
                "available": True,
                "format": "none",
            }
        ],
        "missing_inputs": [],
    }

    assert should_run_missing_input_slot_filler(
        "Can you improve my resume summary?",
        empty_resume_intent,
        {"families": ["general_request"]},
    )
    assert not should_run_missing_input_slot_filler(
        "What is the latest stable React version?",
        current_fact_intent,
        {"families": ["current_fact"]},
    )
    assert not should_run_missing_input_slot_filler(
        "Write a warm birthday email to Alex.",
        writing_intent,
        {"families": ["text_generation"]},
    )


def test_missing_input_slot_filler_prompt_is_narrow():
    messages = build_missing_input_slot_filler_messages(
        "Summarize this article in three bullets.",
        intent_audit={
            "final_user_want": "summarize this article in three bullets",
            "inputs": [],
            "missing_inputs": [],
        },
        task_family={"families": ["general_request"]},
    )
    text = "\n".join(message["content"] for message in messages)

    assert "missing-input slot filler" in text
    assert '"missing_inputs":[]' in text
    assert "Do not propose tools, capabilities, or actions" in text
    assert "current/latest facts" in text


def test_missing_input_slot_filler_adds_unavailable_input_and_rebuilds_plan():
    request = "Can you improve my resume summary?"
    intent = {
        "final_user_want": "improve my resume summary",
        "inputs": [],
        "missing_inputs": [],
    }
    task_family = {"families": ["general_request"]}

    filled_intent, repairs = apply_missing_input_slot_filler(
        request,
        [],
        intent,
        {
            "missing_inputs": [
                {
                    "name": "resume summary text",
                    "reason": "The request asks to improve a resume summary, but no text is provided.",
                    "evidence_span": "my resume summary",
                }
            ]
        },
        task_family,
    )
    rebuilt = build_rules_first_capability_plan(
        request,
        intent_override=filled_intent,
        intent_extra_repairs=repairs,
    )
    transform = rebuilt["passes"]["transformation_externality_audit"]["parsed"]
    requirements = rebuilt["passes"]["capability_requirements"]["parsed"]

    assert any(item["available"] is False and "resume" in item["name"] for item in filled_intent["inputs"])
    assert any("resume" in str(item).lower() for item in filled_intent["missing_inputs"])
    assert any(repair["action"] == "missing_input_slot_filler" for repair in repairs)
    assert {item["action_type"] for item in transform["external_actions"]} == {"user_input"}
    assert any(
        cap["capability_name"] == "request_missing_input"
        for cap in requirements["capabilities_needed"]
    )
    assert rebuilt["validation"]["valid"]


def test_rules_first_external_tool_action_routes_to_other():
    plan = build_rules_first_capability_plan(
        "Available external tool/API documentation is provided by the benchmark. "
        "Plan the external tool/API action needed for this request: "
        "Find the timezone for latitude 48.8584 and longitude 2.2945."
    )

    route = plan["task_route"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]
    caps = requirements["capabilities_needed"]

    assert route["operation"] == "external_tool"
    assert route["route"] == "external_tool_action"
    assert {item["action_type"] for item in transform["external_actions"]} == {"other"}
    assert any(
        cap["capability_name"] == "select_and_execute_external_action"
        and cap["external_action_type"] == "other"
        for cap in caps
    )
    assert all("timezone" not in cap["capability_name"] for cap in caps)
    assert plan["validation"]["valid"]


def test_rules_first_bfcl_irrelevance_math_stays_calculation_not_tool_action():
    plan = build_rules_first_capability_plan(
        "Calculate the area of a triangle given the base is 10 meters and height is 5 meters."
    )

    route = plan["task_route"]
    transform = plan["passes"]["transformation_externality_audit"]["parsed"]
    requirements = plan["passes"]["capability_requirements"]["parsed"]

    assert route["route"] == "numeric_calculation"
    assert {item["action_type"] for item in transform["external_actions"]} == {"calculation"}
    assert any(
        cap["capability_name"] == "compute_numeric_result"
        for cap in requirements["capabilities_needed"]
    )
    assert not any(
        cap["capability_name"] == "select_and_execute_external_action"
        for cap in requirements["capabilities_needed"]
    )
