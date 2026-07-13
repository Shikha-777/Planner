from __future__ import annotations

from taskdecomp.agent_binding import (
    build_agent_binding_plan,
    build_default_agent_registry,
    extract_ordered_subtasks,
)
from taskdecomp.agent_binding_benchmark import build_synthetic_agent_binding_cases
from taskdecomp.agent_binding_eval import score_predictions


def test_agent_binding_assigns_research_then_summarizer():
    plan = build_agent_binding_plan(
        {
            "request": "Summarize this article: https://example.com/article",
            "capability_plan": {
                "ordered_capabilities": [
                    {
                        "id": "s1",
                        "capability_name": "retrieve_external_information",
                        "inputs": ["https://example.com/article"],
                        "outputs": ["article text"],
                        "depends_on": [],
                    },
                    {
                        "id": "s2",
                        "capability_name": "summarize_retrieved_content",
                        "inputs": [],
                        "outputs": ["article summary"],
                        "depends_on": ["s1"],
                    },
                ]
            },
            "agents": build_default_agent_registry(),
        }
    )

    assert plan["assignment_decision"] == "assign"
    assert [item["assigned_agent"] for item in plan["assignments"]] == [
        "research_agent",
        "summarizer_agent",
    ]
    assert plan["assignments"][1]["inputs_passed"] == ["s1.output"]
    assert plan["validation_notes"] == []


def test_agent_binding_distinguishes_code_roles_and_preserves_dependencies():
    plan = build_agent_binding_plan(
        "Add a React component and update src/index.ts, then validate the result.",
        {
            "ordered_capabilities": [
                {
                    "id": "s1",
                    "capability_name": "inspect_existing_code",
                    "inputs": ["src/index.ts", "repository files"],
                    "outputs": ["code context"],
                    "depends_on": [],
                },
                {
                    "id": "s2",
                    "capability_name": "modify_code",
                    "inputs": [],
                    "outputs": ["code changes"],
                    "depends_on": ["s1"],
                },
                {
                    "id": "s3",
                    "capability_name": "validate_output_against_requirements",
                    "inputs": [],
                    "outputs": ["validation result"],
                    "depends_on": ["s2"],
                },
            ]
        },
        build_default_agent_registry(),
    )

    assert [item["assigned_agent"] for item in plan["assignments"]] == [
        "code_inspection_agent",
        "code_modification_agent",
        "test_runner_agent",
    ]
    assert [item["depends_on"] for item in plan["assignments"]] == [[], ["s1"], ["s2"]]


def test_agent_binding_returns_ask_user_for_missing_input_capability():
    plan = build_agent_binding_plan(
        "Summarize the article I mentioned earlier.",
        {
            "ordered_capabilities": [
                {
                    "id": "s1",
                    "capability_name": "request_missing_input",
                    "inputs": ["missing article source"],
                    "outputs": ["missing input request"],
                    "depends_on": [],
                }
            ]
        },
        build_default_agent_registry(),
    )

    assert plan["assignment_decision"] == "ask_user"
    assert plan["assignments"][0]["assigned_agent"] == "clarification_agent"


def test_agent_binding_returns_unsupported_for_unknown_capability():
    plan = build_agent_binding_plan(
        "Do something outside the registry.",
        {
            "ordered_capabilities": [
                {
                    "id": "s1",
                    "capability_name": "invent_new_physical_device",
                    "inputs": ["request"],
                    "outputs": ["device"],
                    "depends_on": [],
                }
            ]
        },
        build_default_agent_registry(),
    )

    assert plan["assignment_decision"] == "unsupported"
    assert plan["assignments"] == []
    assert plan["unassigned_capabilities"] == ["invent_new_physical_device"]


def test_agent_binding_accepts_full_rules_first_capability_plan_shape():
    subtasks = extract_ordered_subtasks(
        {
            "passes": {
                "capability_ordering": {
                    "parsed": {
                        "ordered_capabilities": [
                            {"id": "cap_1", "capability_name": "draft_text"}
                        ]
                    }
                }
            }
        }
    )

    assert len(subtasks) == 1
    assert subtasks[0].subtask_id == "cap_1"
    assert subtasks[0].capability == "draft_text"


def test_agent_binding_scorer_reports_exact_match():
    rows = build_synthetic_agent_binding_cases(3)
    predictions = []
    for row in rows:
        predictions.append(
            {
                "id": row["id"],
                "agent_binding_plan": build_agent_binding_plan(
                    {
                        "request": row["request"],
                        "capability_plan": row["gold_capability_plan"],
                        "agents": row["agents"],
                    }
                ),
            }
        )

    scored = score_predictions(rows, predictions)

    assert scored["metrics"]["case_count"] == 3
    assert scored["metrics"]["agent_assignment_exact_match"] == 1.0
    assert scored["metrics"]["capability_to_agent_match_rate"] == 1.0


def test_synthetic_agent_binding_benchmark_builds_100_cases_with_near_miss_registry():
    rows = build_synthetic_agent_binding_cases(100)

    assert len(rows) == 100
    assert rows[0]["id"] == "agent_binding_001"
    assert rows[-1]["id"] == "agent_binding_100"
    assert len(rows[0]["agents"]) >= 20
    agent_ids = {agent["agent_id"] for agent in rows[0]["agents"]}
    assert {"data_analysis_agent", "spreadsheet_agent", "chart_agent"} <= agent_ids
    assert rows[0]["expected_agent_binding"]["assignments"]
