from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


ASSIGNMENT_DECISIONS = {"assign", "ask_user", "unsupported"}

QUALITY_VALUES = {"low": 0.0, "medium": 0.5, "high": 1.0}
INVERSE_QUALITY_VALUES = {"low": 1.0, "medium": 0.5, "high": 0.0}


@dataclass
class AgentCard:
    agent_id: str
    name: str
    description: str
    capabilities: tuple[str, ...] = ()
    input_contract: dict[str, tuple[str, ...]] = field(default_factory=dict)
    output_contract: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cannot_do: tuple[str, ...] = ()
    cost: str = "medium"
    latency: str = "medium"
    reliability_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "input_contract": _contract_to_dict(self.input_contract),
            "output_contract": _contract_to_dict(self.output_contract),
            "cannot_do": list(self.cannot_do),
            "cost": self.cost,
            "latency": self.latency,
            "reliability_score": self.reliability_score,
        }


@dataclass
class CapabilitySubtask:
    subtask_id: str
    capability: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


CAPABILITY_INPUT_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "analyze_provided_dataset": (("structured_data",),),
    "classify_provided_text": (("text", "document", "request"),),
    "combine_files": (("file", "attached_file", "pdf"),),
    "compare_texts": (("text", "document"),),
    "compute_numeric_result": (("structured_data", "numeric_values", "analysis_summary"),),
    "create_chart": (("structured_data", "chart_data"),),
    "diagnose_issue": (("error_log", "codebase", "text"),),
    "draft_text": (("writing_goal", "request"),),
    "execute_code": (("codebase", "code_changes", "generated_code"),),
    "extract_action_items": (("text", "document"),),
    "extract_information_from_attached_document": (("attached_file", "document", "pdf"),),
    "extract_information_from_file": (("file", "file_path", "codebase"),),
    "extract_information_from_image": (("image",),),
    "generate_code": (("code_request", "request"),),
    "generate_image": (("image_prompt", "request"),),
    "identify_outliers": (("structured_data", "analysis_summary"),),
    "inspect_existing_code": (("codebase", "file_path"),),
    "interpret_image_content": (("image",),),
    "modify_code": (("codebase", "code_context"),),
    "prepare_chart_data": (("structured_data", "analysis_summary"),),
    "report_external_action_result": (("external_action_result",),),
    "report_findings": (("analysis_summary", "computed_result", "request"),),
    "request_missing_input": (("missing_input_request", "request"),),
    "retrieve_current_information": (("current_info_query", "request"),),
    "retrieve_external_information": (("url", "external_query", "request"),),
    "revise_text_for_clarity": (("text", "document"),),
    "search_provided_files": (("codebase", "file_path", "file"),),
    "select_and_execute_external_action": (("request_spec", "request"),),
    "summarize_document": (("text", "document", "pdf", "retrieved_content"),),
    "summarize_findings": (("analysis_summary", "computed_result", "request"),),
    "summarize_provided_text": (("text", "document"),),
    "summarize_retrieved_content": (("retrieved_content", "current_information"),),
    "synthesize_plan": (("planning_goal", "request"),),
    "transform_image": (("image",),),
    "transform_text_format": (("text", "document", "structured_data"),),
    "translate_text": (("text", "document"), ("language", "request")),
    "validate_output_against_requirements": (
        ("code_changes", "generated_code", "draft_text", "request"),
    ),
    "verify_current_information": (("current_information", "current_info_query", "request"),),
    "write_file": (("file_content", "draft_text", "code_changes", "request"),),
}

CAPABILITY_OUTPUT_KINDS: dict[str, tuple[str, ...]] = {
    "analyze_provided_dataset": ("analysis_summary", "structured_data"),
    "classify_provided_text": ("classification_result",),
    "combine_files": ("combined_file", "file"),
    "compare_texts": ("comparison_summary",),
    "compute_numeric_result": ("computed_result", "analysis_summary"),
    "create_chart": ("chart",),
    "diagnose_issue": ("diagnosis", "analysis_summary"),
    "draft_text": ("draft_text", "text"),
    "execute_code": ("execution_result", "validation_result"),
    "extract_action_items": ("extracted_items",),
    "extract_information_from_attached_document": ("document_content", "text"),
    "extract_information_from_file": ("file_content", "text"),
    "extract_information_from_image": ("image_extraction", "text"),
    "generate_code": ("generated_code", "code_changes"),
    "generate_image": ("generated_image", "image"),
    "identify_outliers": ("analysis_summary", "outlier_report"),
    "inspect_existing_code": ("code_context", "codebase"),
    "interpret_image_content": ("image_description", "text"),
    "modify_code": ("code_changes", "codebase"),
    "prepare_chart_data": ("chart_data", "structured_data"),
    "report_external_action_result": ("external_action_report", "text"),
    "report_findings": ("findings_report", "text"),
    "request_missing_input": ("missing_input_request",),
    "retrieve_current_information": ("current_information", "retrieved_content"),
    "retrieve_external_information": ("retrieved_content", "external_source"),
    "revise_text_for_clarity": ("revised_text", "text"),
    "search_provided_files": ("file_search_results", "code_context"),
    "select_and_execute_external_action": ("external_action_result",),
    "summarize_document": ("summary", "text"),
    "summarize_findings": ("summary", "text"),
    "summarize_provided_text": ("summary", "text"),
    "summarize_retrieved_content": ("summary", "text"),
    "synthesize_plan": ("plan", "text"),
    "transform_image": ("transformed_image", "image"),
    "transform_text_format": ("formatted_text", "text"),
    "translate_text": ("translated_text", "text"),
    "validate_output_against_requirements": ("validation_result",),
    "verify_current_information": ("verification_result", "current_information"),
    "write_file": ("written_file", "file"),
}

PREFERRED_AGENT_BY_CAPABILITY: dict[str, str] = {
    "analyze_provided_dataset": "data_analysis_agent",
    "classify_provided_text": "classification_agent",
    "combine_files": "file_conversion_agent",
    "compare_texts": "writer_agent",
    "compute_numeric_result": "data_analysis_agent",
    "create_chart": "chart_agent",
    "diagnose_issue": "debugging_agent",
    "draft_text": "writer_agent",
    "execute_code": "test_runner_agent",
    "extract_action_items": "extraction_agent",
    "extract_information_from_attached_document": "document_reader_agent",
    "extract_information_from_file": "extraction_agent",
    "extract_information_from_image": "image_understanding_agent",
    "generate_code": "code_modification_agent",
    "generate_image": "image_generation_agent",
    "identify_outliers": "data_analysis_agent",
    "inspect_existing_code": "code_inspection_agent",
    "interpret_image_content": "image_understanding_agent",
    "modify_code": "code_modification_agent",
    "prepare_chart_data": "chart_agent",
    "report_external_action_result": "tool_executor_agent",
    "report_findings": "writer_agent",
    "request_missing_input": "clarification_agent",
    "retrieve_current_information": "current_info_agent",
    "retrieve_external_information": "research_agent",
    "revise_text_for_clarity": "rewriter_agent",
    "search_provided_files": "code_search_agent",
    "select_and_execute_external_action": "tool_executor_agent",
    "summarize_document": "summarizer_agent",
    "summarize_findings": "summarizer_agent",
    "summarize_provided_text": "summarizer_agent",
    "summarize_retrieved_content": "summarizer_agent",
    "synthesize_plan": "writer_agent",
    "transform_image": "image_generation_agent",
    "transform_text_format": "rewriter_agent",
    "translate_text": "translator_agent",
    "validate_output_against_requirements": "test_runner_agent",
    "verify_current_information": "current_info_agent",
    "write_file": "file_conversion_agent",
}

KIND_ALIASES: dict[str, tuple[str, ...]] = {
    "analysis_summary": ("request",),
    "attached_file": ("file",),
    "chart_data": ("structured_data", "analysis_summary"),
    "code_context": ("codebase", "file_path"),
    "code_request": ("request",),
    "current_info_query": ("request",),
    "document": ("text", "attached_file"),
    "external_query": ("request",),
    "file_content": ("text", "file"),
    "image_prompt": ("request",),
    "language": ("request",),
    "missing_input_request": ("request",),
    "numeric_values": ("structured_data", "request"),
    "planning_goal": ("request",),
    "request_spec": ("request",),
    "retrieved_content": ("current_information",),
    "structured_data": ("spreadsheet", "csv", "table"),
    "text": (
        "document",
        "draft_text",
        "revised_text",
        "retrieved_content",
        "current_information",
        "verification_result",
        "summary",
    ),
    "url": ("external_source",),
    "writing_goal": ("request",),
}


def build_agent_binding_plan(
    user_request: str | dict[str, Any],
    capability_plan: dict[str, Any] | list[Any] | None = None,
    agents: list[dict[str, Any] | AgentCard] | None = None,
) -> dict[str, Any]:
    """Assign abstract capability subtasks to expert agents.

    The function accepts either the explicit arguments above or the payload shape:
    {"request": "...", "capability_plan": {...}, "agents": [...]}.
    """
    request, plan, agent_rows = _coerce_binding_inputs(user_request, capability_plan, agents)
    agent_cards = [normalize_agent_card(agent) for agent in agent_rows]
    subtasks = extract_ordered_subtasks(plan)

    if not subtasks:
        return _binding_plan(
            "unsupported",
            [],
            [],
            ["no capability subtasks were available for assignment"],
            [],
        )
    if not agent_cards:
        return _binding_plan(
            "unsupported",
            [],
            [],
            ["no agent registry was provided"],
            [subtask.capability for subtask in subtasks],
        )

    context = _initial_assignment_context(request)
    assignments: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    validation_notes: list[str] = []
    unassigned: list[str] = []
    saw_missing_input_capability = False
    saw_unsupported_capability = False

    for subtask in subtasks:
        if subtask.capability == "request_missing_input":
            saw_missing_input_capability = True
        if subtask.capability == "unsupported_request":
            saw_unsupported_capability = True

        missing_dependencies = [
            dep for dep in subtask.depends_on if dep and dep not in context["assigned_subtasks"]
        ]
        if missing_dependencies:
            unassigned.append(subtask.capability)
            missing_text = ", ".join(missing_dependencies)
            validation_notes.append(
                f"{subtask.subtask_id} depends on unassigned subtasks: {missing_text}"
            )
            continue

        ranked = rank_agent_candidates(subtask, agent_cards, request, context)
        eligible = [candidate for candidate in ranked if candidate["eligible"]]
        if not eligible:
            unassigned.append(subtask.capability)
            candidate_missing = _dedupe(
                item
                for candidate in ranked[:3]
                for item in candidate.get("missing_inputs", [])
            )
            missing_inputs.extend(candidate_missing)
            if ranked:
                validation_notes.append(
                    f"no eligible agent for {subtask.capability}; best candidate was "
                    f"{ranked[0]['agent_id']}"
                )
            else:
                validation_notes.append(f"no agent can perform {subtask.capability}")
            continue

        chosen = eligible[0]
        chosen_agent = next(card for card in agent_cards if card.agent_id == chosen["agent_id"])
        assignment = {
            "subtask_id": subtask.subtask_id,
            "capability": subtask.capability,
            "assigned_agent": chosen_agent.agent_id,
            "inputs_passed": _inputs_passed_for_subtask(subtask, context),
            "depends_on": list(subtask.depends_on),
            "candidate_agents": [
                {
                    "agent_id": candidate["agent_id"],
                    "score": candidate["score"],
                    "eligible": candidate["eligible"],
                }
                for candidate in ranked[:3]
            ],
        }
        assignments.append(assignment)
        _record_assignment_context(context, subtask, chosen_agent)

    missing_inputs = _dedupe(missing_inputs)
    if saw_unsupported_capability:
        decision = "unsupported"
    elif unassigned:
        has_missing_agent = _has_missing_agent(unassigned, agent_cards)
        decision = "ask_user" if missing_inputs and not has_missing_agent else "unsupported"
    elif missing_inputs or saw_missing_input_capability:
        decision = "ask_user"
    else:
        decision = "assign"

    result = _binding_plan(decision, assignments, missing_inputs, validation_notes, unassigned)
    result["validation_notes"].extend(
        validate_agent_binding_plan(plan, result, [agent.to_dict() for agent in agent_cards])
    )
    result["validation_notes"] = _dedupe(result["validation_notes"])
    return result


def normalize_agent_card(agent: dict[str, Any] | AgentCard) -> AgentCard:
    if isinstance(agent, AgentCard):
        return agent
    return AgentCard(
        agent_id=str(agent.get("agent_id") or agent.get("id") or agent.get("name") or ""),
        name=str(agent.get("name") or agent.get("agent_id") or ""),
        description=str(agent.get("description") or ""),
        capabilities=tuple(_dedupe(_strings(agent.get("capabilities")))),
        input_contract=_normalize_contract(agent.get("input_contract")),
        output_contract=_normalize_contract(agent.get("output_contract")),
        cannot_do=tuple(_dedupe(_strings(agent.get("cannot_do")))),
        cost=_quality(agent.get("cost"), default="medium"),
        latency=_quality(agent.get("latency"), default="medium"),
        reliability_score=_float_between(agent.get("reliability_score"), 0.0, 1.0, default=0.5),
    )


def extract_ordered_subtasks(
    capability_plan: dict[str, Any] | list[Any] | None,
) -> list[CapabilitySubtask]:
    if isinstance(capability_plan, list):
        raw_caps = capability_plan
    elif isinstance(capability_plan, dict):
        raw_caps = _extract_ordered_capability_dicts(capability_plan)
    else:
        raw_caps = []

    subtasks = []
    for index, item in enumerate(raw_caps, start=1):
        if isinstance(item, str):
            raw = {"capability_name": item}
        elif isinstance(item, dict):
            raw = deepcopy(item)
        else:
            continue
        capability = str(raw.get("capability_name") or raw.get("capability") or "").strip()
        if not capability:
            continue
        subtask_id = str(raw.get("id") or raw.get("subtask_id") or f"s{index}")
        subtasks.append(
            CapabilitySubtask(
                subtask_id=subtask_id,
                capability=capability,
                inputs=tuple(_strings(raw.get("inputs"))),
                outputs=tuple(_strings(raw.get("outputs"))),
                depends_on=tuple(_strings(raw.get("depends_on"))),
                raw=raw,
            )
        )
    return subtasks


def rank_agent_candidates(
    subtask: CapabilitySubtask | dict[str, Any],
    agents: list[dict[str, Any] | AgentCard],
    user_request: str = "",
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    task = _coerce_subtask(subtask)
    agent_cards = [normalize_agent_card(agent) for agent in agents]
    context = context or _initial_assignment_context(user_request)
    available_kinds = _available_input_kinds(user_request, task, context)
    capability_missing = _missing_capability_inputs(task.capability, available_kinds)
    expected_outputs = _expected_output_kinds(task)

    ranked: list[dict[str, Any]] = []
    for agent in agent_cards:
        score = 0.0
        reasons: list[str] = []
        if task.capability in agent.cannot_do:
            ranked.append(
                _candidate(
                    agent,
                    score,
                    False,
                    ["capability is blocked by cannot_do"],
                    capability_missing,
                )
            )
            continue

        if task.capability in agent.capabilities:
            score += 100.0
            reasons.append("exact capability match")
            score += max(0, 8 - _capability_index(agent, task.capability))
        elif _capability_token_overlap(task.capability, agent.capabilities):
            score += 20.0
            reasons.append("related capability wording")
        else:
            ranked.append(
                _candidate(agent, score, False, ["no capability match"], capability_missing)
            )
            continue

        if PREFERRED_AGENT_BY_CAPABILITY.get(task.capability) == agent.agent_id:
            score += 14.0
            reasons.append("preferred agent for capability")
        score += _contextual_preference_bonus(agent.agent_id, task, available_kinds, user_request)

        contract_missing = _missing_contract_inputs(agent, available_kinds)
        missing_inputs = _dedupe(capability_missing + contract_missing)
        if missing_inputs:
            score -= 30.0 * len(missing_inputs)
            reasons.append("missing required inputs")

        produced = set(agent.output_contract.get("produces", ()))
        output_overlap = produced & expected_outputs
        if output_overlap:
            score += 6.0 + len(output_overlap)
            reasons.append("output contract match")

        score += agent.reliability_score * 5.0
        score += INVERSE_QUALITY_VALUES[agent.cost] * 1.5
        score += INVERSE_QUALITY_VALUES[agent.latency] * 1.0
        eligible = not missing_inputs and score >= 75.0
        ranked.append(_candidate(agent, score, eligible, reasons, missing_inputs))

    ranked.sort(key=lambda item: (-item["score"], not item["eligible"], item["agent_id"]))
    return ranked


def validate_agent_binding_plan(
    capability_plan: dict[str, Any] | list[Any] | None,
    assignment_plan: dict[str, Any],
    agents: list[dict[str, Any] | AgentCard],
) -> list[str]:
    subtasks = extract_ordered_subtasks(capability_plan)
    cards = {card.agent_id: card for card in [normalize_agent_card(agent) for agent in agents]}
    expected_ids = {subtask.subtask_id for subtask in subtasks}
    assignments = assignment_plan.get("assignments") or []
    notes: list[str] = []
    seen: set[str] = set()

    for assignment in assignments:
        subtask_id = str(assignment.get("subtask_id") or "")
        capability = str(assignment.get("capability") or "")
        agent_id = str(assignment.get("assigned_agent") or "")
        if subtask_id not in expected_ids:
            notes.append(f"assignment references unknown subtask {subtask_id}")
        if subtask_id in seen:
            notes.append(f"subtask {subtask_id} was assigned more than once")
        seen.add(subtask_id)

        agent = cards.get(agent_id)
        if not agent:
            notes.append(f"assigned agent {agent_id} is not in the registry")
            continue
        if capability in agent.cannot_do:
            notes.append(f"{agent_id} cannot do {capability}")
        if capability not in agent.capabilities:
            notes.append(f"{agent_id} does not advertise {capability}")

    unassigned = set(assignment_plan.get("unassigned_capabilities") or [])
    if assignment_plan.get("assignment_decision") == "assign":
        missing_ids = expected_ids - seen
        if missing_ids:
            notes.append(f"assign decision omitted subtasks: {', '.join(sorted(missing_ids))}")
        if unassigned:
            notes.append("assign decision includes unassigned capabilities")
    return notes


def build_default_agent_registry() -> list[dict[str, Any]]:
    agents = [
        _agent(
            "research_agent",
            "Research Agent",
            "Retrieves stable external sources and prepares source content for downstream use.",
            ["retrieve_external_information", "verify_current_information"],
            requires=["request"],
            produces=["retrieved_content", "external_source"],
            cannot_do=["modify_code", "edit_image", "compute_numeric_result"],
            cost="medium",
            latency="medium",
            reliability=0.9,
        ),
        _agent(
            "current_info_agent",
            "Current Information Agent",
            "Retrieves and verifies time-sensitive information.",
            ["retrieve_current_information", "verify_current_information"],
            requires=["request"],
            produces=["current_information", "retrieved_content", "verification_result"],
            cannot_do=["modify_code", "analyze_provided_dataset"],
            cost="medium",
            latency="medium",
            reliability=0.92,
        ),
        _agent(
            "document_reader_agent",
            "Document Reader Agent",
            "Reads attached documents and extracts their contents.",
            ["extract_information_from_attached_document", "extract_information_from_file"],
            requires=["document"],
            produces=["document_content", "file_content", "text"],
            cannot_do=["retrieve_current_information", "modify_code"],
            cost="low",
            latency="medium",
            reliability=0.88,
        ),
        _agent(
            "pdf_agent",
            "PDF Agent",
            "Handles PDF extraction, splitting, merging, and page-aware document work.",
            ["extract_information_from_attached_document", "combine_files"],
            requires=["pdf"],
            produces=["document_content", "combined_file", "text"],
            cannot_do=["translate_text", "retrieve_current_information"],
            cost="medium",
            latency="medium",
            reliability=0.86,
        ),
        _agent(
            "spreadsheet_agent",
            "Spreadsheet Agent",
            "Reads spreadsheets, preserves formulas, and prepares tabular summaries.",
            ["analyze_provided_dataset", "compute_numeric_result", "prepare_chart_data"],
            requires=["spreadsheet", "structured_data"],
            produces=["analysis_summary", "computed_result", "chart_data"],
            cannot_do=["retrieve_external_information", "modify_code"],
            cost="medium",
            latency="medium",
            reliability=0.87,
        ),
        _agent(
            "data_analysis_agent",
            "Data Analysis Agent",
            "Analyzes structured data, computes metrics, finds outliers, and summarizes trends.",
            [
                "analyze_provided_dataset",
                "compute_numeric_result",
                "identify_outliers",
                "prepare_chart_data",
                "summarize_findings",
            ],
            requires=["structured_data"],
            optional=["metric_request", "chart_type"],
            produces=["analysis_summary", "computed_result", "outlier_report", "chart_data"],
            cannot_do=["retrieve_external_information", "modify_code", "edit_image"],
            cost="low",
            latency="medium",
            reliability=0.91,
        ),
        _agent(
            "chart_agent",
            "Chart Agent",
            "Transforms analysis outputs or datasets into chart-ready specifications.",
            ["prepare_chart_data", "create_chart"],
            requires=["structured_data"],
            produces=["chart_data", "chart"],
            cannot_do=["retrieve_current_information", "modify_code"],
            cost="medium",
            latency="medium",
            reliability=0.85,
        ),
        _agent(
            "code_search_agent",
            "Code Search Agent",
            "Searches repositories for symbols, TODOs, references, and relevant files.",
            ["search_provided_files", "inspect_existing_code"],
            requires=["codebase", "file_path"],
            produces=["file_search_results", "code_context"],
            cannot_do=["modify_code", "execute_code"],
            cost="low",
            latency="low",
            reliability=0.88,
        ),
        _agent(
            "code_inspection_agent",
            "Code Inspection Agent",
            "Inspects existing code and explains relevant implementation context.",
            ["inspect_existing_code", "search_provided_files"],
            requires=["codebase", "file_path"],
            produces=["code_context", "file_search_results"],
            cannot_do=["write_file", "generate_image"],
            cost="low",
            latency="low",
            reliability=0.9,
        ),
        _agent(
            "code_modification_agent",
            "Code Modification Agent",
            "Writes and edits code after repository context has been inspected.",
            ["modify_code", "generate_code", "write_file"],
            requires=["request"],
            produces=["code_changes", "generated_code", "written_file"],
            cannot_do=["retrieve_current_information", "edit_image"],
            cost="medium",
            latency="medium",
            reliability=0.89,
        ),
        _agent(
            "debugging_agent",
            "Debugging Agent",
            "Diagnoses failures from logs, stack traces, and relevant code context.",
            ["diagnose_issue", "inspect_existing_code"],
            requires=["request"],
            produces=["diagnosis", "code_context"],
            cannot_do=["write_file", "generate_image"],
            cost="medium",
            latency="medium",
            reliability=0.86,
        ),
        _agent(
            "test_runner_agent",
            "Test Runner Agent",
            "Executes tests or validation steps and reports whether changes satisfy requirements.",
            ["execute_code", "validate_output_against_requirements"],
            requires=["request"],
            produces=["execution_result", "validation_result"],
            cannot_do=["retrieve_external_information", "draft_text"],
            cost="low",
            latency="medium",
            reliability=0.9,
        ),
        _agent(
            "writer_agent",
            "Writer Agent",
            "Drafts user-facing prose, plans, reports, and explanations.",
            ["draft_text", "report_findings", "synthesize_plan", "compare_texts"],
            requires=["request"],
            produces=["draft_text", "findings_report", "plan", "comparison_summary", "text"],
            cannot_do=["retrieve_current_information", "compute_numeric_result"],
            cost="low",
            latency="low",
            reliability=0.86,
        ),
        _agent(
            "rewriter_agent",
            "Rewriter Agent",
            "Revises supplied text for clarity, tone, format, or concision.",
            ["revise_text_for_clarity", "transform_text_format"],
            requires=["request"],
            produces=["revised_text", "formatted_text", "text"],
            cannot_do=["retrieve_external_information", "compute_numeric_result"],
            cost="low",
            latency="low",
            reliability=0.9,
        ),
        _agent(
            "translator_agent",
            "Translator Agent",
            "Translates provided text while preserving meaning and requested style.",
            ["translate_text"],
            requires=["text", "language"],
            produces=["translated_text", "text"],
            cannot_do=["retrieve_current_information", "modify_code"],
            cost="low",
            latency="low",
            reliability=0.89,
        ),
        _agent(
            "summarizer_agent",
            "Summarizer Agent",
            "Summarizes retrieved, provided, or document-derived content.",
            [
                "summarize_retrieved_content",
                "summarize_document",
                "summarize_provided_text",
                "summarize_findings",
            ],
            requires=["text"],
            produces=["summary", "text"],
            cannot_do=["retrieve_external_information", "modify_code"],
            cost="low",
            latency="low",
            reliability=0.91,
        ),
        _agent(
            "classification_agent",
            "Classification Agent",
            "Classifies provided text into labels or categories.",
            ["classify_provided_text"],
            requires=["text"],
            produces=["classification_result"],
            cannot_do=["retrieve_current_information", "write_file"],
            cost="low",
            latency="low",
            reliability=0.88,
        ),
        _agent(
            "extraction_agent",
            "Extraction Agent",
            "Extracts entities, action items, fields, or structured facts from provided sources.",
            [
                "extract_action_items",
                "extract_information_from_file",
                "extract_information_from_attached_document",
            ],
            requires=["text"],
            produces=["extracted_items", "file_content", "document_content", "text"],
            cannot_do=["retrieve_current_information", "modify_code"],
            cost="low",
            latency="low",
            reliability=0.87,
        ),
        _agent(
            "image_understanding_agent",
            "Image Understanding Agent",
            "Interprets images, screenshots, receipts, diagrams, and visual content.",
            ["extract_information_from_image", "interpret_image_content"],
            requires=["image"],
            produces=["image_extraction", "image_description", "text"],
            cannot_do=["generate_image", "modify_code"],
            cost="medium",
            latency="medium",
            reliability=0.88,
        ),
        _agent(
            "image_generation_agent",
            "Image Generation Agent",
            "Generates or transforms raster images from prompts or source images.",
            ["generate_image", "transform_image"],
            requires=["request"],
            produces=["generated_image", "transformed_image", "image"],
            cannot_do=["interpret_image_content", "retrieve_current_information"],
            cost="high",
            latency="medium",
            reliability=0.84,
        ),
        _agent(
            "file_conversion_agent",
            "File Conversion Agent",
            "Converts, combines, or writes files in requested formats.",
            ["combine_files", "write_file", "transform_text_format"],
            requires=["request"],
            produces=["combined_file", "written_file", "formatted_text", "file"],
            cannot_do=["retrieve_current_information", "analyze_provided_dataset"],
            cost="medium",
            latency="medium",
            reliability=0.85,
        ),
        _agent(
            "tool_executor_agent",
            "Tool Executor Agent",
            "Selects and executes an external action behind an agent boundary.",
            ["select_and_execute_external_action", "report_external_action_result"],
            requires=["request"],
            produces=["external_action_result", "external_action_report"],
            cannot_do=["modify_code", "generate_image"],
            cost="medium",
            latency="medium",
            reliability=0.86,
        ),
        _agent(
            "clarification_agent",
            "Clarification Agent",
            "Asks the user for missing required inputs and blocks unsafe guessing.",
            ["request_missing_input"],
            requires=["request"],
            produces=["missing_input_request"],
            cannot_do=["retrieve_external_information", "modify_code"],
            cost="low",
            latency="low",
            reliability=0.95,
        ),
        _agent(
            "unsupported_request_agent",
            "Unsupported Request Agent",
            "Explains unsupported requests and records why no agent should proceed.",
            ["unsupported_request"],
            requires=["request"],
            produces=["unsupported_notice"],
            cannot_do=["modify_code", "retrieve_external_information", "generate_image"],
            cost="low",
            latency="low",
            reliability=0.93,
        ),
        _agent(
            "finance_research_agent",
            "Finance Research Agent",
            "Near-miss specialist for market data and financial research tasks.",
            ["retrieve_current_information", "retrieve_external_information", "summarize_findings"],
            requires=["request"],
            produces=["current_information", "retrieved_content", "summary"],
            cannot_do=["analyze_provided_dataset", "modify_code"],
            cost="high",
            latency="medium",
            reliability=0.84,
        ),
        _agent(
            "documentation_writer_agent",
            "Documentation Writer Agent",
            "Near-miss specialist for release notes, docs, and code explanations.",
            ["draft_text", "summarize_document", "report_findings"],
            requires=["request"],
            produces=["draft_text", "summary", "findings_report"],
            cannot_do=["modify_code", "execute_code"],
            cost="medium",
            latency="low",
            reliability=0.85,
        ),
    ]
    return [agent.to_dict() for agent in agents]


def _agent(
    agent_id: str,
    name: str,
    description: str,
    capabilities: list[str],
    *,
    requires: list[str] | None = None,
    optional: list[str] | None = None,
    produces: list[str] | None = None,
    cannot_do: list[str] | None = None,
    cost: str = "medium",
    latency: str = "medium",
    reliability: float = 0.5,
) -> AgentCard:
    return AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        capabilities=tuple(capabilities),
        input_contract={
            "requires": tuple(requires or []),
            "optional": tuple(optional or []),
        },
        output_contract={"produces": tuple(produces or [])},
        cannot_do=tuple(cannot_do or []),
        cost=cost,
        latency=latency,
        reliability_score=reliability,
    )


def _coerce_binding_inputs(
    user_request: str | dict[str, Any],
    capability_plan: dict[str, Any] | list[Any] | None,
    agents: list[dict[str, Any] | AgentCard] | None,
) -> tuple[str, dict[str, Any] | list[Any] | None, list[dict[str, Any] | AgentCard]]:
    if isinstance(user_request, dict) and capability_plan is None:
        payload = user_request
        request = str(payload.get("request") or payload.get("user_request") or "")
        plan = payload.get("capability_plan") or payload.get("gold_capability_plan")
        agent_rows = payload.get("agents") if isinstance(payload.get("agents"), list) else None
        return request, plan, agent_rows or build_default_agent_registry()
    return str(user_request), capability_plan, agents or build_default_agent_registry()


def _extract_ordered_capability_dicts(plan: dict[str, Any]) -> list[Any]:
    if isinstance(plan.get("ordered_capabilities"), list):
        return plan["ordered_capabilities"]
    ordered = plan.get("ordered_capability_plan")
    if isinstance(ordered, dict) and isinstance(ordered.get("ordered_capabilities"), list):
        return ordered["ordered_capabilities"]
    passes = plan.get("passes")
    if isinstance(passes, dict):
        ordering_pass = passes.get("capability_ordering") or {}
        if isinstance(ordering_pass, dict):
            parsed = ordering_pass.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("ordered_capabilities"), list):
                return parsed["ordered_capabilities"]
    requirements = plan.get("capability_requirements")
    if isinstance(requirements, dict) and isinstance(requirements.get("capabilities_needed"), list):
        return requirements["capabilities_needed"]
    if isinstance(plan.get("capabilities_needed"), list):
        return plan["capabilities_needed"]
    return []


def _coerce_subtask(subtask: CapabilitySubtask | dict[str, Any]) -> CapabilitySubtask:
    if isinstance(subtask, CapabilitySubtask):
        return subtask
    return extract_ordered_subtasks([subtask])[0]


def _initial_assignment_context(user_request: str) -> dict[str, Any]:
    return {
        "request_kinds": _infer_input_kinds(user_request),
        "assigned_subtasks": set(),
        "subtask_outputs": {},
        "agent_by_subtask": {},
    }


def _record_assignment_context(
    context: dict[str, Any],
    subtask: CapabilitySubtask,
    agent: AgentCard,
) -> None:
    context["assigned_subtasks"].add(subtask.subtask_id)
    context["agent_by_subtask"][subtask.subtask_id] = agent.agent_id
    context["subtask_outputs"][subtask.subtask_id] = _expected_output_kinds(subtask)


def _available_input_kinds(
    user_request: str,
    subtask: CapabilitySubtask,
    context: dict[str, Any],
) -> set[str]:
    kinds = set(context.get("request_kinds") or _infer_input_kinds(user_request))
    kinds.add("request")
    kinds.update(_infer_input_kinds(" ".join(subtask.inputs)))
    kinds.update(_infer_input_kinds(" ".join(subtask.outputs)))
    for dep in subtask.depends_on:
        dep_outputs = context.get("subtask_outputs", {}).get(dep, set())
        kinds.update(dep_outputs)
        if dep_outputs:
            kinds.add(f"{dep}.output")
    if subtask.capability == "request_missing_input":
        kinds.add("missing_input_request")
    for kind in list(kinds):
        kinds.update(KIND_ALIASES.get(kind, ()))
    return kinds


def _missing_capability_inputs(capability: str, available_kinds: set[str]) -> list[str]:
    missing = []
    for group in CAPABILITY_INPUT_REQUIREMENTS.get(capability, ()):
        if not _requirement_group_satisfied(group, available_kinds):
            missing.append("|".join(group))
    return missing


def _missing_contract_inputs(agent: AgentCard, available_kinds: set[str]) -> list[str]:
    missing = []
    for requirement in agent.input_contract.get("requires", ()):
        if _contract_requirement_satisfied(requirement, available_kinds):
            continue
        missing.append(requirement)
    return missing


def _requirement_group_satisfied(requirements: tuple[str, ...], available_kinds: set[str]) -> bool:
    return any(_kind_available(requirement, available_kinds) for requirement in requirements)


def _contract_requirement_satisfied(requirement: str, available_kinds: set[str]) -> bool:
    return _kind_available(requirement, available_kinds)


def _kind_available(requirement: str, available_kinds: set[str]) -> bool:
    if requirement in available_kinds:
        return True
    aliases = KIND_ALIASES.get(requirement, ())
    if any(alias in available_kinds for alias in aliases):
        return True
    return any(requirement in KIND_ALIASES.get(kind, ()) for kind in available_kinds)


def _expected_output_kinds(subtask: CapabilitySubtask) -> set[str]:
    kinds = set(CAPABILITY_OUTPUT_KINDS.get(subtask.capability, ()))
    kinds.update(_infer_input_kinds(" ".join(subtask.outputs)))
    return kinds


def _inputs_passed_for_subtask(subtask: CapabilitySubtask, context: dict[str, Any]) -> list[str]:
    inputs = []
    for dep in subtask.depends_on:
        if dep in context.get("assigned_subtasks", set()):
            inputs.append(f"{dep}.output")
    inputs.extend(subtask.inputs)
    return _dedupe(inputs or ["request"])


def _infer_input_kinds(text: str) -> set[str]:
    lowered = text.lower()
    kinds: set[str] = set()
    if not lowered.strip():
        return kinds
    if re.search(r"https?://\S+", text):
        kinds.update({"url", "external_source"})
    if re.search(r"\b[\w./-]+\.(py|js|ts|tsx|jsx|go|rs|java|rb|php|css|html|sql)\b", lowered):
        kinds.update({"file_path", "codebase"})
    if re.search(r"\b[\w./-]+\.(csv|tsv|xlsx|xls)\b", lowered):
        kinds.update({"file", "attached_file", "structured_data", "spreadsheet"})
    if re.search(r"\b[\w./-]+\.pdf\b", lowered):
        kinds.update({"file", "attached_file", "pdf", "document"})
    if re.search(r"\b[\w./-]+\.(png|jpg|jpeg|gif|webp|tiff)\b", lowered):
        kinds.update({"file", "attached_file", "image"})
    if "," in text and re.search(r"\b[a-z_ ]+,[a-z_ ]+", lowered) and re.search(r"\d", text):
        kinds.update({"structured_data", "csv", "table"})
    keyword_kinds = {
        "article": {"document", "text", "external_query"},
        "attached": {"attached_file", "file"},
        "chart": {"chart_data"},
        "comment": {"text", "document"},
        "comments": {"text", "document"},
        "code": {"codebase", "code_request"},
        "codebase": {"codebase"},
        "csv": {"structured_data", "csv", "table"},
        "current": {"current_info_query"},
        "data": {"structured_data"},
        "dataset": {"structured_data"},
        "diagram": {"image"},
        "document": {"document", "text"},
        "description": {"text", "document"},
        "descriptions": {"text", "document"},
        "draft": {"text", "draft_text"},
        "error": {"error_log"},
        "essay": {"text", "document"},
        "file": {"file"},
        "image": {"image"},
        "latest": {"current_info_query"},
        "memo": {"document", "text"},
        "notes": {"text", "document"},
        "outlier": {"structured_data"},
        "paragraph": {"text", "document"},
        "photo": {"image"},
        "picture": {"image"},
        "pdf": {"pdf", "document", "attached_file"},
        "query": {"external_query"},
        "react": {"codebase", "code_request"},
        "repository": {"codebase"},
        "revenue": {"structured_data", "metric_request"},
        "spreadsheet": {"structured_data", "spreadsheet"},
        "table": {"structured_data", "table"},
        "test": {"codebase"},
        "translate": {"language"},
        "cta": {"text", "document"},
        "url": {"url"},
        "workbook": {"structured_data", "spreadsheet"},
        "write": {"writing_goal"},
    }
    for keyword, values in keyword_kinds.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            kinds.update(values)
    language_pattern = r"\b(to|into)\s+(spanish|french|german|italian|japanese|chinese|hindi)\b"
    if re.search(language_pattern, lowered):
        kinds.add("language")
    if re.search(r"\b(sum|total|average|growth|percent|percentage|calculate|compute)\b", lowered):
        kinds.add("numeric_values")
    if re.search(r"\b(todo|fixme|src/|index\.|component|function|class)\b", lowered):
        kinds.update({"codebase", "file_path"})
    return kinds


def _contextual_preference_bonus(
    agent_id: str,
    subtask: CapabilitySubtask,
    available_kinds: set[str],
    user_request: str,
) -> float:
    capability = subtask.capability
    lowered = user_request.lower()
    bonus = 0.0
    if agent_id == "pdf_agent" and "pdf" in available_kinds:
        bonus += 10.0
    if agent_id == "spreadsheet_agent" and {"spreadsheet", "workbook"} & available_kinds:
        bonus += 9.0
    if agent_id == "chart_agent" and (
        capability in {"prepare_chart_data", "create_chart"} or "chart" in lowered
    ):
        bonus += 10.0
    finance_pattern = r"\b(stock|market|finance|ticker)\b"
    if agent_id == "finance_research_agent" and re.search(finance_pattern, lowered):
        bonus += 8.0
    docs_pattern = r"\b(docs|documentation|release notes)\b"
    if agent_id == "documentation_writer_agent" and re.search(docs_pattern, lowered):
        bonus += 8.0
    return bonus


def _has_missing_agent(unassigned_capabilities: list[str], agents: list[AgentCard]) -> bool:
    advertised = {cap for agent in agents for cap in agent.capabilities}
    return any(capability not in advertised for capability in unassigned_capabilities)


def _candidate(
    agent: AgentCard,
    score: float,
    eligible: bool,
    reasons: list[str],
    missing_inputs: list[str],
) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "score": round(score, 4),
        "eligible": eligible,
        "reasons": reasons,
        "missing_inputs": list(missing_inputs),
    }


def _capability_index(agent: AgentCard, capability: str) -> int:
    try:
        return list(agent.capabilities).index(capability)
    except ValueError:
        return 99


def _capability_token_overlap(capability: str, agent_capabilities: tuple[str, ...]) -> bool:
    cap_tokens = set(capability.split("_"))
    for agent_capability in agent_capabilities:
        agent_tokens = set(agent_capability.split("_"))
        if len(cap_tokens & agent_tokens) >= 2:
            return True
    return False


def _binding_plan(
    decision: str,
    assignments: list[dict[str, Any]],
    missing_inputs: list[str],
    validation_notes: list[str],
    unassigned_capabilities: list[str],
) -> dict[str, Any]:
    return {
        "assignment_decision": decision if decision in ASSIGNMENT_DECISIONS else "unsupported",
        "assignments": assignments,
        "missing_inputs": missing_inputs,
        "validation_notes": validation_notes,
        "unassigned_capabilities": _dedupe(unassigned_capabilities),
    }


def _normalize_contract(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {"requires": (), "optional": ()}
    return {
        "requires": tuple(_dedupe(_strings(value.get("requires")))),
        "optional": tuple(_dedupe(_strings(value.get("optional")))),
        "produces": tuple(_dedupe(_strings(value.get("produces")))),
    }


def _contract_to_dict(contract: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    return {key: list(values) for key, values in contract.items() if values}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _quality(value: Any, *, default: str) -> str:
    text = str(value or default).lower()
    return text if text in QUALITY_VALUES else default


def _float_between(value: Any, minimum: float, maximum: float, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _dedupe(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
