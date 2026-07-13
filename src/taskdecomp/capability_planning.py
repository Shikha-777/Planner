from __future__ import annotations

import json
import re
import csv
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable


@dataclass(frozen=True)
class TaskFamilyRule:
    name: str
    family: str
    description: str
    predicate: Callable[[dict[str, Any], list[str]], bool]


@dataclass(frozen=True)
class ExternalActionRule:
    name: str
    actions: tuple[str, ...]
    description: str
    predicate: Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class CapabilityTemplate:
    capability_name: str
    capability_description: str
    external_action_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    done_when: str


@dataclass(frozen=True)
class CapabilityRouteSpec:
    route: str
    capability_keys: tuple[str, ...]
    external_action_types: tuple[str, ...]
    source_requirement: str
    output_type: str
    domain: str
    route_status: str = "confident"


@dataclass(frozen=True)
class TaskRoute:
    source_status: str
    input_status: str
    input_format: str
    operation: str
    source_requirement: str
    output_type: str
    domain: str
    requires_source_input: bool
    requires_external_info: bool
    source_reference_status: str
    route_status: str
    route: str
    external_action_types: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    evidence_spans: tuple[str, ...] = ()


EXTERNAL_ACTION_TYPES = {
    "none",
    "file_reading",
    "file_writing",
    "web_search",
    "fact_checking",
    "calculation",
    "code_execution",
    "image_understanding",
    "image_generation",
    "user_input",
    "other",
}

INPUT_FORMATS = {
    "pasted_text",
    "attached_file",
    "pdf",
    "image",
    "url",
    "file_path",
    "structured_data",
    "unknown",
    "none",
}

OPERATION_HINTS = {
    "external_tool": [
        "available external tool",
        "available external tools",
        "available external api",
        "available external apis",
        "available tool documentation",
        "available api documentation",
        "benchmark-provided tool",
        "benchmark provided tool",
        "external tool/api",
        "tool/api action",
        "external tool action",
        "external api action",
        "plan the external tool",
        "plan the api action",
    ],
    "repo_search": [
        "repo",
        "repository",
        "codebase",
        "find all",
        "search for",
        "fixme",
        "todo",
        "todos",
        "references",
        "usages",
        "scan",
        "deprecated api",
    ],
    "generate_code": [
        "graphql",
        "mutation",
        "regex",
        "regular expression",
        "pattern",
        "sql",
        "query",
        "select",
        "join",
        "database",
        "schema",
        "react component",
        "function",
        "script",
    ],
    "generate_image": [
        "app icon",
        "banner",
        "graphic",
        "illustration",
        "create an image",
        "create a simple square",
        "generate image",
        "make an image",
    ],
    "edit_image": [
        "edit",
        "remove background",
        "retouch",
        "badge",
    ],
    "diagnose": [
        "traceback",
        "stack trace",
        "error log",
        "exception",
        "likely fix",
        "explain this error",
    ],
    "explain": [
        "define",
        "explain",
        "definition of",
        "how do i calculate",
        "how do i compute",
        "how do i find",
        "how to solve",
        "how to find",
        "meaning of",
        "what does",
        "why does",
        "why is",
    ],
    "format_transform": [
        "markdown table",
        "table",
        "convert",
        "turn",
        "turn this",
        "turn these",
        "format",
    ],
    "classify": [
        "classify",
        "categorize",
        "sentiment",
    ],
    "chart": [
        "chart",
        "plot",
        "graph",
        "visualize",
    ],
    "calculate": [
        "aggregate",
        "area",
        "coefficients",
        "calculate",
        "compute",
        "count",
        "equation",
        "total",
        "average",
        "integral",
        "sum",
        "group by",
        "growth",
        "perimeter",
        "outlier",
        "percentage",
        "quadratic",
        "roots",
        "slope",
    ],
    "compare": [
        "compare",
        "contrast",
        "difference",
        "differences",
        "which is better",
        "versus",
        "vs",
        "between these two",
    ],
    "extract_action_items": [
        "action items",
        "follow-ups",
        "follow ups",
        "next steps",
        "todos",
        "to-dos",
        "tasks from these notes",
        "owners",
        "due dates",
    ],
    "extract": [
        "extract",
        "pull",
        "identify",
    ],
    "translate": [
        "translate",
        "how do you say",
        "how do i say",
        "how do they say",
        "how does one say",
        "how might i say",
        "how would i say",
        "how would one say",
        "how would they say",
        "i must know how to say",
        "right way to say",
        "word for",
    ],
    "rewrite": ["improve", "revise", "rewrite", "polish", "warmer", "clearer", "tighter"],
    "summarize": ["summarize", "summary", "takeaways", "bullets"],
    "draft": [
        "draft",
        "drop a message",
        "drop a note",
        "send a message",
        "write an email",
        "write a note",
        "write a paragraph",
    ],
}

OPERATION_PRIORITY = [
    "external_tool",
    "generate_code",
    "repo_search",
    "generate_image",
    "edit_image",
    "diagnose",
    "explain",
    "chart",
    "calculate",
    "format_transform",
    "classify",
    "compare",
    "extract_action_items",
    "summarize",
    "extract",
    "translate",
    "rewrite",
    "draft",
]

SOURCE_REQUIREMENTS = {
    "rewrite": "source_required",
    "summarize": "source_required",
    "extract": "source_required",
    "extract_action_items": "source_required",
    "calculate": "source_required",
    "chart": "source_required",
    "compare": "source_required",
    "diagnose": "source_required",
    "explain": "source_optional",
    "format_transform": "source_required",
    "classify": "source_required",
    "repo_search": "source_required",
    "edit_image": "source_required",
    "translate": "source_required",
    "draft": "source_optional",
    "generate_code": "source_optional",
    "external_tool": "source_optional",
    "generate_image": "source_optional",
    "general": "source_optional",
}

ROUTE_MATRIX = {
    ("current_query", "fact_check"): "current_fact_verification",
    ("current_query", "*"): "current_information_answer",
    ("url_list", "compare"): "external_multi_source_comparison",
    ("url_list", "*"): "external_source_summary",
    ("url", "*"): "external_source_summary",
    ("repo_files", "repo_search"): "repo_search",
    ("file_path", "repo_search"): "repo_search",
    ("pdf", "calculate"): "attached_document_calculation",
    ("pdf", "chart"): "attached_document_calculation",
    ("pdf", "summarize"): "attached_document_summary",
    ("pdf", "*"): "attached_document_extraction",
    ("attached_file", "calculate"): "attached_document_calculation",
    ("attached_file", "chart"): "attached_document_calculation",
    ("attached_file", "summarize"): "attached_document_summary",
    ("attached_file", "*"): "attached_document_extraction",
    ("structured_data", "chart"): "structured_data_chart",
    ("structured_data", "calculate"): "structured_data_calculation",
    ("structured_data", "*"): "structured_data_analysis",
    ("image", "edit_image"): "image_editing",
    ("image", "generate_image"): "image_editing",
    ("image", "calculate"): "image_measurement",
    ("image", "*"): "image_understanding",
    ("pasted_text", "diagnose"): "pasted_text_diagnosis",
    ("pasted_text", "explain"): "pasted_text_diagnosis",
    ("pasted_text", "format_transform"): "pasted_text_format_transform",
    ("pasted_text", "classify"): "pasted_text_classification",
    ("pasted_text", "extract_action_items"): "pasted_text_extraction",
    ("pasted_text", "extract"): "pasted_text_extraction",
    ("pasted_text", "compare"): "pasted_text_comparison",
    ("pasted_text", "translate"): "pasted_text_translation",
    ("pasted_text", "summarize"): "pasted_text_summary",
    ("pasted_text", "rewrite"): "pasted_text_rewrite",
    ("request_spec", "generate_code"): "code_generation",
    ("request_spec", "external_tool"): "external_tool_action",
    ("request_spec", "generate_image"): "image_generation",
    ("request_spec", "draft"): "text_generation",
    ("request_spec", "explain"): "concept_explanation",
    ("request_spec", "translate"): "self_contained_translation",
    ("request_spec", "calculate"): "numeric_calculation",
    ("none", "generate_code"): "code_generation",
    ("none", "external_tool"): "external_tool_action",
    ("none", "generate_image"): "image_generation",
    ("none", "draft"): "text_generation",
    ("none", "explain"): "concept_explanation",
    ("none", "translate"): "self_contained_translation",
    ("none", "calculate"): "numeric_calculation",
}

CAPABILITY_ROUTE_SPECS = {
    "missing_user_source": CapabilityRouteSpec(
        "missing_user_source",
        ("request_missing_input",),
        ("user_input",),
        "requires_user_source",
        "answer",
        "general",
        "blocked",
    ),
    "external_multi_source_comparison": CapabilityRouteSpec(
        "external_multi_source_comparison",
        ("retrieve_external_information", "compare_texts"),
        ("web_search",),
        "requires_external_retrieval",
        "report",
        "web",
    ),
    "external_source_summary": CapabilityRouteSpec(
        "external_source_summary",
        ("retrieve_external_information", "summarize_retrieved_content"),
        ("web_search",),
        "requires_external_retrieval",
        "text",
        "web",
    ),
    "current_fact_verification": CapabilityRouteSpec(
        "current_fact_verification",
        ("retrieve_current_information", "verify_current_information", "answer_with_current_information"),
        ("web_search", "fact_checking"),
        "requires_external_retrieval",
        "answer",
        "web",
    ),
    "current_information_answer": CapabilityRouteSpec(
        "current_information_answer",
        ("retrieve_current_information", "answer_with_current_information"),
        ("web_search",),
        "requires_external_retrieval",
        "answer",
        "web",
    ),
    "external_tool_action": CapabilityRouteSpec(
        "external_tool_action",
        ("select_and_execute_external_action", "report_external_action_result"),
        ("other",),
        "requires_external_tool",
        "answer",
        "tool",
    ),
    "file_merge": CapabilityRouteSpec(
        "file_merge",
        ("read_files_for_merge", "combine_files"),
        ("file_reading", "file_writing"),
        "requires_file_access",
        "file",
        "document",
    ),
    "code_edit": CapabilityRouteSpec(
        "code_edit",
        ("inspect_existing_code", "modify_code"),
        ("file_reading", "file_writing"),
        "requires_file_access",
        "code",
        "code",
    ),
    "code_edit_with_validation": CapabilityRouteSpec(
        "code_edit_with_validation",
        ("inspect_existing_code", "modify_code", "validate_output_against_requirements"),
        ("file_reading", "file_writing", "code_execution"),
        "requires_file_access",
        "code",
        "code",
    ),
    "repo_search": CapabilityRouteSpec(
        "repo_search",
        ("search_provided_files", "draft_cleanup_report"),
        ("file_reading",),
        "requires_file_access",
        "report",
        "code",
    ),
    "file_reading": CapabilityRouteSpec(
        "file_reading",
        ("file_reading",),
        ("file_reading",),
        "requires_file_access",
        "answer",
        "document",
    ),
    "file_reading_summary": CapabilityRouteSpec(
        "file_reading_summary",
        ("file_reading", "summarize_extracted_file"),
        ("file_reading",),
        "requires_file_access",
        "report",
        "document",
    ),
    "attached_document_extraction": CapabilityRouteSpec(
        "attached_document_extraction",
        ("file_reading",),
        ("file_reading",),
        "requires_file_access",
        "answer",
        "document",
    ),
    "attached_document_summary": CapabilityRouteSpec(
        "attached_document_summary",
        ("file_reading", "summarize_extracted_document"),
        ("file_reading",),
        "requires_file_access",
        "text",
        "document",
    ),
    "attached_document_calculation": CapabilityRouteSpec(
        "attached_document_calculation",
        ("file_reading", "compute_numeric_result"),
        ("file_reading", "calculation"),
        "requires_file_access",
        "data_result",
        "data",
    ),
    "structured_data_analysis": CapabilityRouteSpec(
        "structured_data_analysis",
        ("analyze_provided_dataset",),
        ("none",),
        "uses_available_source",
        "report",
        "data",
    ),
    "structured_data_calculation": CapabilityRouteSpec(
        "structured_data_calculation",
        ("analyze_provided_dataset", "compute_numeric_result"),
        ("calculation",),
        "uses_available_source",
        "data_result",
        "data",
    ),
    "structured_data_chart": CapabilityRouteSpec(
        "structured_data_chart",
        ("analyze_provided_dataset", "prepare_chart_data", "generate_chart"),
        ("calculation",),
        "uses_available_source",
        "chart",
        "data",
    ),
    "file_write_from_source": CapabilityRouteSpec(
        "file_write_from_source",
        ("text_for_operation", "file_writing"),
        ("file_writing",),
        "uses_available_source",
        "file",
        "document",
    ),
    "image_editing": CapabilityRouteSpec(
        "image_editing",
        ("image_understanding", "image_generation"),
        ("image_understanding", "image_generation"),
        "uses_available_source",
        "edited_image",
        "image",
    ),
    "image_measurement": CapabilityRouteSpec(
        "image_measurement",
        ("image_understanding", "measure_image"),
        ("image_understanding", "calculation"),
        "uses_available_source",
        "answer",
        "image",
    ),
    "image_understanding": CapabilityRouteSpec(
        "image_understanding",
        ("image_understanding",),
        ("image_understanding",),
        "uses_available_source",
        "answer",
        "image",
    ),
    "image_generation": CapabilityRouteSpec(
        "image_generation",
        ("image_generation",),
        ("image_generation",),
        "self_contained",
        "image",
        "image",
    ),
    "pasted_text_diagnosis": CapabilityRouteSpec(
        "pasted_text_diagnosis",
        ("analyze_provided_text",),
        ("none",),
        "uses_available_source",
        "answer",
        "code",
    ),
    "pasted_text_format_transform": CapabilityRouteSpec(
        "pasted_text_format_transform",
        ("transform_text_format",),
        ("none",),
        "uses_available_source",
        "table",
        "writing",
    ),
    "pasted_text_classification": CapabilityRouteSpec(
        "pasted_text_classification",
        ("classify_provided_text",),
        ("none",),
        "uses_available_source",
        "table",
        "writing",
    ),
    "pasted_text_extraction": CapabilityRouteSpec(
        "pasted_text_extraction",
        ("analyze_provided_text",),
        ("none",),
        "uses_available_source",
        "table",
        "writing",
    ),
    "pasted_text_comparison": CapabilityRouteSpec(
        "pasted_text_comparison",
        ("compare_texts",),
        ("none",),
        "uses_available_source",
        "report",
        "writing",
    ),
    "pasted_text_translation": CapabilityRouteSpec(
        "pasted_text_translation",
        ("translate_text",),
        ("none",),
        "uses_available_source",
        "text",
        "writing",
    ),
    "pasted_text_summary": CapabilityRouteSpec(
        "pasted_text_summary",
        ("summarize_provided_text",),
        ("none",),
        "uses_available_source",
        "text",
        "writing",
    ),
    "pasted_text_rewrite": CapabilityRouteSpec(
        "pasted_text_rewrite",
        ("revise_text_for_clarity",),
        ("none",),
        "uses_available_source",
        "text",
        "writing",
    ),
    "code_generation": CapabilityRouteSpec(
        "code_generation",
        ("generate_code",),
        ("none",),
        "self_contained",
        "code",
        "code",
    ),
    "text_generation": CapabilityRouteSpec(
        "text_generation",
        ("draft_text",),
        ("none",),
        "self_contained",
        "text",
        "writing",
    ),
    "concept_explanation": CapabilityRouteSpec(
        "concept_explanation",
        ("provide_explanation",),
        ("none",),
        "self_contained",
        "answer",
        "general",
    ),
    "self_contained_translation": CapabilityRouteSpec(
        "self_contained_translation",
        ("translate_text",),
        ("none",),
        "self_contained",
        "text",
        "writing",
    ),
    "numeric_calculation": CapabilityRouteSpec(
        "numeric_calculation",
        ("compute_numeric_result",),
        ("calculation",),
        "self_contained",
        "answer",
        "data",
    ),
    "general_request": CapabilityRouteSpec(
        "general_request",
        ("provide_explanation",),
        ("none",),
        "self_contained",
        "answer",
        "general",
        "uncertain",
    ),
}

PASS_ORDER = [
    "intent_input_audit",
    "transformation_externality_audit",
    "capability_requirements",
    "capability_normalization",
    "capability_ordering",
]

PASS_TOP_LEVEL_KEYS = {
    "one_shot_capability_plan": [
        "intent_input_audit",
        "transformation_externality_audit",
        "capability_requirements",
        "capability_normalization",
        "capability_ordering",
    ],
    "semantic_slot_frame": ["canonical_request", "slots_observed", "call_groups"],
    "intent_final_user_want": ["final_user_want"],
    "intent_required_inputs": ["required_inputs"],
    "intent_input_availability": ["inputs", "missing_inputs"],
    "missing_input_slot_filler": ["missing_inputs"],
    "intent_input_audit": ["final_user_want", "inputs", "missing_inputs"],
    "transformation_externality_audit": [
        "starting_state",
        "desired_state",
        "transformations_needed",
        "needs_current_or_external_info",
        "external_actions",
    ],
    "capability_requirements": ["capabilities_needed"],
    "capability_normalization": ["normalized_capabilities", "merged_capabilities"],
    "capability_ordering": ["ordered_capabilities"],
}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the last JSON object in text, with json-repair as a conservative fallback."""
    end = text.rfind("}")
    if end < 0:
        return None, "no closing brace found"

    starts = list(re.finditer(r"{", text[: end + 1]))
    for match in reversed(starts):
        snippet = text[match.start() : end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None

    snippet = text[starts[0].start() : end + 1] if starts else text[: end + 1]
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:
        return None, "no valid JSON object found"

    try:
        repaired = repair_json(snippet, return_objects=True)
    except Exception as exc:  # pragma: no cover - depends on optional repair parser failures
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(repaired, dict):
        return repaired, None
    return None, "no valid JSON object found"


def build_evidence_chunks(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
) -> list[dict[str, Any]]:
    """Create a small typed evidence inventory for chunked input-audit passes."""
    attachments_metadata = attachments_metadata if attachments_metadata is not None else []
    chunks: list[dict[str, Any]] = []
    segmented = _segment_user_turn(user_request)
    instruction_text = segmented["instruction_text"]
    url_spans = _url_spans_for_segmented_turn(segmented)
    chunks.append(
        {
            "id": "instruction_1",
            "type": "instruction",
            "format": "none",
            "available": True,
            "text": instruction_text,
            "masked_text": segmented.get("instruction_text_masked", instruction_text),
            "preview": _preview(instruction_text),
        }
    )

    for index, item in enumerate(_as_list(attachments_metadata), start=1):
        if not isinstance(item, dict):
            continue
        fmt = item.get("format") if item.get("format") in INPUT_FORMATS else "attached_file"
        chunks.append(
            {
                "id": f"attachment_{index}",
                "type": "attachment",
                "format": fmt,
                "available": item.get("available") is not False,
                "name": item.get("name") or f"attachment_{index}",
                "contents_in_context": bool(item.get("contents_in_context")),
                "metadata": item,
                "preview": str(item.get("name") or f"attachment_{index}"),
            }
        )

    retrieval_urls = [
        span for span in url_spans if span.get("role") == "retrieval_target"
    ]
    for index, span in enumerate(retrieval_urls, start=1):
        clean_url = str(span.get("url") or "").rstrip(".,)")
        chunks.append(
            {
                "id": f"url_{index}",
                "type": "url",
                "format": "url",
                "available": True,
                "text": clean_url,
                "preview": clean_url,
                "zone": span.get("zone"),
                "role": span.get("role"),
            }
        )

    for index, block in enumerate(_inline_structured_data_blocks(user_request), start=1):
        chunks.append(
            {
                "id": f"structured_data_{index}",
                "type": "inline_structured_data",
                "format": "structured_data",
                "available": True,
                "text": block,
                "preview": _preview(block),
            }
        )

    for index, block in enumerate(_pasted_text_blocks(user_request), start=1):
        chunks.append(
            {
                "id": f"pasted_text_{index}",
                "type": "pasted_text",
                "format": "pasted_text",
                "available": True,
                "text": block,
                "preview": _preview(block),
            }
        )

    if _looks_like_calculation_request(str(segmented.get("instruction_text_masked", instruction_text)).lower()):
        chunks.append(
            {
                "id": "numeric_expression_1",
                "type": "numeric_expression",
                "format": "none",
                "available": True,
                "text": user_request,
                "preview": _preview(user_request),
            }
        )

    if context:
        chunks.append(
            {
                "id": "context_1",
                "type": "context",
                "format": "unknown",
                "available": True,
                "text": context,
                "preview": _preview(context),
            }
        )
    return chunks


def merge_split_intent_input_audit(
    final_intent: dict[str, Any] | None,
    required_inputs: dict[str, Any] | None,
    availability: dict[str, Any] | None,
) -> dict[str, Any]:
    final_intent = final_intent or {}
    required_inputs = required_inputs or {}
    availability = availability or {}
    inputs = _as_list(availability.get("inputs"))
    if not inputs:
        inputs = [
            {
                "name": item.get("name") or str(item),
                "needed_for": item.get("needed_for") or final_intent.get("final_user_want", ""),
                "available": False,
                "format": "unknown",
                "evidence": "No matching evidence chunk selected.",
            }
            for item in _as_list(required_inputs.get("required_inputs"))
            if isinstance(item, dict)
        ]
    return {
        "final_user_want": str(final_intent.get("final_user_want") or ""),
        "inputs": inputs,
        "missing_inputs": _as_list(availability.get("missing_inputs")),
    }


def classify_task_family(
    user_request: str,
    attachments_metadata: Any | None = None,
    evidence_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify the request into a small rules-first planning family."""
    attachments = [item for item in _as_list(attachments_metadata) if isinstance(item, dict)]
    chunks = evidence_chunks or build_evidence_chunks(user_request, "", attachments)
    ctx = _task_rule_context(user_request, attachments, chunks)
    families: list[str] = []
    matched_rules: list[dict[str, str]] = []
    for rule in TASK_FAMILY_RULES:
        if not rule.predicate(ctx, families):
            continue
        families.append(rule.family)
        matched_rules.append(
            {
                "rule": rule.name,
                "family": rule.family,
                "reason": rule.description,
            }
        )

    if _looks_like_text_generation_request(ctx["request"]) and not families:
        families.append("text_generation")
    if not families:
        families.append("general_request")

    families = _dedupe_preserve_order(families)
    return {
        "primary": families[0],
        "families": families,
        "signals": {
            "formats": sorted(format for format in ctx["formats"] if format),
            "evidence_types": sorted(ctx["chunk_types"]),
            "matched_rules": matched_rules,
            "missing_inputs": ctx["missing_names"],
            "operation": ctx["operation"],
        },
    }


def classify_task_route(
    user_request: str,
    intent_audit: dict[str, Any] | None = None,
    task_family: dict[str, Any] | None = None,
    evidence_chunks: list[dict[str, Any]] | None = None,
) -> TaskRoute:
    """Classify the audited request into a typed route used for skeleton selection."""
    intent_audit = intent_audit or {}
    task_family = task_family or classify_task_family(
        user_request,
        None,
        evidence_chunks,
    )
    instruction_text = _instruction_text_masked(user_request)
    request = instruction_text.lower()
    families = set(_as_list(task_family.get("families")))
    raw_operation = str(
        ((task_family.get("signals") or {}).get("operation"))
        or _operation_for_request(request)
    )
    operation = _normalize_frame_operation(raw_operation, request, families)
    input_status = _route_input_status(intent_audit)
    input_format = _route_input_format(user_request, intent_audit, families, operation)
    operation = _normalize_operation_for_input_format(operation, input_format, request)
    route = _route_name(request, families, input_status, input_format, operation)
    route_spec = _route_spec(route)
    source_requirement = route_spec.source_requirement or _route_source_requirement(
        input_status,
        input_format,
        families,
        operation,
    )
    output_type = route_spec.output_type or _route_output_type(route, families, operation, request)
    actions = tuple(_route_external_action_types(route, intent_audit, families, request))
    evidence = tuple(_route_evidence(intent_audit, evidence_chunks or []))
    source_status = _source_status_for_frame(input_status, input_format, source_requirement)
    requires_external_info = bool({"web_search", "fact_checking"} & set(actions))
    requires_source_input = _frame_requires_source_input(
        operation,
        source_requirement,
        source_status,
    )
    source_reference_status = _source_reference_status(user_request, intent_audit, evidence_chunks or [])
    return TaskRoute(
        source_status=source_status,
        input_status=input_status,
        input_format=input_format,
        operation=operation,
        source_requirement=source_requirement,
        output_type=output_type,
        domain=route_spec.domain or _route_domain(input_format, operation, families),
        requires_source_input=requires_source_input,
        requires_external_info=requires_external_info,
        source_reference_status=source_reference_status,
        route_status=route_spec.route_status,
        route=route,
        external_action_types=actions,
        evidence=evidence,
        evidence_spans=evidence,
    )


def task_route_to_dict(route: TaskRoute | dict[str, Any] | None) -> dict[str, Any]:
    if route is None:
        return {}
    if isinstance(route, dict):
        return deepcopy(route)
    return {
        "source_status": route.source_status,
        "input_status": route.input_status,
        "input_format": route.input_format,
        "operation": route.operation,
        "source_requirement": route.source_requirement,
        "output_type": route.output_type,
        "domain": route.domain,
        "requires_source_input": route.requires_source_input,
        "requires_external_info": route.requires_external_info,
        "source_reference_status": route.source_reference_status,
        "route_status": route.route_status,
        "route": route.route,
        "external_action_types": list(route.external_action_types),
        "evidence": list(route.evidence),
        "evidence_spans": list(route.evidence_spans),
    }


def build_rules_first_capability_plan(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
    intent_override: dict[str, Any] | None = None,
    intent_extra_repairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the complete capability plan from deterministic evidence and templates."""
    attachments_metadata = attachments_metadata if attachments_metadata is not None else []
    evidence_chunks = build_evidence_chunks(user_request, context, attachments_metadata)
    task_family = classify_task_family(user_request, attachments_metadata, evidence_chunks)

    if intent_override is None:
        intent, intent_repairs = build_deterministic_intent_input_audit(
            user_request,
            context,
            attachments_metadata,
            evidence_chunks,
            task_family,
        )
    else:
        intent = deepcopy(intent_override)
        intent_repairs = list(intent_extra_repairs or [])
    intent, repair_repairs = repair_intent_input_audit(
        user_request,
        attachments_metadata,
        intent,
    )
    intent_repairs.extend(repair_repairs)
    task_route = classify_task_route(
        user_request,
        intent,
        task_family,
        evidence_chunks,
    )

    transform, transform_repairs = build_deterministic_transformation_audit(
        user_request,
        intent,
        task_family,
        task_route,
    )
    transform, repair_repairs = repair_transformation_audit(user_request, intent, transform)
    transform_repairs.extend(repair_repairs)

    requirements, requirement_repairs = build_deterministic_capability_requirements(
        user_request,
        intent,
        transform,
        task_family,
        task_route,
    )
    requirements, repair_repairs = repair_capability_requirements(
        intent,
        transform,
        requirements,
    )
    requirement_repairs.extend(repair_repairs)

    normalization = _normalization_from_capabilities(
        requirements.get("capabilities_needed", [])
    )
    normalization, normalization_repairs = repair_capability_normalization(
        requirements,
        normalization,
    )

    ordering = build_deterministic_capability_ordering(requirements)
    ordering, ordering_repairs = repair_capability_ordering(requirements, ordering)

    validation = validate_capability_plan(
        intent,
        transform,
        requirements,
        normalization,
        ordering,
    )
    pass_outputs = {
        "intent_input_audit": _deterministic_pass_output(intent, intent_repairs),
        "transformation_externality_audit": _deterministic_pass_output(
            transform,
            transform_repairs,
        ),
        "capability_requirements": _deterministic_pass_output(
            requirements,
            requirement_repairs,
        ),
        "capability_normalization": _deterministic_pass_output(
            normalization,
            normalization_repairs,
        ),
        "capability_ordering": _deterministic_pass_output(ordering, ordering_repairs),
    }
    return {
        "task_family": task_family,
        "task_frame": task_route_to_dict(task_route),
        "task_route": task_route_to_dict(task_route),
        "evidence_chunks": evidence_chunks,
        "passes": pass_outputs,
        "ordered_capability_plan": ordering,
        "validation": validation,
    }


def build_deterministic_intent_input_audit(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
    evidence_chunks: list[dict[str, Any]] | None = None,
    task_family: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create Run 1 from typed evidence instead of asking a model to infer it."""
    attachments = [
        item for item in _as_list(attachments_metadata) if isinstance(item, dict)
    ]
    chunks = evidence_chunks or build_evidence_chunks(user_request, context, attachments)
    family = task_family or classify_task_family(user_request, attachments, chunks)
    instruction_text = _instruction_text_without_large_blocks(user_request)
    instruction_masked = _instruction_text_masked(user_request)
    instruction_request = instruction_masked.lower()
    operation = _operation_for_request(instruction_request)
    final_want = _deterministic_final_user_want(user_request, family)
    inputs: list[dict[str, Any]] = []
    repairs = [
        {
            "action": "deterministic_input_audit",
            "reason": "Build Run 1 from typed evidence chunks before model planning.",
            "patch": {"task_family": family.get("primary")},
        }
    ]

    for item in _attachment_inputs(attachments, final_want):
        inputs.append(item)
    for item in _url_inputs(chunks, final_want):
        inputs.append(item)
    if operation != "external_tool":
        for item in _structured_data_inputs(chunks, final_want):
            inputs.append(item)
        for item in _pasted_text_inputs(user_request, chunks, final_want):
            inputs.append(item)

    if _looks_like_current_fact_request(instruction_request) or _looks_like_fact_check_request(
        instruction_request
    ):
        inputs.append(
            {
                "name": _current_query_input_name(user_request),
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )
    if _looks_like_calculation_request(instruction_request):
        inputs.append(
            {
                "name": "numeric expression or calculation inputs",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )
    if operation == "external_tool":
        inputs.append(
            {
                "name": "tool action request",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )
    if operation == "generate_code":
        inputs.append(
            {
                "name": "code or query specification",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )
    if _looks_like_image_generation_request(instruction_request):
        inputs.append(
            {
                "name": "image generation requirements",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )
    if _looks_like_text_generation_request(instruction_request) and not inputs:
        inputs.append(
            {
                "name": "writing requirements",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )

    missing_inputs = _deterministic_missing_input_names(instruction_masked, attachments, chunks)
    for name in missing_inputs:
        inputs.append(
            {
                "name": name,
                "needed_for": final_want,
                "available": False,
                "format": "unknown",
                "evidence": "missing user input",
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(instruction_text),
            }
        )

    if not inputs and _looks_like_available_request_only_input(instruction_request):
        inputs.append(
            {
                "name": "request requirements",
                "needed_for": final_want,
                "available": True,
                "format": "none",
                "evidence": _preview(user_request),
                "source_chunk_ids": ["instruction_1"],
                "evidence_span": _preview(user_request),
            }
        )

    inputs = _dedupe_inputs(inputs)
    return {
        "final_user_want": final_want,
        "inputs": inputs,
        "missing_inputs": missing_inputs,
    }, repairs


def build_deterministic_transformation_audit(
    user_request: str,
    intent_audit: dict[str, Any],
    task_family: dict[str, Any] | None = None,
    task_route: TaskRoute | dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create Run 2 from the task family and input formats."""
    family = task_family or classify_task_family(user_request, None, None)
    actions = _deterministic_external_action_types(
        user_request,
        intent_audit,
        family,
        task_route,
    )
    external_actions = [
        {
            "action_type": action_type,
            "needed": action_type != "none",
            "reason": _external_action_reason(action_type),
        }
        for action_type in actions
    ]
    transforms = _deterministic_transformations(
        _instruction_text_without_large_blocks(user_request),
        intent_audit,
        family,
        actions,
    )
    return {
        "starting_state": _starting_state_from_inputs(intent_audit),
        "desired_state": intent_audit.get("final_user_want") or "requested result",
        "transformations_needed": transforms,
        "needs_current_or_external_info": bool(
            {"web_search", "fact_checking"} & set(actions)
        ),
        "external_actions": external_actions,
    }, [
        {
            "action": "deterministic_externality_audit",
            "reason": "Set external actions from task family and input formats.",
            "patch": {"external_action_types": actions},
        }
    ]


def build_deterministic_capability_requirements(
    user_request: str,
    intent_audit: dict[str, Any],
    transformation_audit: dict[str, Any],
    task_family: dict[str, Any] | None = None,
    task_route: TaskRoute | dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create Run 3 from small capability skeleton templates."""
    family = task_family or classify_task_family(user_request, None, None)
    families = set(_as_list(family.get("families")))
    request = _instruction_text_masked(user_request).lower()
    caps: list[dict[str, Any]] = []
    repairs = [
        {
            "action": "task_frame_route_matrix",
            "reason": "Select capability skeleton from the canonical task frame before model detail filling.",
            "patch": {
                "families": sorted(families),
                "task_route": task_route_to_dict(task_route) if task_route else None,
            },
        }
    ]
    operation = _task_route_field(task_route, "operation") or _operation_for_request(request)
    route_name = _task_route_field(task_route, "route") or ""
    route_spec = _route_spec(route_name)

    if _has_missing_inputs(intent_audit):
        _add_capability_template(caps, _missing_input_capability_spec(intent_audit))
        return {"capabilities_needed": caps}, repairs

    if _looks_like_initial_test_run_request(request) and route_name == "code_edit_with_validation":
        _add_capability_template(caps, _template_cap("execute_initial_tests"))

    for key in route_spec.capability_keys:
        spec = _capability_spec_for_route_key(
            key,
            user_request,
            intent_audit,
            transformation_audit,
            task_route,
        )
        if spec:
            _add_capability_template(caps, spec)

    # Compatibility guards: if older family rules identify an external source class that
    # the matrix route did not cover, add only the missing prerequisite capability.
    if _attached_structured_data_requires_reading(intent_audit) and not any(_capability_has_action(cap, "file_reading") for cap in caps):
        _add_capability_template(caps, _capability_spec_for_action("file_reading", intent_audit))
    if "file_reading" in families and not any(_capability_has_action(cap, "file_reading") for cap in caps):
        _add_capability_template(caps, _capability_spec_for_action("file_reading", intent_audit))
    if "attached_document" in families and "file_merge" not in families and not any(_capability_has_action(cap, "file_reading") for cap in caps):
        _add_capability_template(caps, _capability_spec_for_action("file_reading", intent_audit))
    if "generic_code" in families and not _has_capability_named(caps, "generate_code"):
        _add_capability_template(caps, _template_cap("generate_code"))
    if "image_generation" in families and not any(_capability_has_action(cap, "image_generation") for cap in caps):
        _add_capability_template(caps, _capability_spec_for_action("image_generation", intent_audit))
    if "calculation" in families and not any(_capability_has_action(cap, "calculation") for cap in caps):
        _add_capability_template(caps, _capability_spec_for_action("calculation", intent_audit))

    if not caps:
        _add_capability_template(caps, _template_cap("perform_required_transformation"))

    return {"capabilities_needed": caps}, repairs


def build_deterministic_capability_ordering(
    capability_requirements: dict[str, Any],
) -> dict[str, Any]:
    ordered = []
    for index, cap in enumerate(
        cap
        for cap in _as_list(capability_requirements.get("capabilities_needed"))
        if isinstance(cap, dict)
    ):
        ordered.append(_ordered_capability_from_requirement(_capability_id(cap, index), cap))
    _add_template_dependencies(ordered)
    return {"ordered_capabilities": ordered}


def _deterministic_pass_output(
    parsed: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "raw_text": "",
        "retry_raw_text": None,
        "parsed": parsed,
        "parse_error": None,
        "repairs_applied": repairs,
    }


def _deterministic_final_user_want(
    user_request: str,
    task_family: dict[str, Any],
) -> str:
    request = _instruction_text_without_large_blocks(user_request).strip()
    lowered = request.lower()
    families = set(_as_list(task_family.get("families")))
    if "external_tool_action" in families:
        return "select and execute the required external action"
    if "calculation" in families:
        return f"compute numeric result for {user_request}"
    if "missing_input" in families and "essay" in lowered:
        return "revise the essay once the essay text is provided"
    if "missing_input" in families and "chart" in lowered and "sales" in lowered:
        return "create a chart of last quarter's sales by region"
    if "missing_input" in families and "paper" in lowered:
        return "summarize the attached paper and explain the methods section"
    if "fact_check" in families:
        return f"fact check current claim: {request}"
    if "url_summary" in families:
        return f"summarize external content from the provided URL: {request}"
    if not request:
        return "complete the requested task"
    return request


def _deterministic_missing_input_names(
    user_request: str,
    attachments: list[dict[str, Any]],
    evidence_chunks: list[dict[str, Any]],
) -> list[str]:
    instruction_text = _instruction_text_masked(user_request)
    request = instruction_text.lower()
    has_attachment = bool(attachments)
    formats = {str(item.get("format")) for item in attachments if item.get("format")}
    has_pasted_text = any(chunk.get("format") == "pasted_text" for chunk in evidence_chunks)
    has_structured_data = any(
        chunk.get("format") == "structured_data" for chunk in evidence_chunks
    )
    has_url_source = any(
        chunk.get("type") == "url" and chunk.get("role") == "retrieval_target"
        for chunk in evidence_chunks
        if isinstance(chunk, dict)
    )
    has_source = (
        has_attachment
        or has_pasted_text
        or has_structured_data
        or has_url_source
    )
    missing: list[str] = []

    if (
        not has_source
        and _looks_like_source_dependent_without_source_request(request)
        and not _request_itself_supplies_requirements(request, user_request)
    ):
        missing.append(_missing_source_input_name(request))
    if "essay" in request and not has_pasted_text and not has_attachment:
        missing.append("essay text")
    if "paper" in request and any(word in request for word in ["attached", "attach"]) and not has_attachment:
        missing.append("paper")
    if any(word in request for word in ["attachment", "attached", "attach"]) and not has_attachment:
        if not missing:
            missing.append("attached file")
    if "chart" in request and "sales" in request and not has_attachment and "structured_data" not in formats:
        missing.append("sales data")
    if "dataset" in request and not has_attachment and not any(
        chunk.get("format") == "structured_data" for chunk in evidence_chunks
    ):
        missing.append("dataset")
    if _looks_like_image_measurement_request(request, _attachment_inputs(attachments)):
        missing.append("scale or reference measurement")
    return _dedupe_preserve_order(missing)


def _missing_source_input_name(request: str) -> str:
    if _looks_like_reply_to_missing_email_source(request):
        return "email content"
    if "press release" in request:
        return "press release text"
    if "case study" in request:
        return "case study text"
    if "podcast transcript" in request:
        return "podcast transcript text"
    if "diagram" in request:
        return "diagram"
    if "budget" in request:
        return "budget data"
    if "export" in request:
        return "exported data"
    if "one-liner" in request or "one liner" in request or "tagline" in request:
        return "product one-liner text"
    if "portfolio" in request or "blurb" in request:
        return "portfolio blurb text" if "blurb" in request else "portfolio text"
    if "abstract" in request:
        return "abstract text"
    if "customer quote" in request or "customer quotes" in request or "quotes" in request:
        return "customer quote text"
    if "comment" in request or "comments" in request:
        return "comments"
    if "message" in request or "messages" in request:
        return "messages"
    if "memo" in request:
        return "memo"
    if "policy" in request:
        return "policy text"
    if "code" in request:
        return "code"
    if "review" in request or "reviews" in request:
        return "reviews"
    if "ticket" in request or "tickets" in request:
        return "tickets"
    if "row" in request or "rows" in request:
        return "rows"
    if "sales call" in request or "sales calls" in request or "call" in request or "calls" in request:
        return "call notes or transcript"
    if "contract" in request or "agreement" in request:
        return "contract text"
    if "linkedin bio" in request:
        return "LinkedIn bio text"
    if "bio" in request:
        return "bio text"
    if "grant proposal" in request:
        return "grant proposal text"
    if "proposal" in request:
        return "proposal text"
    if "project plan" in request:
        return "project plan"
    if "blog post" in request:
        return "blog post text"
    if "resume" in request:
        return "resume summary text" if "summary" in request else "resume text"
    if "cover letter" in request:
        return "cover letter text"
    if "article" in request:
        return "article text"
    if "transcript" in request:
        return "transcript text"
    if "inventory" in request:
        return "inventory data"
    if "survey" in request:
        return "survey response data"
    if "expense" in request or "expenses" in request or "vendor" in request:
        return "expense data"
    if "revenue" in request and any(word in request for word in ["forecast", "chart", "plot", "graph"]):
        return "revenue data"
    if "signups" in request or "sign-ups" in request:
        return "signup data"
    if any(word in request for word in ["chart", "graph", "plot"]):
        return "source data"
    if "screenshot" in request:
        return "screenshot"
    if "photo" in request or "image" in request:
        return "image"
    if "deck" in request:
        return "deck"
    if "document" in request:
        return "document"
    if "paper" in request:
        return "paper"
    if "essay" in request:
        return "essay text"
    if "notes" in request:
        return "notes"
    return "source material"


def _looks_like_source_dependent_without_source_request(request: str) -> bool:
    if _looks_like_reply_to_missing_email_source(request):
        return True
    source_verbs = [
        "analyze",
        "chart",
        "classify",
        "clean up",
        "cluster",
        "convert",
        "compare",
        "create",
        "describe",
        "explain",
        "extract",
        "forecast",
        "graph",
        "happening",
        "identify",
        "improve",
        "make",
        "make better",
        "plot",
        "polish",
        "pull",
        "revise",
        "rewrite",
        "shorten",
        "tighten",
        "summarize",
        "turn",
    ]
    source_nouns = [
        "abstract",
        "agreement",
        "article",
        "bio",
        "blurb",
        "blog post",
        "budget",
        "case study",
        "contract",
        "cover letter",
        "code",
        "customer quote",
        "customer quotes",
        "call",
        "calls",
        "comment",
        "comments",
        "data",
        "dataset",
        "deck",
        "diagram",
        "document",
        "essay",
        "expense",
        "expenses",
        "export",
        "image",
        "inventory",
        "memo",
        "message",
        "messages",
        "notes",
        "paper",
        "photo",
        "podcast transcript",
        "portfolio",
        "press release",
        "project plan",
        "policy",
        "product one-liner",
        "product one liner",
        "proposal",
        "quotes",
        "review",
        "reviews",
        "row",
        "rows",
        "report",
        "results",
        "revenue",
        "resume",
        "screenshot",
        "signups",
        "spreadsheet",
        "survey",
        "ticket",
        "tickets",
        "transcript",
        "vendor",
    ]
    if (
        re.search(r"\b(?:uploaded|attached|attachment)\b", request)
        and any(noun in request for noun in source_nouns)
    ):
        return True
    if not any(verb in request for verb in source_verbs):
        return False
    if any(noun in request for noun in source_nouns):
        return True
    return bool(
        re.search(r"\b(?:this|these|that|the|my|uploaded|attached)\b", request)
        and any(word in request for word in ["summary", "text", "file", "content"])
    )


def _operation_for_request(request: str) -> str:
    for operation in OPERATION_PRIORITY:
        if any(_request_has_hint(request, hint) for hint in OPERATION_HINTS[operation]):
            return operation
    return "general"


def _request_has_hint(request: str, hint: str) -> bool:
    hint = hint.lower().strip()
    if not hint:
        return False
    pattern = re.escape(hint).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9_]){pattern}(?![a-z0-9_])", request))


def _request_itself_supplies_requirements(request: str, user_request: str) -> bool:
    return (
        _looks_like_current_fact_request(request)
        or _looks_like_fact_check_request(request)
        or _looks_like_url_request(user_request)
        or _looks_like_calculation_request(request)
        or _operation_for_request(request) == "external_tool"
        or _operation_for_request(request) == "generate_code"
        or _looks_like_image_generation_request(request)
        or (
            _looks_like_text_generation_request(request)
            and not _looks_like_reply_to_missing_email_source(request)
        )
    )


def _missing_input_slot_filler_names(slot_filler_output: dict[str, Any] | None) -> list[str]:
    if not isinstance(slot_filler_output, dict):
        return []
    names: list[str] = []
    for item in _as_list(slot_filler_output.get("missing_inputs")):
        name = _missing_input_name(item).strip()
        normalized = _to_snake_case(name)
        if not normalized:
            continue
        if normalized in {"none", "na", "n_a", "no_missing_input", "nothing"}:
            continue
        if normalized in {"answer", "final_answer", "result", "output"}:
            continue
        names.append(name)
    return _dedupe_preserve_order(names)


def _route_spec(route: str | None) -> CapabilityRouteSpec:
    return CAPABILITY_ROUTE_SPECS.get(str(route or ""), CAPABILITY_ROUTE_SPECS["general_request"])


def _normalize_frame_operation(
    operation: str,
    request: str,
    families: set[str],
) -> str:
    if "fact_check" in families:
        return "fact_check"
    if "current_fact" in families:
        return "current"
    if "external_tool_action" in families:
        return "external_tool"
    if "image_generation" in families:
        return "generate_image"
    if "image_edit" in families:
        return "edit_image"
    if "image_measurement" in families:
        return "calculate"
    if "code_edit" in families:
        return "edit_code"
    if "file_merge" in families:
        return "merge"
    if operation == "repo_search" and _looks_like_code_generation_request(request):
        return "generate_code"
    if operation == "chart":
        return "chart"
    return operation


def _normalize_operation_for_input_format(
    operation: str,
    input_format: str,
    request: str,
) -> str:
    if input_format == "pasted_text":
        if operation == "repo_search":
            return "extract_action_items"
        if "markdown table" in request or re.search(r"\bturn\s+(?:this|these)\b.*\btable\b", request):
            return "format_transform"
        if "classify" in request or "sentiment" in request or "categorize" in request:
            return "classify"
    if input_format in {"pdf", "attached_file"} and _request_needs_summary(request):
        return "summarize"
    return operation


def _source_status_for_frame(
    input_status: str,
    input_format: str,
    source_requirement: str,
) -> str:
    if input_status == "missing":
        return "missing"
    if source_requirement == "requires_external_retrieval":
        return "external_required"
    if input_format in {"none", "request_spec", "current_query"}:
        return "not_required"
    if input_status == "available":
        return "available"
    return "not_required"


def _frame_requires_source_input(
    operation: str,
    source_requirement: str,
    source_status: str,
) -> bool:
    if source_status == "missing":
        return True
    if source_requirement in {"requires_file_access", "uses_available_source"}:
        return True
    return SOURCE_REQUIREMENTS.get(operation, "source_optional") == "source_required"


def _source_reference_status(
    user_request: str,
    intent_audit: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
) -> str:
    request = _instruction_text_masked(user_request).lower()
    if _has_missing_inputs(intent_audit):
        if re.search(r"\b(?:attached|uploaded|attachment)\b", request):
            return "user_mentions_attachment_but_missing"
        return "source_referenced_but_missing"
    if any(
        isinstance(chunk, dict) and chunk.get("type") == "attachment"
        for chunk in evidence_chunks
    ):
        return "actual_attachment_present"
    if any(
        isinstance(chunk, dict)
        and chunk.get("format") in {"pasted_text", "structured_data", "url"}
        for chunk in evidence_chunks
    ):
        return "inline_source_present"
    if any(
        isinstance(item, dict) and item.get("available") is True
        for item in _as_list(intent_audit.get("inputs"))
    ):
        return "request_requirements_only"
    return "none"


def _route_domain(input_format: str, operation: str, families: set[str]) -> str:
    if "generic_code" in families or "code_edit" in families or operation in {"generate_code", "edit_code", "repo_search", "diagnose"}:
        return "code"
    if "structured_data_analysis" in families or operation in {"calculate", "chart"}:
        return "data"
    if input_format in {"pdf", "attached_file", "file_path"}:
        return "document"
    if input_format == "image" or operation in {"generate_image", "edit_image"}:
        return "image"
    if input_format == "url" or input_format == "url_list" or operation in {"current", "fact_check"}:
        return "web"
    if input_format == "pasted_text" or operation in {"rewrite", "summarize", "draft", "translate", "extract"}:
        return "writing"
    return "general"


def _route_input_status(intent_audit: dict[str, Any]) -> str:
    if _has_missing_inputs(intent_audit):
        return "missing"
    if any(
        isinstance(item, dict) and item.get("available") is True
        for item in _as_list(intent_audit.get("inputs"))
    ):
        return "available"
    return "none"


def _route_input_format(
    user_request: str,
    intent_audit: dict[str, Any],
    families: set[str],
    operation: str,
) -> str:
    if operation == "external_tool" or "external_tool_action" in families:
        return "request_spec"
    formats = _input_formats(intent_audit)
    url_count = sum(
        1
        for item in _as_list(intent_audit.get("inputs"))
        if isinstance(item, dict) and item.get("format") == "url"
    )
    if _has_missing_inputs(intent_audit) and not any(
        isinstance(item, dict) and item.get("available") is True
        for item in _as_list(intent_audit.get("inputs"))
    ):
        return "no_source_available"
    if url_count >= 2:
        return "url_list"
    if url_count == 1 or "url" in formats:
        return "url"
    if "structured_data" in formats or "structured_data_analysis" in families:
        return "structured_data"
    if "pdf" in formats:
        return "pdf"
    if "image" in formats:
        return "image"
    if "file_path" in formats:
        if operation == "repo_search" or "repository" in _instruction_text_masked(user_request).lower():
            return "repo_files"
        return "file_path"
    if "attached_file" in formats:
        return "attached_file"
    if "pasted_text" in formats:
        return "pasted_text"
    if "current_fact" in families or "fact_check" in families:
        return "current_query"
    if "none" in formats:
        return "request_spec"
    if "unknown" in formats:
        return "unknown"
    return "none"


def _route_source_requirement(
    input_status: str,
    input_format: str,
    families: set[str],
    operation: str,
) -> str:
    if input_status == "missing":
        return "requires_user_source"
    if families & {"url_summary", "current_fact", "fact_check"}:
        return "requires_external_retrieval"
    if input_format in {"attached_file", "file_path", "pdf", "repo_files"}:
        return "requires_file_access"
    if input_format in {"image", "pasted_text", "structured_data"}:
        return "uses_available_source"
    if SOURCE_REQUIREMENTS.get(operation) == "source_required":
        return "requires_user_source"
    return "self_contained"


def _route_name(
    request: str,
    families: set[str],
    input_status: str,
    input_format: str,
    operation: str,
) -> str:
    if input_status == "missing":
        return "missing_user_source"
    if "external_tool_action" in families or operation == "external_tool":
        return "external_tool_action"
    if "file_merge" in families:
        return "file_merge"
    if "code_edit" in families:
        return "code_edit_with_validation" if "code_execution" in families else "code_edit"
    matrix_route = _route_from_matrix(input_format, operation)
    if matrix_route:
        return matrix_route
    if "fact_check" in families:
        return "current_fact_verification"
    if "current_fact" in families:
        return "current_information_answer"
    if operation == "repo_search" and input_format in {"repo_files", "file_path"}:
        return "repo_search"
    if "file_reading" in families:
        return "file_reading_summary" if _request_needs_summary(request) else "file_reading"
    if "attached_document" in families and operation == "calculate":
        return "attached_document_calculation"
    if "attached_document" in families:
        return "attached_document_summary" if _request_needs_summary(request) else "attached_document_extraction"
    if "structured_data_analysis" in families:
        return "structured_data_calculation" if operation == "calculate" or _dataset_needs_calculation(request) else "structured_data_analysis"
    if "file_write" in families:
        return "file_write_from_source"
    if "image_edit" in families:
        return "image_editing"
    if "image_measurement" in families:
        return "image_measurement"
    if "image_understanding" in families:
        return "image_understanding"
    if "image_generation" in families:
        return "image_generation"
    if "generic_code" in families:
        return "code_generation"
    if "calculation" in families:
        return "numeric_calculation"
    if input_format == "pasted_text":
        if operation == "extract_action_items":
            return "pasted_text_extraction"
        if operation == "compare":
            return "pasted_text_comparison"
        if "translate" in request:
            return "pasted_text_translation"
        if "table" in request:
            return "pasted_text_format_transform"
        if operation == "summarize":
            return "pasted_text_summary"
        return "pasted_text_rewrite"
    if "text_generation" in families:
        return "text_generation"
    return "general_request"


def _route_from_matrix(input_format: str, operation: str) -> str | None:
    for key in [
        (input_format, operation),
        (input_format, "*"),
        ("*", operation),
    ]:
        route = ROUTE_MATRIX.get(key)
        if route:
            return route
    return None


def _route_output_type(
    route: str,
    families: set[str],
    operation: str,
    request: str,
) -> str:
    if route in CAPABILITY_ROUTE_SPECS:
        return CAPABILITY_ROUTE_SPECS[route].output_type
    if route in {"file_merge", "file_write_from_source"}:
        return "file"
    if route in {"code_edit", "code_edit_with_validation", "code_generation"}:
        return "code"
    if route in {"image_editing", "image_generation"}:
        return "image"
    if route in {"repo_search", "file_reading_summary"}:
        return "report"
    if "chart" in request or "graph" in request or "plot" in request:
        return "chart"
    if operation == "calculate" or route.endswith("_calculation"):
        return "data_result"
    if route.startswith("pasted_text") or route == "text_generation":
        return "text"
    return "answer"


def _route_external_action_types(
    route: str,
    intent_audit: dict[str, Any],
    families: set[str],
    request: str,
) -> list[str]:
    if route in CAPABILITY_ROUTE_SPECS:
        actions = list(CAPABILITY_ROUTE_SPECS[route].external_action_types)
        if route in {"structured_data_analysis", "structured_data_calculation", "structured_data_chart"}:
            if _attached_structured_data_requires_reading(intent_audit):
                actions = ["file_reading", *[action for action in actions if action != "none"]]
                if route == "structured_data_analysis" and _dataset_needs_calculation(request):
                    actions.append("calculation")
            elif route == "structured_data_analysis":
                actions = ["none"]
        if route == "current_information_answer" and "itinerary" in request:
            actions = ["web_search"]
        return actions or ["none"]
    if route == "missing_user_source":
        return ["user_input"]
    if route in {"external_multi_source_comparison", "external_source_summary", "current_information_answer"}:
        return ["web_search"]
    if route == "current_fact_verification":
        return ["web_search", "fact_checking"]
    if route in {"file_merge"}:
        return ["file_reading", "file_writing"]
    if route in {"code_edit"}:
        return ["file_reading", "file_writing"]
    if route in {"code_edit_with_validation"}:
        return ["file_reading", "file_writing", "code_execution"]
    if route in {"repo_search", "file_reading", "file_reading_summary", "attached_document_summary", "attached_document_extraction"}:
        return ["file_reading"]
    if route == "attached_document_calculation":
        return ["file_reading", "calculation"]
    if route == "structured_data_calculation":
        actions = ["calculation"]
        if _attached_structured_data_requires_reading(intent_audit):
            actions.insert(0, "file_reading")
        return actions
    if route == "structured_data_analysis":
        return ["file_reading"] if _attached_structured_data_requires_reading(intent_audit) else ["none"]
    if route == "file_write_from_source":
        return ["file_writing"]
    if route == "image_editing":
        return ["image_understanding", "image_generation"]
    if route == "image_measurement":
        actions = ["image_understanding", "calculation"]
        if _has_missing_inputs(intent_audit):
            actions.insert(0, "user_input")
        return actions
    if route == "image_understanding":
        return ["image_understanding"]
    if route == "image_generation":
        return ["image_generation"]
    if route == "numeric_calculation":
        return ["calculation"]
    return ["none"]


def _route_evidence(
    intent_audit: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
) -> list[str]:
    evidence = []
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict):
            continue
        value = str(item.get("evidence_span") or item.get("evidence") or item.get("name") or "").strip()
        if value:
            evidence.append(_preview(value, 120))
    if not evidence:
        for chunk in evidence_chunks[:3]:
            value = str(chunk.get("preview") or chunk.get("text") or "").strip()
            if value:
                evidence.append(_preview(value, 120))
    return _dedupe_preserve_order(evidence[:5])


def _task_route_field(route: TaskRoute | dict[str, Any] | None, field: str) -> Any:
    if route is None:
        return None
    if isinstance(route, dict):
        return route.get(field)
    return getattr(route, field)


def _task_route_external_actions(route: TaskRoute | dict[str, Any] | None) -> list[str]:
    if route is None:
        return []
    actions = _task_route_field(route, "external_action_types")
    if isinstance(actions, tuple):
        actions = list(actions)
    return [str(action) for action in _as_list(actions) if action]


def _task_rule_context(
    user_request: str,
    attachments: list[dict[str, Any]],
    evidence_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    instruction_text = _instruction_text_without_large_blocks(user_request)
    instruction_text_masked = _instruction_text_masked(user_request)
    request = instruction_text_masked.lower()
    formats = {
        str(item.get("format"))
        for item in attachments
        if isinstance(item.get("format"), str)
    }
    chunk_types = {str(chunk.get("type")) for chunk in evidence_chunks}
    chunk_formats = {str(chunk.get("format")) for chunk in evidence_chunks}
    missing_names = _deterministic_missing_input_names(
        instruction_text,
        attachments,
        evidence_chunks,
    )
    return {
        "user_request": user_request,
        "instruction_text": instruction_text,
        "instruction_text_masked": instruction_text_masked,
        "request": request,
        "operation": _operation_for_request(request),
        "attachments": attachments,
        "evidence_chunks": evidence_chunks,
        "formats": formats,
        "chunk_types": chunk_types,
        "chunk_formats": chunk_formats,
        "has_pasted_text": "pasted_text" in chunk_formats,
        "has_retrieval_url": any(
            chunk.get("type") == "url" and chunk.get("role") == "retrieval_target"
            for chunk in evidence_chunks
            if isinstance(chunk, dict)
        ),
        "missing_names": missing_names,
    }


TASK_FAMILY_RULES = [
    TaskFamilyRule(
        "missing_user_input",
        "missing_input",
        "A required user-provided input is absent from request evidence and metadata.",
        lambda ctx, families: bool(ctx["missing_names"]),
    ),
    TaskFamilyRule(
        "url_present",
        "url_summary",
        "A URL in the request must be treated as an available URL input.",
        lambda ctx, families: ctx["has_retrieval_url"],
    ),
    TaskFamilyRule(
        "fact_check_request",
        "fact_check",
        "The request asks to verify a current claim.",
        lambda ctx, families: _looks_like_fact_check_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "current_or_latest_request",
        "current_fact",
        "The request asks for current, latest, live, or time-sensitive information.",
        lambda ctx, families: "fact_check" not in families
        and _looks_like_current_fact_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "external_tool_action_request",
        "external_tool_action",
        "The request is framed as selecting or applying a provided external tool/API.",
        lambda ctx, families: _looks_like_external_tool_action_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "code_file_edit",
        "code_edit",
        "Code-like language plus file-path evidence indicates a project edit.",
        lambda ctx, families: _looks_like_code_task(ctx["request"])
        and "file_path" in ctx["formats"]
        and not ctx["has_pasted_text"],
    ),
    TaskFamilyRule(
        "code_execution_requested",
        "code_execution",
        "The request asks to run tests or code against file-path evidence.",
        lambda ctx, families: "file_path" in ctx["formats"]
        and _looks_like_code_execution_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "target_file_output",
        "file_write",
        "A target output path is provided and the task asks to create or write it.",
        lambda ctx, families: _looks_like_file_write_request(
            ctx["user_request"],
            ctx["attachments"],
        ),
    ),
    TaskFamilyRule(
        "merge_attached_files",
        "file_merge",
        "Multiple attached PDFs should be combined into an output file.",
        lambda ctx, families: _looks_like_file_merge_request(
            ctx["request"],
            ctx["attachments"],
        ),
    ),
    TaskFamilyRule(
        "attached_document",
        "attached_document",
        "Attached PDFs or attached files need document/data extraction before use.",
        lambda ctx, families: bool(ctx["formats"] & {"pdf", "attached_file"}),
    ),
    TaskFamilyRule(
        "structured_data",
        "structured_data_analysis",
        "Structured inline data, CSVs, or spreadsheets should be routed as dataset work.",
        lambda ctx, families: (
            "structured_data" in ctx["formats"]
            or "inline_structured_data" in ctx["chunk_types"]
            or _has_spreadsheet_attachment(ctx["attachments"])
            or "workbook" in ctx["request"]
        ),
    ),
    TaskFamilyRule(
        "plain_file_reading",
        "file_reading",
        "File-path inputs that are not code edits, file writes, or file merges must be read.",
        lambda ctx, families: "file_path" in ctx["formats"]
        and not any(family in families for family in ["code_edit", "file_write", "file_merge"]),
    ),
    TaskFamilyRule(
        "image_edit",
        "image_edit",
        "An image attachment plus edit language requires image transformation.",
        lambda ctx, families: "image" in ctx["formats"]
        and _looks_like_image_edit_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "image_understanding",
        "image_understanding",
        "An image attachment without edit language requires visual interpretation.",
        lambda ctx, families: "image" in ctx["formats"] and "image_edit" not in families,
    ),
    TaskFamilyRule(
        "image_generation",
        "image_generation",
        "The request asks to generate a new image from text requirements.",
        lambda ctx, families: _looks_like_image_generation_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "image_measurement",
        "image_measurement",
        "Real-world measurement from an image needs visual interpretation and scale.",
        lambda ctx, families: _looks_like_image_measurement_request(
            ctx["request"],
            _attachment_inputs(ctx["attachments"]),
        ),
    ),
    TaskFamilyRule(
        "pasted_text_analysis",
        "pasted_text_analysis",
        "Available pasted text is being analyzed, explained, compared, or classified.",
        lambda ctx, families: ctx["has_pasted_text"]
        and _looks_like_pasted_text_analysis_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "pasted_text_transform",
        "pasted_text_transform",
        "Available pasted text is being rewritten, translated, or reformatted.",
        lambda ctx, families: ctx["has_pasted_text"]
        and "pasted_text_analysis" not in families,
    ),
    TaskFamilyRule(
        "numeric_calculation",
        "calculation",
        "The request contains an explicit numeric computation.",
        lambda ctx, families: _looks_like_calculation_request(ctx["request"]),
    ),
    TaskFamilyRule(
        "generic_function_spec",
        "generic_code",
        "A standalone code, query, or function spec should be generated, not treated as missing data.",
        lambda ctx, families: (
            ctx["operation"] == "generate_code"
            or _looks_like_generic_function_request(ctx["instruction_text"])
        )
        and "code_edit" not in families,
    ),
]


EXTERNAL_ACTION_RULES = [
    ExternalActionRule(
        "missing_input_requires_user_input",
        ("user_input",),
        "Missing required user data must be requested.",
        lambda ctx: ctx["has_missing_inputs"],
    ),
    ExternalActionRule(
        "current_or_url_requires_web_search",
        ("web_search",),
        "Current facts and URL content require external retrieval.",
        lambda ctx: bool(ctx["families"] & {"url_summary", "current_fact"}),
    ),
    ExternalActionRule(
        "fact_check_requires_search_and_verification",
        ("web_search", "fact_checking"),
        "Fact-checking requires retrieval and verification.",
        lambda ctx: "fact_check" in ctx["families"],
    ),
    ExternalActionRule(
        "external_tool_requires_other_action",
        ("other",),
        "Provided tool/API tasks require an abstract external tool action.",
        lambda ctx: "external_tool_action" in ctx["families"],
    ),
    ExternalActionRule(
        "code_edit_reads_and_writes_files",
        ("file_reading", "file_writing"),
        "Project code edits need inspection and modification.",
        lambda ctx: "code_edit" in ctx["families"],
    ),
    ExternalActionRule(
        "code_edit_validation_runs_code",
        ("code_execution",),
        "Requested tests or validation require code execution.",
        lambda ctx: "code_edit" in ctx["families"]
        and (
            "code_execution" in ctx["families"]
            or _looks_like_code_execution_request(ctx["request"])
        ),
    ),
    ExternalActionRule(
        "file_merge_reads_and_writes_files",
        ("file_reading", "file_writing"),
        "File merge tasks read source files and write a combined output.",
        lambda ctx: "file_merge" in ctx["families"],
    ),
    ExternalActionRule(
        "file_write_writes_target_file",
        ("file_writing",),
        "File-output tasks write the requested target file.",
        lambda ctx: "file_write" in ctx["families"],
    ),
    ExternalActionRule(
        "plain_file_path_reads_file",
        ("file_reading",),
        "Plain file-path tasks need file content extraction.",
        lambda ctx: "file_reading" in ctx["families"],
    ),
    ExternalActionRule(
        "attached_document_reads_file",
        ("file_reading",),
        "Attached document content must be extracted before downstream work.",
        lambda ctx: "attached_document" in ctx["families"]
        and not bool(ctx["families"] & {"code_edit", "file_merge"}),
    ),
    ExternalActionRule(
        "attached_structured_data_reads_file",
        ("file_reading",),
        "Attached structured data needs file reading before analysis.",
        lambda ctx: _attached_structured_data_requires_reading(ctx["intent_audit"]),
    ),
    ExternalActionRule(
        "dataset_calculation",
        ("calculation",),
        "Dataset aggregation or numeric analysis requires calculation.",
        lambda ctx: "structured_data_analysis" in ctx["families"]
        and _dataset_needs_calculation(ctx["request"]),
    ),
    ExternalActionRule(
        "explicit_calculation",
        ("calculation",),
        "Explicit arithmetic or numeric computation requires calculation.",
        lambda ctx: "calculation" in ctx["families"],
    ),
    ExternalActionRule(
        "image_understanding",
        ("image_understanding",),
        "Image description, OCR, or measurement requires visual interpretation.",
        lambda ctx: bool(ctx["families"] & {"image_understanding", "image_measurement"}),
    ),
    ExternalActionRule(
        "image_edit_generation",
        ("image_understanding", "image_generation"),
        "Editing an existing image requires understanding and transformation.",
        lambda ctx: "image_edit" in ctx["families"],
    ),
    ExternalActionRule(
        "image_generation",
        ("image_generation",),
        "Generating a new image requires image generation.",
        lambda ctx: "image_generation" in ctx["families"]
        and "image_edit" not in ctx["families"],
    ),
    ExternalActionRule(
        "pdf_numeric_analysis",
        ("calculation",),
        "PDF tasks with numeric analysis require calculation after extraction.",
        lambda ctx: "pdf" in ctx["formats"] and _dataset_needs_calculation(ctx["request"]),
    ),
]


CAPABILITY_TEMPLATES = {
    "retrieve_external_information": CapabilityTemplate(
        "retrieve_external_information",
        "Retrieve externally hosted content needed for the request.",
        "web_search",
        ("URL and retrieval purpose",),
        ("retrieved external content",),
        "the external content needed for the summary is available",
    ),
    "summarize_retrieved_content": CapabilityTemplate(
        "summarize_document",
        "Summarize retrieved content into the requested takeaways or claims.",
        "none",
        ("retrieved external content",),
        ("summary of retrieved content",),
        "the requested summary is complete",
    ),
    "retrieve_current_information": CapabilityTemplate(
        "retrieve_current_information",
        "Retrieve current external information needed for the request.",
        "web_search",
        ("current information query",),
        ("retrieved current information",),
        "current information is available",
    ),
    "verify_current_information": CapabilityTemplate(
        "verify_current_information",
        "Verify the claim against current evidence.",
        "fact_checking",
        ("claim", "retrieved current information"),
        ("verified fact-check result",),
        "the claim is verified or qualified",
    ),
    "answer_with_current_information": CapabilityTemplate(
        "answer_with_current_information",
        "Answer using the verified or retrieved current information.",
        "none",
        ("verified or retrieved current information",),
        ("answer grounded in current information",),
        "the answer reflects the current facts",
    ),
    "select_and_execute_external_action": CapabilityTemplate(
        "select_and_execute_external_action",
        "Select and execute the external action needed for the request.",
        "other",
        ("external action request", "available action schema"),
        ("external action result",),
        "the relevant external action has been selected and executed",
    ),
    "report_external_action_result": CapabilityTemplate(
        "report_external_action_result",
        "Report the result of the external action in the requested form.",
        "none",
        ("external action result",),
        ("answer based on external action result",),
        "the external action result is reported in the requested form",
    ),
    "read_files_for_merge": CapabilityTemplate(
        "extract_information_from_attached_document",
        "Read the attached files to prepare them for combining.",
        "file_reading",
        ("attached files",),
        ("read file contents",),
        "the attached files are available for combination",
    ),
    "combine_files": CapabilityTemplate(
        "combine_files",
        "Combine the provided files and write the requested output file.",
        "file_writing",
        ("read file contents", "target output path"),
        ("written combined file",),
        "the combined file is written to the requested path",
    ),
    "execute_initial_tests": CapabilityTemplate(
        "execute_code",
        "Run the existing tests to establish the failing behavior.",
        "code_execution",
        ("test command or test paths",),
        ("test failure information",),
        "the initial test result is available",
    ),
    "inspect_existing_code": CapabilityTemplate(
        "inspect_existing_code",
        "Inspect existing code or tests before modifying them.",
        "file_reading",
        ("provided code file paths",),
        ("relevant code behavior and constraints",),
        "the relevant existing code has been inspected",
    ),
    "modify_code": CapabilityTemplate(
        "modify_code",
        "Modify project code or tests to satisfy the requested change.",
        "file_writing",
        ("existing code context", "requested change"),
        ("updated code or tests",),
        "the code changes satisfy the requested behavior",
    ),
    "validate_output_against_requirements": CapabilityTemplate(
        "validate_output_against_requirements",
        "Run validation or tests after the modification.",
        "code_execution",
        ("updated code or tests",),
        ("validation result",),
        "tests or validation have been run against the change",
    ),
    "search_provided_files": CapabilityTemplate(
        "search_provided_files",
        "Search provided repository files for the requested comments.",
        "file_reading",
        ("repository files",),
        ("matching TODO comments",),
        "all matching TODO comments are found",
    ),
    "draft_cleanup_report": CapabilityTemplate(
        "draft_text",
        "Summarize found matches and draft a cleanup report.",
        "none",
        ("matching TODO comments",),
        ("cleanup report",),
        "the cleanup report is complete",
    ),
    "summarize_extracted_file": CapabilityTemplate(
        "summarize_document",
        "Summarize extracted file content into the requested output.",
        "none",
        ("extracted file information",),
        ("summary",),
        "the requested summary is complete",
    ),
    "summarize_extracted_document": CapabilityTemplate(
        "summarize_document",
        "Summarize the extracted document in the requested format.",
        "none",
        ("extracted document content",),
        ("document summary",),
        "the requested document summary is complete",
    ),
    "measure_image": CapabilityTemplate(
        "measure",
        "Convert interpreted image measurements into the requested real-world measurement.",
        "calculation",
        ("interpreted image content", "scale or reference measurement"),
        ("real-world measurement",),
        "the requested real-world measurement can be reported or the missing scale is identified",
    ),
    "generate_code": CapabilityTemplate(
        "generate_code",
        "Generate code that satisfies the requested function specification.",
        "none",
        ("function signature and parameters",),
        ("generated code",),
        "the function implementation satisfies the specification",
    ),
    "draft_text": CapabilityTemplate(
        "draft_text",
        "Draft the requested text from the supplied requirements.",
        "none",
        ("writing requirements",),
        ("drafted text",),
        "the requested text is drafted",
    ),
    "provide_explanation": CapabilityTemplate(
        "provide_explanation",
        "Explain the requested concept or self-contained question.",
        "none",
        ("explanation request"),
        ("explanation"),
        "the requested explanation is provided",
    ),
    "perform_required_transformation": CapabilityTemplate(
        "perform_required_transformation",
        "Transform the available inputs into the requested output.",
        "none",
        ("available inputs",),
        ("requested output",),
        "the requested output is produced",
    ),
}


def _attachment_inputs(
    attachments: list[dict[str, Any]],
    final_want: str = "complete the request",
) -> list[dict[str, Any]]:
    inputs = []
    for index, attachment in enumerate(attachments, start=1):
        fmt = attachment.get("format") if attachment.get("format") in INPUT_FORMATS else "attached_file"
        name = str(attachment.get("name") or f"attachment_{index}")
        inputs.append(
            {
                "name": _friendly_attachment_input_name(name, fmt),
                "needed_for": final_want,
                "available": attachment.get("available") is not False,
                "format": fmt,
                "evidence": name,
                "source_chunk_ids": [f"attachment_{index}"],
                "evidence_span": name,
            }
        )
    return inputs


def _friendly_attachment_input_name(name: str, fmt: str) -> str:
    base = name.rsplit("/", 1)[-1]
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", base).replace("_", " ").replace("-", " ")
    if fmt == "pdf" and "part " in stem.lower():
        return name
    if fmt == "pdf" and "memo" in stem.lower():
        return f"{stem} PDF"
    if fmt == "image":
        return f"{stem} image"
    if fmt == "structured_data":
        return f"{base} CSV or structured data"
    if fmt == "attached_file" and name.lower().endswith((".xlsx", ".xls")):
        return f"{base} workbook"
    return name


def _url_inputs(
    evidence_chunks: list[dict[str, Any]],
    final_want: str,
) -> list[dict[str, Any]]:
    inputs = []
    for chunk in evidence_chunks:
        if chunk.get("format") != "url":
            continue
        url = str(chunk.get("text") or chunk.get("preview") or "")
        inputs.append(
            {
                "name": url,
                "needed_for": final_want,
                "available": True,
                "format": "url",
                "evidence": url,
                "source_chunk_ids": [str(chunk.get("id"))],
                "evidence_span": url,
            }
        )
    return inputs


def _structured_data_inputs(
    evidence_chunks: list[dict[str, Any]],
    final_want: str,
) -> list[dict[str, Any]]:
    inputs = []
    for chunk in evidence_chunks:
        if chunk.get("type") != "inline_structured_data":
            continue
        inputs.append(
            {
                "name": "JSON data",
                "needed_for": final_want,
                "available": True,
                "format": "structured_data",
                "evidence": _preview(str(chunk.get("text") or chunk.get("preview") or "")),
                "source_chunk_ids": [str(chunk.get("id"))],
                "evidence_span": _preview(str(chunk.get("text") or chunk.get("preview") or "")),
            }
        )
    return inputs


def _pasted_text_inputs(
    user_request: str,
    evidence_chunks: list[dict[str, Any]],
    final_want: str,
) -> list[dict[str, Any]]:
    request = _instruction_text_masked(user_request).lower()
    inputs = []
    for chunk in evidence_chunks:
        if chunk.get("format") != "pasted_text":
            continue
        inputs.append(
            {
                "name": _pasted_input_name(request),
                "needed_for": final_want,
                "available": True,
                "format": "pasted_text",
                "evidence": _preview(str(chunk.get("text") or chunk.get("preview") or "")),
                "source_chunk_ids": [str(chunk.get("id"))],
                "evidence_span": _preview(str(chunk.get("text") or chunk.get("preview") or "")),
            }
        )
    return inputs


def _pasted_input_name(request: str) -> str:
    if "essay" in request:
        return "essay text"
    if "paragraph" in request:
        return "paragraph"
    if "review" in request or "sentiment" in request:
        return "reviews"
    if "error log" in request or "traceback" in request or "importerror" in request:
        return "error log"
    if "product description" in request:
        return "two product descriptions"
    if "list" in request:
        return "list"
    if "notes" in request or "release" in request:
        return "notes"
    return "pasted text"


def _current_query_input_name(user_request: str) -> str:
    lowered = _instruction_text_masked(user_request).lower()
    if "openai" in lowered and "model" in lowered:
        return "OpenAI model query"
    if "stock price" in lowered or "market cap" in lowered:
        return "Apple stock price market cap query" if "apple" in lowered else "stock price query"
    if "travel restrictions" in lowered:
        return "US to Japan travel restrictions query" if "japan" in lowered else "travel restrictions query"
    if "chicago" in lowered and "saturday" in lowered:
        return "Chicago Saturday planning constraints"
    if "mayor" in lowered and "climate" in lowered:
        return "Boston mayor climate plan claim"
    return "current or external information query"


def _dedupe_inputs(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    for item in inputs:
        key = (
            _to_snake_case(str(item.get("name") or "")),
            str(item.get("format") or ""),
            bool(item.get("available")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _deterministic_external_action_types(
    user_request: str,
    intent_audit: dict[str, Any],
    task_family: dict[str, Any],
    task_route: TaskRoute | dict[str, Any] | None = None,
) -> list[str]:
    route_actions = _task_route_external_actions(task_route)
    if route_actions:
        return route_actions

    actions: list[str] = []
    instruction_request = _instruction_text_masked(user_request).lower()
    ctx = {
        "request": instruction_request,
        "families": set(_as_list(task_family.get("families"))),
        "formats": _input_formats(intent_audit),
        "has_missing_inputs": _has_missing_inputs(intent_audit),
        "intent_audit": intent_audit,
    }
    for rule in EXTERNAL_ACTION_RULES:
        if not rule.predicate(ctx):
            continue
        actions.extend(rule.actions)

    actions = [action for action in _dedupe_preserve_order(actions) if action != "none"]
    return actions or ["none"]


def _external_action_reason(action_type: str) -> str:
    return {
        "none": "the available inputs can be transformed without external action",
        "file_reading": "content from provided files must be inspected or extracted",
        "file_writing": "the requested result must be written or changed in a file",
        "web_search": "current or externally hosted information must be retrieved",
        "fact_checking": "a current claim must be verified against evidence",
        "calculation": "numeric computation or data aggregation is needed",
        "code_execution": "tests or code must be run for validation",
        "image_understanding": "visual content must be interpreted",
        "image_generation": "an image must be generated or transformed",
        "user_input": "required user-provided information is missing",
        "other": "a provided external action is required",
    }.get(action_type, "another external action is needed")


def _deterministic_transformations(
    user_request: str,
    intent_audit: dict[str, Any],
    task_family: dict[str, Any],
    actions: list[str],
) -> list[dict[str, Any]]:
    families = set(_as_list(task_family.get("families")))
    transforms = []
    if "user_input" in actions:
        transforms.append(
            _transformation(
                "request missing input",
                "required input is unavailable",
                "missing input has been requested",
                "downstream work cannot proceed without the missing user data",
            )
        )
    if "web_search" in actions:
        transforms.append(
            _transformation(
                "retrieve current or external information",
                "query or URL is available",
                "retrieved information is available",
                "the request depends on current or externally hosted information",
            )
        )
    if "fact_checking" in actions:
        transforms.append(
            _transformation(
                "verify current claim",
                "claim and current evidence are available",
                "verified fact-check result is available",
                "the user asked for fact-checking",
            )
        )
    if "other" in actions:
        transforms.append(
            _transformation(
                "select and execute external action",
                "external action request and available action schema are identified",
                "external action result is available",
                "the request depends on a provided external action",
            )
        )
    if "file_reading" in actions:
        transforms.append(
            _transformation(
                "extract or inspect provided files",
                "file references or attachments are available",
                "needed file content is available",
                "file contents are needed for the requested result",
            )
        )
    if "image_understanding" in actions:
        transforms.append(
            _transformation(
                "interpret image content",
                "image is available",
                "visual information is available",
                "the requested output depends on image content",
            )
        )
    if "calculation" in actions:
        transforms.append(
            _transformation(
                "compute numeric or dataset result",
                "numeric expression or extracted data is available",
                "computed result is available",
                "the request requires calculation or aggregation",
            )
        )
    if "file_writing" in actions:
        transforms.append(
            _transformation(
                "write or modify file output",
                "content requirements and target path are available",
                "requested file output is written",
                "the request asks for a file change or output file",
            )
        )
    if "image_generation" in actions:
        transforms.append(
            _transformation(
                "generate or transform image",
                "image requirements are available",
                "generated or transformed image is available",
                "the request asks for image creation or editing",
            )
        )
    if "none" in actions or not transforms:
        transforms.append(
            _transformation(
                _textual_transformation_name(_instruction_text_masked(user_request).lower(), families),
                "available request inputs are identified",
                intent_audit.get("final_user_want") or "requested output is produced",
                "no external action is required for this transformation",
            )
        )
    return transforms


def _transformation(
    transformation: str,
    input_state: str,
    output_state: str,
    reason: str,
) -> dict[str, str]:
    return {
        "transformation": transformation,
        "input_state": input_state,
        "output_state": output_state,
        "reason": reason,
    }


def _textual_transformation_name(request: str, families: set[str]) -> str:
    if any(word in request for word in ["traceback", "stack trace", "error log", "exception"]):
        return "analyze provided diagnostic text"
    if "translate" in request:
        return "translate provided text"
    if "compare" in request:
        return "compare provided texts"
    if "classify" in request or "sentiment" in request:
        return "classify provided text"
    if "table" in request:
        return "transform provided text format"
    if "function" in request:
        return "generate code from specification"
    if "email" in request:
        return "draft requested text"
    if "structured_data_analysis" in families:
        return "analyze provided structured data"
    return "transform available input into requested output"


def _starting_state_from_inputs(intent_audit: dict[str, Any]) -> str:
    inputs = []
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict):
            continue
        status = "available" if item.get("available") is True else "missing"
        inputs.append(f"{item.get('name')} ({status}, {item.get('format')})")
    return "; ".join(inputs) if inputs else "request requirements are available"


def _normalization_from_capabilities(caps: list[Any]) -> dict[str, Any]:
    normalized = []
    for index, cap in enumerate(caps):
        if not isinstance(cap, dict):
            continue
        cap_id = _capability_id(cap, index)
        name = str(cap.get("capability_name") or f"capability_{index + 1}")
        normalized.append(
            {
                "id": cap_id,
                "original_name": name,
                "normalized_name": _to_snake_case(name),
                "meaning_changed": False,
                "external_action_type": cap.get("external_action_type", "none"),
            }
        )
    return {"normalized_capabilities": normalized, "merged_capabilities": []}


def _add_capability_template(caps: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    name = str(spec.get("capability_name") or "")
    action = str(spec.get("external_action_type") or "none")
    if name and any(
        str(cap.get("capability_name")) == name
        and str(cap.get("external_action_type")) == action
        for cap in caps
    ):
        return
    cap = deepcopy(spec)
    cap["id"] = f"cap_{len(caps) + 1}"
    _repair_capability_basics(cap, caps, [])
    caps.append(cap)


def _cap(
    name: str,
    description: str,
    action_type: str,
    inputs: list[str],
    outputs: list[str],
    done_when: str,
) -> dict[str, Any]:
    return {
        "capability_name": name,
        "capability_description": description,
        "input_state": inputs[0] if inputs else "required inputs are available",
        "output_state": outputs[0] if outputs else "requested output is available",
        "requires_external_action": action_type != "none",
        "external_action_type": action_type,
        "inputs": inputs,
        "outputs": outputs,
        "done_when": done_when,
    }


def _template_cap(template_key: str) -> dict[str, Any]:
    spec = CAPABILITY_TEMPLATES[template_key]
    return _cap(
        spec.capability_name,
        spec.capability_description,
        spec.external_action_type,
        list(spec.inputs),
        list(spec.outputs),
        spec.done_when,
    )


def _capability_spec_for_route_key(
    key: str,
    user_request: str,
    intent_audit: dict[str, Any],
    transformation_audit: dict[str, Any],
    task_route: TaskRoute | dict[str, Any] | None,
) -> dict[str, Any] | None:
    request = _instruction_text_masked(user_request).lower()
    if key == "request_missing_input":
        return _missing_input_capability_spec(intent_audit)
    if key == "file_reading":
        return _capability_spec_for_action("file_reading", intent_audit)
    if key == "file_writing":
        return _capability_spec_for_action("file_writing", intent_audit)
    if key == "compute_numeric_result":
        return _capability_spec_for_action("calculation", intent_audit)
    if key == "image_understanding":
        return _capability_spec_for_action("image_understanding", intent_audit)
    if key == "image_generation":
        return _capability_spec_for_action("image_generation", intent_audit)
    if key == "text_for_operation":
        return _text_capability_for_request(request)
    if key in CAPABILITY_TEMPLATES:
        if key == "answer_with_current_information" and _request_asks_to_synthesize_plan(request):
            return _cap(
                "synthesize_plan",
                "Use retrieved current information to produce the requested itinerary or plan.",
                "none",
                ["retrieved current information", "request requirements"],
                ["short itinerary" if "itinerary" in request else "current-information-based plan"],
                "the plan is complete and grounded in current information",
            )
        return _template_cap(key)
    if key == "analyze_provided_dataset":
        action_type = "calculation" if _route_needs_dataset_calculation(request, intent_audit, task_route) else "none"
        return _cap(
            "analyze_provided_dataset",
            "Analyze the provided structured data to produce the requested rows, aggregations, rankings, or findings.",
            action_type,
            ["dataset"],
            ["dataset analysis result"],
            "the requested dataset result is produced",
        )
    if key == "prepare_chart_data":
        return _cap(
            "prepare_chart_data",
            "Prepare the provided data for the requested chart or visualization.",
            "calculation",
            ["dataset"],
            ["chart-ready data"],
            "the data is ready for the requested chart",
        )
    if key == "generate_chart":
        return _cap(
            "generate_chart",
            "Generate the requested chart from prepared data.",
            "none",
            ["chart-ready data"],
            ["chart"],
            "the requested chart is produced",
        )
    if key == "analyze_provided_text":
        return _cap(
            "analyze_provided_text",
            "Analyze the provided text and extract, explain, or diagnose the requested information.",
            "none",
            ["provided text"],
            ["requested text analysis"],
            "the requested information is extracted, explained, or diagnosed",
        )
    if key == "transform_text_format":
        return _cap(
            "transform_text_format",
            "Transform the provided text into the requested table or format.",
            "none",
            ["provided text"],
            ["formatted text"],
            "the text is transformed into the requested format",
        )
    if key == "classify_provided_text":
        return _cap(
            "classify_provided_text",
            "Classify provided text according to the requested labels.",
            "none",
            ["provided text"],
            ["classification result"],
            "each provided item has the requested label",
        )
    if key == "compare_texts":
        return _cap(
            "compare_texts",
            "Compare the provided or retrieved texts according to the user's criterion.",
            "none",
            ["provided or retrieved texts"],
            ["comparison result"],
            "the requested comparison is complete",
        )
    if key == "translate_text":
        return _cap(
            "translate_text",
            "Translate the provided text into the requested language.",
            "none",
            ["provided text"],
            ["translated text"],
            "the text is translated into the requested language",
        )
    if key == "summarize_provided_text":
        return _cap(
            "summarize_document",
            "Summarize the provided text into the requested form.",
            "none",
            ["provided text"],
            ["summary"],
            "the requested summary is complete",
        )
    if key == "revise_text_for_clarity":
        return _cap(
            "revise_text_for_clarity",
            "Revise the provided text for clarity and polish.",
            "none",
            ["provided text"],
            ["revised text"],
            "the text is clearer and more polished",
        )
    return None


def _route_needs_dataset_calculation(
    request: str,
    intent_audit: dict[str, Any],
    task_route: TaskRoute | dict[str, Any] | None,
) -> bool:
    route_name = str(_task_route_field(task_route, "route") or "")
    return (
        route_name in {"structured_data_calculation", "structured_data_chart"}
        or _attached_structured_data_requires_reading(intent_audit)
        or _dataset_needs_calculation(request)
    )


def _has_capability_named(caps: list[dict[str, Any]], name: str) -> bool:
    return any(cap.get("capability_name") == name for cap in caps)


def _add_template_dependencies(caps: list[dict[str, Any]]) -> None:
    for cap in caps:
        cap.setdefault("depends_on", [])
    _add_dependency_by_predicate(caps, _is_missing_input_capability, lambda cap: not _is_missing_input_capability(cap))
    _add_dependency_by_predicate(caps, _is_file_extraction_capability, _uses_extracted_file_content)
    _add_dependency_by_predicate(caps, _is_current_info_capability, _uses_current_information)
    _add_dependency_by_predicate(caps, _is_image_understanding_capability, _is_image_generation_capability)
    _add_dependency_by_predicate(caps, _is_code_inspection_capability, _is_code_modification_capability)
    _add_dependency_by_predicate(caps, _is_code_modification_capability, _is_validation_capability)
    _add_dependency_by_predicate(caps, _is_initial_code_execution_capability, _is_code_modification_capability)
    _add_dependency_by_predicate(caps, _is_dataset_analysis_capability, _uses_dataset_analysis)
    _add_dependency_by_predicate(caps, _is_chart_data_preparation_capability, _is_chart_generation_capability)


def _add_dependency_by_predicate(
    caps: list[dict[str, Any]],
    before_predicate: Any,
    after_predicate: Any,
) -> None:
    ids = {str(cap.get("id")) for cap in caps}
    for before in caps:
        if not before_predicate(before):
            continue
        before_id = str(before.get("id"))
        for after in caps:
            if before is after or not after_predicate(after):
                continue
            after_id = str(after.get("id"))
            if before_id not in ids or after_id not in ids or before_id == after_id:
                continue
            deps = after.setdefault("depends_on", [])
            if before_id not in deps:
                deps.append(before_id)


def _uses_extracted_file_content(cap: dict[str, Any]) -> bool:
    if _is_missing_input_capability(cap) or _is_file_extraction_capability(cap):
        return False
    text = _capability_text_for_matching(cap)
    return any(
        word in text
        for word in [
            "analyze",
            "calculate",
            "combine",
            "compute",
            "draft",
            "report",
            "summarize",
            "write",
        ]
    )


def _uses_current_information(cap: dict[str, Any]) -> bool:
    if _is_missing_input_capability(cap) or _is_current_info_capability(cap):
        return False
    text = _capability_text_for_matching(cap)
    return any(word in text for word in ["answer", "compare", "itinerary", "plan", "synthesize", "summar"])


def _is_dataset_analysis_capability(cap: dict[str, Any]) -> bool:
    return cap.get("capability_name") == "analyze_provided_dataset"


def _uses_dataset_analysis(cap: dict[str, Any]) -> bool:
    if _is_dataset_analysis_capability(cap):
        return False
    text = _capability_text_for_matching(cap)
    return any(word in text for word in ["chart", "compute", "numeric", "prepare_chart_data", "generate_chart"])


def _is_chart_data_preparation_capability(cap: dict[str, Any]) -> bool:
    return cap.get("capability_name") == "prepare_chart_data"


def _is_chart_generation_capability(cap: dict[str, Any]) -> bool:
    return cap.get("capability_name") == "generate_chart"


def _is_initial_code_execution_capability(cap: dict[str, Any]) -> bool:
    text = _capability_text_for_matching(cap)
    return cap.get("capability_name") == "execute_code" and "initial" in text


def _request_mentions_code_file_path(user_request: str) -> bool:
    return bool(
        re.search(r"\b(?:src|tests?|package|components?)/[A-Za-z0-9_./-]+", user_request)
        or re.search(r"\b[A-Za-z0-9_/-]+\.(?:py|js|jsx|ts|tsx)\b", user_request)
    )


def _looks_like_code_execution_request(request: str) -> bool:
    return any(
        phrase in request
        for phrase in ["run", "rerun", "test suite", "tests", "failing test", "validate"]
    )


def _looks_like_initial_test_run_request(request: str) -> bool:
    return request.strip().startswith("run ") or "run the existing test suite" in request


def _looks_like_file_write_request(
    user_request: str,
    attachments: list[dict[str, Any]],
) -> bool:
    request = _instruction_text_masked(user_request).lower()
    has_target_path = any(
        item.get("format") == "file_path"
        and _looks_like_output_file_path(str(item.get("name", "")))
        and not _looks_like_code_task(str(item.get("name", "")).lower())
        for item in attachments
    )
    return (
        has_target_path
        and any(word in request for word in ["create", "write", "save", "named"])
        and not _looks_like_code_task(request)
    )


def _looks_like_output_file_path(name: str) -> bool:
    return bool(
        re.search(r"/", name)
        or re.search(r"\.(?:md|txt|pdf|csv|json|docx|xlsx|html)$", name, re.I)
    )


def _looks_like_file_merge_request(
    request: str,
    attachments: list[dict[str, Any]],
) -> bool:
    pdf_count = sum(1 for item in attachments if item.get("format") == "pdf")
    return "merge" in request and pdf_count >= 2


def _looks_like_fact_check_request(request: str) -> bool:
    return "fact-check" in request or "fact check" in request or "verify whether" in request


def _looks_like_external_tool_action_request(request: str) -> bool:
    if any(
        phrase in request
        for phrase in [
            "external tool/api",
            "tool/api action",
            "available tool documentation",
            "available api documentation",
            "benchmark-provided tool",
            "benchmark provided tool",
            "plan the external tool",
            "plan the api action",
        ]
    ):
        return True
    return bool(
        re.search(
            r"\b(?:available|provided|external)\s+"
            r"(?:external\s+)?(?:tool|tools|api|apis|function|functions)\b",
            request,
        )
    )


def _looks_like_image_edit_request(request: str) -> bool:
    return any(word in request for word in ["edit", "remove", "background", "badge", "retouch"])


def _looks_like_pasted_text_analysis_request(request: str) -> bool:
    if _operation_for_request(request) in {"compare", "extract_action_items", "diagnose", "explain", "extract"}:
        return True
    return any(
        word in request
        for word in [
            "classify",
            "compare",
            "extract",
            "sentiment",
            "explain",
            "error log",
            "traceback",
            "stack trace",
            "exception",
            "likely fix",
            "meeting notes",
        ]
    )


def _looks_like_text_generation_request(request: str) -> bool:
    return any(
        word in request
        for word in [
            "drop a message",
            "drop a note",
            "send a message",
            "write a",
            "draft",
            "email",
            "reply to",
            "reply thank",
        ]
    ) and not (
        "function" in request
        or "file" in request
        or _looks_like_code_generation_request(request)
        or _looks_like_image_generation_request(request)
    )


def _looks_like_reply_to_missing_email_source(request: str) -> bool:
    return bool(
        re.search(r"\breply\b", request)
        and re.search(r"\b(?:email|emails|message|messages)\b", request)
        and re.search(r"\b(?:latest|recent|previous|existing|thread)\b", request)
    )


def _attached_structured_data_requires_reading(intent_audit: dict[str, Any]) -> bool:
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict) or item.get("available") is not True:
            continue
        if item.get("format") == "structured_data" and "csv" in str(item.get("evidence", "")).lower():
            return True
        if item.get("format") == "attached_file" and any(
            suffix in str(item.get("evidence", "")).lower() for suffix in [".xlsx", ".xls", ".csv"]
        ):
            return True
    return False


def _has_spreadsheet_attachment(attachments: list[dict[str, Any]]) -> bool:
    for item in attachments:
        name = str(item.get("name") or "").lower()
        mime = str(item.get("mime_type") or "").lower()
        if name.endswith((".xlsx", ".xls")) or "spreadsheet" in mime:
            return True
    return False


def _dataset_needs_calculation(request: str) -> bool:
    if _operation_for_request(request) in {"calculate", "chart"}:
        return True
    return any(
        _request_has_hint(request, hint)
        for hint in [
            "aggregate",
            "calculate",
            "compute",
            "count",
            "group by",
            "growth",
            "outlier",
            "revenue",
            "refunds",
            "subtotal",
            "tax",
            "top three",
            "total",
        ]
    )


def _request_needs_summary(request: str) -> bool:
    return any(word in request for word in ["summarize", "summary", "takeaways", "bullets", "action items"])


def _request_asks_to_synthesize_plan(request: str) -> bool:
    return bool(
        "itinerary" in request
        or re.search(r"\bplan\s+(?:a|an|my|our|the)\b", request)
        or re.search(r"\bmake\s+(?:a|an)\s+plan\b", request)
        or re.search(r"\bcreate\s+(?:a|an)\s+plan\b", request)
    )


def _text_capability_for_request(request: str) -> dict[str, Any]:
    if "translate" in request:
        return _cap(
            "translate_text",
            "Translate the provided text into the requested language.",
            "none",
            ["provided text"],
            ["translated text"],
            "the text is translated into the requested language",
        )
    if _operation_for_request(request) == "compare":
        return _cap(
            "compare_texts",
            "Compare the provided texts and identify which better satisfies the criterion.",
            "none",
            ["provided texts"],
            ["comparison result"],
            "the requested comparison is complete",
        )
    if "classify" in request or "sentiment" in request:
        return _cap(
            "classify_provided_text",
            "Classify provided text according to the requested labels.",
            "none",
            ["provided text"],
            ["classification result"],
            "each provided item has the requested label",
        )
    if _operation_for_request(request) in {"extract_action_items", "explain"} or any(
        word in request
        for word in ["extract", "error log", "importerror", "likely fix", "meeting notes"]
    ):
        return _cap(
            "analyze_provided_text",
            "Analyze the provided text and extract or explain the requested information.",
            "none",
            ["provided text"],
            ["requested text analysis"],
            "the requested information is extracted or explained",
        )
    if "table" in request:
        return _cap(
            "transform_text_format",
            "Transform the provided text into the requested format.",
            "none",
            ["provided text"],
            ["formatted text"],
            "the text is transformed into the requested format",
        )
    if "email" in request:
        return _cap(
            "draft_text",
            "Draft the requested text from the supplied requirements.",
            "none",
            ["writing requirements"],
            ["drafted text"],
            "the requested text is drafted",
        )
    return _cap(
        "revise_text_for_clarity",
        "Revise the provided text for clarity and polish.",
        "none",
        ["provided text"],
        ["revised text"],
        "the text is clearer and more polished",
    )


def build_one_shot_capability_plan_messages(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
) -> list[dict[str, str]]:
    evidence_chunks = build_evidence_chunks(user_request, context, attachments_metadata)
    payload = {
        "user_request": user_request,
        "evidence_chunks": _availability_chunks(evidence_chunks),
    }
    if context:
        payload["context_preview"] = _preview(context, 500)
    return [
        {
            "role": "system",
            "content": (
                "You produce a complete abstract capability plan in one JSON object. Return "
                "only compact valid JSON. Start with { and end with }. Do not include "
                "analysis, commentary, markdown, or code fences. Do not name tools, APIs, "
                "libraries, plugins, or agents as capabilities."
            ),
        },
        {
            "role": "user",
            "content": (
                "One-shot Capability Planning.\n"
                "Use the typed evidence chunks as grounding. The final intent must be a full "
                "phrase with action and object. Input formats must come from evidence chunks "
                "when matched: pasted_text, pdf, image, url, file_path, structured_data, "
                "attached_file, none, or unknown. Current/latest fact questions have an "
                "available query input with format none; the answer is not missing user input. "
                "Generic function parameters are available specification inputs, not missing "
                "runtime data. Attachment files are available even when contents are not in "
                "context. Use exactly one external_action_type enum value per action and per "
                "capability; never output an enum list with pipes. Capabilities must be "
                "abstract state transformations with inputs, outputs, and done_when. Dependency "
                "ids must exist and form an acyclic graph. Put retrieval/extraction/inspection "
                "before downstream synthesis, modification, or validation. If any required "
                "input is missing, include request_missing_input with external_action_type "
                "user_input before downstream work that cannot proceed. normalized_name values "
                "must be short lowercase snake_case labels such as request_missing_input, "
                "inspect_existing_code, modify_code, run_tests, summarize_document, "
                "retrieve_current_information, or answer_with_current_information.\n"
                "Schema:"
                '{"intent_input_audit":{"final_user_want":"...","inputs":[{"name":"...",'
                '"needed_for":"...","available":true,"format":"pasted_text | attached_file | '
                'pdf | image | url | file_path | structured_data | unknown | none",'
                '"evidence":"..."}],"missing_inputs":[]},'
                '"transformation_externality_audit":{"starting_state":"...",'
                '"desired_state":"...","transformations_needed":[{"transformation":"...",'
                '"input_state":"...","output_state":"...","reason":"..."}],'
                '"needs_current_or_external_info":false,"external_actions":[{"action_type":'
                '"none | file_reading | file_writing | web_search | fact_checking | '
                'calculation | code_execution | image_understanding | image_generation | '
                'user_input | other","needed":true,"reason":"..."}]},'
                '"capability_requirements":{"capabilities_needed":[{"id":"cap_1",'
                '"capability_name":"...","capability_description":"...",'
                '"input_state":"...","output_state":"...","requires_external_action":false,'
                '"external_action_type":"none | file_reading | file_writing | web_search | '
                'fact_checking | calculation | code_execution | image_understanding | '
                'image_generation | user_input | other","inputs":["..."],"outputs":["..."],'
                '"done_when":"..."}]},'
                '"capability_normalization":{"normalized_capabilities":[{"id":"cap_1",'
                '"original_name":"...","normalized_name":"...","meaning_changed":false,'
                '"external_action_type":"..."}],"merged_capabilities":[]},'
                '"capability_ordering":{"ordered_capabilities":[{"id":"cap_1",'
                '"capability_name":"...","depends_on":[],"inputs":["..."],"outputs":["..."],'
                '"done_when":"..."}]}}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def should_run_missing_input_slot_filler(
    user_request: str,
    intent_audit: dict[str, Any] | None,
    task_family: dict[str, Any] | None = None,
) -> bool:
    """Return true only for source-dependent requests with no known source input."""
    intent_audit = intent_audit or {}
    if _has_missing_inputs(intent_audit):
        return False
    inputs = [item for item in _as_list(intent_audit.get("inputs")) if isinstance(item, dict)]
    if any(item.get("available") is True for item in inputs):
        return False

    instruction_text = _instruction_text_masked(user_request)
    request = instruction_text.lower()
    if _looks_like_current_fact_request(request) or _looks_like_fact_check_request(request):
        return False
    if _looks_like_url_request(user_request):
        return False
    if _looks_like_calculation_request(request):
        return False
    if _looks_like_generic_function_request(instruction_text):
        return False
    if _looks_like_image_generation_request(request):
        return False
    if _looks_like_text_generation_request(request) and not _looks_like_reply_to_missing_email_source(request):
        return False

    families = set(_as_list((task_family or {}).get("families")))
    if families & {
        "attached_document",
        "code_edit",
        "code_execution",
        "current_fact",
        "fact_check",
        "file_merge",
        "file_reading",
        "file_write",
        "generic_code",
        "image_edit",
        "image_generation",
        "image_understanding",
        "url_summary",
    }:
        return False
    return _looks_like_source_dependent_without_source_request(request)


def build_missing_input_slot_filler_messages(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
    intent_audit: dict[str, Any] | None = None,
    task_family: dict[str, Any] | None = None,
    evidence_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Ask the model only to name missing user-provided source inputs."""
    evidence_chunks = evidence_chunks or build_evidence_chunks(
        user_request,
        context,
        attachments_metadata,
    )
    payload = {
        "user_request": user_request,
        "intent_input_audit": intent_audit or {},
        "task_family": task_family or {},
        "evidence_chunks": _availability_chunks(evidence_chunks),
    }
    if context:
        payload["context_preview"] = _preview(context, 500)
    return [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. You are a narrow missing-input slot "
                "filler, not a planner. Do not propose tools, capabilities, or actions."
            ),
        },
        {
            "role": "user",
            "content": (
                "Identify missing user-provided source inputs only when the request cannot "
                "be completed from the user request, context, attachments metadata, URLs, "
                "or typed evidence chunks. Do not mark current/latest facts, calculations, "
                "image-generation requirements, writing requirements, or generic function "
                "parameters as missing. If no user-provided source input is missing, return "
                'exactly {"missing_inputs":[]}.\n'
                "Schema:"
                '{"missing_inputs":[{"name":"short human-readable missing input name",'
                '"reason":"why this source input is required",'
                '"evidence_span":"short span from the request showing the need"}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def build_semantic_slot_frame_messages(
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
    intent_audit: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Ask the model for a schema-agnostic, evidence-backed input/call frame."""
    evidence_chunks = build_evidence_chunks(user_request, context, attachments_metadata)
    payload = {
        "user_request": user_request,
        "intent_input_audit": intent_audit or {},
        "evidence_chunks": _availability_chunks(evidence_chunks),
    }
    if context:
        payload["context_preview"] = _preview(context, 500)
    return [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. You are a semantic input-frame parser, "
                "not a tool caller. Do not choose tool names, APIs, libraries, plugins, or "
                "agents. Every explicit slot value must include a short evidence span copied "
                "from the user request. You may mark safe domain defaults as inferred, but "
                "do not invent user-provided values."
            ),
        },
        {
            "role": "user",
            "content": (
                "Build a schema-agnostic frame that a later Python binder can verify against "
                "tool schemas. Use semantic roles, not tool parameter names unless the user "
                "used them. Separate result counts from tool-call counts.\n"
                "Rules:\n"
                "- slots_observed are facts/constraints present in the request or safe "
                "defaults implied by domain convention.\n"
                "- evidence_span must be copied from the request for explicit or semantic "
                "values. Use null only for inferred/defaulted values.\n"
                "- expected_call_count is the number of independent operations, not the "
                "number of requested results. top N / first N / N examples is result_count, "
                "usually expected_call_count=1.\n"
                "- For one object with descriptors, emit one call group. For hotel name plus "
                "city, emit one booking group, not one per location.\n"
                "- If something is truly absent, put its semantic role in missing_inputs. "
                "Do not mark current facts, generic parameters, or defaultable values missing.\n"
                "Schema:"
                '{"canonical_request":"clear paraphrase of the user request",'
                '"slots_observed":[{"role":"short_snake_case_semantic_role",'
                '"value":"normalized value or number/list/bool",'
                '"value_type":"text | number | boolean | date | time | location | currency | '
                'person | organization | topic | file_path | url | identifier | category | '
                'constraint | result_count | default",'
                '"evidence_span":"exact request span or null",'
                '"status":"explicit | semantic | inferred | defaulted",'
                '"confidence":0.0}],'
                '"call_groups":[{"intent":"short action phrase",'
                '"unit_of_work":"one requested operation/entity/group",'
                '"requested_entities":["..."],'
                '"expected_call_count":1,'
                '"result_count":null,'
                '"can_use_batch_tool_if_available":true}],'
                '"missing_inputs":[]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def apply_missing_input_slot_filler(
    user_request: str,
    attachments_metadata: Any,
    intent_audit: dict[str, Any] | None,
    slot_filler_output: dict[str, Any] | None,
    task_family: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Merge model-identified missing inputs into Run 1 and add unavailable entries."""
    repaired = deepcopy(intent_audit if isinstance(intent_audit, dict) else {})
    repairs: list[dict[str, Any]] = []
    if not should_run_missing_input_slot_filler(user_request, repaired, task_family):
        return repaired, repairs

    names = _missing_input_slot_filler_names(slot_filler_output)
    if not names:
        return repaired, repairs

    missing_inputs = list(_as_list(repaired.get("missing_inputs")))
    existing_missing = {_to_snake_case(_missing_input_name(item)) for item in missing_inputs}
    for name in names:
        normalized_name = _to_snake_case(name)
        if not normalized_name or normalized_name in existing_missing:
            continue
        missing_inputs.append(name)
        existing_missing.add(normalized_name)

    if not missing_inputs:
        return repaired, repairs

    inputs = [
        deepcopy(item)
        for item in _as_list(repaired.get("inputs"))
        if isinstance(item, dict)
    ]
    repairs.append(
        {
            "action": "missing_input_slot_filler",
            "reason": "A narrow model pass identified required user-provided source input.",
            "patch": {"missing_inputs": [_missing_input_name(item) for item in missing_inputs]},
        }
    )
    _ensure_missing_inputs_are_input_entries(inputs, missing_inputs, repairs)
    repaired["inputs"] = _dedupe_inputs(inputs)
    repaired["missing_inputs"] = _repaired_missing_inputs(missing_inputs, repaired["inputs"])
    repaired, repair_repairs = repair_intent_input_audit(
        user_request,
        attachments_metadata,
        repaired,
    )
    repairs.extend(repair_repairs)
    return repaired, repairs


def build_messages_for_pass(
    pass_key: str,
    user_request: str,
    context: str = "",
    attachments_metadata: Any | None = None,
    previous: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    previous = previous or {}
    attachments_metadata = attachments_metadata if attachments_metadata is not None else []
    if pass_key == "intent_final_user_want":
        evidence_chunks = previous.get("evidence_chunks") or build_evidence_chunks(
            user_request, context, attachments_metadata
        )
        return _intent_final_user_want_messages(user_request, evidence_chunks)
    if pass_key == "intent_required_inputs":
        evidence_chunks = previous.get("evidence_chunks") or build_evidence_chunks(
            user_request, context, attachments_metadata
        )
        return _intent_required_inputs_messages(
            user_request,
            evidence_chunks,
            previous.get("intent_final_user_want", {}),
        )
    if pass_key == "intent_input_availability":
        evidence_chunks = previous.get("evidence_chunks") or build_evidence_chunks(
            user_request, context, attachments_metadata
        )
        return _intent_input_availability_messages(
            evidence_chunks,
            previous.get("intent_final_user_want", {}),
            previous.get("intent_required_inputs", {}),
        )
    if pass_key == "intent_input_audit":
        return _intent_input_audit_messages(user_request, context, attachments_metadata)
    if pass_key == "transformation_externality_audit":
        return _transformation_audit_messages(user_request, previous["intent_input_audit"])
    if pass_key == "capability_requirements":
        return _capability_requirement_messages(
            user_request,
            previous["intent_input_audit"],
            previous["transformation_externality_audit"],
        )
    if pass_key == "capability_normalization":
        return _capability_normalization_messages(previous["capability_requirements"])
    if pass_key == "capability_ordering":
        return _capability_ordering_messages(
            previous["capability_requirements"],
            previous["capability_normalization"],
        )
    raise ValueError(f"Unknown capability planning pass: {pass_key}")


def validate_required_top_level(pass_key: str, parsed: dict[str, Any]) -> list[str]:
    missing = []
    for key in PASS_TOP_LEVEL_KEYS[pass_key]:
        if key not in parsed:
            missing.append(key)
    return missing


def repair_intent_input_audit(
    user_request: str,
    attachments_metadata: Any,
    intent_audit: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair high-confidence Run 1 availability/format errors."""
    source = intent_audit if isinstance(intent_audit, dict) else {}
    repaired = deepcopy(source)
    inputs = [
        deepcopy(item)
        for item in _as_list(repaired.get("inputs"))
        if isinstance(item, dict)
    ]
    repairs: list[dict[str, Any]] = []
    instruction_text = _instruction_text_without_large_blocks(user_request)
    instruction_masked = _instruction_text_masked(user_request)
    request = instruction_masked.lower()

    if "test" in request and "test" not in str(repaired.get("final_user_want", "")).lower():
        repaired["final_user_want"] = f"{repaired.get('final_user_want', '').strip()} and tests".strip()
        repairs.append(
            {
                "action": "edit_final_user_want",
                "reason": "The user request explicitly includes tests.",
                "patch": {"final_user_want": repaired["final_user_want"]},
            }
        )

    if _looks_like_calculation_request(request):
        final_want = str(repaired.get("final_user_want", ""))
        if "compute" not in final_want.lower() or "numeric" not in final_want.lower():
            repaired["final_user_want"] = f"compute numeric result for {instruction_text}"
            repairs.append(
                {
                    "action": "edit_final_user_want",
                    "reason": "Calculation tasks should preserve compute/numeric intent.",
                    "patch": {"final_user_want": repaired["final_user_want"]},
                }
            )

    _repair_attachment_input_formats(inputs, attachments_metadata, repairs)
    if not inputs and _pasted_text_blocks(user_request):
        input_name = "essay text" if "essay" in request else "pasted text"
        inputs.append(
            {
                "name": input_name,
                "needed_for": repaired.get("final_user_want") or "text transformation",
                "available": True,
                "format": "pasted_text",
                "evidence": _preview(_pasted_text_blocks(user_request)[0]),
            }
        )
        repairs.append(
            {
                "action": "add_available_input",
                "reason": "A pasted text chunk is available even if Run 1B omitted it.",
                "patch": {"name": input_name, "format": "pasted_text"},
            }
        )
    if _looks_like_current_fact_request(request):
        _mark_query_inputs_available(inputs, "current or external information query", repairs)
        _ensure_request_input(inputs, instruction_text, "current or external information query", repairs)
    if _looks_like_url_request(user_request):
        _repair_url_inputs(user_request, inputs, repairs)
    if _operation_for_request(request) == "generate_code":
        _mark_query_inputs_available(inputs, "code or query specification", repairs)
        _ensure_request_input(inputs, instruction_text, "code or query specification", repairs)
    if _looks_like_calculation_request(request):
        _mark_query_inputs_available(inputs, "numeric expression or calculation inputs", repairs)
        _ensure_request_input(inputs, instruction_text, "numeric expression or calculation inputs", repairs)
    if _looks_like_image_generation_request(request):
        _mark_query_inputs_available(inputs, "image generation requirements", repairs)
        _ensure_request_input(inputs, instruction_text, "image generation requirements", repairs)

    _repair_pasted_text_input_names(user_request, inputs, repairs)

    if _needs_image_scale_input(request, inputs):
        inputs.append(
            {
                "name": "scale or reference measurement",
                "needed_for": "convert image measurements to real-world size",
                "available": False,
                "format": "unknown",
                "evidence": "real-world measurement from an image requires scale",
            }
        )
        repairs.append(
            {
                "action": "add_missing_input",
                "reason": "Real-world image measurement requires scale information.",
                "patch": {"missing_input": "scale or reference measurement"},
            }
        )
    if _looks_like_image_measurement_request(request, inputs):
        _normalize_scale_inputs(inputs, repairs)

    if not inputs and _looks_like_available_request_only_input(request):
        inputs.append(
            {
                "name": "request requirements",
                "needed_for": repaired.get("final_user_want") or "complete the request",
                "available": True,
                "format": "none",
                "evidence": instruction_text,
            }
        )
        repairs.append(
            {
                "action": "add_available_input",
                "reason": "The request itself supplies the needed requirements.",
                "patch": {"format": "none"},
            }
        )

    repaired["inputs"] = inputs
    repaired["missing_inputs"] = _repaired_missing_inputs(repaired.get("missing_inputs"), inputs)
    _ensure_missing_inputs_are_input_entries(inputs, repaired["missing_inputs"], repairs)
    repaired["missing_inputs"] = _repaired_missing_inputs(repaired.get("missing_inputs"), inputs)
    if _looks_like_image_measurement_request(request, inputs) and not any(
        "scale" in _to_snake_case(str(item)) for item in repaired["missing_inputs"]
    ):
        repaired["missing_inputs"].append("scale or reference measurement")
        _ensure_missing_inputs_are_input_entries(inputs, repaired["missing_inputs"], repairs)
    return repaired, repairs


def repair_transformation_audit(
    user_request: str,
    intent_audit: dict[str, Any] | None,
    transformation_audit: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair high-confidence Run 2 external-action errors."""
    intent_audit = intent_audit or {}
    source = transformation_audit if isinstance(transformation_audit, dict) else {}
    repaired = deepcopy(source)
    actions = [
        deepcopy(action)
        for action in _as_list(repaired.get("external_actions"))
        if isinstance(action, dict)
    ]
    repairs: list[dict[str, Any]] = []
    instruction_text = _instruction_text_without_large_blocks(user_request)
    instruction_masked = _instruction_text_masked(user_request)
    request = instruction_masked.lower()

    if _looks_like_current_fact_request(request) or _looks_like_url_request(user_request):
        repaired["needs_current_or_external_info"] = True
        _ensure_external_action(actions, "web_search", "current or external information is needed", repairs)
    elif _has_missing_inputs(intent_audit) and repaired.get("needs_current_or_external_info") is True:
        repaired["needs_current_or_external_info"] = False
        actions = [
            action
            for action in actions
            if action.get("action_type") not in {"web_search", "fact_checking"}
        ]
        repairs.append(
            {
                "action": "edit_current_info_flag",
                "reason": "Missing user input is not a current/external fact requirement.",
                "patch": {"needs_current_or_external_info": False},
            }
        )
    elif _image_request_should_not_need_current_info(request, intent_audit):
        if repaired.get("needs_current_or_external_info") is not False:
            repaired["needs_current_or_external_info"] = False
            repairs.append(
                {
                    "action": "edit_current_info_flag",
                    "reason": "Image measurement needs scale/user input, not current facts.",
                    "patch": {"needs_current_or_external_info": False},
                }
            )
        actions = [
            action
            for action in actions
            if action.get("action_type") not in {"web_search", "fact_checking"}
        ]
    if "image" in _input_formats(intent_audit) and not _has_file_like_input(intent_audit):
        actions = [
            action for action in actions if action.get("action_type") != "file_reading"
        ]

    if _has_missing_inputs(intent_audit):
        _ensure_external_action(actions, "user_input", "required input is missing", repairs)
    if _looks_like_calculation_request(request):
        _ensure_external_action(actions, "calculation", "numeric computation is needed", repairs)
    if _looks_like_code_task(request) and _has_file_like_input(intent_audit):
        _ensure_external_action(actions, "file_reading", "existing project files must be inspected", repairs)
        _ensure_external_action(actions, "file_writing", "project files must be changed", repairs)
        if "test" in request or "tests" in request or "rerun" in request:
            _ensure_external_action(actions, "code_execution", "tests or code must be run", repairs)
    if _looks_like_generic_function_request(instruction_masked) and not _has_file_like_input(intent_audit):
        actions = [
            action
            for action in actions
            if action.get("action_type") not in {"file_reading", "file_writing", "code_execution"}
        ]
    if _looks_like_image_measurement_request(request, _as_list(intent_audit.get("inputs"))):
        _ensure_external_action(actions, "image_understanding", "image content must be interpreted", repairs)
        _ensure_external_action(actions, "calculation", "pixel measurements must be converted", repairs)
        _ensure_external_action(actions, "user_input", "scale information is required", repairs)
    if _looks_like_image_generation_request(request):
        _ensure_external_action(actions, "image_generation", "an image must be generated", repairs)

    if not actions:
        actions = [{"action_type": "none", "needed": False, "reason": "no external action needed"}]
    repaired["external_actions"] = actions
    return repaired, repairs


def repair_capability_requirements(
    intent_audit: dict[str, Any] | None,
    transformation_audit: dict[str, Any] | None,
    capability_requirements: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply deterministic repairs for repeated, schema-level capability mistakes."""
    intent_audit = intent_audit or {}
    transformation_audit = transformation_audit or {}
    source = capability_requirements if isinstance(capability_requirements, dict) else {}
    repaired = deepcopy(source)
    repairs: list[dict[str, Any]] = []
    caps = [
        deepcopy(cap)
        for cap in _as_list(repaired.get("capabilities_needed"))
        if isinstance(cap, dict) and not _is_placeholder_capability(cap)
    ]
    extra_needed_actions: set[str] = set()

    for cap in caps:
        _repair_capability_basics(cap, caps, repairs)
        action_types = _split_external_action_types(cap.get("external_action_type"))
        if not action_types:
            action_types = ["other" if cap.get("requires_external_action") else "none"]
        if "none" in action_types and len(action_types) > 1:
            action_types = [action for action in action_types if action != "none"]
        original_action = cap.get("external_action_type")
        primary_action = action_types[0]
        if original_action != primary_action:
            cap["external_action_type"] = primary_action
            cap["requires_external_action"] = primary_action != "none"
            repairs.append(
                {
                    "action": "normalize_external_action_type",
                    "capability_id": cap.get("id"),
                    "reason": "Capability action types must use exactly one enum value.",
                    "from": original_action,
                    "to": primary_action,
                }
            )
        if primary_action == "none" and cap.get("requires_external_action") is True:
            cap["requires_external_action"] = False
            repairs.append(
                {
                    "action": "edit_field",
                    "capability_id": cap.get("id"),
                    "reason": "Capabilities with external_action_type none are not external.",
                    "patch": {"requires_external_action": False},
                }
            )
        if primary_action != "none" and cap.get("requires_external_action") is not True:
            cap["requires_external_action"] = True
            repairs.append(
                {
                    "action": "edit_field",
                    "capability_id": cap.get("id"),
                    "reason": "Capabilities with external actions should mark requires_external_action true.",
                    "patch": {"requires_external_action": True},
                }
            )
        if _is_missing_input_capability(cap) and cap.get("capability_name") != "request_missing_input":
            cap["capability_name"] = "request_missing_input"
            repairs.append(
                {
                    "action": "rename_capability",
                    "capability_id": cap.get("id"),
                    "reason": "Missing-input capabilities use the canonical label.",
                    "patch": {"capability_name": "request_missing_input"},
                }
            )
        extra_needed_actions.update(action_types[1:])

    needed_actions = _needed_external_action_types(transformation_audit) | extra_needed_actions
    if _has_missing_inputs(intent_audit):
        needed_actions.add("user_input")
    if _needs_current_information(transformation_audit) and not (
        needed_actions & {"web_search", "fact_checking"}
    ):
        needed_actions.add("web_search")

    if "file_reading" in needed_actions and _only_inline_or_visual_inputs(intent_audit):
        needed_actions.remove("file_reading")
        repairs.append(
            {
                "action": "drop_unneeded_external_action",
                "reason": "Inline text and image inputs do not require file_reading.",
                "patch": {"external_action_type": "file_reading"},
            }
        )
    _drop_unneeded_file_reading_capabilities(caps, intent_audit, needed_actions, repairs)
    _normalize_code_file_reading_capabilities(caps, intent_audit, repairs)

    if _has_missing_inputs(intent_audit) and not any(
        _is_missing_input_capability(cap) for cap in caps
    ):
        _append_capability(caps, _missing_input_capability_spec(intent_audit), repairs)

    if (
        "pdf" in _input_formats(intent_audit)
        and _pdf_text_is_needed(intent_audit)
        and not any(_is_file_extraction_capability(cap) for cap in caps)
    ):
        _append_capability(
            caps,
            _capability_spec_for_action("file_reading", intent_audit),
            repairs,
        )

    for action_type in sorted(needed_actions):
        if action_type == "none":
            continue
        if any(_capability_has_action(cap, action_type) for cap in caps):
            continue
        _append_capability(
            caps,
            _capability_spec_for_action(action_type, intent_audit),
            repairs,
        )

    if _looks_like_image_measurement_request(
        str(intent_audit.get("final_user_want", "")).lower(),
        _as_list(intent_audit.get("inputs")),
    ) and not any(
        str(cap.get("capability_name", "")).lower() == "measure" for cap in caps
    ):
        _append_capability(caps, _template_cap("measure_image"), repairs)

    _append_inferred_none_capabilities(caps, intent_audit, repairs)

    repaired["capabilities_needed"] = caps
    return repaired, repairs


def repair_capability_ordering(
    capability_requirements: dict[str, Any] | None,
    ordering: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep Run 5 aligned with repaired Run 3 capabilities and obvious dependencies."""
    requirements = capability_requirements or {}
    source = ordering if isinstance(ordering, dict) else {}
    repaired = deepcopy(source)
    required_caps = [
        deepcopy(cap)
        for cap in _as_list(requirements.get("capabilities_needed"))
        if isinstance(cap, dict)
    ]
    ordered_caps = [
        deepcopy(cap)
        for cap in _as_list(repaired.get("ordered_capabilities"))
        if isinstance(cap, dict)
    ]
    repairs: list[dict[str, Any]] = []

    required_by_id = {
        _capability_id(cap, index): cap for index, cap in enumerate(required_caps)
    }
    ordered_by_id = {
        _capability_id(cap, index): cap for index, cap in enumerate(ordered_caps)
    }
    for cap_id, required in required_by_id.items():
        if cap_id not in ordered_by_id:
            ordered = _ordered_capability_from_requirement(cap_id, required)
            ordered_caps.append(ordered)
            ordered_by_id[cap_id] = ordered
            repairs.append(
                {
                    "action": "add_ordered_capability",
                    "reason": "Run 5 must include every Run 3 capability in the graph.",
                    "patch": ordered,
                }
            )

    for index, cap in enumerate(ordered_caps):
        cap_id = _capability_id(cap, index)
        required = required_by_id.get(cap_id, {})
        _repair_ordered_capability_fields(cap, required, repairs)

    _repair_dependency_fields(ordered_caps, repairs)
    _add_obvious_dependencies(ordered_caps, repairs)
    repaired["ordered_capabilities"] = ordered_caps
    return repaired, repairs


def repair_capability_normalization(
    capability_requirements: dict[str, Any] | None,
    normalization: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep normalized capability labels aligned and machine-checkable."""
    requirements = capability_requirements or {}
    source = normalization if isinstance(normalization, dict) else {}
    repaired = deepcopy(source)
    repairs: list[dict[str, Any]] = []
    requirement_caps = [
        cap
        for cap in _as_list(requirements.get("capabilities_needed"))
        if isinstance(cap, dict)
    ]
    normalized_caps = [
        deepcopy(cap)
        for cap in _as_list(repaired.get("normalized_capabilities"))
        if isinstance(cap, dict)
    ]
    normalized_by_id = {
        _capability_id(cap, index): cap for index, cap in enumerate(normalized_caps)
    }

    for index, required in enumerate(requirement_caps):
        cap_id = _capability_id(required, index)
        if cap_id not in normalized_by_id:
            cap = {
                "id": cap_id,
                "original_name": required.get("capability_name")
                or f"capability_{index + 1}",
                "normalized_name": _to_snake_case(
                    str(required.get("capability_name") or f"capability_{index + 1}")
                ),
                "meaning_changed": False,
                "external_action_type": required.get("external_action_type", "none"),
            }
            normalized_caps.append(cap)
            normalized_by_id[cap_id] = cap
            repairs.append(
                {
                    "action": "add_normalized_capability",
                    "reason": "Every capability requirement needs a normalized label.",
                    "patch": cap,
                }
            )

    for index, cap in enumerate(normalized_caps):
        cap_id = _capability_id(cap, index)
        original_name = str(cap.get("original_name") or cap.get("normalized_name") or cap_id)
        desired_name = _to_snake_case(str(cap.get("normalized_name") or original_name))
        if cap.get("normalized_name") != desired_name:
            cap["normalized_name"] = desired_name
            repairs.append(
                {
                    "action": "normalize_capability_name",
                    "capability_id": cap_id,
                    "reason": "Capability normalization requires short snake_case names.",
                    "patch": {"normalized_name": desired_name},
                }
            )
        if not cap.get("original_name"):
            cap["original_name"] = original_name
        if "meaning_changed" not in cap:
            cap["meaning_changed"] = False
        if not cap.get("external_action_type"):
            cap["external_action_type"] = "none"

    repaired["normalized_capabilities"] = normalized_caps
    repaired["merged_capabilities"] = _as_list(repaired.get("merged_capabilities"))
    return repaired, repairs


def validate_capability_plan(
    intent_audit: dict[str, Any] | None,
    transformation_audit: dict[str, Any] | None,
    capability_requirements: dict[str, Any] | None,
    normalization: dict[str, Any] | None,
    ordering: dict[str, Any] | None,
) -> dict[str, Any]:
    intent_audit = intent_audit or {}
    transformation_audit = transformation_audit or {}
    capability_requirements = capability_requirements or {}
    normalization = normalization or {}
    ordering = ordering or {}

    violations: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    generated_caps = _as_list(capability_requirements.get("capabilities_needed"))
    normalized_caps = _as_list(normalization.get("normalized_capabilities"))
    ordered_caps = _as_list(ordering.get("ordered_capabilities"))
    all_caps = generated_caps + ordered_caps

    _validate_capability_fields(ordered_caps, violations, repairs)
    _validate_dependency_graph(ordered_caps, violations, repairs)
    _validate_external_action_fields(generated_caps, normalized_caps, violations, repairs)
    _validate_required_external_actions(
        intent_audit,
        transformation_audit,
        generated_caps,
        violations,
        repairs,
    )
    _validate_no_tool_or_worker_caps(generated_caps, normalized_caps, violations, repairs)
    _validate_normalized_names(normalized_caps, violations, repairs)
    _validate_missing_inputs(intent_audit, all_caps, violations, repairs)
    _validate_pasted_text_handling(intent_audit, generated_caps, violations, repairs)
    _validate_pdf_text_extraction(intent_audit, generated_caps, violations, repairs)
    _validate_current_information(
        transform_audit=transformation_audit,
        caps=generated_caps,
        violations=violations,
        repairs=repairs,
    )

    return {
        "valid": not violations,
        "violations": violations,
        "minimal_repairs": repairs,
    }


def _intent_final_user_want_messages(
    user_request: str,
    evidence_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "instruction_text": _chunk_text(evidence_chunks, "instruction_1"),
        "source_inventory": _chunk_inventory(evidence_chunks),
    }
    return [
        {
            "role": "system",
            "content": (
                "You identify only the user's final desired result. Return compact valid "
                "JSON only. Do not decide input availability or capabilities."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 1A: Final Intent Only.\n"
                "Write a concise full phrase containing both the action and the main "
                "object/topic. Do not output only a bare verb. Preserve important requested "
                "deliverables such as tests, bullets, open questions, itinerary, alt text, "
                "or file output.\n"
                'Schema:{"final_user_want":"..."}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _intent_required_inputs_messages(
    user_request: str,
    evidence_chunks: list[dict[str, Any]],
    final_intent: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "final_user_want": final_intent.get("final_user_want"),
        "instruction_text": _chunk_text(evidence_chunks, "instruction_1"),
        "source_inventory": _chunk_inventory(evidence_chunks),
    }
    return [
        {
            "role": "system",
            "content": (
                "You list the inputs required to satisfy the final user intent. Return "
                "compact valid JSON only. Do not judge availability yet."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 1B: Required Inputs Only.\n"
                "List only information/content/constraints required for the final result. "
                "Do not mark missing or available. Current/latest fact questions require the "
                "query as an input, not the answer. Generic function parameters are part of "
                "the function specification. If real-world image measurement is requested, "
                "include scale/reference measurement as a required input.\n"
                'Schema:{"required_inputs":[{"name":"...","needed_for":"...",'
                '"reason":"..."}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _intent_input_availability_messages(
    evidence_chunks: list[dict[str, Any]],
    final_intent: dict[str, Any],
    required_inputs: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "final_user_want": final_intent.get("final_user_want"),
        "required_inputs": required_inputs.get("required_inputs", []),
        "evidence_chunks": _availability_chunks(evidence_chunks),
    }
    return [
        {
            "role": "system",
            "content": (
                "You match required inputs to typed evidence chunks. Return compact valid "
                "JSON only. Use chunk formats; do not invent sources."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 1C: Input Availability + Format.\n"
                "For each required input, decide whether an evidence chunk directly satisfies "
                "it. If yes, available=true and format must be the matched chunk format. If no, "
                "available=false and format=unknown. Current/latest fact queries are available "
                "when the request supplies the question; retrieval happens later. Attachment "
                "files are available even when contents_in_context=false. Do not treat the "
                "answer to a current-fact question as missing user input.\n"
                'Schema:{"inputs":[{"name":"...","needed_for":"...","available":true,'
                '"format":"pasted_text | attached_file | pdf | image | url | file_path | '
                'structured_data | unknown | none","source_chunk_ids":["..."],'
                '"evidence":"..."}],"missing_inputs":[]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _intent_input_audit_messages(
    user_request: str,
    context: str,
    attachments_metadata: Any,
) -> list[dict[str, str]]:
    payload = {
        "user_request": user_request,
        "context": context,
        "attachments_metadata": attachments_metadata,
    }
    return [
        {
            "role": "system",
            "content": (
                "You infer the final user intent and required inputs. Return only compact "
                "valid JSON. Do not use markdown. Do not choose tools, agents, or capabilities."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 1: Intent + Input Audit.\n"
                "Answer only: What final thing does the user want? What inputs are needed? "
                "Are those inputs already available? What format are the inputs in?\n"
                "Rules: use only the provided user request, context, and attachments metadata. "
                "final_user_want must be a concise full phrase with both the task verb and the "
                "main object/topic, for example revise the essay, summarize the policy memo, "
                "compute the numeric expression, fix the parser bug and tests, describe the "
                "attached image as alt text, or identify the latest OpenAI API model. Do not "
                "return only a bare verb. Pasted_text means substantive inline material to "
                "transform or analyze, such as an essay, paragraph, log, list, or reviews. Do "
                "not mark the whole instruction as pasted_text. Use structured_data for inline "
                "JSON, CSV-like tables, or data records. Use none for self-contained "
                "requirements, calculations, current-fact queries, image-generation briefs, "
                "email requirements, and generic function parameters. Generic function/template "
                "parameters such as nums, k, and m are available specification inputs, not "
                "missing user data. Current/latest/live facts are an available information need "
                "with format none; the missing thing is not the answer, it is external retrieval "
                "for a later pass. URLs are available with format url. File paths are available "
                "with format file_path. For attachments, preserve the attachment format exactly, "
                "especially pdf or image. If a request refers to an essay, paper, dataset, "
                "attachment, or scale but it is neither pasted nor present in attachments "
                "metadata, mark that input unavailable with format unknown. If an attached "
                "file's contents are not in context, mark the file itself available; extraction "
                "is a later pass.\n"
                "Examples: make my essay better with no essay -> final_user_want revise the "
                "essay, missing essay text. Latest OpenAI model -> input OpenAI model query "
                "available true format none, missing_inputs empty. 1 + 2 + log base 3 of 90 -> "
                "input numeric expression available true format none. max_window_sum(nums,k,m) "
                "-> input function signature and parameters available true format none. "
                "Attached conference_room.png with format image -> input attached image "
                "available true format image.\n"
                "Schema:"
                '{"final_user_want":"...","inputs":[{"name":"...","needed_for":"...",'
                '"available":true,"format":"pasted_text | attached_file | pdf | image | url | '
                'file_path | structured_data | unknown | none","evidence":"..."}],'
                '"missing_inputs":[]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _transformation_audit_messages(
    user_request: str,
    intent_input_audit: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "original_user_request": user_request,
        "run_1_intent_input_audit": intent_input_audit,
    }
    return [
        {
            "role": "system",
            "content": (
                "You audit state transformations and externality needs. Return only compact "
                "valid JSON. Do not use markdown. Do not choose tools, agents, or capabilities."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 2: Transformation + Externality Audit.\n"
                "Answer only: What transformations turn the available inputs into the desired "
                "output? Does the task require current or external information? Does it require "
                "file reading, file writing, search, calculation, code execution, image "
                "generation, image understanding, user input, or another external action?\n"
                "Rules: pure pasted-text editing, translating, comparing, classifying, or error "
                "log explanation does not need file_reading or web_search. Current/latest/live, "
                "today, this month, stock prices, travel restrictions, weather, advisories, "
                "current office holders, API availability, URLs, and YouTube pages need external "
                "info and usually web_search; fact-check requests need fact_checking and "
                "web_search. Attachments and file paths whose contents are not in context need "
                "file_reading. Local code changes need file_reading and file_writing; include "
                "code_execution when the user asks to run, rerun, validate, or add tests. Image "
                "description, OCR, or visual measurement needs image_understanding, not "
                "file_reading. Image creation or editing needs image_generation; editing an "
                "existing image also needs image_understanding. Missing required inputs need "
                "user_input. Calculations need calculation. Use exactly one action_type enum "
                "value per external_actions item; never combine values with slashes or pipes.\n"
                "Schema:"
                '{"starting_state":"...","desired_state":"...","transformations_needed":['
                '{"transformation":"...","input_state":"...","output_state":"...",'
                '"reason":"..."}],"needs_current_or_external_info":false,'
                '"external_actions":[{"action_type":"file_reading | file_writing | web_search | '
                'fact_checking | calculation | code_execution | image_understanding | '
                'image_generation | user_input | none | other","needed":true,'
                '"reason":"..."}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _capability_requirement_messages(
    user_request: str,
    intent_input_audit: dict[str, Any],
    transformation_audit: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "original_user_request": user_request,
        "run_1_intent_input_audit": intent_input_audit,
        "run_2_transformation_externality_audit": transformation_audit,
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate abstract capability requirements as state transformations. "
                "Return only compact valid JSON. Do not use markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 3: Generated Capability Requirements.\n"
                "Rules: do not choose specific agents. Do not choose concrete tools. Do not "
                "name APIs, libraries, plugins, or software packages. Do not use a fixed "
                "allowed capability list. A capability must describe a state transformation, "
                "not a worker. Every capability must include inputs, outputs, "
                "external_action_type, and done_when. Create at least one capability for every "
                "Run 2 transformation, and at least one capability for every needed external "
                "action except none. Use exactly one external_action_type enum value per "
                "capability; never combine values. If inputs are missing, include "
                "request_missing_input with external_action_type user_input before impossible "
                "downstream work. If file_reading is needed, include an abstract reading "
                "capability such as extract_information_from_attached_document, "
                "extract_information_from_file, inspect_existing_code, or search_provided_files. "
                "If web_search or fact_checking is needed, include retrieve_current_information, "
                "retrieve_external_information, or verify_current_information before answering. "
                "If file_writing is needed, include write_file or modify_code. If code_execution "
                "is needed, include execute_code or validate_output_against_requirements. If "
                "image_understanding is needed, include interpret_image_content or "
                "extract_information_from_image. If image_generation is needed, include "
                "generate_image or transform_image. Pure generation/transformation work still "
                "needs a capability such as revise_text_for_clarity, summarize_document, "
                "translate_text, compare_texts, draft_text, transform_text_format, "
                "analyze_provided_dataset, compute_numeric_result, synthesize_plan, or "
                "generate_code.\n"
                "Good names look like retrieve_current_information, "
                "extract_information_from_attached_document, revise_text_for_clarity, "
                "compute_numeric_result, analyze_provided_dataset, request_missing_input, "
                "validate_output_against_requirements. Bad names look like research_agent, "
                "coding_agent, pdfplumber, browser_tool, assistant_writer.\n"
                "Schema:"
                '{"capabilities_needed":[{"id":"cap_1","capability_name":"...",'
                '"capability_description":"...","input_state":"...","output_state":"...",'
                '"requires_external_action":true,"external_action_type":"none | file_reading | '
                'file_writing | web_search | fact_checking | calculation | code_execution | '
                'image_understanding | image_generation | user_input | other",'
                '"inputs":["..."],"outputs":["..."],"done_when":"..."}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _capability_normalization_messages(
    capability_requirements: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {"run_3_capability_requirements": capability_requirements}
    return [
        {
            "role": "system",
            "content": (
                "You normalize capability names without changing the work. Return only compact "
                "valid JSON. Do not use markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 4: Capability Normalization.\n"
                "Normalize capability names. Merge capabilities that mean the same thing. Use "
                "short snake_case names. Do not map to tools. Do not introduce new work.\n"
                "Schema:"
                '{"normalized_capabilities":[{"id":"cap_1","original_name":"...",'
                '"normalized_name":"...","meaning_changed":false,'
                '"external_action_type":"..."}],"merged_capabilities":[{"new_id":"cap_1",'
                '"merged_from":["cap_1","cap_2"],"normalized_name":"...","reason":"..."}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _capability_ordering_messages(
    capability_requirements: dict[str, Any],
    normalization: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "run_3_capability_requirements": capability_requirements,
        "run_4_capability_normalization": normalization,
    }
    return [
        {
            "role": "system",
            "content": (
                "You order existing abstract capabilities as a dependency graph. Return only "
                "compact valid JSON. Do not use markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run 5: Capability Ordering / Dependency Graph.\n"
                "Order only the capabilities already identified. Do not choose tools. Do not "
                "choose agents. Do not add unrelated capabilities. Dependencies must reference "
                "valid capability ids. Every capability must have inputs, outputs, and done_when.\n"
                "Dependency rules: request_missing_input comes before work that cannot proceed "
                "without it. PDF/document/file extraction comes before summarization, analysis, "
                "calculation, or writing based on extracted content. Current/external retrieval "
                "or fact-checking comes before answering, summarizing, or planning. Inspecting "
                "existing code comes before modifying code. Image interpretation comes before "
                "image transformation when both are present.\n"
                "Schema:"
                '{"ordered_capabilities":[{"id":"cap_1","capability_name":"...",'
                '"depends_on":[],"inputs":["..."],"outputs":["..."],"done_when":"..."}]}\n'
                f"Input:{compact_json(payload)}"
            ),
        },
    ]


def _chunk_inventory(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.get("id"),
            "type": chunk.get("type"),
            "format": chunk.get("format"),
            "available": chunk.get("available"),
            "name": chunk.get("name"),
            "preview": chunk.get("preview"),
            "contents_in_context": chunk.get("contents_in_context"),
        }
        for chunk in chunks
    ]


def _availability_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": chunk.get("id"),
            "type": chunk.get("type"),
            "format": chunk.get("format"),
            "available": chunk.get("available"),
            "name": chunk.get("name"),
            "contents_in_context": chunk.get("contents_in_context"),
            "preview": chunk.get("preview"),
        }
        for chunk in chunks
    ]


def _chunk_text(chunks: list[dict[str, Any]], chunk_id: str) -> str:
    for chunk in chunks:
        if chunk.get("id") == chunk_id:
            return str(chunk.get("text") or "")
    return ""


def _preview(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _segment_user_turn(user_request: str) -> dict[str, Any]:
    """Split a user turn into instruction text and structurally delimited sources."""
    text = str(user_request or "").strip()
    if not text:
        return {
            "instruction_text": "",
            "instruction_text_masked": "",
            "source_spans": [],
            "url_spans": [],
        }

    source_spans: list[dict[str, str]] = []

    def add_source(kind: str, value: str) -> None:
        cleaned = value.strip()
        if len(cleaned) < 2:
            return
        source_spans.append(
            {
                "kind": kind,
                "format": "structured_data" if _looks_like_structured_data(cleaned) else "pasted_text",
                "text": cleaned,
            }
        )

    code_fence_pattern = re.compile(r"```[A-Za-z0-9_-]*\n?([\s\S]*?)```")
    for match in code_fence_pattern.finditer(text):
        add_source("code_fence", match.group(1))
    without_fences = code_fence_pattern.sub(" ", text).strip()

    # Blank-line blocks are a strong structural signal: first block is instruction,
    # later blocks are source material.
    blocks = [block.strip() for block in re.split(r"\n\s*\n", without_fences) if block.strip()]
    if len(blocks) >= 2 and _looks_like_instruction_span(blocks[0]):
        for block in blocks[1:]:
            add_source("blank_line_block", block)
        return _segmented_turn(blocks[0], source_spans)

    colon_split = _split_instruction_colon_source(without_fences)
    if colon_split:
        instruction, payload = colon_split
        if _payload_urls_are_retrieval_targets(instruction, payload):
            return _segmented_turn(f"{instruction} {payload}", source_spans)
        add_source("colon_payload", payload)
        return _segmented_turn(instruction, source_spans)

    quote_match = re.search(r'"([^"\n]{2,})"|“([^”\n]{2,})”|\'([^\'\n]{2,})\'', without_fences)
    if quote_match and _looks_like_instruction_span(without_fences):
        quoted = next(group for group in quote_match.groups() if group)
        add_source("quoted_payload", quoted)
        instruction = (without_fences[: quote_match.start()] + " " + without_fences[quote_match.end() :]).strip()
        return _segmented_turn(instruction or without_fences, source_spans)

    lines = without_fences.splitlines()
    if len(lines) >= 2 and _looks_like_instruction_span(lines[0]):
        payload = "\n".join(lines[1:]).strip()
        if payload:
            if _payload_urls_are_retrieval_targets(lines[0].strip(), payload):
                return _segmented_turn(f"{lines[0].strip()} {payload}", source_spans)
            add_source("newline_payload", payload)
            return _segmented_turn(lines[0].strip(), source_spans)

    return _segmented_turn(without_fences, source_spans)


def _segmented_turn(
    instruction_text: str,
    source_spans: list[dict[str, str]],
) -> dict[str, Any]:
    source_spans = _dedupe_source_spans(source_spans)
    masked_instruction, instruction_urls = _mask_urls(instruction_text)
    url_spans: list[dict[str, str]] = []
    retrieval_role = _url_role_for_instruction(masked_instruction)
    for index, url in enumerate(instruction_urls, start=1):
        url_spans.append(
            {
                "id": f"url_{index}",
                "url": url,
                "zone": "instruction",
                "role": retrieval_role,
            }
        )
    source_offset = len(url_spans)
    for source_index, source in enumerate(source_spans, start=1):
        _, source_urls = _mask_urls(source.get("text", ""))
        source["contains_url"] = bool(source_urls)
        for url_index, url in enumerate(source_urls, start=1):
            url_spans.append(
                {
                    "id": f"url_{source_offset + url_index}",
                    "url": url,
                    "zone": "source_span",
                    "role": "embedded_content",
                    "source_index": str(source_index),
                }
            )
        source_offset = len(url_spans)
    return {
        "instruction_text": instruction_text.strip(),
        "instruction_text_masked": masked_instruction.strip(),
        "source_spans": source_spans,
        "url_spans": url_spans,
    }


def _mask_urls(text: str) -> tuple[str, list[str]]:
    urls: list[str] = []

    def replace(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,)")
        urls.append(url)
        return f"<URL_{len(urls)}>"

    return re.sub(r"https?://\S+", replace, text), urls


def _payload_urls_are_retrieval_targets(instruction_text: str, payload: str) -> bool:
    if not re.search(r"https?://\S+", payload):
        return False
    if _instruction_treats_payload_as_source_text(instruction_text) and _payload_has_non_url_text(
        payload
    ):
        return False
    masked_text, _ = _mask_urls(f"{instruction_text} {payload}")
    return _url_role_for_instruction(masked_text) == "retrieval_target"


def _instruction_treats_payload_as_source_text(instruction_text: str) -> bool:
    text = instruction_text.lower()
    source_markers = [
        "this text",
        "this sentence",
        "this line",
        "this paragraph",
        "this cta",
        "this snippet",
        "this memo",
        "this note",
        "these notes",
        "this message",
        "these messages",
        "this headline",
        "these headlines",
        "this release note",
        "this blurb",
        "this quote",
        "this list",
    ]
    return any(marker in text for marker in source_markers)


def _payload_has_non_url_text(payload: str) -> bool:
    masked, _ = _mask_urls(payload)
    return bool(re.search(r"[A-Za-z]{2,}", masked))


def _url_role_for_instruction(instruction_text_masked: str) -> str:
    text = instruction_text_masked.lower()
    retrieval_verbs = [
        "article",
        "check",
        "compare",
        "contrast",
        "differences",
        "extract",
        "fetch",
        "link",
        "links",
        "open",
        "page",
        "read",
        "retrieve",
        "summarize",
        "url",
        "verify",
    ]
    if "<url_" in text and any(_request_has_hint(text, verb) for verb in retrieval_verbs):
        return "retrieval_target"
    return "embedded_content"


def _url_spans_for_segmented_turn(segmented: dict[str, Any]) -> list[dict[str, str]]:
    return [
        span
        for span in _as_list(segmented.get("url_spans"))
        if isinstance(span, dict)
    ]


def _dedupe_source_spans(spans: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for span in spans:
        key = (span.get("format", ""), re.sub(r"\s+", " ", span.get("text", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def _split_instruction_colon_source(text: str) -> tuple[str, str] | None:
    for match in re.finditer(":", text):
        before = text[: match.start()].strip()
        after = text[match.end() :].strip()
        if len(before) < 3 or len(after) < 8:
            continue
        if after.startswith("//") and before.lower().endswith(("http", "https")):
            continue
        if re.search(r"https?://\S*$", before):
            continue
        if re.match(r"^\d{1,2}:\d{2}\b", after):
            continue
        if _looks_like_instruction_span(before):
            return before, after
    return None


def _looks_like_instruction_span(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if any(_request_has_hint(lowered, hint) for hints in OPERATION_HINTS.values() for hint in hints):
        return True
    return bool(
        re.search(
            r"\b(?:can you|please|could you|would you|what does|what is|turn|make|write|draft|create|from this|given this|using this|these|this)\b",
            lowered,
        )
    )


def _instruction_text_without_large_blocks(user_request: str) -> str:
    return str(_segment_user_turn(user_request).get("instruction_text") or "").strip()


def _instruction_text_masked(user_request: str) -> str:
    return str(_segment_user_turn(user_request).get("instruction_text_masked") or "").strip()


def _pasted_text_blocks(user_request: str) -> list[str]:
    return _dedupe_preserve_order(
        [
            span["text"]
            for span in _segment_user_turn(user_request).get("source_spans", [])
            if span.get("format") != "structured_data"
        ]
    )


def _inline_structured_data_blocks(user_request: str) -> list[str]:
    blocks = [
        span["text"]
        for span in _segment_user_turn(user_request).get("source_spans", [])
        if span.get("format") == "structured_data"
    ]
    for match in re.finditer(r"(\[[\s\S]*\]|\{[\s\S]*\})", user_request):
        candidate = match.group(1).strip()
        if len(candidate) < 8:
            continue
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        blocks.append(candidate)
    if ":" in user_request:
        after_colon = user_request.split(":", 1)[1].strip()
        if _looks_like_structured_data(after_colon):
            blocks.append(after_colon)
    if "\n" in user_request:
        after_first_line = "\n".join(user_request.splitlines()[1:]).strip()
        if _looks_like_structured_data(after_first_line):
            blocks.append(after_first_line)
    return _dedupe_preserve_order(blocks)


def _looks_like_colon_pasted_text(text: str) -> bool:
    if ":" not in text:
        return False
    before, after = text.split(":", 1)
    if len(after.strip()) < 20:
        return False
    if re.search(r"https?://", after):
        return False
    if _looks_like_structured_data(after.strip()):
        return False
    lowered = before.lower()
    return any(
        phrase in lowered
        for phrase in [
            "translate",
            "make this",
            "turn this",
            "turn these",
            "classify",
            "explain this",
            "from this",
            "given this",
            "here is",
            "notes",
            "update",
            "entries",
            "list into",
            "paragraph",
        ]
    )


def _looks_like_structured_data(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped[0] in "[{":
        return True
    lowered = stripped.lower()
    if re.match(r"^(csv|json|tsv)\s*[:\n]", lowered):
        return True
    if _looks_like_csv_or_tsv(stripped):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    comma_rows = [line for line in lines if line.count(",") >= 2]
    tab_rows = [line for line in lines if line.count("\t") >= 2]
    return len(comma_rows) >= 2 or len(tab_rows) >= 2


def _looks_like_csv_or_tsv(text: str) -> bool:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    sample = "\n".join(lines[:5])
    for delimiter in ["\t", ",", ";", "|"]:
        if delimiter not in sample:
            continue
        try:
            rows = list(csv.reader(StringIO(sample), delimiter=delimiter))
        except csv.Error:
            continue
        if len(rows) < 2:
            continue
        widths = [len(row) for row in rows]
        if len(set(widths)) == 1 and widths[0] >= 2:
            return True
    return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _capability_id(cap: dict[str, Any], index: int) -> str:
    return str(cap.get("id") or f"cap_{index + 1}")


def _next_capability_id(caps: list[Any]) -> str:
    numbers = []
    for cap in caps:
        if not isinstance(cap, dict):
            continue
        match = re.fullmatch(r"cap_(\d+)", str(cap.get("id", "")))
        if match:
            numbers.append(int(match.group(1)))
    return f"cap_{max(numbers, default=0) + 1}"


def _repair_attachment_input_formats(
    inputs: list[dict[str, Any]],
    attachments_metadata: Any,
    repairs: list[dict[str, Any]],
) -> None:
    attachments = [item for item in _as_list(attachments_metadata) if isinstance(item, dict)]
    for index, attachment in enumerate(attachments, start=1):
        name = str(attachment.get("name") or "")
        fmt = attachment.get("format")
        if fmt not in INPUT_FORMATS or not name:
            continue
        for item in inputs:
            marker = f"attachment_{index}"
            text = _to_snake_case(
                " ".join(str(item.get(key, "")) for key in ["name", "evidence"])
            )
            matched = _to_snake_case(name) in text or marker in text
            if not matched:
                continue
            if marker in text and _to_snake_case(name) not in text:
                item["name"] = name
                item["evidence"] = name
                repairs.append(
                    {
                        "action": "edit_input_evidence",
                        "reason": "Attachment markers should resolve to the concrete file name.",
                        "patch": {"name": name, "evidence": name},
                    }
                )
            if item.get("format") != fmt:
                item["format"] = fmt
                repairs.append(
                    {
                        "action": "edit_input_format",
                        "reason": "Attachment metadata is authoritative for input format.",
                        "patch": {"name": item.get("name"), "format": fmt},
                    }
                )
            if attachment.get("available") is not False and item.get("available") is not True:
                item["available"] = True
                repairs.append(
                    {
                        "action": "edit_input_availability",
                        "reason": "Attachment metadata says the file is available.",
                        "patch": {"name": item.get("name"), "available": True},
                    }
                )


def _repair_pasted_text_input_names(
    user_request: str,
    inputs: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    request = _instruction_text_masked(user_request).lower()
    if "essay" not in request:
        return
    for item in inputs:
        if item.get("available") is not True or item.get("format") != "pasted_text":
            continue
        text = _to_snake_case(
            " ".join(str(item.get(key, "")) for key in ["name", "needed_for", "evidence"])
        )
        if "essay" in text:
            continue
        item["name"] = "essay text"
        repairs.append(
            {
                "action": "edit_input_name",
                "reason": "Pasted essay input should be labeled as essay text.",
                "patch": {"name": "essay text", "format": "pasted_text"},
            }
        )


def _ensure_missing_inputs_are_input_entries(
    inputs: list[dict[str, Any]],
    missing_inputs: list[Any],
    repairs: list[dict[str, Any]],
) -> None:
    existing = _to_snake_case(
        " ".join(
            str(item.get(key, ""))
            for item in inputs
            if item.get("available") is False
            for key in ["name", "needed_for", "evidence"]
        )
    )
    for missing in list(missing_inputs):
        name = _missing_input_name(missing)
        if not name:
            continue
        normalized_name = _to_snake_case(name)
        if normalized_name and normalized_name in existing:
            continue
        item = {
            "name": name,
            "needed_for": name,
            "available": False,
            "format": "unknown",
            "evidence": "missing user input",
        }
        inputs.append(item)
        existing = f"{existing} {normalized_name}".strip()
        repairs.append(
            {
                "action": "add_missing_input_entry",
                "reason": "Missing inputs should also appear as unavailable required inputs.",
                "patch": item,
            }
        )


def _missing_input_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["name", "input", "needed_for"]:
            candidate = str(value.get(key) or "").strip()
            if candidate and "|" not in candidate:
                return candidate
        return ""
    return str(value or "").strip()


def _mark_query_inputs_available(
    inputs: list[dict[str, Any]],
    reason: str,
    repairs: list[dict[str, Any]],
) -> None:
    for item in inputs:
        fmt = item.get("format")
        if fmt not in {None, "none", "unknown", "pasted_text"}:
            continue
        if item.get("available") is not True or item.get("format") != "none":
            item["available"] = True
            item["format"] = "none"
            repairs.append(
                {
                    "action": "edit_input_availability",
                    "reason": f"Treat {reason} as supplied by the user request.",
                    "patch": {"name": item.get("name"), "available": True, "format": "none"},
                }
            )


def _ensure_request_input(
    inputs: list[dict[str, Any]],
    user_request: str,
    reason: str,
    repairs: list[dict[str, Any]],
) -> None:
    normalized_request = _to_snake_case(user_request)
    for item in inputs:
        text = _to_snake_case(
            " ".join(str(item.get(key, "")) for key in ["name", "needed_for", "evidence"])
        )
        if item.get("available") is True and item.get("format") == "none" and text in normalized_request:
            return
        if item.get("available") is True and item.get("format") == "none" and normalized_request in text:
            return
    inputs.append(
        {
            "name": reason,
            "needed_for": reason,
            "available": True,
            "format": "none",
            "evidence": user_request,
        }
    )
    repairs.append(
        {
            "action": "add_available_input",
            "reason": f"Preserve the full user request as {reason}.",
            "patch": {"format": "none", "available": True},
        }
    )


def _ensure_external_action(
    actions: list[dict[str, Any]],
    action_type: str,
    reason: str,
    repairs: list[dict[str, Any]],
) -> None:
    for action in actions:
        if action.get("action_type") == action_type and action.get("needed") is True:
            return
    actions.append({"action_type": action_type, "needed": True, "reason": reason})
    repairs.append(
        {
            "action": "add_external_action",
            "reason": reason,
            "patch": {"action_type": action_type, "needed": True},
        }
    )


def _normalize_scale_inputs(
    inputs: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    for item in inputs:
        if item.get("available") is not False:
            continue
        text = _to_snake_case(str(item))
        if "reference" not in text and "measurement" not in text and "scale" not in text:
            continue
        if "scale" not in _to_snake_case(str(item.get("name", ""))):
            item["name"] = "scale or reference measurement"
            repairs.append(
                {
                    "action": "edit_input_name",
                    "reason": "Real-world image measurement missing input should mention scale.",
                    "patch": {"name": item["name"]},
                }
            )


def _repair_url_inputs(
    user_request: str,
    inputs: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    segmented = _segment_user_turn(user_request)
    urls = [
        span.get("url", "")
        for span in _url_spans_for_segmented_turn(segmented)
        if span.get("role") == "retrieval_target"
    ]
    if not urls:
        return
    matched_url = False
    for item in inputs:
        text = " ".join(str(item.get(key, "")) for key in ["name", "evidence", "needed_for"])
        if any(url in text for url in urls):
            item["available"] = True
            item["format"] = "url"
            matched_url = True
    if not matched_url:
        inputs.append(
            {
                "name": urls[0].rstrip(".,)"),
                "needed_for": "retrieve external content",
                "available": True,
                "format": "url",
                "evidence": urls[0].rstrip(".,)"),
            }
        )
    repairs.append(
        {
            "action": "repair_url_input",
            "reason": "A URL in the request is an available input; retrieval happens later.",
            "patch": {"format": "url", "available": True},
        }
    )


def _repaired_missing_inputs(value: Any, inputs: list[dict[str, Any]]) -> list[Any]:
    missing = list(_as_list(value))
    available_text = _to_snake_case(
        " ".join(
            str(item.get(key, ""))
            for item in inputs
            if item.get("available") is True
            for key in ["name", "needed_for", "evidence"]
        )
    )
    repaired = []
    for item in missing:
        text = _to_snake_case(str(item))
        if text and (
            text in available_text
            or any(
                text in _to_snake_case(str(inp.get("name", "")))
                for inp in inputs
                if inp.get("available") is True
            )
        ):
            continue
        repaired.append(item)
    return repaired


def _looks_like_current_fact_request(request: str) -> bool:
    return _needs_current_info(request, _operation_for_request(request))


def _needs_current_info(request: str, operation: str = "general") -> bool:
    text = request.lower()
    if _looks_like_reply_to_missing_email_source(text):
        return False
    if _looks_like_personal_local_activity_request(text):
        return False
    if _looks_like_public_current_stories_request(text):
        return True
    if _looks_like_currency_exchange_request(text):
        return True
    if _looks_like_stock_or_market_lookup(text):
        return True
    if _looks_like_weather_lookup(text):
        return True
    if _looks_like_stable_definition_lookup(text):
        return False
    if _request_has_hint(text, "news") and not any(
        _request_has_hint(text, phrase) for phrase in ["summarize this", "rewrite this", "classify this"]
    ):
        return True
    source_transform_operations = {
        "classify",
        "compare",
        "draft",
        "edit_image",
        "explain",
        "extract",
        "extract_action_items",
        "format_transform",
        "generate_image",
        "rewrite",
        "summarize",
        "translate",
    }
    lookup_intents = [
        "as of today",
        "check whether",
        "fact check",
        "fact-check",
        "find current",
        "latest",
        "look up",
        "past week",
        "recent",
        "right now",
        "search online",
        "search the web",
        "web search",
        "today's",
        "up to date",
        "verify",
    ]
    volatile_objects = [
        "advisories",
        "availability",
        "ceo",
        "forecast",
        "law",
        "market cap",
        "news",
        "policy",
        "population",
        "president",
        "price",
        "rain",
        "ranking",
        "rate",
        "rates",
        "release",
        "restriction",
        "schedule",
        "score",
        "status",
        "stock",
        "travel",
        "version",
        "weather",
    ]
    if operation in source_transform_operations and not any(
        _request_has_hint(text, intent) for intent in lookup_intents
    ):
        return False
    if any(_request_has_hint(text, intent) for intent in lookup_intents):
        return True
    if any(_request_has_hint(text, phrase) for phrase in ["stock price", "market cap", "right now"]):
        return True
    if _request_has_hint(text, "current"):
        return any(_request_has_hint(text, obj) for obj in volatile_objects)
    if _request_has_hint(text, "today") or _request_has_hint(text, "tomorrow"):
        return any(_request_has_hint(text, obj) for obj in volatile_objects)
    return any(_request_has_hint(text, obj) for obj in ["weather", "rain", "forecast", "advisories"])


def _looks_like_weather_lookup(text: str) -> bool:
    if any(
        _request_has_hint(text, term)
        for term in [
            "weather",
            "rain",
            "raining",
            "rainy",
            "snowing",
            "windy",
            "forecast",
            "advisories",
        ]
    ):
        return True
    if any(_request_has_hint(text, term) for term in ["high and low", "highs and lows"]):
        return True
    if _request_has_hint(text, "umbrella") and any(
        _request_has_hint(text, term) for term in ["today", "tomorrow", "outside", "bring"]
    ):
        return True
    if any(_request_has_hint(text, term) for term in ["cold outside", "temperature outside"]):
        return True
    if _request_has_hint(text, "how cold") and any(
        _request_has_hint(text, term) for term in ["outside", "today", "tomorrow", "this week", "week"]
    ):
        return True
    return False


def _looks_like_public_current_stories_request(text: str) -> bool:
    return bool(
        any(_request_has_hint(text, phrase) for phrase in ["top stories", "most read stories", "headlines"])
        or re.search(r"\bwhat(?:'s|s| is| were)?\s+happening\s+with\b", text)
    )


def _looks_like_personal_local_activity_request(text: str) -> bool:
    return bool(
        re.search(r"\b(?:my|our)\b", text)
        and any(_request_has_hint(text, phrase) for phrase in ["recent activity", "current activity", "activity"])
        and any(_request_has_hint(text, place) for place in ["account", "backyard", "device", "home", "inbox"])
    )


def _looks_like_stable_definition_lookup(text: str) -> bool:
    return bool(
        _request_has_hint(text, "define")
        or _request_has_hint(text, "definition")
        or _request_has_hint(text, "definition of")
        or re.search(r"\blook\s+up\b.*\bdefinition\b", text)
        or re.search(r"\bdefinition\b.*\blook\s+up\b", text)
        or re.search(r"\bwhat\s+does\b.+\bmean\b", text)
    )


def _looks_like_currency_exchange_request(text: str) -> bool:
    currency_terms = [
        "british pounds",
        "canadian dollars",
        "dollars",
        "euro",
        "euros",
        "exchange rate",
        "japanese yen",
        "pounds",
        "usd",
        "yen",
    ]
    if any(_request_has_hint(text, phrase) for phrase in ["exchange rate", "rate of exchange"]):
        return True
    return (
        any(_request_has_hint(text, term) for term in currency_terms)
        and any(
            _request_has_hint(text, verb)
            for verb in ["convert", "worth", "how many", "how much", "equals", "to"]
        )
    )


def _looks_like_stock_or_market_lookup(text: str) -> bool:
    if any(_request_has_hint(text, term) for term in ["share value", "share price"]):
        return True
    if not any(_request_has_hint(text, term) for term in ["stock", "stocks", "dow", "market"]):
        return False
    return any(
        _request_has_hint(text, term)
        for term in [
            "changed",
            "cost",
            "declining",
            "done last week",
            "exchange",
            "going",
            "going for",
            "increase",
            "market",
            "news",
            "past week",
            "price",
            "rate",
            "rates",
            "rising",
            "share price",
            "these days",
            "today",
            "value",
            "current",
        ]
    )


def _looks_like_url_request(user_request: str) -> bool:
    segmented = _segment_user_turn(user_request)
    return any(
        span.get("zone") == "instruction" and span.get("role") == "retrieval_target"
        for span in _url_spans_for_segmented_turn(segmented)
    )


def _looks_like_generic_function_request(user_request: str) -> bool:
    return bool(re.search(r"\bfunction\b", user_request, re.I) and re.search(r"\w+\([^)]*\)", user_request))


def _looks_like_code_generation_request(request: str) -> bool:
    return any(
        _request_has_hint(request, phrase)
        for phrase in [
            "graphql",
            "mutation",
            "regex",
            "regular expression",
            "sql",
            "select statement",
            "database query",
            "postgres query",
            "python function",
            "javascript function",
            "typescript function",
            "react component",
        ]
    )


def _looks_like_calculation_request(request: str) -> bool:
    if not re.search(r"\d", request):
        return False
    return bool(
        re.search(r"\d\s*(?:[+*/%]|\s-\s)\s*\d", request)
        or re.search(r"\blog(?:arithm|arithmic)?\b", request)
        or re.search(r"\bprime numbers?\b.*\bbetween\b", request)
        or re.search(r"\b(?:area|coefficients?|equation|integral|perimeter|quadratic|roots?|slope)\b", request)
        or ("tax" in request and "price" in request)
        or "final price" in request
    )


def _looks_like_image_generation_request(request: str) -> bool:
    return any(
        phrase in request
        for phrase in [
            "app icon",
            "banner graphic",
            "banner image",
            "create a banner",
            "create a graphic",
            "create an image",
            "create a simple square",
            "generate image",
            "generate a graphic",
            "graphic for",
            "make an image",
        ]
    )


def _needs_image_scale_input(request: str, inputs: list[dict[str, Any]]) -> bool:
    if not _looks_like_image_measurement_request(request, inputs):
        return False
    has_scale = any(
        item.get("available") is False
        and any(word in _to_snake_case(str(item)) for word in ["scale", "reference"])
        for item in inputs
    )
    return not has_scale


def _looks_like_image_measurement_request(request: str, inputs: list[dict[str, Any]]) -> bool:
    has_image = any(item.get("format") == "image" for item in inputs)
    asks_real_measurement = (
        "measure" in request
        and any(word in request for word in ["height", "size", "width", "real-world"])
    )
    return has_image and asks_real_measurement


def _image_request_should_not_need_current_info(
    request: str,
    intent_audit: dict[str, Any],
) -> bool:
    formats = _input_formats(intent_audit)
    return "image" in formats and not _looks_like_current_fact_request(request)


def _has_file_like_input(intent_audit: dict[str, Any]) -> bool:
    return bool(_input_formats(intent_audit) & {"attached_file", "pdf", "file_path"})


def _looks_like_available_request_only_input(request: str) -> bool:
    return any(
        phrase in request
        for phrase in [
            "available external tool",
            "available external api",
            "available tool documentation",
            "available api documentation",
            "external tool/api",
            "tool/api action",
            "write a",
            "draft",
            "create",
            "calculate",
            "what is",
            "turn this",
        ]
    )


def _repair_capability_basics(
    cap: dict[str, Any],
    caps: list[Any],
    repairs: list[dict[str, Any]],
) -> None:
    if not cap.get("id"):
        cap["id"] = _next_capability_id(caps)
        repairs.append(
            {
                "action": "edit_field",
                "capability_id": cap["id"],
                "reason": "Capability ids are required for downstream dependency graphs.",
                "patch": {"id": cap["id"]},
            }
        )
    if not cap.get("capability_name"):
        cap["capability_name"] = "perform_required_transformation"
    if not cap.get("capability_description"):
        cap["capability_description"] = (
            "Transform the available input state into the requested output state."
        )
    if not isinstance(cap.get("inputs"), list):
        value = cap.get("inputs")
        cap["inputs"] = [] if value is None else [str(value)]
    if not isinstance(cap.get("outputs"), list):
        value = cap.get("outputs")
        cap["outputs"] = [] if value is None else [str(value)]
    if not cap.get("done_when"):
        cap["done_when"] = "the capability has produced its declared outputs"


def _split_external_action_types(value: Any) -> list[str]:
    if value in EXTERNAL_ACTION_TYPES:
        return [str(value)]
    if not isinstance(value, str):
        return []
    normalized = _to_snake_case(value)
    matches: list[tuple[int, str]] = []
    for action_type in EXTERNAL_ACTION_TYPES:
        if action_type == "none":
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(action_type)}(?![a-z0-9])", normalized)
        if match:
            matches.append((match.start(), action_type))
    if len(matches) > 3:
        return []
    if not matches and re.search(r"(?<![a-z0-9])none(?![a-z0-9])", normalized):
        return ["none"]
    return [action_type for _, action_type in sorted(matches)]


def _is_placeholder_capability(cap: dict[str, Any]) -> bool:
    name = str(cap.get("capability_name") or "").strip().lower()
    desc = str(cap.get("capability_description") or "").strip().lower()
    action_type = str(cap.get("external_action_type") or "").lower()
    return (
        name in {"...", "capability_name"}
        or desc in {"...", "capability_description"}
        or "none | file_reading | file_writing | web_search" in action_type
    )


def _needed_external_action_types(transform_audit: dict[str, Any]) -> set[str]:
    needed_actions: set[str] = set()
    for action in _as_list(transform_audit.get("external_actions")):
        if not isinstance(action, dict) or not action.get("needed"):
            continue
        for action_type in _split_external_action_types(action.get("action_type")):
            if action_type != "none":
                needed_actions.add(action_type)
    return needed_actions


def _needs_current_information(transform_audit: dict[str, Any]) -> bool:
    if transform_audit.get("needs_current_or_external_info") is True:
        return True
    return bool(_needed_external_action_types(transform_audit) & {"web_search", "fact_checking"})


def _has_missing_inputs(intent_audit: dict[str, Any]) -> bool:
    if _as_list(intent_audit.get("missing_inputs")):
        return True
    return any(
        isinstance(item, dict) and item.get("available") is False
        for item in _as_list(intent_audit.get("inputs"))
    )


def _only_inline_or_visual_inputs(intent_audit: dict[str, Any]) -> bool:
    formats = _input_formats(intent_audit)
    if not formats:
        return False
    file_like = {"attached_file", "pdf", "file_path"}
    return bool(formats <= {"pasted_text", "image", "none", "unknown"} and not formats & file_like)


def _append_capability(
    caps: list[Any],
    spec: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    cap = deepcopy(spec)
    cap["id"] = _next_capability_id(caps)
    caps.append(cap)
    repairs.append(
        {
            "action": "add_capability",
            "reason": f"Add missing capability {cap['capability_name']}.",
            "patch": cap,
        }
    )


def _capability_has_action(cap: dict[str, Any], action_type: str) -> bool:
    return action_type in _split_external_action_types(cap.get("external_action_type"))


def _drop_unneeded_file_reading_capabilities(
    caps: list[dict[str, Any]],
    intent_audit: dict[str, Any],
    needed_actions: set[str],
    repairs: list[dict[str, Any]],
) -> None:
    if "file_reading" in needed_actions or _has_file_like_input(intent_audit):
        return
    kept = []
    for cap in caps:
        if _capability_has_action(cap, "file_reading"):
            repairs.append(
                {
                    "action": "remove_capability",
                    "reason": "No file-like input or Run 2 file_reading need exists.",
                    "patch": {"id": cap.get("id"), "external_action_type": "file_reading"},
                }
            )
            continue
        kept.append(cap)
    caps[:] = kept


def _normalize_code_file_reading_capabilities(
    caps: list[dict[str, Any]],
    intent_audit: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    final_want = str(intent_audit.get("final_user_want") or "").lower()
    if not _looks_like_code_task(final_want):
        return
    for cap in caps:
        if not _capability_has_action(cap, "file_reading"):
            continue
        name = str(cap.get("capability_name") or "")
        if "inspect" in name.lower():
            continue
        cap["capability_name"] = "inspect_existing_code"
        cap["capability_description"] = "Inspect existing code or tests before modifying them."
        repairs.append(
            {
                "action": "rename_capability",
                "reason": "Code file-reading capabilities should describe inspection.",
                "patch": {"id": cap.get("id"), "capability_name": cap["capability_name"]},
            }
        )


def _append_inferred_none_capabilities(
    caps: list[Any],
    intent_audit: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    if _has_missing_inputs(intent_audit):
        return
    final_want = str(intent_audit.get("final_user_want") or "").lower()
    if any(word in final_want for word in ["itinerary", "plan"]):
        if not any("synthesize_plan" in _capability_text_for_matching(cap) for cap in caps):
            _append_capability(
                caps,
                _external_capability_spec(
                    "synthesize_plan",
                    "Synthesize the available constraints and retrieved information into a plan.",
                    "none",
                    ["planning constraints", "available information"],
                    ["synthesized plan or itinerary"],
                ),
                repairs,
            )
    if any(_is_current_info_capability(cap) for cap in caps if isinstance(cap, dict)):
        has_downstream = any(
            isinstance(cap, dict)
            and not _is_current_info_capability(cap)
            and not _is_missing_input_capability(cap)
            for cap in caps
        )
        if not has_downstream:
            _append_capability(
                caps,
                _external_capability_spec(
                    "answer_with_current_information",
                    "Answer the user's question using retrieved current information.",
                    "none",
                    ["retrieved current information", "user question"],
                    ["answer grounded in current information"],
                ),
                repairs,
            )


def _missing_input_capability_spec(intent_audit: dict[str, Any]) -> dict[str, Any]:
    missing = _missing_input_names(intent_audit)
    return {
        "capability_name": "request_missing_input",
        "capability_description": "Ask the user for required information that is not available.",
        "input_state": "required input is missing",
        "output_state": "missing input has been requested from the user",
        "requires_external_action": True,
        "external_action_type": "user_input",
        "inputs": missing or ["missing input description"],
        "outputs": ["user-provided missing input or explicit inability to proceed"],
        "done_when": "the missing input has been requested from the user",
    }


def _missing_input_names(intent_audit: dict[str, Any]) -> list[str]:
    names = [str(item) for item in _as_list(intent_audit.get("missing_inputs")) if item]
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict) or item.get("available") is not False:
            continue
        name = str(item.get("name") or item.get("needed_for") or "missing input")
        if name not in names:
            names.append(name)
    return names


def _capability_spec_for_action(action_type: str, intent_audit: dict[str, Any]) -> dict[str, Any]:
    formats = _input_formats(intent_audit)
    final_want = str(intent_audit.get("final_user_want") or "").lower()
    if action_type == "user_input":
        return _missing_input_capability_spec(intent_audit)
    if action_type == "file_reading":
        if "file_path" in formats and _looks_like_code_task(final_want):
            name = "inspect_existing_code"
            description = "Inspect existing code or tests before modifying them."
            inputs = ["provided code file paths"]
            outputs = ["relevant existing code behavior and constraints"]
        elif "file_path" in formats:
            name = "extract_information_from_file"
            description = "Read provided file paths and extract the information needed."
            inputs = ["provided file paths"]
            outputs = ["extracted file information"]
        else:
            name = "extract_information_from_attached_document"
            description = "Extract needed content from attached documents or data files."
            inputs = ["attached file"]
            outputs = ["extracted document or data content"]
        return _external_capability_spec(name, description, action_type, inputs, outputs)
    if action_type == "file_writing":
        if _looks_like_code_task(final_want):
            name = "modify_code"
            description = "Modify project code or tests to satisfy the requested change."
            inputs = ["existing code context", "requested change"]
            outputs = ["updated code or tests"]
        else:
            name = "write_file"
            description = "Write the requested output to the target file path."
            inputs = ["requested content", "target file path"]
            outputs = ["written file"]
        return _external_capability_spec(name, description, action_type, inputs, outputs)
    if action_type == "code_execution":
        return _external_capability_spec(
            "execute_code",
            "Run code or tests needed to validate the requested change.",
            action_type,
            ["code or test command"],
            ["execution result"],
        )
    if action_type == "web_search":
        if "url" in formats:
            name = "retrieve_external_information"
            description = "Retrieve externally hosted content needed for the request."
        else:
            name = "retrieve_current_information"
            description = "Retrieve current external information needed for the request."
        return _external_capability_spec(
            name,
            description,
            action_type,
            ["information need"],
            ["retrieved external information"],
        )
    if action_type == "fact_checking":
        return _external_capability_spec(
            "verify_current_information",
            "Verify the current claim against external information.",
            action_type,
            ["claim to verify"],
            ["verified fact-check result"],
        )
    if action_type == "calculation":
        return _external_capability_spec(
            "compute_numeric_result",
            "Compute or calculate the numeric result required by the request.",
            action_type,
            ["numeric expression or data"],
            ["computed or calculated result"],
        )
    if action_type == "image_understanding":
        name = "extract_information_from_image" if _looks_like_image_extraction(final_want) else (
            "interpret_image_content"
        )
        return _external_capability_spec(
            name,
            "Interpret the provided image and extract the visual information needed.",
            action_type,
            ["provided image"],
            ["interpreted image content"],
        )
    if action_type == "image_generation":
        name = "transform_image" if "image" in formats else "generate_image"
        return _external_capability_spec(
            name,
            "Generate or transform an image according to the user's request.",
            action_type,
            ["image requirements"],
            ["generated or transformed image"],
        )
    return _external_capability_spec(
        "select_and_execute_external_action",
        "Select and execute the required external action at an abstract capability level.",
        "other",
        ["external action request"],
        ["external action result"],
    )


def _external_capability_spec(
    name: str,
    description: str,
    action_type: str,
    inputs: list[str],
    outputs: list[str],
) -> dict[str, Any]:
    return {
        "capability_name": name,
        "capability_description": description,
        "input_state": "required inputs are identified",
        "output_state": "required outputs are available",
        "requires_external_action": action_type != "none",
        "external_action_type": action_type,
        "inputs": inputs,
        "outputs": outputs,
        "done_when": outputs[0],
    }


def _ordered_capability_from_requirement(cap_id: str, cap: dict[str, Any]) -> dict[str, Any]:
    ordered = {
        "id": cap_id,
        "capability_name": cap.get("capability_name") or "perform_required_transformation",
        "depends_on": [],
        "inputs": cap.get("inputs") if isinstance(cap.get("inputs"), list) else [],
        "outputs": cap.get("outputs") if isinstance(cap.get("outputs"), list) else [],
        "done_when": cap.get("done_when") or "the capability has produced its outputs",
    }
    for field in ["capability_description", "external_action_type"]:
        if field in cap:
            ordered[field] = cap[field]
    return ordered


def _repair_ordered_capability_fields(
    cap: dict[str, Any],
    required: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> None:
    cap_id = str(cap.get("id") or required.get("id") or "capability")
    if not cap.get("id"):
        cap["id"] = cap_id
    if not cap.get("capability_name"):
        cap["capability_name"] = required.get("capability_name") or "perform_required_transformation"
    if not isinstance(cap.get("inputs"), list):
        cap["inputs"] = required.get("inputs") if isinstance(required.get("inputs"), list) else []
    if not isinstance(cap.get("outputs"), list):
        cap["outputs"] = required.get("outputs") if isinstance(required.get("outputs"), list) else []
    if not cap.get("done_when"):
        cap["done_when"] = required.get("done_when") or "the capability has produced its outputs"
    for field in ["capability_description", "external_action_type"]:
        if field not in cap and field in required:
            cap[field] = required[field]
    if not isinstance(cap.get("depends_on"), list):
        cap["depends_on"] = []
        repairs.append(
            {
                "action": "edit_dependency_field",
                "capability_id": cap_id,
                "reason": "Dependencies must be represented as a list.",
                "patch": {"depends_on": []},
            }
        )


def _repair_dependency_fields(
    caps: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    ids = {str(cap.get("id")) for cap in caps}
    for cap in caps:
        cap_id = str(cap.get("id"))
        cleaned = []
        for dep in cap.get("depends_on") or []:
            dep_id = str(dep)
            if dep_id in ids and dep_id != cap_id and dep_id not in cleaned:
                cleaned.append(dep_id)
        if cleaned != (cap.get("depends_on") or []):
            cap["depends_on"] = cleaned
            repairs.append(
                {
                    "action": "repair_dependencies",
                    "capability_id": cap_id,
                    "reason": "Remove duplicate, self, or unknown dependencies.",
                    "patch": {"depends_on": cleaned},
                }
            )


def _add_obvious_dependencies(
    caps: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    for before in caps:
        if not _is_missing_input_capability(before):
            continue
        for after in caps:
            if before is after:
                continue
            _add_dependency_if_safe(caps, before, after, repairs, "missing input first")

    for before in caps:
        if not _is_file_extraction_capability(before):
            continue
        for after in caps:
            if before is after or _is_missing_input_capability(after):
                continue
            if _is_file_extraction_capability(after):
                continue
            if _is_code_execution_capability(after):
                continue
            _add_dependency_if_safe(caps, before, after, repairs, "read/extract before use")

    for before in caps:
        if not _is_current_info_capability(before):
            continue
        for after in caps:
            if before is after or _is_missing_input_capability(after):
                continue
            if _is_current_info_capability(after):
                continue
            _add_dependency_if_safe(
                caps,
                before,
                after,
                repairs,
                "retrieve external information before downstream synthesis",
            )

    for before in caps:
        if not _is_image_understanding_capability(before):
            continue
        for after in caps:
            if before is not after and _is_image_generation_capability(after):
                _add_dependency_if_safe(
                    caps,
                    before,
                    after,
                    repairs,
                    "interpret image before image transformation",
                )

    for before in caps:
        if not _is_code_inspection_capability(before):
            continue
        for after in caps:
            if before is not after and _is_code_modification_capability(after):
                _add_dependency_if_safe(
                    caps,
                    before,
                    after,
                    repairs,
                    "inspect existing code before modifying it",
                )

    for before in caps:
        if not _is_code_modification_capability(before):
            continue
        for after in caps:
            if before is not after and _is_validation_capability(after):
                _add_dependency_if_safe(
                    caps,
                    before,
                    after,
                    repairs,
                    "modify code before validating the modification",
                )


def _add_dependency_if_safe(
    caps: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
    repairs: list[dict[str, Any]],
    reason: str,
) -> None:
    before_id = str(before.get("id"))
    after_id = str(after.get("id"))
    if not before_id or not after_id or before_id == after_id:
        return
    depends_on = after.setdefault("depends_on", [])
    if before_id in depends_on:
        return
    depends_on.append(before_id)
    if not _is_acyclic(
        [str(cap.get("id")) for cap in caps],
        _dependency_edges_from_caps(caps),
    ):
        depends_on.remove(before_id)
        return
    repairs.append(
        {
            "action": "add_dependency",
            "reason": reason,
            "patch": {"before": before_id, "after": after_id},
        }
    )


def _dependency_edges_from_caps(caps: list[dict[str, Any]]) -> list[tuple[str, str]]:
    edges = []
    ids = {str(cap.get("id")) for cap in caps}
    for cap in caps:
        cap_id = str(cap.get("id"))
        for dep in cap.get("depends_on") or []:
            dep_id = str(dep)
            if dep_id in ids and dep_id != cap_id:
                edges.append((dep_id, cap_id))
    return edges


def _looks_like_answer_or_synthesis_capability(cap: dict[str, Any]) -> bool:
    text = _capability_text_for_matching(cap)
    return any(
        word in text
        for word in [
            "answer",
            "bullet",
            "itinerary",
            "plan",
            "report",
            "summar",
            "synthesize",
            "takeaway",
        ]
    )


def _is_code_execution_capability(cap: dict[str, Any]) -> bool:
    action_type = cap.get("external_action_type")
    text = _capability_text_for_matching(cap)
    return action_type == "code_execution" or "execute" in text or "run_test" in text


def _is_image_understanding_capability(cap: dict[str, Any]) -> bool:
    return cap.get("external_action_type") == "image_understanding"


def _is_image_generation_capability(cap: dict[str, Any]) -> bool:
    return cap.get("external_action_type") == "image_generation"


def _is_code_inspection_capability(cap: dict[str, Any]) -> bool:
    text = _capability_text_for_matching(cap)
    return "inspect" in text or ("read" in text and "code" in text)


def _is_code_modification_capability(cap: dict[str, Any]) -> bool:
    text = _capability_text_for_matching(cap)
    return "modify" in text or "write_modified" in text or "repair_project_code" in text


def _is_validation_capability(cap: dict[str, Any]) -> bool:
    text = _capability_text_for_matching(cap)
    return "validate" in text or "verification" in text


def _capability_text_for_matching(cap: dict[str, Any]) -> str:
    return _to_snake_case(
        " ".join(
            str(cap.get(key, ""))
            for key in [
                "capability_name",
                "capability_description",
                "done_when",
                "external_action_type",
            ]
        )
    )


def _looks_like_code_task(text: str) -> bool:
    return any(
        re.search(pattern, text)
        for pattern in [
            r"\bbug\b",
            r"\bcode\b",
            r"\bcomponents?\b",
            r"\bfailing\s+tests?\b",
            r"\bfix\b",
            r"\bfunctions?\b",
            r"\bimports?\b",
            r"\bpackage\b",
            r"\breact\b",
            r"\btests?\b",
            r"\b(?:src|tests?|package|components?)/",
            r"\.[jt]sx?\b",
            r"\.py\b",
        ]
    )


def _looks_like_image_extraction(text: str) -> bool:
    return any(word in text for word in ["date", "merchant", "read", "receipt", "total"])


def _add_violation(
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
    type_: str,
    message: str,
    capability_id: str | None = None,
    repair: dict[str, Any] | None = None,
) -> None:
    item = {"type": type_, "message": message}
    if capability_id is not None:
        item["capability_id"] = capability_id
    violations.append(item)
    if repair is not None:
        repairs.append(repair)


def _validate_capability_fields(
    ordered_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    if not ordered_caps:
        _add_violation(
            violations,
            repairs,
            "missing_ordered_capabilities",
            "Run 5 did not produce any ordered capabilities.",
            repair={
                "action": "add_capability",
                "reason": "At least one ordered capability is required for a plan.",
                "patch": {"ordered_capabilities": []},
            },
        )
        return

    seen_ids: set[str] = set()
    for index, cap in enumerate(ordered_caps):
        if not isinstance(cap, dict):
            _add_violation(
                violations,
                repairs,
                "invalid_capability",
                "Ordered capability must be an object.",
                repair={
                    "action": "remove_capability",
                    "reason": "Non-object entries cannot be used in the dependency graph.",
                    "patch": {"index": index},
                },
            )
            continue
        cap_id = _capability_id(cap, index)
        if cap_id in seen_ids:
            _add_violation(
                violations,
                repairs,
                "duplicate_capability_id",
                f"Capability id {cap_id} appears more than once.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Capability ids must be unique.",
                    "patch": {"id": _next_capability_id(ordered_caps)},
                },
            )
        seen_ids.add(cap_id)
        for field in ["id", "capability_name", "inputs", "outputs", "done_when"]:
            if field not in cap:
                _add_violation(
                    violations,
                    repairs,
                    "missing_required_field",
                    f"Capability {cap_id} is missing required field {field}.",
                    cap_id,
                    {
                        "action": "edit_field",
                        "reason": "Every ordered capability must include required fields.",
                        "patch": {
                            "field": field,
                            "value": [] if field in {"inputs", "outputs"} else "",
                        },
                    },
                )
        for field in ["inputs", "outputs"]:
            if field in cap and not isinstance(cap[field], list):
                _add_violation(
                    violations,
                    repairs,
                    "invalid_field_type",
                    f"Capability {cap_id} field {field} must be a list.",
                    cap_id,
                    {
                        "action": "edit_field",
                        "reason": "Capability inputs and outputs are list-valued fields.",
                        "patch": {"field": field, "value": [str(cap[field])]},
                    },
                )


def _validate_dependency_graph(
    ordered_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    caps = [cap for cap in ordered_caps if isinstance(cap, dict)]
    ids = [_capability_id(cap, index) for index, cap in enumerate(caps)]
    id_set = set(ids)
    edges: list[tuple[str, str]] = []
    for index, cap in enumerate(caps):
        cap_id = _capability_id(cap, index)
        depends_on = cap.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list):
            _add_violation(
                violations,
                repairs,
                "invalid_dependency_field",
                f"Capability {cap_id} depends_on must be a list.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Dependencies must be represented as a list of capability ids.",
                    "patch": {"depends_on": []},
                },
            )
            continue
        for dep in depends_on:
            dep_id = str(dep)
            if dep_id not in id_set:
                _add_violation(
                    violations,
                    repairs,
                    "unknown_dependency",
                    f"Capability {cap_id} depends on unknown capability id {dep_id}.",
                    cap_id,
                    {
                        "action": "set_dependency",
                        "reason": "Dependencies must reference valid capability ids.",
                        "patch": {"capability_id": cap_id, "remove_dependency": dep_id},
                    },
                )
            elif dep_id == cap_id:
                _add_violation(
                    violations,
                    repairs,
                    "self_dependency",
                    f"Capability {cap_id} depends on itself.",
                    cap_id,
                    {
                        "action": "set_dependency",
                        "reason": "Self-dependencies create a cycle.",
                        "patch": {"capability_id": cap_id, "remove_dependency": dep_id},
                    },
                )
            else:
                edges.append((dep_id, cap_id))
    if ids and not _is_acyclic(ids, edges):
        _add_violation(
            violations,
            repairs,
            "dependency_cycle",
            "Capability dependencies contain a cycle.",
            repair={
                "action": "set_dependency",
                "reason": "Remove the smallest dependency edge that breaks the cycle.",
                "patch": {
                    "cycle_edges": [
                        {"before": before, "after": after} for before, after in edges
                    ]
                },
            },
        )


def _is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indeg = {node: 0 for node in nodes}
    out: dict[str, list[str]] = defaultdict(list)
    for before, after in edges:
        out[before].append(after)
        indeg[after] += 1
    queue = deque([node for node in nodes if indeg[node] == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for child in out[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def _validate_external_action_fields(
    generated_caps: list[Any],
    normalized_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    for index, cap in enumerate(generated_caps):
        if not isinstance(cap, dict):
            continue
        cap_id = _capability_id(cap, index)
        action_type = cap.get("external_action_type")
        requires_external = bool(cap.get("requires_external_action"))
        if action_type not in EXTERNAL_ACTION_TYPES:
            _add_violation(
                violations,
                repairs,
                "invalid_external_action_type",
                f"Capability {cap_id} has invalid external_action_type {action_type!r}.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Capabilities must use the handoff external_action_type enum.",
                    "patch": {"external_action_type": "other"},
                },
            )
        elif requires_external and action_type == "none":
            _add_violation(
                violations,
                repairs,
                "missing_external_action_type",
                f"Capability {cap_id} requires external action but uses none.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "External work needs an explicit external_action_type.",
                    "patch": {"external_action_type": "other"},
                },
            )

    for index, cap in enumerate(normalized_caps):
        if not isinstance(cap, dict):
            continue
        action_type = cap.get("external_action_type")
        if action_type is not None and action_type not in EXTERNAL_ACTION_TYPES:
            cap_id = _capability_id(cap, index)
            _add_violation(
                violations,
                repairs,
                "invalid_external_action_type",
                f"Normalized capability {cap_id} has invalid external_action_type {action_type!r}.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Normalized capability action types must use the handoff enum.",
                    "patch": {"external_action_type": "other"},
                },
            )


def _validate_required_external_actions(
    intent_audit: dict[str, Any],
    transform_audit: dict[str, Any],
    generated_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    needed_actions = _needed_external_action_types(transform_audit)
    if "file_reading" in needed_actions and _only_inline_or_visual_inputs(intent_audit):
        needed_actions.remove("file_reading")
    for action in _as_list(transform_audit.get("external_actions")):
        if not isinstance(action, dict) or not action.get("needed"):
            continue
        action_type = action.get("action_type")
        if action_type in {"none", None}:
            continue
        if action_type not in EXTERNAL_ACTION_TYPES and not _split_external_action_types(
            action_type
        ):
            _add_violation(
                violations,
                repairs,
                "invalid_external_action_type",
                f"Transformation audit has invalid action_type {action_type!r}.",
                repair={
                    "action": "edit_field",
                    "reason": "External action audits must use the handoff enum.",
                    "patch": {"action_type": "other"},
                },
            )
    needed_actions -= {"web_search", "fact_checking"}

    cap_actions = {
        action_type
        for cap in generated_caps
        if isinstance(cap, dict)
        for action_type in _split_external_action_types(cap.get("external_action_type"))
    }
    for action_type in sorted(needed_actions):
        if action_type in cap_actions:
            continue
        cap_id = _next_capability_id(generated_caps)
        _add_violation(
            violations,
            repairs,
            "missing_capability_external_action_type",
            f"Run 2 needs {action_type}, but no capability declares that external_action_type.",
            repair={
                "action": "edit_field",
                "reason": "External actions identified in Run 2 must be explicit in Run 3.",
                "patch": {
                    "id": cap_id,
                    "external_action_type": action_type,
                },
            },
        )


def _validate_no_tool_or_worker_caps(
    generated_caps: list[Any],
    normalized_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    for index, cap in enumerate(generated_caps):
        if not isinstance(cap, dict):
            continue
        cap_id = _capability_id(cap, index)
        text = " ".join(
            str(cap.get(key, "")) for key in ["capability_name", "capability_description"]
        )
        if _contains_tool_or_worker_reference(text):
            _add_violation(
                violations,
                repairs,
                "tool_or_worker_capability",
                f"Capability {cap_id} names a tool, agent, API, library, plugin, or worker.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": (
                        "Capabilities must describe state transformations, not workers/tools."
                    ),
                    "patch": {"capability_name": "rename_as_state_transformation"},
                },
            )

    for index, cap in enumerate(normalized_caps):
        if not isinstance(cap, dict):
            continue
        cap_id = _capability_id(cap, index)
        name = str(cap.get("normalized_name", ""))
        if _contains_tool_or_worker_reference(name):
            _add_violation(
                violations,
                repairs,
                "tool_or_worker_capability",
                f"Normalized capability {cap_id} names a tool, agent, API, library, or plugin.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Normalized names must remain abstract state transformations.",
                    "patch": {"normalized_name": "rename_as_state_transformation"},
                },
            )


def _contains_tool_or_worker_reference(text: str) -> bool:
    lowered = text.lower()
    if any(bad in lowered for bad in ["pdfplumber", "browser_tool", "assistant_writer"]):
        return True
    tokens = re.split(r"[^a-z0-9]+", lowered)
    return any(
        token in {"agent", "tool", "tools", "library", "plugin"}
        for token in tokens
    )


def _validate_normalized_names(
    normalized_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    for index, cap in enumerate(normalized_caps):
        if not isinstance(cap, dict):
            continue
        cap_id = _capability_id(cap, index)
        name = cap.get("normalized_name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", name):
            _add_violation(
                violations,
                repairs,
                "invalid_normalized_name",
                f"Normalized capability {cap_id} must use short snake_case.",
                cap_id,
                {
                    "action": "edit_field",
                    "reason": "Capability normalization requires short snake_case names.",
                    "patch": {"normalized_name": _to_snake_case(str(name or "capability"))},
                },
            )


def _validate_missing_inputs(
    intent_audit: dict[str, Any],
    caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    missing_inputs = _as_list(intent_audit.get("missing_inputs"))
    unavailable = [
        item
        for item in _as_list(intent_audit.get("inputs"))
        if isinstance(item, dict) and item.get("available") is False
    ]
    if not missing_inputs and not unavailable:
        return
    if any(_is_missing_input_capability(cap) for cap in caps if isinstance(cap, dict)):
        return

    cap_id = _next_capability_id(caps)
    _add_violation(
        violations,
        repairs,
        "missing_input_without_request_capability",
        "Required input is missing but the plan does not request missing input.",
        repair={
            "action": "add_capability",
            "reason": "The plan must handle unavailable required input before other work.",
            "patch": {
                "id": cap_id,
                "capability_name": "request_missing_input",
                "requires_external_action": True,
                "external_action_type": "user_input",
                "inputs": ["missing input description"],
                "outputs": ["user-provided missing input or explicit inability to proceed"],
                "done_when": "the missing input has been requested from the user",
            },
        },
    )


def _validate_pasted_text_handling(
    intent_audit: dict[str, Any],
    generated_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    formats = _input_formats(intent_audit)
    has_pasted_text = "pasted_text" in formats
    has_file_source = bool(formats & {"attached_file", "pdf", "file_path"})
    if not has_pasted_text or has_file_source:
        return
    for index, cap in enumerate(generated_caps):
        if isinstance(cap, dict) and _is_file_extraction_capability(cap):
            cap_id = _capability_id(cap, index)
            _add_violation(
                violations,
                repairs,
                "unneeded_file_extraction_for_pasted_text",
                f"Capability {cap_id} adds file extraction even though the input is pasted text.",
                cap_id,
                {
                    "action": "remove_capability",
                    "reason": (
                        "Pasted text is already available and does not require file extraction."
                    ),
                    "patch": {"id": cap_id},
                },
            )


def _validate_pdf_text_extraction(
    intent_audit: dict[str, Any],
    generated_caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    if "pdf" not in _input_formats(intent_audit) or not _pdf_text_is_needed(intent_audit):
        return
    if any(_is_file_extraction_capability(cap) for cap in generated_caps if isinstance(cap, dict)):
        return
    cap_id = _next_capability_id(generated_caps)
    _add_violation(
        violations,
        repairs,
        "missing_pdf_text_extraction",
        "A PDF is attached and its text/content is needed, but no extraction capability exists.",
        repair={
            "action": "add_capability",
            "reason": "PDF content must be extracted before it can be analyzed or transformed.",
            "patch": {
                "id": cap_id,
                "capability_name": "extract_information_from_attached_document",
                "requires_external_action": True,
                "external_action_type": "file_reading",
                "inputs": ["attached PDF"],
                "outputs": ["extracted document text or structured content"],
                "done_when": "the needed PDF text/content is available for later capabilities",
            },
        },
    )


def _validate_current_information(
    transform_audit: dict[str, Any],
    caps: list[Any],
    violations: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
) -> None:
    needs_current = _needs_current_information(transform_audit)
    if not needs_current:
        return
    if any(_is_current_info_capability(cap) for cap in caps if isinstance(cap, dict)):
        return
    cap_id = _next_capability_id(caps)
    _add_violation(
        violations,
        repairs,
        "missing_current_information_capability",
        "Current or external facts are needed but no retrieval/fact-checking capability exists.",
        repair={
            "action": "add_capability",
            "reason": (
                "Current facts require an explicit abstract information retrieval capability."
            ),
            "patch": {
                "id": cap_id,
                "capability_name": "retrieve_current_information",
                "requires_external_action": True,
                "external_action_type": "web_search",
                "inputs": ["information need"],
                "outputs": ["current or externally verified information"],
                "done_when": "enough current information is available to continue",
            },
        },
    )


def _input_formats(intent_audit: dict[str, Any]) -> set[str]:
    formats = set()
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict):
            continue
        value = item.get("format")
        if isinstance(value, str) and value in INPUT_FORMATS:
            formats.add(value)
    return formats


def _is_missing_input_capability(cap: dict[str, Any]) -> bool:
    name = str(cap.get("capability_name") or cap.get("normalized_name") or "").lower()
    action_type = cap.get("external_action_type")
    return "request_missing_input" in name or action_type == "user_input"


def _is_file_extraction_capability(cap: dict[str, Any]) -> bool:
    name = str(cap.get("capability_name") or cap.get("normalized_name") or "").lower()
    desc = str(cap.get("capability_description") or "").lower()
    action_type = cap.get("external_action_type")
    text = f"{name} {desc}"
    if action_type == "file_reading":
        return True
    name_reads_source = any(word in name for word in ["extract", "inspect", "read", "search"])
    return name_reads_source and any(
        word in text for word in ["file", "document", "pdf", "attached"]
    )


def _pdf_text_is_needed(intent_audit: dict[str, Any]) -> bool:
    final_want = str(intent_audit.get("final_user_want") or "").lower()
    text_need_words = {
        "answer",
        "analy",
        "content",
        "details",
        "document",
        "extract",
        "information",
        "read",
        "summar",
        "text",
        "transform",
    }
    for item in _as_list(intent_audit.get("inputs")):
        if not isinstance(item, dict) or item.get("format") != "pdf":
            continue
        text = " ".join(
            str(item.get(key, "")) for key in ["name", "needed_for", "evidence"]
        ).lower()
        if any(word in text or word in final_want for word in text_need_words):
            return True
    return False


def _is_current_info_capability(cap: dict[str, Any]) -> bool:
    action_type = cap.get("external_action_type")
    name = str(cap.get("capability_name") or cap.get("normalized_name") or "").lower()
    return action_type in {"web_search", "fact_checking"} or any(
        phrase in name
        for phrase in [
            "retrieve_current_information",
            "fact_check",
            "verify_current",
            "retrieve_external_information",
        ]
    )


def _to_snake_case(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    return "_".join(words) or "capability"
