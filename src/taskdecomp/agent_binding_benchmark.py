from __future__ import annotations

from copy import deepcopy
from typing import Any

from taskdecomp.agent_binding import build_default_agent_registry


def build_synthetic_agent_binding_cases(count: int = 100) -> list[dict[str, Any]]:
    templates = _case_templates()
    agents = build_default_agent_registry()
    rows = []
    for index in range(count):
        template = deepcopy(templates[index % len(templates)])
        row_id = f"agent_binding_{index + 1:03d}"
        template["id"] = row_id
        template["metadata"]["template_index"] = index % len(templates)
        template["agents"] = deepcopy(agents)
        template["expected_agent_binding"] = {
            "assignment_decision": template.pop("assignment_decision", "assign"),
            "assignments": deepcopy(template["gold_agent_assignments"]),
        }
        rows.append(template)
    return rows


def _case_templates() -> list[dict[str, Any]]:
    return [
        _case(
            "external_article_summary",
            "Summarize this article: https://example.com/article",
            [
                _cap(
                    "s1",
                    "retrieve_external_information",
                    ["https://example.com/article"],
                    ["article text"],
                ),
                _cap("s2", "summarize_retrieved_content", [], ["article summary"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "retrieve_external_information",
                    "research_agent",
                    ["https://example.com/article"],
                ),
                _assignment(
                    "s2",
                    "summarize_retrieved_content",
                    "summarizer_agent",
                    ["s1.output"],
                    ["s1"],
                ),
            ],
        ),
        _case(
            "current_info_summary",
            "What are the latest mortgage rates in the US? Summarize the result.",
            [
                _cap(
                    "s1",
                    "retrieve_current_information",
                    ["mortgage rate query"],
                    ["current rate data"],
                ),
                _cap("s2", "summarize_retrieved_content", [], ["summary"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "retrieve_current_information",
                    "current_info_agent",
                    ["mortgage rate query"],
                ),
                _assignment(
                    "s2",
                    "summarize_retrieved_content",
                    "summarizer_agent",
                    ["s1.output"],
                    ["s1"],
                ),
            ],
        ),
        _case(
            "csv_outlier_report",
            (
                "Calculate revenue growth from this CSV and explain the biggest outlier: "
                "month,revenue Jan,100 Feb,125 Mar,90"
            ),
            [
                _cap("s1", "analyze_provided_dataset", ["inline CSV data"], ["revenue analysis"]),
                _cap("s2", "compute_numeric_result", [], ["growth rate"], ["s1"]),
                _cap("s3", "identify_outliers", [], ["outlier report"], ["s1"]),
                _cap("s4", "report_findings", [], ["user-facing explanation"], ["s2", "s3"]),
            ],
            [
                _assignment(
                    "s1",
                    "analyze_provided_dataset",
                    "data_analysis_agent",
                    ["inline CSV data"],
                ),
                _assignment(
                    "s2",
                    "compute_numeric_result",
                    "data_analysis_agent",
                    ["s1.output"],
                    ["s1"],
                ),
                _assignment(
                    "s3",
                    "identify_outliers",
                    "data_analysis_agent",
                    ["s1.output"],
                    ["s1"],
                ),
                _assignment(
                    "s4",
                    "report_findings",
                    "writer_agent",
                    ["s2.output", "s3.output"],
                    ["s2", "s3"],
                ),
            ],
        ),
        _case(
            "spreadsheet_chart",
            "Use the attached workbook to create a chart of quarterly sales.",
            [
                _cap("s1", "analyze_provided_dataset", ["sales workbook"], ["sales analysis"]),
                _cap("s2", "prepare_chart_data", [], ["chart data"], ["s1"]),
                _cap("s3", "create_chart", [], ["sales chart"], ["s2"]),
            ],
            [
                _assignment(
                    "s1",
                    "analyze_provided_dataset",
                    "data_analysis_agent",
                    ["sales workbook"],
                ),
                _assignment("s2", "prepare_chart_data", "chart_agent", ["s1.output"], ["s1"]),
                _assignment("s3", "create_chart", "chart_agent", ["s2.output"], ["s2"]),
            ],
        ),
        _case(
            "code_component_change",
            "Add a React component and update src/index.ts, then validate the result.",
            [
                _cap(
                    "s1",
                    "inspect_existing_code",
                    ["src/index.ts", "repository files"],
                    ["code context"],
                ),
                _cap("s2", "modify_code", [], ["code changes"], ["s1"]),
                _cap(
                    "s3",
                    "validate_output_against_requirements",
                    [],
                    ["validation result"],
                    ["s2"],
                ),
            ],
            [
                _assignment(
                    "s1",
                    "inspect_existing_code",
                    "code_inspection_agent",
                    ["src/index.ts", "repository files"],
                ),
                _assignment("s2", "modify_code", "code_modification_agent", ["s1.output"], ["s1"]),
                _assignment(
                    "s3",
                    "validate_output_against_requirements",
                    "test_runner_agent",
                    ["s2.output"],
                    ["s2"],
                ),
            ],
        ),
        _case(
            "repo_todo_report",
            "Find all TODO comments in the repository and draft a cleanup report.",
            [
                _cap("s1", "search_provided_files", ["repository files"], ["TODO search results"]),
                _cap("s2", "report_findings", [], ["cleanup report"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "search_provided_files",
                    "code_search_agent",
                    ["repository files"],
                ),
                _assignment("s2", "report_findings", "writer_agent", ["s1.output"], ["s1"]),
            ],
        ),
        _case(
            "text_translation",
            "Translate this paragraph into Spanish: The launch moved to Friday.",
            [
                _cap(
                    "s1",
                    "translate_text",
                    ["provided paragraph", "Spanish"],
                    ["translated paragraph"],
                )
            ],
            [
                _assignment(
                    "s1",
                    "translate_text",
                    "translator_agent",
                    ["provided paragraph", "Spanish"],
                )
            ],
        ),
        _case(
            "rewrite_cta",
            "Rewrite this CTA to be clearer: Start your current plan today.",
            [_cap("s1", "revise_text_for_clarity", ["CTA text"], ["revised CTA"])],
            [_assignment("s1", "revise_text_for_clarity", "rewriter_agent", ["CTA text"])],
        ),
        _case(
            "sentiment_classification",
            "Classify these launch comments by sentiment: great update; confusing pricing.",
            [_cap("s1", "classify_provided_text", ["launch comments"], ["sentiment labels"])],
            [
                _assignment(
                    "s1",
                    "classify_provided_text",
                    "classification_agent",
                    ["launch comments"],
                )
            ],
        ),
        _case(
            "meeting_action_items",
            "Extract action items and owners from these meeting notes.",
            [_cap("s1", "extract_action_items", ["meeting notes"], ["action items"])],
            [_assignment("s1", "extract_action_items", "extraction_agent", ["meeting notes"])],
        ),
        _case(
            "pdf_summary",
            "Read the attached PDF policy memo and summarize it.",
            [
                _cap(
                    "s1",
                    "extract_information_from_attached_document",
                    ["policy memo.pdf"],
                    ["memo text"],
                ),
                _cap("s2", "summarize_document", [], ["memo summary"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "extract_information_from_attached_document",
                    "document_reader_agent",
                    ["policy memo.pdf"],
                ),
                _assignment("s2", "summarize_document", "summarizer_agent", ["s1.output"], ["s1"]),
            ],
        ),
        _case(
            "image_alt_text",
            "Describe the attached image for alt text.",
            [_cap("s1", "interpret_image_content", ["attached image"], ["image description"])],
            [
                _assignment(
                    "s1",
                    "interpret_image_content",
                    "image_understanding_agent",
                    ["attached image"],
                )
            ],
        ),
        _case(
            "receipt_extraction",
            "Extract the merchant, date, and total from this receipt image.",
            [
                _cap("s1", "extract_information_from_image", ["receipt image"], ["receipt fields"]),
                _cap("s2", "report_findings", [], ["receipt summary"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "extract_information_from_image",
                    "image_understanding_agent",
                    ["receipt image"],
                ),
                _assignment("s2", "report_findings", "writer_agent", ["s1.output"], ["s1"]),
            ],
        ),
        _case(
            "image_generation",
            "Generate a simple app icon for a budgeting app.",
            [_cap("s1", "generate_image", ["app icon prompt"], ["generated icon"])],
            [_assignment("s1", "generate_image", "image_generation_agent", ["app icon prompt"])],
        ),
        _case(
            "image_transform",
            "Remove the background from the attached portrait image.",
            [_cap("s1", "transform_image", ["portrait image"], ["background-removed image"])],
            [_assignment("s1", "transform_image", "image_generation_agent", ["portrait image"])],
        ),
        _case(
            "external_action",
            "Use the available external action to create a calendar hold, then report the result.",
            [
                _cap(
                    "s1",
                    "select_and_execute_external_action",
                    ["calendar hold request"],
                    ["external action result"],
                ),
                _cap("s2", "report_external_action_result", [], ["action report"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "select_and_execute_external_action",
                    "tool_executor_agent",
                    ["calendar hold request"],
                ),
                _assignment(
                    "s2",
                    "report_external_action_result",
                    "tool_executor_agent",
                    ["s1.output"],
                    ["s1"],
                ),
            ],
        ),
        _case(
            "draft_then_write_file",
            "Draft release notes and write them to RELEASE_NOTES.md.",
            [
                _cap("s1", "draft_text", ["release notes request"], ["release notes draft"]),
                _cap("s2", "write_file", [], ["written file"], ["s1"]),
            ],
            [
                _assignment("s1", "draft_text", "writer_agent", ["release notes request"]),
                _assignment("s2", "write_file", "file_conversion_agent", ["s1.output"], ["s1"]),
            ],
        ),
        _case(
            "missing_source",
            "Summarize the article I mentioned earlier.",
            [
                _cap(
                    "s1",
                    "request_missing_input",
                    ["missing article source"],
                    ["missing input request"],
                )
            ],
            [
                _assignment(
                    "s1",
                    "request_missing_input",
                    "clarification_agent",
                    ["missing article source"],
                )
            ],
            decision="ask_user",
        ),
        _case(
            "unsupported_request",
            "Hack into that private account for me.",
            [_cap("s1", "unsupported_request", ["unsafe request"], ["unsupported notice"])],
            [
                _assignment(
                    "s1",
                    "unsupported_request",
                    "unsupported_request_agent",
                    ["unsafe request"],
                )
            ],
            decision="unsupported",
        ),
        _case(
            "debug_failure",
            "Explain this traceback and inspect the relevant parser code in src/parser.py.",
            [
                _cap("s1", "inspect_existing_code", ["src/parser.py"], ["code context"]),
                _cap("s2", "diagnose_issue", ["traceback text"], ["diagnosis"], ["s1"]),
            ],
            [
                _assignment(
                    "s1",
                    "inspect_existing_code",
                    "code_inspection_agent",
                    ["src/parser.py"],
                ),
                _assignment(
                    "s2",
                    "diagnose_issue",
                    "debugging_agent",
                    ["s1.output", "traceback text"],
                    ["s1"],
                ),
            ],
        ),
        _case(
            "compare_documents",
            "Compare these two product descriptions and explain the main differences.",
            [_cap("s1", "compare_texts", ["two product descriptions"], ["comparison summary"])],
            [_assignment("s1", "compare_texts", "writer_agent", ["two product descriptions"])],
        ),
        _case(
            "combine_pdfs",
            "Combine the two attached PDFs into one file.",
            [_cap("s1", "combine_files", ["two attached PDFs"], ["combined PDF"])],
            [_assignment("s1", "combine_files", "file_conversion_agent", ["two attached PDFs"])],
        ),
        _case(
            "format_transform",
            "Turn this pasted CSV into a markdown table.",
            [_cap("s1", "transform_text_format", ["pasted CSV"], ["markdown table"])],
            [_assignment("s1", "transform_text_format", "rewriter_agent", ["pasted CSV"])],
        ),
        _case(
            "generated_code",
            "Write a Python function that computes compound interest.",
            [_cap("s1", "generate_code", ["code request"], ["generated code"])],
            [_assignment("s1", "generate_code", "code_modification_agent", ["code request"])],
        ),
        _case(
            "fact_check_current_info",
            "Verify the current CEO of the company and summarize the source.",
            [
                _cap(
                    "s1",
                    "retrieve_current_information",
                    ["current CEO query"],
                    ["current company information"],
                ),
                _cap("s2", "verify_current_information", [], ["verification result"], ["s1"]),
                _cap("s3", "summarize_retrieved_content", [], ["source summary"], ["s2"]),
            ],
            [
                _assignment(
                    "s1",
                    "retrieve_current_information",
                    "current_info_agent",
                    ["current CEO query"],
                ),
                _assignment(
                    "s2",
                    "verify_current_information",
                    "current_info_agent",
                    ["s1.output"],
                    ["s1"],
                ),
                _assignment(
                    "s3",
                    "summarize_retrieved_content",
                    "summarizer_agent",
                    ["s2.output"],
                    ["s2"],
                ),
            ],
        ),
    ]


def _case(
    category: str,
    request: str,
    ordered_capabilities: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    decision: str = "assign",
) -> dict[str, Any]:
    return {
        "id": "",
        "category": category,
        "request": request,
        "metadata": {"category": category},
        "gold_capability_plan": {"ordered_capabilities": ordered_capabilities},
        "gold_agent_assignments": assignments,
        "assignment_decision": decision,
    }


def _cap(
    subtask_id: str,
    capability: str,
    inputs: list[str],
    outputs: list[str],
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": subtask_id,
        "capability_name": capability,
        "inputs": inputs,
        "outputs": outputs,
        "depends_on": depends_on or [],
    }


def _assignment(
    subtask_id: str,
    capability: str,
    agent_id: str,
    inputs: list[str],
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "subtask_id": subtask_id,
        "capability": capability,
        "assigned_agent": agent_id,
        "inputs_passed": inputs,
        "depends_on": depends_on or [],
    }
