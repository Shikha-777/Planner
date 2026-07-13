#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generate_capability_holdout import case, expected, inp


OUT = Path("data/capability_planning/property_holdout_50.jsonl")


def prop_case(
    id_: str,
    property_: str,
    category: str,
    request: str,
    exp: dict[str, Any],
    attachments: list[dict[str, Any]] | None = None,
    mutation: str | None = None,
) -> dict[str, Any]:
    row = case(id_, category, request, exp, attachments)
    row["property"] = property_
    if mutation:
        row["source_mutation"] = mutation
    return row


CASES = [
    prop_case(
        "property_source_where_select_classify_001",
        "source_text_must_not_route_operation",
        "pasted_text_analysis",
        "Categorize these comments as praise, bug, or question:\n"
        "1. Where is billing history?\n2. SELECT your plan now.\n3. Start today.",
        expected(final=["categorize", "comments"], inputs=[inp("comments", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["classify_provided_text"]),
        mutation="source contains where/select/today",
    ),
    prop_case(
        "property_source_today_compare_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_analysis",
        "Compare these CTAs and pick the more direct one.\n\nA: Start today and save.\n\nB: SELECT your plan now.",
        expected(final=["compare", "CTAs"], inputs=[inp("CTAs", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["compare_texts"]),
        mutation="source contains today/select",
    ),
    prop_case(
        "property_source_current_rewrite_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_transform",
        "Rewrite this policy blurb to be clearer: Users may cancel after the current renewal term has elapsed.",
        expected(final=["rewrite", "policy"], inputs=[inp("policy", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        mutation="source contains current",
    ),
    prop_case(
        "property_source_latest_summarize_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_analysis",
        "Summarize this memo in two bullets: The latest rollout note says support is staffed and monitoring continues today.",
        expected(final=["summarize", "memo"], inputs=[inp("memo", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
        mutation="source contains latest/today",
    ),
    prop_case(
        "property_source_select_translate_001",
        "source_text_must_not_route_operation",
        "pasted_text_transform",
        "Translate this into Spanish: SELECT your plan today.",
        expected(final=["translate", "Spanish"], inputs=[inp("SELECT", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["translate_text"]),
        mutation="source contains select/today",
    ),
    prop_case(
        "property_source_function_rewrite_001",
        "source_text_must_not_route_operation",
        "pasted_text_transform",
        "Make this sentence clearer: The function of the current policy is to reduce refund confusion.",
        expected(final=["clearer", "sentence"], inputs=[inp("sentence", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        mutation="source contains function/current",
    ),
    prop_case(
        "property_source_stock_classify_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_analysis",
        "Classify these snippets as billing, product, or sales:\n1. What is the stock price today?\n2. Invoice export failed.\n3. Demo booked.",
        expected(final=["classify", "snippets"], inputs=[inp("snippets", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["classify_provided_text"]),
        mutation="source contains stock price/today",
    ),
    prop_case(
        "property_source_weather_compare_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_analysis",
        "Compare these two headlines and pick the clearer one.\n\nA: Weather your billing storms today.\n\nB: Resolve invoice issues faster.",
        expected(final=["compare", "headlines"], inputs=[inp("headlines", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["compare_texts"]),
        mutation="source contains weather/today",
    ),
    prop_case(
        "property_source_sql_code_fence_summary_001",
        "source_text_must_not_route_operation",
        "pasted_text_analysis",
        "Summarize this snippet:\n\n```sql\nSELECT user_id FROM events WHERE created_at >= CURRENT_DATE;\n```",
        expected(final=["summarize", "snippet"], inputs=[inp("SELECT", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
        mutation="code fence contains sql/current",
    ),
    prop_case(
        "property_source_todo_markdown_table_001",
        "source_text_must_not_route_operation",
        "pasted_text_transform",
        "Turn these notes into a Markdown table: TODO owner is Alex; current blocker is legal; latest date is Friday.",
        expected(final=["Markdown table"], inputs=[inp("notes", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["transform_text_format"]),
        mutation="source contains todo/current/latest",
    ),
    prop_case(
        "property_source_where_extract_001",
        "source_text_must_not_route_operation",
        "pasted_text_analysis",
        "Extract action items from these notes:\n\nWhere is the final deck? Mira owns the update. Start today if legal approves.",
        expected(final=["action items", "notes"], inputs=[inp("notes", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"]),
        mutation="source contains where/today",
    ),
    prop_case(
        "property_source_current_exception_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_analysis",
        "Explain this exception:\n\nRuntimeError: current transaction closed where status='pending'",
        expected(final=["explain", "exception"], inputs=[inp("exception", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"]),
        mutation="source contains current/where",
    ),
    prop_case(
        "property_source_latest_roster_table_001",
        "source_text_must_not_trigger_current_info",
        "pasted_text_transform",
        "Turn this roster into a Markdown table:\nAri - current owner - starts today\nBo - latest hire - starts Monday",
        expected(final=["Markdown table"], inputs=[inp("roster", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["transform_text_format"]),
        mutation="newline payload contains current/today/latest",
    ),
    prop_case(
        "property_source_select_quote_translate_001",
        "quoted_source_available",
        "pasted_text_transform",
        'Translate "SELECT your plan today" into French.',
        expected(final=["translate", "French"], inputs=[inp("SELECT", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["translate_text"]),
        mutation="quoted source contains select/today",
    ),
    prop_case(
        "property_source_current_quote_rewrite_001",
        "quoted_source_available",
        "pasted_text_transform",
        'Make "The current process starts today" sound warmer.',
        expected(final=["warmer"], inputs=[inp("current process", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        mutation="quoted source contains current/today",
    ),
    prop_case(
        "property_source_where_classify_newline_001",
        "newline_source_available",
        "pasted_text_analysis",
        "Classify each line as question or statement:\nWhere is the invoice?\nThe current plan renews today.",
        expected(final=["classify", "line"], inputs=[inp("invoice", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["classify_provided_text"]),
        mutation="newline source contains where/current/today",
    ),
    prop_case(
        "property_source_current_summary_blankline_001",
        "blank_line_source_available",
        "pasted_text_analysis",
        "Summarize this announcement.\n\nThe current launch date is today. SELECT teams will get early access.",
        expected(final=["summarize", "announcement"], inputs=[inp("announcement", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
        mutation="blank-line source contains current/today/select",
    ),
    prop_case(
        "property_source_sql_no_code_generation_001",
        "source_text_must_not_route_operation",
        "pasted_text_analysis",
        "Explain this customer message:\n\nCan I SELECT a cheaper plan and where do I downgrade today?",
        expected(final=["explain", "customer message"], inputs=[inp("customer message", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"]),
        mutation="source contains select/where/today",
    ),
    prop_case(
        "property_source_url_text_not_retrieval_001",
        "source_text_url_can_be_plain_text_when_instruction_says_rewrite",
        "pasted_text_transform",
        "Rewrite this sentence for clarity: Visit https://example.com/current-plan today to SELECT a package.",
        expected(final=["rewrite", "sentence"], inputs=[inp("Visit", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        mutation="source contains url/current/today/select",
    ),
    prop_case(
        "property_source_percent_calc_not_external_001",
        "source_text_must_not_route_operation",
        "pasted_text_analysis",
        "Summarize this note: Current conversion is 42% today, but the team only needs a concise summary.",
        expected(final=["summarize", "note"], inputs=[inp("conversion", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
        mutation="source contains percent/current/today",
    ),

    prop_case("property_missing_notes_themes_001", "source_consuming_missing_source", "missing_input", "Turn my stakeholder notes into themes.", expected(final=["themes", "notes"], inputs=[inp("notes", False, "unknown")], missing=["notes"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_calls_objections_001", "source_consuming_missing_source", "missing_input", "Extract objections from the sales calls.", expected(final=["extract", "objections"], inputs=[inp("call", False, "unknown")], missing=["call"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_article_summary_001", "source_consuming_missing_source", "missing_input", "Can you summarize my article into three bullets?", expected(final=["summarize", "article"], inputs=[inp("article", False, "unknown")], missing=["article"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_comments_classify_001", "source_consuming_missing_source", "missing_input", "Classify my launch comments by sentiment.", expected(final=["classify", "comments"], inputs=[inp("comments", False, "unknown")], missing=["comments"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_bio_rewrite_001", "source_consuming_missing_source", "missing_input", "Can you make my founder bio sound more confident?", expected(final=["bio", "confident"], inputs=[inp("bio", False, "unknown")], missing=["bio"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_export_analysis_001", "source_consuming_missing_source", "missing_input", "Analyze my billing export for outliers.", expected(final=["analyze", "export"], inputs=[inp("export", False, "unknown")], missing=["export"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_two_proposals_compare_001", "source_consuming_missing_source", "missing_input", "Compare the two grant proposals and pick the stronger one.", expected(final=["compare", "proposals"], inputs=[inp("proposal", False, "unknown")], missing=["proposal"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_inventory_chart_001", "source_consuming_missing_source", "missing_input", "Chart inventory counts by warehouse.", expected(final=["chart", "inventory"], inputs=[inp("inventory", False, "unknown")], missing=["inventory"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_lab_notes_outline_001", "source_consuming_missing_source", "missing_input", "Convert my lab notes into an outline.", expected(final=["outline", "lab notes"], inputs=[inp("notes", False, "unknown")], missing=["notes"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),
    prop_case("property_missing_case_study_tighten_001", "source_consuming_missing_source", "missing_input", "Tighten my customer case study intro.", expected(final=["case study"], inputs=[inp("case study", False, "unknown")], missing=["case study"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"])),

    prop_case("property_current_latest_model_001", "instruction_current_info_still_routes", "current_fact", "What is the latest stable Python release?", expected(final=["latest", "Python"], inputs=[inp("Python", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"])),
    prop_case("property_current_stock_001", "instruction_current_info_still_routes", "current_fact", "What is Apple's stock price right now?", expected(final=["stock", "Apple"], inputs=[inp("stock", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"])),
    prop_case("property_current_weather_001", "instruction_current_info_still_routes", "current_fact", "Will it rain in Denver tomorrow?", expected(final=["rain", "Denver"], inputs=[inp("Denver", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"])),
    prop_case("property_fact_check_claim_001", "instruction_fact_check_still_routes", "fact_check", "Fact-check whether New York announced new subway fares this week.", expected(final=["fact", "subway"], inputs=[inp("subway", True, "none")], current=True, include_actions=["web_search", "fact_checking"], include_caps=["verify_current_information"], precede=[{"before": "verify", "after": "answer"}])),
    prop_case("property_current_travel_001", "instruction_current_info_still_routes", "current_fact", "Are there current travel restrictions for US travelers to Chile?", expected(final=["travel", "Chile"], inputs=[inp("travel", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"])),

    prop_case("property_code_sql_001", "instruction_code_generation_still_routes", "code_generation", "Write a SQL query that selects active customers from customers(id, active).", expected(final=["SQL", "customers"], inputs=[inp("SQL", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"])),
    prop_case("property_code_graphql_001", "instruction_code_generation_still_routes", "code_generation", "Write a GraphQL query for repository name and star count.", expected(final=["GraphQL", "repository"], inputs=[inp("GraphQL", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"])),
    prop_case("property_code_regex_001", "instruction_code_generation_still_routes", "code_generation", "Write a regex for invoice IDs like INV-2026-0042.", expected(final=["regex", "invoice"], inputs=[inp("regex", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"])),
    prop_case("property_url_summary_https_001", "url_scheme_colon_not_source_delimiter", "url_summary", "Summarize https://example.org/blog/current-pricing in four bullets.", expected(final=["summarize", "current pricing"], inputs=[inp("https://example.org/blog/current-pricing", True, "url")], current=True, include_actions=["web_search"], include_caps=["retrieve_external_information"], alt_caps=["summarize_document"])),
    prop_case("property_url_compare_https_001", "url_scheme_colon_not_source_delimiter", "url_summary", "Compare https://example.com/select and https://example.com/current.", expected(final=["compare"], inputs=[inp("https://example.com/select", True, "url")], current=True, include_actions=["web_search"], include_caps=["compare_texts"], precede=[{"before": "retrieve", "after": "compare"}])),

    prop_case("property_attached_pdf_summary_001", "metadata_source_still_routes", "attached_document", "Summarize the attached memo and list decisions.", expected(final=["summarize", "memo"], inputs=[inp("memo", True, "pdf")], current=False, include_actions=["file_reading"], include_caps=["extract_information_from_attached_document"], alt_caps=["summarize_document"], precede=[{"before": "extract", "after": "summarize"}]), attachments=[{"name": "current_select_memo.pdf", "format": "pdf", "available": True}]),
    prop_case("property_attached_csv_current_001", "metadata_source_still_routes", "structured_data", "Compute total revenue by plan from the attached CSV.", expected(final=["revenue", "plan"], inputs=[inp("CSV", True, "structured_data")], include_formats=["structured_data"], current=False, include_actions=["file_reading", "calculation"], include_caps=["analyze_provided_dataset"]), attachments=[{"name": "current_plan_revenue.csv", "format": "structured_data", "available": True}]),
    prop_case("property_repo_search_todo_001", "instruction_repo_search_still_routes", "repo_search", "Scan this repository for TODO comments and summarize files affected.", expected(final=["TODO", "repository"], inputs=[inp("repository", True, "file_path")], current=False, include_actions=["file_reading"], include_caps=["search_provided_files"]), attachments=[{"name": "repository files", "format": "file_path", "available": True}]),
    prop_case("property_code_edit_current_file_001", "metadata_source_still_routes", "code_task", "In src/current_select.ts, fix the failing parser test.", expected(final=["fix", "parser"], inputs=[inp("src/current_select.ts", True, "file_path")], current=False, include_actions=["file_reading", "file_writing", "code_execution"], include_caps=["inspect_existing_code", "modify_code"]), attachments=[{"name": "src/current_select.ts", "format": "file_path", "available": True}]),
    prop_case("property_image_alt_current_001", "metadata_source_still_routes", "image_understanding", "Write alt text for this image.", expected(final=["alt text", "image"], inputs=[inp("image", True, "image")], current=False, include_actions=["image_understanding"], include_caps=["interpret_image_content"]), attachments=[{"name": "today_select_chart.png", "format": "image", "available": True}]),
    prop_case("property_image_generation_instruction_001", "instruction_image_generation_still_routes", "image_generation", "Create a banner graphic for a webinar about current hiring trends.", expected(final=["banner", "webinar"], inputs=[inp("webinar", True, "none")], current=False, include_actions=["image_generation"], include_caps=["generate_image"])),
    prop_case("property_file_write_notes_current_001", "file_write_with_source_text", "file_task", "Create docs/update.md from these notes: current launch is today, SELECT customers notified.", expected(final=["update", "notes"], inputs=[inp("notes", True, "pasted_text"), inp("docs/update.md", True, "file_path")], current=False, include_actions=["file_writing"], include_caps=["write_file"]), attachments=[{"name": "docs/update.md", "format": "file_path", "available": True}]),
    prop_case("property_image_measure_missing_scale_001", "unsupported_missing_scale", "unsupported_missing_scale", "Measure the real-world height of the monitor in this image.", expected(final=["measure", "height"], inputs=[inp("image", True, "image"), inp("scale", False, "unknown")], missing=["scale"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]), attachments=[{"name": "current_monitor.png", "format": "image", "available": True}]),
    prop_case("property_arithmetic_001", "self_contained_calculation", "calculation", "What is 72 / 9 + 11?", expected(final=["72", "11"], inputs=[inp("numeric", True, "none")], current=False, include_actions=["calculation"], include_caps=["compute_numeric_result"])),
    prop_case("property_draft_email_current_word_001", "self_contained_draft_not_current", "text_generation", "Draft a short email saying the current rollout is complete and monitoring continues.", expected(final=["email", "rollout"], inputs=[inp("requirements", True, "none")], current=False, include_actions=["none"], include_caps=["draft_text"])),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in CASES:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
