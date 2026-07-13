#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_capability_holdout import case, expected, inp


DEFAULT_OUT = Path("data/capability_planning/public_mix_wrapped_60.jsonl")


def mixed_case(
    id_: str,
    source_dataset: str,
    wrapper_type: str,
    category: str,
    request: str,
    exp: dict[str, Any],
    attachments: list[dict[str, Any]] | None = None,
    source_text: str | None = None,
) -> dict[str, Any]:
    row = case(id_, category, request, exp, attachments)
    row["source_dataset"] = source_dataset
    row["wrapper_type"] = wrapper_type
    if source_text:
        row["wrapped_source_text"] = source_text
    return row


def source_transform_cases() -> list[dict[str, Any]]:
    contamination = {
        "current": "The current renewal term starts today and SELECT users can switch plans.",
        "sql": "SELECT user_id FROM events WHERE created_at >= CURRENT_DATE;",
        "url": "Visit https://example.com/current/select today to choose a package.",
        "stock": "What is the stock price today? Invoice export failed. Demo booked.",
        "todo": "TODO owner is Alex; current blocker is legal; latest date is Friday.",
        "weather": "Weather your billing storms today. Resolve invoice issues faster.",
    }
    return [
        mixed_case(
            "public_mix_sni_summarize_current_001",
            "super_natural_instructions",
            "summarize_with_source_contamination",
            "pasted_text_analysis",
            f"Summarize this memo in two bullets: {contamination['current']}",
            expected(inputs=[inp("memo", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
            source_text=contamination["current"],
        ),
        mixed_case(
            "public_mix_sni_rewrite_url_001",
            "super_natural_instructions",
            "rewrite_with_source_url",
            "pasted_text_transform",
            f"Rewrite this CTA to be clearer: {contamination['url']}",
            expected(inputs=[inp("CTA", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"], exclude_actions=["web_search"]),
            source_text=contamination["url"],
        ),
        mixed_case(
            "public_mix_sni_classify_stock_001",
            "super_natural_instructions",
            "classify_with_current_fact_decoy",
            "pasted_text_analysis",
            f"Classify these snippets as billing, product, or sales:\n1. {contamination['stock']}\n2. Checkout failed.\n3. Demo booked.",
            expected(inputs=[inp("snippets", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["classify_provided_text"]),
            source_text=contamination["stock"],
        ),
        mixed_case(
            "public_mix_sni_extract_todo_001",
            "super_natural_instructions",
            "extract_with_source_operation_decoys",
            "pasted_text_analysis",
            f"Extract action items from these notes:\n\n{contamination['todo']}",
            expected(inputs=[inp("notes", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"]),
            source_text=contamination["todo"],
        ),
        mixed_case(
            "public_mix_sni_translate_sql_001",
            "super_natural_instructions",
            "translate_with_code_decoy",
            "pasted_text_transform",
            f"Translate this line into Spanish: {contamination['sql']}",
            expected(inputs=[inp("SELECT", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["translate_text"], exclude_caps=["generate_code"]),
            source_text=contamination["sql"],
        ),
        mixed_case(
            "public_mix_sni_explain_message_001",
            "super_natural_instructions",
            "explain_provided_source",
            "pasted_text_analysis",
            "Explain this customer message:\n\nCan I SELECT a cheaper plan and where do I downgrade today?",
            expected(inputs=[inp("customer message", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"], exclude_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_sni_format_todo_001",
            "super_natural_instructions",
            "format_transform_with_current_decoys",
            "pasted_text_transform",
            f"Turn these notes into a Markdown table: {contamination['todo']}",
            expected(inputs=[inp("notes", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["transform_text_format"]),
            source_text=contamination["todo"],
        ),
        mixed_case(
            "public_mix_sni_compare_weather_001",
            "super_natural_instructions",
            "compare_with_weather_decoy",
            "pasted_text_analysis",
            f"Compare these two headlines and pick the clearer one.\n\nA: {contamination['weather']}\n\nB: Resolve invoice issues faster.",
            expected(inputs=[inp("headlines", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["compare_texts"]),
            source_text=contamination["weather"],
        ),
        mixed_case(
            "public_mix_sni_summarize_code_fence_001",
            "super_natural_instructions",
            "summarize_code_fence_as_source",
            "pasted_text_analysis",
            "Summarize this snippet:\n\n```sql\nSELECT account_id FROM invoices WHERE status = 'current';\n```",
            expected(inputs=[inp("SELECT", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"], exclude_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_sni_quote_rewrite_001",
            "super_natural_instructions",
            "quoted_source_transform",
            "pasted_text_transform",
            'Make "The current process starts today" sound warmer.',
            expected(inputs=[inp("current process", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        ),
        mixed_case(
            "public_mix_sni_newline_classify_001",
            "super_natural_instructions",
            "newline_source_classification",
            "pasted_text_analysis",
            "Classify each line as question or statement:\nWhere is the invoice?\nThe current plan renews today.",
            expected(inputs=[inp("invoice", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["classify_provided_text"]),
        ),
        mixed_case(
            "public_mix_sni_summary_percent_001",
            "super_natural_instructions",
            "summarize_metric_source_without_calculation",
            "pasted_text_analysis",
            "Summarize this note: Current conversion is 42% today, but the team only needs a concise summary.",
            expected(inputs=[inp("conversion", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["summarize_document"]),
        ),
    ]


def intent_paraphrase_cases() -> list[dict[str, Any]]:
    return [
        mixed_case(
            "public_mix_clinc_missing_resume_001",
            "clinc150_oos",
            "missing_source_request",
            "missing_input",
            "Can you improve my resume summary?",
            expected(inputs=[inp("resume", False, "unknown")], missing=["resume"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
        ),
        mixed_case(
            "public_mix_massive_missing_reviews_001",
            "massive",
            "missing_source_request",
            "missing_input",
            "Classify my product reviews by sentiment.",
            expected(inputs=[inp("reviews", False, "unknown")], missing=["reviews"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
        ),
        mixed_case(
            "public_mix_clinc_current_exchange_001",
            "clinc150_oos",
            "current_fact_intent",
            "current_facts",
            "What is today's USD to EUR exchange rate?",
            expected(inputs=[inp("exchange rate", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"]),
        ),
        mixed_case(
            "public_mix_massive_current_weather_001",
            "massive",
            "current_fact_intent",
            "current_facts",
            "Will it rain in Denver tomorrow morning?",
            expected(inputs=[inp("Denver", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"]),
        ),
        mixed_case(
            "public_mix_clinc_current_ceo_001",
            "clinc150_oos",
            "current_fact_intent",
            "current_facts",
            "Who is the current CEO of AMD?",
            expected(inputs=[inp("AMD", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"]),
        ),
        mixed_case(
            "public_mix_massive_draft_email_001",
            "massive",
            "self_contained_writing_intent",
            "text_generation",
            "Draft a short email saying onboarding is complete and support coverage starts Monday.",
            expected(inputs=[inp("email", True, "none")], current=False, include_actions=["none"], include_caps=["draft_text"]),
        ),
        mixed_case(
            "public_mix_clinc_missing_chart_001",
            "clinc150_oos",
            "missing_source_request",
            "missing_input",
            "Graph monthly signups by acquisition channel.",
            expected(inputs=[inp("signup", False, "unknown")], missing=["signup"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
        ),
        mixed_case(
            "public_mix_massive_concept_explain_001",
            "massive",
            "self_contained_explanation",
            "general_qa",
            "Explain recursion in simple terms.",
            expected(current=False, include_actions=["none"], include_caps=["provide_explanation"]),
        ),
    ]


def tool_and_code_cases() -> list[dict[str, Any]]:
    return [
        mixed_case(
            "public_mix_bfcl_sql_001",
            "bfcl",
            "function_or_query_generation",
            "generic_coding_parameters",
            "Write a Postgres query that returns monthly revenue from orders(created_at, total).",
            expected(inputs=[inp("Postgres", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_bfcl_graphql_001",
            "bfcl",
            "function_or_query_generation",
            "generic_coding_parameters",
            "Write a GraphQL query that fetches repository name, owner, and star count.",
            expected(inputs=[inp("GraphQL", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_conala_function_001",
            "conala",
            "code_generation_from_intent",
            "generic_coding_parameters",
            "Write a Python function longest_unique_subarray(nums, limit) that returns the longest contiguous slice with all values <= limit.",
            expected(inputs=[inp("function", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_codesearchnet_regex_001",
            "codesearchnet",
            "code_generation_from_intent",
            "generic_coding_parameters",
            "Write a regex pattern that matches invoice IDs like INV-2026-0042.",
            expected(inputs=[inp("regex", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_apibank_url_summary_001",
            "api_bank",
            "external_retrieval",
            "url",
            "Summarize https://example.org/blog/current-pricing in four bullets.",
            expected(inputs=[inp("https://example.org/blog/current-pricing", True, "url")], current=True, include_actions=["web_search"], include_caps=["retrieve_external_information"]),
        ),
        mixed_case(
            "public_mix_apibank_url_extract_001",
            "api_bank",
            "external_retrieval",
            "url",
            "Extract migration steps from https://example.com/releases/v3.",
            expected(inputs=[inp("https://example.com/releases/v3", True, "url")], current=True, include_actions=["web_search"], include_caps=["retrieve_external_information"]),
        ),
        mixed_case(
            "public_mix_toolbench_url_compare_001",
            "toolbench",
            "external_multi_source_comparison",
            "url",
            "Compare https://example.com/select and https://example.com/current.",
            expected(inputs=[inp("https://example.com/select", True, "url")], current=True, include_actions=["web_search"], include_caps=["compare_texts"], precede=[{"before": "retrieve", "after": "compare"}]),
        ),
        mixed_case(
            "public_mix_apibank_fact_check_001",
            "api_bank",
            "fact_checking",
            "fact_checking",
            "Fact-check whether Seattle city council passed a new tenant ordinance this month.",
            expected(inputs=[inp("tenant ordinance", True, "none")], current=True, include_actions=["web_search", "fact_checking"], include_caps=["verify_current_information"]),
        ),
        mixed_case(
            "public_mix_swebench_code_edit_001",
            "swe_bench",
            "code_file_edit",
            "code_file_read_write",
            "In src/parser.py, fix the bug that drops the last CSV column and update tests.",
            expected(inputs=[inp("src/parser.py", True, "file_path")], current=False, include_actions=["file_reading", "file_writing", "code_execution"], include_caps=["inspect_existing_code", "modify_code"]),
            attachments=[{"name": "src/parser.py", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_swebench_run_tests_001",
            "swe_bench",
            "code_execution_then_edit",
            "code_execution",
            "Run tests, fix failing imports in service/app.py, and report the change.",
            expected(inputs=[inp("service/app.py", True, "file_path")], current=False, include_actions=["file_reading", "file_writing", "code_execution"], include_caps=["execute_code", "modify_code"]),
            attachments=[{"name": "service/app.py", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_repobench_repo_search_001",
            "repobench",
            "repo_search",
            "file_reading",
            "Search the codebase for console.log calls and make a cleanup list.",
            expected(inputs=[inp("repository", True, "file_path")], current=False, include_actions=["file_reading"], include_caps=["search_provided_files"]),
            attachments=[{"name": "repository files", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_repobench_config_refs_001",
            "repobench",
            "repo_search",
            "file_reading",
            "Find usages of oldAuth in the repo and write a migration note.",
            expected(inputs=[inp("repository", True, "file_path")], current=False, include_actions=["file_reading"], include_caps=["search_provided_files"]),
            attachments=[{"name": "repository files", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_webarena_web_task_001",
            "webarena",
            "web_retrieval_and_summary",
            "url",
            "Read https://example.com/docs/onboarding and summarize the risky steps.",
            expected(inputs=[inp("https://example.com/docs/onboarding", True, "url")], current=True, include_actions=["web_search"], include_caps=["retrieve_external_information"]),
        ),
        mixed_case(
            "public_mix_taubench_policy_check_001",
            "tau_bench",
            "policy_fact_check",
            "fact_checking",
            "Verify whether the current refund policy allows same-day cancellations.",
            expected(inputs=[inp("refund policy", True, "none")], current=True, include_actions=["web_search", "fact_checking"], include_caps=["verify_current_information"]),
        ),
        mixed_case(
            "public_mix_agentbench_file_write_001",
            "agentbench",
            "file_write_from_source",
            "file_writing",
            "Create docs/status.md from this update: rollout green, support staffed, billing watch continues.",
            expected(inputs=[inp("update", True, "pasted_text"), inp("docs/status.md", True, "file_path")], current=False, include_actions=["file_writing"], include_caps=["write_file"]),
            attachments=[{"name": "docs/status.md", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_agentbench_merge_pdfs_001",
            "agentbench",
            "file_merge",
            "file_read_write",
            "Merge the attached PDFs into final_packet.pdf.",
            expected(inputs=[inp("pdf", True, "pdf")], current=False, include_actions=["file_reading", "file_writing"], include_caps=["combine_files"]),
            attachments=[
                {"name": "part_a.pdf", "format": "pdf", "available": True},
                {"name": "part_b.pdf", "format": "pdf", "available": True},
            ],
        ),
    ]


def document_and_media_cases() -> list[dict[str, Any]]:
    return [
        mixed_case(
            "public_mix_doc_pdf_summary_001",
            "document_public_mix",
            "attached_pdf_summary",
            "attached_pdf",
            "Summarize the attached audit report and list open risks.",
            expected(inputs=[inp("audit", True, "pdf")], current=False, include_actions=["file_reading"], include_caps=["extract_information_from_attached_document"], alt_caps=["summarize_document"]),
            attachments=[{"name": "audit_report.pdf", "format": "pdf", "available": True}],
        ),
        mixed_case(
            "public_mix_doc_pdf_calc_001",
            "document_public_mix",
            "attached_pdf_calculation",
            "pdf_calculation",
            "From the attached invoice PDF, calculate subtotal and tax.",
            expected(inputs=[inp("invoice", True, "pdf")], current=False, include_actions=["file_reading", "calculation"], include_caps=["compute_numeric_result"]),
            attachments=[{"name": "invoice.pdf", "format": "pdf", "available": True}],
        ),
        mixed_case(
            "public_mix_sheet_csv_rank_001",
            "spreadsheet_public_mix",
            "attached_structured_data_ranking",
            "attached_structured_data",
            "Analyze the attached CSV of monthly sales and tell me the top three regions by revenue.",
            expected(inputs=[inp("CSV", True, "structured_data")], current=False, include_actions=["file_reading", "calculation"], include_caps=["analyze_provided_dataset"]),
            attachments=[{"name": "monthly_sales.csv", "format": "structured_data", "available": True}],
        ),
        mixed_case(
            "public_mix_sheet_inline_table_001",
            "spreadsheet_public_mix",
            "inline_structured_data",
            "structured_data",
            "From this TSV, calculate total signups by channel:\nchannel\tsignups\nPaid\t80\nOrganic\t75\nReferral\t40",
            expected(inputs=[inp("signups", True, "structured_data")], current=False, include_actions=["calculation"], include_caps=["analyze_provided_dataset"]),
        ),
        mixed_case(
            "public_mix_image_alt_001",
            "image_public_mix",
            "image_understanding",
            "image_understanding",
            "Write alt text for the attached product dashboard screenshot.",
            expected(inputs=[inp("image", True, "image")], current=False, include_actions=["image_understanding"], include_caps=["interpret_image_content"]),
            attachments=[{"name": "dashboard_current.png", "format": "image", "available": True}],
        ),
        mixed_case(
            "public_mix_image_measure_001",
            "image_public_mix",
            "unsupported_missing_scale",
            "unsupported_missing_scale",
            "Measure the real-world height of the lamp in the attached photo.",
            expected(inputs=[inp("image", True, "image"), inp("scale", False, "unknown")], missing=["scale"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
            attachments=[{"name": "lamp_photo.png", "format": "image", "available": True}],
        ),
        mixed_case(
            "public_mix_image_generate_001",
            "image_public_mix",
            "image_generation",
            "image_generation",
            "Create a banner graphic for a webinar about current hiring trends.",
            expected(inputs=[inp("webinar", True, "none")], current=False, include_actions=["image_generation"], include_caps=["generate_image"]),
        ),
        mixed_case(
            "public_mix_image_edit_001",
            "image_public_mix",
            "image_editing",
            "image_editing",
            "Remove the background from the attached headshot.",
            expected(inputs=[inp("headshot", True, "image")], current=False, include_actions=["image_understanding", "image_generation"], include_caps=["interpret_image_content", "generate_image"]),
            attachments=[{"name": "headshot.png", "format": "image", "available": True}],
        ),
    ]


def extra_public_mix_cases() -> list[dict[str, Any]]:
    return [
        mixed_case(
            "public_mix_sni_rewrite_latest_002",
            "super_natural_instructions",
            "rewrite_with_current_decoy",
            "pasted_text_transform",
            "Rewrite this release note: The latest schedule says the current launch starts today.",
            expected(inputs=[inp("release note", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["revise_text_for_clarity"]),
        ),
        mixed_case(
            "public_mix_sni_extract_ceo_decoy_002",
            "super_natural_instructions",
            "extract_with_current_entity_decoy",
            "pasted_text_analysis",
            "Extract owners from these notes:\n\nCurrent CEO question goes to Mira. Billing export goes to Omar.",
            expected(inputs=[inp("notes", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["analyze_provided_text"]),
        ),
        mixed_case(
            "public_mix_sni_translate_url_quote_002",
            "super_natural_instructions",
            "quoted_url_source_transform",
            "pasted_text_transform",
            'Translate "Visit https://example.com/current/select today" into French.',
            expected(inputs=[inp("Visit", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["translate_text"], exclude_actions=["web_search"]),
        ),
        mixed_case(
            "public_mix_sni_compare_current_status_002",
            "super_natural_instructions",
            "compare_with_current_status_decoy",
            "pasted_text_analysis",
            "Compare these support responses and pick the clearer one.\n\nA: Current status is pending today.\n\nB: Your refund is pending while we review receipts.",
            expected(inputs=[inp("support responses", True, "pasted_text")], current=False, include_actions=["none"], include_caps=["compare_texts"]),
        ),
        mixed_case(
            "public_mix_clinc_current_rate_002",
            "clinc150_oos",
            "current_fact_intent",
            "current_facts",
            "What is the current federal funds rate?",
            expected(inputs=[inp("rate", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"]),
        ),
        mixed_case(
            "public_mix_massive_missing_policy_memo_002",
            "massive",
            "missing_source_request",
            "missing_input",
            "Summarize my policy memo.",
            expected(inputs=[inp("policy", False, "unknown")], missing=["policy"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
        ),
        mixed_case(
            "public_mix_clinc_missing_project_plan_002",
            "clinc150_oos",
            "missing_source_request",
            "missing_input",
            "Extract risks from my project plan.",
            expected(inputs=[inp("plan", False, "unknown")], missing=["plan"], current=False, include_actions=["user_input"], include_caps=["request_missing_input"]),
        ),
        mixed_case(
            "public_mix_massive_draft_note_002",
            "massive",
            "self_contained_writing_intent",
            "text_generation",
            "Write a polite note asking Jordan to send the signed agreement by noon.",
            expected(inputs=[inp("note", True, "none")], current=False, include_actions=["none"], include_caps=["draft_text"]),
        ),
        mixed_case(
            "public_mix_apibank_latest_model_002",
            "api_bank",
            "current_fact_intent",
            "current_facts",
            "What is the latest OpenAI model available in the API?",
            expected(inputs=[inp("OpenAI", True, "none")], current=True, include_actions=["web_search"], include_caps=["retrieve_current_information"]),
        ),
        mixed_case(
            "public_mix_bfcl_bash_script_002",
            "bfcl",
            "function_or_script_generation",
            "generic_coding_parameters",
            "Write a bash script that prints the five largest files under a directory path.",
            expected(inputs=[inp("script", True, "none")], current=False, include_actions=["none"], include_caps=["generate_code"]),
        ),
        mixed_case(
            "public_mix_repobench_todo_search_002",
            "repobench",
            "repo_search",
            "file_reading",
            "Find TODOs in the codebase related to payments.",
            expected(inputs=[inp("repository", True, "file_path")], current=False, include_actions=["file_reading"], include_caps=["search_provided_files"]),
            attachments=[{"name": "repository files", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_toolbench_url_differences_002",
            "toolbench",
            "external_multi_source_comparison",
            "url",
            "What are the differences between https://example.com/free and https://example.com/team?",
            expected(inputs=[inp("https://example.com/free", True, "url")], current=True, include_actions=["web_search"], include_caps=["compare_texts"], precede=[{"before": "retrieve", "after": "compare"}]),
        ),
        mixed_case(
            "public_mix_sheet_workbook_growth_002",
            "spreadsheet_public_mix",
            "attached_spreadsheet_calculation",
            "attached_spreadsheet",
            "Calculate month-over-month growth from the attached workbook.",
            expected(inputs=[inp("workbook", True, "attached_file")], current=False, include_actions=["file_reading", "calculation"], include_caps=["compute_numeric_result"]),
            attachments=[{"name": "growth_workbook.xlsx", "format": "attached_file", "available": True}],
        ),
        mixed_case(
            "public_mix_sheet_inline_json_002",
            "spreadsheet_public_mix",
            "inline_structured_data",
            "structured_data",
            "From this JSON, list admin users: [{\"name\":\"Nia\",\"role\":\"admin\"},{\"name\":\"Sol\",\"role\":\"viewer\"},{\"name\":\"Uma\",\"role\":\"admin\"}]",
            expected(inputs=[inp("JSON", True, "structured_data")], current=False, include_actions=["none"], include_caps=["analyze_provided_dataset"]),
        ),
        mixed_case(
            "public_mix_doc_local_file_002",
            "document_public_mix",
            "file_path_reading",
            "file_path_reading",
            "Read settings/prod.json and summarize enabled integrations.",
            expected(inputs=[inp("settings/prod.json", True, "file_path")], current=False, include_actions=["file_reading"], include_caps=["extract_information_from_file"]),
            attachments=[{"name": "settings/prod.json", "format": "file_path", "available": True}],
        ),
        mixed_case(
            "public_mix_image_receipt_002",
            "image_public_mix",
            "image_understanding",
            "image_understanding",
            "Read the attached receipt image and extract the total.",
            expected(inputs=[inp("receipt", True, "image")], current=False, include_actions=["image_understanding"], include_caps=["interpret_image_content"]),
            attachments=[{"name": "receipt.png", "format": "image", "available": True}],
        ),
    ]


def built_in_cases() -> list[dict[str, Any]]:
    rows = []
    rows.extend(source_transform_cases())
    rows.extend(intent_paraphrase_cases())
    rows.extend(tool_and_code_cases())
    rows.extend(document_and_media_cases())
    rows.extend(extra_public_mix_cases())
    return rows


def load_external_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "request" not in row or "expected" not in row:
                    raise ValueError(
                        f"{path} rows must already contain capability-planning request and expected fields"
                    )
                row.setdefault("source_dataset", path.stem)
                row.setdefault("wrapper_type", "external_jsonl")
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a mixed capability-planning eval: public-dataset-style "
            "examples plus adversarial source-span wrappers."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--external-jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional local JSONL exports already converted to this repo's gold schema. "
            "Can be passed multiple times."
        ),
    )
    args = parser.parse_args()

    rows = built_in_cases()
    rows.extend(load_external_rows(args.external_jsonl))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
