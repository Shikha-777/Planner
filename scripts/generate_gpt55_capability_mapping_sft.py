#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "capability_mapping_sft_v1"
SOURCE = "synthetic_gpt55_style_capability_mapping"

SYSTEM_PROMPT = """You are a capability planner.
Return only valid compact JSON.
Audit which user inputs are available or missing, classify the task route, and produce ordered capabilities.
Capabilities must describe transformations with inputs, outputs, dependency ids, done_when, and explicit external_action_type.
Do not invent attachments, URLs, files, or current facts. Use available tools only when the request semantically requires them."""

REQUEST_CONTEXT_FRAMES = [
    "",
    "For a short team update, plan this request:",
    "For a client handoff, plan this request:",
    "For an internal review, plan this request:",
    "For a support workflow, plan this request:",
    "For a product operations note, plan this request:",
    "For a classroom exercise, plan this request:",
    "For a research assistant task, plan this request:",
    "For a project manager, plan this request:",
    "For an analyst, plan this request:",
    "Treat this as a user request from a busy teammate:",
    "Treat this as a request from a nontechnical user:",
    "Treat this as a request from a developer:",
    "Treat this as a request from a customer-support lead:",
    "Treat this as a request from a data analyst:",
    "The user wants a practical answer. Plan this:",
    "The user wants only the necessary steps. Plan this:",
    "The user is asking from a shared workspace. Plan this:",
    "The user is giving partial context. Plan this:",
    "The user expects you to notice missing inputs. Plan this:",
    "Capability-plan this request:",
    "Route this request carefully:",
    "Audit inputs before planning this request:",
    "Decide whether external action is needed for this:",
    "Decide whether available tools should be called for this:",
    "Classify and plan this task:",
    "Map this user request to capabilities:",
    "Use planner judgment for this request:",
    "Be strict about not inventing sources for this:",
    "Be strict about tool relevance for this:",
    "Assume attachments metadata is authoritative for this:",
    "Assume available tools are the only callable tools for this:",
    "Focus on capability routing for this:",
    "Focus on input availability for this:",
    "Focus on dependency order for this:",
    "Focus on call count and order for this:",
    "Plan at capability level, not argument level:",
    "Return the planning JSON for this:",
    "Prepare a robust planner target for this:",
    "Create supervision for this planner example:",
]

REQUEST_OUTPUT_PREFERENCES = [
    "",
    "Additional preference: keep the final user-facing answer concise.",
    "Additional preference: use bullets if the final answer is structured.",
    "Additional preference: preserve the user's wording where possible.",
    "Additional preference: call out assumptions separately.",
    "Additional preference: do not use outside information unless the task requires it.",
    "Additional preference: prioritize exact source grounding.",
    "Additional preference: include only actionable steps.",
    "Additional preference: make the final response easy to scan.",
    "Additional preference: avoid unnecessary extra analysis.",
    "Additional preference: return a table if comparison is natural.",
    "Additional preference: explain missing inputs plainly.",
    "Additional preference: keep dependencies explicit.",
    "Additional preference: avoid speculative file or attachment access.",
    "Additional preference: do not select irrelevant tools.",
    "Additional preference: distinguish provided data from external data.",
    "Additional preference: mention unsupported parts only when they block the task.",
    "Additional preference: use the requested output format if one is implied.",
    "Additional preference: keep any generated text professional.",
    "Additional preference: focus on the user's final desired outcome.",
    "Additional preference: produce a minimal but complete plan.",
    "Additional preference: do not over-decompose simple requests.",
    "Additional preference: do not under-decompose dependent workflows.",
    "Additional preference: ask for missing source material before downstream work.",
    "Additional preference: retrieve before summarizing external sources.",
    "Additional preference: extract attachments before analyzing them.",
    "Additional preference: read code before modifying it.",
    "Additional preference: validate code changes after modifying them.",
    "Additional preference: plan one call per entity unless a batch action exists.",
    "Additional preference: preserve the user's requested action order.",
    "Additional preference: avoid treating variable names as missing values.",
    "Additional preference: avoid treating output preferences as source inputs.",
    "Additional preference: keep route labels benchmark-agnostic.",
    "Additional preference: cite evidence spans in the target.",
    "Additional preference: prefer user-input requests over invented retrieval.",
    "Additional preference: only mark current facts when freshness matters.",
    "Additional preference: only mark file reading when a real file or attachment exists.",
    "Additional preference: keep capability names transformation-oriented.",
    "Additional preference: keep tool names out of capability names.",
    "Additional preference: keep graph dependencies acyclic.",
]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sequence_from_record_id(record_id: str) -> int:
    digits = "".join(char for char in record_id if char.isdigit())
    return int(digits or "0")


def diversify_request(record_id: str, request: str) -> str:
    seq = sequence_from_record_id(record_id)
    frame = REQUEST_CONTEXT_FRAMES[seq % len(REQUEST_CONTEXT_FRAMES)]
    preference = REQUEST_OUTPUT_PREFERENCES[(seq // max(1, len(REQUEST_CONTEXT_FRAMES))) % len(REQUEST_OUTPUT_PREFERENCES)]
    parts = [part for part in (frame, request, preference) if part]
    return "\n".join(parts)


def make_messages(
    request: str,
    attachments_metadata: list[dict[str, Any]],
    available_tools: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "request": request,
        "attachments_metadata": attachments_metadata,
        "available_tools": available_tools,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Plan the capability mapping for this request. "
                "Return JSON with final_user_intent, input_audit, route, "
                "capability_plan, and tool_binding when available_tools are provided.\n"
                f"{compact_json(payload)}"
            ),
        },
        {"role": "assistant", "content": compact_json(target)},
    ]


def cap(
    index: int,
    name: str,
    description: str,
    inputs: list[str],
    outputs: list[str],
    done_when: str,
    *,
    external_action_type: str = "none",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"cap_{index}",
        "capability_name": name,
        "capability_description": description,
        "inputs": inputs,
        "outputs": outputs,
        "external_action_type": external_action_type,
        "depends_on": depends_on or [],
        "done_when": done_when,
    }


def target(
    *,
    final_user_intent: str,
    available_inputs: list[dict[str, Any]],
    missing_inputs: list[dict[str, Any]],
    task_family: str,
    operation: str,
    input_format: str,
    capabilities: list[dict[str, Any]],
    external_action_type: list[str],
    tool_decision: str,
    requires_external_current_info: bool = False,
    evidence: dict[str, str] | None = None,
    tool_binding: dict[str, Any] | None = None,
    unsupported_reason: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "final_user_intent": final_user_intent,
        "input_audit": {
            "available_inputs": available_inputs,
            "missing_inputs": missing_inputs,
            "input_formats": sorted({input_format} | {str(item.get("format", "")) for item in available_inputs if item.get("format")}),
        },
        "route": {
            "task_family": task_family,
            "operation": operation,
            "input_format": input_format,
            "requires_external_current_info": requires_external_current_info,
            "external_action_type": external_action_type,
            "tool_decision": tool_decision,
        },
        "capability_plan": {
            "ordered_capabilities": capabilities,
            "dependency_edges": [
                {"before": dep, "after": item["id"]}
                for item in capabilities
                for dep in item.get("depends_on", [])
            ],
        },
        "evidence": evidence or {},
    }
    if tool_binding is not None:
        data["tool_binding"] = tool_binding
    if unsupported_reason:
        data["unsupported_reason"] = unsupported_reason
    return data


def available_input(name: str, fmt: str, source: str, evidence_span: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": True,
        "format": fmt,
        "source": source,
        "evidence_span": evidence_span[:220],
    }


def missing_input(name: str, needed_for: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": False,
        "format": "unknown",
        "needed_for": needed_for,
        "reason": reason,
    }


def make_record(
    *,
    record_id: str,
    family: str,
    request: str,
    attachments_metadata: list[dict[str, Any]],
    available_tools: list[dict[str, Any]],
    gold_target: dict[str, Any],
) -> dict[str, Any]:
    request = diversify_request(record_id, request)
    record = {
        "schema_version": SCHEMA_VERSION,
        "id": record_id,
        "source": SOURCE,
        "family": family,
        "request": request,
        "attachments_metadata": attachments_metadata,
        "available_tools": available_tools,
        "target": gold_target,
    }
    record["messages"] = make_messages(request, attachments_metadata, available_tools, gold_target)
    validate_record(record)
    return record


def attachment(name: str, fmt: str, idx: int = 1) -> dict[str, Any]:
    mime = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "jpg": "image/jpeg",
        "mp3": "audio/mpeg",
        "mp4": "video/mp4",
        "txt": "text/plain",
        "json": "application/json",
        "zip": "application/zip",
    }.get(fmt, "application/octet-stream")
    return {"id": f"att_{idx}", "name": name, "format": fmt, "mime_type": mime, "available": True}


TEXT_SNIPPETS = [
    ("school lunch paragraph", "School lunches should be healthier because students need energy and focus. Some cafeterias sell too much fried food, and that makes learning harder."),
    ("product review", "The app is useful but the onboarding felt confusing. I liked the charts, but I could not find the export button without asking support."),
    ("meeting notes", "Maya will update the launch calendar by Friday. Andre needs to confirm vendor pricing. Priya will send the revised customer email."),
    ("support email", "I ordered the blue headphones two weeks ago and the tracking page has not changed. Please help me understand whether the order shipped."),
    ("grant paragraph", "Our program helps first-generation students navigate college applications through workshops, mentoring, and family information sessions."),
    ("incident note", "At 09:10 the checkout service returned elevated 500s. A rollback at 09:24 restored traffic. Follow-up is needed on alert thresholds."),
]

DATASETS = [
    {
        "kind": "CSV",
        "name": "plan revenue",
        "block": "plan,revenue\nBasic,120\nPro,300\nBasic,80\nEnterprise,520\nPro,140",
        "metric": "revenue",
        "group": "plan",
    },
    {
        "kind": "TSV",
        "name": "regional signups",
        "block": "region\tsignups\nNorth\t42\nSouth\t28\nNorth\t35\nWest\t19\nSouth\t44",
        "metric": "signups",
        "group": "region",
    },
    {
        "kind": "pipe-delimited table",
        "name": "ticket priority",
        "block": "priority|tickets\nlow|12\nhigh|7\nmedium|15\nhigh|11",
        "metric": "tickets",
        "group": "priority",
    },
    {
        "kind": "semicolon-delimited table",
        "name": "campaign cost",
        "block": "campaign;cost\nA;1200\nB;980\nA;700\nC;400",
        "metric": "cost",
        "group": "campaign",
    },
    {
        "kind": "JSON",
        "name": "warehouse inventory",
        "block": '[{"sku":"A1","stock":8},{"sku":"B2","stock":2},{"sku":"C3","stock":17}]',
        "metric": "stock",
        "group": "sku",
    },
]

CURRENT_FACT_TOPICS = [
    ("latest OpenAI model for coding", "identify the latest OpenAI coding model"),
    ("current price of Bitcoin", "answer with the current Bitcoin price"),
    ("today's weather in Chicago", "answer with today's Chicago weather"),
    ("latest CPI inflation reading for the United States", "answer with the latest CPI reading"),
    ("current CEO of a public company", "answer with the current CEO"),
    ("next Federal Reserve meeting date", "answer with the next meeting date"),
    ("latest Python stable version", "answer with the latest Python stable version"),
    ("current mortgage rate average", "answer with the current mortgage rate average"),
]

URLS = [
    "https://example.com/product-roadmap",
    "https://example.org/research-summary",
    "https://docs.example.net/api/limits",
    "https://news.example.com/company-update",
    "https://status.example.io/incidents/123",
]

CODE_PATHS = [
    ("src/parser.py", "fix the bug where blank lines create empty tokens", "parser behavior is corrected"),
    ("app/routes/users.ts", "add validation for missing email addresses", "user route validates email input"),
    ("lib/reporting/exporter.py", "make CSV export preserve column order", "CSV export preserves requested order"),
    ("components/SearchBox.jsx", "debounce searches so typing does not trigger too many requests", "search input is debounced"),
    ("tests/test_billing.py", "add coverage for prorated annual plan upgrades", "billing tests cover prorated upgrades"),
]

NAMES = ["Maya", "Andre", "Priya", "Jordan", "Sam", "Nina", "Avery", "Luis", "Chen", "Riley"]
CITIES = ["Boston", "Austin", "Denver", "Tokyo", "Paris", "Chicago", "Seattle", "Miami", "Phoenix", "Toronto"]
DATES = ["tomorrow", "Friday", "July 12", "next Monday", "2026-08-14", "the last weekday of this month"]
PRODUCTS = ["blue headphones", "standing desk", "trail shoes", "espresso machine", "phone case", "laptop sleeve"]
COMPANIES = ["Acme Robotics", "Northstar Health", "Bluebird Bank", "MetroGrid", "Cedar Analytics", "Kite Labs"]
REPO_PATTERNS = [
    ("FIXME comments", "find_cleanup_targets"),
    ("deprecated API usages", "find_deprecated_usage"),
    ("TODOs related to billing", "find_cleanup_targets"),
    ("calls to requests.get without timeout", "find_risky_code_pattern"),
    ("React components missing aria-labels", "find_accessibility_issues"),
    ("SQL queries built with string concatenation", "find_security_risks"),
]

TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for one city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    "convert_currency": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
            "required": ["amount", "from", "to"],
        },
    },
    "solve_quadratic": {
        "name": "solve_quadratic",
        "description": "Find roots of a quadratic equation from coefficients a, b, and c.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "number"}},
            "required": ["a", "b", "c"],
        },
    },
    "calculate_circumference": {
        "name": "calculate_circumference",
        "description": "Calculate a circle circumference from a radius.",
        "parameters": {"type": "object", "properties": {"radius": {"type": "number"}}, "required": ["radius"]},
    },
    "translate_text": {
        "name": "translate_text",
        "description": "Translate text into a requested language.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"],
        },
    },
    "create_calendar_event": {
        "name": "create_calendar_event",
        "description": "Create a calendar event with title, date, time, and attendees.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}},
            "required": ["title", "date", "time"],
        },
    },
    "find_movie_showtimes": {
        "name": "find_movie_showtimes",
        "description": "Find movie showtimes at one theater for a movie and date.",
        "parameters": {
            "type": "object",
            "properties": {"movie": {"type": "string"}, "theater": {"type": "string"}, "date": {"type": "string"}},
            "required": ["movie", "theater", "date"],
        },
    },
    "search_flights": {
        "name": "search_flights",
        "description": "Search flights between two airports for a date.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}},
            "required": ["origin", "destination", "date"],
        },
    },
    "book_restaurant": {
        "name": "book_restaurant",
        "description": "Book a restaurant table for a party size, date, and time.",
        "parameters": {
            "type": "object",
            "properties": {"restaurant": {"type": "string"}, "party_size": {"type": "integer"}, "date": {"type": "string"}, "time": {"type": "string"}},
            "required": ["restaurant", "party_size", "date", "time"],
        },
    },
    "estimate_home_price": {
        "name": "estimate_home_price",
        "description": "Estimate the market price of one property from address and attributes.",
        "parameters": {"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]},
    },
}

TOOLS.update(
    {
        "get_weather_batch": {
            "name": "get_weather_batch",
            "description": "Get current weather for multiple cities in one call.",
            "parameters": {
                "type": "object",
                "properties": {"cities": {"type": "array", "items": {"type": "string"}}},
                "required": ["cities"],
            },
        },
        "lookup_order": {
            "name": "lookup_order",
            "description": "Look up one customer order by order id.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
        "update_shipping_address": {
            "name": "update_shipping_address",
            "description": "Update the shipping address for an existing order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}},
                "required": ["order_id", "address"],
            },
        },
        "issue_refund": {
            "name": "issue_refund",
            "description": "Issue a refund for an eligible order.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
        "cancel_order": {
            "name": "cancel_order",
            "description": "Cancel an order that has not shipped yet.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
        "check_return_policy": {
            "name": "check_return_policy",
            "description": "Retrieve the return policy for a product category.",
            "parameters": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
        },
        "search_products": {
            "name": "search_products",
            "description": "Search a product catalog by query and optional filters.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
        "add_to_cart": {
            "name": "add_to_cart",
            "description": "Add one product to a user's cart.",
            "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
        },
        "checkout_cart": {
            "name": "checkout_cart",
            "description": "Check out the current shopping cart.",
            "parameters": {"type": "object", "properties": {"payment_method": {"type": "string"}}, "required": ["payment_method"]},
        },
        "get_account_balance": {
            "name": "get_account_balance",
            "description": "Retrieve the balance for one bank account.",
            "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]},
        },
        "transfer_funds": {
            "name": "transfer_funds",
            "description": "Transfer money between two accounts.",
            "parameters": {
                "type": "object",
                "properties": {"from_account": {"type": "string"}, "to_account": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["from_account", "to_account", "amount"],
            },
        },
        "update_user_email": {
            "name": "update_user_email",
            "description": "Update a user's email address.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}},
                "required": ["user_id", "email"],
            },
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an email to one recipient.",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "subject", "body"],
            },
        },
        "create_support_ticket": {
            "name": "create_support_ticket",
            "description": "Create a customer support ticket.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}, "issue": {"type": "string"}},
                "required": ["customer_id", "issue"],
            },
        },
        "query_database": {
            "name": "query_database",
            "description": "Run a read-only SQL query against an analytics database.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
        },
        "send_slack_message": {
            "name": "send_slack_message",
            "description": "Send a Slack message to one channel.",
            "parameters": {
                "type": "object",
                "properties": {"channel": {"type": "string"}, "message": {"type": "string"}},
                "required": ["channel", "message"],
            },
        },
        "calculate_card_probability": {
            "name": "calculate_card_probability",
            "description": "Calculate probability for drawing cards from a standard deck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "deck_size": {"type": "integer"},
                    "draw_count": {"type": "integer"},
                },
                "required": ["event", "deck_size", "draw_count"],
            },
        },
        "plan_travel_itinerary": {
            "name": "plan_travel_itinerary",
            "description": "Plan a travel itinerary from destination, dates, budget, and interests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "budget": {"type": "number"},
                },
                "required": ["destination", "start_date", "end_date", "budget"],
            },
        },
        "lookup_court_case": {
            "name": "lookup_court_case",
            "description": "Look up a court case by jurisdiction and case number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "jurisdiction": {"type": "string"},
                    "case_number": {"type": "string"},
                },
                "required": ["jurisdiction", "case_number"],
            },
        },
        "search_courthouse_locations": {
            "name": "search_courthouse_locations",
            "description": "Search courthouse building locations by city or county.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        "analyze_fmri_file": {
            "name": "analyze_fmri_file",
            "description": "Analyze an fMRI image file from a provided local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "analysis": {"type": "string"},
                    "threshold": {"type": "number"},
                },
                "required": ["file_path", "analysis"],
            },
        },
        "find_top_k": {
            "name": "find_top_k",
            "description": "Find the top k items in a named dataset or list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string"},
                    "k": {"type": "integer"},
                },
                "required": ["dataset", "k"],
            },
        },
        "calculate_moving_average": {
            "name": "calculate_moving_average",
            "description": "Calculate a moving average over a numeric list using window k.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nums": {"type": "array", "items": {"type": "number"}},
                    "k": {"type": "integer"},
                },
                "required": ["nums", "k"],
            },
        },
        "estimate_trip_cost": {
            "name": "estimate_trip_cost",
            "description": "Estimate travel cost for origin and destination coordinates with a budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_coordinates": {"type": "string"},
                    "destination_coordinates": {"type": "string"},
                    "budget": {"type": "number"},
                },
                "required": ["origin_coordinates", "destination_coordinates", "budget"],
            },
        },
        "get_historical_weather_by_coordinates": {
            "name": "get_historical_weather_by_coordinates",
            "description": "Get historical weather for latitude, longitude, and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "date": {"type": "string"},
                },
                "required": ["latitude", "longitude", "date"],
            },
        },
        "get_weather_forecast_by_coordinates": {
            "name": "get_weather_forecast_by_coordinates",
            "description": "Get a weather forecast for latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
        "plan_route": {
            "name": "plan_route",
            "description": "Plan a route between an origin and destination address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
        },
        "lookup_chess_club": {
            "name": "lookup_chess_club",
            "description": "Look up local chess clubs and chess organizations.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        "find_chess_tournament_info": {
            "name": "find_chess_tournament_info",
            "description": "Find information about chess tournaments and championships.",
            "parameters": {
                "type": "object",
                "properties": {"tournament": {"type": "string"}},
                "required": ["tournament"],
            },
        },
        "call_rest_api": {
            "name": "call_rest_api",
            "description": "Send an HTTP request to a REST API endpoint.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "method": {"type": "string"}},
                "required": ["url", "method"],
            },
        },
    }
)


def tool_list(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[name] for name in names]


def tool_target(selected: list[str], reason: str) -> dict[str, Any]:
    return {
        "tool_decision": "call" if selected else "no_call",
        "ordered_tool_names": selected,
        "call_count": len(selected),
        "selection_reason": reason,
    }


def gen_missing_input(rng: random.Random, seq: int) -> dict[str, Any]:
    objects = [
        ("essay text", "make my essay better", "revise the essay"),
        ("resume text", "polish my resume", "revise the resume"),
        ("LinkedIn bio", "make my LinkedIn bio sound more professional", "revise the profile bio"),
        ("proposal draft", "summarize my proposal", "summarize the proposal"),
        ("revenue dataset", "plot revenue by region", "analyze the dataset"),
        ("screenshot", "tell me what is wrong in my screenshot", "inspect the screenshot"),
        ("inventory data", "calculate which SKUs are low stock", "analyze inventory"),
        ("article text", "turn my article into a thread", "rewrite the article"),
    ]
    name, phrase, intent = rng.choice(objects)
    request = rng.choice([
        f"Can you {phrase}?",
        f"Please {phrase}.",
        f"I need you to {phrase} for me.",
        f"Could you help me {phrase}?",
    ])
    caps = [
        cap(
            1,
            "request_missing_input",
            f"Ask the user to provide the {name} needed for the requested transformation.",
            [],
            [name],
            f"the user has been asked for the missing {name}",
            external_action_type="user_input",
        )
    ]
    gold = target(
        final_user_intent=intent,
        available_inputs=[],
        missing_inputs=[missing_input(name, intent, f"The request asks to use {name}, but no source text, data, attachment, or URL was provided.")],
        task_family="missing_source_input",
        operation="request_required_source",
        input_format="none",
        capabilities=caps,
        external_action_type=["user_input"],
        tool_decision="ask_user",
        evidence={"operation_span": phrase, "missing_reason": "no source content or attachment metadata"},
    )
    return make_record(record_id=f"capmap_missing_input_{seq:06d}", family="missing_input", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_pasted_text_transform(rng: random.Random, seq: int) -> dict[str, Any]:
    label, text = rng.choice(TEXT_SNIPPETS)
    op = rng.choice(["rewrite", "summarize", "extract_action_items", "extract_entities", "change_tone"])
    prompts = {
        "rewrite": f"Please make this {label} clearer and more polished:\n\n{text}",
        "summarize": f"Summarize this {label} in two bullets:\n\n{text}",
        "extract_action_items": f"Pull out the action items from these notes:\n\n{text}",
        "extract_entities": f"Extract the people, dates, and important objects from this text:\n\n{text}",
        "change_tone": f"Make this text warmer but still concise:\n\n{text}",
    }
    if op == "rewrite":
        caps = [
            cap(1, "revise_text_for_clarity", "Rewrite provided text for clarity while preserving meaning.", ["pasted text"], ["revised text"], "the pasted text is clearer and polished")
        ]
        intent = f"revise the provided {label}"
    elif op == "summarize":
        caps = [
            cap(1, "analyze_provided_text", "Read the pasted text and identify salient points.", ["pasted text"], ["key points"], "the main ideas are identified"),
            cap(2, "summarize_text", "Condense key points into the requested summary format.", ["key points"], ["summary"], "the requested summary is produced", depends_on=["cap_1"]),
        ]
        intent = f"summarize the provided {label}"
    elif op == "extract_action_items":
        caps = [
            cap(1, "analyze_provided_text", "Read the pasted notes for obligations and owners.", ["pasted notes"], ["candidate actions"], "potential action items are identified"),
            cap(2, "extract_action_items", "Convert obligations into clear action items.", ["candidate actions"], ["action item list"], "action items include owners or due dates when present", depends_on=["cap_1"]),
        ]
        intent = "extract action items from pasted notes"
    elif op == "extract_entities":
        caps = [
            cap(1, "analyze_provided_text", "Read the pasted text and detect named entities.", ["pasted text"], ["entity candidates"], "people, dates, and objects are detected"),
            cap(2, "structure_extracted_information", "Organize extracted entities into a structured answer.", ["entity candidates"], ["structured entity list"], "entities are grouped by type", depends_on=["cap_1"]),
        ]
        intent = "extract structured entities from pasted text"
    else:
        caps = [
            cap(1, "revise_text_tone", "Adjust tone of provided text while preserving facts.", ["pasted text"], ["tone-adjusted text"], "the text has the requested warmer concise tone")
        ]
        intent = "adjust the tone of pasted text"
    request = prompts[op]
    gold = target(
        final_user_intent=intent,
        available_inputs=[available_input(label, "pasted_text", "request", text)],
        missing_inputs=[],
        task_family="pasted_text_transform",
        operation=op,
        input_format="pasted_text",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"input_span": text[:120], "operation_span": request.split("\n", 1)[0]},
    )
    return make_record(record_id=f"capmap_pasted_text_{seq:06d}", family="pasted_text_transform", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_structured_data_analysis(rng: random.Random, seq: int) -> dict[str, Any]:
    data = rng.choice(DATASETS)
    op = rng.choice(["compute_total_by_group", "compute_average", "identify_outliers", "rank_groups"])
    if op == "compute_total_by_group":
        instruction = f"From this {data['kind']}, compute total {data['metric']} by {data['group']}:"
        intent = f"compute total {data['metric']} by {data['group']}"
    elif op == "compute_average":
        instruction = f"Using this {data['kind']}, compute the average {data['metric']} and explain the result:"
        intent = f"compute average {data['metric']}"
    elif op == "identify_outliers":
        instruction = f"Find any obvious outliers in this {data['kind']} data:"
        intent = f"identify outliers in {data['name']} data"
    else:
        instruction = f"Rank the {data['group']} values by {data['metric']} using this {data['kind']}:"
        intent = f"rank {data['group']} by {data['metric']}"
    request = f"{instruction}\n{data['block']}"
    caps = [
        cap(1, "parse_structured_input", "Parse the inline structured data into rows and fields.", ["inline structured data"], ["parsed dataset"], "the provided rows and columns are available"),
        cap(2, "compute_numeric_result", "Apply the requested aggregation or numeric calculation.", ["parsed dataset"], ["computed result"], "the requested calculation is complete", depends_on=["cap_1"]),
        cap(3, "report_findings", "Present the result in a concise user-facing form.", ["computed result"], ["answer"], "the answer reports the calculation and key interpretation", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=intent,
        available_inputs=[available_input(data["name"], "structured_data", "request", data["block"])],
        missing_inputs=[],
        task_family="structured_data_analysis",
        operation=op,
        input_format="structured_data",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"input_span": data["block"][:120], "operation_span": instruction},
    )
    return make_record(record_id=f"capmap_structured_data_{seq:06d}", family="structured_data_analysis", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_attached_document(rng: random.Random, seq: int) -> dict[str, Any]:
    docs = [
        ("policy_memo.pdf", "pdf", "summarize the attached policy memo", "summarize_document"),
        ("board_notes.docx", "docx", "extract decisions from the attached board notes", "extract_decisions"),
        ("sales_pipeline.xlsx", "xlsx", "calculate total pipeline by stage from the attached spreadsheet", "compute_dataset_metric"),
        ("survey_results.csv", "csv", "summarize themes in the attached survey results", "analyze_dataset"),
    ]
    filename, fmt, phrase, op = rng.choice(docs)
    request = rng.choice([f"Please {phrase}.", f"Can you {phrase}?", f"{phrase.capitalize()} and keep it concise."])
    att = attachment(filename, fmt)
    if fmt in {"csv", "xlsx"}:
        caps = [
            cap(1, "load_attached_dataset", "Read the attached tabular file into analyzable rows.", [filename], ["loaded dataset"], "the attached dataset is loaded", external_action_type="file_reading"),
            cap(2, "analyze_dataset", "Perform the requested dataset analysis.", ["loaded dataset"], ["analysis result"], "the requested dataset analysis is complete", depends_on=["cap_1"]),
            cap(3, "report_findings", "Report the dataset result in user-facing form.", ["analysis result"], ["answer"], "the answer includes the requested findings", depends_on=["cap_2"]),
        ]
        input_format = "spreadsheet" if fmt == "xlsx" else "csv"
    else:
        caps = [
            cap(1, "extract_information_from_attached_document", "Read and extract relevant text from the attached document.", [filename], ["document text"], "the attachment content is available", external_action_type="file_reading"),
            cap(2, "analyze_document_content", "Analyze the extracted document content for the requested information.", ["document text"], ["document analysis"], "the requested document information is identified", depends_on=["cap_1"]),
            cap(3, "produce_document_answer", "Produce the requested summary or extraction.", ["document analysis"], ["answer"], "the user receives the requested document-based answer", depends_on=["cap_2"]),
        ]
        input_format = fmt
    gold = target(
        final_user_intent=phrase,
        available_inputs=[available_input(filename, input_format, "attachment", filename)],
        missing_inputs=[],
        task_family="attached_document",
        operation=op,
        input_format=input_format,
        capabilities=caps,
        external_action_type=["file_reading"],
        tool_decision="no_tool",
        evidence={"input_span": filename, "operation_span": phrase},
    )
    return make_record(record_id=f"capmap_attached_document_{seq:06d}", family="attached_document", request=request, attachments_metadata=[att], available_tools=[], gold_target=gold)


def gen_current_fact(rng: random.Random, seq: int) -> dict[str, Any]:
    topic, intent = rng.choice(CURRENT_FACT_TOPICS)
    request = rng.choice([
        f"What is the {topic}?",
        f"Can you check the {topic}?",
        f"As of today, what is the {topic}?",
        f"I need the current answer for: {topic}.",
    ])
    caps = [
        cap(1, "retrieve_current_information", "Fetch up-to-date information needed to answer the current-fact query.", ["current-fact query"], ["retrieved current information"], "current relevant sources are checked", external_action_type="web_search"),
        cap(2, "answer_with_current_information", "Use the retrieved current information to answer with appropriate date context.", ["retrieved current information"], ["current answer"], "the answer reflects current information and cites timing", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent=intent,
        available_inputs=[available_input("current-fact query", "none", "request", topic)],
        missing_inputs=[],
        task_family="current_fact",
        operation="retrieve_current_answer",
        input_format="none",
        capabilities=caps,
        external_action_type=["web_search"],
        tool_decision="no_tool",
        requires_external_current_info=True,
        evidence={"operation_span": topic, "current_info_trigger": "latest/current/today/as of today"},
    )
    return make_record(record_id=f"capmap_current_fact_{seq:06d}", family="current_fact", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_url_task(rng: random.Random, seq: int) -> dict[str, Any]:
    urls = rng.sample(URLS, rng.choice([1, 2, 3]))
    if len(urls) == 1:
        request = rng.choice([
            f"Summarize this page: {urls[0]}",
            f"Read {urls[0]} and pull out the main limitations.",
            f"Use this URL to answer what changed: {urls[0]}",
        ])
        caps = [
            cap(1, "retrieve_external_information", "Retrieve content from the provided URL.", ["URL"], ["retrieved page content"], "the URL content is available", external_action_type="web_search"),
            cap(2, "summarize_retrieved_content", "Summarize or answer from retrieved content.", ["retrieved page content"], ["answer"], "the answer is grounded in the retrieved page", depends_on=["cap_1"]),
        ]
        op = "summarize_url"
        intent = "summarize or answer from a provided URL"
        fmt = "url"
        input_name = "URL"
    else:
        request = "Compare these links and tell me where they disagree:\n" + "\n".join(urls)
        caps = [
            cap(1, "retrieve_external_information", "Retrieve content from each provided URL.", ["URL list"], ["retrieved page contents"], "content from all provided URLs is available", external_action_type="web_search"),
            cap(2, "compare_retrieved_content", "Compare retrieved pages for agreement and disagreement.", ["retrieved page contents"], ["comparison"], "differences and overlaps are identified", depends_on=["cap_1"]),
            cap(3, "report_comparison", "Present the comparison in a concise answer.", ["comparison"], ["answer"], "the answer reports agreement and disagreement", depends_on=["cap_2"]),
        ]
        op = "compare_urls"
        intent = "compare information across provided URLs"
        fmt = "url_list"
        input_name = "URL list"
    gold = target(
        final_user_intent=intent,
        available_inputs=[available_input(input_name, fmt, "request", " ".join(urls))],
        missing_inputs=[],
        task_family="url_retrieval",
        operation=op,
        input_format=fmt,
        capabilities=caps,
        external_action_type=["web_search"],
        tool_decision="no_tool",
        evidence={"input_span": " ".join(urls), "operation_span": request.split("\n", 1)[0]},
    )
    return make_record(record_id=f"capmap_url_task_{seq:06d}", family="url_task", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_code_edit(rng: random.Random, seq: int) -> dict[str, Any]:
    path, change, done = rng.choice(CODE_PATHS)
    request = rng.choice([
        f"In {path}, {change}.",
        f"Please update {path} to {change}.",
        f"Can you inspect {path} and {change}?",
    ])
    caps = [
        cap(1, "inspect_existing_code", "Read the referenced code and understand the current behavior.", [path], ["code understanding"], "the relevant code is inspected", external_action_type="file_reading"),
        cap(2, "modify_code", "Apply the requested code change in the appropriate file.", ["code understanding"], ["code changes"], "the code implements the requested behavior", external_action_type="file_writing", depends_on=["cap_1"]),
        cap(3, "validate_output_against_requirements", "Run or reason through focused validation for the change.", ["code changes"], ["validation result"], done, depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=change,
        available_inputs=[available_input(path, "file_path", "request", path)],
        missing_inputs=[],
        task_family="code_edit",
        operation="modify_existing_code",
        input_format="file_path",
        capabilities=caps,
        external_action_type=["file_reading", "file_writing"],
        tool_decision="no_tool",
        evidence={"input_span": path, "operation_span": change},
    )
    return make_record(record_id=f"capmap_code_edit_{seq:06d}", family="code_edit", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_code_generation(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Write a Python function that takes nums and k and returns the k largest values.", "generate a Python function from requirements"),
        ("Write a Postgres SQL query to find monthly revenue by plan from an orders table.", "generate a SQL query from requirements"),
        ("Create a TypeScript helper that validates an email string and returns a typed result.", "generate a TypeScript helper from requirements"),
        ("Implement a function f(n, m) that returns the number of grid paths modulo 1_000_000_007.", "generate an algorithmic function from requirements"),
        ("Write a regex that captures invoice IDs like INV-2026-0042.", "generate a regex from requirements"),
    ]
    request, intent = rng.choice(cases)
    caps = [
        cap(1, "interpret_code_requirements", "Identify the desired language, inputs, outputs, and constraints from the request.", ["code requirements"], ["implementation spec"], "the implementation requirements are clear"),
        cap(2, "generate_code_from_requirements", "Produce code or query matching the implementation spec.", ["implementation spec"], ["generated code"], "the generated code satisfies the requested behavior", depends_on=["cap_1"]),
        cap(3, "validate_generated_code", "Check the generated code against the stated requirements.", ["generated code", "implementation spec"], ["validation result"], "the code is checked for consistency with requirements", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=intent,
        available_inputs=[available_input("code requirements", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="code_generation",
        operation="generate_code",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"operation_span": request, "input_span": request},
    )
    return make_record(record_id=f"capmap_code_generation_{seq:06d}", family="code_generation", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_image_understanding(rng: random.Random, seq: int) -> dict[str, Any]:
    img = rng.choice([attachment("receipt_photo.png", "png"), attachment("dashboard_screenshot.png", "png"), attachment("whiteboard.jpg", "jpg")])
    prompt = rng.choice([
        "What text is visible in the attached image?",
        "Describe the attached screenshot and identify anything unusual.",
        "Extract the main items shown in this image.",
        "Summarize the whiteboard in the attached photo.",
    ])
    caps = [
        cap(1, "interpret_image_content", "Analyze the attached image for visible text, objects, or layout.", [img["name"]], ["image observations"], "the relevant visual content is identified", external_action_type="image_analysis"),
        cap(2, "answer_visual_question", "Answer the user's question using image observations.", ["image observations"], ["visual answer"], "the answer is grounded in the attached image", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent=prompt.rstrip("."),
        available_inputs=[available_input(img["name"], "image", "attachment", img["name"])],
        missing_inputs=[],
        task_family="image_understanding",
        operation="answer_from_image",
        input_format="image",
        capabilities=caps,
        external_action_type=["image_analysis"],
        tool_decision="no_tool",
        evidence={"input_span": img["name"], "operation_span": prompt},
    )
    return make_record(record_id=f"capmap_image_understanding_{seq:06d}", family="image_understanding", request=prompt, attachments_metadata=[img], available_tools=[], gold_target=gold)


def gen_unsupported_measurement(rng: random.Random, seq: int) -> dict[str, Any]:
    img = attachment(rng.choice(["desk_photo.png", "plant_image.jpg", "package_picture.png"]), rng.choice(["png", "jpg"]))
    request = rng.choice([
        "Measure the exact width of the object in the attached image.",
        "How tall is the plant in this photo in centimeters?",
        "Tell me the real-world size of the package in the image.",
    ])
    caps = [
        cap(
            1,
            "request_missing_input",
            "Ask for a scale reference or known dimension before estimating real-world size from the image.",
            [img["name"]],
            ["scale reference or known dimension"],
            "the user is asked for the missing scale information",
            external_action_type="user_input",
        )
    ]
    gold = target(
        final_user_intent="measure real-world object size from image",
        available_inputs=[available_input(img["name"], "image", "attachment", img["name"])],
        missing_inputs=[missing_input("scale reference or known dimension", "measure real-world size from image", "The image metadata does not provide a physical scale, so exact real-world dimensions cannot be inferred.")],
        task_family="unsupported_or_missing_scale",
        operation="request_scale_before_measurement",
        input_format="image",
        capabilities=caps,
        external_action_type=["user_input"],
        tool_decision="ask_user",
        evidence={"input_span": img["name"], "operation_span": request},
        unsupported_reason="Real-world size cannot be measured from an image without a scale reference or camera geometry.",
    )
    return make_record(record_id=f"capmap_measurement_{seq:06d}", family="unsupported_measurement", request=request, attachments_metadata=[img], available_tools=[], gold_target=gold)


def gen_multistep_dependency(rng: random.Random, seq: int) -> dict[str, Any]:
    label, text = rng.choice(TEXT_SNIPPETS)
    request = rng.choice([
        f"From these notes, extract action items and draft a follow-up email:\n\n{text}",
        f"Read this incident note, summarize the cause, then draft a customer update:\n\n{text}",
        f"Analyze this customer feedback, list the complaints, and write a short reply:\n\n{text}",
    ])
    caps = [
        cap(1, "analyze_provided_text", "Read the provided text and identify relevant facts.", ["pasted text"], ["text analysis"], "the relevant facts are identified"),
        cap(2, "extract_intermediate_findings", "Extract the requested intermediate findings from the analysis.", ["text analysis"], ["findings"], "the requested findings are extracted", depends_on=["cap_1"]),
        cap(3, "draft_user_requested_output", "Draft the final requested message or update using the findings.", ["findings"], ["draft output"], "the final requested draft is produced", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent="extract findings and draft a dependent written output",
        available_inputs=[available_input(label, "pasted_text", "request", text)],
        missing_inputs=[],
        task_family="multi_step_text_workflow",
        operation="extract_then_draft",
        input_format="pasted_text",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"input_span": text[:120], "operation_span": request.split("\n", 1)[0]},
    )
    return make_record(record_id=f"capmap_multistep_{seq:06d}", family="multi_step_dependency", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_file_write(rng: random.Random, seq: int) -> dict[str, Any]:
    data = rng.choice(DATASETS)
    output = rng.choice(["CSV", "Markdown table", "JSON file"])
    request = f"Create a {output} from this data and save it as cleaned_output.{output.split()[0].lower()}:\n{data['block']}"
    caps = [
        cap(1, "parse_structured_input", "Parse the provided inline data into rows and fields.", ["inline structured data"], ["parsed dataset"], "the source data is parsed"),
        cap(2, "generate_requested_file_content", "Transform parsed data into the requested output file format.", ["parsed dataset"], ["file content"], "the requested file content is generated", depends_on=["cap_1"]),
        cap(3, "write_requested_file", "Write the generated content to the requested file path.", ["file content"], ["written file"], "the requested file is created", external_action_type="file_writing", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=f"create a {output} file from provided data",
        available_inputs=[available_input(data["name"], "structured_data", "request", data["block"])],
        missing_inputs=[],
        task_family="file_creation",
        operation="write_file_from_provided_data",
        input_format="structured_data",
        capabilities=caps,
        external_action_type=["file_writing"],
        tool_decision="no_tool",
        evidence={"input_span": data["block"][:120], "operation_span": request.split("\n", 1)[0]},
    )
    return make_record(record_id=f"capmap_file_write_{seq:06d}", family="file_write", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_tool_binding_single(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Find the roots of the quadratic with a=3, b=-11, c=-4.", "solve_quadratic", ["calculate_circumference", "translate_text"]),
        ("What is the circumference of a circle with radius 4 inches?", "calculate_circumference", ["solve_quadratic", "get_weather"]),
        ("Translate 'good morning' into Spanish.", "translate_text", ["get_weather", "convert_currency"]),
        ("What is the weather in Tokyo right now?", "get_weather", ["convert_currency", "book_restaurant"]),
        ("Convert 125 USD to EUR.", "convert_currency", ["get_weather", "solve_quadratic"]),
        ("Create a calendar event called design review tomorrow at 2 PM.", "create_calendar_event", ["search_flights", "translate_text"]),
        ("Find showtimes for Dune at the Tivoli tomorrow.", "find_movie_showtimes", ["book_restaurant", "get_weather"]),
    ]
    request, selected, decoys = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Compare the user intent against the available action descriptions.", ["user request", "available tools"], ["matched action"], "the relevant available action is selected", external_action_type="external_tool_call"),
        cap(2, "plan_external_action_sequence", "Plan the required external action call sequence without filling arguments.", ["matched action"], ["ordered external actions"], "the needed action count and order are specified", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent=f"use the available action for {selected}",
        available_inputs=[available_input("tool-call request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding",
        operation="select_single_external_action",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": selected},
        tool_binding=tool_target([selected], f"The request semantically matches {selected}."),
    )
    return make_record(record_id=f"capmap_tool_single_{seq:06d}", family="tool_binding_single", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_binding_multiple(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Check weather in Paris, then convert 50 EUR to USD.", ["get_weather", "convert_currency"], ["translate_text"]),
        ("Search flights from STL to LAX on July 12 and create a calendar event for the trip.", ["search_flights", "create_calendar_event"], ["get_weather"]),
        ("Translate 'see you soon' into French, then create a calendar reminder called language practice tomorrow at 8 AM.", ["translate_text", "create_calendar_event"], ["solve_quadratic"]),
        ("Estimate the home price for 10 Oak St, then convert 400000 USD to EUR.", ["estimate_home_price", "convert_currency"], ["book_restaurant"]),
        ("Find Dune showtimes at the Tivoli, then book a restaurant table at Pastaria for 2 at 7 PM.", ["find_movie_showtimes", "book_restaurant"], ["search_flights"]),
    ]
    request, selected, decoys = rng.choice(cases)
    tools = tool_list(*(selected + decoys))
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Identify each requested external action from the available action catalog.", ["user request", "available tools"], ["matched actions"], "all requested actions are matched", external_action_type="external_tool_call"),
        cap(2, "order_external_actions_by_dependency", "Order the selected actions according to the user's sequence and dependencies.", ["matched actions"], ["ordered external actions"], "the action sequence preserves required order", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="perform multiple requested external actions in order",
        available_inputs=[available_input("multi-action request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding",
        operation="select_ordered_external_actions",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": " -> ".join(selected)},
        tool_binding=tool_target(selected, "The request contains two distinct external actions that should be called in the stated order."),
    )
    return make_record(record_id=f"capmap_tool_multiple_{seq:06d}", family="tool_binding_multiple", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_binding_parallel(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Get weather for Boston, Austin, and Denver.", "get_weather", 3, ["convert_currency", "translate_text"]),
        ("Estimate home prices for 10 Oak St and 99 Pine Ave.", "estimate_home_price", 2, ["get_weather", "book_restaurant"]),
        ("Find Dune showtimes at the Tivoli and the Alamo Drafthouse for tomorrow.", "find_movie_showtimes", 2, ["book_restaurant", "search_flights"]),
        ("Convert 25 USD to EUR, 30 USD to JPY, and 40 USD to CAD.", "convert_currency", 3, ["get_weather", "solve_quadratic"]),
    ]
    request, selected, count, decoys = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    sequence = [selected] * count
    caps = [
        cap(1, "match_request_to_available_actions", "Select the available action that matches the repeated operation.", ["user request", "available tools"], ["matched action"], "the repeated operation is matched to an available action", external_action_type="external_tool_call"),
        cap(2, "expand_repeated_action_calls", "Create one planned call per independent requested entity when the action handles one entity at a time.", ["matched action", "requested entities"], ["repeated external action sequence"], "the planned call count matches the number of requested entities", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="perform the same external action for multiple requested entities",
        available_inputs=[available_input("parallel action request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding",
        operation="select_parallel_repeated_external_actions",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": f"{selected} repeated {count} times"},
        tool_binding=tool_target(sequence, "The request asks the same single-entity action for multiple entities."),
    )
    return make_record(record_id=f"capmap_tool_parallel_{seq:06d}", family="tool_binding_parallel", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_irrelevance(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("What is 1 + 2 + log base 3 of 90? Show your reasoning.", ["get_weather", "book_restaurant"]),
        ("Thanks, that helps.", ["search_flights", "convert_currency"]),
        ("Explain why unit tests are useful in one paragraph.", ["find_movie_showtimes", "estimate_home_price"]),
        ("Rewrite this sentence: The report are late.", ["search_flights", "book_restaurant"]),
        ("Can you brainstorm three names for a coffee shop?", ["solve_quadratic", "get_weather"]),
    ]
    request, names = rng.choice(cases)
    tools = tool_list(*names)
    caps = [
        cap(1, "answer_or_transform_directly", "Handle the self-contained request without selecting unrelated available actions.", ["user request"], ["direct answer"], "the request is answered without irrelevant external actions")
    ]
    gold = target(
        final_user_intent="handle a self-contained request without external action",
        available_inputs=[available_input("self-contained request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="direct_or_no_tool",
        operation="avoid_irrelevant_external_action",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"operation_span": request, "no_tool_reason": "available actions do not match the user request"},
        tool_binding=tool_target([], "No available action is semantically required."),
    )
    return make_record(record_id=f"capmap_tool_no_call_{seq:06d}", family="tool_irrelevance", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_missing_attachment_reference(rng: random.Random, seq: int) -> dict[str, Any]:
    source_name, phrase, intent = rng.choice(
        [
            ("attached paper", "summarize the paper I attached", "summarize the attached paper"),
            ("attached slide deck", "turn the deck I attached into speaker notes", "create speaker notes from the attached deck"),
            ("uploaded contract", "find risky clauses in the contract I uploaded", "review the uploaded contract"),
            ("image attachment", "describe the screenshot I attached", "answer from the attached screenshot"),
            ("spreadsheet attachment", "calculate churn by cohort from the spreadsheet I attached", "analyze the attached spreadsheet"),
        ]
    )
    request = rng.choice([f"Can you {phrase}?", f"Please {phrase}.", f"I need you to {phrase} and keep it short."])
    caps = [
        cap(
            1,
            "request_missing_input",
            f"Ask the user to attach or provide the missing {source_name}.",
            [],
            [source_name],
            f"the user has been asked to provide the missing {source_name}",
            external_action_type="user_input",
        )
    ]
    gold = target(
        final_user_intent=intent,
        available_inputs=[],
        missing_inputs=[missing_input(source_name, intent, "The request refers to an attachment, but attachment metadata is empty.")],
        task_family="missing_attachment",
        operation="request_referenced_attachment",
        input_format="none",
        capabilities=caps,
        external_action_type=["user_input"],
        tool_decision="ask_user",
        evidence={"operation_span": phrase, "missing_reason": "attachment reference without attachment metadata"},
    )
    return make_record(record_id=f"capmap_missing_attachment_{seq:06d}", family="missing_attachment_reference", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_text_classification(rng: random.Random, seq: int) -> dict[str, Any]:
    label, text = rng.choice(TEXT_SNIPPETS)
    op, instruction, output = rng.choice(
        [
            ("classify_sentiment", "Classify the sentiment of this text", "sentiment label"),
            ("classify_urgency", "Label this support message as low, medium, or high urgency", "urgency label"),
            ("classify_topic", "Classify the main topic of this note", "topic label"),
            ("detect_policy_violation", "Decide whether this message contains a policy violation", "classification decision"),
        ]
    )
    request = f"{instruction}:\n\n{text}"
    caps = [
        cap(1, "analyze_provided_text", "Read the provided text and identify classification evidence.", ["pasted text"], ["classification evidence"], "the relevant evidence is identified"),
        cap(2, "classify_text", "Assign the requested classification label from the evidence.", ["classification evidence"], [output], "the requested label is produced", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent=instruction.lower(),
        available_inputs=[available_input(label, "pasted_text", "request", text)],
        missing_inputs=[],
        task_family="text_classification",
        operation=op,
        input_format="pasted_text",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"input_span": text[:120], "operation_span": instruction},
    )
    return make_record(record_id=f"capmap_text_classification_{seq:06d}", family="text_classification", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_math_reasoning(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("What is 17 * 24 minus 39?", "compute_arithmetic_result"),
        ("A plan costs $18 per user for 14 users. What is the monthly total?", "compute_word_problem_result"),
        ("If revenue grew from 1200 to 1560, what is the percent increase?", "compute_percentage_change"),
        ("What is 1 + 2 + log base 3 of 81?", "compute_mixed_expression"),
        ("If a train travels 180 miles in 3 hours, what is the average speed?", "compute_rate"),
    ]
    request, op = rng.choice(cases)
    caps = [
        cap(1, "parse_numeric_problem", "Identify the numbers, units, and requested calculation.", ["math request"], ["calculation spec"], "the calculation requirements are clear"),
        cap(2, "compute_numeric_result", "Carry out the requested calculation.", ["calculation spec"], ["computed result"], "the numeric result is computed", depends_on=["cap_1"]),
        cap(3, "report_findings", "Present the numeric result with concise explanation.", ["computed result"], ["answer"], "the answer includes the result and needed units", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent="solve a self-contained numeric problem",
        available_inputs=[available_input("math problem", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="calculation",
        operation=op,
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"operation_span": request, "input_span": request},
    )
    return make_record(record_id=f"capmap_math_{seq:06d}", family="math_reasoning", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_repo_search(rng: random.Random, seq: int) -> dict[str, Any]:
    pattern, op = rng.choice(REPO_PATTERNS)
    request = rng.choice(
        [
            f"Scan this repo for {pattern} and give me a cleanup list.",
            f"Find {pattern} across the codebase and summarize the risky files.",
            f"Search the project for {pattern}, then suggest fixes.",
        ]
    )
    caps = [
        cap(1, "search_provided_files", "Search the local project files for the requested pattern.", ["workspace files"], ["matching files"], "relevant matches are found", external_action_type="file_reading"),
        cap(2, "extract_matches", "Extract the matching snippets and organize them by file.", ["matching files"], ["organized matches"], "matches are grouped with useful context", depends_on=["cap_1"]),
        cap(3, "draft_cleanup_report", "Summarize the findings and suggest next actions.", ["organized matches"], ["cleanup report"], "the report identifies files and suggested fixes", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=f"search the repository for {pattern}",
        available_inputs=[available_input("workspace", "repo_files", "runtime", "this repo")],
        missing_inputs=[],
        task_family="repo_search",
        operation=op,
        input_format="repo_files",
        capabilities=caps,
        external_action_type=["file_reading"],
        tool_decision="no_tool",
        evidence={"operation_span": pattern, "input_span": "this repo"},
    )
    return make_record(record_id=f"capmap_repo_search_{seq:06d}", family="repo_search", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_multi_attachment_compare(rng: random.Random, seq: int) -> dict[str, Any]:
    left = attachment(rng.choice(["proposal_v1.pdf", "policy_old.docx", "vendor_a.pdf"]), rng.choice(["pdf", "docx"]), 1)
    right = attachment(rng.choice(["proposal_v2.pdf", "policy_new.docx", "vendor_b.pdf"]), rng.choice(["pdf", "docx"]), 2)
    request = rng.choice(
        [
            "Compare the two attached documents and list the important differences.",
            "Find what changed between these two attachments.",
            "Summarize where the attached documents agree and disagree.",
        ]
    )
    caps = [
        cap(1, "extract_information_from_attached_document", "Extract text from each attached document.", [left["name"], right["name"]], ["document texts"], "both attachments are extracted", external_action_type="file_reading"),
        cap(2, "compare_document_content", "Compare the extracted document content.", ["document texts"], ["comparison"], "important differences and overlaps are identified", depends_on=["cap_1"]),
        cap(3, "report_comparison", "Present the comparison clearly.", ["comparison"], ["answer"], "the comparison is reported in the requested form", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent="compare two attached documents",
        available_inputs=[available_input("document pair", "attached_files", "attachment", f"{left['name']} {right['name']}")],
        missing_inputs=[],
        task_family="multi_attachment_document",
        operation="compare_attached_documents",
        input_format="attached_files",
        capabilities=caps,
        external_action_type=["file_reading"],
        tool_decision="no_tool",
        evidence={"input_span": f"{left['name']} {right['name']}", "operation_span": request},
    )
    return make_record(record_id=f"capmap_multi_attachment_{seq:06d}", family="multi_attachment_compare", request=request, attachments_metadata=[left, right], available_tools=[], gold_target=gold)


def gen_media_transcription(rng: random.Random, seq: int) -> dict[str, Any]:
    media = rng.choice([attachment("customer_call.mp3", "mp3"), attachment("townhall_recording.mp4", "mp4"), attachment("interview_clip.mp3", "mp3")])
    op, instruction = rng.choice(
        [
            ("transcribe_audio", "Transcribe the attached recording."),
            ("summarize_media", "Summarize the attached recording and list follow-ups."),
            ("extract_speakers_and_actions", "Extract speakers and action items from the attached media."),
        ]
    )
    caps = [
        cap(1, "extract_audio_or_video_content", "Process the attached media into text or observations.", [media["name"]], ["media transcript"], "media content is available as text", external_action_type="file_reading"),
        cap(2, "analyze_media_transcript", "Analyze the transcript for the requested information.", ["media transcript"], ["media analysis"], "requested media details are identified", depends_on=["cap_1"]),
        cap(3, "produce_media_answer", "Produce the requested transcription, summary, or extraction.", ["media analysis"], ["answer"], "the media-based answer is complete", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=instruction.rstrip(".").lower(),
        available_inputs=[available_input(media["name"], "media", "attachment", media["name"])],
        missing_inputs=[],
        task_family="attached_media",
        operation=op,
        input_format="media",
        capabilities=caps,
        external_action_type=["file_reading"],
        tool_decision="no_tool",
        evidence={"input_span": media["name"], "operation_span": instruction},
    )
    return make_record(record_id=f"capmap_media_{seq:06d}", family="media_transcription", request=instruction, attachments_metadata=[media], available_tools=[], gold_target=gold)


def gen_chart_generation(rng: random.Random, seq: int) -> dict[str, Any]:
    data = rng.choice(DATASETS)
    chart = rng.choice(["bar chart", "line chart", "stacked chart", "summary table with chart notes"])
    request = f"Create a {chart} from this {data['kind']} showing {data['metric']} by {data['group']}:\n{data['block']}"
    caps = [
        cap(1, "parse_structured_input", "Parse the inline structured data into rows and fields.", ["inline structured data"], ["parsed dataset"], "the dataset is parsed"),
        cap(2, "derive_visualization_spec", "Choose fields and chart structure for the requested visualization.", ["parsed dataset"], ["chart spec"], "the chart fields and encoding are specified", depends_on=["cap_1"]),
        cap(3, "generate_visual_output", "Create the requested chart or chart-ready output.", ["chart spec"], ["visual output"], "the requested visualization is produced", external_action_type="file_writing", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=f"create a {chart} from provided data",
        available_inputs=[available_input(data["name"], "structured_data", "request", data["block"])],
        missing_inputs=[],
        task_family="data_visualization",
        operation="create_chart_from_data",
        input_format="structured_data",
        capabilities=caps,
        external_action_type=["file_writing"],
        tool_decision="no_tool",
        evidence={"input_span": data["block"][:120], "operation_span": request.split("\n", 1)[0]},
    )
    return make_record(record_id=f"capmap_chart_{seq:06d}", family="chart_generation", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_database_query_task(rng: random.Random, seq: int) -> dict[str, Any]:
    schemas = [
        "orders(id, plan, amount, created_at), users(id, region)",
        "tickets(id, priority, status, created_at), agents(id, team)",
        "events(user_id, event_name, created_at), accounts(id, signup_date)",
    ]
    schema = rng.choice(schemas)
    metric = rng.choice(["monthly revenue", "open tickets by priority", "weekly active users"])
    request = f"Given this schema, write a SQL query for {metric}:\n{schema}"
    caps = [
        cap(1, "interpret_database_schema", "Read the provided schema and requested metric.", ["database schema", "query request"], ["query spec"], "tables, fields, and target metric are understood"),
        cap(2, "generate_database_query", "Write a query that computes the requested metric.", ["query spec"], ["SQL query"], "the SQL query matches the schema and metric", depends_on=["cap_1"]),
        cap(3, "validate_generated_query", "Check that the query references available tables and columns.", ["SQL query", "query spec"], ["validation result"], "the query is consistent with the schema", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent=f"write a SQL query for {metric}",
        available_inputs=[available_input("database schema", "schema_text", "request", schema)],
        missing_inputs=[],
        task_family="database_query_generation",
        operation="generate_sql_from_schema",
        input_format="schema_text",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"input_span": schema, "operation_span": metric},
    )
    return make_record(record_id=f"capmap_database_query_{seq:06d}", family="database_query_task", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_tool_customer_service(rng: random.Random, seq: int) -> dict[str, Any]:
    order_id = f"ORD-{rng.randint(1000, 9999)}"
    product = rng.choice(PRODUCTS)
    kind = rng.choice(["refund", "address_change", "cancel"])
    if kind == "refund":
        request = f"Customer wants a refund for order {order_id} for {product}. Check eligibility and issue it if allowed."
        selected = ["lookup_order", "check_return_policy", "issue_refund"]
        op = "customer_refund_workflow"
    elif kind == "address_change":
        request = f"Change the shipping address for order {order_id} to 12 Pine St, Austin, TX."
        selected = ["lookup_order", "update_shipping_address"]
        op = "customer_order_update_workflow"
    else:
        request = f"Cancel order {order_id} if it has not shipped yet."
        selected = ["lookup_order", "cancel_order"]
        op = "customer_order_cancel_workflow"
    tools = tool_list(*(selected + ["search_products", "convert_currency"]))
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Identify the customer-service actions required by the request.", ["user request", "available tools"], ["matched actions"], "the needed actions are matched", external_action_type="external_tool_call"),
        cap(2, "order_external_actions_by_dependency", "Order lookup, policy, and mutation actions so state-changing calls happen after checks.", ["matched actions"], ["ordered external actions"], "lookup/check actions precede updates or refunds", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="complete a customer-service order workflow",
        available_inputs=[available_input("customer-service request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_customer_service",
        operation=op,
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": " -> ".join(selected)},
        tool_binding=tool_target(selected, "The workflow requires lookup before any policy check or state-changing action."),
    )
    return make_record(record_id=f"capmap_tool_customer_{seq:06d}", family="tool_customer_service", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_ecommerce_workflow(rng: random.Random, seq: int) -> dict[str, Any]:
    product = rng.choice(PRODUCTS)
    request = rng.choice(
        [
            f"Find a {product} under $100, add the best option to my cart, and check out with my default card.",
            f"Search for a waterproof {product}, add one to cart, then checkout.",
            f"Buy a replacement {product}: search, add it to cart, and complete checkout.",
        ]
    )
    selected = ["search_products", "add_to_cart", "checkout_cart"]
    tools = tool_list(*(selected + ["lookup_order", "get_weather"]))
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Identify search, cart, and checkout actions from the available tool list.", ["user request", "available tools"], ["matched actions"], "shopping actions are matched", external_action_type="external_tool_call"),
        cap(2, "order_external_actions_by_dependency", "Order shopping actions so product search precedes cart update and checkout.", ["matched actions"], ["ordered external actions"], "search, add, and checkout are ordered correctly", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="complete an ecommerce purchase workflow",
        available_inputs=[available_input("shopping request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_ecommerce",
        operation="ecommerce_search_cart_checkout",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": " -> ".join(selected)},
        tool_binding=tool_target(selected, "The request requires a search, then a cart mutation, then checkout."),
    )
    return make_record(record_id=f"capmap_tool_ecommerce_{seq:06d}", family="tool_ecommerce_workflow", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_api_crud(rng: random.Random, seq: int) -> dict[str, Any]:
    user_id = f"user_{rng.randint(100, 999)}"
    amount = rng.choice([25, 40, 75, 120, 250])
    cases = [
        (f"Transfer ${amount} from checking to savings after checking the account balance.", ["get_account_balance", "transfer_funds"], "bank_transfer_workflow"),
        (f"Update {user_id}'s email to new_{user_id}@example.com.", ["update_user_email"], "update_user_record"),
        (f"Check account balance and email the summary to finance@example.com.", ["get_account_balance", "send_email"], "retrieve_then_notify"),
        (f"Run a read-only query for daily active users and send the result to #metrics.", ["query_database", "send_slack_message"], "query_then_notify"),
    ]
    request, selected, op = rng.choice(cases)
    tools = tool_list(*(selected + ["book_restaurant", "find_movie_showtimes"]))
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Map the requested API-style operation to available actions.", ["user request", "available tools"], ["matched actions"], "the relevant API actions are selected", external_action_type="external_tool_call"),
        cap(2, "order_external_actions_by_dependency", "Order retrieval and mutation or notification actions appropriately.", ["matched actions"], ["ordered external actions"], "dependent API actions are ordered correctly", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="perform an API-style account or notification workflow",
        available_inputs=[available_input("API workflow request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_api_workflow",
        operation=op,
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": " -> ".join(selected)},
        tool_binding=tool_target(selected, "The request maps to the selected API actions in dependency order."),
    )
    return make_record(record_id=f"capmap_tool_api_crud_{seq:06d}", family="tool_api_crud", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_batch_capable(rng: random.Random, seq: int) -> dict[str, Any]:
    cities = rng.sample(CITIES, rng.choice([2, 3, 4]))
    request = f"Get weather for {', '.join(cities[:-1])}, and {cities[-1]}."
    selected = ["get_weather_batch"]
    tools = tool_list("get_weather_batch", "get_weather", "convert_currency")
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Detect that a batch-capable action can satisfy the repeated request.", ["user request", "available tools"], ["matched batch action"], "the batch action is selected over repeated single calls", external_action_type="external_tool_call"),
        cap(2, "plan_external_action_sequence", "Plan one call because the selected action accepts multiple entities.", ["matched batch action"], ["one external action"], "one batch call covers all requested cities", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="perform a repeated lookup with a batch-capable action",
        available_inputs=[available_input("batch weather request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_batch",
        operation="select_batch_external_action",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": "get_weather_batch handles multiple cities"},
        tool_binding=tool_target(selected, "Use the batch action once because it accepts multiple cities."),
    )
    return make_record(record_id=f"capmap_tool_batch_{seq:06d}", family="tool_batch_capable", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_parallel_multiple(rng: random.Random, seq: int) -> dict[str, Any]:
    cities = rng.sample(CITIES, 2)
    restaurants = rng.sample(["Pastaria", "Nobu", "Bistro 31", "Blue Hill"], 2)
    addresses = rng.sample(["10 Oak St", "99 Pine Ave", "42 Maple Dr", "7 Cedar Ct"], 2)
    theaters = rng.sample(["Tivoli", "Alamo Drafthouse", "Regal Loop", "Cinema Plaza"], 2)
    cases = [
        (
            f"Get weather for {cities[0]} and {cities[1]}, then book tables at {restaurants[0]} and {restaurants[1]} for Friday at 7 PM.",
            ["get_weather", "get_weather", "book_restaurant", "book_restaurant"],
            ["get_weather", "book_restaurant", "search_flights", "translate_text"],
            "two weather calls then two booking calls",
        ),
        (
            f"Estimate home prices for {addresses[0]} and {addresses[1]}, then email the two results to maya@example.com and priya@example.com.",
            ["estimate_home_price", "estimate_home_price", "send_email", "send_email"],
            ["estimate_home_price", "send_email", "convert_currency", "lookup_order"],
            "two price-estimate calls then two email calls",
        ),
        (
            f"Find Dune showtimes at {theaters[0]} and {theaters[1]}, then get weather for {cities[0]} and {cities[1]}.",
            ["find_movie_showtimes", "find_movie_showtimes", "get_weather", "get_weather"],
            ["find_movie_showtimes", "get_weather", "book_restaurant", "search_flights"],
            "two showtime calls then two weather calls",
        ),
    ]
    request, selected, names, sequence_reason = rng.choice(cases)
    tools = tool_list(*names)
    rng.shuffle(tools)
    caps = [
        cap(1, "match_request_to_available_actions", "Match each distinct repeated operation to available actions.", ["user request", "available tools"], ["matched repeated actions"], "all repeated action groups are matched", external_action_type="external_tool_call"),
        cap(2, "expand_repeated_action_calls", "Plan one call per independent entity because the selected tools are single-entity actions.", ["matched repeated actions"], ["expanded action sequence"], "the sequence has one call for each requested entity", external_action_type="external_tool_call", depends_on=["cap_1"]),
        cap(3, "order_external_actions_by_dependency", "Preserve the user's requested operation order across repeated action groups.", ["expanded action sequence"], ["ordered external actions"], "weather calls precede restaurant bookings", external_action_type="external_tool_call", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent="perform multiple repeated external actions in order",
        available_inputs=[available_input("parallel multiple request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_parallel_multiple",
        operation="expand_and_order_multiple_repeated_external_actions",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": sequence_reason},
        tool_binding=tool_target(selected, "Each selected tool handles one entity, so repeated entities require repeated calls in group order."),
    )
    return make_record(record_id=f"capmap_tool_parallel_multiple_{seq:06d}", family="tool_parallel_multiple", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_missing_required_slot(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Book me a restaurant table tonight.", "restaurant date/time or restaurant name", ["book_restaurant"]),
        ("Transfer money to savings.", "transfer amount and source account", ["transfer_funds"]),
        ("Send an email to Maya.", "email subject and message body", ["send_email"]),
        ("Create a calendar event for the meeting.", "date and time", ["create_calendar_event"]),
        ("Refund my order.", "order id", ["issue_refund", "lookup_order"]),
    ]
    request, missing, names = rng.choice(cases)
    tools = tool_list(*(names + ["get_weather", "translate_text"]))
    rng.shuffle(tools)
    caps = [
        cap(
            1,
            "request_missing_input",
            "Ask for the required slot values before planning an external action call.",
            ["user request", "available tools"],
            [missing],
            "the user has been asked for the missing required slot values",
            external_action_type="user_input",
        )
    ]
    gold = target(
        final_user_intent="prepare an external action request but required slots are missing",
        available_inputs=[available_input("partial tool request", "natural_language_requirements", "request", request)],
        missing_inputs=[missing_input(missing, "external action call", "The available action requires slot values that are not present in the request.")],
        task_family="tool_binding_missing_required_slot",
        operation="request_missing_slots_before_tool_call",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["user_input"],
        tool_decision="ask_user",
        evidence={"operation_span": request, "missing_reason": missing},
        tool_binding=tool_target([], "Do not call an external action until required slots are available."),
    )
    return make_record(record_id=f"capmap_tool_missing_slot_{seq:06d}", family="tool_missing_required_slot", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_false_no_call_hard(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        (
            "Calculate the probability of drawing an ace from a standard 52-card deck in one draw.",
            "calculate_card_probability",
            ["solve_quadratic", "get_weather", "convert_currency"],
            "card probability is an actionable tool request, not a direct-answer no-call case",
        ),
        (
            "Build a 3-day travel itinerary for Tokyo from 2026-08-14 to 2026-08-17 with a $1200 budget.",
            "plan_travel_itinerary",
            ["search_flights", "book_restaurant", "get_weather"],
            "destination, dates, and budget are present, so the itinerary tool should be called",
        ),
        (
            "Send an email to maya@example.com with subject 'Draft ready' and body 'The proposal draft is ready for review.'",
            "send_email",
            ["create_support_ticket", "send_slack_message", "translate_text"],
            "recipient, subject, and body are provided",
        ),
        (
            "Look up court case 24-CV-1022 in Cook County.",
            "lookup_court_case",
            ["search_courthouse_locations", "query_database", "get_weather"],
            "jurisdiction and case number are provided",
        ),
        (
            "Run an fMRI quality check on /data/sub-03/func/sub-03_task-rest_bold.nii.gz.",
            "analyze_fmri_file",
            ["query_database", "translate_text", "get_weather"],
            "the file path is a provided parameter value",
        ),
    ]
    request, selected, decoys, reason = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    caps = [
        cap(1, "recognize_actionable_tool_request", "Detect that the request should be satisfied by an available external action.", ["user request", "available tools"], ["tool-call decision"], "the planner chooses to call rather than answer directly or ask for missing input", external_action_type="external_tool_call"),
        cap(2, "select_exact_external_action", "Select the exact tool matching the requested action.", ["tool-call decision", "available tools"], ["selected external action"], "the selected action matches the requested operation", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="call the matching available tool for an actionable benchmark-style request",
        available_inputs=[available_input("actionable tool request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_false_no_call",
        operation="avoid_false_no_call_for_actionable_tool_request",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": selected, "false_no_call_guard": reason},
        tool_binding=tool_target([selected], reason),
    )
    return make_record(record_id=f"capmap_tool_false_no_call_{seq:06d}", family="tool_false_no_call_hard", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_parameter_not_missing(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        (
            "Run the top-k function on dataset_A with k=5.",
            "find_top_k",
            ["query_database", "translate_text"],
            "dataset_A and k=5 are supplied parameter values",
        ),
        (
            "Calculate the moving average for nums=[5, 8, 13, 21, 34] using k=2.",
            "calculate_moving_average",
            ["find_top_k", "solve_quadratic"],
            "nums and k are supplied parameter values",
        ),
        (
            "Analyze fMRI file_path=/mnt/bfcl/sub-01_task-rest_bold.nii.gz with analysis='motion_summary'.",
            "analyze_fmri_file",
            ["query_database", "lookup_court_case"],
            "file_path and analysis are supplied parameter values",
        ),
        (
            "Estimate trip cost from coordinates 38.6270,-90.1994 to 41.8781,-87.6298 with budget=750.",
            "estimate_trip_cost",
            ["plan_route", "convert_currency"],
            "coordinates and budget are supplied parameter values",
        ),
    ]
    request, selected, decoys, reason = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    caps = [
        cap(1, "audit_tool_parameters_as_available", "Treat symbolic names, arrays, coordinates, budgets, and file paths in the request as available parameter values.", ["user request", "available tools"], ["available tool parameters"], "provided parameter-like strings are not marked missing"),
        cap(2, "plan_external_action_sequence", "Plan the matching external action using those available parameters.", ["available tool parameters"], ["selected external action"], "the tool is selected with no missing-input request", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="call a tool using parameter values already present in the request",
        available_inputs=[available_input("provided tool parameters", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_parameter_audit",
        operation="treat_parameter_names_as_available_values",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": selected, "provided_parameter_evidence": reason},
        tool_binding=tool_target([selected], reason),
    )
    return make_record(record_id=f"capmap_tool_parameter_value_{seq:06d}", family="tool_parameter_not_missing", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_semantic_near_miss_hard(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        (
            "Get historical weather for coordinates 40.7128,-74.0060 on 2020-01-05.",
            "get_historical_weather_by_coordinates",
            ["get_weather_forecast_by_coordinates", "get_weather"],
            "historical weather with a date is not a generic forecast",
        ),
        (
            "Get the forecast for coordinates 34.0522,-118.2437 tomorrow.",
            "get_weather_forecast_by_coordinates",
            ["get_historical_weather_by_coordinates", "get_weather"],
            "forecast weather is not historical weather lookup",
        ),
        (
            "Look up court case 24-CV-1022 in Cook County.",
            "lookup_court_case",
            ["search_courthouse_locations", "query_database"],
            "case lookup is not courthouse location search",
        ),
        (
            "Find courthouse locations near downtown Chicago.",
            "search_courthouse_locations",
            ["lookup_court_case", "plan_route"],
            "courthouse location search is not case lookup",
        ),
    ]
    request, selected, decoys, reason = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    caps = [
        cap(1, "disambiguate_semantic_neighbor_tools", "Compare close tool descriptions against the exact user operation.", ["user request", "available tools"], ["exact tool match"], "near-miss tools are rejected", external_action_type="external_tool_call"),
        cap(2, "plan_external_action_sequence", "Plan only the exact matching action.", ["exact tool match"], ["selected external action"], "the exact tool is selected once", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="select the exact tool instead of a semantically adjacent cousin",
        available_inputs=[available_input("semantically specific tool request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_semantic_near_miss",
        operation="select_exact_tool_over_semantic_cousin",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": selected, "near_miss_guard": reason},
        tool_binding=tool_target([selected], reason),
    )
    return make_record(record_id=f"capmap_tool_near_miss_{seq:06d}", family="tool_semantic_near_miss_hard", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_overcall_related_hard(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        (
            "Plan a route from Union Station to the chess championship venue at 123 Market St.",
            "plan_route",
            ["lookup_chess_club", "find_chess_tournament_info"],
            "chess is context for the destination, not a request for chess lookup",
        ),
        (
            "Send an email to team@example.com about tomorrow's restaurant reservation; do not book anything.",
            "send_email",
            ["book_restaurant", "get_weather"],
            "restaurant is message content, not a booking request",
        ),
        (
            "Plan the route from my hotel to the courthouse for the 9 AM hearing.",
            "plan_route",
            ["lookup_court_case", "search_courthouse_locations"],
            "court is context for the route destination, not a case lookup",
        ),
        (
            "Create a calendar event called flight search review for Friday at 3 PM.",
            "create_calendar_event",
            ["search_flights", "send_email"],
            "flight search is the event title, not a flight-search action",
        ),
    ]
    request, selected, decoys, reason = rng.choice(cases)
    tools = tool_list(selected, *decoys)
    rng.shuffle(tools)
    caps = [
        cap(1, "separate_requested_action_from_context", "Identify which nouns are contextual and which action is actually requested.", ["user request", "available tools"], ["single requested action"], "contextual related tools are not selected"),
        cap(2, "plan_external_action_sequence", "Plan only the requested external action.", ["single requested action"], ["one external action"], "exactly one requested action is selected", external_action_type="external_tool_call", depends_on=["cap_1"]),
    ]
    gold = target(
        final_user_intent="call only the requested tool and suppress related-but-unrequested tools",
        available_inputs=[available_input("context-rich tool request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="tool_binding_overcall_suppression",
        operation="avoid_related_but_unrequested_tool_calls",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["external_tool_call"],
        tool_decision="call",
        evidence={"operation_span": request, "selected_tool_evidence": selected, "overcall_guard": reason},
        tool_binding=tool_target([selected], reason),
    )
    return make_record(record_id=f"capmap_tool_overcall_{seq:06d}", family="tool_overcall_related_hard", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_tool_keyword_no_call_hard(rng: random.Random, seq: int) -> dict[str, Any]:
    cases = [
        ("Explain what a weather API does in plain English.", ["get_weather", "call_rest_api"]),
        ("Write a short poem about booking a restaurant in Paris.", ["book_restaurant", "get_weather"]),
        ("How should I structure an email asking for a deadline extension?", ["send_email", "create_calendar_event"]),
        ("Describe how court case lookup systems usually work.", ["lookup_court_case", "search_courthouse_locations"]),
        ("Give me study tips for understanding REST API authentication.", ["call_rest_api", "query_database"]),
    ]
    request, names = rng.choice(cases)
    tools = tool_list(*names)
    rng.shuffle(tools)
    caps = [
        cap(1, "recognize_non_actionable_keyword_overlap", "Detect that tool-related words are being discussed rather than requested as actions.", ["user request", "available tools"], ["direct response decision"], "keyword overlap does not trigger a tool call")
    ]
    gold = target(
        final_user_intent="answer a self-contained explanatory or creative request without calling tools",
        available_inputs=[available_input("keyword-overlap request", "natural_language_requirements", "request", request)],
        missing_inputs=[],
        task_family="direct_or_no_tool",
        operation="no_call_for_keyword_overlap_without_action",
        input_format="natural_language_requirements",
        capabilities=caps,
        external_action_type=["none"],
        tool_decision="no_tool",
        evidence={"operation_span": request, "no_tool_reason": "tool names or domains are mentioned, but no external action is requested"},
        tool_binding=tool_target([], "No external action should be called for explanatory or creative keyword overlap."),
    )
    return make_record(record_id=f"capmap_tool_keyword_no_call_{seq:06d}", family="tool_keyword_no_call_hard", request=request, attachments_metadata=[], available_tools=tools, gold_target=gold)


def gen_ambiguous_request(rng: random.Random, seq: int) -> dict[str, Any]:
    thing = rng.choice(["the report", "the dashboard", "the model", "the workflow", "the issue"])
    request = rng.choice([f"Can you fix {thing}?", f"Make {thing} better.", f"Look into {thing} and improve it."])
    caps = [
        cap(
            1,
            "request_clarification",
            "Ask the user to specify the source, problem, and desired outcome before planning work.",
            [],
            ["clarified task requirements"],
            "the user is asked for enough detail to proceed",
            external_action_type="user_input",
        )
    ]
    gold = target(
        final_user_intent="clarify an underspecified task",
        available_inputs=[],
        missing_inputs=[missing_input("specific source and desired change", "clarify task", "The request names a vague object without providing the source, problem, or target outcome.")],
        task_family="ambiguous_or_underspecified",
        operation="request_clarification",
        input_format="none",
        capabilities=caps,
        external_action_type=["user_input"],
        tool_decision="ask_user",
        evidence={"operation_span": request, "missing_reason": "underspecified source and desired outcome"},
    )
    return make_record(record_id=f"capmap_ambiguous_{seq:06d}", family="ambiguous_request", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_image_edit_generation(rng: random.Random, seq: int) -> dict[str, Any]:
    has_attachment = rng.choice([True, False])
    if has_attachment:
        img = attachment(rng.choice(["profile_photo.png", "product_mockup.jpg", "room_photo.png"]), rng.choice(["png", "jpg"]))
        request = rng.choice(
            [
                "Remove the background from the attached image.",
                "Make the attached product photo look brighter and cleaner.",
                "Turn this attached photo into a simple illustration.",
            ]
        )
        caps = [
            cap(1, "interpret_image_edit_request", "Identify the requested visual edit and source image.", [img["name"], "edit request"], ["edit specification"], "the edit requirements are clear", external_action_type="image_analysis"),
            cap(2, "generate_or_edit_image", "Apply the requested visual transformation.", ["edit specification"], ["edited image"], "the edited image is produced", external_action_type="image_generation", depends_on=["cap_1"]),
        ]
        available = [available_input(img["name"], "image", "attachment", img["name"])]
        attachments = [img]
        missing = []
        tool_decision = "no_tool"
        action_types = ["image_analysis", "image_generation"]
        input_format = "image"
        task_family = "image_edit"
        operation = "edit_attached_image"
    else:
        request = rng.choice(
            [
                "Create an image of a minimal desk setup for a productivity blog.",
                "Generate a square app icon for a habit tracker.",
                "Make a clean product mockup image for a reusable water bottle.",
            ]
        )
        caps = [
            cap(1, "interpret_image_generation_request", "Identify the visual subject, style, and output constraints.", ["image request"], ["image specification"], "the image generation requirements are clear"),
            cap(2, "generate_or_edit_image", "Create a new image from the specification.", ["image specification"], ["generated image"], "the requested image is generated", external_action_type="image_generation", depends_on=["cap_1"]),
        ]
        available = [available_input("image request", "natural_language_requirements", "request", request)]
        attachments = []
        missing = []
        tool_decision = "no_tool"
        action_types = ["image_generation"]
        input_format = "natural_language_requirements"
        task_family = "image_generation"
        operation = "generate_image_from_prompt"
    gold = target(
        final_user_intent=request.rstrip(".").lower(),
        available_inputs=available,
        missing_inputs=missing,
        task_family=task_family,
        operation=operation,
        input_format=input_format,
        capabilities=caps,
        external_action_type=action_types,
        tool_decision=tool_decision,
        evidence={"operation_span": request, "input_span": available[0]["evidence_span"]},
    )
    return make_record(record_id=f"capmap_image_generation_{seq:06d}", family="image_edit_generation", request=request, attachments_metadata=attachments, available_tools=[], gold_target=gold)


def gen_mixed_provided_and_current(rng: random.Random, seq: int) -> dict[str, Any]:
    label, text = rng.choice(TEXT_SNIPPETS)
    topic, _ = rng.choice(CURRENT_FACT_TOPICS)
    request = f"Compare this note with the latest public information about {topic}:\n\n{text}"
    caps = [
        cap(1, "analyze_provided_text", "Extract the relevant claims from the provided text.", ["pasted text"], ["provided claims"], "claims from the pasted text are identified"),
        cap(2, "retrieve_current_information", "Retrieve current external information for comparison.", ["current-info topic"], ["retrieved current information"], "current relevant information is retrieved", external_action_type="web_search"),
        cap(3, "compare_provided_and_retrieved_information", "Compare the provided claims with current information.", ["provided claims", "retrieved current information"], ["comparison"], "matches, gaps, and conflicts are identified", depends_on=["cap_1", "cap_2"]),
        cap(4, "report_comparison", "Present the comparison with date context.", ["comparison"], ["answer"], "the answer distinguishes provided text from current information", depends_on=["cap_3"]),
    ]
    gold = target(
        final_user_intent="compare provided text with current external information",
        available_inputs=[
            available_input(label, "pasted_text", "request", text),
            available_input("current-info topic", "none", "request", topic),
        ],
        missing_inputs=[],
        task_family="mixed_provided_current_info",
        operation="compare_provided_text_with_current_info",
        input_format="mixed",
        capabilities=caps,
        external_action_type=["web_search"],
        tool_decision="no_tool",
        requires_external_current_info=True,
        evidence={"input_span": text[:120], "operation_span": topic},
    )
    return make_record(record_id=f"capmap_mixed_current_{seq:06d}", family="mixed_provided_and_current", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


def gen_local_file_summary(rng: random.Random, seq: int) -> dict[str, Any]:
    path = rng.choice(["docs/architecture.md", "reports/q2-retrospective.txt", "data/export.json", "logs/service.log"])
    request = rng.choice([f"Summarize {path}.", f"Read {path} and extract the risks.", f"Turn {path} into a short briefing."])
    caps = [
        cap(1, "read_referenced_file", "Read the local file referenced by path.", [path], ["file content"], "the referenced file content is available", external_action_type="file_reading"),
        cap(2, "analyze_file_content", "Analyze the file content for the requested information.", ["file content"], ["file analysis"], "the requested information is identified", depends_on=["cap_1"]),
        cap(3, "produce_file_based_answer", "Produce the requested summary, extraction, or briefing.", ["file analysis"], ["answer"], "the answer is grounded in the file content", depends_on=["cap_2"]),
    ]
    gold = target(
        final_user_intent="answer from a referenced local file",
        available_inputs=[available_input(path, "file_path", "request", path)],
        missing_inputs=[],
        task_family="local_file_task",
        operation="read_and_summarize_file",
        input_format="file_path",
        capabilities=caps,
        external_action_type=["file_reading"],
        tool_decision="no_tool",
        evidence={"input_span": path, "operation_span": request},
    )
    return make_record(record_id=f"capmap_local_file_{seq:06d}", family="local_file_summary", request=request, attachments_metadata=[], available_tools=[], gold_target=gold)


FAMILY_GENERATORS: list[tuple[str, Callable[[random.Random, int], dict[str, Any]]]] = [
    ("missing_input", gen_missing_input),
    ("pasted_text_transform", gen_pasted_text_transform),
    ("structured_data_analysis", gen_structured_data_analysis),
    ("attached_document", gen_attached_document),
    ("current_fact", gen_current_fact),
    ("url_task", gen_url_task),
    ("code_edit", gen_code_edit),
    ("code_generation", gen_code_generation),
    ("image_understanding", gen_image_understanding),
    ("unsupported_measurement", gen_unsupported_measurement),
    ("multi_step_dependency", gen_multistep_dependency),
    ("file_write", gen_file_write),
    ("tool_binding_single", gen_tool_binding_single),
    ("tool_binding_multiple", gen_tool_binding_multiple),
    ("tool_binding_parallel", gen_tool_binding_parallel),
    ("tool_irrelevance", gen_tool_irrelevance),
    ("missing_attachment_reference", gen_missing_attachment_reference),
    ("text_classification", gen_text_classification),
    ("math_reasoning", gen_math_reasoning),
    ("repo_search", gen_repo_search),
    ("multi_attachment_compare", gen_multi_attachment_compare),
    ("media_transcription", gen_media_transcription),
    ("chart_generation", gen_chart_generation),
    ("database_query_task", gen_database_query_task),
    ("tool_customer_service", gen_tool_customer_service),
    ("tool_ecommerce_workflow", gen_tool_ecommerce_workflow),
    ("tool_api_crud", gen_tool_api_crud),
    ("tool_batch_capable", gen_tool_batch_capable),
    ("tool_parallel_multiple", gen_tool_parallel_multiple),
    ("tool_missing_required_slot", gen_tool_missing_required_slot),
    ("tool_false_no_call_hard", gen_tool_false_no_call_hard),
    ("tool_parameter_not_missing", gen_tool_parameter_not_missing),
    ("tool_semantic_near_miss_hard", gen_tool_semantic_near_miss_hard),
    ("tool_overcall_related_hard", gen_tool_overcall_related_hard),
    ("tool_keyword_no_call_hard", gen_tool_keyword_no_call_hard),
    ("ambiguous_request", gen_ambiguous_request),
    ("image_edit_generation", gen_image_edit_generation),
    ("mixed_provided_and_current", gen_mixed_provided_and_current),
    ("local_file_summary", gen_local_file_summary),
]


def validate_record(record: dict[str, Any]) -> None:
    target_data = record["target"]
    compact_json(target_data)
    required_top = {"final_user_intent", "input_audit", "route", "capability_plan", "evidence"}
    missing_top = required_top - set(target_data)
    if missing_top:
        raise ValueError(f"{record['id']} target missing keys: {sorted(missing_top)}")

    caps = target_data["capability_plan"].get("ordered_capabilities", [])
    if not caps:
        raise ValueError(f"{record['id']} has no capabilities")
    ids = [item.get("id") for item in caps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{record['id']} has duplicate capability ids")
    id_set = set(ids)
    for item in caps:
        for field in ("capability_name", "capability_description", "inputs", "outputs", "external_action_type", "depends_on", "done_when"):
            if field not in item:
                raise ValueError(f"{record['id']} capability {item.get('id')} missing {field}")
        if not item["inputs"] and item["capability_name"] not in {"request_missing_input", "request_clarification"}:
            raise ValueError(f"{record['id']} capability {item.get('id')} has no inputs")
        for dep in item.get("depends_on", []):
            if dep not in id_set:
                raise ValueError(f"{record['id']} dependency {dep} does not exist")

    position = {item["id"]: index for index, item in enumerate(caps)}
    for item in caps:
        for dep in item.get("depends_on", []):
            if position[dep] >= position[item["id"]]:
                raise ValueError(f"{record['id']} has dependency order error: {dep} -> {item['id']}")

    route_actions = target_data["route"].get("external_action_type", [])
    cap_actions = {item.get("external_action_type") for item in caps}
    if "none" not in route_actions and not any(action in cap_actions for action in route_actions):
        if route_actions != ["external_tool_call"]:
            raise ValueError(f"{record['id']} route external actions not represented by capabilities")

    if record.get("available_tools") and "tool_binding" not in target_data:
        raise ValueError(f"{record['id']} has tools but no tool_binding target")
    if target_data["route"]["tool_decision"] == "call":
        selected = target_data.get("tool_binding", {}).get("ordered_tool_names", [])
        available = {tool["name"] for tool in record.get("available_tools", [])}
        if not selected or any(name not in available for name in selected):
            raise ValueError(f"{record['id']} selected unavailable tool")


def build_records(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for index in range(count):
        _, generator = FAMILY_GENERATORS[index % len(FAMILY_GENERATORS)]
        records.append(generator(rng, index + 1))
    rng.shuffle(records)
    return records


def sft_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "source": record["source"],
        "family": record["family"],
        "request": record["request"],
        "attachments_metadata": record["attachments_metadata"],
        "available_tools": record["available_tools"],
        "messages": record["messages"],
        "target": record["target"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


def split_records(records: list[dict[str, Any]], validation_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_count = max(1, round(len(records) * validation_fraction)) if records else 0
    validation = records[:validation_count]
    train = records[validation_count:]
    for row in train:
        row["split"] = "train"
    for row in validation:
        row["split"] = "validation"
    return train, validation


def summarize(records: list[dict[str, Any]], train: list[dict[str, Any]], validation: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_family = Counter(row["family"] for row in records)
    by_tool_decision = Counter(row["target"]["route"]["tool_decision"] for row in records)
    by_action = Counter(action for row in records for action in row["target"]["route"]["external_action_type"])
    by_operation = Counter(row["target"]["route"]["operation"] for row in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "seed": seed,
        "total_records": len(records),
        "train_records": len(train),
        "validation_records": len(validation),
        "family_count": len(by_family),
        "by_family": dict(sorted(by_family.items())),
        "by_tool_decision": dict(sorted(by_tool_decision.items())),
        "by_external_action_type": dict(sorted(by_action.items())),
        "operation_count": len(by_operation),
        "by_operation_top": dict(by_operation.most_common(30)),
        "format": {
            "records": "full records with target and messages",
            "sft": "chat-style JSONL rows with messages and assistant target JSON",
        },
    }


def generate_dataset(count: int, out_dir: Path, validation_fraction: float, seed: int) -> dict[str, Any]:
    records = build_records(count, seed)
    train, validation = split_records(records, validation_fraction)
    all_records = train + validation

    write_jsonl(out_dir / "all.records.jsonl", all_records)
    write_jsonl(out_dir / "train.records.jsonl", train)
    write_jsonl(out_dir / "validation.records.jsonl", validation)
    write_jsonl(out_dir / "train.sft.jsonl", [sft_row(row) for row in train])
    write_jsonl(out_dir / "validation.sft.jsonl", [sft_row(row) for row in validation])

    summary = summarize(all_records, train, validation, seed)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50000, help="Total records to generate.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/gpt55_capability_mapping_sft_v3"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=55)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    summary = generate_dataset(args.count, args.out_dir, args.validation_fraction, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
