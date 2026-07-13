#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_capability_holdout import case, expected, inp


DEFAULT_OUT = Path("data/capability_planning/benchmark_adapted.jsonl")

DATASET_ALIASES = {
    "api-bank": "api_bank",
    "apibank": "api_bank",
    "bfcl-tool": "bfcl_tool",
    "bfcl_tool_action": "bfcl_tool",
    "bfcl_v4": "bfcl",
    "clinc150": "clinc",
    "clinc150_oos": "clinc",
    "code_search_net": "codesearchnet",
    "code-search-net": "codesearchnet",
    "code_searchnet": "codesearchnet",
    "super-natural-instructions": "superni",
    "super_natural_instructions": "superni",
    "supernaturalinstructions": "superni",
    "swe-bench": "swe_bench",
    "swebench": "swe_bench",
    "tau-bench": "tau_bench",
    "taubench": "tau_bench",
    "web-arena": "webarena",
    "web_arena": "webarena",
}

PROFILE_BY_DATASET = {
    "agentbench": "agent",
    "api_bank": "tool_external",
    "bfcl": "tool_code",
    "clinc": "intent",
    "codesearchnet": "tool_code",
    "conala": "tool_code",
    "massive": "intent",
    "repobench": "repo_code",
    "superni": "source_task",
    "swe_bench": "repo_code",
    "tau_bench": "agent",
    "toolbench": "tool_external",
    "bfcl_tool": "tool_external",
    "tool_action": "tool_external",
    "toolsandbox": "tool_external",
    "vakra": "tool_external",
    "webarena": "web",
}

TEXT_KEYS = [
    "request",
    "task",
    "user_request",
    "prompt",
    "query",
    "question",
    "utterance",
    "utt",
    "text",
    "instruction",
    "problem_statement",
    "command",
    "goal",
    "intent",
]

SOURCE_KEYS = [
    "source_text",
    "source",
    "input",
    "passage",
    "document",
    "context",
    "article",
    "content",
]

ID_KEYS = [
    "id",
    "uid",
    "qid",
    "question_id",
    "instance_id",
    "task_id",
    "example_id",
]


def canonical_dataset(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if normalized == "auto":
        return "generic"
    return DATASET_ALIASES.get(normalized, normalized or "generic")


def dataset_profile(dataset: str) -> str:
    return PROFILE_BY_DATASET.get(canonical_dataset(dataset), "generic")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.extend(flatten_json(json.loads(line)))
        return records
    return flatten_json(json.loads(path.read_text(encoding="utf-8")))


def flatten_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(flatten_json(item))
        return rows

    if not isinstance(value, dict):
        return [{"text": str(value)}]

    if isinstance(value.get("Instances"), list):
        definition = first_text(value, ["Definition", "definition"])
        categories = value.get("Categories") or value.get("categories") or []
        task_name = first_text(value, ["Task", "Name", "task_name", "name"])
        rows = []
        for instance in value["Instances"]:
            row = dict(instance) if isinstance(instance, dict) else {"input": str(instance)}
            row.setdefault("_superni_definition", definition)
            row.setdefault("_superni_categories", categories)
            row.setdefault("_superni_task_name", task_name)
            rows.append(row)
        return rows

    nested_keys = ["data", "examples", "records", "rows", "train", "validation", "dev", "test"]
    rows = []
    for key in nested_keys:
        nested = value.get(key)
        if isinstance(nested, list):
            rows.extend(flatten_json(nested))
        elif isinstance(nested, dict):
            rows.extend(flatten_json(nested))
    if rows:
        return rows

    return [value]


def adapt_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    dataset: str,
    id_prefix: str | None = None,
    source_wrapper: str = "auto",
) -> list[dict[str, Any]]:
    canonical = canonical_dataset(dataset)
    rows = []
    for index, record in enumerate(raw_records):
        record_dataset = infer_dataset_for_record(record, canonical)
        adapted = adapt_record(
            record,
            dataset=record_dataset,
            index=index,
            id_prefix=id_prefix or canonical,
            source_wrapper=source_wrapper,
        )
        if adapted is not None:
            rows.append(adapted)
    return rows


def adapt_record(
    record: dict[str, Any],
    *,
    dataset: str,
    index: int,
    id_prefix: str,
    source_wrapper: str,
) -> dict[str, Any] | None:
    if "request" in record and "expected" in record:
        row = dict(record)
        row.setdefault("source_dataset", dataset)
        row.setdefault("benchmark_adapter", "passthrough_gold_schema")
        return row

    profile = dataset_profile(dataset)
    row_id = f"{slug(id_prefix)}_{slug(record_id(record, index))}"

    if profile == "source_task":
        request, operation, source_text = build_source_task_request(record, source_wrapper)
        if not request:
            return None
        category, exp = expected_for_source_task(operation, source_text)
        adapted = case(row_id, category, request, exp)
        adapted["wrapped_source_text"] = source_text
        adapted["operation"] = operation
    else:
        request = extract_request_text(record)
        if not request:
            return None
        category, exp, attachments = expected_for_request(request, dataset=dataset, profile=profile)
        adapted = case(row_id, category, request, exp, attachments)

    adapted["source_dataset"] = dataset
    adapted["benchmark_profile"] = profile
    adapted["raw_label"] = first_text(record, ["label", "intent", "category", "domain", "task_name"])
    adapted["source_record_id"] = record_id(record, index)
    return adapted


def build_source_task_request(
    record: dict[str, Any],
    source_wrapper: str,
) -> tuple[str, str, str]:
    source_text = first_text(record, SOURCE_KEYS)
    instruction = first_text(record, ["instruction", "definition", "_superni_definition", "task"])
    operation = infer_operation(" ".join([source_wrapper, instruction, _json_text(record)]))
    if source_wrapper != "auto":
        operation = source_wrapper
    if not source_text:
        source_text = first_text(record, TEXT_KEYS)
    if not source_text:
        return "", operation, ""
    return wrap_source_request(operation, source_text), operation, source_text


def wrap_source_request(operation: str, source_text: str) -> str:
    wrappers = {
        "classify": "Classify this text: {source}",
        "compare": "Compare the alternatives in this text: {source}",
        "extract": "Extract the requested information from this text: {source}",
        "explain": "Explain this text: {source}",
        "format": "Turn this text into a structured format: {source}",
        "rewrite": "Rewrite this text to be clearer: {source}",
        "summarize": "Summarize this text: {source}",
        "translate": "Translate this text into English: {source}",
    }
    template = wrappers.get(operation, "Analyze this text: {source}")
    return template.format(source=source_text)


def expected_for_source_task(operation: str, source_text: str) -> tuple[str, dict[str, Any]]:
    cap_by_operation = {
        "classify": "classify_provided_text",
        "compare": "compare_texts",
        "extract": "analyze_provided_text",
        "explain": "analyze_provided_text",
        "format": "transform_text_format",
        "rewrite": "revise_text_for_clarity",
        "summarize": "summarize_document",
        "translate": "translate_text",
    }
    category_by_operation = {
        "classify": "pasted_text_analysis",
        "compare": "pasted_text_analysis",
        "extract": "pasted_text_analysis",
        "explain": "pasted_text_analysis",
        "format": "pasted_text_transform",
        "rewrite": "pasted_text_transform",
        "summarize": "pasted_text_analysis",
        "translate": "pasted_text_transform",
    }
    cap = cap_by_operation.get(operation, "analyze_provided_text")
    category = category_by_operation.get(operation, "pasted_text_analysis")
    exclude_caps = ["generate_code"] if _looks_like_code_generation(source_text) else None
    return (
        category,
        expected(
            inputs=[inp(source_keyword(source_text), True, "pasted_text")],
            current=False,
            include_actions=["none"],
            exclude_actions=["web_search", "file_reading"],
            include_caps=[cap],
            exclude_caps=exclude_caps,
        ),
    )


def expected_for_request(
    request: str,
    *,
    dataset: str,
    profile: str,
) -> tuple[str, dict[str, Any], list[dict[str, Any]] | None]:
    lowered = request.lower()
    urls = extract_urls(request)

    if profile == "tool_external" and not urls:
        return (
            "tool_external",
            expected(
                inputs=[inp("tool action request", True, "none")],
                current=False,
                include_actions=["other"],
                include_caps=["select_and_execute_external_action"],
            ),
            None,
        )

    if profile == "web" or urls:
        compare_requested = _has_any(lowered, ["compare", "contrast", "differences", "difference between"])
        cap = (
            "compare_texts"
            if compare_requested or (profile != "web" and len(urls) > 1)
            else "retrieve_external_information"
        )
        precede = [{"before": "retrieve", "after": "compare"}] if cap == "compare_texts" else None
        return (
            "url",
            expected(
                inputs=[inp(urls[0] if urls else "web page", True, "url")],
                current=True,
                include_actions=["web_search"],
                include_caps=[cap],
                precede=precede,
            ),
            None,
        )

    if profile == "repo_code" or _looks_like_repo_search(lowered) or _looks_like_code_edit(lowered):
        attachments = [{"name": "repository files", "format": "file_path", "available": True}]
        if _looks_like_repo_search(lowered):
            return (
                "file_reading",
                expected(
                    inputs=[inp("repository", True, "file_path")],
                    current=False,
                    include_actions=["file_reading"],
                    include_caps=["search_provided_files"],
                ),
                attachments,
            )
        actions = ["file_reading", "file_writing"]
        caps = ["inspect_existing_code", "modify_code"]
        if _looks_like_code_validation_request(lowered):
            actions.append("code_execution")
            caps.append("execute_code")
        file_hint = first_file_path(request) or "repository"
        attachments = [{"name": file_hint, "format": "file_path", "available": True}]
        return (
            "code_file_read_write",
            expected(
                inputs=[inp(file_hint, True, "file_path")],
                current=False,
                include_actions=actions,
                include_caps=caps,
            ),
            attachments,
        )

    if profile == "tool_code" or _looks_like_code_generation(lowered):
        return (
            "generic_coding_parameters",
            expected(
                inputs=[inp(code_input_keyword(request), True, "none")],
                current=False,
                include_actions=["none"],
                include_caps=["generate_code"],
            ),
            None,
        )

    if _looks_like_missing_source(lowered):
        missing = missing_source_name(lowered)
        return (
            "missing_input",
            expected(
                inputs=[inp(missing, False, "unknown")],
                missing=[missing.split()[0]],
                current=False,
                include_actions=["user_input"],
                include_caps=["request_missing_input"],
            ),
            None,
        )

    if _looks_like_fact_check(lowered):
        return (
            "fact_checking",
            expected(
                inputs=[inp(fact_keyword(request), True, "none")],
                current=True,
                include_actions=["web_search", "fact_checking"],
                include_caps=["verify_current_information"],
            ),
            None,
        )

    if _looks_like_current_info(lowered):
        return (
            "current_facts",
            expected(
                inputs=[inp(current_info_keyword(request), True, "none")],
                current=True,
                include_actions=["web_search"],
                include_caps=["retrieve_current_information"],
            ),
            None,
        )

    if _looks_like_inline_structured_data(request):
        action = "calculation" if _looks_like_calculation(lowered) else "none"
        return (
            "structured_data",
            expected(
                inputs=[inp("data", True, "structured_data")],
                current=False,
                include_actions=[action],
                include_caps=["analyze_provided_dataset"],
            ),
            None,
        )

    if _looks_like_calculation(lowered):
        return (
            "calculation",
            expected(
                inputs=[inp("numeric", True, "none")],
                current=False,
                include_actions=["calculation"],
                include_caps=["compute_numeric_result"],
            ),
            None,
        )

    operation = infer_operation(lowered)
    if operation == "translate" and not _looks_like_source_transform(lowered):
        return (
            "translation",
            expected(
                current=False,
                include_actions=["none"],
                include_caps=["translate_text"],
            ),
            None,
        )

    if operation in {"summarize", "classify", "extract", "rewrite", "translate", "format", "compare"} and _looks_like_source_transform(lowered):
        category, exp = expected_for_source_task(operation, request)
        return category, exp, None

    if _has_any(
        lowered,
        [
            "draft",
            "write",
            "compose",
            "email",
            "reply to",
            "reply thank",
            "drop a message",
            "drop a note",
            "send a message",
        ],
    ):
        return (
            "text_generation",
            expected(
                inputs=[inp("writing requirements", True, "none")],
                current=False,
                include_actions=["none"],
                include_caps=["draft_text"],
            ),
            None,
        )

    return (
        "general_qa",
        expected(
            current=False,
            include_actions=["none"],
            include_caps=["provide_explanation"],
        ),
        None,
    )


def infer_operation(text: str) -> str:
    lowered = text.lower()
    checks = [
        ("summarize", [r"\bsummarize\b", r"\bsummary\b", r"\bsummarization\b"]),
        ("classify", [r"\bclassify\b", r"\bcategorize\b", r"\blabel\b", r"\bsentiment\b"]),
        (
            "translate",
            [
                r"\btranslate\b",
                r"\btranslation\b",
                r"\bhow do you say\b",
                r"\bhow do i say\b",
                r"\bhow do they say\b",
                r"\bhow does one say\b",
                r"\bhow might i say\b",
                r"\bhow would i say\b",
                r"\bhow would one say\b",
                r"\bhow would they say\b",
                r"\bi must know how to say\b",
                r"\bright way to say\b",
                r"\bword for\b",
            ],
        ),
        ("rewrite", [r"\brewrite\b", r"\brephrase\b", r"\bparaphrase\b", r"\bpolish\b", r"\bimprove\b"]),
        ("extract", [r"\bextract\b", r"\bpull\b", r"\bidentify\b", r"\blist\b"]),
        ("compare", [r"\bcompare\b", r"\bdifference", r"\bwhich .* clearer\b"]),
        ("format", [r"\bformat\b", r"\bmarkdown table\b", r"\btable\b"]),
        ("explain", [r"\bdefine\b", r"\bdefinition of\b", r"\bhow to solve\b", r"\bmeaning of\b", r"\bexplain\b", r"\bwhy\b"]),
        ("calculate", [r"\bcalculate\b", r"\bcompute\b"]),
    ]
    for operation, patterns in checks:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return operation
    return "analyze"


def extract_request_text(record: dict[str, Any]) -> str:
    messages = record.get("messages") or record.get("conversation") or record.get("user_messages")
    if isinstance(messages, list):
        parts = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("speaker") or "user").lower()
            if role in {"user", "human", "customer"}:
                parts.append(str(message.get("content") or message.get("text") or message.get("utterance") or ""))
        if parts:
            return "\n\n".join(part for part in parts if part).strip()
    return first_text(record, TEXT_KEYS)


def first_text(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        text = value_to_text(value)
        if text:
            return text
    return ""


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [value_to_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in TEXT_KEYS + SOURCE_KEYS:
            if key in value:
                text = value_to_text(value[key])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def record_id(record: dict[str, Any], index: int) -> str:
    for key in ID_KEYS:
        text = value_to_text(record.get(key))
        if text:
            return text
    return f"{index + 1:04d}"


def source_keyword(source_text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", source_text)
    if not words:
        return "text"
    return " ".join(words[: min(2, len(words))])


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(").,;]\"'") for url in re.findall(r"https?://[^\s)>,]+", text)]


def first_file_path(text: str) -> str:
    match = re.search(r"\b[\w./-]+\.(?:py|js|ts|tsx|jsx|json|md|yml|yaml|toml|java|go|rs)\b", text)
    return match.group(0) if match else ""


def code_input_keyword(request: str) -> str:
    lowered = request.lower()
    for keyword in ["postgres", "sql", "graphql", "python", "bash", "regex", "function", "script"]:
        if keyword in lowered:
            return keyword
    return "code"


def current_info_keyword(request: str) -> str:
    lowered = request.lower()
    for keyword in ["exchange", "weather", "ceo", "stock", "price", "rate", "model", "policy"]:
        if keyword in lowered:
            return keyword
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", request)
    return " ".join(words[:2]) if words else "current information"


def fact_keyword(request: str) -> str:
    lowered = request.lower()
    for keyword in ["policy", "ordinance", "law", "claim", "refund"]:
        if keyword in lowered:
            return keyword
    return "claim"


def missing_source_name(lowered: str) -> str:
    if _looks_like_reply_to_missing_email_source(lowered):
        return "email content"
    if "revenue" in lowered:
        return "revenue data"
    names = [
        "resume",
        "essay",
        "article",
        "policy memo",
        "memo",
        "project plan",
        "reviews",
        "spreadsheet",
        "signups",
        "dataset",
        "data",
        "screenshot",
        "image",
        "notes",
        "transcript",
        "proposal",
        "bio",
    ]
    for name in names:
        if name in lowered:
            if name == "signups":
                return "signup data"
            return name
    return "source material"


def _looks_like_missing_source(lowered: str) -> bool:
    if _looks_like_reply_to_missing_email_source(lowered):
        return True
    if _has_any(lowered, ["attached", "this ", "following", "below"]):
        return False
    if re.search(r"\b(?:graph|chart|plot)\b", lowered) and _has_any(
        lowered, ["signups", "revenue", "sales", "data", "channel"]
    ):
        return True
    verbs = [
        "summarize",
        "improve",
        "rewrite",
        "polish",
        "classify",
        "analyze",
        "extract",
        "graph",
        "chart",
        "plot",
    ]
    nouns = [
        "linkedin bio",
        "my resume",
        "my essay",
        "my article",
        "my memo",
        "my policy memo",
        "my project plan",
        "my reviews",
        "my spreadsheet",
        "my dataset",
        "my data",
        "the data",
        "the spreadsheet",
        "the screenshot",
        "the image",
        "my notes",
        "the transcript",
        "my proposal",
        "my bio",
    ]
    return _has_any(lowered, verbs) and _has_any(lowered, nouns)


def _looks_like_fact_check(lowered: str) -> bool:
    return _has_any(lowered, ["fact-check", "fact check", "verify whether", "verify if"])


def _looks_like_current_info(lowered: str) -> bool:
    if _looks_like_weather_lookup(lowered):
        return True
    if _looks_like_currency_exchange_request(lowered):
        return True
    if _looks_like_stock_or_market_lookup(lowered):
        return True
    if _has_any(lowered, ["most read stories", "today's most read", "top stories", "current headlines", "headlines"]):
        return True
    if re.search(r"\bwhat(?:'s|s| is| were)?\s+happening\s+with\b", lowered):
        return True
    if re.search(r"(?<![a-z0-9_])news(?![a-z0-9_])", lowered):
        return True
    current_terms = ["latest", "current", "today", "tomorrow", "this month", "now"]
    current_domains = [
        "weather",
        "exchange rate",
        "stock",
        "price",
        "ceo",
        "model",
        "federal funds rate",
        "mortgage",
        "policy",
    ]
    questionish = lowered.startswith(("what ", "who ", "when ", "where ", "will ", "is ", "are ", "did ", "does "))
    return (_has_any(lowered, current_terms) and (questionish or _has_any(lowered, current_domains)))


def _looks_like_inline_structured_data(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"```(?:csv|tsv|json)?", stripped, re.I):
        return True
    if re.search(r"\b(csv|tsv|json)\b", stripped, re.I) and any(delim in stripped for delim in [",", "\t", "|", ";", "{", "["]):
        return True
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 3 and any(delim in lines[0] for delim in [",", "\t", "|", ";"]):
        return True
    return False


def _looks_like_calculation(lowered: str) -> bool:
    if re.search(r"\bprime numbers?\b.*\bbetween\b", lowered):
        return True
    if not re.search(r"\d", lowered) and not _has_any(
        lowered,
        ["column", "csv", "data", "dataset", "revenue", "rows", "sales", "table"],
    ):
        return False
    return any(
        re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", lowered)
        for term in [
            "area",
            "calculate",
            "coefficients",
            "compute",
            "equation",
            "growth",
            "integral",
            "median",
            "perimeter",
            "quadratic",
            "roots",
            "slope",
            "sum",
            "total",
            "average",
        ]
    )


def _looks_like_weather_lookup(lowered: str) -> bool:
    if _has_any(lowered, ["weather", "rain", "raining", "rainy", "snowing", "windy", "forecast", "advisories"]):
        return True
    if _has_any(lowered, ["high and low", "highs and lows"]):
        return True
    if "umbrella" in lowered and _has_any(lowered, ["today", "tomorrow", "outside", "bring"]):
        return True
    if _has_any(lowered, ["cold outside", "temperature outside"]):
        return True
    if "how cold" in lowered and _has_any(lowered, ["outside", "today", "tomorrow", "this week", "week"]):
        return True
    return False


def _looks_like_reply_to_missing_email_source(lowered: str) -> bool:
    return bool(
        re.search(r"\breply\b", lowered)
        and re.search(r"\b(?:email|emails|message|messages)\b", lowered)
        and re.search(r"\b(?:latest|recent|previous|existing|thread)\b", lowered)
    )


def _looks_like_currency_exchange_request(lowered: str) -> bool:
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
    if _has_any(lowered, ["exchange rate", "rate of exchange"]):
        return True
    return _has_any(lowered, currency_terms) and _has_any(
        lowered, ["convert", "worth", "how many", "how much", "equals", " to "]
    )


def _looks_like_stock_or_market_lookup(lowered: str) -> bool:
    if _has_any(lowered, ["share value", "share price"]):
        return True
    return _has_any(lowered, ["stock", "dow", "market"]) and _has_any(
        lowered,
        [
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
        ],
    )


def _looks_like_source_transform(lowered: str) -> bool:
    return bool(
        re.search(r"\b(?:following|below|provided)\b", lowered)
        or ":" in lowered
        or "\n" in lowered
    )


def _looks_like_repo_search(lowered: str) -> bool:
    return _has_any(lowered, ["search the codebase", "find usages", "find todos", "in the repo", "codebase for"])


def _looks_like_code_edit(lowered: str) -> bool:
    return bool(first_file_path(lowered)) and _has_any(lowered, ["fix", "update", "modify", "change", "refactor"])


def _looks_like_code_validation_request(lowered: str) -> bool:
    return bool(
        re.search(r"\b(?:run|update|add|fix)\s+(?:the\s+)?tests?\b", lowered)
        or re.search(r"\bfailing\s+tests?\b", lowered)
        or re.search(r"\bpytest\b", lowered)
    )


def _looks_like_code_generation(text: str) -> bool:
    lowered = text.lower()
    code_terms = [
        "write a postgres query",
        "write a sql",
        "write an sql",
        "graphql query",
        "python function",
        "bash script",
        "regex",
        "regular expression",
        "function ",
        "class ",
    ]
    return _has_any(lowered, code_terms) or bool(re.search(r"\w+\([^)]*\)", text))


def _looks_like_structured_generation(lowered: str) -> bool:
    return _has_any(lowered, ["write", "draft", "compose"])


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return value[:80] or "row"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt public benchmark exports into this repo's capability-planning gold schema. "
            "The adapter labels planner properties, not full task answers."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--dataset",
        default="auto",
        help=(
            "Benchmark family, e.g. clinc, massive, superni, bfcl, api_bank, "
            "toolbench, swe_bench, repobench, webarena. Use auto to infer from filename."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id-prefix")
    parser.add_argument(
        "--source-wrapper",
        choices=["auto", "summarize", "classify", "extract", "rewrite", "translate", "compare", "format", "explain"],
        default="auto",
        help="For source-task datasets, wrap source text in this operation.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "auto":
        dataset = infer_dataset_from_path(args.input)
    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    rows = adapt_records(
        records,
        dataset=dataset,
        id_prefix=args.id_prefix,
        source_wrapper=args.source_wrapper,
    )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "dataset": canonical_dataset(dataset),
                "raw_records": len(records),
                "adapted_rows": len(rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def infer_dataset_from_path(path: Path) -> str:
    lowered = path.stem.lower()
    for key in sorted(set(DATASET_ALIASES) | set(PROFILE_BY_DATASET)):
        if key.replace("_", "-") in lowered or key in lowered:
            return canonical_dataset(key)
    return "generic"


def infer_dataset_for_record(record: dict[str, Any], fallback: str) -> str:
    if fallback != "generic":
        return fallback
    if "_superni_definition" in record or "_superni_categories" in record:
        return "superni"
    if "problem_statement" in record:
        return "swe_bench"
    if "repo" in record and "prompt" in record:
        return "repobench"
    if "site" in record or "start_url" in record:
        return "webarena"
    if "question" in record and _looks_like_code_generation(value_to_text(record.get("question"))):
        return "bfcl"
    if "query" in record and extract_urls(value_to_text(record.get("query"))):
        return "api_bank"
    if "utterance" in record:
        return "clinc"
    if "utt" in record:
        return "massive"
    return fallback


if __name__ == "__main__":
    main()
