#!/usr/bin/env python3
import argparse
import calendar
import datetime
import json
import os
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path


READINESS_POLICY = """Grounding and readiness policy:
- Before emitting a call, verify that every required argument is grounded in the original user request or an actual prior tool result.
- Do not fill required arguments with symbolic placeholders such as x, y, a, b, c, v, t, or theta; empty strings; guessed zeros or ones; or common defaults unless the user explicitly provided them or the schema makes the argument optional.
- Do not call a tool merely because it is topically related to the request.
- Emit no tool call when the request is conceptual or explanatory, no listed tool is directly applicable, or any required argument is not explicitly grounded."""


TOOL_SYSTEM_PROMPT = """You are a careful, general-purpose tool-calling assistant. Use your own judgment to decide whether a tool call is needed.

Follow these principles:
1. Assume the available tools have already been implemented by the user or runtime.
2. Assume the user or runtime will execute tool calls. Do not ask the user to call a tool and report back; provide the tool name and arguments when a tool call is appropriate.
3. Never call the same tool twice with identical arguments. Do not repeat tool calls.
4. If none of the available tools are relevant to the user's request, do not make an unnecessary tool call.
5. Do not assume access to tools that are not listed in the available tool schema. Do not invent tools, tool names, parameters, or a code interpreter.
6. Use only information provided by the user, prior tool results, and the available tool schemas. If required arguments are missing, ask a concise clarification instead of guessing.
7. When a tool result is provided in a later user message, incorporate that result into the answer if it resolves the user's request.

When explaining tool-use decisions in text-only settings, place concise reasoning inside <thinking> tags. When emitting manual tool calls in text-only settings, use:
<tool_call>[{"name":"tool_name","arguments":{}}]</tool_call>

The content inside <tool_call> must be valid JSON and must be a list of objects with "name" and "arguments" keys. In native tool-calling APIs, use the platform's tool-call mechanism instead of printing manual XML tags.""" + "\n\n" + READINESS_POLICY


STRICT_VALUE_COPY_INSTRUCTION = """Strict value-copy mode for tool arguments:
- Treat the original user request as the source of truth for argument values.
- For string-like arguments such as formulas, mathematical expressions, names, titles, locations, dates, IDs, categories, units, enum-like labels, and free-text snippets, copy the shortest exact span from the user request whenever possible.
- Do not rewrite formulas or expressions. Preserve written forms such as 3x**2, C6H12O6, x=4, 5%, dates, IDs, and units exactly unless the schema requires a primitive numeric/boolean conversion.
- Do not add descriptive adjectives or surrounding context to copied values. For example, copy blue, not vibrant blue; copy New York, not New York Marriott hotel.
- Do not singularize, pluralize, abbreviate, expand, translate, or canonicalize explicit strings unless the user explicitly used that form.
- For array/list arguments, include all and only the list items assigned to this subtask; do not split one list item per call when the schema expects a list.
- For optional arguments, include them only when their value is explicitly grounded in the request or clearly implied by the exact assigned subtask. Do not invent defaults."""


def strict_value_copy_instruction(args):
    return (
        STRICT_VALUE_COPY_INSTRUCTION + "\n\n"
        if getattr(args, "strict_value_copy", False)
        else ""
    )


VALUE_COPY_FEWSHOT_INSTRUCTION = """Selector examples for exact argument values.
These are illustrative examples only; do not call these example tools unless they are in the current schema.

Example A: copy formulas exactly.
Request span: "Calculate the derivative of 3x**2 at x=2."
Schema expects function:string and value:number.
Correct arguments: {"function": "3x**2", "value": 2}
Wrong arguments: {"function": "3*x**2", "value": 2}

Example B: do not embellish explicit strings.
Request span: "Identify a small blue bird in a forest."
Schema expects color:string, size:string, habitat:string.
Correct arguments: {"color": "blue", "size": "small", "habitat": "forest"}
Wrong arguments: {"color": "vibrant blue", "size": "small", "habitat": "forest"}

Example C: keep location/name spans short.
Request span: "Book the Marriott hotel in New York."
Schema expects hotel_name:string and location:string.
Correct arguments: {"hotel_name": "Marriott", "location": "New York"}
Wrong arguments: {"hotel_name": "Marriott hotel", "location": "New York, Marriott hotel"}

Example D: convert percentages only when the schema expects a number.
Request span: "Use an interest rate of 5%."
If schema expects number, correct argument: {"interest_rate": 0.05}
If schema expects string, correct argument: {"interest_rate": "5%"}
Wrong numeric argument: {"interest_rate": 5}

Example E: array arguments should contain the full requested list assigned to the subtask.
Request span: "Fetch Personal Info, Job History, Payroll, and Attendance for employee 12345."
Schema expects data_field:array.
Correct arguments: {"employee_id": 12345, "data_field": ["Personal Info", "Job History", "Payroll", "Attendance"]}
Wrong arguments: {"employee_id": 12345, "data_field": ["Personal Info"]}

Example F: omit optional defaults that are not stated.
Request span: "Find the waiting time for the Louvre."
Schema has optional day:string.
Correct arguments: {"museum_name": "Louvre"}
Wrong arguments: {"museum_name": "Louvre", "day": "Sunday"}"""


SPAN_INVENTORY_INSTRUCTION = """Grounded span inventory from the original user request.
This inventory is advisory, not exhaustive. Prefer these exact prompt spans for
tool argument values when they fit the schema. Do not invent values outside the
inventory unless the schema requires a primitive conversion or the value is
clearly implied by the request."""


SPAN_INVENTORY_SECTIONS = (
    ("quoted_strings", "quoted strings"),
    ("formulas", "formulas/expressions"),
    ("dates", "dates/times/ranges"),
    ("numbers", "numbers/percentages"),
    ("ids", "IDs/codes"),
    ("entities", "names/titles/entities"),
    ("list_items", "list/coordinated items"),
)


def clean_span(value, max_len=80):
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:()[]{}")
    text = re.sub(
        r"^(?:fetch|get|find|list|retrieve|calculate|compute|check|compare|show|tell me|give me)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(?:in|for|with|and|or|of|the)$", "", text, flags=re.IGNORECASE)
    if not text:
        return ""
    if len(text) > max_len:
        return ""
    return text


def add_inventory_span(inventory, section, value, max_items=24):
    span = clean_span(value)
    if not span:
        return
    if len(span) == 1 and not span.isdigit():
        return
    normalized = " ".join(span.lower().split())
    existing = {" ".join(item.lower().split()) for item in inventory[section]}
    if normalized not in existing and len(inventory[section]) < max_items:
        inventory[section].append(span)


def split_list_candidate(value):
    text = clean_span(value, max_len=140)
    if not text:
        return []
    text = re.sub(r"\s+(?:and|or)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [clean_span(part, max_len=60) for part in text.split(",")]
    stop = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "for",
        "with",
        "in",
        "on",
        "of",
    }
    return [
        part
        for part in parts
        if part and part.lower() not in stop and len(part) <= 60
    ]


def extract_span_inventory(prompt):
    inventory = {key: [] for key, _ in SPAN_INVENTORY_SECTIONS}
    text = prompt or ""

    for match in re.finditer(r"[\"“”`]([^\"“”`]{2,80})[\"“”`]", text):
        add_inventory_span(inventory, "quoted_strings", match.group(1))

    date_patterns = [
        r"\b(?:19|20)\d{2}\s*[-/]\s*(?:(?:19|20)?\d{2})\b",
        r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_inventory_span(inventory, "dates", match.group(0))

    for match in re.finditer(r"\b[$€£]?\d+(?:,\d{3})*(?:\.\d+)?%?\b", text):
        add_inventory_span(inventory, "numbers", match.group(0))

    operand = r"(?:[A-Za-z]\w*|\d+(?:\.\d+)?[A-Za-z]\w*|\d+(?:\.\d+)?)"
    expression_pattern = (
        r"\b" + operand + r"(?:\s*(?:\*\*|[+\-*/=^])\s*" + operand + r")+"
    )
    for match in re.finditer(expression_pattern, text):
        add_inventory_span(inventory, "formulas", match.group(0))
    for match in re.finditer(r"\b(?:[A-Z][a-z]?\d*){2,}\b", text):
        span = match.group(0)
        if any(ch.isdigit() for ch in span):
            add_inventory_span(inventory, "formulas", span)

    for match in re.finditer(r"\b[A-Z]{1,}[A-Z0-9]*[-_]?\d+[A-Z0-9_-]*\b", text):
        add_inventory_span(inventory, "ids", match.group(0))

    entity_pattern = (
        r"\b[A-Z][A-Za-z0-9&'-]+"
        r"(?:\s+(?:[A-Z][A-Za-z0-9&'-]+|of|and|the|for|in)){1,5}"
    )
    for match in re.finditer(entity_pattern, text):
        span = clean_span(match.group(0))
        if (
            span
            and span.lower() not in {"the", "a", "an", "and", "or", "for", "in", "of"}
            and not re.match(r"^(The|This|That|Please|Can|Could|What|When|Where|Why|How)\b", span)
        ):
            add_inventory_span(inventory, "entities", span)
            for part in re.split(r"\b(?:for|in|of|with|and)\b", span):
                subspan = clean_span(part)
                if (
                    subspan
                    and subspan != span
                    and subspan.lower()
                    not in {"the", "a", "an", "and", "or", "for", "in", "of"}
                    and re.search(r"\b[A-Z][A-Za-z0-9&'-]+", subspan)
                ):
                    add_inventory_span(inventory, "entities", subspan)

    list_patterns = [
        r"\b[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+)*"
        r"(?:\s*,\s*(?:and\s+)?[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+)*){1,}",
        r"\b[a-z][a-z0-9&.'-]+(?:\s+[a-z][a-z0-9&.'-]+){0,2}"
        r"(?:\s*,\s*(?:and\s+)?[a-z][a-z0-9&.'-]+(?:\s+[a-z][a-z0-9&.'-]+){0,2}){2,}",
    ]
    for pattern in list_patterns:
        for match in re.finditer(pattern, text):
            for item in split_list_candidate(match.group(0)):
                add_inventory_span(inventory, "list_items", item)

    return inventory


def format_span_inventory(prompt):
    inventory = extract_span_inventory(prompt)
    lines = []  # type: ignore[var-annotated]
    for key, label in SPAN_INVENTORY_SECTIONS:
        values = inventory.get(key) or []
        if values:
            lines.append("- %s: %s" % (label, "; ".join(values)))
    if not lines:
        return ""
    return SPAN_INVENTORY_INSTRUCTION + "\n" + "\n".join(lines)


def selector_value_guidance(args, prompt=None):
    blocks = []
    if getattr(args, "strict_value_copy", False) or getattr(
        args,
        "value_copy_fewshot",
        False,
    ):
        blocks.append(STRICT_VALUE_COPY_INSTRUCTION)
    if getattr(args, "value_copy_fewshot", False):
        blocks.append(VALUE_COPY_FEWSHOT_INSTRUCTION)
    if getattr(args, "span_inventory", False):
        inventory_text = format_span_inventory(prompt or "")
        if inventory_text:
            blocks.append(inventory_text)
    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


REACT_SYSTEM_PROMPT = """You are a careful, general-purpose tool-calling assistant using a ReAct-style single-pass decision process.

First reason internally about the user's request, the needed subgoals, and which available tools are relevant. Then emit the final tool calls using the native tool-calling API.

Important constraints:
1. This is a no-execution setting: do not wait for observations and do not claim that tools have been run.
2. Assume the user or runtime will execute emitted tool calls. Do not ask the user to call a tool and report back.
3. Never call the same tool twice with identical arguments. Do not repeat tool calls.
4. If no available tool is relevant, do not emit a tool call.
5. Use only tools and parameters in the available tool schemas. Do not invent tools, tool names, parameters, observations, or a code interpreter.
6. Use the original request and full tool schemas for exact argument names and values. If required arguments are missing, ask a concise clarification instead of guessing.

In native tool-calling APIs, use the platform's tool-call mechanism. Do not print manual XML tool-call tags.""" + "\n\n" + READINESS_POLICY


DECOMPOSITION_SYSTEM_PROMPT = """Classify the user's tool-use intent, then produce a compact call-set only when multiple distinct calls are needed.

The first output line must be exactly one of:
verdict=no_intent
verdict=single_intent
verdict=multi_intent

Definitions:
- no_intent: no listed tool is directly needed or relevant.
- single_intent: exactly one tool call is needed.
- multi_intent: two or more distinct tool calls are needed.

For no_intent or single_intent, output only the verdict line.

For multi_intent, output the verdict followed by at most 4 distinct call-set lines:
1. tool=<candidate_tool>; argument_group=<plain-language facts identifying this distinct call>; objective=<specific objective>

Each line represents one distinct (tool, argument_group) pair. Use a tool once unless the user clearly requests separate calls with genuinely different argument groups. Deduplicate equivalent calls before output.

Plan only the final independent calls directly requested by the user:
- Preserve each requested operation as written. Do not replace it with a related tool or reinterpret it as a request about another entity.
- Do not add calls for intermediate calculations, derived expressions, follow-up transformations, or imagined tool results.
- Do not split one requested call into a chain. A phrase such as "evaluate the derivative at x=4" is one derivative call, not separate calls to derive and then evaluate.
- Repeated requests with different grounded argument groups are separate calls, even when they use the same tool.
- Treat each explicitly requested output variant, format, condition, or option as a separate call when the tool accepts only one variant per call. Do not collapse requests for "percentage and fraction" into an invented "both" call.
- Copy every fact that distinguishes a call into its argument_group, including expressions, evaluation points, formats, boolean conditions, sorting instructions, meal/type labels, and other explicit modifiers.
- Prefer the tool whose accepted arguments directly represent the user's requested subject and operation. Do not substitute a related entity's tool; for example, a request for a player's championship record remains a player-record request.
- Every call-set line must be independently executable from facts in the original request. If it would require the output of another planned call, omit it.
- Terse, misspelled, command-like, or list-like user text can still be a tool-use request. Do not output no_intent merely because the request is short, ungrammatical, or lacks an explicit question.

Use the provided tool descriptions and compact argument hints to choose the exact tool, but do not copy parameter names into the output. Do not output executable tool calls, JSON, schemas, parameter names, key=value arguments, explanations, or extra lines. Do not choose a tool merely because it is topically related."""


CARDINALITY_REPAIR_SYSTEM_PROMPT = """Repair only the cardinality of an existing call-set plan.

The first output line must be exactly one of:
verdict=no_intent
verdict=single_intent
verdict=multi_intent

For no_intent or single_intent, output only the verdict line.

For multi_intent, output the verdict followed by at most 8 distinct call-set lines:
1. tool=<candidate_tool>; argument_group=<plain-language facts identifying this distinct call>; objective=<specific objective>

Do not emit executable tool calls, JSON, schemas, parameter names as formal arguments, explanations, or extra lines.

Your job is not to improve wording. Your job is to fix only these count/cardinality mistakes:
- If a schema field is scalar (string, number, integer, boolean), different grounded values for that field usually require separate calls.
- If a schema field is array/list, multiple grounded values for that field usually belong in one call.
- If the request asks for every combination across two or more scalar dimensions, expand to the Cartesian product.
- If the current plan split values that should be one array argument, collapse those lines.
- If the current plan combined values that belong in separate scalar calls, split them.
- If the current plan added a related but unrequested operation, remove it.
- Preserve every grounded value exactly; do not compute intermediate results and do not invent values.

If the current plan is multi_intent, keep verdict=multi_intent after repair unless the user truly requested no tool calls.
Repeated calls to the same tool with different grounded argument groups are still multi_intent.
If the repaired plan would have the same number of call-set lines as the current plan, copy the current plan exactly.
Keep the existing plan unchanged when cardinality is already correct or the request is ambiguous."""


CARDINALITY_PARALLEL_REPAIR_SYSTEM_PROMPT = """Repair only the cardinality of an existing parallel call-set plan.

This is a parallel or parallel_multiple BFCL task. The existing plan already decided that calls are needed.

The first output line must be exactly:
verdict=multi_intent

Never output verdict=no_intent or verdict=single_intent for this repair.
If no repair is needed, repeat the existing multi_intent plan unchanged.

Output the verdict followed by 2 to 8 distinct final call-set lines:
1. tool=<candidate_tool>; argument_group=<plain-language facts identifying this distinct call>; objective=<specific objective>

Do not emit executable tool calls, JSON, schemas, parameter names as formal arguments, explanations, or extra lines.

Your job is not to improve wording. Your job is to fix only these count/cardinality mistakes:
- If a schema field is scalar (string, number, integer, boolean), different grounded values for that field usually require separate calls.
- If a schema field is array/list, multiple grounded values for that field usually belong in one call.
- If the request asks for every combination across two or more scalar dimensions, expand to the Cartesian product.
- If the current plan split values that should be one array argument, collapse those lines.
- If the current plan combined values that belong in separate scalar calls, split them.
- If the current plan added a related but unrequested operation, remove it.
- Preserve every grounded value exactly; do not compute intermediate results and do not invent values.

Repeated calls to the same tool with different grounded argument groups are still multi_intent.
If the repaired plan would have the same number of call-set lines as the current plan, copy the current plan exactly.
Keep the existing plan unchanged when cardinality is already correct or the request is ambiguous."""


VERIFICATION_SYSTEM_PROMPT = """Verify candidate tool calls conservatively against the original request and the supplied full tool schemas.

Return only JSON in this exact shape:
{"drop":[{"index":0,"reason":"clearly_irrelevant"}]}

Only include a candidate in drop when it is clearly irrelevant to the user's request. Do not drop for uncertainty, style, wording, optional omitted arguments, or because another tool might also work. Missing required arguments, placeholder values, unexpected arguments, and type mismatches are checked separately by deterministic code.

If every candidate is relevant or you are uncertain, return:
{"drop":[]}"""


ABSTENTION_GUARD_SYSTEM_PROMPT = """You are a strict no-call verifier for a function-calling benchmark.

Your job is to decide whether the original user request should execute any tool at all.

Return only JSON in this exact shape:
{"decision":"keep","reason":"short reason"}

Allowed decisions:
- keep: the user is directly asking for one or more executable tool calls, and at least one candidate call is grounded by the request.
- abstain: the user is asking a conceptual, explanatory, conversational, unrelated, or merely topical question; no listed candidate tool should be executed; or executing the call would require fabricating required facts.

Important rules:
- Candidate tools may be topically related but still unnecessary. Abstain in that case.
- Do not execute a tool merely to answer a general knowledge, explanation, definition, comparison, recommendation, or chat request unless the user explicitly asks for the listed tool action.
- Keep when the request clearly asks for calculation, lookup, booking, ordering, retrieval, search, conversion, or another action represented by a candidate tool.
- Do not abstain just because optional arguments are omitted, wording is imperfect, or another listed tool might be better.
- For real executable requests, prefer keep when uncertain. For clearly non-executable or irrelevant requests, abstain."""


ABSTENTION_PREFILTER_SYSTEM_PROMPT = """You are a strict pre-selector no-call classifier for a function-calling benchmark.

Decide whether the user's request should execute at least one of the listed tools.

Return only JSON in this exact shape:
{"decision":"keep","reason":"short reason"}

Allowed decisions:
- keep: the user directly asks for an action represented by a listed tool, with enough grounded information for at least one required call.
- abstain: the user is asking a conceptual, explanatory, conversational, hypothetical, comparative, definitional, recommendation, or unrelated question; the listed tools are only topically related; or required facts would have to be guessed.

Use these rules strictly:
- A topical match is not enough. The user must request the tool action itself.
- Do not call lookup/search/calculation tools merely to answer "what is", "why", "how does", "tell me about", "compare", "recommend", or general advice questions.
- Keep executable requests such as calculate, convert, book, order, search, retrieve, fetch, find, list, update, or get specific data when the required facts are stated.
- If the request can be answered in ordinary text without executing a listed tool, abstain.
- Prefer abstain for benchmark irrelevance-style prompts. Prefer keep only when the requested operation is explicit and grounded."""


def openai_tools(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def tool_inventory(tools):
    return "\n".join(
        f"- {tool['name']}: {tool.get('description', 'No description provided.')}"
        for tool in tools
    )


def planning_tool_inventory(tools):
    lines = []
    for tool in tools:
        properties = (tool.get("parameters") or {}).get("properties") or {}
        hints = []
        for name, schema in properties.items():
            hint = name
            enum_values = schema.get("enum")
            if enum_values:
                hint += f" (allowed: {', '.join(map(str, enum_values))})"
            hints.append(hint)
        argument_hint = ", ".join(hints) if hints else "none"
        lines.append(
            f"- {tool['name']}: {tool.get('description', 'No description provided.')} "
            f"Accepted arguments: {argument_hint}."
        )
    return "\n".join(lines)


def schema_cardinality_type(schema):
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if any(item in ("array", "list", "tuple") for item in expected_type):
            return "array"
        non_null = [item for item in expected_type if item != "null"]
        return "/".join(map(str, non_null)) if non_null else "null"
    if expected_type in ("array", "list", "tuple"):
        return "array"
    if expected_type in ("object", "dict"):
        return "object"
    return str(expected_type or "any")


def cardinality_tool_inventory(tools):
    lines = []
    for tool in tools:
        properties = (tool.get("parameters") or {}).get("properties") or {}
        required = set((tool.get("parameters") or {}).get("required") or [])
        hints = []
        for name, schema in properties.items():
            cardinality = schema_cardinality_type(schema)
            required_text = "required" if name in required else "optional"
            enum_values = schema.get("enum")
            enum_text = (
                f"; allowed={', '.join(map(str, enum_values))}"
                if enum_values
                else ""
            )
            hints.append(f"{name}: {cardinality}, {required_text}{enum_text}")
        argument_hint = "; ".join(hints) if hints else "none"
        lines.append(
            f"- {tool['name']}: {tool.get('description', 'No description provided.')} "
            f"Arguments: {argument_hint}."
        )
    return "\n".join(lines)


def post_chat(endpoint, api_key, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    retry_count = int(os.environ.get("QWEN_HTTP_RETRIES", "2"))
    retry_delay = float(os.environ.get("QWEN_RETRY_DELAY", "10"))
    started = time.perf_counter()

    for attempt in range(retry_count + 1):
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_text = response.read().decode("utf-8")
            latency_ms = round((time.perf_counter() - started) * 1000)
            return json.loads(raw_text), latency_ms
        except urllib.error.HTTPError as exc:
            raw_text = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code >= 500
            if not retryable or attempt == retry_count:
                raise RuntimeError(f"HTTP {exc.code}: {raw_text}") from exc
        except urllib.error.URLError as exc:
            if attempt == retry_count:
                raise RuntimeError(f"Could not reach endpoint {endpoint}: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt == retry_count:
                raise RuntimeError(
                    f"Timed out waiting for endpoint {endpoint}. Increase QWEN_TIMEOUT for CPU-only runs."
                ) from exc

        time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"Request to {endpoint} failed after retries.")


def ollama_native_chat_endpoint(endpoint):
    return re.sub(r"/v1/chat/completions/?$", "/api/chat", endpoint)


def post_ollama_chat(endpoint, api_key, payload, timeout):
    return post_chat(
        ollama_native_chat_endpoint(endpoint),
        api_key,
        payload,
        timeout,
    )


def response_usage(raw):
    usage = raw.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def native_response_usage(raw):
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    completion_tokens = int(raw.get("eval_count") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def native_request_usage(stage, raw):
    usage = native_response_usage(raw)
    usage["stage"] = stage
    return usage


def request_usage(stage, raw):
    usage = response_usage(raw)
    usage["stage"] = stage
    return usage


def aggregate_request_usages(usages):
    return {
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usages),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in usages),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usages),
        "max_request_prompt_tokens": max(
            [item.get("prompt_tokens", 0) for item in usages] or [0]
        ),
        "max_request_total_tokens": max(
            [item.get("total_tokens", 0) for item in usages] or [0]
        ),
        "request_count": len(usages),
    }


def normalize_tool_calls(raw):
    message = (((raw.get("choices") or [{}])[0]).get("message") or {})
    calls = message.get("tool_calls") or message.get("function_call") or []
    if isinstance(calls, dict):
        calls = [calls]
    normalized = []
    for call in calls:
        fn = call.get("function") or call
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        normalized.append(
            {
                "name": fn.get("name") or call.get("name") or "unknown_function",
                "arguments": arguments,
            }
        )
    return dedupe_tool_calls(normalized)


def dedupe_tool_calls(calls):
    seen = set()
    deduped = []
    for call in calls:
        key = canonical(call)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_json_object(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def parse_json_array(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, list) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
        return value if isinstance(value, list) else None
    except json.JSONDecodeError:
        return None


def normalize_constrained_calls(value):
    if not isinstance(value, list):
        return []
    calls = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments})
    return dedupe_tool_calls(calls)


def normalize_text(value):
    return " ".join(str(value).strip().lower().split())


def values_match(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and values_match(actual[key], value) for key, value in expected.items())

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(values_match(a, e) for a, e in zip(actual, expected))

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-6

    if isinstance(expected, str) and isinstance(actual, str):
        actual_norm = normalize_text(actual)
        expected_norm = normalize_text(expected)
        return (
            actual_norm == expected_norm
            or actual_norm.startswith(f"{expected_norm},")
            or expected_norm.startswith(f"{actual_norm},")
        )

    return actual == expected


def calls_match(actual, expected):
    return actual.get("name") == expected.get("name") and values_match(
        actual.get("arguments", {}),
        expected.get("arguments", {}),
    )


def score_calls(actual, expected):
    if not expected and not actual:
        return {"exact": True, "precision": 1.0, "recall": 1.0}

    matched_expected = set()
    true_positive = 0
    for actual_call in actual:
        for index, expected_call in enumerate(expected):
            if index not in matched_expected and calls_match(actual_call, expected_call):
                matched_expected.add(index)
                true_positive += 1
                break

    precision = true_positive / len(actual) if actual else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    return {
        "exact": true_positive == len(actual) == len(expected),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


PLACEHOLDER_STRINGS = {
    "",
    "a",
    "b",
    "c",
    "x",
    "y",
    "z",
    "v",
    "t",
    "theta",
    "tbd",
    "unknown",
    "unspecified",
    "n/a",
    "none",
    "null",
}


def literal_is_grounded(value, source_text):
    if not isinstance(value, str) or not value or not source_text:
        return False
    return re.search(
        rf"(?<![\w]){re.escape(value)}(?![\w])",
        source_text,
    ) is not None


def phrase_norm(value):
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[_/\\,;:()\[\]{}\"'`]+", " ", text)
    text = re.sub(r"[^\w.+%-]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def value_appears_in_source(value, source_text):
    if source_text is None:
        source_text = ""
    if value is None or value == "" or value == []:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (list, dict)):
        return False
    needle = phrase_norm(value)
    if not needle:
        return False
    haystack = " " + phrase_norm(source_text) + " "
    if len(needle) == 1:
        return re.search(
            r"(^|[^a-z0-9])" + re.escape(needle) + r"([^a-z0-9]|$)",
            haystack,
        ) is not None
    if " " + needle + " " in haystack:
        return True
    if any(ord(ch) > 127 for ch in needle) and needle in haystack:
        return True
    return any(ch.isdigit() for ch in needle) and needle in haystack


def absence_value_is_grounded(value, source_text):
    if not isinstance(value, str) or not source_text:
        return False
    if value.strip().lower() not in {"none", "no"}:
        return False
    return re.search(
        r"\b(no|none|without|exclude|remove|omit)\b",
        str(source_text).lower(),
    ) is not None


def contains_placeholder(value, source_text=""):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if stripped.lower() not in PLACEHOLDER_STRINGS:
            return False
        return not (
            literal_is_grounded(stripped, source_text)
            or absence_value_is_grounded(stripped, source_text)
        )
    if isinstance(value, list):
        return any(contains_placeholder(item, source_text) for item in value)
    if isinstance(value, dict):
        return any(contains_placeholder(item, source_text) for item in value.values())
    return False


def schema_type_matches(value, schema):
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        return any(
            schema_type_matches(value, dict(schema, type=item))
            for item in expected_type
        )
    if expected_type in (None, "any"):
        return True
    if expected_type in ("object", "dict"):
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties") or {}
        return all(
            key not in properties or schema_type_matches(item, properties[key])
            for key, item in value.items()
        )
    if expected_type in ("array", "list", "tuple"):
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items") or {}
        return all(schema_type_matches(item, item_schema) for item in value)
    if expected_type in ("integer", "int"):
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type in ("number", "float"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type in ("boolean", "bool"):
        return isinstance(value, bool)
    if expected_type in ("string", "str"):
        return isinstance(value, str)
    if expected_type == "null":
        return value is None
    return True


def schema_allows_null(schema):
    expected_type = schema.get("type")
    return expected_type == "null" or (
        isinstance(expected_type, list) and "null" in expected_type
    )


def schema_declares_default(schema):
    if not isinstance(schema, dict):
        return False
    if "default" in schema:
        return True
    description = str(schema.get("description") or "")
    return re.search(r"\bdefault\b", description, flags=re.IGNORECASE) is not None


def schema_describes_positive_numeric(schema):
    if not isinstance(schema, dict):
        return False
    description = str(schema.get("description") or "")
    return re.search(
        r"\b(positive|greater than zero|non[-\s]?zero)\b",
        description,
        flags=re.IGNORECASE,
    ) is not None


def schema_describes_separator(schema):
    if not isinstance(schema, dict):
        return False
    description = str(schema.get("description") or "")
    return re.search(
        r"\b(separator|delimiter|joiner|between each string)\b",
        description,
        flags=re.IGNORECASE,
    ) is not None


def schema_enum_matches(value, schema):
    enum_values = schema.get("enum")
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        expected_type = [item for item in expected_type if item != "null"]
        expected_type = expected_type[0] if len(expected_type) == 1 else expected_type
    if expected_type in ("array", "list", "tuple") and isinstance(value, list):
        item_enum_values = (schema.get("items") or {}).get("enum") or enum_values
        if item_enum_values is None:
            return True
        return all(item in item_enum_values for item in value)
    if enum_values is None:
        return True
    return value in enum_values


def normalize_call_for_schema(call, tool_by_name):
    name = call.get("name")
    tool = tool_by_name.get(name)
    arguments = call.get("arguments")
    if tool is None or not isinstance(arguments, dict):
        return call

    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    normalized_arguments = {
        key: value
        for key, value in arguments.items()
        if not (
            value is None
            and key not in required
            and key in properties
            and not schema_allows_null(properties[key])
        )
    }
    return {**call, "arguments": normalized_arguments}


def _clean_optional_value(
    value,
    schema,
    source_text,
    path,
    drop_unmentioned_scalars=True,
    drop_defaulted_optional_scalars=False,
):
    if isinstance(value, dict):
        properties = (schema or {}).get("properties") or {}
        required = set((schema or {}).get("required") or [])
        cleaned = dict(value)
        dropped = []
        for key, item in list(value.items()):
            if key not in properties:
                continue
            child_path = f"{path}.{key}" if path else key
            cleaned_item, child_drops = _clean_optional_value(
                item,
                properties[key],
                source_text,
                child_path,
                drop_unmentioned_scalars=drop_unmentioned_scalars,
                drop_defaulted_optional_scalars=drop_defaulted_optional_scalars,
            )
            dropped.extend(child_drops)
            if key in required:
                cleaned[key] = cleaned_item
                continue
            if _optional_value_is_ungrounded(
                cleaned_item,
                properties[key],
                source_text,
                drop_unmentioned_scalars=drop_unmentioned_scalars,
                drop_defaulted_optional_scalars=drop_defaulted_optional_scalars,
            ):
                dropped.append(
                    {
                        "arg": child_path,
                        "removed_value": cleaned_item,
                        "reason": f"ungrounded_optional_argument:{child_path}",
                    }
                )
                cleaned.pop(key, None)
            else:
                cleaned[key] = cleaned_item
        return cleaned, dropped
    return value, []


def _optional_value_is_ungrounded(
    value,
    schema,
    source_text,
    drop_unmentioned_scalars=True,
    drop_defaulted_optional_scalars=False,
):
    if value is None:
        return not schema_allows_null(schema or {})
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        return False
    if isinstance(value, str) and not value.strip():
        if value == " " and schema_describes_separator(schema or {}):
            return False
        return True
    grounded = value_appears_in_source(value, source_text) or absence_value_is_grounded(
        value,
        source_text,
    )
    if isinstance(value, str) and value.strip().lower() in {
        "dontcare",
        "don't care",
        "do not care",
        "no preference",
    }:
        return not grounded
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == 0.0
        and schema_describes_positive_numeric(schema or {})
    ):
        return not grounded
    if (
        not drop_unmentioned_scalars
        and drop_defaulted_optional_scalars
        and schema_declares_default(schema or {})
    ):
        return not grounded
    if not drop_unmentioned_scalars:
        return contains_placeholder(value, source_text)
    return not grounded


def drop_ungrounded_optional_arguments(
    call,
    tool_by_name,
    source_text="",
    drop_unmentioned_scalars=True,
    drop_defaulted_optional_scalars=False,
):
    name = call.get("name")
    tool = tool_by_name.get(name)
    arguments = call.get("arguments")
    if tool is None or not isinstance(arguments, dict):
        return call, []

    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    cleaned = dict(arguments)
    dropped = []

    for key, value in list(cleaned.items()):
        if key not in properties:
            continue
        cleaned_value, child_drops = _clean_optional_value(
            value,
            properties[key],
            source_text,
            key,
            drop_unmentioned_scalars=drop_unmentioned_scalars,
            drop_defaulted_optional_scalars=drop_defaulted_optional_scalars,
        )
        dropped.extend(child_drops)
        if key in required:
            cleaned[key] = cleaned_value
            continue
        if not _optional_value_is_ungrounded(
            cleaned_value,
            properties[key],
            source_text,
            drop_unmentioned_scalars=drop_unmentioned_scalars,
            drop_defaulted_optional_scalars=drop_defaulted_optional_scalars,
        ):
            cleaned[key] = cleaned_value
            continue
        dropped.append(
            {
                "arg": key,
                "removed_value": cleaned_value,
                "reason": f"ungrounded_optional_argument:{key}",
            }
        )
        cleaned.pop(key, None)

    if not dropped:
        return call, []
    return {**call, "arguments": cleaned}, dropped


def deterministic_call_check(call, tool_by_name, source_text=""):
    name = call.get("name")
    tool = tool_by_name.get(name)
    if tool is None:
        return False, "unknown_tool"

    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments_not_object"

    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []

    for key in required:
        if key not in arguments:
            return False, f"missing_required:{key}"
        if contains_placeholder(arguments[key], source_text):
            return False, f"placeholder_required:{key}"

    for key, value in arguments.items():
        if key not in properties:
            return False, f"unexpected_argument:{key}"
        if contains_placeholder(value, source_text):
            return False, f"placeholder_argument:{key}"
        if not schema_type_matches(value, properties[key]):
            return False, f"type_mismatch:{key}"
        if not schema_enum_matches(value, properties[key]):
            return False, f"enum_mismatch:{key}"

    return True, ""


def deterministic_verify_calls(
    calls,
    tools,
    source_text="",
    drop_ungrounded_optional_args=False,
    drop_unmentioned_optional_scalars=True,
):
    tool_by_name = {tool["name"]: tool for tool in tools}
    kept = []
    dropped = []
    argument_drops = []
    seen = set()
    merged_count = 0

    for index, call in enumerate(calls):
        call = normalize_call_for_schema(call, tool_by_name)
        if drop_ungrounded_optional_args:
            call, call_argument_drops = drop_ungrounded_optional_arguments(
                call,
                tool_by_name,
                source_text,
                drop_unmentioned_scalars=drop_unmentioned_optional_scalars,
            )
            for item in call_argument_drops:
                argument_drops.append({"index": index, "call": call, **item})
        valid, reason = deterministic_call_check(call, tool_by_name, source_text)
        if not valid:
            dropped.append({"index": index, "call": call, "reason": reason})
            continue
        key = canonical(call)
        if key in seen:
            merged_count += 1
            continue
        seen.add(key)
        kept.append({"original_index": index, "call": call})

    return kept, dropped, merged_count, argument_drops


def drop_invalid_optional_enum_arguments(call, tool_by_name):
    name = call.get("name")
    tool = tool_by_name.get(name) or {}
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict):
        return call, []
    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    cleaned = dict(arguments)
    dropped = []
    for key, value in list(arguments.items()):
        if key in required or key not in properties:
            continue
        schema = properties.get(key) or {}
        if schema_enum_matches(value, schema):
            continue
        cleaned.pop(key, None)
        dropped.append(
            {
                "arg": key,
                "removed_value": value,
                "reason": f"invalid_optional_enum:{key}",
            }
        )
    if not dropped:
        return call, []
    return {**call, "arguments": cleaned}, dropped


def sanitize_candidate_for_router(task, candidate, drop_defaulted_optional_args=False):
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    cleaned_calls = []
    argument_drops = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        cleaned_call = normalize_call_for_schema(call, tool_by_name)
        cleaned_call, enum_argument_drops = drop_invalid_optional_enum_arguments(
            cleaned_call,
            tool_by_name,
        )
        for item in enum_argument_drops:
            argument_drops.append({"index": index, "call": cleaned_call, **item})
        cleaned_call, call_argument_drops = drop_ungrounded_optional_arguments(
            cleaned_call,
            tool_by_name,
            task["prompt"],
            drop_unmentioned_scalars=False,
            drop_defaulted_optional_scalars=drop_defaulted_optional_args,
        )
        cleaned_calls.append(cleaned_call)
        for item in call_argument_drops:
            argument_drops.append({"index": index, "call": cleaned_call, **item})
    sanitized = {
        **candidate,
        "normalized_calls": cleaned_calls,
    }
    if argument_drops:
        sanitized["router_cleanup"] = {
            "argument_drops": argument_drops,
            "argument_dropped_count": len(argument_drops),
        }
    return sanitized


_OPTIONAL_DEFAULT_PLACEHOLDER_RE = re.compile(
    r"^(your[_ -]?api[_ -]?key(?:[_ -]?here)?|api[_ -]?key|YOUR_API_KEY_HERE)$",
    re.IGNORECASE,
)

_UNIT_VALUE_ALIASES = {
    "celsius": ["celsius", "°c", "degrees c", "degree c", "centigrade", "metric", "摄氏", "摄氏度"],
    "fahrenheit": ["fahrenheit", "°f", "degrees f", "degree f", "imperial", "华氏", "华氏度"],
    "metric": ["metric", "celsius", "°c", "degrees c", "degree c", "centigrade", "摄氏", "摄氏度"],
    "imperial": ["imperial", "fahrenheit", "°f", "degrees f", "degree f", "华氏", "华氏度"],
    "standard": ["standard", "kelvin", "开尔文"],
    "meters": ["meters", "meter", "metres", "metre", "m"],
    "meter": ["meters", "meter", "metres", "metre", "m"],
    "metres": ["meters", "meter", "metres", "metre", "m"],
    "metre": ["meters", "meter", "metres", "metre", "m"],
    "m": ["meters", "meter", "metres", "metre", "m"],
    "feet": ["feet", "foot", "ft"],
    "foot": ["feet", "foot", "ft"],
    "ft": ["feet", "foot", "ft"],
    "cm": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    "centimeter": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    "centimeters": ["cm", "centimeter", "centimeters", "centimetre", "centimetres"],
    "inches": ["inches", "inch", "in"],
    "inch": ["inches", "inch", "in"],
}

_TEMPERATURE_UNIT_VALUES = {
    "celsius",
    "fahrenheit",
    "metric",
    "imperial",
    "standard",
    "kelvin",
}

_GENERIC_ENTITY_SUFFIXES = {
    "area",
    "city",
    "county",
    "district",
    "hotel",
    "province",
    "state",
    "subdistrict",
    "sub-district",
}

_ENTITY_SUFFIX_ARGUMENT_RE = re.compile(
    r"(?:^|_)(?:name|title|artist|customer|receiver|recipient|merchant|company|"
    r"event|movie|show|restaurant|hotel|location|city|district|sub_district|"
    r"province|state|region)(?:_|$)",
    re.IGNORECASE,
)

_THEATER_NAME_ARGUMENT_RE = re.compile(
    r"(?:^|_)theat(?:er|re)_?name(?:_|$)",
    re.IGNORECASE,
)

_ACCOUNT_OWNER_ARGUMENT_RE = re.compile(
    r"(?:^|_)(?:receiver|recipient|payee|payer|sender|customer|user|person|"
    r"account_holder)(?:_|$)",
    re.IGNORECASE,
)

_MUSEUM_LOCATION_ARGUMENT_RE = re.compile(
    r"^(?:museum_location|.*_museum_location)$",
    re.IGNORECASE,
)

_POSSESSIVE_ACCOUNT_RE = re.compile(
    r"^(.+?)['\u2019]s\s+account$",
    re.IGNORECASE,
)

_DATETIME_ARGUMENT_RE = re.compile(
    r"(?:^|_)(?:date_?or_?time|datetime|date_time|new_datetime)(?:_|$)",
    re.IGNORECASE,
)

_UTC_SUFFIXED_LOCAL_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)

_CITY_ARGUMENT_RE = re.compile(r"^(?:city|.*_city)$", re.IGNORECASE)

_ITEM_LIST_ARGUMENT_RE = re.compile(
    r"(?:^|_)(?:items?|products?|goods|groceries)(?:_|$)",
    re.IGNORECASE,
)

_PICKUP_ARGUMENT_RE = re.compile(
    r"^(?:pick_?up|pickup|pickup_location|pick_up_location)$",
    re.IGNORECASE,
)

_PICKUP_PHRASE_RE = re.compile(
    r"\bpick(?:ing)?\s+(?:it|them|the\s+car|the\s+vehicle)?\s*up\s+from\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,30})",
    re.IGNORECASE,
)

_DERIVED_RESULT_PLACEHOLDER_RE = re.compile(
    r"^(?:derivative_)?result(?:_of_.*)?$|^<[^>]*result[^>]*>$|"
    r"^\$\{[^}]*result[^}]*\}$",
    re.IGNORECASE,
)

_CITY_NAME_ONLY_DESCRIPTION_RE = re.compile(
    r"\bname\s+of\s+(?:the\s+)?city\b",
    re.IGNORECASE,
)

_CITY_FORMAT_DESCRIPTION_RE = re.compile(
    r"\b(?:city,\s*(?:state|country)|city\s+and\s+(?:state|country)|"
    r"formatted\s+as|format\s+of|state\s+abbreviation|such\s+as|"
    r"(?:state|country)\b|abbr)\b",
    re.IGNORECASE,
)

_DETAIL_BOOLEAN_ARGUMENT_RE = re.compile(
    r"(?:^|_)(?:details?|detail_level|verbose)(?:_|$)",
    re.IGNORECASE,
)

_GROUNDED_BOOLEAN_FLAG_PREFIX_RE = re.compile(
    r"^(?:has|is|include|includes|available_for|in_unit)_",
    re.IGNORECASE,
)

_BOOLEAN_FLAG_TOKEN_PREFIXES = {
    "has",
    "is",
    "include",
    "includes",
    "available",
    "for",
    "in",
    "unit",
}

_BOOLEAN_FLAG_TOKEN_STOPWORDS = {"a", "an", "and", "or", "the", "to", "of", "by", "on"}

_DETAIL_REQUEST_RE = re.compile(
    r"\b(details?|detailed|full\s+details?|verbose)\b",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"\b(no|not|without|exclude|excluding|omit|omitting|skip|disable|disabled|false)\b",
    re.IGNORECASE,
)


def _prompt_has_term(text, term):
    term = str(term or "").strip().lower()
    if not term:
        return False
    lowered = str(text or "").lower()
    if re.search(r"[\u4e00-\u9fff°]", term):
        return term in lowered
    if len(term) <= 2:
        return re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
            lowered,
        ) is not None
    return term in lowered


def _unit_value_grounded(value, schema, source_text):
    if value_appears_in_source(value, source_text) or absence_value_is_grounded(
        value,
        source_text,
    ):
        return True
    if not isinstance(value, str):
        return False
    for alias in _UNIT_VALUE_ALIASES.get(value.strip().lower(), [value]):
        if _prompt_has_term(source_text, alias):
            return True
    return False


def _prompt_mentions_schema_unit(schema, source_text):
    values = set()
    if isinstance(schema, dict):
        if isinstance(schema.get("default"), str):
            values.add(schema["default"].strip().lower())
        for item in schema.get("enum") or []:
            if isinstance(item, str):
                values.add(item.strip().lower())
    for value in values:
        for alias in _UNIT_VALUE_ALIASES.get(value, [value]):
            if _prompt_has_term(source_text, alias):
                return True
    return False


def _is_temperature_unit_value(value):
    return str(value or "").strip().lower() in _TEMPERATURE_UNIT_VALUES


def _is_result_limit_argument(key, schema):
    text = f"{key} {schema.get('description') if isinstance(schema, dict) else ''}".lower()
    return re.search(
        r"\b(max(?:imum)?[_\s-]?(results|items|records)|limit|top[_\s-]?k)\b",
        text,
    ) is not None


def _schema_expects_boolean(schema):
    expected_type = (schema or {}).get("type")
    if isinstance(expected_type, list):
        return "boolean" in expected_type or "bool" in expected_type
    return expected_type in {"boolean", "bool"}


def _prompt_negates_detail_request(source_text):
    text = str(source_text or "")
    for match in _DETAIL_REQUEST_RE.finditer(text):
        window = text[max(0, match.start() - 40) : match.end() + 40]
        if _NEGATION_RE.search(window):
            return True
    return False


def _detail_boolean_fill_requested(key, schema, source_text):
    return (
        _schema_expects_boolean(schema)
        and _DETAIL_BOOLEAN_ARGUMENT_RE.search(str(key or "")) is not None
        and _DETAIL_REQUEST_RE.search(str(source_text or "")) is not None
        and not _prompt_negates_detail_request(source_text)
    )


def _boolean_flag_tokens(key):
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(key or ""))
    tokens = [item.lower() for item in re.split(r"[^A-Za-z0-9]+", text) if item]
    return [
        item
        for item in tokens
        if item not in _BOOLEAN_FLAG_TOKEN_PREFIXES
        and item not in _BOOLEAN_FLAG_TOKEN_STOPWORDS
        and len(item) > 2
    ]


def _prompt_affirms_boolean_flag_token(source_text, token):
    text = str(source_text or "")
    pattern = r"(?<![A-Za-z0-9])" + re.escape(str(token)) + r"s?(?![A-Za-z0-9])"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        window = text[max(0, match.start() - 45) : match.end() + 45]
        if _NEGATION_RE.search(window):
            continue
        return True
    return False


def _prompt_grounded_boolean_flag_fill_requested(key, schema, source_text):
    if not _schema_expects_boolean(schema):
        return False
    if _GROUNDED_BOOLEAN_FLAG_PREFIX_RE.search(str(key or "")) is None:
        return False
    tokens = _boolean_flag_tokens(key)
    return bool(tokens) and all(
        _prompt_affirms_boolean_flag_token(source_text, token)
        for token in tokens
    )


def _optional_default_repair_action(key, value, schema, source_text):
    if (
        not isinstance(schema, dict)
        or "default" not in schema
        or isinstance(value, (dict, list))
        or _schema_default_matches_value(schema, value)
    ):
        return None
    default = schema.get("default")
    if (
        isinstance(default, str)
        and isinstance(value, str)
        and _OPTIONAL_DEFAULT_PLACEHOLDER_RE.match(default)
        and _OPTIONAL_DEFAULT_PLACEHOLDER_RE.match(value)
    ):
        return "placeholder_default_casing"
    if (
        isinstance(default, (int, float))
        and not isinstance(default, bool)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and _is_result_limit_argument(key, schema)
        and not (
            value_appears_in_source(value, source_text)
            or absence_value_is_grounded(value, source_text)
        )
    ):
        return "unmentioned_result_limit_default"
    if (
        key.lower() in {"unit", "units"}
        and isinstance(default, str)
        and isinstance(value, str)
        and not _is_temperature_unit_value(default)
        and not _is_temperature_unit_value(value)
        and not _unit_value_grounded(value, schema, source_text)
        and not _prompt_mentions_schema_unit(schema, source_text)
    ):
        return "unmentioned_unit_default"
    return None


def repair_prompt_grounded_optional_arguments(task, candidate):
    """Fill omitted optional arguments only when the prompt strongly grounds them."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    source_text = task.get("prompt", "")
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, schema in properties.items():
            if key in required or key in repaired_arguments:
                continue
            reason = None
            if _detail_boolean_fill_requested(key, schema or {}, source_text):
                reason = "detail_boolean_requested"
            elif _prompt_grounded_boolean_flag_fill_requested(
                key,
                schema or {},
                source_text,
            ):
                reason = "prompt_grounded_boolean_flag"
            if reason is None:
                continue
            repaired_arguments[key] = True
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "to": True,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "prompt_grounded_optional_fill": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


WEEKDAY_INDEX = {day.lower(): index for index, day in enumerate(calendar.day_name)}
MONTH_INDEX = {
    month.lower(): index
    for index, month in enumerate(calendar.month_name)
    if month
}
MONTH_INDEX.update(
    {
        month.lower(): index
        for index, month in enumerate(calendar.month_abbr)
        if month
    }
)


def _parse_prompt_today_anchor(source_text):
    text = str(source_text or "").lower()
    month_name_match = re.search(
        r"\btoday\s+is\s+"
        r"(?:(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+)?"
        r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b",
        text,
    )
    if month_name_match:
        weekday, month, day, year = month_name_match.groups()
        if month in MONTH_INDEX:
            date = datetime.date(int(year), MONTH_INDEX[month], int(day))
            return date, WEEKDAY_INDEX.get(weekday, date.weekday())

    numeric_match = re.search(
        r"\btoday\s+is\s+(\d{4})[./-](\d{1,2})[./-](\d{1,2})"
        r"(?:\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))?\b",
        text,
    )
    if numeric_match:
        year, month, day, weekday = numeric_match.groups()
        date = datetime.date(int(year), int(month), int(day))
        return date, WEEKDAY_INDEX.get(weekday, date.weekday())
    return None


def _prompt_relative_weekday_date(source_text):
    text = str(source_text or "").lower()
    anchor = _parse_prompt_today_anchor(source_text)
    if not anchor:
        return None
    anchor_date, anchor_weekday = anchor
    for weekday_name, target_weekday in WEEKDAY_INDEX.items():
        next_before = re.search(
            rf"\b(?:upcoming|coming|next)\s+{weekday_name}\b",
            text,
        )
        next_after = re.search(
            rf"\b{weekday_name}\s+next\s+week\b",
            text,
        )
        if not next_before and not next_after:
            continue
        if next_after:
            delta_days = 7 + ((target_weekday - anchor_weekday) % 7)
        else:
            delta_days = (target_weekday - anchor_weekday) % 7
            if delta_days == 0:
                delta_days = 7
        return (anchor_date + datetime.timedelta(days=delta_days)).isoformat()
    return None


def _schema_accepts_iso_date(schema):
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    description = str(schema.get("description") or "").lower()
    return (
        schema_type in {None, "string"}
        and ("yyyy-mm-dd" in description or "format 'yyyy-mm-dd'" in description)
    )


def repair_prompt_relative_date_arguments(task, candidate):
    """Fill a single omitted date when the prompt gives an anchored relative day."""
    derived_date = _prompt_relative_weekday_date(task.get("prompt", ""))
    if not derived_date:
        return candidate
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        fillable_keys = [
            key
            for key, schema in properties.items()
            if str(key).lower() == "date"
            and key not in required
            and key not in arguments
            and _schema_accepts_iso_date(schema)
        ]
        if len(fillable_keys) != 1:
            repaired_calls.append(call)
            continue
        key = fillable_keys[0]
        repaired_arguments = dict(arguments)
        repaired_arguments[key] = derived_date
        repaired_calls.append({**call, "arguments": repaired_arguments})
        changes.append(
            {
                "index": index,
                "tool": call.get("name"),
                "arg": key,
                "to": derived_date,
                "reason": "anchored_relative_weekday",
            }
        )
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "relative_date_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


RECURRENCE_CUE_RE = re.compile(
    r"\b(repeat|repeats|recurring|every|daily|weekly|monthly|annually|yearly)\b",
    re.I,
)


def _prompt_lacks_recurrence_cue(source_text):
    return RECURRENCE_CUE_RE.search(str(source_text or "")) is None


def _parse_iso_timestamp(value):
    if not isinstance(value, str):
        return None, False
    for fmt, has_z in (
        ("%Y-%m-%dT%H:%M:%SZ", True),
        ("%Y-%m-%dT%H:%M:%S", False),
    ):
        try:
            return datetime.datetime.strptime(value, fmt), has_z
        except ValueError:
            pass
    return None, False


def _prompt_mentions_end_time(source_text, end_time):
    if not isinstance(end_time, datetime.datetime):
        return False
    hour = end_time.hour
    minute = end_time.minute
    variants = {
        f"{hour}:{minute:02d}",
        f"{hour:02d}:{minute:02d}",
    }
    return any(
        re.search(rf"(?i)\b(?:to|until|through)\s+{re.escape(variant)}\b", str(source_text or ""))
        for variant in variants
    )


def _derive_t1_from_t0_timespan(arguments, source_text):
    if not isinstance(arguments, dict):
        return None
    if "t1" in arguments or "t0" not in arguments or "timespan" not in arguments:
        return None
    timespan = arguments.get("timespan")
    if not isinstance(timespan, int) or isinstance(timespan, bool):
        return None
    start, has_z = _parse_iso_timestamp(arguments.get("t0"))
    if start is None:
        return None
    end = start + datetime.timedelta(seconds=timespan)
    if not _prompt_mentions_end_time(source_text, end):
        return None
    return end.strftime("%Y-%m-%dT%H:%M:%S") + ("Z" if has_z else "")


def _prompt_requests_start_date(source_text):
    return re.search(
        r"(?i)\b(start date|started|date when it started)\b",
        str(source_text or ""),
    ) is not None


def repair_prompt_temporal_default_arguments(task, candidate):
    """Fill narrowly grounded temporal/default fields omitted by the model."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    source_text = task.get("prompt", "")
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)

        rrule_schema = properties.get("rrule") or {}
        if (
            "rrule" not in repaired_arguments
            and rrule_schema.get("default") == "Does not repeat"
            and _prompt_lacks_recurrence_cue(source_text)
        ):
            repaired_arguments["rrule"] = "Does not repeat"
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": "rrule",
                    "to": "Does not repeat",
                    "reason": "non_recurring_default",
                }
            )

        if "t1" in properties and "t1" not in repaired_arguments:
            derived_t1 = _derive_t1_from_t0_timespan(repaired_arguments, source_text)
            if derived_t1 is not None:
                repaired_arguments["t1"] = derived_t1
                changes.append(
                    {
                        "index": index,
                        "tool": call.get("name"),
                        "arg": "t1",
                        "to": derived_t1,
                        "reason": "t0_plus_timespan_end_time",
                    }
                )

        start_date_schema = properties.get("start_date") or {}
        if (
            "start_date" not in repaired_arguments
            and "start_date" in properties
            and start_date_schema.get("default") is None
            and _prompt_requests_start_date(source_text)
        ):
            repaired_arguments["start_date"] = None
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": "start_date",
                    "to": None,
                    "reason": "requested_unknown_start_date_default",
                }
            )

        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "temporal_default_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def repair_optional_default_arguments(task, candidate):
    """Restore conservative schema defaults for ungrounded optional arguments."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if key in required or key not in properties:
                continue
            schema = properties.get(key) or {}
            action = _optional_default_repair_action(
                key,
                value,
                schema,
                task.get("prompt", ""),
            )
            if not action:
                continue
            default = schema.get("default")
            repaired_arguments[key] = default
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": default,
                    "reason": action,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "optional_default_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


ADDRESS_LABEL_LOCATION_KEYS = {
    "address",
    "drop_off_location",
    "dropoff_location",
    "loc",
    "location",
    "pick_up_location",
    "pickup_location",
}

EVENT_TYPE_SUFFIXES = {"concert", "event", "play", "show"}


def _strip_diacritics(value):
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value or ""))
        if not unicodedata.combining(char)
    )


def _has_diacritic(value):
    return _strip_diacritics(value) != str(value or "")


def _prompt_diacritic_span(value, source_text):
    if not isinstance(value, str) or value_appears_in_source(value, source_text):
        return None
    source = str(source_text or "")
    normalized_value = _strip_diacritics(value).lower()
    normalized_source = _strip_diacritics(source).lower()
    offset = normalized_source.find(normalized_value)
    if offset < 0:
        return None
    restored = source[offset : offset + len(value)]
    if (
        restored != value
        and _has_diacritic(restored)
        and _strip_diacritics(restored).lower() == normalized_value
    ):
        return restored
    return None


def _prompt_address_label_span(source_text):
    match = re.search(
        r"(?i)\baddress\s*:\s*([^\n.;]+)",
        str(source_text or ""),
    )
    if not match:
        return None
    span = _normalize_space(match.group(1))
    return span if len(span) >= 3 else None


def _schema_describes_artist_or_title(schema):
    description = str((schema or {}).get("description") or "").lower()
    return (
        "artist" in description
        and (
            "title" in description
            or "play" in description
            or "performer" in description
        )
    )


def _prompt_location_suffix_span(value, source_text):
    text = _normalize_space(value)
    if "," not in text or value_appears_in_source(text, source_text):
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) < 3:
        return None
    candidate = ", ".join(parts[-2:])
    if len(candidate) < 4:
        return None
    if re.search(
        rf"(?i)\b(?:at|in|near|located at|locates at|location is|for)\s+{re.escape(candidate)}\b",
        str(source_text or ""),
    ):
        return candidate
    return None


def _initials(value):
    parts = re.findall(r"[A-Za-z]+", str(value or ""))
    return "".join(part[0] for part in parts if part).lower()


def _looks_like_region_abbreviation(short_value, long_value):
    short = re.sub(r"[^A-Za-z]", "", str(short_value or "")).lower()
    if len(short) < 2 or len(short) > 4:
        return False
    long_text = str(long_value or "").strip()
    if len(long_text) <= len(short_value or ""):
        return False
    if not re.search(r"[A-Za-z]", long_text):
        return False
    long_compact = re.sub(r"[^A-Za-z]", "", long_text).lower()
    return short == _initials(long_text)[: len(short)] or long_compact.startswith(short)


def _prompt_location_abbreviation_expansion_span(value, source_text):
    text = _normalize_space(value)
    if "," not in text or value_appears_in_source(text, source_text):
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        return None
    city, region = parts
    compact_region = re.sub(r"[^A-Za-z]", "", region)
    if len(city) < 2 or len(compact_region) < 2 or len(compact_region) > 4:
        return None

    source = str(source_text or "")
    city_re = re.compile(rf"(?i)\b{re.escape(city)}\s*,\s*")
    for match in city_re.finditer(source):
        tail_start = match.end()
        tail = source[tail_start : tail_start + 80]
        stop = re.search(
            r"(?:,\s+(?:and|or)\b|\s+(?:in|using|with|please|pls|show|display|tell|give|provide|could|can|would)\b|[.?!;\n])",
            tail,
            flags=re.IGNORECASE,
        )
        tail = tail[: stop.start()] if stop else tail
        tail = tail.strip(" ,")
        if not tail:
            continue
        candidate = _normalize_space(source[match.start() : tail_start] + tail)
        if (
            candidate != text
            and len(candidate) > len(text)
            and _looks_like_region_abbreviation(region, tail)
        ):
            return candidate
    return None


def _prompt_span_trim_value(key, value, schema, source_text):
    if not isinstance(value, str):
        return None
    key_text = str(key or "").lower()
    text = _normalize_space(value)

    restored = _prompt_diacritic_span(text, source_text)
    if restored:
        return restored, "prompt_diacritic_span"

    address_span = _prompt_address_label_span(source_text)
    if (
        address_span
        and key_text in ADDRESS_LABEL_LOCATION_KEYS
        and text.lower().startswith(address_span.lower() + ",")
    ):
        return address_span, "address_label_prefix"

    if key_text in ADDRESS_LABEL_LOCATION_KEYS:
        location_suffix = _prompt_location_suffix_span(text, source_text)
        if location_suffix:
            return location_suffix, "prompt_location_suffix"
        location_expansion = _prompt_location_abbreviation_expansion_span(
            text, source_text
        )
        if location_expansion:
            return location_expansion, "prompt_location_abbreviation_expansion"

    if key_text == "event_name" and _schema_describes_artist_or_title(schema):
        trimmed = _trim_trailing_entity_suffix(text, EVENT_TYPE_SUFFIXES)
        if (
            trimmed
            and value_appears_in_source(text, source_text)
            and value_appears_in_source(trimmed, source_text)
        ):
            return trimmed, "event_type_suffix_for_artist_title"
    return None


def repair_prompt_span_trimmed_values(task, candidate):
    """Trim model-expanded strings when the prompt/schema identify the shorter span."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    source_text = task.get("prompt", "")
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties") or {})
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if key not in properties:
                continue
            result = _prompt_span_trim_value(
                key,
                value,
                properties.get(key) or {},
                source_text,
            )
            if not result:
                continue
            trimmed, reason = result
            if trimmed == value:
                continue
            repaired_arguments[key] = trimmed
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": trimmed,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "prompt_span_trim_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


API_KEY_PLACEHOLDER_VALUES = {
    "api_key",
    "apikey",
    "my_api_key",
    "myapikey",
    "your_api_key",
}


def _prompt_quoted_values(source_text):
    values = []
    for match in re.finditer(r"[\"']([^\"']{2,80})[\"']", str(source_text or "")):
        values.append(match.group(1).strip())
    return values


def _pathish_prompt_parts(value):
    return [
        part.strip("\"'")
        for part in str(value or "").split()
        if "/" in part and len(part) >= 3
    ]


def _all_pathish_parts_in_prompt(value, source_text):
    parts = _pathish_prompt_parts(value)
    return bool(parts) and all(part in str(source_text or "") for part in parts)


def _narrow_value_repair(key, value, schema, source_text):
    if not isinstance(value, str):
        return None
    key_text = str(key or "").lower()
    text = _normalize_space(value)

    if any(token in key_text for token in ("path", "file")) and "/" in text:
        basename = text.rsplit("/", 1)[-1]
        if (
            basename
            and value_appears_in_source(basename, source_text)
            and not value_appears_in_source(text, source_text)
        ):
            return basename, "prompt_basename_path"

    if key_text in {"cmd", "command"} and "\\" in value:
        normalized = value.replace("\\", "/")
        if _all_pathish_parts_in_prompt(
            normalized,
            source_text,
        ):
            return normalized, "command_slash_normalization"

    if (
        key_text == "api_key"
        and text.lower() in API_KEY_PLACEHOLDER_VALUES
        and not value_appears_in_source(text, source_text)
    ):
        return (schema or {}).get("default") or "YOUR_API_KEY_HERE", "api_key_placeholder_default"

    if key_text.endswith("id") and text.lower().endswith("_id"):
        stem = text[:-3]
        if stem in _prompt_quoted_values(source_text):
            return stem, "quoted_identifier_id_suffix"

    description = str((schema or {}).get("description") or "").lower()
    if (
        key_text in {"city", "loc", "location"}
        and text.endswith(", US")
        and "city, country" in description
        and "state" not in description
    ):
        return text[:-3] + " USA", "us_country_expansion"

    return None


def repair_narrow_prompt_schema_values(task, candidate):
    """Apply high-precision prompt/schema value normalizations."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    source_text = task.get("prompt", "")
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties") or {})
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if key not in properties:
                continue
            result = _narrow_value_repair(
                key,
                value,
                properties.get(key) or {},
                source_text,
            )
            if not result:
                continue
            repaired, reason = result
            if repaired == value:
                continue
            repaired_arguments[key] = repaired
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": repaired,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "narrow_value_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def _normalize_space(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def _entity_suffix_trim(value, source_text):
    text = _normalize_space(value)
    if not text or value_appears_in_source(text, source_text):
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    suffix = parts[-1].strip(" ,.;:()[]{}").lower()
    if suffix not in _GENERIC_ENTITY_SUFFIXES:
        return None
    trimmed = _normalize_space(" ".join(parts[:-1]))
    if len(trimmed) < 3 or not value_appears_in_source(trimmed, source_text):
        return None
    return trimmed


def _trim_trailing_entity_suffix(value, suffixes):
    text = _normalize_space(value)
    parts = text.split()
    if len(parts) < 2:
        return None
    suffix = parts[-1].strip(" ,.;:()[]{}").lower()
    if suffix not in suffixes:
        return None
    trimmed = _normalize_space(" ".join(parts[:-1]))
    return trimmed if len(trimmed) >= 2 else None


def _prompt_mentions_bare_museum_location(location, source_text):
    loc = phrase_norm(location)
    source = phrase_norm(source_text)
    if not loc or not source:
        return False
    return (
        re.search(r"\bmuseum of " + re.escape(loc) + r"\b", source) is not None
        or re.search(r"\b" + re.escape(loc) + r" museum\b", source) is not None
    )


def _context_entity_value_repair(key, value, source_text):
    if not isinstance(value, str):
        return None
    if _THEATER_NAME_ARGUMENT_RE.search(str(key or "")):
        trimmed = _trim_trailing_entity_suffix(
            value,
            {"theater", "theatre", "cinema"},
        )
        if trimmed and value_appears_in_source(trimmed, source_text):
            return trimmed, "theater_name_suffix"
    if _ACCOUNT_OWNER_ARGUMENT_RE.search(str(key or "")):
        match = _POSSESSIVE_ACCOUNT_RE.match(_normalize_space(value))
        if match:
            owner = _normalize_space(match.group(1))
            if owner and value_appears_in_source(owner, source_text):
                return owner, "possessive_account_owner"
    if _MUSEUM_LOCATION_ARGUMENT_RE.match(str(key or "")):
        match = re.match(r"^(.+?)\s+Museum$", _normalize_space(value), re.IGNORECASE)
        if match:
            location = _normalize_space(match.group(1))
            if _prompt_mentions_bare_museum_location(location, source_text):
                return location, "museum_location_suffix"
    return None


def _schema_describes_local_datetime(schema):
    text = str((schema or {}).get("description") or "").lower()
    return (
        "timezone is not included" in text
        or "time zone is not included" in text
        or "timezone should be specified separately" in text
        or "time zone should be specified separately" in text
        or "timezone is specified separately" in text
        or "time zone is specified separately" in text
    )


_MONTH_ABBREVIATION_ALIASES = {
    "jan": "January",
    "jan.": "January",
    "feb": "February",
    "feb.": "February",
    "mar": "March",
    "mar.": "March",
    "apr": "April",
    "apr.": "April",
    "jun": "June",
    "jun.": "June",
    "jul": "July",
    "jul.": "July",
    "aug": "August",
    "aug.": "August",
    "sep": "September",
    "sep.": "September",
    "sept": "September",
    "sept.": "September",
    "oct": "October",
    "oct.": "October",
    "nov": "November",
    "nov.": "November",
    "dec": "December",
    "dec.": "December",
}


def _month_argument_key(key):
    key_text = str(key or "").lower()
    return key_text == "month" or key_text.endswith("_month")


def _temporal_value_repair(key, value, schema):
    if not isinstance(value, str):
        return None
    if _month_argument_key(key):
        alias = _MONTH_ABBREVIATION_ALIASES.get(value.strip().lower())
        if alias:
            return alias, "expand_month_abbreviation"
    if not _DATETIME_ARGUMENT_RE.search(str(key or "")):
        return None
    if not _UTC_SUFFIXED_LOCAL_DATETIME_RE.match(value):
        return None
    if not _schema_describes_local_datetime(schema or {}):
        return None
    return value[:-1], "trim_utc_suffix_for_local_datetime"


def _schema_expects_city_name_only(schema):
    description = str((schema or {}).get("description") or "")
    return (
        _CITY_NAME_ONLY_DESCRIPTION_RE.search(description) is not None
        and _CITY_FORMAT_DESCRIPTION_RE.search(description) is None
    )


def _city_value_repair(key, value, schema, source_text):
    if not isinstance(value, str):
        return None
    if not _CITY_ARGUMENT_RE.match(str(key or "")):
        return None
    if not _schema_expects_city_name_only(schema or {}):
        return None
    parts = [_normalize_space(part) for part in value.split(",")]
    if len(parts) < 2 or not parts[0]:
        return None
    city = parts[0]
    if len(city) < 2:
        return None
    if not (
        value_appears_in_source(city, source_text)
        or value_appears_in_source(value, source_text)
    ):
        return None
    return city, "city_name_only_trim_region"


def _simple_plural_variant(value):
    text = _normalize_space(value)
    if not text or not re.match(r"^[A-Za-z][A-Za-z -]*$", text):
        return None
    if text != text.lower():
        return None
    words = text.split()
    last = words[-1]
    lower = last.lower()
    if len(lower) < 3 or lower.endswith("s"):
        return None
    if re.search(r"[^aeiou]y$", lower):
        plural_last = last[:-1] + ("IES" if last.isupper() else "ies")
    elif re.search(r"(?:s|x|z|ch|sh)$", lower):
        plural_last = last + "es"
    else:
        plural_last = last + "s"
    return " ".join(words[:-1] + [plural_last])


def _prompt_plural_item_repair(key, value, source_text):
    if not isinstance(value, str):
        return None
    if not _ITEM_LIST_ARGUMENT_RE.search(str(key or "")):
        return None
    plural = _simple_plural_variant(value)
    if not plural:
        return None
    if value_appears_in_source(plural, source_text):
        return plural, "prompt_plural_list_item"
    return None


def repair_prompt_plural_list_items(task, candidate):
    """Use prompt plural forms for item/product list entries when explicit."""
    source_text = task.get("prompt", "")
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if not isinstance(value, list):
                continue
            repaired_items = []
            item_changes = []
            for item_index, item in enumerate(value):
                result = _prompt_plural_item_repair(key, item, source_text)
                if result is None:
                    repaired_items.append(item)
                    continue
                repaired, reason = result
                repaired_items.append(repaired)
                item_changes.append((item_index, item, repaired, reason))
            if not item_changes:
                continue
            repaired_arguments[key] = repaired_items
            for item_index, original, repaired, reason in item_changes:
                changes.append(
                    {
                        "index": index,
                        "tool": call.get("name"),
                        "arg": key,
                        "item_index": item_index,
                        "from": original,
                        "to": repaired,
                        "reason": reason,
                    }
                )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "prompt_plural_list_item_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def _prompt_sentences(source_text):
    text = _normalize_space(source_text)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\s+(?:Then|After that|Lastly|Finally),?\s+", text)
        if part.strip()
    ]


def _clean_pickup_location(value):
    text = _normalize_space(value)
    text = re.sub(r"\b(?:and|then|for|on|at|with|from)\b.*$", "", text, flags=re.IGNORECASE)
    text = text.strip(" .,;:!?\"'")
    if not re.match(r"^[A-Za-z][A-Za-z -]{1,30}$", text):
        return ""
    return text.lower()


def _prompt_grounded_pickup_value(arguments, source_text):
    sentences = _prompt_sentences(source_text)
    location_values = [
        value
        for key, value in (arguments or {}).items()
        if isinstance(value, str)
        and re.search(r"(?:^|_)(?:location|city|destination)(?:_|$)", str(key), re.IGNORECASE)
    ]
    candidates = []
    windows = list(sentences or [source_text])
    if location_values and sentences:
        windows.extend(
            _normalize_space(sentences[index] + " " + sentences[index + 1])
            for index in range(0, len(sentences) - 1)
            if any(value_appears_in_source(value, sentences[index]) for value in location_values)
        )
    elif sentences:
        windows.extend(
            _normalize_space(sentences[index] + " " + sentences[index + 1])
            for index in range(0, len(sentences) - 1)
        )
    for window in windows:
        if location_values and not any(value_appears_in_source(value, window) for value in location_values):
            continue
        for match in _PICKUP_PHRASE_RE.finditer(window):
            pickup = _clean_pickup_location(match.group(1))
            if pickup:
                candidates.append(pickup)
    if len(set(candidates)) == 1:
        return candidates[0]
    if not candidates and not location_values:
        all_candidates = [
            _clean_pickup_location(match.group(1))
            for match in _PICKUP_PHRASE_RE.finditer(source_text)
        ]
        all_candidates = [candidate for candidate in all_candidates if candidate]
        if len(set(all_candidates)) == 1:
            return all_candidates[0]
    return ""


def repair_prompt_grounded_pickup_arguments(task, candidate):
    """Fill missing pickup locations when grounded in the matching prompt sentence."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    source_text = task.get("prompt", "")
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, schema in properties.items():
            if key in repaired_arguments:
                continue
            if not _PICKUP_ARGUMENT_RE.match(str(key or "")):
                continue
            if not _schema_expects_string(schema or {}):
                continue
            pickup = _prompt_grounded_pickup_value(arguments, source_text)
            if not pickup:
                continue
            repaired_arguments[key] = pickup
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "to": pickup,
                    "reason": "prompt_grounded_pickup_location",
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "prompt_grounded_pickup_fill": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def _schema_expects_polynomial_function(schema):
    text = str((schema or {}).get("description") or "").lower()
    return "polynomial" in text or "function" in text


def _tool_supports_derivative_repair(tool, key, schema):
    text = " ".join(
        [
            str((tool or {}).get("name") or ""),
            str((tool or {}).get("description") or ""),
            str(key or ""),
            str((schema or {}).get("description") or ""),
        ]
    ).lower()
    return "derivative" in text and str(key or "").lower() == "function"


def _format_polynomial_number(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("%0.10f" % value).rstrip("0").rstrip(".")


def _format_polynomial_term(coefficient, power):
    magnitude = abs(coefficient)
    coeff = _format_polynomial_number(magnitude)
    if power == 0:
        return coeff
    if power == 1:
        return "x" if coeff == "1" else coeff + "*x"
    return "x**%d" % power if coeff == "1" else coeff + "*x**%d" % power


def _derive_simple_polynomial(function_text):
    text = _normalize_space(function_text)
    if not text:
        return None
    text = re.sub(r"^lambda\s+x\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("^", "**")
    if re.search(r"[^0-9xX+\-*.()\s]", text):
        return None
    compact = text.replace(" ", "").replace("X", "x")
    compact = compact.replace("(", "").replace(")", "")
    if not compact or "x" not in compact:
        return None
    terms = re.findall(r"[+-]?[^+-]+", compact)
    if not terms:
        return None
    derived = []
    for term in terms:
        if not term:
            continue
        sign = -1.0 if term.startswith("-") else 1.0
        body = term[1:] if term[0] in "+-" else term
        if "x" not in body:
            if re.fullmatch(r"\d+(?:\.\d+)?", body):
                continue
            return None
        match = re.fullmatch(
            r"(?:(\d+(?:\.\d+)?)\*?)?x(?:\*\*(\d+))?",
            body,
        )
        if not match:
            return None
        coefficient = float(match.group(1)) if match.group(1) else 1.0
        power = int(match.group(2)) if match.group(2) else 1
        if power <= 0:
            continue
        derived.append((sign * coefficient * power, power - 1))
    if not derived:
        return "0"
    parts = []
    for coefficient, power in derived:
        if abs(coefficient) < 1e-9:
            continue
        term_text = _format_polynomial_term(coefficient, power)
        if not parts:
            parts.append(("-" if coefficient < 0 else "") + term_text)
        else:
            parts.append((" - " if coefficient < 0 else " + ") + term_text)
    return "".join(parts) if parts else "0"


def _derived_polynomial_repair_value(value, previous_function):
    if not isinstance(value, str) or not isinstance(previous_function, str):
        return None
    if _DERIVED_RESULT_PLACEHOLDER_RE.search(value.strip()) is None:
        return None
    derived = _derive_simple_polynomial(previous_function)
    if not derived:
        return None
    return derived, "previous_polynomial_derivative"


def repair_derived_polynomial_arguments(task, candidate):
    """Replace derivative result placeholders with simple derived polynomials."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    previous_function_by_tool = {}
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            schema = properties.get(key) or {}
            if not _tool_supports_derivative_repair(tool, key, schema):
                continue
            if not _schema_expects_polynomial_function(schema):
                continue
            previous_function = previous_function_by_tool.get(call.get("name"))
            result = _derived_polynomial_repair_value(value, previous_function)
            if result is None:
                continue
            repaired, reason = result
            repaired_arguments[key] = repaired
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": repaired,
                    "reason": reason,
                }
            )
        if isinstance(repaired_arguments.get("function"), str):
            previous_function_by_tool[call.get("name")] = repaired_arguments["function"]
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "derived_polynomial_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def repair_generic_entity_suffixes(task, candidate):
    """Trim ungrounded generic suffixes from entity-like string arguments."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    source_text = task.get("prompt", "")
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if not isinstance(value, str):
                continue
            if not _ENTITY_SUFFIX_ARGUMENT_RE.search(str(key)):
                continue
            if key in properties and not _schema_expects_string(properties.get(key) or {}):
                continue
            trimmed = _entity_suffix_trim(value, source_text)
            if not trimmed:
                continue
            repaired_arguments[key] = trimmed
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": trimmed,
                    "reason": "ungrounded_generic_entity_suffix",
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "entity_suffix_span_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def repair_temporal_values(task, candidate):
    """Repair datetime strings when the schema explicitly excludes timezone suffixes."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            schema = properties.get(key) if isinstance(properties, dict) else {}
            if key in properties and not _schema_expects_string(schema or {}):
                continue
            result = _temporal_value_repair(key, value, schema or {})
            if result is None:
                continue
            repaired, reason = result
            repaired_arguments[key] = repaired
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": repaired,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "temporal_value_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def repair_city_values(task, candidate):
    """Trim state/country qualifiers only when the schema asks for a bare city name."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    source_text = task.get("prompt", "")
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            schema = properties.get(key) if isinstance(properties, dict) else {}
            if key in properties and not _schema_expects_string(schema or {}):
                continue
            result = _city_value_repair(key, value, schema or {}, source_text)
            if result is None:
                continue
            repaired, reason = result
            repaired_arguments[key] = repaired
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": repaired,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "city_value_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def repair_context_entity_values(task, candidate):
    """Fix narrow entity value expansions that are contradicted by prompt spans."""
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    repaired_calls = []
    changes = []
    source_text = task.get("prompt", "")
    for index, call in enumerate(candidate.get("normalized_calls") or []):
        tool = tool_by_name.get(call.get("name")) or {}
        properties = (tool.get("parameters") or {}).get("properties") or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            repaired_calls.append(call)
            continue
        repaired_arguments = dict(arguments)
        for key, value in list(arguments.items()):
            if key in properties and not _schema_expects_string(properties.get(key) or {}):
                continue
            result = _context_entity_value_repair(key, value, source_text)
            if result is None:
                continue
            repaired, reason = result
            repaired_arguments[key] = repaired
            changes.append(
                {
                    "index": index,
                    "tool": call.get("name"),
                    "arg": key,
                    "from": value,
                    "to": repaired,
                    "reason": reason,
                }
            )
        repaired_calls.append({**call, "arguments": repaired_arguments})
    if not changes:
        return candidate
    return {
        **candidate,
        "normalized_calls": repaired_calls,
        "context_entity_value_repair": {
            "changes": changes,
            "changed_count": len(changes),
        },
    }


def run_relevance_verification(task, args, indexed_calls):
    if not indexed_calls:
        return {
            "calls": [],
            "dropped": [],
            "latency_ms": 0,
            "raw": None,
            "usage": None,
            "parse_failed": False,
        }

    candidate_names = {item["call"]["name"] for item in indexed_calls}
    schemas = [
        tool
        for tool in task["tools"]
        if tool["name"] in candidate_names
    ]
    candidates = [
        {
            "index": item["original_index"],
            "name": item["call"]["name"],
            "arguments": item["call"]["arguments"],
        }
        for item in indexed_calls
    ]
    payload = {
        "model": args.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{task['prompt']}\n\n"
                    f"Candidate calls:\n{json.dumps(candidates, sort_keys=True)}\n\n"
                    f"Full schemas for candidate tools:\n{json.dumps(schemas, sort_keys=True)}"
                ),
            },
        ],
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    parsed = parse_json_object(assistant_text(raw))
    if parsed is None or not isinstance(parsed.get("drop"), list):
        return {
            "calls": [item["call"] for item in indexed_calls],
            "dropped": [],
            "latency_ms": latency_ms,
            "raw": raw,
            "usage": request_usage("verification", raw),
            "parse_failed": True,
        }

    allowed_indices = {item["original_index"] for item in indexed_calls}
    drop_indices = set()
    dropped = []
    for item in parsed["drop"]:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        reason = str(item.get("reason") or "")
        if index in allowed_indices and reason == "clearly_irrelevant":
            drop_indices.add(index)
            dropped.append({"index": index, "reason": reason})

    return {
        "calls": [
            item["call"]
            for item in indexed_calls
            if item["original_index"] not in drop_indices
        ],
        "dropped": dropped,
        "latency_ms": latency_ms,
        "raw": raw,
        "usage": request_usage("verification", raw),
        "parse_failed": False,
    }


def verify_candidate_calls(task, args, calls):
    indexed, deterministic_drops, merged_count, argument_drops = deterministic_verify_calls(
        calls,
        task["tools"],
        task["prompt"],
        drop_ungrounded_optional_args=getattr(
            args,
            "drop_ungrounded_optional_args",
            False,
        ),
    )
    relevance = run_relevance_verification(task, args, indexed)
    return {
        "calls": dedupe_tool_calls(relevance["calls"]),
        "dropped": deterministic_drops + relevance["dropped"],
        "dropped_count": len(deterministic_drops) + len(relevance["dropped"]),
        "argument_drops": argument_drops,
        "argument_dropped_count": len(argument_drops),
        "merged_count": merged_count,
        "latency_ms": relevance["latency_ms"],
        "raw": relevance["raw"],
        "usage": relevance["usage"],
        "parse_failed": relevance["parse_failed"],
    }


def run_abstention_guard(task, args, calls):
    if not getattr(args, "irrelevance_guard", False):
        return {
            "calls": calls,
            "decision": "not_run",
            "reason": "",
            "latency_ms": 0,
            "raw": None,
            "usage": None,
            "parse_failed": False,
        }

    if not calls:
        return {
            "calls": [],
            "decision": "no_calls",
            "reason": "",
            "latency_ms": 0,
            "raw": None,
            "usage": None,
            "parse_failed": False,
        }

    candidate_names = {call.get("name") for call in calls}
    schemas = [
        tool
        for tool in task["tools"]
        if tool["name"] in candidate_names
    ]
    payload = {
        "model": args.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": ABSTENTION_GUARD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{task['prompt']}\n\n"
                    f"Candidate calls after normal verification:\n"
                    f"{json.dumps(calls, sort_keys=True)}\n\n"
                    f"Schemas for candidate tools:\n"
                    f"{json.dumps(schemas, sort_keys=True)}"
                ),
            },
        ],
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    parsed = parse_json_object(assistant_text(raw))
    if parsed is None:
        return {
            "calls": calls,
            "decision": "keep",
            "reason": "parse_failed_keep",
            "latency_ms": latency_ms,
            "raw": raw,
            "usage": request_usage("irrelevance_guard", raw),
            "parse_failed": True,
        }

    decision = normalize_text(
        parsed.get("decision")
        or parsed.get("verdict")
        or parsed.get("action")
        or "keep"
    )
    reason = str(parsed.get("reason") or "")
    if decision == "abstain":
        return {
            "calls": [],
            "decision": "abstain",
            "reason": reason,
            "latency_ms": latency_ms,
            "raw": raw,
            "usage": request_usage("irrelevance_guard", raw),
            "parse_failed": False,
        }

    return {
        "calls": calls,
        "decision": "keep",
        "reason": reason,
        "latency_ms": latency_ms,
        "raw": raw,
        "usage": request_usage("irrelevance_guard", raw),
        "parse_failed": False,
    }


def run_abstention_prefilter(task, args, plan):
    if not getattr(args, "irrelevance_prefilter", False):
        return {
            "decision": "not_run",
            "reason": "",
            "latency_ms": 0,
            "raw": None,
            "usage": None,
            "parse_failed": False,
        }

    if plan.get("verdict") == "no_intent":
        return {
            "decision": "already_no_intent",
            "reason": "",
            "latency_ms": 0,
            "raw": None,
            "usage": None,
            "parse_failed": False,
        }

    payload = {
        "model": args.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": ABSTENTION_PREFILTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{task['prompt']}\n\n"
                    f"Planner verdict:\n{plan.get('verdict')}\n\n"
                    f"Available tool schemas:\n"
                    f"{json.dumps(task['tools'], sort_keys=True)}"
                ),
            },
        ],
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    parsed = parse_json_object(assistant_text(raw))
    if parsed is None:
        return {
            "decision": "keep",
            "reason": "parse_failed_keep",
            "latency_ms": latency_ms,
            "raw": raw,
            "usage": request_usage("irrelevance_prefilter", raw),
            "parse_failed": True,
        }

    decision = normalize_text(
        parsed.get("decision")
        or parsed.get("verdict")
        or parsed.get("action")
        or "keep"
    )
    reason = str(parsed.get("reason") or "")
    if decision == "abstain":
        return {
            "decision": "abstain",
            "reason": reason,
            "latency_ms": latency_ms,
            "raw": raw,
            "usage": request_usage("irrelevance_prefilter", raw),
            "parse_failed": False,
        }

    return {
        "decision": "keep",
        "reason": reason,
        "latency_ms": latency_ms,
        "raw": raw,
        "usage": request_usage("irrelevance_prefilter", raw),
        "parse_failed": False,
    }


def parse_decomposition(text, tools, max_calls=4):
    available = {tool["name"]: tool for tool in tools}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    verdict = None
    if lines:
        match = re.search(
            r"\bverdict\s*=\s*(no_intent|single_intent|multi_intent)\b",
            lines[0],
            flags=re.IGNORECASE,
        )
        if match:
            verdict = match.group(1).lower()

    call_set = []
    seen = set()
    for line in lines[1:]:
        tool_match = re.search(r"\btool\s*=\s*([^;\s]+)", line)
        if not tool_match:
            continue
        tool_name = tool_match.group(1).strip()
        if tool_name not in available:
            continue

        group_match = re.search(
            r"\bargument_group\s*=\s*(.*?)(?:;\s*objective\s*=|$)",
            line,
            flags=re.IGNORECASE,
        )
        objective_match = re.search(
            r"\bobjective\s*=\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )
        argument_group = (
            group_match.group(1).strip()
            if group_match and group_match.group(1).strip()
            else "unspecified"
        )
        objective = (
            objective_match.group(1).strip()
            if objective_match and objective_match.group(1).strip()
            else "unspecified"
        )
        key = (tool_name, normalize_text(argument_group))
        if key in seen:
            continue
        seen.add(key)
        call_set.append(
            {
                "tool": available[tool_name],
                "tool_name": tool_name,
                "argument_group": argument_group,
                "objective": objective,
                "line": (
                    f"{len(call_set) + 1}. tool={tool_name}; "
                    f"argument_group={argument_group}; objective={objective}"
                ),
            }
        )
        if len(call_set) >= max_calls:
            break

    if verdict not in {"no_intent", "single_intent", "multi_intent"}:
        verdict = "single_intent"
    if verdict == "multi_intent" and len(call_set) < 2:
        verdict = "single_intent"
        call_set = []
    elif verdict != "multi_intent":
        call_set = []

    normalized_lines = [f"verdict={verdict}"]
    normalized_lines.extend(item["line"] for item in call_set)
    return {
        "verdict": verdict,
        "call_set": call_set,
        "decomposition": "\n".join(normalized_lines),
    }


def assistant_text(raw):
    message = (((raw.get("choices") or [{}])[0]).get("message") or {})
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


def run_warmup(args):
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 4,
        "messages": [
            {"role": "system", "content": "Reply with OK."},
            {"role": "user", "content": "OK"},
        ],
    }
    _, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    return latency_ms


def run_baseline(task, args):
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ],
        "tools": openai_tools(task["tools"]),
        "tool_choice": "auto",
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    calls = normalize_tool_calls(raw)
    usages = [request_usage("baseline_selector", raw)]
    return {
        "latency_ms": latency_ms,
        "normalized_calls": calls,
        "score": score_calls(calls, task.get("expected_calls", [])),
        "raw": raw,
        "request_usages": usages,
        "usage": aggregate_request_usages(usages),
    }


def run_pipeline_single_selector(task, args):
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Pipeline single-intent selector. The original user request "
                    "is authoritative. Emit zero or one tool call using only the "
                    "provided schema. "
                    f"{selector_value_guidance(args, task['prompt'])}"
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ],
        "tools": openai_tools(task["tools"]),
        "tool_choice": "auto",
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    calls = normalize_tool_calls(raw)[:1]
    usages = [request_usage("pipeline_single_selector", raw)]
    return {
        "latency_ms": latency_ms,
        "normalized_calls": calls,
        "score": score_calls(calls, task.get("expected_calls", [])),
        "raw": raw,
        "request_usages": usages,
        "usage": aggregate_request_usages(usages),
    }


def run_react_baseline(task, args):
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ],
        "tools": openai_tools(task["tools"]),
        "tool_choice": "auto",
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    calls = normalize_tool_calls(raw)
    usages = [request_usage("react_selector", raw)]
    return {
        "latency_ms": latency_ms,
        "normalized_calls": calls,
        "score": score_calls(calls, task.get("expected_calls", [])),
        "raw": raw,
        "request_usages": usages,
        "usage": aggregate_request_usages(usages),
    }


def run_decomposition(task, args):
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Available tools:\n{planning_tool_inventory(task['tools'])}\n\nUser request:\n{task['prompt']}",
            },
        ],
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    parsed = parse_decomposition(assistant_text(raw), task["tools"])
    return {
        "latency_ms": latency_ms,
        "raw": raw,
        "usage": request_usage("intent_planner", raw),
        **parsed,
    }


def _multseq_step_tokens(step):
    text = "%s %s" % (
        step.get("argument_group", ""),
        step.get("objective", ""),
    )
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def _multseq_step_similarity(left, right):
    if left.get("tool_name") != right.get("tool_name"):
        return 0.0
    left_tokens = _multseq_step_tokens(left)
    right_tokens = _multseq_step_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return float(len(left_tokens & right_tokens)) / float(len(union) or 1)


def _multseq_plan_similarity(left, right):
    left_steps = list(left.get("call_set") or [])
    right_steps = list(right.get("call_set") or [])
    verdict_bonus = 0.2 if left.get("verdict") == right.get("verdict") else 0.0
    if not left_steps and not right_steps:
        return verdict_bonus + 0.8
    if not left_steps or not right_steps:
        return verdict_bonus

    left_match = sum(
        max(_multseq_step_similarity(step, other) for other in right_steps)
        for step in left_steps
    ) / float(len(left_steps))
    right_match = sum(
        max(_multseq_step_similarity(step, other) for other in left_steps)
        for step in right_steps
    ) / float(len(right_steps))
    return verdict_bonus + 0.4 * (left_match + right_match)


def select_multseq_medoid(plans):
    """Return the coherent sampled plan with the highest cross-plan agreement."""
    if not plans:
        return None, [], None
    totals = []
    for index, plan in enumerate(plans):
        pair_scores = [
            _multseq_plan_similarity(plan, other)
            for other_index, other in enumerate(plans)
            if other_index != index
        ]
        totals.append(sum(pair_scores) / float(len(pair_scores) or 1))
    selected_index = max(range(len(plans)), key=lambda index: (totals[index], -index))
    return plans[selected_index], totals, selected_index


def _multseq_step_grounding(step, prompt):
    tokens = _multseq_step_tokens(step)
    if not tokens:
        return 0.5
    prompt_tokens = _router_tokens(prompt)
    if not prompt_tokens:
        return 0.0
    return len(tokens & prompt_tokens) / float(len(tokens))


def _multseq_duplicate_penalty(plan):
    steps = list(plan.get("call_set") or [])
    penalty = 0.0
    for left_index, left in enumerate(steps):
        for right in steps[left_index + 1 :]:
            if left.get("tool_name") != right.get("tool_name"):
                continue
            similarity = _multseq_step_similarity(left, right)
            if similarity >= 0.85:
                penalty += 0.5
            elif similarity >= 0.6:
                penalty += 0.2
    return penalty


def select_multseq_score_filter(task, plans):
    """Select a sampled plan with inference-time score-and-filter signals."""
    if not plans:
        return None, [], None, []

    _, consensus_scores, _ = select_multseq_medoid(plans)
    verdict_counts = {}
    call_counts = []
    for plan in plans:
        verdict_counts[plan.get("verdict")] = verdict_counts.get(plan.get("verdict"), 0) + 1
        call_counts.append(len(plan.get("call_set") or []))
    sorted_counts = sorted(call_counts)
    median_count = sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0

    scored = []
    for index, plan in enumerate(plans):
        steps = list(plan.get("call_set") or [])
        verdict = plan.get("verdict")
        count = len(steps)
        grounding = (
            sum(_multseq_step_grounding(step, task.get("prompt", "")) for step in steps)
            / float(len(steps))
            if steps
            else 0.5
        )
        verdict_support = verdict_counts.get(verdict, 0) / float(len(plans))
        count_distance = abs(count - median_count) / float(max(median_count, 1))
        duplicate_penalty = _multseq_duplicate_penalty(plan)
        score = (
            1.2 * consensus_scores[index]
            + 0.7 * grounding
            + 0.4 * verdict_support
            - 0.35 * count_distance
            - duplicate_penalty
        )
        if verdict == "multi_intent" and count < 2:
            score -= 1.0
        if median_count >= 2 and count >= median_count:
            score += 0.15
        scored.append(
            {
                "index": index,
                "score": round(score, 6),
                "consensus_score": round(consensus_scores[index], 6),
                "grounding": round(grounding, 6),
                "verdict_support": round(verdict_support, 6),
                "call_set_size": count,
                "count_distance": round(count_distance, 6),
                "duplicate_penalty": round(duplicate_penalty, 6),
            }
        )

    selected_index = max(
        range(len(plans)),
        key=lambda index: (scored[index]["score"], scored[index]["consensus_score"], -index),
    )
    return plans[selected_index], consensus_scores, selected_index, scored


def run_multseq_decomposition(task, args):
    """Sample plans, then combine subtasks or select a consensus medoid plan.

    The vote strategy keeps subtasks that appear in enough plans. The medoid
    strategy returns one sampled plan with the highest average similarity to all
    other plans, preserving its original grouping instead of synthesizing a plan.

    Falls back to a single run_decomposition() call when multseq_k <= 1.
    """
    k = getattr(args, "multseq_k", 1)
    if k <= 1:
        return run_decomposition(task, args)

    diversity_temp = getattr(args, "multseq_temperature", 0.5)
    min_votes = getattr(args, "multseq_min_votes", 2)
    strategy = getattr(args, "multseq_strategy", "vote")

    plans = []
    total_latency = 0
    all_usages = []
    for _ in range(k):
        payload = {
            "model": args.model,
            "temperature": diversity_temp,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Available tools:\n{planning_tool_inventory(task['tools'])}\n\n"
                        f"User request:\n{task['prompt']}"
                    ),
                },
            ],
        }
        raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
        parsed = parse_decomposition(assistant_text(raw), task["tools"])
        usage = request_usage("intent_planner_multseq", raw)
        total_latency += latency_ms
        all_usages.append(usage)
        plans.append({"latency_ms": latency_ms, "raw": raw, "usage": usage, **parsed})

    if strategy in {"medoid", "score_filter", "saf"}:
        filter_scores = []
        if strategy == "medoid":
            selected, consensus_scores, selected_index = select_multseq_medoid(plans)
        else:
            selected, consensus_scores, selected_index, filter_scores = (
                select_multseq_score_filter(task, plans)
            )
        if selected is None:
            return run_decomposition(task, args)
        return {
            "latency_ms": total_latency,
            "raw": selected["raw"],
            "usage": aggregate_request_usages(all_usages),
            "verdict": selected["verdict"],
            "call_set": selected["call_set"],
            "decomposition": selected["decomposition"],
            "multseq_strategy": strategy,
            "multseq_selected_plan": selected_index,
            "multseq_consensus_scores": [
                round(score, 6) for score in consensus_scores
            ],
            "multseq_filter_scores": filter_scores,
            "multseq_plans": [
                {
                    "verdict": plan["verdict"],
                    "call_set_size": len(plan.get("call_set") or []),
                    "consensus_score": round(consensus_scores[index], 6),
                    "filter_score": (
                        filter_scores[index]["score"] if filter_scores else None
                    ),
                }
                for index, plan in enumerate(plans)
            ],
        }

    # Merged verdict: prefer multi_intent if any plan says so, else majority.
    verdicts = [p["verdict"] for p in plans]
    if "multi_intent" in verdicts:
        merged_verdict = "multi_intent"
    else:
        merged_verdict = max(set(verdicts), key=verdicts.count)

    # Majority-vote on subtasks: count by (tool_name, normalized argument_group).
    vote_counts: dict = {}
    key_to_step: dict = {}
    key_order: list = []
    for plan in plans:
        for step in (plan.get("call_set") or []):
            key = (step["tool_name"], normalize_text(step["argument_group"]))
            if key not in vote_counts:
                vote_counts[key] = 0
                key_to_step[key] = step
                key_order.append(key)
            vote_counts[key] += 1

    # Keep subtasks that meet the vote threshold (capped so min_votes <= k).
    threshold = min(min_votes, k)
    kept = [key_to_step[key] for key in key_order if vote_counts[key] >= threshold]

    # Rebuild sequential line numbers.
    kept = [
        {
            **step,
            "line": (
                f"{i + 1}. tool={step['tool_name']}; "
                f"argument_group={step['argument_group']}; objective={step['objective']}"
            ),
        }
        for i, step in enumerate(kept)
    ]

    # Enforce multi_intent requires >= 2 subtasks (mirrors parse_decomposition logic).
    if merged_verdict == "multi_intent" and len(kept) < 2:
        merged_verdict = "single_intent"
        kept = []
    elif merged_verdict != "multi_intent":
        kept = []

    decomposition_text = f"verdict={merged_verdict}\n" + "\n".join(
        s["line"] for s in kept
    )

    return {
        "latency_ms": total_latency,
        "raw": plans[0]["raw"],
        "usage": aggregate_request_usages(all_usages),
        "verdict": merged_verdict,
        "call_set": kept,
        "decomposition": decomposition_text,
        "multseq_strategy": strategy,
        # Store per-plan summaries for post-hoc analysis.
        "multseq_plans": [
            {
                "verdict": p["verdict"],
                "call_set_size": len(p.get("call_set") or []),
                "vote_counts": {
                    f"{s['tool_name']}|{normalize_text(s['argument_group'])}": vote_counts.get(
                        (s["tool_name"], normalize_text(s["argument_group"])), 0
                    )
                    for s in (p.get("call_set") or [])
                },
            }
            for p in plans
        ],
    }


def run_cardinality_repair(task, args, plan, max_calls=8):
    category = bfcl_category(task)
    system_prompt = (
        CARDINALITY_PARALLEL_REPAIR_SYSTEM_PROMPT
        if category in {"parallel", "parallel_multiple"}
        and plan.get("verdict") == "multi_intent"
        else CARDINALITY_REPAIR_SYSTEM_PROMPT
    )
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"BFCL category:\n{category or 'unknown'}\n\n"
                    f"Original user request:\n{task['prompt']}\n\n"
                    f"Schema cardinality hints:\n{cardinality_tool_inventory(task['tools'])}\n\n"
                    f"Current call-set plan:\n{plan['decomposition']}"
                ),
            },
        ],
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    parsed = parse_decomposition(assistant_text(raw), task["tools"], max_calls=max_calls)
    return {
        "latency_ms": latency_ms,
        "raw": raw,
        "usage": request_usage("cardinality_repair", raw),
        **parsed,
    }


def bfcl_category(task):
    task_id = str(task.get("id", ""))
    for category in (
        "parallel_multiple",
        "parallel",
        "multiple",
        "irrelevance",
    ):
        if task_id.startswith(f"{category}_"):
            return category
    return ""


def select_cardinality_plan(task, original_plan, repair):
    """Choose whether the cardinality-repaired plan is safe to use.

    The repair model is advisory. In parallel categories, collapsing a multi-call
    plan to single_intent sends the lane through the direct selector and loses
    the planned call-set structure. Reject those downgrades.
    """

    if repair is None:
        return original_plan, False, "not_run"

    category = bfcl_category(task)
    original_verdict = original_plan.get("verdict")
    repair_verdict = repair.get("verdict")
    original_count = len(original_plan.get("call_set") or [])
    repair_count = len(repair.get("call_set") or [])

    if original_verdict == "no_intent":
        return original_plan, False, "original_no_intent"

    if category in {"parallel", "parallel_multiple"}:
        if original_verdict == "multi_intent" and repair_verdict != "multi_intent":
            return original_plan, False, "rejected_parallel_verdict_downgrade"
        if repair_verdict == "multi_intent" and repair_count < 2:
            return original_plan, False, "rejected_parallel_too_few_calls"
        if repair_verdict == "multi_intent" and repair_count == original_count:
            return original_plan, False, "rejected_parallel_same_count"

    if repair_verdict == "multi_intent" and repair_count >= 2:
        return repair, True, "accepted_multi_intent"

    if category == "multiple" and repair_verdict in {"single_intent", "no_intent"}:
        return repair, True, f"accepted_{repair_verdict}"

    return original_plan, False, "rejected_unhelpful_repair"


def planned_steps(plan, tools):
    return list(plan.get("call_set") or [])


def run_context_pipeline(task, args, plan):
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Task plan context generated before tool selection. "
                    "This is not a tool result and no tools have been executed yet. "
                    "The original user request is authoritative if the plan is incomplete or ambiguous. "
                    "Treat this plan as advisory context only. Do not copy field names from the plan. "
                    "Use the original request and full tool schemas when deciding final tool calls and exact argument names. "
                    "Use the plan as a checklist for coverage: consider every line with tool=<available tool>, "
                    "and do not stop after the first tool if the original request contains multiple independent tasks. "
                    "For each checklist line with tool=<available tool>, make zero or one call: zero only when the original request lacks required arguments or the tool is not actually relevant, one when the original request provides the needed facts. "
                    "Never repeat the same tool call with identical arguments.\n"
                    f"{selector_value_guidance(args, task['prompt'])}"
                    f"{plan['decomposition']}"
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ],
        "tools": openai_tools(task["tools"]),
        "tool_choice": "auto",
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    calls = normalize_tool_calls(raw)
    usages = [plan["usage"], request_usage("context_selector", raw)]
    return {
        "latency_ms": latency_ms,
        "total_latency_ms": latency_ms + plan["latency_ms"],
        "decomposition_latency_ms": plan["latency_ms"],
        "decomposition": plan["decomposition"],
        "normalized_calls": calls,
        "score": score_calls(calls, task.get("expected_calls", [])),
        "raw": raw,
        "decomposition_raw": plan["raw"],
        "mode": "context",
        "request_usages": usages,
        "usage": aggregate_request_usages(usages),
    }


def run_subtask_tool_selection(task, args, step, prior_calls, retry=False):
    prior_context = json.dumps(prior_calls, sort_keys=True)
    retry_instruction = (
        "A previous attempt for this same subtask emitted no tool call. "
        "Re-evaluate the original request and schema. If every required argument is grounded, "
        "emit exactly one call now. Omit ungrounded optional arguments. "
        if retry
        else ""
    )
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "messages": [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "You are selecting tools for exactly one planned subtask. "
                    "This subtask plan is not a tool result. The original user request is authoritative. "
                    "Use the provided tool schema for exact argument names. Do not copy field names from the plan. "
                    "Previously selected calls are not executed results; use them only as hints for consistent argument values across subtasks. "
                    "Prior selected calls are planning context, not actual tool results. "
                    f"{retry_instruction}"
                    "When the original request explicitly provides a literal number, string, list item, identifier, URL-like token, date, unit, or named value, preserve that value exactly unless the schema requires a different primitive type. "
                    "Do not paraphrase, singularize, pluralize, normalize, compute, or replace explicit user-provided values. "
                    "Variable-like identifiers explicitly written by the user, such as file names, URLs, record IDs, or text placeholders, are grounded values and should be passed through exactly. "
                    "Include every explicitly stated constraint that applies to this call, including optional arguments such as formats, sorting flags, modes, categories, and meal/type labels. "
                    "Do not infer a combined enum or option such as 'both' when the request names separate variants; emit only the one variant assigned to this subtask. "
                    "Never emit null for an omitted optional argument. Never fabricate the output of a prior tool call. If an optional argument would require an unexecuted tool result or an unstated value, omit that optional argument. "
                    "If all required arguments for the planned tool are explicitly grounded, emit the call instead of declining it. "
                    "Emit zero or one tool call for this subtask.\n\n"
                    f"{selector_value_guidance(args, task['prompt'])}"
                    f"{READINESS_POLICY}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{task['prompt']}\n\n"
                    f"Prior selected calls for this request, not executed results:\n{prior_context}\n\n"
                    f"Current planned subtask:\n{step['line']}"
                ),
            },
        ],
        "tools": openai_tools([step["tool"]]),
        "tool_choice": "auto",
    }
    raw, latency_ms = post_chat(args.endpoint, args.api_key, payload, args.timeout)
    normalized_calls = normalize_tool_calls(raw)
    return {
        "latency_ms": latency_ms,
        "raw": raw,
        "calls": normalized_calls[:1],
        "discarded_call_count": max(0, len(normalized_calls) - 1),
        "usage": request_usage(
            f"subtask_selector{'_retry' if retry else ''}:{step['tool']['name']}",
            raw,
        ),
        "step": step["line"],
        "tool": step["tool"]["name"],
        "retry": retry,
    }


def run_per_subtask_pipeline(task, args, plan):
    steps = planned_steps(plan, task["tools"])
    subtask_results = []
    prior_calls = []
    for step in steps:
        result = run_subtask_tool_selection(task, args, step, prior_calls)
        result["request_usages"] = [result["usage"]]
        result["retry_attempted"] = False
        if not result["calls"]:
            retry_result = run_subtask_tool_selection(
                task,
                args,
                step,
                prior_calls,
                retry=True,
            )
            result["retry_attempted"] = True
            result["retry_raw"] = retry_result["raw"]
            result["latency_ms"] += retry_result["latency_ms"]
            result["calls"] = retry_result["calls"]
            result["discarded_call_count"] += retry_result["discarded_call_count"]
            result["request_usages"].append(retry_result["usage"])
        subtask_results.append(result)
        prior_calls = dedupe_tool_calls(prior_calls + result["calls"])
    calls = dedupe_tool_calls([call for result in subtask_results for call in result["calls"]])
    latency_ms = sum(result["latency_ms"] for result in subtask_results)
    usages = [plan["usage"]] + [
        usage
        for result in subtask_results
        for usage in result["request_usages"]
    ]
    return {
        "latency_ms": latency_ms,
        "total_latency_ms": latency_ms + plan["latency_ms"],
        "decomposition_latency_ms": plan["latency_ms"],
        "decomposition": plan["decomposition"],
        "normalized_calls": calls,
        "score": score_calls(calls, task.get("expected_calls", [])),
        "raw": {"mode": "per-subtask", "subtask_results": subtask_results},
        "decomposition_raw": plan["raw"],
        "mode": "per-subtask",
        "request_usages": usages,
        "usage": aggregate_request_usages(usages),
    }


def run_pipeline(task, args):
    plan = (
        run_multseq_decomposition(task, args)
        if getattr(args, "multseq_k", 1) > 1
        else run_decomposition(task, args)
    )
    verdict = plan["verdict"]
    planner_metadata = {
        key: plan[key]
        for key in (
            "multseq_strategy",
            "multseq_selected_plan",
            "multseq_consensus_scores",
            "multseq_filter_scores",
            "multseq_plans",
        )
        if key in plan
    }
    prefilter = run_abstention_prefilter(task, args, plan)
    if prefilter["decision"] == "abstain":
        usages = [plan["usage"]]
        if prefilter["usage"] is not None:
            usages.append(prefilter["usage"])
        return {
            "latency_ms": 0,
            "total_latency_ms": plan["latency_ms"] + prefilter["latency_ms"],
            "decomposition_latency_ms": plan["latency_ms"],
            "irrelevance_prefilter_latency_ms": prefilter["latency_ms"],
            "decomposition": plan["decomposition"],
            "normalized_calls": [],
            "score": score_calls([], task.get("expected_calls", [])),
            "raw": {"mode": "irrelevance_prefilter_abstain"},
            "decomposition_raw": plan["raw"],
            "mode": "irrelevance_prefilter_abstain",
            "request_usages": usages,
            "usage": aggregate_request_usages(usages),
            "verdict": verdict,
            **planner_metadata,
            "irrelevance_prefilter": {
                "enabled": getattr(args, "irrelevance_prefilter", False),
                "decision": prefilter["decision"],
                "reason": prefilter["reason"],
                "parse_failed": prefilter["parse_failed"],
                "raw": prefilter["raw"],
            },
        }

    if verdict == "no_intent":
        candidate = {
            "latency_ms": 0,
            "total_latency_ms": plan["latency_ms"],
            "decomposition_latency_ms": plan["latency_ms"],
            "irrelevance_prefilter_latency_ms": prefilter["latency_ms"],
            "decomposition": plan["decomposition"],
            "normalized_calls": [],
            "raw": {"mode": "no_intent"},
            "decomposition_raw": plan["raw"],
            "mode": "no_intent",
            "request_usages": [plan["usage"]],
            "usage": aggregate_request_usages([plan["usage"]]),
        }
    elif verdict == "single_intent":
        selector = (
            run_pipeline_single_selector(task, args)
            if getattr(args, "strict_value_copy", False)
            or getattr(args, "value_copy_fewshot", False)
            or getattr(args, "span_inventory", False)
            else run_baseline(task, args)
        )
        usages = [plan["usage"]] + selector["request_usages"]
        candidate = {
            "latency_ms": selector["latency_ms"],
            "total_latency_ms": plan["latency_ms"] + selector["latency_ms"],
            "decomposition_latency_ms": plan["latency_ms"],
            "irrelevance_prefilter_latency_ms": prefilter["latency_ms"],
            "decomposition": plan["decomposition"],
            "normalized_calls": selector["normalized_calls"],
            "raw": {"mode": "single_intent", "selector": selector["raw"]},
            "decomposition_raw": plan["raw"],
            "mode": "single_intent",
            "request_usages": usages,
            "usage": aggregate_request_usages(usages),
        }
    elif args.pipeline_mode == "context":
        candidate = run_context_pipeline(task, args, plan)
    else:
        candidate = run_per_subtask_pipeline(task, args, plan)

    candidate.update(planner_metadata)
    verification = verify_candidate_calls(
        task,
        args,
        candidate["normalized_calls"],
    )
    usages = list(candidate["request_usages"])
    if prefilter["usage"] is not None and prefilter["usage"] not in usages:
        usages.append(prefilter["usage"])
    if verification["usage"] is not None:
        usages.append(verification["usage"])
    guard = run_abstention_guard(task, args, verification["calls"])
    if guard["usage"] is not None:
        usages.append(guard["usage"])
    calls = guard["calls"]
    candidate.update(
        {
            "normalized_calls": calls,
            "score": score_calls(calls, task.get("expected_calls", [])),
            "total_latency_ms": candidate["total_latency_ms"]
            + verification["latency_ms"]
            + prefilter["latency_ms"]
            + guard["latency_ms"],
            "verification_latency_ms": verification["latency_ms"],
            "verification": {
                "dropped": verification["dropped"],
                "dropped_count": verification["dropped_count"],
                "argument_drops": verification["argument_drops"],
                "argument_dropped_count": verification["argument_dropped_count"],
                "merged_count": verification["merged_count"],
                "parse_failed": verification["parse_failed"],
                "raw": verification["raw"],
            },
            "irrelevance_guard_latency_ms": guard["latency_ms"],
            "irrelevance_prefilter_latency_ms": prefilter["latency_ms"],
            "irrelevance_prefilter": {
                "enabled": getattr(args, "irrelevance_prefilter", False),
                "decision": prefilter["decision"],
                "reason": prefilter["reason"],
                "parse_failed": prefilter["parse_failed"],
                "raw": prefilter["raw"],
            },
            "irrelevance_guard": {
                "enabled": getattr(args, "irrelevance_guard", False),
                "decision": guard["decision"],
                "reason": guard["reason"],
                "parse_failed": guard["parse_failed"],
                "raw": guard["raw"],
            },
            "request_usages": usages,
            "usage": aggregate_request_usages(usages),
            "verdict": verdict,
        }
    )
    return candidate


BOOL_GROUNDING_STOPWORDS = {
    "allow",
    "allowed",
    "enable",
    "enabled",
    "flag",
    "has",
    "include",
    "included",
    "is",
    "requested",
    "requires",
    "use",
    "with",
}


def _bool_optional_flag_grounded(key, value, source_text):
    bool_value = _bool_like_value(value)
    if bool_value is None:
        return False
    action_text = _action_source_text(source_text)
    prompt_tokens = _router_tokens(action_text)
    key_tokens = _router_tokens(key) - BOOL_GROUNDING_STOPWORDS
    if not key_tokens:
        return False
    if bool_value is True:
        return bool(prompt_tokens & key_tokens)
    if bool_value is False:
        negation_tokens = {"no", "not", "without", "exclude", "disable", "disabled"}
        return bool(prompt_tokens & key_tokens) and bool(prompt_tokens & negation_tokens)
    return False


def _bool_like_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _schema_is_bool_like_optional_flag(schema):
    enum_values = schema.get("enum") if isinstance(schema, dict) else None
    if not isinstance(enum_values, list):
        return False
    lowered = {str(item).strip().lower() for item in enum_values}
    return {"true", "false"}.issubset(lowered)


def _router_leaf_grounding(value, schema, source_text, key=""):
    enum_values = schema.get("enum") if isinstance(schema, dict) else None
    if isinstance(enum_values, list) and value in enum_values:
        return 1.0
    if isinstance(value, list):
        item_schema = (schema or {}).get("items") or {}
        if not value:
            return 0.5
        return sum(
            _router_leaf_grounding(item, item_schema, source_text, key=key)
            for item in value
        ) / float(len(value))
    if isinstance(value, dict):
        if not value:
            return 0.5
        properties = (schema or {}).get("properties") or {}
        return sum(
            _router_leaf_grounding(
                item,
                properties.get(child_key) or {},
                source_text,
                key=child_key,
            )
            for child_key, item in value.items()
        ) / float(len(value))
    if value_appears_in_source(value, source_text):
        return 1.0
    if isinstance(value, bool):
        return 0.6 if _bool_optional_flag_grounded(key, value, source_text) else 0.2
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0.35
    if isinstance(value, str):
        return 0.2
    return 0.25


def _router_call_grounding(call, tool_by_name, source_text):
    tool = tool_by_name.get(call.get("name")) or {}
    properties = ((tool.get("parameters") or {}).get("properties")) or {}
    arguments = call.get("arguments") or {}
    if not isinstance(arguments, dict):
        return 0.0
    if not arguments:
        return 1.0
    return sum(
        _router_leaf_grounding(
            value,
            properties.get(key) or {},
            source_text,
            key=key,
        )
        for key, value in arguments.items()
    ) / float(len(arguments))


def _percent_literal_pattern(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if value <= 1:
        return None
    if isinstance(value, int) or float(value).is_integer():
        number = str(int(value))
        return r"(?<![\d.])%s(?:\.0+)?\s*%%" % re.escape(number)
    return r"(?<![\d.])%s\s*%%" % re.escape(str(value))


def _percent_scale_mismatch_count(value, schema, source_text):
    if isinstance(value, dict):
        properties = (schema or {}).get("properties") or {}
        return sum(
            _percent_scale_mismatch_count(item, properties.get(key) or {}, source_text)
            for key, item in value.items()
        )
    if isinstance(value, list):
        item_schema = (schema or {}).get("items") or {}
        return sum(
            _percent_scale_mismatch_count(item, item_schema, source_text)
            for item in value
        )
    pattern = _percent_literal_pattern(value)
    if pattern is None:
        return 0
    expected_type = (schema or {}).get("type")
    if isinstance(expected_type, list):
        expected_type = [item for item in expected_type if item != "null"]
    numeric_schema = expected_type in ("number", "float", "integer", "int") or (
        isinstance(expected_type, list)
        and any(item in ("number", "float", "integer", "int") for item in expected_type)
    )
    if not numeric_schema:
        return 0
    return 1 if re.search(pattern, str(source_text or "")) else 0


def _schema_uses_smallest_currency_unit(schema):
    description = str((schema or {}).get("description") or "").lower()
    return bool(
        re.search(
            r"\b(cent|cents|pennies|smallest\s+unit|minor\s+unit)\b",
            description,
        )
    )


def _currency_literals(source_text):
    text = str(source_text or "")
    values = []
    for match in re.finditer(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text):
        values.append(float(match.group(1)))
    for match in re.finditer(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:usd|dollars?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        values.append(float(match.group(1)))
    return values


def _currency_scale_mismatch_count(value, schema, source_text):
    if isinstance(value, dict):
        properties = (schema or {}).get("properties") or {}
        return sum(
            _currency_scale_mismatch_count(item, properties.get(key) or {}, source_text)
            for key, item in value.items()
        )
    if isinstance(value, list):
        item_schema = (schema or {}).get("items") or {}
        return sum(
            _currency_scale_mismatch_count(item, item_schema, source_text)
            for item in value
        )
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    if _schema_uses_smallest_currency_unit(schema):
        return 0
    expected_type = (schema or {}).get("type")
    if isinstance(expected_type, list):
        expected_type = [item for item in expected_type if item != "null"]
    numeric_schema = expected_type in ("number", "float", "integer", "int") or (
        isinstance(expected_type, list)
        and any(item in ("number", "float", "integer", "int") for item in expected_type)
    )
    if not numeric_schema:
        return 0
    for literal in _currency_literals(source_text):
        if literal <= 0:
            continue
        scaled = literal * 100
        if abs(float(value) - scaled) < 1e-6 and abs(float(value) - literal) > 1e-6:
            return 1
    return 0


def _router_percent_scale_mismatch_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties")) or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            total += _percent_scale_mismatch_count(
                value,
                properties.get(key) or {},
                source_text,
            )
    return total


def _router_currency_scale_mismatch_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties")) or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            total += _currency_scale_mismatch_count(
                value,
                properties.get(key) or {},
                source_text,
            )
    return total


def _ungrounded_default_true_flag_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key in required or _bool_like_value(value) is not True:
                continue
            schema = properties.get(key) or {}
            if not isinstance(value, bool) and not _schema_is_bool_like_optional_flag(schema):
                continue
            if not schema_declares_default(schema):
                continue
            if _bool_optional_flag_grounded(key, value, source_text):
                continue
            total += 1
    return total


def _ungrounded_optional_bool_flag_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key in required:
                continue
            schema = properties.get(key) or {}
            bool_value = _bool_like_value(value)
            if bool_value is None:
                continue
            if not isinstance(value, bool) and not _schema_is_bool_like_optional_flag(schema):
                continue
            if bool_value is True and schema_declares_default(schema):
                continue
            if _bool_optional_flag_grounded(key, value, source_text):
                continue
            total += 1
    return total


def _schema_default_matches_value(schema, value):
    if not isinstance(schema, dict) or "default" not in schema:
        return False
    default = schema.get("default")
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        if isinstance(default, bool) or isinstance(value, bool):
            return default is value
        return abs(float(default) - float(value)) < 1e-9
    if isinstance(default, str) and isinstance(value, str):
        return default.strip().lower() == value.strip().lower()
    return default == value


def _ungrounded_nondefault_optional_scalar_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key in required:
                continue
            if isinstance(value, bool) or isinstance(value, (list, dict)):
                continue
            schema = properties.get(key) or {}
            if not isinstance(schema, dict) or "default" not in schema:
                continue
            if _schema_default_matches_value(schema, value):
                continue
            if value_appears_in_source(value, source_text) or absence_value_is_grounded(
                value,
                source_text,
            ):
                continue
            total += 1
    return total


def _ungrounded_optional_collection_item_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key in required or not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, (list, dict)) or isinstance(item, bool):
                    continue
                if _value_has_prompt_token_support(item, source_text):
                    continue
                total += 1
    return total


def _schema_expects_string(schema):
    expected_type = (schema or {}).get("type")
    if isinstance(expected_type, list):
        return "string" in expected_type or "str" in expected_type
    return expected_type in ("string", "str")


def _entity_like_argument_name(key):
    return re.search(
        r"(?:^|_)(name|title|artist|customer|receiver|recipient|merchant|company|event|movie|show|restaurant|hotel)(?:_|$)",
        str(key or ""),
        flags=re.IGNORECASE,
    ) is not None


def _ungrounded_required_entity_string_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key not in required:
                continue
            if not isinstance(value, str):
                continue
            if not _entity_like_argument_name(key):
                continue
            schema = properties.get(key) or {}
            if not _schema_expects_string(schema):
                continue
            if _value_has_prompt_token_support(value, source_text):
                continue
            total += 1
    return total


def _ungrounded_required_query_string_count(calls, tool_by_name, source_text):
    total = 0
    for call in calls:
        tool = tool_by_name.get(call.get("name")) or {}
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            if key not in required or str(key).lower() != "query":
                continue
            if not isinstance(value, str) or not _schema_expects_string(properties.get(key) or {}):
                continue
            if _value_has_prompt_token_support(value, source_text):
                continue
            total += 1
    return total


ACTION_REQUEST_TERMS = {
    "add",
    "book",
    "calculate",
    "check",
    "combine",
    "compare",
    "convert",
    "create",
    "delete",
    "execute",
    "fetch",
    "find",
    "freeze",
    "get",
    "help",
    "list",
    "locate",
    "make",
    "open",
    "remove",
    "reserve",
    "retrieve",
    "rotate",
    "run",
    "search",
    "set",
    "show",
    "turn",
    "update",
}

QUESTION_ONLY_TERMS = {
    "explain",
    "why",
    "what",
    "when",
    "where",
    "who",
    "whom",
}

ROUTER_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "to",
    "what",
    "whats",
    "with",
    "you",
}


def _router_tokens(text):
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return {
        token
        for token in re.findall(r"[a-z0-9]+", phrase_norm(spaced))
        if len(token) > 1 and token not in ROUTER_STOPWORDS
    }


def _tool_affinity(call, tool_by_name, source_text, include_parameters=True):
    tool = tool_by_name.get(call.get("name")) or {}
    prompt_tokens = _router_tokens(source_text)
    if not prompt_tokens:
        return 0.0

    chunks = [
        tool.get("name", ""),
        tool.get("description", ""),
    ]
    if include_parameters:
        for key, schema in ((tool.get("parameters") or {}).get("properties") or {}).items():
            chunks.append(key)
            chunks.append((schema or {}).get("description", ""))
    tool_tokens = _router_tokens(" ".join(chunks))
    if not tool_tokens:
        return 0.0

    overlap = prompt_tokens & tool_tokens
    return len(overlap) / float((len(prompt_tokens) * len(tool_tokens)) ** 0.5)


def _action_source_text(source_text):
    text = str(source_text or "")
    matches = list(re.finditer(r"(?:^|\n)\s*user:\s*", text, flags=re.IGNORECASE))
    if matches:
        return text[matches[-1].end():]
    return re.sub(r"^\s*(system|assistant|user):\s*", "", text, flags=re.IGNORECASE)


def _prompt_requests_tool_action(source_text):
    action_text = _action_source_text(source_text)
    tokens = _router_tokens(action_text)
    if tokens & ACTION_REQUEST_TERMS:
        return True
    stripped = action_text.lstrip().lower()
    if stripped.startswith(("how many ", "how much ")):
        return True
    if stripped.startswith(("can you ", "could you ", "please ", "i need ", "i want ")):
        return True
    if tokens and tokens.isdisjoint(QUESTION_ONLY_TERMS) and len(tokens) <= 5:
        return True
    return False


def _direct_empty_pipeline_rescue_allowed(source_text):
    action_text = _action_source_text(source_text).lstrip().lower()
    if action_text.startswith(("what ", "what's ", "who ", "where ", "when ", "why ")):
        return False
    return _prompt_requests_tool_action(source_text)


def _direct_has_grounded_collection_call(direct_calls, tool_by_name, source_text):
    for call in direct_calls:
        tool = tool_by_name.get(call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties")) or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        for key, value in arguments.items():
            schema = properties.get(key) or {}
            if not isinstance(value, list) or len(value) < 2:
                continue
            if all(_value_has_prompt_token_support(item, source_text) for item in value):
                return True
    return False


def _direct_has_command_execution_call(direct_calls, tool_by_name):
    for call in direct_calls:
        tool = tool_by_name.get(call.get("name")) or {}
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict) or "command" not in arguments:
            continue
        text = " ".join(
            [
                str(call.get("name") or ""),
                str(tool.get("description") or ""),
            ]
        ).lower()
        if re.search(r"\b(command|execute|system|shell|terminal|os\.system)\b", text):
            return True
    return False


def _prompt_drive_letters(source_text):
    text = str(source_text or "").lower()
    letters = {
        match.group(1)
        for match in re.finditer(r"\b([a-z])\s*(?:drive|disk)\b", text)
    }
    letters.update(
        match.group(1)
        for match in re.finditer(r"\b([a-z]):[\\/]?\b", text)
    )
    return letters


def _normalized_command_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _pipeline_has_command_target_refinement(
    direct_calls,
    pipeline_calls,
    tool_by_name,
    source_text,
):
    if len(direct_calls) != 1 or len(pipeline_calls) != 1:
        return False
    if direct_calls[0].get("name") != pipeline_calls[0].get("name"):
        return False
    if not _direct_has_command_execution_call(direct_calls, tool_by_name):
        return False

    direct_args = direct_calls[0].get("arguments") or {}
    pipeline_args = pipeline_calls[0].get("arguments") or {}
    if set(direct_args) != set(pipeline_args) or "command" not in direct_args:
        return False

    direct_command = _normalized_command_text(direct_args.get("command"))
    pipeline_command = _normalized_command_text(pipeline_args.get("command"))
    if not direct_command or not pipeline_command or direct_command == pipeline_command:
        return False

    drive_letters = _prompt_drive_letters(source_text)
    if not drive_letters:
        return False

    for letter in drive_letters:
        drive_prefix = f"{letter}:"
        if drive_prefix not in pipeline_command:
            continue
        if pipeline_command.startswith(direct_command):
            added = pipeline_command[len(direct_command):].strip()
            if re.match(r"^[\\/]*$", added):
                return True
            if re.match(r"^[a-z]:[\\/]?$", added):
                return True
        direct_words = direct_command.split()
        pipeline_words = pipeline_command.split()
        if (
            len(direct_words) == 1
            and pipeline_words
            and pipeline_words[0] == direct_words[0]
            and any(word.startswith(drive_prefix) for word in pipeline_words[1:])
        ):
            return True
    return False


def _is_code_like_string(value):
    if not isinstance(value, str):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{2,6}", value.strip()))


def _alternative_expands_direct_code_like_value(direct_calls, alternative_calls):
    if len(direct_calls) != len(alternative_calls):
        return False
    for direct_call, alternative_call in zip(direct_calls, alternative_calls):
        if direct_call.get("name") != alternative_call.get("name"):
            return False
        direct_args = direct_call.get("arguments") or {}
        alternative_args = alternative_call.get("arguments") or {}
        if not isinstance(direct_args, dict) or not isinstance(alternative_args, dict):
            continue
        for key, direct_value in direct_args.items():
            if key not in alternative_args:
                continue
            alternative_value = alternative_args[key]
            if (
                _is_code_like_string(direct_value)
                and isinstance(alternative_value, str)
                and alternative_value.strip() != direct_value.strip()
            ):
                return True
            if isinstance(direct_value, list) and isinstance(alternative_value, list):
                for direct_item, alternative_item in zip(direct_value, alternative_value):
                    if (
                        _is_code_like_string(direct_item)
                        and isinstance(alternative_item, str)
                        and alternative_item.strip() != direct_item.strip()
                    ):
                        return True
    return False


def _direct_has_state_change_call(direct_calls, tool_by_name):
    for call in direct_calls:
        tool = tool_by_name.get(call.get("name")) or {}
        text = " ".join(
            [
                str(call.get("name") or ""),
                str(tool.get("description") or ""),
            ]
        )
        if re.search(
            r"(?:^|[_.\s-])(?:set|update|change|modify|rotate|resize|edit|freeze)(?:[_.\s-]|$|[A-Z])",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _strong_direct_tool_evidence(task, direct, direct_score):
    direct_calls = direct.get("normalized_calls") or []
    if not direct_calls:
        return False
    if direct_score["valid_call_count"] != len(direct_calls):
        return False
    if direct_score["deterministic_drop_count"] > 0:
        return False

    source_text = task.get("prompt", "")
    action_text = _action_source_text(source_text)
    tool_by_name = {tool["name"]: tool for tool in task.get("tools") or []}
    action_like = _prompt_requests_tool_action(source_text)
    stripped = action_text.lstrip().lower()
    quantitative_lookup = stripped.startswith(("how many ", "how much "))
    collection_like = _direct_has_grounded_collection_call(
        direct_calls,
        tool_by_name,
        source_text,
    )
    command_like = _direct_has_command_execution_call(direct_calls, tool_by_name)
    state_change_like = _direct_has_state_change_call(direct_calls, tool_by_name)
    if not action_like and not collection_like and not command_like and not state_change_like:
        return False
    if direct_score.get("ungrounded_optional_collection_item_count", 0) > 0:
        return False

    affinities = [
        _tool_affinity(call, tool_by_name, source_text, include_parameters=False)
        for call in direct_calls
    ]
    affinity = sum(affinities) / float(len(affinities) or 1)
    grounding = direct_score["grounding"]
    if collection_like and direct_score.get("unsupported_call_count", 0) == 0:
        return True
    if command_like and action_like and len(_router_tokens(action_text)) <= 6:
        return True
    if state_change_like and action_like and grounding >= 0.6:
        return True
    if quantitative_lookup and grounding >= 0.75 and affinity >= 0.05:
        return True
    if (
        action_like
        and direct_score.get("ungrounded_default_true_flag_count", 0) == 1
        and grounding >= 0.7
        and affinity >= 0.05
    ):
        return True
    if action_like and grounding >= 0.3 and affinity >= 0.18:
        return True
    return action_like and grounding >= 0.75 and affinity >= 0.25


def _high_precision_abstention_override(task, calls, score):
    if not calls:
        return False
    if score["valid_call_count"] != len(calls):
        return False
    if score["deterministic_drop_count"] > 0:
        return False
    if score.get("ungrounded_optional_collection_item_count", 0) > 0:
        return False

    source_text = task.get("prompt", "")
    tool_by_name = {tool["name"]: tool for tool in task.get("tools") or []}
    if _direct_has_grounded_collection_call(calls, tool_by_name, source_text):
        return True
    return (
        _prompt_requests_tool_action(source_text)
        and (
            _direct_has_command_execution_call(calls, tool_by_name)
            or _direct_has_state_change_call(calls, tool_by_name)
        )
    )


def _router_plan_tool_coverage(calls, decomposition):
    planned = []
    for line in str(decomposition or "").splitlines():
        match = re.search(r"\btool=([^;\s]+)", line)
        if match:
            planned.append(match.group(1))
    if not planned:
        return 0.5
    called_counts = {}
    for call in calls:
        name = call.get("name")
        called_counts[name] = called_counts.get(name, 0) + 1
    covered = 0
    for name in planned:
        if called_counts.get(name, 0) <= 0:
            continue
        covered += 1
        called_counts[name] -= 1
    return covered / float(len(planned))


def _router_plan_stability(pipeline):
    scores = pipeline.get("multseq_consensus_scores") or []
    selected_index = pipeline.get("multseq_selected_plan")
    if (
        isinstance(selected_index, int)
        and 0 <= selected_index < len(scores)
    ):
        return max(0.0, min(1.0, float(scores[selected_index])))

    plans = pipeline.get("multseq_plans") or []
    if plans:
        vote_values = []
        for plan in plans:
            vote_values.extend((plan.get("vote_counts") or {}).values())
        if vote_values:
            k = max(len(plans), 1)
            return max(0.0, min(1.0, sum(vote_values) / float(len(vote_values) * k)))
    return 0.5


def _call_multiset(calls):
    counts = {}
    for call in calls:
        key = canonical([call])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _calls_are_strict_superset(superset_calls, subset_calls):
    if len(superset_calls) <= len(subset_calls):
        return False
    superset_counts = _call_multiset(superset_calls)
    subset_counts = _call_multiset(subset_calls)
    return all(superset_counts.get(key, 0) >= count for key, count in subset_counts.items())


def _calls_are_strict_subset(subset_calls, superset_calls):
    return _calls_are_strict_superset(superset_calls, subset_calls)


def _call_argument_shape(calls):
    shape = []
    for call in calls:
        arguments = call.get("arguments") or {}
        keys = tuple(sorted(arguments)) if isinstance(arguments, dict) else ()
        shape.append((call.get("name"), keys))
    return shape


def _call_tool_sequence(calls):
    return [call.get("name") for call in calls]


def _call_tool_counts(calls):
    counts = {}
    for call in calls:
        name = call.get("name")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _react_same_tool_pipeline_strict_win_allowed(
    selected,
    selected_reason,
    selected_calls,
    react_calls,
    selected_score,
    react_score,
):
    if selected != "pipeline":
        return False
    if selected_reason != "pipeline_over_single_intent_direct":
        return False
    if not selected_calls or not react_calls:
        return False
    if len(selected_calls) != len(react_calls):
        return False
    if _call_tool_counts(selected_calls) != _call_tool_counts(react_calls):
        return False
    if react_score is None:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > selected_score["unsupported_call_count"]:
        return False
    if react_score["grounding"] < selected_score["grounding"]:
        return False
    return react_score["total"] > selected_score["total"]


def _pipeline_can_override_single_intent_direct(
    direct_calls,
    pipeline_calls,
    direct_score,
    pipeline_score,
    margin,
    source_text,
):
    if not pipeline_calls:
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if pipeline_score["deterministic_drop_count"] > 0:
        return False
    if not direct_calls:
        return (
            _direct_empty_pipeline_rescue_allowed(source_text)
            and pipeline_score["total"] >= direct_score["total"] + margin
        )
    same_tool_sequence = _call_tool_sequence(direct_calls) == _call_tool_sequence(pipeline_calls)
    direct_suspect_count = (
        (
            direct_score.get("deterministic_drop_count", 0)
            if not _deterministic_drops_are_grounded_enum_only(
                direct_score,
                source_text,
            )
            else 0
        )
        + direct_score.get("ungrounded_required_query_string_count", 0)
        + direct_score.get("ungrounded_optional_bool_flag_count", 0)
        + direct_score.get("ungrounded_nondefault_optional_scalar_count", 0)
        + direct_score.get("ungrounded_optional_collection_item_count", 0)
    )
    pipeline_suspect_count = (
        pipeline_score.get("deterministic_drop_count", 0)
        + pipeline_score.get("ungrounded_required_query_string_count", 0)
        + pipeline_score.get("ungrounded_optional_bool_flag_count", 0)
        + pipeline_score.get("ungrounded_nondefault_optional_scalar_count", 0)
        + pipeline_score.get("ungrounded_optional_collection_item_count", 0)
    )
    if same_tool_sequence and pipeline_score["total"] >= direct_score["total"] + margin:
        return True
    if (
        direct_suspect_count > pipeline_suspect_count
        and pipeline_score["total"] >= direct_score["total"] - 0.25
    ):
        return True
    return False


def _source_contains_non_ascii(source_text):
    return any(ord(ch) > 127 for ch in str(source_text or ""))


def _value_has_prompt_token_support(value, source_text):
    if value_appears_in_source(value, source_text) or absence_value_is_grounded(
        value,
        source_text,
    ):
        return True
    if value is None or isinstance(value, bool) or isinstance(value, (list, dict)):
        return False
    value_tokens = _router_tokens(value)
    source_tokens = _router_tokens(source_text)
    return bool(value_tokens and source_tokens and value_tokens & source_tokens)


def _react_omits_agreed_supported_argument(
    direct_calls,
    pipeline_calls,
    react_calls,
    source_text,
):
    if not direct_calls or canonical(direct_calls) != canonical(pipeline_calls):
        return False
    if len(direct_calls) != len(react_calls):
        return False
    for reference_call, react_call in zip(direct_calls, react_calls):
        if reference_call.get("name") != react_call.get("name"):
            return False
        reference_args = reference_call.get("arguments") or {}
        react_args = react_call.get("arguments") or {}
        if not isinstance(reference_args, dict) or not isinstance(react_args, dict):
            return False
        for key, value in reference_args.items():
            if key in react_args:
                continue
            if _value_has_prompt_token_support(value, source_text):
                return True
    return False


def _react_cardinality_expansion_allowed(
    selected_calls,
    react_calls,
    selected_score,
    react_score,
):
    if not selected_calls or not react_calls:
        return False
    if len(react_calls) < len(selected_calls) + 2:
        return False
    if react_score is None:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > 0:
        return False
    if react_score["percent_scale_mismatch_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if (
        react_score["ungrounded_default_true_flag_count"]
        or react_score["ungrounded_optional_bool_flag_count"]
        or react_score["ungrounded_nondefault_optional_scalar_count"]
        or react_score["ungrounded_optional_collection_item_count"]
        or react_score["ungrounded_required_entity_string_count"]
        or react_score["ungrounded_required_query_string_count"]
    ):
        return False
    if react_score["grounding"] < selected_score["grounding"] - 0.1:
        return False
    return react_score["total"] >= selected_score["total"] - 0.3


def _candidate_is_clean_grounded(calls, score):
    if not calls or score is None:
        return False
    if score["valid_call_count"] != len(calls):
        return False
    if score["deterministic_drop_count"] > 0:
        return False
    if score["unsupported_call_count"] > 0:
        return False
    if score["percent_scale_mismatch_count"] > 0:
        return False
    if score["currency_scale_mismatch_count"] > 0:
        return False
    return True


def _pipeline_react_corroborated_alternative_allowed(
    direct_calls,
    pipeline_calls,
    react_calls,
    direct_score,
    pipeline_score,
    react_score,
    source_text,
):
    if not _prompt_requests_tool_action(source_text):
        return False
    if not _candidate_is_clean_grounded(pipeline_calls, pipeline_score):
        return False
    if not _candidate_is_clean_grounded(react_calls, react_score):
        return False
    if canonical(pipeline_calls) != canonical(react_calls):
        return False
    if canonical(pipeline_calls) == canonical(direct_calls):
        return False
    if _alternative_expands_direct_code_like_value(direct_calls, pipeline_calls):
        return False
    if pipeline_score["grounding"] < direct_score["grounding"] - 0.05:
        return False
    if react_score["grounding"] < direct_score["grounding"] - 0.05:
        return False
    return min(pipeline_score["total"], react_score["total"]) >= direct_score["total"] - 0.35


def _react_grounded_superset_allowed(
    selected_calls,
    react_calls,
    selected_score,
    react_score,
):
    if not selected_calls:
        return False
    if not react_calls or react_score is None:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if len(react_calls) < len(selected_calls) + 1:
        return False
    if not _calls_are_strict_superset(react_calls, selected_calls):
        return False
    if (
        react_score["ungrounded_default_true_flag_count"]
        or react_score["ungrounded_optional_bool_flag_count"]
        or react_score["ungrounded_nondefault_optional_scalar_count"]
        or react_score["ungrounded_optional_collection_item_count"]
        or react_score["ungrounded_required_entity_string_count"]
        or react_score["ungrounded_required_query_string_count"]
    ):
        return False
    if react_score["percent_scale_mismatch_count"] > selected_score["percent_scale_mismatch_count"]:
        return False
    if react_score["grounding"] < selected_score["grounding"] - 0.05:
        return False
    return react_score["total"] >= selected_score["total"] - 0.1


def _minimum_tool_affinity(calls, tool_by_name, source_text):
    if not calls:
        return 0.0
    return min(
        _tool_affinity(call, tool_by_name, source_text, include_parameters=False)
        for call in calls
    )


def _pipeline_cleaner_subset_of_react_allowed(
    direct_calls,
    pipeline_calls,
    react_calls,
    direct_score,
    pipeline_score,
    task,
):
    if not pipeline_calls or not react_calls:
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if pipeline_score["deterministic_drop_count"] > 0:
        return False
    if pipeline_score["unsupported_call_count"] > 0:
        return False
    if pipeline_score["currency_scale_mismatch_count"] > 0:
        return False
    if not _calls_are_strict_subset(direct_calls, react_calls):
        return False
    if not _calls_are_strict_subset(pipeline_calls, react_calls):
        return False
    if canonical(direct_calls) == canonical(pipeline_calls):
        return False
    if pipeline_score["total"] < direct_score["total"] - 0.35:
        return False

    tool_by_name = {tool["name"]: tool for tool in task.get("tools") or []}
    source_text = task.get("prompt", "")
    return _minimum_tool_affinity(
        pipeline_calls,
        tool_by_name,
        source_text,
    ) >= _minimum_tool_affinity(
        direct_calls,
        tool_by_name,
        source_text,
    ) + 0.15


def _deterministic_drops_are_grounded_enum_only(score, source_text):
    drops = score.get("deterministic_drops") or []
    if not drops:
        return False
    for drop in drops:
        reason = str(drop.get("reason") or "")
        match = re.match(r"enum_mismatch:(.+)$", reason)
        if not match:
            return False
        key = match.group(1)
        arguments = (drop.get("call") or {}).get("arguments") or {}
        if not isinstance(arguments, dict) or key not in arguments:
            return False
        if not _value_has_prompt_token_support(arguments[key], source_text):
            return False
    return True


def _value_contains_result_placeholder(value):
    if isinstance(value, str):
        return bool(
            re.search(
                r"(?:\$\{[^}]*result[^}]*\}|<[^>]*result[^>]*>|\b\w+_result\b|\bresult_of\b)",
                value,
                flags=re.IGNORECASE,
            )
        )
    if isinstance(value, dict):
        return any(_value_contains_result_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_value_contains_result_placeholder(item) for item in value)
    return False


def _deterministic_drops_are_result_placeholders(score):
    drops = score.get("deterministic_drops") or []
    if not drops:
        return False
    for drop in drops:
        reason = str(drop.get("reason") or "")
        if not reason.startswith("type_mismatch:"):
            return False
        arguments = (drop.get("call") or {}).get("arguments") or {}
        if not _value_contains_result_placeholder(arguments):
            return False
    return True


def _pipeline_has_multilingual_copy_advantage(
    task,
    direct_calls,
    pipeline_calls,
    direct_score,
    pipeline_score,
):
    if not _source_contains_non_ascii(task.get("prompt", "")):
        return False
    if not direct_calls or not pipeline_calls:
        return False
    if _call_argument_shape(direct_calls) != _call_argument_shape(pipeline_calls):
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if pipeline_score["deterministic_drop_count"] > 0:
        return False
    return (
        pipeline_score["grounding"] >= direct_score["grounding"] + 0.25
        and pipeline_score["total"] >= direct_score["total"] + 0.25
    )


def _corroborated_pipeline_superset(
    direct_calls,
    pipeline_calls,
    react_calls,
    pipeline_score,
    react_score,
):
    if not react_calls:
        return False
    if canonical(pipeline_calls) != canonical(react_calls):
        return False
    if not _calls_are_strict_superset(pipeline_calls, direct_calls):
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if react_score is None or react_score["valid_call_count"] != len(react_calls):
        return False
    if pipeline_score["deterministic_drop_count"] or react_score["deterministic_drop_count"]:
        return False
    if pipeline_score["grounding"] < 0.75 or react_score["grounding"] < 0.75:
        return False
    if pipeline_score["unsupported_call_count"] or react_score["unsupported_call_count"]:
        return False
    if pipeline_score["percent_scale_mismatch_count"] or react_score["percent_scale_mismatch_count"]:
        return False
    if pipeline_score["currency_scale_mismatch_count"] or react_score["currency_scale_mismatch_count"]:
        return False
    if (
        pipeline_score["ungrounded_default_true_flag_count"]
        or pipeline_score["ungrounded_optional_bool_flag_count"]
    ):
        return False
    return True


def _corroborated_pipeline_over_suspect_direct(
    direct_calls,
    pipeline_calls,
    react_calls,
    direct_score,
    pipeline_score,
    react_score,
):
    if not direct_calls or not pipeline_calls or not react_calls:
        return False
    if canonical(pipeline_calls) != canonical(react_calls):
        return False
    if canonical(pipeline_calls) == canonical(direct_calls):
        return False
    direct_suspect = (
        direct_score["deterministic_drop_count"] > 0
        or direct_score["ungrounded_default_true_flag_count"] > 0
        or direct_score["ungrounded_optional_bool_flag_count"] > 0
        or direct_score["currency_scale_mismatch_count"] > 0
    )
    if not direct_suspect:
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if react_score is None or react_score["valid_call_count"] != len(react_calls):
        return False
    if pipeline_score["deterministic_drop_count"] or react_score["deterministic_drop_count"]:
        return False
    if pipeline_score["grounding"] < 0.6 or react_score["grounding"] < 0.6:
        return False
    if pipeline_score["unsupported_call_count"] or react_score["unsupported_call_count"]:
        return False
    if pipeline_score["percent_scale_mismatch_count"] or react_score["percent_scale_mismatch_count"]:
        return False
    if pipeline_score["currency_scale_mismatch_count"] or react_score["currency_scale_mismatch_count"]:
        return False
    if (
        pipeline_score["ungrounded_default_true_flag_count"]
        or pipeline_score["ungrounded_optional_bool_flag_count"]
    ):
        return False
    return pipeline_score["total"] >= direct_score["total"] - 0.25


_MULTISTEP_PROMPT_RE = re.compile(
    r"\b(?:after|then|lastly|finally|another|second|also)\b",
    re.IGNORECASE,
)

_NEW_TOOL_STEP_MIN_AFFINITY = 0.05


def _string_values_are_prompt_grounded(value, source_text):
    if isinstance(value, str):
        if _value_contains_result_placeholder(value):
            return False
        return value_appears_in_source(value, source_text) or absence_value_is_grounded(
            value,
            source_text,
        )
    if isinstance(value, list):
        return all(_string_values_are_prompt_grounded(item, source_text) for item in value)
    if isinstance(value, dict):
        return all(
            _string_values_are_prompt_grounded(item, source_text)
            for item in value.values()
        )
    return True


def _pipeline_adds_grounded_new_tool_step(
    task,
    direct_calls,
    pipeline_calls,
    direct_score,
    pipeline_score,
):
    source_text = task.get("prompt", "")
    if not direct_calls or len(pipeline_calls) <= len(direct_calls):
        return False
    if _MULTISTEP_PROMPT_RE.search(source_text) is None:
        return False
    direct_tool_names = {call.get("name") for call in direct_calls}
    new_tool_calls = [
        call for call in pipeline_calls if call.get("name") not in direct_tool_names
    ]
    if not new_tool_calls:
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if pipeline_score["deterministic_drop_count"] > 0:
        return False
    if pipeline_score["unsupported_call_count"] > 0:
        return False
    if pipeline_score["currency_scale_mismatch_count"] > 0:
        return False
    if pipeline_score["percent_scale_mismatch_count"] > 0:
        return False
    if pipeline_score["total"] < direct_score["total"] - 0.25:
        return False
    tool_by_name = {tool["name"]: tool for tool in task.get("tools") or []}
    for call in new_tool_calls:
        if (
            _tool_affinity(call, tool_by_name, source_text, include_parameters=False)
            < _NEW_TOOL_STEP_MIN_AFFINITY
        ):
            return False
        if not _string_values_are_prompt_grounded(
            call.get("arguments") or {},
            source_text,
        ):
            return False
    return True


def _clean_react_covers_pipeline(pipeline_calls, react_calls, react_score):
    if not react_calls or react_score is None:
        return False
    if len(react_calls) < len(pipeline_calls):
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if react_score["percent_scale_mismatch_count"] > 0:
        return False
    return True


def _pipeline_over_under_calling_react_allowed(
    task,
    direct_calls,
    pipeline_calls,
    react_calls,
    pipeline_score,
    react_score,
):
    if not direct_calls or not pipeline_calls or not react_calls:
        return False
    if len(pipeline_calls) <= len(react_calls):
        return False
    if len(pipeline_calls) != len(direct_calls):
        return False
    if _MULTISTEP_PROMPT_RE.search(task.get("prompt", "")) is None:
        return False
    if pipeline_score["valid_call_count"] != len(pipeline_calls):
        return False
    if pipeline_score["deterministic_drop_count"] > 0:
        return False
    if pipeline_score["unsupported_call_count"] > 0:
        return False
    if pipeline_score["currency_scale_mismatch_count"] > 0:
        return False
    if pipeline_score["percent_scale_mismatch_count"] > 0:
        return False
    if pipeline_score["ungrounded_required_entity_string_count"] > 0:
        return False
    if pipeline_score["ungrounded_required_query_string_count"] > 0:
        return False
    if react_score is None:
        return False
    if pipeline_score["total"] < react_score["total"] - 1.0:
        return False
    return True


def _react_intent_switch_reduces_ungrounded_scalars_allowed(
    selected_calls,
    react_calls,
    selected_score,
    react_score,
):
    if not selected_calls or not react_calls or react_score is None:
        return False
    if len(selected_calls) != len(react_calls):
        return False
    if [call.get("name") for call in selected_calls] == [
        call.get("name") for call in react_calls
    ]:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if react_score["percent_scale_mismatch_count"] > 0:
        return False
    if selected_score["ungrounded_nondefault_optional_scalar_count"] <= react_score[
        "ungrounded_nondefault_optional_scalar_count"
    ]:
        return False
    return react_score["total"] >= selected_score["total"] - 1.0


LOCATION_NORMALIZATION_KEYS = {
    "address",
    "city",
    "destination",
    "drop_off_location",
    "dropoff_location",
    "location",
    "origin",
    "pick_up_location",
    "pickup_location",
    "place",
}


def _router_edit_distance(left, right):
    left = phrase_norm(left)
    right = phrase_norm(right)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i] + [0] * len(right)
        for j, right_char in enumerate(right, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            )
        previous = current
    return previous[-1]


def _location_normalization_key(path):
    if not path:
        return False
    key = str(path[-1]).lower()
    return (
        key in LOCATION_NORMALIZATION_KEYS
        or key.endswith("_city")
        or key.endswith("_location")
        or key.endswith("_place")
        or key.endswith("_address")
    )


def _collect_location_normalization_changes(selected_value, react_value, path=()):
    if isinstance(selected_value, str) and isinstance(react_value, str):
        if selected_value == react_value:
            return []
        if not _location_normalization_key(path):
            return None
        if len(selected_value) < 4 or len(react_value) < 4:
            return None
        if _router_edit_distance(selected_value, react_value) <= 2:
            return [
                {
                    "path": path,
                    "from": selected_value,
                    "to": react_value,
                    "reason": "location_typo_correction",
                }
            ]
        selected_raw = str(selected_value).strip().lower()
        react_raw = str(react_value).strip().lower()
        if react_raw.startswith(selected_raw + ","):
            suffix = react_value[len(selected_value) + 1 :].strip()
            if suffix and len(_router_tokens(suffix)) <= 3:
                return [
                    {
                        "path": path,
                        "from": selected_value,
                        "to": react_value,
                        "reason": "location_comma_suffix",
                    }
                ]
        return None
    if isinstance(selected_value, dict) and isinstance(react_value, dict):
        if set(selected_value) != set(react_value):
            return None
        changes = []
        for key in selected_value:
            child_changes = _collect_location_normalization_changes(
                selected_value[key],
                react_value[key],
                path + (key,),
            )
            if child_changes is None:
                return None
            changes.extend(child_changes)
        return changes
    if isinstance(selected_value, list) and isinstance(react_value, list):
        if len(selected_value) != len(react_value):
            return None
        changes = []
        for index, (selected_item, react_item) in enumerate(
            zip(selected_value, react_value)
        ):
            child_changes = _collect_location_normalization_changes(
                selected_item,
                react_item,
                path + (str(index),),
            )
            if child_changes is None:
                return None
            changes.extend(child_changes)
        return changes
    if selected_value == react_value:
        return []
    return None


def _react_location_normalization_allowed(
    selected_reason,
    selected_calls,
    react_calls,
    react_score,
):
    if selected_reason != "candidates_agree":
        return False
    if not selected_calls or not react_calls or react_score is None:
        return False
    if len(selected_calls) != len(react_calls):
        return False
    if [call.get("name") for call in selected_calls] != [
        call.get("name") for call in react_calls
    ]:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if react_score["percent_scale_mismatch_count"] > 0:
        return False
    changes = []
    for selected_call, react_call in zip(selected_calls, react_calls):
        selected_args = selected_call.get("arguments") or {}
        react_args = react_call.get("arguments") or {}
        call_changes = _collect_location_normalization_changes(
            selected_args,
            react_args,
        )
        if call_changes is None:
            return False
        changes.extend(call_changes)
    if len(changes) != 1:
        return False
    return react_score["unsupported_call_count"] <= len(changes)


def _schema_id_examples(description):
    examples = []
    for match in re.finditer(
        r"(?i)(\d+)\s+for\s+([a-z][a-z0-9 _-]{2,40}?)(?=,|\.|$)",
        str(description or ""),
    ):
        examples.append((int(match.group(1)), match.group(2).strip()))
    return examples


def _react_schema_example_id_replaces_boolean_allowed(
    task,
    selected_reason,
    selected_calls,
    react_calls,
    react_score,
):
    if selected_reason != "candidates_agree":
        return False
    if not selected_calls or not react_calls or react_score is None:
        return False
    if len(selected_calls) != len(react_calls):
        return False
    if [call.get("name") for call in selected_calls] != [
        call.get("name") for call in react_calls
    ]:
        return False
    if react_score["valid_call_count"] != len(react_calls):
        return False
    if react_score["deterministic_drop_count"] > 0:
        return False
    if react_score["unsupported_call_count"] > 0:
        return False
    if react_score["currency_scale_mismatch_count"] > 0:
        return False
    if react_score["percent_scale_mismatch_count"] > 0:
        return False

    tool_by_name = {tool["name"]: tool for tool in task.get("tools") or []}
    prompt_tokens = _router_tokens(task.get("prompt", ""))
    changes = []
    for selected_call, react_call in zip(selected_calls, react_calls):
        selected_args = selected_call.get("arguments") or {}
        react_args = react_call.get("arguments") or {}
        if not isinstance(selected_args, dict) or not isinstance(react_args, dict):
            return False
        common = set(selected_args) & set(react_args)
        if any(selected_args[key] != react_args[key] for key in common):
            return False
        added = {key: react_args[key] for key in set(react_args) - set(selected_args)}
        removed = {
            key: selected_args[key]
            for key in set(selected_args) - set(react_args)
        }
        if not added and not removed:
            continue
        if len(added) != 1 or len(removed) != 1:
            return False
        added_key, added_value = next(iter(added.items()))
        removed_key, removed_value = next(iter(removed.items()))
        if (
            not str(added_key).lower().endswith("_id")
            or not isinstance(added_value, int)
            or isinstance(added_value, bool)
        ):
            return False
        if not str(removed_key).lower().startswith("is_") or removed_value is not True:
            return False

        tool = tool_by_name.get(selected_call.get("name")) or {}
        properties = ((tool.get("parameters") or {}).get("properties") or {})
        schema = properties.get(added_key) or {}
        matching_labels = [
            label
            for value, label in _schema_id_examples(schema.get("description", ""))
            if value == added_value
        ]
        if not matching_labels:
            return False
        label_token_sets = [_router_tokens(label) for label in matching_labels]
        if not any(tokens and tokens <= prompt_tokens for tokens in label_token_sets):
            return False
        removed_tokens = _router_tokens(str(removed_key).replace("_", " "))
        removed_tokens -= {"is", "has", "available", "for"}
        if not any(removed_tokens & tokens for tokens in label_token_sets):
            return False
        changes.append((added_key, removed_key))

    return len(changes) == 1


def score_router_candidate(task, candidate, source, pipeline_context=None):
    calls = list(candidate.get("normalized_calls") or [])
    indexed, dropped, merged_count, _ = deterministic_verify_calls(
        calls,
        task["tools"],
        task["prompt"],
    )
    valid_calls = [item["call"] for item in indexed]
    tool_by_name = {tool["name"]: tool for tool in task["tools"]}
    validity = (
        len(valid_calls) / float(len(calls))
        if calls
        else 1.0
    )
    grounding_values = [
        _router_call_grounding(call, tool_by_name, task["prompt"])
        for call in valid_calls
    ]
    grounding = (
        sum(grounding_values) / float(len(grounding_values))
        if grounding_values
        else 0.0
    )
    uniqueness = (
        len(valid_calls) / float(len(valid_calls) + merged_count)
        if valid_calls or merged_count
        else 1.0
    )
    unsupported_calls = sum(1 for value in grounding_values if value <= 0.25)
    percent_scale_mismatch_count = _router_percent_scale_mismatch_count(
        valid_calls,
        tool_by_name,
        task["prompt"],
    )
    currency_scale_mismatch_count = _router_currency_scale_mismatch_count(
        valid_calls,
        tool_by_name,
        task["prompt"],
    )
    ungrounded_default_true_flag_count = _ungrounded_default_true_flag_count(
        valid_calls,
        tool_by_name,
        task["prompt"],
    )
    ungrounded_optional_bool_flag_count = _ungrounded_optional_bool_flag_count(
        valid_calls,
        tool_by_name,
        task["prompt"],
    )
    ungrounded_nondefault_optional_scalar_count = (
        _ungrounded_nondefault_optional_scalar_count(
            valid_calls,
            tool_by_name,
            task["prompt"],
        )
    )
    ungrounded_optional_collection_item_count = (
        _ungrounded_optional_collection_item_count(
            valid_calls,
            tool_by_name,
            task["prompt"],
        )
    )
    ungrounded_required_entity_string_count = (
        _ungrounded_required_entity_string_count(
            valid_calls,
            tool_by_name,
            task["prompt"],
        )
    )
    ungrounded_required_query_string_count = (
        _ungrounded_required_query_string_count(
            valid_calls,
            tool_by_name,
            task["prompt"],
        )
    )
    extra_call_penalty = max(0, len(valid_calls) - 1) * 0.1

    plan_coverage = 0.5
    plan_stability = 0.5
    if pipeline_context is not None:
        plan_coverage = _router_plan_tool_coverage(
            valid_calls,
            pipeline_context.get("decomposition"),
        )
        plan_stability = _router_plan_stability(pipeline_context)

    if not calls:
        if source == "pipeline" and candidate.get("verdict") == "no_intent":
            total = 5.0 + 0.5 * plan_stability
        else:
            total = 2.5
    else:
        total = (
            3.0 * validity
            + 2.0 * grounding
            + 0.5 * uniqueness
            - 0.5 * unsupported_calls
            - 1.0 * percent_scale_mismatch_count
            - 1.0 * currency_scale_mismatch_count
            - 0.75 * ungrounded_default_true_flag_count
            - 1.0 * ungrounded_optional_bool_flag_count
            - 0.7 * ungrounded_nondefault_optional_scalar_count
            - 0.7 * ungrounded_optional_collection_item_count
            - 1.2 * ungrounded_required_entity_string_count
            - 1.2 * ungrounded_required_query_string_count
            - extra_call_penalty
        )
        total += 2.0 * plan_coverage * plan_stability
        if source != "pipeline":
            total += 0.15

    return {
        "total": round(total, 6),
        "call_count": len(calls),
        "valid_call_count": len(valid_calls),
        "validity": round(validity, 6),
        "grounding": round(grounding, 6),
        "uniqueness": round(uniqueness, 6),
        "unsupported_call_count": unsupported_calls,
        "percent_scale_mismatch_count": percent_scale_mismatch_count,
        "currency_scale_mismatch_count": currency_scale_mismatch_count,
        "ungrounded_default_true_flag_count": ungrounded_default_true_flag_count,
        "ungrounded_optional_bool_flag_count": ungrounded_optional_bool_flag_count,
        "ungrounded_nondefault_optional_scalar_count": ungrounded_nondefault_optional_scalar_count,
        "ungrounded_optional_collection_item_count": ungrounded_optional_collection_item_count,
        "ungrounded_required_entity_string_count": ungrounded_required_entity_string_count,
        "ungrounded_required_query_string_count": ungrounded_required_query_string_count,
        "extra_call_penalty": round(extra_call_penalty, 6),
        "plan_coverage": round(plan_coverage, 6),
        "plan_stability": round(plan_stability, 6),
        "deterministic_drop_count": len(dropped),
        "deterministic_drops": dropped,
    }


def route_tool_call_candidates(
    task,
    direct,
    pipeline,
    react=None,
    margin=0.45,
    abstention_policy="strict",
    include_react=False,
    react_policy="score_margin",
    drop_defaulted_optional_args=False,
):
    """Choose direct or decomposed calls using inference-time-only signals."""
    direct = sanitize_candidate_for_router(
        task,
        direct,
        drop_defaulted_optional_args=drop_defaulted_optional_args,
    )
    pipeline = sanitize_candidate_for_router(
        task,
        pipeline,
        drop_defaulted_optional_args=drop_defaulted_optional_args,
    )
    if react is not None:
        react = sanitize_candidate_for_router(
            task,
            react,
            drop_defaulted_optional_args=drop_defaulted_optional_args,
        )

    direct = repair_optional_default_arguments(task, direct)
    pipeline = repair_optional_default_arguments(task, pipeline)
    if react is not None:
        react = repair_optional_default_arguments(task, react)

    direct_score = score_router_candidate(
        task,
        direct,
        "direct",
        pipeline_context=pipeline,
    )
    pipeline_score = score_router_candidate(
        task,
        pipeline,
        "pipeline",
        pipeline_context=pipeline,
    )
    react_score = None
    if include_react and react is not None:
        react_score = score_router_candidate(
            task,
            react,
            "react",
            pipeline_context=pipeline,
        )
    direct_calls = direct.get("normalized_calls") or []
    pipeline_calls = pipeline.get("normalized_calls") or []
    react_calls_for_corroboration = (
        (react.get("normalized_calls") or [])
        if react is not None
        else []
    )
    if react is not None and react_score is None:
        react_score_for_corroboration = score_router_candidate(
            task,
            react,
            "react",
            pipeline_context=pipeline,
        )
    else:
        react_score_for_corroboration = react_score
    if direct_calls and pipeline_calls:
        missing_relative_calls = max(
            0,
            max(direct_score["valid_call_count"], len(direct_calls))
            - max(pipeline_score["valid_call_count"], len(pipeline_calls)),
        )
        if missing_relative_calls and direct_score["validity"] >= 0.5:
            penalty = 1.0 * missing_relative_calls
            pipeline_score = dict(pipeline_score)
            pipeline_score["relative_missing_call_count"] = missing_relative_calls
            pipeline_score["relative_missing_call_penalty"] = round(penalty, 6)
            pipeline_score["total_before_relative_missing_call_penalty"] = pipeline_score["total"]
            pipeline_score["total"] = round(pipeline_score["total"] - penalty, 6)
        else:
            pipeline_score = dict(pipeline_score)
            pipeline_score["relative_missing_call_penalty"] = 0.0
            pipeline_score["relative_missing_call_count"] = missing_relative_calls
    pipeline_over_under_calling_react = _pipeline_over_under_calling_react_allowed(
        task,
        direct_calls,
        pipeline_calls,
        react_calls_for_corroboration,
        pipeline_score,
        react_score_for_corroboration,
    )
    pipeline_abstention_signal = (
        not pipeline_calls
        and (
            pipeline.get("verdict") == "no_intent"
            or pipeline.get("mode") in {"no_intent", "irrelevance_prefilter_abstain"}
            or pipeline.get("irrelevance_prefilter", {}).get("decision") == "abstain"
            or pipeline.get("irrelevance_guard", {}).get("decision") == "abstain"
            or pipeline.get("verification", {}).get("dropped_count", 0) > 0
        )
    )
    trust_planner_no_intent = (
        abstention_policy == "calibrated_trust_no_intent"
        and pipeline_abstention_signal
        and pipeline.get("verdict") == "no_intent"
    )
    high_precision_no_intent_override = (
        trust_planner_no_intent
        and _high_precision_abstention_override(task, direct_calls, direct_score)
    )

    if pipeline_abstention_signal and not direct_calls:
        selected = "pipeline"
        reason = "pipeline_abstention_signal"
    elif canonical(direct_calls) == canonical(pipeline_calls):
        selected = "direct"
        reason = "candidates_agree"
    elif pipeline_abstention_signal:
        if (
            (not trust_planner_no_intent or high_precision_no_intent_override)
            and abstention_policy in {"calibrated", "calibrated_trust_no_intent"}
            and _strong_direct_tool_evidence(task, direct, direct_score)
        ):
            selected = "direct"
            reason = "direct_over_uncertain_abstention"
        else:
            selected = "pipeline"
            reason = "pipeline_abstention_signal"
    elif (
        abstention_policy in {"calibrated", "calibrated_trust_no_intent"}
        and pipeline.get("verdict") == "single_intent"
        and not pipeline_calls
        and not _strong_direct_tool_evidence(task, direct, direct_score)
    ):
        selected = "pipeline"
        reason = "pipeline_empty_single_intent"
    elif _corroborated_pipeline_superset(
        direct_calls,
        pipeline_calls,
        react_calls_for_corroboration,
        pipeline_score,
        react_score_for_corroboration,
    ):
        selected = "pipeline"
        reason = "pipeline_react_corroborated_superset"
    elif _corroborated_pipeline_over_suspect_direct(
        direct_calls,
        pipeline_calls,
        react_calls_for_corroboration,
        direct_score,
        pipeline_score,
        react_score_for_corroboration,
    ):
        selected = "pipeline"
        reason = "pipeline_react_over_suspect_direct"
    elif _pipeline_react_corroborated_alternative_allowed(
        direct_calls,
        pipeline_calls,
        react_calls_for_corroboration,
        direct_score,
        pipeline_score,
        react_score_for_corroboration,
        task.get("prompt", ""),
    ):
        selected = "pipeline"
        reason = "pipeline_react_corroborated_alternative"
    elif _pipeline_has_multilingual_copy_advantage(
        task,
        direct_calls,
        pipeline_calls,
        direct_score,
        pipeline_score,
    ):
        selected = "pipeline"
        reason = "pipeline_multilingual_copy_advantage"
    elif pipeline_over_under_calling_react:
        selected = "pipeline"
        reason = "pipeline_over_under_calling_react"
    elif (
        not _clean_react_covers_pipeline(
            pipeline_calls,
            react_calls_for_corroboration,
            react_score_for_corroboration,
        )
        and _pipeline_adds_grounded_new_tool_step(
            task,
            direct_calls,
            pipeline_calls,
            direct_score,
            pipeline_score,
        )
    ):
        selected = "pipeline"
        reason = "pipeline_grounded_new_tool_step"
    elif _pipeline_can_override_single_intent_direct(
        direct_calls,
        pipeline_calls,
        direct_score,
        pipeline_score,
        margin,
        task.get("prompt", ""),
    ):
        selected = "pipeline"
        reason = "pipeline_over_single_intent_direct"
    elif (
        _pipeline_has_command_target_refinement(
            direct_calls,
            pipeline_calls,
            {tool["name"]: tool for tool in task["tools"]},
            task.get("prompt", ""),
        )
        and pipeline_score["valid_call_count"] == len(pipeline_calls)
        and pipeline_score["deterministic_drop_count"] == 0
    ):
        selected = "pipeline"
        reason = "pipeline_command_target_refinement"
    elif (
        direct_score["deterministic_drop_count"] > 0
        and _deterministic_drops_are_result_placeholders(direct_score)
        and pipeline_calls
        and pipeline_score["valid_call_count"] == len(pipeline_calls)
        and pipeline_score["deterministic_drop_count"] == 0
        and len(pipeline_calls) <= len(direct_calls)
        and pipeline_score["valid_call_count"] >= direct_score["valid_call_count"]
    ):
        selected = "pipeline"
        reason = "pipeline_clean_subset_after_deterministic_drop"
    elif (
        direct_score["deterministic_drop_count"] > 0
        and not _deterministic_drops_are_grounded_enum_only(
            direct_score,
            task.get("prompt", ""),
        )
        and pipeline_calls
        and pipeline_score["valid_call_count"] == len(pipeline_calls)
        and pipeline_score["deterministic_drop_count"] == 0
        and pipeline_score["total"] >= direct_score["total"] - 0.25
    ):
        selected = "pipeline"
        reason = "pipeline_over_deterministic_drop"
    elif pipeline.get("verdict") == "single_intent":
        selected = "direct"
        reason = "direct_for_single_intent"
    elif pipeline_score["total"] >= direct_score["total"] + margin:
        selected = "pipeline"
        reason = "pipeline_exceeds_margin"
    else:
        selected = "direct"
        reason = "direct_fallback"

    score_by_source = {"direct": direct_score, "pipeline": pipeline_score}
    candidate_by_source = {"direct": direct, "pipeline": pipeline}
    if react_score is not None:
        react_calls = react.get("normalized_calls") or []
        react_evidence = _strong_direct_tool_evidence(task, react, react_score)
        react_abstention_rescue_evidence = _high_precision_abstention_override(
            task,
            react_calls,
            react_score,
        )
        selected_score = score_by_source[selected]
        strict_abstention_selected = (
            selected == "pipeline"
            and pipeline_abstention_signal
            and (
                abstention_policy == "strict"
                or trust_planner_no_intent
            )
        )
        react_policy_allows = (
            react_policy == "score_margin"
            or (
                react_policy == "empty_only"
                and not direct_calls
                and not pipeline_calls
            )
        )
        react_omits_agreed_supported_argument = (
            _react_omits_agreed_supported_argument(
                direct_calls,
                pipeline_calls,
                react_calls,
                task.get("prompt", ""),
            )
        )
        react_cardinality_expansion = _react_cardinality_expansion_allowed(
            (candidate_by_source[selected].get("normalized_calls") or []),
            react_calls,
            selected_score,
            react_score,
        )
        react_grounded_superset = _react_grounded_superset_allowed(
            (candidate_by_source[selected].get("normalized_calls") or []),
            react_calls,
            selected_score,
            react_score,
        )
        pipeline_cleaner_subset_of_react = (
            (react_cardinality_expansion or react_grounded_superset)
            and _pipeline_cleaner_subset_of_react_allowed(
                direct_calls,
                pipeline_calls,
                react_calls,
                direct_score,
                pipeline_score,
                task,
            )
        )
        react_same_tool_pipeline_strict_win = (
            _react_same_tool_pipeline_strict_win_allowed(
                selected,
                reason,
                (candidate_by_source[selected].get("normalized_calls") or []),
                react_calls,
                selected_score,
                react_score,
            )
        )
        react_intent_switch_reduces_ungrounded_scalars = (
            _react_intent_switch_reduces_ungrounded_scalars_allowed(
                (candidate_by_source[selected].get("normalized_calls") or []),
                react_calls,
                selected_score,
                react_score,
            )
        )
        react_location_normalization = _react_location_normalization_allowed(
            reason,
            (candidate_by_source[selected].get("normalized_calls") or []),
            react_calls,
            react_score,
        )
        react_schema_example_id_replaces_boolean = (
            _react_schema_example_id_replaces_boolean_allowed(
                task,
                reason,
                (candidate_by_source[selected].get("normalized_calls") or []),
                react_calls,
                react_score,
            )
        )
        if (
            react_calls
            and not strict_abstention_selected
            and react_policy_allows
            and (
                pipeline_calls
                or len(task.get("tools") or []) != 1
                or react_abstention_rescue_evidence
            )
            and not pipeline_over_under_calling_react
            and (
                not react_omits_agreed_supported_argument
                or react_intent_switch_reduces_ungrounded_scalars
            )
            and react_score["valid_call_count"] == len(react_calls)
            and react_score["deterministic_drop_count"] == 0
            and (
                react_score["total"] >= selected_score["total"] + margin
                or react_cardinality_expansion
                or react_grounded_superset
                or react_same_tool_pipeline_strict_win
                or react_intent_switch_reduces_ungrounded_scalars
                or react_location_normalization
                or react_schema_example_id_replaces_boolean
                or (
                    selected == "pipeline"
                    and pipeline_abstention_signal
                    and abstention_policy in {"calibrated", "calibrated_trust_no_intent"}
                    and react_abstention_rescue_evidence
                )
            )
        ):
            if pipeline_cleaner_subset_of_react:
                selected = "pipeline"
                reason = "pipeline_cleaner_subset_of_react"
            else:
                selected = "react"
                if react_cardinality_expansion:
                    reason = "react_cardinality_expansion"
                elif react_grounded_superset:
                    reason = "react_grounded_superset"
                elif react_same_tool_pipeline_strict_win:
                    reason = "react_same_tool_pipeline_strict_win"
                elif react_intent_switch_reduces_ungrounded_scalars:
                    reason = "react_intent_switch_reduces_ungrounded_scalars"
                elif react_location_normalization:
                    reason = "react_location_normalization"
                elif react_schema_example_id_replaces_boolean:
                    reason = "react_schema_example_id_replaces_boolean"
                else:
                    reason = "react_exceeds_margin" if not pipeline_abstention_signal else "react_over_uncertain_abstention"
        if react_omits_agreed_supported_argument:
            react_score = dict(react_score)
            react_score["omits_agreed_supported_argument"] = True
        score_by_source["react"] = react_score
        candidate_by_source["react"] = react

    chosen = repair_prompt_grounded_optional_arguments(task, candidate_by_source[selected])
    chosen = repair_prompt_relative_date_arguments(task, chosen)
    chosen = repair_prompt_temporal_default_arguments(task, chosen)
    chosen = repair_optional_default_arguments(task, chosen)
    chosen = repair_prompt_span_trimmed_values(task, chosen)
    chosen = repair_narrow_prompt_schema_values(task, chosen)
    chosen = repair_generic_entity_suffixes(task, chosen)
    chosen = repair_context_entity_values(task, chosen)
    chosen = repair_city_values(task, chosen)
    chosen = repair_temporal_values(task, chosen)
    chosen = repair_prompt_plural_list_items(task, chosen)
    chosen = repair_prompt_grounded_pickup_arguments(task, chosen)
    chosen = repair_derived_polynomial_arguments(task, chosen)
    request_usages = list(direct.get("request_usages") or []) + list(
        pipeline.get("request_usages") or []
    )
    if react is not None:
        request_usages += list(react.get("request_usages") or [])
    return {
        "latency_ms": direct.get("latency_ms", 0)
        + pipeline.get("total_latency_ms", pipeline.get("latency_ms", 0)),
        "total_latency_ms": direct.get("latency_ms", 0)
        + pipeline.get("total_latency_ms", pipeline.get("latency_ms", 0)),
        "normalized_calls": chosen.get("normalized_calls") or [],
        "score": score_calls(
            chosen.get("normalized_calls") or [],
            task.get("expected_calls", []),
        ),
        "raw": None,
        "request_usages": request_usages,
        "usage": aggregate_request_usages(request_usages),
        "mode": "selective_router",
        "decomposition": pipeline.get("decomposition", ""),
        "verdict": pipeline.get("verdict"),
        "router": {
            "selected": selected,
            "reason": reason,
            "margin": margin,
            "abstention_policy": abstention_policy,
            "react_policy": react_policy,
            "direct": direct_score,
            "pipeline": pipeline_score,
            **({"react": react_score} if react_score is not None else {}),
        },
        **(
            {
                "prompt_grounded_optional_fill": chosen[
                    "prompt_grounded_optional_fill"
                ]
            }
            if "prompt_grounded_optional_fill" in chosen
            else {}
        ),
        **(
            {"optional_default_repair": chosen["optional_default_repair"]}
            if "optional_default_repair" in chosen
            else {}
        ),
        **(
            {"relative_date_repair": chosen["relative_date_repair"]}
            if "relative_date_repair" in chosen
            else {}
        ),
        **(
            {"temporal_default_repair": chosen["temporal_default_repair"]}
            if "temporal_default_repair" in chosen
            else {}
        ),
        **(
            {"prompt_span_trim_repair": chosen["prompt_span_trim_repair"]}
            if "prompt_span_trim_repair" in chosen
            else {}
        ),
        **(
            {"narrow_value_repair": chosen["narrow_value_repair"]}
            if "narrow_value_repair" in chosen
            else {}
        ),
        **(
            {"entity_suffix_span_repair": chosen["entity_suffix_span_repair"]}
            if "entity_suffix_span_repair" in chosen
            else {}
        ),
        **(
            {"context_entity_value_repair": chosen["context_entity_value_repair"]}
            if "context_entity_value_repair" in chosen
            else {}
        ),
        **(
            {"city_value_repair": chosen["city_value_repair"]}
            if "city_value_repair" in chosen
            else {}
        ),
        **(
            {"temporal_value_repair": chosen["temporal_value_repair"]}
            if "temporal_value_repair" in chosen
            else {}
        ),
        **(
            {
                "prompt_plural_list_item_repair": chosen[
                    "prompt_plural_list_item_repair"
                ]
            }
            if "prompt_plural_list_item_repair" in chosen
            else {}
        ),
        **(
            {"prompt_grounded_pickup_fill": chosen["prompt_grounded_pickup_fill"]}
            if "prompt_grounded_pickup_fill" in chosen
            else {}
        ),
        **(
            {"derived_polynomial_repair": chosen["derived_polynomial_repair"]}
            if "derived_polynomial_repair" in chosen
            else {}
        ),
    }


def run_cardinality_pipeline(task, args):
    original_plan = run_decomposition(task, args)
    repair = None
    plan = original_plan
    repair_applied = False
    repair_decision = "not_run"
    repair_latency_ms = 0
    plan_usages = [original_plan["usage"]]
    if original_plan["verdict"] != "no_intent":
        repair = run_cardinality_repair(task, args, original_plan)
        repair_latency_ms = repair["latency_ms"]
        plan_usages.append(repair["usage"])
        plan, repair_applied, repair_decision = select_cardinality_plan(
            task,
            original_plan,
            repair,
        )

    verdict = plan["verdict"]
    if verdict == "no_intent":
        candidate = {
            "latency_ms": 0,
            "total_latency_ms": original_plan["latency_ms"] + repair_latency_ms,
            "decomposition_latency_ms": original_plan["latency_ms"],
            "cardinality_repair_latency_ms": repair_latency_ms,
            "decomposition": plan["decomposition"],
            "normalized_calls": [],
            "raw": {"mode": "cardinality_no_intent"},
            "decomposition_raw": original_plan["raw"],
            "cardinality_repair_raw": repair["raw"] if repair else None,
            "mode": "cardinality",
            "request_usages": plan_usages,
            "usage": aggregate_request_usages(plan_usages),
        }
    elif verdict == "single_intent":
        selector = (
            run_pipeline_single_selector(task, args)
            if getattr(args, "strict_value_copy", False)
            or getattr(args, "value_copy_fewshot", False)
            or getattr(args, "span_inventory", False)
            else run_baseline(task, args)
        )
        usages = plan_usages + selector["request_usages"]
        candidate = {
            "latency_ms": selector["latency_ms"],
            "total_latency_ms": (
                original_plan["latency_ms"]
                + repair_latency_ms
                + selector["latency_ms"]
            ),
            "decomposition_latency_ms": original_plan["latency_ms"],
            "cardinality_repair_latency_ms": repair_latency_ms,
            "decomposition": plan["decomposition"],
            "normalized_calls": selector["normalized_calls"],
            "raw": {"mode": "cardinality_single_intent", "selector": selector["raw"]},
            "decomposition_raw": original_plan["raw"],
            "cardinality_repair_raw": repair["raw"] if repair else None,
            "mode": "cardinality",
            "request_usages": usages,
            "usage": aggregate_request_usages(usages),
        }
    elif args.pipeline_mode == "context":
        candidate = run_context_pipeline(task, args, plan)
        candidate["total_latency_ms"] += original_plan["latency_ms"] + repair_latency_ms - plan["latency_ms"]
        candidate["decomposition_latency_ms"] = original_plan["latency_ms"]
        candidate["cardinality_repair_latency_ms"] = repair_latency_ms
        candidate["decomposition_raw"] = original_plan["raw"]
        candidate["cardinality_repair_raw"] = repair["raw"] if repair else None
        candidate["request_usages"] = plan_usages + candidate["request_usages"][1:]
        candidate["usage"] = aggregate_request_usages(candidate["request_usages"])
        candidate["mode"] = "cardinality"
    else:
        candidate = run_per_subtask_pipeline(task, args, plan)
        candidate["total_latency_ms"] += original_plan["latency_ms"] + repair_latency_ms - plan["latency_ms"]
        candidate["decomposition_latency_ms"] = original_plan["latency_ms"]
        candidate["cardinality_repair_latency_ms"] = repair_latency_ms
        candidate["decomposition_raw"] = original_plan["raw"]
        candidate["cardinality_repair_raw"] = repair["raw"] if repair else None
        candidate["request_usages"] = plan_usages + candidate["request_usages"][1:]
        candidate["usage"] = aggregate_request_usages(candidate["request_usages"])
        candidate["mode"] = "cardinality"

    verification = verify_candidate_calls(
        task,
        args,
        candidate["normalized_calls"],
    )
    usages = list(candidate["request_usages"])
    if verification["usage"] is not None:
        usages.append(verification["usage"])
    calls = verification["calls"]
    candidate.update(
        {
            "normalized_calls": calls,
            "score": score_calls(calls, task.get("expected_calls", [])),
            "total_latency_ms": candidate["total_latency_ms"]
            + verification["latency_ms"],
            "verification_latency_ms": verification["latency_ms"],
            "verification": {
                "dropped": verification["dropped"],
                "dropped_count": verification["dropped_count"],
                "argument_drops": verification["argument_drops"],
                "argument_dropped_count": verification["argument_dropped_count"],
                "merged_count": verification["merged_count"],
                "parse_failed": verification["parse_failed"],
                "raw": verification["raw"],
            },
            "request_usages": usages,
            "usage": aggregate_request_usages(usages),
            "verdict": verdict,
            "original_verdict": original_plan["verdict"],
            "original_decomposition": original_plan["decomposition"],
            "cardinality_repair": {
                "applied": repair_applied,
                "decision": repair_decision,
                "raw": repair["raw"] if repair else None,
                "decomposition": repair["decomposition"] if repair else None,
            },
        }
    )
    return candidate


def lane_order(order, index):
    if order == "baseline-first":
        return ["baseline", "pipeline"]
    if order == "pipeline-first":
        return ["pipeline", "baseline"]
    return ["baseline", "pipeline"] if index % 2 == 0 else ["pipeline", "baseline"]


def result_lanes(results):
    lane_order = [
        "baseline",
        "react",
        "pipeline",
        "selective_pipeline",
        "grammar_pipeline",
        "cardinality_pipeline",
    ]
    if not results:
        return []
    return [lane for lane in lane_order if lane in results[0]]


def summarize_runtime_diagnostics(results, context_length=None):
    diagnostics = {}
    for lane in result_lanes(results):
        lane_rows = [row[lane] for row in results]
        request_usages = [
            usage
            for lane_row in lane_rows
            for usage in lane_row.get("request_usages", [])
        ]
        latency_key = (
            "total_latency_ms"
            if lane
            in {
                "pipeline",
                "selective_pipeline",
                "grammar_pipeline",
                "cardinality_pipeline",
            }
            else "latency_ms"
        )
        lane_diagnostics = {
            "avg_latency_ms": round(
                sum(row.get(latency_key, 0) for row in lane_rows) / len(lane_rows),
                1,
            ),
            "avg_prompt_tokens": round(
                sum(row.get("usage", {}).get("prompt_tokens", 0) for row in lane_rows)
                / len(lane_rows),
                1,
            ),
            "max_request_prompt_tokens": max(
                [usage.get("prompt_tokens", 0) for usage in request_usages] or [0]
            ),
            "max_request_total_tokens": max(
                [usage.get("total_tokens", 0) for usage in request_usages] or [0]
            ),
            "request_count": len(request_usages),
            "abstention_count": sum(
                1 for row in lane_rows if not row.get("normalized_calls")
            ),
            "abstention_rate": round(
                sum(1 for row in lane_rows if not row.get("normalized_calls"))
                / len(lane_rows),
                3,
            ),
        }
        if context_length:
            lane_diagnostics["requests_at_or_above_context"] = sum(
                1
                for usage in request_usages
                if usage.get("total_tokens", 0) >= context_length
            )
        if lane == "pipeline":
            verdict_counts = {}
            for row in lane_rows:
                verdict = row.get("verdict", "unknown")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            lane_diagnostics.update(
                {
                    "verdict_counts": verdict_counts,
                    "verification_dropped_count": sum(
                        row.get("verification", {}).get("dropped_count", 0)
                        for row in lane_rows
                    ),
                    "verification_argument_dropped_count": sum(
                        row.get("verification", {}).get(
                            "argument_dropped_count",
                            0,
                        )
                        for row in lane_rows
                    ),
                    "verification_merged_count": sum(
                        row.get("verification", {}).get("merged_count", 0)
                        for row in lane_rows
                    ),
                    "verification_parse_failures": sum(
                        1
                        for row in lane_rows
                        if row.get("verification", {}).get("parse_failed")
                    ),
                    "irrelevance_guard_abstain_count": sum(
                        1
                        for row in lane_rows
                        if row.get("irrelevance_guard", {}).get("decision")
                        == "abstain"
                    ),
                    "irrelevance_guard_parse_failures": sum(
                        1
                        for row in lane_rows
                        if row.get("irrelevance_guard", {}).get("parse_failed")
                    ),
                    "irrelevance_prefilter_abstain_count": sum(
                        1
                        for row in lane_rows
                        if row.get("irrelevance_prefilter", {}).get("decision")
                        == "abstain"
                    ),
                    "irrelevance_prefilter_parse_failures": sum(
                        1
                        for row in lane_rows
                        if row.get("irrelevance_prefilter", {}).get("parse_failed")
                    ),
                }
            )
        if lane == "selective_pipeline":
            selected_counts = {}
            reason_counts = {}
            for row in lane_rows:
                selected = row.get("router", {}).get("selected", "unknown")
                reason = row.get("router", {}).get("reason", "unknown")
                selected_counts[selected] = selected_counts.get(selected, 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            lane_diagnostics.update(
                {
                    "router_selected_counts": selected_counts,
                    "router_reason_counts": reason_counts,
                }
            )
        if lane == "grammar_pipeline":
            lane_diagnostics.update(
                {
                    "grammar_parse_failures": sum(
                        row.get("grammar_diagnostics", {}).get(
                            "parse_failure_count",
                            0,
                        )
                        for row in lane_rows
                    ),
                    "grammar_hallucinated_tools": sum(
                        row.get("grammar_diagnostics", {}).get(
                            "hallucinated_tool_count",
                            0,
                        )
                        for row in lane_rows
                    ),
                    "verification_dropped_count": sum(
                        row.get("verification", {}).get("dropped_count", 0)
                        for row in lane_rows
                    ),
                    "verification_argument_dropped_count": sum(
                        row.get("verification", {}).get(
                            "argument_dropped_count",
                            0,
                        )
                        for row in lane_rows
                    ),
                    "no_intent_count": sum(
                        row.get("verdict") == "no_intent" for row in lane_rows
                    ),
                    "grammar_enabled": all(
                        row.get("grammar_enabled") for row in lane_rows
                    ),
                }
            )
        if lane == "cardinality_pipeline":
            verdict_counts = {}
            original_verdict_counts = {}
            for row in lane_rows:
                verdict = row.get("verdict", "unknown")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                original = row.get("original_verdict", "unknown")
                original_verdict_counts[original] = (
                    original_verdict_counts.get(original, 0) + 1
                )
            lane_diagnostics.update(
                {
                    "verdict_counts": verdict_counts,
                    "original_verdict_counts": original_verdict_counts,
                    "verification_dropped_count": sum(
                        row.get("verification", {}).get("dropped_count", 0)
                        for row in lane_rows
                    ),
                    "verification_argument_dropped_count": sum(
                        row.get("verification", {}).get(
                            "argument_dropped_count",
                            0,
                        )
                        for row in lane_rows
                    ),
                    "verification_merged_count": sum(
                        row.get("verification", {}).get("merged_count", 0)
                        for row in lane_rows
                    ),
                    "cardinality_repair_applied_count": sum(
                        row.get("cardinality_repair", {}).get("applied", False)
                        for row in lane_rows
                    ),
                }
            )
        diagnostics[lane] = lane_diagnostics
    return diagnostics


def summarize(results):
    lanes = result_lanes(results)
    summary = {}
    for lane in lanes:
        exact = [1 if row[lane]["score"]["exact"] else 0 for row in results]
        precision = [row[lane]["score"]["precision"] for row in results]
        recall = [row[lane]["score"]["recall"] for row in results]
        latency_key = (
            "total_latency_ms"
            if lane
            in {
                "pipeline",
                "selective_pipeline",
                "grammar_pipeline",
                "cardinality_pipeline",
            }
            else "latency_ms"
        )
        latency = [row[lane][latency_key] for row in results]
        summary[lane] = {
            "exact_accuracy": round(sum(exact) / len(exact), 3) if exact else 0,
            "avg_precision": round(sum(precision) / len(precision), 3) if precision else 0,
            "avg_recall": round(sum(recall) / len(recall), 3) if recall else 0,
            "avg_latency_ms": round(sum(latency) / len(latency), 1) if latency else 0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Smoke-test baseline vs decomposition tool calling.")
    parser.add_argument("--tasks", default="benchmark_tasks/smoke_tool_tasks.json")
    parser.add_argument("--output-dir", default=os.environ.get("RESULTS_DIR", "benchmark_results/smoke_tool_eval"))
    parser.add_argument("--endpoint", default=os.environ.get("QWEN_ENDPOINT", "http://localhost:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", os.environ.get("MODEL", "qwen2.5:14b")))
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("QWEN_TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("QWEN_TIMEOUT", "120")))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("TASK_LIMIT", "0")))
    parser.add_argument(
        "--pipeline-mode",
        choices=["per-subtask", "context"],
        default=os.environ.get("PIPELINE_MODE", "per-subtask"),
    )
    parser.add_argument(
        "--order",
        choices=["alternate", "baseline-first", "pipeline-first"],
        default=os.environ.get("LANE_ORDER", "alternate"),
    )
    parser.add_argument(
        "--drop-ungrounded-optional-args",
        action="store_true",
        default=os.environ.get("DROP_UNGROUNDED_OPTIONAL_ARGS", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--strict-value-copy",
        action="store_true",
        default=os.environ.get("STRICT_VALUE_COPY", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--value-copy-fewshot",
        action="store_true",
        default=os.environ.get("VALUE_COPY_FEWSHOT", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--irrelevance-guard",
        action="store_true",
        default=os.environ.get("IRRELEVANCE_GUARD", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--irrelevance-prefilter",
        action="store_true",
        default=os.environ.get("IRRELEVANCE_PREFILTER", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--span-inventory",
        action="store_true",
        default=os.environ.get("SPAN_INVENTORY", "0")
        in {"1", "true", "TRUE", "yes", "YES"},
    )
    parser.add_argument(
        "--multseq-k",
        type=int,
        default=int(os.environ.get("MULTSEQ_K", "1")),
    )
    parser.add_argument(
        "--multseq-temperature",
        type=float,
        default=float(os.environ.get("MULTSEQ_TEMPERATURE", "0.5")),
    )
    parser.add_argument(
        "--multseq-min-votes",
        type=int,
        default=int(os.environ.get("MULTSEQ_MIN_VOTES", "2")),
    )
    parser.add_argument(
        "--multseq-strategy",
        choices=["vote", "medoid", "score_filter", "saf"],
        default=os.environ.get("MULTSEQ_STRATEGY", "vote"),
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    if args.limit:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "endpoint": args.endpoint, "model": args.model}, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_latency_ms = None
    if not args.skip_warmup:
        print("warming model...", flush=True)
        warmup_latency_ms = run_warmup(args)
        print(f"  warmup latency={warmup_latency_ms} ms", flush=True)

    results = []
    for index, task in enumerate(tasks):
        print(f"running {task['id']}...", flush=True)
        baseline = None
        pipeline = None
        order = lane_order(args.order, index)
        print(f"  lane order: {' -> '.join(order)}", flush=True)
        for lane in order:
            if lane == "baseline":
                baseline = run_baseline(task, args)
            else:
                pipeline = run_pipeline(task, args)
        assert baseline is not None and pipeline is not None
        results.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "expected_calls": task.get("expected_calls", []),
                "baseline": baseline,
                "pipeline": pipeline,
            }
        )
        print(
            f"  baseline exact={baseline['score']['exact']} calls={len(baseline['normalized_calls'])}; "
            f"pipeline exact={pipeline['score']['exact']} calls={len(pipeline['normalized_calls'])}",
            flush=True,
        )

    report = {
        "model": args.model,
        "endpoint": args.endpoint,
        "lane_order": args.order,
        "warmup_latency_ms": warmup_latency_ms,
        "summary": summarize(results),
        "diagnostics": summarize_runtime_diagnostics(
            results,
            context_length=int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "0") or 0)
            or None,
        ),
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
