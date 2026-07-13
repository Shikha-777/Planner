#!/usr/bin/env python3
"""Run a custom agent against VAKRA MCP containers.

This runner is intentionally independent of VAKRA's Python package imports so it
can run from a lightweight environment while the MCP servers run inside VAKRA's
Python 3.11 Docker image.  It writes VAKRA's submission JSON schema:

  output/<domain>.json

The `ensemble` agent path reuses the GPT-OSS adapter bank from
tau_ensemble_agent.py.  The `scripted-smoke` path is only for harness testing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


CAPABILITY_DIRS = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
    3: "capability_3_multihop_reasoning",
    4: "capability_4_multiturn",
}

DEFAULT_CONTAINERS = {
    1: "capability_1_bi_apis",
    2: "capability_2_dashboard_apis",
    3: "capability_3_multihop_reasoning",
    4: "capability_4_multiturn",
}

RESPOND_NAMES = {"respond", "final", "final_answer", "answer"}
GETTER_SCHEMA = {
    "type": "object",
    "properties": {
        "data_label": {
            "type": "string",
            "description": "Handle string returned by get_data or a previous data-manipulation tool.",
        }
    },
    "required": ["data_label"],
    "additionalProperties": False,
}
FIELD_MATCH_STOP_TOKENS = {
    "brand",
    "brands",
    "name",
    "names",
    "page",
    "data",
    "value",
    "root",
    "beer",
    "rootbeer",
    "customer",
    "customers",
    "description",
    "descriptions",
    "city",
    "cities",
    "state",
    "country",
    "many",
    "purchased",
    "consumed",
}


def short_json(value: Any, limit: int = 28000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def parse_jsonish(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def result_to_value(result: Any) -> Any:
    content = getattr(result, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if text is not None:
            return parse_jsonish(text)
    return str(result)


def compact_observation(value: Any, *, depth: int = 0, list_limit: int = 8, text_limit: int = 1800) -> Any:
    if depth > 4:
        return short_json(value, limit=1200)
    if isinstance(value, str):
        return value if len(value) <= text_limit else value[:text_limit] + "...<truncated>"
    if isinstance(value, list):
        items = [compact_observation(item, depth=depth + 1, list_limit=list_limit, text_limit=text_limit) for item in value[:list_limit]]
        if len(value) > list_limit:
            items.append({"_truncated_items": len(value) - list_limit})
        return items
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            compact[str(key)] = compact_observation(item, depth=depth + 1, list_limit=list_limit, text_limit=text_limit)
        return compact
    return value


def _tokens(text: Any) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    return [tok for tok in re.findall(r"[a-z0-9]+", spaced.lower()) if len(tok) >= 4]


def _field_names(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = _first_json_object(value)
        if parsed is not None:
            return _field_names(parsed)
        names = re.findall(r'"(?:name|key_name)"\s*:\s*"([^"]+)"', value)
        for known in ("BrandName", "FacebookPage", "Website", "Twitter"):
            if known in value and known not in names:
                names.append(known)
        return names
    names: list[str] = []
    if isinstance(value, dict):
        for key in ("key_details", "columns", "fields"):
            raw = value.get(key)
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and (item.get("name") or item.get("key_name")):
                        names.append(str(item.get("name") or item["key_name"]))
                    elif isinstance(item, str):
                        names.append(item)
        for item in value.values():
            names.extend(_field_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(_field_names(item))
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _first_json_object(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _explicit_final_action(text: str) -> Optional["AgentStep"]:
    for value in reversed(_json_objects(text)):
        if not isinstance(value, dict):
            continue
        name = str(value.get("name", ""))
        if name not in RESPOND_NAMES:
            continue
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        answer = arguments.get("answer") or arguments.get("content") or arguments.get("message")
        if answer is None:
            continue
        return AgentStep(name=name, arguments={"answer": str(answer)}, raw=text)
    return None


def _result_handle(value: Any) -> str:
    if isinstance(value, str):
        parsed = _first_json_object(value)
        if parsed is not None:
            return _result_handle(parsed)
        match = re.search(r"\b(?:filtered|retrieved)_data_\d+\b", value)
        return match.group(0) if match else ""
    if isinstance(value, dict):
        for key in ("handle", "data_label", "label"):
            if value.get(key):
                return str(value[key])
    return ""


def _num_records(value: Any) -> Optional[int]:
    if isinstance(value, str):
        parsed = _first_json_object(value)
        if parsed is not None:
            return _num_records(parsed)
        match = re.search(r'"?num_records"?\s*[:=]\s*(\d+)', value)
        if not match:
            match = re.search(r"\bwith\s+(\d+)\s+records\b", value, re.IGNORECASE)
        return int(match.group(1)) if match else None
    if isinstance(value, dict):
        for key in ("num_records", "count", "n_records"):
            raw = value.get(key)
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str) and raw.isdigit():
                return int(raw)
    return None


def _extract_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        parsed = _first_json_object(value)
        if parsed is not None:
            return _extract_values(parsed)
        return []
    if isinstance(value, dict):
        for key in ("unique_values", "unique_array", "values", "result", "data"):
            raw = value.get(key)
            if isinstance(raw, list):
                values.extend(str(item) for item in raw if item not in (None, ""))
        for item in value.values():
            values.extend(_extract_values(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (str, int, float)) and item not in (None, ""):
                values.append(str(item))
            else:
                values.extend(_extract_values(item))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _extract_scalar(value: Any) -> str:
    if isinstance(value, str):
        parsed = _first_json_object(value)
        if parsed is not None:
            return _extract_scalar(parsed)
        stripped = value.strip()
        return stripped
    if isinstance(value, dict):
        for key in ("mean", "average", "result", "value", "count"):
            raw = value.get(key)
            if isinstance(raw, (str, int, float)) and raw != "":
                return str(raw)
        for raw in value.values():
            scalar = _extract_scalar(raw)
            if scalar:
                return scalar
    if isinstance(value, list) and value:
        return _extract_scalar(value[0])
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _getter_for_field(tool_names: set[str], field_name: str) -> str:
    candidates = [
        f"get_{field_name}s",
        f"get_{field_name}",
        f"get_{field_name.lower()}s",
        f"get_{field_name.lower()}",
    ]
    for candidate in candidates:
        if candidate in tool_names:
            return candidate
    field_tokens = set(_tokens(field_name))
    for tool_name in sorted(tool_names):
        if not tool_name.startswith("get_"):
            continue
        if field_tokens and field_tokens <= set(_tokens(tool_name)):
            return tool_name
    return ""


def _hidden_getter_tools(initial_data: Any, known_tool_names: set[str]) -> list[Any]:
    tools: list[Any] = []
    for field_name in _field_names(initial_data):
        name = f"get_{field_name}s"
        if name in known_tool_names:
            continue
        tools.append(
            SimpleNamespace(
                name=name,
                description=f"Retrieve the {field_name} column from a data handle.",
                inputSchema=GETTER_SCHEMA,
            )
        )
        known_tool_names.add(name)
    return tools


def _target_field_for_query(query: str, field_names: list[str]) -> str:
    query_tokens = set(_tokens(query))
    preferences: list[set[str]] = []
    if "description" in query_tokens:
        preferences.append({"description"})
    if {"city", "cities"} & query_tokens:
        preferences.append({"city"})
    if {"brand", "brands", "name", "names"} & query_tokens:
        if not preferences:
            brand_name = _field_named(field_names, "BrandName")
            if brand_name:
                return brand_name
        preferences.append({"brand", "name"})
    for wanted in preferences:
        for field_name in field_names:
            field_tokens = set(_tokens(field_name))
            if wanted <= field_tokens or field_tokens & wanted:
                return field_name
    return ""


def _brand_entity_from_query(query: str) -> str:
    match = re.search(r"\bbrand\s+([A-Za-z0-9&'. -]+?)(?:\?|$)", query, re.IGNORECASE)
    if not match:
        known = re.search(r"\b(A&W)\b", query, re.IGNORECASE)
        if known:
            return known.group(1)
        return ""
    value = match.group(1).strip(" .?")
    value = re.sub(r"^the\s+", "", value, flags=re.IGNORECASE).strip()
    return value


def _field_named(field_names: list[str], *wanted: str) -> str:
    wanted_lower = {item.lower() for item in wanted}
    for field_name in field_names:
        lowered = field_name.lower()
        if lowered in wanted_lower or any(lowered.endswith(item.lower()) for item in wanted):
            return field_name
    for field_name in field_names:
        field_tokens = set(_tokens(field_name))
        if any(item.lower() in field_tokens for item in wanted):
            return field_name
    return ""


def _field_with_tokens(field_names: list[str], *wanted: str) -> str:
    wanted_tokens = {item.lower() for item in wanted}
    for field_name in field_names:
        if wanted_tokens <= set(_tokens(field_name)):
            return field_name
    return ""


def _year_from_query(query: str) -> Optional[int]:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    return int(match.group(1)) if match else None


def _month_range_from_query(query: str) -> tuple[str, str]:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    lowered = query.lower()
    year = _year_from_query(query)
    if not year:
        return "", ""
    for name, month in months.items():
        if re.search(rf"\b{name}\b", lowered):
            next_year = year + 1 if month == 12 else year
            next_month = 1 if month == 12 else month + 1
            return f"{year:04d}-{month:02d}-01", f"{next_year:04d}-{next_month:02d}-01"
    return "", ""


def _slash_date(month: str, day: str, year: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _before_date_from_query(query: str) -> str:
    match = re.search(r"\bbefore\s+(\d{1,2})/(\d{1,2})/(\d{4})\b", query, re.IGNORECASE)
    return _slash_date(match.group(1), match.group(2), match.group(3)) if match else ""


def _slash_date_range_from_query(query: str) -> tuple[str, str]:
    match = re.search(
        r"\bfrom\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+to\s+(\d{1,2})/(\d{1,2})/(\d{4})\b",
        query,
        re.IGNORECASE,
    )
    if not match:
        return "", ""
    return (
        _slash_date(match.group(1), match.group(2), match.group(3)),
        _slash_date(match.group(4), match.group(5), match.group(6)),
    )


def _brand_id_from_query(query: str) -> str:
    match = re.search(r"\bbrand\s+ID\s+(\d+)\b", query, re.IGNORECASE)
    return match.group(1) if match else ""


def _star_rating_from_query(query: str) -> Optional[int]:
    match = re.search(r"\b([1-5])[-\s]+stars?\b", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _rating_threshold_from_query(query: str) -> Optional[int]:
    match = re.search(r"\bstar\s+rating\s+of\s+more\s+than\s+(\d+)\b", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _mentions_can_container(query: str) -> bool:
    return bool(re.search(r"\b(can|cans|canned)\b", query, re.IGNORECASE))


def _latest_handle(history: list[dict[str, Any]]) -> str:
    for rec in reversed(history):
        handle = _result_handle(rec.get("result"))
        if handle:
            return handle
    return ""


def _values_from_tool(history: list[dict[str, Any]], tool_fragment: str) -> list[str]:
    for rec in reversed(history):
        if tool_fragment in str(rec.get("tool_name", "")):
            return _extract_values(rec.get("result"))
    return []


def _format_pairs(left: list[str], right: list[str], sep: str = " ") -> str:
    pairs: list[str] = []
    seen: set[str] = set()
    for first, second in zip(left, right):
        item = f"{first}{sep}{second}".strip()
        if item and item not in seen:
            pairs.append(item)
            seen.add(item)
    return ", ".join(pairs)


def _person_from_query(query: str) -> tuple[str, str]:
    for match in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", query):
        first, last = match.group(1), match.group(2)
        if first in {"What", "From", "How", "Please"}:
            continue
        if last.lower() in {"beer", "brands", "brand", "cities", "customers"}:
            continue
        return first, last
    return "", ""


def clean_arguments(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return {str(k): v for k, v in args.items()}
    if args is None:
        return {}
    return {"value": args}


def coerce_value_to_schema(value: Any, schema: Optional[dict[str, Any]]) -> Any:
    if not isinstance(schema, dict):
        return value
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null = [s for s in schema["anyOf"] if isinstance(s, dict) and s.get("type") != "null"]
        schema = non_null[0] if non_null else schema
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0] if schema_type else None)

    if schema_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
                return int(stripped)
    elif schema_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
    elif schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
    elif schema_type == "string":
        if not isinstance(value, str):
            return str(value)
    elif schema_type == "array":
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        values = value if isinstance(value, list) else [value]
        return [coerce_value_to_schema(v, item_schema) for v in values]
    elif schema_type == "object" and isinstance(value, dict):
        return coerce_args_to_schema(value, schema)
    return value


def coerce_args_to_schema(args: dict[str, Any], schema: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return args
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return args
    filtered = {}
    for key, value in args.items():
        if key in {"args", "kwargs", "config"}:
            continue
        if schema.get("additionalProperties") is False and key not in properties:
            continue
        filtered[key] = coerce_value_to_schema(value, properties.get(key))
    return filtered


def tool_to_openai(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(getattr(tool, "description", "") or ""),
            "parameters": schema,
        },
    }


def compact_tool(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None) or {}
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    compact_props = {}
    if isinstance(props, dict):
        for name, prop in props.items():
            compact_props[name] = {"type": prop.get("type")} if isinstance(prop, dict) else {}
    return {
        "name": str(tool.name),
        "description": str(getattr(tool, "description", "") or "")[:240],
        "required": schema.get("required", []) if isinstance(schema, dict) else [],
        "properties": compact_props,
    }


def compact_data_context(value: Any) -> Any:
    handle = _result_handle(value)
    fields = _field_names(value)
    records = _num_records(value)
    if handle or fields or records is not None:
        return {
            "handle": handle,
            "num_records": records,
            "fields": fields,
        }
    return compact_observation(value, text_limit=900)


@dataclass
class BenchmarkItem:
    uuid: str
    domain: str
    query: str
    turn_id: int = 0
    context: list[dict[str, str]] = field(default_factory=list)
    additional_instructions: str = ""


@dataclass
class AgentStep:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


class BaseVakraAgent:
    async def next_step(
        self,
        *,
        item: BenchmarkItem,
        initial_data: Any,
        tools: list[Any],
        history: list[dict[str, Any]],
    ) -> AgentStep:
        raise NotImplementedError


class ScriptedSmokeAgent(BaseVakraAgent):
    """Tiny deterministic agent for validating the VAKRA MCP/output plumbing."""

    async def next_step(
        self,
        *,
        item: BenchmarkItem,
        initial_data: Any,
        tools: list[Any],
        history: list[dict[str, Any]],
    ) -> AgentStep:
        handle = initial_data.get("handle") if isinstance(initial_data, dict) else None
        query = item.query.lower()
        if not history and handle and "facebook" in query:
            return AgentStep(
                name="select_data_contains",
                arguments={"data_label": handle, "key_name": "FacebookPage", "value": "facebook"},
            )
        if not history and handle and "how many" in query:
            return AgentStep(
                name="compute_data_count",
                arguments={"data_label": handle, "key_name": "BrandID", "distinct": False},
            )
        return AgentStep(
            name="final_answer",
            arguments={"answer": self._last_observation_answer(history)},
        )

    def _last_observation_answer(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "No tool call was made."
        obs = history[-1].get("result")
        if isinstance(obs, dict) and "num_records" in obs:
            return str(obs.get("num_records"))
        return str(obs)


class GPTOSSVakraAgent(BaseVakraAgent):
    def __init__(self, args: argparse.Namespace) -> None:
        from tau_ensemble_agent import (
            DEFAULT_EXECUTOR,
            DEFAULT_PLANNER,
            DEFAULT_RECOVERY,
            GPTOSSAdapterBank,
            parse_action_text,
        )

        self.parse_action_text = parse_action_text
        self.bank = GPTOSSAdapterBank(
            args.executor_adapter or os.environ.get("TAU_ENSEMBLE_EXECUTOR_ADAPTER", DEFAULT_EXECUTOR),
            args.planner_adapter or os.environ.get("TAU_ENSEMBLE_PLANNER_ADAPTER", DEFAULT_PLANNER),
            args.recovery_adapter or os.environ.get("TAU_ENSEMBLE_RECOVERY_ADAPTER", DEFAULT_RECOVERY),
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.capability_id = args.capability_id

    async def next_step(
        self,
        *,
        item: BenchmarkItem,
        initial_data: Any,
        tools: list[Any],
        history: list[dict[str, Any]],
    ) -> AgentStep:
        visible_tools = [t for t in tools if t.name != "get_data"]
        scripted = self._bi_scripted_step(item=item, initial_data=initial_data, tools=visible_tools, history=history)
        if scripted is not None:
            return scripted
        visible_tools = [t for t in visible_tools if not str(t.name).startswith("get_")]
        openai_tools = [tool_to_openai(t) for t in visible_tools]
        prompt = self._messages(item=item, initial_data=initial_data, tools=visible_tools, history=history)
        result = None
        action = None
        for attempt in range(2):
            result = self.bank.generate(
                self.bank.executor_adapter,
                prompt,
                tools=openai_tools,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            print(
                f"[vakra-agent] raw attempt={attempt + 1} "
                f"{result.text[:800].replace(chr(10), ' ')}",
                flush=True,
            )
            action = self.parse_action_text(result.text)
            explicit_final = _explicit_final_action(result.text)
            if explicit_final is not None and history:
                action = explicit_final
                break
            if str(action.name) not in RESPOND_NAMES or history or not visible_tools or attempt:
                break
            prompt = [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        "Do not produce a final answer yet. This is a local synthetic database task, "
                        "not a request for web browsing. Use the initial_data handle and output exactly "
                        "one JSON tool call for one available MCP tool. Treat initial_data as a preview only; "
                        "for user-mentioned words or site names, prefer contains/search-style filters using a string value."
                    ),
                },
            ]
        assert result is not None and action is not None
        action_args = getattr(action, "kwargs", None)
        if action_args is None:
            action_args = getattr(action, "arguments", {})
        return AgentStep(name=action.name, arguments=clean_arguments(action_args), raw=result.text)

    def _bi_scripted_step(
        self,
        *,
        item: BenchmarkItem,
        initial_data: Any,
        tools: list[Any],
        history: list[dict[str, Any]],
    ) -> Optional[AgentStep]:
        if self.capability_id != 1:
            return None
        tool_names = {str(t.name) for t in tools}
        query_tokens_all = set(_tokens(item.query))
        if history:
            last = history[-1]
            last_result = last.get("result")
            last_tool_name = str(last.get("tool_name", ""))
            handle = _result_handle(last_result)
            source_handle = handle or _latest_handle(history)
            field_names = _field_names(last_result) or _field_names(initial_data)
            name_field = next((name for name in field_names if name.lower() in {"brandname", "name"}), "")
            target_field = _target_field_for_query(item.query, field_names) or name_field
            asks_for_names = bool({"list", "names", "name", "brands", "brand"} & set(_tokens(item.query)))
            asks_how_many = "many" in query_tokens_all
            asks_review_cities = bool({"city", "cities"} & query_tokens_all) and "reviews" in query_tokens_all and "stars" in query_tokens_all
            asks_five_star_customer_names = (
                "customers" in query_tokens_all
                and bool({"name", "names"} & query_tokens_all)
                and _star_rating_from_query(item.query) is not None
                and bool({"review", "reviews"} & query_tokens_all)
            )
            asks_contact_tally = (
                "email" in query_tokens_all
                and "phone" in query_tokens_all
                and "sacramento" in query_tokens_all
                and _rating_threshold_from_query(item.query) is not None
            )
            asks_canned_before_brewery_brand = (
                "brewery" in query_tokens_all
                and "brand" in query_tokens_all
                and _mentions_can_container(item.query)
                and bool(_before_date_from_query(item.query))
            )
            asks_average_female_brand = (
                "average" in query_tokens_all
                and "female" in query_tokens_all
                and bool(_brand_id_from_query(item.query))
                and bool(_slash_date_range_from_query(item.query)[0])
            )
            print(
                f"[vakra-agent] scripted_probe history={len(history)} last={last_tool_name} "
                f"handle={handle} target_field={target_field} records={_num_records(last_result)} "
                f"getter={_getter_for_field(tool_names, target_field) if target_field else ''}",
                flush=True,
            )
            if asks_how_many:
                year = _year_from_query(item.query)
                date_field = _field_with_tokens(field_names, "purchase", "date") or _field_with_tokens(_field_names(initial_data), "purchase", "date")
                container_field = _field_with_tokens(field_names, "container", "type") or _field_with_tokens(_field_names(initial_data), "container", "type")
                brand_field = _field_named(field_names, "BrandName") or _field_named(_field_names(initial_data), "BrandName")
                count_field = _field_named(field_names, "RootBeerID") or _field_named(_field_names(initial_data), "RootBeerID")
                brand_entity = _brand_entity_from_query(item.query)
                if history and last_tool_name == "compute_data_count":
                    values = _extract_values(last_result)
                    count = _num_records(last_result)
                    answer = str(count if count is not None else (values[0] if values else last_result))
                    return AgentStep(name="final_answer", arguments={"answer": answer}, raw="scripted_bi_final_count")
                if len(history) == 1 and handle and year and date_field and "select_data_less_than" in tool_names:
                    return AgentStep(
                        name="select_data_less_than",
                        arguments={"data_label": handle, "key_name": date_field, "value": f"{year + 1}-01-01"},
                        raw=f"scripted_bi_year_upper:{date_field}:{year + 1}",
                    )
                if len(history) == 2 and handle and container_field and _mentions_can_container(item.query) and "select_data_contains" in tool_names:
                    return AgentStep(
                        name="select_data_contains",
                        arguments={"data_label": handle, "key_name": container_field, "value": "can"},
                        raw=f"scripted_bi_container_filter:{container_field}:can",
                    )
                if len(history) == 3 and handle and brand_field and brand_entity and "select_data_contains" in tool_names:
                    return AgentStep(
                        name="select_data_contains",
                        arguments={"data_label": handle, "key_name": brand_field, "value": brand_entity},
                        raw=f"scripted_bi_brand_filter:{brand_field}:{brand_entity}",
                    )
                if len(history) >= 4 and handle and count_field and "compute_data_count" in tool_names:
                    return AgentStep(
                        name="compute_data_count",
                        arguments={"data_label": handle, "key_name": count_field, "distinct": False},
                        raw=f"scripted_bi_count:{count_field}",
                    )
            if asks_review_cities:
                start_date, end_date = _month_range_from_query(item.query)
                rating = _star_rating_from_query(item.query)
                date_field = _field_with_tokens(field_names, "review", "date") or _field_with_tokens(_field_names(initial_data), "review", "date")
                rating_field = _field_with_tokens(field_names, "star", "rating") or _field_with_tokens(_field_names(initial_data), "star", "rating")
                city_field = _field_named(field_names, "customers_City") or _field_named(_field_names(initial_data), "customers_City")
                if len(history) == 1 and handle and end_date and date_field and "select_data_less_than" in tool_names:
                    return AgentStep(
                        name="select_data_less_than",
                        arguments={"data_label": handle, "key_name": date_field, "value": end_date},
                        raw=f"scripted_bi_review_month_upper:{date_field}:{end_date}",
                    )
                if len(history) == 2 and handle and rating is not None and rating_field and "select_data_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_equal_to",
                        arguments={"data_label": handle, "key_name": rating_field, "value": rating},
                        raw=f"scripted_bi_review_rating:{rating_field}:{rating}",
                    )
                if len(history) >= 3 and handle and city_field and _getter_for_field(tool_names, city_field):
                    getter_name = _getter_for_field(tool_names, city_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": handle},
                        raw=f"scripted_bi_review_city_getter:{getter_name}",
                    )
            if asks_five_star_customer_names:
                first_field = _field_named(field_names, "customers_First") or _field_named(_field_names(initial_data), "customers_First")
                last_field = _field_named(field_names, "customers_Last") or _field_named(_field_names(initial_data), "customers_Last")
                if len(history) == 1 and source_handle and first_field and _getter_for_field(tool_names, first_field):
                    getter_name = _getter_for_field(tool_names, first_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_review_first_names:{getter_name}",
                    )
                if len(history) == 2 and source_handle and last_field and _getter_for_field(tool_names, last_field):
                    getter_name = _getter_for_field(tool_names, last_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_review_last_names:{getter_name}",
                    )
                if len(history) >= 3:
                    answer = _format_pairs(
                        _values_from_tool(history, "customers_First"),
                        _values_from_tool(history, "customers_Last"),
                    )
                    if answer:
                        return AgentStep(name="final_answer", arguments={"answer": answer}, raw="scripted_bi_final_customer_names")
            if asks_contact_tally:
                year = _year_from_query(item.query)
                threshold = _rating_threshold_from_query(item.query)
                rating_field = _field_with_tokens(field_names, "star", "rating") or _field_with_tokens(_field_names(initial_data), "star", "rating")
                date_field = _field_with_tokens(field_names, "review", "date") or _field_with_tokens(_field_names(initial_data), "review", "date")
                email_field = _field_named(field_names, "customers_Email") or _field_named(_field_names(initial_data), "customers_Email")
                phone_field = _field_named(field_names, "customers_PhoneNumber") or _field_named(_field_names(initial_data), "customers_PhoneNumber")
                if len(history) == 1 and source_handle and threshold is not None and rating_field and "select_data_greater_than" in tool_names:
                    return AgentStep(
                        name="select_data_greater_than",
                        arguments={"data_label": source_handle, "key_name": rating_field, "value": threshold},
                        raw=f"scripted_bi_contact_rating:{rating_field}:{threshold}",
                    )
                if len(history) == 2 and source_handle and year and date_field and "select_data_greater_than_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_greater_than_equal_to",
                        arguments={"data_label": source_handle, "key_name": date_field, "value": f"{year}-01-01"},
                        raw=f"scripted_bi_contact_year_lower:{date_field}:{year}",
                    )
                if len(history) == 3 and source_handle and year and date_field and "select_data_less_than" in tool_names:
                    return AgentStep(
                        name="select_data_less_than",
                        arguments={"data_label": source_handle, "key_name": date_field, "value": f"{year + 1}-01-01"},
                        raw=f"scripted_bi_contact_year_upper:{date_field}:{year + 1}",
                    )
                if len(history) == 4 and source_handle and email_field and _getter_for_field(tool_names, email_field):
                    getter_name = _getter_for_field(tool_names, email_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_contact_emails:{getter_name}",
                    )
                if len(history) == 5 and source_handle and phone_field and _getter_for_field(tool_names, phone_field):
                    getter_name = _getter_for_field(tool_names, phone_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_contact_phones:{getter_name}",
                    )
                if len(history) >= 6:
                    answer = _format_pairs(
                        _values_from_tool(history, "Email"),
                        _values_from_tool(history, "Phone"),
                        sep=" / ",
                    )
                    if answer:
                        return AgentStep(name="final_answer", arguments={"answer": answer}, raw="scripted_bi_final_contacts")
            if asks_canned_before_brewery_brand:
                container_field = _field_with_tokens(field_names, "container", "type") or _field_with_tokens(_field_names(initial_data), "container", "type")
                brewery_field = _field_named(field_names, "rootbeerbrand_BreweryName") or _field_named(_field_names(initial_data), "rootbeerbrand_BreweryName")
                brand_field = _field_named(field_names, "rootbeerbrand_BrandName") or _field_named(_field_names(initial_data), "rootbeerbrand_BrandName")
                if len(history) == 1 and source_handle and container_field and "select_data_contains" in tool_names:
                    return AgentStep(
                        name="select_data_contains",
                        arguments={"data_label": source_handle, "key_name": container_field, "value": "can"},
                        raw=f"scripted_bi_canned_before_container:{container_field}",
                    )
                if len(history) == 2 and source_handle and brewery_field and _getter_for_field(tool_names, brewery_field):
                    getter_name = _getter_for_field(tool_names, brewery_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_canned_before_breweries:{getter_name}",
                    )
                if len(history) == 3 and source_handle and brand_field and _getter_for_field(tool_names, brand_field):
                    getter_name = _getter_for_field(tool_names, brand_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": source_handle},
                        raw=f"scripted_bi_canned_before_brands:{getter_name}",
                    )
                if len(history) >= 4:
                    answer = _format_pairs(
                        _values_from_tool(history, "BreweryName"),
                        _values_from_tool(history, "BrandName"),
                        sep=" - ",
                    )
                    if answer:
                        return AgentStep(name="final_answer", arguments={"answer": answer}, raw="scripted_bi_final_brewery_brands")
            if asks_average_female_brand:
                if _num_records(last_result) == 0:
                    return AgentStep(
                        name="final_answer",
                        arguments={"answer": "No matching records."},
                        raw="scripted_bi_final_avg_empty",
                    )
                start_date, end_date = _slash_date_range_from_query(item.query)
                brand_id = _brand_id_from_query(item.query)
                brand_field = _field_named(field_names, "rootbeerreview_BrandID") or _field_named(_field_names(initial_data), "rootbeerreview_BrandID")
                date_field = _field_with_tokens(field_names, "review", "date") or _field_with_tokens(_field_names(initial_data), "review", "date")
                rating_field = _field_with_tokens(field_names, "star", "rating") or _field_with_tokens(_field_names(initial_data), "star", "rating")
                if len(history) == 1 and source_handle and brand_id and brand_field and "select_data_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_equal_to",
                        arguments={"data_label": source_handle, "key_name": brand_field, "value": brand_id},
                        raw=f"scripted_bi_avg_brand:{brand_field}:{brand_id}",
                    )
                if len(history) == 2 and source_handle and start_date and date_field and "select_data_greater_than_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_greater_than_equal_to",
                        arguments={"data_label": source_handle, "key_name": date_field, "value": start_date},
                        raw=f"scripted_bi_avg_start:{date_field}:{start_date}",
                    )
                if len(history) == 3 and source_handle and end_date and date_field and "select_data_less_than_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_less_than_equal_to",
                        arguments={"data_label": source_handle, "key_name": date_field, "value": end_date},
                        raw=f"scripted_bi_avg_end:{date_field}:{end_date}",
                    )
                if len(history) == 4 and source_handle and rating_field and "compute_data_mean" in tool_names:
                    return AgentStep(
                        name="compute_data_mean",
                        arguments={"data_label": source_handle, "key_name": rating_field},
                        raw=f"scripted_bi_avg_mean:{rating_field}",
                    )
                if len(history) >= 5 and last_tool_name == "compute_data_mean":
                    answer = _extract_scalar(last_result) or str(last_result)
                    return AgentStep(name="final_answer", arguments={"answer": answer}, raw="scripted_bi_final_avg")
            person_first, person_last = _person_from_query(item.query)
            if person_first and person_last and {"consumed", "brands"} & query_tokens_all:
                first_field = _field_named(field_names, "customers_First") or _field_named(_field_names(initial_data), "customers_First")
                last_field = _field_named(field_names, "customers_Last") or _field_named(_field_names(initial_data), "customers_Last")
                consumed_brand_field = _field_named(field_names, "rootbeerbrand_BrandName") or _field_named(_field_names(initial_data), "rootbeerbrand_BrandName") or _field_named(field_names, "BrandName")
                if len(history) == 1 and handle and last_field and "select_data_equal_to" in tool_names:
                    return AgentStep(
                        name="select_data_equal_to",
                        arguments={"data_label": handle, "key_name": last_field, "value": person_last},
                        raw=f"scripted_bi_person_last:{last_field}:{person_last}",
                    )
                if len(history) >= 2 and handle and consumed_brand_field and _getter_for_field(tool_names, consumed_brand_field):
                    getter_name = _getter_for_field(tool_names, consumed_brand_field)
                    return AgentStep(
                        name=getter_name,
                        arguments={"data_label": handle},
                        raw=f"scripted_bi_consumed_brand_getter:{getter_name}",
                    )
            if last_tool_name.startswith("get_"):
                values = _extract_values(last_result)
                if values and "select_unique_values" in tool_names:
                    return AgentStep(
                        name="select_unique_values",
                        arguments={"unique_array": values},
                        raw=f"scripted_bi_unique_array:{last_tool_name}",
                    )
                if values:
                    return AgentStep(
                        name="final_answer",
                        arguments={"answer": ", ".join(values)},
                        raw=f"scripted_bi_final_from_getter:{last_tool_name}",
                    )
            if (
                (asks_for_names or bool(target_field))
                and handle
                and target_field
                and _getter_for_field(tool_names, target_field)
                and not last_tool_name.startswith("get_")
                and last_tool_name != "select_unique_values"
                and (_num_records(last_result) or 0) > 0
            ):
                getter_name = _getter_for_field(tool_names, target_field)
                return AgentStep(
                    name=getter_name,
                    arguments={"data_label": handle},
                    raw=f"scripted_bi_getter:{getter_name}:{target_field}",
                )
            if last_tool_name == "select_unique_values":
                values = _extract_values(last_result)
                if values:
                    return AgentStep(
                        name="final_answer",
                        arguments={"answer": ", ".join(values)},
                        raw="scripted_bi_final_from_unique_values",
                    )
            return None
        if "select_data_contains" not in tool_names:
            return None
        handle = _result_handle(initial_data)
        if not handle:
            return None
        query_tokens = set(_tokens(item.query))
        field_names = _field_names(initial_data)
        if (
            "customers" in query_tokens
            and bool({"name", "names"} & query_tokens)
            and _star_rating_from_query(item.query) is not None
            and bool({"review", "reviews"} & query_tokens)
        ):
            rating_field = _field_with_tokens(field_names, "star", "rating")
            rating = _star_rating_from_query(item.query)
            if rating is not None and rating_field and "select_data_equal_to" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"reviewer_rating={rating_field}:{rating}",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_equal_to",
                    arguments={"data_label": str(handle), "key_name": rating_field, "value": rating},
                    raw=f"scripted_bi_reviewer_rating:{rating_field}:{rating}",
                )
        if (
            "email" in query_tokens
            and "phone" in query_tokens
            and "sacramento" in query_tokens
            and _rating_threshold_from_query(item.query) is not None
        ):
            city_field = _field_named(field_names, "customers_City")
            if city_field and "select_data_contains" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"contact_city={city_field}:Sacramento",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_contains",
                    arguments={"data_label": str(handle), "key_name": city_field, "value": "Sacramento"},
                    raw=f"scripted_bi_contact_city:{city_field}:Sacramento",
                )
        if (
            "brewery" in query_tokens
            and "brand" in query_tokens
            and _mentions_can_container(item.query)
            and _before_date_from_query(item.query)
        ):
            date_field = _field_with_tokens(field_names, "purchase", "date")
            before_date = _before_date_from_query(item.query)
            if date_field and "select_data_less_than" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"canned_before_date={date_field}:{before_date}",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_less_than",
                    arguments={"data_label": str(handle), "key_name": date_field, "value": before_date},
                    raw=f"scripted_bi_canned_before_date:{date_field}:{before_date}",
                )
        if (
            "average" in query_tokens
            and "female" in query_tokens
            and _brand_id_from_query(item.query)
            and _slash_date_range_from_query(item.query)[0]
        ):
            gender_field = _field_named(field_names, "customers_Gender")
            if gender_field and "select_data_equal_to" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"avg_gender={gender_field}:F",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_equal_to",
                    arguments={"data_label": str(handle), "key_name": gender_field, "value": "F"},
                    raw=f"scripted_bi_avg_gender:{gender_field}:F",
                )
        if "many" in query_tokens:
            year = _year_from_query(item.query)
            date_field = _field_with_tokens(field_names, "purchase", "date")
            if year and date_field and "select_data_greater_than_equal_to" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"count_year_lower={date_field}:{year}",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_greater_than_equal_to",
                    arguments={"data_label": str(handle), "key_name": date_field, "value": f"{year}-01-01"},
                    raw=f"scripted_bi_year_lower:{date_field}:{year}",
                )
        if bool({"city", "cities"} & query_tokens) and "reviews" in query_tokens and "stars" in query_tokens:
            start_date, _ = _month_range_from_query(item.query)
            date_field = _field_with_tokens(field_names, "review", "date")
            if start_date and date_field and "select_data_greater_than_equal_to" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"review_month_lower={date_field}:{start_date}",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_greater_than_equal_to",
                    arguments={"data_label": str(handle), "key_name": date_field, "value": start_date},
                    raw=f"scripted_bi_review_month_lower:{date_field}:{start_date}",
                )
        person_first, person_last = _person_from_query(item.query)
        if person_first and person_last and {"consumed", "brands"} & query_tokens:
            first_field = _field_named(field_names, "customers_First")
            if first_field and "select_data_equal_to" in tool_names:
                print(
                    f"[vakra-agent] scripted_probe history=0 handle={handle} "
                    f"person_first={first_field}:{person_first}",
                    flush=True,
                )
                return AgentStep(
                    name="select_data_equal_to",
                    arguments={"data_label": str(handle), "key_name": first_field, "value": person_first},
                    raw=f"scripted_bi_person_first:{first_field}:{person_first}",
                )
        target_field = _target_field_for_query(item.query, field_names)
        brand_field = _field_named(field_names, "BrandName")
        brand_entity = _brand_entity_from_query(item.query)
        if target_field and brand_field and brand_entity and "select_data_contains" in tool_names:
            print(
                f"[vakra-agent] scripted_probe history=0 handle={handle} "
                f"brand_filter={brand_field}:{brand_entity} target={target_field}",
                flush=True,
            )
            return AgentStep(
                name="select_data_contains",
                arguments={"data_label": str(handle), "key_name": brand_field, "value": brand_entity},
                raw=f"scripted_bi_brand_filter:{brand_field}:{brand_entity}",
            )
        best: tuple[int, str, str] | None = None
        for field in field_names:
            field_tokens = [tok for tok in _tokens(field) if tok not in FIELD_MATCH_STOP_TOKENS]
            for token in field_tokens:
                if token in query_tokens:
                    score = 10 + len(token)
                    if best is None or score > best[0]:
                        best = (score, field, token)
        print(
            f"[vakra-agent] scripted_probe history=0 handle={handle} "
            f"fields={field_names[:8]} query_tokens={sorted(query_tokens)[:12]} best={best}",
            flush=True,
        )
        if best is None:
            return None
        _, field, token = best
        return AgentStep(
            name="select_data_contains",
            arguments={"data_label": str(handle), "key_name": field, "value": token},
            raw=f"scripted_bi_column_match:{field}:{token}",
        )

    def _messages(
        self,
        *,
        item: BenchmarkItem,
        initial_data: Any,
        tools: list[Any],
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        system = (
            "You are an agent being evaluated on VAKRA live MCP tools. "
            "The tasks are synthetic local database tasks, even when field values mention websites, social media, "
            "private records, or dashboards. Do not refuse because you cannot browse the web; use the MCP data tools. "
            "Answer the user by making one tool call at a time, using only the available tools. "
            "Tool results are handles/peeks; pass returned handle strings as data_label in later calls. "
            "Initial data and tool peeks are incomplete previews, not proof that no later rows match. "
            "When a user asks for records matching a word or site name, prefer contains/search-style tools with that "
            "query word as a lowercase string before using missing-value or not-equal filters. "
            "After a successful contains/search filter, do not repeat a missing-value or not-equal filter on that "
            "same field; answer from the filtered result preview or use a column/value extraction tool if available. "
            "Never emit NaN as a JSON value; use strings, numbers, booleans, arrays, or null only. "
            "If a tool errors, correct the next call using the error message and schema. "
            "If initial_data contains a handle and non-get_data tools are available, make at least one tool call "
            "before final_answer unless the final answer is explicitly present in initial_data. "
            "When ready to answer, output exactly one JSON object: "
            '{"name":"final_answer","arguments":{"answer":"..."}}. '
            "For a tool call, output exactly: "
            '{"name":"tool_name","arguments":{...}}. '
            "Do not use ground-truth answers or invent data not supported by tool observations."
        )
        user = {
            "query": item.query,
            "turn_id": item.turn_id,
            "conversation_context": item.context,
            "additional_instructions": item.additional_instructions,
            "initial_data": compact_data_context(initial_data),
            "available_tools": [compact_tool(t) for t in tools],
            "tool_history": history[-12:],
            "output_contract": {
                "tool_call": {"name": "one available tool", "arguments": "schema-valid object"},
                "final": {"name": "final_answer", "arguments": {"answer": "string"}},
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": short_json(user)},
        ]


def load_items(data_root: Path, capability_id: int, domain: str, max_samples: Optional[int]) -> list[BenchmarkItem]:
    input_path = data_root / CAPABILITY_DIRS[capability_id] / "input" / f"{domain}.json"
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    items: list[BenchmarkItem] = []
    for rec in raw:
        turns = (rec.get("dialogue") or {}).get("turns") or []
        if not turns:
            continue
        context = []
        for turn in turns[:-1]:
            context.append({"role": "user", "content": str(turn.get("query", ""))})
            if "answer" in turn:
                context.append({"role": "assistant", "content": str(turn["answer"])})
        last = turns[-1]
        items.append(
            BenchmarkItem(
                uuid=str(rec.get("uuid", "")),
                domain=str(rec.get("domain", domain)),
                query=str(last.get("query", "")),
                turn_id=int(last.get("turn_id", 0)),
                context=context,
                additional_instructions=str(rec.get("additional_instructions", "") or ""),
            )
        )
    return items[:max_samples] if max_samples else items


def mcp_env(args: argparse.Namespace, domain: str) -> dict[str, str]:
    exec_env = {
        "MCP_DOMAIN": domain,
        "CAPABILITY_ID": str(args.capability_id),
        "MCP_DB_ROOT": args.mcp_db_root,
    }
    if args.capability_id == 1:
        exec_env["MCP_SERVER_TYPE"] = "router"
    if args.capability_id == 4:
        exec_env["PRELOAD_COLLECTIONS"] = "false"
    return exec_env


def mcp_server_params(args: argparse.Namespace, domain: str) -> StdioServerParameters:
    exec_env = mcp_env(args, domain)
    if args.mcp_launcher == "local":
        env = os.environ.copy()
        env.update(exec_env)
        return StdioServerParameters(
            command=args.mcp_command,
            args=args.mcp_args,
            env=env,
        )
    docker_args = ["exec", "-i"]
    for key, value in exec_env.items():
        docker_args.extend(["-e", f"{key}={value}"])
    docker_args.extend([args.container_name, "python", "/app/mcp_dispatch.py"])
    return StdioServerParameters(command=args.container_runtime, args=docker_args, env=None)


async def call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    return result_to_value(await session.call_tool(name, arguments))


async def run_item(
    *,
    session: ClientSession,
    item: BenchmarkItem,
    tools: list[Any],
    agent: BaseVakraAgent,
    max_steps: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    tool_calls: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    answer = ""
    status = "success"
    error = ""

    try:
        current_tools = tools
        schema_map = {
            str(t.name): getattr(t, "inputSchema", None)
            for t in current_tools
        }
        tool_names = set(schema_map)
        if "get_data" in tool_names:
            initial_data = await call_tool(session, "get_data", {"tool_universe_id": item.uuid})
            current_tools = (await session.list_tools()).tools
            current_tools = [
                *current_tools,
                *_hidden_getter_tools(initial_data, {str(t.name) for t in current_tools}),
            ]
            schema_map = {
                str(t.name): getattr(t, "inputSchema", None)
                for t in current_tools
            }
            tool_names = set(schema_map)
            print(
                "[vakra-runner] active_tools="
                + ", ".join(str(tool.name) for tool in current_tools),
                flush=True,
            )
        else:
            initial_data = {
                "note": "This VAKRA capability did not expose get_data; use domain tools directly."
            }
        for step_idx in range(max_steps):
            step = await agent.next_step(item=item, initial_data=initial_data, tools=current_tools, history=history)
            name = str(step.name)
            arguments = coerce_args_to_schema(clean_arguments(step.arguments), schema_map.get(name))
            if name in RESPOND_NAMES:
                answer = str(
                    arguments.get("answer")
                    or arguments.get("content")
                    or arguments.get("message")
                    or step.raw
                    or ""
                )
                if "<|channel|>" in answer or "<|start|>" in answer:
                    status = "error"
                    error = "Malformed final answer contained raw Harmony trace."
                break
            if name not in tool_names:
                history.append(
                    {
                        "step": step_idx,
                        "tool_name": name,
                        "arguments": arguments,
                        "result": {
                            "error": f"Unknown tool {name!r}. Choose one of the available MCP tools."
                        },
                    }
                )
                continue
            if name == "get_data":
                history.append(
                    {
                        "step": step_idx,
                        "tool_name": name,
                        "arguments": arguments,
                        "result": {"error": "get_data is managed by the runner; use the initial handle."},
                    }
                )
                continue
            result = await call_tool(session, name, arguments)
            tool_calls.append({"name": name, "arguments": arguments})
            history.append({"step": step_idx, "tool_name": name, "arguments": arguments, "result": result})
            history[-1]["result"] = compact_observation(history[-1]["result"])
        else:
            answer = "Reached max tool steps before producing a final answer."
            status = "error"
            error = answer
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        print("[vakra-runner] exception:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        answer = answer or error

    return {
        "uuid": item.uuid,
        "domain": item.domain,
        "status": status,
        "error": error,
        "duration_s": time.perf_counter() - start,
        "output": [
            {
                "turn_id": item.turn_id,
                "query": item.query,
                "answer": answer,
                "sequence": {"tool_call": tool_calls},
            }
        ],
    }


async def run_domain(args: argparse.Namespace) -> Path:
    data_root = args.data_root or (args.vakra_root / "data" / "test")
    items = load_items(data_root, args.capability_id, args.domain, args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.domain}.json"

    if args.agent == "scripted-smoke":
        agent: BaseVakraAgent = ScriptedSmokeAgent()
    else:
        agent = GPTOSSVakraAgent(args)

    params = mcp_server_params(args, args.domain)
    results: list[dict[str, Any]] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(
                "[vakra-runner] tools="
                + ", ".join(str(tool.name) for tool in tools),
                flush=True,
            )
            for idx, item in enumerate(items, start=1):
                print(f"[vakra-runner] {args.domain} {idx}/{len(items)} {item.uuid}: {item.query}", flush=True)
                result = await run_item(
                    session=session,
                    item=item,
                    tools=tools,
                    agent=agent,
                    max_steps=args.max_steps,
                )
                print(
                    f"[vakra-runner] status={result['status']} "
                    f"tools={len(result['output'][0]['sequence']['tool_call'])} "
                    f"answer={str(result['output'][0]['answer'])[:160]}",
                    flush=True,
                )
                results.append(result)
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vakra-root", type=Path, default=Path("external/vakra"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--capability-id", type=int, choices=sorted(CAPABILITY_DIRS), required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--agent", choices=["ensemble", "scripted-smoke"], default="ensemble")
    parser.add_argument(
        "--mcp-launcher",
        choices=["docker-exec", "local"],
        default=os.environ.get("VAKRA_MCP_LAUNCHER", "docker-exec"),
        help=(
            "docker-exec talks to a running VAKRA container from the host; "
            "local spawns the MCP server in this process environment, useful inside HPC containers."
        ),
    )
    parser.add_argument("--mcp-command", default=os.environ.get("VAKRA_MCP_COMMAND", sys.executable))
    parser.add_argument(
        "--mcp-args",
        nargs="*",
        default=os.environ.get("VAKRA_MCP_ARGS", "/app/mcp_dispatch.py").split(),
    )
    parser.add_argument("--container-runtime", default=os.environ.get("VAKRA_CONTAINER_RUNTIME", "docker"))
    parser.add_argument("--container-name", default=None)
    parser.add_argument("--mcp-db-root", default="/app/db")
    parser.add_argument("--executor-adapter", default=None)
    parser.add_argument("--planner-adapter", default=None)
    parser.add_argument("--recovery-adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VAKRA_ENSEMBLE_MAX_NEW_TOKENS", "700")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("VAKRA_ENSEMBLE_TEMPERATURE", "0.0")))
    args = parser.parse_args()
    if args.container_name is None:
        args.container_name = DEFAULT_CONTAINERS[args.capability_id]
    # Make tau_ensemble_agent.py importable when this script is launched by path.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return args


def main() -> None:
    args = parse_args()
    out_path = asyncio.run(run_domain(args))
    print(json.dumps({"output_file": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
