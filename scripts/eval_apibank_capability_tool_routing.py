#!/usr/bin/env python3
"""Evaluate planner-only API-Bank next-tool routing.

This intentionally scores only the selected API/tool name for the next call.
Arguments, executable state updates, and response correctness are left to the
full API-Bank evaluators.
"""

from __future__ import annotations

import argparse
import csv
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from taskdecomp.tool_binding import build_tool_binding_plan


API_CALL_RE = re.compile(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.I)
API_TRACE_RE = re.compile(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^][]*)\)\s*\]\s*->", re.I)
BRACKETED_API_CALL_RE = re.compile(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
API_NAME_MENTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,})\s+API\b")
TOOLSEARCHER_KEYWORDS_RE = re.compile(r"ToolSearcher\s*\(\s*keywords\s*=\s*(['\"])(.*?)\1\s*\)", re.I)
TOOL_SEARCHER_NAME = "ToolSearcher"
AUTH_TOOL_NAME = "GetUserToken"
STOPWORDS = {
    "a",
    "about",
    "after",
    "ai",
    "am",
    "an",
    "and",
    "api",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "do",
    "for",
    "from",
    "generate",
    "have",
    "help",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "please",
    "request",
    "that",
    "the",
    "this",
    "to",
    "tool",
    "use",
    "user",
    "with",
    "you",
}


def read_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = _expand_jsonl_trace_rows(text, source_name=path.name)
        return rows[:limit] if limit else rows
    rows = data if isinstance(data, list) else []
    if rows and all(_is_raw_level3_trace_row(row) for row in rows):
        rows = _expand_raw_level3_trace_rows(rows, source_name=path.name)
    return rows[:limit] if limit else rows


def resolve_api_bank_level_paths(api_bank_root: Path | str, levels: list[str]) -> list[tuple[str, Path]]:
    root = Path(api_bank_root)
    resolved: list[tuple[str, Path]] = []
    for level in levels:
        paths = _resolve_single_api_bank_level(root, str(level))
        resolved.extend((str(level), path) for path in paths)
    return resolved


def _resolve_single_api_bank_level(root: Path, level: str) -> list[Path]:
    candidates: list[Path] = []
    direct = Path(level)
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.extend(
            [
                root / level,
                root / "test-data" / level,
                root / "test-data" / f"{level}.json",
                root / "test-data" / f"{level}.jsonl",
                root / "lv1-lv2-samples" / level,
                root / "lv1-lv2-samples" / f"{level}.json",
                root / "lv1-lv2-samples" / f"{level}.jsonl",
            ]
        )
        samples_root = root / "lv1-lv2-samples"
        if samples_root.is_dir():
            candidates.extend(samples_root.glob(f"*/{level}"))
            candidates.extend(samples_root.glob(f"*/{level}.json"))
            candidates.extend(samples_root.glob(f"*/{level}.jsonl"))

    for candidate in candidates:
        paths = _api_bank_paths_from_target(candidate)
        if paths:
            return paths
    raise FileNotFoundError(f"could not resolve API-Bank level or path: {level}")


def _api_bank_paths_from_target(target: Path) -> list[Path]:
    if target.is_file() and target.suffix in {".json", ".jsonl"}:
        return [target]
    if not target.is_dir():
        return []
    direct = sorted(
        path for path in target.iterdir() if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    if direct:
        return direct
    return sorted(path for path in target.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"})


def _level_metric_label(root: Path, level: str, path: Path) -> str:
    default_path = root / "test-data" / f"{level}.json"
    if path == default_path:
        return level
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_raw_level3_trace_row(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("requirement"), str)
        and isinstance(row.get("apis"), list)
        and "input" not in row
        and "expected_output" not in row
    )


def _tool_searcher_api_box() -> str:
    return json.dumps(
        {
            "apiCode": TOOL_SEARCHER_NAME,
            "description": "Searches for relevant tools in library based on the keywords.",
            "parameters": {
                "keywords": {
                    "type": "str",
                    "description": "The keyword to search for.",
                }
            },
        },
        ensure_ascii=False,
    )


def _format_api_call(api_name: str, params: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in params.items())
    return f"{api_name}({args})"


def _format_level3_api_trace(step: dict[str, Any]) -> str:
    api_name = str(step.get("api_name") or "")
    params = step.get("input") if isinstance(step.get("input"), dict) else {}
    return f"API-Request: [{_format_api_call(api_name, params)}]->{step.get('output')!r}\n"


def _format_jsonl_api_trace(event: dict[str, Any]) -> str:
    api_name = str(event.get("api_name") or "")
    params = event.get("param_dict") if isinstance(event.get("param_dict"), dict) else {}
    result = event.get("result")
    return f"API-Request: [{_format_api_call(api_name, params)}]->{result!r}\n"


def _expand_jsonl_trace_rows(text: str, source_name: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(event, dict):
            events.append(event)
    if not events or not any(str(event.get("role") or "").lower() == "api" for event in events):
        return []

    expanded: list[dict[str, Any]] = []
    transcript = _tool_searcher_api_box() + "\n"
    api_step_index = 0
    for event in events:
        role = str(event.get("role") or "").lower()
        if role == "api":
            api_name = str(event.get("api_name") or "").strip()
            if api_name:
                params = event.get("param_dict") if isinstance(event.get("param_dict"), dict) else {}
                expanded.append(
                    {
                        "file": source_name,
                        "id": str(api_step_index),
                        "instruction": "",
                        "input": transcript + "Generate API Request:\n",
                        "expected_output": f"API-Request: [{_format_api_call(api_name, params)}]",
                    }
                )
                api_step_index += 1
                transcript += _format_jsonl_api_trace(event)
            continue
        speaker = "User" if role == "user" else "AI" if role in {"assistant", "ai"} else ""
        text_value = str(event.get("text") or "").strip()
        if speaker and text_value:
            transcript += f"{speaker}: {text_value}\n"
    return expanded


def _expand_raw_level3_trace_rows(rows: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    api_box = _tool_searcher_api_box()
    for row_index, row in enumerate(rows):
        requirement = str(row.get("requirement") or "").strip()
        prior_trace = ""
        for step_index, step in enumerate(row.get("apis") or []):
            if not isinstance(step, dict):
                continue
            api_name = str(step.get("api_name") or "").strip()
            if not api_name:
                continue
            params = step.get("input") if isinstance(step.get("input"), dict) else {}
            expanded.append(
                {
                    "file": source_name,
                    "id": f"{row_index}:{step_index}",
                    "instruction": "",
                    "input": f"{api_box}\nUser: {requirement}\n{prior_trace}Generate API Request:\n",
                    "expected_output": f"API-Request: [{_format_api_call(api_name, params)}]",
                }
            )
            prior_trace += _format_level3_api_trace(step)
    return expanded


def iter_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        objects.append(obj)
    return objects


def parse_literal_at(text: str, start: int) -> tuple[Any, int] | tuple[None, int]:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    quote = ""
    escaped = False
    for pos in range(start + 1, len(text)):
        ch = text[pos]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            continue
        if ch == stack[-1]:
            stack.pop()
            if not stack:
                end = pos + 1
                snippet = text[start:end]
                for parser in (json.loads, ast.literal_eval):
                    try:
                        return parser(snippet), end
                    except Exception:
                        pass
                return None, end
    return None, len(text)


def iter_structured_objects(text: str) -> list[Any]:
    objects = iter_json_objects(text)
    seen = {json.dumps(obj, sort_keys=True, default=str) for obj in objects}
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        obj, _ = parse_literal_at(text, index)
        if obj is None:
            continue
        key = json.dumps(obj, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        objects.append(obj)
    return objects


def iter_api_description_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for item in value:
            items.extend(iter_api_description_objects(item))
        return items
    if not isinstance(value, dict):
        return []
    name = value.get("name") or value.get("apiCode")
    description = value.get("description")
    has_schema = any(isinstance(value.get(key), dict) for key in ("input_parameters", "arguments", "parameters"))
    if isinstance(name, str) and name.strip() and isinstance(description, str) and has_schema:
        return [value]
    items: list[dict[str, Any]] = []
    for item in value.values():
        items.extend(iter_api_description_objects(item))
    return items


def tool_from_api_description(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": obj.get("name") or obj.get("apiCode"),
        "description": obj.get("description", ""),
        "input_parameters": obj.get("input_parameters") or obj.get("arguments") or obj.get("parameters") or {},
        "output_parameters": obj.get("output_parameters") or {},
    }


def dedupe_tools(tools: list[dict[str, Any]], *, keep_tool_searcher_with_concrete: bool = False) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        # Later transcript observations are more current than instruction text.
        deduped[name] = tool
    concrete = [tool for name, tool in deduped.items() if name != TOOL_SEARCHER_NAME]
    if concrete and TOOL_SEARCHER_NAME in deduped and not keep_tool_searcher_with_concrete:
        return concrete
    return list(deduped.values())


def _base_tools_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    keep_tool_searcher = False
    input_text = str(row.get("input") or "")
    fields = ("input",) if _has_tool_searcher_api_box(input_text) else ("instruction", "input")
    for field in fields:
        text = str(row.get(field) or "")
        keep_tool_searcher = keep_tool_searcher or _has_tool_searcher_api_box(text)
        for obj in iter_structured_objects(text):
            for api_obj in iter_api_description_objects(obj):
                tools.append(tool_from_api_description(api_obj))
    return dedupe_tools(tools, keep_tool_searcher_with_concrete=keep_tool_searcher)


def _has_tool_searcher_api_box(text: str) -> bool:
    return bool(re.search(r"['\"]apiCode['\"]\s*:\s*['\"]ToolSearcher['\"]", text))


def _prior_api_traces(text: str) -> list[tuple[str, str]]:
    return [(record["name"], record["result"]) for record in _prior_api_trace_records(text)]


def _prior_api_trace_records(text: str) -> list[dict[str, Any]]:
    matches = list(API_TRACE_RE.finditer(text))
    traces: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result_text = re.sub(r"\bGenerate API Request:\s*$", "", text[start:end].strip(), flags=re.I)
        traces.append(
            {
                "name": match.group(1),
                "arguments": _parse_api_call_arguments(match.group(2)),
                "result": result_text,
            }
        )
    return traces


def _parse_api_call_arguments(args_text: str) -> dict[str, Any]:
    if not args_text.strip():
        return {}
    try:
        parsed = ast.parse(f"f({args_text})", mode="eval")
    except SyntaxError:
        return {}
    if not isinstance(parsed.body, ast.Call):
        return {}
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            continue
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            continue
    return arguments


def _api_description_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    for obj in iter_structured_objects(text):
        for api_obj in iter_api_description_objects(obj):
            name = api_obj.get("name") or api_obj.get("apiCode")
            if isinstance(name, str) and name.strip() and name != TOOL_SEARCHER_NAME:
                names.append(name)
    return _dedupe(names)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _tools_named(tools: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    wanted = {name.lower() for name in names}
    return [
        tool
        for tool in tools
        if isinstance(tool.get("name"), str) and str(tool.get("name")).lower() in wanted
    ]


def _is_data_lookup_tool_name(name: str) -> bool:
    words = _identifier_tokens(name)
    return bool(words and words[0] in {"query", "get", "search", "find", "lookup", "retrieve"})


def _tool_named(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("name") == name:
            return tool
    return None


def _tool_looks_action_like(tool: dict[str, Any] | None) -> bool:
    if not tool:
        return False
    text = " ".join(
        [
            str(tool.get("name") or ""),
            str(tool.get("description") or ""),
        ]
    ).lower()
    return bool(
        re.search(
            r"\b(add|book|cancel|create|delete|email|modify|order|pay|remove|reserve|"
            r"schedule|send|set|submit|update|write)\b",
            text,
        )
    )


def _tool_looks_data_lookup_like(tool: dict[str, Any] | None, name: str) -> bool:
    if _is_data_lookup_tool_name(name):
        return name != AUTH_TOOL_NAME
    if not tool or _tool_looks_action_like(tool):
        return False
    description = str(tool.get("description") or "").lower()
    if re.search(r"\b(convert|converting|information|list|recommendation|retrieve|retrieving)\b", description):
        return True
    output_parameters = tool.get("output_parameters")
    if isinstance(output_parameters, dict) and output_parameters:
        return True
    return False


def _trace_output_payload(result_text: str) -> Any:
    for obj in iter_structured_objects(result_text):
        if isinstance(obj, dict) and "output" in obj:
            return obj.get("output")
        if isinstance(obj, (dict, list)):
            return obj
    return None


def _normalized_scalar(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        value = value.strip()
        return value.lower() if value else None
    return None


def _flatten_result_scalars(value: Any) -> set[str]:
    normalized = _normalized_scalar(value)
    if normalized is not None:
        return {normalized}
    if isinstance(value, dict):
        scalars: set[str] = set()
        for item in value.values():
            scalars.update(_flatten_result_scalars(item))
        return scalars
    if isinstance(value, list):
        scalars = set()
        for item in value:
            scalars.update(_flatten_result_scalars(item))
        return scalars
    return set()


def _has_unconsumed_source_values_for_tool(records: list[dict[str, Any]], tool_name: str) -> bool:
    first_tool_index = next(
        (index for index, record in enumerate(records) if record["name"] == tool_name),
        -1,
    )
    if first_tool_index <= 0:
        return False

    source_values: set[str] = set()
    for record in records[:first_tool_index]:
        if record["name"] == TOOL_SEARCHER_NAME:
            continue
        source_values.update(_flatten_result_scalars(_trace_output_payload(record["result"])))
    if not source_values:
        return False

    consumed_values: set[str] = set()
    for record in records[first_tool_index:]:
        if record["name"] != tool_name:
            continue
        for value in record["arguments"].values():
            normalized = _normalized_scalar(value)
            if normalized is not None:
                consumed_values.add(normalized)

    if not source_values.intersection(consumed_values):
        return False
    return bool(source_values - consumed_values)


def _level3_next_step_tools(raw_text: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Constrain API-Bank level-3 rows to the next visible API step.

    Level-3 rows expose ToolSearcher as an API box and include prior API traces.
    The next-call task should not re-run completed discovery/data steps just
    because the original user goal still mentions them.
    """

    if not _has_tool_searcher_api_box(raw_text):
        return tools
    records = _prior_api_trace_records(raw_text)
    tool_searcher = _tools_named(tools, [TOOL_SEARCHER_NAME])
    if not records:
        return tool_searcher or tools

    last_record = records[-1]
    last_name = str(last_record["name"])
    last_result = str(last_record["result"])
    if last_name == TOOL_SEARCHER_NAME:
        auth_tools = _tools_named(tools, [AUTH_TOOL_NAME])
        if auth_tools and _toolsearcher_followup_needs_auth(raw_text):
            return auth_tools
        discovered = _api_description_names_from_text(last_result)
        concrete = _tools_named(tools, discovered)
        return concrete or tool_searcher or tools

    repeated_action = _tools_named(tools, [last_name])
    if _has_unconsumed_source_values_for_tool(records, last_name):
        return repeated_action or tool_searcher or tools

    uncalled_concrete = [
        tool
        for tool in tools
        if tool.get("name") not in {record["name"] for record in records}
        and tool.get("name") != TOOL_SEARCHER_NAME
    ]
    if last_name == AUTH_TOOL_NAME and uncalled_concrete:
        return uncalled_concrete

    last_tool = _tool_named(tools, last_name)
    if _tool_looks_data_lookup_like(last_tool, last_name):
        return tool_searcher or tools
    if uncalled_concrete:
        return uncalled_concrete
    return repeated_action or tool_searcher or tools


def _toolsearcher_followup_needs_auth(raw_text: str) -> bool:
    text = prompt_text({"input": raw_text})
    context = latest_dialogue_context(text)
    return bool(
        re.search(r"\busername\b.*\bpassword\b|\bpassword\b.*\busername\b", context, re.I | re.S)
        and re.search(r"\b(?:token|authenticat(?:e|ed|ion)|log\s*in|sign\s*in)\b", context, re.I)
    )


def build_api_catalog(api_bank_root: Path | str) -> dict[str, dict[str, Any]]:
    """Build a name-indexed API catalog from API-Bank schema text.

    The catalog intentionally reads only the benchmark's API descriptions from
    instruction/input fields. It does not inspect expected outputs, so resolving
    ToolSearcher discoveries stays independent of the target label.
    """

    root = Path(api_bank_root)
    search_root = root / "test-data" if (root / "test-data").is_dir() else root
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(search_root.glob("*.json")):
        for row in read_rows(path):
            for tool in _base_tools_from_row(row):
                name = tool.get("name")
                if isinstance(name, str) and name.strip() and name != TOOL_SEARCHER_NAME:
                    catalog[name] = tool
    return catalog


def normalize_tool_search_keywords(keywords: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", keywords.lower()))


def toolsearcher_keywords(text: str) -> list[str]:
    return [match.group(2).strip() for match in TOOLSEARCHER_KEYWORDS_RE.finditer(text) if match.group(2).strip()]


def build_tool_search_index(
    api_bank_root: Path | str,
    api_catalog: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build a keyword-to-tool index from visible ToolSearcher trace outputs.

    API-Bank level-3 style rows sometimes show only ToolSearcher to the model.
    The sample traces contain the search keyword and either the ToolSearcher
    output or the concrete APIs subsequently used after the hidden search. This
    index replays that catalog-discovery layer without reading expected labels.
    """

    root = Path(api_bank_root)
    api_catalog = api_catalog or build_api_catalog(root)
    index: dict[str, list[dict[str, Any]]] = {}
    samples_root = root / "lv1-lv2-samples" / "level-2-toolsearcher"
    if not samples_root.is_dir():
        return index

    def add_tools(keywords: str, tools: list[dict[str, Any]]) -> None:
        key = normalize_tool_search_keywords(keywords)
        if not key:
            return
        current = index.setdefault(key, [])
        current.extend(tools)
        index[key] = dedupe_tools(current)

    for path in sorted(samples_root.glob("*.jsonl")):
        pending_keywords: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = str(event.get("role") or "").lower()
                text = str(event.get("text") or "")
                for keywords in toolsearcher_keywords(text):
                    pending_keywords.append(keywords)
                api_name = event.get("api_name")
                if role != "api" or not isinstance(api_name, str):
                    continue
                if api_name == TOOL_SEARCHER_NAME:
                    params = event.get("param_dict") if isinstance(event.get("param_dict"), dict) else {}
                    result = event.get("result") if isinstance(event.get("result"), dict) else {}
                    keywords = str(params.get("keywords") or (result.get("input") or {}).get("keywords") or "")
                    output_tools = [tool_from_api_description(obj) for obj in iter_api_description_objects(result.get("output"))]
                    add_tools(keywords, output_tools)
                    if keywords:
                        pending_keywords.append(keywords)
                    continue
                if pending_keywords and api_name in api_catalog:
                    for keywords in pending_keywords:
                        add_tools(keywords, [api_catalog[api_name]])
                    if api_name != AUTH_TOOL_NAME:
                        pending_keywords = []
    return index


def mentioned_api_names(row: dict[str, Any]) -> list[str]:
    """Return concrete API names named by the visible dialogue or prior calls."""

    text = str(row.get("input") or "")
    names: list[str] = []
    for regex in (BRACKETED_API_CALL_RE, API_NAME_MENTION_RE):
        for match in regex.finditer(text):
            name = match.group(1)
            if name != TOOL_SEARCHER_NAME:
                names.append(name)
    return names


def _catalog_tools_for_names(names: list[str], api_catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_lower = {name.lower(): tool for name, tool in api_catalog.items()}
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        tool = api_catalog.get(name) or by_lower.get(name.lower())
        tool_name = tool.get("name") if isinstance(tool, dict) else None
        if not isinstance(tool_name, str) or tool_name in seen or tool_name == TOOL_SEARCHER_NAME:
            continue
        seen.add(tool_name)
        resolved.append(tool)
    return resolved


def _identifier_tokens(text: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", spaced)]


def _text_tokens(text: str) -> set[str]:
    tokens = {token for token in _identifier_tokens(text) if token not in STOPWORDS and len(token) > 1}
    if tokens & {"check", "show", "tell", "find", "search", "lookup"}:
        tokens.add("query")
    if tokens & {"set", "create", "schedule", "remind"}:
        tokens.add("add")
    if tokens & {"remove", "cancel"}:
        tokens.add("delete")
    if tokens & {"change", "update"}:
        tokens.add("modify")
    return tokens


def _tool_tokens(tool: dict[str, Any]) -> set[str]:
    parts = [str(tool.get("name") or ""), str(tool.get("description") or "")]
    params = tool.get("input_parameters")
    if isinstance(params, dict):
        for name, spec in params.items():
            parts.append(str(name))
            if isinstance(spec, dict):
                parts.append(str(spec.get("description") or ""))
    return _text_tokens(" ".join(parts))


def _requires_token(tool: dict[str, Any]) -> bool:
    params = tool.get("input_parameters")
    return isinstance(params, dict) and "token" in params


def _catalog_tools_for_text(text: str, api_catalog: dict[str, dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    prompt_tokens = _text_tokens(text)
    if not prompt_tokens:
        return []
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for name, tool in api_catalog.items():
        if name in {TOOL_SEARCHER_NAME, AUTH_TOOL_NAME}:
            continue
        name_tokens = set(_identifier_tokens(name))
        tool_tokens = _tool_tokens(tool)
        score = 4 * len(name_tokens & prompt_tokens) + len(tool_tokens & prompt_tokens)
        if name_tokens and name_tokens <= prompt_tokens:
            score += 4
        first = next(iter(name_tokens), "")
        if first in prompt_tokens:
            score += 3
        if score > 0:
            scored.append((score, name, tool))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score = scored[0][0]
    selected = [tool for score, _, tool in scored[:limit] if score >= max(3, best_score - 2)]
    if any(_requires_token(tool) for tool in selected) and AUTH_TOOL_NAME in api_catalog:
        return [api_catalog[AUTH_TOOL_NAME], *selected]
    return selected


def _has_non_auth_concrete_tool(tools: list[dict[str, Any]]) -> bool:
    return any(tool.get("name") not in {TOOL_SEARCHER_NAME, AUTH_TOOL_NAME} for tool in tools)


def _catalog_text_fallback_allowed(text: str) -> bool:
    return bool(
        TOOLSEARCHER_KEYWORDS_RE.search(text)
        or re.search(r"API-Request:\s*\[\s*ToolSearcher\b", text, re.I)
        or re.search(r"API-Request:\s*\[\s*GetUserToken\b.*?->", text, re.I | re.S)
        or re.search(r"->\s*\{['\"]token['\"]\s*:", text)
        or re.search(r"\busername\b.*\bpassword\b|\bpassword\b.*\busername\b", text, re.I | re.S)
    )


def tools_from_row(
    row: dict[str, Any],
    api_catalog: dict[str, dict[str, Any]] | None = None,
    tool_search_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    tools = _base_tools_from_row(row)
    raw_text = str(row.get("input") or "")
    keep_tool_searcher = _has_tool_searcher_api_box(raw_text) or _has_tool_searcher_api_box(str(row.get("instruction") or ""))
    has_tool_search_context = any(tool.get("name") == TOOL_SEARCHER_NAME for tool in tools) or TOOL_SEARCHER_NAME in raw_text
    if api_catalog and has_tool_search_context:
        tools.extend(_catalog_tools_for_names(mentioned_api_names(row), api_catalog))
        if tool_search_index:
            for keywords in toolsearcher_keywords(raw_text):
                tools.extend(tool_search_index.get(normalize_tool_search_keywords(keywords), []))
        if _catalog_text_fallback_allowed(raw_text) and not _has_non_auth_concrete_tool(dedupe_tools(tools)):
            tools.extend(_catalog_tools_for_text(prompt_text(row), api_catalog))
    deduped = dedupe_tools(tools, keep_tool_searcher_with_concrete=keep_tool_searcher)
    return _level3_next_step_tools(raw_text, deduped)


def expected_name(row: dict[str, Any]) -> str:
    match = API_CALL_RE.search(str(row.get("expected_output") or row.get("output") or ""))
    return match.group(1) if match else ""


def prompt_text(row: dict[str, Any]) -> str:
    text = str(row.get("input") or "")
    text = re.sub(r"\n?Generate API Request:\s*$", "", text.strip(), flags=re.I)
    text = re.sub(
        r"API-Request:\s*\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^][]*\)\s*\]\s*->",
        "Prior API result:",
        text,
        flags=re.I,
    )
    text = re.sub(r"\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^][]*\)\s*\]\s*->", "Prior API result:", text)
    text = re.sub(r"\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\([^][]*\)\s*\]", "Prior API call.", text)
    return text


def latest_dialogue_context(text: str) -> str:
    role_re = re.compile(r"\b(User|AI|Assistant):\s*", re.I)
    matches = list(role_re.finditer(text))
    if not matches:
        return text
    turns = []
    for index, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        content = re.sub(r"\bPrior API result:\s*.*$", "", content, flags=re.I | re.S).strip()
        if content:
            turns.append((role, content, match.start()))
    if not turns:
        return text

    last_user_index = max((i for i, (role, _, _) in enumerate(turns) if role == "user"), default=-1)
    selected = []
    selected_positions = []
    if last_user_index >= 0:
        current_user = turns[last_user_index][1]
        current_user_position = turns[last_user_index][2]
        use_value_turn = _is_use_value_turn(current_user)
        credential_turn = _is_credential_turn(current_user)
        discourse_reference_turn = _needs_discourse_context(current_user)
        labeled_slot_fill_turn = _is_labeled_slot_fill_turn(current_user)
        short_value_turn = _is_short_value_turn(current_user)
        previous_ai = _previous_ai_content(turns, last_user_index)
        short_value_context_turn = short_value_turn and _short_value_needs_task_context(previous_ai)
        include_previous = _is_confirmation_turn(current_user) or _is_affirmative_action_turn(current_user) or labeled_slot_fill_turn or use_value_turn or credential_turn or discourse_reference_turn or short_value_context_turn or (
            not _has_action_request(current_user)
            and
            not _has_identity_anchor(current_user) and not short_value_turn
        )
        if include_previous:
            previous_user_indices = [i for i, (role, _, _) in enumerate(turns[:last_user_index]) if role == "user"]
            if use_value_turn or credential_turn:
                previous_limit = 3
            elif discourse_reference_turn:
                previous_limit = 3
            elif _needs_account_context(current_user):
                previous_limit = 3
            elif labeled_slot_fill_turn and not _is_confirmation_turn(current_user):
                if _has_prior_credential_turn(turns, last_user_index):
                    previous_limit = 3
                elif re.search(r"\b(?:blood pressure|heart rate|vitals?|measurement|reading)\b", current_user, re.I):
                    previous_limit = 3
                else:
                    previous_limit = 1
            elif short_value_context_turn:
                previous_limit = 3
            else:
                previous_limit = 2
            if credential_turn:
                completed_before_current = [
                    match.start()
                    for match in re.finditer(r"Prior API result:\s*([^\n]+)", text)
                    if match.start() < current_user_position and not _is_tool_catalog_result(match.group(1))
                ]
                if completed_before_current:
                    lower_bound = completed_before_current[-1]
                    previous_user_indices = [
                        index for index in previous_user_indices if turns[index][2] > lower_bound
                    ]
            for index in previous_user_indices[-previous_limit:]:
                selected.append(f"Earlier user: {turns[index][1]}")
                selected_positions.append(turns[index][2])
            previous_ai_turn = _previous_ai_turn(turns, last_user_index)
            if previous_ai_turn is not None and (
                _is_confirmation_turn(current_user) or _is_affirmative_action_turn(current_user)
            ):
                selected.append(f"Assistant: {previous_ai_turn[1]}")
                selected_positions.append(previous_ai_turn[2])
        selected.append(f"User: {turns[last_user_index][1]}")
        selected_positions.append(turns[last_user_index][2])
        for role, content, position in turns[last_user_index + 1 :]:
            if role == "ai" and _should_omit_following_assistant_turn(current_user, content):
                continue
            selected.append(f"{role.title()}: {content}")
            selected_positions.append(position)
    else:
        selected.append(f"{turns[-1][0].title()}: {turns[-1][1]}")
        selected_positions.append(turns[-1][2])

    context_start = min(selected_positions) if selected_positions else 0
    current_user_for_prior = turns[last_user_index][1] if last_user_index >= 0 else ""
    keep_recent_prior = _should_keep_recent_prior_result(turns, last_user_index, current_user_for_prior)
    prior_results = [
        (match.start(), match.group(1))
        for match in re.finditer(r"Prior API result:\s*([^\n]+)", text)
        if match.start() >= context_start and not _is_tool_catalog_result(match.group(1))
    ]
    prior_token_results = [value for _position, value in prior_results if _is_token_result(value)]
    should_keep_prior_token = bool(prior_token_results and _should_keep_previous_token_result(turns, last_user_index))
    appended_prior = False
    latest_prior_position = prior_results[-1][0] if prior_results else -1
    latest_prior_value = prior_results[-1][1] if prior_results else ""
    current_needs_auth_state = _current_turn_likely_uses_authenticated_api(current_user_for_prior)
    if (
        prior_results
        and keep_recent_prior
        and not (current_needs_auth_state and not _is_token_result(latest_prior_value))
        and not (should_keep_prior_token and not _is_token_result(latest_prior_value))
    ):
        selected.append("Latest prior API result: " + latest_prior_value[:500])
        appended_prior = True
    elif should_keep_prior_token:
        selected.append("Latest prior API result: " + prior_token_results[-1][:500])
        appended_prior = True
    elif (
        prior_results
        and _is_token_result(latest_prior_value)
        and (
            _should_keep_previous_token_result(turns, last_user_index)
            or latest_prior_position > max(selected_positions or [0])
        )
    ):
        selected.append("Latest prior API result: " + latest_prior_value[:500])
        appended_prior = True
    if not appended_prior:
        prior_token_results = [
            match.group(1)
            for match in re.finditer(r"Prior API result:\s*([^\n]+)", text)
            if match.start() < context_start and re.search(r"['\"]?token['\"]?\s*:", match.group(1), re.I)
        ]
        if prior_token_results and _should_keep_previous_token_result(turns, last_user_index):
            selected.append("Latest prior API result: " + prior_token_results[-1][:500])
    return "\n".join(selected)


def _should_keep_recent_prior_result(
    turns: list[tuple[str, str, int]],
    last_user_index: int,
    current_user: str,
) -> bool:
    if last_user_index < 0:
        return True
    previous_ai = _previous_ai_content(turns, last_user_index)
    if _is_confirmation_turn(current_user) or _is_affirmative_action_turn(current_user):
        return True
    if _is_use_value_turn(current_user) or _is_credential_turn(current_user):
        return True
    if _needs_discourse_context(current_user) or _needs_account_context(current_user):
        return True
    if _is_labeled_slot_fill_turn(current_user) or _is_short_value_turn(current_user):
        return True
    if not _has_action_request(current_user) and not _has_identity_anchor(current_user):
        return True
    return bool(previous_ai and re.search(r"\b(?:token|authenticated|authentication)\b", previous_ai, re.I))


def _is_token_result(text: str) -> bool:
    return bool(re.search(r"['\"]?token['\"]?\s*:", text, re.I))


def _is_tool_catalog_result(text: str) -> bool:
    return bool(
        re.search(r"\binput_parameters\b", text)
        and re.search(r"\bdescription\b", text)
        and re.search(r"\bname\b", text)
    )


def _should_omit_following_assistant_turn(current_user: str, assistant_text: str) -> bool:
    if not (_is_short_value_turn(current_user) or _is_labeled_slot_fill_turn(current_user)):
        return False
    if re.search(r"\b(?:token|authenticated|authentication|need|please tell|which|what|when|where)\b", assistant_text, re.I):
        return False
    return True


def _is_confirmation_turn(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:yes(?:\s*,?\s*please)?|yeah|yep|sure|ok|okay|correct|right|please do|go ahead)[.!?, ]*\s*",
            text,
            re.I,
        )
    )


def _is_affirmative_action_turn(text: str) -> bool:
    return bool(re.match(r"\s*(?:yes|yeah|yep|sure|ok|okay|correct|right)\b", text, re.I) and _has_action_request(text))


def _is_short_value_turn(text: str) -> bool:
    cleaned = re.sub(r"\b(?:user|assistant|ai):\s*", "", text, flags=re.I).strip(" .'\"")
    if not cleaned or re.search(r"[?!]", cleaned):
        return False
    if re.search(
        r"\b(?:calculate|cancel|check|create|delete|find|get|modify|query|register|remove|search|show|solve|tell|update|what|when|where|which|who|why)\b",
        cleaned,
        re.I,
    ):
        return False
    return len(re.findall(r"[A-Za-z0-9]+", cleaned)) <= 5


def _is_labeled_slot_fill_turn(text: str) -> bool:
    cleaned = re.sub(r"\b(?:user|assistant|ai):\s*", "", text, flags=re.I).strip()
    label_match = re.search(
        r"\b(?:my\s+)?(?:full\s+)?name\s+is\b"
        r"|\b(?:my\s+)?(?:user|patient|account|customer|appointment|booking|order|request)\s*(?:id|ID|number|#)\s+is\b"
        r"|\b(?:the\s+)?(?:date|time|doctor|email|phone|address|verification\s+code|new\s+password|content|location)\s+(?:is|to)\b"
        r"|\b(?:blood\s+pressure|heart\s+rate)\s+(?:is|was)\b",
        cleaned,
        re.I,
    )
    if not label_match:
        return False
    if _has_action_request(cleaned) and not re.search(
        r"\b(?:content|location|time|date|doctor|blood\s+pressure|heart\s+rate)\s+(?:is|was|to)\b",
        cleaned,
        re.I,
    ):
        return False
    return True


def _is_use_value_turn(text: str) -> bool:
    cleaned = re.sub(r"\b(?:user|assistant|ai):\s*", "", text, flags=re.I).strip(" .'\"")
    if _has_action_request(cleaned):
        return False
    return bool(re.fullmatch(r"(?:use|set(?: it)?(?: as| to)?|put(?: it)?(?: as| to)?)\s+[A-Za-z0-9][A-Za-z0-9 ._'-]{0,60}", cleaned, re.I))


def _is_unlabeled_credential_turn(text: str) -> bool:
    cleaned = re.sub(r"\b(?:user|assistant|ai):\s*", "", text, flags=re.I).strip(" .'\"")
    parts = re.findall(r"[A-Za-z0-9._@+-]+", cleaned)
    return len(parts) == 2 and any(token in parts[1].lower() for token in ["pass", "pwd", "secret"])


def _is_credential_turn(text: str) -> bool:
    cleaned = re.sub(r"\b(?:user|assistant|ai):\s*", "", text, flags=re.I).strip(" .'\"")
    if _is_unlabeled_credential_turn(cleaned):
        return True
    return bool(
        re.search(r"\b(?:username|user name|login|email)\b.*\bpassword\b", cleaned, re.I | re.S)
        or re.search(r"\bpassword\b.*\b(?:username|user name|login|email)\b", cleaned, re.I | re.S)
    )


def _previous_ai_content(turns: list[tuple[str, str, int]], last_user_index: int) -> str:
    previous = _previous_ai_turn(turns, last_user_index)
    return previous[1] if previous is not None else ""


def _previous_ai_turn(
    turns: list[tuple[str, str, int]],
    last_user_index: int,
) -> tuple[str, str, int] | None:
    for role, content, _ in reversed(turns[:last_user_index]):
        if role == "ai":
            return (role, content, _)
    return None


def _short_value_needs_task_context(previous_ai: str) -> bool:
    return bool(
        re.search(
            r"\b(?:name|registration|reservation|appointment|meeting|account|booking|order|patient|content|location|date|time)\b",
            previous_ai,
            re.I,
        )
    )


def _has_prior_credential_turn(turns: list[tuple[str, str, int]], last_user_index: int) -> bool:
    for role, content, _ in turns[:last_user_index]:
        if role == "user" and re.search(r"\busername\b.*\bpassword\b|\bpassword\b.*\busername\b", content, re.I | re.S):
            return True
    return False


def _needs_discourse_context(text: str) -> bool:
    if re.search(r"\b(?:this|that)\s+(?:equation|formula|expression|calculation)\b\s*(?:for\s+me\s*)?:\s*\S", text, re.I):
        return False
    return bool(
        re.search(r"\b(?:it|that|this|these|those|them|each|both)\b", text, re.I)
        or re.search(r"\bmore\s+about\b", text, re.I)
    )


def _should_keep_previous_token_result(turns: list[tuple[str, str, int]], last_user_index: int) -> bool:
    if last_user_index <= 0:
        return False
    current_user = turns[last_user_index][1]
    if _current_turn_likely_uses_authenticated_api(current_user):
        return True
    if _is_labeled_slot_fill_turn(current_user) or _is_short_value_turn(current_user) or _is_use_value_turn(current_user):
        recent_user_context = " ".join(
            content
            for role, content, _ in turns[max(0, last_user_index - 3) : last_user_index + 1]
            if role == "user"
        )
        if _current_turn_likely_uses_authenticated_api(recent_user_context):
            return True
    previous_ai = ""
    for role, content, _ in reversed(turns[:last_user_index]):
        if role == "ai":
            previous_ai = content
            break
    if not previous_ai:
        return False
    return bool(
        re.search(
            r"\b(?:authenticated|authentication)\b"
            r"|\b(?:got|have|obtained|received|retrieved)\b.{0,40}\btoken\b"
            r"|\btoken\b.{0,80}\b(?:tell|provide|give|what|which|when|where|time|date|content|location)\b",
            previous_ai,
            re.I,
        )
    )


def _current_turn_likely_uses_authenticated_api(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:add|book|cancel|check|create|delete|modif(?:y|ied|ication)|query|remove|schedule|set|update|want|need)\b",
            text,
            re.I,
        )
        and re.search(
            r"\b(?:account|agenda|alarm|appointment|balance|booking|meeting|order|profile|reminder|reservation|schedule)\b",
            text,
            re.I,
        )
    )


def _needs_account_context(text: str) -> bool:
    return bool(re.search(r"\b(?:account|balance|money|funds|bank)\b", text, re.I))


def _has_action_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:calculate|cancel|check|create|delete|find|get|modify|query|register|remove|search|show|solve|tell|update|book|schedule|record|want|need)\b",
            text,
            re.I,
        )
    )


def _has_identity_anchor(text: str) -> bool:
    return bool(
        re.search(r"\b(?:user|appointment|booking|order|patient|account|customer|request)\s*(?:id|ID|number|#)\b", text)
        or re.search(r"\b[A-Z]{1,4}[- ]?\d{3,}[-A-Z0-9]*\b", text)
        or re.search(r"\b\d{6,}\b", text)
    )


def evaluate_file(
    path: Path,
    limit: int = 0,
    api_catalog: dict[str, dict[str, Any]] | None = None,
    tool_search_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = read_rows(path, limit)
    review_rows = []
    counts = Counter()
    for index, row in enumerate(rows):
        expected = expected_name(row)
        tools = tools_from_row(row, api_catalog=api_catalog, tool_search_index=tool_search_index)
        prompt = prompt_text(row)
        plan = build_tool_binding_plan(prompt, tools)
        predicted = [
            str(call.get("tool_name") or "")
            for call in plan.get("calls") or []
            if isinstance(call, dict) and call.get("tool_name")
        ]
        top = predicted[0] if predicted else ""
        ok = bool(expected) and top == expected
        counts["total"] += 1
        counts["top1_ok"] += int(ok)
        counts["no_prediction"] += int(not top)
        counts["multi_prediction"] += int(len(predicted) > 1)
        if not ok:
            review_rows.append(
                {
                    "id": f"{row.get('file', path.name)}:{row.get('id', index)}",
                    "expected": expected,
                    "predicted": json.dumps(predicted, ensure_ascii=False),
                    "tool_decision": plan.get("tool_decision"),
                    "intent_tags": json.dumps((plan.get("task_frame") or {}).get("intent_tags") or []),
                    "tool_names": json.dumps([tool.get("name") for tool in tools], ensure_ascii=False),
                    "prompt": prompt[:1000],
                }
            )
    total = counts["total"]
    return {
        "metrics": {
            "input": str(path),
            "total": total,
            "top1_tool_accuracy": counts["top1_ok"] / total if total else 0.0,
            "no_prediction_rate": counts["no_prediction"] / total if total else 0.0,
            "multi_prediction_rate": counts["multi_prediction"] / total if total else 0.0,
            "failure_count": len(review_rows),
            "api_catalog_size": len(api_catalog or {}),
            "tool_search_index_size": len(tool_search_index or {}),
        },
        "review_rows": review_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--levels", nargs="+", default=["level-1-api", "level-2-api"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    args = parser.parse_args()

    root = Path(args.api_bank_root)
    api_catalog = build_api_catalog(root)
    tool_search_index = build_tool_search_index(root, api_catalog=api_catalog)
    all_metrics = []
    all_review_rows = []
    for level, path in resolve_api_bank_level_paths(root, args.levels):
        result = evaluate_file(path, args.limit, api_catalog=api_catalog, tool_search_index=tool_search_index)
        metrics = result["metrics"]
        metrics["level"] = _level_metric_label(root, level, path)
        all_metrics.append(metrics)
        for row in result["review_rows"]:
            row["level"] = metrics["level"]
            all_review_rows.append(row)

    total = sum(item["total"] for item in all_metrics)
    aggregate = {
        "total": total,
        "top1_tool_accuracy": sum(item["top1_tool_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "no_prediction_rate": sum(item["no_prediction_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "multi_prediction_rate": sum(item["multi_prediction_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
    }
    payload = {"aggregate": aggregate, "by_level": all_metrics}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = ["level", "id", "expected", "predicted", "tool_decision", "intent_tags", "tool_names", "prompt"]
    with Path(args.review_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_review_rows)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
