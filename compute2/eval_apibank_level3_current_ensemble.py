#!/usr/bin/env python3
"""Run API-Bank level-3 with the current ToolACE+xLAM+APIGen+TaskBench ensemble."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import eval_apibank_lv12_current_ensemble as current
import eval_apibank_testdata_ensemble as ens
import eval_apibank_testdata_toolace as toolace_eval
import eval_apibank_toolace_official as base


STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "generate",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "request",
    "the",
    "to",
    "tool",
    "user",
    "with",
}

READ_WORDS = {
    "available",
    "availability",
    "count",
    "find",
    "get",
    "info",
    "information",
    "list",
    "query",
    "retrieve",
    "search",
    "status",
}
ACTION_WORDS = {
    "add",
    "book",
    "email",
    "recommend",
    "recommendation",
    "send",
    "update",
}
FREEFORM_ARG_WORDS = {
    "body",
    "content",
    "description",
    "message",
    "note",
    "subject",
    "summary",
    "text",
    "title",
    "topic",
}


@dataclass(frozen=True)
class StateFeatureGates:
    prior_tool_outputs: bool = False
    retrieval_tool_visible: bool = False
    needs_retrieval: bool = False
    repeated_or_list_action: bool = False
    producer_consumer_dependency: bool = False
    candidate_value_repair: bool = False
    mutable_state: bool = False

    @property
    def active(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, bool]:
        data = asdict(self)
        data["active"] = self.active
        return data


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[_\.\-/]+", " ", text)


def tokens(text: Any) -> set[str]:
    raw = split_identifier(str(text or "")).lower()
    out: set[str] = set()
    for tok in re.findall(r"[a-z0-9]{2,}", raw):
        if tok in STOPWORDS:
            continue
        out.add(tok)
        if len(tok) > 3 and tok.endswith("s") and tok[:-1] not in STOPWORDS:
            out.add(tok[:-1])
        if len(tok) > 5 and tok.endswith("ing") and tok[:-3] not in STOPWORDS:
            out.add(tok[:-3])
        if tok in {"employee", "employees"}:
            out.add("member")
        if tok in {"member", "members"}:
            out.add("employee")
    return out


def norm_name(text: Any) -> str:
    return "".join(ch for ch in str(text or "") if ch.isalnum()).lower()


_TOOL_LIBRARY_CACHE: dict[str, list[dict[str, Any]]] = {}
_TOKEN_DF_CACHE: dict[str, dict[str, int]] = {}


def tool_library(api_bank_root: str) -> list[dict[str, Any]]:
    root = str(Path(api_bank_root).resolve())
    if root in _TOOL_LIBRARY_CACHE:
        return _TOOL_LIBRARY_CACHE[root]
    try:
        from tool_manager import ToolManager

        manager = ToolManager(str(Path(root) / "lv3_apis"))
        apis = []
        for api in manager.apis:
            if isinstance(api, dict) and api.get("name"):
                apis.append({k: v for k, v in api.items() if k not in {"class", "init_database"}})
    except Exception:
        apis = []
    _TOOL_LIBRARY_CACHE[root] = apis
    return apis


def library_by_norm(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {norm_name(api.get("name")): api for api in tool_library(args.api_bank_root)}


def library_token_df(api_bank_root: str) -> dict[str, int]:
    root = str(Path(api_bank_root).resolve())
    if root in _TOKEN_DF_CACHE:
        return _TOKEN_DF_CACHE[root]
    df: dict[str, int] = {}
    for api in tool_library(root):
        for token in tokens(tool_text(api)):
            df[token] = df.get(token, 0) + 1
    _TOKEN_DF_CACHE[root] = df
    return df


def weighted_overlap_score(left: set[str], right: set[str], api_bank_root: str) -> float:
    if not left or not right:
        return 0.0
    df = library_token_df(api_bank_root)

    def weight(token: str) -> float:
        return 1.0 / max(1.0, float(df.get(token, 1)))

    overlap = sum(weight(token) for token in (left & right))
    denom = min(sum(weight(token) for token in left), sum(weight(token) for token in right))
    return overlap / max(1e-9, denom)


def tool_text(api: dict[str, Any]) -> str:
    parts = [api.get("name", ""), api.get("description", "")]
    for group in ("input_parameters", "output_parameters", "parameters"):
        params = api.get(group)
        if isinstance(params, dict):
            for key, spec in params.items():
                parts.append(str(key))
                if isinstance(spec, dict):
                    parts.append(str(spec.get("description") or ""))
                    parts.append(str(spec.get("type") or ""))
    return " ".join(parts)


def parse_pythonish_dict(text: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(text)
    except Exception:
        try:
            value = json.loads(text)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def prior_observations(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(row.get("input") or "")
    observations: list[dict[str, Any]] = []
    pattern = re.compile(
        r"API-Request:\s*\[[^\n]+?\]\s*->\s*(\{[\s\S]*?\})(?=\nAPI-Request:|\nGenerate API Request:|\Z)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        parsed = parse_pythonish_dict(match.group(1).strip())
        if parsed:
            observations.append(parsed)
    return observations


def prior_output_text(row: dict[str, Any], include_toolsearcher: bool = False) -> str:
    parts: list[str] = []
    for obs in prior_observations(row):
        if not include_toolsearcher and obs.get("api_name") == "ToolSearcher":
            continue
        output = obs.get("output")
        if output is not None:
            parts.append(json.dumps(output, ensure_ascii=False, sort_keys=True, default=str))
    return " ".join(parts)


def walk_output_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(walk_output_values(child, child_prefix))
    elif isinstance(value, list):
        for item in value:
            found.extend(walk_output_values(item, prefix))
    else:
        found.append((prefix, value))
    return found


def prior_output_bindings(row: dict[str, Any]) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for obs in prior_observations(row):
        if obs.get("api_name") == "ToolSearcher":
            continue
        api_name = str(obs.get("api_name") or "")
        for path, value in walk_output_values(obs.get("output")):
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            bindings.append(
                {
                    "api_name": api_name,
                    "path": path,
                    "key": path.split(".")[-1] if path else "",
                    "value": value,
                    "text": text,
                    "tokens": tokens(" ".join([api_name, path, text])),
                }
            )
    return bindings


def binding_score_for_param(binding: dict[str, Any], arg_name: str, spec: dict[str, Any]) -> float:
    key_tokens = tokens(binding.get("key") or "")
    path_tokens = tokens(binding.get("path") or "")
    arg_tokens = tokens(arg_name)
    desc_tokens = tokens(spec.get("description") or "")
    target = arg_tokens | desc_tokens
    if not target:
        return 0.0
    name_overlap = len((key_tokens | path_tokens) & arg_tokens) / max(1, len(arg_tokens))
    binding_tokens = key_tokens | path_tokens | set(binding.get("tokens") or set())
    desc_overlap = len(binding_tokens & target) / max(1, min(len(target), len(binding_tokens)) or 1)
    alias_bonus = 0.0
    if "condition" in arg_tokens and "description" in key_tokens:
        alias_bonus += 0.8
    if "weather" in arg_tokens and "description" in key_tokens:
        alias_bonus += 0.5
    return (2.0 * name_overlap) + desc_overlap + alias_bonus


def is_value_grounded(value: Any, row: dict[str, Any]) -> bool:
    if value is None:
        return True
    text = str(value).strip().strip("\"'")
    if not text:
        return True
    context = str(row.get("input") or "").lower()
    if text.lower() in context:
        return True
    for binding in prior_output_bindings(row):
        if str(binding.get("text") or "").lower() == text.lower():
            return True
    return False


def prior_context_text(row: dict[str, Any]) -> str:
    return " ".join([user_utterance(row), prior_output_text(row), str(row.get("input") or "")])


def api_by_name(args: argparse.Namespace, name: str) -> dict[str, Any] | None:
    return library_by_norm(args).get(norm_name(name))


def param_specs(api: dict[str, Any], groups: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    for group in groups:
        params = api.get(group)
        if isinstance(params, dict):
            for key, spec in params.items():
                specs.append((str(key), spec if isinstance(spec, dict) else {}))
    return specs


def param_tokens(api: dict[str, Any], groups: tuple[str, ...]) -> set[str]:
    parts: list[str] = []
    for key, spec in param_specs(api, groups):
        parts.append(key)
        parts.append(str(spec.get("description") or ""))
        parts.append(str(spec.get("type") or ""))
    return tokens(" ".join(parts))


def param_name_tokens(api: dict[str, Any], groups: tuple[str, ...]) -> set[str]:
    return tokens(" ".join(key for key, _spec in param_specs(api, groups)))


def dependency_strength(producer: dict[str, Any], consumer: dict[str, Any]) -> float:
    output_names = param_name_tokens(producer, ("output_parameters", "response"))
    input_names = param_name_tokens(consumer, ("input_parameters", "parameters"))
    output_all = param_tokens(producer, ("output_parameters", "response"))
    input_all = param_tokens(consumer, ("input_parameters", "parameters"))
    if not output_all or not input_all:
        return 0.0
    name_overlap = len(output_names & input_names) / max(1, min(len(output_names), len(input_names)))
    text_overlap = len(output_all & input_all) / max(1, min(len(output_all), len(input_all)))
    return max(name_overlap, text_overlap)


def called_non_search_names(row: dict[str, Any]) -> list[str]:
    return [str(call.get("name") or "") for call in prior_calls(row) if call.get("name") != "ToolSearcher"]


def direct_input_grounding_ratio(api: dict[str, Any], row: dict[str, Any]) -> float:
    params = param_specs(api, ("input_parameters", "parameters"))
    if not params:
        return 1.0
    context = tokens(prior_context_text(row))
    hits = 0
    for key, spec in params:
        key_tokens = tokens(key)
        desc_tokens = tokens(spec.get("description") or "")
        if key_tokens and key_tokens <= context:
            hits += 1
        elif key_tokens and key_tokens & context and (desc_tokens & context):
            hits += 1
    return hits / max(1, len(params))


def called_producer_strength(api: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> float:
    strength = 0.0
    for name in called_non_search_names(row):
        producer = api_by_name(args, name)
        if producer:
            strength = max(strength, dependency_strength(producer, api))
    return strength


def max_prior_list_length(value: Any) -> int:
    if isinstance(value, list):
        nested = [max_prior_list_length(item) for item in value]
        return max([len(value), *nested], default=len(value))
    if isinstance(value, dict):
        return max((max_prior_list_length(item) for item in value.values()), default=0)
    return 0


def prior_output_list_capacity(row: dict[str, Any]) -> int:
    capacity = 0
    for obs in prior_observations(row):
        if obs.get("api_name") == "ToolSearcher":
            continue
        capacity = max(capacity, max_prior_list_length(obs.get("output")))
    return capacity


def prior_same_api_argument_text(row: dict[str, Any], api_name: str) -> str:
    parts = []
    for call in prior_calls(row):
        if call.get("name") == api_name:
            parts.append(json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, default=str))
    return " ".join(parts).lower()


def repeated_call_supported(call: dict[str, Any], row: dict[str, Any]) -> bool:
    name = str(call.get("name") or "")
    priors = prior_calls(row)
    if name not in {str(item.get("name") or "") for item in priors}:
        return False
    if call_signature(call) in {call_signature(item) for item in priors}:
        return False
    if prior_output_list_capacity(row) <= sum(1 for item in priors if item.get("name") == name):
        return False
    if list_cursor_score(call, row) <= 0.0:
        return False
    if argument_grounding(call, row) < 2.0:
        return False
    return True


def list_cursor_score(call: dict[str, Any], row: dict[str, Any]) -> float:
    name = str(call.get("name") or "")
    if not name:
        return 0.0
    priors = prior_calls(row)
    same_api_count = sum(1 for item in priors if item.get("name") == name)
    if same_api_count <= 0:
        return 0.0
    if prior_output_list_capacity(row) <= same_api_count:
        return 0.0
    output_text = prior_output_text(row).lower()
    same_api_args = prior_same_api_argument_text(row, name)
    output_tokens = tokens(output_text)
    used_tokens = tokens(same_api_args)
    score = 0.0
    for value in (call.get("arguments") or {}).values():
        text = (
            json.dumps(value, ensure_ascii=False, default=str).lower()
            if isinstance(value, (dict, list))
            else str(value).lower()
        )
        text = text.strip().strip('"')
        if len(text) >= 3 and text in output_text and text not in same_api_args:
            score += 1.75
            continue
        value_tokens = tokens(text)
        if value_tokens and (value_tokens & output_tokens) and not (value_tokens <= used_tokens):
            score += 0.75
    return score


def api_kind(api: dict[str, Any] | None) -> tuple[bool, bool]:
    if not api:
        return False, False
    text_tokens = tokens(tool_text(api))
    return bool(text_tokens & READ_WORDS), bool(text_tokens & ACTION_WORDS)


def user_utterance(row: dict[str, Any]) -> str:
    try:
        return ens.first_user_utterance(row)
    except Exception:
        match = re.search(r"User:\s*(.*?)(?:\n|$)", str(row.get("input") or ""), re.IGNORECASE)
        return match.group(1).strip() if match else str(row.get("input") or "")


def prior_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"API-Request:\s*(\[[^\n]+?\])", str(row.get("input") or "")):
        call = base.parse_first_call(match.group(1))
        if call:
            calls.append(call)
    return calls


def call_signature(call: dict[str, Any] | None) -> str:
    if not call:
        return ""
    return json.dumps(
        {"name": call.get("name"), "arguments": call.get("arguments") or {}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def available_api_names(tools: list[dict[str, Any]]) -> set[str]:
    return {str(tool.get("name")) for tool in tools if tool.get("name")}


def searched_tool_names(row: dict[str, Any], args: argparse.Namespace) -> set[str]:
    by_norm = library_by_norm(args)
    searched: set[str] = set()
    for call in prior_calls(row):
        if call.get("name") != "ToolSearcher":
            continue
        keyword = ((call.get("arguments") or {}).get("keywords") or "")
        api = by_norm.get(norm_name(keyword))
        searched.add(str(api.get("name") if api else keyword))
    return searched


def retrieval_score(api: dict[str, Any], row: dict[str, Any], already_called_non_search: bool, api_bank_root: str) -> float:
    context_tokens = tokens(user_utterance(row))
    context_tokens |= tokens(re.sub(r"API descriptions:[\s\S]*?User:", "User:", str(row.get("input") or ""), flags=re.IGNORECASE))
    api_tokens = tokens(tool_text(api))
    if not context_tokens or not api_tokens:
        return 0.0
    overlap = len(context_tokens & api_tokens) / max(1, min(len(context_tokens), len(api_tokens)))
    weighted_overlap = weighted_overlap_score(context_tokens, api_tokens, api_bank_root)
    name_tokens = tokens(api.get("name", ""))
    score = (6.0 * overlap) + (12.0 * weighted_overlap)
    if context_tokens & name_tokens:
        score += 3.0
    text_tokens = tokens(tool_text(api))
    is_read = bool(text_tokens & READ_WORDS)
    is_action = bool(text_tokens & ACTION_WORDS)
    if not already_called_non_search and is_read:
        score += 1.5
    if not already_called_non_search and is_action:
        score -= 2.5
    if already_called_non_search and is_action:
        score += 0.75
    return score


def best_retrieval_call(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if "ToolSearcher" not in available_api_names(record["tools"]):
        return None
    row = record["row"]
    available = available_api_names(record["tools"])
    searched = searched_tool_names(row, args)
    prior_sigs = {call_signature(item) for item in prior_calls(row)}
    prior_names = {str(item.get("name") or "") for item in prior_calls(row)}
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        if not call or (candidate.get("issues") or ens.schema_issues(call, record["tools"])):
            continue
        name = str(call.get("name") or "")
        if name == "ToolSearcher":
            continue
        if name in available and call_signature(call) not in prior_sigs and name not in prior_names:
            return None
        if name in available and repeated_call_supported(call, row):
            return None

    search_keywords = []
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        if not call or call.get("name") != "ToolSearcher":
            continue
        keyword = str((call.get("arguments") or {}).get("keywords") or "").strip()
        exact = exact_library_api(keyword, args)
        if exact and (exact in available or exact in searched):
            continue
        if keyword:
            search_keywords.append(keyword)
    if not search_keywords:
        return None

    called_non_search = any(call.get("name") != "ToolSearcher" for call in prior_calls(row))
    candidate_items: list[dict[str, Any]] = []
    for api in tool_library(args.api_bank_root):
        name = str(api.get("name") or "")
        if not name or name == "ToolSearcher" or name in available or name in searched:
            continue
        score = retrieval_score(api, row, called_non_search, args.api_bank_root)
        api_tokens = tokens(tool_text(api))
        name_tokens = tokens(name)
        keyword_support = 0.0
        for keyword in search_keywords:
            keyword_tokens = tokens(keyword)
            if not keyword_tokens:
                continue
            keyword_support = max(
                keyword_support,
                len(keyword_tokens & (api_tokens | name_tokens)) / max(1, min(len(keyword_tokens), len(api_tokens | name_tokens))),
            )
        weighted_context_support = weighted_overlap_score(tokens(prior_context_text(row)), api_tokens | name_tokens, args.api_bank_root)
        if keyword_support < 0.20:
            continue
        score += 8.0 * keyword_support
        score += 2.0 * weighted_context_support
        if score > 0:
            candidate_items.append({"score": score, "name": name, "api": api, "keyword_support": keyword_support})
    if not candidate_items:
        return None

    called_names = called_non_search_names(row)
    called_apis = [api for api_name in called_names if (api := api_by_name(args, api_name))]
    for item in candidate_items:
        api = item["api"]
        score = float(item["score"])
        called_dep = max((dependency_strength(producer, api) for producer in called_apis), default=0.0)
        input_grounding = direct_input_grounding_ratio(api, row)
        missing_producer_dep = 0.0
        missing_producer_name = ""
        producer_for_relevant_consumer = 0.0
        producer_consumer_name = ""
        for other in candidate_items:
            if other is item:
                continue
            other_api = other["api"]
            other_name = str(other["name"])
            dep_to_item = dependency_strength(other_api, api)
            if dep_to_item >= 0.20 and input_grounding < 0.75 and called_dep < 0.20:
                weighted = dep_to_item * max(0.0, float(other["score"]))
                if weighted > missing_producer_dep:
                    missing_producer_dep = weighted
                    missing_producer_name = other_name
            dep_from_item = dependency_strength(api, other_api)
            other_grounding = direct_input_grounding_ratio(other_api, row)
            other_called_dep = called_producer_strength(other_api, row, args)
            if dep_from_item >= 0.20 and other_grounding < 0.75 and other_called_dep < 0.20:
                weighted = dep_from_item * max(0.0, float(other["score"]))
                if weighted > producer_for_relevant_consumer:
                    producer_for_relevant_consumer = weighted
                    producer_consumer_name = other_name

        if called_dep >= 0.20:
            score += 6.0 * called_dep
        if missing_producer_dep >= 1.0:
            score -= min(16.0, 0.75 * missing_producer_dep)
        if producer_for_relevant_consumer >= 1.0:
            score += min(8.0, 0.35 * producer_for_relevant_consumer)

        item["score"] = score
        item["dependency_debug"] = {
            "called_producer_strength": round(called_dep, 3),
            "input_grounding": round(input_grounding, 3),
            "missing_producer": missing_producer_name,
            "producer_for_relevant_consumer": producer_consumer_name,
        }

    candidate_items.sort(key=lambda item: (float(item["score"]), str(item["name"])), reverse=True)
    best = candidate_items[0]
    score = float(best["score"])
    name = str(best["name"])
    if score < 1.25:
        return None
    call = {"name": "ToolSearcher", "arguments": {"keywords": name}}
    candidate = current.make_candidate(
        "retrieval_repair",
        {"raw_text": f"state-aware retrieval: {name}", "latency_ms": 0, "generated_tokens": 0},
        record["tools"],
    )
    candidate["call"] = call
    candidate["calls"] = [call]
    candidate["pred"] = ens.call_to_text(call)
    candidate["issues"] = ens.schema_issues(call, record["tools"])
    candidate["retrieval_score"] = round(score, 3)
    candidate["retrieval_debug"] = best.get("dependency_debug")
    return candidate


def row_prompt(row: dict[str, Any]) -> str:
    return (str(row.get("instruction") or "").strip() + "\n" + str(row.get("input") or "").strip()).strip()


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    api_bank_root = Path(args.api_bank_root).resolve()
    os.chdir(api_bank_root)
    sys.path.insert(0, str(api_bank_root))
    base.install_optional_dependency_stubs()
    rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rows = select_rows(rows, args)
    records = []
    for row in rows:
        tools = toolace_eval.tools_from_row(row)
        records.append(
            {
                "row": row,
                "tools": tools,
                "prompt": row_prompt(row),
                "candidates": [],
            }
        )
    return records


def run_toolace(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.toolace_mode == "off":
        return
    tool_args = argparse.Namespace(
        base=args.toolace_model,
        precision=args.precision if args.precision in {"bf16", "fp16"} else "bf16",
        device_map=args.device_map,
        load_4bit=args.load_4bit,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[apibank-current-l3] loading ToolACE candidate: {args.toolace_model}", flush=True)
    model, tokenizer = toolace_eval.load_model(tool_args)
    try:
        for idx, record in enumerate(records, start=1):
            try:
                result = toolace_eval.generate_api(model, tokenizer, record["row"], tool_args)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"raw_text": "", "pred": "", "latency_ms": 0, "generated_tokens": 0}
                error = f"{type(exc).__name__}: {exc}"
            record["candidates"].append(current.make_candidate("toolace", result, record["tools"], error))
            print(json.dumps({"phase": "toolace", "n": idx, "total": len(records), "error": error}, ensure_ascii=False), flush=True)
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def write_and_score(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    preds = []
    for record in records:
        row = record["row"]
        chosen = record.get("chosen") or {}
        candidates = record.get("candidates") or []
        pred = {
            **{key: row[key] for key in ("file", "id", "sample_id", "api_id") if key in row},
            "pred": chosen.get("pred", ""),
            "raw_text": chosen.get("raw_text", ""),
            "expected_output": row.get("expected_output") or row.get("output", ""),
            "expected_call": toolace_eval.expected_call(row.get("expected_output") or row.get("output", "")),
            "latency_ms": round(sum(float((cand or {}).get("latency_ms") or 0) for cand in candidates), 3),
            "generated_tokens": int(sum(int((cand or {}).get("generated_tokens") or 0) for cand in candidates)),
            "error": chosen.get("error", ""),
            "ensemble": {
                "implementation": "current_toolace_xlam_apigen_taskbench_xlam_verifier",
                "chosen_source": chosen.get("source"),
                "verifier": record.get("verifier"),
                "state_features": record.get("state_features"),
                "candidates": [current.compact_candidate(candidate) for candidate in candidates],
            },
        }
        preds.append(pred)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pred in preds:
            handle.write(json.dumps(pred, ensure_ascii=False) + "\n")

    score = ens.score_level3(preds, Path(args.level3_gt), Path(args.api_bank_root).resolve(), args.max_error_details)
    score_path = Path(args.score_output)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: score[key]
                for key in (
                    "paper_metric",
                    "paper_ability",
                    "total_api_calls",
                    "correct_api_calls",
                    "accuracy",
                    "sample_total",
                    "sample_correct",
                    "sample_accuracy",
                    "errors",
                )
            },
            indent=2,
        ),
        flush=True,
    )


def exact_library_api(keyword: Any, args: argparse.Namespace) -> str:
    api = library_by_norm(args).get(norm_name(keyword))
    return str(api.get("name")) if api else ""


def state_repaired_call(call: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if not call or call.get("name") == "ToolSearcher":
        return None
    name = str(call.get("name") or "")
    api = api_by_name(args, name)
    if not api:
        return None
    call_args = call.get("arguments") or {}
    if not isinstance(call_args, dict):
        return None
    bindings = prior_output_bindings(row)
    if not bindings:
        return None

    new_args = dict(call_args)
    changed = False
    for arg_name, spec in param_specs(api, ("input_parameters", "parameters")):
        candidates = [
            (binding_score_for_param(binding, arg_name, spec), binding)
            for binding in bindings
        ]
        candidates = [(score, binding) for score, binding in candidates if score >= 1.15]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, binding = candidates[0]
        replacement = binding.get("value")
        current_value = new_args.get(arg_name)
        if current_value is not None and str(current_value).strip().lower() == str(replacement).strip().lower():
            continue
        if (
            isinstance(current_value, str)
            and isinstance(replacement, str)
            and current_value.strip()
            and tokens(arg_name) & FREEFORM_ARG_WORDS
        ):
            continue
        if isinstance(current_value, (int, float)) and isinstance(replacement, (int, float)):
            if abs(float(current_value)) == abs(float(replacement)) and float(current_value) != float(replacement):
                continue
        should_replace = arg_name not in new_args or not is_value_grounded(current_value, row)
        if not should_replace and isinstance(current_value, (int, float)) and isinstance(replacement, (int, float)):
            # Common model error: percent-like values copied as 80 instead of a prior state value 0.8.
            should_replace = abs(float(current_value)) > 1.0 and 0.0 <= abs(float(replacement)) <= 1.0 and score >= 1.5
        if should_replace:
            new_args[arg_name] = replacement
            changed = True
    if not changed:
        return None
    return {"name": name, "arguments": new_args}


def add_state_value_repairs(record: dict[str, Any], args: argparse.Namespace) -> None:
    seen = {call_signature(candidate.get("call")) for candidate in record.get("candidates") or []}
    additions: list[dict[str, Any]] = []
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        repaired = state_repaired_call(call, record["row"], args) if call else None
        if not repaired or call_signature(repaired) in seen:
            continue
        repair_candidate = current.make_candidate(
            "state_value_repair",
            {"raw_text": f"state value repair from {candidate.get('source')}", "latency_ms": 0, "generated_tokens": 0},
            record["tools"],
        )
        repair_candidate["call"] = repaired
        repair_candidate["calls"] = [repaired]
        repair_candidate["pred"] = ens.call_to_text(repaired)
        repair_candidate["issues"] = ens.schema_issues(repaired, record["tools"])
        repair_candidate["repaired_from"] = candidate.get("source")
        additions.append(repair_candidate)
        seen.add(call_signature(repaired))
    record.setdefault("candidates", []).extend(additions)


def unresolved_retrieval_keywords(record: dict[str, Any], args: argparse.Namespace) -> list[str]:
    available = available_api_names(record["tools"])
    searched = searched_tool_names(record["row"], args)
    keywords: list[str] = []
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        if not call:
            continue
        name = str(call.get("name") or "")
        if name == "ToolSearcher":
            keyword = str((call.get("arguments") or {}).get("keywords") or "").strip()
            exact = exact_library_api(keyword, args)
            if keyword and exact not in available and exact not in searched:
                keywords.append(keyword)
            continue
        if name not in available and exact_library_api(name, args):
            keywords.append(name)
    return keywords


def candidate_api_names(record: dict[str, Any], args: argparse.Namespace) -> set[str]:
    names: set[str] = set()
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        if not call:
            continue
        name = str(call.get("name") or "")
        if name == "ToolSearcher":
            exact = exact_library_api((call.get("arguments") or {}).get("keywords"), args)
            if exact:
                names.add(exact)
        elif name:
            names.add(name)
    return names


def has_producer_consumer_dependency(record: dict[str, Any], args: argparse.Namespace) -> bool:
    row = record["row"]
    prior_apis = [api for name in called_non_search_names(row) if (api := api_by_name(args, name))]
    visible_apis = [
        api
        for name in available_api_names(record["tools"])
        if name != "ToolSearcher" and (api := api_by_name(args, name))
    ]
    candidate_apis = [api for name in candidate_api_names(record, args) if (api := api_by_name(args, name))]

    for producer in prior_apis:
        for consumer in candidate_apis or visible_apis:
            if dependency_strength(producer, consumer) >= 0.20:
                return True

    related = candidate_apis or visible_apis
    for left_idx, producer in enumerate(related):
        for right_idx, consumer in enumerate(related):
            if left_idx != right_idx and dependency_strength(producer, consumer) >= 0.20:
                return True
    return False


def state_feature_gates(record: dict[str, Any], args: argparse.Namespace) -> StateFeatureGates:
    row = record["row"]
    prior_obs = [obs for obs in prior_observations(row) if obs.get("api_name") != "ToolSearcher"]
    prior_tool_outputs = any(obs.get("output") is not None for obs in prior_obs)
    retrieval_tool_visible = "ToolSearcher" in available_api_names(record["tools"])
    needs_retrieval = retrieval_tool_visible and bool(unresolved_retrieval_keywords(record, args))
    prior_sigs = {call_signature(item) for item in prior_calls(row)}
    prior_names = {str(item.get("name") or "") for item in prior_calls(row)}
    repeated_or_list_action = False
    for candidate in record.get("candidates") or []:
        call = candidate.get("call")
        if not call:
            continue
        name = str(call.get("name") or "")
        if name and name in prior_names and call_signature(call) not in prior_sigs:
            repeated_or_list_action = True
            break
    if not repeated_or_list_action and prior_output_list_capacity(row) > len(prior_names):
        repeated_or_list_action = bool(candidate_api_names(record, args) & prior_names)

    candidate_value_repair = any(
        state_repaired_call(candidate.get("call"), row, args) is not None
        for candidate in record.get("candidates") or []
        if candidate.get("call")
    )
    mutable_state = prior_tool_outputs and (prior_output_list_capacity(row) > 0 or bool(prior_output_bindings(row)))
    return StateFeatureGates(
        prior_tool_outputs=prior_tool_outputs,
        retrieval_tool_visible=retrieval_tool_visible,
        needs_retrieval=needs_retrieval,
        repeated_or_list_action=repeated_or_list_action,
        producer_consumer_dependency=has_producer_consumer_dependency(record, args),
        candidate_value_repair=candidate_value_repair,
        mutable_state=mutable_state,
    )


def state_mode_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "chronology_mode", "state") == "state"


def argument_grounding(call: dict[str, Any], row: dict[str, Any]) -> float:
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return 0.0
    context = str(row.get("input") or "").lower()
    score = 0.0
    for value in args.values():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)
        text_l = text.lower().strip()
        if not text_l:
            continue
        if len(text_l) <= 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(text_l)}(?![a-z0-9])", context):
                score += 1.0
        elif text_l in context:
            score += 1.5
        elif any(str(binding.get("text") or "").lower() == text_l for binding in prior_output_bindings(row)):
            score += 1.75
        else:
            value_tokens = tokens(text_l)
            if value_tokens:
                score += min(1.0, len(value_tokens & tokens(context)) / max(1, len(value_tokens)))
    return score


def state_score(
    candidate: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
    verifier_chosen: str | None,
    features: StateFeatureGates | None = None,
) -> float:
    call = candidate.get("call")
    issues = candidate.get("issues") or ens.schema_issues(call, record["tools"])
    if not call:
        return -100.0
    score = {
        "retrieval_repair": 7.0,
        "state_value_repair": 5.5,
        "verifier_repair": 5.0,
        "xlam": 4.0,
        "toolace": 3.0,
        "gptoss_apigen": 2.5,
        "taskbench": 1.5,
    }.get(str(candidate.get("source")), 0.0)
    score -= 40.0 * len(issues)
    if issues:
        return score
    name = str(call.get("name") or "")
    row = record["row"]
    priors = prior_calls(row)
    prior_sigs = {call_signature(item) for item in priors}
    prior_names = [str(item.get("name") or "") for item in priors]
    available = available_api_names(record["tools"])
    if name == "ToolSearcher":
        if features is not None and not features.needs_retrieval:
            score -= 20.0
        keyword = (call.get("arguments") or {}).get("keywords")
        exact = exact_library_api(keyword, args)
        if exact:
            score += 12.0
        else:
            score -= 12.0
        if exact and exact in available:
            score -= 18.0
        if exact and exact in searched_tool_names(row, args):
            score -= 14.0
        if candidate.get("retrieval_score") is not None:
            score += float(candidate["retrieval_score"])
        return score

    if name in available:
        score += 12.0
    else:
        score -= 25.0
    sig = call_signature(call)
    if sig in prior_sigs:
        score -= 30.0
    elif name in prior_names:
        api = api_by_name(args, name)
        is_read, is_action = api_kind(api)
        if repeated_call_supported(call, row):
            score += 7.0
        elif is_read and not is_action:
            score -= 12.0
        else:
            score -= 5.0
    else:
        score += 8.0
    score += min(8.0, argument_grounding(call, row))
    score += min(4.0, list_cursor_score(call, row))
    if verifier_chosen and candidate.get("source") == verifier_chosen:
        score += 1.5
    return score


def verifier_repair_candidate(
    repair_call: dict[str, Any],
    verifier_raw: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    repair = current.make_candidate(
        "verifier_repair",
        {"raw_text": verifier_raw, "latency_ms": 0, "generated_tokens": 0},
        tools,
    )
    repair["call"] = repair_call
    repair["calls"] = [repair_call]
    repair["pred"] = ens.call_to_text(repair_call, verifier_raw)
    repair["issues"] = []
    return repair


def choose_static_record(
    record: dict[str, Any],
    chosen_source: str | None,
    repair_call: dict[str, Any] | None,
    verifier_raw: str,
) -> None:
    if repair_call and not ens.schema_issues(repair_call, record["tools"]):
        chosen = verifier_repair_candidate(repair_call, verifier_raw, record["tools"])
        record.setdefault("candidates", []).append(chosen)
    else:
        chosen = None
        if chosen_source:
            for candidate in record.get("candidates") or []:
                if candidate.get("source") == chosen_source and not candidate.get("issues"):
                    chosen = candidate
                    break
        if chosen is None:
            ranked = sorted(record.get("candidates") or [], key=current.static_score, reverse=True)
            chosen = ranked[0] if ranked else current.make_candidate("none", {"raw_text": ""}, record["tools"])

    for candidate in record.get("candidates") or []:
        candidate["selector_score"] = round(current.static_score(candidate), 3)
        candidate["selector_issues"] = candidate.get("issues") or ens.schema_issues(candidate.get("call"), record["tools"])
    record["chosen"] = chosen


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.chronology_mode == "prefer":
        fallback_records: list[dict[str, Any]] = []
        for idx, record in enumerate(records, start=1):
            hint = ens.chronology_hint_candidate(record["row"], record["tools"])
            if hint:
                hint["selector_score"] = 999.0
                hint["selector_issues"] = ens.schema_issues(hint.get("call"), record["tools"])
                record["candidates"].append(hint)
            if hint and not hint["selector_issues"]:
                record["chosen"] = hint
                record["verifier"] = {
                    "chosen_source": "chronology",
                    "raw_text": "skipped: chronology controller selected schema-valid next call",
                }
                print(
                    json.dumps(
                        {"phase": "select", "n": idx, "total": len(records), "chosen": "chronology", "verifier_chosen": "chronology"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                fallback_records.append(record)
        if fallback_records:
            current.select_records(fallback_records, args)
        return

    verifier = None
    if args.verifier_model and args.verifier_model != "none":
        print(f"[apibank-current-l3] loading verifier: {args.verifier_model}", flush=True)
        verifier = current.APIBankVerifier(args.verifier_model, args)
    try:
        for idx, record in enumerate(records, start=1):
            pre_features = state_feature_gates(record, args) if state_mode_enabled(args) else StateFeatureGates()
            if state_mode_enabled(args) and pre_features.needs_retrieval:
                retrieval = best_retrieval_call(record, args)
                if retrieval:
                    record.setdefault("candidates", []).append(retrieval)
            features = state_feature_gates(record, args) if state_mode_enabled(args) else StateFeatureGates()
            if state_mode_enabled(args) and features.candidate_value_repair:
                add_state_value_repairs(record, args)
                features = state_feature_gates(record, args)
            record["state_features"] = features.to_dict()
            chosen_source = None
            verifier_raw = ""
            repair_call = None
            if verifier is not None:
                try:
                    chosen_source, repair_call, verifier_raw = verifier.choose(record)
                except Exception as exc:  # noqa: BLE001
                    verifier_raw = f"{type(exc).__name__}: {exc}"
            if not (state_mode_enabled(args) and features.active):
                choose_static_record(record, chosen_source, repair_call, verifier_raw)
                chosen = record["chosen"]
            else:
                if repair_call and not ens.schema_issues(repair_call, record["tools"]):
                    record.setdefault("candidates", []).append(verifier_repair_candidate(repair_call, verifier_raw, record["tools"]))
                candidates = record.get("candidates") or []
                for candidate in candidates:
                    candidate["selector_score"] = round(state_score(candidate, record, args, chosen_source, features), 3)
                    candidate["selector_issues"] = candidate.get("issues") or ens.schema_issues(candidate.get("call"), record["tools"])
                chosen = max(
                    candidates,
                    key=lambda item: item.get("selector_score", -1000.0),
                    default=current.make_candidate("none", {"raw_text": ""}, record["tools"]),
                )
                record["chosen"] = chosen
            record["verifier"] = {"chosen_source": chosen_source, "raw_text": verifier_raw}
            print(
                json.dumps(
                    {
                        "phase": "select",
                        "n": idx,
                        "total": len(records),
                        "chosen": chosen.get("source"),
                        "score": chosen.get("selector_score"),
                        "verifier_chosen": chosen_source,
                        "state_features": record["state_features"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if verifier is not None:
            verifier.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--level3-gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--toolace-mode", choices=("off", "always"), default="always")
    parser.add_argument("--toolace-model", default=current.DEFAULT_TOOLACE)
    parser.add_argument("--xlam-model", default=current.DEFAULT_XLAM)
    parser.add_argument("--verifier-model", default=current.DEFAULT_XLAM)
    parser.add_argument("--gptoss-base", default=current.DEFAULT_GPTOSS_BASE)
    parser.add_argument("--apigen-adapter", default=current.DEFAULT_APIGEN)
    parser.add_argument("--taskbench-adapter", default=current.DEFAULT_TASKBENCH)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chronology-mode", choices=("state", "prefer", "off"), default="state")
    parser.add_argument("--max-error-details", type=int, default=80)
    args = parser.parse_args()

    start = time.time()
    records = load_records(args)
    print(
        json.dumps(
            {
                "rows": len(records),
                "input_json": args.input_json,
                "implementation": "current_toolace_xlam_apigen_taskbench_xlam_verifier",
                "toolace_model": args.toolace_model,
                "xlam_model": args.xlam_model,
                "verifier_model": args.verifier_model,
                "apigen_adapter": args.apigen_adapter,
                "taskbench_adapter": args.taskbench_adapter,
            },
            indent=2,
        ),
        flush=True,
    )
    run_toolace(records, args)
    current.run_hf_candidate(records, "xlam", args.xlam_model, "", args)
    current.run_hf_candidate(records, "gptoss_apigen", args.gptoss_base, args.apigen_adapter, args)
    current.run_hf_candidate(records, "taskbench", args.gptoss_base, args.taskbench_adapter, args)
    select_records(records, args)
    write_and_score(records, args)
    print(json.dumps({"wall_seconds": round(time.time() - start, 3)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
