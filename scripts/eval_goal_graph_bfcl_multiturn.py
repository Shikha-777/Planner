#!/usr/bin/env python3
"""Evaluate BFCL multi-turn categories with the goal-graph runtime.

This runner keeps BFCL's official multi-turn execution and scoring loop, but
uses the goal-graph stepwise planner to produce the next function-call batch for
each active turn step.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
COMPUTE2_DIR = ROOT_DIR / "compute2"
for path in (SRC_DIR, SCRIPT_DIR, COMPUTE2_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_bfcl_multiturn_current_ensemble as mt
from bfcl_compare_eval import normalize_tools
from goal_graph_eval_common import benchmark_compile_tools, plan_and_compile_goal_graph
from run_gptoss_capability_plan import generate_text, load_model


def unload_model(model: Any = None, tokenizer: Any = None) -> None:
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


def normalized_tools_for_state(state: mt.EntryState) -> list[dict[str, Any]]:
    synthetic_record = {"function": state.available_tools}
    try:
        tools = normalize_tools(synthetic_record)
    except Exception:
        tools = [tool for tool in state.available_tools if isinstance(tool, dict)]
    return benchmark_compile_tools(tools)


EXISTING_FS_ARG_KINDS: dict[str, dict[str, str]] = {
    "cat": {"file_name": "file", "path": "file"},
    "cd": {"folder": "directory", "path": "directory"},
    "cp": {"source": "any", "src": "any"},
    "diff": {"file_name1": "file", "file_name2": "file", "path1": "file", "path2": "file"},
    "find": {"path": "directory"},
    "grep": {"file_name": "file", "path": "file"},
    "ls": {"directory": "directory", "folder": "directory", "path": "directory"},
    "mv": {"source": "any", "src": "any"},
    "rm": {"file_name": "any", "path": "any"},
    "sort": {"file_name": "file", "path": "file"},
    "tail": {"file_name": "file", "path": "file"},
    "wc": {"file_name": "file", "path": "file"},
}


def _fs_entries_from_initial_config(initial_config: Any) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, path)
            return
        if not isinstance(node, dict):
            return

        node_type = str(node.get("type") or "").lower()
        if node_type in {"directory", "folder"}:
            if path:
                entries.append((path[-1], "directory"))
            contents = node.get("contents")
            if isinstance(contents, dict):
                for child_name, child in contents.items():
                    walk(child, path + [str(child_name)])
            return
        if node_type == "file":
            if path:
                entries.append((path[-1], "file"))
            return

        for key, child in node.items():
            if key in {"contents", "type"}:
                continue
            if isinstance(child, dict) and child.get("type"):
                walk(child, path + [str(key)])
            else:
                walk(child, path)

    walk(initial_config, [])
    return entries


def _fs_paths_from_initial_config(initial_config: Any) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, path)
            return
        if not isinstance(node, dict):
            return
        node_type = str(node.get("type") or "").lower()
        if node_type in {"directory", "folder"}:
            if path:
                entries.append(("/".join(path), path[-1], "directory"))
            contents = node.get("contents")
            if isinstance(contents, dict):
                for child_name, child in contents.items():
                    walk(child, path + [str(child_name)])
            return
        if node_type == "file":
            if path:
                entries.append(("/".join(path), path[-1], "file"))
            return
        for key, child in node.items():
            if key in {"contents", "type"}:
                continue
            if isinstance(child, dict) and child.get("type"):
                walk(child, path + [str(key)])
            else:
                walk(child, path)

    walk(initial_config, [])
    return entries


def _direct_child_files(initial_config: Any, directory_name: str) -> list[str]:
    target = directory_name.strip().lower()
    files: list[str] = []

    def walk(node: Any, path: list[str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, path)
            return
        if not isinstance(node, dict):
            return
        node_type = str(node.get("type") or "").lower()
        if node_type in {"directory", "folder"}:
            contents = node.get("contents")
            if path and path[-1].lower() == target and isinstance(contents, dict):
                for child_name, child in contents.items():
                    if isinstance(child, dict) and str(child.get("type") or "").lower() == "file":
                        files.append(str(child_name))
                return
            if isinstance(contents, dict):
                for child_name, child in contents.items():
                    walk(child, path + [str(child_name)])
            return
        for key, child in node.items():
            if key in {"contents", "type"}:
                continue
            if isinstance(child, dict) and child.get("type"):
                walk(child, path + [str(key)])
            else:
                walk(child, path)

    walk(initial_config, [])
    return files


def _fs_aliases(name: str) -> set[str]:
    lower = name.strip().lower()
    aliases = {lower}
    if "." in lower:
        stem = lower.rsplit(".", 1)[0]
        if stem:
            aliases.add(stem)
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    if compact:
        aliases.add(compact)
    spaced = re.sub(r"[_-]+", " ", lower).strip()
    if spaced:
        aliases.add(spaced)
    underscored = re.sub(r"[\s-]+", "_", lower).strip("_")
    if underscored:
        aliases.add(underscored)
    if len(lower) > 3 and lower.endswith("s") and not lower.endswith("ss"):
        aliases.add(lower[:-1])
    elif len(lower) > 2:
        aliases.add(f"{lower}s")
    if len(lower) > 4 and lower.endswith("ed"):
        aliases.add(lower[:-1])
    return {alias for alias in aliases if alias}


def _fs_entry_index(state: mt.EntryState, kind: str) -> dict[str, set[str]]:
    wanted = {kind}
    if kind == "any":
        wanted = {"directory", "file"}
    index: dict[str, set[str]] = {}
    for name, entry_kind in _fs_entries_from_initial_config(state.record.get("initial_config", {})):
        if entry_kind not in wanted:
            continue
        for alias in _fs_aliases(name):
            index.setdefault(alias, set()).add(name)
    return index


def _history_directory_names(state: mt.EntryState) -> set[str]:
    names: set[str] = set()
    for batch in state.executed_call_history:
        for call_text in batch:
            for call in mt.parse_python_calls(f"[{call_text}]"):
                call_name = str(call.get("name") or "").lower()
                args = mt.selector.maybe_json(call.get("arguments") or {})
                if not isinstance(args, dict):
                    continue
                if call_name == "mkdir" and isinstance(args.get("dir_name"), str):
                    names.add(args["dir_name"])
                if call_name == "cd" and isinstance(args.get("folder"), str):
                    names.add(args["folder"])
    return names


def _known_directory_names(state: mt.EntryState) -> set[str]:
    names = {
        name
        for name, kind in _fs_entries_from_initial_config(state.record.get("initial_config", {}))
        if kind == "directory"
    }
    names.update(_history_directory_names(state))
    return {name for name in names if name}


def _resolve_known_directory_name(state: mt.EntryState, value: Any) -> str:
    if not isinstance(value, str):
        return ""
    aliases = _fs_aliases(value)
    matches = [
        name
        for name in _known_directory_names(state)
        if aliases & _fs_aliases(name)
    ]
    if len(matches) == 1:
        return matches[0]
    stripped = value.strip()
    return stripped if stripped else ""


def _directory_exists(state: mt.EntryState, value: Any) -> bool:
    resolved = _resolve_known_directory_name(state, value)
    return bool(resolved and any(resolved.lower() == name.lower() for name in _known_directory_names(state)))


def _resolve_existing_fs_name(state: mt.EntryState, value: Any, kind: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped in {".", ".."}:
        return value

    index = _fs_entry_index(state, kind)

    def resolve_segment(segment: str) -> str:
        if not segment or segment in {".", ".."}:
            return segment
        aliases = _fs_aliases(segment)
        matches: set[str] = set()
        for alias in aliases:
            matches.update(index.get(alias, set()))
        if len(matches) == 1:
            return next(iter(matches))
        return segment

    if "/" not in stripped:
        resolved = resolve_segment(stripped)
    else:
        prefix = "/" if stripped.startswith("/") else ""
        parts = [resolve_segment(part) for part in stripped.split("/") if part != ""]
        resolved = prefix + "/".join(parts)

    if stripped != value:
        return value.replace(stripped, resolved, 1)
    return resolved


def repair_filesystem_call_arguments(
    state: mt.EntryState,
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = copy.deepcopy(call)
        name = str(item.get("name") or "").lower()
        arg_kinds = EXISTING_FS_ARG_KINDS.get(name, {})
        args = mt.selector.maybe_json(item.get("arguments") or {})
        if not arg_kinds or not isinstance(args, dict):
            repaired.append(item)
            continue
        changed = False
        new_args = copy.deepcopy(args)
        if name == "find" and isinstance(new_args.get("name"), str):
            stripped_name = new_args["name"].strip("*")
            if stripped_name and stripped_name != new_args["name"]:
                new_args["name"] = stripped_name
                changed = True
        for arg_name, kind in arg_kinds.items():
            if arg_name not in new_args:
                continue
            new_value = _resolve_existing_fs_name(state, new_args[arg_name], kind)
            if new_value != new_args[arg_name]:
                new_args[arg_name] = new_value
                changed = True
        if changed:
            item["arguments"] = new_args
        repaired.append(item)
    return repaired


def repair_filesystem_candidate(
    state: mt.EntryState,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    item = copy.deepcopy(candidate)
    calls = item.get("calls") or []
    if isinstance(calls, list):
        item["calls"] = normalize_multiturn_calls(state, calls)
        item["issues"] = mt.selector.call_issues(item["calls"], state.available_tools)
    return item


def normalize_multiturn_calls(state: mt.EntryState, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _should_defer_for_missing_requested_tool(state, _current_turn_user_text_raw(state)):
        return []
    repaired = repair_filesystem_call_arguments(state, calls)
    repaired = _drop_redundant_existing_mkdirs(state, repaired)
    repaired = _drop_unrequested_touch_echo_writes(state, repaired)
    repaired = _repair_echo_content_from_quoted_request(state, repaired)
    repaired = _repair_folder_rename_sequence(state, repaired)
    repaired = _repair_reads_after_recent_file_move(state, repaired)
    repaired = _repair_text_args_from_prior_result(state, repaired)
    current_text = _effective_user_text_raw(state).lower()
    if re.search(r"\bhidden\b|\bvisible and hidden\b|\ball\b.{0,20}\bcontents?\b", current_text):
        for call in repaired:
            if str(call.get("name") or "").lower() == "ls":
                args = mt.selector.maybe_json(call.get("arguments") or {})
                if isinstance(args, dict):
                    args["a"] = True
                    call["arguments"] = args
    if len(repaired) == 1 and str(repaired[0].get("name") or "").lower() == "find":
        args = mt.selector.maybe_json(repaired[0].get("arguments") or {})
        if (
            isinstance(args, dict)
            and isinstance(args.get("path"), str)
            and args.get("path") not in {"", "."}
            and isinstance(args.get("name"), str)
            and re.search(r"\.[A-Za-z0-9]{1,8}$", args["name"])
            and not re.search(r"\b(?:find|search|locate|gather|list)\b", current_text)
            and _tool_available(state, "cd")
        ):
            folder = args["path"].strip("./").split("/")[-1]
            if folder:
                repaired = [{"name": "cd", "arguments": {"folder": folder}}]
    names = [str(call.get("name") or "").lower() for call in repaired if isinstance(call, dict)]
    if "pwd" in names and "ls" in names:
        repaired = [call for call in repaired if str(call.get("name") or "").lower() != "pwd"]
    return repaired


def _tool_available(state: mt.EntryState, name: str) -> bool:
    wanted = name.lower()
    return any(str(tool.get("name") or "").lower() == wanted for tool in state.available_tools)


def _current_turn_has_new_functions(state: mt.EntryState) -> bool:
    missed = state.record.get("missed_function") if isinstance(state.record, dict) else {}
    return isinstance(missed, dict) and str(getattr(state, "current_turn_index", "")) in missed


def _additional_function_prompt(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip().lower()
    if not stripped:
        return True
    default = str(getattr(mt, "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC", "") or "")
    default = re.sub(r"\s+", " ", default).strip().lower()
    if default and stripped == default:
        return True
    return bool(re.search(r"\b(?:additional|new|missing)\s+function\b|\bfunction\s+(?:is|was)\s+now\s+available\b", stripped))


def _previous_real_user_text(state: mt.EntryState) -> str:
    for message in reversed(state.history[: state.current_turn_start]):
        if str(message.get("role", "user")).lower() != "user":
            continue
        content = str(message.get("content") or "")
        if content.strip() and not _additional_function_prompt(content):
            return content
    return ""


def _requested_tool_names(text: str) -> set[str]:
    lowered = text.lower()
    requested: set[str] = set()
    patterns = {
        "sort": r"\b(?:sort|sorting)\b|\blines?\s+sorted\b|\bcluttered\b",
        "mv": r"\b(?:move|archive|transfer|relocate)\b",
        "cp": r"\b(?:copy|duplicate|backup|secure)\b",
        "grep": r"\b(?:grep|search|occurrence|occurrences|keyword|investigate|identify sections?)\b",
        "diff": r"\b(?:diff|compare|juxtapose|differences?|alterations?)\b",
        "tail": r"\b(?:last|tail)\s+\d+\s+lines?\b",
        "cat": r"\b(?:cat|view|peek|show|display)\b.{0,50}\b(?:content|contents|file)\b",
        "post_tweet": r"\b(?:social media|tweet|post|share)\b",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lowered):
            requested.add(name)
    return requested


def _should_defer_for_missing_requested_tool(state: mt.EntryState, raw_text: str) -> bool:
    if not raw_text.strip() or _additional_function_prompt(raw_text):
        return False
    requested = _requested_tool_names(raw_text)
    if not requested:
        return False
    missing = {name for name in requested if not _tool_available(state, name)}
    return bool(missing)


def _effective_user_text_raw(state: mt.EntryState) -> str:
    current = _current_turn_user_text_raw(state)
    if not _current_turn_has_new_functions(state) or not _additional_function_prompt(current):
        return current
    previous = _previous_real_user_text(state)
    if not previous:
        return current
    requested = _requested_tool_names(previous)
    if requested and any(_tool_available(state, name) for name in requested):
        return previous
    return current


def _mentioned_fs_entry(state: mt.EntryState, text: str, kind: str) -> tuple[str, str] | None:
    wanted = {kind} if kind != "any" else {"file", "directory"}
    lowered = text.lower()
    matches: list[tuple[int, str, str]] = []
    for path, name, entry_kind in _fs_paths_from_initial_config(state.record.get("initial_config", {})):
        if entry_kind not in wanted:
            continue
        aliases = sorted(_fs_aliases(name), key=len, reverse=True)
        for alias in aliases:
            if not alias or len(alias) < 2:
                continue
            match = re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered)
            if match:
                matches.append((match.start(), path, name))
                break
    if not matches:
        return None
    _pos, path, name = sorted(matches, key=lambda item: (item[0], len(item[2])))[0]
    return path, name


def _mentioned_directory_names(state: mt.EntryState, raw_text: str) -> list[tuple[int, str]]:
    lowered = raw_text.lower()
    matches: list[tuple[int, str]] = []
    for name in _known_directory_names(state):
        aliases = sorted(_fs_aliases(name), key=len, reverse=True)
        for alias in aliases:
            if not alias or len(alias) < 2:
                continue
            match = re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", lowered)
            if match:
                matches.append((match.start(), name))
                break
    return sorted(matches, key=lambda item: item[0])


def _action_destination_directory(state: mt.EntryState, raw_text: str) -> str:
    mentions = _mentioned_directory_names(state, raw_text)
    if not mentions:
        return ""
    lowered = raw_text.lower()
    if len(mentions) == 1:
        return mentions[0][1]
    action_pos = -1
    action_match = re.search(r"\b(?:copy|duplicate|move|transfer|archive|place|put)\b", lowered)
    if action_match:
        action_pos = action_match.start()
    after_action = [(pos, name) for pos, name in mentions if pos >= action_pos]
    if after_action:
        return after_action[-1][1]
    return mentions[-1][1]


def _parent_dir(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    return parts[-2] if len(parts) >= 2 else ""


def _current_working_directory(state: mt.EntryState) -> str:
    for message in reversed(state.history):
        if str(message.get("role", "")).lower() != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        cwd = payload.get("current_working_directory") if isinstance(payload, dict) else None
        if isinstance(cwd, str) and cwd:
            return cwd.split("/")[-1]
    return ""


def _current_turn_user_text_raw(state: mt.EntryState) -> str:
    parts = []
    for message in state.history[state.current_turn_start :]:
        if str(message.get("role", "user")).lower() == "user":
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


def _explicit_content_write_request(text: str) -> bool:
    lowered = text.lower()
    if _quoted_strings(text) and re.search(r"\b(?:write|jot|put|add|store|append|echo|say|containing|content)\b", lowered):
        return True
    return bool(
        re.search(
            r"\b(?:write|jot|put|add|store|append|echo)\b.{0,80}\b(?:into|to|in|file|document)\b",
            lowered,
        )
    )


def _filename_literals(text: str) -> list[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9]{1,8}\b", text)


def _quoted_strings(text: str) -> list[str]:
    return [left or right for left, right in re.findall(r"'([^']+)'|\"([^\"]+)\"", text) if left or right]


def _latest_json_tool_value(state: mt.EntryState, key: str) -> Any:
    for message in reversed(state.history):
        if str(message.get("role", "")).lower() != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    return None


def _repair_echo_content_from_quoted_request(
    state: mt.EntryState,
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = _effective_user_text_raw(state)
    if not re.search(r"\b(?:write|jot|put|add|store|append|echo)\b", text, re.I):
        return calls
    quoted = [
        item
        for item in _quoted_strings(text)
        if not re.search(r"\.[A-Za-z0-9]{1,8}$", item)
    ]
    if not quoted:
        return calls
    content = max(quoted, key=len)
    repaired: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = copy.deepcopy(call)
        if str(item.get("name") or "").lower() != "echo":
            repaired.append(item)
            continue
        args = mt.selector.maybe_json(item.get("arguments") or {})
        if isinstance(args, dict) and isinstance(args.get("content"), str) and args["content"] != content:
            args["content"] = content
            item["arguments"] = args
        repaired.append(item)
    return repaired


def _latest_referenced_text_result(state: mt.EntryState, text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bsorted\b.{0,40}\b(?:result|output|content|report|file|body)\b", lowered):
        value = _latest_json_tool_value(state, "sorted_content")
        if isinstance(value, str) and value:
            return value
    if re.search(r"\b(?:result|output|content|contents|body|message)\b", lowered):
        for key in ("sorted_content", "file_content", "content", "result"):
            value = _latest_json_tool_value(state, key)
            if isinstance(value, str) and value:
                return value
    return ""


def _repair_text_args_from_prior_result(
    state: mt.EntryState,
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = _current_turn_user_text_raw(state)
    result_text = _latest_referenced_text_result(state, text)
    if not result_text:
        return calls
    text_arg_names = ("content", "body", "message", "text")
    repaired: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = copy.deepcopy(call)
        args = mt.selector.maybe_json(item.get("arguments") or {})
        if not isinstance(args, dict):
            repaired.append(item)
            continue
        changed = False
        for arg_name in text_arg_names:
            if arg_name in args and isinstance(args.get(arg_name), str):
                if args[arg_name] != result_text:
                    args[arg_name] = result_text
                    changed = True
                break
        if changed:
            item["arguments"] = args
        repaired.append(item)
    return repaired


def _latest_find_matches(state: mt.EntryState) -> list[str]:
    matches = _latest_json_tool_value(state, "matches")
    if not isinstance(matches, list):
        return []
    return [str(match) for match in matches if isinstance(match, str)]


def _recent_file_from_history(state: mt.EntryState) -> str:
    for batch in reversed(state.executed_call_history):
        for call_text in reversed(batch):
            for call in reversed(mt.parse_python_calls(f"[{call_text}]")):
                args = mt.selector.maybe_json(call.get("arguments") or {})
                if not isinstance(args, dict):
                    continue
                for key in ("file_name", "source", "destination"):
                    value = args.get(key)
                    if isinstance(value, str) and re.search(r"\.[A-Za-z0-9]{1,8}$", value):
                        return value
    return ""


def _cd_calls_for_parent(parent_path: str, current_dir: str) -> list[dict[str, Any]]:
    parts = [part for part in parent_path.strip("./").split("/") if part and part != "."]
    if not parts:
        return []
    if current_dir and parts and current_dir.lower() == parts[-1].lower():
        return []
    return [{"name": "cd", "arguments": {"folder": part}} for part in parts]


def _copy_prior_find_matches_candidate(state: mt.EntryState, text: str) -> list[dict[str, Any]]:
    if not (_tool_available(state, "cp") and re.search(r"\b(?:copy|duplicate|safely copied|backup)\b", text)):
        return []
    matches = _latest_find_matches(state)
    file_paths = [match.strip("./") for match in matches if re.search(r"\.[A-Za-z0-9]{1,8}$", match)]
    if not file_paths:
        return []
    destination = _action_destination_directory(state, _current_turn_user_text_raw(state))
    if not destination:
        dir_entry = _mentioned_fs_entry(state, text, "directory")
        destination = dir_entry[1] if dir_entry else ""
    if not destination:
        return []
    parent_counts: dict[str, int] = {}
    for path in file_paths:
        parent = "/".join(path.split("/")[:-1])
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    parent = max(parent_counts, key=parent_counts.get)
    calls = _cd_calls_for_parent(parent, _current_working_directory(state))
    for path in file_paths:
        if "/".join(path.split("/")[:-1]) == parent:
            calls.append({"name": "cp", "arguments": {"source": path.split("/")[-1], "destination": destination}})
    return calls


def _copy_directory_files_candidate(state: mt.EntryState, text: str) -> list[dict[str, Any]]:
    if not (_tool_available(state, "cp") and re.search(r"\b(?:copy|duplicate|transfer)\b", text)):
        return []
    if not re.search(r"\ball\b.{0,40}\bfiles?\b", text):
        return []
    dir_entry = _mentioned_fs_entry(state, text, "directory")
    if dir_entry is None:
        return []
    directory = dir_entry[1]
    files = _direct_child_files(state.record.get("initial_config", {}), directory)
    if re.search(r"\btext\s+files?\b", text):
        files = [name for name in files if name.lower().endswith(".txt")]
    if not files:
        return []
    destination = ""
    quoted = _quoted_strings(_current_turn_user_text_raw(state))
    for item in reversed(quoted):
        if item != directory and not re.search(r"\.[A-Za-z0-9]{1,8}$", item):
            destination = item
            break
    if not destination:
        dest_entry = _mentioned_fs_entry(state, text, "directory")
        destination = dest_entry[1] if dest_entry else ""
    if not destination:
        return []
    calls: list[dict[str, Any]] = []
    if _tool_available(state, "mkdir") and not _directory_exists(state, destination):
        calls.append({"name": "mkdir", "arguments": {"dir_name": destination}})
    for file_name in files:
        calls.append({"name": "cp", "arguments": {"source": file_name, "destination": destination}})
    return calls


def _drop_redundant_existing_mkdirs(state: mt.EntryState, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = {name.lower() for name in _known_directory_names(state)}
    kept: list[dict[str, Any]] = []
    for call in calls:
        if str(call.get("name") or "").lower() != "mkdir":
            kept.append(call)
            continue
        args = mt.selector.maybe_json(call.get("arguments") or {})
        dir_name = args.get("dir_name") if isinstance(args, dict) else None
        resolved = _resolve_known_directory_name(state, dir_name)
        if resolved and resolved.lower() in existing:
            continue
        if isinstance(dir_name, str) and dir_name:
            existing.add(dir_name.lower())
        kept.append(call)
    return kept


def _drop_unrequested_touch_echo_writes(state: mt.EntryState, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _current_turn_user_text_raw(state)
    if _explicit_content_write_request(text):
        return calls
    touched = set()
    for call in calls:
        if str(call.get("name") or "").lower() != "touch":
            continue
        args = mt.selector.maybe_json(call.get("arguments") or {})
        if isinstance(args, dict) and isinstance(args.get("file_name"), str):
            touched.add(args["file_name"].lower())
    if not touched:
        return calls
    kept = []
    for call in calls:
        if str(call.get("name") or "").lower() == "echo":
            args = mt.selector.maybe_json(call.get("arguments") or {})
            if isinstance(args, dict) and str(args.get("file_name") or "").lower() in touched:
                continue
        kept.append(call)
    return kept


def _repair_folder_rename_sequence(state: mt.EntryState, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(calls) != 2 or not (_tool_available(state, "cd") and _tool_available(state, "mv")):
        return calls
    first, second = calls
    first_name = str(first.get("name") or "").lower()
    second_name = str(second.get("name") or "").lower()
    if first_name != "cd" or second_name not in {"cp", "mv"}:
        return calls
    text = _current_turn_user_text_raw(state).lower()
    if not re.search(r"\b(?:rename|renamed|name it|under the name|as)\b", text):
        return calls
    if second_name == "cp" and not re.search(r"\b(?:copy|duplicate|transfer)\b", text):
        return calls
    first_args = mt.selector.maybe_json(first.get("arguments") or {})
    second_args = mt.selector.maybe_json(second.get("arguments") or {})
    if not isinstance(first_args, dict) or not isinstance(second_args, dict):
        return calls
    folder = _resolve_known_directory_name(state, first_args.get("folder"))
    source = second_args.get("source")
    destination = second_args.get("destination")
    if not (folder and isinstance(source, str) and isinstance(destination, str)):
        return calls
    if "/" in source or "/" in destination or not re.search(r"\.[A-Za-z0-9]{1,8}$", destination):
        return calls
    return [
        {"name": second_name, "arguments": {"source": source, "destination": folder}},
        {"name": "cd", "arguments": {"folder": folder}},
        {"name": "mv", "arguments": {"source": source.split("/")[-1], "destination": destination}},
    ]


def _repair_reads_after_recent_file_move(state: mt.EntryState, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not calls or not _tool_available(state, "cd"):
        return calls
    first = calls[0]
    if str(first.get("name") or "").lower() not in {"grep", "cat", "tail", "sort", "wc", "diff"}:
        return calls
    args = mt.selector.maybe_json(first.get("arguments") or {})
    if not isinstance(args, dict):
        return calls
    names = [value for key, value in args.items() if "file" in key and isinstance(value, str)]
    if not names:
        return calls
    for batch in reversed(state.executed_call_history):
        for call_text in reversed(batch):
            for prior in reversed(mt.parse_python_calls(f"[{call_text}]")):
                if str(prior.get("name") or "").lower() not in {"mv", "cp"}:
                    continue
                prior_args = mt.selector.maybe_json(prior.get("arguments") or {})
                if not isinstance(prior_args, dict):
                    continue
                source = prior_args.get("source")
                destination = prior_args.get("destination")
                if not isinstance(source, str) or not isinstance(destination, str):
                    continue
                if source.split("/")[-1] not in names:
                    continue
                folder = _resolve_known_directory_name(state, destination)
                if folder and folder.lower() != _current_working_directory(state).lower():
                    return [{"name": "cd", "arguments": {"folder": folder}}] + calls
    return calls


def goal_graph_controller_candidate(state: mt.EntryState) -> dict[str, Any] | None:
    raw_text = _effective_user_text_raw(state)
    text = raw_text.lower()
    calls: list[dict[str, Any]] = []

    file_entry = _mentioned_fs_entry(state, text, "file")
    dir_entry = _mentioned_fs_entry(state, text, "directory")
    action_dir = _action_destination_directory(state, raw_text)
    file_literals = _filename_literals(raw_text)
    new_names = [name for name in file_literals if file_entry is None or name.lower() != file_entry[1].lower()]
    source_file = file_entry[1] if file_entry else (file_literals[0] if file_literals else "")

    if not calls:
        calls = _copy_prior_find_matches_candidate(state, text)

    if not calls:
        calls = _copy_directory_files_candidate(state, text)

    if not calls and source_file and (action_dir or dir_entry) and _tool_available(state, "cp") and re.search(r"\b(?:copy|duplicate)\b", text):
        destination_dir = action_dir or (dir_entry[1] if dir_entry else "")
        parent = _parent_dir(file_entry[0]) if file_entry else ""
        if parent and parent.lower() != _current_working_directory(state).lower() and _tool_available(state, "cd"):
            calls.append({"name": "cd", "arguments": {"folder": parent}})
        calls.append({"name": "cp", "arguments": {"source": source_file, "destination": destination_dir}})
        if new_names and _tool_available(state, "mv") and _tool_available(state, "cd"):
            calls.append({"name": "cd", "arguments": {"folder": destination_dir}})
            calls.append({"name": "mv", "arguments": {"source": source_file, "destination": new_names[-1]}})

    if not calls and source_file and (action_dir or dir_entry) and _tool_available(state, "mv") and re.search(
        r"\b(?:move|archive|transfer|rename|safekeeping)\b",
        text,
    ):
        destination_dir = action_dir or (dir_entry[1] if dir_entry else "")
        parent = _parent_dir(file_entry[0]) if file_entry else ""
        if parent and parent.lower() != _current_working_directory(state).lower() and _tool_available(state, "cd"):
            calls.append({"name": "cd", "arguments": {"folder": parent}})
        calls.append({"name": "mv", "arguments": {"source": source_file, "destination": destination_dir}})
        if new_names and _tool_available(state, "cd"):
            calls.append({"name": "cd", "arguments": {"folder": destination_dir}})
            calls.append({"name": "mv", "arguments": {"source": source_file, "destination": new_names[-1]}})

    if not calls and _tool_available(state, "echo") and re.search(r"\b(?:jot|write|put|add|store)\b", text):
        quoted = _quoted_strings(raw_text)
        content = max((item for item in quoted if not re.search(r"\.[A-Za-z0-9]{1,8}$", item)), key=len, default="")
        target = source_file or _recent_file_from_history(state)
        if content and target:
            calls = [{"name": "echo", "arguments": {"content": content, "file_name": target}}]

    if not calls and _tool_available(state, "echo") and re.search(r"\bstore\b.+\b(?:number|count|words?)\b", text):
        target = file_literals[-1] if file_literals else ""
        count = _latest_json_tool_value(state, "count")
        directory = dir_entry[1] if dir_entry else ""
        if target and count is not None:
            if directory and _tool_available(state, "cd") and directory.lower() != _current_working_directory(state).lower():
                calls.extend([{"name": "cd", "arguments": {"folder": ".."}}, {"name": "cd", "arguments": {"folder": directory}}])
            if _tool_available(state, "touch"):
                calls.append({"name": "touch", "arguments": {"file_name": target}})
            calls.append({"name": "echo", "arguments": {"content": str(count), "file_name": target}})

    if not calls and source_file and re.search(r"\b(?:display|show|view|peek|contents?)\b", text):
        if _tool_available(state, "cat"):
            calls.append({"name": "cat", "arguments": {"file_name": source_file}})
        if _tool_available(state, "sort") and "sort" in text:
            calls.append({"name": "sort", "arguments": {"file_name": source_file}})

    if not calls and source_file and _tool_available(state, "sort") and "sort" in text:
        calls = [{"name": "sort", "arguments": {"file_name": source_file}}]

    if not calls and _tool_available(state, "find") and re.search(r"\b(?:find|search|locate)\b", text):
        keyword = ""
        for quoted in _quoted_strings(raw_text):
            keyword = quoted
            break
        if not keyword:
            match = re.search(r"\b(?:named?|called|matching|for)\s+([a-z0-9_.-]+)", text)
            keyword = match.group(1) if match else ""
        if keyword:
            calls = [{"name": "find", "arguments": {"path": ".", "name": keyword.strip("*")}}]

    if not calls and _tool_available(state, "find") and re.search(r"\b(?:has|with|containing)\s+['\"]?([a-z0-9_.-]+)['\"]?\s+in\s+(?:its|their|the)\s+file\s+name\b", text):
        keyword = re.search(r"\b(?:has|with|containing)\s+['\"]?([a-z0-9_.-]+)['\"]?\s+in\s+(?:its|their|the)\s+file\s+name\b", text)
        if keyword:
            calls = [{"name": "find", "arguments": {"path": ".", "name": keyword.group(1).strip("*")}}]

    if not calls and _tool_available(state, "ls") and re.search(
        r"\b(?:list|show|display)\b.{0,40}\b(?:files?|contents?|directory)\b",
        text,
    ) and not source_file and "file name" not in text:
        calls = [{"name": "ls", "arguments": {"a": True}}]

    if not calls and _tool_available(state, "cd") and source_file and dir_entry and re.search(r"\bin\s+(?:the\s+)?['\"]?[a-z0-9_. -]+['\"]?\s+folder\b", text):
        calls = [{"name": "cd", "arguments": {"folder": dir_entry[1]}}]

    if not calls and _tool_available(state, "cd") and re.search(r"\b(?:cd|go|navigate|switch|open|enter)\b", text):
        directory = _mentioned_fs_entry(state, text, "directory")
        if directory is not None:
            if source_file and _tool_available(state, "touch") and re.search(r"\b(?:create|draft|document titled|file creation)\b", text):
                calls = [
                    {"name": "cd", "arguments": {"folder": directory[1]}},
                    {"name": "touch", "arguments": {"file_name": source_file}},
                ]
            else:
                calls = [{"name": "cd", "arguments": {"folder": directory[1]}}]

    if not calls:
        return None
    return {
        "source": "goal_graph_controller",
        "calls": calls,
        "issues": mt.selector.call_issues(calls, state.available_tools),
        "error": "",
        "latency_ms": 0,
        "raw_text": "goal_graph_controller",
        "generated_tokens": 0,
    }


def goal_graph_calls(result: dict[str, Any], tools: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    calls = []
    for call in result.get("calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("tool_name") or call.get("name")
        if not name:
            continue
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        calls.append({"name": str(name), "arguments": arguments})
    calls = mt.dedupe_tool_calls(calls)
    return mt.selector.semantic_repair_calls(calls, tools, prompt)


def generate_goal_graph_step(
    model: Any,
    tokenizer: Any,
    state: mt.EntryState,
    step_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = mt.render_prompt(state, step_index)
    tools = normalized_tools_for_state(state)
    start = time.time()
    try:
        result = plan_and_compile_goal_graph(
            model,
            tokenizer,
            generate_text,
            prompt,
            tools,
            max_new_tokens=args.max_new_tokens,
            repair_attempts=args.repair_attempts,
            allow_side_effects=not args.strict_read_only,
            use_binder_fallback=not args.no_binder_fallback,
            planner_mode=args.planner_mode,
        )
        calls = goal_graph_calls(result, tools, prompt)
        calls = normalize_multiturn_calls(state, calls)
        error = ""
    except Exception as exc:  # noqa: BLE001
        result = {
            "planner_mode": args.planner_mode,
            "calls": [],
            "verification_ok": False,
            "diagnostic_codes": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        calls = []
        error = result["error"]
    return {
        "source": "goal_graph",
        "calls": calls,
        "issues": mt.selector.call_issues(calls, tools),
        "error": error,
        "latency_ms": round((time.time() - start) * 1000, 3),
        "raw_text": mt.trim_text(result.get("raw_text", ""), args.raw_log_chars),
        "generated_tokens": 0,
        "goal_graph_result": result,
    }


def duplicate_guarded_calls(state: mt.EntryState, calls: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    duplicate_count = mt.duplicate_call_count(state, calls, mt.duplicate_penalty_exempt_tools(args))
    if duplicate_count and duplicate_count >= len(calls):
        return []
    return calls


def run_goal_graph_step(
    model: Any,
    tokenizer: Any,
    active_states: list[mt.EntryState],
    step_index: int,
    args: argparse.Namespace,
) -> int:
    continued = 0
    for state in active_states:
        prompt = mt.render_prompt(state, step_index)
        output = generate_goal_graph_step(model, tokenizer, state, step_index, args)
        output = repair_filesystem_candidate(state, output)
        candidates = [mt.empty_turn_candidate(), output]
        local_controller = goal_graph_controller_candidate(state)
        if local_controller:
            candidates.append(repair_filesystem_candidate(state, local_controller))
        controller = mt.controller_candidate(state)
        if controller:
            candidates.append(repair_filesystem_candidate(state, controller))
        selected, selection_diag = mt.select_multiturn_candidate(
            candidates,
            state.available_tools,
            prompt,
            None,
            state,
            step_index,
            args,
        )
        selected = normalize_multiturn_calls(state, selected)
        selected = duplicate_guarded_calls(state, selected, args)
        response = mt.calls_to_response(selected)
        decoded = mt.decode_response(response)
        empty = mt.is_empty_execute_response(decoded)
        state.current_turn_responses.append(response if not empty else "[]")

        step_log = {
            "turn": state.current_turn_index,
            "step": step_index,
            "selected_source": selection_diag.get("selected_source", "goal_graph"),
            "selected_calls": selected,
            "decoded": decoded,
            "empty": empty,
            "goal_graph": output.get("goal_graph_result", {}),
            "candidates": candidates,
            "selection_diag": selection_diag,
            "issues": output.get("issues", []),
            "error": output.get("error", ""),
            "latency_ms": output.get("latency_ms", 0),
        }

        if empty:
            state.active = False
            state.logs.append(step_log)
            print(
                json.dumps(
                    {
                        "id": state.rid,
                        "turn": state.current_turn_index,
                        "step": step_index,
                        "selected": selection_diag.get("selected_source", "goal_graph"),
                        "decoded_calls": 0,
                        "continue": False,
                        "error": output.get("error", ""),
                    }
                ),
                flush=True,
            )
            continue

        execution_results = mt.execute_selected_calls(state, response, decoded, args)
        step_log["execution_results"] = execution_results
        state.logs.append(step_log)
        if should_stop_goal_graph_turn(state, decoded, execution_results, args):
            state.active = False
        else:
            continued += 1
        print(
            json.dumps(
                {
                    "id": state.rid,
                    "turn": state.current_turn_index,
                    "step": step_index,
                    "selected": selection_diag.get("selected_source", "goal_graph"),
                    "decoded_calls": len(decoded),
                    "continue": state.active,
                    "verification_ok": bool((output.get("goal_graph_result") or {}).get("verification_ok")),
                    "diagnostic_codes": (output.get("goal_graph_result") or {}).get("diagnostic_codes") or [],
                }
            ),
            flush=True,
        )
    return continued


def should_stop_goal_graph_turn(
    state: mt.EntryState,
    decoded: list[str],
    execution_results: list[str],
    args: argparse.Namespace,
) -> bool:
    if mt.should_stop_after_success(state, decoded, execution_results, args):
        return True
    if not args.stop_after_multicall_success:
        return False
    if any(mt.execution_result_has_error(result) for result in execution_results):
        return False
    names = mt.decoded_call_names(decoded)
    text = mt.current_turn_user_text(state)
    if names & {"post_tweet", "comment", "echo", "touch", "mkdir", "mv", "cp", "rm"}:
        return True
    if names == {"cd"} and not re.search(
        r"\b(?:move|copy|archive|transfer|rename|create|touch|write|jot|put|store|sort|grep|diff|tail|count|tweet|post|comment)\b",
        text,
    ):
        return True
    if "ls" in names and re.search(r"\b(?:list|show|display)\b.{0,40}\b(?:files?|contents?|directory)\b", text):
        return True
    if "grep" in names and re.search(r"\b(?:search|grep|occurrence|occurrences|find|contains?|investigate|keyword|anomal)\b", text):
        return any("[]" not in str(result) for result in execution_results)
    if "find" in names and re.search(r"\b(?:find|search|locate|gather|list).{0,80}\b(?:name|file|project|document)\b", text):
        return any('"matches": [' in str(result) and '"matches": []' not in str(result) for result in execution_results)
    if "cat" in names and re.search(r"\b(?:content|contents|view|show|display|output|peek|terminal)\b", text):
        return any("file_content" in str(result) for result in execution_results)
    return False


def run_goal_graph_inference(states: list[mt.EntryState], model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    mt.init_backend_state(states, args)
    max_turns = max((len(state.record.get("question", [])) for state in states), default=0)
    for turn_index in range(max_turns):
        active_states = []
        for state in states:
            if turn_index >= len(state.record.get("question", [])):
                continue
            mt.start_turn(state, turn_index)
            active_states.append(state)

        for step_index in range(args.max_steps_per_turn):
            still_active = [state for state in active_states if state.active]
            if not still_active:
                break
            continued = run_goal_graph_step(model, tokenizer, still_active, step_index, args)
            if continued == 0:
                break

        for state in active_states:
            if state.active:
                state.logs.append(
                    {
                        "turn": turn_index,
                        "step": args.max_steps_per_turn,
                        "empty": True,
                        "forced_stop": True,
                    }
                )
                state.active = False
            mt.finish_turn(state)


def write_predictions(states: list[mt.EntryState], output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for state in states:
            row = {
                "id": state.rid,
                "result": state.result,
                "model": "goal_graph_runtime",
                "category": args.category,
                "planner_mode": args.planner_mode,
            }
            if args.include_logs:
                row["inference_log"] = state.logs
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--repair-attempts", type=int, default=0)
    parser.add_argument("--planner-mode", choices=["stepwise", "one_shot"], default="stepwise")
    parser.add_argument("--max-steps-per-turn", type=int, default=4)
    parser.add_argument("--strict-read-only", action="store_true")
    parser.add_argument("--no-binder-fallback", action="store_true")
    parser.add_argument("--execution-error-penalty", type=float, default=35.0)
    parser.add_argument("--execution-success-bonus", type=float, default=2.0)
    parser.add_argument("--duplicate-call-penalty", type=float, default=45.0)
    parser.add_argument("--duplicate-penalty-exempt-tools", default="cd")
    parser.add_argument("--empty-turn-action-penalty", type=float, default=12.0)
    parser.add_argument("--no-execution-aware-selection", dest="execution_aware_selection", action="store_false")
    parser.set_defaults(execution_aware_selection=True)
    parser.add_argument("--stop-after-multicall-success", dest="stop_after_multicall_success", action="store_true")
    parser.add_argument("--no-stop-after-multicall-success", dest="stop_after_multicall_success", action="store_false")
    parser.set_defaults(stop_after_multicall_success=True)
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--raw-log-chars", type=int, default=1200)
    parser.add_argument("--generation-state-name", default="bfcl_mt_goal_graph_generation")
    parser.add_argument("--scoring-model-name", default="bfcl_mt_goal_graph_score")
    parser.add_argument("--sources", default="goal_graph_runtime")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    states, category = mt.load_entries(args)
    print(
        json.dumps(
            {
                "event": "start",
                "category": category,
                "rows": len(states),
                "model": args.model,
                "planner_mode": args.planner_mode,
                "max_steps_per_turn": args.max_steps_per_turn,
            },
            indent=2,
        ),
        flush=True,
    )
    model, tokenizer = load_model(args.model)
    try:
        run_goal_graph_inference(states, model, tokenizer, args)
    finally:
        unload_model(model, tokenizer)
    write_predictions(states, Path(args.output), args)
    scored = mt.score_predictions(states, Path(args.score_output), args)
    summary = scored.get("summary") or {}
    summary["model"] = "goal_graph_runtime"
    Path(args.score_output).write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
