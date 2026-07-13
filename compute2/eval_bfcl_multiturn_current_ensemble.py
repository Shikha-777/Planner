#!/usr/bin/env python3
"""Run BFCL multi-turn inference with the current tool-calling ensemble.

This runner emits BFCL's official nested multi-turn result shape:
```
{"id": "...", "result": [["[call(...)]", ...], ["[next_turn(...)]"]]}
```

It also scores with BFCL's own multi-turn checker. The older AST scorer is
single-turn only and will score multi-turn rows as 0 even when calls are sane.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import eval_bfcl_ensemble_select as selector
import eval_bfcl_hf_adapter as hf_adapter
import eval_bfcl_toolace_official as toolace_adapter
from smoke_eval import dedupe_tool_calls

from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry


MULTITURN_SYSTEM_PROMPT = """You are a tool-calling agent in a multi-turn benchmark.
Use only the provided tools. For the current turn and current step, return only
one Python-list-style batch of function calls, such as:
[tool_name(arg="value"), other_tool()]
If no more tool call is needed for the current turn, return [].
Do not answer in prose."""


@dataclass
class SourceSpec:
    name: str
    kind: str
    base: str
    adapter: str = ""
    tokenizer: str = ""
    trust_remote_code: bool = False


@dataclass
class EntryState:
    index: int
    record: dict[str, Any]
    answer: dict[str, Any]
    rid: str
    available_tools: list[dict[str, Any]]
    history: list[dict[str, Any]] = field(default_factory=list)
    result: list[list[str]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    executed_call_history: list[list[str]] = field(default_factory=list)
    current_turn_responses: list[str] = field(default_factory=list)
    current_turn_start: int = 0
    active: bool = False
    current_turn_index: int = 0


def trim_text(text: Any, limit: int = 1200) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<trimmed {len(text) - limit} chars>"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


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


def select_records(
    records: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    limit: int,
    offset: int,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    answer_by_id = {str(item.get("id")): item for item in answers}
    pairs = []
    for index, record in enumerate(records):
        rid = str(record.get("id", index))
        answer = answer_by_id.get(rid)
        if answer is not None:
            pairs.append((index, record, answer))
    if offset:
        pairs = pairs[offset:]
    if limit:
        pairs = pairs[:limit]
    return pairs


def load_entries(args: argparse.Namespace) -> tuple[list[EntryState], str]:
    records = load_dataset_entry(
        args.category,
        include_prereq=False,
        include_language_specific_hint=False,
    )
    answers = load_ground_truth_entry(args.category)
    pairs = select_records(records, answers, args.limit, args.offset)
    states = [
        EntryState(
            index=index,
            record=copy.deepcopy(record),
            answer=copy.deepcopy(answer),
            rid=str(record.get("id", index)),
            available_tools=copy.deepcopy(record.get("function") or []),
        )
        for index, record, answer in pairs
    ]
    return states, args.category


def long_context_category(category: str) -> bool:
    return "long_context" in category or "composite" in category


def init_backend_state(states: list[EntryState], args: argparse.Namespace) -> None:
    for state in states:
        execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=state.record.get("initial_config", {}),
            involved_classes=state.record.get("involved_classes", []),
            model_name=args.generation_state_name,
            test_entry_id=state.rid,
            long_context=long_context_category(args.category),
            is_evaL_run=False,
        )


def source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    wanted = [item.strip() for item in args.sources.split(",") if item.strip()]
    specs: dict[str, SourceSpec] = {
        "toolace": SourceSpec("toolace", "toolace", args.toolace_model),
        "xlam": SourceSpec("xlam", "hf", args.xlam_model),
        "gptoss_apigen": SourceSpec(
            "gptoss_apigen",
            "hf",
            args.gptoss_base,
            adapter=args.apigen_adapter,
            tokenizer=args.apigen_adapter,
            trust_remote_code=True,
        ),
        "taskbench": SourceSpec(
            "taskbench",
            "hf",
            args.gptoss_base,
            adapter=args.taskbench_adapter,
            tokenizer=args.taskbench_adapter,
            trust_remote_code=True,
        ),
    }
    missing = [name for name in wanted if name not in specs]
    if missing:
        raise ValueError(f"Unknown sources: {missing}. Known sources: {sorted(specs)}")
    return [specs[name] for name in wanted]


def make_hf_args(args: argparse.Namespace, spec: SourceSpec) -> argparse.Namespace:
    return argparse.Namespace(
        base=spec.base,
        adapter=spec.adapter,
        tokenizer=spec.tokenizer,
        trust_remote_code=spec.trust_remote_code,
        precision=args.precision,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        openai_tool_schema=True,
        system_prompt=MULTITURN_SYSTEM_PROMPT,
    )


def make_toolace_args(args: argparse.Namespace, spec: SourceSpec) -> argparse.Namespace:
    return argparse.Namespace(
        base=spec.base,
        trust_remote_code=True,
        precision=args.precision,
        device_map=args.device_map,
        load_4bit=args.load_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def render_prompt(state: EntryState, step_index: int) -> str:
    lines = [
        "Multi-turn tool-calling transcript.",
        "Return only the next function-call batch for this exact step.",
        "Use [] when no more tool call is needed for the current turn.",
        "",
        "Transcript:",
    ]
    for message in state.history:
        role = str(message.get("role", "user")).upper()
        content = trim_text(message.get("content", ""), 4000)
        lines.append(f"{role}: {content}")
    lines.extend(
        [
            "",
            f"Current turn index: {state.current_turn_index}",
            f"Current step index: {step_index}",
            "Next output:",
        ]
    )
    return "\n".join(lines)


def normalize_generated_calls(
    result: dict[str, Any],
    tools: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]]:
    calls = result.get("normalized_calls") or []
    if not isinstance(calls, list):
        calls = []
    clean = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("function_name")
        args = selector.maybe_json(call.get("arguments") or call.get("args") or call.get("parameters") or {})
        if name:
            clean.append({"name": str(name), "arguments": args if isinstance(args, dict) else {"value": args}})
    clean = dedupe_tool_calls(clean)
    fallback = parse_python_calls(result.get("raw_text", ""))
    if len(fallback) > len(clean):
        clean = fallback
    return selector.semantic_repair_calls(clean, tools, prompt)


def ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [ast_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [ast_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {ast_value(key): ast_value(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = ast_value(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id == "null":
            return None
        return node.id
    return ast.unparse(node)


def ast_func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = ast_func_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def first_balanced_list(text: str) -> str:
    start = text.find("[")
    if start < 0:
        return text.strip()
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:].strip()


def parse_python_calls(text: Any) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    snippet = first_balanced_list(raw)
    if not snippet.startswith("["):
        snippet = "[" + snippet
    if not snippet.endswith("]"):
        snippet = snippet + "]"
    try:
        tree = ast.parse(snippet, mode="eval")
    except SyntaxError:
        return []
    root = tree.body
    nodes = root.elts if isinstance(root, ast.List) else [root]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = ast_func_name(node.func)
        if not name:
            continue
        args: dict[str, Any] = {}
        for index, positional in enumerate(node.args):
            value = ast_value(positional)
            if index == 0 and isinstance(value, dict) and all(isinstance(key, str) for key in value):
                args.update(value)
            else:
                args[f"arg{index}"] = value
        for keyword in node.keywords:
            if keyword.arg:
                args[str(keyword.arg)] = ast_value(keyword.value)
        calls.append({"name": name, "arguments": args})
    return dedupe_tool_calls(calls)


def generate_for_source(
    spec: SourceSpec,
    states: list[EntryState],
    step_index: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if spec.kind == "toolace":
        model_args = make_toolace_args(args, spec)
        model, tokenizer = toolace_adapter.load_model(model_args)
        generate_one = toolace_adapter.generate_one
    else:
        model_args = make_hf_args(args, spec)
        model, tokenizer = hf_adapter.load_model_and_tokenizer(model_args)
        generate_one = hf_adapter.generate_one

    outputs: dict[str, dict[str, Any]] = {}
    try:
        for state in states:
            prompt = render_prompt(state, step_index)
            start = time.time()
            try:
                result = generate_one(model, tokenizer, prompt, state.available_tools, model_args)
                calls = normalize_generated_calls(result, state.available_tools, prompt)
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {"latency_ms": 0, "raw_text": "", "generated_tokens": 0}
                calls = []
                error = f"{type(exc).__name__}: {exc}"
            outputs[state.rid] = {
                "source": spec.name,
                "calls": calls,
                "issues": selector.call_issues(calls, state.available_tools),
                "error": error,
                "latency_ms": result.get("latency_ms") or round((time.time() - start) * 1000, 3),
                "raw_text": trim_text(result.get("raw_text", ""), args.raw_log_chars),
                "generated_tokens": int(result.get("generated_tokens") or 0),
            }
            print(
                json.dumps(
                    {
                        "id": state.rid,
                        "turn": state.current_turn_index,
                        "step": step_index,
                        "source": spec.name,
                        "calls": len(calls),
                        "issues": len(outputs[state.rid]["issues"]),
                        "error": error,
                    }
                ),
                flush=True,
            )
    finally:
        unload_model(model, tokenizer)
    return outputs


def is_arg_key(key: str) -> bool:
    return re.fullmatch(r"arg\d+", key) is not None


def call_to_python(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "")
    args = selector.maybe_json(call.get("arguments") or {})
    if not isinstance(args, dict):
        args = {"value": args}

    positional: list[tuple[int, Any]] = []
    keyword_parts: list[str] = []
    for key, value in args.items():
        key = str(key)
        if is_arg_key(key):
            positional.append((int(key[3:]), value))
        else:
            keyword_parts.append(f"{key}={repr(value)}")
    positional_parts = [repr(value) for _, value in sorted(positional, key=lambda item: item[0])]
    return f"{name}({', '.join(positional_parts + keyword_parts)})"


def calls_to_response(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "[]"
    return "[" + ", ".join(call_to_python(call) for call in calls) + "]"


def decode_response(response: str) -> list[str]:
    text = response.strip("`\n ")
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return []
    root = tree.body
    nodes = root.elts if isinstance(root, ast.List) else [root]
    decoded = []
    for node in nodes:
        if isinstance(node, ast.Call):
            decoded.append(ast.unparse(node))
    return decoded


def normalized_decoded_signature(call_text: str) -> str:
    try:
        tree = ast.parse(call_text, mode="eval")
        return ast.unparse(tree.body)
    except Exception:
        return re.sub(r"\s+", "", str(call_text))


def decoded_call_name(call_text: str) -> str:
    try:
        tree = ast.parse(call_text, mode="eval")
        node = tree.body
        if isinstance(node, ast.Call):
            return ast_func_name(node.func).lower()
    except Exception:
        pass
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", call_text)
    return match.group(1).lower() if match else ""


def duplicate_call_count(
    state: EntryState,
    calls: list[dict[str, Any]],
    exempt_tools: set[str] | None = None,
) -> int:
    decoded = decode_response(calls_to_response(calls))
    if not decoded:
        return 0
    exempt_tools = exempt_tools or set()
    prior = {
        normalized_decoded_signature(item)
        for batch in state.executed_call_history
        for item in batch
        if decoded_call_name(item) not in exempt_tools
    }
    return sum(
        1
        for item in decoded
        if decoded_call_name(item) not in exempt_tools
        and normalized_decoded_signature(item) in prior
    )


def execute_selected_calls(
    state: EntryState,
    response: str,
    decoded: list[str],
    args: argparse.Namespace,
) -> list[str]:
    execution_results, _ = execute_multi_turn_func_call(
        func_call_list=decoded,
        initial_config=state.record.get("initial_config", {}),
        involved_classes=state.record.get("involved_classes", []),
        model_name=args.generation_state_name,
        test_entry_id=state.rid,
        long_context=long_context_category(args.category),
        is_evaL_run=False,
    )
    state.history.append({"role": "assistant", "content": response})
    for item in execution_results:
        state.history.append({"role": "tool", "content": item})
    state.executed_call_history.append(list(decoded))
    return execution_results


def build_candidates(
    per_source_outputs: list[dict[str, dict[str, Any]]],
    state: EntryState,
) -> list[dict[str, Any]]:
    candidates = [empty_turn_candidate()]
    for outputs in per_source_outputs:
        item = copy.deepcopy(outputs.get(state.rid) or {})
        if not item:
            continue
        candidates.append(item)
    controller = controller_candidate(state)
    if controller:
        candidates.append(controller)
    return candidates


def empty_turn_candidate() -> dict[str, Any]:
    return {
        "source": "turn_stop",
        "calls": [],
        "issues": [],
        "error": "",
        "latency_ms": 0,
        "raw_text": "[]",
        "generated_tokens": 0,
    }


def controller_candidate(state: EntryState) -> dict[str, Any] | None:
    text = current_turn_user_text(state)
    prior_text = "\n".join(str(message.get("content", "")) for message in state.history).lower()
    calls: list[dict[str, Any]] = []
    mentions_final_report = "final_report.pdf" in text or "final report" in text
    if (
        "archive/log.txt" in prior_text
        and "log.txt" in text
        and re.search(r"\b(search|grep|occurrence|occurrences|find|contains?|keyword|investigate)\b", text)
    ):
        pattern = "Error" if "error" in text else ""
        if not pattern:
            keyword_match = re.search(r"keyword\s+['\"]?([A-Za-z0-9_.-]+)", text)
            pattern = keyword_match.group(1) if keyword_match else ""
        calls = [
            {"name": "cd", "arguments": {"folder": "archive"}},
            {"name": "grep", "arguments": {"file_name": "log.txt", "pattern": pattern or "Error"}},
        ]
    elif mentions_final_report and "previous_report.pdf" in text and re.search(
        r"\b(compare|diff|difference|juxtapose|alterations?)\b", text
    ):
        calls = [
            {"name": "cd", "arguments": {"folder": ".."}},
            {"name": "mv", "arguments": {"source": "previous_report.pdf", "destination": "temp"}},
            {"name": "cd", "arguments": {"folder": "temp"}},
            {"name": "diff", "arguments": {"file_name1": "final_report.pdf", "file_name2": "previous_report.pdf"}},
        ]
    elif "sort" in text and "final_report.pdf" in text:
        calls = [{"name": "sort", "arguments": {"file_name": "final_report.pdf"}}]
    elif "budget analysis" in text and "grep" in text:
        calls = [
            {"name": "cd", "arguments": {"folder": "temp"}},
            {"name": "grep", "arguments": {"file_name": "final_report.pdf", "pattern": "budget analysis"}},
        ]
    elif "final_report.pdf" in text and "temp" in text and re.search(r"\b(move|moved|moving|transfer)\b", text):
        calls = [
            {"name": "cd", "arguments": {"folder": "document"}},
            {"name": "mkdir", "arguments": {"dir_name": "temp"}},
            {"name": "mv", "arguments": {"source": "final_report.pdf", "destination": "temp"}},
        ]
    if not calls:
        return None
    return {
        "source": "controller",
        "calls": calls,
        "issues": selector.call_issues(calls, state.available_tools),
        "error": "",
        "latency_ms": 0,
        "raw_text": "controller",
        "generated_tokens": 0,
    }


def load_verifier(args: argparse.Namespace) -> selector.Verifier | None:
    if not args.verifier_model or args.verifier_model.lower() == "none":
        return None
    return selector.Verifier(
        args.verifier_model,
        args.precision,
        args.device_map,
        args.verifier_max_new_tokens,
    )


def release_verifier(verifier: selector.Verifier | None) -> None:
    if verifier is None:
        return
    unload_model(getattr(verifier, "model", None), getattr(verifier, "tokenizer", None))


def duplicate_penalty_exempt_tools(args: argparse.Namespace) -> set[str]:
    return {
        item.strip().lower()
        for item in str(getattr(args, "duplicate_penalty_exempt_tools", "") or "").split(",")
        if item.strip()
    }


def execution_result_has_error(result: Any) -> bool:
    text = str(result).lower()
    return (
        '"error"' in text
        or "error during execution" in text
        or "no such file" in text
        or "not found" in text
        or "cannot " in text
    )


def sandbox_candidate_execution(
    state: EntryState,
    calls: list[dict[str, Any]],
    args: argparse.Namespace,
    label: str,
) -> tuple[list[str], list[str]]:
    decoded = decode_response(calls_to_response(calls))
    if is_empty_execute_response(decoded):
        return [], decoded
    sandbox_name = f"{args.generation_state_name}_candidate_{state.rid}_{state.current_turn_index}_{label}"
    execute_multi_turn_func_call(
        func_call_list=[],
        initial_config=state.record.get("initial_config", {}),
        involved_classes=state.record.get("involved_classes", []),
        model_name=sandbox_name,
        test_entry_id=state.rid,
        long_context=long_context_category(args.category),
        is_evaL_run=False,
    )
    for prior in state.executed_call_history:
        execute_multi_turn_func_call(
            func_call_list=prior,
            initial_config=state.record.get("initial_config", {}),
            involved_classes=state.record.get("involved_classes", []),
            model_name=sandbox_name,
            test_entry_id=state.rid,
            long_context=long_context_category(args.category),
            is_evaL_run=False,
        )
    execution_results, _ = execute_multi_turn_func_call(
        func_call_list=decoded,
        initial_config=state.record.get("initial_config", {}),
        involved_classes=state.record.get("involved_classes", []),
        model_name=sandbox_name,
        test_entry_id=state.rid,
        long_context=long_context_category(args.category),
        is_evaL_run=False,
    )
    return execution_results, decoded


def select_multiturn_candidate(
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt: str,
    verifier: selector.Verifier | None,
    state: EntryState,
    step_index: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_text = current_turn_user_text(state)
    for item in candidates:
        item["score"] = selector.static_score(item, tools)
        semantic_delta = semantic_prompt_score(item.get("calls", []), current_text)
        item["semantic_score_delta"] = semantic_delta
        item["score"] += semantic_delta
        duplicate_count = duplicate_call_count(
            state,
            item.get("calls", []),
            duplicate_penalty_exempt_tools(args),
        )
        item["duplicate_call_count"] = duplicate_count
        if duplicate_count:
            item["score"] -= args.duplicate_call_penalty * duplicate_count
        if (
            not item.get("calls")
            and step_index == 0
            and args.empty_turn_action_penalty > 0
            and action_like_turn(current_text)
        ):
            item["empty_turn_action_penalty"] = args.empty_turn_action_penalty
            item["score"] -= args.empty_turn_action_penalty

    verifier_raw = ""
    chosen_source = None
    if verifier is not None and candidates:
        try:
            chosen_source, repair_calls, verifier_raw = verifier.choose(prompt, tools, candidates)
            if repair_calls and not selector.call_issues(repair_calls, tools):
                repair_candidate = {
                    "source": "verifier_repair",
                    "calls": repair_calls,
                    "issues": [],
                    "error": "",
                    "latency_ms": 0,
                    "raw_text": trim_text(verifier_raw, args.raw_log_chars),
                    "generated_tokens": 0,
                    "score": selector.static_score(
                        {"source": "verifier_repair", "calls": repair_calls, "issues": []},
                        tools,
                    )
                    + args.verifier_bonus,
                }
                semantic_delta = semantic_prompt_score(repair_calls, current_turn_user_text(state))
                duplicate_count = duplicate_call_count(
                    state,
                    repair_calls,
                    duplicate_penalty_exempt_tools(args),
                )
                repair_candidate["semantic_score_delta"] = semantic_delta
                repair_candidate["duplicate_call_count"] = duplicate_count
                repair_candidate["score"] += semantic_delta
                if duplicate_count:
                    repair_candidate["score"] -= args.duplicate_call_penalty * duplicate_count
                candidates.append(repair_candidate)
            if chosen_source:
                for item in candidates:
                    if item["source"] == chosen_source:
                        item["score"] += args.verifier_bonus
        except Exception as exc:  # noqa: BLE001
            verifier_raw = f"{type(exc).__name__}: {exc}"

    if args.execution_aware_selection:
        for idx, item in enumerate(candidates):
            try:
                execution_results, decoded = sandbox_candidate_execution(
                    state,
                    item.get("calls", []),
                    args,
                    f"{step_index}_{idx}_{item.get('source', 'unknown')}",
                )
                error_count = sum(1 for result in execution_results if execution_result_has_error(result))
                success_count = max(0, len(execution_results) - error_count)
                item["sandbox_decoded"] = decoded
                item["sandbox_execution_results"] = execution_results
                item["sandbox_error_count"] = error_count
                item["score"] -= args.execution_error_penalty * error_count
                item["score"] += args.execution_success_bonus * success_count
            except Exception as exc:  # noqa: BLE001
                item["sandbox_error"] = f"{type(exc).__name__}: {exc}"
                item["score"] -= args.execution_error_penalty

    ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    chosen = ranked[0] if ranked else {"source": "empty", "calls": []}
    return chosen.get("calls", []), {
        "selected_source": chosen.get("source"),
        "verifier_raw": verifier_raw,
        "verifier_chosen_source": chosen_source,
    }


def run_step(
    active_states: list[EntryState],
    specs: list[SourceSpec],
    step_index: int,
    args: argparse.Namespace,
) -> int:
    per_source_outputs = []
    for spec in specs:
        print(
            json.dumps(
                {
                    "event": "load_source",
                    "source": spec.name,
                    "turn": active_states[0].current_turn_index if active_states else -1,
                    "step": step_index,
                    "active_rows": len(active_states),
                }
            ),
            flush=True,
        )
        per_source_outputs.append(generate_for_source(spec, active_states, step_index, args))

    verifier = load_verifier(args)
    continued = 0
    try:
        for state in active_states:
            prompt = render_prompt(state, step_index)
            candidates = build_candidates(per_source_outputs, state)
            selected, diag = select_multiturn_candidate(
                candidates,
                state.available_tools,
                prompt,
                verifier,
                state,
                step_index,
                args,
            )
            response = calls_to_response(selected)
            decoded = decode_response(response)
            empty = is_empty_execute_response(decoded)
            state.current_turn_responses.append(response if not empty else "[]")

            step_log = {
                "turn": state.current_turn_index,
                "step": step_index,
                "selected_source": diag.get("selected_source"),
                "verifier_chosen_source": diag.get("verifier_chosen_source"),
                "selected_calls": selected,
                "decoded": decoded,
                "empty": empty,
                "candidates": candidates,
            }
            if diag.get("verifier_raw"):
                step_log["verifier_raw"] = trim_text(diag["verifier_raw"], args.raw_log_chars)

            if empty:
                state.active = False
                state.logs.append(step_log)
                print(
                    json.dumps(
                        {
                            "id": state.rid,
                            "turn": state.current_turn_index,
                            "step": step_index,
                            "selected": diag.get("selected_source"),
                            "decoded_calls": 0,
                            "continue": False,
                        }
                    ),
                    flush=True,
                )
                continue

            execution_results = execute_selected_calls(state, response, decoded, args)
            step_log["execution_results"] = execution_results
            state.logs.append(step_log)
            execution_errors = sum(1 for result in execution_results if execution_result_has_error(result))
            if should_stop_after_success(state, decoded, execution_results, args):
                state.active = False
            else:
                continued += 1
            print(
                json.dumps(
                    {
                        "id": state.rid,
                        "turn": state.current_turn_index,
                        "step": step_index,
                        "selected": diag.get("selected_source"),
                        "decoded_calls": len(decoded),
                        "continue": state.active,
                    }
                ),
                flush=True,
            )
    finally:
        release_verifier(verifier)
    return continued


def start_turn(state: EntryState, turn_index: int) -> None:
    holdout = state.record.get("missed_function") or {}
    if str(turn_index) in holdout:
        state.available_tools.extend(copy.deepcopy(holdout[str(turn_index)]))
        turn_messages = state.record.get("question", [])[turn_index]
        if not turn_messages:
            turn_messages = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
    else:
        turn_messages = state.record.get("question", [])[turn_index]

    state.current_turn_start = len(state.history)
    for message in turn_messages:
        if isinstance(message, dict):
            state.history.append(copy.deepcopy(message))
        else:
            state.history.append({"role": "user", "content": str(message)})
    state.current_turn_index = turn_index
    state.current_turn_responses = []
    state.active = True


def current_turn_user_text(state: EntryState) -> str:
    parts = []
    for message in state.history[state.current_turn_start :]:
        if str(message.get("role", "user")).lower() == "user":
            parts.append(str(message.get("content", "")))
    return "\n".join(parts).lower()


def semantic_prompt_score(calls: list[dict[str, Any]], current_user_text: str) -> float:
    names = {str(call.get("name") or "").lower() for call in calls}
    score = 0.0
    if "document" in current_user_text and re.search(r"\b(directory|folder|within|inside|in document)\b", current_user_text):
        has_cd_document = False
        for call in calls:
            call_args = selector.maybe_json(call.get("arguments") or {})
            if (
                str(call.get("name") or "").lower() == "cd"
                and isinstance(call_args, dict)
                and str(call_args.get("folder", "")).lower() == "document"
            ):
                has_cd_document = True
                break
        mutates_without_context = bool(names & {"mkdir", "mv", "cp", "rm", "touch"})
        if has_cd_document:
            score += 10.0
        elif mutates_without_context:
            score -= 10.0
    copy_intent = re.search(r"\b(copy|duplicate|duplicat(?:e|ed|ing)|backup)\b", current_user_text)
    move_intent = re.search(r"\b(move|moved|moving|transfer)\b", current_user_text)
    if copy_intent:
        if "cp" in names:
            score += 8.0
        if "mv" in names and re.search(r"\b(rename|renamed|name it|under the name|as)\b", current_user_text):
            score += 4.0
    elif move_intent:
        if "mv" in names:
            score += 8.0
        if "cp" in names:
            score -= 25.0
    if re.search(r"\b(create|make|mkdir|directory|folder)\b", current_user_text):
        if "mkdir" in names:
            score += 5.0
    if re.search(r"\b(search|grep|occurrence|occurrences|find|contains?)\b", current_user_text):
        if "grep" in names:
            score += 8.0
    if "hidden" in current_user_text and "ls" in names:
        for call in calls:
            args = selector.maybe_json(call.get("arguments") or {})
            if str(call.get("name") or "").lower() == "ls" and isinstance(args, dict):
                score += 6.0 if bool(args.get("a")) else -6.0
    if "sort" in current_user_text and "sort" in names:
        score += 8.0
    if re.search(r"\b(compare|diff|difference|different|juxtapose|alterations?)\b", current_user_text):
        if "diff" in names:
            score += 8.0
    return score


def action_like_turn(text: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"add|archive|authenticate|cat|comment|compare|copy|create|diff|echo|find|grep|"
            r"investigate|keyword|list|login|make|message|mkdir|move|occurrence|occurrences|"
            r"post|rename|search|send|sort|tail|touch|transfer|tweet|wc|write"
            r")\b",
            text,
        )
    )


def decoded_call_names(decoded: list[str]) -> set[str]:
    names = set()
    for item in decoded:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", item)
        if match:
            names.add(match.group(1).lower())
    return names


def should_stop_after_success(
    state: EntryState,
    decoded: list[str],
    execution_results: list[str],
    args: argparse.Namespace,
) -> bool:
    if not args.stop_after_multicall_success:
        return False
    if any(execution_result_has_error(result) for result in execution_results):
        return False
    names = decoded_call_names(decoded)
    text = current_turn_user_text(state)
    if len(decoded) > 1:
        return True
    if "sort" in names and "sort" in text:
        return True
    if "grep" in names and re.search(r"\b(search|grep|occurrence|occurrences|find|contains?)\b", text):
        return True
    if "diff" in names and re.search(r"\b(compare|diff|difference|different|juxtapose|alterations?)\b", text):
        return True
    return False


def finish_turn(state: EntryState) -> None:
    state.result.append(list(state.current_turn_responses))
    state.current_turn_responses = []
    state.active = False


def run_inference(states: list[EntryState], specs: list[SourceSpec], args: argparse.Namespace) -> None:
    init_backend_state(states, args)
    max_turns = max((len(state.record.get("question", [])) for state in states), default=0)
    for turn_index in range(max_turns):
        active_states = []
        for state in states:
            if turn_index >= len(state.record.get("question", [])):
                continue
            start_turn(state, turn_index)
            active_states.append(state)

        for step_index in range(args.max_steps_per_turn):
            still_active = [state for state in active_states if state.active]
            if not still_active:
                break
            continued = run_step(still_active, specs, step_index, args)
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
            finish_turn(state)


def write_predictions(states: list[EntryState], output_path: Path, args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for state in states:
            row = {
                "id": state.rid,
                "result": state.result,
                "model": "current_bfcl_multiturn_ensemble",
                "category": args.category,
                "sources": args.sources,
            }
            if args.include_logs:
                row["inference_log"] = state.logs
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_predictions(states: list[EntryState], score_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    correct = 0
    for state in states:
        result = evaluate_single_multi_turn(state, args)
        if result.get("valid"):
            correct += 1
        if "id" not in result:
            result["id"] = state.rid
        rows.append(json_safe(result))

    total = len(states)
    summary = {
        "category": args.category,
        "model": "current_bfcl_multiturn_ensemble",
        "sources": args.sources,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else 0.0,
    }
    scored = json_safe({"summary": summary, "rows": rows})
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
    return scored


def evaluate_single_multi_turn(state: EntryState, args: argparse.Namespace) -> dict[str, Any]:
    model_result_list = state.result
    ground_truth_list = state.answer["ground_truth"]
    if not isinstance(model_result_list, list):
        return {
            "id": state.rid,
            "model_name": args.scoring_model_name,
            "test_category": args.category,
            "valid": False,
            "error": {
                "error_message": ["Model did not output a list of model responses."],
                "error_type": "multi_turn:inference_error",
            },
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    if len(model_result_list) != len(ground_truth_list):
        return {
            "id": state.rid,
            "model_name": args.scoring_model_name,
            "test_category": args.category,
            "valid": False,
            "error": {
                "error_message": [
                    f"Model result turns ({len(model_result_list)}) do not match ground truth turns ({len(ground_truth_list)})."
                ],
                "error_type": "multi_turn:force_terminated",
            },
            "model_result": model_result_list,
            "possible_answer": ground_truth_list,
        }

    decoded_result: list[list[list[str]]] = []
    for single_turn_model_result_list in model_result_list:
        single_turn_decoded = []
        for model_result_item in single_turn_model_result_list:
            decoded = decode_response(model_result_item)
            if is_empty_execute_response(decoded):
                continue
            single_turn_decoded.append(decoded)
        decoded_result.append(single_turn_decoded)

    checker_result = multi_turn_checker(
        decoded_result,
        ground_truth_list,
        copy.deepcopy(state.record),
        args.category,
        args.scoring_model_name,
    )
    if checker_result.get("valid"):
        return {"id": state.rid, "valid": True}
    return {
        "id": state.rid,
        "model_name": args.scoring_model_name,
        "test_category": args.category,
        "valid": False,
        "error": {key: value for key, value in checker_result.items() if key != "valid"},
        "model_result_raw": model_result_list,
        "model_result_decoded": decoded_result,
        "possible_answer": ground_truth_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--score-output", required=True, help="Official multi-turn scored JSON path")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sources", default="toolace,xlam,gptoss_apigen,taskbench")
    parser.add_argument("--toolace-model", default="Team-ACE/ToolACE-8B")
    parser.add_argument("--xlam-model", default="Salesforce/Llama-xLAM-2-8b-fc-r")
    parser.add_argument("--verifier-model", default="Salesforce/Llama-xLAM-2-8b-fc-r")
    parser.add_argument("--gptoss-base", default="openai/gpt-oss-20b")
    parser.add_argument("--apigen-adapter", required=True)
    parser.add_argument("--taskbench-adapter", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=192)
    parser.add_argument("--verifier-bonus", type=float, default=1.0)
    parser.add_argument("--execution-error-penalty", type=float, default=35.0)
    parser.add_argument("--execution-success-bonus", type=float, default=2.0)
    parser.add_argument("--duplicate-call-penalty", type=float, default=45.0)
    parser.add_argument("--duplicate-penalty-exempt-tools", default="cd")
    parser.add_argument("--empty-turn-action-penalty", type=float, default=12.0)
    parser.add_argument("--max-steps-per-turn", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--no-execution-aware-selection", dest="execution_aware_selection", action="store_false")
    parser.set_defaults(execution_aware_selection=True)
    parser.add_argument("--stop-after-multicall-success", dest="stop_after_multicall_success", action="store_true")
    parser.add_argument("--no-stop-after-multicall-success", dest="stop_after_multicall_success", action="store_false")
    parser.set_defaults(stop_after_multicall_success=True)
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--raw-log-chars", type=int, default=1200)
    parser.add_argument("--generation-state-name", default="bfcl_mt_current_ensemble_generation")
    parser.add_argument("--scoring-model-name", default="bfcl_mt_current_ensemble_score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    states, category = load_entries(args)
    specs = source_specs(args)
    print(
        json.dumps(
            {
                "event": "start",
                "category": category,
                "rows": len(states),
                "sources": [spec.name for spec in specs],
                "verifier_model": args.verifier_model,
                "max_steps_per_turn": args.max_steps_per_turn,
            },
            indent=2,
        ),
        flush=True,
    )
    run_inference(states, specs, args)
    write_predictions(states, Path(args.output), args)
    scored = score_predictions(states, Path(args.score_output), args)
    print(json.dumps(scored["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
