#!/usr/bin/env python3
"""Evaluate API-Bank next-tool routing through the goal-graph runtime."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
for path in (SRC_DIR, SCRIPT_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_apibank_capability_tool_routing import (
    build_api_catalog,
    build_tool_search_index,
    expected_name,
    latest_dialogue_context,
    prompt_text,
    read_rows,
    resolve_api_bank_level_paths,
    tools_from_row,
    _level_metric_label,
)
from goal_graph_eval_common import benchmark_compile_tools, plan_and_compile_goal_graph
from run_gptoss_capability_plan import generate_text, load_model
from taskdecomp.tool_binding import build_tool_binding_plan


def predicted_api_tool_names(result: dict[str, Any], tools: list[dict[str, Any]], prompt: str = "") -> list[str]:
    """Return next-API route predictions without requiring unsafe execution.

    API-Bank scores the next API name. The goal-graph runtime should still refuse
    to compile calls with missing required inputs, but those ask-user plans often
    contain the correct routed API. Use compiled calls first, then verified
    binding-plan calls, then missing-input route candidates.
    """
    plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    audits = [audit for audit in plan.get("candidate_tool_audits") or [] if isinstance(audit, dict)]
    auth_name = _auth_tool_name(tools) or _auth_tool_name_from_audits(audits)
    compiled = [
        str(call.get("tool_name") or call.get("name") or "")
        for call in result.get("calls") or []
        if isinstance(call, dict) and (call.get("tool_name") or call.get("name"))
    ]
    compiled = _dedupe_api_route_names(compiled)
    if compiled:
        auth_step = _prompt_indicates_auth_step(prompt)
        if auth_step and auth_name:
            return [auth_name]
        latest = _prefer_latest_prompt_route(compiled, prompt, tools, result)
        if latest:
            return [latest]
        protected = _complete_token_protected_route_from_audits(audits, auth_name)
        if protected and all(name == auth_name for name in compiled):
            return [protected]
        if protected and _compiled_routes_are_context_helpers(compiled, tools):
            return [protected]
        return compiled

    planned = [
        str(call.get("tool_name") or "")
        for call in plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name") and not call.get("missing_arguments")
    ]
    planned = _dedupe_api_route_names(planned)
    if planned:
        latest = _prefer_latest_prompt_route(planned, prompt, tools, result)
        if latest:
            return [latest]
        return planned

    prior_route = _prior_result_schema_route(prompt, tools)
    if prior_route:
        return [prior_route]

    if auth_name and any(_audit_missing_token(audit) for audit in audits):
        if _prompt_has_available_token(prompt) and not _prompt_indicates_auth_step(prompt):
            action = _best_missing_token_action(audits, prompt, tools)
            if not action:
                action = _first_missing_token_action(audits, auth_name)
            if action:
                return [action]
        return [auth_name]

    raw_terminal_route = _raw_route_for_terminal_empty_plan(result, tools, prompt)
    if raw_terminal_route:
        return [raw_terminal_route]

    for audit in audits:
        name = str(audit.get("tool_name") or "")
        if name and str(audit.get("semantic_fit") or "").lower() in {"exact", "partial"}:
            return [name]
    return []


def _prefer_latest_prompt_route(
    names: list[str],
    prompt: str,
    tools: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    if len(names) <= 1 or not prompt:
        return ""
    latest_user = _latest_user_text(prompt)
    if not latest_user:
        return ""
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    calls = [
        call
        for call in (result.get("calls") or [])
        if isinstance(call, dict) and str(call.get("tool_name") or call.get("name") or "") in names
    ]
    scored = []
    for index, name in enumerate(names):
        tool = tools_by_name.get(name, {})
        call_args = [
            call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            for call in calls
            if str(call.get("tool_name") or call.get("name") or "") == name
        ]
        score = _latest_route_score(name, tool, call_args, latest_user)
        scored.append((score, -index, name))
    scored.sort(reverse=True)
    if not scored or scored[0][0] <= 0:
        return ""
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return ""
    return scored[0][2]


def _latest_route_score(name: str, tool: dict[str, Any], call_args: list[dict[str, Any]], latest_user: str) -> int:
    latest = latest_user.lower()
    tool_words = _identifier_words(name)
    description_words = _content_words(str(tool.get("description") or ""))
    score = 0
    for word in tool_words:
        if len(word) >= 3 and re.search(rf"\b{re.escape(word)}\b", latest):
            score += 4
    for word in description_words:
        if len(word) >= 4 and re.search(rf"\b{re.escape(word)}\b", latest):
            score += 1
    for args in call_args:
        for value in args.values():
            for scalar in _argument_scalars(value):
                if len(scalar) >= 3 and scalar.lower() in latest:
                    score += 3
    return score


def _raw_route_for_terminal_empty_plan(result: dict[str, Any], tools: list[dict[str, Any]], prompt: str) -> str:
    if not prompt or not tools:
        return ""
    plan = result.get("tool_binding_plan") if isinstance(result.get("tool_binding_plan"), dict) else {}
    decision = str(plan.get("tool_decision") or "").strip().lower()
    if decision not in {"ask_user", "no_tool"}:
        return ""
    if result.get("calls") or plan.get("calls"):
        return ""
    raw_plan = build_tool_binding_plan(prompt, tools)
    raw_audits = [audit for audit in raw_plan.get("candidate_tool_audits") or [] if isinstance(audit, dict)]
    auth_name = _auth_tool_name(tools) or _auth_tool_name_from_audits(raw_audits)
    if auth_name and any(_audit_missing_token(audit) for audit in raw_audits):
        if _prompt_has_available_token(prompt) and not _prompt_indicates_auth_step(prompt):
            action = _best_missing_token_action(raw_audits, prompt, tools)
            if not action:
                action = _first_missing_token_action(raw_audits, auth_name)
            if action:
                return action
        return auth_name
    raw_calls = [
        call
        for call in raw_plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name") and not call.get("missing_arguments")
    ]
    if raw_calls:
        names = _dedupe_api_route_names([str(call.get("tool_name") or "") for call in raw_calls])
        latest = _prefer_latest_prompt_route(names, prompt, tools, {"calls": raw_calls})
        if latest:
            return latest
        if names:
            return names[0]
    for audit in raw_audits:
        name = str(audit.get("tool_name") or "")
        if name and str(audit.get("semantic_fit") or "").lower() in {"exact", "partial"}:
            return name
    return ""


def _best_missing_token_action(audits: list[dict[str, Any]], prompt: str, tools: list[dict[str, Any]]) -> str:
    latest_user = _latest_user_text(prompt)
    if not latest_user:
        return ""
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    candidates = []
    for index, audit in enumerate(audits):
        name = str(audit.get("tool_name") or "")
        missing = {str(slot).lower() for slot in audit.get("missing_slots") or []}
        if not name or missing != {"token"}:
            continue
        if str(audit.get("semantic_fit") or "").lower() not in {"exact", "partial"}:
            continue
        score = _latest_route_score(
            name,
            tools_by_name.get(name, {}),
            [call.get("arguments") for call in audit.get("planned_calls") or [] if isinstance(call, dict)],
            latest_user,
        )
        if score > 0:
            candidates.append((score, float(audit.get("score") or 0.0), -index, name))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][3]


def _first_missing_token_action(audits: list[dict[str, Any]], auth_name: str) -> str:
    for audit in audits:
        name = str(audit.get("tool_name") or "")
        if not name or name == auth_name:
            continue
        missing = {str(slot).lower() for slot in audit.get("missing_slots") or []}
        if missing != {"token"}:
            continue
        if str(audit.get("semantic_fit") or "").lower() not in {"exact", "partial"}:
            continue
        return name
    return ""


def _latest_user_text(prompt: str) -> str:
    matches = re.findall(r"(?im)^User:\s*(.+)$", prompt)
    return matches[-1].strip() if matches else ""


def _prompt_indicates_auth_step(prompt: str) -> bool:
    if not prompt:
        return False
    latest_user = _latest_user_text(prompt).lower()
    latest_window = "\n".join(line for line in prompt.splitlines()[-4:])
    if "username" not in latest_user and "password" not in latest_user:
        return False
    needs_token = bool(
        re.search(
            r"\b(?:need|needs|need to|get|get your|get the|obtain|retrieve).{0,40}\btoken\b",
            latest_window,
            re.I,
        )
    )
    if _latest_prior_token_result_available(prompt) and _latest_window_says_token_obtained(latest_window):
        return False
    if _latest_prior_token_result_available(prompt) and not needs_token:
        return False
    return bool(needs_token or re.search(r"\bauthenticat(?:e|ed|ing|ion)\b", latest_window, re.I))


def _latest_prior_token_result_available(prompt: str) -> bool:
    match = re.search(r"(?im)^Latest prior API result:\s*(.+)$", prompt)
    if not match:
        return False
    return bool(re.search(r"['\"]?token['\"]?\s*:", match.group(1), re.I))


def _latest_window_says_token_obtained(text: str) -> bool:
    return bool(
        re.search(r"\b(?:got|have|received|obtained|retrieved)\s+(?:the\s+)?token\b", text, re.I)
        or re.search(r"\b(?:got|have|received|obtained|retrieved)\b.{0,40}\btoken\b", text, re.I)
        or re.search(r"\b(?:able to|successfully|already)\b.{0,40}\b(?:retrieve|get|obtain)\b.{0,30}\btoken\b", text, re.I)
        or re.search(r"\btoken\b.{0,40}\b(?:retrieved|obtained|received)\b", text, re.I)
        or re.search(r"\b(?:authenticated|authentication\s+(?:complete|successful))\b", text, re.I)
    )


def _prompt_has_available_token(prompt: str) -> bool:
    if not prompt:
        return False
    return _latest_prior_token_result_available(prompt) or _latest_window_says_token_obtained(
        "\n".join(line for line in prompt.splitlines()[-4:])
    )


def _prior_result_schema_route(prompt: str, tools: list[dict[str, Any]]) -> str:
    if not prompt or not _latest_user_is_prior_result_followup(prompt):
        return ""
    prior_value = _latest_prior_result_value(prompt)
    prior_keys = _prior_result_keys(prior_value) - {"result", "results", "status", "output"}
    if not prior_keys:
        return ""
    scored = []
    for index, tool in enumerate(tools):
        name = str(tool.get("name") or "")
        output_keys = _tool_output_schema_keys(tool) - {"result", "results", "status", "output"}
        overlap = prior_keys & output_keys
        if overlap:
            scored.append((len(overlap), len(output_keys & prior_keys), -index, name))
    if not scored:
        return ""
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return ""
    return scored[0][3]


def _latest_user_is_prior_result_followup(prompt: str) -> bool:
    latest = _latest_user_text(prompt).lower()
    if not latest:
        return False
    return bool(
        re.search(r"\b(?:yes|yeah|yep|please|tell me more|more information|more info|provide more|details?)\b", latest)
        or re.search(r"\b(?:these|those|any of these|each of these|one of these|about .+)\b", latest)
    )


def _latest_prior_result_value(prompt: str) -> Any:
    matches = re.findall(r"(?im)^(?:Latest prior API result|Prior API result):\s*(.+)$", prompt)
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None


def _prior_result_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_prior_result_keys(item))
    elif isinstance(value, list):
        for item in value[:5]:
            keys.update(_prior_result_keys(item))
    return keys


def _tool_output_schema_keys(tool: dict[str, Any]) -> set[str]:
    outputs = tool.get("output_parameters") if isinstance(tool.get("output_parameters"), dict) else {}
    keys = {str(key).lower() for key in outputs}
    for name, spec in outputs.items():
        keys.update(_content_words(str(name)))
        if isinstance(spec, dict):
            keys.update(_content_words(str(spec.get("description") or "")))
            for match in re.finditer(
                r'"([A-Za-z_][A-Za-z0-9_]*)"|\'([A-Za-z_][A-Za-z0-9_]*)\'',
                str(spec.get("description") or ""),
            ):
                key = match.group(1) or match.group(2)
                if key:
                    keys.add(key.lower())
    return keys


def _identifier_words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name.replace("_", " ").replace("-", " "))
    return _content_words(spaced)


def _content_words(text: str) -> list[str]:
    stop = {"api", "the", "for", "and", "with", "this", "that", "user", "data", "info"}
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9]+", text) if word.lower() not in stop]


def _argument_scalars(value: Any) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float, str)):
        return [str(value).strip().lower()]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_argument_scalars(item))
        return values
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(_argument_scalars(item))
        return values
    return []


def _dedupe_api_route_names(names: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _complete_token_protected_route_from_audits(audits: list[dict[str, Any]], auth_name: str) -> str:
    for audit in audits:
        name = str(audit.get("tool_name") or "")
        if not name or name == auth_name:
            continue
        if "token" not in {str(slot).lower() for slot in audit.get("required_slots") or []}:
            continue
        if audit.get("missing_slots"):
            continue
        if not _audit_has_bound_token(audit):
            continue
        if not audit.get("planned_calls"):
            continue
        return name
    return ""


def _audit_has_bound_token(audit: dict[str, Any]) -> bool:
    bindings = audit.get("slot_bindings") if isinstance(audit.get("slot_bindings"), dict) else {}
    token_value = bindings.get("token") or bindings.get("access_token")
    if token_value:
        return True
    for call in audit.get("planned_calls") or []:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if args.get("token") or args.get("access_token"):
            return True
    return False


def _audit_missing_token(audit: dict[str, Any]) -> bool:
    missing = {str(value).lower() for value in audit.get("missing_slots") or []}
    if "token" not in missing:
        return False
    if str(audit.get("semantic_fit") or "").lower() in {"exact", "partial"}:
        return True
    if len(missing) == 1 and float(audit.get("score") or 0.0) >= 3.0:
        return True
    return False


def _compiled_routes_are_context_helpers(compiled: list[str], tools: list[dict[str, Any]]) -> bool:
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    for name in compiled:
        tool = tools_by_name.get(name, {})
        params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        text = f"{name} {tool.get('description') or ''}".lower()
        if required:
            return False
        if not re.search(r"\b(?:today|date|time|current)\b", text):
            return False
    return bool(compiled)


def _auth_tool_name(tools: list[dict[str, Any]]) -> str:
    for tool in tools:
        name = str(tool.get("name") or "")
        normalized = name.replace("_", "").replace("-", "").lower()
        if normalized in {"getusertoken", "gettoken", "authenticateuser"}:
            return name
    for tool in tools:
        name = str(tool.get("name") or "")
        text = f"{name} {tool.get('description') or ''}".lower()
        if "token" in text and ("username" in text or "password" in text or "auth" in text):
            return name
    return ""


def _auth_tool_name_from_audits(audits: list[dict[str, Any]]) -> str:
    for audit in audits:
        name = str(audit.get("tool_name") or "")
        normalized = name.replace("_", "").replace("-", "").lower()
        if normalized in {"getusertoken", "gettoken", "authenticateuser"}:
            return name
    for audit in audits:
        name = str(audit.get("tool_name") or "")
        required = {str(slot).lower() for slot in audit.get("required_slots") or []}
        missing = {str(slot).lower() for slot in audit.get("missing_slots") or []}
        if "token" in name.lower() and ({"username", "password"} <= required or {"username", "password"} <= missing):
            return name
    return ""


def evaluate_file(
    path: Path,
    limit: int,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    repair_attempts: int,
    plan_handle: Any | None,
    allow_side_effects: bool,
    use_latest_context: bool,
    use_binder_fallback: bool,
    planner_mode: str,
    api_catalog: dict[str, dict[str, Any]] | None = None,
    tool_search_index: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = read_rows(path, limit)
    review_rows = []
    counts = Counter()

    for index, row in enumerate(rows):
        expected = expected_name(row)
        tools = benchmark_compile_tools(
            tools_from_row(row, api_catalog=api_catalog, tool_search_index=tool_search_index)
        )
        prompt = prompt_text(row)
        if use_latest_context:
            prompt = latest_dialogue_context(prompt)
        result = plan_and_compile_goal_graph(
            model,
            tokenizer,
            generate_text,
            prompt,
            tools,
            max_new_tokens=max_new_tokens,
            repair_attempts=repair_attempts,
            allow_side_effects=allow_side_effects,
            use_binder_fallback=use_binder_fallback,
            planner_mode=planner_mode,
        )
        predicted = predicted_api_tool_names(result, tools, prompt)
        top = predicted[0] if predicted else ""
        ok = bool(expected) and top == expected
        counts["total"] += 1
        counts["top1_ok"] += int(ok)
        counts["no_prediction"] += int(not top)
        counts["multi_prediction"] += int(len(predicted) > 1)
        counts["parse_error"] += int(bool(result.get("parse_error")))
        counts["verification_error"] += int(not result.get("verification_ok"))

        row_file = row.get("file") or row.get("api_id") or path.name
        row_index = row.get("id") if "id" in row else row.get("sample_id", index)
        row_id = f"{row_file}:{row_index}"
        if plan_handle is not None:
            plan_handle.write(
                json.dumps(
                    {
                        "id": row_id,
                        "level": path.stem,
                        "prompt": prompt,
                        "expected": expected,
                        "predicted": predicted,
                        "goal_graph_result": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            plan_handle.flush()

        if not ok:
            review_rows.append(
                {
                    "id": row_id,
                    "expected": expected,
                    "predicted": json.dumps(predicted, ensure_ascii=False),
                    "top": top,
                    "verification_ok": bool(result.get("verification_ok")),
                    "parse_error": result.get("parse_error") or "",
                    "diagnostic_codes": json.dumps(result.get("diagnostic_codes") or []),
                    "diagnostics": json.dumps(result.get("diagnostics") or [], ensure_ascii=False),
                    "tool_names": json.dumps([tool.get("name") for tool in tools], ensure_ascii=False),
                    "api_catalog_size": len(api_catalog or {}),
                    "tool_search_index_size": len(tool_search_index or {}),
                    "capability_count": result.get("capability_count", 0),
                    "compiled_calls": json.dumps(result.get("calls") or [], ensure_ascii=False),
                    "prompt": prompt[:1200],
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
            "parse_error_rate": counts["parse_error"] / total if total else 0.0,
            "verification_error_rate": counts["verification_error"] / total if total else 0.0,
            "failure_count": len(review_rows),
        },
        "review_rows": review_rows,
    }


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "level",
        "id",
        "expected",
        "predicted",
        "top",
        "verification_ok",
        "parse_error",
        "diagnostic_codes",
        "diagnostics",
        "tool_names",
        "api_catalog_size",
        "tool_search_index_size",
        "capability_count",
        "compiled_calls",
        "prompt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--api-bank-root", required=True)
    parser.add_argument("--levels", nargs="+", default=["level-1-api", "level-2-api"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--plans-jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--planner-mode", choices=["stepwise", "one_shot"], default="stepwise")
    parser.add_argument(
        "--full-prompt",
        action="store_true",
        help="Use the full sanitized API-Bank prompt instead of latest dialogue context.",
    )
    parser.add_argument(
        "--strict-read-only",
        action="store_true",
        help="Keep runtime side-effect gates active during offline benchmark scoring.",
    )
    parser.add_argument(
        "--no-binder-fallback",
        action="store_true",
        help="Disable frozen-pipeline binder fallback in one_shot planner mode.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    allow_side_effects = not args.strict_read_only
    use_latest_context = not args.full_prompt
    use_binder_fallback = not args.no_binder_fallback

    root = Path(args.api_bank_root)
    api_catalog = build_api_catalog(root)
    tool_search_index = build_tool_search_index(root, api_catalog=api_catalog)
    all_metrics = []
    all_review_rows = []
    plan_path = Path(args.plans_jsonl) if args.plans_jsonl else None
    plan_handle = None
    try:
        if plan_path is not None:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_handle = plan_path.open("w", encoding="utf-8")
        for level, path in resolve_api_bank_level_paths(root, args.levels):
            result = evaluate_file(
                path,
                args.limit,
                model,
                tokenizer,
                args.max_new_tokens,
                args.repair_attempts,
                plan_handle,
                allow_side_effects,
                use_latest_context,
                use_binder_fallback,
                args.planner_mode,
                api_catalog=api_catalog,
                tool_search_index=tool_search_index,
            )
            metrics = result["metrics"]
            metrics["level"] = _level_metric_label(root, level, path)
            all_metrics.append(metrics)
            for row in result["review_rows"]:
                row["level"] = metrics["level"]
                all_review_rows.append(row)
    finally:
        if plan_handle is not None:
            plan_handle.close()

    total = sum(item["total"] for item in all_metrics)
    aggregate = {
        "total": total,
        "top1_tool_accuracy": sum(item["top1_tool_accuracy"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "no_prediction_rate": sum(item["no_prediction_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "multi_prediction_rate": sum(item["multi_prediction_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "parse_error_rate": sum(item["parse_error_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
        "verification_error_rate": sum(item["verification_error_rate"] * item["total"] for item in all_metrics) / total if total else 0.0,
    }
    payload = {
        "pipeline": "goal_graph_runtime",
        "aggregate": aggregate,
        "by_level": all_metrics,
        "limit": args.limit,
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "repair_attempts": args.repair_attempts,
        "planner_mode": args.planner_mode,
        "latest_dialogue_context": use_latest_context,
        "benchmark_compile_allows_side_effect_names": allow_side_effects,
        "stepwise_binding_enabled": args.planner_mode == "stepwise",
        "binder_fallback_enabled": use_binder_fallback if args.planner_mode == "one_shot" else False,
        "api_catalog_size": len(api_catalog),
        "tool_search_index_size": len(tool_search_index),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_review_csv(Path(args.review_csv), all_review_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
