from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "category",
        "decision_ok",
        "call_count_ok",
        "tool_names_ok",
        "arg_keys_ok",
        "arg_values_ok",
        "exact_call_match",
        "main_failure_type",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def score_predictions(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = {str(row.get("id")): row for row in prediction_rows}
    cases = [score_case(gold, predictions.get(str(gold.get("id")), {})) for gold in gold_rows]
    return {"metrics": summarize_cases(cases), "cases": cases}


def score_case(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = gold.get("expected_tool_binding") or {}
    expected_calls = _expected_calls(expected)
    plan = prediction.get("tool_binding_plan") or prediction.get("prediction") or {}
    predicted_calls = _predicted_calls(plan)
    expected_decision = expected.get("tool_decision") or ("call" if expected_calls else "no_tool")
    predicted_decision = plan.get("tool_decision") or ("call" if predicted_calls else "no_tool")

    decision_ok = expected_decision == predicted_decision
    call_count_ok = len(expected_calls) == len(predicted_calls)
    tool_names_ok = Counter(call["tool_name"] for call in expected_calls) == Counter(
        call.get("tool_name") for call in predicted_calls
    )
    arg_scores = _argument_scores(expected_calls, predicted_calls)
    exact_call_match = (
        decision_ok
        and call_count_ok
        and tool_names_ok
        and arg_scores["arg_keys_ok"]
        and arg_scores["arg_values_ok"]
    )
    failures = []
    if not decision_ok:
        failures.append("wrong_decision")
    if not call_count_ok:
        failures.append("wrong_call_count")
    if not tool_names_ok:
        failures.append("wrong_tool_name")
    if not arg_scores["arg_keys_ok"]:
        failures.append("missing_or_extra_args")
    if not arg_scores["arg_values_ok"]:
        failures.append("wrong_arg_value")
    return {
        "id": gold.get("id"),
        "category": gold.get("category"),
        "decision_ok": decision_ok,
        "call_count_ok": call_count_ok,
        "tool_names_ok": tool_names_ok,
        "arg_keys_ok": arg_scores["arg_keys_ok"],
        "arg_values_ok": arg_scores["arg_values_ok"],
        "arg_key_recall": arg_scores["arg_key_recall"],
        "arg_value_accuracy": arg_scores["arg_value_accuracy"],
        "exact_call_match": exact_call_match,
        "main_failure_type": failures[0] if failures else "",
        "failure_types": failures,
        "notes": "; ".join(_notes(expected_calls, predicted_calls, expected_decision, predicted_decision, failures)),
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    if not total:
        return {
            "case_count": 0,
            "decision_accuracy": 0,
            "call_count_accuracy": 0,
            "tool_name_accuracy": 0,
            "argument_key_recall": 0,
            "argument_value_accuracy": 0,
            "exact_call_match_rate": 0,
            "failure_counts": {},
        }
    failures: Counter[str] = Counter()
    for case in cases:
        failures.update(case.get("failure_types") or [])
    return {
        "case_count": total,
        "decision_accuracy": _rate(cases, "decision_ok"),
        "call_count_accuracy": _rate(cases, "call_count_ok"),
        "tool_name_accuracy": _rate(cases, "tool_names_ok"),
        "argument_key_recall": round(sum(float(case.get("arg_key_recall", 0)) for case in cases) / total, 4),
        "argument_value_accuracy": round(sum(float(case.get("arg_value_accuracy", 0)) for case in cases) / total, 4),
        "exact_call_match_rate": _rate(cases, "exact_call_match"),
        "failure_counts": dict(failures),
    }


def _argument_scores(expected_calls: list[dict[str, Any]], predicted_calls: list[dict[str, Any]]) -> dict[str, Any]:
    if not expected_calls:
        return {"arg_keys_ok": True, "arg_values_ok": True, "arg_key_recall": 1.0, "arg_value_accuracy": 1.0}

    matched = _match_calls(expected_calls, predicted_calls)
    total_keys = 0
    present_keys = 0
    total_values = 0
    matched_values = 0
    all_keys_ok = True
    all_values_ok = True
    for expected_call, predicted_call in matched:
        expected_args = expected_call.get("arguments") or {}
        predicted_args = predicted_call.get("arguments") if predicted_call else {}
        if not isinstance(predicted_args, dict):
            predicted_args = {}
        for key, accepted in expected_args.items():
            accepted_values = accepted if isinstance(accepted, list) else [accepted]
            optional_empty = any(_is_empty_allowed(value) for value in accepted_values)
            total_keys += 0 if optional_empty else 1
            if key in predicted_args:
                present_keys += 0 if optional_empty else 1
            elif not optional_empty:
                all_keys_ok = False
            if key not in predicted_args and optional_empty:
                continue
            total_values += 1
            if key in predicted_args and any(_values_match(predicted_args[key], value) for value in accepted_values):
                matched_values += 1
            else:
                all_values_ok = False
    key_recall = present_keys / total_keys if total_keys else 1.0
    value_accuracy = matched_values / total_values if total_values else 1.0
    return {
        "arg_keys_ok": all_keys_ok,
        "arg_values_ok": all_values_ok,
        "arg_key_recall": round(key_recall, 4),
        "arg_value_accuracy": round(value_accuracy, 4),
    }


def _match_calls(
    expected_calls: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    unused = list(predicted_calls)
    pairs = []
    for expected in expected_calls:
        match_index = next(
            (
                index
                for index, predicted in enumerate(unused)
                if predicted.get("tool_name") == expected.get("tool_name")
            ),
            None,
        )
        if match_index is None:
            pairs.append((expected, None))
        else:
            pairs.append((expected, unused.pop(match_index)))
    return pairs


def _expected_calls(expected: dict[str, Any]) -> list[dict[str, Any]]:
    calls = expected.get("calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _predicted_calls(plan: dict[str, Any]) -> list[dict[str, Any]]:
    calls = plan.get("calls") if isinstance(plan, dict) else []
    return [call for call in calls if isinstance(call, dict)]


def _values_match(predicted: Any, expected: Any) -> bool:
    if _is_empty_allowed(expected):
        return predicted in (None, "", [], {})
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return math.isclose(float(predicted), float(expected), rel_tol=1e-4, abs_tol=1e-4)
    if isinstance(expected, list):
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return False
        return all(_values_match(p, e) for p, e in zip(predicted, expected))
    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False
        return all(key in predicted and _values_match(predicted[key], value) for key, value in expected.items())
    return str(predicted).strip().lower() == str(expected).strip().lower()


def _is_empty_allowed(value: Any) -> bool:
    return value in ("", None)


def _rate(cases: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for case in cases if case.get(field)) / len(cases), 4)


def _notes(
    expected_calls: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
    expected_decision: str,
    predicted_decision: str,
    failures: list[str],
) -> list[str]:
    if not failures:
        return []
    notes = [f"expected_decision={expected_decision}, predicted_decision={predicted_decision}"]
    notes.append(f"expected_tools={[call.get('tool_name') for call in expected_calls]}")
    notes.append(f"predicted_tools={[call.get('tool_name') for call in predicted_calls]}")
    return notes
