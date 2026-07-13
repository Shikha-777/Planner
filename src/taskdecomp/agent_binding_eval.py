from __future__ import annotations

import csv
import json
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
        "subtask_decomposition_ok",
        "agent_assignment_exact_match",
        "capability_to_agent_match_rate",
        "agent_assignment_recall_at_3",
        "handoff_input_accuracy",
        "dependency_graph_accuracy",
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
    expected = _expected_binding(gold)
    predicted = prediction.get("agent_binding_plan") or prediction.get("prediction") or {}
    expected_assignments = _assignments(expected)
    predicted_assignments = _assignments(predicted)
    expected_decision = str(expected.get("assignment_decision") or "assign")
    predicted_decision = str(predicted.get("assignment_decision") or "unsupported")

    expected_by_key = {_assignment_key(item): item for item in expected_assignments}
    predicted_by_key = {_assignment_key(item): item for item in predicted_assignments}

    decision_ok = expected_decision == predicted_decision
    expected_sequence = [
        (str(item.get("subtask_id")), str(item.get("capability"))) for item in expected_assignments
    ]
    predicted_sequence = [
        (str(item.get("subtask_id")), str(item.get("capability"))) for item in predicted_assignments
    ]
    subtask_decomposition_ok = expected_sequence == predicted_sequence

    matched_agent_count = 0
    recall_at_3_count = 0
    input_ok_count = 0
    dependency_ok_count = 0
    expected_count = len(expected_assignments)
    for key, expected_assignment in expected_by_key.items():
        predicted_assignment = predicted_by_key.get(key)
        if not predicted_assignment:
            continue
        expected_agent = str(expected_assignment.get("assigned_agent"))
        predicted_agent = str(predicted_assignment.get("assigned_agent"))
        if expected_agent == predicted_agent:
            matched_agent_count += 1
        candidate_agent_ids = _candidate_agent_ids(predicted_assignment)
        if expected_agent == predicted_agent or expected_agent in candidate_agent_ids:
            recall_at_3_count += 1
        if _inputs_match(expected_assignment, predicted_assignment):
            input_ok_count += 1
        if set(_strings(expected_assignment.get("depends_on"))) == set(
            _strings(predicted_assignment.get("depends_on"))
        ):
            dependency_ok_count += 1

    matched_keys = expected_by_key.keys() & predicted_by_key.keys()
    missing_assignments = max(expected_count - len(matched_keys), 0)
    extra_assignments = max(len(predicted_by_key.keys() - expected_by_key.keys()), 0)
    wrong_agent_count = expected_count - matched_agent_count - missing_assignments
    exact_match = (
        decision_ok
        and subtask_decomposition_ok
        and matched_agent_count == expected_count
        and input_ok_count == expected_count
        and dependency_ok_count == expected_count
        and extra_assignments == 0
    )
    failures = []
    if not decision_ok:
        failures.append("wrong_decision")
    if not subtask_decomposition_ok:
        failures.append("wrong_subtask_decomposition")
    if missing_assignments:
        failures.append("missing_agent_assignment")
    if extra_assignments:
        failures.append("extra_agent_assignment")
    if wrong_agent_count > 0:
        failures.append("wrong_agent_type")
    if input_ok_count < expected_count:
        failures.append("wrong_handoff_inputs")
    if dependency_ok_count < expected_count:
        failures.append("wrong_dependency_graph")

    return {
        "id": gold.get("id"),
        "category": gold.get("category") or (gold.get("metadata") or {}).get("category"),
        "decision_ok": decision_ok,
        "subtask_decomposition_ok": subtask_decomposition_ok,
        "agent_assignment_exact_match": exact_match,
        "agent_assignment_recall_at_3": _rate_count(recall_at_3_count, expected_count),
        "capability_to_agent_match_rate": _rate_count(matched_agent_count, expected_count),
        "wrong_agent_type_rate": _rate_count(wrong_agent_count, expected_count),
        "missing_agent_assignment_rate": _rate_count(missing_assignments, expected_count),
        "extra_agent_assignment_rate": _rate_count(
            extra_assignments,
            max(len(predicted_assignments), 1),
        ),
        "handoff_input_accuracy": _rate_count(input_ok_count, expected_count),
        "dependency_graph_accuracy": _rate_count(dependency_ok_count, expected_count),
        "ask_user_ok": expected_decision != "ask_user" or predicted_decision == "ask_user",
        "unsupported_ok": expected_decision != "unsupported" or predicted_decision == "unsupported",
        "overdelegated": len(predicted_assignments) > len(expected_assignments),
        "end_to_end_success": exact_match,
        "main_failure_type": failures[0] if failures else "",
        "failure_types": failures,
        "notes": "; ".join(
            _notes(
                expected_assignments,
                predicted_assignments,
                expected_decision,
                predicted_decision,
                failures,
            )
        ),
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    if not total:
        return {
            "case_count": 0,
            "subtask_decomposition_accuracy": 0,
            "agent_assignment_exact_match": 0,
            "agent_assignment_recall_at_3": 0,
            "capability_to_agent_match_rate": 0,
            "wrong_agent_type_rate": 0,
            "missing_agent_assignment_rate": 0,
            "extra_agent_assignment_rate": 0,
            "handoff_input_accuracy": 0,
            "dependency_graph_accuracy": 0,
            "ask_user_accuracy": 0,
            "unsupported_accuracy": 0,
            "overdelegation_rate": 0,
            "end_to_end_success_rate": 0,
            "failure_counts": {},
        }
    failures: Counter[str] = Counter()
    for case in cases:
        failures.update(case.get("failure_types") or [])
    return {
        "case_count": total,
        "subtask_decomposition_accuracy": _bool_rate(cases, "subtask_decomposition_ok"),
        "agent_assignment_exact_match": _bool_rate(cases, "agent_assignment_exact_match"),
        "agent_assignment_recall_at_3": _avg(cases, "agent_assignment_recall_at_3"),
        "capability_to_agent_match_rate": _avg(cases, "capability_to_agent_match_rate"),
        "wrong_agent_type_rate": _avg(cases, "wrong_agent_type_rate"),
        "missing_agent_assignment_rate": _avg(cases, "missing_agent_assignment_rate"),
        "extra_agent_assignment_rate": _avg(cases, "extra_agent_assignment_rate"),
        "handoff_input_accuracy": _avg(cases, "handoff_input_accuracy"),
        "dependency_graph_accuracy": _avg(cases, "dependency_graph_accuracy"),
        "ask_user_accuracy": _bool_rate(cases, "ask_user_ok"),
        "unsupported_accuracy": _bool_rate(cases, "unsupported_ok"),
        "overdelegation_rate": _bool_rate(cases, "overdelegated"),
        "end_to_end_success_rate": _bool_rate(cases, "end_to_end_success"),
        "failure_counts": dict(failures),
    }


def _expected_binding(gold: dict[str, Any]) -> dict[str, Any]:
    if isinstance(gold.get("expected_agent_binding"), dict):
        return gold["expected_agent_binding"]
    assignments = gold.get("gold_agent_assignments") or []
    return {
        "assignment_decision": gold.get("assignment_decision") or "assign",
        "assignments": assignments if isinstance(assignments, list) else [],
    }


def _assignments(plan: dict[str, Any]) -> list[dict[str, Any]]:
    assignments = plan.get("assignments") if isinstance(plan, dict) else []
    return [item for item in assignments if isinstance(item, dict)]


def _assignment_key(assignment: dict[str, Any]) -> tuple[str, str]:
    return (str(assignment.get("subtask_id")), str(assignment.get("capability")))


def _candidate_agent_ids(assignment: dict[str, Any]) -> set[str]:
    candidates = assignment.get("candidate_agents") or []
    return {
        str(candidate.get("agent_id"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("agent_id")
    }


def _inputs_match(expected: dict[str, Any], predicted: dict[str, Any]) -> bool:
    expected_inputs = set(_strings(expected.get("inputs_passed")))
    predicted_inputs = set(_strings(predicted.get("inputs_passed")))
    if not expected_inputs:
        return True
    return expected_inputs <= predicted_inputs


def _notes(
    expected_assignments: list[dict[str, Any]],
    predicted_assignments: list[dict[str, Any]],
    expected_decision: str,
    predicted_decision: str,
    failures: list[str],
) -> list[str]:
    if not failures:
        return []
    return [
        f"expected_decision={expected_decision}, predicted_decision={predicted_decision}",
        f"expected_agents={[item.get('assigned_agent') for item in expected_assignments]}",
        f"predicted_agents={[item.get('assigned_agent') for item in predicted_assignments]}",
    ]


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _rate_count(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


def _bool_rate(cases: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for case in cases if case.get(field)) / len(cases), 4)


def _avg(cases: list[dict[str, Any]], field: str) -> float:
    return round(sum(float(case.get(field, 0)) for case in cases) / len(cases), 4)
