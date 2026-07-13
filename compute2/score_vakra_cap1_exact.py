#!/usr/bin/env python3
"""Deterministic development scorer for VAKRA/Live API-Bench capability 1."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def norm_scalar(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(sorted(norm_scalar(v) for v in value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).lower()
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    return text


def split_items(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {norm_scalar(v) for v in value if norm_scalar(v)}
    text = norm_scalar(value)
    if not text:
        return set()
    if "|" in text:
        return {x.strip() for x in text.split("|") if x.strip()}
    if "," in text:
        return {x.strip() for x in text.split(",") if x.strip()}
    return {text}


def answer_match(gold: Any, pred: Any) -> bool:
    gold_norm = norm_scalar(gold)
    pred_norm = norm_scalar(pred)
    if gold_norm == pred_norm:
        return True
    gold_items = split_items(gold)
    pred_items = split_items(pred)
    return bool(gold_items) and gold_items == pred_items


def calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    seq = turn.get("sequence") or {}
    raw = seq.get("tool_call") or []
    return [c for c in raw if c.get("name") not in {"initialize_active_data", "get_data"}]


def tool_name_score(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]) -> tuple[bool, float]:
    gold_names = [c.get("name") for c in gold_calls]
    pred_names = [c.get("name") for c in pred_calls]
    exact = gold_names == pred_names
    if not gold_names and not pred_names:
        return exact, 1.0
    matched = sum(1 for g, p in zip(gold_names, pred_names) if g == p)
    denom = max(len(gold_names), len(pred_names), 1)
    return exact, matched / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gold_records = json.loads(args.gold.read_text(encoding="utf-8"))
    pred_records = json.loads(args.pred.read_text(encoding="utf-8")) if args.pred.exists() else []
    pred_by_uuid = {str(r.get("uuid")): r for r in pred_records}

    details = []
    answer_correct = 0
    status_success = 0
    exact_tool_names = 0
    tool_name_scores = []
    missing = 0

    for gold in gold_records:
        uuid = str(gold.get("uuid"))
        if uuid not in pred_by_uuid:
            missing += 1
            continue
        gold_turn = (gold.get("output") or [{}])[0]
        pred = pred_by_uuid[uuid]
        pred_turn = ((pred or {}).get("output") or [{}])[0]
        status = (pred or {}).get("status", "missing")
        if status == "success":
            status_success += 1

        answer_ok = bool(pred) and answer_match(gold_turn.get("answer"), pred_turn.get("answer"))
        answer_correct += int(answer_ok)

        exact_tools, name_score = tool_name_score(calls(gold_turn), calls(pred_turn))
        exact_tool_names += int(exact_tools)
        tool_name_scores.append(name_score)
        details.append(
            {
                "uuid": uuid,
                "status": status,
                "answer_match": answer_ok,
                "tool_names_exact": exact_tools,
                "tool_name_score": name_score,
                "gold_answer": gold_turn.get("answer"),
                "pred_answer": pred_turn.get("answer"),
                "gold_tools": [c.get("name") for c in calls(gold_turn)],
                "pred_tools": [c.get("name") for c in calls(pred_turn)],
                "error": (pred or {}).get("error", "missing prediction"),
            }
        )

    evaluated = len(details)
    total_gold = len(gold_records)
    summary = {
        "total_gold": total_gold,
        "evaluated_predictions": evaluated,
        "predictions": len(pred_records),
        "missing_predictions": missing,
        "status_success": status_success,
        "answer_exact_or_set_match": answer_correct,
        "answer_accuracy_on_predictions": answer_correct / evaluated if evaluated else 0.0,
        "answer_accuracy_on_gold": answer_correct / total_gold if total_gold else 0.0,
        "tool_name_exact": exact_tool_names,
        "tool_name_exact_rate_on_predictions": exact_tool_names / evaluated if evaluated else 0.0,
        "tool_name_exact_rate_on_gold": exact_tool_names / total_gold if total_gold else 0.0,
        "tool_name_mean_position_score": sum(tool_name_scores) / evaluated if evaluated else 0.0,
    }
    report = {"summary": summary, "details": details}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
