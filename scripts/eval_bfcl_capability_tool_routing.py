#!/usr/bin/env python3
"""Evaluate capability-planner-only BFCL tool routing.

This intentionally ignores arguments and BFCL call serialization. It measures
only whether a planner-only projection chooses the right tool names, number of
calls, and order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from taskdecomp.tool_binding import build_tool_binding_plan

try:
    from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
except ImportError:  # pragma: no cover
    from compute2.bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id


STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "help",
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
    "provide",
    "show",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "want",
    "what",
    "with",
    "you",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[_\.\-/]+", " ", text)


def toks(text: Any) -> list[str]:
    raw = split_identifier(str(text or "")).lower()
    return [tok for tok in re.findall(r"[a-z0-9]{2,}", raw) if tok not in STOPWORDS]


def tool_schema_text(tool: dict[str, Any]) -> str:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    parts = [str(tool.get("name") or ""), str(tool.get("description") or "")]
    for key, spec in props.items():
        parts.append(str(key))
        if isinstance(spec, dict):
            parts.append(str(spec.get("description") or ""))
            enum = spec.get("enum")
            if isinstance(enum, list):
                parts.extend(str(item) for item in enum)
    return " ".join(parts)


def bm25_scores(prompt: str, tools: list[dict[str, Any]]) -> list[tuple[str, float]]:
    query = toks(prompt)
    if not query:
        return [(str(tool.get("name") or ""), 0.0) for tool in tools]
    docs = [toks(tool_schema_text(tool)) for tool in tools]
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))
    avg_len = sum(len(doc) for doc in docs) / max(1, len(docs))
    k1 = 1.4
    b = 0.75
    scored = []
    for tool, doc in zip(tools, docs):
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / max(1.0, avg_len))
            score += idf * tf[term] * (k1 + 1) / denom
        name_bonus = len(set(query) & set(toks(tool.get("name") or ""))) * 0.75
        scored.append((str(tool.get("name") or ""), score + name_bonus))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def expected_names(answer_row: dict[str, Any]) -> list[str]:
    names = []
    for call in answer_row.get("ground_truth") or []:
        if isinstance(call, dict) and call:
            names.append(str(next(iter(call.keys()))))
    return names


def prompt_text(record: dict[str, Any]) -> str:
    return latest_user_turn(extract_prompt(record).strip())


def latest_user_turn(text: str) -> str:
    """Return the latest user turn from chat-style benchmark prompt text."""
    role_pattern = re.compile(r"(?im)^(system|user|assistant|tool):\s*")
    matches = list(role_pattern.finditer(text))
    if not matches:
        return re.sub(r"^user:\s*", "", text.strip(), flags=re.I)

    last_user: str | None = None
    for index, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if role == "user" and segment:
            last_user = segment
    return last_user if last_user is not None else text[matches[-1].end() :].strip()


def count_hint_from_prompt(prompt: str, expected_len_cap: int = 8) -> int:
    lowered = prompt.lower()
    strong_multi = re.search(
        r"\b(for each|for both|each of|both of|respectively|separately|simultaneously|for every)\b",
        lowered,
    )
    if not strong_multi:
        return 1
    quoted = re.findall(r"['\"][^'\"]+['\"]", prompt)
    if len(quoted) >= 2:
        return min(expected_len_cap, len(quoted))
    return 2


def predict_names(prompt: str, tools: list[dict[str, Any]], no_call_threshold: float) -> tuple[list[str], dict[str, Any]]:
    # no_call_threshold is retained for CLI compatibility with earlier runs.
    del no_call_threshold
    plan = build_tool_binding_plan(prompt, tools)
    predicted = [
        str(call.get("tool_name") or "")
        for call in plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]
    ranked = bm25_scores(prompt, tools)
    best_name, best_score = ranked[0] if ranked else ("", 0.0)
    diagnostics = {
        "task_route": {
            "route": plan.get("tool_decision"),
            "operation": "tool_binding",
            "source_requirement": "tool_schema",
        },
        "missing_inputs": plan.get("missing_inputs") or [],
        "best_tool": best_name,
        "best_score": round(best_score, 4),
        "ranked_tools": [{"name": name, "score": round(score, 4)} for name, score in ranked[:5]],
        "predicted_count": len(predicted),
        "tool_binding_plan": plan,
    }
    return predicted, diagnostics


def multiset_equal(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row.get("id")): row for row in read_jsonl(path)}


def evaluate_file(input_path: Path, answer_path: Path, limit: int, no_call_threshold: float) -> dict[str, Any]:
    records = load_records(str(input_path))
    if limit:
        records = records[:limit]
    answers = load_answers(answer_path)

    review_rows = []
    counts = Counter()
    total = 0
    expected_call_total = 0
    for index, record in enumerate(records):
        rid = record_id(record, index)
        answer = answers.get(rid, {"ground_truth": []})
        expected = expected_names(answer)
        if expected:
            expected_call_total += 1
        prompt = prompt_text(record)
        tools = normalize_tools(record)
        predicted, diagnostics = predict_names(prompt, tools, no_call_threshold)
        total += 1

        tool_set_ok = set(predicted) == set(expected)
        tool_multiset_ok = multiset_equal(predicted, expected)
        count_ok = len(predicted) == len(expected)
        ordered_ok = predicted == expected
        unordered_ok = tool_multiset_ok
        top1_tool_ok = bool(expected) and bool(predicted) and predicted[0] in set(expected)
        no_call_ok = not predicted and not expected

        counts["tool_set_ok"] += int(tool_set_ok)
        counts["tool_multiset_ok"] += int(tool_multiset_ok)
        counts["count_ok"] += int(count_ok)
        counts["ordered_ok"] += int(ordered_ok)
        counts["unordered_ok"] += int(unordered_ok)
        counts["top1_tool_ok"] += int(top1_tool_ok)
        counts["no_call_ok"] += int(no_call_ok)

        if not ordered_ok:
            review_rows.append(
                {
                    "id": rid,
                    "prompt": prompt,
                    "expected": json.dumps(expected, ensure_ascii=False),
                    "predicted": json.dumps(predicted, ensure_ascii=False),
                    "tool_set_ok": tool_set_ok,
                    "tool_multiset_ok": tool_multiset_ok,
                    "count_ok": count_ok,
                    "ordered_ok": ordered_ok,
                    "best_tool": diagnostics["best_tool"],
                    "best_score": diagnostics["best_score"],
                    "route": diagnostics["task_route"].get("route"),
                    "operation": diagnostics["task_route"].get("operation"),
                    "source_requirement": diagnostics["task_route"].get("source_requirement"),
                    "missing_inputs": json.dumps(diagnostics["missing_inputs"], ensure_ascii=False),
                    "ranked_tools": json.dumps(diagnostics["ranked_tools"], ensure_ascii=False),
                }
            )

    metrics = {
        "category": input_path.stem.replace("BFCL_v4_", ""),
        "input": str(input_path),
        "answers": str(answer_path),
        "total": total,
        "tool_set_accuracy": counts["tool_set_ok"] / total if total else 0.0,
        "tool_multiset_accuracy": counts["tool_multiset_ok"] / total if total else 0.0,
        "count_accuracy": counts["count_ok"] / total if total else 0.0,
        "ordered_sequence_accuracy": counts["ordered_ok"] / total if total else 0.0,
        "unordered_sequence_accuracy": counts["unordered_ok"] / total if total else 0.0,
        "top1_tool_hit_rate_when_called": counts["top1_tool_ok"] / max(1, expected_call_total),
        "no_call_accuracy_count": counts["no_call_ok"],
        "failure_count": len(review_rows),
    }
    return {"metrics": metrics, "review_rows": review_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--categories", nargs="+", default=["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--no-call-threshold", type=float, default=0.35)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_metrics = []
    all_review_rows = []
    for category in args.categories:
        input_path = data_dir / f"BFCL_v4_{category}.json"
        answer_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
        result = evaluate_file(input_path, answer_path, args.limit, args.no_call_threshold)
        all_metrics.append(result["metrics"])
        for row in result["review_rows"]:
            row["category"] = category
            all_review_rows.append(row)

    total = sum(item["total"] for item in all_metrics)
    aggregate = {
        "total": total,
        "tool_set_accuracy": sum(item["tool_set_accuracy"] * item["total"] for item in all_metrics) / total,
        "tool_multiset_accuracy": sum(item["tool_multiset_accuracy"] * item["total"] for item in all_metrics) / total,
        "count_accuracy": sum(item["count_accuracy"] * item["total"] for item in all_metrics) / total,
        "ordered_sequence_accuracy": sum(item["ordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total,
        "unordered_sequence_accuracy": sum(item["unordered_sequence_accuracy"] * item["total"] for item in all_metrics) / total,
    }
    payload = {"aggregate": aggregate, "by_category": all_metrics}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "category",
        "id",
        "prompt",
        "expected",
        "predicted",
        "tool_set_ok",
        "tool_multiset_ok",
        "count_ok",
        "ordered_ok",
        "best_tool",
        "best_score",
        "route",
        "operation",
        "source_requirement",
        "missing_inputs",
        "ranked_tools",
    ]
    with Path(args.review_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_review_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
