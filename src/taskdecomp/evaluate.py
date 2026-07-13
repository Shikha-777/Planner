from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - exercised only in minimal local environments.
    rouge_scorer = None


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def rouge_f(a: str, b: str, metric: str = "rougeL") -> float:
    if rouge_scorer is None:
        return rouge_l_fallback(a, b)
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    return scorer.score(a, b)[metric].fmeasure


def rouge_l_fallback(a: str, b: str) -> float:
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    if not a_tokens or not b_tokens:
        return 0.0
    dp = [[0] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]
    for i, atok in enumerate(a_tokens, 1):
        for j, btok in enumerate(b_tokens, 1):
            if atok == btok:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(a_tokens)
    recall = lcs / len(b_tokens)
    return fbeta(precision, recall, beta=1.0)


def match_steps(pred_steps: list[str], gold_steps: list[str]) -> tuple[list[tuple[int, int]], float, float]:
    if not pred_steps and not gold_steps:
        return [], 1.0, 1.0
    if not pred_steps or not gold_steps:
        return [], 0.0, 0.0
    sim = np.zeros((len(pred_steps), len(gold_steps)))
    for i, pred in enumerate(pred_steps):
        for j, gold in enumerate(gold_steps):
            sim[i, j] = rouge_f(pred, gold)
    rows, cols = linear_sum_assignment(-sim)
    matched = [(int(i), int(j)) for i, j in zip(rows, cols) if sim[i, j] >= 0.35]
    precision = len(matched) / len(pred_steps)
    recall = len(matched) / len(gold_steps)
    return matched, precision, recall


def edge_f1(pred: dict[str, Any], gold: dict[str, Any], matches: list[tuple[int, int]]) -> float:
    pred_steps = pred.get("subtasks", [])
    pred_id_to_idx = {s["id"]: i for i, s in enumerate(pred_steps)}
    gold_steps = gold["subtasks"]
    gold_text_to_idx = {s: i for i, s in enumerate(gold_steps)}
    p_to_g = {p: g for p, g in matches}

    pred_edges = set()
    for dep in pred.get("dependencies", []):
        a = pred_id_to_idx.get(dep.get("before"))
        b = pred_id_to_idx.get(dep.get("after"))
        if a in p_to_g and b in p_to_g:
            pred_edges.add((p_to_g[a], p_to_g[b]))

    gold_edges = {
        (gold_text_to_idx[a], gold_text_to_idx[b])
        for a, b in gold.get("dependencies", [])
        if a in gold_text_to_idx and b in gold_text_to_idx
    }
    if not pred_edges and not gold_edges:
        return 1.0
    if not pred_edges or not gold_edges:
        return 0.0
    tp = len(pred_edges & gold_edges)
    precision = tp / len(pred_edges)
    recall = tp / len(gold_edges)
    return fbeta(precision, recall, beta=1.0)


def evaluate(predictions: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, float]:
    scores = []
    gate_errors = 0
    for pred, ref in zip(predictions, references):
        gold = ref["target"]
        if pred.get("decision") != gold.get("decision", "decompose"):
            gate_errors += 1
        pred_steps = [s["text"] for s in pred.get("subtasks", [])]
        gold_steps = gold.get("subtasks", [])
        matches, precision, recall = match_steps(pred_steps, gold_steps)
        scores.append(
            {
                "hungarian_f1": fbeta(precision, recall, 1.0),
                "hungarian_f2": fbeta(precision, recall, 2.0),
                "edge_f1": edge_f1(pred, gold, matches),
            }
        )
    return {
        "n": float(len(scores)),
        "gate_accuracy": 1.0 - gate_errors / max(len(scores), 1),
        "hungarian_f1": float(np.mean([s["hungarian_f1"] for s in scores])),
        "hungarian_f2": float(np.mean([s["hungarian_f2"] for s in scores])),
        "edge_f1": float(np.mean([s["edge_f1"] for s in scores])),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=Path("data/processed/test.eval.jsonl"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    metrics = evaluate(read_jsonl(args.predictions), read_jsonl(args.references))
    text = json.dumps(metrics, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
