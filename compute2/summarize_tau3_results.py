#!/usr/bin/env python3
"""Summarize tau2/tau3 simulation result JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def reward_value(simulation: dict[str, Any]) -> float:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward", 0.0)
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    simulations = data.get("simulations") or []
    rewards = [reward_value(sim) for sim in simulations]
    terminations: dict[str, int] = {}
    for sim in simulations:
        reason = str(sim.get("termination_reason") or "unknown")
        terminations[reason] = terminations.get(reason, 0) + 1

    agent_info = (data.get("info") or {}).get("agent_info") or {}
    user_info = (data.get("info") or {}).get("user_info") or {}
    environment_info = (data.get("info") or {}).get("environment_info") or {}
    return {
        "path": str(path),
        "domain": environment_info.get("domain_name"),
        "agent": agent_info.get("implementation"),
        "agent_llm": agent_info.get("llm"),
        "user": user_info.get("implementation"),
        "user_llm": user_info.get("llm"),
        "n": len(simulations),
        "avg_reward": round(mean(rewards), 4) if rewards else 0.0,
        "success_rate": round(sum(1 for reward in rewards if reward >= 1.0) / len(rewards), 4)
        if rewards
        else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "terminations": terminations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of a table.")
    args = parser.parse_args()

    rows = [summarize(path) for path in args.paths]
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0

    headers = ["n", "avg_reward", "success_rate", "domain", "agent_llm", "user_llm", "path"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
