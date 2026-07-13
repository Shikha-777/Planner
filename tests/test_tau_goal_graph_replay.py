from __future__ import annotations

import json

from scripts.tau_goal_graph_replay import load_replay_cases, write_replay_results


def test_load_replay_cases_reads_logged_state_and_manifest(tmp_path):
    debug_path = tmp_path / "task_3.jsonl"
    (tmp_path / "goal_graph_replay_manifest.json").write_text(
        json.dumps({"goal_graph_tools": [{"name": "lookup", "parameters": {"type": "object"}}]}),
        encoding="utf-8",
    )
    debug_path.write_text(
        json.dumps(
            {
                "step": 4,
                "final_action": {"name": "lookup", "kwargs": {"id": "A"}},
                "goal_graph_result": {
                    "planning_request": "Current task: retrieve A.",
                    "binding_request": "Binding evidence: A.",
                    "stateful_execution_history": [{"tool_name": "authenticate", "outcome": "success"}],
                    "stateful_goal_ledger_input": {"goals": [{"id": "1", "status": "pending"}]},
                },
            }
        )
        + "\n"
        + json.dumps({"step": 5, "goal_graph_result": {"binding_request": "missing planning request"}})
        + "\n",
        encoding="utf-8",
    )

    tools, cases = load_replay_cases(debug_path, {4})

    assert tools[0]["name"] == "lookup"
    assert cases == [
        {
            "step": 4,
            "planning_request": "Current task: retrieve A.",
            "binding_request": "Binding evidence: A.",
            "execution_history": [{"tool_name": "authenticate", "outcome": "success"}],
            "stateful_goal_ledger": {"goals": [{"id": "1", "status": "pending"}]},
            "original_action": {"name": "lookup", "kwargs": {"id": "A"}},
        }
    ]


def test_write_replay_results_flushes_each_case_with_elapsed_time(monkeypatch, tmp_path):
    def fake_replay(_model, _tokenizer, _tools, case, _max_new_tokens):
        return {"source_step": case["step"]}

    monkeypatch.setattr("scripts.tau_goal_graph_replay.replay_case", fake_replay)
    output = tmp_path / "replay.jsonl"

    write_replay_results(None, None, [], [{"step": 4}, {"step": 7}], 900, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["source_step"] for row in rows] == [4, 7]
    assert all(isinstance(row["elapsed_seconds"], float) for row in rows)
    assert all(row["elapsed_seconds"] >= 0 for row in rows)
