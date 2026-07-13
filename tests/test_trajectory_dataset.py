from taskdecomp.trajectory_dataset import (
    make_code_record,
    make_mind2web_record,
    record_to_annotation_row,
    record_to_sft_row,
)


def test_mind2web_record_uses_action_sequence_as_decomposition():
    row = {
        "website": "example",
        "domain": "Travel",
        "subdomain": "Restaurant",
        "annotation_id": "abc",
        "confirmed_task": "Find a pickup restaurant for one guest",
        "action_reprs": [
            "[combobox] Reservation type -> SELECT: Pickup",
            "[searchbox] Find a location -> TYPE: Boston",
            "[button] Update search -> CLICK",
        ],
    }

    record = make_mind2web_record(row, 0)

    assert record["id"] == "mind2web:abc"
    assert record["verification"]["status"] == "success"
    assert [step["text"] for step in record["target"]["subtasks"]] == [
        "Set Reservation type to Pickup.",
        "Type Boston into Find a location.",
        "Click Update search.",
    ]
    assert record["target"]["dependencies"] == [
        {"before": "s1", "after": "s2"},
        {"before": "s2", "after": "s3"},
    ]


def test_code_record_marks_unresolved_trace_for_annotation():
    row = {
        "trajectory_id": "traj-1",
        "instance_id": "repo-1",
        "repo": "owner/repo",
        "trajectory": [
            {
                "role": "user",
                "content": "Consider the following issue description:\n<issue_description>Fix bad date parsing.</issue_description>",
            },
            {"role": "assistant", "content": "I will inspect the parser.\n```bash\nrg parse_date\n```"},
            {"role": "tool", "content": "tests failed with AssertionError"},
        ],
        "model_patch": "diff --git a/pkg/date.py b/pkg/date.py\n--- a/pkg/date.py\n+++ b/pkg/date.py\n",
        "exit_status": "submit",
        "resolved": 0,
        "gen_tests_correct": 1.0,
        "pred_passes_gen_tests": 0.0,
    }

    record = make_code_record("openhands", row, 0, max_trace_events=8, max_text_chars=300)
    sft = record_to_sft_row(record)
    annotation = record_to_annotation_row(record)

    assert record["task"] == "Fix bad date parsing."
    assert record["verification"]["status"] == "failure"
    assert record["failure_analysis"]["needs_annotation"] is True
    assert record["target"]["subtasks"][0]["id"] == "s1"
    assert "messages" in sft
    assert annotation["annotation_schema"]["bad_decomposition"]
