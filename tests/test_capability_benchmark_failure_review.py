from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_capability_benchmark_failures.py"
SPEC = importlib.util.spec_from_file_location("review_capability_benchmark_failures", SCRIPT)
assert SPEC and SPEC.loader
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


def test_build_review_scaffold_keeps_manual_owner_blank():
    rows = [
        {
            "id": "ok",
            "category": "general",
            "request": "Explain recursion.",
            "main_failure_type": "",
            "notes": "",
        },
        {
            "id": "bad",
            "category": "current_facts",
            "request": "What is the latest model?",
            "main_failure_type": "missed_current_info",
            "notes": "run2 missing external action 'web_search'",
        },
    ]

    review_rows, summary = reviewer.build_review(rows)

    assert summary["case_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["failure_counts"] == {"missed_current_info": 1}
    assert review_rows[0]["suggested_owner"] == "needs_manual_review"
    assert review_rows[0]["final_owner"] == ""
