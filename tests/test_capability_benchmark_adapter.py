from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adapt_capability_benchmark.py"
SPEC = importlib.util.spec_from_file_location("adapt_capability_benchmark", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_superni_task_expands_instances_and_marks_url_as_source_text():
    raw = {
        "Definition": ["Write a short summary of the provided text."],
        "Categories": ["Summarization"],
        "Instances": [
            {
                "id": "url_source",
                "input": "Visit https://example.com/current/select today to choose a package.",
            }
        ],
    }

    records = adapter.flatten_json(raw)
    rows = adapter.adapt_records(records, dataset="auto")

    assert len(rows) == 1
    row = rows[0]
    assert row["source_dataset"] == "superni"
    assert row["benchmark_profile"] == "source_task"
    assert row["expected"]["run1"]["inputs"][0]["format"] == "pasted_text"
    assert row["expected"]["run2"]["needs_current_or_external_info"] is False
    assert "web_search" in row["expected"]["run2"]["must_not_include_external_action_type"]


def test_mixed_auto_records_route_by_shape():
    rows = adapter.adapt_records(
        [
            {"id": "weather", "utterance": "Will it rain in Denver tomorrow morning?"},
            {
                "id": "code",
                "question": "Write a Python function normalize_names(names).",
            },
            {
                "id": "issue",
                "problem_statement": "In src/parser.py, fix the bug and update tests.",
            },
        ],
        dataset="auto",
    )

    assert [row["source_dataset"] for row in rows] == ["clinc", "bfcl", "swe_bench"]
    assert rows[0]["expected"]["run2"]["must_include_external_action_type"] == ["web_search"]
    assert rows[1]["expected"]["run3"]["must_include_capability"] == ["generate_code"]
    assert "file_reading" in rows[2]["expected"]["run2"]["must_include_external_action_type"]


def test_intent_adapter_handles_public_intent_edge_cases():
    cases = [
        ("define elaborate", "general_qa", "provide_explanation", ["none"]),
        ("what's the week's forecast", "current_facts", "retrieve_current_information", ["web_search"]),
        ("what is the current temperature outside", "current_facts", "retrieve_current_information", ["web_search"]),
        ("is it raining", "current_facts", "retrieve_current_information", ["web_search"]),
        ("is it currently snowing", "current_facts", "retrieve_current_information", ["web_search"]),
        ("is it going to be windy tomorrow", "current_facts", "retrieve_current_information", ["web_search"]),
        ("should i bring an umbrella tomorrow", "current_facts", "retrieve_current_information", ["web_search"]),
        ("how cold will it get each night this week", "current_facts", "retrieve_current_information", ["web_search"]),
        ("what is the exchange rate between us dollars and japanese yen", "current_facts", "retrieve_current_information", ["web_search"]),
        ("were the stocks rising or declining", "current_facts", "retrieve_current_information", ["web_search"]),
        ("what is the current share value for m. s. n. b. c.", "current_facts", "retrieve_current_information", ["web_search"]),
        ("what were the top stories this week", "current_facts", "retrieve_current_information", ["web_search"]),
        ("please find today's most read stories from the new york times", "current_facts", "retrieve_current_information", ["web_search"]),
        ("look up the definition of blunder", "general_qa", "provide_explanation", ["none"]),
        ("how do i compute the median of a set of numbers", "general_qa", "provide_explanation", ["none"]),
        ("how would they say butter in zambia", "translation", "translate_text", ["none"]),
        ("how does one say wonderful in german", "translation", "translate_text", ["none"]),
        ("could you translate this into chinese for me, please", "translation", "translate_text", ["none"]),
        ("what is the right way to say excuse me in spanish", "translation", "translate_text", ["none"]),
        ("reply to dad write i will be late", "text_generation", "draft_text", ["none"]),
        ("drop a message to jan write i will be late", "text_generation", "draft_text", ["none"]),
        ("send this email to reply to the latest emails", "missing_input", "request_missing_input", ["user_input"]),
        ("Can you polish my LinkedIn bio?", "missing_input", "request_missing_input", ["user_input"]),
        ("Graph monthly signups by acquisition channel.", "missing_input", "request_missing_input", ["user_input"]),
        ("Plot revenue by region.", "missing_input", "request_missing_input", ["user_input"]),
    ]

    for request, category, capability, actions in cases:
        actual_category, exp, _ = adapter.expected_for_request(
            request,
            dataset="massive",
            profile="intent",
        )

        assert actual_category == category
        assert exp["run3"]["must_include_capability"] == [capability]
        assert exp["run2"]["must_include_external_action_type"] == actions


def test_webarena_url_values_do_not_force_comparison():
    category, exp, _ = adapter.expected_for_request(
        "Open http://gitlab.com/ and complete this web task: set the homepage URL on my GitLab profile to https://egg.tart.com",
        dataset="webarena",
        profile="web",
    )

    assert category == "url"
    assert exp["run3"]["must_include_capability"] == ["retrieve_external_information"]
    assert exp["graph"]["must_precede"] == []
