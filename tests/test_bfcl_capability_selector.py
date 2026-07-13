from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def load_selector(monkeypatch):
    bfcl = types.ModuleType("bfcl_compare_eval")
    bfcl.extract_prompt = lambda record: record.get("prompt", "")
    bfcl.load_records = lambda path: []
    bfcl.normalize_tools = lambda record, function_doc_dir=None: []
    bfcl.record_id = lambda record, index: str(record.get("id", index))

    smoke = types.ModuleType("smoke_eval")
    smoke.dedupe_tool_calls = lambda calls: calls
    smoke.openai_tools = lambda tools: tools

    monkeypatch.setitem(sys.modules, "bfcl_compare_eval", bfcl)
    monkeypatch.setitem(sys.modules, "smoke_eval", smoke)

    script = Path(__file__).resolve().parents[1] / "compute2" / "eval_bfcl_ensemble_select.py"
    spec = importlib.util.spec_from_file_location("bfcl_selector_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def weather_tool():
    return {
        "name": "get_weather",
        "description": "Get the current weather forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    }


def text_tool():
    return {
        "name": "improve_text",
        "description": "Improve provided essay or resume text.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to improve"}},
            "required": ["text"],
        },
    }


def itinerary_tool():
    return {
        "name": "travel_itinerary_generator",
        "description": "Generate a travel itinerary for a destination, duration, budget, and exploration style.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "travel destination"},
                "days": {"type": "integer", "description": "number of travel days"},
                "daily_budget": {"type": "number", "description": "daily budget in dollars"},
                "exploration_type": {"type": "string", "description": "preferred exploration type"},
            },
            "required": ["destination", "days", "daily_budget", "exploration_type"],
        },
    }


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_capability_hint_marks_self_contained_irrelevance_as_no_tool(monkeypatch):
    selector = load_selector(monkeypatch)
    hint = selector.build_capability_selection_hint(
        "Write a haiku about winter.",
        [weather_tool()],
        "auto",
    )

    assert hint["decision"] == "no_tool_expected"
    assert hint["influence_selection"] is False
    assert hint["route"] == "text_generation"


def test_auto_capability_hint_is_diagnostic_only_for_irrelevant_prompt(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [weather_tool()]
    prompt = "Write a haiku about winter."
    hint = selector.build_capability_selection_hint(prompt, tools, "auto")
    candidates = [
        {"source": "xlam", "calls": [call("get_weather", city="Paris")], "issues": []},
        {"source": "toolace", "calls": [], "issues": []},
    ]

    baseline_selected, _ = selector.select_candidate(
        [dict(item) for item in candidates],
        tools,
        prompt,
        None,
        None,
    )
    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == baseline_selected
    assert diag["capability_planner"]["decision"] == "no_tool_expected"
    assert diag["capability_planner"]["influence_selection"] is False
    assert not diag.get("capability_no_call_gate")


def test_forced_capability_hint_prefers_empty_candidate_for_irrelevant_prompt(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [weather_tool()]
    prompt = "Write a haiku about winter."
    hint = selector.build_capability_selection_hint(prompt, tools, "on")
    candidates = [
        {"source": "xlam", "calls": [call("get_weather", city="Paris")], "issues": []},
        {"source": "toolace", "calls": [], "issues": []},
    ]

    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == []
    assert diag["selected_source"] == "capability_no_call_gate:toolace"
    assert diag["capability_planner"]["decision"] == "no_tool_expected"
    assert diag["capability_planner"]["influence_selection"] is True


def test_capability_hint_does_not_block_clear_tool_call(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [weather_tool()]
    prompt = "What is the weather in Paris?"
    hint = selector.build_capability_selection_hint(prompt, tools, "auto")
    candidates = [
        {"source": "xlam", "calls": [call("get_weather", city="Paris")], "issues": []},
        {"source": "toolace", "calls": [call("get_weather", city="Paris")], "issues": []},
        {"source": "taskbench", "calls": [], "issues": []},
    ]

    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == [call("get_weather", city="Paris")]
    assert diag["selected_source"] == "xlam"
    assert diag["capability_planner"]["decision"] == "tool_expected"
    assert not diag.get("no_call_gate")


def test_capability_hint_prefers_empty_candidate_for_missing_source_input(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [text_tool()]
    prompt = "Can you make my essay better?"
    hint = selector.build_capability_selection_hint(prompt, tools, "on")
    candidates = [
        {"source": "xlam", "calls": [call("improve_text", text="my essay")], "issues": []},
        {"source": "toolace", "calls": [], "issues": []},
    ]

    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == []
    assert hint["decision"] == "ask_user_expected"
    assert diag["selected_source"] == "capability_no_call_gate:toolace"


def test_capability_hint_cannot_veto_reliable_tool_call_consensus(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [itinerary_tool()]
    prompt = "Generate a 7-day Tokyo itinerary with a daily budget of $100 focused on nature."
    good_call = call(
        "travel_itinerary_generator",
        destination="Tokyo",
        days=7,
        daily_budget=100,
        exploration_type="nature",
    )
    hint = {
        "enabled": True,
        "decision": "ask_user_expected",
        "reason": "capability planner found missing user input",
        "missing_inputs": ["budget data"],
        "multi_call_expected": False,
    }
    candidates = [
        {"source": "toolace", "calls": [good_call], "issues": []},
        {"source": "xlam", "calls": [good_call], "issues": []},
        {"source": "gptoss_apigen", "calls": [good_call], "issues": []},
        {"source": "taskbench", "calls": [], "issues": []},
    ]

    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == [good_call]
    assert diag["selected_source"] == "xlam"
    assert not diag.get("capability_no_call_gate")


def test_capability_hint_bonuses_multi_call_candidate_for_each_requests(monkeypatch):
    selector = load_selector(monkeypatch)
    tools = [weather_tool()]
    prompt = "Get the weather for each city: Paris and Berlin."
    hint = selector.build_capability_selection_hint(prompt, tools, "on")
    candidates = [
        {"source": "xlam", "calls": [call("get_weather", city="Paris")], "issues": []},
        {
            "source": "toolace",
            "calls": [
                call("get_weather", city="Paris"),
                call("get_weather", city="Berlin"),
            ],
            "issues": [],
        },
    ]

    selected, diag = selector.select_candidate(candidates, tools, prompt, None, hint)

    assert selected == [
        call("get_weather", city="Paris"),
        call("get_weather", city="Berlin"),
    ]
    assert diag["selected_source"] == "toolace"
    assert diag["capability_planner"]["multi_call_expected"] is True
