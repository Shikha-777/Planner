from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_multiturn_module():
    def parse_python_calls(text):
        tree = ast.parse(str(text), mode="eval")
        calls = []
        for node in tree.body.elts:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
            calls.append({"name": node.func.id, "arguments": args})
        return calls

    selector = types.SimpleNamespace(
        maybe_json=lambda value: value,
        call_issues=lambda calls, tools: [],
    )
    sys.modules["eval_bfcl_multiturn_current_ensemble"] = types.SimpleNamespace(
        selector=selector,
        parse_python_calls=parse_python_calls,
    )
    sys.modules["bfcl_compare_eval"] = types.SimpleNamespace(normalize_tools=lambda record: record.get("function") or [])
    sys.modules["goal_graph_eval_common"] = types.SimpleNamespace(
        benchmark_compile_tools=lambda tools: tools,
        plan_and_compile_goal_graph=lambda *args, **kwargs: {"calls": []},
    )
    sys.modules["run_gptoss_capability_plan"] = types.SimpleNamespace(
        generate_text=lambda *args, **kwargs: "",
        load_model=lambda model: (None, None),
    )
    spec = importlib.util.spec_from_file_location(
        "goal_graph_bfcl_multiturn_adapter_for_test",
        ROOT / "scripts" / "eval_goal_graph_bfcl_multiturn.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_state(initial_config, *, tools=None, history=None, executed=None, current_turn_start=0, current_turn_index=0, record_extra=None):
    record = {"initial_config": initial_config}
    if record_extra:
        record.update(record_extra)
    return types.SimpleNamespace(
        record=record,
        available_tools=tools or [],
        history=history or [],
        executed_call_history=executed or [],
        current_turn_start=current_turn_start,
        current_turn_index=current_turn_index,
    )


def test_filesystem_repair_resolves_existing_plural_and_case_names_only():
    module = load_multiturn_module()
    state = make_state(
        {
            "GorillaFileSystem": {
                "root": {
                    "simona": {
                        "type": "directory",
                        "contents": {
                            "documents": {
                                "type": "directory",
                                "contents": {"report.txt": {"type": "file", "content": "x"}},
                            }
                        },
                    }
                }
            }
        }
    )
    calls = [
        {"name": "cd", "arguments": {"folder": "document"}},
        {"name": "grep", "arguments": {"file_name": "REPORT.TXT", "pattern": "x"}},
        {"name": "touch", "arguments": {"file_name": "document"}},
        {"name": "mkdir", "arguments": {"dir_name": "document"}},
    ]

    repaired = module.repair_filesystem_call_arguments(state, calls)

    assert repaired[0]["arguments"]["folder"] == "documents"
    assert repaired[1]["arguments"]["file_name"] == "report.txt"
    assert repaired[2]["arguments"]["file_name"] == "document"
    assert repaired[3]["arguments"]["dir_name"] == "document"
    assert calls[0]["arguments"]["folder"] == "document"


def test_filesystem_repair_resolves_archive_alias_when_unique():
    module = load_multiturn_module()
    state = make_state(
        {
            "GorillaFileSystem": {
                "root": {
                    "simona": {
                        "type": "directory",
                        "contents": {
                            "Archived": {"type": "directory", "contents": {}},
                        },
                    }
                }
            }
        }
    )

    repaired = module.repair_filesystem_call_arguments(
        state,
        [{"name": "cd", "arguments": {"folder": "archive"}}],
    )

    assert repaired == [{"name": "cd", "arguments": {"folder": "Archived"}}]


def test_filesystem_repair_strips_find_wildcards():
    module = load_multiturn_module()
    state = make_state({"root": {"type": "directory", "contents": {}}})

    repaired = module.repair_filesystem_call_arguments(
        state,
        [{"name": "find", "arguments": {"path": ".", "name": "*goal*"}}],
    )

    assert repaired == [{"name": "find", "arguments": {"path": ".", "name": "goal"}}]


def test_multiturn_normalizer_drops_pwd_when_listing_same_batch():
    module = load_multiturn_module()
    state = make_state({"root": {"type": "directory", "contents": {}}})

    normalized = module.normalize_multiturn_calls(
        state,
        [{"name": "pwd", "arguments": {}}, {"name": "ls", "arguments": {"a": True}}],
    )

    assert normalized == [{"name": "ls", "arguments": {"a": True}}]


def test_multiturn_normalizer_drops_existing_mkdir_before_move():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "log.txt": {"type": "file", "content": ""},
                            "archive": {"type": "directory", "contents": {}},
                        },
                    }
                },
            }
        }
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [
            {"name": "cd", "arguments": {"folder": "workspace"}},
            {"name": "mkdir", "arguments": {"dir_name": "archive"}},
            {"name": "mv", "arguments": {"source": "log.txt", "destination": "archive"}},
        ],
    )

    assert normalized == [
        {"name": "cd", "arguments": {"folder": "workspace"}},
        {"name": "mv", "arguments": {"source": "log.txt", "destination": "archive"}},
    ]


def test_multiturn_normalizer_drops_unrequested_echo_after_touch():
    module = load_multiturn_module()
    state = make_state(
        {"root": {"type": "directory", "contents": {"documents": {"type": "directory", "contents": {}}}}},
        history=[
            {
                "role": "user",
                "content": "Create a document titled 'TeamNotes.txt' for keeping track of fresh ideas.",
            }
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [
            {"name": "touch", "arguments": {"file_name": "TeamNotes.txt"}},
            {
                "name": "echo",
                "arguments": {
                    "content": "for keeping track of fresh ideas",
                    "file_name": "TeamNotes.txt",
                },
            },
        ],
    )

    assert normalized == [{"name": "touch", "arguments": {"file_name": "TeamNotes.txt"}}]


def test_multiturn_normalizer_defers_when_requested_tool_is_missing():
    module = load_multiturn_module()
    state = make_state(
        {"root": {"type": "directory", "contents": {"final_report.pdf": {"type": "file", "content": ""}}}},
        tools=[{"name": "cat"}],
        history=[
            {
                "role": "user",
                "content": "Sort the 'final_report.pdf' by line for improved clarity.",
            }
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [{"name": "cat", "arguments": {"file_name": "final_report.pdf"}}],
    )

    assert normalized == []


def test_controller_resumes_previous_request_when_missing_function_arrives():
    module = load_multiturn_module()
    state = make_state(
        {"root": {"type": "directory", "contents": {"final_report.pdf": {"type": "file", "content": ""}}}},
        tools=[{"name": "sort"}],
        history=[
            {
                "role": "user",
                "content": "Sort the 'final_report.pdf' by line for improved clarity.",
            },
            {"role": "user", "content": "You now have access to an additional function."},
        ],
        current_turn_start=1,
        current_turn_index=3,
        record_extra={"missed_function": {"3": [{"name": "sort"}]}},
    )

    candidate = module.goal_graph_controller_candidate(state)

    assert candidate["calls"] == [{"name": "sort", "arguments": {"file_name": "final_report.pdf"}}]


def test_multiturn_normalizer_repairs_echo_content_from_quoted_text():
    module = load_multiturn_module()
    state = make_state(
        {"root": {"type": "directory", "contents": {}}},
        history=[
            {
                "role": "user",
                "content": "Could you jot down 'Collaboration leads to success. Innovation ignites growth.' into the previous file?",
            }
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [
            {
                "name": "echo",
                "arguments": {"content": "Could you jot down ", "file_name": "TeamNotes.txt"},
            }
        ],
    )

    assert normalized == [
        {
            "name": "echo",
            "arguments": {
                "content": "Collaboration leads to success. Innovation ignites growth.",
                "file_name": "TeamNotes.txt",
            },
        }
    ]


def test_multiturn_normalizer_repairs_copy_to_folder_with_rename_order():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "test_document.txt": {"type": "file", "content": ""},
                    "archives": {"type": "directory", "contents": {}},
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "cp"}, {"name": "mv"}],
        history=[
            {
                "role": "user",
                "content": "Transfer a duplicate of 'test_document.txt' over to the archives folder and rename it 'final_document.txt'.",
            }
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [
            {"name": "cd", "arguments": {"folder": "archives"}},
            {"name": "cp", "arguments": {"source": "test_document.txt", "destination": "final_document.txt"}},
        ],
    )

    assert normalized == [
        {"name": "cp", "arguments": {"source": "test_document.txt", "destination": "archives"}},
        {"name": "cd", "arguments": {"folder": "archives"}},
        {"name": "mv", "arguments": {"source": "test_document.txt", "destination": "final_document.txt"}},
    ]


def test_multiturn_normalizer_prepends_cd_for_file_read_after_move_to_folder():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "archive": {"type": "directory", "contents": {}},
                        },
                    }
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "grep"}],
        history=[{"role": "user", "content": "Investigate within 'log.txt' for the keyword Error."}],
        executed=[
            ["cd(folder='workspace')", "mv(source='log.txt', destination='archive')"],
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [{"name": "grep", "arguments": {"file_name": "log.txt", "pattern": "Error"}}],
    )

    assert normalized == [
        {"name": "cd", "arguments": {"folder": "archive"}},
        {"name": "grep", "arguments": {"file_name": "log.txt", "pattern": "Error"}},
    ]


def test_multiturn_normalizer_uses_prior_sorted_result_as_text_body():
    module = load_multiturn_module()
    state = make_state(
        {"root": {"type": "directory", "contents": {}}},
        tools=[{"name": "post_tweet"}],
        history=[
            {
                "role": "tool",
                "content": '{"sorted_content":"Initial report content More unsorted data Unsorted data"}',
            },
            {
                "role": "user",
                "content": (
                    "Share the sorted result as the message body on social media, "
                    "tagging currenttechtrend and mentioning Julia."
                ),
            },
        ],
        current_turn_start=1,
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [
            {
                "name": "post_tweet",
                "arguments": {
                    "content": "Initial report content Unsorted data More unsorted data",
                    "tags": ["#currenttechtrend"],
                    "mentions": ["@Julia"],
                },
            }
        ],
    )

    assert normalized == [
        {
            "name": "post_tweet",
            "arguments": {
                "content": "Initial report content More unsorted data Unsorted data",
                "tags": ["#currenttechtrend"],
                "mentions": ["@Julia"],
            },
        }
    ]


def test_controller_expands_copy_all_text_files_from_directory():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "Quarter1_Reports": {
                        "type": "directory",
                        "contents": {
                            "report1.txt": {"type": "file", "content": ""},
                            "report2.txt": {"type": "file", "content": ""},
                            "MonthlySummary.docx": {"type": "file", "content": ""},
                            "Archived_Quarter1": {"type": "directory", "contents": {}},
                        },
                    }
                },
            }
        },
        tools=[{"name": "mkdir"}, {"name": "cp"}],
        history=[
            {
                "role": "user",
                "content": "Copy all the text files of the 'Quarter1_Reports' directory and place it in a new directory naming it 'Archived_Quarter1'.",
            }
        ],
    )

    candidate = module.goal_graph_controller_candidate(state)

    assert candidate["calls"] == [
        {"name": "cp", "arguments": {"source": "report1.txt", "destination": "Archived_Quarter1"}},
        {"name": "cp", "arguments": {"source": "report2.txt", "destination": "Archived_Quarter1"}},
    ]


def test_multiturn_normalizer_turns_unasked_file_probe_into_folder_cd():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "Documentation": {
                        "type": "directory",
                        "contents": {"FinalReport.txt": {"type": "file", "content": ""}},
                    }
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "find"}],
        history=[
            {
                "role": "user",
                "content": "I remember I should have a document called 'FinalReport.txt' in the 'Documentation' folder.",
            }
        ],
    )

    normalized = module.normalize_multiturn_calls(
        state,
        [{"name": "find", "arguments": {"path": "Documentation", "name": "FinalReport.txt"}}],
    )

    assert normalized == [{"name": "cd", "arguments": {"folder": "Documentation"}}]


def test_controller_copies_prior_find_matches_one_directory_level_at_a_time():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "projects": {
                        "type": "directory",
                        "contents": {
                            "photography": {
                                "type": "directory",
                                "contents": {
                                    "backup_tests": {"type": "directory", "contents": {}},
                                    "test_image1.jpg": {"type": "file", "content": ""},
                                    "test_document.txt": {"type": "file", "content": ""},
                                },
                            }
                        },
                    }
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "cp"}],
        history=[
            {
                "role": "user",
                "content": "After identifying them, copy the images and text files into a backup_tests folder.",
            },
            {
                "role": "tool",
                "content": '{"matches":["./projects/photography/test_image1.jpg","./projects/photography/test_document.txt","./projects/photography/backup_tests"]}',
            },
        ],
        current_turn_start=0,
    )

    candidate = module.goal_graph_controller_candidate(state)

    assert candidate["calls"] == [
        {"name": "cd", "arguments": {"folder": "projects"}},
        {"name": "cd", "arguments": {"folder": "photography"}},
        {"name": "cp", "arguments": {"source": "test_image1.jpg", "destination": "backup_tests"}},
        {"name": "cp", "arguments": {"source": "test_document.txt", "destination": "backup_tests"}},
    ]


def test_controller_uses_later_action_destination_for_move_after_navigation():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "alex": {
                        "type": "directory",
                        "contents": {
                            "workspace": {
                                "type": "directory",
                                "contents": {
                                    "log.txt": {"type": "file", "content": ""},
                                    "archive": {"type": "directory", "contents": {}},
                                },
                            }
                        },
                    }
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "mv"}],
        history=[
            {
                "role": "user",
                "content": "Go to workspace directory and move one of the 'log.txt' files into a new directory 'archive'.",
            }
        ],
    )

    candidate = module.goal_graph_controller_candidate(state)

    assert candidate["calls"] == [
        {"name": "cd", "arguments": {"folder": "workspace"}},
        {"name": "mv", "arguments": {"source": "log.txt", "destination": "archive"}},
    ]


def test_controller_preserves_requested_rename_case():
    module = load_multiturn_module()
    state = make_state(
        {
            "root": {
                "type": "directory",
                "contents": {
                    "Documentation": {
                        "type": "directory",
                        "contents": {
                            "FinalReport.txt": {"type": "file", "content": ""},
                            "Archives": {"type": "directory", "contents": {}},
                        },
                    }
                },
            }
        },
        tools=[{"name": "cd"}, {"name": "cp"}, {"name": "mv"}],
        history=[
            {
                "role": "user",
                "content": (
                    "Make a copy of 'FinalReport.txt' to the 'Archives' directory inside the "
                    "'Documentation' folder, while ensuring the duplicate is preserved as "
                    "'ArchivedFinalReport2024.txt'."
                ),
            }
        ],
        current_turn_start=0,
    )

    candidate = module.goal_graph_controller_candidate(state)

    assert candidate["calls"][-1] == {
        "name": "mv",
        "arguments": {"source": "FinalReport.txt", "destination": "ArchivedFinalReport2024.txt"},
    }
