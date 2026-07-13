from __future__ import annotations

import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


FAILURE_PRIORITY = [
    "wrong_final_intent",
    "missed_missing_input",
    "input_audit_error",
    "invented_file_reading",
    "missed_current_info",
    "wrong_external_action",
    "used_tool_name",
    "used_agent_name",
    "vague_capability",
    "under_decomposed",
    "over_decomposed",
    "bad_dependency",
    "invalid_json",
]

TOOL_TERMS = {
    "browser_tool",
    "pdfplumber",
    "beautifulsoup",
    "selenium",
    "playwright",
    "pandas",
    "numpy",
}

AGENT_TERMS = {
    "research_agent",
    "coding_agent",
    "assistant_writer",
    "agent",
}

VAGUE_CAPABILITY_NAMES = {
    "answer_user",
    "complete_request",
    "do_task",
    "handle_task",
    "process_input",
    "process_request",
    "respond_to_user",
    "solve_problem",
    "use_information",
}

KEYWORD_ALIASES = {
    "alex extension data arrived late": ["recipient name extension duration reason"],
    "billing code": ["src billing py", "billing"],
    "compare": ["determine"],
    "describe": ["description", "alt text"],
    "failing test": ["test"],
    "json": ["json data"],
    "numeric": ["value", "result"],
    "openai model": ["openai api documentation", "model info"],
    "plan": ["itinerary"],
    "read": ["extract"],
    "revise": ["polish", "improve", "better", "make better", "clearer", "clarify"],
    "tests": ["test"],
    "two attached pdfs": ["part a pdf part b pdf"],
    "two product descriptions": ["description a description b"],
    "workbook": ["xlsx"],
}

CAPABILITY_ALIASES = {
    "analyze_provided_dataset": [
        "analyze provided dataset",
        "aggregate revenue",
        "calculate growth",
        "filter active users",
        "sort users",
    ],
    "classify_provided_text": ["classify provided text", "classify sentiment"],
    "combine_files": ["combine files", "merge pdfs"],
    "compare_texts": ["compare texts", "compare descriptions"],
    "compute_numeric_result": ["compute numeric result", "compute log", "sum numeric"],
    "draft_text": ["draft text", "write email"],
    "execute_code": ["execute code", "run test", "run test suite"],
    "extract_information_from_attached_document": [
        "extract information attached document",
        "extract text pdf",
        "read pdf",
        "read workbook",
    ],
    "extract_information_from_file": [
        "extract information file",
        "read file",
        "read repository files",
    ],
    "extract_information_from_image": [
        "extract information image",
        "read receipt",
        "extract merchant date total",
    ],
    "generate_code": ["generate code", "define function", "write python function"],
    "generate_image": ["generate image", "create app icon"],
    "inspect_existing_code": [
        "inspect existing code",
        "read parser file",
        "read billing test files",
        "read code",
    ],
    "interpret_image_content": [
        "interpret image content",
        "generate alt text",
        "describe image",
    ],
    "modify_code": [
        "modify code",
        "modify parser file",
        "modify file",
        "write modified billing file",
    ],
    "measure": [
        "measure",
        "calculate real world height",
        "calculate lamp height",
        "compute lamp height",
    ],
    "request_missing_input": ["request missing input", "request reference object"],
    "select_and_execute_external_action": [
        "select execute external action",
        "perform required external action",
        "invoke external action",
        "apply external action",
    ],
    "retrieve_current_information": [
        "retrieve current information",
        "retrieve latest model info",
        "fetch weather data",
        "fetch transit advisories",
        "latest stock price",
    ],
    "retrieve_external_information": [
        "retrieve external information",
        "fetch article content",
        "fetch video content",
    ],
    "revise_text_for_clarity": [
        "revise text clarity",
        "polish clarify essay",
        "make essay clearer",
    ],
    "search_provided_files": ["search provided files", "find todo comments"],
    "summarize_document": [
        "summarize document",
        "summarize text",
        "summarize bullets",
    ],
    "synthesize_plan": ["synthesize plan", "create itinerary"],
    "report_external_action_result": [
        "report external action result",
        "answer from external action result",
    ],
    "transform_image": ["transform image", "edit portrait", "remove background"],
    "transform_text_format": ["transform text format", "markdown table"],
    "translate_text": ["translate text", "translate paragraph"],
    "validate_output_against_requirements": [
        "validate output requirements",
        "run test",
        "validate test",
    ],
    "verify_current_information": ["verify current information", "fact check"],
    "write_file": ["write file", "create release notes"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "request",
        "run1_ok",
        "run2_ok",
        "run3_ok",
        "graph_ok",
        "structural_valid",
        "auto_plan_acceptable",
        "human_acceptable",
        "main_failure_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def score_predictions(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    predicted_only: bool = False,
) -> dict[str, Any]:
    predictions = {str(row.get("id")): row for row in prediction_rows}
    if predicted_only:
        gold_rows = [row for row in gold_rows if str(row.get("id")) in predictions]
    case_scores = []
    for gold in gold_rows:
        pred = predictions.get(str(gold.get("id")), {})
        case_scores.append(score_case(gold, pred))
    metrics = summarize_scores(case_scores)
    return {"metrics": metrics, "cases": case_scores}


def score_case(gold: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = gold.get("expected", {})
    passes = prediction.get("passes", {})
    run1 = _parsed_pass(passes, "intent_input_audit")
    run2 = _parsed_pass(passes, "transformation_externality_audit")
    run3 = _parsed_pass(passes, "capability_requirements")
    run4 = _parsed_pass(passes, "capability_normalization")
    run5 = _parsed_pass(passes, "capability_ordering")

    failures: list[str] = []
    notes: list[str] = []

    structural_graph_valid = _graph_structurally_valid(prediction)
    structural_valid = _all_passes_json_valid(passes) and structural_graph_valid
    if not _all_passes_json_valid(passes):
        failures.append("invalid_json")
        notes.append("one or more planning passes did not parse as JSON")

    run1_ok = _score_run1(run1, expected.get("run1", {}), failures, notes)
    run2_ok = _score_run2(run2, expected.get("run2", {}), failures, notes)
    run3_ok = _score_run3(run3, run4, expected.get("run3", {}), failures, notes)
    graph_ok = _score_graph(run5, expected.get("graph", {}), prediction, failures, notes)

    auto_plan_acceptable = all([structural_valid, run1_ok, run2_ok, run3_ok, graph_ok])
    main_failure_type = _main_failure(failures)
    return {
        "id": gold.get("id"),
        "category": gold.get("category"),
        "request": gold.get("request"),
        "run1_ok": run1_ok,
        "run2_ok": run2_ok,
        "run3_ok": run3_ok,
        "graph_ok": graph_ok,
        "structural_graph_valid": structural_graph_valid,
        "structural_valid": structural_valid,
        "auto_plan_acceptable": auto_plan_acceptable,
        "human_acceptable": "",
        "main_failure_type": main_failure_type,
        "failure_types": sorted(set(failures)),
        "notes": "; ".join(notes),
    }


def summarize_scores(case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_scores)
    if total == 0:
        return {
            "case_count": 0,
            "structural_valid_rate": 0.0,
            "input_audit_accuracy": 0.0,
            "valid_dependency_graph_rate": 0.0,
            "capability_plan_accept_rate": 0.0,
            "auto_capability_plan_accept_rate": 0.0,
            "human_capability_plan_accept_rate": None,
            "failure_counts": {},
        }
    failure_counts: dict[str, int] = defaultdict(int)
    for score in case_scores:
        for failure in score["failure_types"]:
            failure_counts[failure] += 1
    return {
        "case_count": total,
        "structural_valid_rate": _rate(case_scores, "structural_valid"),
        "input_audit_accuracy": _rate(case_scores, "run1_ok"),
        "valid_dependency_graph_rate": _rate(case_scores, "structural_graph_valid"),
        "capability_plan_accept_rate": _rate(case_scores, "auto_plan_acceptable"),
        "auto_capability_plan_accept_rate": _rate(case_scores, "auto_plan_acceptable"),
        "human_capability_plan_accept_rate": None,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _parsed_pass(passes: dict[str, Any], pass_name: str) -> dict[str, Any]:
    item = passes.get(pass_name, {})
    parsed = item.get("parsed") if isinstance(item, dict) else None
    return parsed if isinstance(parsed, dict) else {}


def _all_passes_json_valid(passes: dict[str, Any]) -> bool:
    required = [
        "intent_input_audit",
        "transformation_externality_audit",
        "capability_requirements",
        "capability_normalization",
        "capability_ordering",
    ]
    for pass_name in required:
        item = passes.get(pass_name)
        if not isinstance(item, dict) or item.get("parsed") is None or item.get("parse_error"):
            return False
    return True


def _graph_structurally_valid(prediction: dict[str, Any]) -> bool:
    ordered = _parsed_pass(prediction.get("passes", {}), "capability_ordering")
    caps = ordered.get("ordered_capabilities")
    if not isinstance(caps, list):
        return False
    return _dependency_graph_ok(caps)


def _dependency_graph_ok(caps: list[Any]) -> bool:
    ids = []
    edges = []
    for cap in caps:
        if not isinstance(cap, dict):
            return False
        cap_id = str(cap.get("id", ""))
        if not cap_id or cap_id in ids:
            return False
        ids.append(cap_id)
    id_set = set(ids)
    for cap in caps:
        cap_id = str(cap.get("id"))
        depends_on = cap.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list):
            return False
        for dep in depends_on:
            dep = str(dep)
            if dep not in id_set or dep == cap_id:
                return False
            edges.append((dep, cap_id))
    return _is_acyclic(ids, edges)


def _score_run1(
    run1: dict[str, Any],
    expected: dict[str, Any],
    failures: list[str],
    notes: list[str],
) -> bool:
    ok = True
    final_want = str(run1.get("final_user_want", ""))
    for keyword in expected.get("final_intent_keywords", []):
        if not _contains(final_want, keyword):
            ok = False
            failures.append("wrong_final_intent")
            notes.append(f"final intent missing {keyword!r}")

    inputs = run1.get("inputs", [])
    if not isinstance(inputs, list):
        ok = False
        failures.append("input_audit_error")
        notes.append("run1 inputs is not a list")
        inputs = []

    for expected_input in expected.get("inputs", []):
        if not _matching_input(inputs, expected_input):
            ok = False
            failure = (
                "missed_missing_input"
                if expected_input.get("available") is False
                else "input_audit_error"
            )
            failures.append(failure)
            notes.append(f"missing input audit match for {expected_input}")

    missing_text = _json_text(run1.get("missing_inputs", []))
    for keyword in expected.get("missing_input_keywords", []):
        if not (_contains(missing_text, keyword) or _unavailable_input_contains(inputs, keyword)):
            ok = False
            failures.append("missed_missing_input")
            notes.append(f"missing input list does not mention {keyword!r}")

    input_formats = {
        item.get("format") for item in inputs if isinstance(item, dict) and item.get("format")
    }
    for fmt in expected.get("must_include_format", []):
        if fmt not in input_formats:
            ok = False
            failures.append("input_audit_error")
            notes.append(f"run1 did not include input format {fmt!r}")
    for fmt in expected.get("must_not_include_format", []):
        if fmt in input_formats:
            ok = False
            failures.append("input_audit_error")
            notes.append(f"run1 unexpectedly included input format {fmt!r}")
    return ok


def _score_run2(
    run2: dict[str, Any],
    expected: dict[str, Any],
    failures: list[str],
    notes: list[str],
) -> bool:
    ok = True
    if "needs_current_or_external_info" in expected:
        actual = bool(run2.get("needs_current_or_external_info"))
        if actual != bool(expected["needs_current_or_external_info"]):
            ok = False
            if expected["needs_current_or_external_info"]:
                failure = "missed_current_info"
            else:
                failure = "wrong_external_action"
            failures.append(failure)
            notes.append(
                "run2 needs_current_or_external_info="
                f"{actual}, expected {expected['needs_current_or_external_info']}"
            )

    action_types = _run2_action_types(run2)
    for action_type in expected.get("must_include_external_action_type", []):
        if action_type not in action_types:
            ok = False
            if action_type in {"web_search", "fact_checking"}:
                failure = "missed_current_info"
            else:
                failure = "wrong_external_action"
            failures.append(failure)
            notes.append(f"run2 missing external action {action_type!r}")
    for action_type in expected.get("must_not_include_external_action_type", []):
        if action_type in action_types:
            ok = False
            if action_type == "file_reading":
                failure = "invented_file_reading"
            else:
                failure = "wrong_external_action"
            failures.append(failure)
            notes.append(f"run2 unexpectedly included external action {action_type!r}")

    transform_text = _json_text(run2.get("transformations_needed", []))
    for keyword in expected.get("transformation_keywords", []):
        if not _contains(transform_text, keyword):
            ok = False
            failures.append("under_decomposed")
            notes.append(f"run2 transformations missing {keyword!r}")
    return ok


def _score_run3(
    run3: dict[str, Any],
    run4: dict[str, Any],
    expected: dict[str, Any],
    failures: list[str],
    notes: list[str],
) -> bool:
    ok = True
    caps = run3.get("capabilities_needed", [])
    if not isinstance(caps, list) or not caps:
        failures.append("under_decomposed")
        notes.append("run3 produced no capabilities")
        return False

    names_and_text = _capability_text(caps)
    normalized_text = _json_text(run4.get("normalized_capabilities", []))
    all_cap_text = f"{names_and_text} {normalized_text}"

    for cap in caps:
        if not isinstance(cap, dict):
            ok = False
            failures.append("under_decomposed")
            notes.append("run3 capability is not an object")
            continue
        for field in ["inputs", "outputs", "done_when", "external_action_type"]:
            if field not in cap:
                ok = False
                failures.append("under_decomposed")
                notes.append(f"capability {cap.get('id')} missing {field}")
        name = _normalize_name(str(cap.get("capability_name", "")))
        if name in VAGUE_CAPABILITY_NAMES:
            ok = False
            failures.append("vague_capability")
            notes.append(f"vague capability name {name!r}")

    tool_hit = _first_term_hit(all_cap_text, TOOL_TERMS)
    if tool_hit:
        ok = False
        failures.append("used_tool_name")
        notes.append(f"capability text used tool/library term {tool_hit!r}")
    agent_hit = _first_term_hit(all_cap_text, AGENT_TERMS)
    if agent_hit:
        ok = False
        failures.append("used_agent_name")
        notes.append(f"capability text used agent/worker term {agent_hit!r}")

    for required in expected.get("must_include_capability", []):
        if not _contains_capability(caps, required):
            ok = False
            failures.append("under_decomposed")
            notes.append(f"run3 missing expected capability {required!r}")
    for required in expected.get("must_include_capability_alt", []):
        if not _contains_capability(caps, required):
            ok = False
            failures.append("under_decomposed")
            notes.append(f"run3 missing secondary expected capability {required!r}")
    for forbidden in expected.get("must_not_include_capability", []):
        if _contains_capability(caps, forbidden):
            ok = False
            failures.append("over_decomposed")
            notes.append(f"run3 unexpectedly included capability {forbidden!r}")

    action_types = {
        cap.get("external_action_type")
        for cap in caps
        if isinstance(cap, dict) and cap.get("external_action_type")
    }
    for action_type in expected.get("must_include_external_action_type", []):
        if action_type not in action_types:
            ok = False
            failures.append("wrong_external_action")
            notes.append(f"run3 missing external_action_type {action_type!r}")
    for action_type in expected.get("must_not_include_external_action_type", []):
        if action_type in action_types:
            ok = False
            if action_type == "file_reading":
                failure = "invented_file_reading"
            else:
                failure = "wrong_external_action"
            failures.append(failure)
            notes.append(f"run3 unexpectedly included external_action_type {action_type!r}")
    return ok


def _score_graph(
    run5: dict[str, Any],
    expected: dict[str, Any],
    prediction: dict[str, Any],
    failures: list[str],
    notes: list[str],
) -> bool:
    ok = True
    caps = run5.get("ordered_capabilities", [])
    if not isinstance(caps, list) or not _dependency_graph_ok(caps):
        ok = False
        failures.append("bad_dependency")
        notes.append("run5 dependency graph is invalid")
    validation = prediction.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is False:
        ok = False
        for violation in validation.get("violations", []):
            if isinstance(violation, dict):
                failure = _failure_from_validator_violation(violation.get("type"))
                failures.append(failure)
                notes.append(f"validator: {violation.get('message')}")

    for rule in expected.get("must_precede", []):
        before = rule.get("before")
        after = rule.get("after")
        if before and after and not _capability_precedes(caps, before, after):
            ok = False
            failures.append("bad_dependency")
            notes.append(f"expected {before!r} to precede {after!r}")
    return ok


def _matching_input(inputs: list[Any], expected_input: dict[str, Any]) -> bool:
    keyword = expected_input.get("keyword")
    for item in inputs:
        if not isinstance(item, dict):
            continue
        text = _json_text(item)
        if keyword and not _contains(text, keyword):
            continue
        if "available" in expected_input and bool(item.get("available")) != bool(
            expected_input["available"]
        ):
            continue
        if "format" in expected_input and item.get("format") != expected_input["format"]:
            continue
        return True
    matching_items = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if "available" in expected_input and bool(item.get("available")) != bool(
            expected_input["available"]
        ):
            continue
        if "format" in expected_input and item.get("format") != expected_input["format"]:
            continue
        matching_items.append(item)
    if matching_items and keyword:
        return _contains(_json_text(matching_items), keyword)
    return False


def _unavailable_input_contains(inputs: list[Any], keyword: str) -> bool:
    for item in inputs:
        if (
            isinstance(item, dict)
            and item.get("available") is False
            and _contains(_json_text(item), keyword)
        ):
            return True
    return False


def _run2_action_types(run2: dict[str, Any]) -> set[str]:
    actions = run2.get("external_actions", [])
    if not isinstance(actions, list):
        return set()
    action_types = set()
    has_needed_action = False
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = action.get("action_type")
        needed = bool(action.get("needed"))
        if needed and action_type and action_type != "none":
            has_needed_action = True
            action_types.update(_split_action_types(action_type) or {str(action_type)})
        elif action_type == "none" and not needed:
            action_types.add("none")
    if not has_needed_action:
        action_types.add("none")
    return action_types


def _capability_text(caps: list[Any]) -> str:
    return _json_text(
        [
            {
                "capability_name": cap.get("capability_name"),
                "capability_description": cap.get("capability_description"),
                "external_action_type": cap.get("external_action_type"),
            }
            for cap in caps
            if isinstance(cap, dict)
        ]
    )


def _contains_capability(caps: list[Any], expected: str) -> bool:
    for cap in caps:
        if not isinstance(cap, dict):
            continue
        text = _json_text(cap)
        if _capability_text_matches(text, expected):
            return True
    return False


def _capability_precedes(caps: list[Any], before_name: str, after_name: str) -> bool:
    before_ids = _matching_capability_ids(caps, before_name)
    after_ids = _matching_capability_ids(caps, after_name)
    if not before_ids or not after_ids:
        return False
    edges = []
    ids = []
    for cap in caps:
        if not isinstance(cap, dict):
            continue
        cap_id = str(cap.get("id", ""))
        ids.append(cap_id)
        for dep in cap.get("depends_on") or []:
            edges.append((str(dep), cap_id))
    for before_id in before_ids:
        for after_id in after_ids:
            if before_id != after_id and _has_path(before_id, after_id, edges):
                return True
    return False


def _matching_capability_ids(caps: list[Any], expected: str) -> list[str]:
    ids = []
    for cap in caps:
        if not isinstance(cap, dict):
            continue
        text = _json_text(cap)
        if _capability_text_matches(text, expected):
            ids.append(str(cap.get("id")))
    return ids


def _has_path(src: str, dst: str, edges: list[tuple[str, str]]) -> bool:
    out: dict[str, list[str]] = defaultdict(list)
    for before, after in edges:
        out[before].append(after)
    queue = deque([src])
    visited = set()
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        queue.extend(out[node])
    return False


def _is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indeg = {node: 0 for node in nodes}
    out: dict[str, list[str]] = defaultdict(list)
    for before, after in edges:
        out[before].append(after)
        indeg[after] = indeg.get(after, 0) + 1
    queue = deque([node for node in nodes if indeg.get(node, 0) == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for child in out[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def _failure_from_validator_violation(violation_type: Any) -> str:
    if violation_type in {"unneeded_file_extraction_for_pasted_text"}:
        return "invented_file_reading"
    if violation_type in {"missing_current_information_capability"}:
        return "missed_current_info"
    if violation_type in {"tool_or_worker_capability"}:
        return "used_tool_name"
    if violation_type in {"dependency_cycle", "unknown_dependency", "self_dependency"}:
        return "bad_dependency"
    if violation_type in {"missing_input_without_request_capability"}:
        return "missed_missing_input"
    return "input_audit_error"


def _main_failure(failures: list[str]) -> str:
    if not failures:
        return ""
    failure_set = set(failures)
    for failure in FAILURE_PRIORITY:
        if failure in failure_set:
            return failure
    return sorted(failure_set)[0]


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in rows if row.get(field)) / len(rows), 4)


def _contains(text: str, keyword: str) -> bool:
    normalized_keyword = _normalize_name(keyword)
    normalized_text = _normalize_name(text)
    if normalized_keyword and re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
        normalized_text,
    ):
        return True
    text_tokens = set(_meaningful_tokens(text))
    for variant in _keyword_variants(keyword):
        keyword_tokens = _meaningful_tokens(variant)
        if keyword_tokens and all(
            token in text_tokens or token in normalized_text for token in keyword_tokens
        ):
            return True
    return False


def _keyword_variants(keyword: str) -> list[str]:
    normalized = _normalize_name(keyword)
    aliases = []
    for key, values in KEYWORD_ALIASES.items():
        if _normalize_name(key) == normalized:
            aliases = values
            break
    return [keyword, *aliases]


def _capability_text_matches(text: str, expected: str) -> bool:
    normalized_expected = _normalize_name(expected)
    variants = CAPABILITY_ALIASES.get(normalized_expected, [expected])
    for variant in variants:
        tokens = _meaningful_tokens(variant)
        if tokens and all(_contains(text, token) for token in tokens):
            return True
    return False


def _split_action_types(value: Any) -> set[str]:
    valid = {
        "none",
        "file_reading",
        "file_writing",
        "web_search",
        "fact_checking",
        "calculation",
        "code_execution",
        "image_understanding",
        "image_generation",
        "user_input",
        "other",
    }
    if value in valid:
        return {str(value)}
    normalized = _normalize_name(str(value))
    return {action_type for action_type in valid if action_type != "none" and action_type in normalized}


def _meaningful_tokens(value: str) -> list[str]:
    stop = {"a", "an", "and", "for", "from", "in", "my", "of", "or", "the", "to", "with"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_name(str(value)))
        if token not in stop and len(token) > 1
    ]


def _first_term_hit(text: str, terms: set[str]) -> str:
    normalized = _normalize_name(text)
    for term in sorted(terms):
        if _normalize_name(term) in normalized:
            return term
    return ""


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
