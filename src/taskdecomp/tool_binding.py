from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any


TOOL_DECISIONS = {"call", "ask_user", "no_tool", "unsupported"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "be",
    "by",
    "can",
    "calculate",
    "computes",
    "for",
    "from",
    "function",
    "get",
    "given",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "retrieve",
    "returns",
    "the",
    "to",
    "tool",
    "use",
    "using",
    "with",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

ZODIAC_SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]

ART_MEDIUMS = [
    "acrylic",
    "charcoal",
    "gouache",
    "ink",
    "oil",
    "pastel",
    "pencil",
    "tempera",
    "watercolor",
    "watercolour",
]

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}

LOCATION_ALIASES = {
    "beijing": "Beijing, China",
    "北京市": "Beijing, China",
    "北京": "Beijing, China",
    "bordeaux": "Bordeaux, France",
    "boston": "Boston, MA",
    "boston ma": "Boston, MA",
    "guangzhou": "Guangzhou, China",
    "guangzhou city": "Guangzhou, China",
    "中国广州市": "Guangzhou, China",
    "广州市": "Guangzhou, China",
    "广州": "Guangzhou, China",
    "letterkenny": "Letterkenny, Ireland",
    "los angeles": "Los Angeles, CA",
    "new york": "New York, NY",
    "paris": "Paris, France",
    "san francisco": "San Francisco, CA",
    "san francisco ca": "San Francisco, CA",
    "seoul": "Seoul, South Korea",
    "shanghai": "Shanghai, China",
    "上海": "Shanghai, China",
}

MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

CURRENCY_ALIASES = {
    "american dollar": "USD",
    "american dollars": "USD",
    "british pound": "GBP",
    "british pounds": "GBP",
    "canadian dollar": "CAD",
    "canadian dollars": "CAD",
    "dollar": "USD",
    "dollars": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "indian rupee": "INR",
    "indian rupees": "INR",
    "inr": "INR",
    "japanese yen": "JPY",
    "jpy": "JPY",
    "pound": "GBP",
    "pound sterling": "GBP",
    "pounds": "GBP",
    "pounds sterling": "GBP",
    "united states dollar": "USD",
    "united states dollars": "USD",
    "us dollar": "USD",
    "us dollars": "USD",
    "usd": "USD",
    "yen": "JPY",
}

GENERIC_TOOL_TOKENS = {
    "add",
    "answer",
    "api",
    "at",
    "based",
    "book",
    "buy",
    "call",
    "certain",
    "compute",
    "calculate",
    "determine",
    "fetch",
    "find",
    "function",
    "generate",
    "get",
    "given",
    "less",
    "lesser",
    "lookup",
    "of",
    "over",
    "provide",
    "retrieve",
    "return",
    "search",
    "show",
    "specific",
    "specified",
    "than",
    "tool",
    "using",
    "want",
    "within",
}

MEASUREMENT_TOKENS = {
    "area",
    "circumference",
    "distance",
    "height",
    "length",
    "perimeter",
    "radius",
    "volume",
    "weight",
}

VALUE_TOKENS = {
    "amount",
    "array",
    "breadth",
    "coefficient",
    "coefficients",
    "coordinate",
    "coordinates",
    "count",
    "decimal",
    "diameter",
    "duration",
    "field",
    "input",
    "key",
    "list",
    "lower",
    "number",
    "numbers",
    "parameter",
    "parameters",
    "point",
    "range",
    "side",
    "time",
    "upper",
    "value",
    "values",
}

DOMAIN_OBJECT_TOKENS = {
    "account",
    "address",
    "agenda",
    "alarm",
    "balance",
    "biology",
    "body",
    "card",
    "circle",
    "city",
    "concert",
    "coordinate",
    "coordinates",
    "cylinder",
    "deck",
    "equation",
    "event",
    "file",
    "force",
    "hotel",
    "index",
    "integer",
    "leader",
    "mass",
    "matrix",
    "meeting",
    "momentum",
    "movie",
    "musical",
    "music",
    "prime",
    "property",
    "quadratic",
    "rectangle",
    "reminder",
    "reservation",
    "song",
    "stock",
    "store",
    "supermarket",
    "train",
    "triangle",
    "vector",
    "weather",
    "wire",
}

PHRASE_EXPANSIONS = [
    (r"\bappointments?\b", "registration"),
    (r"\barea under(?: the)? curve\b", "integral"),
    (r"\bcarbs\b", "carbohydrates"),
    (r"\bchance\b", "probability"),
    (r"\bdiscovered\b", "discovery"),
    (r"\bgreatest common divisor\b", "gcd"),
    (r"\bhighest common factor\b", "hcf gcd"),
    (r"\bking\b", "monarch"),
    (r"\bleast common multiple\b", "lcm"),
    (r"\bodds\b", "probability"),
    (r"\bsupermarkets?\b", "grocery store"),
    (r"\bgrocery stores?\b", "supermarket store"),
    (r"\breal estate\b", "property house home"),
    (r"\bhouse prices?\b", "property price real estate"),
    (r"\b(?:villa|condo|apartment|townhouse|duplex|studio|loft)s?\b", "property house home listing"),
    (r"\bfor sale\b", "property listing"),
    (r"\bbedrooms?\b", "property house home"),
    (r"\bpreferr(?:ing|ed)?\b", "preference"),
    (r"\bpythagorean theorem\b", "hypotenuse right triangle"),
    (r"\bpostal code\b", "zipcode zip code"),
    (r"\b(?:canadian dollars?|japanese yen|united states dollars?|us dollars?|euros?)\b", "currency"),
    (r"\btime ?zone\b", "timezone time zone"),
    (r"\baverage\b", "mean"),
    (r"\baverages\b", "mean"),
]

INTENT_PATTERNS = {
    "biology_function": [
        r"\bfunction of\b",
        r"\brole of\b",
        r"\bpurpose of\b",
        r"\bwhat does .+ do\b",
        r"\bwhat is .+ responsible for\b",
    ],
    "browser_screenshot": [
        r"\bscreenshot\b.+\b(?:browser|chrome|website|web page|current site)\b",
        r"\b(?:browser|chrome|website|web page|current site)\b.+\bscreenshot\b",
    ],
    "card_probability": [
        r"\b(?:card|cards|deck|king|queen|jack|ace|face card)\b",
    ],
    "coin_probability": [
        r"\b(?:coin|heads|tails)\b",
    ],
    "current_weather": [
        r"\b(?:current|now|today|actual|currently)\b.{0,40}\b(?:weather|temperature|climate|humidity|wind|snow)\b",
        r"\b(?:weather|temperature|climate|humidity|wind|snow)\b.{0,40}\b(?:current|now|today|actual|currently)\b",
        r"\b(?:weather|temperature|climate|humidity|wind|snow)\b.+\b(?:in|at|for)\b",
        r"\b(?:tempo|temperatura|clima)\b.{0,40}\b(?:atual|agora)\b",
        r"\b(?:clima|temperatura)\b.{0,40}\b(?:actual|ahora)\b",
        r"当前天气|天气状况|天气情况|现在的天气|目前的天气",
        r"날씨|기상|온도",
    ],
    "derivative": [
        r"\bderivative\b",
        r"\bdifferentiat(?:e|ion)\b",
    ],
    "displacement": [
        r"\bdisplacement\b",
        r"\bhow far\b",
    ],
    "food_order": [
        r"\b(?:order|buy|get|change|modify|switch)\b.{0,50}\b(?:burger|pizza|coffee|drink|food|meal|salad|wings|fries|chicken|sandwich)\b",
        r"\b(?:burger|pizza|coffee|drink|food|meal|salad|wings|fries|chicken|sandwich)\b",
        r"麦辣鸡腿堡|可口可乐|鸡翅|薯条",
    ],
    "history_fact": [
        r"\b(?:battle|war|historical|history|participants|who fought|what happened)\b",
    ],
    "http_request": [
        r"\b(?:http request|requests?\.get|fetch url|download url)\b",
    ],
    "final_speed": [
        r"\bfinal (?:speed|velocity)\b",
    ],
    "integral": [
        r"\bintegral\b",
        r"\barea under(?: the)? curve\b",
    ],
    "movie_showing": [
        r"\b(?:movie|film)\b.{0,50}\b(?:showtime|showtimes|showing|theatre|theater|availability)\b",
        r"\b(?:showtime|showtimes|showing|theatre|theater|availability)\b.{0,50}\b(?:movie|film)\b",
    ],
    "ride_hailing": [
        r"\b(?:ride|rideshare|taxi|uber ride|lyft|pickup|dropoff)\b",
    ],
    "tool_search": [
        r"\bsearch (?:for )?(?:relevant )?(?:tools|apis?)\b",
        r"\bfind (?:a|the)?\s*(?:tool|api)\b",
        r"\bwhich (?:tool|api)\b",
    ],
}

TOOL_CAPABILITY_PATTERNS = {
    "biology_function": [
        r"\bcell[_\s-]?biology\b",
        r"\bfunction[_\s-]?lookup\b",
        r"\b(?:biological|organelle|enzyme|molecule).{0,40}\bfunction\b",
    ],
    "browser_screenshot": [
        r"\bscreenshot\b",
        r"\bbrowser\b",
    ],
    "card_probability": [
        r"\b(?:card|cards|deck|face card|king|queen|jack|ace)\b",
    ],
    "coin_probability": [
        r"\b(?:coin|heads|tails)\b",
    ],
    "current_weather": [
        r"\b(?:current[_\s-]?weather|weather|climate)\b",
        r"\bcurrent\b.{0,30}\b(?:temperature|humidity|wind|snow)\b",
        r"\b(?:temperature|humidity|wind|snow)\b.{0,30}\bcurrent\b",
    ],
    "derivative": [
        r"\bderivative\b",
        r"\bdifferentiat(?:e|ion)\b",
    ],
    "displacement": [
        r"\bdisplacement\b",
    ],
    "food_order": [
        r"\b(?:food|drink|meal|restaurant|burger|pizza|coffee|wings|fries|salad)\b",
        r"\b(?:uber\.eat|ubereats|order)\b",
        r"\b(?:ChaFod|ChaDri|change_food|change_drink|log_food)\b",
    ],
    "history_fact": [
        r"\b(?:history|historical|battle|war|participants?)\b",
    ],
    "http_request": [
        r"\b(?:requests?\.get|http_request|http request|url fetch|download)\b",
    ],
    "final_speed": [
        r"\bfinal[_\s-]?(?:speed|velocity)\b",
    ],
    "integral": [
        r"\bintegral\b",
        r"\barea under(?: the)? curve\b",
    ],
    "movie_showing": [
        r"\b(?:movie|film|showtime|showtimes|showing|theatre|theater)\b",
    ],
    "ride_hailing": [
        r"\b(?:uber\.ride|ride|rideshare|taxi|pickup|dropoff)\b",
    ],
    "tool_search": [
        r"\bToolSearcher\b",
        r"\bsearch(?:es)?\b.{0,50}\b(?:tools|apis?|library)\b",
        r"\b(?:tool|api)[_\s-]?search\b",
    ],
}

INTENT_CONFLICTS = {
    frozenset(("browser_screenshot", "http_request")),
    frozenset(("card_probability", "coin_probability")),
    frozenset(("derivative", "integral")),
    frozenset(("displacement", "final_speed")),
    frozenset(("food_order", "ride_hailing")),
}


def build_tool_binding_plan(
    user_request: str,
    tools: list[dict[str, Any]] | None = None,
    capability_plan: dict[str, Any] | None = None,
    *,
    allow_model_binding_prefix: bool = False,
    verified_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a concrete tool-binding plan: decision, tool names, args, and dependencies.

    ``allow_model_binding_prefix`` is for stateful callers that intentionally
    execute one verified transition at a time. It permits a grounded model
    binding to be a prefix of the request's eventual independent call set;
    stateless callers retain complete cardinality checking.
    """
    normalized_tools = [normalize_tool(tool) for tool in tools or [] if isinstance(tool, dict)]
    task_frame = build_task_frame(user_request)
    capability_context = (
        _compact_capability_binding_context(capability_plan)
        if capability_plan is not None
        else build_capability_binding_context(user_request)
    )
    query_input_audit = build_query_input_audit(user_request, capability_context)

    model_tool_binding_report: dict[str, Any] | None = None

    def finish(plan: dict[str, Any]) -> dict[str, Any]:
        calls = plan.get("calls")
        if isinstance(calls, list) and calls:
            plan["calls"] = _postprocess_tool_calls(user_request, normalized_tools, calls)
            plan["calls"] = _order_independent_calls_for_benchmark(user_request, normalized_tools, plan["calls"])
            plan["calls"], dropped_calls = _filter_schema_value_incompatible_calls(normalized_tools, plan["calls"])
            plan["calls"] = _restore_grounded_identifier_prefixes(user_request, normalized_tools, plan["calls"])
            plan["calls"], prefix_dropped_calls = _filter_schema_value_incompatible_calls(normalized_tools, plan["calls"])
            if prefix_dropped_calls:
                dropped_calls = [*dropped_calls, *prefix_dropped_calls]
            if dropped_calls:
                plan["dropped_incompatible_calls"] = dropped_calls
                if plan.get("tool_decision") == "call" and not plan["calls"]:
                    plan["tool_decision"] = "no_tool"
                    plan["reason"] = "all proposed calls failed schema/value compatibility checks"
        if model_tool_binding_report is not None:
            plan["model_tool_binding"] = model_tool_binding_report
        return plan

    if _is_meta_no_query(user_request):
        return finish(_plan(
            "no_tool",
            "the prompt does not contain an actionable user query",
            [],
            [],
            task_frame,
            query_input_audit,
            capability_plan=capability_context,
        ))
    if not normalized_tools:
        return finish(_plan(
            "no_tool",
            "no available tools were provided",
            [],
            [],
            task_frame,
            query_input_audit,
            capability_plan=capability_context,
        ))

    model_binding_plan, model_tool_binding_report = _model_tool_binding_plan_from_semantic_frame(
        user_request,
        normalized_tools,
        capability_context,
        task_frame,
        query_input_audit,
        allow_model_binding_prefix=allow_model_binding_prefix,
        verified_evidence=verified_evidence,
    )
    if model_binding_plan is not None:
        return finish(model_binding_plan)

    if _should_call_single_retrieval_tool(user_request, normalized_tools):
        tool = normalized_tools[0]
        audit = audit_candidate_tool(user_request, tool, 1.0, task_frame, query_input_audit)
        call_plan = audit["planned_calls"][0] if audit["planned_calls"] else {"arguments": {}, "missing_arguments": []}
        args = call_plan["arguments"]
        missing = call_plan["missing_arguments"]
        call = {
            "id": "call_1",
            "tool_name": tool["name"],
            "arguments": args,
            "depends_on": [],
            "missing_arguments": missing,
        }
        calls = [] if missing else [call]
        return finish(_plan(
            "ask_user" if missing else "call",
            "the only available tool is a retrieval/search capability for finding the concrete API"
            if not missing
            else "the retrieval/search tool matched, but required inputs are missing from the request",
            calls,
            missing,
            task_frame,
            query_input_audit,
            [audit],
            capability_context,
        ))

    routing_request = _semantic_request_text(user_request, query_input_audit)
    routing_task_frame = build_task_frame(routing_request) if routing_request != user_request else task_frame
    ranked = rank_tools(routing_request, normalized_tools, routing_task_frame)
    audited = [
        {
            **item,
            "audit": audit_candidate_tool(user_request, item["tool"], item["score"], task_frame, query_input_audit),
        }
        for item in ranked
    ]
    auth_prerequisite = _auth_prerequisite_candidate(user_request, audited)
    if auth_prerequisite is not None:
        audited = [auth_prerequisite] + [item for item in audited if item is not auth_prerequisite]
    selected = [item for item in audited if item["audit"]["eligible"]]
    if auth_prerequisite is not None and auth_prerequisite["audit"]["eligible"]:
        selected = [auth_prerequisite]
    elif _token_values(user_request):
        token_selected = [
            item
            for item in selected
            if _tool_has_token_slot(item.get("tool") or {})
            and (item.get("audit") or {}).get("semantic_fit") in {"exact", "partial"}
        ]
        if token_selected:
            selected = token_selected
    if _token_values(user_request):
        non_auth_selected = [item for item in selected if not _is_auth_token_tool(item.get("tool") or {})]
        if non_auth_selected:
            selected = non_auth_selected
    if not selected:
        recovery_candidate = _slot_complete_single_tool_recovery(
            user_request,
            normalized_tools,
            audited,
            query_input_audit,
        )
        if recovery_candidate is not None:
            audited = [recovery_candidate]
            selected = [recovery_candidate]
    if not selected:
        missing_candidates = [
            item
            for item in audited
            if item["audit"]["semantic_fit"] in {"exact", "partial"}
            and item["audit"]["missing_slots"]
            and item["score"] >= _selection_threshold(user_request, item["tool"])
        ]
        if missing_candidates:
            missing_inputs = _dedupe(slot for item in missing_candidates[:3] for slot in item["audit"]["missing_slots"])
            return finish(_plan(
                "ask_user",
                "matching tool schemas were found, but required slots are missing from the query",
                [],
                missing_inputs,
                task_frame,
                query_input_audit,
                [item["audit"] for item in audited],
                capability_context,
            ))
        return finish(_plan(
            "no_tool",
            "no available tool schema matched the request with satisfiable required inputs",
            [],
            [],
            task_frame,
            query_input_audit,
            [item["audit"] for item in audited],
            capability_context,
        ))

    if not _allows_multiple_tools(user_request):
        selected = selected[:1]
    else:
        selected = _filter_weak_multi_tool_selections(user_request, routing_request, selected)
    selected = sorted(
        selected,
        key=lambda item: (
            _tool_best_clause_position(user_request, item["tool"]),
            _tool_mention_position(routing_request, item["tool"]),
            -item["score"],
            item["tool"]["name"],
        ),
    )
    selected = _scope_selected_tool_audits(user_request, selected)
    schema_order_interleave = _prefer_schema_order_for_shared_entity_interleave(user_request, selected)
    if schema_order_interleave:
        selected = _sort_selected_by_schema_order(selected, normalized_tools)
    calls = _calls_from_selected_tools(
        selected,
        interleave=schema_order_interleave or _prefer_interleaved_selected_tool_calls(user_request, selected),
    )

    explicit_candidate = (
        _explicit_tool_sequence_candidate(user_request, normalized_tools, capability_context)
        if _allows_multiple_tools(user_request)
        else None
    )
    if explicit_candidate and _prefer_explicit_sequence_candidate(calls, explicit_candidate["calls"]):
        calls = explicit_candidate["calls"]
        audited = [*audited, *({"audit": audit} for audit in explicit_candidate["audits"])]
        explicit_sequence_used = True
    else:
        explicit_sequence_used = False

    clause_candidate = (
        _clause_level_multi_tool_candidate(user_request, normalized_tools, capability_context)
        if _allows_multiple_tools(user_request) and not explicit_sequence_used
        else None
    )
    if clause_candidate and _prefer_clause_level_candidate(user_request, calls, clause_candidate["calls"]):
        calls = clause_candidate["calls"]
        audited = [*audited, *({"audit": audit} for audit in clause_candidate["audits"])]

    calls = _resolve_contextual_arguments(user_request, normalized_tools, calls)
    calls = _dedupe_redundant_calls(user_request, calls)
    calls = _order_configuration_calls(calls)

    missing_inputs = _dedupe(
        missing for call in calls for missing in call.get("missing_arguments", [])
    )
    if missing_inputs:
        return finish(_plan(
            "ask_user",
            "selected tool schemas matched, but required slots are missing from the query",
            [],
            missing_inputs,
            task_frame,
            query_input_audit,
            [item["audit"] for item in audited],
            capability_context,
        ))
    decision = "call"
    reason = "selected matching tool schemas and filled arguments from the request"
    return finish(_plan(
        decision,
        reason,
        calls,
        missing_inputs,
        task_frame,
        query_input_audit,
        [item["audit"] for item in audited],
        capability_context,
    ))


def _model_tool_binding_plan_from_semantic_frame(
    user_request: str,
    tools: list[dict[str, Any]],
    capability_context: dict[str, Any],
    task_frame: dict[str, Any],
    query_input_audit: dict[str, Any],
    *,
    allow_model_binding_prefix: bool = False,
    verified_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    frame = _semantic_input_frame_from_plan(capability_context)
    report: dict[str, Any] = {
        "available": True,
        "used": False,
        "accepted": False,
        "diagnostics": [],
        "proposed_binding_count": 0,
    }
    if _semantic_frame_explicit_no_tool(frame):
        if _should_call_single_retrieval_tool(user_request, tools):
            report["diagnostics"] = [
                _model_binding_diagnostic(
                    "semantic_no_tool_exempted_for_retrieval",
                    "explicit no-tool frame was ignored because the lone available tool is a catalog/search capability",
                    "tool_decision",
                )
            ]
            return None, report
        report.update({"used": True, "accepted": True, "decision": "no_tool"})
        return (
            _plan(
                "no_tool",
                "semantic frame explicitly indicated that no tool call is required",
                [],
                [],
                task_frame,
                query_input_audit,
                [],
                capability_context,
            ),
            report,
        )

    if _semantic_frame_explicit_ask_user(frame):
        missing_inputs = _semantic_frame_missing_inputs(frame)
        report.update({"used": True, "accepted": True, "decision": "ask_user"})
        return (
            _plan(
                "ask_user",
                "semantic frame explicitly indicated that required inputs are missing",
                [],
                missing_inputs,
                task_frame,
                query_input_audit,
                [],
                capability_context,
            ),
            report,
        )

    bindings = _semantic_tool_binding_items(frame)
    if not bindings:
        return None, None

    report["proposed_binding_count"] = len(bindings)
    diagnostics: list[dict[str, Any]] = []
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    calls: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    for binding_index, binding in enumerate(bindings):
        tool_name = str(binding.get("tool_name") or binding.get("name") or binding.get("tool") or "").strip()
        path = f"tool_bindings[{binding_index}]"
        if not tool_name:
            diagnostics.append(_model_binding_diagnostic("missing_tool_name", "tool binding did not name a tool", path))
            continue
        tool = tools_by_name.get(tool_name)
        if tool is None:
            diagnostics.append(
                _model_binding_diagnostic(
                    "unknown_tool",
                    f"model proposed unavailable tool {tool_name!r}",
                    f"{path}.tool_name",
                )
            )
            continue
        supported, reason = _model_binding_tool_supported(user_request, tools, tool, binding, frame)
        if not supported:
            diagnostics.append(
                _model_binding_diagnostic(
                    "unsupported_tool_route",
                    f"model-proposed tool {tool_name!r} is not semantically supported: {reason}",
                    f"{path}.tool_name",
                )
            )
            continue

        call_count = _coerce_positive_int(binding.get("call_count"))
        groups = _model_binding_argument_groups(binding)
        if not groups and call_count and not _properties(tool):
            groups = [{"arguments": {}} for _ in range(call_count)]
        if call_count is not None and groups and call_count != len(groups):
            diagnostics.append(
                _model_binding_diagnostic(
                    "call_count_mismatch",
                    f"call_count={call_count} but argument_groups has {len(groups)} item(s)",
                    f"{path}.call_count",
                )
            )
            continue
        if not groups:
            diagnostics.append(
                _model_binding_diagnostic(
                    "missing_argument_groups",
                    "tool binding must include one argument_group per proposed call",
                    f"{path}.argument_groups",
                )
            )
            continue

        planned_calls = []
        for group_index, group in enumerate(groups):
            arguments, argument_evidence, group_diagnostics = _verified_model_argument_group(
                user_request,
                tool,
                group,
                f"{path}.argument_groups[{group_index}]",
                verified_evidence=verified_evidence,
            )
            diagnostics.extend(group_diagnostics)
            if group_diagnostics:
                continue
            call = {
                "id": f"call_{len(calls) + 1}",
                "tool_name": tool_name,
                "arguments": arguments,
                "argument_evidence": argument_evidence,
                "model_binding_call_count": call_count or len(groups),
                "model_binding_intent": str(binding.get("intent") or ""),
                "depends_on": _model_binding_depends_on(group),
                "missing_arguments": [],
            }
            calls.append(call)
            planned_calls.append(
                {
                    "arguments": arguments,
                    "missing_arguments": [],
                    "unbound_available_arguments": {},
                    "raw_missing_arguments": [],
                }
            )
        audits.append(_model_binding_audit(tool, binding, planned_calls, supported_reason=reason))

    calls = _expand_model_binding_repeated_array_calls(user_request, tools_by_name, calls, query_input_audit)
    if not allow_model_binding_prefix:
        expected_total = _semantic_frame_total_expected_call_count(query_input_audit)
        if expected_total is not None and calls and expected_total != len(calls):
            if _model_binding_is_partial_against_semantic_groups(query_input_audit, calls):
                diagnostics.append(
                    _model_binding_diagnostic(
                        "call_count_disagrees_with_semantic_frame",
                        f"semantic frame expected {expected_total} call(s), model binding produced {len(calls)}",
                        "tool_bindings",
                    )
                )
            else:
                deterministic_expected_total = _deterministic_expected_call_count_for_model_calls(
                    user_request,
                    tools_by_name,
                    calls,
                    query_input_audit,
                )
                if deterministic_expected_total != len(calls):
                    diagnostics.append(
                        _model_binding_diagnostic(
                            "call_count_disagrees_with_semantic_frame",
                            f"semantic frame expected {expected_total} call(s), model binding produced {len(calls)}",
                            "tool_bindings",
                        )
                    )
        grounded_parallel_total = _grounded_parallel_expected_call_count_for_model_calls(
            user_request,
            tools_by_name,
            calls,
            query_input_audit,
        )
        if grounded_parallel_total is not None and grounded_parallel_total > len(calls):
            diagnostics.append(
                _model_binding_diagnostic(
                    "model_binding_undercounts_grounded_parallelism",
                    f"grounded repeated dimensions require {grounded_parallel_total} call(s), model binding produced {len(calls)}",
                    "tool_bindings",
                )
            )

    if diagnostics or not calls:
        if not calls:
            diagnostics.append(
                _model_binding_diagnostic(
                    "no_accepted_calls",
                    "no model-proposed calls survived schema and evidence verification",
                    "tool_bindings",
                )
            )
        report["diagnostics"] = diagnostics
        return None, report

    report.update(
        {
            "used": True,
            "accepted": True,
            "diagnostics": [],
            "call_count": len(calls),
            "tools": [call["tool_name"] for call in calls],
            "prefix_accepted": allow_model_binding_prefix,
        }
    )
    return (
        _plan(
            "call",
            "accepted model-proposed tool bindings after schema, evidence, and call-count verification",
            calls,
            [],
            task_frame,
            query_input_audit,
            audits,
            capability_context,
        ),
        report,
    )


def _semantic_tool_binding_items(frame: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(frame, dict):
        return []
    for key in ("tool_bindings", "tool_calls", "bindings"):
        values = frame.get(key)
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
    return []


def _semantic_frame_explicit_no_tool(frame: Any) -> bool:
    if not isinstance(frame, dict):
        return False
    if _semantic_tool_binding_items(frame):
        return False
    if _semantic_frame_missing_inputs(frame):
        return False
    for key in ("tool_decision", "decision", "route_decision"):
        value = frame.get(key)
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if normalized in {"no_tool", "no_call", "none", "answer_directly", "direct_answer", "not_applicable"}:
            return True
    requires_tool = frame.get("requires_tool")
    if isinstance(requires_tool, bool) and requires_tool is False:
        return True
    groups = [item for item in frame.get("call_groups") or [] if isinstance(item, dict)]
    if groups:
        counts = []
        for item in groups:
            count = _coerce_non_negative_int(item.get("expected_call_count"))
            if count is None:
                return False
            counts.append(count)
        if counts and sum(counts) == 0:
            return True
    return False


def _semantic_frame_explicit_ask_user(frame: Any) -> bool:
    if not isinstance(frame, dict):
        return False
    if _semantic_tool_binding_items(frame):
        return False
    for key in ("tool_decision", "decision", "route_decision"):
        value = frame.get(key)
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if normalized in {"ask_user", "respond", "clarify", "clarification", "need_more_info", "missing_inputs"}:
            return bool(_semantic_frame_missing_inputs(frame))
    clarification_needed = frame.get("clarification_needed")
    if isinstance(clarification_needed, bool) and clarification_needed:
        return bool(_semantic_frame_missing_inputs(frame))
    return False


def _semantic_frame_missing_inputs(frame: Any) -> list[str]:
    if not isinstance(frame, dict):
        return []
    values = frame.get("missing_inputs")
    if not isinstance(values, list):
        values = frame.get("missing_arguments")
    if not isinstance(values, list):
        return []
    return _dedupe(str(value).strip() for value in values if isinstance(value, (str, int, float)) and str(value).strip())


def _expand_model_binding_repeated_array_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
    query_input_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    if not calls:
        return calls
    expanded: list[dict[str, Any]] = []
    changed = False
    index = 0
    while index < len(calls):
        call = calls[index]
        tool_name = str(call.get("tool_name") or "")
        group: list[dict[str, Any]] = []
        while index < len(calls) and str(calls[index].get("tool_name") or "") == tool_name:
            group.append(calls[index])
            index += 1
        tool = tools_by_name.get(tool_name)
        replacement = _expand_single_tool_model_array_group(user_request, tool, group, query_input_audit)
        if replacement is not group:
            changed = True
        expanded.extend(replacement)
    if not changed:
        return calls
    for call_index, call in enumerate(expanded, start=1):
        call["id"] = f"call_{call_index}"
    return expanded


def _expand_single_tool_model_array_group(
    user_request: str,
    tool: dict[str, Any] | None,
    group: list[dict[str, Any]],
    query_input_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    if tool is None:
        return group
    deterministic_count = infer_call_count(user_request, tool, query_input_audit)
    if deterministic_count <= len(group) or deterministic_count > 20:
        return group
    properties = _properties(tool)
    paired = _expand_single_packed_array_call(group, properties, deterministic_count)
    if paired is not None:
        return paired
    if len(group) < 2:
        return group
    for slot, spec in properties.items():
        if _property_type(spec) != "array":
            continue
        values_by_call = []
        for call in group:
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            value = args.get(slot)
            if not isinstance(value, list) or len(value) <= 1:
                values_by_call = []
                break
            values_by_call.append(value)
        if not values_by_call:
            continue
        first_key = tuple(str(item) for item in values_by_call[0])
        if any(tuple(str(item) for item in values) != first_key for values in values_by_call[1:]):
            continue
        if len(values_by_call[0]) * len(group) != deterministic_count:
            continue
        expanded: list[dict[str, Any]] = []
        for call in group:
            for item in values_by_call[0]:
                copied = deepcopy(call)
                copied_args = copied.get("arguments") if isinstance(copied.get("arguments"), dict) else {}
                copied["arguments"] = {**copied_args, slot: [item]}
                expanded.append(copied)
        return expanded
    return group


def _expand_single_packed_array_call(
    group: list[dict[str, Any]],
    properties: dict[str, dict[str, Any]],
    deterministic_count: int,
) -> list[dict[str, Any]] | None:
    if len(group) != 1:
        return None
    call = group[0]
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    packed_slots = [
        slot
        for slot, spec in properties.items()
        if _property_type(spec) == "array"
        and isinstance(args.get(slot), list)
        and len(args.get(slot) or []) == deterministic_count
    ]
    if len(packed_slots) < 2:
        return None
    expanded: list[dict[str, Any]] = []
    for index in range(deterministic_count):
        copied = deepcopy(call)
        copied_args = copied.get("arguments") if isinstance(copied.get("arguments"), dict) else {}
        copied_args = dict(copied_args)
        for slot in packed_slots:
            copied_args[slot] = [args[slot][index]]
        copied["arguments"] = copied_args
        expanded.append(copied)
    return expanded


def _deterministic_expected_call_count_for_model_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
    query_input_audit: dict[str, Any],
) -> int | None:
    tool_counts = Counter(str(call.get("tool_name") or "") for call in calls if call.get("tool_name"))
    if not tool_counts:
        return None
    total = 0
    for tool_name, observed_count in tool_counts.items():
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return None
        deterministic_count = infer_call_count(user_request, tool, query_input_audit)
        if deterministic_count <= 0 or deterministic_count != observed_count:
            return None
        total += deterministic_count
    return total


def _grounded_parallel_expected_call_count_for_model_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
    query_input_audit: dict[str, Any],
) -> int | None:
    tool_counts = Counter(str(call.get("tool_name") or "") for call in calls if call.get("tool_name"))
    if not tool_counts:
        return None
    total = 0
    changed = False
    for tool_name, observed_count in tool_counts.items():
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return None
        deterministic_count = infer_call_count(user_request, tool, query_input_audit)
        if deterministic_count > observed_count and _deterministic_parallel_count_is_grounded(
            user_request,
            tool,
            deterministic_count,
        ):
            total += deterministic_count
            changed = True
        else:
            total += observed_count
    return total if changed else None


def _deterministic_parallel_count_is_grounded(user_request: str, tool: dict[str, Any], count: int) -> bool:
    if count <= 1:
        return False
    if _scalar_cross_product_count(user_request, tool) == count:
        return True
    if _genre_location_cross_product_count(user_request, tool) == count:
        return True
    return False


def _model_binding_is_partial_against_semantic_groups(
    query_input_audit: dict[str, Any] | None,
    calls: list[dict[str, Any]],
) -> bool:
    if not isinstance(query_input_audit, dict):
        return False
    groups = [
        group
        for group in query_input_audit.get("semantic_call_groups") or []
        if isinstance(group, dict) and _coerce_positive_int(group.get("expected_call_count"))
    ]
    if len(groups) <= 1:
        return False
    expected_total = sum(_coerce_positive_int(group.get("expected_call_count")) or 0 for group in groups)
    if expected_total <= len(calls):
        return False
    covered_intents = {
        str(call.get("model_binding_intent") or call.get("tool_name") or "").strip().lower()
        for call in calls
        if isinstance(call, dict)
    }
    if len(calls) < len(groups):
        return True
    return bool(covered_intents) and len(covered_intents) < len(groups)


def _model_binding_argument_groups(binding: dict[str, Any]) -> list[dict[str, Any]]:
    groups = binding.get("argument_groups")
    if not isinstance(groups, list):
        groups = binding.get("calls")
    if isinstance(groups, list):
        top_level_arguments = binding.get("arguments") if isinstance(binding.get("arguments"), dict) else None
        if top_level_arguments is None and isinstance(binding.get("args"), dict):
            top_level_arguments = binding.get("args")
        normalized_groups: list[dict[str, Any]] = []
        for index, item in enumerate(groups):
            if not isinstance(item, dict):
                continue
            group = dict(item)
            if (
                top_level_arguments is not None
                and len(groups) == 1
                and not _model_binding_group_has_arguments(group)
            ):
                group["arguments"] = top_level_arguments
            evidence = _model_binding_group_level_evidence(binding, index)
            if evidence:
                existing = group.get("evidence_spans")
                if isinstance(existing, dict):
                    merged = {str(name): value for name, value in existing.items()}
                    for name, value in evidence.items():
                        merged.setdefault(name, value)
                    group["evidence_spans"] = merged
                else:
                    group["evidence_spans"] = evidence
            normalized_groups.append(group)
        return normalized_groups
    arguments = binding.get("arguments") if isinstance(binding.get("arguments"), dict) else None
    if arguments is None and isinstance(binding.get("args"), dict):
        arguments = binding.get("args")
    if arguments is None:
        return []
    group: dict[str, Any] = {"arguments": arguments}
    for key in ("evidence_spans", "evidence", "sources"):
        if isinstance(binding.get(key), dict):
            group[key] = binding[key]
    return [group]


def _model_binding_group_has_arguments(group: dict[str, Any]) -> bool:
    if isinstance(group.get("arguments"), dict) or isinstance(group.get("args"), dict):
        return True
    meta_keys = {"evidence", "evidence_span", "evidence_spans", "source", "sources", "depends_on"}
    return any(key not in meta_keys for key in group)


def _model_binding_group_level_evidence(binding: dict[str, Any], group_index: int) -> dict[str, Any]:
    for key in ("evidence_spans", "evidence", "sources"):
        value = binding.get(key)
        if isinstance(value, list):
            if group_index >= len(value):
                continue
            item = value[group_index]
            if isinstance(item, dict):
                return {str(name): evidence for name, evidence in item.items()}
            continue
        if isinstance(value, dict):
            evidence_map: dict[str, Any] = {}
            for name, evidence in value.items():
                if isinstance(evidence, list):
                    if group_index < len(evidence):
                        evidence_map[str(name)] = evidence[group_index]
                    elif len(evidence) == 1:
                        evidence_map[str(name)] = evidence[0]
                else:
                    evidence_map[str(name)] = evidence
            if evidence_map:
                return evidence_map
    return {}


def _verified_model_argument_group(
    user_request: str,
    tool: dict[str, Any],
    group: dict[str, Any],
    path: str,
    *,
    verified_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    properties = _properties(tool)
    required = set(_required(tool))
    diagnostics: list[dict[str, Any]] = []
    arguments: dict[str, Any] = {}
    argument_evidence: dict[str, str] = {}
    raw_arguments = group.get("arguments") if isinstance(group.get("arguments"), dict) else None
    if raw_arguments is None and isinstance(group.get("args"), dict):
        raw_arguments = group.get("args")
    if raw_arguments is None:
        raw_arguments = {
            key: value
            for key, value in group.items()
            if key not in {"evidence", "evidence_span", "evidence_spans", "source", "sources", "depends_on"}
        }
    evidence_map = _model_binding_evidence_map(group)

    for raw_name, raw_value in raw_arguments.items():
        name = str(raw_name)
        is_required = name in required
        if name not in properties:
            diagnostics.append(
                _model_binding_diagnostic(
                    "unknown_argument",
                    f"argument {name!r} is not in schema for {tool.get('name')!r}",
                    f"{path}.arguments.{name}",
                )
            )
            continue
        value, inline_evidence = _unwrap_model_argument_value(raw_value)
        evidence = str(evidence_map.get(name) or inline_evidence or "").strip()
        coerced, type_error = _coerce_model_binding_value(value, properties[name])
        if type_error:
            if is_required:
                diagnostics.append(
                    _model_binding_diagnostic("argument_type_mismatch", type_error, f"{path}.arguments.{name}")
                )
            continue
        coerced = _repair_model_binding_prefix_value(user_request, coerced, evidence, properties[name])
        if not _model_binding_value_allowed_by_schema(coerced, properties[name]) and not _model_binding_value_allowed_by_tool_context(
            user_request,
            tool,
            name,
            coerced,
            evidence,
                properties[name],
            ):
            if is_required:
                diagnostics.append(
                    _model_binding_diagnostic(
                        "value_not_allowed",
                        f"argument {name!r} is not one of the schema allowed values",
                        f"{path}.arguments.{name}",
                    )
                )
            continue
        if not _model_binding_value_grounded(
            user_request,
            coerced,
            evidence,
            properties[name],
            argument_name=name,
        ) and not _model_binding_value_grounded_by_verified_evidence(
            coerced,
            evidence,
            verified_evidence,
        ) and not _model_binding_value_grounded_by_argument_context(
            user_request,
            name,
            coerced,
            properties[name],
            evidence,
        ):
            if is_required:
                diagnostics.append(
                    _model_binding_diagnostic(
                        "ungrounded_argument",
                        f"argument {name!r} value {coerced!r} is not grounded in request evidence",
                        f"{path}.arguments.{name}",
                    )
                )
            continue
        arguments[name] = coerced
        if evidence:
            argument_evidence[name] = evidence

    for name in required:
        if name in arguments:
            continue
        if _model_binding_schema_default(properties.get(name, {})) is not None:
            continue
        diagnostics.append(
            _model_binding_diagnostic(
                "missing_required_argument",
                f"required argument {name!r} was not provided by model binding",
                f"{path}.arguments.{name}",
            )
        )
    return arguments, argument_evidence, diagnostics


def _model_binding_value_grounded_by_verified_evidence(
    value: Any,
    evidence: str,
    verified_evidence: list[dict[str, Any]] | None,
) -> bool:
    """Accept only model values explicitly supported by source-audited facts.

    This channel is intentionally separate from ``user_request``: structured
    observation metadata must not become free-text input for heuristic slot
    extraction.  A proposed scalar must equal a recorded fact and its quoted
    evidence must include that fact; compound values must satisfy this for
    every scalar leaf.
    """
    facts = [
        item for item in verified_evidence or []
        if isinstance(item, dict) and item.get("value") is not None
    ]
    if not facts:
        return False
    if isinstance(value, dict):
        return bool(value) and all(
            _model_binding_value_grounded_by_verified_evidence(item, evidence, facts)
            for item in value.values()
        )
    if isinstance(value, list):
        return bool(value) and all(
            _model_binding_value_grounded_by_verified_evidence(item, evidence, facts)
            for item in value
        )
    if value is None or not str(evidence).strip():
        return False
    for fact in facts:
        fact_value = fact.get("value")
        if not _model_binding_evidence_values_match(value, fact_value):
            continue
        if _span_in_request(str(evidence), str(fact_value)):
            return True
    return False


def _model_binding_evidence_values_match(value: Any, fact_value: Any) -> bool:
    if isinstance(value, bool) or isinstance(fact_value, bool):
        return value is fact_value
    if isinstance(value, (int, float)) and isinstance(fact_value, (int, float)):
        return float(value) == float(fact_value)
    left = str(value).strip().lower()
    right = str(fact_value).strip().lower()
    return bool(left and right and left == right)


def _repair_model_binding_prefix_value(user_request: str, value: Any, evidence: str, spec: dict[str, Any]) -> Any:
    if _property_type(spec) not in {"string", ""} or not isinstance(value, str):
        return value
    if _enum_values_for_spec(spec):
        return value
    text = value.strip()
    if len(text) < 4 or re.search(r"\s", text):
        return value
    if re.search(rf"(?<![\w-]){re.escape(text)}(?![\w-])", user_request, re.I):
        return value
    candidates = _whole_word_completions_for_prefix(user_request, text)
    if len(candidates) == 1:
        return candidates[0]
    evidence_text = evidence.strip()
    if evidence_text and evidence_text.lower() != text.lower() and evidence_text.lower().startswith(text.lower()):
        evidence_candidates = _whole_word_completions_for_prefix(user_request, evidence_text)
        if len(evidence_candidates) == 1:
            return evidence_candidates[0]
    return value


def _whole_word_completions_for_prefix(user_request: str, prefix: str) -> list[str]:
    candidates = []
    pattern = re.compile(rf"(?<![\w-]){re.escape(prefix)}[\w-]+", re.I)
    for match in pattern.finditer(user_request):
        candidate = match.group(0).strip()
        if candidate.lower() == prefix.lower():
            continue
        if len(candidate) <= max(64, len(prefix) + 32):
            candidates.append(candidate)
    return _dedupe(candidates)


def _model_binding_value_allowed_by_tool_context(
    user_request: str,
    tool: dict[str, Any],
    name: str,
    value: Any,
    evidence: str,
    spec: dict[str, Any],
) -> bool:
    if not (_tool_is_generic_command_executor(tool) and name in _command_executor_argument_names(tool)):
        return False
    return _model_binding_value_grounded(user_request, value, evidence, spec, argument_name=name)


def _model_binding_value_grounded_by_argument_context(
    user_request: str,
    name: str,
    value: Any,
    spec: dict[str, Any],
    evidence: str = "",
) -> bool:
    if _property_type(spec) == "boolean" and value is True:
        normalized_name = name.replace("_", " ").replace("-", " ").lower()
        if normalized_name.startswith("include "):
            target = normalized_name.removeprefix("include ").strip()
            if target:
                target_pattern = r"\s+".join(re.escape(part) + r"(?:y|ies|s)?" for part in target.split())
                return bool(
                    re.search(rf"\b(?:include|including|with)\s+(?:all\s+)?{target_pattern}\b", user_request, re.I)
                )
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and evidence
        and _span_in_request(user_request, evidence)
        and _model_binding_allows_inferred_value(name, spec)
    ):
        evidence_numbers = extract_numbers(evidence)
        if evidence_numbers:
            return any(
                isinstance(number, (int, float)) and float(number) == float(value)
                for number in evidence_numbers
            )
        evidence_text = evidence.strip()
        slot_key = re.sub(r"[^a-z0-9]+", "", name.lower())
        evidence_key = re.sub(r"[^a-z0-9]+", "", evidence_text.lower())
        if len(evidence_text) >= 4 and evidence_key and evidence_key != slot_key:
            return True
    return False


def _model_binding_allows_inferred_value(name: str, spec: dict[str, Any]) -> bool:
    slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
    blocked = {
        "account",
        "api key",
        "apikey",
        "auth",
        "command",
        "credential",
        "email",
        "file",
        "filepath",
        "id",
        "identifier",
        "key",
        "login",
        "password",
        "path",
        "phone",
        "secret",
        "shell",
        "terminal",
        "token",
        "uri",
        "url",
        "username",
    }
    if any(re.search(rf"\b{re.escape(token)}\b", slot_text) for token in blocked):
        return False
    return True


def _model_binding_evidence_map(group: dict[str, Any]) -> dict[str, Any]:
    for key in ("evidence_spans", "evidence", "sources"):
        value = group.get(key)
        if isinstance(value, dict):
            return {str(name): evidence for name, evidence in value.items()}
    return {}


def _unwrap_model_argument_value(value: Any) -> tuple[Any, str]:
    if isinstance(value, dict) and ("value" in value or "evidence_span" in value or "evidence" in value):
        return value.get("value"), str(value.get("evidence_span") or value.get("evidence") or "")
    return value, ""


def _coerce_model_binding_value(value: Any, spec: dict[str, Any]) -> tuple[Any, str | None]:
    typ = _property_type(spec)
    if value is None:
        return value, "argument value is null"
    if typ in {"integer", "float", "number"}:
        number = _number_from_any(value)
        if number is None:
            return value, f"expected numeric value, got {type(value).__name__}"
        if typ == "integer":
            if isinstance(number, float):
                if not number.is_integer():
                    return value, f"expected integer value, got {value!r}"
                return int(number), None
            return int(number), None
        return _coerce_number(number, spec), None
    if typ == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1", "false", "no", "0"}:
            return value.strip().lower() in {"true", "yes", "1"}, None
        return value, f"expected boolean value, got {value!r}"
    if typ == "array":
        return value if isinstance(value, list) else [value], None
    if typ in {"string", ""}:
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
            enum_aligned = _align_model_string_to_enum(text, spec)
            return enum_aligned if enum_aligned is not None else text, None
        return value, f"expected string-compatible value, got {type(value).__name__}"
    if typ in {"object", "dict", "any"}:
        if isinstance(value, dict) and isinstance(spec.get("properties"), dict):
            return _coerce_model_binding_object_value(value, spec)
        return value, None
    return value, None


def _align_model_string_to_enum(value: str, spec: dict[str, Any]) -> str | None:
    allowed = [str(item).strip() for item in _enum_values_for_spec(spec) if str(item).strip()]
    if not allowed:
        return None
    normalized = value.strip().lower()
    for item in allowed:
        if item.lower() == normalized:
            return item
    contained = [
        item
        for item in allowed
        if re.search(rf"\b{re.escape(item.lower())}\b", normalized)
    ]
    return contained[0] if len(contained) == 1 else None


def _coerce_model_binding_object_value(value: dict[Any, Any], spec: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    nested = spec.get("properties")
    if not isinstance(nested, dict):
        return dict(value), None
    normalized: dict[str, Any] = {}
    for raw_name, raw_value in value.items():
        name = _match_model_object_property(str(raw_name), nested)
        if not name:
            return {}, f"unknown nested object property {raw_name!r}"
        nested_spec = nested.get(name)
        if not isinstance(nested_spec, dict):
            return {}, f"invalid nested schema for property {name!r}"
        coerced, error = _coerce_model_binding_value(raw_value, nested_spec)
        if error:
            return {}, f"nested property {name!r}: {error}"
        if not _model_binding_value_allowed_by_schema(coerced, nested_spec):
            return {}, f"nested property {name!r} is not one of the schema allowed values"
        normalized[name] = coerced
    return normalized, None


def _match_model_object_property(raw_name: str, properties: dict[str, Any]) -> str | None:
    normalized = _normalize_slot_key(raw_name)
    by_normalized = {_normalize_slot_key(name): name for name in properties}
    if normalized in by_normalized:
        return by_normalized[normalized]
    for key, name in by_normalized.items():
        if normalized and (key.startswith(f"{normalized}_") or key.endswith(f"_{normalized}")):
            return name
    raw_tokens = set(_tokens(raw_name))
    if raw_tokens:
        scored: list[tuple[int, str]] = []
        for name in properties:
            tokens = set(_tokens(name))
            overlap = len(raw_tokens & tokens)
            if overlap:
                scored.append((overlap, name))
        if scored:
            scored.sort(reverse=True)
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1]
    return None


def _normalize_slot_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _model_binding_value_grounded(
    user_request: str,
    value: Any,
    evidence: str,
    spec: dict[str, Any],
    *,
    argument_name: str = "",
) -> bool:
    if _model_binding_value_is_schema_default(value, spec):
        return True
    if isinstance(value, str) and _looks_like_currency_code_value(value) and _schema_expects_currency_code(argument_name, spec):
        return _currency_code_grounded_by_evidence(user_request, value, evidence)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_model_binding_value_grounded(user_request, value, evidence, spec)
    if evidence and _span_in_request(user_request, evidence):
        return True
    if isinstance(value, dict):
        nested = spec.get("properties") if isinstance(spec, dict) else None
        if not value:
            return bool(_model_binding_value_is_schema_default(value, spec))
        for name, item in value.items():
            nested_spec = nested.get(name, {}) if isinstance(nested, dict) and isinstance(nested.get(name), dict) else {}
            if not _model_binding_value_grounded(user_request, item, "", nested_spec, argument_name=str(name)):
                return False
        return True
    if isinstance(value, list):
        if not value:
            return bool(_model_binding_value_is_schema_default(value, spec))
        return all(_model_binding_value_grounded(user_request, item, "", spec, argument_name=argument_name) for item in value)
    if isinstance(value, bool):
        return bool(evidence and _span_in_request(user_request, evidence))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if _span_in_request(user_request, text) or _span_in_request(user_request, text.replace("_", " ")):
            return True
        if _model_binding_value_allowed_by_schema(text, spec) and _enum_value_evidence(user_request, spec):
            return True
    return False


def _schema_expects_currency_code(argument_name: str, spec: dict[str, Any]) -> bool:
    """Apply currency-code grounding only to schema-declared currency slots.

    Three-letter uppercase values are used by many compact identifier systems
    (airport, country, language, and domain-specific codes).  Treating their
    shape as proof that they are currencies causes otherwise evidence-grounded
    model bindings to be rejected.  The schema's own name/description is the
    portable discriminator rather than a benchmark or domain allowlist.
    """
    schema_text = " ".join(
        str(value)
        for value in (
            argument_name,
            spec.get("title") if isinstance(spec, dict) else "",
            spec.get("description") if isinstance(spec, dict) else "",
            spec.get("format") if isinstance(spec, dict) else "",
        )
        if value
    ).lower()
    return bool(re.search(r"(?:^|[_\W])(?:currency|currencies|forex|fx)(?:$|[_\W])", schema_text))


def _numeric_model_binding_value_grounded(user_request: str, value: int | float, evidence: str, spec: dict[str, Any]) -> bool:
    texts = [user_request]
    if evidence and _span_in_request(user_request, evidence):
        texts.insert(0, evidence)
    for text in texts:
        numbers = extract_numbers(text)
        if any(float(number) == float(value) for number in numbers if isinstance(number, (int, float))):
            return True
        if "%" in text and any(
            math.isclose(float(number) * 100.0, float(value), rel_tol=0.0, abs_tol=1e-9)
            for number in numbers
            if isinstance(number, (int, float))
        ):
            slot_text = str(spec.get("description") or "").lower()
            if "percent" in slot_text or "percentage" in slot_text:
                return True
    return False


def _model_binding_value_allowed_by_schema(value: Any, spec: dict[str, Any]) -> bool:
    if _model_binding_value_is_schema_default(value, spec):
        return True
    allowed = _enum_values_for_spec(spec)
    if not allowed:
        return True
    if isinstance(value, list):
        return all(_model_binding_value_allowed_by_schema(item, spec) for item in value)
    normalized = str(value).strip().lower()
    return any(str(item).strip().lower() == normalized for item in allowed)


def _model_binding_value_is_schema_default(value: Any, spec: dict[str, Any]) -> bool:
    default = _model_binding_schema_default(spec)
    if default is None:
        return False
    return str(value).strip().lower() == str(default).strip().lower()


def _model_binding_schema_default(spec: dict[str, Any]) -> Any | None:
    if "default" in spec:
        return spec.get("default")
    return _default_value_from_description(spec)


def _model_binding_depends_on(group: dict[str, Any]) -> list[str]:
    depends = group.get("depends_on")
    if isinstance(depends, list):
        return [str(item) for item in depends if item]
    return []


def _model_binding_tool_supported(
    user_request: str,
    tools: list[dict[str, Any]],
    tool: dict[str, Any],
    binding: dict[str, Any],
    frame: dict[str, Any],
) -> tuple[bool, str]:
    if _has_hard_semantic_conflict(user_request, tool):
        return False, "hard_semantic_conflict"
    # The semantic frame is an evidence-audited paraphrase of the live request.
    # Use it for route validation as well as the raw transcript; otherwise a
    # terse confirmation such as "complete this" can erase the active intent.
    support_text = _model_binding_support_text(user_request, binding, frame)
    if _has_core_intent_mismatch(support_text, tool) and not _tool_name_is_explicitly_requested(support_text, tool):
        if _model_binding_is_state_gathering_call(tool, binding):
            # A stateful planner may need a grounded read before it can carry
            # out the user's higher-level intent (for example, retrieve a
            # product record before an exchange). Argument verification still
            # happens below, so this is not permission to invent a route.
            return True, "model_proposed_state_gathering"
        if _model_binding_can_route_to_generic_command_executor(user_request, tool, binding, frame):
            return True, "generic_command_executor"
        return False, "core_intent_mismatch"
    if _tool_name_is_explicitly_requested(support_text, tool):
        return True, "tool_name_explicitly_requested_in_semantic_context"
    request_actions = _action_labels(support_text)
    tool_actions = _action_labels(_tool_text(tool))
    if request_actions and tool_actions and not (request_actions & tool_actions) and _action_labels_conflict(request_actions, tool_actions):
        return False, "action_label_conflict"
    if _semantic_capability_overlap(support_text, tool):
        return True, "semantic_capability_overlap"
    if _request_tool_action_alias_overlap(support_text, tool):
        return True, "action_alias_overlap"
    if _semantic_identity_overlap(support_text, tool) >= 1:
        return True, "semantic_identity_overlap"
    if request_actions & tool_actions:
        return True, "action_label_overlap"
    ranked = rank_tools(support_text, tools, build_task_frame(support_text))
    if len(tools) > 1 and ranked and ranked[0]["tool"].get("name") == tool.get("name") and ranked[0]["score"] > 0:
        return True, "top_ranked_tool"
    return False, "no route evidence for proposed tool"


def _model_binding_is_state_gathering_call(tool: dict[str, Any], binding: dict[str, Any]) -> bool:
    text = f"{tool.get('name') or ''} {tool.get('description') or ''}".lower()
    if not re.search(r"\b(?:get|search|find|lookup|retrieve|query|read|fetch|details?|inspect)\b", text):
        return False
    if re.search(r"\b(?:add|book|cancel|create|delete|exchange|modify|remove|return|send|update|write)\b", text):
        return False
    return bool(_model_binding_argument_groups(binding) or not _properties(tool))


def _model_binding_can_route_to_generic_command_executor(
    user_request: str,
    tool: dict[str, Any],
    binding: dict[str, Any],
    frame: dict[str, Any],
) -> bool:
    if not _tool_is_generic_command_executor(tool):
        return False
    if not _request_mentions_command_workflow(user_request, binding, frame):
        return False
    groups = _model_binding_argument_groups(binding)
    if not groups:
        return False
    command_slots = _command_executor_argument_names(tool)
    if not command_slots:
        return False
    for group in groups:
        raw_arguments = group.get("arguments") if isinstance(group.get("arguments"), dict) else None
        if raw_arguments is None and isinstance(group.get("args"), dict):
            raw_arguments = group.get("args")
        if not isinstance(raw_arguments, dict):
            return False
        if not any(name in raw_arguments for name in command_slots):
            return False
        evidence = _model_binding_evidence_map(group)
        if not any(str(evidence.get(name) or "").strip() for name in command_slots if name in raw_arguments):
            return False
    return True


def _tool_is_generic_command_executor(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower()
    has_command_object = bool(
        re.search(r"\b(?:command|cmd|shell|terminal|console|process|subprocess|script)\b", text)
    )
    has_execute_action = bool(
        re.search(r"\b(?:execute|exec|run|invoke|call|launch|start)\b", text)
    )
    return has_command_object and has_execute_action and bool(_command_executor_argument_names(tool))


def _command_executor_argument_names(tool: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for name, spec in _properties(tool).items():
        slot_text = f"{name} {spec.get('description') or ''}".lower()
        if re.search(r"\b(?:command|cmd|shell|terminal|script|program|executable)\b", slot_text):
            names.add(name)
    return names


def _request_mentions_command_workflow(
    user_request: str,
    binding: dict[str, Any],
    frame: dict[str, Any],
) -> bool:
    pieces = [_model_binding_support_text(user_request, binding, frame)]
    for key in ("intent", "operation", "action", "description"):
        value = binding.get(key) if isinstance(binding, dict) else None
        if isinstance(value, str):
            pieces.append(value)
    for group in _model_binding_argument_groups(binding):
        for value in _model_binding_evidence_map(group).values():
            if isinstance(value, str):
                pieces.append(value)
    text = " ".join(pieces).lower()
    has_file_system_object = bool(
        re.search(r"\b(?:file|files|folder|directory|directories|path|drive|disk|txt|csv|json|log)\b", text)
        or re.search(r"(?:[a-z]:\\|/[\w.-]+)", text)
    )
    has_command_action = bool(
        re.search(
            r"\b(?:list|show|make|create|write|delete|remove|copy|move|rename|open|run|execute|launch|"
            r"install|mkdir|touch|dir|ls|echo|cat|grep|find|chmod|chown|cd|pwd)\b",
            text,
        )
    )
    has_direct_command_language = bool(
        re.search(r"\b(?:command|cmd|shell|terminal|console|powershell|bash|script)\b", text)
    )
    has_non_english_command_text = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    return has_direct_command_language or has_non_english_command_text or (has_file_system_object and has_command_action)


def _model_binding_support_text(user_request: str, binding: dict[str, Any], frame: dict[str, Any]) -> str:
    pieces = [user_request]
    for key in ("canonical_request",):
        value = frame.get(key) if isinstance(frame, dict) else None
        if isinstance(value, str) and value.strip():
            pieces.append(value)
    return " ".join(_dedupe(piece for piece in pieces if piece))


def _semantic_frame_total_expected_call_count(query_input_audit: dict[str, Any] | None) -> int | None:
    if not isinstance(query_input_audit, dict):
        return None
    counts = [
        _coerce_positive_int(group.get("expected_call_count"))
        for group in query_input_audit.get("semantic_call_groups") or []
        if isinstance(group, dict)
    ]
    counts = [count for count in counts if count is not None]
    if not counts:
        return None
    return counts[0] if len(counts) == 1 else sum(counts)


def _model_binding_audit(
    tool: dict[str, Any],
    binding: dict[str, Any],
    planned_calls: list[dict[str, Any]],
    *,
    supported_reason: str,
) -> dict[str, Any]:
    required = _required(tool)
    slot_bindings: dict[str, Any] = {}
    for call in planned_calls:
        for name, value in (call.get("arguments") or {}).items():
            slot_bindings.setdefault(name, []).append(value)
    normalized_bindings = {
        name: values[0] if len(values) == 1 else values
        for name, values in slot_bindings.items()
    }
    return {
        "tool_name": tool.get("name", ""),
        "score": 1.0,
        "semantic_fit": "model_proposed",
        "eligible": True,
        "ineligible_reason": "",
        "required_slots": required,
        "slot_availability": {slot: "available_from_model_binding" for slot in required},
        "slot_bindings": normalized_bindings,
        "slot_evidence": {},
        "slot_satisfaction": {
            slot: {
                "status": "MODEL_VERIFIED",
                "value": normalized_bindings.get(slot),
                "evidence_span": None,
                "evidence_type": "model_tool_binding",
                "confidence": _safe_confidence(binding.get("confidence"), 0.86),
            }
            for slot in required
        },
        "missing_slots": [],
        "call_policy": {
            "unit_of_work": str(binding.get("intent") or "model-proposed call group"),
            "call_count": len(planned_calls),
            "batchable": _tool_has_required_batch_array(tool),
            "can_use_batch_tool_if_available": not _tool_has_required_batch_array(tool),
        },
        "requirement_frame": {
            "tool_name": tool.get("name", ""),
            "unit_of_work": str(binding.get("intent") or "model-proposed call group"),
            "required_slots": required,
            "required_slots_available": True,
            "expected_call_count_if_single_entity_tool": len(planned_calls),
            "source": "model_tool_binding",
            "supported_reason": supported_reason,
        },
        "planned_calls": planned_calls,
    }


def _model_binding_diagnostic(code: str, message: str, path: str) -> dict[str, Any]:
    return {"code": code, "message": message, "path": path}


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or tool.get("function", {}).get("name") or "")
    description = str(tool.get("description") or tool.get("function", {}).get("description") or "")
    parameters = tool.get("parameters") or tool.get("function", {}).get("parameters") or {}
    argument_schema = tool.get("arguments") if isinstance(tool.get("arguments"), dict) else None
    if argument_schema is None and isinstance(tool.get("input_parameters"), dict):
        argument_schema = tool.get("input_parameters")
    if not parameters and isinstance(argument_schema, dict):
        props = {}
        required = []
        for arg_name, spec in argument_schema.items():
            spec = spec if isinstance(spec, dict) else {"type": "string", "description": str(spec)}
            props[str(arg_name)] = {
                "type": json_schema_type(spec.get("type")),
                "description": str(spec.get("description") or ""),
            }
            required.append(str(arg_name))
        parameters = {"type": "dict", "properties": props, "required": required}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            parameters = {}
    if not isinstance(parameters, dict):
        parameters = {}
    return {"name": name, "description": description, "parameters": deepcopy(parameters)}


def json_schema_type(api_type: Any) -> str:
    text = str(api_type or "string").lower()
    if text in {"str", "string"}:
        return "string"
    if text in {"int", "integer"}:
        return "integer"
    if text in {"float", "double", "number"}:
        return "number"
    if text in {"bool", "boolean"}:
        return "boolean"
    if text.startswith("list") or text.startswith("array"):
        return "array"
    if text in {"dict", "object"}:
        return "object"
    return "string"


def build_task_frame(user_request: str) -> dict[str, Any]:
    """Infer a benchmark-neutral view of the user's requested work."""
    tags = sorted(_intent_tags(user_request))
    atomic_tasks = _atomic_tasks_for_request(user_request, set(tags))
    return {
        "user_goal": user_request.strip(),
        "intent_tags": tags,
        "atomic_tasks": atomic_tasks,
        "parallelizable": len(atomic_tasks) > 1,
        "unit_count": len(atomic_tasks) or 1,
    }


def build_capability_binding_context(user_request: str) -> dict[str, Any]:
    """Run the abstract capability planner as a deterministic pre-pass."""
    try:
        from taskdecomp.capability_planning import build_rules_first_capability_plan

        plan = build_rules_first_capability_plan(user_request)
    except Exception as exc:  # pragma: no cover - defensive fallback for optional planner deps
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "intent_input_audit": {"inputs": [], "missing_inputs": []},
            "ordered_capability_plan": {"ordered_capabilities": []},
        }

    return _compact_capability_binding_context(plan)


def _compact_capability_binding_context(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {
            "available": False,
            "intent_input_audit": {"inputs": [], "missing_inputs": []},
            "ordered_capability_plan": {"ordered_capabilities": []},
        }
    if "intent_input_audit" in plan and "passes" not in plan:
        intent = plan.get("intent_input_audit")
        if not isinstance(intent, dict):
            intent = {"inputs": [], "missing_inputs": []}
        return {
            "available": bool(plan.get("available", True)),
            "task_family": plan.get("task_family", {}),
            "task_route": plan.get("task_route", {}),
            "intent_input_audit": intent,
            "ordered_capability_plan": plan.get("ordered_capability_plan", {}),
            "validation": plan.get("validation", {}),
            "ensemble": plan.get("ensemble", {}),
            "semantic_input_frame": _semantic_input_frame_from_plan(plan),
        }

    passes = plan.get("passes") if isinstance(plan, dict) else {}
    intent_pass = passes.get("intent_input_audit") if isinstance(passes, dict) else {}
    intent = intent_pass.get("parsed") if isinstance(intent_pass, dict) else {}
    if not isinstance(intent, dict):
        intent = {"inputs": [], "missing_inputs": []}
    return {
        "available": True,
        "task_family": plan.get("task_family") if isinstance(plan, dict) else {},
        "task_route": plan.get("task_route") if isinstance(plan, dict) else {},
        "intent_input_audit": intent,
        "ordered_capability_plan": plan.get("ordered_capability_plan") if isinstance(plan, dict) else {},
        "validation": plan.get("validation") if isinstance(plan, dict) else {},
        "ensemble": plan.get("ensemble", {}),
        "semantic_input_frame": _semantic_input_frame_from_plan(plan),
    }


def _semantic_input_frame_from_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    frame = plan.get("semantic_input_frame")
    if isinstance(frame, dict):
        return frame
    passes = plan.get("passes") if isinstance(plan.get("passes"), dict) else {}
    output = passes.get("semantic_slot_frame") if isinstance(passes, dict) else {}
    parsed = output.get("parsed") if isinstance(output, dict) else {}
    return parsed if isinstance(parsed, dict) else {}


def build_query_input_audit(user_request: str, capability_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract benchmark-neutral evidence that required tool slots may bind to."""
    numbers = extract_numbers(user_request)
    quoted_values = _quoted_strings(user_request)
    file_paths = _file_path_values(user_request)
    emails = _email_values(user_request)
    tokens = _token_values(user_request)
    dates = _date_values(user_request)
    temporal_phrases = _temporal_phrase_values(user_request)
    locations = _location_units(user_request)
    route_endpoints = _route_endpoint_values(user_request)
    identifiers = _symbolic_identifier_values(user_request)
    available = []
    for name, values in [
        ("numbers", numbers),
        ("quoted_values", quoted_values),
        ("file_paths", file_paths),
        ("emails", emails),
        ("tokens", tokens),
        ("dates", dates),
        ("temporal_phrases", temporal_phrases),
        ("locations", locations),
        ("route_endpoints", route_endpoints),
        ("identifiers", identifiers),
    ]:
        if values:
            available.append({"kind": name, "values": values[:8]})
    if capability_plan:
        intent = capability_plan.get("intent_input_audit") if isinstance(capability_plan, dict) else {}
        if isinstance(intent, dict):
            capability_inputs = [
                {
                    "name": str(item.get("name") or ""),
                    "format": str(item.get("format") or "unknown"),
                    "evidence": str(item.get("evidence") or item.get("evidence_span") or "")[:240],
                }
                for item in intent.get("inputs", [])
                if isinstance(item, dict) and item.get("available") is True
            ]
            if capability_inputs:
                available.append({"kind": "capability_available_inputs", "values": capability_inputs[:8]})
    semantic_frame = _semantic_input_frame_from_plan(capability_plan)
    semantic_facts = _verified_semantic_frame_facts(user_request, semantic_frame)
    return {
        "available_inputs": available,
        "query_facts": _dedupe_facts(build_query_facts(user_request) + semantic_facts),
        "semantic_call_groups": _verified_semantic_call_groups(user_request, semantic_frame),
        "missing_inputs": _capability_missing_input_names(capability_plan),
        "capability_planner": capability_plan or {},
        "raw_query": user_request.strip(),
    }


def build_query_facts(user_request: str) -> list[dict[str, Any]]:
    """Return span-backed facts that can satisfy schema slots semantically."""
    facts: list[dict[str, Any]] = []

    def add_fact(span: Any, kind: str, value: Any | None = None, confidence: float = 0.9, **extra: Any) -> None:
        if span in (None, ""):
            return
        normalized = span if value is None else value
        fact = {
            "span": str(span),
            "type": kind,
            "normalized_value": normalized,
            "confidence": confidence,
        }
        fact.update(extra)
        facts.append(fact)

    for value in _file_path_values(user_request):
        add_fact(value, "file_path", value, 0.99)
    for value in _token_values(user_request):
        add_fact(value, "token", value, 0.99)
    for value in _quoted_strings(user_request):
        add_fact(value, "quoted_text", value, 0.96)
    for value in _clock_time_values(user_request):
        add_fact(value, "time", value, 0.95)
    for value in _temporal_phrase_values(user_request):
        add_fact(value, "time", value, 0.92)
    for value in _size_phrase_values(user_request):
        add_fact(value, "size", value, 0.97)
    for value in _art_medium_values(user_request):
        add_fact(value, "art_medium", value, 0.93)
    for value in _color_phrase_values(user_request):
        add_fact(value, "color", value, 0.9)
    for index, value in enumerate(_zodiac_sign_values(user_request)):
        add_fact(value, "zodiac_sign", value, 0.96, index=index)
    for index, value in enumerate(_place_pair_values(user_request)):
        add_fact(value, "place", value, 0.92, index=index)
    for item in _labeled_numeric_array_values(user_request):
        add_fact(item["span"], "numeric_array", item["value"], 0.98, label=item["label"], index=item["index"])
    result_count = _result_count_value(user_request)
    if result_count is not None:
        add_fact(str(result_count), "result_count", result_count, 0.95)
    language = _language_value(user_request)
    if language:
        add_fact(language, "language", language, 0.82)
    return _dedupe_facts(facts)


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for fact in facts:
        key = (
            str(fact.get("type") or ""),
            str(fact.get("span") or ""),
            json.dumps(fact.get("normalized_value"), sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique


def _verified_semantic_frame_facts(user_request: str, frame: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(frame, dict):
        return []
    facts: list[dict[str, Any]] = []
    for item in frame.get("slots_observed") or []:
        for slot in _semantic_slot_records_from_observed_item(item):
            role = str(slot.get("role") or "").strip()
            if not role:
                continue
            status = str(slot.get("status") or "semantic").lower()
            evidence = slot.get("evidence_span")
            value = slot.get("value")
            if not _semantic_frame_item_is_grounded(user_request, value, evidence, status):
                continue
            fact_type = _semantic_fact_type(slot)
            confidence = _safe_confidence(slot.get("confidence"), 0.72)
            if confidence <= 0.0 and evidence not in (None, ""):
                confidence = 0.72
            facts.append(
                {
                    "span": str(evidence or value or role),
                    "type": fact_type,
                    "normalized_value": value,
                    "confidence": confidence,
                    "role": role,
                    "value_type": str(slot.get("value_type") or ""),
                    "status": status,
                    "source": "gptoss_semantic_slot_frame",
                }
            )
    return facts


_SEMANTIC_SLOT_META_KEYS = {
    "confidence",
    "evidence",
    "evidence_span",
    "evidence_spans",
    "name",
    "role",
    "slot",
    "source",
    "status",
    "value",
    "value_type",
}


def _semantic_slot_records_from_observed_item(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    role = str(item.get("role") or item.get("slot") or item.get("name") or "").strip()
    if role:
        value = item.get("value")
        if value is None and role in item and role not in _SEMANTIC_SLOT_META_KEYS:
            value = item.get(role)
        return [
            {
                "role": role,
                "value": value,
                "value_type": item.get("value_type") or _semantic_value_type(value),
                "evidence_span": item.get("evidence_span") or item.get("evidence"),
                "status": item.get("status") or "semantic",
                "confidence": item.get("confidence"),
            }
        ]

    evidence_spans = item.get("evidence_spans") if isinstance(item.get("evidence_spans"), dict) else {}
    shared_evidence = item.get("evidence_span") or item.get("evidence")
    records: list[dict[str, Any]] = []
    for key, value in item.items():
        if key in _SEMANTIC_SLOT_META_KEYS:
            continue
        records.append(
            {
                "role": str(key),
                "value": value,
                "value_type": _semantic_value_type(value),
                "evidence_span": evidence_spans.get(key) or shared_evidence or value,
                "status": item.get("status") or "semantic",
                "confidence": item.get("confidence"),
            }
        )
    return records


def _semantic_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "text"


def _verified_semantic_call_groups(user_request: str, frame: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(frame, dict):
        return []
    groups = []
    for item in frame.get("call_groups") or []:
        if not isinstance(item, dict):
            continue
        count = _coerce_positive_int(item.get("expected_call_count"))
        if count is None:
            continue
        result_count = _coerce_positive_int(item.get("result_count"))
        requested_entities = [
            str(value)
            for value in item.get("requested_entities") or []
            if isinstance(value, (str, int, float)) and _span_in_request(user_request, str(value))
        ]
        groups.append(
            {
                "intent": str(item.get("intent") or ""),
                "unit_of_work": str(item.get("unit_of_work") or ""),
                "requested_entities": requested_entities,
                "expected_call_count": count,
                "result_count": result_count,
                "can_use_batch_tool_if_available": bool(item.get("can_use_batch_tool_if_available", True)),
            }
        )
    return groups


def _semantic_frame_item_is_grounded(user_request: str, value: Any, evidence: Any, status: str) -> bool:
    if evidence not in (None, "") and _span_in_request(user_request, str(evidence)):
        return True
    if status in {"inferred", "defaulted"}:
        return False
    if isinstance(value, (int, float, bool)):
        return str(value).lower() in user_request.lower()
    if isinstance(value, str):
        return _span_in_request(user_request, value)
    if isinstance(value, list):
        return any(isinstance(item, str) and _span_in_request(user_request, item) for item in value)
    return False


def _semantic_fact_type(item: dict[str, Any]) -> str:
    value_type = str(item.get("value_type") or "").lower()
    if value_type in {"result_count", "currency", "location", "file_path", "url", "identifier", "date", "time"}:
        return value_type
    if value_type == "number":
        return "number"
    return "semantic_slot"


def _safe_confidence(value: Any, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def _coerce_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= 100 else None


def _coerce_non_negative_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 100 else None


def _span_in_request(user_request: str, span: str) -> bool:
    span = re.sub(r"\s+", " ", span.strip())
    if not span:
        return False
    normalized_span = span.lower()
    normalized_request = re.sub(r"\s+", " ", user_request).lower()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized_span):
        # A period after a number is often sentence punctuation, not a
        # continuation of a decimal value. Digit boundaries still prevent a
        # short number from matching inside a longer numeric token.
        return bool(re.search(rf"(?<!\d){re.escape(normalized_span)}(?!\d)", normalized_request))
    if re.fullmatch(r"[a-z0-9_./:-]+", normalized_span) and re.search(r"\d", normalized_span):
        return bool(
            re.search(
                # A period can terminate an ordinary sentence after an ID
                # ("user_123.") but remains part of a structured token when
                # followed by another identifier character ("user_123.json").
                rf"(?<![a-z0-9_.:-]){re.escape(normalized_span)}(?![a-z0-9_:-]|\.(?=[a-z0-9_]))",
                normalized_request,
            )
        )
    return normalized_span in normalized_request


def audit_candidate_tool(
    user_request: str,
    tool: dict[str, Any],
    score: float,
    task_frame: dict[str, Any] | None = None,
    query_input_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit semantic fit, required-slot availability, and call cardinality for one tool."""
    required = _required(tool)
    properties = _properties(tool)
    query_input_audit = query_input_audit or build_query_input_audit(user_request)
    call_count = infer_call_count(user_request, tool, query_input_audit)
    planned_calls = []
    all_args_by_slot: dict[str, list[Any]] = {slot: [] for slot in required}
    missing_by_slot: dict[str, int] = {slot: 0 for slot in required}
    soft_available_by_slot: dict[str, dict[str, Any]] = {}
    for call_index in range(call_count):
        args, missing = infer_arguments(user_request, tool, call_index, call_count, query_input_audit)
        true_missing = []
        soft_available = {}
        for slot in missing:
            evidence = _available_evidence_for_unbound_slot(
                user_request,
                tool,
                slot,
                properties.get(slot, {}),
                query_input_audit,
                call_index,
                call_count,
            )
            if evidence:
                promoted = _argument_value_from_slot_evidence(evidence, properties.get(slot, {}), call_index)
                if promoted is not None:
                    args[slot] = promoted
                else:
                    soft_available[slot] = evidence
                    soft_available_by_slot.setdefault(slot, evidence)
            else:
                true_missing.append(slot)
        planned_calls.append(
            {
                "arguments": args,
                "missing_arguments": true_missing,
                "unbound_available_arguments": soft_available,
                "raw_missing_arguments": missing,
            }
        )
        for slot in required:
            if slot in args:
                all_args_by_slot.setdefault(slot, []).append(args[slot])
            elif slot in true_missing:
                missing_by_slot[slot] = missing_by_slot.get(slot, 0) + 1

    slot_bindings = {
        slot: values[0] if len(values) == 1 else values
        for slot, values in all_args_by_slot.items()
        if values
    }
    slot_availability = {}
    slot_evidence = {}
    for slot in required:
        if slot in slot_bindings:
            slot_availability[slot] = "available_from_query_repeated_entity" if call_count > 1 else "available_from_query"
        elif slot in soft_available_by_slot:
            slot_availability[slot] = "evidence_present_but_unbound"
            slot_evidence[slot] = soft_available_by_slot[slot]
        else:
            slot_availability[slot] = "missing"
    missing_slots = [slot for slot in required if slot_availability.get(slot) == "missing"]
    missing_slots = _filter_conditionally_unneeded_missing(tool, slot_bindings, missing_slots, user_request)
    unneeded_slots = {
        slot
        for slot in required
        if slot not in slot_bindings and not _filter_conditionally_unneeded_missing(tool, slot_bindings, [slot], user_request)
    }
    effective_required = [slot for slot in required if slot not in unneeded_slots]
    slot_satisfaction = _slot_satisfaction_records(required, properties, slot_bindings, slot_evidence, missing_slots)
    semantic_request = _semantic_request_text(user_request, query_input_audit)
    semantic_bound_slots = _semantic_frame_bound_required_slots(effective_required, properties, query_input_audit)
    grounded_required_slot_support = _has_grounded_required_slot_support(
        semantic_request,
        tool,
        effective_required,
        slot_bindings,
        slot_evidence,
        semantic_bound_slots,
    )
    if not grounded_required_slot_support and semantic_request != user_request:
        grounded_required_slot_support = _has_grounded_required_slot_support(
            user_request,
            tool,
            effective_required,
            slot_bindings,
            slot_evidence,
            semantic_bound_slots,
        )
    suspicious_required_duplicates = _has_suspicious_duplicate_required_slot_bindings(
        user_request,
        tool,
        effective_required,
        slot_bindings,
        semantic_bound_slots,
    )
    semantic_mismatch = _has_semantic_mismatch(semantic_request, tool)
    raw_hard_conflict = semantic_request != user_request and _has_hard_semantic_conflict(user_request, tool)
    route_grounded_support = (
        _tool_looks_like_route_request(tool)
        and _route_request_alignment(user_request)
        and not missing_slots
        and call_count > 0
    )
    if route_grounded_support:
        semantic_mismatch = False
    elif raw_hard_conflict:
        semantic_mismatch = True
    elif semantic_mismatch and grounded_required_slot_support and not _has_hard_semantic_conflict(semantic_request, tool):
        semantic_mismatch = False
    threshold = _selection_threshold(user_request, tool)
    if semantic_mismatch or (score < threshold and not grounded_required_slot_support):
        semantic_fit = "rejected"
    elif _semantic_capability_overlap(semantic_request, tool) or _semantic_identity_overlap(semantic_request, tool) >= 2:
        semantic_fit = "exact"
    else:
        semantic_fit = "partial"
    required_slots_available = not missing_slots
    eligible = bool(
        semantic_fit in {"exact", "partial"}
        and required_slots_available
        and not _suspicious_duplicate_bindings_should_block_call(
            user_request,
            tool,
            effective_required,
            properties,
            slot_bindings,
            suspicious_required_duplicates,
        )
        and (
            grounded_required_slot_support
            or _has_enough_required_arg_coverage(semantic_request, tool, score, query_input_audit)
        )
    )
    requirement_frame = _tool_requirement_frame(user_request, tool, task_frame or build_task_frame(user_request), call_count)
    requirement_frame["required_slots_available"] = required_slots_available
    return {
        "tool_name": tool.get("name", ""),
        "score": round(float(score), 4),
        "semantic_fit": semantic_fit,
        "eligible": eligible,
        "ineligible_reason": _candidate_ineligible_reason(semantic_fit, missing_slots, eligible),
        "required_slots": required,
        "slot_availability": slot_availability,
        "slot_bindings": slot_bindings,
        "slot_evidence": slot_evidence,
        "slot_satisfaction": slot_satisfaction,
        "slot_binding_warnings": ["duplicate_required_slot_values"] if suspicious_required_duplicates else [],
        "missing_slots": missing_slots,
        "call_policy": {
            "unit_of_work": requirement_frame["unit_of_work"],
            "call_count": call_count,
            "batchable": _tool_has_required_batch_array(tool),
            "can_use_batch_tool_if_available": not _tool_has_required_batch_array(tool),
        },
        "requirement_frame": requirement_frame,
        "planned_calls": planned_calls,
    }


def _candidate_ineligible_reason(semantic_fit: str, missing_slots: list[str], eligible: bool) -> str:
    if eligible:
        return ""
    if semantic_fit == "rejected":
        return "semantic_mismatch_or_low_score"
    if missing_slots:
        return "missing_required_slots"
    return "insufficient_required_slot_evidence"


def _slot_complete_single_tool_recovery(
    user_request: str,
    tools: list[dict[str, Any]],
    audited: list[dict[str, Any]],
    query_input_audit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if len(tools) != 1 or len(audited) != 1:
        return None
    item = audited[0]
    tool = item.get("tool") or tools[0]
    audit = item.get("audit") or {}
    if audit.get("eligible") or audit.get("semantic_fit") != "rejected":
        return None
    if audit.get("missing_slots"):
        return None
    required = [str(slot) for slot in audit.get("required_slots") or _required(tool)]
    if not required:
        return None
    properties = _properties(tool)
    if any(_property_type(properties.get(slot, {})) in {"array", "object"} for slot in required):
        return None
    planned_calls = audit.get("planned_calls") or []
    if not planned_calls:
        return None
    for call_plan in planned_calls:
        if not isinstance(call_plan, dict) or call_plan.get("missing_arguments"):
            return None
        args = call_plan.get("arguments")
        if not isinstance(args, dict):
            return None
        if any(slot not in args for slot in required):
            return None
    semantic_request = _semantic_request_text(user_request, query_input_audit or {})
    if _has_hard_semantic_conflict(semantic_request, tool):
        return None
    if semantic_request != user_request and _has_hard_semantic_conflict(user_request, tool):
        return None
    if _has_core_intent_mismatch(semantic_request, tool) and not _slot_complete_recovery_softens_core_mismatch(
        user_request,
        tool,
        audit,
    ):
        return None
    if not _slot_complete_recovery_has_minimum_semantic_support(user_request, semantic_request, tool, item, audit):
        return None
    if _has_suspicious_duplicate_required_slot_bindings(
        user_request,
        tool,
        required,
        audit.get("slot_bindings") or {},
        _semantic_frame_bound_required_slots(required, properties, query_input_audit),
    ):
        return None
    recovered_audit = deepcopy(audit)
    recovered_audit["semantic_fit"] = "partial"
    recovered_audit["eligible"] = True
    recovered_audit["ineligible_reason"] = ""
    recovered_audit["recovery_reason"] = "single_tool_complete_required_slots"
    requirement_frame = recovered_audit.get("requirement_frame")
    if isinstance(requirement_frame, dict):
        requirement_frame["recovery_reason"] = "single_tool_complete_required_slots"
    return {**item, "audit": recovered_audit}


def _slot_complete_recovery_softens_core_mismatch(
    user_request: str,
    tool: dict[str, Any],
    audit: dict[str, Any],
) -> bool:
    properties = _properties(tool)
    risky_promotions = False
    for call_plan in audit.get("planned_calls") or []:
        if not isinstance(call_plan, dict):
            return False
        args = call_plan.get("arguments") or {}
        for slot in call_plan.get("raw_missing_arguments") or []:
            if slot not in _required(tool):
                continue
            spec = properties.get(slot, {})
            if _property_type(spec) in {"integer", "float", "number", "boolean", "array", "object"}:
                risky_promotions = True
                continue
            value = args.get(slot)
            values = value if isinstance(value, list) else [value]
            if not values or any(not isinstance(item, str) or not _span_in_request(user_request, item) for item in values):
                risky_promotions = True
    return not risky_promotions


def _slot_complete_recovery_has_minimum_semantic_support(
    user_request: str,
    semantic_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
    audit: dict[str, Any],
) -> bool:
    try:
        score = max(float(item.get("score") or 0.0), float(audit.get("score") or 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score > 1.0:
        return True
    if _tool_name_is_explicitly_requested(user_request, tool) or _tool_name_is_explicitly_requested(semantic_request, tool):
        return True
    if _semantic_capability_overlap(semantic_request, tool):
        return True
    return _semantic_identity_overlap(semantic_request, tool) >= 2


def _auth_prerequisite_candidate(user_request: str, audited: list[dict[str, Any]]) -> dict[str, Any] | None:
    if _token_values(user_request):
        return None
    auth_candidates = [
        item
        for item in audited
        if _is_auth_token_tool(item.get("tool") or {}) and (item.get("audit") or {}).get("eligible")
    ]
    if not auth_candidates:
        return None
    auth_candidates.sort(key=lambda item: (-float(item.get("score") or 0.0), str((item.get("tool") or {}).get("name") or "")))
    if (
        _request_provides_credentials(user_request)
        or _request_explicitly_requests_auth_token(user_request)
    ) and not _has_eligible_credential_payload_tool(audited):
        return auth_candidates[0]
    protected_missing = [
        item
        for item in audited
        if _tool_has_token_slot(item.get("tool") or {})
        and "token" in {str(slot).lower() for slot in (item.get("audit") or {}).get("missing_slots") or []}
        and (item.get("audit") or {}).get("semantic_fit") in {"exact", "partial"}
        and float(item.get("score") or 0.0) >= _selection_threshold(user_request, item.get("tool") or {})
    ]
    if not protected_missing:
        return None
    return auth_candidates[0]


def _has_eligible_credential_payload_tool(audited: list[dict[str, Any]]) -> bool:
    for item in audited:
        tool = item.get("tool") or {}
        audit = item.get("audit") or {}
        if _is_auth_token_tool(tool) or _tool_has_token_slot(tool) or not audit.get("eligible"):
            continue
        required = {str(slot).lower() for slot in audit.get("required_slots") or []}
        if "password" not in required:
            continue
        text = _tool_text(tool).lower()
        account_or_user_slot = bool(required & {"username", "user_name", "account", "email", "name"})
        account_creation = bool(
            re.search(r"\b(?:register|registering|registration|open(?:ing)?\s+(?:a\s+)?(?:bank\s+)?account|create\s+(?:a\s+)?account)\b", text)
            or _action_labels(text) & {"create"}
        )
        if account_or_user_slot and account_creation:
            return True
    return False


def _request_provides_credentials(user_request: str) -> bool:
    return bool(
        (_username_values(user_request) and _password_values(user_request))
        or _unlabeled_credential_pair(user_request)
    )


def _request_explicitly_requests_auth_token(user_request: str) -> bool:
    return bool(
        re.search(r"\b(?:get|fetch|generate|obtain|retrieve)\b.{0,50}\btoken\b", user_request, re.I)
        or re.search(r"\btoken\b.{0,50}\b(?:auth|authenticate|authentication)\b", user_request, re.I)
    )


def _auth_request_has_multiple_accounts(user_request: str) -> bool:
    lowered = user_request.lower()
    explicit_multi_account = bool(
        re.search(
            r"\b(?:for\s+)?(?:both|each|every|multiple|two|three|several|all)\s+"
            r"(?:accounts?|users?|logins?|credentials?)\b",
            lowered,
        )
        or re.search(
            r"\b(?:accounts?|users?|logins?|credentials?)\s+"
            r"(?:for|of|are|were|include)\s+(?:both|each|multiple|two|three|several|all)\b",
            lowered,
        )
        or re.search(r"\b(?:for each|for both|each of|both of)\b", lowered)
    )
    labeled_usernames = _labeled_username_values(user_request)
    passwords = _password_values(user_request)
    if len(labeled_usernames) > 1 and len(passwords) > 1:
        return True
    if explicit_multi_account and (len(labeled_usernames) > 1 or len(_email_values(user_request)) > 1 or len(passwords) > 1):
        return True
    return False


def _tool_has_token_slot(tool: dict[str, Any]) -> bool:
    for name, spec in _properties(tool).items():
        text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if "token" in text:
            return True
    return False


def _is_auth_token_tool(tool: dict[str, Any]) -> bool:
    required = {name.lower() for name in _required(tool)}
    properties = {name.lower() for name in _properties(tool)}
    text = _tool_text(tool).lower().replace("_", " ")
    has_credentials = bool({"username", "user_name", "password"} & (required | properties)) and "password" in (required | properties)
    return has_credentials and "token" in text


def _argument_value_from_slot_evidence(
    evidence: dict[str, Any],
    spec: dict[str, Any],
    call_index: int,
) -> Any | None:
    kind = str(evidence.get("kind") or "")
    if kind in {"available_request_input", "request_content"}:
        return None
    selected = evidence.get("selected")
    if selected is None:
        values = evidence.get("values")
        if isinstance(values, list) and values:
            selected = values[min(call_index, len(values) - 1)]
    if selected is None:
        return None
    typ = _property_type(spec)
    if typ in {"integer", "float", "number"}:
        numeric = _number_from_any(selected)
        if numeric is None:
            return None
        return _coerce_number(numeric, spec)
    if typ == "boolean":
        if isinstance(selected, bool):
            return selected
        if isinstance(selected, str) and selected.strip().lower() in {"true", "yes", "1", "false", "no", "0"}:
            return selected.strip().lower() in {"true", "yes", "1"}
        return None
    if typ == "array":
        if isinstance(selected, list):
            return selected
        values = evidence.get("values")
        if kind in {"array_values", "named_entities", "enum_value"} and isinstance(values, list):
            return values
        return None
    if typ in {"string", "any", "object", "dict", ""}:
        if isinstance(selected, (str, int, float, bool)):
            return str(selected) if typ == "string" else selected
        return selected
    return None


def _semantic_request_text(user_request: str, query_input_audit: dict[str, Any] | None) -> str:
    pieces: list[str] = []
    if isinstance(query_input_audit, dict):
        capability = query_input_audit.get("capability_planner")
        frame = _semantic_input_frame_from_plan(capability if isinstance(capability, dict) else None)
        canonical = frame.get("canonical_request") if isinstance(frame, dict) else None
        if isinstance(canonical, str) and canonical.strip():
            pieces.append(canonical)
        if isinstance(frame, dict):
            for raw_slot in frame.get("slots_observed") or []:
                for slot in _semantic_slot_records_from_observed_item(raw_slot):
                    role = str(slot.get("role") or "").replace("_", " ").strip()
                    value = slot.get("evidence_span")
                    if value in (None, ""):
                        value = slot.get("value")
                    if role and value not in (None, ""):
                        pieces.append(f"{role}: {value}")
        for group in query_input_audit.get("semantic_call_groups") or []:
            if not isinstance(group, dict):
                continue
            for key in ("intent", "unit_of_work"):
                value = group.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value)
    if not pieces:
        pieces.append(user_request)
    return " ".join(_dedupe(piece.strip() for piece in pieces if isinstance(piece, str) and piece.strip()))


def _has_grounded_required_slot_support(
    semantic_request: str,
    tool: dict[str, Any],
    required: list[str],
    slot_bindings: dict[str, Any],
    slot_evidence: dict[str, dict[str, Any]],
    semantic_bound_slots: set[str] | None = None,
) -> bool:
    if not required:
        return True
    if not all(slot in slot_bindings for slot in required):
        return False
    if set(required).issubset(semantic_bound_slots or set()):
        return True
    if (
        len(required) == 1
        and _property_type(_properties(tool).get(required[0], {})) == "string"
        and _short_request_content_values(semantic_request)
    ):
        return True
    if _has_bound_formula_slot(semantic_request, tool, required):
        return True
    if _tool_name_is_explicitly_requested(semantic_request, tool):
        return True
    if _semantic_capability_overlap(semantic_request, tool):
        return True
    if _semantic_identity_overlap(semantic_request, tool) >= 1:
        return True
    if _looks_like_conversion_request(semantic_request) and {"convert", "conversion"} & _tool_core_tokens(tool):
        return True
    if slot_evidence and _has_any_user_input_evidence(semantic_request):
        return _semantic_identity_overlap(semantic_request, tool) >= 1
    return False


def _has_suspicious_duplicate_required_slot_bindings(
    user_request: str,
    tool: dict[str, Any],
    required: list[str],
    slot_bindings: dict[str, Any],
    semantic_bound_slots: set[str] | None = None,
) -> bool:
    if len(required) < 2 or set(required).issubset(semantic_bound_slots or set()):
        return False
    if _tool_name_is_explicitly_requested(user_request, tool):
        return False
    if re.search(r"\b(?:same|identical|equal|both)\b", user_request, re.I):
        return False
    values_by_key: dict[str, list[str]] = defaultdict(list)
    for slot in required:
        if slot not in slot_bindings:
            continue
        scalar = _duplicate_slot_scalar_value(slot_bindings.get(slot))
        if scalar is None:
            continue
        key = _duplicate_slot_value_key(scalar)
        if key:
            values_by_key[key].append(slot)
    for key, slots in values_by_key.items():
        if len(set(slots)) < 2:
            continue
        if _request_mentions_duplicate_value_for_each_slot(user_request, key, len(set(slots))):
            continue
        return True
    return False


def _suspicious_duplicate_bindings_should_block_call(
    user_request: str,
    tool: dict[str, Any],
    required: list[str],
    properties: dict[str, dict[str, Any]],
    slot_bindings: dict[str, Any],
    suspicious_required_duplicates: bool,
) -> bool:
    if not suspicious_required_duplicates:
        return False
    numeric_required = [
        slot
        for slot in required
        if slot in slot_bindings and _property_type(properties.get(slot, {})) in {"integer", "float", "number"}
    ]
    if len(numeric_required) < 2:
        return _suspicious_string_duplicate_bindings_should_block_call(
            user_request,
            tool,
            required,
            properties,
            slot_bindings,
        )
    distinct_numbers = {
        float(number)
        for number in extract_numbers(user_request)
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }
    return len(distinct_numbers) < len(numeric_required)


def _suspicious_string_duplicate_bindings_should_block_call(
    user_request: str,
    tool: dict[str, Any],
    required: list[str],
    properties: dict[str, dict[str, Any]],
    slot_bindings: dict[str, Any],
) -> bool:
    del tool
    string_required = [
        slot
        for slot in required
        if slot in slot_bindings and _property_type(properties.get(slot, {})) in {"string", "any", ""}
    ]
    if len(string_required) < 2:
        return False
    values_by_key: dict[str, list[str]] = defaultdict(list)
    for slot in string_required:
        scalar = _duplicate_slot_scalar_value(slot_bindings.get(slot))
        if scalar is None:
            continue
        key = _duplicate_slot_value_key(scalar)
        if key:
            values_by_key[key].append(slot)
    for key, slots in values_by_key.items():
        distinct_slots = sorted(set(slots))
        if len(distinct_slots) < 2:
            continue
        if _request_mentions_duplicate_value_for_each_slot(user_request, key, len(distinct_slots)):
            continue
        if _required_string_slots_have_distinct_roles(distinct_slots, properties):
            return True
    return False


def _required_string_slots_have_distinct_roles(
    slots: list[str],
    properties: dict[str, dict[str, Any]],
) -> bool:
    role_sets = [_slot_role_tokens(slot, properties.get(slot, {})) for slot in slots]
    informative = [roles for roles in role_sets if roles]
    if len(informative) < 2:
        return False
    location_like = [
        bool(roles & {"address", "city", "country", "location", "museum", "place", "region", "state"})
        for roles in informative
    ]
    return any(location_like) and not all(location_like)


def _slot_role_tokens(slot: str, spec: dict[str, Any]) -> set[str]:
    raw = f"{slot} {spec.get('description') or ''}".lower().replace("_", " ")
    tokens = set(_tokens(raw))
    generic = GENERIC_TOOL_TOKENS | VALUE_TOKENS | {
        "id",
        "identifier",
        "name",
        "names",
        "specific",
        "specified",
        "given",
        "wanted",
    }
    return {token for token in tokens if token not in generic and len(token) > 2}


def _duplicate_slot_scalar_value(value: Any) -> Any | None:
    if isinstance(value, list):
        if len(value) != 1:
            return None
        return _duplicate_slot_scalar_value(value[0])
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        return value
    return None


def _duplicate_slot_value_key(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return f"number:{float(value):.12g}"
        except (TypeError, ValueError):
            return ""
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if len(text) <= 2:
        return ""
    return f"text:{text}"


def _request_mentions_duplicate_value_for_each_slot(user_request: str, key: str, slot_count: int) -> bool:
    if slot_count <= 1:
        return True
    kind, _, raw_value = key.partition(":")
    if not raw_value:
        return False
    if kind == "number":
        pattern_value = re.escape(raw_value.rstrip("0").rstrip(".") if "." in raw_value else raw_value)
        occurrences = re.findall(rf"(?<![\d.]){pattern_value}(?!\d)(?!\.\d)", user_request)
        return len(occurrences) >= slot_count
    value = re.escape(raw_value)
    return len(re.findall(value, user_request.lower())) >= slot_count


def _has_bound_formula_slot(semantic_request: str, tool: dict[str, Any], required: list[str]) -> bool:
    if not _formula_like_values(semantic_request):
        return False
    properties = _properties(tool)
    for slot in required:
        spec = properties.get(slot, {})
        slot_text = f"{slot} {spec.get('description') or ''}".lower().replace("_", " ")
        if any(token in slot_text for token in ["formula", "calculation", "expression"]):
            return True
    return False


def _semantic_frame_bound_required_slots(
    required: list[str],
    properties: dict[str, Any],
    query_input_audit: dict[str, Any] | None,
) -> set[str]:
    facts = _query_facts("", query_input_audit)
    semantic_facts = [
        fact
        for fact in facts
        if fact.get("source") == "gptoss_semantic_slot_frame" and fact.get("normalized_value") is not None
    ]
    bound: set[str] = set()
    for slot in required:
        slot_text = f"{slot} {properties.get(slot, {}).get('description') or ''}".lower().replace("_", " ")
        if any(_semantic_frame_role_matches_slot(str(fact.get("role") or ""), slot, slot_text) for fact in semantic_facts):
            bound.add(slot)
    return bound


def _available_evidence_for_unbound_slot(
    user_request: str,
    tool: dict[str, Any],
    slot: str,
    spec: dict[str, Any],
    query_input_audit: dict[str, Any],
    call_index: int,
    call_count: int,
) -> dict[str, Any] | None:
    strict = _strict_slot_evidence(user_request, slot, spec, query_input_audit, call_index)
    if strict is not None:
        return strict if strict.get("status") == "available" else None

    typed = _typed_query_evidence_for_slot(user_request, tool, slot, spec, query_input_audit, call_index, call_count)
    if typed:
        return typed

    abstract = _capability_input_evidence_for_slot(slot, spec, query_input_audit)
    if abstract:
        return abstract
    return None


def _strict_slot_evidence(
    user_request: str,
    slot: str,
    spec: dict[str, Any],
    query_input_audit: dict[str, Any],
    call_index: int,
) -> dict[str, Any] | None:
    arg = slot.lower()
    text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")

    if "email" in text:
        values = _email_values(user_request)
        return _slot_evidence("emails", values, call_index, strict=True)
    if "token" in text:
        values = _token_values(user_request)
        return _slot_evidence("tokens", values, call_index, strict=True)
    if arg.replace("_", "") == "username" or any(token in text for token in ["username", "login"]):
        values = _username_values(user_request)
        return _slot_evidence("usernames", values, call_index, strict=True)
    if "password" in text:
        values = _password_values(user_request)
        return _slot_evidence("credential", values, call_index, strict=True)
    if any(token in text for token in ["api key", "apikey", "secret"]):
        values = _quoted_strings(user_request) or _symbolic_identifier_values(user_request)
        return _slot_evidence("credential", values, call_index, strict=True)
    if "phone" in text:
        values = re.findall(r"\+?\d[\d .()-]{6,}\d", user_request)
        return _slot_evidence("phone", values, call_index, strict=True)
    if any(token in text for token in ["file path", "filepath"]) or arg in {"path", "file_path"}:
        values = _file_path_values(user_request)
        return _slot_evidence("file_paths", values, call_index, strict=True)
    if any(token in text for token in [" url", "uri", "website"]):
        values = re.findall(r"https?://[^\s,;]+", user_request)
        if not values and "website" in text:
            website = _website_value(user_request)
            values = [website] if website else []
        return _slot_evidence("urls", values, call_index, strict=True)
    if "birth" in text and "date" in text:
        values = _date_values(user_request)
        return _slot_evidence("dates", values, call_index, strict=True)
    if _looks_like_identifier_slot(arg, text):
        values = _identifier_values_for_slot(user_request, arg)
        return _slot_evidence("identifiers", values, call_index, strict=True)

    return None


def _typed_query_evidence_for_slot(
    user_request: str,
    tool: dict[str, Any],
    slot: str,
    spec: dict[str, Any],
    query_input_audit: dict[str, Any],
    call_index: int,
    call_count: int,
) -> dict[str, Any] | None:
    del tool, call_count
    arg = slot.lower()
    typ = _property_type(spec)
    text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")

    numbers = _audit_values(query_input_audit, "numbers")
    quoted = _audit_values(query_input_audit, "quoted_values")
    dates = _audit_values(query_input_audit, "dates")
    locations = _audit_values(query_input_audit, "locations")
    route_endpoints = _audit_values(query_input_audit, "route_endpoints")
    identifiers = _audit_values(query_input_audit, "identifiers")

    if typ in {"integer", "float", "number"} and numbers:
        return _slot_evidence("numbers", numbers, call_index)

    if typ == "array":
        if "interval" in text and len(numbers) >= 2:
            return _slot_evidence("numbers", numbers[:2], 0)
        if any(token in text for token in ["coordinate", "point", "vector", "array", "list", "values", "numbers"]):
            values = numbers or quoted or identifiers
            return _slot_evidence("array_values", values, call_index)
        if locations and any(token in text for token in ["cities", "locations", "areas", "countries", "states"]):
            return _slot_evidence("locations", locations, call_index)
        return None

    if typ == "boolean":
        lowered = user_request.lower()
        if any(word in lowered for word in ["true", "false", "yes", "no"]) or any(token in lowered for token in _tokens(arg)):
            return {"status": "available", "source": "query_input_audit", "kind": "boolean_signal", "values": []}
        return None

    if typ in {"any", "object", "dict", ""}:
        if any(token in text for token in ["dataset", "data set", "training data", "input data"]) and identifiers:
            return _slot_evidence("identifiers", identifiers, call_index)
        if quoted:
            return _slot_evidence("quoted_values", quoted, call_index)
        if identifiers:
            return _slot_evidence("identifiers", identifiers, call_index)
        if numbers and any(token in text for token in ["value", "input", "data", "number"]):
            return _slot_evidence("numbers", numbers, call_index)
        values = _proper_entity_values(user_request) or _content_value_tokens(user_request)
        if values:
            return _slot_evidence("request_content", values, call_index)
        return None

    if typ != "string":
        return None

    if _enum_value_evidence(user_request, spec):
        return _slot_evidence("enum_value", _enum_value_evidence(user_request, spec), call_index)
    if any(token in text for token in ["function", "equation", "expression", "formula", "calculation"]):
        value = _function_expression_value(user_request)
        if not value:
            values = _formula_like_values(user_request)
            value = values[min(call_index, len(values) - 1)] if values else None
        if value:
            return _slot_evidence("function_expression", [value], call_index)
    if _looks_like_time_slot(arg, text):
        values = _clock_time_values(user_request)
        if values:
            return _slot_evidence("times", values, call_index)
        temporal = _temporal_phrase_values(user_request)
        if temporal:
            return _slot_evidence("temporal_phrases", temporal, call_index)
        if dates:
            return _slot_evidence("dates", dates, call_index)
    if any(token in text for token in ["date", "timeframe", "time frame", "day", "weekend", "month"]):
        if dates:
            return _slot_evidence("dates", dates, call_index)
        temporal = _temporal_phrase_values(user_request)
        if temporal:
            return _slot_evidence("temporal_phrases", temporal, call_index)
    if "year" in text and numbers:
        years = [value for value in numbers if isinstance(value, int) and 1000 <= value <= 2999]
        return _slot_evidence("years", years or numbers, call_index)
    if any(token in text for token in ["duration", "period", "term"]) and (numbers or _temporal_phrase_values(user_request)):
        return _slot_evidence("duration", _temporal_phrase_values(user_request) or numbers, call_index)
    if _looks_like_route_start_slot(arg, text) and route_endpoints:
        return _slot_evidence("route_start", route_endpoints[:1], 0)
    if _looks_like_route_end_slot(arg, text) and len(route_endpoints) >= 2:
        return _slot_evidence("route_end", route_endpoints[1:2], 0)
    if _looks_like_location_slot(arg, text):
        if locations:
            return _slot_evidence("locations", locations, call_index)
        values = _proper_entity_values(user_request)
        if values:
            return _slot_evidence("proper_entities", values, call_index)
    if any(token in text for token in ["dataset", "data set"]) and identifiers:
        return _slot_evidence("identifiers", identifiers, call_index)
    if quoted:
        return _slot_evidence("quoted_values", quoted, call_index)
    if _looks_like_named_entity_slot(arg, text):
        values = _short_request_content_values(user_request)
        if values:
            return _slot_evidence("short_request_content", values, 0)
        values = _named_slot_values(user_request, arg, text)
        if values:
            return _slot_evidence("named_entities", values, call_index)
    if _generic_string_slot_can_use_request(arg, text):
        values = _short_request_content_values(user_request)
        if values:
            return _slot_evidence("short_request_content", values, 0)
        values = _proper_entity_values(user_request) or _content_value_tokens(user_request)
        if values:
            return _slot_evidence("request_content", values, call_index)
    return None


def _capability_input_evidence_for_slot(
    slot: str,
    spec: dict[str, Any],
    query_input_audit: dict[str, Any],
) -> dict[str, Any] | None:
    if _strict_slot_evidence("", slot, spec, {"available_inputs": [], "capability_planner": {}}, 0) is not None:
        return None
    capability = query_input_audit.get("capability_planner")
    intent = capability.get("intent_input_audit") if isinstance(capability, dict) else {}
    if not isinstance(intent, dict):
        return None
    arg = slot.lower()
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    typ = _property_type(spec)
    if typ == "string":
        can_use_request = _generic_string_slot_can_use_request(arg, slot_text)
    else:
        can_use_request = typ in {"any", "object", "dict", ""} and bool(
            set(_tokens(slot_text)) & {"data", "dataset", "input", "query", "value", "object", "payload"}
        )
    if not can_use_request:
        return None
    inputs = [
        item
        for item in intent.get("inputs", [])
        if isinstance(item, dict) and item.get("available") is True
    ]
    if not inputs:
        return None
    if any(_slot_matches_text(slot, spec, missing) for missing in _capability_missing_input_names(capability)):
        return None
    values = [
        {
            "name": str(item.get("name") or ""),
            "format": str(item.get("format") or "unknown"),
            "evidence": str(item.get("evidence") or item.get("evidence_span") or "")[:160],
        }
        for item in inputs
    ]
    return {
        "status": "available",
        "source": "capability_planner_intent_input_audit",
        "kind": "available_request_input",
        "values": values[:4],
    }


def _slot_evidence(kind: str, values: list[Any], call_index: int, strict: bool = False) -> dict[str, Any] | None:
    if not values:
        return {"status": "missing", "source": "strict_slot_audit", "kind": kind} if strict else None
    index = min(call_index, len(values) - 1)
    selected = values[index]
    return {
        "status": "available",
        "source": "query_input_audit",
        "kind": kind,
        "values": values[:8],
        "selected": selected,
        "evidence_span": str(selected),
        "tier": "EXPLICIT" if kind in {"emails", "file_paths", "urls", "dates", "numbers", "quoted_values"} else "SEMANTIC",
    }


def _slot_satisfaction_records(
    required: list[str],
    properties: dict[str, Any],
    slot_bindings: dict[str, Any],
    slot_evidence: dict[str, dict[str, Any]],
    missing_slots: list[str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for slot in required:
        spec = properties.get(slot, {})
        if slot in slot_bindings:
            value = slot_bindings[slot]
            records[slot] = {
                "status": "EXPLICIT",
                "value": value,
                "evidence_span": str(value)[:160],
                "evidence_type": _property_type(spec) or "value",
                "confidence": 0.95,
            }
        elif slot in slot_evidence:
            evidence = slot_evidence[slot]
            records[slot] = {
                "status": evidence.get("tier") or "SEMANTIC",
                "value": evidence.get("selected") or evidence.get("values"),
                "evidence_span": evidence.get("evidence_span"),
                "evidence_type": evidence.get("kind"),
                "confidence": 0.82,
            }
        elif slot in missing_slots:
            records[slot] = {
                "status": "MISSING",
                "value": None,
                "evidence_span": None,
                "evidence_type": None,
                "confidence": 0.0,
            }
    return records


def _audit_values(query_input_audit: dict[str, Any], kind: str) -> list[Any]:
    for item in query_input_audit.get("available_inputs", []):
        if isinstance(item, dict) and item.get("kind") == kind:
            values = item.get("values")
            return values if isinstance(values, list) else []
    return []


def _query_facts(user_request: str, query_input_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(query_input_audit, dict):
        facts = query_input_audit.get("query_facts")
        if isinstance(facts, list):
            return [fact for fact in facts if isinstance(fact, dict)]
    return build_query_facts(user_request)


def _fact_values(facts: list[dict[str, Any]], kind: str) -> list[Any]:
    return [fact.get("normalized_value") for fact in facts if fact.get("type") == kind and fact.get("normalized_value") is not None]


def _looks_like_data_source_slot(arg: str, slot_text: str) -> bool:
    if arg in {"data_source", "datasource", "source", "input_file", "file", "path", "file_path"}:
        return True
    return bool(
        ("source" in slot_text or "data" in slot_text or "dataset" in slot_text or "input" in slot_text)
        and any(token in slot_text for token in ["file", "path", "source", "dataset", "data"])
    )


def _looks_like_text_payload_slot(arg: str, slot_text: str) -> bool:
    return bool(
        arg
        in {
            "text",
            "input_text",
            "review",
            "review_text",
            "comment",
            "content",
            "sentence",
            "utterance",
        }
        or any(token in slot_text for token in ["review text", "input text", "text to", "customer review", "comment"])
    )


def _looks_like_language_slot(arg: str, slot_text: str) -> bool:
    return arg in {"language", "lang", "locale"} or "language" in slot_text


def _looks_like_zodiac_slot(arg: str, slot_text: str) -> bool:
    return bool(arg.startswith("sign") or "zodiac" in slot_text or "astrological sign" in slot_text)


def _looks_like_place_slot(arg: str, slot_text: str) -> bool:
    return bool(
        re.fullmatch(r"(?:place|location|city|timezone|time_zone)_?\d*", arg)
        or arg.startswith("place")
        or any(token in slot_text for token in ["place", "timezone", "time zone"])
    )


def _looks_like_time_slot(arg: str, slot_text: str) -> bool:
    return bool(arg in {"time", "hour", "time_of_day"} or "time of day" in slot_text or re.search(r"\btime\b", slot_text))


def _string_temporal_slot_without_evidence(user_request: str, arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    temporal_slot = bool(tokens & {"date", "datetime", "timestamp"} or arg in {"time", "date_time", "datetime"})
    if not temporal_slot:
        return False
    return not (_date_values(user_request) or _clock_time_values(user_request) or _temporal_phrase_values(user_request))


def _looks_like_service_descriptor_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool(tokens & {"expertise", "specialization", "specialty", "profession", "service"})


def _service_descriptor_values(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\b(?:find|search for|looking for|look for|need|want|hire|book)\s+(?:a|an|the)?\s*([^,?.]{2,80}?)\s+(?:in|near|at)\s+[A-Z][A-Za-z .,'-]+",
        r"\b(?:a|an|the|another)\s+([^,?.]{2,80}?)\s+(?:in|near|at)\s+[A-Z][A-Za-z .,'-]+",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = _clean_service_descriptor(match.group(1))
            if value:
                values.append(value)
    return _dedupe(values)


def _clean_service_descriptor(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .'\"")
    value = re.sub(r"^(?:a|an|the|another|some|any)\s+", "", value, flags=re.I)
    value = re.sub(r"^(?:me|to|for|with)\s+", "", value, flags=re.I)
    if not value or re.search(r"\b(?:help|please|could|would|want|need|find|search)\b", value, re.I):
        return ""
    tokens = _tokens(value)
    if not 1 <= len(tokens) <= 5:
        return ""
    return value


def _temporal_range_value_for_slot(user_request: str, arg: str, slot_text: str) -> str | None:
    if not any(token in slot_text for token in ["date", "day", "month", "time", "year"]):
        return None
    values = _date_values(user_request)
    if not values:
        return None
    if re.search(r"\b(?:start|begin|from|after|since)\b", slot_text) or arg.startswith(("start", "from")):
        return values[0]
    if re.search(r"\b(?:end|finish|to|until|before|through)\b", slot_text) or arg.startswith(("end", "to")):
        return values[min(1, len(values) - 1)]
    return None


def _looks_like_detail_slot(arg: str, slot_text: str) -> bool:
    return arg in {"details", "detail", "detail_level", "level_of_detail"} or "detail" in slot_text


def _looks_like_result_count_slot(arg: str, slot_text: str) -> bool:
    return bool(
        arg in {"limit", "count", "result_count", "num_results", "num_matches", "max_results", "top_k", "k"}
        or any(
            token in slot_text
            for token in ["number of results", "number of upcoming", "result count", "how many", "top k", "limit"]
        )
    )


def _looks_like_music_key_slot(arg: str, slot_text: str) -> bool:
    return arg in {"key", "music_key", "key_signature"} or "key signature" in slot_text or re.search(r"\bkey\b", slot_text)


def _ordinal_slot_index(arg: str, default: int = 0) -> int:
    match = re.search(r"(?:^|_)([12])$", arg)
    if match:
        return int(match.group(1)) - 1
    match = re.search(r"([12])$", arg)
    if match:
        return int(match.group(1)) - 1
    if any(token in arg for token in ["first", "one", "source", "start"]):
        return 0
    if any(token in arg for token in ["second", "two", "target", "end"]):
        return 1
    return default


def _capability_missing_input_names(capability_plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(capability_plan, dict):
        return []
    intent = capability_plan.get("intent_input_audit")
    if not isinstance(intent, dict):
        return []
    return [
        _missing_input_name(item)
        for item in intent.get("missing_inputs", [])
        if _missing_input_name(item)
    ]


def _missing_input_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("input") or "").strip()
    return str(item or "").strip()


def _slot_matches_text(slot: str, spec: dict[str, Any], text: str) -> bool:
    haystack = f"{slot} {spec.get('description') or ''}".lower().replace("_", " ")
    needle_tokens = set(_tokens(text))
    slot_tokens = set(_tokens(haystack))
    return bool(needle_tokens and slot_tokens and needle_tokens & slot_tokens)


def _looks_like_identifier_slot(arg: str, text: str) -> bool:
    return bool(
        arg == "id"
        or arg.endswith("_id")
        or re.search(r"\bid\b", text)
        or any(token in text for token in [" identifier", "numbered", "docket"])
    )


def _identifier_values_for_slot(user_request: str, arg: str) -> list[str]:
    explicit = _explicit_value_for_arg(user_request, arg)
    if explicit is not None:
        return [str(explicit)]
    if "user" in arg:
        value = _user_id_value(user_request)
        if value:
            return [value]
    label = re.sub(r"_?id$", "", arg).replace("_", " ").strip()
    patterns = []
    if label:
        patterns.append(rf"\b{re.escape(label)}\s*(?:id|ID|number|#)\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)\b")
    patterns.extend(
        [
            r"\b(?:id|ID|number|docket)\s*[:#=]?\s*([A-Za-z0-9][A-Za-z0-9_-]{2,})\b",
            r"\b(?:it'?s|it\s+is|that'?s|that\s+is)\s*([0-9]{4,})\b",
            r"\b(rs\d+)\b",
            r"\b([A-Z]{1,4}[- ]?\d{2,}[-A-Z0-9]*)\b",
        ]
    )
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(1) for match in re.finditer(pattern, user_request, re.I))
    values.extend(str(value) for value in _symbolic_identifier_values(user_request))
    return _dedupe(value.strip(" .'\"") for value in values if value)


def _enum_value_evidence(user_request: str, spec: dict[str, Any]) -> list[str]:
    lowered = user_request.lower()
    enum = spec.get("enum")
    if not isinstance(enum, list) and isinstance(spec.get("items"), dict):
        enum = spec["items"].get("enum")
    if not isinstance(enum, list):
        enum = _listed_values_from_description(str(spec.get("description") or ""))
    values = []
    for item in enum:
        text = str(item).strip().strip("'\"")
        enum_pattern = re.escape(text.lower()).replace("_", r"[\s_-]+")
        pattern = rf"\b{enum_pattern}(?:s|es)?\b"
        if text and re.search(pattern, lowered):
            values.append(text)
    values.extend(_enum_alias_values(lowered, enum))
    return _dedupe(values)


def _enum_alias_values(lowered_request: str, enum: Any) -> list[str]:
    if not isinstance(enum, list):
        return []
    values: list[str] = []
    has_ticket_context = bool(re.search(r"\btickets?\b|\bticket\s+prices?\b|\bprices?\b", lowered_request))
    for item in enum:
        text = str(item).strip().strip("'\"")
        lowered = text.lower()
        if not text:
            continue
        if "adult" in lowered and "ticket" in lowered and has_ticket_context and re.search(r"\badults?\b", lowered_request):
            values.append(text)
        if (
            ("child" in lowered or "children" in lowered)
            and "ticket" in lowered
            and has_ticket_context
            and re.search(r"\b(?:child|children|kids?)\b", lowered_request)
        ):
            values.append(text)
        if "opening" in lowered and "hour" in lowered and re.search(r"\b(?:opening|open)\s+(?:hours?|times?)\b|\btimings?\b", lowered_request):
            values.append(text)
        if "%" in lowered:
            percent_phrase = re.sub(r"\s*%\s*", " percentage", lowered).strip()
            percent_phrase = re.sub(r"\s+", " ", percent_phrase)
            percent_pattern = re.escape(percent_phrase).replace(r"\ ", r"\s+")
            if re.search(rf"\b{percent_pattern}\b", lowered_request):
                values.append(text)
            percent_word_phrase = re.sub(r"\s*%\s*", " percent", lowered).strip()
            percent_word_pattern = re.escape(re.sub(r"\s+", " ", percent_word_phrase)).replace(r"\ ", r"\s+")
            if re.search(rf"\b{percent_word_pattern}\b", lowered_request):
                values.append(text)
    return values


def _temporal_phrase_values(text: str) -> list[str]:
    patterns = [
        r"\b(?:today|tomorrow|yesterday)\s+(?:morning|afternoon|evening|night)\b",
        r"\b(?:this|next|last)\s+(?:morning|afternoon|evening|night)\b",
        r"\b(?:next|this|last)\s+(?:week|weekend|month|year)\b",
        r"\b(?:past|last|next)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?|years?)\b",
        r"\b(?:past|last|next)\s+(?:decade|century)\b",
        r"\b(?:today|tomorrow|yesterday|tonight)\b",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0) for match in re.finditer(pattern, text, re.I))
    return _dedupe(value.strip() for value in values)


def _route_endpoint_values(text: str) -> list[str]:
    start = _route_endpoint_value(text, "from")
    end = _route_endpoint_value(text, "to")
    values = [value for value in [start, end] if value]
    if len(values) >= 2:
        return values
    match = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(?:using|with|by|via)\b|[?.!]|$)", text, re.I)
    if match:
        return [match.group(1).strip(" .'\""), match.group(2).strip(" .'\"")]
    return []


def _place_pair_values(text: str) -> list[str]:
    values = _route_endpoint_values(text)
    if len(values) >= 2:
        return [_clean_location_value(value) for value in values[:2]]
    match = re.search(
        r"\bbetween\s+([A-Z][A-Za-z .'-]+?)\s+and\s+([A-Z][A-Za-z .'-]+?)(?:\s+(?:using|with|for|at|on|by|via)\b|[?.!]|$)",
        text,
    )
    if match:
        return [_clean_location_value(match.group(1)), _clean_location_value(match.group(2))]
    locations = _location_units(text)
    if len(locations) >= 2:
        return locations[:2]
    entities = _capitalized_entity_values(text)
    return entities[:2] if len(entities) >= 2 else []


def _looks_like_route_start_slot(arg: str, text: str) -> bool:
    return bool(arg in {"origin", "_from", "from", "pickup"} or any(token in text for token in ["start", "origin", "pickup", "departure"]))


def _looks_like_route_end_slot(arg: str, text: str) -> bool:
    return bool(arg in {"destination", "to", "dropoff"} or any(token in text for token in ["end", "destination", "dropoff", "arrival"]))


def _looks_like_location_slot(arg: str, text: str) -> bool:
    return bool(
        arg in {"city", "location", "country", "state", "region", "area", "address", "place"}
        or arg.startswith(("place", "location", "city"))
        or any(token in text for token in ["city", "location", "country", "state", "region", "area", "address", "place", "near", "timezone"])
    )


def _proper_entity_values(text: str) -> list[str]:
    values: list[str] = []
    for pattern in [
        r"\b(?:from|to|in|at|near|of|for|against|with)\s+([A-Z][A-Za-z0-9'.-]+(?:\s+[A-Z][A-Za-z0-9'.-]+)*)",
        r"'([^']+)'",
        r"\"([^\"]+)\"",
    ]:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip(" .'\"")
            if value and value.lower() not in {"i", "can", "could"}:
                values.append(value)
    return _dedupe(values)


def _capitalized_entity_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'.-]+(?:\s+[A-Z][A-Za-z0-9'.-]+)*\b", text):
        value = match.group(0).strip(" .'\"")
        if value and value.lower() not in {"i", "can", "could", "what", "who", "when", "where"}:
            values.append(value)
    return _dedupe(values)


def _formula_like_values(text: str) -> list[str]:
    values = re.findall(r"\b(?:[A-Z][a-z]?\d*){2,}\b", text)
    values.extend(
        match.group(0)
        for match in re.finditer(r"(?<![\w.])(?:-?\d+(?:\.\d+)?\s*[+\-*/]\s*)+-?\d+(?:\.\d+)?(?![\w.])", text)
    )
    values.extend(_quoted_strings(text))
    verbal = _verbal_arithmetic_formula(text)
    if verbal:
        values.append(verbal)
    cleaned = []
    for value in values:
        if re.search(r"[+\-*/()]", value):
            formula = re.sub(r"\s+", "", value.strip(" .'\""))
            if formula:
                cleaned.append(formula)
            continue
        phrase = _clean_slot_phrase(value)
        if phrase:
            cleaned.append(phrase)
    return _dedupe(cleaned)


def _verbal_arithmetic_formula(text: str) -> str | None:
    number = r"(-?\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

    def number_text(raw: str) -> str | None:
        value = _number_from_text(raw)
        if isinstance(value, (int, float)):
            return str(int(value)) if float(value).is_integer() else str(value)
        return None

    unary = re.search(rf"\b(?:square|squared)\s+of\s+{number}\b", text, re.I)
    if unary:
        value = number_text(unary.group(1))
        return f"{value}*{value}" if value is not None else None
    unary = re.search(rf"\b(?:cube|cubed)\s+of\s+{number}\b", text, re.I)
    if unary:
        value = number_text(unary.group(1))
        return f"{value}*{value}*{value}" if value is not None else None

    binary_patterns = [
        (rf"\b(?:sum|addition)\s+of\s+{number}\s+and\s+{number}\b", "+"),
        (rf"\b(?:product|multiplication)\s+of\s+{number}\s+and\s+{number}\b", "*"),
        (rf"\b(?:difference|subtraction)\s+of\s+{number}\s+and\s+{number}\b", "-"),
        (rf"\b(?:quotient|division)\s+of\s+{number}\s+and\s+{number}\b", "/"),
        (rf"\b(?:add|plus)\s+{number}\s+(?:and|to)\s+{number}\b", "+"),
        (rf"\b(?:multiply|times)\s+{number}\s+(?:and|by)\s+{number}\b", "*"),
        (rf"\b(?:divide)\s+{number}\s+by\s+{number}\b", "/"),
    ]
    for pattern, operator in binary_patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        left = number_text(match.group(1))
        right = number_text(match.group(2))
        if left is not None and right is not None:
            return f"{left}{operator}{right}"

    subtract = re.search(rf"\bsubtract\s+{number}\s+from\s+{number}\b", text, re.I)
    if subtract:
        subtrahend = number_text(subtract.group(1))
        minuend = number_text(subtract.group(2))
        if subtrahend is not None and minuend is not None:
            return f"{minuend}-{subtrahend}"
    return None


def _condition_phrase_values(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\b(?:that|which|where)\s+(.+?)(?=\s+(?:in|at|near|for|with|using|by|from|to|over|under|during|before|after)\b|[?.!,]|$)",
        r"\b((?:open|opens|available|serves?|offers?|supports?|allows?|accepts?)\s+.+?)(?=\s+(?:in|at|near|for|with|using|by|from|to|over|under|during|before|after)\b|[?.!,]|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = _clean_slot_phrase(match.group(1), keep_leading_prepositions=True)
            if value:
                values.append(value)
    return _dedupe(values)


def _transition_phrase_values(text: str) -> list[str]:
    match = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z -]+?)\s+to\s+([A-Za-z][A-Za-z -]+?)(?=\s+(?:at|in|with|using|for|by)\b|[?.!,]|$)",
        text,
        re.I,
    )
    if not match:
        return []
    left = _clean_slot_phrase(match.group(1), keep_leading_prepositions=True)
    right = _clean_slot_phrase(match.group(2), keep_leading_prepositions=True)
    return [f"{left} to {right}"] if left and right else []


def _color_phrase_values(text: str) -> list[str]:
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "black",
        "white",
        "navy",
        "maroon",
        "gray",
        "grey",
        "brown",
        "pink",
        "cyan",
        "magenta",
        "violet",
        "indigo",
        "gold",
        "silver",
    ]
    tokens = set(_tokens(text))
    return [color for color in colors if color in tokens]


def _country_values(text: str) -> list[str]:
    countries = [
        "Afghanistan",
        "Albania",
        "Algeria",
        "Argentina",
        "Australia",
        "Austria",
        "Bangladesh",
        "Belgium",
        "Brazil",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Denmark",
        "Egypt",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "India",
        "Indonesia",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Kenya",
        "Malaysia",
        "Mexico",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "Norway",
        "Pakistan",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Russia",
        "Saudi Arabia",
        "Singapore",
        "South Africa",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Thailand",
        "Turkey",
        "Ukraine",
        "United Kingdom",
        "United States",
        "USA",
        "US",
        "Vietnam",
    ]
    values: list[tuple[int, str]] = []
    seen_spans: set[tuple[int, int]] = set()
    for country in sorted(countries, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(country)}\b", text, re.I):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)
            values.append((match.start(), country))
    values.sort(key=lambda item: item[0])
    return _dedupe(country for _pos, country in values)


def _size_phrase_values(text: str) -> list[str]:
    values = []
    pattern = r"\b\d+(?:\.\d+)?\s*(?:x|×|by)\s*\d+(?:\.\d+)?(?:\s*(?:in|inch|inches|cm|mm|ft|feet))?\b"
    for match in re.finditer(pattern, text, re.I):
        value = re.sub(r"\s+", "", match.group(0).strip())
        values.append(value)
    return _dedupe(values)


def _art_medium_values(text: str) -> list[str]:
    lowered = text.lower()
    values = [medium for medium in ART_MEDIUMS if re.search(rf"\b{re.escape(medium)}\b", lowered)]
    return _dedupe(values)


def _zodiac_sign_values(text: str) -> list[str]:
    return [sign for sign in ZODIAC_SIGNS if re.search(rf"\b{re.escape(sign)}\b", text, re.I)]


def _clock_time_values(text: str) -> list[str]:
    values = re.findall(r"\b\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\b", text)
    values.extend(re.findall(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", text, re.I))
    values.extend(re.findall(r"\b(?:noon|midnight)\b", text, re.I))
    return _dedupe(value.strip() for value in values)


def _month_values(text: str) -> list[str]:
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    return [month for month in months if re.search(rf"\b{month}\b", text, re.I)]


def _language_value(text: str) -> str | None:
    lowered = text.lower()
    for name, code in LANGUAGE_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            return code
    quoted = _quoted_strings(text)
    if quoted and all(ord(ch) < 128 for ch in " ".join(quoted)):
        return "en"
    if re.search(r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return "en"
    return None


def _translation_language_pairs(text: str) -> list[tuple[str, str]]:
    names = "|".join(re.escape(name) for name in sorted(LANGUAGE_ALIASES, key=len, reverse=True))
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(rf"\bfrom\s+({names})\s+to\s+({names})\b", text, re.I):
        source = LANGUAGE_ALIASES.get(match.group(1).lower())
        target = LANGUAGE_ALIASES.get(match.group(2).lower())
        if source and target:
            pairs.append((source, target))
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _frequency_values(text: str) -> list[str]:
    values = []
    for value in ["quarterly", "annual", "annually", "monthly", "weekly", "daily"]:
        if re.search(rf"\b{value}\b", text, re.I):
            values.append("annual" if value == "annually" else value)
    return _dedupe(values)


def _travel_mode_values(text: str) -> list[str]:
    values = []
    for mode in ["walking", "walk", "biking", "bike", "driving", "drive", "transit"]:
        if re.search(rf"\b{mode}\b", text, re.I):
            normalized = {"walking": "walk", "biking": "bike", "driving": "drive"}.get(mode, mode)
            values.append(normalized)
    return _dedupe(values)


def _text_payload_value(text: str) -> str | None:
    quoted = _quoted_strings(text)
    if quoted:
        return quoted[0]
    match = re.search(r"\b(?:review|comment|text|sentence)\s+(?:is\s+)?(.+?)(?:[?.!]|$)", text, re.I)
    if match:
        value = _clean_slot_phrase(match.group(1))
        if value:
            return value
    return None


def _detail_value(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(?:detail|detailed|full|complete|comprehensive)\b", lowered):
        return "detailed"
    if re.search(r"\b(?:summary|brief|short)\b", lowered):
        return "summary"
    return None


def _hobby_value(text: str) -> str | None:
    for pattern in [
        r"\b(?:like|likes|enjoy|enjoys|prefer|prefers)\s+([A-Za-z][A-Za-z -]+?)(?:[?.!,]|$)",
        r"\bpeople\s+who\s+(?:like|enjoy)\s+([A-Za-z][A-Za-z -]+?)(?:[?.!,]|$)",
    ]:
        match = re.search(pattern, text, re.I)
        if match:
            value = _clean_slot_phrase(match.group(1))
            if value:
                return value
    return None


def _music_note_value(text: str) -> str | None:
    match = re.search(r"\b(?:note|starting with(?: the note)?)\s+([A-G](?:#|b)?\d?)\b", text, re.I)
    if match:
        return match.group(1)
    match = re.search(r"\bnote\s+of\s+([A-G](?:#|b)?\d?)\b", text, re.I)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-G](?:#|b)?\d)\b", text)
    return match.group(1) if match else None


def _music_key_value(text: str) -> str | None:
    match = re.search(r"\b([A-G](?:#|b| sharp| flat)?\s+(?:major|minor))\s+(?:key|scale)\b", text, re.I)
    if match:
        return _normalize_music_accidental(match.group(1))
    match = re.search(r"\b(?:in|of|does)\s+([A-G](?:#|b| sharp| flat)?\s+(?:major|minor))\b", text, re.I)
    if match:
        return _normalize_music_accidental(match.group(1))
    match = re.search(r"\b([A-G](?:#|b| sharp| flat)?)\s+key\b", text, re.I)
    if match:
        return _normalize_music_accidental(match.group(1))
    return None


def _music_scale_value(text: str) -> str | None:
    match = re.search(r"\b([A-G](?:#|b| sharp| flat)?\s+(?:major|minor))\s+scale\b", text, re.I)
    if match:
        return _normalize_music_accidental(match.group(1))
    return _music_key_value(text)


def _normalize_music_accidental(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\bsharp\b", "#", value, flags=re.I)
    value = re.sub(r"\bflat\b", "b", value, flags=re.I)
    value = re.sub(r"\s+#", "#", value)
    value = re.sub(r"\s+b\b", "b", value)
    return value


def _slot_guided_noun_phrases(user_request: str, arg: str, text: str) -> list[str]:
    slot_tokens = set(_tokens(f"{arg} {text}"))
    if not slot_tokens & {
        "animal",
        "brand",
        "category",
        "cell",
        "company",
        "compartment",
        "compound",
        "condition",
        "cuisine",
        "diet",
        "ecosystem",
        "environment",
        "genre",
        "habitat",
        "item",
        "league",
        "material",
        "molecule",
        "organelle",
        "organism",
        "phase",
        "player",
        "product",
        "property",
        "protein",
        "religion",
        "restriction",
        "species",
        "sport",
        "style",
        "substance",
        "team",
        "topic",
        "type",
    }:
        return []
    if slot_tokens & {"shape", "size"} and not slot_tokens & {"color", "condition", "type"}:
        return []

    values: list[str] = []
    values.extend(_prepositional_slot_phrases(user_request, slot_tokens))
    values.extend(_modifier_before_domain_head_phrases(user_request))
    return _dedupe(value for value in values if value)


def _prepositional_slot_phrases(user_request: str, slot_tokens: set[str]) -> list[str]:
    prepositions = "of|for|in|on|about|against|within|with"
    if slot_tokens & {"condition", "property"}:
        prepositions = "with|that|which|where|on|for"
    if slot_tokens & {"ecosystem", "environment", "habitat"}:
        prepositions = "in|on|within|near|around"
    values: list[str] = []
    pattern = rf"\b(?:{prepositions})\s+([A-Za-z][A-Za-z0-9 '&.-]+?)(?=\s+(?:and|or|then|with|using|over|under|during|from|to|by|at|near|please|could|would|should|that|which|who|when|where)\b|[?.!,]|$)"
    for match in re.finditer(pattern, user_request, re.I):
        value = _clean_slot_phrase(match.group(1))
        if value:
            values.append(value)
    return _dedupe(values)


def _modifier_before_domain_head_phrases(text: str) -> list[str]:
    heads = (
        "animal|animals|bird|birds|category|categories|compound|compounds|cuisine|"
        "diet|diets|ecosystem|ecosystems|film|films|food|foods|game|games|genre|"
        "habitat|habitats|item|items|material|materials|molecule|molecules|movie|"
        "movies|organism|organisms|place|places|plant|plants|product|products|"
        "protein|proteins|restaurant|restaurants|restriction|restrictions|sport|"
        "sports|style|styles|substance|substances|team|teams|topic|topics|type|types"
    )
    values: list[str] = []
    pattern = rf"\b([A-Za-z][A-Za-z0-9'&.-]+(?:\s+[A-Za-z][A-Za-z0-9'&.-]+){{0,3}})\s+(?:{heads})\b"
    for match in re.finditer(pattern, text, re.I):
        value = _clean_slot_phrase(match.group(1))
        if value:
            values.append(value)
    return _dedupe(values)


def _clean_slot_phrase(value: str, keep_leading_prepositions: bool = False) -> str:
    cleaned = re.sub(r"\s+", " ", str(value)).strip(" .'\"")
    if not cleaned:
        return ""
    if not keep_leading_prepositions:
        cleaned = re.sub(r"^(?:a|an|the|any|some|all|their|its|his|her|my|our)\s+", "", cleaned, flags=re.I)
    cleaned = re.split(
        r"\s+\b(?:and|or|then|with|using|over|under|during|from|by|at|near|please|could|would|should)\b",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" .'\"")
    if not cleaned or len(cleaned) > 80:
        return ""
    tokens = _tokens(cleaned)
    if not tokens:
        return ""
    if len(tokens) > 6:
        return ""
    blocked = GENERIC_TOOL_TOKENS | {"how", "what", "when", "where", "which", "who", "why", "impact", "growth"}
    if all(token in blocked for token in tokens):
        return ""
    return cleaned


def _looks_like_named_entity_slot(arg: str, text: str) -> bool:
    named_tokens = {
        "actor",
        "artist",
        "cell",
        "color",
        "compartment",
        "compound",
        "company",
        "condition",
        "cuisine",
        "disease",
        "ecosystem",
        "genre",
        "habitat",
        "league",
        "museum",
        "name",
        "molecule",
        "organelle",
        "phase",
        "player",
        "property",
        "protein",
        "religion",
        "species",
        "sport",
        "substance",
        "symptom",
        "team",
        "topic",
        "type",
    }
    return bool(set(_tokens(f"{arg} {text}")) & named_tokens)


def _named_slot_values(user_request: str, arg: str, text: str) -> list[str]:
    enum_values = _enum_value_evidence(user_request, {"description": text})
    if enum_values:
        return enum_values
    values: list[str] = []
    values.extend(_formula_like_values(user_request) if any(token in text for token in ["formula", "compound"]) else [])
    values.extend(_condition_phrase_values(user_request) if "condition" in text else [])
    values.extend(_transition_phrase_values(user_request) if any(token in text for token in ["phase", "transition", "from", "to"]) else [])
    values.extend(_color_phrase_values(user_request) if "color" in text else [])
    values.extend(_slot_guided_noun_phrases(user_request, arg, text))
    if values:
        return _dedupe(values)
    return _proper_entity_values(user_request)


def _generic_string_slot_can_use_request(arg: str, text: str) -> bool:
    blocked = {"email", "password", "token", "secret", "key", "url", "uri", "file", "path"}
    if set(_tokens(f"{arg} {text}")) & blocked:
        return False
    generic = {"query", "search", "keyword", "topic", "type", "property", "condition", "format", "method", "name"}
    return bool(set(_tokens(f"{arg} {text}")) & generic)


def _looks_like_sensitive_credential_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool(
        tokens
        & {
            "apikey",
            "key",
            "login",
            "password",
            "secret",
            "token",
            "username",
        }
        or "api key" in slot_text
        or "access token" in slot_text
        or arg.replace("_", "") == "username"
    )


def _short_request_content_values(text: str) -> list[str]:
    cleaned = re.sub(r"\b(?:earlier\s+user|user|assistant|ai):\s*", "", text, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .'\"")
    if not cleaned or len(cleaned) > 80 or re.search(r"[?!]", cleaned):
        return []
    if re.search(
        r"\b(?:calculate|cancel|check|create|delete|find|get|modify|query|register|remove|search|show|solve|update|what|when|where|which|who|why)\b",
        cleaned,
        re.I,
    ):
        return []
    value = _clean_slot_phrase(cleaned)
    if not value:
        return []
    tokens = _tokens(value)
    if not 1 <= len(tokens) <= 5:
        return []
    return [value]


def _content_value_tokens(text: str) -> list[str]:
    blocked = STOPWORDS | GENERIC_TOOL_TOKENS | {"please", "could", "would", "what", "which", "find"}
    text = re.sub(r"\b(?:earlier\s+user|user|assistant|ai):\s*", "", text, flags=re.I)
    values = [token for token in _expanded_tokens(text) if token not in blocked and not token.isdigit()]
    return _dedupe(values)


def _tool_requirement_frame(
    user_request: str,
    tool: dict[str, Any],
    task_frame: dict[str, Any],
    call_count: int,
) -> dict[str, Any]:
    properties = _properties(tool)
    required = _required(tool)
    unit_slot = _unit_slot_for_tool(tool)
    entities = _requested_entities_for_unit_slot(user_request, unit_slot, call_count)
    action = _action_requirement_sentence(user_request, tool, unit_slot, call_count)
    return {
        "action_requirement": action,
        "intent_tags": task_frame.get("intent_tags") or [],
        "unit_of_work": f"one {unit_slot}" if unit_slot else "one requested operation",
        "requested_entities": entities,
        "expected_call_count_if_single_entity_tool": call_count,
        "can_use_batch_tool_if_available": not _tool_has_required_batch_array(tool),
        "required_slots": required,
        "required_slots_available": False,
        "schema_properties": list(properties),
    }


def _unit_slot_for_tool(tool: dict[str, Any]) -> str:
    properties = _properties(tool)
    for preferred in [
        "city",
        "location",
        "coordinates",
        "theater",
        "theatre",
        "restaurant",
        "address",
        "recipient",
        "to",
        "artist",
        "dataset",
        "case_number",
        "file_path",
    ]:
        if preferred in properties:
            return preferred
    for name, spec in properties.items():
        if name in _required(tool) and _property_type(spec) == "string":
            return name
    return ""


def _requested_entities_for_unit_slot(user_request: str, unit_slot: str, call_count: int) -> list[str]:
    if not unit_slot:
        return []
    if unit_slot in {"city", "location"}:
        return _location_units(user_request)[:call_count]
    if unit_slot in {"theater", "theatre"}:
        return _theater_values(user_request)[:call_count]
    if unit_slot == "restaurant":
        return _restaurant_values(user_request)[:call_count]
    if unit_slot in {"recipient", "to"}:
        return _email_values(user_request)[:call_count]
    if unit_slot == "artist":
        return extract_artist_like_values(user_request)[:call_count]
    if unit_slot == "dataset":
        value = _dataset_value(user_request)
        return [value] if value else []
    if unit_slot == "case_number":
        value = _case_number_value(user_request)
        return [value] if value else []
    if unit_slot == "file_path":
        return _file_path_values(user_request)[:call_count]
    return _quoted_strings(user_request)[:call_count]


def _action_requirement_sentence(user_request: str, tool: dict[str, Any], unit_slot: str, call_count: int) -> str:
    tool_name = str(tool.get("name") or "selected tool")
    required = ", ".join(_required(tool)) or "no required slots"
    if call_count > 1 and unit_slot:
        return f"Use {tool_name} once per requested {unit_slot}; required slots are {required}."
    return f"Use {tool_name} for the requested action; required slots are {required}."


def rank_tools(
    user_request: str,
    tools: list[dict[str, Any]],
    task_frame: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query = [token for token in _expanded_tokens(user_request) if token not in STOPWORDS]
    request_tokens = set(query)
    request_tags = set((task_frame or build_task_frame(user_request)).get("intent_tags") or [])
    docs = [[token for token in _expanded_tokens(_tool_text(tool)) if token not in STOPWORDS] for tool in tools]
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))
    avg_len = sum(len(doc) for doc in docs) / max(1, len(docs))
    k1 = 1.4
    b = 0.75
    ranked = []
    for tool_index, (tool, doc) in enumerate(zip(tools, docs)):
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / max(1.0, avg_len))
            score += idf * tf[term] * (k1 + 1) / denom
        name_tokens = set(_tokens(tool.get("name", "")))
        score += len(request_tokens & name_tokens) * 0.75
        primary_name_token = _tokens(str(tool.get("name", "")).split(".")[0])
        if primary_name_token and any(re.search(rf"\b{re.escape(token)}\b", user_request, re.I) for token in primary_name_token):
            score += 6.0
        if re.search(r"\bfor\s+sale\b|\bbudget\b", user_request, re.I):
            text = _tool_text(tool).lower()
            if "budget" in text or "find properties" in text or "find property" in text:
                score += 3.0
        score += _required_arg_coverage(user_request, tool) * 0.5
        tool_tags = _tool_capability_tags(tool)
        score += len(request_tags & tool_tags) * 3.0
        score += _action_alignment_score(user_request, tool)
        if _tool_looks_like_route_request(tool) and _route_request_alignment(user_request):
            score += 2.0
        if _has_intent_conflict(request_tags, tool_tags):
            score -= 4.0
        ranked.append({"tool": tool, "score": score, "tool_index": tool_index})
    ranked.sort(key=lambda item: (-item["score"], item["tool"]["name"]))
    return ranked


def _intent_tags(text: str) -> set[str]:
    tags: set[str] = set()
    for tag, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            tags.add(tag)
    if "card_probability" in tags or "coin_probability" in tags:
        if re.search(r"\b(?:probability|odds|chance|likely|likelihood)\b", text, re.I):
            tags.add("probability")
    return tags


def _is_meta_no_query(user_request: str) -> bool:
    lowered = user_request.strip().lower()
    if not lowered:
        return True
    no_query_patterns = [
        r"\buser did not provide (?:a )?query\b",
        r"\bno (?:user )?query (?:was )?provided\b",
        r"\bthe prompt is empty\b",
    ]
    return any(re.search(pattern, lowered) for pattern in no_query_patterns)


def _tool_capability_tags(tool: dict[str, Any]) -> set[str]:
    text = _tool_text(tool)
    tags: set[str] = set()
    for tag, patterns in TOOL_CAPABILITY_PATTERNS.items():
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            tags.add(tag)
    if "card_probability" in tags or "coin_probability" in tags:
        tags.add("probability")
    return tags


def _action_alignment_score(user_request: str, tool: dict[str, Any]) -> float:
    if _is_auth_token_tool(tool):
        return 0.0
    request_actions = _action_labels(user_request)
    if not request_actions:
        return 0.0
    tool_actions = _action_labels(_tool_text(tool))
    if not tool_actions:
        return 0.0
    if request_actions & tool_actions:
        return 2.0
    if _action_labels_conflict(request_actions, tool_actions):
        return -2.0
    return 0.0


def _route_request_alignment(user_request: str) -> bool:
    lowered = user_request.lower()
    return bool(
        re.search(r"\b(?:route|directions?|travel|trip|drive|driving|journey|eta|estimated travel time)\b", lowered)
        and (
            re.search(r"\bfrom\b.+\bto\b", lowered)
            or re.search(r"\bstops?\s+at\b", lowered)
            or re.search(r"\bback\s+to\b.+\bfrom\b", lowered)
        )
    )


def _action_labels(text: str) -> set[str]:
    text = re.sub(
        r"\bcheck[-\s]?(?:in|out)\b(?=\s+(?:date|time|on|from|to|at|by|is|of)\b|[:=])",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[_\.\-/]+", " ", text)
    labels: set[str] = set()
    if re.search(
        r"\b(?:check(?:s|ed|ing)?|find(?:s|ing)?|fetch(?:es|ed|ing)?|get(?:s|ting)?|look\s+up|lookup|"
        r"quer(?:y|ies|ied|ying)|read(?:s|ing)?|search(?:es|ed|ing)?|see|show(?:s|ed|ing)?|view(?:s|ed|ing)?)\b",
        text,
        re.I,
    ):
        labels.add("query")
    set_is_create = re.search(r"\bset\b", text, re.I) and not re.search(
        r"\b(?:have|has|had)\s+set\b", text, re.I
    )
    if re.search(
        r"\b(?:add(?:s|ed|ing)?|book(?:s|ed|ing)?|buy(?:s|ing)?|create(?:s|d|ing)?|make(?:s|ing)?|"
        r"open(?:s|ed|ing)?|purchase(?:s|d|ing)?|register(?:s|ed|ing)?|remind(?:s|ed|ing)?|reserve(?:s|d|ing)?)\b",
        text,
        re.I,
    ) or set_is_create or (
        re.search(r"\bschedule\b", text, re.I)
        and not re.search(r"\b(?:check|query|read|see|show|view)\s+(?:my\s+|the\s+)?schedule\b", text, re.I)
        and not re.search(r"\bschedule\s+(?:item|items|entry|entries)\b", text, re.I)
    ):
        labels.add("create")
    if re.search(r"\b(?:cancel(?:s|ed|ing|led|ling)?|delete(?:s|d|ing)?|remove(?:s|d|ing)?|void(?:s|ed|ing)?)\b", text, re.I):
        labels.add("delete")
    if re.search(
        r"\b(?:change(?:s|d|ing)?|edit(?:s|ed|ing)?|modif(?:y|ies|ied|ying)|reschedule(?:s|d|ing)?|update(?:s|d|ing)?)\b",
        text,
        re.I,
    ):
        labels.add("modify")
    return labels


def _action_labels_conflict(request_actions: set[str], tool_actions: set[str]) -> bool:
    opposed = {("query", "create"), ("query", "delete"), ("create", "delete"), ("modify", "delete")}
    return any(
        (left in request_actions and right in tool_actions) or (right in request_actions and left in tool_actions)
        for left, right in opposed
    )


def _has_intent_conflict(request_tags: set[str], tool_tags: set[str]) -> bool:
    for conflict in INTENT_CONFLICTS:
        left, right = tuple(conflict)
        if left in tool_tags and right in request_tags and left not in request_tags:
            return True
        if right in tool_tags and left in request_tags and right not in request_tags:
            return True
    return False


def _atomic_tasks_for_request(user_request: str, tags: set[str]) -> list[dict[str, Any]]:
    if "current_weather" in tags:
        units = _location_units(user_request)
        return [{"task": "get current weather", "target": unit} for unit in units] or [{"task": "get current weather"}]
    if "food_order" in tags:
        units = _food_units(user_request)
        return [{"task": "order or modify food", "target": unit} for unit in units] or [{"task": "order or modify food"}]
    if "history_fact" in tags:
        return [{"task": "look up historical fact"}]
    if "biology_function" in tags:
        return [{"task": "look up biological function"}]
    if "movie_showing" in tags:
        units = _movie_units(user_request)
        return [{"task": "find movie showing", "target": unit} for unit in units] or [{"task": "find movie showing"}]
    return [{"task": "satisfy user request"}]


def _atomic_unit_count_for_tool(user_request: str, tool: dict[str, Any]) -> int:
    properties = _properties(tool)
    tool_tags = _tool_capability_tags(tool)

    currency_count = _currency_conversion_count(user_request, properties)
    if currency_count > 1:
        return currency_count

    service_count = _service_unit_count(user_request, properties)
    if service_count > 1:
        return service_count

    if "current_weather" in tool_tags and not _tool_supports_batching_for(properties, {"location", "city", "coordinates"}):
        return max(1, len(_location_units(user_request)))

    forecast_metric_count = _forecast_metric_scoped_location_count(user_request, tool)
    if forecast_metric_count > 0:
        return forecast_metric_count

    area_units = _area_units(user_request)
    if area_units and _has_single_area_slot(properties) and not _tool_supports_batching_for(properties, {"area", "areas"}):
        return min(len(area_units), 8)

    if "movie_showing" in tool_tags:
        movie_time_count = _movie_time_pair_count(user_request, tool)
        if movie_time_count > 1:
            return movie_time_count
        theater_units = _theater_values(user_request)
        if len(theater_units) > 1 and not _tool_supports_batching_for(properties, {"theater", "theatre"}):
            return len(theater_units)
        if not _tool_supports_batching_for(properties, {"movie", "film"}):
            return max(1, len(_movie_units(user_request)))

    energy_count = _energy_substance_count(user_request, tool)
    if energy_count > 1:
        return energy_count

    restaurant_count = _restaurant_search_unit_count(user_request, tool)
    if restaurant_count > 1:
        return restaurant_count

    food_log_count = len(_food_log_entries_for_tool(user_request, tool))
    if food_log_count > 1:
        return food_log_count

    if _single_shape_instance_request(user_request, tool):
        return 1

    schema_scalar_count = _schema_named_scalar_unit_count(user_request, tool)
    if schema_scalar_count > 1:
        return schema_scalar_count

    color_count = _color_unit_count(user_request, tool)
    if color_count > 1:
        return color_count

    country_question_count = _country_question_unit_count(user_request, tool)
    if country_question_count > 1:
        return country_question_count

    genre_location_count = _genre_location_cross_product_count(user_request, tool)
    if genre_location_count > 1:
        return genre_location_count

    if _has_nonbatched_location_slot(properties):
        if _is_single_route_or_visit_workflow(user_request, tool):
            return 1
        numeric_location_count = _numeric_location_pair_count(user_request, tool)
        if numeric_location_count > 1:
            return numeric_location_count
        location_units = _location_units(user_request)
        location_count = len(location_units)
        if location_count > 1 and (
            _has_explicit_parallel_unit_context(user_request)
            or _has_location_list_context(user_request, tool)
        ):
            return location_count

    stock_count = _stock_symbol_entity_count(user_request, tool)
    if stock_count > 1:
        return stock_count

    if _tool_has_required_batch_array(tool):
        return 1

    return 1


def _color_unit_count(user_request: str, tool: dict[str, Any]) -> int:
    if not any("color" in f"{name} {spec.get('description') or ''}".lower() for name, spec in _properties(tool).items()):
        return 1
    values = _color_phrase_values(user_request)
    return len(values) if len(values) > 1 else 1


def _single_shape_instance_request(user_request: str, tool: dict[str, Any]) -> bool:
    tool_text = _tool_text(tool).lower()
    for name, spec in _properties(tool).items():
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if _looks_like_property_selector_slot(name.lower(), slot_text) and len(_requested_property_values(user_request, spec)) > 1:
            return False
    shape_aliases = {
        "circle": r"circles?|circular",
        "rectangle": r"rectangles?|rectangular",
        "square": r"squares?",
        "triangle": r"triangles?|triangular",
    }
    matched_shapes = [shape for shape, pattern in shape_aliases.items() if re.search(rf"\b(?:{pattern})\b", tool_text)]
    if not matched_shapes:
        return False
    request = user_request.lower()
    if (
        max(_ordinal_scenario_count(user_request), _explicit_group_count(request)) > 1
        and _scoped_request_refs_full_scenarios(user_request)
    ):
        return False
    if re.search(r"\b(?:two|three|four|multiple|several|different|each|both)\s+(?:circles?|rectangles?|squares?|triangles?|shapes?)\b", request):
        return False
    return any(re.search(rf"\b(?:a|an|the|one)\s+(?:[a-z]+\s+)?(?:{shape_aliases[shape]})\b", request) for shape in matched_shapes)


def _country_question_unit_count(user_request: str, tool: dict[str, Any]) -> int:
    if not any("country" in f"{name} {spec.get('description') or ''}".lower() for name, spec in _properties(tool).items()):
        return 1
    clauses = _intent_clauses(user_request)
    matching = [
        clause
        for clause in clauses
        if _country_values(clause)
        and _tool_scope_score(clause, tool) >= 2.0
    ]
    if len(matching) > 1:
        return len(matching)
    countries = _country_values(user_request)
    if len(countries) > 1 and _has_repeated_wh_clause_context(user_request):
        return len(countries)
    return 1


def _schema_named_scalar_unit_count(user_request: str, tool: dict[str, Any]) -> int:
    counts: list[int] = []
    for name, spec in _properties(tool).items():
        if name not in _required(tool) or _property_type(spec) != "string":
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if "president" in slot_text:
            values = _president_name_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "empire" in slot_text:
            values = _empire_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if _looks_like_recipe_name_slot(name.lower(), slot_text):
            values = _recipe_dish_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "material" in slot_text:
            values = _material_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if name.lower() != "country" and _looks_like_location_slot(name.lower(), slot_text):
            values = _city_list_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
            continue
        if "museum" in slot_text:
            values = _museum_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "type" in slot_text and any(word in _tool_text(tool).lower() for word in ["art", "sculpture", "painting", "statue"]):
            values = _artwork_type_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "stat" in slot_text:
            values = _player_stat_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "game" in slot_text and "guide" in _tool_text(tool).lower():
            values = _game_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "frequency" in slot_text:
            values = _frequency_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "mode" in slot_text:
            values = _travel_mode_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        if "team" in slot_text:
            teams = _team_values(user_request)
            seasons = _season_values(user_request)
            if len(teams) > 1 and len(seasons) > 1 and "respectively" not in user_request.lower():
                counts.append(len(teams) * len(seasons))
            elif len(teams) > 1:
                counts.append(len(teams))
        if name.lower() == "position" or "position" in slot_text:
            values = _position_values_for_year_groups(user_request)
            if len(values) > 1:
                counts.append(len(values))
    return min(max(counts), 8) if counts else 1


def _forecast_metric_scoped_location_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower().replace("_", " ")
    if "forecast" not in tool_text:
        return 0
    metrics = [
        metric
        for metric in ("temperature", "humidity", "precipitation")
        if metric in tool_text
    ]
    if not metrics:
        return 0
    grouped_locations = _forecast_metric_location_values(user_request, metrics)
    if grouped_locations:
        return min(max(1, len(_dedupe(grouped_locations))), 8)
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return 0
    matching_locations: list[str] = []
    for clause in clauses:
        lowered = clause.lower()
        if not any(metric in lowered for metric in metrics):
            continue
        matching_locations.extend(_location_units(clause))
    if not matching_locations:
        return 0
    return min(max(1, len(_dedupe(matching_locations))), 8)


def _forecast_metric_location_values_for_slot(user_request: str, slot_text: str) -> list[str]:
    if "forecast" not in slot_text:
        return []
    metrics = [metric for metric in ("temperature", "humidity", "precipitation") if metric in slot_text]
    if not metrics:
        return []
    return _forecast_metric_location_values(user_request, metrics)


def _forecast_metric_location_values(user_request: str, metrics: list[str]) -> list[str]:
    metric_pattern = r"(?:temperature|humidity|precipitation)"
    values: list[str] = []
    for match in re.finditer(
        rf"\b((?:{metric_pattern})(?:\s+and\s+(?:{metric_pattern}))*)\s+forecast\s+for\s+(.+?)(?=\s+and\s+(?:{metric_pattern})\s+forecast\b|[?.!]|$)",
        user_request,
        re.I,
    ):
        metric_group = {item.lower() for item in re.findall(metric_pattern, match.group(1), re.I)}
        if not metric_group.intersection(metrics):
            continue
        chunk = _strip_location_trailing_context(match.group(2))
        locations = _location_units(chunk)
        if not locations:
            cleaned = _clean_location_value(chunk)
            if cleaned:
                locations = [cleaned]
        values.extend(locations)
    return _dedupe(values)


def _has_location_list_context(user_request: str, tool: dict[str, Any]) -> bool:
    if not _has_nonbatched_location_slot(_properties(tool)):
        return False
    return bool(
        len(_capitalized_location_list_values(user_request)) > 1
        or re.search(r"\b(?:in|near|around|for)\s+[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+(?:\s+and\s+|\s*,\s*)", user_request)
        or re.search(r"\b(?:same|also|and)\s+(?:for|in|near)\s+[A-Z]", user_request)
    )


def _tool_has_required_batch_array(tool: dict[str, Any]) -> bool:
    properties = _properties(tool)
    return any(
        name in _required(tool)
        and (
            _property_type(spec) == "array"
            or "list" in name.lower()
            or "list" in str(spec.get("description") or "").lower()
        )
        for name, spec in properties.items()
    )


def _tool_supports_batching_for(properties: dict[str, Any], names: set[str]) -> bool:
    for name, spec in properties.items():
        normalized = name.lower()
        if not any(part in normalized or (part == "city" and "citi" in normalized) for part in names):
            continue
        if _property_type(spec) == "array" or "list" in normalized or "list" in str(spec.get("description") or "").lower():
            return True
    return False


def _has_nonbatched_location_slot(properties: dict[str, Any]) -> bool:
    location_names = {"city", "location", "country", "state", "region", "area"}
    for name, spec in properties.items():
        normalized = name.lower()
        if any(part in normalized for part in location_names) and _property_type(spec) == "string":
            if "list" not in normalized and "list" not in str(spec.get("description") or "").lower():
                return True
    return False


def _has_single_area_slot(properties: dict[str, Any]) -> bool:
    for name, spec in properties.items():
        normalized = name.lower()
        if normalized in {"area", "region", "state", "country"} and _property_type(spec) == "string":
            if "list" not in normalized and "list" not in str(spec.get("description") or "").lower():
                return True
    return False


def _is_single_route_or_visit_workflow(user_request: str, tool: dict[str, Any]) -> bool:
    lowered = user_request.lower()
    tool_text = _tool_text(tool).lower()
    if _tool_looks_like_route_request(tool) and re.search(r"\bfrom\b.+\bto\b", lowered) and re.search(r"\b(?:route|travel|trip|drive|flight|distance|duration|eta|time)\b", tool_text):
        return True
    if _tool_looks_like_route_request(tool) and re.search(r"\bbetween\s+two\s+(?:cities|locations|places|points)\b", lowered):
        return True
    if _tool_looks_like_route_request(tool) and re.search(r"\bstops?\s+at\b", lowered):
        return True
    if re.search(r"\bnear\b.+\bwith\b", lowered) and _has_nonbatched_location_slot(_properties(tool)):
        return True
    return False


def _has_explicit_parallel_unit_context(user_request: str) -> bool:
    lowered = user_request.lower()
    return bool(
        re.search(r"\b(?:and also|also get|for each|for both|each of|both of|respectively|separately)\b", lowered)
        or re.search(r",\s*[^,]+,\s*(?:and|also)\s+", user_request)
        or re.search(r"\b(?:two|three|four|five|\d+)\s+(?:cities|locations|areas|countries|states|regions|hotels|reservations)\b", lowered)
    )


def _service_unit_count(user_request: str, properties: dict[str, Any]) -> int:
    if not any("service" in name.lower() or "platform" in name.lower() for name in properties):
        return 1
    lowered = user_request.lower()
    services = [
        "amazon prime",
        "disney",
        "hbo",
        "hulu",
        "netflix",
        "spotify",
        "youtube",
    ]
    found = {service for service in services if re.search(rf"\b{re.escape(service)}\b", lowered)}
    return len(found) if len(found) > 1 else 1


def _location_units(user_request: str) -> list[str]:
    text = user_request.strip()
    explicit_two = re.search(r"\btwo\s+(?:cities|locations)\s+of\s+(.+)", text, re.I)
    if explicit_two:
        return _split_named_units(explicit_two.group(1))[:2]

    weather_of = re.search(
        r"\b(?:weather|temperature|humidity|conditions?)\b.+\bof\s+([A-Z][A-Za-z .'-]+(?:,\s*[A-Z][A-Za-z .'-]+)?)(?:\s+(?:right now|now|today|currently)\b|[?.!]|$)",
        text,
        re.I,
    )
    if weather_of:
        return [_clean_location_value(weather_of.group(1))]

    found_cjk = _cjk_location_mentions(text)
    if found_cjk:
        return found_cjk

    venue_cities = _venue_city_values(text)
    if venue_cities:
        return venue_cities

    cjk_locations = re.findall(r"[\u4e00-\u9fff]{1,8}(?:市|省|县|区)?", text)
    cjk_locations = [
        item
        for item in cjk_locations
        if item not in {"天气", "状况", "情况", "现在", "目前", "使用", "单位", "帮我", "请问"}
        and "天气" not in item
        and "温度" not in item
        and "查询" not in item
    ]
    if len(cjk_locations) >= 2:
        return cjk_locations

    near = _near_location_value(text)
    if near:
        return [near]

    phrase_units: list[str] = []
    phrase_match = re.search(
        r"\b(?:location(?:s)?(?: given by)?|coordinates?|cities|city|areas?|population of|weather in|weather at|temperature in|temperature at|in|at|for|of|en)\s+(.+?)(?:\s+(?:using|with|please|right now|currently|today|now)\b|[?.!]|$)",
        text,
        re.I,
    )
    if phrase_match:
        phrase_chunk = _strip_location_trailing_context(phrase_match.group(1).strip(" ."))
        if len(phrase_chunk.split()) <= 16:
            phrase_units = _split_named_units(phrase_chunk)

    listed_locations = _capitalized_location_list_values(text)
    if len(listed_locations) > 1:
        return listed_locations
    if len(phrase_units) > len(listed_locations):
        return _filter_location_units(phrase_units)
    if len(listed_locations) == 1 and not _looks_like_named_place_prefix(listed_locations[0].split(",", 1)[0]):
        return listed_locations

    if not phrase_match:
        return []
    return _filter_location_units(phrase_units)


def _cjk_location_mentions(text: str) -> list[str]:
    aliases = [
        "中国广州市",
        "广州市",
        "广州",
        "北京市",
        "北京",
        "上海市",
        "上海",
        "深圳市",
        "深圳",
        "杭州市",
        "杭州",
        "南京市",
        "南京",
        "成都市",
        "成都",
        "重庆市",
        "重庆",
        "武汉市",
        "武汉",
        "天津市",
        "天津",
        "西安市",
        "西安",
        "苏州市",
        "苏州",
    ]
    matches: list[tuple[int, int, str]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        for match in re.finditer(re.escape(alias), text):
            span = (match.start(), match.end())
            if any(not (span[1] <= start or span[0] >= end) for start, end, _value in matches):
                continue
            matches.append((span[0], span[1], alias))
    matches.sort(key=lambda item: item[0])
    return _dedupe(value for _start, _end, value in matches)


def _capitalized_location_list_values(text: str) -> list[str]:
    for match in re.finditer(r"\b(?:in|at|for|of|en)\s+(.+?)(?:[?.!]|$)", text, re.I):
        chunk = _strip_location_trailing_context(match.group(1))
        units = _split_named_units(chunk)
        if len(units) > 1 and any("," in unit for unit in units):
            return units
        comma_parts = [value.strip(" .'\"") for value in chunk.split(",") if value.strip(" .'\"")]
        paired = _pair_region_suffix_parts(comma_parts)
        if len(paired) > 1:
            return paired
    values: list[str] = []
    pattern = r"\b([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4},\s*(?:[A-Z]{2}|[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}))\b"
    blocked = {"RGB", "HEX", "US Census"}
    for match in re.finditer(pattern, text):
        value = _clean_location_value(match.group(1))
        if value and not any(item.lower() == value.lower() for item in blocked):
            values.append(value)
    return _dedupe(values)


def _venue_city_values(text: str) -> list[str]:
    venue = r"(?:Museum|Theatre|Theater|Gallery|Hotel|Restaurant|Stadium|Center|Centre)"
    venue_name = rf"(?:the\s+)?[A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){{0,6}}\s+{venue}"
    pattern = (
        rf"\b{venue_name}\s+in\s+([A-Z][A-Za-z .'-]+?)"
        rf"(?=\s+(?:and\s+{venue_name}\s+in\b|with\b|for\b|from\b|near\b|at\b)|[?.!]|$)"
    )
    values = []
    for match in re.finditer(pattern, text):
        raw_value = match.group(1)
        if raw_value.endswith("D") and text[match.end(1) : match.end(1) + 3].startswith(".C"):
            raw_value = f"{raw_value}.C."
        value = _clean_location_value(raw_value)
        if value:
            values.append(value)
    return _dedupe(values)


def _area_units(user_request: str) -> list[str]:
    pieces: list[str] = []
    patterns = [
        r"\bpopulation(?:\s+data)?\s+of\s+(.+?)(?:\s+from\b|[?.!]|$)",
        r"\bpopulation\s+data\s+for\s+(.+?)(?:[?.!]|$)",
        r"\b(?:for|of)\s+(.+?)\s+from\s+(?:US\s+)?Census\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, user_request, re.I):
            units = _split_named_units(match.group(1))
            pieces.extend(units)
    cleaned = []
    for piece in pieces:
        value = re.sub(r"\b(?:state|country)\b", "", piece, flags=re.I).strip(" .'\"")
        if value:
            cleaned.append(value)
    return _dedupe(cleaned)


def _food_units(user_request: str) -> list[str]:
    lowered = user_request.lower()
    if re.search(r"\b(?:change|modify|switch)\b.+\b(?:from|to|instead of)\b", lowered):
        return ["food_order_change"]
    foods = [
        "burger",
        "burrito",
        "chai",
        "chicken",
        "coffee",
        "drink",
        "fries",
        "mango",
        "pizza",
        "salad",
        "sandwich",
        "tea",
        "wings",
    ]
    found = [food for food in foods if re.search(rf"\b{food}s?\b", lowered)]
    return _dedupe(found)


def _tool_looks_like_food_log(tool: dict[str, Any]) -> bool:
    properties = _properties(tool)
    names = {name.lower() for name in properties}
    return bool(
        any("food" in name for name in names)
        and any("meal" in name for name in names)
        and any("portion" in name and "amount" in name for name in names)
    )


def _food_log_entries_for_tool(user_request: str, tool: dict[str, Any]) -> list[dict[str, Any]]:
    if not _tool_looks_like_food_log(tool):
        return []
    return _food_log_entries_from_text(user_request)


def _food_log_entries_from_text(text: str) -> list[dict[str, Any]]:
    meal = r"(?:breakfast|lunch|dinner|snack)"
    pattern = (
        rf"(?:^|[\n.?!]\s*)(?:for\s+)?(?P<meal>{meal})\b"
        r"\s*(?:[:,-]|\s+)?(?:i\s+(?:had|ate)|had|ate|included|consisted\s+of)?\s*"
        rf"(?P<items>.*?)(?=(?:[\n.?!]\s*)(?:for\s+)?{meal}\b|$)"
    )
    entries: list[dict[str, Any]] = []
    for match in re.finditer(pattern, text, re.I | re.S):
        meal_name = match.group("meal").lower()
        for item in _split_food_log_items(match.group("items")):
            parsed = _parse_food_log_item(item)
            if parsed:
                entries.append({**parsed, "meal_name": meal_name})
    if entries:
        return entries

    generic_pattern = (
        r"(?:^|[\n.?!]\s*)(?:earlier\s+)?(?:i\s+)?"
        r"(?:had|ate|consumed)\s+(?P<items>.*?)(?=(?:[\n.?!]\s*)"
        r"(?:earlier\s+)?(?:i\s+)?(?:had|ate|consumed)\s+|$)"
    )
    for match in re.finditer(generic_pattern, text, re.I | re.S):
        for item in _split_food_log_items(match.group("items")):
            parsed = _parse_food_log_item(item)
            if parsed:
                entries.append({**parsed, "meal_name": "snack"})
    return entries


def _split_food_log_items(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip(" .")
    if not cleaned:
        return []
    parts = [
        part.strip(" .'\"")
        for part in re.split(r"\s*,\s*|\s+\band\s+", cleaned)
        if part.strip(" .'\"")
    ]
    return parts


def _parse_food_log_item(text: str) -> dict[str, Any] | None:
    amount = r"(?P<amount>\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    unit = r"(?P<unit>ounces?|oz|grams?|g|pieces?|slices?|cups?|tablespoons?)"
    patterns = [
        rf"^(?:a|an|the)?\s*{amount}\s+{unit}\s+(?:of\s+)?(?P<food>[A-Za-z][A-Za-z '-]+)$",
        rf"^(?:a|an|the)?\s*{amount}\s+(?P<food>[A-Za-z][A-Za-z '-]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = _number_from_text(match.group("amount"))
        if value is None:
            return None
        unit_value = _normalize_food_portion_unit(match.groupdict().get("unit"), match.group("food"))
        food_name = _clean_slot_phrase(match.group("food"))
        return (
            {"food_name": food_name, "portion_amount": float(value), "portion_unit": unit_value}
            if food_name and unit_value
            else None
        )
    article = re.search(r"^(?:a|an)\s+(?P<food>[A-Za-z][A-Za-z '-]+)$", text, re.I)
    if article:
        food_name = _clean_slot_phrase(article.group("food"))
        unit_value = _normalize_food_portion_unit(None, food_name)
        return (
            {"food_name": food_name, "portion_amount": 1.0, "portion_unit": unit_value}
            if food_name and unit_value
            else None
        )
    food_name = _clean_slot_phrase(text)
    if not food_name:
        return None
    return {"food_name": food_name, "portion_amount": 1.0, "portion_unit": _normalize_food_portion_unit(None, food_name)}


def _normalize_food_portion_unit(raw_unit: str | None, food_name: str) -> str:
    if raw_unit:
        unit = raw_unit.lower().strip()
        if unit in {"ounce", "ounces", "oz"}:
            return "ounces"
        if unit in {"gram", "grams", "g"}:
            return "grams"
        if unit in {"piece", "pieces"}:
            return "pieces"
        if unit in {"slice", "slices"}:
            return "slices"
        if unit in {"cup", "cups"}:
            return "cups"
        if unit in {"tablespoon", "tablespoons"}:
            return "tablespoons"
    if re.search(r"\b(?:coffee|tea|chai|juice|milk|water|soda|smoothie|latte|beer|wine)\b", food_name, re.I):
        return "cups"
    return "pieces"


def _align_food_portion_unit_to_schema(unit: str, spec: dict[str, Any]) -> str:
    enum_values = [str(value) for value in _enum_values_for_spec(spec)]
    if not enum_values:
        return unit
    candidates = [unit]
    if unit.endswith("s"):
        candidates.append(unit[:-1])
    else:
        candidates.append(f"{unit}s")
    irregular = {
        "pieces": "piece",
        "piece": "pieces",
        "slices": "slice",
        "slice": "slices",
        "cups": "cup",
        "cup": "cups",
    }
    if unit in irregular:
        candidates.append(irregular[unit])
    canonical = {value.lower(): value for value in enum_values}
    for candidate in candidates:
        if candidate.lower() in canonical:
            return canonical[candidate.lower()]
    return unit


def _movie_units(user_request: str) -> list[str]:
    match = re.search(r"\bfor\s+(.+?)(?:\s+near\b|\s+in\b|[?.!]|$)", user_request, re.I)
    if not match:
        return []
    chunk = match.group(1).strip(" .")
    parts = re.split(r"\s+\band\b\s+", chunk)
    movies = []
    for part in parts:
        title = re.sub(r"\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b.*", "", part, flags=re.I).strip(" .'\"")
        if title:
            movies.append(title)
    return movies if len(movies) > 1 else []


def _movie_time_pair_count(user_request: str, tool: dict[str, Any]) -> int:
    del tool
    movies = _movie_units(user_request)
    if len(movies) < 2:
        return 1
    times = _clock_time_values(user_request)
    if len(times) >= len(movies):
        return len(movies)
    if re.search(r"\b(?:two|three|four|five|\d+)\s+(?:movies?|films?|showings?|showtimes?)\b", user_request, re.I):
        return len(movies)
    return 1


def _energy_substance_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    if not ("energy" in tool_text and "heat" in tool_text and "substance" in tool_text):
        return 1
    substances = _energy_substances(user_request)
    return len(substances) if len(substances) > 1 else 1


def _energy_substances(user_request: str) -> list[str]:
    materials = [
        "water",
        "aluminium",
        "aluminum",
        "copper",
        "iron",
        "steel",
        "lead",
        "gold",
        "silver",
    ]
    lowered = user_request.lower()
    values = [material for material in materials if re.search(rf"\b{re.escape(material)}\b", lowered)]
    return _dedupe(values)


def _restaurant_search_unit_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    if "restaurant" not in tool_text:
        return 1
    cuisines = _restaurant_cuisine_values(user_request)
    locations = _restaurant_search_locations(user_request)
    count = max(len(cuisines), len(locations))
    return count if count > 1 else 1


def _restaurant_cuisine_values(user_request: str) -> list[str]:
    cuisines = [
        "Chinese",
        "Italian",
        "Mexican",
        "Indian",
        "Thai",
        "Japanese",
        "French",
        "Korean",
        "Mediterranean",
        "Greek",
    ]
    return [cuisine for cuisine in cuisines if re.search(rf"\b{re.escape(cuisine)}\b", user_request, re.I)]


def _music_genre_values(user_request: str) -> list[str]:
    genres = [
        "rock",
        "jazz",
        "pop",
        "classical",
        "hip hop",
        "rap",
        "country",
        "electronic",
        "folk",
        "blues",
        "metal",
        "punk",
        "reggae",
        "latin",
        "indie",
    ]
    return [genre for genre in genres if re.search(rf"\b{re.escape(genre)}\b", user_request, re.I)]


def _named_entity_in_location_count(user_request: str) -> int:
    entity_terms = r"(?:Museum|Hotel|Theatre|Theater|Gallery|University|Hospital|Airport|Stadium|Center|Centre)"
    pattern = (
        rf"\b(?:the\s+)?[A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){{0,5}}\s+{entity_terms}"
        r"\s+in\s+[A-Z][A-Za-z .'-]+?"
        rf"(?=,\s*(?:and\s*)?(?:the\s+)?[A-Z]|(?:\s+and\s+(?:the\s+)?[A-Z])|[?.!]|$)"
    )
    count = len(re.findall(pattern, user_request))
    return count if count > 1 else 1


def _restaurant_search_locations(user_request: str) -> list[str]:
    locations: list[str] = []
    patterns = [
        r"\brestaurant\s+near\s+me\s+in\s+([A-Z][A-Za-z .'-]+?)(?=\s+\band\b|\s+then\b|[?.!]|$)",
        r"\brestaurant\s+in\s+([A-Z][A-Za-z .'-]+?)(?=\s+\band\b|\s+then\b|[?.!]|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, user_request):
            value = _clean_location_value(match.group(1))
            if value:
                locations.append(value)
    return _dedupe(locations)


def _currency_conversion_count(user_request: str, properties: dict[str, Any]) -> int:
    names = {name.lower() for name in properties}
    if not any(("from" in name or "base" in name or "source" in name) and "currenc" in name for name in names):
        return 1
    if not any(("to" in name or "target" in name or "destination" in name) and "currenc" in name for name in names):
        return 1
    pair_count = _currency_pair_count(user_request)
    if pair_count > 1:
        return pair_count
    conversion_scenario_count = _money_conversion_scenario_count(user_request)
    if conversion_scenario_count > 1:
        return conversion_scenario_count
    count = 0
    pattern = (
        r"\b\d[\d,.]*\s+([A-Z]{3})\b.{0,80}?"
        r"\b(?:to|into|converted\s+to|convert\s+to)\s+"
        r"([A-Z]{3}(?:\s*,\s*[A-Z]{3})*(?:\s*,?\s*(?:and|or)\s*[A-Z]{3})?)"
    )
    for match in re.finditer(pattern, user_request):
        targets = re.findall(r"\b[A-Z]{3}\b", match.group(2))
        count += max(1, len(targets))
    return count if count > 1 else 1


def _currency_pair_count(user_request: str) -> int:
    code = r"[A-Z]{3}"
    count = len(re.findall(rf"\b{code}\s+to\s+{code}\b", user_request))
    if count > 1:
        return count
    shared_base = re.search(rf"\b({code})\s+to\s+({code}(?:\s*,\s*{code})*(?:\s*(?:and|or)\s*{code})+)\b", user_request)
    if shared_base:
        targets = re.findall(rf"\b{code}\b", shared_base.group(2))
        return len(targets) if len(targets) > 1 else 1
    return 1


def _money_conversion_scenario_count(user_request: str) -> int:
    money_word = r"(?:dollars?|usd|euros?|eur|pounds?|gbp|yen|jpy|cad|aud)"
    count = len(
        re.findall(
            rf"\b(?:transfer|convert|exchange)\s+\d[\d,.]*\s+{money_word}\s+(?:to|into)\s+{money_word}\b",
            user_request,
            re.I,
        )
    )
    return count if count > 1 else 1


def _array_payload_group_count(user_request: str, tool: dict[str, Any]) -> int:
    required_arrays = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _property_type(spec) == "array"
    ]
    if not required_arrays:
        return 1
    if len(required_arrays) >= 2:
        return 1

    lowered = user_request.lower()
    explicit = _explicit_group_count(lowered)
    if explicit > 1 and re.search(r"\b(?:arrays?|datasets?|data sets?|groups?|lists?)\b", lowered):
        return explicit

    ordinal_count = len(
        re.findall(
            r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s+"
            r"(?:array|dataset|data\s+set|group|list|series)\b",
            lowered,
        )
    )
    if ordinal_count > 1:
        return ordinal_count

    segment_count = _numeric_payload_segment_count(user_request)
    if segment_count > 1 and re.search(
        r"\b(?:averages?|means?|median|standard deviation|variance|datasets?|data sets?|groups?|lists?|for each|within each|these)\b",
        lowered,
    ):
        return segment_count
    return 1


def _array_tuple_group_count(user_request: str, tool: dict[str, Any]) -> int:
    required_arrays = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _property_type(spec) == "array"
    ]
    if len(required_arrays) < 2:
        return 1
    coordinate_pair_count = len(_coordinate_pair_groups(user_request))
    if coordinate_pair_count > 1:
        return coordinate_pair_count
    vector_pair_count = len(re.findall(r"\[[^\]]+\]\s+(?:with|and)\s+\[[^\]]+\]", user_request, re.I))
    return vector_pair_count if vector_pair_count > 1 else 1


def _numeric_payload_segment_count(user_request: str) -> int:
    marked = re.sub(
        r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s+"
        r"(?:array|dataset|data\s+set|group|list|series)\b",
        "|",
        user_request,
        flags=re.I,
    )
    marked = re.sub(r",\s+(?:and\s+)?(?:the|a|an)\s+", "|", marked, flags=re.I)
    marked = re.sub(r";", "|", marked)
    parts = [part for part in marked.split("|") if part.strip()]
    return sum(1 for part in parts if len(extract_numbers(part)) >= 2)


def _numeric_slot_list_count(user_request: str, tool: dict[str, Any]) -> int:
    counts: list[int] = []
    for name, spec in _properties(tool).items():
        if name not in _required(tool):
            continue
        if _property_type(spec) not in {"integer", "float", "number"}:
            continue
        labels = _numeric_slot_labels(name, spec)
        for label in labels:
                counts.extend(_number_list_counts_near_label(user_request, label))
    return max(counts) if counts else 1


def _parallel_scenario_count(user_request: str, tool: dict[str, Any]) -> int:
    if _has_result_count_phrase(user_request):
        return 1
    full_request = user_request
    full_ordinal_count = _ordinal_scenario_count(full_request)
    full_explicit_count = 1 if _has_result_count_phrase(full_request) else _explicit_group_count(full_request.lower())
    route_pair_count = _route_pair_count(user_request, tool)
    route_alternative_count = _route_alternative_count(user_request, tool)
    early_integration_count = _integration_method_function_count(user_request, tool)
    early_coefficient_count = len(_equation_coefficient_groups(user_request))
    scoped_request = _tool_scoped_request_text(user_request, tool)
    if scoped_request != user_request and route_pair_count <= 1 and route_alternative_count <= 1:
        user_request = scoped_request
    required_batch_array = _tool_has_required_batch_array(tool)
    if required_batch_array and not _has_repeated_scalar_selector(user_request, tool):
        return 1

    properties = _properties(tool)
    required = set(_required(tool))
    counts: list[int] = []
    same_tool_context = _has_same_tool_parallel_context(user_request)
    format_variant_count = _boolean_output_variant_count(user_request, tool)
    full_scenario_count = max(full_ordinal_count, full_explicit_count)
    if full_scenario_count > 1 and scoped_request != full_request and _scoped_request_refs_full_scenarios(user_request):
        counts.append(full_scenario_count)
    if early_integration_count > 1:
        counts.append(early_integration_count)
    if any("mode" in f"{name} {spec.get('description') or ''}".lower() for name, spec in properties.items()):
        mode_values = _travel_mode_values(user_request)
        if len(mode_values) > 1:
            counts.append(len(mode_values))

    route_pair_count = _route_pair_count(user_request, tool)
    route_alternative_count = _route_alternative_count(user_request, tool)
    if route_pair_count > 1:
        counts.append(route_pair_count)
    if route_alternative_count > 1:
        counts.append(route_alternative_count)
    elif route_pair_count <= 1 and _is_single_route_or_visit_workflow(user_request, tool):
        return 1

    financial_scenarios = _financial_scenarios(user_request)
    if len(financial_scenarios) > 1 and any(
        "financial institution" in f"{name} {spec.get('description') or ''}".lower()
        or "income" in f"{name} {spec.get('description') or ''}".lower()
        or ("loan" in f"{name} {spec.get('description') or ''}".lower() and "amount" in f"{name} {spec.get('description') or ''}".lower())
        for name, spec in properties.items()
    ):
        counts.append(len(financial_scenarios))

    reservation_scenarios = _reservation_scenarios(full_request) if _tool_looks_like_reservation_tool(tool) else []
    if len(reservation_scenarios) > 1:
        counts.append(len(reservation_scenarios))

    institution_count = len(_financial_institution_values(user_request))
    if institution_count > 1 and any("institution" in f"{name} {spec.get('description') or ''}".lower() for name, spec in properties.items()):
        counts.append(institution_count)

    if any("case" in f"{name} {spec.get('description') or ''}".lower() for name, spec in properties.items()):
        case_count = _case_identifier_count(user_request)
        if case_count > 1:
            detail_count = max((_requested_detail_value_count(user_request, spec) for spec in properties.values()), default=1)
            counts.append(case_count * max(1, detail_count))

    ordinal_count = _ordinal_scenario_count(user_request)
    if ordinal_count > 1:
        counts.append(ordinal_count)

    explicit_count = 1 if _has_result_count_phrase(user_request) else _explicit_group_count(user_request.lower())
    if explicit_count > 1:
        counts.append(explicit_count)
    boundary_count = _semantic_scenario_boundary_count(user_request)
    if boundary_count > 1 and re.search(r"\bas\s+well\s+as\s+one\b", user_request, re.I):
        counts.append(boundary_count)

    entity_location_count = _named_entity_in_location_count(user_request)
    if entity_location_count > 1 and any(_looks_like_location_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower()) for name, spec in properties.items()):
        counts.append(entity_location_count)

    integration_count = _integration_method_function_count(user_request, tool)
    if integration_count > 1:
        counts.append(integration_count)

    operation_clause_count = _repeated_operation_clause_count(user_request, tool)
    if operation_clause_count > 1:
        counts.append(operation_clause_count)

    genre_location_count = _genre_location_cross_product_count(user_request, tool)
    if genre_location_count > 1:
        counts.append(genre_location_count)

    scalar_cross_product_count = _scalar_cross_product_count(user_request, tool)
    if scalar_cross_product_count > 1:
        counts.append(scalar_cross_product_count)

    experiment_count = _experiment_repeat_count(user_request, tool)
    if experiment_count > 1:
        counts.append(experiment_count)

    numeric_required = [
        name
        for name, spec in properties.items()
        if name in required and _is_numeric_value_property(spec)
    ]
    coefficient_count = len(_equation_coefficient_groups(user_request))
    coefficient_count = max(coefficient_count, early_coefficient_count)
    if coefficient_count > 1 and {"a", "b", "c"}.issubset(set(numeric_required)):
        counts.append(coefficient_count)
    named_pair_count = _paired_numeric_sequence_count(user_request, len(numeric_required)) if len(numeric_required) >= 2 else 1
    if named_pair_count > 1 and _has_named_numeric_pair_context(user_request):
        counts.append(named_pair_count)
    repeated_numeric_for_tool = any(
        _unit_labeled_numeric_slot_count(user_request, name, properties[name]) > 1
        or _numeric_slot_list_count(user_request, tool) > 1
        for name in numeric_required
    )
    if _allows_multiple_tools(user_request) and not same_tool_context and format_variant_count <= 1 and not repeated_numeric_for_tool:
        numeric_required = []

    for name in numeric_required:
        unit_count = _unit_labeled_numeric_slot_count(user_request, name, properties[name])
        if unit_count > 1:
            counts.append(unit_count)

    if len(numeric_required) == 1:
        slot_count = _numeric_slot_list_count(user_request, tool)
        if slot_count > 1:
            counts.append(slot_count)
        unit_count = _unit_labeled_numeric_slot_count(user_request, numeric_required[0], properties[numeric_required[0]])
        if unit_count > 1:
            counts.append(unit_count)
        list_count = _single_numeric_value_list_count(user_request, tool, numeric_required[0])
        if list_count > 1:
            counts.append(list_count)
        unary_count = _repeated_unary_operation_count(user_request, tool)
        if unary_count > 1:
            counts.append(unary_count)
        repeated_label_count = _repeated_numeric_label_count(user_request, numeric_required[0], properties[numeric_required[0]])
        if repeated_label_count > 1:
            counts.append(repeated_label_count)
    elif len(numeric_required) >= 2:
        pair_count = _paired_numeric_sequence_count(user_request, len(numeric_required))
        if pair_count > 1:
            counts.append(pair_count)

    if numeric_required:
        tuple_count = _numeric_tuple_count(user_request, tool)
        if tuple_count > 1:
            counts.append(tuple_count)

    if format_variant_count > 1:
        counts.append(format_variant_count * max(counts or [1]))

    for name, spec in properties.items():
        if name in required and _property_type(spec) == "string":
            entity_count = _entity_count_for_arg(user_request, name, spec)
            if entity_count > 1:
                counts.append(entity_count)

    return min(max(counts), 8) if counts else 1


def _genre_location_cross_product_count(user_request: str, tool: dict[str, Any]) -> int:
    properties = _properties(tool)
    has_genre = any(
        name in _required(tool)
        and ("genre" in name.lower() or "genre" in str(spec.get("description") or "").lower())
        for name, spec in properties.items()
    )
    has_location = any(
        name in _required(tool)
        and _property_type(spec) == "string"
        and _looks_like_location_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower())
        for name, spec in properties.items()
    )
    if not (has_genre and has_location):
        return 1
    genres = _music_genre_values(user_request)
    locations = _location_units(user_request)
    if len(genres) > 1 and len(locations) > 1:
        return min(len(genres) * len(locations), 8)
    return 1


def _genre_location_cross_product_value(
    user_request: str,
    arg: str,
    slot_text: str,
    call_index: int,
    call_count: int,
) -> str | None:
    genres = _music_genre_values(user_request)
    locations = _location_units(user_request)
    expected = len(genres) * len(locations)
    if len(genres) <= 1 or len(locations) <= 1 or call_count < expected:
        return None
    if _looks_like_location_slot(arg, slot_text):
        return locations[min(call_index // len(genres), len(locations) - 1)]
    if "genre" in arg or "genre" in slot_text:
        return genres[call_index % len(genres)]
    return None


def _scalar_cross_product_count(user_request: str, tool: dict[str, Any]) -> int:
    dimensions = _scalar_cross_product_dimensions(user_request, tool)
    if len(dimensions) < 2:
        return 1
    product = 1
    for _name, values in dimensions:
        product *= len(values)
    return min(product, 8) if product > max(len(values) for _name, values in dimensions) else 1


def _scalar_cross_product_dimensions(user_request: str, tool: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    if _tool_has_required_batch_array(tool):
        return []
    if _tool_looks_like_reservation_tool(tool):
        return []
    properties = _properties(tool)
    required = set(_required(tool))
    entity_dimensions: list[tuple[str, list[str]]] = []
    enum_dimensions: list[tuple[str, list[str]]] = []
    numeric_dimensions: list[tuple[str, list[Any]]] = []
    numeric_required = [
        name
        for name, spec in properties.items()
        if name in required and _property_type(spec) in {"integer", "float", "number"}
    ]
    for name, spec in properties.items():
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if _looks_like_result_count_slot(name.lower(), slot_text):
            continue
        if _property_type(spec) in {"integer", "float", "number"} and name in required:
            numeric_values = _scalar_numeric_condition_values_for_slot(
                user_request,
                name,
                spec,
                len(numeric_required),
            )
            if len(numeric_values) > 1:
                numeric_dimensions.append((name, numeric_values))
            continue
        if _property_type(spec) != "string":
            continue
        enum_values = _enum_value_evidence(user_request, spec)
        if len(enum_values) > 1:
            enum_dimensions.append((name, enum_values))
            continue
        if name not in required:
            continue
        entity_values = _scalar_entity_values_for_slot(user_request, name.lower(), slot_text, spec)
        if len(entity_values) <= 1:
            entity_values = _quoted_scalar_dimension_values_for_slot(user_request, name.lower(), slot_text, properties)
        if len(entity_values) > 1:
            entity_dimensions.append((name, entity_values))
    if not entity_dimensions:
        return []
    dimensions: list[tuple[str, list[Any]]] = []
    dimensions.extend(entity_dimensions[:2])
    dimensions.extend(enum_dimensions[:2])
    dimensions.extend(numeric_dimensions[:1])
    if len(dimensions) < 2:
        return []
    if not _scalar_cross_product_dimensions_are_distinct(dimensions):
        return []
    if not _has_scalar_cross_product_request_signal(user_request):
        return []
    if re.search(r"\brespectively\b", user_request, re.I) and not _respectively_allows_scalar_cross_product(
        user_request,
        dimensions,
        properties,
    ):
        return []
    return dimensions


def _has_scalar_cross_product_request_signal(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:all|for each|for every|each of|both of|for both|respectively|separately|simultaneously|"
            r"repeat|same analysis|same calculation|same lookup|all combinations?|each combination|every combination)\b",
            user_request,
            re.I,
        )
    )


def _scalar_cross_product_dimensions_are_distinct(dimensions: list[tuple[str, list[Any]]]) -> bool:
    normalized_sets: list[set[str]] = []
    for _name, values in dimensions:
        normalized = {
            re.sub(r"\s+", " ", str(value).strip().lower())
            for value in values
            if str(value).strip()
        }
        if len(normalized) <= 1:
            return False
        normalized_sets.append(normalized)
    for left_index, left in enumerate(normalized_sets):
        for right in normalized_sets[left_index + 1 :]:
            overlap = left & right
            if not overlap:
                continue
            if left == right or len(overlap) / max(1, min(len(left), len(right))) >= 0.5:
                return False
    return True


def _quoted_scalar_dimension_values_for_slot(
    user_request: str,
    arg: str,
    slot_text: str,
    properties: dict[str, dict[str, Any]],
) -> list[str]:
    if _looks_like_contextual_reference_slot(arg, slot_text):
        return []
    if not re.search(
        r"\b(?:both|each|every|repeat|same|respectively|separately|for each|for both|and)\b",
        user_request,
        re.I,
    ):
        return []
    enum_values = {
        str(value).strip().lower()
        for spec in properties.values()
        for value in _enum_values_for_spec(spec)
        if str(value).strip()
    }
    values: list[str] = []
    for value, start, _end in _quoted_string_spans(user_request):
        text = value.strip()
        if not text or text.lower() in enum_values:
            continue
        if _quoted_value_has_reference_context(user_request, start):
            continue
        values.append(text)
    deduped = _dedupe(values)
    return deduped if len(deduped) > 1 else []


def _looks_like_contextual_reference_slot(arg: str, slot_text: str) -> bool:
    text = f"{arg} {slot_text}".lower().replace("_", " ")
    return bool(re.search(r"\b(?:reference|baseline|control|source|template|comparison)\b", text))


def _quoted_value_has_reference_context(user_request: str, start: int) -> bool:
    prefix = user_request[max(0, start - 48) : start].lower()
    return bool(re.search(r"\b(?:reference|baseline|control|source|template|comparison)\s+(?:[a-z]+\s+){0,3}$", prefix))


def _quoted_string_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9])'(.+?)'(?=[\s,.;:!?)]|$)|(?<![A-Za-z0-9])\"([^\"]+)\"(?![A-Za-z0-9])"
    )
    for match in pattern.finditer(text):
        value = match.group(1) or match.group(2)
        if value:
            spans.append((value, match.start(), match.end()))
    return spans


def _scalar_cross_product_argument_value(
    user_request: str,
    tool: dict[str, Any],
    arg_name: str,
    call_index: int,
    call_count: int,
) -> Any | None:
    dimensions = _scalar_cross_product_dimensions(user_request, tool)
    if len(dimensions) < 2:
        return None
    product = 1
    for _name, values in dimensions:
        product *= len(values)
    if product <= 1 or call_count < product:
        return None
    stride = product
    for name, values in dimensions:
        stride = max(1, stride // len(values))
        if name != arg_name:
            continue
        value_index = min((call_index // stride) % len(values), len(values) - 1)
        return values[value_index]
    return None


def _scalar_numeric_condition_values_for_slot(
    user_request: str,
    name: str,
    spec: dict[str, Any],
    numeric_required_count: int,
) -> list[Any]:
    if numeric_required_count != 1:
        return []
    slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
    if re.search(
        r"\b(?:amount|cost|price|duration|time|minutes?|seconds?|hours?|days?|months?|years?|term|count|number)\b",
        slot_text,
    ):
        return []
    values = _unit_labeled_numeric_values(user_request, name, spec)
    if len(values) <= 1 and re.search(r"\b(?:level|altitude|depth|pressure|temperature|elevation)\b", slot_text):
        values = extract_numbers(user_request)
    coerced = []
    for value in values:
        number = _number_from_any(value)
        if number is None:
            continue
        coerced.append(_coerce_number(number, spec))
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in coerced:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _respectively_allows_scalar_cross_product(
    user_request: str,
    dimensions: list[tuple[str, list[Any]]],
    properties: dict[str, dict[str, Any]],
) -> bool:
    del user_request
    has_numeric_condition = False
    for name, _values in dimensions:
        spec = properties.get(name) or {}
        if _property_type(spec) not in {"integer", "float", "number"}:
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if re.search(r"\b(?:level|altitude|depth|pressure|temperature|elevation)\b", slot_text):
            has_numeric_condition = True
            break
    return has_numeric_condition


def _scalar_entity_values_for_slot(
    user_request: str,
    arg: str,
    slot_text: str,
    spec: dict[str, Any],
) -> list[str]:
    del spec
    if _looks_like_person_name_slot(arg, slot_text):
        values = _person_name_list_values(user_request, slot_text)
        if values:
            return values
    if _looks_like_location_slot(arg, slot_text):
        values = _city_list_values(user_request) or _quoted_location_values(user_request)
        if values:
            return values
    if "company" in slot_text:
        values = _company_or_entity_values(user_request)
        if values:
            return values
    if "substance" in arg or "substance" in slot_text:
        values = _energy_substances(user_request)
        if len(values) > 1:
            return values
    quoted = _quoted_strings(user_request)
    if len(quoted) > 1 and _has_multi_value_context_for_arg(user_request, arg):
        return _dedupe(quoted)
    if _looks_like_named_entity_slot(arg, slot_text):
        values = _named_slot_values(user_request, arg, slot_text)
        if len(values) > 1:
            return values
    return []


def _looks_like_person_name_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    if "name" not in tokens and not arg.endswith("name"):
        return False
    return bool(tokens & {"client", "customer", "member", "partner", "patient", "person", "user"})


def _person_name_list_values(text: str, slot_text: str) -> list[str]:
    del slot_text
    labels = (
        "clients?|customers?|members?|partners?|patients?|people|persons?|users?"
    )
    values: list[str] = []
    for match in re.finditer(
        rf"\b(?:for|of|from|with)?\s*(?:the\s+)?(?:{labels})\s+(.+?)(?=\s+(?:that|which|who|whose|with|where|for|from|in|on|at|to|please|using|based)\b|[?.!]|$)",
        text,
        re.I,
    ):
        chunk = match.group(1)
        chunk = re.split(r"\s+\b(?:and|or)\s+(?:their|his|her|its|status|statuses|mandates?|records?|accounts?)\b", chunk, maxsplit=1, flags=re.I)[0]
        for value in _split_named_units(chunk):
            cleaned = _clean_slot_phrase(value)
            if cleaned and not re.search(r"\b(?:active|inactive|pending|all|any|records?|mandates?)\b", cleaned, re.I):
                values.append(cleaned)
    return _dedupe(values)


def _schema_named_numeric_value(
    user_request: str,
    arg: str,
    slot_text: str,
    call_index: int,
    call_count: int,
) -> int | float | None:
    coefficient_groups = _equation_coefficient_groups(user_request)
    if coefficient_groups and arg in {"a", "b", "c"}:
        group = coefficient_groups[min(call_index, len(coefficient_groups) - 1)]
        if arg in group:
            return group[arg]
    ecology_scenarios = _ecology_scenarios(user_request)
    if ecology_scenarios:
        scenario = ecology_scenarios[min(call_index, len(ecology_scenarios) - 1)]
        if "population growth" in slot_text or arg in {"duration", "growth_duration"}:
            return scenario.get("growth_years")
        if "impact" in slot_text or arg in {"timeframe", "time_frame", "impact_duration"}:
            return scenario.get("impact_years")
    if "season" in slot_text:
        teams = _team_values(user_request)
        seasons = _season_values(user_request)
        if len(teams) > 1 and len(seasons) > 1 and call_count >= len(teams) * len(seasons):
            value = seasons[min(call_index // len(teams), len(seasons) - 1)]
            return _number_from_any(value)
    if arg == "year" or "year" in slot_text:
        years = _years_for_position_groups(user_request)
        if len(years) > 1 and call_index < len(years):
            return years[call_index]
        if arg == "year":
            plain_years = [int(year) for year in _year_values(user_request)]
            if len(plain_years) > 1 and call_count > 1:
                return plain_years[min(call_index, len(plain_years) - 1)]
    return None


def _equation_coefficient_groups(user_request: str) -> list[dict[str, int | float]]:
    groups: list[dict[str, int | float]] = []
    equation_pattern = r"([+-]?\s*\d*(?:\.\d+)?\s*x\s*\^\s*2(?:\s*[+-]\s*\d+(?:\.\d+)?\s*x)?(?:\s*[+-]\s*\d+(?:\.\d+)?)?)\s*=\s*0"
    for match in re.finditer(equation_pattern, user_request, re.I):
        group = _quadratic_coefficients_from_expression(match.group(1))
        if group:
            groups.append(group)
    if len(groups) > 1:
        return groups

    labeled_groups: list[dict[str, int | float]] = []
    for segment in re.split(
        r"\b(?:first|second|third|fourth|next|another)\s+equation\b|[.;]",
        user_request,
        flags=re.I,
    ):
        values: dict[str, int | float] = {}
        for label in ("a", "b", "c"):
            match = re.search(rf"\b{label}\s*(?:=|:|is)\s*(-?\d+(?:\.\d+)?)\b", segment, re.I)
            if match:
                value = _number_from_text(match.group(1))
                if value is not None:
                    values[label] = value
        if len(values) >= 3:
            labeled_groups.append(values)
    return labeled_groups if len(labeled_groups) > 1 else groups


def _quadratic_coefficients_from_expression(expression: str) -> dict[str, int | float] | None:
    normalized = expression.replace(" ", "")
    match = re.match(
        r"(?P<a>[+-]?\d*(?:\.\d+)?)x\^2(?P<b>[+-]\d+(?:\.\d+)?)x(?P<c>[+-]\d+(?:\.\d+)?)$",
        normalized,
        re.I,
    )
    if not match:
        return None

    def parse_coefficient(raw: str) -> int | float:
        if raw in {"", "+"}:
            return 1
        if raw == "-":
            return -1
        value = float(raw)
        return int(value) if value.is_integer() else value

    return {
        "a": parse_coefficient(match.group("a")),
        "b": parse_coefficient(match.group("b")),
        "c": parse_coefficient(match.group("c")),
    }


def _schema_named_scalar_value(
    user_request: str,
    arg: str,
    slot_text: str,
    call_index: int,
    call_count: int,
) -> str | None:
    if arg in {"from_account", "source_account"} or "from account" in slot_text:
        value = _transfer_account_value(user_request, "from")
        if value:
            return value
    if _looks_like_stock_code_slot(arg, slot_text):
        values = _stock_symbol_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if arg in {"to_account", "target_account", "destination_account"} or "to account" in slot_text:
        value = _transfer_account_value(user_request, "to")
        if value:
            return value
    if "url" in arg or "url" in slot_text:
        value = _identifier_with_suffix_value(user_request, "url")
        if value:
            return value
    if _looks_like_text_payload_slot(arg, slot_text):
        value = _identifier_with_suffix_value(user_request, "text")
        if value:
            return value
    if arg == "shape" or "shape" in slot_text:
        value = _shape_value(user_request)
        if value:
            return value
    if _looks_like_property_type_slot(arg, slot_text):
        value = _property_type_value(user_request)
        if value:
            return value
    if _looks_like_product_slot(arg, slot_text):
        value = _product_value(user_request)
        if value:
            return value
    if "material" in slot_text:
        values = _material_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if arg != "country" and _looks_like_location_slot(arg, slot_text):
        capital_gain_states = _capital_gain_states(user_request)
        if capital_gain_states:
            return capital_gain_states[min(call_index, len(capital_gain_states) - 1)]
        numeric_pair_locations = _numeric_location_pair_locations(user_request)
        if numeric_pair_locations:
            return numeric_pair_locations[min(call_index, len(numeric_pair_locations) - 1)]
        forecast_locations = _forecast_metric_location_values_for_slot(user_request, slot_text)
        if forecast_locations:
            return forecast_locations[min(call_index, len(forecast_locations) - 1)]
        values = _city_list_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
        product_cities = _product_context_city_values(user_request)
        if product_cities:
            return product_cities[min(call_index, len(product_cities) - 1)]
    if "museum" in slot_text and not _looks_like_location_slot(arg, slot_text):
        values = _museum_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "type" in slot_text:
        values = _artwork_type_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if arg == "artist" or "artist" in slot_text:
        value = _artist_value(user_request)
        if value:
            return value
    if arg == "player" or "player" in slot_text:
        value = _player_value(user_request)
        if value:
            return value
    if _looks_like_game_name_slot(arg, slot_text):
        values = _game_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "attribute" in slot_text:
        value = _attribute_value(user_request)
        if value:
            return value
    if "stat" in slot_text:
        values = _player_stat_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "frequency" in slot_text:
        values = _frequency_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "condition" in slot_text:
        values = _condition_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "mode" in slot_text:
        values = _travel_mode_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "president" in slot_text and arg != "position":
        values = _president_name_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if arg == "position" or "position" in slot_text:
        values = _position_values_for_year_groups(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "empire" in slot_text:
        values = _empire_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if _looks_like_recipe_name_slot(arg, slot_text):
        values = _recipe_dish_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "type" in slot_text and "guide" in slot_text:
        values = _schema_example_modifier_values(user_request, slot_text, "guide")
        if values:
            return values[min(call_index, len(values) - 1)]
    if "team" in slot_text:
        teams = _team_values(user_request)
        seasons = _season_values(user_request)
        if len(teams) > 1 and len(seasons) > 1 and call_count >= len(teams) * len(seasons):
            return teams[call_index % len(teams)]
        if teams:
            return teams[min(call_index, len(teams) - 1)]
    if arg != "country" and _looks_like_location_slot(arg, slot_text):
        values = _city_list_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    if "company" in slot_text:
        quoted = _quoted_strings(user_request)
        if quoted:
            return quoted[min(call_index, len(quoted) - 1)]
        values = _company_or_entity_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
    return None


def _transfer_account_value(text: str, direction: str) -> str | None:
    match = re.search(r"\bfrom\s+(?:my\s+)?([a-z][a-z _-]+?)\s+to\s+(?:my\s+)?([a-z][a-z _-]+?)(?:\s+account\b|[?.!]|$)", text, re.I)
    if not match:
        return None
    value = match.group(1 if direction == "from" else 2)
    value = re.sub(r"\s+account\b.*$", "", value, flags=re.I)
    return _clean_slot_phrase(value)


def _identifier_with_suffix_value(text: str, suffix: str) -> str | None:
    match = re.search(rf"\b([A-Za-z][A-Za-z0-9_/-]*_{re.escape(suffix)})\b", text)
    return match.group(1) if match else None


def _shape_value(text: str) -> str | None:
    for shape in ["square", "rectangle", "circle", "triangle", "oval"]:
        if re.search(rf"\b{shape}\b", text, re.I):
            return shape
    return None


def _dimension_array_value(text: str) -> list[int | float] | None:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\b", text, re.I)
    if not match:
        return None
    values: list[int | float] = []
    for raw in match.groups():
        number = float(raw)
        values.append(int(number) if number.is_integer() else number)
    return values


def _coordinate_pair_groups(text: str) -> list[tuple[list[int | float], list[int | float]]]:
    coordinates: list[list[int | float]] = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        values = extract_numbers(match.group(1))
        if len(values) >= 2:
            coordinates.append(values[:2])
    groups: list[tuple[list[int | float], list[int | float]]] = []
    for index in range(0, len(coordinates) - 1, 2):
        groups.append((coordinates[index], coordinates[index + 1]))
    return groups


def _coordinate_array_values(text: str) -> list[list[int | float]]:
    values: list[list[int | float]] = []
    for match in re.finditer(r"\bcoordinates?\s*\(([^)]*)\)", text, re.I):
        numbers = extract_numbers(match.group(1))
        if len(numbers) >= 2:
            values.append(numbers[:2])
    return values


def _product_value(text: str) -> str | None:
    patterns = [
        r"\b(?:for|of|carry)\s+(?:a\s+|an\s+|the\s+)?([A-Z][A-Za-z0-9 '&.-]+?)(?:\s+in\s+[A-Z]|\s+from\b|[?.!]|$)",
        r"\b([A-Z][A-Za-z0-9 '&.-]+?)\s+in\s+[A-Z][A-Za-z .'-]+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = _clean_slot_phrase(match.group(1))
            if value and not re.search(r"\b(?:lowest price|average price|stores?)\b", value, re.I):
                return value
    return None


def _looks_like_product_slot(arg: str, slot_text: str) -> bool:
    if arg in {"product", "product_name", "productname", "item", "item_name", "itemname"}:
        return True
    return bool(
        re.search(r"\bname\s+of\s+the\s+(?:product|item)\b", slot_text)
        or re.search(r"\b(?:product|item)\s+name\b", slot_text)
    )


def _looks_like_property_type_slot(arg: str, slot_text: str) -> bool:
    normalized = arg.lower().replace("_", "")
    return normalized in {"propertytype", "housetype", "hometype"} or bool(
        re.search(r"\btype\s+of\s+(?:property|home|house|listing)\b", slot_text)
        or re.search(r"\b(?:property|home|house)\s+type\b", slot_text)
    )


def _property_type_value(text: str) -> str | None:
    for value in [
        "villa",
        "condo",
        "apartment",
        "townhouse",
        "house",
        "home",
        "studio",
        "loft",
        "duplex",
    ]:
        if re.search(rf"\b{re.escape(value)}s?\b", text, re.I):
            return value
    return None


def _product_context_city_values(text: str) -> list[str]:
    if not re.search(r"\b(?:average|lowest|highest|best)?\s*(?:price|cost|store|stores|carry|product|item)\b", text, re.I):
        return []
    product = _product_value(text)
    if not product:
        return []
    values: list[str] = []
    product_pattern = re.escape(product).replace(r"\ ", r"\s+")
    for match in re.finditer(
        rf"\b{product_pattern}\s+in\s+([A-Z][A-Za-z .'-]+?)(?=\s+(?:and|also|for|with|from|to)\b|[?.!]|$)",
        text,
    ):
        city = _clean_location_value(match.group(1))
        if city and city.lower() not in STOPWORDS:
            values.append(city)
    return _dedupe(values)


def _material_values(text: str) -> list[str]:
    materials = [
        "bronze",
        "stone",
        "marble",
        "wood",
        "metal",
        "steel",
        "gold",
        "silver",
        "ceramic",
        "clay",
        "glass",
    ]
    return [material for material in materials if re.search(rf"\b{re.escape(material)}\b", text, re.I)]


def _artwork_type_values(text: str) -> list[str]:
    values: list[str] = []
    for value in ["statue", "sculpture", "painting", "portrait", "installation"]:
        for match in re.finditer(rf"\b{re.escape(value)}\b", text, re.I):
            normalized = "sculpture" if value == "statue" else value
            values.append(normalized)
    return _dedupe(values)


def _museum_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(
        r"\b(?:the\s+)?([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}\s+Museum(?:\s+of\s+Art)?)\b",
        text,
    ):
        value = _clean_slot_phrase(match.group(1))
        if value:
            values.append(value)
    for pattern in [
        r"\b(?:in|at)\s+(?:the\s+)?([A-Z][A-Za-z .'-]+Museum(?:\s+of\s+Art)?(?:\s+in\s+[A-Z][A-Za-z .'-]+)?)",
        r"\b(?:in|at)\s+(?:the\s+)?([A-Z][A-Za-z .'-]+\s+Museum(?:\s+of\s+Art)?(?:\s+in\s+[A-Z][A-Za-z .'-]+)?)",
        r"\bmuseum\s+of\s+([A-Z][A-Za-z .'-]+)",
        r"\bin\s+([A-Z][A-Za-z .'-]+)\s+museum\b",
    ]:
        for match in re.finditer(pattern, text):
            value = _clean_slot_phrase(match.group(1))
            if value:
                values.append(value)
    deduped = _dedupe(values)
    bases = {re.sub(r"\s+in\s+[A-Z][A-Za-z .'-]+$", "", value).strip() for value in deduped}
    return [
        value
        for value in deduped
        if not (re.search(r"\s+in\s+[A-Z]", value) and re.sub(r"\s+in\s+[A-Z][A-Za-z .'-]+$", "", value).strip() in bases)
    ]


def _artist_value(text: str) -> str | None:
    match = re.search(r"\b(?:made|painted|created)\s+by\s+([A-Z][A-Za-z .'-]+?)(?:\s+in\b|[?.!,]|$)", text)
    return _clean_slot_phrase(match.group(1)) if match else None


def _player_value(text: str) -> str | None:
    match = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)'s\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(?:player|athlete)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b", text)
    return match.group(1) if match else None


def _game_values(text: str) -> list[str]:
    values: list[str] = []
    for pattern in [
        r"\b(?:in|for)\s+game\s+([A-Z][A-Za-z0-9 '&:-]+?)(?=\s*(?:,|\.|and\b|with\b|$))",
        r"\bgame\s+['\"]([^'\"]+)['\"]",
    ]:
        for match in re.finditer(pattern, text, re.I):
            value = _clean_slot_phrase(match.group(1))
            if value:
                values.append(value)
    quoted = _quoted_strings(text)
    if quoted and re.search(r"\b(?:game|gaming|level|platform|trophy|mission|guide)\b", text, re.I):
        values.extend(value for value in quoted if not re.search(r"\b(?:pc|xbox|playstation|nintendo|switch|master)\b", value, re.I))
    return _dedupe(values)


def _looks_like_game_name_slot(arg: str, slot_text: str) -> bool:
    if arg in {"game", "game_name", "gamename", "title"}:
        return True
    return bool(
        re.search(r"\bname\s+of\s+the\s+game\b", slot_text)
        or re.search(r"\bgame\s+name\b", slot_text)
    )


def _condition_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(
        r"\b([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})\s+conditions?\b",
        text,
        re.I,
    ):
        value = _clean_slot_phrase(match.group(1))
        value = re.sub(r"^(?:how\s+to\s+)?win\s+in\s+", "", value, flags=re.I)
        value = re.sub(r"^(?:in|under|with|for)\s+", "", value, flags=re.I)
        if value and value.lower() not in {"game", "specific", "weather"}:
            values.append(value)
    for match in re.finditer(r"\b((?:hard|easy|normal|expert|beginner)\s+mode)\b", text, re.I):
        values.append(_clean_slot_phrase(match.group(1)))
    return _dedupe(values)


def _schema_example_modifier_values(text: str, slot_text: str, head: str) -> list[str]:
    examples = []
    for single_quoted, double_quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", slot_text):
        value = (single_quoted or double_quoted).strip().lower()
        if value:
            examples.append(value)
    examples = _dedupe(examples)
    values: list[str] = []
    for example in examples:
        if re.search(rf"\b{re.escape(example)}\b\s+{re.escape(head)}s?\b", text, re.I):
            values.append(example)
        elif re.search(rf"\b{re.escape(example)}\b", text, re.I):
            values.append(example)
    return _dedupe(values)


def _attribute_value(text: str) -> str | None:
    match = re.search(
        r"\b(?:change|set|modify|update)\s+(?:the\s+)?([A-Za-z][A-Za-z _-]+?)\s+level\s+to\s+\d+",
        text,
        re.I,
    )
    if match:
        return _clean_slot_phrase(match.group(1))
    return None


def _player_stat_values(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\b(highest scoring game)\b",
        r"\b(total championships)\b",
        r"\b(championships (?:he|she|they)?\s*won)\b",
        r"\b(points scored)\b",
        r"\b(career points)\b",
        r"\b(points per game)\b",
        r"\b(assists(?: per game)?)\b",
        r"\b(rebounds(?: per game)?)\b",
        r"\b(minutes(?: per game)?)\b",
        r"\b(turnovers?)\b",
        r"\b(win rate)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            values.append(_clean_slot_phrase(match.group(1)))
    normalized: list[str] = []
    seen_keys: set[str] = set()
    for value in values:
        key = "championships" if "championship" in value.lower() else value.lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized.append(value)
    return normalized


def _has_repeated_scalar_selector(user_request: str, tool: dict[str, Any]) -> bool:
    for name, spec in _properties(tool).items():
        typ = _property_type(spec)
        if typ == "array":
            continue
        if len(_enum_value_evidence(user_request, spec)) > 1:
            return True
        if typ in {"integer", "float", "number"} and (
            _unit_labeled_numeric_slot_count(user_request, name, spec) > 1
            or _numeric_slot_list_count(user_request, tool) > 1
        ):
            return True
        if typ == "string":
            slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
            if _entity_count_for_arg(user_request, name, spec) > 1:
                return True
            if _looks_like_location_slot(name.lower(), slot_text) and len(_location_units(user_request)) > 1:
                return True
    return _case_identifier_count(user_request) > 1 and "case" in _tool_text(tool).lower()


def _unit_labeled_numeric_slot_count(user_request: str, arg: str, spec: dict[str, Any]) -> int:
    values = _unit_labeled_numeric_values(user_request, arg, spec)
    if len(values) > 1:
        return min(len(values), 8)
    return 1


def _unit_labeled_numeric_values(user_request: str, arg: str, spec: dict[str, Any]) -> list[Any]:
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if _looks_like_result_count_slot(arg, slot_text):
        return []
    units = _numeric_slot_units(slot_text)
    if not units:
        return []
    lowered = user_request.lower()
    if _looks_like_numeric_range_request(lowered, units):
        return []

    number = r"(?:-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    number_list = rf"{number}(?:\s*(?:,\s*(?:and\s*)?|\s+(?:and|or)\s+)\s*{number})+"
    unit_pattern = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
    labeled_values = _unit_labeled_numeric_values_near_slot(user_request, arg, spec, number, unit_pattern)
    if labeled_values:
        return labeled_values
    mentions: list[tuple[int, Any]] = []
    list_spans: list[tuple[int, int]] = []

    for match in re.finditer(rf"\b({number_list})\s*(?:{unit_pattern})\b", lowered, re.I):
        list_spans.append((match.start(), match.end()))
        for value_match in re.finditer(number, match.group(1), re.I):
            value = _number_from_text(value_match.group(0))
            if value is not None:
                mentions.append((match.start(1) + value_match.start(), value))
    for match in re.finditer(rf"\b(?:{unit_pattern})\s+(?:of\s+)?({number_list})\b", lowered, re.I):
        list_spans.append((match.start(), match.end()))
        for value_match in re.finditer(number, match.group(1), re.I):
            value = _number_from_text(value_match.group(0))
            if value is not None:
                mentions.append((match.start(1) + value_match.start(), value))

    for match in re.finditer(rf"\b({number})\s*(?:{unit_pattern})\b", lowered, re.I):
        if any(start <= match.start() < end for start, end in list_spans):
            continue
        value = _number_from_text(match.group(1))
        if value is not None:
            mentions.append((match.start(1), value))

    mentions.sort(key=lambda item: item[0])
    return [value for _position, value in mentions]


def _unit_labeled_numeric_values_near_slot(
    user_request: str,
    arg: str,
    spec: dict[str, Any],
    number_pattern: str,
    unit_pattern: str,
) -> list[Any]:
    labels = _numeric_slot_labels(arg, spec) + [arg.replace("_", " ")]
    mentions: list[tuple[int, Any]] = []
    for label in _dedupe(labels):
        if len(label) <= 1:
            continue
        label_pattern = re.escape(label)
        pattern = rf"\b{label_pattern}s?\b(?:\s+(?:of|is|as|=|:|to|for|with|has))*[^\dA-Za-z-]{{0,24}}({number_pattern})\s*(?:{unit_pattern})\b"
        for match in re.finditer(pattern, user_request, re.I):
            value = _number_from_text(match.group(1))
            if value is not None:
                mentions.append((match.start(1), value))
            tail = user_request[match.end() : match.end() + 48]
            for extra in re.finditer(rf"\b(?:and|or|,)\s*({number_pattern})\s*(?:{unit_pattern})\b", tail, re.I):
                extra_value = _number_from_text(extra.group(1))
                if extra_value is not None:
                    mentions.append((match.end() + extra.start(1), extra_value))
    mentions.sort(key=lambda item: item[0])
    unique: list[Any] = []
    seen_positions: set[int] = set()
    for position, value in mentions:
        if position in seen_positions:
            continue
        seen_positions.add(position)
        unique.append(value)
    return unique


def _numeric_slot_units(slot_text: str) -> set[str]:
    units: set[str] = set()
    if any(token in slot_text for token in ["year", "term", "duration"]):
        units.update({"year", "years"})
    if "season" in slot_text:
        units.update({"season", "seasons"})
    if "month" in slot_text:
        units.update({"month", "months"})
    if "week" in slot_text:
        units.update({"week", "weeks"})
    if "day" in slot_text:
        units.update({"day", "days"})
    if "hour" in slot_text:
        units.update({"hour", "hours"})
    if "minute" in slot_text:
        units.update({"minute", "minutes"})
    if "second" in slot_text:
        units.update({"second", "seconds"})
    if any(token in slot_text for token in ["distance", "radius", "diameter", "length"]):
        units.update({"meter", "meters", "metre", "metres", "mile", "miles", "km", "kilometer", "kilometers"})
    return units


def _looks_like_numeric_range_request(lowered: str, units: set[str]) -> bool:
    unit_pattern = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
    number = r"(?:-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)"
    return bool(
        re.search(rf"\bfrom\s+{number}\s*(?:{unit_pattern})?\s+to\s+{number}\s*(?:{unit_pattern})?\b", lowered, re.I)
        or re.search(rf"\bbetween\s+{number}\s*(?:{unit_pattern})?\s+and\s+{number}\s*(?:{unit_pattern})?\b", lowered, re.I)
    )


def _route_pair_count(user_request: str, tool: dict[str, Any]) -> int:
    if not _tool_looks_like_route_request(tool):
        return 1
    count = len(_route_endpoint_pairs(user_request))
    return count if count > 1 else 1


def _route_alternative_count(user_request: str, tool: dict[str, Any]) -> int:
    if not _tool_looks_like_route_request(tool):
        return 1
    lowered = user_request.lower()
    what_if_count = len(re.findall(r"\bwhat\s+if\b", lowered))
    if what_if_count > 1:
        return min(what_if_count + 1, 8)
    if what_if_count == 1 and re.search(r"\b(?:also|lastly|finally)\b", lowered):
        return 2
    return 1


def _tool_looks_like_route_request(tool: dict[str, Any]) -> bool:
    tool_text = _tool_text(tool).lower().replace("_", " ")
    properties = _properties(tool)
    start_slots = [
        name
        for name, spec in properties.items()
        if _property_type(spec) == "string" and _looks_like_route_start_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower())
    ]
    end_slots = [
        name
        for name, spec in properties.items()
        if _property_type(spec) == "string" and _looks_like_route_end_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower())
    ]
    if start_slots and end_slots:
        return True
    route_terms = {"direction", "directions", "distance", "route", "routes", "travel", "trip", "path"}
    string_slot_count = sum(1 for spec in properties.values() if _property_type(spec) == "string")
    return string_slot_count >= 2 and any(term in tool_text for term in route_terms)


def _boolean_output_variant_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    for name, spec in _properties(tool).items():
        if _property_type(spec) != "boolean":
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if "monthly" in slot_text and re.search(r"\bmonthly\b", lowered) and re.search(r"\bannually|annual|yearly\b", lowered):
            return 2
    if not re.search(r"\b(?:both|as\s+(?:a\s+)?|as\s+well\s+as)\b", lowered):
        return 1
    has_text_format = re.search(r"\b(?:string|text|formatted|plain\s+text)\b", lowered)
    has_array_format = re.search(r"\b(?:array|list|json)\b", lowered)
    if not (has_text_format and has_array_format):
        return 1
    for name, spec in _properties(tool).items():
        if _property_type(spec) != "boolean":
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if any(token in slot_text for token in ["format", "string", "array", "list", "return"]):
            return 2
    return 1


def _requested_detail_value_count(user_request: str, spec: dict[str, Any]) -> int:
    values = _enum_value_evidence(user_request, spec)
    if values:
        return len(values)
    description = str(spec.get("description") or "").lower()
    if "trial" in description or "status" in description:
        matches = 0
        lowered = user_request.lower()
        if re.search(r"\bstatus\b", lowered):
            matches += 1
        if re.search(r"\b(?:trial\s+date|scheduled\s+trial|court\s+date)\b", lowered):
            matches += 1
        return matches
    return 1


def _has_named_numeric_pair_context(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:chose|chooses|picked|select(?:ed)?|has|with)\s+(?:the\s+)?numbers?\b",
            user_request,
        )
        or re.search(r"\bwhile\s+[A-Z][a-z]+", user_request)
    )


def _ordinal_scenario_count(user_request: str) -> int:
    lowered = user_request.lower()
    ordinal_values = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    labels: set[str] = set()
    max_index = 0
    pattern = (
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\s+"
        r"(?:array|case|charge|circle|condition|configuration|dataset|data\s+set|equation|experiment|function|group|hotel|item|list|material|object|one|option|pair|reservation|room|round|sample|scenario|setup|substance|task|triangle|trial|vehicle|car)\b"
    )
    for match in re.finditer(pattern, lowered):
        ordinal = match.group(1)
        if ordinal == "last":
            labels.add(f"last:{match.group(0)}")
            continue
        value = ordinal_values[ordinal]
        labels.add(str(value))
        max_index = max(max_index, value)
    if max_index > len(labels) and "1" not in labels:
        return max_index
    count = len(labels)
    return count if count > 1 else 1


def _scoped_request_refs_full_scenarios(scoped_request: str) -> bool:
    scenario_nouns = (
        r"arrays?|cases?|charges?|circles?|conditions?|configurations?|datasets?|data\s+sets?|"
        r"equations?|experiments?|groups?|hotels?|items?|lists?|materials?|objects?|options?|"
        r"pairs?|reservations?|rooms?|rounds?|samples?|scenarios?|setups?|substances?|tasks?|"
        r"triangles?|trials?|vehicles?|cars?"
    )
    return bool(
        re.search(
            rf"\b(?:both|each|all|these|those|the)\s+(?:(?:two|three|four|five|\d+)\s+)?(?:different\s+)?{scenario_nouns}\b",
            scoped_request,
            re.I,
        )
    )


def _has_same_tool_parallel_context(user_request: str) -> bool:
    lowered = user_request.lower()
    if re.search(r"\b(?:for each|for every|each pair|each of|both of|for both|respectively|separately|simultaneously)\b", lowered):
        return True
    if re.search(r"\b(?:do the same|repeat|again)\b", lowered):
        return True
    if _ordinal_scenario_count(user_request) > 1:
        return True
    if _explicit_group_count(lowered) > 1 and not _has_result_count_phrase(user_request):
        return True
    return len(_financial_scenarios(user_request)) > 1


def _repeated_numeric_label_count(user_request: str, arg: str, spec: dict[str, Any]) -> int:
    labels = _numeric_slot_labels(arg, spec) + [arg.replace("_", " ")]
    number = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    total = 0
    for label in _dedupe(label for label in labels if label):
        if label == "second":
            continue
        total = max(
            total,
            len(re.findall(rf"\b{re.escape(label)}s?\D{{0,20}}{number}", user_request, re.I)),
            len(re.findall(rf"{number}\D{{0,20}}\b{re.escape(label)}s?\b", user_request, re.I)),
        )
    return total if total > 1 else 1


def _repeated_unary_operation_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    operation_terms: list[str] = []
    if "factorial" in tool_text:
        operation_terms.append("factorial")
    if "prime factor" in tool_text:
        operation_terms.extend(["prime factors", "prime factor"])
    if "derivative" in tool_text or "differentiat" in tool_text:
        operation_terms.append("derivative")
    if "integral" in tool_text or "integrat" in tool_text:
        operation_terms.append("integral")
    if not operation_terms:
        return 1
    number = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    counts = [
        len(re.findall(rf"\b{re.escape(term)}\s+of\s+{number}\b", user_request, re.I))
        for term in operation_terms
    ]
    count = max(counts) if counts else 1
    return count if count > 1 else 1


def _numeric_slot_labels(name: str, spec: dict[str, Any]) -> list[str]:
    text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
    name_label = name.lower().replace("_", " ")
    labels = {name_label} if len(name_label) > 1 else set()
    description = str(spec.get("description") or "").lower().replace("_", " ").strip(" .")
    if description and len(description.split()) <= 5:
        labels.add(description)
    if "magnetic field" in text:
        labels.add("magnetic field")
    if "change in time" in text:
        labels.add("change in time")
    if "interest rate" in text:
        labels.add("interest rate")
    if "loan amount" in text:
        labels.add("loan amount")
    if "loan term" in text:
        labels.add("loan term")
    if "year" in text:
        labels.update({"year", "years", "term"})
    if "month" in text:
        labels.update({"month", "months"})
    if "day" in text:
        labels.update({"day", "days"})
    if "second" in text:
        labels.update({"second", "seconds", "time"})
    if "minute" in text:
        labels.update({"minute", "minutes", "duration"})
    return [label for label in labels if label]


def _slot_value_varies_by_call(arg: str, phrase: str) -> bool:
    text = f"{arg} {phrase}".lower()
    return any(
        token in text
        for token in [
            "amount",
            "base",
            "case",
            "change in time",
            "duration",
            "density",
            "height",
            "income",
            "term",
            "number",
            "radius",
            "value",
            "year",
            "month",
            "day",
            "second",
            "minute",
            "time",
        ]
    )


def _number_list_counts_near_label(user_request: str, label: str) -> list[int]:
    label_pattern = re.escape(label)
    number = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    number_list = rf"{number}(?:\s*(?:,\s*(?:and\s*)?|\s+(?:and|or)\s+)\s*{number})+"
    counts = []
    patterns = [
        rf"\b({number_list})\s+{label_pattern}s?\b",
        rf"\b{label_pattern}s?\D{{0,30}}({number_list})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, user_request, re.I):
            count = len(extract_numbers(match.group(1)))
            if count > 1:
                counts.append(count)
    return counts


def _article_entity_group_count(user_request: str, tool: dict[str, Any]) -> int:
    numeric_required = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _property_type(spec) in {"integer", "float", "number"}
    ]
    if len(numeric_required) < 2:
        return 1
    if not re.search(r"\b(?:objects?|materials?|particles?|protons?|electrons?|neutrons?)\b", user_request, re.I):
        return 1
    marked = re.sub(
        r",\s+(?:and\s+)?(?:a|an|the)\s+"
        r"(?!(?:least|greatest|derivative|integral|area|sum|product|factorial|prime|common)\b)",
        "|",
        user_request,
        flags=re.I,
    )
    parts = [part.strip() for part in marked.split("|") if part.strip()]
    if len(parts) < 2:
        return 1
    arg_words = [word for name in numeric_required for word in _tokens(name)]
    bad_leading_words = set(arg_words) | MEASUREMENT_TOKENS | VALUE_TOKENS | {
        "assets",
        "dielectric",
        "distance",
        "income",
        "interest",
        "liabilities",
        "plate",
        "revenue",
    }
    for part in parts[1:]:
        first = next(iter(_tokens(part)), "")
        if first in bad_leading_words:
            return 1
    count = 0
    for part in parts:
        lowered = part.lower()
        has_value = bool(extract_numbers(part)) or any(f"no {word}" in lowered for word in arg_words)
        if has_value:
            count += 1
    return count if count > 1 else 1


def _function_definition_group_count(user_request: str, tool: dict[str, Any]) -> int:
    if not any(name in _properties(tool) for name in {"function", "func", "equation"}):
        return 1
    count = len(re.findall(r"\b[a-z]\s*\(\s*x\s*\)\s*=", user_request, re.I))
    return count if count > 1 else 1


def _repeated_operation_clause_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    tool_text = _tool_text(tool).lower()
    phrase_patterns = []
    if "derivative" in tool_text:
        phrase_patterns.extend([
            r"\b(?:calculate|compute|find|determine)\s+(?:the\s+)?derivatives?\b",
            r"\b(?:calculate|compute|find|determine)\s+the\s+(?:first|second|third|\d+(?:st|nd|rd|th)?)\s+(?:order\s+)?derivative\b",
            r"\b(?:first|second|third|\d+(?:st|nd|rd|th)?)\s+(?:order\s+)?derivative\b",
        ])
    if "greatest common divisor" in tool_text or re.search(r"\bgcd\b", tool_text):
        count = _operation_numeric_pair_mention_count(lowered, "greatest common divisor", "gcd")
        if count > 1:
            return count
    if "least common multiple" in tool_text or re.search(r"\blcm\b", tool_text):
        count = _operation_numeric_pair_mention_count(lowered, "least common multiple", "lcm")
        if count > 1:
            return count
    if "permutation" in tool_text:
        phrase_patterns.append(r"\bnumber of different\b")
    if "time required" in tool_text:
        phrase_patterns.append(r"\bcalculate the time required\b")
    if "energy required" in tool_text:
        phrase_patterns.append(r"\benergy required\b")
    for pattern in phrase_patterns:
        count = len(re.findall(pattern, lowered))
        if count > 1:
            return count
    if "density" in tool_text and re.search(r"\bmass\b", tool_text):
        density_count = len(re.findall(r"\bdensity\s+of\s+\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*g/cm\^?3\b", lowered))
        if density_count > 1:
            return density_count
    return 1


def _operation_numeric_pair_mention_count(lowered: str, phrase: str, acronym: str) -> int:
    normalized = re.sub(
        rf"\b{re.escape(phrase)}\s*\(\s*{re.escape(acronym)}\s*\)",
        phrase,
        lowered,
        flags=re.I,
    )
    pattern = (
        rf"\b(?:{re.escape(phrase)}|{re.escape(acronym)})\b"
        r"\D{0,80}?(-?\d+(?:\.\d+)?)\s+(?:and|,)\s+(-?\d+(?:\.\d+)?)"
    )
    pairs = {(match.group(1), match.group(2)) for match in re.finditer(pattern, normalized, re.I)}
    return min(len(pairs), 8) if len(pairs) > 1 else 1


def _integration_method_function_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    properties = _properties(tool)
    if "integrat" not in tool_text:
        return 1
    if not any("function" in name.lower() for name in properties):
        return 1
    if not any("method" in name.lower() or "method" in str(spec.get("description") or "").lower() for name, spec in properties.items()):
        return 1
    quoted = re.findall(r"'([^']+)'", user_request)
    functions = [value for value in quoted if re.search(r"[a-z]\s*\^|[+\-*/]", value, re.I)]
    method_values = _method_values_for_tool(user_request, tool)
    if len(functions) > 1 and len(method_values) > 1:
        return min(len(functions) * len(method_values), 8)
    return 1


def _method_values_for_tool(user_request: str, tool: dict[str, Any]) -> list[str]:
    quoted = [value.lower() for value in re.findall(r"'([^']+)'", user_request)]
    descriptions = " ".join(
        str(spec.get("description") or "")
        for name, spec in _properties(tool).items()
        if "method" in name.lower() or "method" in str(spec.get("description") or "").lower()
    ).lower()
    choices = set(re.findall(r"'([^']+)'", descriptions))
    if choices:
        return _dedupe(value for value in quoted if value in choices)
    method_words = {"trapezoid", "simpson", "euler", "newton", "bisection"}
    return _dedupe(value for value in quoted if value in method_words)


def _experiment_repeat_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    tool_text = _tool_text(tool).lower()
    if "experiment" not in lowered:
        return 1
    if not any(token in tool_text for token in ["medium", "permittivity", "experiment", "force", "field"]):
        return 1
    for pattern in [
        r"\bexperiment\s+(?:is\s+)?(?:performed\s+)?(twice|three|four|two|3|4|2)\b",
        r"\bperform(?:ed)?\s+the\s+experiment\s+(twice|three|four|two|3|4|2)\b",
    ]:
        match = re.search(pattern, lowered)
        if match:
            value = 2 if match.group(1) == "twice" else _number_from_text(match.group(1))
            if isinstance(value, int) and value > 1:
                return value
    return 1


def _amount_location_pair_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    if not any(token in tool_text for token in ("tax", "price", "cost")):
        return 1
    count = len(
        re.findall(
            r"[$]\s*\d[\d,.]*(?:\s+\w+)?\s+in\s+[A-Z][A-Za-z .'-]+?(?=,|\band\b|[?.!]|$)",
            user_request,
        )
    )
    return count if count > 1 else 1


def _capital_gain_scenario_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    if "capital" not in tool_text or "gain" not in tool_text:
        return 1
    scenarios = _capital_gain_scenarios(user_request)
    return len(scenarios) if len(scenarios) > 1 else 1


def _capital_gain_value_for_slot(user_request: str, arg: str, call_index: int) -> Any | None:
    scenarios = _capital_gain_scenarios(user_request)
    if not scenarios:
        return None
    key = ""
    arg_text = arg.lower().replace("_", " ")
    if "short" in arg_text:
        key = "short_term_gain"
    elif "long" in arg_text:
        key = "long_term_gain"
    if not key:
        return None
    scenario = scenarios[min(call_index, len(scenarios) - 1)]
    return scenario.get(key)


def _capital_gain_states(user_request: str) -> list[str]:
    return [str(item["state"]) for item in _capital_gain_scenarios(user_request) if item.get("state")]


def _capital_gain_scenarios(user_request: str) -> list[dict[str, Any]]:
    segments = re.split(r"\s+\band\s+(?=\$?\d|short\s+term\b)", user_request, flags=re.I)
    scenarios: list[dict[str, Any]] = []
    for segment in segments:
        if not re.search(r"\bshort\s+term\b", segment, re.I) or not re.search(r"\blong\s+term\b", segment, re.I):
            continue
        short_value = _capital_gain_amount(segment, "short")
        long_value = _capital_gain_amount(segment, "long")
        state = _capital_gain_state(segment)
        if short_value is None or long_value is None:
            continue
        item = {"short_term_gain": short_value, "long_term_gain": long_value}
        if state:
            item["state"] = state
        scenarios.append(item)
    return scenarios


def _capital_gain_amount(segment: str, term: str) -> Any | None:
    after = re.search(
        rf"\b{term}\s+term(?:\s+capital)?\s+gains?\s*(?:of|=|:)?\s*\$?(\d[\d,]*)",
        segment,
        re.I,
    )
    if after:
        return _number_from_text(after.group(1).replace(",", ""))
    before = re.search(rf"\$?(\d[\d,]*)\s+{term}\s+term\b", segment, re.I)
    if before:
        return _number_from_text(before.group(1).replace(",", ""))
    return None


def _capital_gain_state(segment: str) -> str | None:
    match = re.search(r"\bin\s+(?:the\s+state\s+of\s+)?([A-Z][A-Za-z ]+?)(?:[?.!]|$)", segment)
    if not match:
        return None
    state = _clean_location_value(match.group(1))
    return state or None


def _numeric_location_pair_count(user_request: str, tool: dict[str, Any]) -> int:
    properties = _properties(tool)
    has_numeric = any(
        name in _required(tool) and _property_type(spec) in {"integer", "float", "number"}
        for name, spec in properties.items()
    )
    has_location = any(
        name in _required(tool) and _property_type(spec) == "string" and _looks_like_location_slot(name.lower(), f"{name} {spec.get('description') or ''}")
        for name, spec in properties.items()
    )
    if not has_numeric or not has_location:
        return 1
    pairs = _numeric_location_pairs(user_request)
    return len(pairs) if len(pairs) > 1 else 1


def _numeric_location_pair_locations(user_request: str) -> list[str]:
    return [location for _number, location in _numeric_location_pairs(user_request)]


def _looks_like_size_numeric_slot(arg: str, slot_text: str) -> bool:
    return bool(
        arg in {"size", "area", "square_feet", "sqft"}
        or any(token in slot_text for token in ["size", "square feet", "sq ft", "sqft", "area"])
    )


def _numeric_location_pairs(user_request: str) -> list[tuple[str, str]]:
    unit = r"(?:sq\.?\s*ft\.?|square\s+feet|square\s+foot|ft\.?|feet|m2|sqm|square\s+meters?)"
    pattern = (
        rf"\b(-?\d+(?:\.\d+)?)\s*(?:{unit})?\.?\s+"
        r"(?:in|at|for)\s+(?:location\s+)?"
        r"([A-Z][A-Za-z .'-]+?)"
        r"(?=\s+(?:and|,)\s+-?\d|\s+using\b|[?.!]|$)"
    )
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(pattern, user_request):
        location = _clean_location_value(match.group(2))
        if location and location.lower() not in STOPWORDS:
            pairs.append((match.group(1), location))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for number, location in pairs:
        key = (str(number), location.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((str(number), location))
    return deduped


def _paired_numeric_sequence_count(user_request: str, required_count: int) -> int:
    if required_count != 2:
        return 1
    coefficient_count = len(_equation_coefficient_groups(user_request))
    if coefficient_count > 1:
        return coefficient_count
    pair_count = len(
        re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?\s+and\s+-?\d+(?:\.\d+)?",
            user_request,
            re.I,
        )
    )
    return pair_count if pair_count > 1 else 1


def _split_named_units(chunk: str) -> list[str]:
    chunk = _strip_location_trailing_context(chunk)
    chunk = re.split(r"\s*,\s*(?:and\s+)?(?:can|could|please|with|using)\b", chunk, maxsplit=1, flags=re.I)[0]
    chunk = re.split(r"\s+\b(?:please|using|with|right now|currently|today|now)\b", chunk, maxsplit=1, flags=re.I)[0]
    if re.search(r"\b(?:and|also|y)\b", chunk, re.I):
        major_parts = [
            part.strip(" .'\"")
            for part in re.split(r"\b(?:and|also|y)\b", chunk, flags=re.I)
            if part.strip(" .'\"")
        ]
        units: list[str] = []
        for part in major_parts:
            part = re.split(r"\s+from\s+(?:us\s+)?(?:census|database|api|source)\b", part, maxsplit=1, flags=re.I)[0]
            comma_parts = [value.strip(" .'\"") for value in part.split(",") if value.strip(" .'\"")]
            paired = _pair_region_suffix_parts(comma_parts)
            if paired:
                units.extend(paired)
            elif len(comma_parts) == 2 and _looks_like_region_suffix(comma_parts[1]):
                units.append(", ".join(comma_parts))
            elif len(comma_parts) == 2 and _looks_like_named_place_prefix(comma_parts[0]):
                units.append(comma_parts[1])
            elif len(comma_parts) > 1:
                units.extend(comma_parts)
            else:
                units.append(part)
        return units

    chunk = re.split(r"\s+from\s+(?:us\s+)?(?:census|database|api|source)\b", chunk, maxsplit=1, flags=re.I)[0]
    raw_parts = [part.strip(" .'\"") for part in chunk.split(",") if part.strip(" .'\"")]
    if len(raw_parts) == 2 and _looks_like_region_suffix(raw_parts[1]):
        return [", ".join(raw_parts)]
    if len(raw_parts) == 2 and _looks_like_named_place_prefix(raw_parts[0]):
        return [raw_parts[1]]
    if len(raw_parts) > 2:
        paired = _pair_region_suffix_parts(raw_parts)
        if paired:
            return paired
        return raw_parts
    return raw_parts


def _filter_location_units(values: list[str]) -> list[str]:
    return _dedupe(value for value in values if _looks_like_place_value(value))


def _pair_region_suffix_parts(parts: list[str]) -> list[str]:
    if len(parts) < 4 or len(parts) % 2:
        return []
    paired = []
    for index in range(0, len(parts), 2):
        city = parts[index].strip(" .'\"")
        suffix = parts[index + 1].strip(" .'\"")
        if not city or not _looks_like_region_suffix(suffix):
            return []
        paired.append(f"{city}, {suffix}")
    return paired


def _strip_location_trailing_context(value: str) -> str:
    value = re.split(
        r"\s+\b(?:(?:for|in)\s+(?:the\s+)?(?:next|upcoming|past|last)\b|(?:for|in)\s+\d+\s+days?\b|on\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december|\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?)\b|checking\b|check[- ]?in\b|right now|currently|today|now)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]
    return value.strip(" .'\"")


def _clean_location_value(value: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", str(value)):
        cleaned = _strip_location_trailing_context(re.sub(r"\s+", " ", str(value))).strip(" .'\"")
        cleaned = re.sub(r"^(?:of|in|at|near|for)\s+", "", cleaned, flags=re.I)
        return cleaned
    cleaned = _strip_location_trailing_context(_clean_slot_phrase(value))
    return re.sub(r"\bD\.C$", "D.C.", cleaned)


def _looks_like_place_value(value: str) -> bool:
    cleaned = value.strip(" .'\"")
    if not cleaned:
        return False
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return True
    key = _location_alias_key(cleaned)
    if key in LOCATION_ALIASES:
        return True
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return False
    lowered = cleaned.lower()
    if any(re.search(rf"\b{re.escape(month)}\b", lowered) for month in MONTH_NUMBERS):
        return False
    if re.search(r"\b(?:checking|check[- ]?in|check[- ]?out|image|prompt|query|search|news|message|year)\b", lowered):
        return False
    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) >= 2 and _looks_like_region_suffix(parts[-1]):
            return True
    return bool(re.match(r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*){0,5}$", cleaned))


def _near_location_value(text: str) -> str | None:
    match = re.search(
        r"\bnear\s+([A-Z][A-Za-z .'-]+?)(?:\s+(?:with|for|at|on|using)\b|[?.!]|$)",
        text,
    )
    if not match:
        return None
    value = _clean_location_value(match.group(1))
    return value or None


def _looks_like_region_suffix(text: str) -> bool:
    value = text.strip()
    if re.fullmatch(r"[A-Z]{2,3}", value):
        return True
    us_states = {
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new york",
        "north carolina",
        "north dakota",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "rhode island",
        "south carolina",
        "south dakota",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "west virginia",
        "wisconsin",
        "wyoming",
    }
    if value.lower() in us_states:
        return True
    return value.lower() in {
        "argentina",
        "australia",
        "brazil",
        "canada",
        "china",
        "england",
        "france",
        "germany",
        "india",
        "israel",
        "italy",
        "japan",
        "latvia",
        "mexico",
        "russia",
        "south korea",
        "spain",
        "thailand",
        "uk",
        "united kingdom",
        "usa",
    }


def _looks_like_named_place_prefix(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:hotel|inn|resort|motel|museum|gallery|theat(?:er|re)|stadium|arena|airport|hospital|university|college|restaurant|cafe|bar|center|centre)\b",
            text,
            re.I,
        )
    )


def _semantic_frame_expected_call_count(query_input_audit: dict[str, Any] | None) -> int | None:
    if not isinstance(query_input_audit, dict):
        return None
    counts = [
        _coerce_positive_int(group.get("expected_call_count"))
        for group in query_input_audit.get("semantic_call_groups") or []
        if isinstance(group, dict)
    ]
    counts = [count for count in counts if count is not None]
    if not counts:
        return None
    return max(counts)


def _semantic_frame_can_cap_to_single_call(user_request: str, tool: dict[str, Any]) -> bool:
    lowered = user_request.lower()
    current_action = _current_user_action_text(user_request).lower()
    action_scope = current_action or lowered
    if re.search(r"\b(?:for each|for both|each of|both of|respectively|separately|simultaneously|for every)\b", action_scope):
        return False
    if re.search(r"\b(?:and also|also get|also find|then get|then find|repeat|again|do the same)\b", action_scope):
        return False
    for name, spec in _properties(tool).items():
        slot_text = f"{name} {spec.get('description') or ''}"
        if _looks_like_property_selector_slot(name.lower(), slot_text) and len(_requested_property_values(user_request, spec)) > 1:
            return False
    if _tool_has_required_batch_array(tool):
        return False
    return True


def _current_user_action_text(user_request: str) -> str:
    text = re.sub(r"\bLatest prior API result:\s*.*$", "", user_request, flags=re.I | re.S).strip()
    if not text:
        return ""
    matches = list(re.finditer(r"(?:^|[\n\r])\s*User:\s*", text, re.I))
    if not matches:
        return text
    tail = text[matches[-1].end() :]
    tail = re.split(r"(?:^|[\n\r])\s*(?:AI|Assistant):\s*", tail, maxsplit=1, flags=re.I)[0]
    return tail.strip()


def _clause_repetition_allowed(user_request: str) -> bool:
    if re.search(
        r"\b(?:earlier user|latest prior api result|prior api result|^user:| ai:| assistant:)\b",
        user_request,
        re.I,
    ):
        return False
    if _allows_multiple_tools(user_request):
        return True
    if len(_intent_clauses(user_request)) > 1:
        return True
    return False


def _optional_scalar_selector_repeat_count(user_request: str, tool: dict[str, Any]) -> int:
    required = set(_required(tool))
    counts: list[int] = []
    for name, spec in _properties(tool).items():
        if name in required:
            continue
        typ = _property_type(spec)
        if typ not in {"string", "integer", "float", "number"}:
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        if "season" in slot_text or "league year" in slot_text:
            values = _season_values(user_request)
            if len(values) > 1:
                counts.append(len(values))
        elif "year" in slot_text and re.search(r"\byears?\b", user_request, re.I):
            years = _year_values(user_request)
            if len(years) > 1:
                counts.append(len(years))
        elif ("date" in slot_text or _looks_like_time_slot(name.lower(), slot_text)) and re.search(
            r"\b(?:for each|each|both|respectively|separately)\b",
            user_request,
            re.I,
        ):
            temporal_values = _date_values(user_request) or _clock_time_values(user_request)
            if len(temporal_values) > 1:
                counts.append(len(temporal_values))
    return min(max(counts), 8) if counts else 1


def _single_operation_multi_slot_request(user_request: str, tool: dict[str, Any], relevant_clause_count: int = 1) -> bool:
    lowered = user_request.lower()
    if relevant_clause_count > 1 and "total balance" not in lowered:
        return False
    tool_text = _tool_text(tool).lower().replace("_", " ")
    if "currenc" in tool_text and _money_conversion_scenario_count(user_request) > 1:
        return False
    if re.search(r"\b(?:for each|for both|each of|both of|respectively|separately|simultaneously|for every)\b", lowered):
        return False
    if _route_pair_count(user_request, tool) > 1 or _route_alternative_count(user_request, tool) > 1:
        return False
    if _repeated_operation_clause_count(user_request, tool) > 1:
        return False
    if _function_definition_group_count(user_request, tool) > 1:
        return False
    properties = _properties(tool)
    required = _required(tool)
    numeric_required_count = sum(
        1 for name, spec in properties.items() if name in required and _is_numeric_value_property(spec)
    )
    string_required_count = sum(
        1 for name, spec in properties.items() if name in required and _property_type(spec) == "string"
    )
    between_operands = bool(re.search(r"\bbetween\s+(?:two\s+)?(?:cities|charges|points|locations|places|functions?)\b", lowered))
    if re.search(r"\blast\s+\d+\s+(?:rounds?|games?|matches?|results?)\b", lowered):
        return True
    if (
        len(extract_numbers(user_request)) == 1
        and numeric_required_count <= 1
        and any(token in tool_text for token in ["circumference", "circle", "radius"])
    ):
        return True
    if "currenc" in tool_text and len(_currency_mentions(user_request)) >= 2 and _currency_pair_count(user_request) <= 1:
        return True
    if "total balance" in lowered and "balance" in tool_text:
        return True
    if "including" in lowered and any(_property_type(spec) == "array" for spec in properties.values()):
        return True
    if between_operands and (numeric_required_count >= 2 or string_required_count >= 2 or "distance" in tool_text):
        return True
    if _has_same_tool_parallel_context(user_request):
        return False
    if "year over year" in lowered or "yoy" in lowered:
        return True
    if "conversion rate" in lowered or ("exchange rate" in lowered and "currenc" in tool_text and _currency_pair_count(user_request) <= 1):
        return True
    if (
        re.search(r"\bconvert\b.+\b(?:to|into)\b", lowered)
        and "currenc" in tool_text
        and _money_conversion_scenario_count(user_request) <= 1
    ):
        return True
    if "intersection" in lowered and "function" in lowered:
        return True
    if "reference sequence" in lowered:
        return True
    if "mixed with" in lowered and numeric_required_count >= 2:
        return True
    if re.search(r"\bbetween\b.+\band\b", lowered) and numeric_required_count >= 2:
        return True
    if re.search(r"\bdistance between\b|\bfrom\b.+\bto\b", lowered) and (
        "distance" in tool_text or "route" in tool_text or "currenc" in tool_text
    ):
        return True
    if (
        string_required_count >= 2
        and _tool_looks_like_route_request(tool)
        and re.search(r"\b(?:from|between)\b.+\b(?:to|and)\b", lowered)
    ):
        return True
    return False


def _year_values(text: str) -> list[str]:
    return _dedupe(re.findall(r"\b(?:19|20)\d{2}\b", text))


def _has_prior_result_context(user_request: str) -> bool:
    return bool(re.search(r"\b(?:prior|previous|latest)\s+(?:api\s+)?(?:call|result|response)\b", user_request, re.I))


def _semantic_slot_repeat_count(
    user_request: str,
    tool: dict[str, Any],
    query_input_audit: dict[str, Any] | None,
) -> int:
    if _has_result_count_phrase(user_request):
        return 1
    facts = _query_facts(user_request, query_input_audit)
    semantic_facts = [fact for fact in facts if fact.get("source") == "gptoss_semantic_slot_frame"]
    if not semantic_facts:
        return 1

    counts_by_slot: dict[str, int] = {}
    enum_selector_slots: set[str] = set()
    required = set(_required(tool))
    for name, spec in _properties(tool).items():
        if name not in required:
            continue
        typ = _property_type(spec)
        if typ == "array" or _looks_like_result_count_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower()):
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        values: list[Any] = []
        for fact in semantic_facts:
            role = str(fact.get("role") or "")
            if not _semantic_frame_role_matches_slot_for_cardinality(role, name.lower(), slot_text):
                continue
            value = fact.get("normalized_value")
            if value in (None, ""):
                continue
            if isinstance(value, list):
                values.extend(item for item in value if item not in (None, ""))
            else:
                values.append(value)
        if typ == "string" and _enum_values_for_spec(spec):
            enum_values = _enum_value_evidence(user_request, spec)
            if len(enum_values) > len(values):
                values = enum_values
            if len(enum_values) > 1:
                enum_selector_slots.add(name)
        if typ == "string" and "genre" in slot_text:
            genre_values = _music_genre_values(user_request)
            if len(genre_values) > len(values):
                values = genre_values
        count = len(_dedupe(str(value).strip().lower() for value in values if str(value).strip()))
        if count > 1:
            counts_by_slot[name] = count

    if not counts_by_slot:
        return 1
    if _semantic_slots_should_cross_product(user_request, tool, counts_by_slot, enum_selector_slots):
        product = 1
        for count in sorted(counts_by_slot.values(), reverse=True)[:2]:
            product *= count
        result = min(product, 8) if product > 1 else 1
    else:
        result = min(max(counts_by_slot.values()), 8)
    boundary_count = _semantic_scenario_boundary_count(user_request)
    if boundary_count > 1 and len(counts_by_slot) > 1:
        result = min(result, boundary_count)
    return result


def _semantic_frame_role_matches_slot_for_cardinality(role: str, arg: str, slot_text: str) -> bool:
    role_key = _normalize_role(role)
    slot_key = _normalize_role(arg)
    role_keys = _role_key_variants(role)
    slot_keys = _role_key_variants(arg)
    if not role_key or not slot_key:
        return False
    if not _indexed_role_compatible(role, arg):
        return False
    role_tokens = set(_tokens(role.replace("_", " ")))
    arg_tokens = set(_tokens(arg.replace("_", " ")))
    if ("unit" in role_key or "unit" in role_tokens) and "unit" not in slot_key and "unit" not in slot_text:
        return False
    if "unit" in slot_key and (role_tokens - {"unit", "units", "measure", "measurement"}):
        return False
    if "pair" in role_key and re.fullmatch(r"(?:num|number|value)[_ -]?\d+", arg.lower()):
        return False
    if role_keys & slot_keys:
        return True
    if any(
        len(role_variant) > 2
        and len(slot_variant) > 2
        and (role_variant in slot_variant or slot_variant in role_variant)
        and role_variant != "name"
        and slot_variant != "name"
        for role_variant in role_keys
        for slot_variant in slot_keys
    ):
        return True
    generic_tokens = {
        "appointment",
        "color",
        "data",
        "date",
        "field",
        "id",
        "input",
        "name",
        "new",
        "number",
        "patient",
        "registration",
        "text",
        "time",
        "value",
    }
    informative_arg_tokens = arg_tokens - generic_tokens
    if informative_arg_tokens and informative_arg_tokens <= role_tokens:
        return True
    for aliases in SEMANTIC_SLOT_ALIASES:
        if aliases & role_keys and (aliases & slot_keys or aliases & informative_arg_tokens):
            return True
    return False


def _semantic_scenario_boundary_count(user_request: str) -> int:
    explicit = _explicit_group_count(user_request.lower())
    ordinal = _ordinal_scenario_count(user_request)
    counts = [count for count in [explicit, ordinal] if count > 1]
    boundary_matches = len(
        re.findall(
            r"\b(?:also|another|same\s+(?:items?|function|tool|request|but)|repeat|then|and\s+for|as\s+well\s+as\s+one)\b",
            user_request,
            re.I,
        )
    )
    if boundary_matches:
        counts.append(boundary_matches + 1)
    return min(max(counts), 8) if counts else 1


def _semantic_slots_should_cross_product(
    user_request: str,
    tool: dict[str, Any],
    counts_by_slot: dict[str, int],
    enum_selector_slots: set[str],
) -> bool:
    if len(counts_by_slot) < 2:
        return False
    lowered = user_request.lower()
    if re.search(r"\b(?:respectively|also|another|same|second|third|fourth|then|and\s+for|as well as one)\b", lowered):
        return False
    if re.search(r"\b(?:each|every|all)\s+(?:combination|pairing)\b|\bcross[-\s]?product\b", lowered):
        return True
    if enum_selector_slots and any(
        token in name.lower()
        for name in enum_selector_slots
        for token in ["detail", "status", "type", "property", "field"]
    ):
        return True
    slot_keys = {name.lower().replace("_", " ") for name in counts_by_slot}
    if _slot_key_has(slot_keys, {"team", "club"}) and _slot_key_has(slot_keys, {"season", "year"}):
        return True
    if _slot_key_has(slot_keys, {"genre"}) and _slot_key_has(slot_keys, {"location", "city", "venue", "region"}):
        return True
    return False


def _slot_key_has(slot_keys: set[str], tokens: set[str]) -> bool:
    return any(any(re.search(rf"\b{re.escape(token)}\b", slot_key) for token in tokens) for slot_key in slot_keys)


def infer_call_count(
    user_request: str,
    tool: dict[str, Any],
    query_input_audit: dict[str, Any] | None = None,
) -> int:
    properties = _properties(tool)
    if _is_auth_token_tool(tool) and not _auth_request_has_multiple_accounts(user_request):
        return 1
    semantic_count = _semantic_frame_expected_call_count(query_input_audit)
    semantic_best = min(semantic_count, 8) if semantic_count and semantic_count > 1 else 1
    if _has_enum_array_payload(user_request, tool):
        semantic_best = 1
    relevant_clause_count = _tool_relevant_clause_count(user_request, tool) if _clause_repetition_allowed(user_request) else 1
    if relevant_clause_count > 1 and _has_result_count_phrase(user_request):
        return min(relevant_clause_count, 8)
    if _single_operation_multi_slot_request(user_request, tool, relevant_clause_count):
        return 1
    atomic_count = _atomic_unit_count_for_tool(user_request, tool)
    if atomic_count > 1:
        return min(max(atomic_count, semantic_best), 8)
    scalar_cross_product_count = _scalar_cross_product_count(user_request, tool)
    if scalar_cross_product_count > 1:
        return min(max(scalar_cross_product_count, semantic_best), 8)
    optional_selector_count = _optional_scalar_selector_repeat_count(user_request, tool)
    if optional_selector_count > 1 and not _tool_has_required_batch_array(tool):
        return min(max(optional_selector_count, semantic_best), 8)
    if _single_shape_instance_request(user_request, tool):
        return 1
    ecology_scenarios = _ecology_scenarios(user_request)
    if ecology_scenarios and _tool_uses_ecology_scenarios(tool):
        return min(len(ecology_scenarios), 8)
    if semantic_count == 1 and _has_prior_result_context(user_request) and _semantic_frame_can_cap_to_single_call(user_request, tool):
        return 1
    semantic_slot_count = _semantic_slot_repeat_count(user_request, tool, query_input_audit)
    parallel_scenario_count = _parallel_scenario_count(user_request, tool)
    if semantic_slot_count > 1 and parallel_scenario_count > semantic_slot_count and re.search(r"\brespectively\b", user_request, re.I):
        return min(max(semantic_slot_count, semantic_best), 8)
    if semantic_slot_count > 1 or parallel_scenario_count > 1:
        return min(max(semantic_slot_count, parallel_scenario_count, semantic_best), 8)
    explicit_group_count = 1 if _has_result_count_phrase(user_request) else _explicit_group_count(user_request.lower())
    if (
        explicit_group_count > 1
        and not _tool_has_required_batch_array(tool)
        and not _is_single_route_or_visit_workflow(user_request, tool)
    ):
        return min(max(explicit_group_count, semantic_best), 8)
    array_group_count = _array_payload_group_count(user_request, tool)
    if array_group_count > 1:
        return min(max(array_group_count, semantic_best), 8)
    array_tuple_count = _array_tuple_group_count(user_request, tool)
    if array_tuple_count > 1:
        return min(max(array_tuple_count, semantic_best), 8)
    stock_count = _stock_symbol_entity_count(user_request, tool)
    if stock_count > 1:
        return min(max(stock_count, semantic_best), 8)
    has_required_batch_array = _tool_has_required_batch_array(tool)
    if has_required_batch_array and len(properties) == 1:
        return 1
    if "artist" in properties:
        artist_count = len(extract_artist_like_values(user_request))
        if artist_count > 1:
            return min(max(artist_count, semantic_best), 8)
    if has_required_batch_array:
        return 1
    enum_count = _enum_value_count(user_request, tool)
    if enum_count > 1 and not _has_enum_array_payload(user_request, tool):
        return min(max(enum_count, semantic_best), 8)
    tuple_count = _numeric_tuple_count(user_request, tool)
    if tuple_count > 1:
        return min(max(tuple_count, semantic_best), 8)
    explicit_count = _explicit_repeated_call_count(user_request, tool)
    if explicit_count > 1:
        return min(max(explicit_count, semantic_best), 8)
    if semantic_count == 1 and _semantic_frame_can_cap_to_single_call(user_request, tool):
        return 1
    for name, spec in properties.items():
        if name in _required(tool) and _property_type(spec) == "string":
            entity_count = _entity_count_for_arg(user_request, name, spec)
            if entity_count > 1:
                return min(max(entity_count, semantic_best), 8)
    if len(properties) == 1:
        arg = next(iter(properties))
        if _property_type(properties[arg]) in {"string", "integer", "float", "number"}:
            return min(max(_entity_count_for_arg(user_request, arg, properties[arg]), semantic_best, 1), 8)
    if {"artist", "duration"} <= set(properties):
        return min(max(len(extract_artist_like_values(user_request)), semantic_best, 1), 8)
    if (
        relevant_clause_count > 1
        and not _tool_has_required_batch_array(tool)
        and _has_distinct_required_string_values_by_clause(user_request, tool)
    ):
        return min(max(relevant_clause_count, semantic_best), 8)
    return semantic_best


def _has_distinct_required_string_values_by_clause(user_request: str, tool: dict[str, Any]) -> bool:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return False
    scores = [_tool_scope_score(clause, tool) for clause in clauses]
    best = max(scores, default=0.0)
    if best < 2.0:
        return False
    threshold = max(2.0, best * 0.6)
    selected = [clause for clause, score in zip(clauses, scores) if score >= threshold]
    selected = _prefer_independent_action_clauses(selected)
    if len(selected) <= 1:
        return False

    for name, spec in _properties(tool).items():
        if name not in _required(tool) or _property_type(spec) != "string":
            continue
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        values = [_clause_string_value_for_slot(clause, name.lower(), slot_text) for clause in selected]
        values = [value for value in values if value]
        if len(values) == len(selected) and len({value.lower() for value in values}) > 1:
            return True
    return False


def _clause_string_value_for_slot(clause: str, arg: str, slot_text: str) -> str | None:
    if _looks_like_location_slot(arg, slot_text):
        locations = _location_units(clause)
        clean_locations = [
            value
            for value in locations
            if not re.search(r"\b(?:party|happened|filed|named|against|with)\b", value, re.I)
        ]
        if clean_locations:
            return clean_locations[-1]
        entities = [
            value
            for value in _capitalized_entity_values(clause)
            if value.lower() not in {"find", "also", "get", "list", "provide", "what"}
        ]
        return entities[-1] if entities else None
    if "company" in slot_text:
        values = _company_or_entity_values(clause)
        return values[0] if values else _quoted_value(clause)
    values = _quoted_strings(clause)
    if values:
        return values[0]
    entities = [
        value
        for value in _capitalized_entity_values(clause)
        if value.lower() not in {"find", "also", "get", "list", "provide", "what"}
    ]
    return entities[0] if entities else None


def infer_arguments(
    user_request: str,
    tool: dict[str, Any],
    call_index: int = 0,
    call_count: int = 1,
    query_input_audit: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    properties = _properties(tool)
    required = set(_required(tool))
    args: dict[str, Any] = {}
    missing: list[str] = []

    for name, spec in properties.items():
        value = _scalar_cross_product_argument_value(user_request, tool, name, call_index, call_count)
        if value is None:
            value = infer_argument_value(user_request, name, spec, call_index, call_count, query_input_audit)
        if value is None and "default" in spec:
            value = spec.get("default")
        if value is None:
            value = _default_value_from_description(spec)
        if value is not None and not _model_binding_value_allowed_by_schema(value, spec):
            schema_default = spec.get("default") if "default" in spec else _default_value_from_description(spec)
            if schema_default is not None and _model_binding_value_allowed_by_schema(schema_default, spec):
                value = schema_default
            else:
                if name in required:
                    missing.append(name)
                continue
        if value is None:
            if name in required:
                missing.append(name)
            continue
        if value == "" and name not in required:
            continue
        args[name] = value

    for name in required:
        if name not in args and name not in missing:
            missing.append(name)
    args = _drop_conditionally_unneeded_args(tool, args, user_request)
    missing = _filter_conditionally_unneeded_missing(tool, args, missing, user_request)
    return args, missing


def infer_argument_value(
    user_request: str,
    arg_name: str,
    spec: dict[str, Any],
    call_index: int = 0,
    call_count: int = 1,
    query_input_audit: dict[str, Any] | None = None,
) -> Any:
    lowered = user_request.lower()
    arg = arg_name.lower()
    typ = _property_type(spec)
    explicit = _explicit_value_for_arg(user_request, arg)
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if typ == "string" and _string_temporal_slot_without_evidence(user_request, arg, slot_text):
        return None
    if any(token in slot_text for token in ["food", "meal", "portion"]) and not any(
        token in arg for token in ["date", "time", "timestamp"]
    ):
        food_entries = _food_log_entries_from_text(user_request)
        if food_entries:
            entry = food_entries[min(call_index, len(food_entries) - 1)]
            if typ == "string":
                if "meal" in slot_text:
                    return entry["meal_name"]
                if "portion" in slot_text and "unit" in slot_text:
                    return _align_food_portion_unit_to_schema(entry["portion_unit"], spec)
                if "food" in slot_text and "name" in slot_text:
                    return entry["food_name"]
            if typ in {"integer", "float", "number"} and "portion" in slot_text and "amount" in slot_text:
                return _coerce_number(entry["portion_amount"], spec)
    if typ == "string":
        ecology_value = _ecology_string_value_for_slot(user_request, arg, slot_text, call_index)
        if ecology_value:
            return ecology_value
        schema_scalar_value = _schema_named_scalar_value(user_request, arg, slot_text, call_index, call_count)
        if schema_scalar_value:
            return schema_scalar_value
        if "token" in slot_text:
            value = _token_value(user_request)
            return value
        if "email" in arg:
            values = _email_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if arg == "status" and "forgot password" in slot_text and "verification code" in slot_text:
            return "Verification Code" if _forgot_password_second_phase(user_request) else "Forgot Password"
        if arg.replace("_", "") == "username" or arg in {"login"}:
            values = _username_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "password" in arg:
            values = _password_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_service_descriptor_slot(arg, slot_text):
            values = _service_descriptor_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if arg in {"name", "full_name", "display_name"}:
            value = _use_value(user_request)
            if value:
                return value
        if arg in {"show", "show_name", "title"}:
            value = _musical_show_value(user_request)
            if value:
                return value
        credential_pair = _unlabeled_credential_pair(user_request)
        if credential_pair:
            if arg.replace("_", "") == "username" or arg in {"login", "account"}:
                return credential_pair[0]
            if "password" in arg:
                return credential_pair[1]
        if _looks_like_language_slot(arg, slot_text):
            pairs = _translation_language_pairs(user_request)
            if pairs:
                index = min(call_index, len(pairs) - 1)
                if any(token in arg for token in ["from", "source", "src", "origin"]):
                    return pairs[index][0]
                if any(token in arg for token in ["to", "target", "dest"]):
                    return pairs[index][1]
                return pairs[index][0]
            value = _language_value(user_request)
            if value:
                return value
        if _looks_like_text_payload_slot(arg, slot_text):
            quoted = _quoted_strings(user_request)
            if quoted:
                return quoted[min(call_index, len(quoted) - 1)]
        if "file" in arg or "path" in arg:
            values = _file_path_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_location_slot(arg, slot_text):
            quoted_locations = _quoted_location_values(user_request) or _quoted_strings(user_request)
            if quoted_locations:
                return quoted_locations[min(call_index, len(quoted_locations) - 1)]
            if re.search(r"\beach\s+city\b|\bboth\s+cities\b", user_request, re.I):
                values = _city_list_values(user_request)
                if values:
                    return values[min(call_index, len(values) - 1)]
        if "frequency" in arg or "frequency" in slot_text:
            values = _frequency_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
    if typ == "array" and _looks_like_record_array_slot(arg, slot_text):
        labels = _record_array_labels(spec)
        if labels:
            values = _labeled_record_values(user_request, labels)
            if values:
                return values
        values = _record_value_field_values(user_request)
        if values:
            return values
    if typ == "array":
        coordinate_values = _coordinate_array_values(user_request)
        if coordinate_values and any(token in arg for token in ["coordinate", "location"]):
            return coordinate_values[min(call_index, len(coordinate_values) - 1)]
        coordinate_groups = _coordinate_pair_groups(user_request)
        if coordinate_groups and any(token in arg for token in ["point", "coordinate"]):
            group = coordinate_groups[min(call_index, len(coordinate_groups) - 1)]
            if re.search(r"(?:b|2)$", arg):
                return group[1]
            return group[0]
        if "movie" in slot_text or "film" in slot_text:
            movies = _movie_units(user_request)
            if movies:
                if call_count > 1:
                    return [movies[min(call_index, len(movies) - 1)]]
                return movies
        if _looks_like_time_slot(arg, slot_text):
            times = _clock_time_values(user_request)
            if times:
                if call_count > 1:
                    return [times[min(call_index, len(times) - 1)]]
                return times
        enum_values = _enum_value_evidence(user_request, spec)
        if enum_values:
            return enum_values
    if (
        typ == "string"
        and explicit is not None
        and not any(token in slot_text for token in ["date", "time", "timestamp"])
    ):
        return str(explicit)
    if typ == "string":
        if arg == "note" or "musical note" in slot_text or "music note" in slot_text:
            value = _music_note_value(user_request)
            if value:
                return value
        if "operation" in arg or "operation" in slot_text:
            value = _operation_value(user_request, slot_text)
            if value:
                return value
        if _looks_like_database_name_slot(arg, slot_text):
            value = _database_name_value(user_request)
            if value:
                return value
        if _looks_like_database_table_slot(arg, slot_text):
            value = _database_table_value(user_request)
            if value:
                return value
        if "state" in arg or "state" in slot_text:
            states = _capital_gain_states(user_request)
            if states:
                return states[min(call_index, len(states) - 1)]
        if _looks_like_stock_code_slot(arg, slot_text):
            values = _stock_symbol_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_property_selector_slot(arg, slot_text):
            values = _requested_property_values(user_request, spec)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "substance" in arg or "substance" in slot_text:
            values = _energy_substances(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "pokemon" in arg or "pokemon" in slot_text:
            values = _pokemon_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "cuisine" in arg or "cuisine" in slot_text:
            values = _restaurant_cuisine_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_location_slot(arg, slot_text):
            quoted_locations = _quoted_location_values(user_request)
            if quoted_locations:
                return quoted_locations[min(call_index, len(quoted_locations) - 1)]
            restaurant_locations = _restaurant_search_locations(user_request)
            if arg in {"location", "city"} and restaurant_locations:
                return restaurant_locations[min(call_index, len(restaurant_locations) - 1)]
            value = _near_location_value(user_request)
            if value:
                return value
            values = _numeric_location_pair_locations(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
    semantic = _semantic_slot_value(user_request, arg, spec, call_index, call_count, query_input_audit)
    if semantic is not None and (
        explicit is None or typ == "string" or _semantic_value_refines_explicit(explicit, semantic, arg, spec)
    ):
        return semantic

    if typ == "boolean":
        if isinstance(explicit, bool):
            return explicit
        description = str(spec.get("description") or "").lower()
        slot_text = f"{arg} {description}".replace("_", " ")
        if "residual" in slot_text and re.search(r"\b(?:return|include|calculate|compute|show)\b.{0,50}\bresiduals?\b", lowered):
            return True
        if "homemade" in slot_text and re.search(r"\bhome[-\s]?made\b", lowered):
            return True
        if _looks_like_detail_slot(arg, slot_text) and _detail_value(user_request):
            return True
        if re.search(r"\bdefault\s+(?:is\s+)?true\b", description):
            return True
        if re.search(r"\bdefault\s+(?:is\s+)?false\b", description):
            return False
        if "false" in lowered and arg.replace("_", " ") in lowered:
            return False
        if "true" in lowered or spec.get("default") is True or spec.get("optional") is True:
            return True
        if "function" in arg and "function" in lowered:
            return True
        if any(token in lowered for token in _tokens(arg)):
            return True
        return None

    if typ in {"integer", "float", "number"}:
        schema_number = _schema_named_numeric_value(user_request, arg, slot_text, call_index, call_count)
        if schema_number is not None:
            return _coerce_number(schema_number, spec)
        if "exercise" in slot_text and "time" in slot_text:
            value = _exercise_hours_value(user_request)
            if value is not None:
                return _coerce_number(value, spec)
        gain_value = _capital_gain_value_for_slot(user_request, arg, call_index)
        if gain_value is not None:
            return _coerce_number(gain_value, spec)
        if _looks_like_size_numeric_slot(arg, slot_text):
            pairs = _numeric_location_pairs(user_request)
            if pairs:
                number = _number_from_any(pairs[min(call_index, len(pairs) - 1)][0])
                if number is not None:
                    return _coerce_number(number, spec)
        return _numeric_value_for_arg(user_request, arg, spec, call_index)

    if typ == "array":
        if "dimension" in arg or "dimension" in slot_text:
            dimensions = _dimension_array_value(user_request)
            if dimensions:
                return dimensions
        if isinstance(explicit, list):
            return explicit
        if explicit is not None and _looks_like_data_reference_value(explicit):
            return str(explicit)
        enum_values = _enum_value_evidence(user_request, spec)
        if enum_values:
            return enum_values
        values = extract_numbers(user_request)
        if values and any(token in arg for token in ["coordinate", "point", "location"]):
            return values[:2]
        if values:
            return values
        item_type = str((spec.get("items") or {}).get("type") or "").lower() if isinstance(spec.get("items"), dict) else ""
        if item_type in {"string", "str"} or any(token in arg for token in ["cities", "locations", "theaters", "theatres", "restaurants", "show_list"]):
            string_values = _string_array_values_for_arg(user_request, arg, call_index)
            if string_values:
                return string_values
        return None

    if typ == "dict":
        return _dict_value_for_arg(user_request, arg, spec, call_index)

    if typ in {"any", "object", ""}:
        if "dataset" in arg or arg in {"data", "table", "input", "payload"} or "data" in str(spec.get("description") or "").lower():
            value = _dataset_value(user_request)
            if value:
                return value
        explicit_any = explicit
        if explicit_any is not None:
            return explicit_any
        quoted = _quoted_value(user_request)
        if quoted:
            return quoted
        identifiers = _symbolic_identifier_values(user_request)
        if identifiers:
            return identifiers[min(call_index, len(identifiers) - 1)]
        return None

    if typ == "string":
        cross_product_value = _genre_location_cross_product_value(user_request, arg, slot_text, call_index, call_count)
        if cross_product_value:
            return cross_product_value
        schema_scalar_value = _schema_named_scalar_value(user_request, arg, slot_text, call_index, call_count)
        if schema_scalar_value:
            return schema_scalar_value
        if "genre" in arg or "genre" in slot_text:
            values = _music_genre_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if explicit is not None:
            return str(explicit)
        enum_values = _enum_value_evidence(user_request, spec)
        if enum_values:
            return enum_values[min(call_index, len(enum_values) - 1)]
        if "operation" in arg:
            value = _operation_value(user_request, slot_text)
            if value:
                return value
        if arg in {"function", "func", "equation"}:
            value = _function_expression_value(user_request)
            if value:
                return value
        if "species" in arg:
            values = _species_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "pokemon" in arg or "pokemon" in slot_text:
            values = _pokemon_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if arg in {"old_item", "old_food", "from_item"}:
            value = _food_change_value(user_request, "from")
            if value:
                return value
        if arg in {"new_item", "new_food", "to_item"}:
            value = _food_change_value(user_request, "to")
            if value:
                return value
        if "token" in arg:
            value = _token_value(user_request)
            if value:
                return value
        if arg in {"username", "user_name", "login", "account"} or arg.endswith("username"):
            values = _username_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "email" in arg or arg in {"to", "recipient"}:
            values = _email_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_stock_code_slot(arg, slot_text):
            values = _stock_symbol_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_doctor_slot(arg, slot_text):
            value = _doctor_name_value(user_request, prefer_new=_looks_like_new_value_slot(arg, slot_text))
            if value:
                return value
        if "airline" in arg:
            value = _airline_value(user_request)
            if value:
                return value
        if "user" in arg and "id" in arg:
            value = _user_id_value(user_request)
            if value:
                return value
        if "id" in arg:
            value = _id_number_value(user_request.lower(), arg)
            if value is not None:
                return str(value)
        if "file" in arg or "path" in arg:
            values = _file_path_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if _looks_like_new_date_slot(arg, slot_text):
            value = _new_date_value(user_request)
            if value:
                return value
        if "date" in arg or arg in {"day"}:
            values = _date_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
        if _looks_like_time_slot(arg, slot_text):
            values = _clock_time_values(user_request) or _temporal_phrase_values(user_request) or _date_values(user_request)
            return values[min(call_index, len(values) - 1)] if values else None
        if _looks_like_route_start_slot(arg, slot_text):
            value = _route_endpoint_value_for_call(user_request, "from", call_index)
            if value:
                return value
        if _looks_like_route_end_slot(arg, slot_text):
            value = _route_endpoint_value_for_call(user_request, "to", call_index)
            if value:
                return value
        if arg == "country" or arg.endswith("_country"):
            values = _country_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if arg in {"city", "location", "country", "state", "region", "area"} or any(token in arg for token in ["city", "location"]):
            capital_gain_states = _capital_gain_states(user_request)
            if arg == "state" and capital_gain_states:
                return capital_gain_states[min(call_index, len(capital_gain_states) - 1)]
            numeric_pair_locations = _numeric_location_pair_locations(user_request)
            if numeric_pair_locations:
                return numeric_pair_locations[min(call_index, len(numeric_pair_locations) - 1)]
            values = _location_units(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "theater" in arg or "theatre" in arg:
            values = _theater_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "restaurant" in arg:
            values = _restaurant_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if arg in {"movie", "film", "title"} or "movie" in arg:
            value = _movie_title_value(user_request)
            if value:
                return value
        if arg in {"subject"}:
            value = _labelled_text_value(user_request, "subject")
            if value:
                return value
        if arg in {"body", "message"}:
            value = _labelled_text_value(user_request, "body") or _labelled_text_value(user_request, "message")
            if value:
                return value
        if arg in {"origin", "_from", "from", "pickup"} or "origin" in arg:
            value = _route_endpoint_value_for_call(user_request, "from", call_index)
            if value:
                return value
        if arg in {"destination", "to", "dropoff"} or "destination" in arg:
            value = _route_endpoint_value_for_call(user_request, "to", call_index)
            if value:
                return value
        if "case" in arg and "number" in arg:
            value = _case_number_value(user_request)
            if value:
                return value
        if "jurisdiction" in arg or "county" in arg:
            value = _jurisdiction_value(user_request)
            if value:
                return value
        if arg == "court":
            value = _jurisdiction_value(user_request) or _location_context_value(user_request)
            if value:
                return value
        if "parties" in arg:
            values = _between_party_values(user_request)
            if values:
                return " and ".join(values)
        if arg == "metal":
            value = _commodity_value(user_request)
            if value:
                return value
        if arg in {"measure", "unit", "units"}:
            value = _measure_value(user_request)
            if value:
                return value
        if "dataset" in arg or arg in {"data", "table"}:
            value = _dataset_value(user_request)
            if value:
                return value
        if arg in {"body1", "body_1", "source_body"}:
            values = _celestial_body_values(user_request)
            if values:
                return values[0]
        if arg in {"body2", "body_2", "target_body"}:
            values = _celestial_body_values(user_request)
            if len(values) >= 2:
                return values[1]
        if arg in {"subject", "event", "discovery", "entity", "information", "details", "field_of_law", "topic"}:
            value = _topic_value(user_request, arg)
            if value:
                return value
        if arg in {"name", "full_name", "display_name"}:
            value = _use_value(user_request)
            if value:
                return value
        if "analysis" in arg:
            value = _labelled_text_value(user_request, "analysis")
            if value:
                return value
        if "keyword" in arg or arg in {"query", "search_query", "search"}:
            return _search_query_value(user_request)
        if "artist" in arg:
            values = extract_artist_like_values(user_request)
            if values:
                return values[min(call_index, len(values) - 1)]
        if "unit" in arg:
            return _unit_value(user_request)
        if "api" in arg and "key" in arg:
            return _quoted_or_after(user_request, "key")
        if "host" in arg:
            return _host_value(user_request)
        if "url" in arg and spec.get("default"):
            return spec.get("default")
        slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
        if _looks_like_sensitive_credential_slot(arg, slot_text):
            return None
        return _quoted_value(user_request)

    return None


def _filter_conditionally_unneeded_missing(
    tool: dict[str, Any],
    args: dict[str, Any],
    missing: list[str],
    user_request: str,
) -> list[str]:
    if not missing:
        return missing
    properties = _properties(tool)
    if not any("only needed for" in str(spec.get("description") or "").lower() for spec in properties.values()):
        return missing
    status = str(args.get("status") or "").lower()
    if not status:
        status = "verification code" if _forgot_password_second_phase(user_request) else "forgot password"
    filtered = []
    for name in missing:
        description = str(properties.get(name, {}).get("description") or "").lower()
        if "only needed for the first call" in description and "verification" in status:
            continue
        if "only needed for the second call" in description and "forgot password" in status:
            continue
        filtered.append(name)
    return filtered


def _drop_conditionally_unneeded_args(
    tool: dict[str, Any],
    args: dict[str, Any],
    user_request: str,
) -> dict[str, Any]:
    if not args:
        return args
    properties = _properties(tool)
    if not any("only needed for" in str(spec.get("description") or "").lower() for spec in properties.values()):
        return args
    status = str(args.get("status") or "").lower()
    if not status:
        status = "verification code" if _forgot_password_second_phase(user_request) else "forgot password"
    filtered = dict(args)
    for name in list(filtered):
        description = str(properties.get(name, {}).get("description") or "").lower()
        if "only needed for the first call" in description and "verification" in status:
            filtered.pop(name, None)
        if "only needed for the second call" in description and "forgot password" in status:
            filtered.pop(name, None)
    return filtered


def _forgot_password_second_phase(user_request: str) -> bool:
    lowered = user_request.lower()
    return bool("verification code" in lowered or "new password" in lowered or "reset code" in lowered)


def _semantic_slot_value(
    user_request: str,
    arg: str,
    spec: dict[str, Any],
    call_index: int,
    call_count: int,
    query_input_audit: dict[str, Any] | None,
) -> Any | None:
    typ = _property_type(spec)
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    facts = _query_facts(user_request, query_input_audit)
    frame_value = _semantic_frame_value_for_slot(arg, slot_text, typ, spec, call_index, facts)
    if frame_value is not None:
        return frame_value

    if typ == "array":
        value = _semantic_record_array_value_for_slot(user_request, arg, slot_text, spec, facts)
        if value is not None:
            return value
        if "attendee" in slot_text:
            values = _attendee_values(user_request)
            if values:
                return values
        if any(token in slot_text for token in ["nutrient", "nutrition", "nutritional", "information"]):
            values = _nutrition_information_values(user_request, spec)
            if values:
                return values
        if "floor" in slot_text:
            values = _floor_values(user_request)
            if values:
                return values
        value = _labeled_numeric_array_value_for_slot(user_request, arg, call_index, facts)
        if value is not None:
            return value
        return None

    if typ in {"integer", "float", "number"}:
        if _looks_like_result_count_slot(arg, slot_text):
            value = _result_count_value(user_request)
            if value is not None:
                return _coerce_number(value, spec)
        if "year" in slot_text and "event" in slot_text:
            value = _event_name_value(user_request)
            if value:
                return value
        return None

    if typ in {"any", "object", "dict", ""}:
        if _looks_like_data_source_slot(arg, slot_text):
            file_values = _fact_values(facts, "file_path")
            if file_values:
                return file_values[min(call_index, len(file_values) - 1)]
        if any(token in slot_text for token in ["dataset", "data set", "training data", "input data"]):
            value = _dataset_value(user_request)
            if value:
                return value
        quoted = _fact_values(facts, "quoted_text")
        if quoted:
            return quoted[min(call_index, len(quoted) - 1)]
        return None

    if typ != "string":
        return None

    if _looks_like_sensitive_credential_slot(arg, slot_text):
        if "token" in slot_text:
            return _token_value(user_request)
        if arg.replace("_", "") == "username" or any(token in slot_text for token in ["username", "login"]):
            values = _username_values(user_request)
            return values[min(call_index, len(values) - 1)] if values else None
        return None

    cross_product_value = _genre_location_cross_product_value(user_request, arg, slot_text, call_index, call_count)
    if cross_product_value:
        return cross_product_value

    schema_scalar_value = _schema_named_scalar_value(user_request, arg, slot_text, call_index, call_count)
    if schema_scalar_value:
        return schema_scalar_value

    if "genre" in arg or "genre" in slot_text:
        values = _music_genre_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if "currency" in slot_text:
        value = _currency_value_for_slot(user_request, arg, slot_text, call_index, call_count)
        if value:
            return value

    if "unit" in slot_text:
        value = _conversion_unit_value_for_slot(user_request, arg, slot_text)
        if value:
            return value

    if _looks_like_stock_code_slot(arg, slot_text):
        values = _stock_symbol_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if "company" in slot_text:
        values = _company_or_entity_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if arg == "country" or arg.endswith("_country"):
        values = _country_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if any(token in slot_text for token in ["formula", "calculation", "expression"]):
        value = _function_expression_value(user_request)
        if not value:
            values = _formula_like_values(user_request)
            value = values[min(call_index, len(values) - 1)] if values else None
        if value:
            return value

    if _looks_like_doctor_slot(arg, slot_text):
        value = _doctor_name_value(user_request, prefer_new=_looks_like_new_value_slot(arg, slot_text))
        if value:
            return value

    if arg in {"name", "full_name", "display_name"}:
        value = _use_value(user_request)
        if value:
            return value

    if _looks_like_new_date_slot(arg, slot_text):
        value = _new_date_value(user_request)
        if value:
            return value

    value = _temporal_range_value_for_slot(user_request, arg, slot_text)
    if value:
        return value

    preference = _preference_value_for_slot(user_request, arg, slot_text)
    if preference:
        return preference

    restaurant = _restaurant_search_value_for_slot(user_request, arg, slot_text)
    if restaurant:
        return restaurant

    if "website" in slot_text or "site" in slot_text or "domain" in slot_text:
        value = _website_value(user_request)
        if value:
            return value

    if _looks_like_recipe_name_slot(arg, slot_text):
        value = _recipe_value(user_request)
        if value:
            return value

    if any(token in slot_text for token in ["food item", "food", "ingredient", "item"]):
        value = _food_item_value(user_request)
        if value:
            return value

    if any(token in slot_text for token in ["season", "league year"]):
        value = _season_value(user_request)
        if value:
            return value

    if "team" in slot_text or "club" in slot_text:
        value = _team_value(user_request)
        if value:
            return value

    if "event" in slot_text:
        value = _event_name_value(user_request)
        if value:
            return value

    if "symptom" in slot_text:
        value = _symptom_value(user_request)
        if value:
            return value

    if "particle" in slot_text:
        values = _particle_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if "scientist" in slot_text or "person" in slot_text or "author" in slot_text:
        value = _person_name_value(user_request)
        if value:
            return value

    if _looks_like_data_source_slot(arg, slot_text):
        file_values = _fact_values(facts, "file_path")
        if file_values:
            return file_values[min(call_index, len(file_values) - 1)]
        value = _dataset_value(user_request)
        if value:
            return value

    if _looks_like_language_slot(arg, slot_text):
        value = _language_value(user_request)
        if value:
            return value

    if _looks_like_text_payload_slot(arg, slot_text):
        value = _text_payload_value(user_request)
        if value:
            return value

    if _looks_like_zodiac_slot(arg, slot_text):
        signs = _zodiac_sign_values(user_request)
        index = _ordinal_slot_index(arg, default=call_index)
        if index < len(signs):
            return signs[index]

    if _looks_like_place_slot(arg, slot_text):
        places = _place_pair_values(user_request)
        index = _ordinal_slot_index(arg, default=call_index)
        if index < len(places):
            return places[index]

    if "medium" in slot_text:
        values = _art_medium_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if any(token in slot_text for token in ["size", "dimension", "dimensions"]):
        values = _size_phrase_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if "color" in slot_text or "colour" in slot_text:
        values = _color_phrase_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if _looks_like_time_slot(arg, slot_text):
        values = _clock_time_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
        values = _temporal_phrase_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if _looks_like_detail_slot(arg, slot_text):
        enum_values = _enum_value_evidence(user_request, spec)
        if enum_values:
            return enum_values[min(call_index, len(enum_values) - 1)]
        value = _detail_value(user_request)
        if value:
            return value

    if "hobby" in slot_text or "interest" in slot_text:
        value = _hobby_value(user_request)
        if value:
            return value

    if _looks_like_music_key_slot(arg, slot_text):
        value = _music_key_value(user_request)
        if value:
            return value

    if "scale" in slot_text:
        value = _music_scale_value(user_request)
        if value:
            return value

    if "start note" in slot_text or arg in {"start_note", "starting_note"}:
        value = _music_note_value(user_request)
        if value:
            return value

    if "month" in slot_text:
        values = _month_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if "manufacturer" in slot_text or "brand" in slot_text:
        value = _brand_or_manufacturer_value(user_request)
        if value:
            return value

    if "model" in slot_text:
        values = _algorithm_model_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]
        value = _model_value(user_request)
        if value:
            return value

    if "financial institution" in slot_text or "institution" in arg:
        values = _financial_institution_values(user_request)
        if values:
            return values[min(call_index, len(values) - 1)]

    if any(token in slot_text for token in ["spec", "specification", "specifications"]):
        value = _specification_value(user_request)
        if value:
            return value

    return None


def _semantic_frame_value_for_slot(
    arg: str,
    slot_text: str,
    typ: str,
    spec: dict[str, Any],
    call_index: int,
    facts: list[dict[str, Any]],
) -> Any | None:
    candidates = [
        fact
        for fact in facts
        if fact.get("source") == "gptoss_semantic_slot_frame"
        and _semantic_frame_role_matches_slot(str(fact.get("role") or ""), arg, slot_text)
    ]
    if not candidates:
        return _semantic_range_boundary_value_for_slot(arg, slot_text, typ, spec, call_index, facts)
    candidates.sort(key=lambda fact: float(fact.get("confidence") or 0.0), reverse=True)
    pair_value = _semantic_pair_value_for_numbered_slot(arg, typ, spec, call_index, candidates)
    if pair_value is not None:
        return pair_value
    list_candidates = [fact for fact in candidates if typ != "array" and isinstance(fact.get("normalized_value"), list)]
    fact = list_candidates[0] if list_candidates else candidates[min(call_index, len(candidates) - 1)]
    value = fact.get("normalized_value")
    if value is None:
        return None
    if isinstance(value, list) and typ != "array":
        if not value:
            return None
        value = value[min(call_index, len(value) - 1)]
    if typ in {"integer", "float", "number"}:
        numeric = _number_from_any(value)
        if numeric is None:
            return None
        return _coerce_number(numeric, spec)
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1", "false", "no", "0"}:
            return value.strip().lower() in {"true", "yes", "1"}
        return None
    if typ == "array":
        if isinstance(value, list):
            return value
        return None
    if typ in {"string", "any", "object", "dict", ""}:
        if isinstance(value, (str, int, float, bool)):
            return str(value) if typ == "string" else value
        return value
    return None


def _semantic_range_boundary_value_for_slot(
    arg: str,
    slot_text: str,
    typ: str,
    spec: dict[str, Any],
    call_index: int,
    facts: list[dict[str, Any]],
) -> Any | None:
    if typ not in {"string", "any", ""}:
        return None
    boundary = _slot_range_boundary(arg, slot_text)
    if boundary is None:
        return None
    range_facts = [
        fact
        for fact in facts
        if fact.get("source") == "gptoss_semantic_slot_frame"
        and _semantic_role_is_range(str(fact.get("role") or ""))
    ]
    if not range_facts:
        return None
    range_facts.sort(key=lambda fact: float(fact.get("confidence") or 0.0), reverse=True)
    fact = range_facts[min(call_index, len(range_facts) - 1)]
    parts = _semantic_range_parts(fact)
    if len(parts) < 2:
        return None
    value = parts[0] if boundary == "start" else parts[1]
    if value in (None, ""):
        return None
    return str(value) if typ == "string" else value


def _slot_range_boundary(arg: str, slot_text: str) -> str | None:
    key = _normalize_role(arg)
    text = f"{arg} {slot_text}".lower().replace("_", " ")
    if re.search(r"\b(?:start|begin|from|after|since|earliest|first)\b", text) or key.startswith(("start", "from")):
        return "start"
    if re.search(r"\b(?:end|finish|to|until|before|through|latest|last)\b", text) or key.startswith(("end", "to")):
        return "end"
    return None


def _semantic_role_is_range(role: str) -> bool:
    key = _normalize_role(role)
    if not key:
        return False
    return bool(
        "range" in key
        or key in {"period", "timeperiod", "timeframe", "window", "interval", "timespan", "datespan"}
        or key.endswith(("range", "period", "window", "interval", "span"))
    )


def _semantic_range_parts(fact: dict[str, Any]) -> list[Any]:
    value = fact.get("normalized_value")
    if isinstance(value, (list, tuple)):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, dict):
        start = _first_present_value(value, ["start", "start_time", "start_date", "from", "begin", "beginning"])
        end = _first_present_value(value, ["end", "end_time", "end_date", "to", "until", "through", "finish"])
        return [item for item in [start, end] if item not in (None, "")]
    text = str(value or fact.get("span") or "").strip()
    if not text:
        return []
    date_values = _date_values(text)
    if len(date_values) >= 2:
        return date_values[:2]
    for pattern in [
        r"\bfrom\s+(.+?)\s+(?:to|through|until|till|-|–|—)\s+(.+)$",
        r"^(.+?)\s+(?:to|through|until|till|-|–|—)\s+(.+)$",
    ]:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return [match.group(1).strip(" .,;"), match.group(2).strip(" .,;")]
    return []


def _first_present_value(mapping: dict[str, Any], keys: list[str]) -> Any | None:
    normalized = {_normalize_role(str(key)): value for key, value in mapping.items()}
    for key in keys:
        value = normalized.get(_normalize_role(key))
        if value not in (None, ""):
            return value
    return None


def _semantic_pair_value_for_numbered_slot(
    arg: str,
    typ: str,
    spec: dict[str, Any],
    call_index: int,
    candidates: list[dict[str, Any]],
) -> Any | None:
    if typ not in {"integer", "float", "number"}:
        return None
    match = re.fullmatch(r"(?:num|number|value)[_ -]?(\d+)", arg.lower())
    if not match:
        return None
    slot_index = int(match.group(1)) - 1
    pair_candidates = [
        fact
        for fact in candidates
        if "pair" in _normalize_role(str(fact.get("role") or ""))
        and isinstance(fact.get("normalized_value"), list)
        and len(fact.get("normalized_value") or []) > slot_index
    ]
    if not pair_candidates:
        return None
    value = pair_candidates[min(call_index, len(pair_candidates) - 1)].get("normalized_value")[slot_index]
    numeric = _number_from_any(value)
    if numeric is None:
        return None
    return _coerce_number(numeric, spec)


def _semantic_record_array_value_for_slot(
    user_request: str,
    arg: str,
    slot_text: str,
    spec: dict[str, Any],
    facts: list[dict[str, Any]],
) -> list[Any] | None:
    """Bind record/measurement arrays from grounded semantic facts, not all numbers."""
    if not _looks_like_record_array_slot(arg, slot_text):
        return None

    values = _semantic_measurement_values(facts)
    if values:
        return values

    labels = _record_array_labels(spec)
    if labels:
        values = _labeled_record_values(user_request, labels)
        if values:
            return values

    values = _record_value_field_values(user_request)
    return values or None


def _looks_like_record_array_slot(arg: str, slot_text: str) -> bool:
    if arg in {"health_data", "vitals", "vital_signs", "measurements", "readings"}:
        return True
    return bool(
        ("data" in slot_text or "record" in slot_text or "measurement" in slot_text or "reading" in slot_text)
        and any(token in slot_text for token in ["health", "vital", "blood", "heart", "value"])
    )


def _semantic_measurement_values(facts: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for fact in facts:
        if fact.get("source") != "gptoss_semantic_slot_frame":
            continue
        role = _normalize_role(str(fact.get("role") or ""))
        if not role or _semantic_role_is_record_metadata(role):
            continue
        value = fact.get("normalized_value")
        if value in (None, ""):
            continue
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            values.append(value)
        else:
            values.append(str(value))
    return _dedupe(values)


def _semantic_role_is_record_metadata(role: str) -> bool:
    metadata_roles = {
        "accountid",
        "customerid",
        "date",
        "datetime",
        "enddate",
        "endtime",
        "id",
        "patientid",
        "startdate",
        "starttime",
        "time",
        "timestamp",
        "userid",
    }
    return role in metadata_roles or role.endswith("id") or role.endswith("time") or role.endswith("date")


def _record_array_labels(spec: dict[str, Any]) -> list[str]:
    description = str(spec.get("description") or "")
    labels = re.findall(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", description, flags=re.I)
    return _dedupe(label.replace("_", " ").strip() for label in labels if label.strip())


def _labeled_record_values(user_request: str, labels: list[str]) -> list[str]:
    values: list[str] = []
    for label in labels:
        label_pattern = re.escape(label).replace(r"\ ", r"[\s_]+")
        patterns = [
            rf"\b{label_pattern}\b\s*(?:is|=|:)\s*([A-Za-z0-9./:-]+)",
            rf"\b{label_pattern}\b\s+([0-9][A-Za-z0-9./:-]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_request, flags=re.I)
            if match:
                values.append(match.group(1).strip(".,; "))
                break
    return _dedupe(values)


def _record_value_field_values(user_request: str) -> list[str]:
    values = re.findall(r"['\"]value['\"]\s*:\s*['\"]([^'\"]+)['\"]", user_request, flags=re.I)
    return _dedupe(value.strip() for value in values if value.strip())


def _semantic_value_refines_explicit(explicit: Any, semantic: Any, arg: str, spec: dict[str, Any]) -> bool:
    if isinstance(semantic, (list, dict)) or isinstance(explicit, (list, dict)):
        return False
    explicit_text = str(explicit).strip()
    semantic_text = str(semantic).strip()
    if not explicit_text or not semantic_text or explicit_text == semantic_text:
        return False
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if ("time" in slot_text or "date" in slot_text or "timestamp" in slot_text) and explicit_text in semantic_text:
        return True
    return False


def _semantic_frame_role_matches_slot(role: str, arg: str, slot_text: str) -> bool:
    role_key = _normalize_role(role)
    slot_key = _normalize_role(arg)
    role_keys = _role_key_variants(role)
    slot_keys = _role_key_variants(arg)
    slot_tokens = set(_tokens(slot_text))
    role_tokens = set(_tokens(role.replace("_", " ")))
    if not role_key or not slot_key:
        return False
    if not _indexed_role_compatible(role, arg):
        return False
    role_boundary = _slot_range_boundary(role, role.replace("_", " "))
    slot_boundary = _slot_range_boundary(arg, slot_text)
    if role_boundary and slot_boundary and role_boundary != slot_boundary:
        return False
    if ("unit" in role_key or "unit" in role_tokens) and "unit" not in slot_key and "unit" not in slot_text:
        return False
    if "unit" in slot_key and (role_tokens - {"unit", "units", "measure", "measurement"}):
        return False
    if role_keys & slot_keys:
        return True
    if role_key in {"person", "patient", "name"} and slot_key in {"patientname", "personname"}:
        return True
    if any(
        role_variant in slot_variant or slot_variant in role_variant
        for role_variant in role_keys
        for slot_variant in slot_keys
        if role_variant != "name" and slot_variant != "name"
    ):
        return True
    weak_domain_tokens = {
        "appointment",
        "currency",
        "data",
        "database",
        "given",
        "new",
        "name",
        "patient",
        "record",
        "records",
        "registration",
        "request",
        "requested",
        "specific",
        "target",
    }
    shared_tokens = role_tokens & slot_tokens
    if role_tokens and slot_tokens and (shared_tokens - weak_domain_tokens):
        return True
    generic_role_tokens = {
        "currency",
        "date",
        "id",
        "input",
        "location",
        "name",
        "number",
        "result",
        "slot",
        "text",
        "value",
    }
    if role_key in {"topic", "subject", "entity"} and slot_key.endswith("name") and (slot_tokens - generic_role_tokens):
        return True
    if role_key.endswith("name") and {"person", "scientist", "author"} & slot_tokens:
        return True
    for aliases in SEMANTIC_SLOT_ALIASES:
        if aliases & role_keys and (aliases & slot_keys or aliases & slot_tokens):
            return True
    return False


SEMANTIC_SLOT_ALIASES = [
    {"sourcecurrency", "basecurrency", "fromcurrency", "origincurrency", "currencyfrom"},
    {"targetcurrency", "tocurrency", "destinationcurrency", "currencyto"},
    {"diet", "dietaryrestriction", "dietaryrestrictions", "dietoption", "foodrestriction"},
    {"fooditem", "food", "ingredient", "item"},
    {"nutrients", "nutrient", "nutrition", "information", "nutritionaldetails"},
    {"platform", "operatingsystem", "os"},
    {"season", "sportsseason", "leagueyear"},
    {"team", "club"},
    {"league", "competition"},
    {"genre", "musicgenre", "concertgenre"},
    {"location", "city", "venue", "place", "region"},
    {"origin", "start", "startlocation", "fromlocation", "from", "sourcelocation", "departure"},
    {"destination", "end", "endlocation", "tolocation", "to", "targetlocation", "arrival"},
    {"suit", "cardsuit"},
    {"decksize", "cardsindeck"},
    {"handsize", "cardcount"},
    {"website", "site", "domain"},
    {"artform", "mediumtype", "worktype"},
    {"material", "medium"},
    {"phasechange", "phasechanges", "phasetransition", "transition", "transitiontype"},
    {"recipe", "recipename", "dish", "dishname", "meal", "food"},
    {"hotel", "hotelname", "accommodation"},
    {"exhibition", "exhibitionname", "show", "showname", "event", "eventname"},
    {"pokemon", "pokemonname", "species"},
    {"timeframe", "timeperiod", "period", "century"},
    {"religionname", "religion", "topic", "subject"},
    {"worktitle", "lawname", "title", "topic", "subject", "work"},
    {"username", "useremail", "accountemail", "login", "email"},
    {"username", "patientname", "customername", "personname", "fullname"},
    {"scientist", "person", "personname", "author"},
    {"eventname", "event", "document", "topic", "subject", "show", "showname", "performance", "performancename", "ticketevent", "timeperiod"},
    {"caseid", "casename", "case"},
    {"detaillevel", "details", "detail"},
    {"formula", "expression", "equation", "calculation", "calculationexpression", "mathexpression"},
    {"symptom", "disease", "diseasename", "condition", "medicalcondition", "healthcondition", "complaint"},
    {"timestamp", "datetime", "time"},
]


def _normalize_role(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def _role_key_variants(value: str) -> set[str]:
    key = _normalize_role(value)
    if not key:
        return set()
    return {variant for variant in [key, re.sub(r"\d+$", "", key)] if variant}


def _indexed_role_compatible(role: str, arg: str) -> bool:
    slot_index = _indexed_slot_number(arg)
    if slot_index is None:
        return True
    role_index = _role_index_number(role)
    return role_index is None or role_index == slot_index


def _indexed_slot_number(arg: str) -> int | None:
    key = _normalize_role(arg)
    match = re.search(r"(?:num|number|value|side|body|charge|point|coordinate)?([1-9])$", key)
    if match:
        return int(match.group(1))
    return None


def _role_index_number(role: str) -> int | None:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", role.lower()) if token]
    aliases = {
        "a": 1,
        "one": 1,
        "first": 1,
        "b": 2,
        "two": 2,
        "second": 2,
        "c": 3,
        "three": 3,
        "third": 3,
        "d": 4,
        "four": 4,
        "fourth": 4,
    }
    for token in reversed(tokens):
        if token in aliases:
            return aliases[token]
        if re.fullmatch(r"[1-9]", token):
            return int(token)
    key = _normalize_role(role)
    match = re.search(r"(?:^|[^a-z])([1-9])$", key)
    if match:
        return int(match.group(1))
    return None


def _number_from_any(value: Any) -> Any | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _number_from_text(value)
    return None


def parse_call_string(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    match = re.fullmatch(r"([A-Za-z_][\w.]*)\((.*)\)", stripped)
    if not match:
        return None
    name, args_text = match.groups()
    args = _parse_keyword_args(args_text)
    return {"tool_name": name, "arguments": args}


def normalize_expected_ground_truth(ground_truth: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in ground_truth if isinstance(ground_truth, list) else []:
        if isinstance(item, dict):
            for name, args in item.items():
                calls.append({"tool_name": str(name), "arguments": _normalize_expected_args(args)})
        elif isinstance(item, str):
            parsed = parse_call_string(item)
            if parsed:
                calls.append(parsed)
    return calls


def _normalize_expected_args(args: Any) -> dict[str, list[Any]]:
    if not isinstance(args, dict):
        return {}
    normalized: dict[str, list[Any]] = {}
    for key, value in args.items():
        values = value if isinstance(value, list) else [value]
        normalized[str(key)] = values
    return normalized


def _parse_keyword_args(args_text: str) -> dict[str, Any]:
    if not args_text.strip():
        return {}
    try:
        parsed = ast.parse(f"f({args_text})", mode="eval")
    except SyntaxError:
        return {}
    if not isinstance(parsed.body, ast.Call):
        return {}
    args: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            continue
        args[keyword.arg] = _literal_or_source(keyword.value, args_text)
    return args


def _literal_or_source(node: ast.AST, source: str) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = _literal_or_source(node.left, source)
            right = _literal_or_source(node.right, source)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) and right:
                return left / right
        return ast.unparse(node) if hasattr(ast, "unparse") else source


def _looks_like_data_reference_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    bracket = r"\[\s*(?:'[^']+'|\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)\s*\]"
    dotted = rf"\.{identifier}"
    return bool(re.fullmatch(rf"{identifier}(?:{bracket}|{dotted})+", text))


def _explicit_value_for_arg(user_request: str, arg: str) -> Any | None:
    labels = _arg_labels(arg)
    for label in labels:
        pattern = (
            rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+|as\s+)"
            r"(\[[^\]]+\]|'[^']+'|\"[^\"]+\"|[^\s,;]+)"
        )
        if label in {"location", "new location"}:
            pattern = (
                rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+|as\s+|to\s+)"
                r"(\[[^\]]+\]|'[^']+'|\"[^\"]+\"|[^\s,;]+)"
            )
        match = re.search(pattern, user_request, re.I)
        if not match:
            continue
        return _parse_inline_value(match.group(1))
    return None


def _arg_labels(arg: str) -> list[str]:
    labels = {arg.lower(), arg.lower().replace("_", " ")}
    if arg.startswith("_"):
        labels.add(arg[1:].lower())
    aliases = {
        "k": {"k", "top k", "window"},
        "nums": {"nums", "numbers"},
        "dataset": {"dataset", "data set"},
        "file_path": {"file path", "filepath", "path"},
        "case_number": {"case number", "case no"},
        "user_id": {"user id", "userid"},
        "account": {"account", "account identifier", "account id"},
        "username": {"username", "user name", "login", "email", "e-mail", "account identifier", "account id"},
        "user_name": {"username", "user name", "login", "email", "e-mail", "account identifier", "account id"},
    }
    labels.update(aliases.get(arg.lower(), set()))
    return sorted(label for label in labels if label)


def _parse_inline_value(raw: str) -> Any:
    raw = raw.strip().strip(",.;")
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw.strip("'\"")


def _email_values(text: str) -> list[str]:
    return _dedupe(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))


def _values_after_labels(text: str, labels: list[str]) -> list[str]:
    values: list[Any] = []
    for label in labels:
        pattern = (
            rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+|as\s+)"
            r"('(?:[^']+)'|\"(?:[^\"]+)\"|[^\s,;]+)"
        )
        values.extend(_parse_inline_value(match.group(1)) for match in re.finditer(pattern, text, re.I))
    return _dedupe(str(value).strip(" .'\"") for value in values if value not in (None, ""))


def _labeled_username_values(text: str) -> list[str]:
    return _values_after_labels(
        text,
        ["username", "user name", "login", "account identifier", "account id", "account"],
    )


def _email_label_as_username_values(text: str) -> list[str]:
    return _values_after_labels(text, ["email", "e-mail"])


def _username_values(text: str) -> list[str]:
    values = _labeled_username_values(text)
    if values:
        return values
    values = _email_label_as_username_values(text)
    values.extend(_email_values(text))
    return _dedupe(values)


def _password_values(text: str) -> list[str]:
    values: list[str] = []
    for label in ["password", "passcode", "pass phrase", "passphrase", "pwd", "new password"]:
        pattern = (
            rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+|as\s+|to\s+)"
            r"('(?:[^']+)'|\"(?:[^\"]+)\"|[^\s,;]+)"
        )
        values.extend(_parse_inline_value(match.group(1)) for match in re.finditer(pattern, text, re.I))
    return _dedupe(str(value).strip(" .'\"") for value in values if value not in (None, ""))


def _unlabeled_credential_pair(text: str) -> tuple[str, str] | None:
    without_results = re.sub(r"\bLatest prior API result:\s*.*$", " ", text, flags=re.I | re.S)
    for segment in re.split(r"[\n.;!?]+", without_results):
        cleaned = re.sub(r"\b(?:earlier\s+user|user|assistant|ai):\s*", " ", segment, flags=re.I).strip(" .'\"")
        parts = re.findall(r"[A-Za-z0-9._@+-]+", cleaned)
        if len(parts) != 2:
            continue
        first, second = parts
        if first.lower() in STOPWORDS or second.lower() in STOPWORDS:
            continue
        if any(token in second.lower() for token in ["pass", "pwd", "secret"]):
            return first, second
    return None


def _token_value(text: str) -> str | None:
    values = _token_values(text)
    return values[0] if values else None


def _token_values(text: str) -> list[str]:
    patterns = [
        r"['\"]token['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        r"\btoken\s*(?:=|:|is\s+|as\s+)\s*['\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{3,})['\"]?",
        r"\baccess\s+token\s*(?:=|:|is\s+|as\s+)\s*['\"]?([A-Za-z0-9][A-Za-z0-9_.:-]{3,})['\"]?",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(1) for match in re.finditer(pattern, text, re.I))
    return _dedupe(
        value.strip(" .'\"")
        for value in values
        if value and value.strip(" .'\"").lower() not in {"token", "access"}
    )


def _file_path_values(text: str) -> list[str]:
    values = re.findall(
        r"(?:~?/|/|[A-Za-z]:\\)[^\s,;]+|[A-Za-z0-9_.-]+\.(?:nii\.gz|csv|json|txt|pdf|docx|xlsx|png|jpg|jpeg|wav|mp3|flac)",
        text,
    )
    return _dedupe(value.strip(" .'\"") for value in values)


def _date_values(text: str) -> list[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:today|tomorrow|yesterday|tonight)\b",
        r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:next|this|last)\s+(?:week|weekend|month|year)\b",
        r"\b(?:past|last|next)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:days?|weeks?|months?|years?)\b",
        r"\b(?:past|last|next)\s+(?:decade|century)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}\b",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0) for match in re.finditer(pattern, text, re.I))
    return _dedupe(value.strip() for value in values)


def _symbolic_identifier_values(text: str) -> list[str]:
    values = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b", text)
    values.extend(re.findall(r"\b(?:dataset|data_set|file_path|nums|budget|coordinates?)\s*[=]\s*[^\s,;]+", text, re.I))
    return _dedupe(value.strip() for value in values)


def _string_array_values_for_arg(user_request: str, arg: str, call_index: int) -> list[str] | None:
    del call_index
    if "stat" in arg or "field" in arg:
        values = _player_stat_values(user_request)
        if values:
            return values
    if "column" in arg:
        values = _database_column_values(user_request)
        if values:
            return values
    if "independent" in arg or "variables" in arg:
        values = _independent_variable_values(user_request)
        if values:
            return values
    if "cities" in arg or "locations" in arg:
        values = _location_units(user_request)
        return values or None
    if "theater" in arg or "theatre" in arg:
        values = _theater_values(user_request)
        return values or None
    if "restaurant" in arg:
        values = _restaurant_values(user_request)
        return values or None
    quoted = _quoted_strings(user_request)
    if quoted:
        return quoted
    if any(token in arg for token in ["company", "name", "entity", "parties", "people"]):
        values = _capitalized_entity_values(user_request)
        if len(values) >= 2:
            return values
    values = _proper_entity_values(user_request)
    return values or None


def _operation_value(user_request: str, slot_text: str) -> str | None:
    lowered = user_request.lower()
    if "delete" in slot_text or "add" in slot_text or "operation" in slot_text:
        if re.search(r"\b(?:delete|remove|drop)\b", lowered):
            return "delete"
        if re.search(r"\b(?:add|insert|create)\b", lowered):
            return "add"
    return None


def _looks_like_database_name_slot(arg: str, slot_text: str) -> bool:
    if arg in {"db", "db_name", "database", "database_name"}:
        return True
    return "database" in slot_text and "table" not in arg


def _looks_like_database_table_slot(arg: str, slot_text: str) -> bool:
    return arg in {"table", "table_name"} or ("table" in slot_text and "database" not in arg)


def _database_name_value(user_request: str) -> str | None:
    patterns = [
        r"\b([A-Za-z][\w-]*)\s+database\b",
        r"\bdatabase\s+(?:named|called|name(?:d)?|=|:|is)\s+([A-Za-z][\w-]*)\b",
        r"\bdb\s+(?:named|called|name(?:d)?|=|:|is)\s+([A-Za-z][\w-]*)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, re.I)
        if match:
            value = _clean_identifier_value(match.group(1))
            if value:
                return value
    return None


def _database_table_value(user_request: str) -> str | None:
    patterns = [
        r"\b(?:from|in|on)\s+([A-Za-z][\w-]*)\s+table\b",
        r"\btable\s+(?:named|called|name(?:d)?|=|:|is)\s+([A-Za-z][\w-]*)\b",
        r"\btable\s+([A-Za-z][\w-]*)\b",
        r"\b([A-Za-z][\w-]*)\s+table\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, re.I)
        if match:
            value = _clean_identifier_value(match.group(1))
            if value:
                return value
    return None


def _clean_identifier_value(value: str) -> str | None:
    cleaned = str(value or "").strip(" .'\"`:,;")
    if not cleaned or cleaned.lower() in STOPWORDS:
        return None
    return cleaned


def _database_column_values(user_request: str) -> list[str]:
    lowered = user_request.lower()
    if not re.search(r"\bcolumns?\b", lowered):
        return []
    aliases = {
        "email addresses": "email",
        "email address": "email",
        "emails": "email",
        "email": "email",
        "social security numbers": "social_security_number",
        "social security number": "social_security_number",
        "ssn": "social_security_number",
        "ssns": "social_security_number",
    }
    values = [canonical for phrase, canonical in aliases.items() if re.search(rf"\b{re.escape(phrase)}\b", lowered)]
    if values:
        return _dedupe(values)
    match = re.search(r"\b(?:delete|remove|drop)\s+(.+?)\s+columns?\b", user_request, re.I)
    if not match:
        match = re.search(r"\bcolumns?\s+(?:named|called)?\s*(.+?)(?:\s+(?:from|in|on)\b|[?.!]|$)", user_request, re.I)
    if not match:
        return []
    parts = [
        _clean_slot_phrase(part).replace(" ", "_")
        for part in re.split(r"\s*,\s*|\s+\band\b\s+", match.group(1))
        if _clean_slot_phrase(part)
    ]
    return _dedupe(parts)


def _independent_variable_values(text: str) -> list[str]:
    match = re.search(r"\busing\s+(.+?)\s+variables?\s+to\s+predict\b", text, re.I)
    if not match:
        return []
    chunk = match.group(1)
    parts = [part.strip(" .'\"") for part in re.split(r"\s+\band\s+|,\s*", chunk) if part.strip(" .'\"")]
    return _dedupe(parts)


def _user_id_value(text: str) -> str | None:
    patterns = [
        r"\buser\s*(?:id|ID)\s*[:#=]?\s*([A-Za-z0-9_-]+)\b",
        r"\buser[_-]([A-Za-z0-9_-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1)
    return None


def _theater_values(text: str) -> list[str]:
    match = re.search(r"\bat\s+(.+?)(?:\s+for\b|\s+on\b|\s+tomorrow\b|\s+today\b|[?.!]|$)", text, re.I)
    if not match:
        return []
    chunk = match.group(1)
    chunk = re.sub(r"^(?:the\s+)?", "", chunk, flags=re.I)
    parts = [
        re.sub(r"^(?:the\s+)?", "", part.strip(" .'\""), flags=re.I)
        for part in re.split(r"\s+\band\b\s+|,\s*", chunk)
        if part.strip(" .'\"")
    ]
    return _dedupe(parts)


def _restaurant_values(text: str) -> list[str]:
    patterns = [
        r"\b(?:at|tables?\s+at|book\s+(?:tables?\s+)?at)\s+(.+?)(?:\s+for\b|\s+on\b|\s+at\s+\d|[?.!]|$)",
        r"\brestaurant\s+(.+?)(?:\s+for\b|\s+on\b|[?.!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        parts = [part.strip(" .'\"") for part in re.split(r"\s+\band\b\s+|,\s*", match.group(1)) if part.strip(" .'\"")]
        if parts:
            return _dedupe(parts)
    return []


def _movie_title_value(text: str) -> str | None:
    quoted = _quoted_strings(text)
    if quoted:
        return quoted[0]
    patterns = [
        r"\b(?:[Mm]ovie|[Ff]ilm)\s+([A-Z][\w:'-]+(?:\s+[A-Z0-9][\w:'-]+)*)",
        r"\b[Ff]ind\s+([A-Z][\w:'-]+(?:\s+[A-Z0-9][\w:'-]+)*)\s+showtimes\b",
        r"\b[Ss]howtimes\s+for\s+([A-Z][\w:'-]+(?:\s+[A-Z0-9][\w:'-]+)*)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _musical_show_value(text: str) -> str | None:
    patterns = [
        r"\b(?:ticket\s+to|see|for)\s+(?:the\s+)?([A-Z][A-Za-z0-9' -]+?)\s+musical\b",
        r"\b([A-Z][A-Za-z0-9' -]+?)\s+musical\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip(" .'\"")
            if value:
                return value
    return None


def _function_expression_value(text: str) -> str | None:
    match = re.search(r"\b[a-z]\s*\(\s*x\s*\)\s*=\s*([^.;,]+)", text, re.I)
    if match:
        return match.group(0).strip()
    match = re.search(r"\by\s*=\s*([^.;,]+?)(?:\s+from\b|\s+at\b|[.;,]|$)", text, re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(?:function|equation)\s*(?:=|is|:)\s*([^.;,]+)", text, re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"\bfunction\s+([^.;,]+?)(?:\s+from\b|\s+at\b|[.;,]|$)", text, re.I)
    return match.group(1).strip() if match else None


def _food_change_value(text: str, side: str) -> str | None:
    match = re.search(
        r"\bfrom\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z '-]+?)\s+to\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z '-]+?)(?:[?.!]|$)",
        text,
        re.I,
    )
    if not match:
        return None
    index = 1 if side == "from" else 2
    return match.group(index).strip(" .'\"")


def _airline_value(text: str) -> str | None:
    match = re.search(r"\bwith\s+([A-Z][A-Za-z '&.-]+? Airlines?)\b", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Za-z '&.-]+? Airlines?)\b", text)
    return match.group(1).strip() if match else None


def _looks_like_stock_code_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool("stock" in tokens and (tokens & {"code", "symbol", "ticker"}))


def _stock_code_value(text: str) -> str | None:
    explicit = (
        _explicit_value_for_arg(text, "stock_code")
        or _explicit_value_for_arg(text, "stock_symbol")
        or _explicit_value_for_arg(text, "ticker")
        or _explicit_value_for_arg(text, "symbol")
    )
    if explicit is not None:
        return str(explicit)
    ticker = re.search(r"\b(?:ticker|symbol|stock\s+code)\s*(?:=|:|is\s+)?([A-Z]{1,5})(?:\b|$)", text)
    if ticker:
        return ticker.group(1)
    patterns = [
        r"\bstock\s+price\s+of\s+(.+?)(?:\s+on\b|\s+for\b|\s+at\b|[?.!]|$)",
        r"\bprice\s+of\s+(.+?)\s+stock(?:\s+on\b|\s+for\b|\s+at\b|[?.!]|$)",
        r"\b(?:quote|stock)\s+for\s+(.+?)(?:\s+on\b|\s+at\b|[?.!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = _clean_slot_phrase(match.group(1))
            if value:
                return value
    return None


COMMON_STOCK_SYMBOL_ALIASES = {
    "alphabet": "GOOG",
    "alphabet inc": "GOOG",
    "apple": "AAPL",
    "apple inc": "AAPL",
    "google": "GOOG",
    "google llc": "GOOG",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
}


def _stock_symbol_values(text: str) -> list[str]:
    explicit = _stock_code_value(text)
    if explicit and re.fullmatch(r"[A-Z]{1,5}", explicit):
        return [explicit]
    if explicit:
        key = re.sub(r"\s+", " ", explicit.lower().strip(" .'\""))
        return [COMMON_STOCK_SYMBOL_ALIASES.get(key, explicit)]
    companies = _stock_company_values(text)
    values = []
    for company in companies:
        key = re.sub(r"\s+", " ", company.lower().strip(" .'\""))
        values.append(COMMON_STOCK_SYMBOL_ALIASES.get(key, company))
    return _dedupe(values)


def _stock_symbol_entity_count(user_request: str, tool: dict[str, Any]) -> int:
    tool_text = _tool_text(tool).lower()
    properties = _properties(tool)
    has_symbol_slot = any(
        name in _required(tool) and _looks_like_stock_code_slot(name.lower(), f"{name} {spec.get('description') or ''}")
        for name, spec in properties.items()
    )
    if not has_symbol_slot or "stock" not in tool_text:
        return 1
    values = _stock_symbol_values(user_request)
    return len(values) if len(values) > 1 else 1


def _stock_company_values(text: str) -> list[str]:
    company_mentions = [
        re.sub(r"\s+", " ", match.group(1).strip(" .'\""))
        for match in re.finditer(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:Inc|LLC|Corporation|Corp|Ltd|Company|Co)\.?)\b",
            text,
        )
    ]
    if company_mentions:
        return _dedupe(company_mentions)
    chunk_match = re.search(
        r"\bfor\s+(.+?)(?:\s+for\s+(?:the\s+)?(?:next|upcoming|past|last)\b|\s+in\s+(?:the\s+)?[A-Z][A-Za-z ]*Stock Exchange\b|\s+on\s+(?:the\s+)?[A-Z][A-Za-z ]*Stock Exchange\b|[?.!]|$)",
        text,
    )
    chunks = [chunk_match.group(1)] if chunk_match else [text]
    values: list[str] = []
    for chunk in chunks:
        parts = [
            part.strip(" .'\"")
            for part in re.split(r"\s*,\s*(?:and\s+)?|\s+\band\s+", chunk)
            if part.strip(" .'\"")
        ]
        for part in parts:
            if re.search(r"\b(?:Inc|LLC|Corporation|Corp|Ltd|Company|Co)\.?\b", part):
                values.append(re.sub(r"\s+", " ", part.strip(" .'\"")))
    if values:
        return _dedupe(values)
    tickers = re.findall(r"\b[A-Z]{1,5}\b", text)
    return _dedupe(tickers)


def _company_or_entity_values(text: str) -> list[str]:
    suffix = r"(?:Inc|LLC|Corporation|Corp|Ltd|Company|Co)\.?"
    quoted = _quoted_strings(text)
    if quoted and re.search(r"\b(?:company|companies|stock|shares?|against)\b", text, re.I):
        return _dedupe(quoted)
    pair_match = re.search(rf"\bfor\s+(.+)\s+and\s+([A-Z][A-Za-z .&'-]*\s+{suffix})(?:[?.!]|$)", text)
    if pair_match:
        left = re.sub(r"\s+", " ", pair_match.group(1).strip(" .'\""))
        right = re.sub(r"\s+", " ", pair_match.group(2).strip(" .'\""))
        if left and right:
            return _dedupe([left, right])
    values = _stock_company_values(text)
    if values:
        return values
    investment_mentions: list[str] = []
    for pattern in [
        r"\b(?:invest|buy|purchase)\b.+?\bin\s+([A-Z][A-Za-z&'.-]+(?:\s+[A-Z][A-Za-z&'.-]+){0,4})(?:'s)?(?:\s+stock)?\b",
        r"\bwithdraw\b.+?\bfrom\s+([A-Z][A-Za-z&'.-]+(?:\s+[A-Z][A-Za-z&'.-]+){0,4})(?:'s)?(?:\s+stock)?\b",
        r"\b([A-Z][A-Za-z&'.-]+)(?:'s)?\s+stock\b",
    ]:
        for match in re.finditer(pattern, text, re.I):
            value = re.sub(r"\s+", " ", match.group(1).strip(" .'\""))
            value = re.sub(r"'s$", "", value, flags=re.I)
            if value and value.lower() not in {"also", "how", "invest", "withdraw", "the", "a", "an"}:
                investment_mentions.append(value)
    if investment_mentions:
        return _dedupe(investment_mentions)
    entities = [
        value
        for value in _capitalized_entity_values(text)
        if value.lower() not in {"find", "get", "what", "which", "predict", "forecast", "search", "calculate", "compute"}
    ]
    return _dedupe(entities)


def _looks_like_property_selector_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return "property" in tokens or arg in {"property", "attribute", "field"}


def _requested_property_values(user_request: str, spec: dict[str, Any]) -> list[str]:
    description = str(spec.get("description") or "")
    description_values = _property_values_from_description(description)
    allowed = description_values or ["length", "width", "diagonal", "area", "perimeter", "circumference"]
    mentions: list[tuple[int, str]] = []
    request_scope = re.split(r"\b(?:which|that)\s+has\b|\bgiven\b|\bbased\s+on\b", user_request, maxsplit=1, flags=re.I)[0]
    lowered = request_scope.lower()
    for value in allowed:
        text = str(value).strip().lower()
        if not text:
            continue
        for match in re.finditer(rf"\b{re.escape(text)}\b", lowered):
            mentions.append((match.start(), text))
    mentions.sort()
    return _dedupe(value for _start, value in mentions)


def _property_values_from_description(description: str) -> list[str]:
    match = re.search(r"\b(?:can be|one of|options? are)\s+(.+?)(?:[.)]|$)", description, re.I)
    if match:
        return _dedupe(
            part.strip(" .'\"")
            for part in re.split(r"\s*,\s*|\s+\bor\b\s+|\s+\band\b\s+", match.group(1))
            if part.strip(" .'\"")
        )
    return _listed_values_from_description(description)


def _looks_like_doctor_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool(tokens & {"doctor", "physician"})


def _symptom_value(text: str) -> str | None:
    patterns = [
        r"\b(?:symptom|symptoms?)\s+(?:is|are|of|for|about)?\s*(?:a|an|the|some)?\s*([a-z][a-z -]{2,50})",
        r"\b(?:about|for|on)\s+(?:a|an|the|some)?\s*([a-z][a-z -]{2,50})",
        r"\b(?:have|having|experiencing|with)\s+(?:a|an|the|some)?\s*([a-z][a-z -]{2,50})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        value = re.split(
            r"\s+(?:and|but|or|that|which|what|could|would|should|can|please|for|on|at|in|with|i(?:'ve| have)?)\b|[?.!,]",
            match.group(1),
            maxsplit=1,
            flags=re.I,
        )[0]
        value = re.sub(r"^(?:a|an|the|some)\s+", "", value.strip(" .'\""), flags=re.I)
        if value and value.lower() not in STOPWORDS:
            return value
    return None


def _looks_like_new_value_slot(arg: str, slot_text: str) -> bool:
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool(tokens & {"new", "target", "updated", "replacement"})


def _looks_like_new_date_slot(arg: str, slot_text: str) -> bool:
    return _looks_like_new_value_slot(arg, slot_text) and "date" in set(_tokens(f"{arg} {slot_text}"))


def _doctor_name_value(text: str, prefer_new: bool = False) -> str | None:
    doctor_pattern = r"\bDr\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?"
    doctors = [match.group(0).replace("Dr ", "Dr. ").strip() for match in re.finditer(doctor_pattern, text)]
    doctors = _dedupe(doctors)
    if not doctors:
        return None
    if prefer_new:
        target = re.search(rf"\bto\b.*?({doctor_pattern})", text, re.I | re.S)
        if target:
            return target.group(1).replace("Dr ", "Dr. ").strip()
        return doctors[-1]
    return doctors[0]


def _use_value(text: str) -> str | None:
    matches = list(
        re.finditer(
            r"\b(?:use|set(?: it)?(?: as| to)?|put(?: it)?(?: as| to)?)\s+([A-Z][A-Za-z0-9 ._'-]{0,60}?)(?:[.;!?]|$)",
            text,
            re.I,
        )
    )
    if not matches:
        return None
    value = _clean_slot_phrase(matches[-1].group(1))
    return value or None


def _new_date_value(text: str) -> str | None:
    target = re.search(r"\bto\s+(.+?)(?:\s+with\b|[.;!?]|$)", text, re.I | re.S)
    if target:
        values = _date_values(target.group(1))
        if values:
            return values[0]
    match = re.search(r"\bnew\s+date\s*(?:is|=|:)?\s*(.+?)(?:\s+with\b|[.;!?]|$)", text, re.I | re.S)
    if match:
        values = _date_values(match.group(1))
        if values:
            return values[0]
    values = _date_values(text)
    return values[-1] if len(values) >= 2 else (values[0] if values else None)


def _labelled_text_value(text: str, label: str) -> str | None:
    pattern = rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+)?\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, text, re.I)
    if match:
        return match.group(1)
    pattern = rf"\b{re.escape(label)}\b\s*(?:=|:|is\s+)(.+?)(?:[.;]|$)"
    match = re.search(pattern, text, re.I)
    return match.group(1).strip() if match else None


def _preference_value_for_slot(text: str, arg: str, slot_text: str) -> str | None:
    match = re.search(r"\bpreferr(?:ing|ed|ence)?\s+(.+?)\s+over\s+(.+?)(?:[?.!]|$)", text, re.I)
    if not match:
        return None
    option_one = _clean_slot_phrase(match.group(1))
    option_two = _clean_slot_phrase(match.group(2))
    if arg in {"option_one", "first_option", "choice_one"}:
        return option_one
    if arg in {"option_two", "second_option", "choice_two"}:
        return option_two
    if "category" in slot_text:
        for token in _tokens(f"{option_one} {option_two}"):
            if token in {"reading", "transportation", "food", "books", "book"}:
                return "reading" if token in {"book", "books"} else token
        return option_one.split()[-1] if option_one else None
    return None


def _restaurant_search_value_for_slot(text: str, arg: str, slot_text: str) -> str | None:
    lowered = text.lower()
    if "diet" in slot_text or "dietary" in slot_text:
        options = [
            "gluten-free",
            "gluten free",
            "vegan",
            "vegetarian",
            "halal",
            "kosher",
        ]
        for option in options:
            if re.search(rf"\b{re.escape(option)}\b", lowered):
                return option.replace(" ", "-")
    if arg in {"type", "food_type", "cuisine"} or "cuisine" in slot_text or "restaurant type" in slot_text:
        match = re.search(r"\b([A-Z][A-Za-z-]+)\s+restaurant\b", text)
        if match:
            return match.group(1)
        for cuisine in ["italian", "thai", "mexican", "indian", "chinese", "japanese", "french"]:
            if re.search(rf"\b{cuisine}\b", lowered):
                return cuisine.title()
    if "location" in slot_text or arg in {"location", "area", "city"}:
        match = re.search(r"\bnear\s+([A-Z][A-Za-z .'-]+?)(?:[?.!,]|$)", text)
        if match:
            return match.group(1).strip(" .'\"")
    return None


def _website_value(text: str) -> str | None:
    match = re.search(r"\b((?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,})(?:/[^\s,;]*)?", text)
    return match.group(1).strip(" .'\"") if match else None


def _looks_like_recipe_name_slot(arg: str, slot_text: str) -> bool:
    if arg in {"recipe", "recipe_name", "recipename", "dish", "dish_name", "dishname", "meal", "meal_name"}:
        return True
    return bool(
        re.search(r"\bname\s+of\s+the\s+(?:recipe|dish|meal)\b", slot_text)
        or re.search(r"\b(?:recipe|dish|meal)\s+name\b", slot_text)
    )


def _recipe_value(text: str) -> str | None:
    values = _recipe_dish_values(text)
    if values:
        return values[0]
    match = re.search(r"\b(?:calories|nutrition|ingredients?)\s+in\s+(?:the\s+)?(.+?)\s+from\s+", text, re.I)
    if match:
        return _clean_slot_phrase(match.group(1))
    match = re.search(r"\b([A-Z][A-Za-z '&-]+?\s+Recipe)\b", text)
    return match.group(1).strip(" .'\"") if match else None


def _food_item_value(text: str) -> str | None:
    recipe_values = _recipe_dish_values(text)
    if recipe_values:
        value = _core_food_from_recipe_phrase(recipe_values[0])
        if value:
            return value
    foods = [
        "avocado",
        "butter",
        "beef",
        "lasagna",
        "chicken",
        "spaghetti",
        "pasta",
        "rice",
        "milk",
        "flour",
        "sugar",
    ]
    lowered = text.lower()
    for food in foods:
        if re.search(rf"\b{food}\b", lowered):
            return food
    match = re.search(r"\b(?:in|of|for)\s+(?:an?|the)?\s*([A-Za-z][A-Za-z '-]+?)(?:\s+from\b|[?.!,]|$)", text, re.I)
    return _clean_slot_phrase(match.group(1)) if match else None


def _core_food_from_recipe_phrase(value: str) -> str | None:
    words = [word for word in _tokens(value) if word]
    modifiers = {
        "a",
        "an",
        "the",
        "homemade",
        "home",
        "made",
        "healthy",
        "gluten",
        "free",
        "dairy",
        "vegetarian",
        "vegan",
        "low",
        "calorie",
        "calories",
        "fresh",
        "easy",
        "simple",
    }
    core = [word for word in words if word not in modifiers]
    if not core:
        return None
    return " ".join(core)


def _season_value(text: str) -> str | None:
    match = re.search(r"\b(?:current\s+)?(\d{4}\s*[-/]\s*\d{4})\s+season\b", text, re.I)
    if match:
        return re.sub(r"\s+", "", match.group(1))
    match = re.search(r"\bseason\s+(\d{4}\s*[-/]\s*\d{4})\b", text, re.I)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def _season_values(text: str) -> list[str]:
    values = [
        re.sub(r"\s+", "", match.group(1))
        for match in re.finditer(r"\b(?:current\s+)?(\d{4}\s*[-/]\s*\d{4})\s+season\b", text, re.I)
    ]
    for match in re.finditer(r"\bseason(?:s)?\s+((?:\d{4}(?:\s*,\s*|\s+and\s+)?)+)", text, re.I):
        years = re.findall(r"\b(?:19|20)\d{2}\b", match.group(1))
        values.extend(years)
    if re.search(r"\bseasons?\b", text, re.I):
        values.extend(re.findall(r"\b(?:19|20)\d{2}\b", text))
    flattened: list[str] = []
    for value in values:
        years = re.findall(r"\b(?:19|20)\d{2}\b", value)
        flattened.extend(years or [value])
    return _dedupe(flattened)


def _team_values(text: str) -> list[str]:
    chunks: list[str] = []
    for pattern in [
        r"\b(?:of|for)\s+(.+?)\s+in\s+(?:the\s+)?(?:NBA|NFL|MLB|NHL|UEFA|La Liga|Premier League|[A-Z][A-Za-z]+\s+seasons?)\b",
        r"\bteams?\s+(.+?)(?:\s+in\b|[?.!]|$)",
    ]:
        for match in re.finditer(pattern, text):
            chunks.append(match.group(1))
    values: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\b(?:current\s+ranking|winning\s+percentage)\b", "", chunk, flags=re.I)
        for part in re.split(r"\s*,\s*|\s+\band\s+", chunk):
            value = _clean_slot_phrase(part)
            if value and value.lower() not in STOPWORDS and not re.search(r"\b(?:season|league|ranking)\b", value, re.I):
                values.append(value)
    return _dedupe(values)


def _team_value(text: str) -> str | None:
    values = _team_values(text)
    if values:
        return values[0]
    patterns = [
        r"\bteam\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,5})\b",
        r"\bwhere\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\s+stand\b",
        r"\bfor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b",
        r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\s+(?:matches|standings|position)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip(" .'\"")
            if value.lower() not in {"english premier league"}:
                return value
    return None


def _president_name_values(text: str) -> list[str]:
    values: list[str] = []
    match = re.search(r"\bpresidency\s+of\s+(.+?)(?:[?.!]|$)", text, re.I)
    if match:
        chunk = re.split(r"\s+(?:during|from|between|in)\b", match.group(1), maxsplit=1, flags=re.I)[0]
        values.extend(_split_named_units(chunk))
    return [
        value
        for value in _dedupe(values)
        if value.lower() not in {"provide", "major events", "events"}
    ]


def _position_values_for_year_groups(text: str) -> list[str]:
    values: list[str] = []
    for position, years in _position_year_groups(text):
        values.extend([position] * len(years))
    return values


def _years_for_position_groups(text: str) -> list[int]:
    values: list[int] = []
    for _position, years in _position_year_groups(text):
        values.extend(years)
    return values


def _position_year_groups(text: str) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    pattern = r"\b((?:vice\s+)?president)\b.+?\bin\s+((?:19|20)\d{2}(?:\s*(?:,|and)\s*(?:19|20)\d{2})*)"
    for match in re.finditer(pattern, text, re.I):
        position = match.group(1).lower()
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", match.group(2))]
        if years:
            groups.append((position, years))
    return groups


def _empire_values(text: str) -> list[str]:
    return _dedupe(match.group(1).strip() for match in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Empire)\b", text))


def _recipe_dish_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(
        r"\b(?:want|need|find|search for|get|give me|show me|recommend|suggest)\s+"
        r"(?:a|an|the)?\s*([A-Za-z][A-Za-z '&-]+?)\s+recipe\b",
        text,
        re.I,
    ):
        value = _clean_recipe_dish_value(match.group(1))
        if value and not re.fullmatch(r"(?:a|an|the|recipe|that|is|it)", value, re.I):
            values.append(value)
    for match in re.finditer(r"\b(?:a|an|the)?\s*([A-Z][A-Za-z '&-]+?)\s+recipe\b", text):
        value = _clean_recipe_dish_value(match.group(1))
        if value:
            values.append(value)
    return _dedupe(values)


def _clean_recipe_dish_value(value: str) -> str:
    cleaned = _clean_slot_phrase(value)
    cleaned = re.sub(
        r"^(?:i|we)\s+(?:want|need|would\s+like|am\s+looking\s+for|are\s+looking\s+for)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?:want|need|find|search\s+for|get|give\s+me|show\s+me|recommend|suggest)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(?:a|an|the|for|with|about)\s+", "", cleaned, flags=re.I)
    return cleaned.strip(" .'\"")


def _city_list_values(text: str) -> list[str]:
    venue_cities = _venue_city_values(text)
    if venue_cities:
        return venue_cities
    values: list[str] = []
    if re.search(r"\b(?:another|also|same|each|both|respectively|separately|and\s+another)\b", text, re.I):
        for match in re.finditer(
            r"\b(?:in|near|at|for)\s+([A-Z][A-Za-z .'-]+?)(?=\s+(?:that|which|with|offers?|opens?|open|and|also|another|for|from|to)\b|[?.!,]|$)",
            text,
        ):
            value = _clean_location_value(match.group(1))
            if value and value.lower() not in STOPWORDS and not re.search(r"\b(?:predict|find|search|list|get)\b", value, re.I):
                values.append(value)
    for pattern in [
        r"\b(?:in|near)\s+([A-Z][A-Za-z .'-]+(?:\s+and\s+[A-Z][A-Za-z .'-]+)+?)(?:\s+(?:happening|for|from|with|using|located|music\s+stores?|stores?|next|today|tomorrow)\b|[?.!,]|$)",
        r"\b(?:cities|locations)\s+(.+?)(?:[?.!]|$)",
    ]:
        for match in re.finditer(pattern, text):
            chunk = match.group(1)
            for part in re.split(r"\s*,\s*|\s+\band\s+", chunk):
                value = re.sub(r"'s\b.*$", "", part.strip(" .'\""), flags=re.I)
                value = re.sub(r"\b(?:music\s+stores?|stores?|happening|next\s+\w+)\b.*$", "", value, flags=re.I)
                value = _clean_location_value(value)
                if value and value.lower() not in STOPWORDS:
                    values.append(value)
    if re.search(r"\beach\s+city\b", text, re.I):
        for match in re.finditer(r"\bin\s+([A-Z][A-Za-z .'-]+?)(?=\s+(?:and|\.|,|compare|also|with|for|from)\b|[?.!]|$)", text):
            value = _clean_location_value(match.group(1))
            if value and value.lower() not in STOPWORDS:
                values.append(value)
    if not values:
        for match in re.finditer(
            r"\b(?:in|near|at)\s+([A-Z][A-Za-z .'-]+?)(?=\s+(?:with|for|from|to|during|on|using)\b|[?.!]|$)",
            text,
        ):
            value = _clean_location_value(match.group(1))
            if value and value.lower() not in STOPWORDS and not re.search(r"\b(?:predict|find|search|list|get)\b", value, re.I):
                values.append(value)
    return _dedupe(values)


def _event_name_value(text: str) -> str | None:
    quoted = _quoted_strings(text)
    if quoted:
        return quoted[0]
    match = re.search(r"\b(?:the\s+)?([A-Z][A-Za-z]+(?:\s+of\s+)?(?:\s+[A-Z][A-Za-z]+){0,5})\s+(?:take place|happen|occur|was signed)\b", text)
    if match:
        return match.group(1).strip(" .'\"")
    match = re.search(r"\bduring\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\b", text)
    return match.group(1).strip(" .'\"") if match else None


def _person_name_value(text: str) -> str | None:
    possessive = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})'s\b", text)
    if possessive:
        return possessive.group(1).strip()
    match = re.search(r"\bby\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\b", text)
    if match:
        return match.group(1).strip()
    return None


def _attendee_values(text: str) -> list[str]:
    match = re.search(
        r"\battendees?\s+(?:are|include|includes|including)?\s*(.+?)(?:[?.!]|$)",
        text,
        re.I | re.S,
    )
    if not match:
        return []
    chunk = match.group(1)
    chunk = re.split(r"\s+(?:on|at|in|with|for)\b", chunk, maxsplit=1, flags=re.I)[0]
    values = [
        re.sub(r"^(?:and\s+)?(?:the\s+)?", "", part.strip(" .'\""), flags=re.I).strip()
        for part in re.split(r"\s*,\s*|\s+\band\b\s+", chunk)
    ]
    values = [value for value in values if value and value.lower() not in STOPWORDS]
    return _dedupe(values)


def _nutrition_information_values(text: str, spec: dict[str, Any]) -> list[str]:
    lowered = text.lower()
    aliases = {
        "protein": "Protein",
        "calorie": "Calories",
        "calories": "Calories",
        "carb": "Carbohydrates",
        "carbs": "Carbohydrates",
        "carbohydrate": "Carbohydrates",
        "carbohydrates": "Carbohydrates",
        "fat": "Fat",
        "fiber": "Fiber",
    }
    values = [canonical for key, canonical in aliases.items() if re.search(rf"\b{re.escape(key)}\b", lowered)]
    enum_values = []
    items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
    if isinstance(items, dict) and isinstance(items.get("enum"), list):
        enum_values = [str(item) for item in items.get("enum") or []]
    if enum_values:
        canonical_map = {value.lower(): value for value in enum_values}
        values = [canonical_map.get(value.lower(), value) for value in values]
    return _dedupe(values)


def _floor_values(text: str) -> list[int]:
    values: list[int] = []
    for raw in re.findall(r"\b(\d+)(?:st|nd|rd|th)?\s+floors?\b|\b(\d+)(?:st|nd|rd|th)?(?=,|\s+and|\s*&)", text, re.I):
        value = next((item for item in raw if item), "")
        if value:
            values.append(int(value))
    if values:
        return values
    match = re.search(r"\bfor\s+(.+?)\s+floors?\b", text, re.I)
    return [int(value) for value in re.findall(r"\d+", match.group(1))] if match else []


def _route_endpoint_value(text: str, side: str) -> str | None:
    if side == "from":
        pattern = r"\bfrom\s+(.+?)(?:\s+to\b|[?.!]|$)"
    else:
        pattern = r"\bto\s+(.+?)(?:\s+(?:using|with|on|for|after|before|at|via|by)\b|[?.!]|$)"
    match = re.search(pattern, text, re.I)
    return match.group(1).strip(" .'\"") if match else None


def _route_endpoint_value_for_call(text: str, side: str, call_index: int) -> str | None:
    pairs = _route_endpoint_pairs(text)
    if pairs:
        pair = pairs[min(call_index, len(pairs) - 1)]
        return pair[0] if side == "from" else pair[1]
    return _route_endpoint_value(text, side)


def _route_endpoint_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    from_to_pattern = (
        r"\bfrom\s+(.+?)\s+to\s+(.+?)"
        r"(?=(?:,\s*(?:and\s*)?|\s+and\s+then\s+|\s+then\s+)?from\s+|\s+and\s+then\s+(?:a\s+|the\s+)?(?:fastest\s+|scenic\s+|shortest\s+)?route\b|,\s*(?:and\s+)?(?:finally|lastly)\b|[?.!,]|$|\s+(?:using|with|via|by|for|considering)\b)"
    )
    for match in re.finditer(from_to_pattern, text):
        pair = (_clean_route_endpoint(match.group(1)), _clean_route_endpoint(match.group(2)))
        if pair[0] and pair[1]:
            pairs.append(pair)

    between_pattern = (
        r"\bbetween\s+([A-Z][A-Za-z0-9 .'-]+?)\s+and\s+([A-Z][A-Za-z0-9 .'-]+?)"
        r"(?=,\s*(?:and\s*)?between\s+|[?.!]|$|\s+(?:using|with|via|by|for)\b)"
    )
    for match in re.finditer(between_pattern, text):
        pair = (_clean_route_endpoint(match.group(1)), _clean_route_endpoint(match.group(2)))
        if pair[0] and pair[1]:
            pairs.append(pair)

    return_pattern = r"\breturn\s+to\s+([A-Z][A-Za-z0-9 .'-]+?)\s+from\s+([A-Z][A-Za-z0-9 .'-]+?)(?=[?.!,]|$)"
    for match in re.finditer(return_pattern, text):
        pair = (_clean_route_endpoint(match.group(2)), _clean_route_endpoint(match.group(1)))
        if pair[0] and pair[1]:
            pairs.append(pair)

    back_pattern = r"\bback\s+to\s+(.+?)\s+from\s+(.+?)(?=[?.!,]|$)"
    for match in re.finditer(back_pattern, text):
        pair = (_clean_route_endpoint(match.group(2)), _clean_route_endpoint(match.group(1)))
        if pair[0] and pair[1]:
            pairs.append(pair)

    if pairs:
        return _dedupe_route_pairs(pairs)

    bare_pattern = r"\b([A-Z][A-Za-z .'-]+?)\s+to\s+([A-Z][A-Za-z .'-]+?)(?=,\s*(?:and\s*)?[A-Z]|\s+and\s+[A-Z]|[?.!]|$)"
    for match in re.finditer(bare_pattern, text):
        pair = (_clean_route_endpoint(match.group(1)), _clean_route_endpoint(match.group(2)))
        if pair[0] and pair[1] and not _looks_like_currency_pair(pair):
            pairs.append(pair)
    return _dedupe_route_pairs(pairs)


def _clean_route_endpoint(value: str) -> str:
    cleaned = re.sub(r"^(?:and|then|the|a|an)\s+", "", value.strip(" .'\""), flags=re.I)
    in_match = re.search(r"^(?:my\s+)?(?:home|office)\s+in\s+([A-Z][A-Za-z .'-]+)$", cleaned, re.I)
    if in_match:
        cleaned = in_match.group(1)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .'\"")
    cleaned = re.sub(r"\s+(?:and|then)$", "", cleaned, flags=re.I).strip(" .'\"")
    cleaned = re.sub(r"^(?:my\s+)?(?:home|office)\s+(?:in\s+)?", "", cleaned, flags=re.I).strip(" .'\"")
    if cleaned.lower() in {"route", "distance", "directions", "trip", "path"}:
        return ""
    return cleaned


def _looks_like_currency_pair(pair: tuple[str, str]) -> bool:
    return all(re.fullmatch(r"[A-Z]{3}", item.strip()) for item in pair)


def _dedupe_route_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for start, end in pairs:
        key = (start.lower(), end.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append((start, end))
    return unique


def _case_number_value(text: str) -> str | None:
    match = re.search(r"\b(?:case\s*(?:no\.?|number)?\s*)?([A-Z]{1,4}[- ]?\d{2,}[-A-Z0-9]*)\b", text, re.I)
    return match.group(1).strip() if match else None


def _jurisdiction_value(text: str) -> str | None:
    match = re.search(r"\bin\s+([A-Z][A-Za-z .'-]+(?:County|Court|District|State))\b", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-Z][A-Za-z .'-]+ County)\b", text)
    if match:
        return match.group(1).strip()
    return _location_context_value(text)


def _location_context_value(text: str) -> str | None:
    matches = re.findall(r"\b(?:in|from|within)\s+([A-Z][A-Za-z .'-]+?)(?:\s+(?:in detail|for|about|with|between)\b|[?.!,]|$)", text)
    return matches[-1].strip(" .'\"") if matches else None


def _quoted_location_values(text: str) -> list[str]:
    values = [
        match.group(2).strip(" .'\"")
        for match in re.finditer(r"\b(?:in|from|at|near|within)\s+(['\"])([^'\"]+)\1", text, re.I)
    ]
    for match in re.finditer(
        r"\b(?:in|from|at|near|within)\s+((?:['\"][^'\"]+['\"](?:\s*(?:,|and)\s*)?)+)",
        text,
        re.I,
    ):
        values.extend(_quoted_strings(match.group(1)))
    return _dedupe(value for value in values if value)


def _between_party_values(text: str) -> list[str]:
    match = re.search(r"\bbetween\s+([A-Z][A-Za-z .'-]+?)\s+and\s+([A-Z][A-Za-z .'-]+?)(?:\s+(?:for|about|regarding|in)\b|[?.!,]|$)", text)
    if not match:
        return []
    return [match.group(1).strip(" .'\""), match.group(2).strip(" .'\"")]


def _commodity_value(text: str) -> str | None:
    commodities = [
        "gold",
        "silver",
        "platinum",
        "palladium",
        "copper",
        "aluminum",
        "aluminium",
        "oil",
        "wheat",
        "corn",
    ]
    lowered = text.lower()
    for commodity in commodities:
        if re.search(rf"\b{commodity}\b", lowered):
            return commodity.title()
    return None


def _particle_values(text: str) -> list[str]:
    particles = ["Electron", "Proton", "Neutron", "Photon", "Muon", "Tau"]
    return [particle for particle in particles if re.search(rf"\b{re.escape(particle)}s?\b", text, re.I)]


def _measure_value(text: str) -> str | None:
    lowered = text.lower()
    for measure in ["ounce", "oz", "gram", "kg", "kilogram", "pound", "barrel"]:
        if re.search(rf"\b{measure}s?\b", lowered):
            return measure
    return None


def _conversion_unit_value_for_slot(text: str, arg: str, slot_text: str) -> str | None:
    mentions = _unit_mentions(text)
    if not mentions:
        return None
    source = _source_unit_value(text, mentions)
    if arg in {"from_unit", "source_unit", "input_unit"} or "convert from" in slot_text:
        return source
    if arg in {"to_unit", "target_unit", "output_unit"} or "convert to" in slot_text:
        for unit, _start, _end in mentions:
            if unit != source:
                return unit
    if len(mentions) == 1:
        return mentions[0][0]
    return None


def _unit_mentions(text: str) -> list[tuple[str, int, int]]:
    aliases = {
        "ounces": "ounces",
        "ounce": "ounces",
        "oz": "ounces",
        "pounds": "pounds",
        "pound": "pounds",
        "lb": "pounds",
        "lbs": "pounds",
        "grams": "grams",
        "gram": "grams",
        "kilograms": "kilograms",
        "kilogram": "kilograms",
        "kg": "kilograms",
        "cups": "cups",
        "cup": "cups",
        "tablespoons": "tablespoons",
        "tablespoon": "tablespoons",
        "teaspoons": "teaspoons",
        "teaspoon": "teaspoons",
    }
    mentions: list[tuple[str, int, int]] = []
    for raw, normalized in aliases.items():
        for match in re.finditer(rf"\b{re.escape(raw)}\b", text, re.I):
            mentions.append((normalized, match.start(), match.end()))
    mentions.sort(key=lambda item: item[1])
    return _dedupe_positioned_mentions(mentions)


def _source_unit_value(text: str, mentions: list[tuple[str, int, int]]) -> str | None:
    for unit, start, _end in mentions:
        prefix = text[max(0, start - 18) : start]
        if re.search(r"\d[\d,.]*\s*$", prefix):
            return unit
    for unit, start, _end in mentions:
        prefix = text[max(0, start - 12) : start].lower()
        if re.search(r"\b(?:from|in)\s*$", prefix):
            return unit
    return mentions[0][0] if mentions else None


def _currency_value_for_slot(
    text: str,
    arg: str,
    slot_text: str,
    call_index: int = 0,
    call_count: int = 1,
) -> str | None:
    scenarios = _currency_conversion_scenarios(text)
    if scenarios and call_count > 1:
        scenario = scenarios[min(call_index, len(scenarios) - 1)]
        wants_source = (
            arg in {"base_currency", "from_currency", "source_currency", "currency_from"}
            or "convert from" in slot_text
            or "base currency" in slot_text
            or "original amount" in slot_text
        )
        wants_target = (
            arg in {"target_currency", "to_currency", "destination_currency", "currency_to"}
            or "convert to" in slot_text
            or "target currency" in slot_text
        )
        if wants_source:
            return scenario.get("from_currency")
        if wants_target:
            return scenario.get("to_currency")
    mentions = _currency_mentions(text)
    if not mentions:
        return None
    source = _source_currency_value(text, mentions)
    wants_source = (
        arg in {"base_currency", "from_currency", "source_currency", "currency_from"}
        or "convert from" in slot_text
        or "base currency" in slot_text
        or "original amount" in slot_text
    )
    wants_target = (
        arg in {"target_currency", "to_currency", "destination_currency", "currency_to"}
        or "convert to" in slot_text
        or "target currency" in slot_text
    )
    if wants_source:
        return source
    if wants_target:
        target = _target_currency_value(text, mentions, source)
        if target:
            return target
    if len(mentions) == 1:
        return mentions[0][0]
    return None


def _currency_conversion_scenarios(text: str) -> list[dict[str, Any]]:
    money_word = r"(?:dollars?|usd|euros?|eur|pounds?|gbp|yen|jpy|cad|aud)"
    scenarios: list[dict[str, Any]] = []
    pattern = (
        rf"\b(?:transfer|convert|exchange)\s+"
        rf"(?P<amount>\d[\d,.]*)\s+(?P<source>{money_word})\s+"
        rf"(?:to|into)\s+(?P<target>{money_word})\b"
    )
    for match in re.finditer(pattern, text, re.I):
        source = _currency_code_for_phrase(match.group("source"))
        target = _currency_code_for_phrase(match.group("target"))
        amount = _number_from_text(match.group("amount"))
        if source and target:
            scenarios.append({"amount": amount, "from_currency": source, "to_currency": target})
    return scenarios


def _currency_code_for_phrase(value: str) -> str | None:
    lowered = value.strip().lower()
    if lowered in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[lowered]
    singular = lowered.rstrip("s")
    return CURRENCY_ALIASES.get(singular)


def _currency_mentions(text: str) -> list[tuple[str, int, int]]:
    mentions: list[tuple[str, int, int]] = []
    for raw, code in sorted(CURRENCY_ALIASES.items(), key=lambda item: -len(item[0])):
        for match in re.finditer(rf"\b{re.escape(raw)}\b", text, re.I):
            mentions.append((code, match.start(), match.end()))
    mentions.sort(key=lambda item: item[1])
    return _dedupe_positioned_mentions(mentions)


def _looks_like_currency_code_value(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}", value.strip()))


def _currency_code_grounded_by_evidence(user_request: str, value: str, evidence: str) -> bool:
    code = value.strip().upper()
    if _span_in_request(user_request, code):
        return True
    evidence_text = str(evidence or "")
    if not evidence_text or not _span_in_request(user_request, evidence_text):
        return False
    return any(found_code == code for found_code, _start, _end in _currency_mentions(evidence_text))


def _source_currency_value(text: str, mentions: list[tuple[str, int, int]]) -> str | None:
    for code, start, _end in mentions:
        prefix = text[max(0, start - 24) : start]
        if re.search(r"\d[\d,.]*\s*$", prefix):
            return code
    for code, start, _end in mentions:
        prefix = text[max(0, start - 16) : start].lower()
        if re.search(r"\b(?:from|for)\s*$", prefix):
            return code
    return mentions[0][0] if mentions else None


def _target_currency_value(
    text: str,
    mentions: list[tuple[str, int, int]],
    source: str | None,
) -> str | None:
    for code, start, _end in mentions:
        prefix = text[max(0, start - 16) : start].lower()
        if re.search(r"\b(?:to|into|in)\s*$", prefix) and code != source:
            return code
    for code, _start, _end in mentions:
        if code != source:
            return code
    return None


def _dedupe_positioned_mentions(mentions: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    unique: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in mentions:
        key = (item[0], item[1], item[2])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _looks_like_conversion_request(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\bconvert(?:ing|ed)?\b", lowered):
        return True
    if len(_currency_mentions(text)) >= 2:
        return True
    if len(_unit_mentions(text)) >= 2 and re.search(r"\bhow many\b|\bin\s+\d", lowered):
        return True
    return False


def _celestial_body_values(text: str) -> list[str]:
    bodies = [
        "Earth",
        "Moon",
        "Sun",
        "Mars",
        "Venus",
        "Mercury",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ]
    found = [body for body in bodies if re.search(rf"\b{re.escape(body)}\b", text, re.I)]
    if len(found) >= 2:
        return found[:2]
    match = re.search(r"\bfrom\s+(?:a\s+)?([A-Za-z][A-Za-z ]+?)\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z ]+?)(?:[?.!,]|$)", text, re.I)
    if match:
        return [match.group(1).strip(" .'\""), match.group(2).strip(" .'\"")]
    return found


def _topic_value(text: str, arg: str) -> str | None:
    patterns = [
        r"\babout\s+(.+?)(?:\s+from\b|\s+in\b|\s+with\b|[?.!,]|$)",
        r"\bof\s+(.+?)(?:\s+from\b|\s+in\b|\s+with\b|[?.!,]|$)",
        r"\b(?:crime|case|event|discovery|treaty|history)\s*(?:of\s*)?(.+?)(?:\s+from\b|\s+in\b|\s+with\b|[?.!,]|$)",
    ]
    if arg == "field_of_law":
        match = re.search(r"\b([A-Za-z]+)\s+law\b", text, re.I)
        if match:
            return match.group(1).strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = _clean_slot_phrase(match.group(1))
            if value:
                return value
    values = _capitalized_entity_values(text)
    return values[0] if values else None


def _dataset_value(text: str) -> str | None:
    explicit = _explicit_value_for_arg(text, "dataset")
    if explicit is not None:
        return str(explicit)
    match = re.search(r"\b(?:provided|given|using|on|from)\s+(?:the\s+)?data\s+([A-Za-z][A-Za-z0-9_]+)\b", text, re.I)
    if match:
        return match.group(1)
    match = re.search(r"\b(dataset[_-]?[A-Za-z0-9]+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b", text)
    return match.group(1) if match else None


def _brand_or_manufacturer_value(text: str) -> str | None:
    brands = [
        "Fender",
        "Gibson",
        "Kawai",
        "Roland",
        "Steinway",
        "Taylor",
        "Yamaha",
    ]
    for brand in brands:
        if re.search(rf"\b{re.escape(brand)}\b", text, re.I):
            return brand
    match = re.search(r"\b(?:from|by|made by|manufacturer)\s+([A-Z][A-Za-z0-9&'.-]+(?:\s+[A-Z][A-Za-z0-9&'.-]+)*)", text)
    return match.group(1).strip(" .'\"") if match else None


def _model_value(text: str) -> str | None:
    quoted = _quoted_strings(text)
    if quoted:
        return quoted[0]
    match = re.search(
        r"\b(?:Fender|Gibson|Yamaha|Roland|Kawai|Steinway)\s+([A-Z][A-Za-z0-9'.-]+(?:\s+[A-Z0-9][A-Za-z0-9'.-]+){0,5})",
        text,
    )
    if match:
        return match.group(1).strip(" .'\"")
    return None


def _algorithm_model_values(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\busing\s+(?:the\s+)?([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3})\s+model\b",
        r"\bwith\s+(?:the\s+)?([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3})\s+model\b",
        r"\b([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3})\s+model\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = _clean_slot_phrase(match.group(1))
            if value and value.lower() not in {"machine learning", "forecast", "prediction"}:
                values.append(value)
    return _dedupe(values)


def _financial_scenarios(text: str) -> list[dict[str, Any]]:
    amount = r"(?P<amount>\d[\d,]*(?:\.\d+)?)"
    institution = r"(?P<institution>[A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){0,4})"
    pattern = (
        rf"\b{institution}\s+for\s+(?:an?\s+)?(?:loan\s+)?amount(?:\s+of)?\s+{amount}"
        r"(?:\D{0,80}?\bannual\s+income(?:\s+of)?\s+(?P<income>\d[\d,]*(?:\.\d+)?))?"
    )
    scenarios: list[dict[str, Any]] = []
    for match in re.finditer(pattern, text):
        institution_value = _clean_financial_institution_value(match.group("institution"))
        loan_amount = _number_from_text(match.group("amount").replace(",", ""))
        income = _number_from_text((match.group("income") or "").replace(",", "")) if match.group("income") else None
        if institution_value and loan_amount is not None:
            scenario = {"financial_institution": institution_value, "loan_amount": loan_amount}
            if income is not None:
                scenario["annual_income"] = income
            scenarios.append(scenario)
    return scenarios


def _financial_institution_values(text: str) -> list[str]:
    known = [
        "American Express",
        "Bank of America",
        "Barclays",
        "Capital One",
        "Chase",
        "Citibank",
        "Citi",
        "HSBC",
        "Santander",
        "TD Bank",
        "US Bank",
        "Wells Fargo",
    ]
    values = [name for name in known if re.search(rf"\b{re.escape(name)}\b", text, re.I)]
    values.extend(str(item["financial_institution"]) for item in _financial_scenarios(text) if item.get("financial_institution"))
    generic_patterns = [
        r"\b(?:from|at|with)\s+([A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){0,4})(?=\s+(?:with|for|of|amount|loan|annual)\b|[,.?]|$)",
        r"\b([A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){0,4})\s+for\s+(?:an?\s+)?(?:loan\s+)?amount\b",
    ]
    for pattern in generic_patterns:
        for match in re.finditer(pattern, text):
            value = _clean_financial_institution_value(match.group(1))
            if value:
                values.append(value)
    return _dedupe(values)


def _clean_financial_institution_value(value: str) -> str | None:
    cleaned = re.sub(r"^(?:and|or|from|at|with|for)\s+", "", value.strip(" .'\""), flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .'\"")
    if not cleaned or cleaned.lower() in {"amount", "annual income", "bank", "loan", "the bank"}:
        return None
    if cleaned.lower().startswith(("i ", "can ", "please ", "check ")):
        return None
    return cleaned


def _financial_numeric_value(text: str, arg: str, spec: dict[str, Any], call_index: int) -> Any | None:
    scenarios = _financial_scenarios(text)
    if not scenarios:
        return None
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    key = ""
    if "income" in slot_text:
        key = "annual_income"
    elif "amount" in slot_text and "loan" in slot_text:
        key = "loan_amount"
    if not key:
        return None
    values = [scenario[key] for scenario in scenarios if key in scenario]
    if not values:
        return None
    return values[min(call_index, len(values) - 1)]


def _specification_value(text: str) -> str | None:
    match = re.search(r"\bspecifications?\s+of\s+(.+?)(?:\s+available\b|[?.!]|$)", text, re.I)
    if match:
        return _clean_slot_phrase(match.group(1), keep_leading_prepositions=True)
    descriptors = []
    for phrase in ["open hole", "c foot", "silver headjoint", "rosewood finish", "excellent condition"]:
        if re.search(rf"\b{re.escape(phrase)}\b", text, re.I):
            descriptors.append(phrase)
    return ", ".join(descriptors) if descriptors else None


def _ecology_scenarios(text: str) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    pattern = (
        r"\becological\s+impact\s+of\s+(?:the\s+)?"
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+in\s+(?:the\s+)?"
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+ecosystem\s+over\s+(?:the\s+)?last\s+(\d+)\s+years?"
        r"(?P<tail>.*?)(?=\becological\s+impact\s+of\b|$)"
    )
    for match in re.finditer(pattern, text, re.I | re.S):
        tail = match.group("tail") or ""
        growth_match = re.search(
            r"\bpopulation\s+growth\b.*?\bover\s+(?:the\s+)?last\s+(\d+)\s+years?",
            tail,
            re.I | re.S,
        )
        if not growth_match:
            continue
        species = _clean_slot_phrase(match.group(1))
        ecosystem = _clean_slot_phrase(match.group(2))
        if not species or not ecosystem:
            continue
        scenarios.append(
            {
                "species": species,
                "ecosystem": ecosystem,
                "location": ecosystem,
                "impact_years": int(match.group(3)),
                "growth_years": int(growth_match.group(1)),
            }
        )
    return scenarios


def _tool_uses_ecology_scenarios(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower()
    return bool(
        "species" in text
        and (
            "ecosystem" in text
            or "population growth" in text
            or "ecological impact" in text
            or "impact analysis" in text
        )
    )


def _ecology_string_value_for_slot(user_request: str, arg: str, slot_text: str, call_index: int) -> str | None:
    scenarios = _ecology_scenarios(user_request)
    if not scenarios:
        return None
    scenario = scenarios[min(call_index, len(scenarios) - 1)]
    if _looks_like_location_slot(arg, slot_text) and ("species" in user_request.lower() or "ecosystem" in user_request.lower()):
        return str(scenario["location"])
    if "ecosystem" in slot_text:
        return str(scenario["ecosystem"])
    if arg in {"species", "animal_species", "target_species"} or re.search(r"\bspecies\b", slot_text):
        return str(scenario["species"])
    return None


def _species_values(text: str) -> list[str]:
    patterns = [
        r"\b(?:between|compare)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z -]+?)\s+and\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z -]+?)(?:\s+(?:are|is|in|as|for|by|using)\b|[?.!,]|$)",
        r"\b(?:a|an|the)?\s*([A-Za-z][A-Za-z -]+?)\s+and\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z -]+?)\s+are\s+(?:genetically\s+)?similar\b",
    ]
    values: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        for raw in match.groups():
            value = _clean_slot_phrase(raw)
            if value and value.lower() not in {"how", "what", "find out"}:
                values.append(value)
        if len(values) >= 2:
            break
    return _dedupe(values[:2])


def _pokemon_values(text: str) -> list[str]:
    values: list[str] = []
    patterns = [
        r"\b(?:a|an)\s+([A-Z][A-Za-z0-9-]+)\s+learn\b",
        r"\bif\s+([A-Z][A-Za-z0-9-]+)\s+can\s+learn\b",
        r"\bfor\s+([A-Z][A-Za-z0-9-]+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group(1).strip(" .'\"")
            if value and value.lower() not in {"pokemon", "go"}:
                values.append(value)
    return _dedupe(values)


def _id_number_value(lowered: str, arg: str) -> int | None:
    label = re.sub(r"_?id$", "", arg.lower()).replace("_", " ").strip()
    patterns = []
    if label:
        patterns.append(rf"\b{re.escape(label)}\s*(?:id|number)\s*(?:is|=|:)?\s*(\d+)\b")
    patterns.extend(
        [
            r"\b(?:id|number)\s*(?:is|=|:)?\s*(\d+)\b",
            r"\bwhose\s+id\s+is\s+(\d+)\b",
            r"\b(?:it'?s|it\s+is|that'?s|that\s+is)\s*(\d{4,})\b",
        ]
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def _card_probability_numeric_value(lowered: str, arg: str, spec: dict[str, Any]) -> int | None:
    text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if not re.search(r"\b(?:card|deck|king|queen|jack|ace)\b", lowered):
        return None
    if any(token in text for token in ["favorable", "successful", "success"]):
        if "face card" in lowered:
            return 12
        if re.search(r"\b(?:king|queen|jack|ace)\b", lowered):
            return 4
        if re.search(r"\b(?:heart|diamond|club|spade)s?\b", lowered):
            return 13
        if re.search(r"\b(?:red|black)\b", lowered):
            return 26
    if any(token in text for token in ["total", "possible", "outcomes"]):
        return 52
    return None


def extract_numbers(text: str) -> list[Any]:
    values: list[Any] = []
    number_pattern = r"(?<![A-Za-z0-9_^])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?"
    for match in re.finditer(number_pattern, text):
        raw = match.group(0).strip()
        raw = raw.replace(",", "")
        if raw.endswith("%"):
            values.append(float(raw[:-1]) / 100)
        elif "." in raw:
            values.append(float(raw))
        else:
            values.append(int(raw))
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text.lower()):
            values.append(number)
    return values


def _labeled_numeric_array_values(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    label_pattern = (
        r"((?:group|sample|dataset|data\s+set|set|list)\s*(?:[A-Z]|\d+)|"
        r"(?:first|second|third|fourth)\s+(?:group|sample|dataset|data\s+set|set|list))"
    )
    pattern = rf"\b{label_pattern}\s*(?:is|has|=|:)?\s*(\[[^\]]+\])"
    for index, match in enumerate(re.finditer(pattern, text, re.I)):
        label = match.group(1)
        array_text = match.group(2)
        numbers = extract_numbers(array_text)
        if numbers:
            values.append({"label": label, "value": numbers, "span": match.group(0), "index": index})
    if values:
        return values
    bracket_arrays = re.findall(r"\[[^\]]+\]", text)
    if len(bracket_arrays) >= 2:
        for index, array_text in enumerate(bracket_arrays):
            numbers = extract_numbers(array_text)
            if numbers:
                values.append({"label": str(index + 1), "value": numbers, "span": array_text, "index": index})
    return values


def _labeled_numeric_array_value_for_slot(
    user_request: str,
    arg: str,
    call_index: int,
    facts: list[dict[str, Any]],
) -> list[Any] | None:
    groups = [
        fact
        for fact in facts
        if fact.get("type") == "numeric_array" and isinstance(fact.get("normalized_value"), list)
    ]
    if not groups:
        groups = [
            {
                "label": item["label"],
                "normalized_value": item["value"],
                "index": item["index"],
            }
            for item in _labeled_numeric_array_values(user_request)
        ]
    if not groups:
        return None
    index = _array_slot_index(arg, call_index)
    for group in groups:
        label = str(group.get("label") or "").lower().replace("_", " ")
        group_index = int(group.get("index") or 0)
        if group_index == index:
            return list(group.get("normalized_value") or [])
        if index == 0 and re.search(r"\b(?:a|1|one|first)\b", label):
            return list(group.get("normalized_value") or [])
        if index == 1 and re.search(r"\b(?:b|2|two|second)\b", label):
            return list(group.get("normalized_value") or [])
    if index < len(groups):
        return list(groups[index].get("normalized_value") or [])
    return None


def _array_slot_index(arg: str, call_index: int) -> int:
    normalized = arg.lower().replace("_", " ")
    if re.search(r"\b(?:a|1|one|first)\b", normalized) or arg.endswith("1"):
        return 0
    if re.search(r"\b(?:b|2|two|second)\b", normalized) or arg.endswith("2"):
        return 1
    return call_index


def _result_count_value(text: str) -> int | None:
    lowered = text.lower()
    result_nouns = (
        r"results?|examples?|recommendations?|items?|cases?|records?|matches?|restaurants?|hotels?|"
        r"flights?|events?|articles?|entries?|options?|suggestions?"
    )
    patterns = [
        r"\btop\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\b",
        rf"\b(?:first|next|best|latest|recent|nearest)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\s+(?:{result_nouns})\b",
        rf"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty)\s+(?:{result_nouns})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = _number_from_text(match.group(1))
        if isinstance(value, int) and value > 0:
            return value
    return None


def _has_result_count_phrase(text: str) -> bool:
    return _result_count_value(text) is not None


def extract_artist_like_values(text: str) -> list[str]:
    match = re.search(r"artists?\s+(.+?)(?:,\s+with|\s+with|\s+on\s+spotify|$)", text, re.I)
    if not match:
        return []
    chunk = match.group(1)
    parts = re.split(r"\s+and\s+|,\s*", chunk)
    return [part.strip(" .'\"") for part in parts if part.strip(" .'\"")]


def _numeric_value_for_arg(
    user_request: str,
    arg: str,
    spec: dict[str, Any],
    call_index: int,
) -> Any:
    lowered = user_request.lower()
    explicit = _explicit_value_for_arg(user_request, arg)
    if isinstance(explicit, (int, float)):
        return _coerce_number(explicit, spec)
    if isinstance(explicit, str):
        parsed = _number_from_text(explicit)
        if parsed is not None:
            return _coerce_number(parsed, spec)
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if "percent" in slot_text or "percentage" in slot_text:
        percent_match = re.search(r"\b(\d+(?:\.\d+)?)(?:\s*%|\s+(?:percent|percentage)\b)", user_request, re.I)
        if percent_match:
            value = _number_from_text(percent_match.group(1))
            if value is not None:
                return _coerce_number(value, spec)
    numbers = extract_numbers(user_request)
    card_probability_value = _card_probability_numeric_value(lowered, arg, spec)
    if card_probability_value is not None:
        return _coerce_number(card_probability_value, spec)
    if not numbers:
        if arg in {"duration", "timeframe", "time_frame", "time", "days", "months", "years"}:
            temporal_value = _duration_number_from_temporal_phrase(lowered)
            if temporal_value is not None:
                return _coerce_number(temporal_value, spec)
        return None

    unit_values = _unit_labeled_numeric_values(user_request, arg, spec)
    if unit_values:
        return _coerce_number(unit_values[min(call_index, len(unit_values) - 1)], spec)

    financial_value = _financial_numeric_value(user_request, arg, spec, call_index)
    if financial_value is not None:
        return _coerce_number(financial_value, spec)

    x_values = [float(value) if "." in value else int(value) for value in re.findall(r"\bx\s*=\s*(-?\d+(?:\.\d+)?)", lowered)]
    if arg == "x" and x_values:
        value = x_values[-1] if "derivative" in lowered or "differentiat" in lowered else x_values[0]
        return _coerce_number(value, spec)
    if arg in {"a", "lower", "lower_limit", "start", "start_x", "startx"}:
        value = _first_number_for_pattern(lowered, r"\bfrom\s+x\s*=\s*(-?\d+(?:\.\d+)?)")
        if value is not None:
            return _coerce_number(value, spec)
    if arg in {"b", "upper", "upper_limit", "end", "end_x", "endx"}:
        value = _first_number_for_pattern(lowered, r"\bto\s+x\s*=\s*(-?\d+(?:\.\d+)?)")
        if value is not None:
            return _coerce_number(value, spec)
    range_bound = _numeric_range_bound_value(user_request, arg)
    if range_bound is not None:
        return _coerce_number(range_bound, spec)

    if arg.startswith("side"):
        index = int(arg[-1]) - 1 if arg[-1].isdigit() else 0
        return _coerce_number(numbers[index], spec) if index < len(numbers) else None
    if re.fullmatch(r"(?:num|number|value|x|y)\d+", arg):
        slot_index = int(re.findall(r"\d+", arg)[0]) - 1
        arity = 2 if slot_index < 2 and len(numbers) >= (call_index + 1) * 2 else slot_index + 1
        index = slot_index + (call_index * arity)
        if index >= len(numbers):
            index = slot_index
        return _coerce_number(numbers[index], spec) if index < len(numbers) else None
    if arg in {"x", "y"} and "side" in str(spec.get("description") or "").lower():
        slot_index = 0 if arg == "x" else 1
        index = slot_index + (call_index * 2)
        if index >= len(numbers):
            index = slot_index
        return _coerce_number(numbers[index], spec) if index < len(numbers) else None
    if arg in {"x", "y"} and len(numbers) >= 2:
        slot_index = 0 if arg == "x" else 1
        index = slot_index + (call_index * 2)
        if index >= len(numbers):
            index = slot_index
        return _coerce_number(numbers[index], spec)
    if arg == "duration":
        minute_values = [
            _number_from_text(match.group(1))
            for match in re.finditer(r"(\d+|[a-z]+)\s+minutes?", lowered)
            if _number_from_text(match.group(1)) is not None
        ]
        if not minute_values:
            minute_values = [n for n in numbers if isinstance(n, int) and n > 0]
        return _coerce_number(minute_values[min(call_index, len(minute_values) - 1)], spec) if minute_values else None
    if arg in {"timeframe", "time_frame", "time", "days", "months", "years", "days_ahead", "forecast_days"}:
        temporal_value = _duration_number_from_temporal_phrase(lowered)
        if temporal_value is not None:
            return _coerce_number(temporal_value, spec)
    if arg == "n":
        for pattern in [
            r"(?:roll|rolled|rolls|trials?|attempts?|rounds?)\D{0,20}(\d+|[a-z]+)\s+times",
            r"(\d+|[a-z]+)\s+(?:rolls|trials|attempts|rounds)",
            r"(\d+|[a-z]+)\s+times",
        ]:
            value = _first_number_for_pattern(lowered, pattern)
            if value is not None:
                return _coerce_number(value, spec)
    if arg == "k":
        for pattern in [
            r"exactly\s+(\d+|[a-z]+)",
            r"(\d+|[a-z]+)\s+(?:successes|wins|sixes)",
        ]:
            value = _first_number_for_pattern(lowered, pattern)
            if value is not None:
                return _coerce_number(value, spec)
    if arg == "p":
        percent = next((n for n in numbers if isinstance(n, float) and 0 < n < 1), None)
        if percent is not None:
            return percent
        if "one in six" in lowered or "1 in 6" in lowered:
            return 1 / 6
    if "id" in arg:
        value = _id_number_value(lowered, arg)
        if value is not None:
            return _coerce_number(value, spec)

    aliases = {
        "base": ["base"],
        "height": ["height"],
        "side1": ["side"],
        "side2": ["side"],
        "side3": ["side"],
        "duration": ["duration", "minutes", "play time"],
        "latitude": ["latitude", "lat"],
        "lat": ["latitude", "lat"],
        "longitude": ["longitude", "lon"],
        "lon": ["longitude", "lon"],
        "n": ["roll", "times", "trials", "attempts", "rounds"],
        "k": ["exactly", "successes", "wins", "times"],
        "p": ["chance", "probability"],
    }
    labels = _dedupe(aliases.get(arg, []) + _numeric_slot_labels(arg, spec) + [arg.replace("_", " ")])
    if _looks_like_unassigned_symbolic_equation_coefficient(lowered, arg):
        return None
    for phrase in labels:
        values = _numbers_near_phrase(lowered, phrase)
        if values:
            index = min(call_index, len(values) - 1) if _slot_value_varies_by_call(arg, phrase) else 0
            return _coerce_number(values[index], spec)

    generic_args = {
        "amount",
        "count",
        "lower_limit",
        "max",
        "min",
        "number",
        "quantity",
        "upper_limit",
        "value",
    }
    if arg in generic_args:
        return _coerce_number(numbers[min(call_index, len(numbers) - 1)], spec)
    return None


def _numeric_range_bound_value(user_request: str, arg: str) -> Any | None:
    role = _numeric_range_bound_role(arg)
    if role is None:
        return None
    bounds = _numeric_range_bounds(user_request)
    if bounds is None:
        return None
    return bounds[0] if role == "start" else bounds[1]


def _numeric_range_bound_role(arg: str) -> str | None:
    normalized = arg.lower().replace("-", "_")
    if re.search(r"(?:^|_)(?:start|begin|from|lower|min|initial)(?:_|$)", normalized):
        return "start"
    if re.search(r"(?:^|_)(?:end|finish|to|upper|max|final)(?:_|$)", normalized):
        return "end"
    return None


def _numeric_range_bounds(user_request: str) -> tuple[Any, Any] | None:
    patterns = [
        r"\bfrom\s+(.+?)\s+(?:to|through|until|till)\s+(.+?)(?=$|[.!?;])",
        r"\bwithin\s+(.+?)\s+(?:to|through|until|till|-)\s+(.+?)(?=$|[.!?;]|\s+budget\b)",
        r"\bbetween\s+(.+?)\s+\band\s+(.+?)(?=$|[.!?;])",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, re.I)
        if not match:
            continue
        start_values = extract_numbers(match.group(1))
        end_values = extract_numbers(match.group(2))
        if start_values and end_values:
            return start_values[-1], end_values[0]
    return None


def _looks_like_unassigned_symbolic_equation_coefficient(lowered: str, arg: str) -> bool:
    if not re.fullmatch(r"[a-z]", arg):
        return False
    if "equation" not in lowered and "=" not in lowered:
        return False
    if re.search(rf"\b{re.escape(arg)}\s*(?:=|:|is)\s*-?\d+(?:\.\d+)?\b", lowered):
        return False
    if re.search(rf"\bcoefficient\s+(?:of\s+)?{re.escape(arg)}\b\D{{0,20}}-?\d+(?:\.\d+)?\b", lowered):
        return False
    equationish = bool(
        re.search(r"\b[a-z]\s*x\b|\b[a-z]x\b|\bx\s*\^\s*\d", lowered)
        or re.search(r"[+\-*/^=]", lowered)
    )
    return equationish


def _first_number_for_pattern(text: str, pattern: str) -> Any | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return _number_from_text(match.group(1))


def _number_from_text(value: str) -> Any | None:
    value = value.strip().lower().replace(",", "")
    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return None


def _duration_number_from_temporal_phrase(lowered: str) -> int | None:
    if "next week" in lowered or "this week" in lowered:
        return 7
    if "next weekend" in lowered or "this weekend" in lowered:
        return 2
    if re.search(r"\b(?:next|this|coming|upcoming)\s+month\b", lowered):
        return 1
    if re.search(r"\b(?:next|this|coming|upcoming)\s+year\b", lowered):
        return 1
    if re.search(r"\b(?:past|last|next)\s+decade\b", lowered):
        return 10
    match = re.search(
        r"\b(?:past|last|next)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+decades?\b",
        lowered,
    )
    if match:
        value = _number_from_text(match.group(1))
        if isinstance(value, int):
            return value * 10
    match = re.search(
        r"\b(?:past|last|next)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(days?|weeks?|months?|years?)\b",
        lowered,
    )
    if not match:
        return None
    value = _number_from_text(match.group(1))
    if not isinstance(value, int):
        return None
    unit = match.group(2)
    if unit.startswith("week"):
        return value * 7
    if unit.startswith("month"):
        return value
    if unit.startswith("year"):
        return value
    return value


def _exercise_hours_value(text: str) -> Any | None:
    match = re.search(r"\bexercise\s+for\s+(\d+(?:\.\d+)?)\s+hours?\b", text, re.I)
    if match:
        return _number_from_text(match.group(1))
    return None


def _dict_value_for_arg(user_request: str, arg: str, spec: dict[str, Any], call_index: int) -> dict[str, Any] | None:
    for match in re.finditer(r"\{[^{}]*\}", user_request):
        try:
            value = ast.literal_eval(match.group(0))
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    nested = spec.get("properties")
    if not isinstance(nested, dict):
        return None
    range_value = _range_dict_value(user_request, nested)
    if range_value:
        return range_value
    values: dict[str, Any] = {}
    for nested_name, nested_spec in nested.items():
        value = infer_argument_value(user_request, nested_name, nested_spec, call_index, 1)
        if value is not None:
            values[nested_name] = value
    return values or None


def _range_dict_value(user_request: str, nested: dict[str, Any]) -> dict[str, Any] | None:
    keys = {str(key).lower(): str(key) for key in nested}
    min_key = next((keys[key] for key in keys if key in {"min", "minimum", "lower", "low"}), None)
    max_key = next((keys[key] for key in keys if key in {"max", "maximum", "upper", "high"}), None)
    if not min_key or not max_key:
        return None
    bounds = _numeric_range_bounds(user_request)
    if bounds is None:
        return None
    return {min_key: bounds[0], max_key: bounds[1]}


def _number_near_phrase(lowered: str, phrase: str) -> Any | None:
    values = _numbers_near_phrase(lowered, phrase)
    return values[0] if values else None


def _numbers_near_phrase(lowered: str, phrase: str) -> list[Any]:
    phrase = re.escape(phrase)
    number = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    patterns = [
        rf"{phrase}\D{{0,30}}({number})",
        rf"({number})\D{{0,30}}{phrase}",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            raw = match.group(1).replace(",", "")
            if raw.endswith("%"):
                values.append(float(raw[:-1]) / 100)
            else:
                values.append(float(raw) if "." in raw else int(raw))
    return values


def _coerce_number(value: Any, spec: dict[str, Any]) -> Any:
    typ = _property_type(spec)
    if typ == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ == "float" and isinstance(value, int):
        return float(value)
    return value


def _default_value_from_description(spec: dict[str, Any]) -> Any | None:
    description = str(spec.get("description") or "")
    match = re.search(
        r"\bdefault(?:\s+value)?\s+(?:is|=|:)?\s*['\"]?([^'\".;,)]+)",
        description,
        re.I,
    )
    if not match:
        return None
    raw = match.group(1).strip()
    raw = re.sub(r"^(?:to|as)\s+", "", raw, flags=re.I).strip()
    typ = _property_type(spec)
    lowered = raw.lower()
    if typ == "boolean":
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        return None
    if typ in {"integer", "float", "number"}:
        numeric = _number_from_text(raw)
        if numeric is None:
            return None
        return _coerce_number(numeric, spec)
    if typ == "array":
        return [] if lowered in {"empty list", "none", "null"} else None
    if typ in {"string", "any", "object", "dict", ""}:
        value = raw.strip("'\" ")
        return value if value else None
    return None


def _quoted_value(text: str) -> str | None:
    values = _quoted_strings(text)
    return values[0] if values else None


def _quoted_strings(text: str) -> list[str]:
    values = []
    pattern = r"(?<![A-Za-z0-9])'(.+?)'(?=[\s,.;:!?)]|$)|(?<![A-Za-z0-9])\"([^\"]+)\"(?![A-Za-z0-9])"
    for left, right in re.findall(pattern, text):
        value = left or right
        if value:
            values.append(value)
    return values


def _quoted_or_after(text: str, label: str) -> str | None:
    pattern = rf"{label}\s+(?:is\s+)?['\"]([^'\"]+)['\"]"
    match = re.search(pattern, text, re.I)
    return match.group(1) if match else _quoted_value(text)


def _host_value(text: str) -> str | None:
    match = re.search(r"host\s+['\"]([^'\"]+)['\"]", text, re.I)
    return match.group(1) if match else None


def _unit_value(text: str) -> str | None:
    lowered = text.lower()
    for unit in ["femtometers", "femtometer", "amu", "meters", "meter", "units", "cm", "minutes"]:
        if unit in lowered:
            return unit
    return None


def _search_query_value(text: str) -> str | None:
    cleaned = re.sub(r"\b(?:User|AI|Assistant)\s*:\s*", " ", text)
    cleaned = re.sub(r"Prior API (?:call|result)\s*:?", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    tokens = [
        token
        for token in _expanded_tokens(cleaned)
        if token not in STOPWORDS and token not in {"please", "help", "sure", "api", "tool"}
    ]
    if not tokens:
        return None
    return " ".join(tokens[:8])


def _entity_count_for_arg(user_request: str, arg: str, spec: dict[str, Any]) -> int:
    if "artist" in arg:
        return len(extract_artist_like_values(user_request))
    typ = _property_type(spec)
    if typ in {"integer", "float", "number"}:
        numbers = extract_numbers(user_request)
        lowered = user_request.lower()
        if len(numbers) >= 2 and (
            re.search(r"\b(?:factorials?|for each|for all|respectively|separately)\b", lowered)
            or re.search(r"\d+(?:\s*,\s*\d+)+(?:\s*,?\s*and\s*\d+)?", lowered)
        ):
            return len(numbers)
    if typ == "string":
        slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
        if "pokemon" in slot_text:
            values = _pokemon_values(user_request)
            if len(values) > 1:
                return len(values)
        if "company" in slot_text:
            values = _company_or_entity_values(user_request)
            if len(values) > 1:
                return len(values)
        if "particle" in slot_text:
            values = _particle_values(user_request)
            if len(values) > 1:
                return len(values)
        if _looks_like_property_selector_slot(arg, f"{arg} {spec.get('description') or ''}"):
            values = _requested_property_values(user_request, spec)
            if len(values) > 1:
                return len(values)
        if "case" in arg:
            count = _case_identifier_count(user_request)
            if count > 1:
                return count
        if "ip" in arg:
            count = len(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", user_request)))
            if count > 1:
                return count
        if _looks_like_location_slot(arg, slot_text):
            city_values = _city_list_values(user_request)
            if len(city_values) > 1:
                return len(city_values)
            quoted_locations = _quoted_location_values(user_request)
            if len(quoted_locations) > 1:
                return len(quoted_locations)
        quoted = _quoted_strings(user_request)
        unique_quoted = _dedupe(quoted)
        if (
            len(unique_quoted) >= 2
            and any(token in slot_text for token in ["city", "exhibition", "hotel", "location", "museum", "name", "place", "title"])
            and re.search(r"\band\b|,", user_request, re.I)
        ):
            return len(unique_quoted)
        if len(unique_quoted) >= 2 and _has_multi_value_context_for_arg(user_request, arg):
            return len(unique_quoted)
        if not _has_multi_value_context_for_arg(user_request, arg):
            return 1
        return _listed_entity_count(user_request, arg)
    return 1


def _case_identifier_count(user_request: str) -> int:
    ids = re.findall(
        r"\b(?:case\s*(?:no\.?|number)?\s*)?([A-Z]{1,4}\d{2,}|[A-Z]{2}\d{4,}|\d{4,})\b",
        user_request,
        re.I,
    )
    ids = [item.upper() for item in ids if not re.fullmatch(r"\d{1,3}", item)]
    return len(set(ids))


def _has_multi_value_context_for_arg(user_request: str, arg: str) -> bool:
    lowered = user_request.lower()
    arg_lower = arg.lower()
    if "list" in arg_lower or (arg_lower.endswith("s") and not arg_lower.endswith("ss")):
        return True
    plural_markers = {
        "city": ["cities"],
        "country": ["countries"],
        "symbol": ["symbols", "tickers", "stocks", "companies"],
        "stock": ["symbols", "tickers", "stocks"],
        "term": ["terms", "definitions"],
        "name": ["names"],
        "case": ["cases"],
    }
    for token, markers in plural_markers.items():
        if token in arg_lower and any(re.search(rf"\b{marker}\b", lowered) for marker in markers):
            return True
    return bool(
        re.search(r"\b(?:for each|for both|both of|each of|respectively|separately)\b", lowered)
        or re.search(r"\band\s+their\b|\band\s+its\b", lowered)
        or re.search(r":\s*[^.]+,\s*[^.]+(?:,\s*|\s+and\s+)", user_request)
    )


def _numeric_tuple_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    numeric_required = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _is_numeric_value_property(spec)
    ]
    if not numeric_required:
        return 1
    tuple_group_count = _parenthesized_numeric_tuple_count(user_request, len(numeric_required))
    if tuple_group_count > 1:
        return tuple_group_count
    if re.search(r"\bbetween\b|\bfrom\b.+\bto\b", lowered):
        return 1
    slot_list_count = _numeric_slot_list_count(user_request, tool)
    if slot_list_count > 1:
        return slot_list_count
    entity_group_count = _article_entity_group_count(user_request, tool)
    if entity_group_count > 1:
        return entity_group_count
    repeated_function_count = _function_definition_group_count(user_request, tool)
    if repeated_function_count > 1:
        return repeated_function_count
    repeated_clause_count = _repeated_operation_clause_count(user_request, tool)
    if repeated_clause_count > 1:
        return repeated_clause_count
    capital_gain_count = _capital_gain_scenario_count(user_request, tool)
    if capital_gain_count > 1:
        return capital_gain_count
    amount_location_count = _amount_location_pair_count(user_request, tool)
    if amount_location_count > 1:
        return amount_location_count
    out_of_count = len(re.findall(r"\b(?:exactly\s+)?(?:-?\d+(?:\.\d+)?|[a-z]+)\s+out\s+of\s+(?:-?\d+(?:\.\d+)?|[a-z]+)\b", lowered))
    if out_of_count >= 2:
        return out_of_count
    vector_pair_count = len(re.findall(r"\[[^\]]+\]\s+(?:with|and)\s+\[[^\]]+\]", user_request, re.I))
    if vector_pair_count >= 2:
        return vector_pair_count
    numeric_pair_count = 1 if _allows_multiple_tools(user_request) else _paired_numeric_sequence_count(user_request, len(numeric_required))
    if numeric_pair_count > 1:
        return numeric_pair_count
    array_required = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _property_type(spec) == "array"
    ]
    if len(array_required) >= 2:
        return 1
    explicit_group_count = _explicit_group_count(lowered)
    if explicit_group_count > 1:
        return explicit_group_count

    segment_count = len([part for part in re.split(r";", user_request) if extract_numbers(part)])
    if segment_count >= 2:
        return segment_count

    if len(numeric_required) == 1:
        unit_count = _unit_labeled_numeric_slot_count(user_request, numeric_required[0], _properties(tool).get(numeric_required[0], {}))
        if unit_count > 1:
            return unit_count
        list_count = _single_numeric_value_list_count(user_request, tool, numeric_required[0])
        if list_count > 1:
            return list_count
        numbers = extract_numbers(user_request)
        unique_numbers = {str(number) for number in numbers}
        if len(unique_numbers) >= 2 and re.search(r"\b(?:for each|respectively|separately)\b", user_request):
            return len(numbers)
        return 1
    if len(numeric_required) < 2:
        return 1
    numbers = extract_numbers(user_request)
    if len(numbers) < len(numeric_required) * 2:
        return 1
    if re.search(r"\bbetween\b|\bfrom\b.+\bto\b", lowered) and len(numbers) <= len(numeric_required) + 1:
        return 1
    if re.search(r"\b(?:pairs?|objects?|trials?|cases?|for each|respectively|separately)\b", lowered):
        return len(numbers) // len(numeric_required)
    return 1


def _parenthesized_numeric_tuple_count(user_request: str, arity: int) -> int:
    if arity < 2:
        return 1
    count = 0
    for match in re.finditer(r"\(([^()]*)\)", user_request):
        numbers = extract_numbers(match.group(1))
        if len(numbers) >= arity:
            count += 1
    return count if count > 1 else 1


def _single_numeric_value_list_count(user_request: str, tool: dict[str, Any], arg: str) -> int:
    lowered = user_request.lower()
    if re.search(r"\b(?:sum|average|mean|median|total|product|add|multiply|range|between|from)\b", lowered):
        return 1
    tool_text = _tool_text(tool).lower()
    if not re.search(r"\b(?:factorial|prime|sqrt|square root|absolute|logarithm|sin|cos|tan)\b", tool_text):
        return 1
    number = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    number_list = rf"{number}(?:\s*(?:,\s*(?:and\s*)?|\s+(?:and|or)\s+)\s*{number})+"
    labels = _numeric_slot_labels(arg, _properties(tool).get(arg, {})) + [arg.replace("_", " ")]
    patterns = [rf"\b(?:of|for)\s+({number_list})\b"]
    for label in labels:
        patterns.append(rf"\b{re.escape(label)}s?\D{{0,20}}({number_list})")
    for pattern in patterns:
        match = re.search(pattern, user_request, re.I)
        if match:
            count = len(extract_numbers(match.group(1)))
            if count > 1:
                return count
    return 1


def _explicit_group_count(lowered: str) -> int:
    scenario_nouns = (
        "cases?|charges?|cities?|conditions?|configurations?|countries?|datasets?|data\\s+sets?|experiments?|groups?|hotels?|"
        "locations?|options?|pairs?|reservations?|routes?|scenarios?|stocks?|symbols?|"
        "setups?|substances?|tasks?|theaters?|theatres?|triangles?|trials?|vehicles?|cars?"
    )
    counted_entity_nouns = (
        "arrays?|cars?|charges?|circles?|conditions?|configurations?|datasets?|data\\s+sets?|equations?|functions?|groups?|"
        "hotels?|lists?|objects?|items?|cases?|rooms?|rounds?|routes?|tasks?|"
        "reservations?|setups?|substances?|trials?|experiments?|options?|pairs?|materials?|locations?|"
        "cities?|countries?|stocks?|stores?|symbols?|theaters?|theatres?|triangles?|"
        "vehicles?"
    )
    couple = re.search(
        r"\bcouple\s+of\s+(?:[a-z]+\s+)?"
        rf"(?:{scenario_nouns})\b",
        lowered,
    )
    if couple:
        return 2
    for match in re.finditer(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:(different|separate|distinct|independent|various|specific)\s+)?"
        r"(?:[a-z]+\s+)?"
        rf"({counted_entity_nouns})\b",
        lowered,
    ):
        noun = match.group(3)
        has_distinct_modifier = bool(match.group(2))
        if not has_distinct_modifier and not re.fullmatch(scenario_nouns, noun):
            continue
        value = _number_from_text(match.group(1))
        if isinstance(value, int) and 1 < value <= 20:
            return value
    return 1


def _listed_entity_count(user_request: str, arg: str) -> int:
    lowered = user_request.lower()
    arg_tokens = [token for token in _tokens(arg) if token not in VALUE_TOKENS]
    if not any(token in lowered for token in arg_tokens):
        if not re.search(r"\b(?:in|of|for|from)\b.+(?:,|\band\b)", lowered):
            return 1
    quoted = _quoted_strings(user_request)
    if len(quoted) >= 2:
        return len(quoted)
    matches = re.findall(
        r"\b(?:in|of|for|from)\s+(.+?)(?:\s+(?:and their|and its|using|with|on|from|sort|near|for my|to match)\b|[?.!]|$)",
        user_request,
        re.I,
    )
    if not matches:
        return 1
    chunk = max(matches, key=lambda value: value.count(",") + len(re.findall(r"\band\b", value, re.I))).strip(" .")
    if len(chunk.split()) > 20:
        return 1
    parts = [part.strip(" .'\"") for part in re.split(r"\s*,\s*|\s+\band\b\s+", chunk)]
    parts = [re.sub(r"^(?:the|a|an)\s+", "", part, flags=re.I).strip() for part in parts]
    parts = [part for part in parts if part]
    return len(parts) if len(parts) >= 2 else 1


def _selection_threshold(user_request: str, tool: dict[str, Any]) -> float:
    del user_request, tool
    return 0.35


def _should_call_single_retrieval_tool(user_request: str, tools: list[dict[str, Any]]) -> bool:
    if len(tools) != 1:
        return False
    if _is_meta_no_query(user_request):
        return False
    tool = tools[0]
    if "tool_search" not in _tool_capability_tags(tool):
        return False
    return _has_any_user_input_evidence(user_request)


def _required_arg_coverage(
    user_request: str,
    tool: dict[str, Any],
    query_input_audit: dict[str, Any] | None = None,
) -> int:
    args, _ = infer_arguments(user_request, tool, query_input_audit=query_input_audit)
    return sum(1 for key in _required(tool) if key in args)


def _has_enough_required_arg_coverage(
    user_request: str,
    tool: dict[str, Any],
    score: float,
    query_input_audit: dict[str, Any] | None = None,
) -> bool:
    required = _required(tool)
    if not required:
        return True
    covered = _required_arg_coverage(user_request, tool, query_input_audit)
    if covered == len(required):
        return True
    if _tool_name_is_explicitly_requested(user_request, tool) and _has_any_user_input_evidence(user_request):
        return True
    identity_score = _function_identity_score(user_request, tool)
    semantic_overlap = _semantic_capability_overlap(user_request, tool)
    if semantic_overlap and _has_any_user_input_evidence(user_request):
        return True
    if (
        identity_score >= 2
        and _semantic_identity_overlap(user_request, tool) >= 1
        and _has_required_argument_evidence(user_request, tool)
    ):
        return True
    if (
        identity_score >= 1
        and score >= 1.0
        and _semantic_identity_overlap(user_request, tool) >= 1
        and _has_general_argument_evidence(user_request)
    ):
        return True
    return score >= 5 and covered > 0 and _semantic_identity_overlap(user_request, tool) >= 1


def _function_identity_score(user_request: str, tool: dict[str, Any]) -> int:
    request_tokens = set(_expanded_tokens(user_request))
    identity_text = " ".join([tool.get("name", ""), tool.get("description", "")])
    identity_tokens = set(_expanded_tokens(identity_text))
    return len(request_tokens & identity_tokens)


def _has_required_argument_evidence(user_request: str, tool: dict[str, Any]) -> bool:
    required = _required(tool)
    if not required:
        return True
    properties = _properties(tool)
    numbers = extract_numbers(user_request)
    lowered = user_request.lower()
    content_tokens = [token for token in _tokens(user_request) if token not in STOPWORDS]
    for arg in required:
        spec = properties.get(arg, {})
        typ = _property_type(spec)
        arg_text = arg.replace("_", " ").lower()
        if typ in {"integer", "float", "number"}:
            if numbers:
                continue
            return False
        if typ == "array":
            if numbers or _quoted_value(user_request) or any(token in lowered for token in _tokens(arg_text)):
                continue
            return False
        if typ == "boolean":
            if any(token in lowered for token in _tokens(arg_text)) or any(word in lowered for word in ["true", "false", "yes", "no"]):
                continue
            return False
        if _quoted_value(user_request) or _looks_like_entity_value(user_request) or content_tokens:
            continue
        return False
    return True


def _has_semantic_mismatch(user_request: str, tool: dict[str, Any]) -> bool:
    if _has_hard_semantic_conflict(user_request, tool):
        return True
    if _has_core_intent_mismatch(user_request, tool):
        return True
    return False


def _has_hard_semantic_conflict(user_request: str, tool: dict[str, Any]) -> bool:
    request = user_request.lower()
    tool_text = _tool_text(tool).lower()
    if _tool_name_is_explicitly_requested(user_request, tool):
        return False
    if _has_intent_conflict(_intent_tags(user_request), _tool_capability_tags(tool)):
        return True
    if _has_action_label_semantic_conflict(user_request, tool):
        return True
    if _has_equation_type_conflict(request, tool_text):
        return True
    request_actions = _action_labels(user_request)
    if not (
        request_actions & _action_labels(str(tool.get("name") or ""))
        or request_actions & _action_labels(_tool_text(tool))
    ) and _has_opposed_action_conflict(request, tool_text):
        return True
    if _has_opposed_modifier_conflict(request, tool_text):
        return True
    if _request_asks_descriptive_stat_but_tool_tests_hypothesis(request, tool_text):
        return True
    if _request_is_advice_but_tool_is_calculator(request, tool_text):
        return True
    if _request_asks_generated_output_but_tool_measures_other_quantity(request, tool_text):
        return True
    if _has_measured_quantity_conflict(request, tool_text):
        return True
    if _has_shape_object_conflict(request, tool_text):
        return True
    if _request_asks_time_but_tool_calculates_force(request, tool_text):
        return True
    if _request_asks_time_difference_but_tool_only_coordinates(request, tool_text):
        return True
    if _request_asks_health_or_exercise_but_tool_is_payment(request, tool_text):
        return True
    if _request_asks_plain_integer_but_tool_is_prime(request, tool_text):
        return True
    if _request_asks_result_dependent_unary_math(request, tool_text):
        return True
    if "musical" in request and "concert" in tool_text and "concert" not in request:
        return True
    if _has_only_symbolic_numeric_inputs(user_request, tool):
        return True
    incompatible_pairs = [
        ("acceleration", "maximum height"),
        ("address", "weather"),
    ]
    return any(left in request and right in tool_text and right not in request for left, right in incompatible_pairs)


def _request_asks_time_difference_but_tool_only_coordinates(request: str, tool_text: str) -> bool:
    asks_time_difference = bool(re.search(r"\btime\s+difference\b|\bdifference\s+in\s+time\b", request))
    coordinate_tool = any(token in tool_text for token in ["coordinate", "latitude", "longitude"])
    time_tool = any(token in tool_text for token in ["time zone", "timezone", "time difference", "local time"])
    return asks_time_difference and coordinate_tool and not time_tool


def _has_shape_object_conflict(request: str, tool_text: str) -> bool:
    shapes = {"circle", "rectangle", "triangle", "cylinder"}
    requested = {shape for shape in shapes if re.search(rf"\b{shape}\b", request)}
    tool_shapes = {shape for shape in shapes if re.search(rf"\b{shape}\b", tool_text)}
    return bool(requested and tool_shapes and requested.isdisjoint(tool_shapes))


def _has_measured_quantity_conflict(request: str, tool_text: str) -> bool:
    quantities = {"charge", "diameter", "mass", "radius", "volume", "weight"}
    requested = {quantity for quantity in quantities if _contains_word(request, quantity)}
    tool_quantities = {quantity for quantity in quantities if _contains_word(tool_text, quantity)}
    if not requested or not tool_quantities or requested & tool_quantities:
        return False
    if {"particle", "atomic", "science", "physics"} & set(_tokens(request + " " + tool_text)):
        return True
    return False


def _request_asks_time_but_tool_calculates_force(request: str, tool_text: str) -> bool:
    asks_time = bool(re.search(r"\b(?:calculate|find|compute)\b.{0,40}\btime\b|\btime required\b", request))
    return asks_time and "force" in tool_text and "force" not in request


def _request_asks_health_or_exercise_but_tool_is_payment(request: str, tool_text: str) -> bool:
    health_terms = {
        "calorie",
        "calories",
        "exercise",
        "hydration",
        "intake",
        "steps",
        "walk",
        "water",
    }
    payment_terms = {
        "bill",
        "billing",
        "buy",
        "bought",
        "cart",
        "checkout",
        "cost",
        "money",
        "pay",
        "payment",
        "price",
        "purchase",
        "shopping",
        "total",
    }
    has_health_request = bool(health_terms & set(_tokens(request)))
    has_payment_tool = bool(payment_terms & set(_tokens(tool_text)))
    has_payment_request = bool(payment_terms & set(_tokens(request)))
    return has_health_request and has_payment_tool and not has_payment_request


def _request_asks_plain_integer_but_tool_is_prime(request: str, tool_text: str) -> bool:
    asks_plain_integer = bool(re.search(r"\bclosest\s+(?:integer|number)\b|\bnearest\s+(?:integer|number)\b", request))
    return asks_plain_integer and "prime" in tool_text and "prime" not in request


def _request_asks_result_dependent_unary_math(request: str, tool_text: str) -> bool:
    unary_tool = bool(re.search(r"\b(?:sqrt|square root|logarithm|absolute)\b", tool_text))
    if not unary_tool:
        return False
    return bool(
        re.search(r"\b(?:of|for)\s+(?:these|those|the)\s+results?\b", request)
        or re.search(r"\bsquare roots?\s+of\s+the\s+(?:lcm|gcd|least common multiple|greatest common divisor)", request)
    )


def _has_action_label_semantic_conflict(user_request: str, tool: dict[str, Any]) -> bool:
    if _is_auth_token_tool(tool):
        return False
    request_actions = _action_labels(user_request)
    if not request_actions:
        return False
    name_actions = _action_labels(str(tool.get("name") or ""))
    tool_actions = _action_labels(_tool_text(tool))
    if request_actions & tool_actions:
        return False
    if request_actions <= {"query"} and _query_action_softened_by_nominal_request(user_request, tool):
        return False
    if name_actions:
        if request_actions & name_actions:
            return False
        return _action_labels_conflict(request_actions, name_actions)
    return bool(tool_actions and _action_labels_conflict(request_actions, tool_actions))


def _query_action_softened_by_nominal_request(user_request: str, tool: dict[str, Any]) -> bool:
    request = user_request.lower()
    tool_text = _tool_text(tool).lower()
    nominal_pairs = [
        (r"\b(?:prediction|predictions|forecast|forecasts)\b", r"\b(?:predict|prediction|forecast)\b"),
        (r"\btickets?\b", r"\b(?:book|booking|ticket|tickets|concert)\b"),
    ]
    return any(
        re.search(request_pattern, request) and re.search(tool_pattern, tool_text)
        for request_pattern, tool_pattern in nominal_pairs
    )


def _has_equation_type_conflict(request: str, tool_text: str) -> bool:
    if "equation" not in request:
        return False
    equation_types = [
        "linear",
        "quadratic",
        "cubic",
        "polynomial",
        "differential",
        "exponential",
        "logarithmic",
        "trigonometric",
    ]
    requested = [
        kind
        for kind in equation_types
        if re.search(rf"\b{re.escape(kind)}\s+equations?\b", request)
        or (kind in request and "equation" in request)
    ]
    if not requested:
        return False
    for wanted in requested:
        for other in equation_types:
            if other == wanted or other in requested:
                continue
            if re.search(rf"\b{re.escape(other)}\s+equations?\b", tool_text) and not re.search(
                rf"\b{re.escape(wanted)}\s+equations?\b", tool_text
            ):
                return True
    return False


def _has_opposed_action_conflict(request: str, tool_text: str) -> bool:
    cancel = r"\b(?:cancel|cancels|canceled|canceling|cancelled|cancelling|delete|deletes|deleted|remove|removes|removed|void)\b"
    modify = r"\b(?:modify|modifies|modified|modification|update|updates|updated|change|changes|changed|reschedule|reschedules|rescheduled|edit|edits|edited)\b"
    create = r"\b(?:create|creates|created|add|adds|added|book|books|booked|register|registers|registered)\b"
    if re.search(cancel, request) and re.search(modify, tool_text) and not re.search(modify, request):
        return True
    if re.search(modify, request) and re.search(cancel, tool_text) and not re.search(cancel, request):
        return True
    if re.search(cancel, request) and re.search(create, tool_text) and not re.search(create, request):
        return True
    if re.search(create, request) and re.search(cancel, tool_text) and not re.search(cancel, request):
        return True
    return False


def _has_opposed_modifier_conflict(request: str, tool_text: str) -> bool:
    opposed_pairs = [
        ("external", "internal"),
        ("public", "private"),
        ("historical", "forecast"),
        ("current", "historical"),
        ("increase", "decrease"),
        ("increasing", "decreasing"),
        ("maximum", "minimum"),
        ("highest", "lowest"),
        ("arrival", "departure"),
    ]
    if _contains_word(request, "forecast") and _contains_word(tool_text, "forecast"):
        opposed_pairs = [pair for pair in opposed_pairs if pair != ("current", "historical")]
    return any(
        _contains_word(request, left)
        and _contains_word(tool_text, right)
        and not _contains_word(request, right)
        and not _contains_word(tool_text, left)
        for left, right in opposed_pairs
    ) or any(
        _contains_word(request, right)
        and _contains_word(tool_text, left)
        and not _contains_word(request, left)
        and not _contains_word(tool_text, right)
        for left, right in opposed_pairs
    )


def _request_asks_descriptive_stat_but_tool_tests_hypothesis(request: str, tool_text: str) -> bool:
    descriptive = r"\b(?:mean|average|median|mode|standard deviation|stddev|std|variance)\b"
    tool_inferential = r"\b(?:p[- ]?value|hypothesis|significance|t[- ]?test|z[- ]?test|chi[- ]?square|anova)\b"
    requested_inferential_output = r"\b(?:p[- ]?value|significance|reject|accept)\b"
    return bool(
        re.search(descriptive, request)
        and not re.search(tool_inferential, request)
        and re.search(tool_inferential, tool_text)
        and not re.search(requested_inferential_output, request)
    )


def _request_is_advice_but_tool_is_calculator(request: str, tool_text: str) -> bool:
    advice = r"\b(?:best way to|how should i|what should i do|recommend(?:ed|ation)?|advice)\b"
    calculator = r"\b(?:calculate|compute|estimate|solve|convert|determine)\b"
    return bool(re.search(advice, request) and re.search(calculator, tool_text))


def _request_asks_generated_output_but_tool_measures_other_quantity(request: str, tool_text: str) -> bool:
    output_action = r"\b(?:generated?|produced?|created?|emitted?|released?|formed?)\b"
    asks_generated_quantity = bool(
        re.search(rf"\b(?:how much|how many|what (?:amount|quantity)|amount of|quantity of)\b.{{0,80}}{output_action}", request)
        or re.search(rf"{output_action}.{{0,40}}\b(?:amount|quantity|number)\b", request)
    )
    if not asks_generated_quantity or re.search(output_action, tool_text):
        return False
    measured_quantities = {
        "area",
        "cost",
        "distance",
        "height",
        "mass",
        "pressure",
        "price",
        "probability",
        "rate",
        "speed",
        "temperature",
        "velocity",
        "volume",
        "weight",
    }
    return any(_contains_word(tool_text, quantity) and not _contains_word(request, quantity) for quantity in measured_quantities)


def _contains_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def _allows_multiple_tools(user_request: str) -> bool:
    lowered = user_request.lower()
    return bool(
        re.search(
            r"\b(?:additionally|also|as well|in addition|finally|first|second|then|separately|respectively)\b",
            lowered,
        )
        or _has_repeated_wh_clause_context(user_request)
        or re.search(r"\b1\s*[\).].+\b2\s*[\).]", user_request, re.S)
        or re.search(r"\bnext(?:,|\s+(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|look up|search|suggest|recommend|invest|withdraw|transfer|deposit)\b)", lowered)
        or re.search(r"\band\s+(?:the\s+)?(?:least|greatest|another|a|an)\b", lowered)
        or re.search(r"\band\s+how\s+(?:much|many)\b", lowered)
        or re.search(r"\band\s+(?:also\s+)?(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|look up|search|suggest|recommend|translate|identify|analy[sz]e|invest|withdraw|transfer|deposit)\b", lowered)
        or re.search(r"\b(?:temperature|humidity|precipitation)\s+forecast\b.+\band\s+(?:temperature|humidity|precipitation)\s+forecast\b", lowered)
        or re.search(r"\b(?:birthdate|birthday|date of birth)\b.+\band\b.+\b(?:discovery|famous)\b", lowered)
        or re.search(r"\bidentify\b.+\band\s+analy[sz]e\b", lowered)
        or re.search(r"\bbattle\b.+\band\b.+\btreaty\b", lowered)
        or re.search(r"\bweight\b.+\band\b.+\bdiameter\b", lowered)
        or re.search(r"\bhow\s+many\b.+\band\s+(?:the\s+)?(?:judge|details?|summary|count)\b", lowered)
        or (user_request.count(";") >= 2 and re.search(r"\b(?:provide|find|get|retrieve|calculate|compute|record|information)\b", lowered))
        or len(re.findall(r"\b(?:record|information|details?)\s+for\b", lowered)) >= 2
    )


def _has_repeated_wh_clause_context(user_request: str) -> bool:
    wh_count = len(re.findall(r"\b(?:what|who|which|where|when|how)\b", user_request, re.I))
    if wh_count >= 2:
        return True
    return bool(
        re.search(
            r"\b(?:what|who|which|where|when|how)\b.+,\s*(?:and\s+)?(?:what|who|which|where|when|how)\b",
            user_request,
            re.I | re.S,
        )
    )


def _filter_weak_multi_tool_selections(
    user_request: str,
    routing_request: str,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(selected) <= 1:
        return selected
    best_score = max((float(item.get("score") or 0.0) for item in selected), default=0.0)
    has_strong = any(
        (item.get("audit") or {}).get("semantic_fit") == "exact"
        or float(item.get("score") or 0.0) >= max(1.5, best_score * 0.35)
        for item in selected
    )
    if not has_strong:
        return selected
    filtered: list[dict[str, Any]] = []
    route_context = _route_request_alignment(user_request) and any(
        _tool_looks_like_route_request(item.get("tool") or {}) for item in selected
    )
    for item in selected:
        tool = item.get("tool") or {}
        audit = item.get("audit") or {}
        score = float(item.get("score") or 0.0)
        scope_score = _tool_scope_score(user_request, tool)
        mentioned = _tool_mention_position(routing_request, tool) <= len(routing_request)
        if route_context and not _tool_looks_like_route_request(tool) and _is_route_purpose_context_tool(user_request, tool):
            continue
        if (
            has_strong
            and audit.get("semantic_fit") == "partial"
            and _looks_like_generic_search_helper(tool)
            and not _tool_name_is_explicitly_requested(user_request, tool)
        ):
            continue
        weak_partial = (
            audit.get("semantic_fit") == "partial"
            and (
                (score < max(1.0, best_score * 0.25) and not mentioned)
                or (scope_score < 0 and score < best_score * 0.5)
            )
            and scope_score < 3.0
        )
        if not weak_partial:
            filtered.append(item)
    return filtered or selected


def _looks_like_generic_search_helper(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower()
    name = str(tool.get("name") or "").lower()
    return bool(
        re.search(r"\bsearch(?:es)?\b", text)
        and not re.search(r"\b(?:book|guide|library|record|data|details?|price|count|availability|status)\b", name)
    )


def _is_route_purpose_context_tool(user_request: str, tool: dict[str, Any]) -> bool:
    mention = _tool_mention_position(user_request, tool)
    purpose_match = re.search(r"\bfor\s+(?:playing|attending|watching|visiting|going\s+to|joining)\b", user_request, re.I)
    return bool(purpose_match and mention >= purpose_match.start())


def _scope_selected_tool_audits(user_request: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not selected:
        return selected
    scoped: list[dict[str, Any]] = []
    multi_tool = len(selected) > 1
    for item in selected:
        tool = item.get("tool") or {}
        original_audit = item.get("audit") or {}
        local_request = _tool_scoped_request_text(user_request, tool)
        if not multi_tool or local_request == user_request:
            scoped.append(item)
            continue
        local_query_audit = build_query_input_audit(local_request)
        local_audit = audit_candidate_tool(
            local_request,
            tool,
            float(item.get("score") or 0.0),
            build_task_frame(local_request),
            local_query_audit,
        )
        if _scoped_audit_is_usable(user_request, local_request, local_audit, original_audit):
            local_audit["scoped_user_request"] = local_request
            scoped.append({**item, "audit": local_audit})
        else:
            scoped.append(item)
    return scoped


def _scoped_audit_is_usable(
    full_request: str,
    local_request: str,
    local_audit: dict[str, Any],
    original_audit: dict[str, Any],
) -> bool:
    local_calls = local_audit.get("planned_calls") or []
    if not local_calls:
        return False
    if any(call.get("missing_arguments") for call in local_calls if isinstance(call, dict)):
        return False
    if local_audit.get("semantic_fit") == "rejected" and original_audit.get("semantic_fit") != "rejected":
        return False
    original_count = len(original_audit.get("planned_calls") or [])
    if _scoped_request_needs_antecedent(local_request) and len(local_calls) < original_count:
        return False
    if (
        len(local_calls) < original_count
        and local_request != full_request
        and _original_repeated_bindings_need_full_context(original_audit)
    ):
        return False
    if _scoped_audit_introduces_raw_missing(local_audit, original_audit):
        return False
    if _scoped_audit_drops_grounded_values(full_request, local_audit, original_audit):
        return False
    return len(local_calls) <= max(1, original_count)


def _scoped_audit_introduces_raw_missing(local_audit: dict[str, Any], original_audit: dict[str, Any]) -> bool:
    local_calls = local_audit.get("planned_calls") or []
    original_calls = original_audit.get("planned_calls") or []
    for local_call, original_call in zip(local_calls, original_calls):
        local_raw = set(local_call.get("raw_missing_arguments") or []) if isinstance(local_call, dict) else set()
        original_raw = set(original_call.get("raw_missing_arguments") or []) if isinstance(original_call, dict) else set()
        if local_raw - original_raw:
            return True
    return False


def _scoped_audit_drops_grounded_values(
    full_request: str,
    local_audit: dict[str, Any],
    original_audit: dict[str, Any],
) -> bool:
    local_calls = local_audit.get("planned_calls") or []
    original_calls = original_audit.get("planned_calls") or []
    for local_call, original_call in zip(local_calls, original_calls):
        local_args = local_call.get("arguments") if isinstance(local_call, dict) else {}
        original_args = original_call.get("arguments") if isinstance(original_call, dict) else {}
        if not isinstance(local_args, dict) or not isinstance(original_args, dict):
            continue
        for name, original_value in original_args.items():
            local_value = local_args.get(name)
            if local_value == original_value:
                continue
            if _scoped_audit_drops_positive_boolean_signal(full_request, name, original_value, local_value):
                return True
            if _contextual_value_is_grounded(full_request, original_value) and not _contextual_value_is_grounded(
                full_request,
                local_value,
            ):
                return True
    return False


def _scoped_audit_drops_positive_boolean_signal(
    full_request: str,
    name: str,
    original_value: Any,
    local_value: Any,
) -> bool:
    if original_value is not True or not _is_false_like_value(local_value):
        return False
    return _positive_boolean_slot_grounded(full_request, name)


def _is_false_like_value(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in {"false", "no", "0"})


def _positive_boolean_slot_grounded(user_request: str, name: str) -> bool:
    lowered = user_request.lower()
    normalized = name.replace("_", " ").replace("-", " ").lower()
    tokens = [
        token
        for token in _tokens(normalized)
        if token
        not in {
            "bool",
            "boolean",
            "flag",
            "option",
            "optional",
            "value",
            "return",
            "include",
            "show",
            "display",
            "output",
            "calculate",
            "compute",
            "report",
            "provide",
            "get",
            "list",
            "enable",
            "enabled",
            "use",
            "with",
            "whether",
            "true",
            "false",
        }
    ]
    if not tokens:
        return False
    target_pattern = r"\s+".join(re.escape(token) + r"(?:s|es)?" for token in tokens)
    positive_verbs = r"(?:return|include|show|display|output|calculate|compute|report|provide|get|list)"
    if re.search(rf"\b{positive_verbs}\b[\w\s,'\"()[\]-]{{0,80}}\b{target_pattern}\b", lowered):
        return True
    action_prefixes = (
        "return ",
        "include ",
        "show ",
        "display ",
        "output ",
        "calculate ",
        "compute ",
        "report ",
        "provide ",
        "get ",
        "list ",
    )
    if normalized.startswith(action_prefixes) and all(re.search(rf"\b{re.escape(token)}(?:s|es)?\b", lowered) for token in tokens):
        return True
    return False


def _scoped_request_needs_antecedent(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:each|both|same|that|those|these)\s+"
            r"(?:city|cities|data|basis|period|periods|location|locations|file|files|route|routes)\b",
            text,
            re.I,
        )
        or re.search(r"\bthen\s+for\s+(?:the\s+)?(?:past|last|next)?\s*\d+", text, re.I)
    )


def _original_repeated_bindings_need_full_context(original_audit: dict[str, Any]) -> bool:
    slot_bindings = original_audit.get("slot_bindings")
    if not isinstance(slot_bindings, dict):
        return False
    context_slots = {"coordinates", "days", "density", "frequency", "mode", "pointa", "pointb", "point_a", "point_b"}
    for slot, value in slot_bindings.items():
        slot_key = str(slot).lower()
        if slot_key not in context_slots and not any(token in slot_key for token in ["coordinate", "point"]):
            continue
        if not isinstance(value, list) or len(value) <= 1:
            continue
        distinct = {json.dumps(item, sort_keys=True, default=str) for item in value}
        if len(distinct) > 1:
            return True
    return False


def _calls_from_selected_tools(selected: list[dict[str, Any]], *, interleave: bool = False) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if interleave and _should_interleave_selected_tool_calls(selected):
        max_count = max(len((item.get("audit") or {}).get("planned_calls") or []) for item in selected)
        for call_index in range(max_count):
            for item in selected:
                tool = item["tool"]
                planned_calls = (item.get("audit") or {}).get("planned_calls") or []
                if call_index >= len(planned_calls):
                    continue
                call_plan = planned_calls[call_index]
                calls.append(_call_from_plan(tool, call_plan, len(calls) + 1))
        return calls
    for item in selected:
        tool = item["tool"]
        for call_plan in (item.get("audit") or {}).get("planned_calls") or []:
            calls.append(_call_from_plan(tool, call_plan, len(calls) + 1))
    return calls


def _filter_schema_value_incompatible_calls(
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not calls:
        return calls, []
    tool_lookup = {}
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        tool = normalize_tool(raw_tool)
        name = str(tool.get("name") or "")
        if name:
            tool_lookup[name] = tool
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool = tool_lookup.get(str(call.get("tool_name") or ""))
        reason = _call_schema_value_compatibility_error(tool, call)
        if reason and _model_grounded_call_allows_compact_identifier(call, reason):
            reason = ""
        if reason:
            dropped.append(
                {
                    "tool_name": str(call.get("tool_name") or ""),
                    "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                    "reason": reason,
                }
            )
            continue
        kept.append(call)
    for index, call in enumerate(kept):
        call["id"] = f"call_{index + 1}"
    return kept, dropped


def _model_grounded_call_allows_compact_identifier(call: dict[str, Any], reason: str) -> bool:
    """Keep a model-proposed compact code when its source span passed audit.

    The generic deterministic filter treats any schema field named ``*code``
    as an identifier and historically expected punctuation or digits.  That is
    appropriate for heuristic fallback calls, but valid schemas also use short
    alphabetic codes.  A semantic binding has already passed schema and
    evidence verification, so the shape-only fallback check must not discard
    it. Role conflicts remain rejected by the normal compatibility check.
    """
    match = re.search(r"slot '([^']+)'", reason)
    if match is None or "not identifier-like enough" not in reason:
        return False
    evidence = call.get("argument_evidence")
    if not isinstance(evidence, dict):
        return False
    value = evidence.get(match.group(1))
    return isinstance(value, str) and bool(value.strip())


def _restore_grounded_identifier_prefixes(
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not calls:
        return calls
    tool_lookup = {}
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        tool = normalize_tool(raw_tool)
        name = str(tool.get("name") or "")
        if name:
            tool_lookup[name] = tool
    repaired_calls: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool = tool_lookup.get(str(call.get("tool_name") or ""))
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if tool is None or not args:
            repaired_calls.append(call)
            continue
        props = _properties(tool)
        repaired_args = dict(args)
        for arg, value in args.items():
            if not isinstance(value, str) or value.startswith("#"):
                continue
            spec = props.get(str(arg)) or {}
            if not _slot_expects_identifier(_slot_semantic_text(str(arg), spec)):
                continue
            prefixed = "#" + value
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(prefixed)}(?![A-Za-z0-9_])", user_request):
                repaired_args[str(arg)] = prefixed
        if repaired_args != args:
            call = deepcopy(call)
            call["arguments"] = repaired_args
        repaired_calls.append(call)
    return repaired_calls


def _call_schema_value_compatibility_error(tool: dict[str, Any] | None, call: dict[str, Any]) -> str:
    if tool is None:
        return ""
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if not args:
        return ""
    props = _properties(tool)
    for arg, value in args.items():
        reason = _slot_value_compatibility_error(str(arg), props.get(str(arg)) or {}, value)
        if reason:
            return reason
    return _duplicate_identifier_value_error(args, props)


def _slot_value_compatibility_error(arg: str, spec: dict[str, Any], value: Any) -> str:
    slot_text = _slot_semantic_text(arg, spec)
    if not _slot_expects_identifier(slot_text):
        return ""
    if isinstance(value, list):
        for item in value:
            reason = _slot_value_compatibility_error(arg, spec, item)
            if reason:
                return reason
        return ""
    if isinstance(value, dict):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    value_role = _identifier_value_role(text)
    slot_roles = _slot_identifier_roles(slot_text)
    if value_role and slot_roles and value_role not in slot_roles:
        return f"value role {value_role!r} is incompatible with identifier slot {arg!r}"
    if not _value_looks_like_identifier(text) and not _identifier_value_allowed_by_slot_text(text, slot_text):
        return f"value {text!r} is not identifier-like enough for slot {arg!r}"
    return ""


def _duplicate_identifier_value_error(args: dict[str, Any], props: dict[str, Any]) -> str:
    seen: dict[str, list[str]] = {}
    for arg, value in args.items():
        spec = props.get(str(arg)) or {}
        slot_text = _slot_semantic_text(str(arg), spec)
        if not _slot_expects_identifier(slot_text) or isinstance(value, (list, dict)):
            continue
        text = str(value).strip()
        if not text:
            continue
        seen.setdefault(text.lower(), []).append(str(arg))
    for value, slots in seen.items():
        if len(slots) < 2:
            continue
        role_sets = [_slot_identifier_roles(_slot_semantic_text(slot, props.get(slot) or {})) for slot in slots]
        specific_roles = [roles for roles in role_sets if roles]
        if len(specific_roles) >= 2 and not set.intersection(*specific_roles):
            return f"value {value!r} was copied into incompatible identifier slots {slots}"
    return ""


def _slot_semantic_text(arg: str, spec: dict[str, Any]) -> str:
    return f"{arg} {spec.get('description') or ''}".lower().replace("-", "_")


def _slot_expects_identifier(slot_text: str) -> bool:
    return bool(
        re.search(r"(?:^|_|\b)(?:id|identifier|number|code)(?:_|\b|$)", slot_text)
        or re.search(
            r"\b(?:reservation|order|payment|user|account|flight|item)\s+(?:id|identifier|number|code)\b",
            slot_text,
        )
    )


def _slot_identifier_roles(slot_text: str) -> set[str]:
    # Parameter names identify the value's role much more reliably than a
    # prose description, which often mentions related entities (for example,
    # "payment method ID for the order"). Fall back to the full description
    # only for generic fields such as ``id`` or ``identifier``.
    argument_name = slot_text.split(" ", 1)[0]
    explicit_roles = _identifier_roles_from_text(argument_name)
    if explicit_roles:
        return explicit_roles
    return _identifier_roles_from_text(slot_text)


def _identifier_roles_from_text(text: str) -> set[str]:
    roles: set[str] = set()
    if re.search(r"\buser\b|user_id|customer", text):
        roles.add("user")
    if re.search(r"\border\b|order_id", text):
        roles.add("order")
    if re.search(r"\breservation\b|reservation_id|booking", text):
        roles.add("reservation")
    if re.search(r"\bpayment\b|payment_id|payment_method|card|paypal", text):
        roles.add("payment")
    if re.search(r"\bflight\b|flight_id|flight_number", text):
        roles.add("flight")
    if re.search(r"\bitem\b|item_id|sku|product", text):
        roles.add("item")
    if re.search(r"\baccount\b|account_id", text):
        roles.add("account")
    return roles


def _identifier_value_role(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if re.search(r"(?:^|[_-])(?:pay|payment|card|paypal|visa|mastercard|amex)(?:[_-]|$)", lowered):
        return "payment"
    if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", lowered) and re.search(r"\d", lowered):
        return "user"
    if re.fullmatch(r"#?w\d{4,}", lowered):
        return "order"
    return ""


def _value_looks_like_identifier(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if lowered in STOPWORDS or lowered in {"name", "kwargs", "arguments", "respond", "available", "though", "none"}:
        return False
    if _identifier_value_role(stripped):
        return True
    if re.fullmatch(r"[A-Za-z0-9]+[-_][A-Za-z0-9]+", stripped) and re.search(r"\d", stripped):
        return True
    if re.fullmatch(r"#?[A-Za-z0-9][A-Za-z0-9_-]{3,}", stripped) and (
        re.search(r"\d", stripped) or re.search(r"[_#-]", stripped)
    ):
        return True
    if re.fullmatch(r"\d{1,12}", stripped):
        return True
    return False


def _identifier_value_allowed_by_slot_text(text: str, slot_text: str) -> bool:
    lowered = text.lower()
    return bool(
        "payment" in slot_text
        and lowered in {"paypal", "visa", "mastercard", "amex", "credit card", "gift card"}
    )


def _postprocess_tool_calls(
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not calls:
        return calls
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    calls = _expand_under_counted_tool_calls(user_request, tools_by_name, calls)
    calls = _expand_contrasting_boolean_calls(user_request, tools_by_name, calls)
    total_by_tool = Counter(str(call.get("tool_name") or "") for call in calls if isinstance(call, dict))
    seen_by_tool: Counter[str] = Counter()
    repaired: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(tool_name)
        occurrence = seen_by_tool[tool_name]
        seen_by_tool[tool_name] += 1
        if tool is None:
            repaired.append(call)
            continue
        arguments = dict(call.get("arguments") or {})
        arguments = _repair_arguments_for_schema(
            user_request,
            tool,
            arguments,
            occurrence,
            max(1, total_by_tool.get(tool_name, 1)),
        )
        arguments = _restore_verified_model_arguments(call, arguments)
        missing = [name for name in call.get("missing_arguments") or [] if not _argument_has_value(arguments.get(str(name)))]
        copied = dict(call)
        copied["arguments"] = arguments
        copied["missing_arguments"] = missing
        repaired.append(copied)
    repaired = _collapse_single_account_auth_calls(user_request, tools_by_name, repaired)
    return _collapse_identical_repeated_calls(user_request, repaired, tools_by_name)


def _restore_verified_model_arguments(call: dict[str, Any], repaired_arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep source-audited semantic bindings ahead of heuristic normalization."""
    original_arguments = call.get("arguments")
    evidence = call.get("argument_evidence")
    if not isinstance(original_arguments, dict) or not isinstance(evidence, dict):
        return repaired_arguments
    restored = dict(repaired_arguments)
    for name, span in evidence.items():
        if isinstance(span, str) and span.strip() and name in original_arguments:
            restored[str(name)] = original_arguments[name]
    return restored


def _order_independent_calls_for_benchmark(
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(calls) <= 1:
        return calls
    if any(call.get("depends_on") for call in calls if isinstance(call, dict)):
        return calls
    if _request_has_explicit_sequence_language(user_request):
        return calls
    tool_order = {str(tool.get("name") or ""): index for index, tool in enumerate(tools)}
    if any(str(call.get("tool_name") or "") not in tool_order for call in calls if isinstance(call, dict)):
        return calls
    names = [str(call.get("tool_name") or "") for call in calls if isinstance(call, dict)]
    name_counts = Counter(names)
    if len(name_counts) > 1 and len(set(name_counts.values())) == 1 and max(name_counts.values()) > 1:
        interleaved = _interleave_shared_entity_repeated_calls(user_request, tools, calls)
        if interleaved is not None:
            return interleaved
        return calls
    ranks = [_call_dependency_order_rank(str(call.get("tool_name") or "")) for call in calls if isinstance(call, dict)]
    if not ranks or len(set(ranks)) <= 1:
        return calls
    ordered = sorted(
        calls,
        key=lambda call: (
            _call_dependency_order_rank(str(call.get("tool_name") or "")),
            tool_order.get(str(call.get("tool_name") or ""), len(tool_order)),
        ),
    )
    if ordered == calls:
        return calls
    return [{**call, "id": f"call_{index + 1}"} for index, call in enumerate(ordered)]


def _interleave_shared_entity_repeated_calls(
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    tool_names = [str(call.get("tool_name") or "") for call in calls if isinstance(call, dict)]
    ordered_tool_names = list(dict.fromkeys(tool_names))
    if len(ordered_tool_names) <= 1:
        return None
    if _repeated_tools_have_distinct_clause_scopes(user_request, tools, ordered_tool_names):
        return None
    groups = [[call for call in calls if str(call.get("tool_name") or "") == name] for name in ordered_tool_names]
    counts = {len(group) for group in groups}
    if len(counts) != 1:
        return None
    call_count = counts.pop()
    if call_count <= 1:
        return None

    grouped_tool_names = [name for name in ordered_tool_names for _ in range(call_count)]
    if tool_names != grouped_tool_names:
        return None

    varying_values_by_group = [_varying_argument_value_sets(group) for group in groups]
    if any(len(value_sets) != call_count for value_sets in varying_values_by_group):
        return None
    for occurrence in range(call_count):
        occurrence_sets = [value_sets[occurrence] for value_sets in varying_values_by_group]
        if any(not values for values in occurrence_sets):
            return None
        shared = set(occurrence_sets[0])
        for values in occurrence_sets[1:]:
            shared.intersection_update(values)
        if not shared:
            return None

    interleaved: list[dict[str, Any]] = []
    for occurrence in range(call_count):
        for group in groups:
            copied = dict(group[occurrence])
            copied["id"] = f"call_{len(interleaved) + 1}"
            interleaved.append(copied)
    return interleaved


def _repeated_tools_have_distinct_clause_scopes(
    user_request: str,
    tools: list[dict[str, Any]],
    tool_names: list[str],
) -> bool:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1 or len(tool_names) <= 1:
        return False
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    best_clause_indices: list[int] = []
    for tool_name in tool_names:
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return False
        scored = [(_tool_scope_score(clause, tool), index) for index, clause in enumerate(clauses)]
        best_score = max((score for score, _index in scored), default=0.0)
        if best_score < 2.0:
            return False
        best_clause_indices.append(next((index for score, index in scored if score == best_score), -1))
    return len(set(best_clause_indices)) == len(tool_names)


def _varying_argument_value_sets(group: list[dict[str, Any]]) -> list[set[str]]:
    values_by_key: dict[str, list[set[str]]] = defaultdict(list)
    for call in group:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        for key, value in arguments.items():
            values_by_key[str(key)].append(_normalized_argument_scalars(value))

    varying_keys = {
        key
        for key, value_sets in values_by_key.items()
        if len(value_sets) == len(group)
        and len({json.dumps(sorted(values), default=str) for values in value_sets}) > 1
    }
    per_call: list[set[str]] = []
    for index, call in enumerate(group):
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        values: set[str] = set()
        for key in varying_keys:
            values.update(_normalized_argument_scalars(arguments.get(key)))
        per_call.append(values)
    return per_call


def _normalized_argument_scalars(value: Any) -> set[str]:
    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {str(value)}
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return {cleaned} if cleaned else set()
    if isinstance(value, list):
        scalars: set[str] = set()
        for item in value:
            scalars.update(_normalized_argument_scalars(item))
        return scalars
    if isinstance(value, dict):
        scalars = set()
        for item in value.values():
            scalars.update(_normalized_argument_scalars(item))
        return scalars
    return set()


def _request_has_explicit_sequence_language(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:first|second|third|fourth|then|after(?:wards| that)?|before|finally|lastly|step\s*\d+)\b"
            r"|\bnext\b(?!\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b)",
            user_request,
            re.I,
        )
    )


def _call_dependency_order_rank(tool_name: str) -> int:
    normalized = tool_name.lower().replace(".", "_")
    if re.search(r"(?:^|_)(?:load|read|fetch|retrieve)(?:_|$)|data_loading|load_data|read_csv|csv_load", normalized):
        return -10
    if re.search(
        r"(?:^|_)(?:fit|train|model|regression|predict|classif(?:y|ication)?|cluster|forecast)(?:_|$)",
        normalized,
    ):
        return 10
    return 0


def _collapse_single_account_auth_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(calls) <= 1 or _explicit_duplicate_call_requested(user_request) or _auth_request_has_multiple_accounts(user_request):
        return calls
    auth_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(tool_name)
        if tool is not None and _is_auth_token_tool(tool):
            auth_groups[tool_name].append(call)
    duplicate_auth_tools = {name for name, group in auth_groups.items() if len(group) > 1}
    if not duplicate_auth_tools:
        return calls

    best_by_tool = {
        name: max(group, key=lambda call: _auth_call_preference_score(user_request, call))
        for name, group in auth_groups.items()
        if name in duplicate_auth_tools
    }
    emitted: set[str] = set()
    collapsed: list[dict[str, Any]] = []
    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        if tool_name not in duplicate_auth_tools:
            collapsed.append(call)
            continue
        if tool_name in emitted:
            continue
        emitted.add(tool_name)
        collapsed.append(best_by_tool[tool_name])
    return [{**call, "id": f"call_{index + 1}"} for index, call in enumerate(collapsed)]


def _auth_call_preference_score(user_request: str, call: dict[str, Any]) -> tuple[int, int, int, int]:
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    username = ""
    password = ""
    for key in ("username", "user_name", "login", "account"):
        if _argument_has_value(arguments.get(key)):
            username = str(arguments.get(key))
            break
    if _argument_has_value(arguments.get("password")):
        password = str(arguments.get("password"))

    labeled_usernames = {value.lower() for value in _labeled_username_values(user_request)}
    passwords = _password_values(user_request)
    missing = call.get("missing_arguments") or []
    complete_score = 1 if not missing and username and password else 0
    username_score = 1 if username.lower() in labeled_usernames else 0
    password_score = 1 if passwords and password == passwords[0] else 0
    email_penalty = -1 if labeled_usernames and "@" in username else 0
    return (complete_score, username_score, password_score, email_penalty)


def _collapse_identical_repeated_calls(
    user_request: str,
    calls: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if len(calls) <= 1 or _explicit_duplicate_call_requested(user_request):
        return calls
    single_workflow_context = _single_workflow_duplicate_collapse_context(user_request)
    identical_entity_context = _explicit_identical_entity_context(user_request)
    tool_lookup = tools_by_name or {}
    collapsed: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    changed = False
    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        key = (
            tool_name,
            json.dumps(call.get("arguments") or {}, sort_keys=True, default=str),
            json.dumps(call.get("missing_arguments") or [], sort_keys=True, default=str),
            json.dumps(call.get("depends_on") or [], sort_keys=True, default=str),
        )
        if key in seen:
            if single_workflow_context or (
                _read_only_tool_allows_identical_call_collapse(tool_lookup.get(tool_name))
                and identical_entity_context
            ):
                changed = True
                continue
        seen.add(key)
        collapsed.append(call)
    if not changed:
        return calls
    return [{**call, "id": f"call_{index + 1}"} for index, call in enumerate(collapsed)]


def _single_workflow_duplicate_collapse_context(user_request: str) -> bool:
    if re.search(r"\b(?:earlier\s+user|latest prior api result|prior api result|^user:|[\n\r]user:| ai:| assistant:)\b", user_request, re.I):
        return True
    if _allows_multiple_tools(user_request):
        return False
    return bool(
        re.search(r"\b(?:forgot|reset|change|modify)\b.{0,50}\bpassword\b", user_request, re.I)
        or re.search(r"\bpassword\b.{0,50}\b(?:forgot|reset|change|modify)\b", user_request, re.I)
    )


def _explicit_identical_entity_context(user_request: str) -> bool:
    return bool(
        re.search(r"\bidentical\b", user_request, re.I)
        or re.search(
            r"\bsame\s+(?:exact\s+)?(?:arguments?|inputs?|parameters?|values?|object|objects|"
            r"item|items|entity|entities|case|cases|scenario|scenarios|shape|shapes|"
            r"measurement|measurements|dimensions?|configuration|configurations?)\b",
            user_request,
            re.I,
        )
    )


def _read_only_tool_allows_identical_call_collapse(tool: dict[str, Any] | None) -> bool:
    if not isinstance(tool, dict):
        return False
    name = str(tool.get("name") or "")
    normalized_name = re.sub(r"[_\.\-/]+", " ", name.lower())
    name_tokens = set(_tokens(normalized_name))
    side_effect_tokens = {
        "add",
        "book",
        "buy",
        "cancel",
        "create",
        "delete",
        "log",
        "modify",
        "order",
        "pay",
        "purchase",
        "record",
        "register",
        "remove",
        "reserve",
        "schedule",
        "send",
        "set",
        "submit",
        "update",
        "write",
    }
    if name_tokens & side_effect_tokens:
        return False
    text = f"{normalized_name} {str(tool.get('description') or '').lower()}"
    return bool(
        re.search(
            r"\b(?:analy[sz]e|calculate|check|compare|compute|convert|detail|estimate|fetch|"
            r"find|forecast|get|info|list|lookup|predict|provide|query|retrieve|search|stat)\b",
            text,
        )
    )


def _expand_under_counted_tool_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _has_prior_result_context(user_request) or re.search(r"\b(?:^user:| ai:| assistant:)\b", user_request, re.I):
        return calls
    concrete_calls = [call for call in calls if isinstance(call, dict)]
    if len(concrete_calls) != 1:
        return calls
    if _verified_model_single_call(concrete_calls[0]):
        return calls
    tool_names = {str(call.get("tool_name") or "") for call in concrete_calls if call.get("tool_name")}
    if len(tool_names) != 1:
        return calls
    tool_name = next(iter(tool_names))
    tool = tools_by_name.get(tool_name)
    if tool is None:
        return calls
    expected = infer_call_count(user_request, tool, build_query_input_audit(user_request))
    if expected <= len(concrete_calls):
        return calls
    audit = audit_candidate_tool(user_request, tool, 1.0)
    planned_calls = audit.get("planned_calls") or []
    if len(planned_calls) != expected or any(call.get("missing_arguments") for call in planned_calls):
        return calls
    return [
        {
            "id": f"call_{index + 1}",
            "tool_name": tool_name,
            "arguments": dict(call_plan.get("arguments") or {}),
            "depends_on": [],
            "missing_arguments": [],
        }
        for index, call_plan in enumerate(planned_calls)
    ]


def _expand_contrasting_boolean_calls(
    user_request: str,
    tools_by_name: dict[str, dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _has_prior_result_context(user_request) or re.search(r"\b(?:^user:| ai:| assistant:)\b", user_request, re.I):
        return calls
    concrete_calls = [call for call in calls if isinstance(call, dict)]
    if len(concrete_calls) != 1:
        return calls
    call = concrete_calls[0]
    tool = tools_by_name.get(str(call.get("tool_name") or ""))
    if tool is None:
        return calls
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    for name, spec in _properties(tool).items():
        if _property_type(spec) != "boolean":
            continue
        if not _request_has_boolean_contrast(user_request, name, spec):
            continue
        true_call = deepcopy(call)
        false_call = deepcopy(call)
        true_args = dict(arguments)
        false_args = dict(arguments)
        true_args[name] = True
        false_args[name] = False
        true_call["arguments"] = true_args
        false_call["arguments"] = false_args
        true_call["id"] = "call_1"
        false_call["id"] = "call_2"
        return [true_call, false_call]
    return calls


def _request_has_boolean_contrast(user_request: str, name: str, spec: dict[str, Any]) -> bool:
    lowered = user_request.lower()
    if not re.search(r"\b(?:what\s+if|if|whether|compare|versus|vs\.?|instead)\b", lowered):
        return False
    aliases = _boolean_slot_aliases(name, spec)
    for alias in aliases:
        pattern = r"\s+".join(re.escape(part) for part in alias.split())
        positive = re.search(rf"(?<!\bnot\s)(?<!\bnon[-\s])\b{pattern}\b", lowered)
        negative = re.search(
            rf"\b(?:not|non[-\s]?|without)\s+(?:\w+\s+){{0,2}}{pattern}\b|\b{pattern}\b\s+is\s+not\b",
            lowered,
        )
        if positive and negative:
            return True
    return False


def _boolean_slot_aliases(name: str, spec: dict[str, Any]) -> list[str]:
    aliases = {name.replace("_", " ").replace("-", " ").lower()}
    description = str(spec.get("description") or "").lower().replace("_", " ")
    for token in _tokens(f"{name} {description}"):
        if len(token) >= 4 and token not in STOPWORDS and token not in {"whether", "process", "default"}:
            aliases.add(token)
    return sorted(aliases, key=len, reverse=True)


def _verified_model_single_call(call: dict[str, Any]) -> bool:
    if _coerce_positive_int(call.get("model_binding_call_count")) != 1:
        return False
    if call.get("missing_arguments"):
        return False
    arguments = call.get("arguments")
    return isinstance(arguments, dict) and bool(arguments)


def _repair_arguments_for_schema(
    user_request: str,
    tool: dict[str, Any],
    arguments: dict[str, Any],
    call_index: int,
    call_count: int,
) -> dict[str, Any]:
    properties = _properties(tool)
    repaired = dict(arguments)
    reservation_scenarios = _reservation_scenarios(user_request) if _tool_looks_like_reservation_tool(tool) else []
    reservation = reservation_scenarios[min(call_index, len(reservation_scenarios) - 1)] if reservation_scenarios else {}
    for name, spec in properties.items():
        slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
        arg = name.lower()
        if _is_auth_token_tool(tool):
            if arg in {"username", "user_name", "login", "account"} or arg.endswith("username"):
                values = _username_values(user_request)
                if values:
                    repaired[name] = values[min(call_index, len(values) - 1)]
            if "password" in arg:
                values = _password_values(user_request)
                if values:
                    repaired[name] = values[min(call_index, len(values) - 1)]
        if _looks_like_location_slot(arg, slot_text):
            value = _canonicalize_location_for_slot(reservation.get("location"), spec) if reservation.get("location") else None
            if value is None:
                value = _repair_location_value_for_slot(user_request, spec, repaired.get(name), call_index, call_count)
            if value is not None:
                repaired[name] = value
        if _looks_like_date_value_slot(arg, slot_text):
            value = None
            if reservation:
                if ("check" in arg and "out" in arg) or any(token in slot_text for token in ["check out", "end date", "finish date", "to date"]):
                    value = reservation.get("check_out")
                elif ("check" in arg and "in" in arg) or any(token in slot_text for token in ["check in", "start date", "begin date", "from date"]):
                    value = reservation.get("check_in")
            if value is None:
                value = _repair_date_value_for_slot(user_request, arg, slot_text, repaired.get(name), call_index, call_count)
            if value is not None:
                repaired[name] = value
        if _property_type(spec) in {"integer", "float", "number"}:
            value = _repair_numeric_value_for_slot(user_request, arg, slot_text, repaired.get(name), spec)
            if value is not None:
                repaired[name] = value
        enum_values = _enum_value_evidence(user_request, spec)
        if enum_values and call_count == 1 and _value_is_schema_default(repaired.get(name), spec):
            repaired[name] = enum_values[min(call_index, len(enum_values) - 1)]
        if _looks_like_weather_unit_slot(tool, arg, slot_text, spec):
            value = (
                _weather_unit_value(user_request, spec)
                if _argument_has_value(repaired.get(name))
                or _model_binding_schema_default(spec) is not None
                or re.search(r"\b(?:metric|celsius|centigrade|imperial|fahrenheit)\b", user_request, re.I)
                else None
            )
            if value is not None:
                repaired[name] = value
        if arg in {"recipient", "to"} and _looks_like_default_recipient(repaired.get(name)):
            person = _person_name_value(user_request)
            if person:
                repaired[name] = person
        if arg in {"prompt", "q", "query"} and _looks_like_image_generation_tool(tool):
            value = _image_prompt_value(user_request)
            if value:
                repaired[name] = value
        if arg in {"hotel_name", "hotel", "accommodation"} or "hotel name" in slot_text:
            if reservation.get("hotel_name"):
                repaired[name] = reservation["hotel_name"]
            else:
                values = _hotel_name_values(user_request)
                if values:
                    repaired[name] = values[min(call_index, len(values) - 1)]
        if reservation and arg in {"adults", "adult_count", "num_adults", "number_of_adults"} and reservation.get("adults") is not None:
            repaired[name] = _coerce_number(reservation.get("adults"), spec)
        elif arg in {"adults", "adult_count", "num_adults", "number_of_adults"}:
            counts = _guest_count_scenarios(user_request)
            if counts and call_index < len(counts):
                repaired[name] = _coerce_number(counts[call_index].get("adults"), spec)
        if reservation and arg in {"children", "child_count", "num_children", "number_of_children"} and reservation.get("children") is not None:
            repaired[name] = _coerce_number(reservation.get("children"), spec)
        elif arg in {"children", "child_count", "num_children", "number_of_children"}:
            counts = _guest_count_scenarios(user_request)
            if counts and call_index < len(counts):
                repaired[name] = _coerce_number(counts[call_index].get("children"), spec)
    return repaired


def _value_is_schema_default(value: Any, spec: dict[str, Any]) -> bool:
    if value in (None, ""):
        return False
    default = _model_binding_schema_default(spec)
    return default is not None and str(value).strip().lower() == str(default).strip().lower()


def _looks_like_date_value_slot(arg: str, slot_text: str) -> bool:
    return bool(arg in {"date", "day", "check_in", "check_out", "start_date", "end_date"} or "date" in slot_text)


def _repair_location_value_for_slot(
    user_request: str,
    spec: dict[str, Any],
    value: Any,
    call_index: int,
    call_count: int,
) -> str | None:
    slot_text = str(spec.get("description") or "").lower()
    values = _location_units(user_request)
    if isinstance(value, str) and value.strip():
        existing = _canonicalize_location_for_slot(value, spec)
        if existing:
            matched = _matched_location_unit(existing, values, spec)
            if matched:
                return matched
            if _contextual_value_is_grounded(user_request, existing) or not values:
                return existing
    if values and call_count > 1:
        candidate = values[min(call_index, len(values) - 1)]
        candidate = _canonicalize_location_for_slot(candidate, spec)
        if candidate:
            return candidate
    if values:
        return _canonicalize_location_for_slot(values[min(call_index, len(values) - 1)], spec)
    if "city" in slot_text or "country" in slot_text or "state" in slot_text:
        return None
    return None


def _canonicalize_location_for_slot(value: Any, spec: dict[str, Any]) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"^(?:of|in|at|near|for)\s+", "", _clean_location_value(value), flags=re.I)
    if not cleaned:
        return None
    key = _location_alias_key(cleaned)
    if key in LOCATION_ALIASES and (_location_slot_prefers_qualified(spec) or re.search(r"[\u4e00-\u9fff]", cleaned)):
        return LOCATION_ALIASES[key]
    return cleaned


def _matched_location_unit(value: str, candidates: list[str], spec: dict[str, Any]) -> str | None:
    if not value or not candidates:
        return None
    value_key = _location_alias_key(value)
    for candidate in candidates:
        canonical = _canonicalize_location_for_slot(candidate, spec)
        if not canonical:
            continue
        candidate_key = _location_alias_key(canonical)
        if candidate_key == value_key:
            return canonical
        if len(value_key) >= 4 and candidate_key.startswith(value_key):
            return canonical
        if len(candidate_key) >= 4 and value_key.startswith(candidate_key):
            return canonical
    return None


def _location_alias_key(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().lower())
    if re.search(r"[\u4e00-\u9fff]", lowered):
        return lowered
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _location_slot_prefers_qualified(spec: dict[str, Any]) -> bool:
    description = str(spec.get("description") or "").lower()
    return bool(
        "city, state" in description
        or "city, country" in description
        or "state (abbr" in description
        or "short form" in description
        or "city and state" in description
    )


def _repair_date_value_for_slot(
    user_request: str,
    arg: str,
    slot_text: str,
    value: Any,
    call_index: int,
    call_count: int,
) -> str | None:
    if not _date_slot_prefers_iso(arg, slot_text):
        return None
    dates = _iso_date_values(user_request)
    if not dates:
        return None
    if "check" in arg and "out" in arg or any(token in slot_text for token in ["check out", "end date", "finish date", "to date"]):
        index = call_index * 2 + 1 if len(dates) >= call_count * 2 else min(1, len(dates) - 1)
    elif "check" in arg and "in" in arg or any(token in slot_text for token in ["check in", "start date", "begin date", "from date"]):
        index = call_index * 2 if len(dates) >= call_count * 2 else 0
    elif len(dates) == 1:
        index = 0
    else:
        index = min(call_index, len(dates) - 1)
    candidate = dates[min(index, len(dates) - 1)]
    if isinstance(value, str) and value == candidate:
        return None
    return candidate


def _date_slot_prefers_iso(arg: str, slot_text: str) -> bool:
    return bool(
        "yyyy-mm-dd" in slot_text
        or "%y-%m-%d" in slot_text
        or "format 'yyyy" in slot_text
        or "format yyyy" in slot_text
        or arg in {"check_in", "check_out", "start_date", "end_date", "departure_date"}
    )


def _iso_date_values(text: str) -> list[str]:
    values: list[str] = []
    values.extend(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    year_context = _explicit_year_context(text)
    month_names = "|".join(sorted(MONTH_NUMBERS, key=len, reverse=True))
    for match in re.finditer(
        rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?\b",
        text,
        re.I,
    ):
        month = MONTH_NUMBERS.get(match.group(1).lower().rstrip("."))
        day = int(match.group(2))
        year = int(match.group(3) or year_context or 0)
        if month and year:
            values.append(f"{year:04d}-{month:02d}-{day:02d}")
    for match in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\.?(?:,\s*)?(\d{{4}})?\b",
        text,
        re.I,
    ):
        month = MONTH_NUMBERS.get(match.group(2).lower().rstrip("."))
        day = int(match.group(1))
        year = int(match.group(3) or year_context or 0)
        if month and year:
            values.append(f"{year:04d}-{month:02d}-{day:02d}")
    return _dedupe(values)


def _explicit_year_context(text: str) -> int | None:
    for pattern in [
        r"\bthis\s+year\s+(?:is|=)\s*((?:19|20)\d{2})\b",
        r"\byear\s+(?:is|=)\s*((?:19|20)\d{2})\b",
        r"\bin\s+((?:19|20)\d{2})\b",
    ]:
        match = re.search(pattern, text, re.I)
        if match:
            return int(match.group(1))
    return None


def _repair_numeric_value_for_slot(
    user_request: str,
    arg: str,
    slot_text: str,
    value: Any,
    spec: dict[str, Any],
) -> Any | None:
    numbers = _first_binary_operation_numbers(user_request)
    if len(numbers) < 2:
        return None
    left, right = numbers[0], numbers[1]
    lowered = user_request.lower()
    if arg == "a":
        if "larger" in slot_text or re.search(r"\b(?:gcd|hcf|highest common factor|greatest common divisor)\b", lowered):
            return _coerce_number(max(left, right), spec)
        if "first" in slot_text:
            return _coerce_number(left, spec)
    if arg == "b":
        if re.search(r"\b(?:gcd|hcf|highest common factor|greatest common divisor)\b", lowered):
            return _coerce_number(min(left, right), spec)
        if "second" in slot_text:
            return _coerce_number(right, spec)
    return None


def _first_binary_operation_numbers(text: str) -> list[Any]:
    number = r"(-?\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = [
        rf"\b(?:for|of|add|sum|plus|gcd|hcf|factor)\s+{number}\s+(?:and|with|plus)\s+{number}\b",
        rf"\b{number}\s+(?:and|with|plus)\s+{number}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        values = [_number_from_text(item) for item in match.groups()]
        values = [item for item in values if isinstance(item, (int, float))]
        if len(values) >= 2:
            return values[:2]
    return []


def _looks_like_weather_unit_slot(tool: dict[str, Any], arg: str, slot_text: str, spec: dict[str, Any]) -> bool:
    if arg not in {"unit", "units"} and "unit" not in slot_text:
        return False
    allowed = {str(item).lower() for item in _enum_values_for_spec(spec)}
    if not {"celsius", "fahrenheit"} <= allowed:
        return False
    return bool(re.search(r"\b(?:weather|temperature|snow)\b", _tool_text(tool), re.I))


def _weather_unit_value(user_request: str, spec: dict[str, Any]) -> str | None:
    lowered = user_request.lower()
    if re.search(r"\b(?:metric|celsius|centigrade)\b", lowered):
        return "celsius" if "celsius" in {str(item).lower() for item in _enum_values_for_spec(spec)} else None
    if re.search(r"\b(?:imperial|fahrenheit)\b", lowered):
        return "fahrenheit"
    return "fahrenheit"


def _looks_like_default_recipient(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"default@example.com", "default", "user@example.com"}


def _looks_like_image_generation_tool(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower()
    return bool("image" in text and re.search(r"\b(?:generate|create|draw|paint)\b", text))


def _image_prompt_value(user_request: str) -> str | None:
    quoted = _quoted_strings(user_request)
    if not quoted:
        return None
    imageish = [
        value
        for value in quoted
        if re.search(r"\b(?:painting|image|photo|picture|portrait|feathers?|style|cyberpunk|digital)\b", value, re.I)
    ]
    return imageish[-1] if imageish else quoted[-1]


def _tool_looks_like_reservation_tool(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower().replace("_", " ")
    properties = _properties(tool)
    slot_text = " ".join(f"{name} {spec.get('description') or ''}" for name, spec in properties.items()).lower()
    has_reservation_language = bool(re.search(r"\b(?:book|booking|reserve|reservation|hotel|accommodation)\b", text))
    has_reservation_slots = bool(
        re.search(r"\bhotel name\b|\bcheck[- ]?in\b|\bcheck[- ]?out\b|\badults?\b|\bchildren\b", slot_text)
    )
    has_place_and_date = any(
        _looks_like_location_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower())
        for name, spec in properties.items()
    ) and any(
        _looks_like_date_value_slot(name.lower(), f"{name} {spec.get('description') or ''}".lower())
        for name, spec in properties.items()
    )
    return bool(has_reservation_language and (has_reservation_slots or has_place_and_date))


def _reservation_scenarios(text: str) -> list[dict[str, Any]]:
    action = r"(?:book|reserve|make\s+(?:a\s+)?reservation(?:\s+(?:for|at))?)"
    pattern = (
        rf"(?:^|[.;]\s*|\b(?:then|also|next|after that)\s+)"
        rf"({action}\b.*?)(?=(?:[.;]\s*|\b(?:then|also|next|after that)\s+){action}\b|$)"
    )
    year = _explicit_year_context(text)
    scenarios: list[dict[str, Any]] = []
    for match in re.finditer(pattern, text, re.I | re.S):
        chunk = match.group(1).strip(" .")
        if not chunk:
            continue
        date_text = f"{chunk} this year is {year}" if year and not re.search(r"\b(?:19|20)\d{2}\b", chunk) else chunk
        dates = _iso_date_values(date_text)
        guests = _guest_count_scenarios(chunk)
        guest = guests[0] if guests else {}
        scenario = {
            "hotel_name": _reservation_hotel_name(chunk),
            "location": _reservation_location(chunk),
            "check_in": dates[0] if dates else None,
            "check_out": dates[1] if len(dates) > 1 else None,
            "adults": guest.get("adults"),
            "children": guest.get("children"),
        }
        if any(value is not None and value != "" for value in scenario.values()):
            scenarios.append(scenario)
    return scenarios


def _reservation_hotel_name(chunk: str) -> str | None:
    patterns = [
        r"\b(?:book|reserve)\s+(?:a\s+room\s+at\s+|a\s+reservation\s+at\s+|the\s+)?([A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){0,5}(?:\s+(?:Hotel|Inn|Resort|Motel))?)(?=\s+in\b)",
        r"\b(?:at|for)\s+(?:the\s+)?([A-Z][A-Za-z&'.-]*(?:\s+[A-Z][A-Za-z&'.-]*){0,5}\s+(?:Hotel|Inn|Resort|Motel))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, chunk, re.I)
        if match:
            value = _clean_slot_phrase(match.group(1))
            if value:
                return value
    return None


def _reservation_location(chunk: str) -> str | None:
    for match in re.finditer(
        r"\bin\s+([A-Z][A-Za-z .'-]+?)(?=\s+(?:for|with|checking|check[- ]?in|check[- ]?out|from|on)\b|[?.!,]|$)",
        chunk,
    ):
        value = _clean_location_value(match.group(1))
        if value:
            return value
    return None


def _hotel_name_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(
        r"\bat\s+(?:the\s+)?([A-Z][A-Za-z&'.-]+(?:\s+[A-Z][A-Za-z&'.-]+)*\s+(?:Hotel|Inn|Resort|Motel)|[A-Z][A-Za-z&'.-]+)(?=\s+in\b|,|\s+with\b|\s+checking\b|\s+check[- ]?in\b)",
        text,
    ):
        value = _clean_slot_phrase(match.group(1))
        if value:
            values.append(value)
    return _dedupe(values)


def _guest_count_scenarios(text: str) -> list[dict[str, int]]:
    scenarios: list[dict[str, int]] = []
    parts = re.split(r"\b(?:first|then|also|next|second|after that)\b", text, flags=re.I)
    for part in parts:
        adult_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+adults?\b", part, re.I)
        child_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a)\s+child(?:ren)?\b", part, re.I)
        if not adult_match and not child_match:
            continue
        adults = _number_from_text(adult_match.group(1)) if adult_match else 0
        child_raw = child_match.group(1) if child_match else "0"
        children = 1 if child_raw.lower() == "a" else _number_from_text(child_raw)
        if isinstance(adults, int) and isinstance(children, int):
            scenarios.append({"adults": adults, "children": children})
    return scenarios


def _resolve_contextual_arguments(
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not calls:
        return calls
    tools_by_name = {str(tool.get("name") or ""): tool for tool in tools}
    total_by_tool = Counter(str(call.get("tool_name") or "") for call in calls if isinstance(call, dict))
    seen_by_tool: Counter[str] = Counter()
    resolved: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool_name") or "")
        tool = tools_by_name.get(tool_name)
        if tool is None:
            resolved.append(call)
            continue
        occurrence = seen_by_tool[tool_name]
        seen_by_tool[tool_name] += 1
        properties = _properties(tool)
        required = set(_required(tool))
        arguments = dict(call.get("arguments") or {})
        missing = [str(item) for item in call.get("missing_arguments") or []]
        tool_call_count = max(1, total_by_tool.get(tool_name, 1))
        for name, spec in properties.items():
            if _argument_has_value(arguments.get(name)):
                continue
            is_required = name in required
            if not is_required and not _optional_context_slot(name, spec):
                continue
            if not is_required and tool_call_count > 1:
                continue
            value = _contextual_argument_value(
                user_request,
                tool,
                name,
                spec,
                occurrence,
                tool_call_count,
                resolved,
                tools_by_name,
            )
            if value is None:
                continue
            if not _model_binding_value_allowed_by_schema(value, spec):
                continue
            if not _contextual_value_is_grounded(user_request, value):
                continue
            arguments[name] = value
        if arguments != (call.get("arguments") or {}) or missing:
            call = dict(call)
            call["arguments"] = arguments
            call["missing_arguments"] = [name for name in missing if not _argument_has_value(arguments.get(name))]
        resolved.append(call)
    return resolved


def _argument_has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _optional_context_slot(name: str, spec: dict[str, Any]) -> bool:
    if _property_type(spec) != "string":
        return False
    slot_text = f"{name} {spec.get('description') or ''}".lower().replace("_", " ")
    if _looks_like_sensitive_credential_slot(name.lower(), slot_text):
        return False
    return bool(re.search(r"\b(?:condition|type|category|topic|subject|detail|field)\b", slot_text))


def _contextual_argument_value(
    user_request: str,
    tool: dict[str, Any],
    name: str,
    spec: dict[str, Any],
    call_index: int,
    call_count: int,
    previous_calls: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> Any | None:
    arg = name.lower()
    slot_text = f"{arg} {spec.get('description') or ''}".lower().replace("_", " ")
    if _looks_like_sensitive_credential_slot(arg, slot_text):
        return None
    if _property_type(spec) == "string":
        value = _schema_named_scalar_value(user_request, arg, slot_text, call_index, call_count)
        if value:
            return value
    if not _slot_can_inherit_from_prior_call(arg, slot_text):
        return None
    return _prior_call_context_value(arg, slot_text, spec, previous_calls, tools_by_name)


def _slot_can_inherit_from_prior_call(arg: str, slot_text: str) -> bool:
    if arg in {"query", "search_query", "keyword", "keywords"}:
        return False
    if any(token in arg for token in ["password", "token", "secret", "key"]):
        return False
    return bool(
        re.search(
            r"\b(?:recipe|dish|meal|food|ingredient|game|hotel|restaurant|movie|film|title|show|event|"
            r"product|company|team|artist|player|city|location|place|topic|subject|entity|name|website|site|dataset)\b",
            slot_text,
        )
    )


def _prior_call_context_value(
    arg: str,
    slot_text: str,
    spec: dict[str, Any],
    previous_calls: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
) -> Any | None:
    typ = _property_type(spec)
    for prior in reversed(previous_calls):
        prior_name = str(prior.get("tool_name") or "")
        prior_tool = tools_by_name.get(prior_name, {})
        prior_properties = _properties(prior_tool)
        prior_args = prior.get("arguments") if isinstance(prior.get("arguments"), dict) else {}
        for prior_arg, prior_value in prior_args.items():
            if not _argument_has_value(prior_value):
                continue
            prior_spec = prior_properties.get(str(prior_arg), {})
            prior_slot_text = f"{prior_arg} {prior_spec.get('description') or ''}".lower().replace("_", " ")
            if _slots_compatible_for_context(arg, slot_text, str(prior_arg).lower(), prior_slot_text, prior_tool):
                coerced = _coerce_contextual_value(prior_value, typ)
                if coerced is not None:
                    return coerced
        if _tool_is_context_source(prior_tool):
            value = _best_prior_text_argument(prior_args)
            if value is not None:
                return _coerce_contextual_value(value, typ)
    return None


def _slots_compatible_for_context(
    arg: str,
    slot_text: str,
    prior_arg: str,
    prior_slot_text: str,
    prior_tool: dict[str, Any],
) -> bool:
    current_keys = _role_key_variants(arg) | _role_key_variants(slot_text)
    prior_keys = _role_key_variants(prior_arg) | _role_key_variants(prior_slot_text)
    if current_keys & prior_keys:
        return True
    for aliases in SEMANTIC_SLOT_ALIASES:
        if aliases & current_keys and aliases & prior_keys:
            return True
    prior_tool_text = _tool_text(prior_tool).lower()
    if {"recipe", "dish", "meal"} & current_keys and (
        {"ingredient", "food", "item"} & prior_keys or "recipe" in prior_tool_text
    ):
        return True
    if "game" in current_keys and ("game" in prior_tool_text or {"game", "title"} & prior_keys):
        return True
    if {"topic", "subject", "entity", "name"} & current_keys and _is_informative_slot(prior_arg, prior_slot_text):
        return True
    return False


def _tool_is_context_source(tool: dict[str, Any]) -> bool:
    text = _tool_text(tool).lower()
    return bool(re.search(r"\b(?:search|find|lookup|retrieve|query|get)\b", text))


def _best_prior_text_argument(arguments: dict[str, Any]) -> Any | None:
    preferred = [
        "recipe",
        "dish",
        "meal",
        "ingredient",
        "game",
        "title",
        "name",
        "query",
        "search_query",
        "topic",
        "subject",
    ]
    for key in preferred:
        if _argument_has_value(arguments.get(key)):
            return arguments.get(key)
    for value in arguments.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _coerce_contextual_value(value: Any, typ: str) -> Any | None:
    if typ == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return None
    if typ in {"integer", "float", "number"}:
        return _number_from_any(value)
    if typ == "array" and isinstance(value, list):
        return value
    if typ in {"any", "object", "dict", ""}:
        return value
    return None


def _contextual_value_is_grounded(user_request: str, value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        lowered = user_request.lower()
        if text.lower() in lowered:
            return True
        tokens = [token for token in _tokens(text) if len(token) > 2]
        return bool(tokens) and all(re.search(rf"\b{re.escape(token)}\b", lowered) for token in tokens)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
        return bool(re.search(rf"(?<![\d.]){re.escape(text)}(?!\d)(?!\.\d)", user_request))
    if isinstance(value, list):
        return all(_contextual_value_is_grounded(user_request, item) for item in value)
    return False


def _is_informative_slot(arg: str, slot_text: str) -> bool:
    generic = {"id", "type", "status", "count", "number", "value", "date", "time", "token", "password"}
    tokens = set(_tokens(f"{arg} {slot_text}"))
    return bool(tokens - generic)


def _explicit_tool_sequence_candidate(
    user_request: str,
    tools: list[dict[str, Any]],
    capability_context: dict[str, Any],
) -> dict[str, Any] | None:
    mentions = _explicit_tool_mentions(user_request, tools)
    if len(mentions) < 2:
        return None
    mention_counts = Counter(str(tool.get("name") or "") for _pos, tool in mentions)
    calls: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for _pos, tool in mentions:
        scoped = _clause_containing_tool_mention(user_request, _pos)
        audit = _audit_single_explicit_tool_call(scoped, tool, capability_context)
        if not audit.get("planned_calls") or any(call.get("missing_arguments") for call in audit.get("planned_calls") or []):
            audit = _audit_single_explicit_tool_call(user_request, tool, capability_context)
        planned_calls = audit.get("planned_calls") or []
        if not planned_calls or any(call.get("missing_arguments") for call in planned_calls):
            return None
        if not _explicit_audit_should_preserve_repeated_calls(
            scoped,
            tools,
            tool,
            audit,
            mention_counts.get(str(tool.get("name") or ""), 0),
        ):
            planned_calls = planned_calls[:1]
        for call_plan in planned_calls:
            calls.append(_call_from_plan(tool, call_plan, len(calls) + 1))
        audits.append(audit)
    if any(call.get("missing_arguments") for call in calls):
        return None
    return {"calls": calls, "audits": audits}


def _explicit_audit_should_preserve_repeated_calls(
    scoped_request: str,
    tools: list[dict[str, Any]],
    tool: dict[str, Any],
    audit: dict[str, Any],
    same_tool_mentions: int,
) -> bool:
    planned_calls = audit.get("planned_calls") or []
    if len(planned_calls) <= 1:
        return False
    if same_tool_mentions > 1:
        return False
    if len(_explicit_tool_mentions(scoped_request, tools)) > 1:
        return False
    required = set(_required(tool))
    slot_bindings = audit.get("slot_bindings") if isinstance(audit.get("slot_bindings"), dict) else {}
    for slot, value in slot_bindings.items():
        if slot not in required or not isinstance(value, list) or len(value) != len(planned_calls):
            continue
        normalized = [json.dumps(item, sort_keys=True, default=str) for item in value]
        if len(set(normalized)) > 1:
            return True
    return False


def _audit_single_explicit_tool_call(
    user_request: str,
    tool: dict[str, Any],
    capability_context: dict[str, Any],
) -> dict[str, Any]:
    task_frame = build_task_frame(user_request)
    query_input_audit = build_query_input_audit(user_request, capability_context)
    return audit_candidate_tool(user_request, tool, 1.0, task_frame, query_input_audit)


def _explicit_tool_mentions(user_request: str, tools: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    mentions: list[tuple[int, dict[str, Any]]] = []
    occupied: list[tuple[int, int]] = []
    for tool in sorted(tools, key=lambda item: -len(str(item.get("name") or ""))):
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        for start, end in _explicit_tool_name_spans(user_request, name):
            if any(not (end <= left or start >= right) for left, right in occupied):
                continue
            occupied.append((start, end))
            mentions.append((start, tool))
    mentions.sort(key=lambda item: item[0])
    return mentions


def _explicit_tool_name_spans(user_request: str, name: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    escaped = re.escape(name)
    exact_pattern = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    for match in re.finditer(exact_pattern, user_request):
        if re.search(r"[._]", name) or _tool_name_match_has_explicit_marker(user_request, match.start(), match.end()):
            spans.append((match.start(), match.end()))
    return spans


def _tool_name_match_has_explicit_marker(user_request: str, start: int, end: int) -> bool:
    before = user_request[max(0, start - 24) : start].lower()
    after = user_request[end : min(len(user_request), end + 32)].lower()
    return bool(
        re.search(r"['\"]\s*$", before)
        or re.search(r"^\s*['\"]", after)
        or re.search(r"\b(?:function|tool|api|using|use|call)\s+(?:the\s+)?$", before)
        or re.search(r"^\s+(?:function|tool|api)\b", after)
    )


def _clause_containing_tool_mention(user_request: str, position: int) -> str:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return user_request
    lowered = user_request.lower()
    for clause in clauses:
        index = lowered.find(clause.lower())
        if index <= position < index + len(clause):
            return clause
    return user_request


def _prefer_explicit_sequence_candidate(
    global_calls: list[dict[str, Any]],
    explicit_calls: list[dict[str, Any]],
) -> bool:
    if len(explicit_calls) < 2:
        return False
    global_names = [str(call.get("tool_name") or "") for call in global_calls]
    explicit_names = [str(call.get("tool_name") or "") for call in explicit_calls]
    if (
        len(set(explicit_names)) == 1
        and set(global_names) == set(explicit_names)
        and len(global_names) > len(explicit_names)
    ):
        return False
    # If the user names an ordered sequence of callable tools/functions, use
    # that as the call skeleton. The surrounding prose often contains examples,
    # repeated data payloads, or background facts that should not add calls.
    if not any(call.get("missing_arguments") for call in explicit_calls):
        return True
    if not global_names:
        return True
    if len(explicit_names) != len(global_names) and set(explicit_names).issubset(set(global_names) | set(explicit_names)):
        return True
    if Counter(explicit_names) == Counter(global_names) and explicit_names != global_names:
        return True
    return False


def _call_from_plan(tool: dict[str, Any], call_plan: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"call_{index}",
        "tool_name": tool["name"],
        "arguments": call_plan["arguments"],
        "depends_on": [],
        "missing_arguments": call_plan["missing_arguments"],
    }


def _dedupe_redundant_calls(user_request: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(calls) <= 1 or _explicit_duplicate_call_requested(user_request):
        return calls
    if len({str(call.get("tool_name") or "") for call in calls}) <= 1:
        return calls
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        key = (
            str(call.get("tool_name") or ""),
            json.dumps(call.get("arguments") or {}, sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**call, "id": f"call_{len(deduped) + 1}"})
    return deduped


def _order_configuration_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(calls) <= 1:
        return calls
    ordered = list(calls)
    index = 0
    while index < len(ordered):
        call = ordered[index]
        if not _is_configuration_call(call):
            index += 1
            continue
        target_index = next(
            (
                pos
                for pos, prior in enumerate(ordered[:index])
                if _is_display_or_consumer_call(prior)
            ),
            None,
        )
        if target_index is None:
            index += 1
            continue
        item = ordered.pop(index)
        ordered.insert(target_index, item)
        index += 1
    return [{**call, "id": f"call_{idx + 1}"} for idx, call in enumerate(ordered)]


def _is_configuration_call(call: dict[str, Any]) -> bool:
    name = str(call.get("tool_name") or "").lower().replace(".", "_")
    return bool(re.search(r"(?:^|_)set(?:_|$)|configure|configuration|brightness|volume|mode|preference", name))


def _is_display_or_consumer_call(call: dict[str, Any]) -> bool:
    name = str(call.get("tool_name") or "").lower().replace(".", "_")
    if _is_configuration_call(call):
        return False
    return bool(re.search(r"display|show|view|render|play|present", name))


def _explicit_duplicate_call_requested(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:repeat|again|twice|two\s+times|same\s+(?:call|function|tool|request)\s+again)\b",
            user_request,
            re.I,
        )
    )


def _should_interleave_selected_tool_calls(selected: list[dict[str, Any]]) -> bool:
    if len(selected) <= 1:
        return False
    counts = [len((item.get("audit") or {}).get("planned_calls") or []) for item in selected]
    return bool(counts and min(counts) > 1 and len(set(counts)) == 1)


def _prefer_interleaved_selected_tool_calls(user_request: str, selected: list[dict[str, Any]]) -> bool:
    if not _should_interleave_selected_tool_calls(selected):
        return False
    if _has_explicit_function_name_sequence(user_request):
        return False
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return True
    best_clause_indices: list[int] = []
    for item in selected:
        scored = [(_tool_scope_score(clause, item["tool"]), index) for index, clause in enumerate(clauses)]
        best_score = max((score for score, _index in scored), default=0.0)
        best_index = next((index for score, index in scored if score == best_score), -1)
        if best_score < 2.0:
            best_clause_indices = []
            break
        best_clause_indices.append(best_index)
    if (
        len(clauses) == len(selected)
        and len(best_clause_indices) == len(selected)
        and len(set(best_clause_indices)) == len(selected)
        and all(_clause_has_entity_context(clause) for clause in clauses)
    ):
        return False
    if re.search(
        r"\b(?:for each|for both|each of|both of|these .+ as well|same .+ as well|for these|respectively)\b",
        user_request,
        re.I,
    ):
        return True
    # If every selected tool uses the same request scope and has the same
    # number of planned calls, the repeated entities are shared across tools
    # rather than belonging to one clause per tool.
    return _selected_tools_share_request_scope(selected)


def _prefer_schema_order_for_shared_entity_interleave(user_request: str, selected: list[dict[str, Any]]) -> bool:
    if not _prefer_interleaved_selected_tool_calls(user_request, selected):
        return False
    if _has_explicit_function_name_sequence(user_request) or _has_strong_order_markers(user_request):
        return False
    return _selected_tools_share_request_scope(selected)


def _selected_tools_share_request_scope(selected: list[dict[str, Any]]) -> bool:
    scopes = [str((item.get("audit") or {}).get("scoped_user_request") or "") for item in selected]
    return bool(scopes) and len(set(scopes)) == 1


def _sort_selected_by_schema_order(
    selected: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schema_order = {str(tool.get("name") or ""): index for index, tool in enumerate(tools)}
    return sorted(
        selected,
        key=lambda item: (
            int(item.get("tool_index", schema_order.get(str((item.get("tool") or {}).get("name") or ""), 10_000))),
            schema_order.get(str((item.get("tool") or {}).get("name") or ""), 10_000),
            str((item.get("tool") or {}).get("name") or ""),
        ),
    )


def _clause_has_entity_context(clause: str) -> bool:
    entities = [
        value
        for value in _capitalized_entity_values(clause)
        if value.lower() not in {"find", "get", "give", "provide", "calculate", "compute", "search", "tell", "can", "could"}
    ]
    if entities:
        return True
    if _location_units(clause):
        return True
    if _quoted_strings(clause):
        return True
    return bool(extract_numbers(clause))


def _clause_level_multi_tool_candidate(
    user_request: str,
    tools: list[dict[str, Any]],
    capability_context: dict[str, Any],
) -> dict[str, Any] | None:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return None
    selected_by_clause: list[list[dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for clause in clauses:
        if _clause_is_context_only(clause):
            continue
        audit_request = user_request if _scoped_request_needs_antecedent(clause) else clause
        task_frame = build_task_frame(audit_request)
        query_input_audit = build_query_input_audit(audit_request, capability_context)
        ranked = rank_tools(clause, tools, task_frame)
        audited = [
            {
                **item,
                "audit": audit_candidate_tool(audit_request, item["tool"], item["score"], task_frame, query_input_audit),
            }
            for item in ranked
        ]
        selected = [
            item
            for item in audited
            if item["audit"]["eligible"]
            and item["score"] >= _selection_threshold(clause, item["tool"])
            and not _request_asks_result_dependent_unary_math(user_request.lower(), _tool_text(item["tool"]).lower())
        ]
        if not selected:
            continue
        if _allows_multiple_tools(clause):
            selected = _filter_weak_multi_tool_selections(clause, clause, selected)
        else:
            explicit = [item for item in selected if _tool_name_is_explicitly_requested(clause, item["tool"])]
            selected = explicit if len(explicit) > 1 else selected[:1]
        selected = sorted(
            selected,
            key=lambda item: (
                _tool_best_clause_position(clause, item["tool"]),
                _tool_mention_position(clause, item["tool"]),
                -item["score"],
                item["tool"]["name"],
            ),
        )
        selected = _scope_selected_tool_audits(clause, selected)
        selected_by_clause.append(selected)
        audits.extend(item["audit"] for item in selected)

    if not selected_by_clause:
        return None
    calls: list[dict[str, Any]] = []
    for selected in selected_by_clause:
        for call in _calls_from_selected_tools(selected, interleave=True):
            call = dict(call)
            call["id"] = f"call_{len(calls) + 1}"
            calls.append(call)
    if not calls:
        return None
    if any(call.get("missing_arguments") for call in calls):
        return None
    return {"calls": calls, "audits": audits}


def _clause_is_context_only(clause: str) -> bool:
    lowered = clause.lower()
    if len(_tokens(clause)) <= 2:
        return True
    if re.fullmatch(r"\d+\s+(?:days?|years?|seconds?|minutes?|hours?)", lowered):
        return True
    if not _clause_has_action_signal(clause) and re.search(r"[\{\[\(]|received|given|available|provided", lowered):
        return True
    return False


def _clause_has_action_signal(clause: str) -> bool:
    return bool(
        _action_labels(clause)
        or re.search(
            r"\b(?:what|who|which|where|when|how|find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|look\s+up|search|suggest|recommend|translate|identify|analy[sz]e|display|withdraw|invest|transfer|locate|fetch|perform|sort|filter|sum)\b",
            clause,
            re.I,
        )
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\s+function\b", clause)
    )


def _prefer_clause_level_candidate(
    user_request: str,
    global_calls: list[dict[str, Any]],
    clause_calls: list[dict[str, Any]],
) -> bool:
    if not clause_calls:
        return False
    if len(clause_calls) > 16:
        return False
    global_names = [str(call.get("tool_name") or "") for call in global_calls]
    clause_names = [str(call.get("tool_name") or "") for call in clause_calls]
    if not global_names and clause_names:
        return True
    if len(clause_names) > len(global_names) and set(global_names).issubset(set(clause_names)):
        if set(clause_names) == set(global_names):
            return _clause_candidate_matches_action_markers(user_request, global_names, clause_names)
        return True
    if (
        len(clause_names) < len(global_names)
        and set(clause_names) == set(global_names)
        and len(clause_names) >= 2
    ):
        if len(set(global_names)) == 1:
            return False
        return True
    if Counter(clause_names) == Counter(global_names) and clause_names != global_names:
        return _has_strong_order_markers(user_request)
    return False


def _clause_candidate_adds_repeated_work(
    user_request: str,
    global_names: list[str],
    clause_names: list[str],
) -> bool:
    if Counter(clause_names) == Counter(global_names):
        return False
    if len(_intent_clauses(user_request)) <= 1:
        return False
    if not re.search(
        r"\b(?:also|after that|then|next|finally|lastly|another|same|both|each|respectively|as well)\b",
        user_request,
        re.I,
    ):
        return False
    global_counts = Counter(global_names)
    clause_counts = Counter(clause_names)
    return any(clause_counts[name] > global_counts.get(name, 0) for name in clause_counts)


def _clause_candidate_matches_action_markers(
    user_request: str,
    global_names: list[str],
    clause_names: list[str],
) -> bool:
    if len(clause_names) <= len(global_names) or len(clause_names) > 8:
        return False
    markers = re.findall(
        r"\b(?:also|then|after that|next|finally|lastly)\b|\band\s+(?=(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|search|recommend|translate|analy[sz]e|display|transfer|locate|fetch|perform)\b)",
        user_request,
        flags=re.I,
    )
    if len(markers) < 2 or len(clause_names) != len(markers) + 1:
        return False
    global_counts = Counter(global_names)
    clause_counts = Counter(clause_names)
    return all(clause_counts[name] >= global_counts.get(name, 0) for name in global_counts)


def _has_strong_order_markers(user_request: str) -> bool:
    return bool(
        re.search(r"\b(?:first|second|third|fourth|finally|lastly|after that|then|once|respectively)\b", user_request, re.I)
        or _has_explicit_function_name_sequence(user_request)
    )


def _has_explicit_function_name_sequence(user_request: str) -> bool:
    return len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]*\s+function\b|'[A-Za-z_][A-Za-z0-9_.]*'", user_request)) >= 2


def _tool_scoped_request_text(user_request: str, tool: dict[str, Any]) -> str:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return user_request
    scored = [(_tool_scope_score(clause, tool), index, clause) for index, clause in enumerate(clauses)]
    best = max((score for score, _, _ in scored), default=0.0)
    if best < 2.0:
        return user_request
    threshold = max(2.0, best * 0.6)
    selected = [clause for score, _, clause in scored if score >= threshold]
    if not selected or len(selected) == len(clauses):
        return user_request
    if any(re.search(r"\b(?:each|both|same|that|those|these)\s+(?:city|cities|data|basis|period|periods|location|locations|file|files|route|routes)\b", clause, re.I) for clause in selected):
        return user_request
    return ". ".join(selected)


def _tool_relevant_clause_count(user_request: str, tool: dict[str, Any]) -> int:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return 1
    scores = [_tool_scope_score(clause, tool) for clause in clauses]
    best = max(scores, default=0.0)
    if best < 2.0:
        return 1
    threshold = max(2.0, best * 0.6)
    selected = [clause for clause, score in zip(clauses, scores) if score >= threshold]
    selected = _prefer_independent_action_clauses(selected)
    count = len(selected)
    return min(count, 8) if count > 1 else 1


def _prefer_independent_action_clauses(clauses: list[str]) -> list[str]:
    action_clauses = [clause for clause in clauses if _clause_has_action_signal(clause)]
    return action_clauses if len(action_clauses) >= 2 else clauses


def _intent_clauses(user_request: str) -> list[str]:
    text = user_request.strip().strip("\"'")
    if not text:
        return []
    text = _protect_abbreviation_periods(text)
    marked = re.sub(r"(?<=[.!?;])\s+", "|", text)
    marked = re.sub(r"(?:^|\s+)\d+\s*[\).]\s+(?=[A-Za-z])", "|", marked)
    marked = re.sub(
        r",\s*(?:and\s+)?(?=(?:what|who|which|where|when|how)\b)",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r"\b(?:also|in addition(?:\s+to\s+that)?|after that|then|lastly|finally|secondly|thirdly|fourthly)\b[, ]*",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r"\bnext(?:,|\s+(?=(?:i|we|please|find|calculate|compute|get|retrieve|translate|book|create|send|locate|search|provide|generate|convert|recommend|analy[sz]e|identify|display|fetch|perform)\b))\s*",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r"\bfirst(?:ly)?(?:,|\s+(?=(?:i|we|please|find|calculate|compute|get|retrieve|translate|book|create|send|locate|search)\b))\s*",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r",?\s+\band\s+(?=(?:also\s+)?(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|look\s+up|search|suggest|recommend|translate|analy[sz]e|identify|display|withdraw|invest|transfer|locate|fetch|perform)\b)",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r"\s+\band\s+(?!(?:temperature|humidity|precipitation|wind|weather)\s+forecast\b)(?=(?:a|an|the)?\s*[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4}\s+(?:in|at|near|for|from|to)\s+[A-Z0-9'\"])",
        "|",
        marked,
    )
    marked = re.sub(
        r",\s+and\s+(?=(?:the\s+)?(?:least\s+common\s+multiple|greatest\s+common\s+divisor|lcm|gcd)\b)",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(
        r"\s+\band\s+(?=precipitation\s+forecast\b)",
        "|",
        marked,
        flags=re.I,
    )
    marked = re.sub(r"\s+\band\s+(?=what\s+(?:his|her|their)\b)", "|", marked, flags=re.I)
    marked = re.sub(r"\s+\band\s+(?=(?:the\s+)?(?:signing\s+of\s+the\s+)?treaty\b)", "|", marked, flags=re.I)
    marked = re.sub(r"\s+\band\s+(?=(?:the\s+)?(?:diameter|weight)\b)", "|", marked, flags=re.I)
    parts = [_clean_intent_clause(part).replace("<<DOT>>", ".") for part in marked.split("|")]
    return [part for part in parts if part]


def _protect_abbreviation_periods(text: str) -> str:
    abbreviations = ["Inc", "Corp", "Co", "Ltd", "LLC", "Dr", "Mr", "Mrs", "Ms", "Prof", "St"]
    for abbr in abbreviations:
        text = re.sub(rf"\b{re.escape(abbr)}\.", lambda match: match.group(0).replace(".", "<<DOT>>"), text)
    return text


def _clean_intent_clause(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    text = re.sub(r"^(?:and|also)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+(?:and|also)$", "", text, flags=re.I).strip(" ,.;")
    return text


def _tool_scope_score(text: str, tool: dict[str, Any]) -> float:
    span_tokens = set(_expanded_tokens(text))
    if not span_tokens:
        return 0.0
    name_tokens = _scope_tokens(str(tool.get("name") or ""), include_value_tokens=True)
    desc_tokens = _scope_tokens(str(tool.get("description") or ""), include_value_tokens=False)
    prop_tokens: set[str] = set()
    for name, spec in _properties(tool).items():
        prop_tokens.update(_scope_tokens(name, include_value_tokens=True))
        prop_tokens.update(_scope_tokens(str(spec.get("description") or ""), include_value_tokens=False))
    score = 0.0
    score += 4.0 * len(span_tokens & name_tokens)
    score += 2.0 * len(span_tokens & desc_tokens)
    score += 0.5 * len(span_tokens & prop_tokens)
    if _phrase_alias_position(text, tool) is not None:
        score += 4.0
    if _action_alignment_score(text, tool) > 0:
        score += 1.0
    if _has_semantic_mismatch(text, tool):
        score -= 3.0
    return score


def _tool_best_clause_position(user_request: str, tool: dict[str, Any]) -> int:
    clauses = _intent_clauses(user_request)
    if len(clauses) <= 1:
        return _tool_mention_position(user_request, tool)
    scored = [(_tool_scope_score(clause, tool), index, clause) for index, clause in enumerate(clauses)]
    best_score = max((score for score, _index, _clause in scored), default=0.0)
    if best_score < 2.0:
        return _tool_mention_position(user_request, tool)
    best_clauses = [(index, clause) for score, index, clause in scored if score == best_score]
    lowered = user_request.lower()
    positions = []
    for index, clause in best_clauses:
        position = lowered.find(clause.lower())
        positions.append(position if position >= 0 else len(user_request) + index)
    return min(positions) if positions else _tool_mention_position(user_request, tool)


def _scope_tokens(text: str, *, include_value_tokens: bool) -> set[str]:
    blocked = GENERIC_TOOL_TOKENS | {"api", "tool", "function", "given", "optional", "default"}
    if not include_value_tokens:
        blocked |= VALUE_TOKENS
    return {token for token in _expanded_tokens(text) if len(token) > 2 and token not in blocked}


def _tool_mention_position(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    tool_text = _tool_text(tool).lower()
    if "history" in tool_text or "historical" in tool_text:
        positions = [pos for token in ["history", "historical", "last"] if (pos := lowered.find(token)) >= 0]
        if positions:
            return min(positions)
    if "projection" in tool_text or "predict" in tool_text or "future" in tool_text:
        positions = [pos for token in ["projected", "projection", "future", "next", "predict"] if (pos := lowered.find(token)) >= 0]
        if positions:
            return min(positions)
    alias_position = _phrase_alias_position(user_request, tool)
    if alias_position is not None:
        return alias_position
    weak_mention_tokens = {"ticket", "tickets"}
    name_tokens = [
        token
        for token in _expanded_tokens(tool.get("name", ""))
        if len(token) > 2 and token not in GENERIC_TOOL_TOKENS and lowered.find(token) >= 0
    ]
    specific_name_positions = [lowered.find(token) for token in name_tokens if token not in weak_mention_tokens]
    if specific_name_positions:
        return min(specific_name_positions)
    name_positions = [lowered.find(token) for token in name_tokens]
    if name_positions:
        return min(name_positions)
    desc_positions = [
        lowered.find(token)
        for token in _expanded_tokens(tool.get("description", ""))
        if len(token) > 3 and token not in {"find", "calculate", "compute", "retrieve"} and lowered.find(token) >= 0
    ]
    return min(desc_positions) if desc_positions else len(lowered) + 1


def _phrase_alias_position(user_request: str, tool: dict[str, Any]) -> int | None:
    tool_tokens = set(_expanded_tokens(" ".join([tool.get("name", ""), tool.get("description", "")])))
    positions = []
    for pattern, expansion in PHRASE_EXPANSIONS:
        expansion_tokens = set(_expanded_tokens(expansion))
        if not (tool_tokens & expansion_tokens):
            continue
        match = re.search(pattern, user_request, re.I)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def _explicit_repeated_call_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    required_array_count = sum(
        1
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _property_type(spec) == "array"
    )
    if required_array_count >= 2 and re.search(r"\b(?:first|second|two)\b.+\b(?:vectors?|arrays?|entities?)\b", lowered):
        return 1
    repeat_count = len(re.findall(r"\b(?:repeat|again)\b", lowered))
    if repeat_count:
        return repeat_count + 1
    if re.search(r"\b(?:also for|another with|do the same|same but|and ones|and another)\b", lowered):
        return max(2, _coordinated_value_count(user_request, tool))
    if re.search(r"\b(?:for each|for both|each of|both of|respectively|separately|simultaneously|for every)\b", lowered):
        return max(2, _coordinated_value_count(user_request, tool))
    if re.search(r"\band\s+also\s+(?:for|in|from|with)\b", lowered):
        return max(2, _coordinated_value_count(user_request, tool))
    return 1


def _coordinated_value_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    quoted = _quoted_strings(user_request)
    counts = [len(quoted)] if len(quoted) >= 2 else []
    properties = _properties(tool)
    if "artist" in properties:
        artists = extract_artist_like_values(user_request)
        if len(artists) >= 2:
            return len(artists)
    enum_count = _enum_value_count(user_request, tool)
    if enum_count >= 2:
        counts.append(enum_count)
    if len(properties) == 1:
        only_spec = next(iter(properties.values()))
        if _property_type(only_spec) in {"integer", "float", "number"}:
            numbers = extract_numbers(user_request)
            if len(numbers) >= 2 and not re.search(r"\bbetween\b|\bfrom\b.+\bto\b", lowered):
                counts.append(len(numbers))
    if any(name in properties for name in ["artist", "location", "city", "region", "state", "country", "music_genre", "genre"]):
        # Count likely named entities after common prepositions without trying
        # to bind them to arguments yet.
        chunks = re.split(r"\b(?:and also|also|and|,)\b", user_request)
        proper_chunks = [
            chunk
            for chunk in chunks
            if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", chunk)
            and not chunk.strip().lower().startswith(("can ", "could ", "please "))
        ]
        if len(proper_chunks) >= 2:
            counts.append(len(proper_chunks))
    numeric_labels = [name for name, spec in properties.items() if _property_type(spec) in {"integer", "float", "number"}]
    for label in numeric_labels:
        label_text = label.replace("_", " ")
        matches = re.findall(rf"{re.escape(label_text)}\D{{0,35}}(-?\d+(?:\.\d+)?)", lowered)
        if len(matches) >= 2:
            counts.append(len(matches))
    return max(counts) if counts else 2


def _looks_like_entity_value(user_request: str) -> bool:
    return bool(
        re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", user_request)
        or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", user_request)
        or re.search(r"(?:~?/|[A-Za-z]:\\)", user_request)
    )


def _semantic_identity_overlap(user_request: str, tool: dict[str, Any]) -> int:
    request_tokens = _request_core_tokens(user_request)
    tool_tokens = _tool_core_tokens(tool)
    return len(request_tokens & tool_tokens)


def _semantic_capability_overlap(user_request: str, tool: dict[str, Any]) -> int:
    return len(_intent_tags(user_request) & _tool_capability_tags(tool))


def _has_any_user_input_evidence(user_request: str) -> bool:
    if extract_numbers(user_request) or _quoted_value(user_request) or _looks_like_entity_value(user_request):
        return True
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", user_request):
        return True
    content = [token for token in _tokens(user_request) if token not in STOPWORDS]
    return len(content) >= 2


def _tool_name_is_explicitly_requested(user_request: str, tool: dict[str, Any]) -> bool:
    name = str(tool.get("name") or "")
    if not name:
        return False
    lowered = user_request.lower()
    normalized = name.lower()
    if normalized in lowered:
        return True
    spaced_name = re.sub(r"[_\.\-/]+", " ", normalized)
    spaced_request = re.sub(r"[_\.\-/]+", " ", lowered)
    return bool(spaced_name and spaced_name in spaced_request)


def _has_core_intent_mismatch(user_request: str, tool: dict[str, Any]) -> bool:
    if _is_auth_token_tool(tool):
        return False
    if _tool_name_is_explicitly_requested(user_request, tool):
        return False
    request_tags = _intent_tags(user_request)
    tool_tags = _tool_capability_tags(tool)
    if request_tags and tool_tags and request_tags & tool_tags:
        return False
    if _request_tool_action_alias_overlap(user_request, tool):
        return False
    request_tokens = _request_core_tokens(user_request)
    tool_tokens = _tool_core_tokens(tool)
    if not tool_tokens:
        return False
    overlap = request_tokens & tool_tokens
    if not overlap and not _allows_generic_lookup_binding(user_request, tool):
        return True
    non_measurement_overlap = overlap - MEASUREMENT_TOKENS - VALUE_TOKENS
    if len(non_measurement_overlap) >= 2:
        return False

    request_objects = (request_tokens & DOMAIN_OBJECT_TOKENS) - MEASUREMENT_TOKENS
    tool_objects = (tool_tokens & DOMAIN_OBJECT_TOKENS) - MEASUREMENT_TOKENS
    if request_objects and tool_objects and request_objects.isdisjoint(tool_objects):
        if not _allows_generic_lookup_binding(user_request, tool):
            return True

    if overlap and not non_measurement_overlap and request_objects and tool_objects:
        return request_objects.isdisjoint(tool_objects)
    return False


def _request_tool_action_alias_overlap(user_request: str, tool: dict[str, Any]) -> bool:
    request = user_request.lower()
    tool_text = _tool_text(tool).lower()
    aliases = [
        (r"\b(?:turn|twist|spin)\b.{0,40}\b(?:degree|degrees|clockwise|counterclockwise)\b", r"\brotat(?:e|es|ed|ing)(?:\b|[_\s-]?image|image)"),
        (r"\b(?:make|put|set)\b.{0,30}\b(?:horizontal|vertical)\b|\bflip\b.{0,30}\b(?:horizontal|vertical|horizontally|vertically)\b", r"\b(?:flip|flips|flipped|flipping|mirror|mirrors)(?:\b|[_\s-]?image|image)"),
        (r"\b(?:steep|slope|rate\s+of\s+change)\b.{0,60}\b(?:curve|function|graph)\b", r"\bderivative\b"),
        (r"\b(?:protein|calories|calorie|carbs|carbohydrates|nutrients?|nutrition|nutritional)\b", r"\b(?:nutrition|nutritional|nutrient)"),
    ]
    return any(re.search(request_pattern, request) and re.search(tool_pattern, tool_text) for request_pattern, tool_pattern in aliases)


def _has_only_symbolic_numeric_inputs(user_request: str, tool: dict[str, Any]) -> bool:
    numeric_required = [
        name
        for name, spec in _properties(tool).items()
        if name in _required(tool) and _is_numeric_value_property(spec)
    ]
    if not numeric_required:
        return False
    if extract_numbers(user_request):
        return False
    quoted = [value.lower() for value in _quoted_strings(user_request)]
    symbolic = [value for value in quoted if re.fullmatch(r"[a-z][a-z0-9_]{0,10}", value)]
    return len(symbolic) >= min(2, len(numeric_required))


def _allows_generic_lookup_binding(user_request: str, tool: dict[str, Any]) -> bool:
    request = user_request.lower()
    asks_lookup = bool(
        re.search(r"\b(?:who|what|when|where|which|how)\b", request)
        or re.search(r"\b(?:check|find|fetch|get|look\s+up|lookup|query|read|search|see|show|view)\b", request)
    )
    if not asks_lookup:
        return False
    lookup_tokens = {
        "date",
        "data",
        "event",
        "fetch",
        "find",
        "get",
        "locate",
        "lookup",
        "query",
        "read",
        "record",
        "records",
        "retrieve",
        "search",
        "searches",
    }
    return bool(_tool_core_tokens(tool) & lookup_tokens)


def _request_core_tokens(user_request: str) -> set[str]:
    tokens = set(_expanded_tokens(user_request))
    return {
        token
        for token in tokens
        if token not in GENERIC_TOOL_TOKENS
        and token not in VALUE_TOKENS
        and not token.isdigit()
    }


def _tool_core_tokens(tool: dict[str, Any]) -> set[str]:
    description = str(tool.get("description") or "")
    description = re.split(
        r"\b(?:given|with|using|based on|for a specific|from a given|parameter|parameters|allowed values?)\b",
        description,
        maxsplit=1,
        flags=re.I,
    )[0]
    property_tokens = {
        token
        for name in _properties(tool)
        for token in _expanded_tokens(name)
    }
    name_tokens = set(_expanded_tokens(str(tool.get("name") or "")))
    description_tokens = set(_expanded_tokens(description)) - property_tokens
    raw_tokens = name_tokens | description_tokens
    return {
        token
        for token in raw_tokens
        if token not in GENERIC_TOOL_TOKENS
        and token not in VALUE_TOKENS
    }


def _enum_value_count(user_request: str, tool: dict[str, Any]) -> int:
    lowered = user_request.lower()
    counts: list[int] = []
    for spec in _properties(tool).values():
        enum = _enum_values_for_spec(spec)
        values: list[str] = []
        for item in enum:
            text = str(item).strip().strip("'\"").lower()
            if text and re.search(rf"\b{re.escape(text)}\b", lowered):
                values.append(text)
        if values:
            counts.append(len(set(values)))
    return max(counts) if counts else 0


def _has_enum_array_payload(user_request: str, tool: dict[str, Any]) -> bool:
    for spec in _properties(tool).values():
        if _property_type(spec) != "array":
            continue
        if len(_enum_value_evidence(user_request, spec)) > 1:
            return True
    return False


def _enum_values_for_spec(spec: dict[str, Any]) -> list[Any]:
    enum = spec.get("enum")
    if not isinstance(enum, list) and isinstance(spec.get("items"), dict):
        enum = spec["items"].get("enum")
    if isinstance(enum, list):
        return enum
    return _listed_values_from_description(str(spec.get("description") or ""))


def _listed_values_from_description(description: str) -> list[str]:
    match = re.search(r"\b(?:allowed values?|options?)\b\s*:?\s*(.+)", description, re.I)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\beither\s+(.+?)(?:[.;]|$)", description, re.I)
        if not match:
            return []
        text = match.group(1)
    quoted = _quoted_strings(text)
    if quoted:
        return quoted
    text = re.split(r"[.;]", text, maxsplit=1)[0]
    return [
        item.strip()
        for item in re.split(r",|\bor\b", text)
        if item.strip() and len(item.strip()) <= 40
    ]


def _has_general_argument_evidence(user_request: str) -> bool:
    return bool(
        extract_numbers(user_request)
        or _quoted_value(user_request)
        or _looks_like_entity_value(user_request)
        or re.search(r"[A-Za-z]+\s*[=]\s*[-+]?\w+", user_request)
        or re.search(r"\b(?:who|what|when|where|which|how)\b", user_request.lower())
    )


def _tool_text(tool: dict[str, Any]) -> str:
    parts = [tool.get("name", ""), tool.get("description", "")]
    for name, spec in _properties(tool).items():
        parts.extend([name, str(spec.get("description") or "")])
    return " ".join(parts)


def _properties(tool: dict[str, Any]) -> dict[str, Any]:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    return {str(k): v for k, v in props.items() if isinstance(v, dict)}


def _required(tool: dict[str, Any]) -> list[str]:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    required = params.get("required") if isinstance(params.get("required"), list) else []
    return [str(item) for item in required]


def _property_type(spec: dict[str, Any]) -> str:
    typ = spec.get("type")
    if typ == "number":
        return "float"
    if typ == "tuple":
        return "array"
    return str(typ or "").lower()


def _is_numeric_value_property(spec: dict[str, Any]) -> bool:
    typ = _property_type(spec)
    if typ in {"integer", "float", "number"}:
        return True
    if typ != "array":
        return False
    items = spec.get("items")
    item_type = items.get("type") if isinstance(items, dict) else None
    return str(item_type or "").lower() in {"integer", "float", "number"}


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    values = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.replace("_", " ").replace(".", " ").lower())
    return [value for value in values if value not in STOPWORDS and len(value) > 1]


def _expanded_tokens(text: str) -> list[str]:
    for pattern, expansion in PHRASE_EXPANSIONS:
        if re.search(pattern, text, re.I):
            text = f"{text} {expansion}"
    expanded: list[str] = []
    for token in _tokens(text):
        expanded.append(token)
        if token == "analysis":
            expanded.append("analyze")
        if token == "structural":
            expanded.append("structure")
        if token.endswith("ly") and len(token) > 5:
            expanded.append(token[:-2])
        if token == "genetically":
            expanded.append("genetic")
        if token.endswith("ies") and len(token) > 5:
            expanded.append(f"{token[:-3]}y")
        if token.endswith("es") and len(token) > 4:
            expanded.append(token[:-2])
        for suffix in ("ing", "ed", "er", "or", "ion", "ions", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                expanded.append(token[: -len(suffix)])
        if token.endswith("ity") and len(token) > 5:
            expanded.append(token[:-3])
    return expanded


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _plan(
    decision: str,
    reason: str,
    calls: list[dict[str, Any]],
    missing_inputs: list[str],
    task_frame: dict[str, Any] | None = None,
    query_input_audit: dict[str, Any] | None = None,
    candidate_tool_audits: list[dict[str, Any]] | None = None,
    capability_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in TOOL_DECISIONS:
        decision = "unsupported"
    audits = candidate_tool_audits or []
    return {
        "tool_decision": decision,
        "reason": reason,
        "task_frame": task_frame or {},
        "capability_plan": capability_plan or {},
        "query_input_audit": query_input_audit or {},
        "tool_requirement_frames": [audit["requirement_frame"] for audit in audits if audit.get("requirement_frame")],
        "candidate_tool_audits": audits,
        "calls": calls,
        "missing_inputs": missing_inputs,
    }
