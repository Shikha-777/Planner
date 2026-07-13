from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from taskdecomp.capability_planning import extract_json_object
from taskdecomp.goal_graph_runtime import (
    GoalGraphRuntime,
    build_goal_graph_planner_messages,
    compiled_calls_to_dicts,
)
from taskdecomp.tool_binding import (
    _command_executor_argument_names,
    _filter_schema_value_incompatible_calls,
    _has_hard_semantic_conflict,
    _order_independent_calls_for_benchmark,
    _semantic_frame_explicit_ask_user,
    _semantic_frame_explicit_no_tool,
    _should_call_single_retrieval_tool,
    _tool_is_generic_command_executor,
    _verified_model_argument_group,
    audit_candidate_tool,
    build_query_input_audit,
    build_task_frame,
    build_tool_binding_plan,
    normalize_tool,
)


GenerateTextFn = Callable[[Any, Any, list[dict[str, str]], int], str]


def benchmark_compile_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool schemas prepared for offline benchmark-call scoring.

    BFCL/API-Bank predictions are not executed against live user accounts, so
    tools whose names look mutating should still be allowed to compile. The
    runtime side-effect gates remain the default for real execution.
    """
    prepared = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        item = copy.deepcopy(tool)
        item["risk"] = "read_only"
        item["requires_confirmation"] = False
        item["requires_unique_target"] = False
        item["effects"] = []
        prepared.append(item)
    return prepared


def diagnostics_to_dicts(verification: Any) -> list[dict[str, str]]:
    return [
        {
            "code": str(item.code),
            "message": str(item.message),
            "path": str(item.path),
            "severity": str(item.severity),
        }
        for item in getattr(verification, "diagnostics", [])
    ]


def diagnostic_codes(diagnostics: list[dict[str, str]]) -> list[str]:
    return [str(item.get("code") or "") for item in diagnostics if item.get("code")]


def plan_and_compile_goal_graph(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repair_attempts: int = 1,
    policies: list[str] | None = None,
    failure_lessons: list[str] | None = None,
    allow_side_effects: bool = False,
    use_binder_fallback: bool = False,
    planner_mode: str = "stepwise",
    stateful: bool = False,
    binding_request: str | None = None,
    execution_history: list[dict[str, Any]] | None = None,
    stateful_goal_ledger: dict[str, Any] | None = None,
    stateful_goal_ledger_required: bool = False,
    stateful_semantic_only: bool = False,
    stateful_semantic_review: bool = False,
) -> dict[str, Any]:
    if planner_mode == "stepwise":
        return _plan_and_compile_goal_graph_stepwise(
            model,
            tokenizer,
            generate_text,
            user_request,
            tools,
            max_new_tokens=max_new_tokens,
            repair_attempts=repair_attempts,
            allow_side_effects=allow_side_effects,
            stateful=stateful,
            binding_request=binding_request,
            execution_history=execution_history,
            stateful_goal_ledger=stateful_goal_ledger,
            stateful_goal_ledger_required=stateful_goal_ledger_required,
            stateful_semantic_only=stateful_semantic_only,
            stateful_semantic_review=stateful_semantic_review,
        )
    if planner_mode != "one_shot":
        raise ValueError(f"unsupported goal-graph planner mode: {planner_mode}")
    return _plan_and_compile_goal_graph_one_shot(
        model,
        tokenizer,
        generate_text,
        user_request,
        tools,
        max_new_tokens=max_new_tokens,
        repair_attempts=repair_attempts,
        policies=policies,
        failure_lessons=failure_lessons,
        allow_side_effects=allow_side_effects,
        use_binder_fallback=use_binder_fallback,
    )


def _plan_and_compile_goal_graph_one_shot(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repair_attempts: int,
    policies: list[str] | None,
    failure_lessons: list[str] | None,
    allow_side_effects: bool,
    use_binder_fallback: bool,
) -> dict[str, Any]:
    runtime = GoalGraphRuntime(tools)
    messages = build_goal_graph_planner_messages(
        user_request,
        runtime.registry,
        policies=policies,
        failure_lessons=failure_lessons,
    )
    raw_text = generate_text(model, tokenizer, messages, max_new_tokens)
    parsed, parse_error = extract_goal_graph_json_object(raw_text)
    retry_outputs: list[dict[str, Any]] = []

    calls = []
    verification = None
    diagnostics: list[dict[str, str]] = []
    if parsed is not None:
        output = runtime.compile(
            parsed,
            user_request,
            allow_side_effects=allow_side_effects,
        )
        calls = output.calls
        verification = output.verification
        diagnostics = diagnostics_to_dicts(output.verification)

    for attempt in range(max(0, repair_attempts)):
        if parsed is not None and verification is not None and verification.ok:
            break
        repair_messages = _build_repair_messages(
            user_request=user_request,
            previous_output=raw_text if not retry_outputs else str(retry_outputs[-1].get("raw_text") or ""),
            parse_error=parse_error,
            diagnostics=diagnostics,
            attempt=attempt + 1,
        )
        retry_text = generate_text(model, tokenizer, repair_messages, max_new_tokens)
        retry_parsed, retry_parse_error = extract_goal_graph_json_object(retry_text)
        retry_record: dict[str, Any] = {
            "raw_text": retry_text,
            "parse_error": retry_parse_error,
            "verification_ok": False,
            "diagnostics": [],
        }
        if retry_parsed is not None:
            retry_output = runtime.compile(
                retry_parsed,
                user_request,
                allow_side_effects=allow_side_effects,
            )
            retry_record["verification_ok"] = retry_output.verification.ok
            retry_record["diagnostics"] = diagnostics_to_dicts(retry_output.verification)
            parsed = retry_parsed
            parse_error = retry_parse_error
            calls = retry_output.calls
            verification = retry_output.verification
            diagnostics = retry_record["diagnostics"]
        else:
            if parsed is None:
                parse_error = retry_parse_error
                diagnostics = []
                calls = []
                verification = None
        retry_outputs.append(retry_record)

    binder_fallback: dict[str, Any] | None = None
    if use_binder_fallback and not calls:
        binder_fallback = _compile_binder_fallback_graph(
            runtime,
            user_request,
            tools,
            allow_side_effects=allow_side_effects,
        )
        if binder_fallback.get("verification_ok"):
            parsed = binder_fallback.get("graph")
            calls = binder_fallback.get("compiled_call_objects", [])
            verification = binder_fallback.get("verification")
            diagnostics = binder_fallback.get("diagnostics", [])

    verification_ok = bool(verification and verification.ok)
    return {
        "planner_mode": "one_shot",
        "raw_text": raw_text,
        "graph": parsed,
        "parse_error": parse_error if parsed is None else None,
        "verification_ok": verification_ok,
        "diagnostics": diagnostics,
        "diagnostic_codes": diagnostic_codes(diagnostics),
        "calls": compiled_calls_to_dicts(calls),
        "retry_outputs": retry_outputs,
        "steps": [
            {
                "step": "one_shot_goal_graph",
                "model_call": True,
                "ok": parsed is not None,
                "parse_error": parse_error if parsed is None else None,
            },
            {
                "step": "runtime_verify_compile",
                "model_call": False,
                "ok": verification_ok,
                "diagnostic_codes": diagnostic_codes(diagnostics),
                "call_count": len(calls),
            },
        ],
        "binder_fallback": _serializable_binder_fallback(binder_fallback),
        "capability_count": len(runtime.registry),
        "capabilities": list(runtime.registry.keys()),
    }


def _plan_and_compile_goal_graph_stepwise(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repair_attempts: int,
    allow_side_effects: bool,
    stateful: bool,
    binding_request: str | None,
    execution_history: list[dict[str, Any]] | None,
    stateful_goal_ledger: dict[str, Any] | None,
    stateful_goal_ledger_required: bool,
    stateful_semantic_only: bool,
    stateful_semantic_review: bool,
) -> dict[str, Any]:
    runtime = GoalGraphRuntime(tools)
    completed_calls = _successful_execution_history(execution_history)
    binding_text = binding_request or user_request
    verified_binding_evidence = (
        _stateful_verified_observation_facts(
            completed_calls,
            max_facts=48,
            per_observation_max=16,
        )
        if stateful
        else []
    )
    semantic_output = _generate_semantic_frame(
        model,
        tokenizer,
        generate_text,
        user_request,
        tools,
        max_new_tokens,
        execution_history=completed_calls,
        stateful=stateful,
        stateful_goal_ledger=stateful_goal_ledger,
        stateful_goal_ledger_required=stateful_goal_ledger_required,
    )
    semantic_frame = semantic_output.get("parsed")
    semantic_frame_has_routing_evidence = _semantic_frame_has_routing_evidence(semantic_frame)
    semantic_no_tool_veto = (
        _semantic_frame_explicit_no_tool(semantic_frame)
        and not _should_call_single_retrieval_tool(
            binding_text,
            [normalize_tool(tool) for tool in tools or [] if isinstance(tool, dict)],
        )
    )
    semantic_ask_user_request = _semantic_frame_requests_user(semantic_frame)
    semantic_ask_user_veto = _semantic_frame_explicit_ask_user(semantic_frame) or (
        stateful and semantic_ask_user_request
    )
    semantic_ask_user_has_missing_inputs = bool(
        isinstance(semantic_frame, dict)
        and any(str(item).strip() for item in semantic_frame.get("missing_inputs") or [])
    )
    semantic_terminal_veto = semantic_no_tool_veto or semantic_ask_user_veto
    semantic_terminal_veto_active = semantic_terminal_veto
    capability_plan = {"semantic_input_frame": semantic_frame} if isinstance(semantic_frame, dict) else {}
    binding_plan = build_tool_binding_plan(
        binding_text,
        tools,
        capability_plan=capability_plan,
        allow_model_binding_prefix=stateful,
        verified_evidence=verified_binding_evidence,
    )
    semantic_enum_grounding: dict[str, Any] = {"attempted": False, "used": False, "approved": []}
    if (
        stateful
        and stateful_semantic_only
        and isinstance(semantic_frame, dict)
        and not bool((binding_plan.get("model_tool_binding") or {}).get("accepted"))
    ):
        semantic_frame, semantic_enum_grounding = _adjudicate_semantic_enum_grounding(
            model,
            tokenizer,
            generate_text,
            semantic_frame,
            binding_text,
            tools,
            max_new_tokens,
        )
        if semantic_enum_grounding["used"]:
            capability_plan = {"semantic_input_frame": semantic_frame}
            binding_plan = build_tool_binding_plan(
                binding_text,
                tools,
                capability_plan=capability_plan,
                allow_model_binding_prefix=True,
                verified_evidence=verified_binding_evidence,
            )
    binding_recovery_plan: dict[str, Any] | None = None
    raw_binding_plan: dict[str, Any] | None = None
    semantic_binding_accepted = bool(
        isinstance(binding_plan.get("model_tool_binding"), dict)
        and binding_plan["model_tool_binding"].get("accepted")
    )
    if stateful_semantic_only:
        if not semantic_binding_accepted:
            binding_plan = _stateful_semantic_only_no_call_plan(binding_plan)
    elif semantic_terminal_veto and binding_plan.get("tool_decision") in {"no_tool", "ask_user"}:
        raw_binding_plan = build_tool_binding_plan(binding_text, tools)
        preserve_stateful_semantic_ask = (
            stateful and semantic_ask_user_request and not semantic_ask_user_has_missing_inputs
        )
        if not preserve_stateful_semantic_ask and _prefer_raw_plan_over_semantic_terminal_veto(
            binding_text,
            binding_plan,
            raw_binding_plan,
        ):
            binding_recovery_plan = raw_binding_plan
            binding_plan = raw_binding_plan
            semantic_terminal_veto_active = False
        else:
            raw_binding_plan = {}
    elif not binding_plan.get("calls") or not semantic_frame_has_routing_evidence:
        raw_binding_plan = build_tool_binding_plan(binding_text, tools)
        if raw_binding_plan.get("calls"):
            binding_recovery_plan = raw_binding_plan
            binding_plan = raw_binding_plan
    else:
        raw_binding_plan = build_tool_binding_plan(binding_text, tools)
        if _prefer_raw_binding_plan(binding_text, binding_plan, raw_binding_plan):
            binding_recovery_plan = raw_binding_plan
            binding_plan = raw_binding_plan
    stateful_raw_safety_filter: dict[str, Any] = {"used": False, "dropped_calls": []}
    if stateful and not stateful_semantic_only:
        binding_plan, stateful_raw_safety_filter = _drop_stateful_unverified_fallback_calls(
            binding_plan,
            tools,
            binding_text,
        )
        if stateful_raw_safety_filter["used"]:
            binding_recovery_plan = binding_plan

    stateful_progress_filter: dict[str, Any] = {"used": False, "dropped_calls": []}
    stateful_progress_repair: dict[str, Any] = {"attempted": False, "used": False}
    stateful_schema_recovery: dict[str, Any] = {"used": False}
    semantic_terminal_decision = str(binding_plan.get("tool_decision") or "").strip().lower() in {"ask_user", "no_tool"}
    if (
        stateful
        and stateful_semantic_only
        and not binding_plan.get("calls")
        and (not isinstance(semantic_frame, dict) or semantic_terminal_decision)
    ):
        schema_recovery = _stateful_unique_schema_grounded_readonly_plan(
            binding_text,
            runtime,
            tools,
            completed_calls,
        )
        if schema_recovery is not None:
            binding_plan = schema_recovery
            binding_recovery_plan = schema_recovery
            stateful_schema_recovery = dict(schema_recovery.get("stateful_schema_recovery") or {"used": True})
    if stateful and completed_calls:
        binding_plan, stateful_progress_filter = _drop_successfully_replayed_calls(
            binding_plan,
            completed_calls,
        )
        if stateful_progress_filter["used"]:
            binding_recovery_plan = binding_plan
    stateful_single_call_execution: dict[str, Any] = {"used": False, "deferred_calls": []}
    if stateful:
        binding_plan, stateful_single_call_execution = _limit_stateful_plan_to_one_call(binding_plan)
    stateful_terminal_progress_needed = bool(
        stateful
        and completed_calls
        and _stateful_plan_defers_to_user_or_terminal(binding_plan, tools)
    )
    if (
        stateful
        and not binding_plan.get("calls")
        and (
            stateful_progress_filter["dropped_calls"]
            or stateful_raw_safety_filter["dropped_calls"]
            or stateful_terminal_progress_needed
        )
    ):
        repaired_plan, stateful_progress_repair = _stateful_progress_repair_plan(
            model,
            tokenizer,
            generate_text,
            user_request,
            binding_text,
            tools,
            completed_calls,
            max_new_tokens,
            semantic_only=stateful_semantic_only,
            stateful_goal_ledger=stateful_goal_ledger,
            stateful_goal_ledger_required=stateful_goal_ledger_required,
        )
        if repaired_plan is not None:
            binding_plan = repaired_plan
            binding_recovery_plan = repaired_plan
            semantic_terminal_veto_active = False
    stateful_collection_disambiguation: dict[str, Any] = {"used": False}
    if (
        stateful
        and not binding_plan.get("calls")
        and _stateful_plan_defers_to_user_or_terminal(binding_plan, tools)
    ):
        collection_read_plan = _stateful_next_collection_readonly_plan(
            binding_text,
            runtime,
            completed_calls,
        )
        if collection_read_plan is not None:
            binding_plan = collection_read_plan
            binding_recovery_plan = collection_read_plan
            semantic_terminal_veto_active = False
            stateful_collection_disambiguation = dict(
                collection_read_plan.get("stateful_collection_disambiguation") or {"used": True}
            )
    if stateful and not stateful_semantic_only:
        read_first_plan = _stateful_readonly_progress_plan(binding_text, binding_plan, runtime, tools)
        if read_first_plan is not None:
            binding_recovery_plan = read_first_plan
            binding_plan = read_first_plan
    stateful_action_recovery_plan: dict[str, Any] | None = None
    if stateful and not stateful_semantic_only:
        stateful_candidate = _stateful_nonterminal_recovery_plan(
            binding_text,
            tools,
            binding_plan,
            runtime,
            completed_calls,
        )
        if stateful_candidate is not None:
            candidate_graph = _graph_from_binding_plan(runtime, binding_text, stateful_candidate)
            candidate_output = runtime.compile(
                candidate_graph,
                binding_text,
                allow_side_effects=allow_side_effects,
            )
            if candidate_output.verification.ok:
                stateful_action_recovery_plan = stateful_candidate
                binding_recovery_plan = stateful_candidate
                binding_plan = stateful_candidate
                semantic_terminal_veto_active = False
                graph = candidate_graph
                output = candidate_output
            else:
                graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
        else:
            graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
            output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
    else:
        graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
        output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
    stateful_semantic_review_result: dict[str, Any] = {"attempted": False, "allowed": True}
    terminal_review: dict[str, Any] | None = None
    stateful_missing_input_evidence = _stateful_missing_input_evidence_adjudication(
        model,
        tokenizer,
        generate_text,
        binding_plan,
        completed_calls,
        max_new_tokens,
    ) if stateful and _stateful_terminal_decision_needs_review(binding_plan) else {
        "attempted": False,
        "available_missing_inputs": [],
        "ambiguous_missing_inputs": [],
    }
    resolved_missing_inputs = [
        str(item.get("missing_input") or "")
        for item in stateful_missing_input_evidence.get("available_missing_inputs") or []
        if isinstance(item, dict) and str(item.get("missing_input") or "").strip()
    ]
    resolved_missing_input_feedback = _stateful_resolved_missing_input_feedback(
        stateful_missing_input_evidence.get("available_missing_inputs") or []
    )
    ambiguous_missing_input_feedback = _stateful_ambiguous_missing_input_feedback(
        stateful_missing_input_evidence.get("ambiguous_missing_inputs") or []
    )
    if (
        stateful
        and stateful_semantic_review
        and not output.calls
        and _stateful_terminal_decision_needs_review(binding_plan)
    ):
        if _stateful_plan_requests_resolved_missing_input(binding_plan, resolved_missing_inputs):
            terminal_review = {
                "attempted": True,
                "allowed": False,
                "reason": "candidate asks for an input already resolved by a source-audited observation",
                "available_missing_inputs": stateful_missing_input_evidence["available_missing_inputs"],
            }
        elif stateful_missing_input_evidence.get("ambiguous_missing_inputs"):
            terminal_review = {
                "attempted": True,
                "allowed": False,
                "reason": "candidate asks for an identifier with multiple source-audited candidates",
                "ambiguous_missing_inputs": stateful_missing_input_evidence["ambiguous_missing_inputs"],
            }
        else:
            terminal_review = _review_stateful_terminal_decision(
                model,
                tokenizer,
                generate_text,
                user_request,
                tools,
                binding_plan,
                completed_calls,
                max_new_tokens,
                available_missing_inputs=stateful_missing_input_evidence.get("available_missing_inputs") or [],
            )
        stateful_semantic_review_result = terminal_review
        if not terminal_review["allowed"]:
            repaired_plan, repair_info = _stateful_progress_repair_plan(
                model,
                tokenizer,
                generate_text,
                user_request,
                binding_text,
                tools,
                completed_calls,
                max_new_tokens,
                semantic_only=stateful_semantic_only,
                reviewer_feedback=(
                    str(terminal_review.get("reason") or "terminal decision rejected")
                    + resolved_missing_input_feedback
                    + ambiguous_missing_input_feedback
                ),
                stateful_goal_ledger=stateful_goal_ledger,
                stateful_goal_ledger_required=stateful_goal_ledger_required,
                no_action_was_rejected=True,
                resolved_missing_inputs=resolved_missing_inputs,
            )
            terminal_review["replan"] = repair_info
            if repaired_plan is not None and repaired_plan.get("calls"):
                repaired_graph = _graph_from_binding_plan(runtime, binding_text, repaired_plan)
                repaired_output = runtime.compile(
                    repaired_graph,
                    binding_text,
                    allow_side_effects=allow_side_effects,
                )
                if repaired_output.verification.ok:
                    binding_plan = repaired_plan
                    binding_recovery_plan = repaired_plan
                    graph = repaired_graph
                    output = repaired_output
                else:
                    binding_plan = _stateful_semantic_only_no_call_plan(
                        binding_plan,
                        reason="semantic terminal repair did not compile",
                    )
                    graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                    output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
            elif repaired_plan is not None and _stateful_semantic_no_call_is_actionable(repaired_plan):
                repair_provenance = repaired_plan.get("stateful_provenance_repair")
                if isinstance(repair_provenance, dict) and repair_provenance.get("used"):
                    repair_terminal_review = {
                        "attempted": False,
                        "allowed": True,
                        "reason": "semantic repair selected a distinct clarification after provenance validation",
                        "provenance_repair": repair_provenance,
                    }
                else:
                    repair_terminal_review = _review_stateful_terminal_decision(
                        model,
                        tokenizer,
                        generate_text,
                        user_request,
                        tools,
                        repaired_plan,
                        completed_calls,
                        max_new_tokens,
                        available_missing_inputs=stateful_missing_input_evidence.get("available_missing_inputs") or [],
                    )
                terminal_review["repair_terminal_review"] = repair_terminal_review
                if repair_terminal_review["allowed"]:
                    binding_plan = repaired_plan
                    binding_recovery_plan = repaired_plan
                    graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                    output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
                else:
                    secondary_plan, secondary_repair = _stateful_progress_repair_plan(
                        model,
                        tokenizer,
                        generate_text,
                        user_request,
                        binding_text,
                        tools,
                        completed_calls,
                        max_new_tokens,
                        semantic_only=stateful_semantic_only,
                        reviewer_feedback=(
                            str(repair_terminal_review.get("reason") or "repaired terminal decision rejected")
                            + resolved_missing_input_feedback
                            + ambiguous_missing_input_feedback
                        ),
                        stateful_goal_ledger=stateful_goal_ledger,
                        stateful_goal_ledger_required=stateful_goal_ledger_required,
                        no_action_was_rejected=True,
                        resolved_missing_inputs=resolved_missing_inputs,
                    )
                    terminal_review["secondary_replan"] = secondary_repair
                    if secondary_plan is not None and secondary_plan.get("calls"):
                        secondary_graph = _graph_from_binding_plan(runtime, binding_text, secondary_plan)
                        secondary_output = runtime.compile(
                            secondary_graph,
                            binding_text,
                            allow_side_effects=allow_side_effects,
                        )
                        if secondary_output.verification.ok:
                            binding_plan = secondary_plan
                            binding_recovery_plan = secondary_plan
                            graph = secondary_graph
                            output = secondary_output
                        else:
                            binding_plan = _stateful_semantic_only_no_call_plan(
                                binding_plan,
                                reason="secondary semantic terminal repair did not compile",
                            )
                            graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                            output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
                    else:
                        binding_plan = _stateful_semantic_only_no_call_plan(
                            binding_plan,
                            reason="semantic terminal review rejected the repaired no-action decision",
                        )
                        graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                        output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
            else:
                binding_plan = _stateful_semantic_only_no_call_plan(
                    binding_plan,
                    reason="semantic state-transition review rejected a no-progress terminal decision",
                )
                graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
    if stateful and stateful_semantic_review and output.calls:
        stateful_semantic_review_result = _review_stateful_candidate_calls(
            model,
            tokenizer,
            generate_text,
            user_request,
            tools,
            compiled_calls_to_dicts(output.calls),
            completed_calls,
            max_new_tokens,
            runtime=runtime,
        )
        if terminal_review is not None:
            stateful_semantic_review_result["terminal_review"] = terminal_review
        if not stateful_semantic_review_result["allowed"]:
            repaired_plan, repair_info = _stateful_progress_repair_plan(
                model,
                tokenizer,
                generate_text,
                user_request,
                binding_text,
                tools,
                completed_calls,
                max_new_tokens,
                semantic_only=stateful_semantic_only,
                reviewer_feedback=str(stateful_semantic_review_result.get("reason") or "candidate rejected"),
                stateful_goal_ledger=stateful_goal_ledger,
                stateful_goal_ledger_required=stateful_goal_ledger_required,
            )
            stateful_semantic_review_result["replan"] = repair_info
            if repaired_plan is not None and repaired_plan.get("calls"):
                repaired_graph = _graph_from_binding_plan(runtime, binding_text, repaired_plan)
                repaired_output = runtime.compile(
                    repaired_graph,
                    binding_text,
                    allow_side_effects=allow_side_effects,
                )
                if repaired_output.verification.ok:
                    repair_review = _review_stateful_candidate_calls(
                        model,
                        tokenizer,
                        generate_text,
                        user_request,
                        tools,
                        compiled_calls_to_dicts(repaired_output.calls),
                        completed_calls,
                        max_new_tokens,
                        runtime=runtime,
                    )
                    stateful_semantic_review_result["repair_review"] = repair_review
                    if repair_review["allowed"]:
                        binding_plan = repaired_plan
                        binding_recovery_plan = repaired_plan
                        graph = repaired_graph
                        output = repaired_output
                    else:
                        binding_plan = _stateful_semantic_only_no_call_plan(
                            binding_plan,
                            reason="semantic state-transition review rejected both bounded candidates",
                        )
                        graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                        output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
                else:
                    binding_plan = _stateful_semantic_only_no_call_plan(
                        binding_plan,
                        reason="semantic repair did not compile",
                    )
                    graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                    output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
            elif repaired_plan is not None and _stateful_semantic_no_call_is_actionable(repaired_plan):
                repair_terminal_review = _review_stateful_terminal_decision(
                    model,
                    tokenizer,
                    generate_text,
                    user_request,
                    tools,
                    repaired_plan,
                    completed_calls,
                    max_new_tokens,
                    available_missing_inputs=stateful_missing_input_evidence.get("available_missing_inputs") or [],
                )
                stateful_semantic_review_result["repair_terminal_review"] = repair_terminal_review
                if repair_terminal_review["allowed"]:
                    binding_plan = repaired_plan
                    binding_recovery_plan = repaired_plan
                    graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                    output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
                else:
                    binding_plan = _stateful_semantic_only_no_call_plan(
                        binding_plan,
                        reason="semantic candidate review rejected the repaired no-action decision",
                    )
                    graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                    output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
            else:
                binding_plan = _stateful_semantic_only_no_call_plan(
                    binding_plan,
                    reason="semantic state-transition review rejected the candidate action",
                )
                graph = _graph_from_binding_plan(runtime, binding_text, binding_plan)
                output = runtime.compile(graph, binding_text, allow_side_effects=allow_side_effects)
    if not output.verification.ok and binding_recovery_plan is None and not stateful_semantic_only:
        raw_binding_plan = raw_binding_plan or build_tool_binding_plan(binding_text, tools)
        if raw_binding_plan.get("calls"):
            raw_graph = _graph_from_binding_plan(runtime, binding_text, raw_binding_plan)
            raw_output = runtime.compile(raw_graph, binding_text, allow_side_effects=allow_side_effects)
            if raw_output.verification.ok:
                binding_recovery_plan = raw_binding_plan
                binding_plan = raw_binding_plan
                graph = raw_graph
                output = raw_output
    skeleton_output: dict[str, Any] = {"used": False}
    if (
        not stateful
        and not semantic_terminal_veto_active
        and _should_generate_call_skeleton(binding_text, tools, binding_plan)
    ):
        skeleton_output = _generate_call_skeleton(
            model,
            tokenizer,
            generate_text,
            binding_text,
            tools,
            binding_plan,
            max_new_tokens,
        )
        skeleton_plan = _binding_plan_from_call_skeleton(
            binding_text,
            tools,
            skeleton_output.get("parsed"),
            binding_plan,
        )
        if skeleton_plan is not None and _prefer_skeleton_binding_plan(binding_text, binding_plan, skeleton_plan):
            skeleton_graph = _graph_from_binding_plan(runtime, binding_text, skeleton_plan)
            skeleton_compile = runtime.compile(
                skeleton_graph,
                binding_text,
                allow_side_effects=allow_side_effects,
            )
            skeleton_output["verification_ok"] = skeleton_compile.verification.ok
            skeleton_output["diagnostics"] = diagnostics_to_dicts(skeleton_compile.verification)
            if skeleton_compile.verification.ok:
                binding_plan = skeleton_plan
                graph = skeleton_graph
                output = skeleton_compile
                skeleton_output["used"] = True
    parsed = graph
    calls = output.calls
    verification = output.verification
    diagnostics = diagnostics_to_dicts(output.verification)
    retry_outputs: list[dict[str, Any]] = []

    for attempt in range(max(0, repair_attempts)):
        if verification.ok:
            break
        retry_messages = _build_repair_messages(
            user_request=binding_text,
            previous_output=json.dumps(parsed, ensure_ascii=False),
            parse_error=None,
            diagnostics=diagnostics,
            attempt=attempt + 1,
        )
        retry_text = generate_text(model, tokenizer, retry_messages, max_new_tokens)
        retry_parsed, retry_parse_error = extract_goal_graph_json_object(retry_text)
        retry_record: dict[str, Any] = {
            "raw_text": retry_text,
            "parse_error": retry_parse_error,
            "verification_ok": False,
            "diagnostics": [],
        }
        if retry_parsed is not None:
            retry_output = runtime.compile(
                retry_parsed,
                binding_text,
                allow_side_effects=allow_side_effects,
            )
            retry_record["verification_ok"] = retry_output.verification.ok
            retry_record["diagnostics"] = diagnostics_to_dicts(retry_output.verification)
            if retry_output.verification.ok:
                parsed = retry_parsed
                calls = retry_output.calls
                verification = retry_output.verification
                diagnostics = retry_record["diagnostics"]
        retry_outputs.append(retry_record)

    verification_ok = bool(verification and verification.ok)
    output_goal_ledger = _stateful_goal_ledger_from_binding_plan(binding_plan, stateful_goal_ledger)
    output_goal_delta = _stateful_goal_delta_from_binding_plan(binding_plan)
    output_requested_fact_delta = _stateful_requested_fact_delta_from_binding_plan(binding_plan)
    return {
        "planner_mode": "stepwise",
        "stateful": stateful,
        "stateful_execution_history": completed_calls,
        "stateful_goal_ledger": output_goal_ledger,
        "stateful_goal_delta": output_goal_delta,
        "stateful_requested_fact_delta": output_requested_fact_delta,
        "binding_request": binding_text,
        "binding_request_separate": binding_text != user_request,
        "raw_text": semantic_output.get("raw_text") or "",
        "graph": parsed,
        "parse_error": None,
        "verification_ok": verification_ok,
        "diagnostics": diagnostics,
        "diagnostic_codes": diagnostic_codes(diagnostics),
        "calls": compiled_calls_to_dicts(calls),
        "retry_outputs": retry_outputs,
        "steps": [
            {
                "step": "semantic_frame",
                "model_call": True,
                "ok": isinstance(semantic_frame, dict),
                "parse_error": semantic_output.get("parse_error"),
            },
            {
                "step": "tool_binding",
                "model_call": False,
                "ok": bool(binding_plan.get("calls")) or binding_plan.get("tool_decision") in {"no_tool", "ask_user"},
                "tool_decision": binding_plan.get("tool_decision"),
                "call_count": len(binding_plan.get("calls") or []),
                "model_binding_used": bool((binding_plan.get("model_tool_binding") or {}).get("used")),
                "model_binding_accepted": bool((binding_plan.get("model_tool_binding") or {}).get("accepted")),
                "raw_query_recovery_used": binding_recovery_plan is not None,
                "semantic_no_tool_veto": semantic_no_tool_veto,
                "semantic_ask_user_veto": semantic_ask_user_veto,
                "semantic_terminal_veto_active": semantic_terminal_veto_active,
                "stateful_action_recovery_used": stateful_action_recovery_plan is not None,
                "stateful_raw_safety_filter_used": stateful_raw_safety_filter["used"],
                "stateful_replayed_call_filter_used": stateful_progress_filter["used"],
                "stateful_schema_recovery_used": stateful_schema_recovery["used"],
                "stateful_single_call_execution_used": stateful_single_call_execution["used"],
                "stateful_terminal_progress_repair_needed": stateful_terminal_progress_needed,
                "stateful_progress_repair_used": stateful_progress_repair["used"],
                "semantic_enum_grounding_used": semantic_enum_grounding["used"],
            },
            {
                "step": "graph_synthesis",
                "model_call": False,
                "ok": True,
                "node_count": len(parsed.get("nodes") or []) if isinstance(parsed, dict) else 0,
            },
            {
                "step": "call_skeleton",
                "model_call": bool(skeleton_output.get("attempted")),
                "ok": bool(skeleton_output.get("used")),
                "call_count": len((skeleton_output.get("plan") or {}).get("calls") or []),
                "parse_error": skeleton_output.get("parse_error"),
            },
            {
                "step": "runtime_verify_compile",
                "model_call": False,
                "ok": verification_ok,
                "diagnostic_codes": diagnostic_codes(diagnostics),
                "call_count": len(calls),
            },
        ],
        "semantic_frame_output": semantic_output,
        "tool_binding_plan": binding_plan,
        "tool_binding_recovery_plan": binding_recovery_plan or {},
        "call_skeleton_output": skeleton_output,
        "stateful_action_recovery_plan": stateful_action_recovery_plan or {},
        "stateful_raw_safety_filter": stateful_raw_safety_filter,
        "stateful_progress_filter": stateful_progress_filter,
        "stateful_schema_recovery": stateful_schema_recovery,
        "stateful_collection_disambiguation": stateful_collection_disambiguation,
        "stateful_single_call_execution": stateful_single_call_execution,
        "stateful_progress_repair": stateful_progress_repair,
        "semantic_enum_grounding": semantic_enum_grounding,
        "stateful_semantic_only": stateful_semantic_only,
        "stateful_semantic_review": stateful_semantic_review_result,
        "stateful_missing_input_evidence": stateful_missing_input_evidence,
        "binder_fallback": {"used": False},
        "capability_count": len(runtime.registry),
        "capabilities": list(runtime.registry.keys()),
    }


def _drop_stateful_unverified_fallback_calls(
    binding_plan: dict[str, Any],
    tools: list[dict[str, Any]],
    user_request: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep raw fallback from executing non-progress tools in live workflows.

    A semantic model binding has already passed schema and evidence checks. A
    deterministic fallback has not established enough intent to safely invoke
    terminal, escalation, zero-input, or private-reasoning tools on its own.
    """
    plan = copy.deepcopy(binding_plan) if isinstance(binding_plan, dict) else {}
    model_report = plan.get("model_tool_binding") if isinstance(plan.get("model_tool_binding"), dict) else {}
    if model_report.get("accepted"):
        return plan, {"used": False, "dropped_calls": []}
    tool_lookup = {
        str(normalize_tool(raw).get("name") or ""): normalize_tool(raw)
        for raw in tools or []
        if isinstance(raw, dict) and str(normalize_tool(raw).get("name") or "")
    }
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for call in plan.get("calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or "").strip()
        tool = tool_lookup.get(name)
        reason = _stateful_unverified_fallback_reason(tool, user_request)
        if reason:
            dropped.append(
                {
                    "tool_name": name,
                    "arguments": copy.deepcopy(call.get("arguments") if isinstance(call.get("arguments"), dict) else {}),
                    "reason": reason,
                }
            )
            continue
        kept.append(call)
    if not dropped:
        return plan, {"used": False, "dropped_calls": []}
    for index, call in enumerate(kept, start=1):
        call["id"] = f"call_{index}"
    plan["calls"] = kept
    if not kept and plan.get("tool_decision") == "call":
        plan["tool_decision"] = "no_tool"
        plan["reason"] = "all raw fallback calls lacked stateful execution evidence"
    return plan, {"used": True, "dropped_calls": dropped}


def _stateful_unverified_fallback_reason(tool: dict[str, Any] | None, user_request: str) -> str:
    if tool is None:
        return "raw fallback referenced an unavailable tool"
    name = str(tool.get("name") or "").lower()
    description = str(tool.get("description") or "").lower()
    combined = f"{name} {description}"
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    if _is_terminal_response_tool(tool):
        return "raw fallback cannot synthesize a stateful terminal response"
    if re.search(r"(?:transfer|escalat|handoff|hand_off|human|supervisor)", combined):
        return "raw fallback cannot escalate without an accepted semantic binding"
    if re.search(r"\b(?:think|reason|reflect|internal plan)\b", combined):
        return "raw fallback cannot execute a private-reasoning tool"
    if not props:
        return "raw fallback cannot invoke a zero-input tool without an accepted semantic binding"
    if re.search(r"\b(?:calculate|compute|math)\b", combined) and not re.search(
        r"\b(?:calculate|compute|math|sum|difference|total|average|percent|factorial|multiply|divide)\b",
        user_request,
        re.I,
    ):
        return "raw fallback cannot invoke a generic calculator without calculation intent"
    return ""


def _successful_execution_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize observed calls supplied by a stateful benchmark adapter.

    The history keeps the action signature plus a bounded structured observation.
    That observation is evidence for a later stateful replan; it is never a
    substitute for schema validation or grounding.
    """
    completed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in history or []:
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome") or item.get("status") or "").strip().lower()
        if outcome not in {"success", "succeeded", "ok", "completed", "failure", "failed", "error"}:
            continue
        name = str(item.get("tool_name") or item.get("name") or "").strip()
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if not name:
            continue
        signature = _stateful_call_signature(name, arguments)
        if signature in seen:
            continue
        seen.add(signature)
        normalized_outcome = "success" if outcome in {"success", "succeeded", "ok", "completed"} else "failure"
        normalized = {
            "tool_name": name,
            "arguments": copy.deepcopy(arguments),
            "outcome": normalized_outcome,
        }
        observation = _compact_stateful_observation(item.get("observation"))
        if observation is not None:
            normalized["observation"] = observation
        completed.append(normalized)
    return completed


def _stateful_verified_observation_facts(
    history: list[dict[str, Any]] | None,
    *,
    max_facts: int = 80,
    per_observation_max: int | None = None,
) -> list[dict[str, Any]]:
    """Flatten successful observations into compact, source-attributed state facts."""
    facts: list[dict[str, Any]] = []
    observation_start = 0

    def append_value(tool_name: str, path: str, value: Any) -> None:
        if (
            len(facts) >= max_facts
            or value is None
            or (per_observation_max is not None and len(facts) - observation_start >= per_observation_max)
        ):
            return
        if isinstance(value, str):
            value = value[:240]
        if not isinstance(value, (str, int, float, bool)):
            return
        facts.append({"tool_name": tool_name, "path": path[:180], "value": value})

    def visit(tool_name: str, value: Any, path: str) -> None:
        """Preserve shallow record fields before nested collections exhaust a budget."""
        if len(facts) >= max_facts:
            return
        if not isinstance(value, (dict, list)):
            append_value(tool_name, path, value)
            return

        queue: list[tuple[Any, str, int]] = []
        if isinstance(value, dict):
            children = [
                (child, f"{path}.{key}" if path else str(key), 1)
                for key, child in list(value.items())[:40]
            ]
        else:
            children = [(child, f"{path}[{index}]", 1) for index, child in enumerate(value[:20])]

        # Direct scalar fields identify the record itself; keep them even when a
        # preceding nested object (for example, preferences or payment methods)
        # is large enough to fill the evidence cap on its own.
        for child, child_path, depth in children:
            if not isinstance(child, (dict, list)) and depth <= 5:
                append_value(tool_name, child_path, child)

        queue.extend((child, child_path, depth) for child, child_path, depth in children if isinstance(child, (dict, list)))
        index = 0
        while index < len(queue):
            current, current_path, depth = queue[index]
            index += 1
            if (
                len(facts) >= max_facts
                or depth > 5
                or (per_observation_max is not None and len(facts) - observation_start >= per_observation_max)
            ):
                break
            if isinstance(current, dict):
                nested_children = [
                    (child, f"{current_path}.{key}" if current_path else str(key), depth + 1)
                    for key, child in list(current.items())[:40]
                ]
            else:
                nested_children = [
                    (child, f"{current_path}[{item_index}]", depth + 1)
                    for item_index, child in enumerate(current[:20])
                ]
            for child, child_path, child_depth in nested_children:
                if child_depth > 5:
                    continue
                if isinstance(child, (dict, list)):
                    queue.append((child, child_path, child_depth))
                else:
                    append_value(tool_name, child_path, child)

    for item in history or []:
        if not isinstance(item, dict) or str(item.get("outcome") or "").lower() != "success":
            continue
        observation_start = len(facts)
        observation = item.get("observation")
        if isinstance(observation, str):
            try:
                parsed_observation = json.loads(observation)
            except (TypeError, ValueError):
                parsed_observation = observation
            observation = parsed_observation
        tool_name = str(item.get("tool_name") or item.get("name") or "observation")[:120]
        visit(tool_name, observation, "")
    return facts


def _stateful_execution_summary(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep prior action provenance in prompts without duplicating observations.

    Full observations remain in the runtime history for verification.  The semantic
    prompt receives source-attributed facts separately, so repeating nested raw
    observations here only consumes context without adding new evidence.
    """
    summary: list[dict[str, Any]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool_name") or item.get("name") or "").strip()
        if not name:
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        summary.append(
            {
                "tool_name": name,
                "arguments": copy.deepcopy(arguments),
                "outcome": str(item.get("outcome") or "").strip().lower() or "unknown",
            }
        )
    return summary


def _compact_stateful_observation(value: Any, *, depth: int = 0) -> Any:
    """Keep serializable environment evidence small enough for planner context."""
    if depth > 5:
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            compact = _compact_stateful_observation(item, depth=depth + 1)
            if compact is not None:
                result[str(key)[:120]] = compact
        return result
    if isinstance(value, list):
        result = []
        for item in value[:20]:
            compact = _compact_stateful_observation(item, depth=depth + 1)
            if compact is not None:
                result.append(compact)
        return result
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:500]


def _normalize_stateful_goal_ledger(value: Any) -> dict[str, Any]:
    """Bound model-proposed semantic continuity state for live tool use."""
    if not isinstance(value, dict):
        return {}
    raw_goals = value.get("goals")
    if not isinstance(raw_goals, list):
        return {}
    goals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_goal in enumerate(raw_goals[:16], start=1):
        if not isinstance(raw_goal, dict):
            continue
        goal_id = str(raw_goal.get("id") or f"goal_{index}").strip()[:80]
        objective = str(raw_goal.get("objective") or raw_goal.get("description") or "").strip()[:600]
        if not goal_id or not objective or goal_id in seen_ids:
            continue
        seen_ids.add(goal_id)
        status = str(raw_goal.get("status") or "pending").strip().lower()
        if status not in {"pending", "blocked", "completed"}:
            status = "pending"
        raw_dependencies = raw_goal.get("depends_on")
        if not isinstance(raw_dependencies, (list, tuple, set)):
            raw_dependencies = []
        depends_on = [
            str(item).strip()[:80]
            for item in raw_dependencies
            if isinstance(item, (str, int, float)) and str(item).strip()
        ][:8]
        goals.append(
            {
                "id": goal_id,
                "objective": objective,
                "status": status,
                "depends_on": depends_on,
            }
        )
    if not goals:
        return {}
    next_goal_id = str(value.get("next_goal_id") or "").strip()[:80]
    goal_ids = {goal["id"] for goal in goals}
    if next_goal_id not in goal_ids:
        next_goal_id = next((goal["id"] for goal in goals if goal["status"] != "completed"), "")
    return {"goals": goals, "next_goal_id": next_goal_id}


def _stateful_goal_ledger_from_binding_plan(
    binding_plan: dict[str, Any],
    fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else None
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan, dict) else None
    proposed = _normalize_stateful_goal_ledger(frame.get("goal_ledger") if isinstance(frame, dict) else None)
    return _merge_stateful_goal_ledgers(fallback, proposed)


def _stateful_goal_delta_from_binding_plan(binding_plan: dict[str, Any]) -> dict[str, Any]:
    """Expose an additive goal proposal without giving it ledger authority.

    The live runtime validates and applies this object.  Static benchmarks can
    ignore it, and the compiler never treats it as tool-call evidence.
    """
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else None
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan, dict) else None
    delta = frame.get("goal_delta") if isinstance(frame, dict) else None
    if not isinstance(delta, dict):
        return {}
    additions = delta.get("add")
    if not isinstance(additions, list):
        return {}
    return {"add": copy.deepcopy(additions[:16])}


def _stateful_requested_fact_delta_from_binding_plan(binding_plan: dict[str, Any]) -> dict[str, Any]:
    """Return only proposed requested-state updates for runtime validation."""
    capability_plan = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else None
    frame = capability_plan.get("semantic_input_frame") if isinstance(capability_plan, dict) else None
    delta = frame.get("requested_fact_delta") if isinstance(frame, dict) else None
    if not isinstance(delta, dict):
        return {}
    updates = delta.get("set")
    if not isinstance(updates, list):
        return {}
    return {"set": copy.deepcopy(updates[:16])}


def _merge_stateful_goal_ledgers(
    previous: dict[str, Any] | None,
    proposed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve model-declared goal order across stateful planning turns.

    The model remains responsible for creating goals and updating their status.
    The runtime preserves goal identity and queue order so later user requests
    cannot silently preempt an unresolved earlier goal.
    """
    prior = _normalize_stateful_goal_ledger(previous)
    candidate = _normalize_stateful_goal_ledger(proposed)
    if not prior:
        return candidate
    if not candidate:
        return prior

    proposed_by_id = {goal["id"]: goal for goal in candidate["goals"]}
    merged: list[dict[str, Any]] = []
    for prior_goal in prior["goals"]:
        updated = proposed_by_id.pop(prior_goal["id"], None)
        if updated is None:
            merged.append(copy.deepcopy(prior_goal))
            continue
        merged.append(
            {
                "id": prior_goal["id"],
                "objective": prior_goal["objective"],
                "status": updated["status"],
                "depends_on": updated["depends_on"] or prior_goal["depends_on"],
            }
        )
    for proposed_goal in candidate["goals"]:
        if proposed_goal["id"] in proposed_by_id:
            merged.append(copy.deepcopy(proposed_goal))

    prior_next = str(prior.get("next_goal_id") or "")
    merged_by_id = {goal["id"]: goal for goal in merged}
    if prior_next in merged_by_id and merged_by_id[prior_next]["status"] != "completed":
        next_goal_id = prior_next
    else:
        next_goal_id = next((goal["id"] for goal in merged if goal["status"] != "completed"), "")
    return {"goals": merged, "next_goal_id": next_goal_id}


def _stateful_call_signature(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    return name, json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


def _drop_successfully_replayed_calls(
    binding_plan: dict[str, Any],
    completed_calls: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove exact calls whose previous observation cannot add new state."""
    plan = copy.deepcopy(binding_plan) if isinstance(binding_plan, dict) else {}
    completed_signatures = {
        _stateful_call_signature(
            str(item.get("tool_name") or ""),
            item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
        )
        for item in completed_calls or []
        if isinstance(item, dict)
    }
    calls = [call for call in plan.get("calls") or [] if isinstance(call, dict)]
    if not completed_signatures or not calls:
        return plan, {"used": False, "dropped_calls": []}

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("tool_name") or call.get("name") or "").strip()
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if _stateful_call_signature(name, arguments) in completed_signatures:
            dropped.append(
                {
                    "tool_name": name,
                    "arguments": copy.deepcopy(arguments),
                    "reason": "identical call already has an observation in stateful execution",
                }
            )
            continue
        kept.append(call)
    if not dropped:
        return plan, {"used": False, "dropped_calls": []}
    for index, call in enumerate(kept, start=1):
        call["id"] = f"call_{index}"
    plan["calls"] = kept
    if not kept and plan.get("tool_decision") == "call":
        plan["tool_decision"] = "no_tool"
        plan["reason"] = "all candidate calls would repeat completed stateful actions"
    return plan, {"used": True, "dropped_calls": dropped}


def _limit_stateful_plan_to_one_call(
    binding_plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one model-ordered transition per stateful environment turn.

    This does not select a tool semantically: GPT-OSS already supplied the
    ordering.  It makes the state-machine boundary explicit so every later
    action can incorporate the observation caused by the preceding action.
    """
    plan = copy.deepcopy(binding_plan) if isinstance(binding_plan, dict) else {}
    calls = [call for call in plan.get("calls") or [] if isinstance(call, dict)]
    if len(calls) <= 1:
        return plan, {"used": False, "deferred_calls": []}
    plan["calls"] = [calls[0]]
    plan["calls"][0]["id"] = "call_1"
    return plan, {
        "used": True,
        "selected_call": {
            "tool_name": str(calls[0].get("tool_name") or calls[0].get("name") or ""),
            "arguments": copy.deepcopy(calls[0].get("arguments") if isinstance(calls[0].get("arguments"), dict) else {}),
        },
        "deferred_calls": [
            {
                "tool_name": str(call.get("tool_name") or call.get("name") or ""),
                "arguments": copy.deepcopy(call.get("arguments") if isinstance(call.get("arguments"), dict) else {}),
            }
            for call in calls[1:]
        ],
    }


def _stateful_semantic_only_no_call_plan(
    binding_plan: dict[str, Any],
    *,
    reason: str = "stateful semantic binding was unavailable or rejected",
) -> dict[str, Any]:
    plan = copy.deepcopy(binding_plan) if isinstance(binding_plan, dict) else {}
    plan["calls"] = []
    plan["tool_decision"] = "no_tool"
    plan["reason"] = reason
    return plan


def _review_stateful_candidate_calls(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    completed_calls: list[dict[str, Any]],
    max_new_tokens: int,
    *,
    runtime: GoalGraphRuntime | None = None,
) -> dict[str, Any]:
    """Ask GPT-OSS to judge a candidate transition; it cannot create a call."""
    candidate_payload = [
        {
            "tool_name": str(call.get("tool_name") or ""),
            "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
        }
        for call in calls
    ]
    payload = {
        "candidate_calls": candidate_payload,
        "available_tools": [_compact_tool_schema(normalize_tool(tool)) for tool in tools if isinstance(tool, dict)],
        "observations": [
            item.get("observation")
            for item in completed_calls
            if isinstance(item, dict) and item.get("outcome") == "success" and "observation" in item
        ],
        "verified_observation_facts": _stateful_verified_observation_facts(completed_calls),
        "user_request": user_request,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one minified JSON object and nothing else. You are a state-transition "
                "reviewer, not an action generator. Judge only the supplied candidate calls. Allow a "
                "candidate only when it is a grounded immediate step toward the user request and is "
                "compatible with observed environment records and the tool schema. Reject unrelated, "
                "replayed, speculative, or state-incompatible actions. candidate_calls is authoritative: "
                "do not substitute an argument from the request or observations for a candidate argument. "
                "verified_observation_facts is authoritative state derived from successful tool results. "
                "A user declining to restate a fact is not a conflicting replacement. A read-only "
                "retrieval over one audited member of an unresolved observed collection is a bounded "
                "disambiguation step, not a choice of target: allow it when its result can identify the "
                "matching record. Do not apply that exception to a write; a write still needs a verified "
                "target match. A rejection is actionable only when checks identifies a defined false "
                "predicate, failed_check names that predicate, and evidence_ids contains at least one "
                "source identifier. Do not reject a property that deterministic schema/evidence checks "
                "already proved."
            ),
        },
        {
            "role": "user",
            "content": (
                'Output fields: verdict ("allow" or "reject"), checks, failed_check, evidence_ids, reason, candidate_calls. '
                "checks may use only supports_active_goal, target_uniquely_resolved, "
                "arguments_match_current_request, facts_are_current, policy_preconditions_satisfied, "
                "confirmation_valid, effect_not_already_completed; each value is true, false, or unknown. "
                "candidate_calls must exactly reproduce the supplied candidate_calls before you judge it. "
                f"Input:{_compact_json(payload)}"
            ),
        },
    ]
    raw_text, parsed, parse_error, format_recovery = _generate_reviewer_json(
        model,
        tokenizer,
        generate_text,
        messages,
        min(max(max_new_tokens, 1000), 1200),
        'verdict ("allow" or "reject"), checks, failed_check, evidence_ids, reason, candidate_calls',
        payload_validator=_reviewer_has_allow_reject_verdict,
    )
    if not isinstance(parsed, dict):
        return {
            "attempted": True,
            "allowed": False,
            "reason": "semantic reviewer did not return valid JSON",
            "parse_error": parse_error,
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    echoed_calls = parsed.get("candidate_calls")
    if echoed_calls != candidate_payload:
        # The compiler already checked the actual candidate's schema and
        # evidence. A reviewer that evaluates a different call is not a valid
        # safety signal, so retain the verified candidate rather than repairing
        # toward the reviewer hallucination.
        return {
            "attempted": True,
            "allowed": True,
            "abstained": True,
            "reason": "semantic reviewer did not faithfully identify the supplied candidate calls",
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    verdict = str(parsed.get("verdict") or "").strip().lower()
    structured_rejection = _structured_candidate_rejection(parsed)
    if verdict == "reject" and structured_rejection is None:
        return {
            "attempted": True,
            "allowed": True,
            "abstained": True,
            "verdict": verdict,
            "reason": "semantic reviewer rejection lacked an actionable structured predicate",
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    bounded_disambiguation = _stateful_bounded_readonly_disambiguation(
        candidate_payload,
        completed_calls,
        runtime,
    )
    if verdict != "allow" and bounded_disambiguation["allowed"]:
        # The compiler has already validated the schema and evidence.  For this
        # one safe transition, source-attributed ambiguity is stronger than a
        # reviewer that mistakes a read for an unrequested target selection.
        return {
            "attempted": True,
            "allowed": True,
            "verdict": verdict,
            "overridden": True,
            "reason": bounded_disambiguation["reason"],
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    return {
        "attempted": True,
        "allowed": verdict == "allow",
        "verdict": verdict,
        "reason": str(parsed.get("reason") or ""),
        **(structured_rejection or {}),
        "raw_text": raw_text,
        "format_recovery": format_recovery,
    }


_CANDIDATE_REVIEW_CHECKS = {
    "supports_active_goal",
    "target_uniquely_resolved",
    "arguments_match_current_request",
    "facts_are_current",
    "policy_preconditions_satisfied",
    "confirmation_valid",
    "effect_not_already_completed",
}


def _structured_candidate_rejection(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept only predicate-level reviewer rejections with cited state."""
    checks = payload.get("checks")
    failed_check = str(payload.get("failed_check") or "").strip()
    evidence_ids = payload.get("evidence_ids")
    if not isinstance(checks, dict) or failed_check not in _CANDIDATE_REVIEW_CHECKS:
        return None
    if str(checks.get(failed_check) or "").strip().lower() != "false":
        return None
    if any(str(value).strip().lower() not in {"true", "false", "unknown"} for value in checks.values()):
        return None
    if not isinstance(evidence_ids, list):
        return None
    normalized_evidence = [str(value).strip()[:240] for value in evidence_ids if str(value).strip()]
    if not normalized_evidence:
        return None
    return {
        "checks": {str(key): str(value).strip().lower() for key, value in checks.items()},
        "failed_check": failed_check,
        "evidence_ids": normalized_evidence[:16],
    }


def _stateful_bounded_readonly_disambiguation(
    calls: list[dict[str, Any]],
    completed_calls: list[dict[str, Any]],
    runtime: GoalGraphRuntime | None,
) -> dict[str, Any]:
    """Recognize one provenance-grounded read over an ambiguous record collection.

    This is a verifier rule rather than a route-selection rule.  It cannot
    choose a tool or value: it only preserves a compiled, single read-only
    candidate when its identifier is an audited member of a collection with
    multiple possible members.  Writes remain subject to normal review.
    """
    if runtime is None or len(calls) != 1:
        return {"allowed": False}
    call = calls[0] if isinstance(calls[0], dict) else {}
    tool_name = str(call.get("tool_name") or "")
    capability = runtime.registry.get(tool_name)
    if (
        capability is None
        or capability.risk != "read_only"
        or capability.kind not in {"resolve", "retrieve", "search", "rank", "decide"}
    ):
        return {"allowed": False}
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    if not arguments:
        return {"allowed": False}
    facts = _stateful_verified_observation_facts(completed_calls)
    for input_name, value in arguments.items():
        if not isinstance(value, (str, int, float, bool)):
            continue
        for fact in facts:
            if fact.get("value") != value:
                continue
            candidates = _stateful_collection_candidates_for_missing_input(
                str(input_name),
                fact,
                facts,
            )
            if len(candidates) > 1:
                return {
                    "allowed": True,
                    "reason": (
                        "verified bounded read-only disambiguation over a "
                        "source-audited ambiguous collection"
                    ),
                }
    return {"allowed": False}


def _stateful_next_collection_readonly_plan(
    binding_text: str,
    runtime: GoalGraphRuntime,
    completed_calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Continue one already-started, source-audited read-only record search.

    A candidate collection is not a target selection.  After one member has
    been read, a terminal decision must not strand the search while a single
    compatible read-only route has uninspected members.  This helper only
    resumes that existing search with the next audited value; it never emits a
    mutation and it requires a unique tool/input route.
    """
    facts = _stateful_verified_observation_facts(completed_calls)
    if not facts:
        return None

    routes: list[tuple[Any, str, list[Any], set[str]]] = []
    for capability in runtime.registry.values():
        if (
            capability.risk != "read_only"
            or capability.kind not in {"resolve", "retrieve", "search", "rank", "decide"}
            or len(capability.required_inputs) != 1
        ):
            continue
        input_name = next(iter(capability.required_inputs))
        attempted = {
            json.dumps(
                (item.get("arguments") if isinstance(item.get("arguments"), dict) else {}).get(input_name),
                sort_keys=True,
                default=str,
            )
            for item in completed_calls
            if isinstance(item, dict)
            and str(item.get("tool_name") or "") == capability.tool_name
            and str(item.get("outcome") or "").lower() in {"success", "succeeded", "ok", "completed"}
            and input_name in (item.get("arguments") if isinstance(item.get("arguments"), dict) else {})
        }
        if not attempted:
            continue
        seen_collections: set[tuple[str, tuple[str, ...]]] = set()
        for fact in facts:
            candidates = _stateful_collection_candidates_for_missing_input(input_name, fact, facts)
            values = [item.get("value") for item in candidates if isinstance(item, dict)]
            values = [value for value in values if isinstance(value, (str, int, float, bool))]
            signature = (
                str(fact.get("path") or "").split("[", 1)[0],
                tuple(json.dumps(value, sort_keys=True, default=str) for value in values),
            )
            if len(values) > 1 and signature not in seen_collections:
                seen_collections.add(signature)
                routes.append((capability, input_name, values, attempted))

    # Several unrelated collections are genuinely ambiguous and remain
    # model-owned. A single route can safely enumerate its audited records.
    unique_routes: dict[tuple[str, str, tuple[str, ...]], tuple[Any, str, list[Any], set[str]]] = {}
    for capability, input_name, values, attempted in routes:
        key = (
            capability.tool_name,
            input_name,
            tuple(json.dumps(value, sort_keys=True, default=str) for value in values),
        )
        unique_routes[key] = (capability, input_name, values, attempted)
    if len(unique_routes) != 1:
        return None
    capability, input_name, values, attempted = next(iter(unique_routes.values()))
    next_value = next(
        (value for value in values if json.dumps(value, sort_keys=True, default=str) not in attempted),
        None,
    )
    if next_value is None:
        return None
    evidence = _fallback_evidence(next_value, binding_text)
    return {
        "tool_decision": "call",
        "reason": "continue source-audited read-only disambiguation before requesting unrelated input",
        "calls": [
            {
                "id": "call_1",
                "tool_name": capability.tool_name,
                "arguments": {input_name: next_value},
                "argument_evidence": {input_name: evidence},
                "depends_on": [],
                "missing_arguments": [],
            }
        ],
        "missing_inputs": [],
        "stateful_collection_disambiguation": {
            "used": True,
            "tool_name": capability.tool_name,
            "input_name": input_name,
            "candidate_ids": [str(value) for value in values],
            "attempted_ids": [
                str(value)
                for value in values
                if json.dumps(value, sort_keys=True, default=str) in attempted
            ],
            "attempted_count": len(attempted),
            "remaining_count": sum(
                1 for value in values if json.dumps(value, sort_keys=True, default=str) not in attempted
            ),
        },
    }


def _terminal_reviewer_restates_candidate_decision(
    payload: dict[str, Any] | None,
    binding_plan: dict[str, Any],
) -> bool:
    """Recognize a reviewer echoing its supplied terminal choice as a verdict alias.

    This preserves the reviewer boundary: the alias is accepted only when it
    names the already-bound candidate decision and that candidate independently
    satisfies the normal terminal-action shape.  It cannot introduce a tool,
    argument, value, or alternative next step.
    """
    if not isinstance(payload, dict):
        return False
    alias = re.sub(r"[^a-z0-9]+", "_", str(payload.get("verdict") or "").strip().lower()).strip("_")
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    direct_missing_inputs = binding_plan.get("missing_inputs") if isinstance(binding_plan.get("missing_inputs"), list) else []
    has_missing_inputs = _binding_plan_semantic_missing_inputs(binding_plan) or any(
        str(item).strip() for item in direct_missing_inputs
    )
    if alias in {"ask_user", "respond", "clarify", "clarification", "need_more_info", "missing_inputs"}:
        return decision == "ask_user" and has_missing_inputs
    if alias in {"no_tool", "no_call", "answer_directly", "direct_answer"}:
        return decision == "no_tool" and bool(_binding_plan_semantic_response_message(binding_plan))
    return False


def _stateful_terminal_decision_needs_review(binding_plan: dict[str, Any]) -> bool:
    """Decide whether a stateful no-action proposal needs semantic scrutiny."""
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    return decision in {"no_tool", "ask_user"}


def _stateful_missing_input_evidence_adjudication(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    binding_plan: dict[str, Any],
    completed_calls: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Let the model match proposed missing inputs to audited observation facts.

    The adjudicator cannot select tools, arguments, or values.  It can only
    identify a candidate missing-input label as already available when it
    quotes a value from the source-attributed fact store.  The runtime then
    prevents that label from being asked of the user again.
    """
    if str(binding_plan.get("tool_decision") or "").strip().lower() != "ask_user":
        return {"attempted": False, "available_missing_inputs": [], "ambiguous_missing_inputs": []}
    candidate_inputs = [
        str(item).strip()
        for item in binding_plan.get("missing_inputs") or []
        if str(item).strip()
    ]
    facts = _stateful_verified_observation_facts(
        completed_calls,
        max_facts=48,
        per_observation_max=16,
    )
    if not candidate_inputs or not facts:
        return {"attempted": False, "available_missing_inputs": [], "ambiguous_missing_inputs": []}
    payload = {
        "candidate_missing_inputs": candidate_inputs,
        "verified_observation_facts": facts,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one minified JSON object and nothing else. You are an evidence "
                "adjudicator, not a planner or action generator. For each candidate missing input, "
                "list it only when a verified observation directly supplies that information. "
                "Use semantic field meaning, not only identical spelling. Every evidence_span must "
                "quote a value from verified_observation_facts. Do not infer a value, choose a tool, "
                "or decide what action should happen next."
            ),
        },
        {
            "role": "user",
            "content": (
                "Output fields: available_missing_inputs list of objects with missing_input and "
                "evidence_span. Omit inputs that are genuinely unavailable. "
                f"Input:{_compact_json(payload)}"
            ),
        },
    ]
    raw_text, parsed, parse_error, format_recovery = _generate_reviewer_json(
        model,
        tokenizer,
        generate_text,
        messages,
        min(max(max_new_tokens, 600), 900),
        "available_missing_inputs list of objects with missing_input and evidence_span",
        payload_validator=lambda value: isinstance(value.get("available_missing_inputs"), list),
    )
    if not isinstance(parsed, dict):
        return {
            "attempted": True,
            "available_missing_inputs": [],
            "ambiguous_missing_inputs": [],
            "parse_error": parse_error,
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    allowed_inputs = {item.lower(): item for item in candidate_inputs}
    available: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed.get("available_missing_inputs") or []:
        if not isinstance(item, dict):
            continue
        requested = str(item.get("missing_input") or "").strip()
        canonical_requested = allowed_inputs.get(requested.lower())
        evidence_span = str(item.get("evidence_span") or "").strip()
        if not canonical_requested or not evidence_span or canonical_requested.lower() in seen:
            continue
        supporting_fact = next(
            (
                fact
                for fact in facts
                if _evidence_in_request(str(fact.get("value") or ""), evidence_span)
            ),
            None,
        )
        if supporting_fact is None:
            continue
        collection_candidates = _stateful_collection_candidates_for_missing_input(
            canonical_requested,
            supporting_fact,
            facts,
        )
        if len(collection_candidates) > 1:
            seen.add(canonical_requested.lower())
            ambiguous.append(
                {
                    "missing_input": canonical_requested,
                    "candidate_facts": collection_candidates,
                }
            )
            continue
        seen.add(canonical_requested.lower())
        available.append(
            {
                "missing_input": canonical_requested,
                "evidence_span": evidence_span,
                "source_fact": supporting_fact,
            }
        )
    return {
        "attempted": True,
        "available_missing_inputs": available,
        "ambiguous_missing_inputs": ambiguous,
        "raw_text": raw_text,
        "format_recovery": format_recovery,
    }


def _stateful_collection_candidates_for_missing_input(
    missing_input: str,
    source_fact: dict[str, Any],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep a collection of opaque candidates ambiguous until a read resolves it."""
    path = str(source_fact.get("path") or "")
    collection_match = re.match(r"^(.*)\[\d+\](?:\.|$)", path)
    if collection_match is None:
        return []
    collection_path = collection_match.group(1)
    input_tokens = {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", str(missing_input).lower())
        if len(token) >= 3 and token not in {"the", "for", "with"}
    }
    collection_tokens = {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", collection_path.lower())
        if len(token) >= 3 and token not in {"the", "for", "with"}
    }
    if not input_tokens.intersection(collection_tokens):
        return []
    candidates = [
        fact
        for fact in facts
        if str(fact.get("path") or "").startswith(f"{collection_path}[")
        and isinstance(fact.get("value"), (str, int, float, bool))
    ]
    distinct_values = {json.dumps(fact.get("value"), sort_keys=True, default=str) for fact in candidates}
    return candidates[:12] if len(distinct_values) > 1 else []


def _stateful_plan_requests_resolved_missing_input(
    binding_plan: dict[str, Any],
    resolved_missing_inputs: list[str] | None,
) -> bool:
    resolved = {
        str(item).strip().lower()
        for item in resolved_missing_inputs or []
        if str(item).strip()
    }
    if not resolved:
        return False
    requested = {
        str(item).strip().lower()
        for item in binding_plan.get("missing_inputs") or []
        if str(item).strip()
    }
    return bool(resolved & requested)


def _stateful_resolved_missing_input_feedback(items: list[Any]) -> str:
    """Render only audited provenance needed by a semantic correction."""
    details: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        missing_input = str(item.get("missing_input") or "").strip()
        evidence_span = str(item.get("evidence_span") or "").strip()
        source_fact = item.get("source_fact") if isinstance(item.get("source_fact"), dict) else {}
        source_path = str(source_fact.get("path") or "").strip()
        if missing_input and evidence_span:
            suffix = f" at {source_path}" if source_path else ""
            details.append(f"{missing_input} is already available as observed evidence {evidence_span!r}{suffix}")
    if not details:
        return ""
    return " Source-audited availability: " + "; ".join(details[:6]) + "."


def _stateful_ambiguous_missing_input_feedback(items: list[Any]) -> str:
    """Tell a semantic repair that observed candidates are not a resolved identifier."""
    details: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        missing_input = str(item.get("missing_input") or "").strip()
        candidates = item.get("candidate_facts") if isinstance(item.get("candidate_facts"), list) else []
        values = [str(candidate.get("value") or "").strip() for candidate in candidates if isinstance(candidate, dict)]
        values = [value for value in values if value]
        if missing_input and len(values) > 1:
            details.append(f"{missing_input} has multiple observed candidates ({', '.join(values[:8])})")
    if not details:
        return ""
    return (
        " Source-audited ambiguity: "
        + "; ".join(details[:4])
        + ". No candidate is resolved. Use a compatible read-only retrieval to identify the matching record "
        "before any write; do not ask the user to restate the identifier."
    )


def _review_stateful_terminal_decision(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
    completed_calls: list[dict[str, Any]],
    max_new_tokens: int,
    *,
    available_missing_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Judge whether ending a stateful turn without a call can make progress."""
    payload = {
        "user_request": user_request,
        "active_user_turn": _stateful_active_user_turn(user_request),
        "candidate_decision": str(binding_plan.get("tool_decision") or ""),
        "candidate_missing_inputs": list(binding_plan.get("missing_inputs") or []),
        "candidate_response_message": _binding_plan_semantic_response_message(binding_plan),
        "already_available_inputs": available_missing_inputs or [],
        "available_tools": [_compact_tool_schema(normalize_tool(tool)) for tool in tools if isinstance(tool, dict)],
        "observations": [
            item.get("observation")
            for item in completed_calls
            if isinstance(item, dict) and item.get("outcome") == "success" and "observation" in item
        ],
        "verified_observation_facts": _stateful_verified_observation_facts(completed_calls),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one minified JSON object and nothing else. You are a state-transition "
                "reviewer, not an action generator. Judge only the supplied no-action decision. Allow "
                "ask_user only when candidate_missing_inputs names information or confirmation that is "
                "actually necessary for the next schema-valid action. Allow no_tool only when the user "
                "request is complete or no available tool can advance it, and candidate_response_message "
                "is a concrete customer-facing final answer. If user confirmation is required before a "
                "later action, require ask_user with confirmation in candidate_missing_inputs rather than "
                "allowing no_tool. Reject an empty clarification, missing response message, or terminal "
                "decision when a grounded immediate tool step can make progress, especially "
                "after an observed prerequisite has completed. If you believe any tool call should occur, "
                "you must return verdict reject for this no-action candidate; never allow it based on a "
                "hypothetical replacement action. A successful observation is authoritative for "
                "its fields: reject ask_user when it requests a field already present in an "
                "observation unless the user explicitly supplied a conflicting replacement. "
                "verified_observation_facts is the compact authoritative record of those fields; "
                "a user declining to restate a fact is not a conflicting replacement. Reject a no-action "
                "response that declares a constrained option unavailable before available retrieval schemas "
                "for the user's stated fallback alternatives have been used. active_user_turn is the newest "
                "user input and takes precedence over earlier dialogue: reject a candidate that asks for "
                "confirmation when that turn already gives clear confirmation for the active operation. "
                "already_available_inputs are independently provenance-audited and must be treated as "
                "satisfied prerequisites, even if the user declined to restate them."
            ),
        },
        {
            "role": "user",
            "content": (
                'Output fields: verdict ("allow" or "reject"), reason. '
                f"Input:{_compact_json(payload)}"
            ),
        },
    ]
    raw_text, parsed, parse_error, format_recovery = _generate_reviewer_json(
        model,
        tokenizer,
        generate_text,
        messages,
        min(max(max_new_tokens, 1000), 1200),
        'verdict ("allow" or "reject"), reason',
        payload_validator=_reviewer_has_allow_reject_verdict,
    )
    if not isinstance(parsed, dict):
        reviewer_alias, _ = extract_json_object(raw_text)
        if _terminal_reviewer_restates_candidate_decision(reviewer_alias, binding_plan):
            return {
                "attempted": True,
                "allowed": True,
                "verdict": "allow",
                "reason": str(reviewer_alias.get("reason") or "reviewer restated the supplied terminal decision"),
                "raw_text": raw_text,
                "format_recovery": format_recovery,
                "verdict_alias_normalized": True,
            }
        return {
            "attempted": True,
            "allowed": False,
            "reason": "semantic terminal reviewer did not return valid JSON",
            "parse_error": parse_error,
            "raw_text": raw_text,
            "format_recovery": format_recovery,
        }
    verdict = str(parsed.get("verdict") or "").strip().lower()
    return {
        "attempted": True,
        "allowed": verdict == "allow",
        "verdict": verdict,
        "reason": str(parsed.get("reason") or ""),
        "raw_text": raw_text,
        "format_recovery": format_recovery,
    }


def _stateful_active_user_turn(user_request: str) -> str:
    """Extract the newest labelled user turn when an adapter supplies a transcript."""
    matches = list(re.finditer(r"(?im)^user:\s*", user_request))
    if not matches:
        return user_request[-2000:]
    start = matches[-1].end()
    following = re.search(r"(?im)^\s*(?:user|assistant(?:_action)?|tool(?::[^\n]*)?):\s*", user_request[start:])
    end = start + following.start() if following else len(user_request)
    return user_request[start:end].strip()[:2000]


def _generate_reviewer_json(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    required_fields: str,
    payload_validator: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str, dict[str, Any] | None, str | None, dict[str, Any]]:
    """Generate a compact reviewer verdict with one format-only recovery pass."""
    raw_text = generate_text(model, tokenizer, messages, max_new_tokens)
    parsed, parse_error = extract_json_object(raw_text)
    recovery: dict[str, Any] = {"attempted": False}
    if isinstance(parsed, dict) and (payload_validator is None or payload_validator(parsed)):
        return raw_text, parsed, None, recovery
    if isinstance(parsed, dict):
        parse_error = "reviewer JSON failed the required output contract"

    recovery_messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one minified JSON object and nothing else. Convert the prior reviewer "
                "response into the required verdict format. Preserve its judgment; do not create, replace, "
                "or modify any candidate tool call."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Required fields: {required_fields}.\n"
                f"Original reviewer input:\n{str(messages[-1].get('content') or '')[:18000]}\n\n"
                f"Prior invalid response:\n{str(raw_text)[-12000:]}"
            ),
        },
    ]
    recovery_text = generate_text(model, tokenizer, recovery_messages, min(max(max_new_tokens, 600), 800))
    recovery_parsed, recovery_error = extract_json_object(recovery_text)
    if isinstance(recovery_parsed, dict) and payload_validator is not None and not payload_validator(recovery_parsed):
        recovery_parsed = None
        recovery_error = "reviewer JSON failed the required output contract"
    recovery = {
        "attempted": True,
        "max_new_tokens": min(max(max_new_tokens, 600), 800),
        "initial_parse_error": parse_error,
        "raw_text": recovery_text,
        "parse_error": recovery_error if recovery_parsed is None else None,
        "used": recovery_parsed is not None,
    }
    return recovery_text, recovery_parsed, recovery_error, recovery


def _reviewer_has_allow_reject_verdict(payload: dict[str, Any]) -> bool:
    verdict = str(payload.get("verdict") or "").strip().lower()
    return verdict in {"allow", "reject"}


def _stateful_progress_repair_plan(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    binding_request: str,
    tools: list[dict[str, Any]],
    completed_calls: list[dict[str, Any]],
    max_new_tokens: int,
    *,
    semantic_only: bool = False,
    reviewer_feedback: str | None = None,
    stateful_goal_ledger: dict[str, Any] | None = None,
    stateful_goal_ledger_required: bool = False,
    no_action_was_rejected: bool = False,
    resolved_missing_inputs: list[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Use one bounded semantic replan when deterministic binding stalls.

    This is intentionally a semantic proposal followed by the same binder and
    verifier, rather than allowing a second model call to execute tools.
    """
    semantic_output = _generate_semantic_frame(
        model,
        tokenizer,
        generate_text,
        user_request,
        tools,
        max_new_tokens,
        execution_history=completed_calls,
        require_stateful_progress=True,
        stateful_feedback=reviewer_feedback,
        stateful=True,
        stateful_goal_ledger=stateful_goal_ledger,
        stateful_goal_ledger_required=stateful_goal_ledger_required,
        stateful_no_action_forbidden=no_action_was_rejected,
    )
    frame = semantic_output.get("parsed")
    info: dict[str, Any] = {
        "attempted": True,
        "used": False,
        "semantic_frame_output": semantic_output,
    }
    if not isinstance(frame, dict):
        return None, info
    def build_repair_plan(proposed_frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        proposed_plan = build_tool_binding_plan(
            binding_request,
            tools,
            capability_plan={"semantic_input_frame": proposed_frame},
            allow_model_binding_prefix=True,
            verified_evidence=_stateful_verified_observation_facts(
                completed_calls,
                max_facts=48,
                per_observation_max=16,
            ),
        )
        if semantic_only:
            report = proposed_plan.get("model_tool_binding") if isinstance(proposed_plan.get("model_tool_binding"), dict) else {}
            if not report.get("accepted"):
                proposed_plan = _stateful_semantic_only_no_call_plan(proposed_plan)
        return _drop_successfully_replayed_calls(proposed_plan, completed_calls)

    plan, replay_filter = build_repair_plan(frame)
    info["replayed_call_filter"] = replay_filter
    repair_missing_input_evidence = _stateful_missing_input_evidence_adjudication(
        model,
        tokenizer,
        generate_text,
        plan,
        completed_calls,
        max_new_tokens,
    ) if no_action_was_rejected else {"attempted": False, "available_missing_inputs": []}
    newly_resolved_inputs = [
        str(item.get("missing_input") or "")
        for item in repair_missing_input_evidence.get("available_missing_inputs") or []
        if isinstance(item, dict) and str(item.get("missing_input") or "").strip()
    ]
    all_resolved_inputs = list(dict.fromkeys([
        *[str(item).strip() for item in resolved_missing_inputs or [] if str(item).strip()],
        *newly_resolved_inputs,
    ]))
    resolved_provenance = [
        *[
            item
            for item in (resolved_missing_inputs or [])
            if isinstance(item, dict)
        ],
        *[
            item
            for item in repair_missing_input_evidence.get("available_missing_inputs") or []
            if isinstance(item, dict)
        ],
    ]
    info["missing_input_evidence"] = repair_missing_input_evidence
    correction_info: dict[str, Any] = {"attempted": False, "used": False}
    if _stateful_repair_needs_binding_correction(
        frame,
        plan,
        semantic_only=semantic_only,
        no_action_was_rejected=no_action_was_rejected,
        resolved_missing_inputs=all_resolved_inputs,
    ):
        correction_feedback = _stateful_binding_correction_feedback(
            plan,
            reviewer_feedback,
            resolved_missing_inputs=all_resolved_inputs,
        )
        correction_feedback += _stateful_resolved_missing_input_feedback(
            repair_missing_input_evidence.get("available_missing_inputs") or []
        )
        correction_output = _generate_semantic_frame(
            model,
            tokenizer,
            generate_text,
            user_request,
            tools,
            max_new_tokens,
            execution_history=completed_calls,
            require_stateful_progress=True,
            stateful_feedback=correction_feedback,
            stateful=True,
            stateful_goal_ledger=stateful_goal_ledger,
            stateful_goal_ledger_required=stateful_goal_ledger_required,
            stateful_no_action_forbidden=no_action_was_rejected,
        )
        correction_frame = correction_output.get("parsed")
        correction_info = {
            "attempted": True,
            "used": False,
            "semantic_frame_output": correction_output,
        }
        if isinstance(correction_frame, dict):
            corrected_plan, corrected_replay_filter = build_repair_plan(correction_frame)
            correction_info["replayed_call_filter"] = corrected_replay_filter
            corrected_missing_input_evidence = _stateful_missing_input_evidence_adjudication(
                model,
                tokenizer,
                generate_text,
                corrected_plan,
                completed_calls,
                max_new_tokens,
            ) if no_action_was_rejected else {"attempted": False, "available_missing_inputs": []}
            correction_info["missing_input_evidence"] = corrected_missing_input_evidence
            corrected_resolved = [
                str(item.get("missing_input") or "")
                for item in corrected_missing_input_evidence.get("available_missing_inputs") or []
                if isinstance(item, dict) and str(item.get("missing_input") or "").strip()
            ]
            all_resolved_inputs = list(dict.fromkeys([*all_resolved_inputs, *corrected_resolved]))
            resolved_provenance.extend(
                item
                for item in corrected_missing_input_evidence.get("available_missing_inputs") or []
                if isinstance(item, dict)
            )
            if corrected_plan.get("calls") or _stateful_repair_plan_is_actionable(
                corrected_plan,
                semantic_only,
                no_action_was_rejected=no_action_was_rejected,
            ):
                plan = corrected_plan
                replay_filter = corrected_replay_filter
                info["replayed_call_filter"] = replay_filter
                correction_info["used"] = True
    info["binding_correction"] = correction_info
    plan, single_call_execution = _limit_stateful_plan_to_one_call(plan)
    info["single_call_execution"] = single_call_execution
    decision = str(plan.get("tool_decision") or "").strip().lower()
    valid_user_clarification = decision == "ask_user" and (
        not semantic_only or bool(_binding_plan_semantic_missing_inputs(plan))
    ) and not _stateful_plan_requests_resolved_missing_input(plan, all_resolved_inputs)
    valid_semantic_response = (
        not no_action_was_rejected
        and decision == "no_tool"
        and _binding_plan_semantic_response_message(plan)
    )
    if plan.get("calls") or valid_user_clarification or valid_semantic_response:
        if resolved_provenance and decision == "ask_user":
            plan["stateful_provenance_repair"] = {
                "used": True,
                "available_missing_inputs": resolved_provenance,
            }
        plan["stateful_progress_repair"] = {
            "used": True,
            "reason": "semantic replan requested after stateful candidates could not make verified progress",
        }
        info["used"] = True
        return plan, info
    return None, info


def _stateful_repair_needs_binding_correction(
    frame: dict[str, Any],
    plan: dict[str, Any],
    *,
    semantic_only: bool,
    no_action_was_rejected: bool,
    resolved_missing_inputs: list[str] | None = None,
) -> bool:
    """Detect a semantic repair that named work but could not become an action."""
    if not semantic_only:
        return False
    decision = str(frame.get("tool_decision") or "").strip().lower()
    if no_action_was_rejected and decision not in {"call", "ask_user"}:
        return True
    if decision == "ask_user":
        if _stateful_plan_requests_resolved_missing_input(plan, resolved_missing_inputs):
            return True
        missing_inputs = frame.get("missing_inputs") if isinstance(frame.get("missing_inputs"), list) else []
        return not any(str(item).strip() for item in missing_inputs)
    if decision != "call":
        return False
    report = plan.get("model_tool_binding") if isinstance(plan.get("model_tool_binding"), dict) else {}
    return bool(frame.get("tool_bindings")) and not report.get("accepted")


def _stateful_binding_correction_feedback(
    plan: dict[str, Any],
    reviewer_feedback: str | None,
    *,
    resolved_missing_inputs: list[str] | None = None,
) -> str:
    """Give one semantic retry verifier feedback without granting execution authority."""
    report = plan.get("model_tool_binding") if isinstance(plan.get("model_tool_binding"), dict) else {}
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), list) else []
    details = []
    for diagnostic in diagnostics[:4]:
        if not isinstance(diagnostic, dict):
            continue
        code = str(diagnostic.get("code") or "verification_error").strip()
        message = str(diagnostic.get("message") or "semantic binding did not verify").strip()
        details.append(f"{code}: {message}")
    prior_feedback = str(reviewer_feedback or "candidate did not make verified progress").strip()
    feedback = (
        f"{prior_feedback[:600]} The prior semantic repair could not be compiled after schema and "
        "evidence verification. Return one corrected next transition with every required argument and "
        "evidence span, or ask_user only with a concrete missing_inputs item."
    )
    if details:
        feedback += " Verifier diagnostics: " + "; ".join(details)[:900]
    resolved = [str(item).strip() for item in resolved_missing_inputs or [] if str(item).strip()]
    if resolved:
        feedback += (
            " These inputs are already available from source-audited successful observations and must not "
            "be requested again: " + ", ".join(resolved[:8]) + "."
        )
    return feedback


def _stateful_repair_plan_is_actionable(
    plan: dict[str, Any],
    semantic_only: bool,
    *,
    no_action_was_rejected: bool = False,
) -> bool:
    decision = str(plan.get("tool_decision") or "").strip().lower()
    if plan.get("calls"):
        return True
    if decision == "ask_user":
        return not semantic_only or bool(_binding_plan_semantic_missing_inputs(plan))
    return (
        not no_action_was_rejected
        and decision == "no_tool"
        and bool(_binding_plan_semantic_response_message(plan))
    )


def _stateful_plan_defers_to_user_or_terminal(
    binding_plan: dict[str, Any],
    tools: list[dict[str, Any]],
) -> bool:
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    if decision in {"ask_user", "no_tool"}:
        return True
    terminal_names = _terminal_response_tool_names(tools)
    calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    return bool(calls) and all(str(call.get("tool_name") or "") in terminal_names for call in calls)


def _stateful_nonterminal_recovery_plan(
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
    runtime: GoalGraphRuntime,
    completed_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    terminal_names = _terminal_response_tool_names(tools)
    if not terminal_names and not binding_plan.get("dropped_incompatible_calls"):
        return None
    if _binding_plan_semantic_requests_user_without_missing(binding_plan):
        return None
    calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    if decision == "ask_user" and not _binding_plan_semantic_missing_inputs(binding_plan):
        return None
    first_tool = str((calls[0] if calls else {}).get("tool_name") or "")
    all_terminal = bool(calls) and all(str(call.get("tool_name") or "") in terminal_names for call in calls)
    terminal_first = bool(first_tool and first_tool in terminal_names)
    needs_recovery = decision in {"ask_user", "no_tool"} or terminal_first or all_terminal
    if not needs_recovery:
        return None

    nonterminal_tools = [
        tool
        for tool in tools
        if str(normalize_tool(tool).get("name") or "") not in terminal_names
    ]
    if not nonterminal_tools:
        return None
    candidate = build_tool_binding_plan(user_request, nonterminal_tools)
    candidate, raw_safety_filter = _drop_stateful_unverified_fallback_calls(
        candidate,
        nonterminal_tools,
        user_request,
    )
    candidate, replay_filter = _drop_successfully_replayed_calls(candidate, completed_calls)
    candidate_calls = [call for call in candidate.get("calls") or [] if isinstance(call, dict)]
    if candidate.get("tool_decision") != "call" or not candidate_calls:
        candidate = _stateful_readonly_retry_after_incompatible_plan(
            user_request,
            tools,
            binding_plan,
            runtime,
            completed_calls,
        )
        candidate_calls = [call for call in (candidate or {}).get("calls") or [] if isinstance(call, dict)]
        if not isinstance(candidate, dict) or candidate.get("tool_decision") != "call" or not candidate_calls:
            return None
    candidate = copy.deepcopy(candidate)
    candidate.setdefault(
        "stateful_terminal_recovery",
        {
            "used": True,
            "reason": "terminal response deferred because non-terminal tools produced a grounded call",
            "excluded_terminal_tools": sorted(terminal_names),
            "previous_decision": decision,
            "previous_first_tool": first_tool,
        },
    )
    if replay_filter["used"]:
        candidate["stateful_replayed_call_filter"] = replay_filter
    if raw_safety_filter["used"]:
        candidate["stateful_raw_safety_filter"] = raw_safety_filter
    return candidate


def _stateful_readonly_retry_after_incompatible_plan(
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
    runtime: GoalGraphRuntime,
    completed_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not binding_plan.get("dropped_incompatible_calls"):
        return None
    read_only_names = {
        capability.tool_name
        for capability in runtime.registry.values()
        if capability.risk == "read_only" and capability.kind in {"resolve", "retrieve", "search", "rank", "decide"}
    }
    terminal_names = _terminal_response_tool_names(tools)
    retry_tools = [
        tool
        for tool in tools
        if str(normalize_tool(tool).get("name") or "") in read_only_names
        and str(normalize_tool(tool).get("name") or "") not in terminal_names
        and _stateful_readonly_tool_can_gather_state(normalize_tool(tool))
    ]
    if not retry_tools:
        return None
    candidate = build_tool_binding_plan(user_request, retry_tools)
    candidate, raw_safety_filter = _drop_stateful_unverified_fallback_calls(
        candidate,
        retry_tools,
        user_request,
    )
    candidate, replay_filter = _drop_successfully_replayed_calls(candidate, completed_calls)
    if candidate.get("tool_decision") != "call" or not candidate.get("calls"):
        candidate = _slot_complete_readonly_candidate(user_request, retry_tools)
        if candidate is None:
            return None
        candidate, slot_safety_filter = _drop_stateful_unverified_fallback_calls(
            candidate,
            retry_tools,
            user_request,
        )
        if slot_safety_filter["used"]:
            raw_safety_filter = slot_safety_filter
        candidate, retry_filter = _drop_successfully_replayed_calls(candidate, completed_calls)
        if retry_filter["used"]:
            replay_filter = retry_filter
        if candidate.get("tool_decision") != "call" or not candidate.get("calls"):
            return None
    candidate = copy.deepcopy(candidate)
    candidate["stateful_readonly_retry"] = {
        "used": True,
        "reason": "retry with read-only tools after incompatible call arguments were dropped",
        "dropped_incompatible_calls": binding_plan.get("dropped_incompatible_calls") or [],
    }
    if replay_filter["used"]:
        candidate["stateful_replayed_call_filter"] = replay_filter
    if raw_safety_filter["used"]:
        candidate["stateful_raw_safety_filter"] = raw_safety_filter
    return candidate


def _stateful_readonly_tool_can_gather_state(tool: dict[str, Any]) -> bool:
    text = f"{tool.get('name') or ''} {tool.get('description') or ''}".lower()
    if re.search(r"\b(?:calculate|compute|math|think|reason|respond|final)\b", text):
        return False
    return bool(re.search(r"\b(?:get|search|find|lookup|retrieve|query|list|details?|read|fetch)\b", text))


def _slot_complete_readonly_candidate(user_request: str, tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    task_frame = build_task_frame(user_request)
    query_audit = build_query_input_audit(user_request, {})
    for raw_tool in tools:
        tool = normalize_tool(raw_tool)
        audit = audit_candidate_tool(user_request, tool, 1.0, task_frame, query_audit)
        planned_calls = audit.get("planned_calls") or []
        complete = [
            call
            for call in planned_calls
            if isinstance(call, dict) and call.get("arguments") and not call.get("missing_arguments")
        ]
        if not complete:
            continue
        calls = [
            {
                "id": f"call_{index + 1}",
                "tool_name": str(tool.get("name") or ""),
                "arguments": _preserve_grounded_identifier_prefixes(
                    call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                    tool,
                    user_request,
                ),
                "depends_on": [],
                "missing_arguments": [],
            }
            for index, call in enumerate(complete[:1])
        ]
        calls, dropped = _filter_schema_value_incompatible_calls([tool], calls)
        if not calls:
            continue
        return {
            "tool_decision": "call",
            "reason": "stateful read-only slot-complete recovery after incompatible write binding",
            "calls": calls,
            "missing_inputs": [],
            "candidate_tool_audits": [audit],
            **({"dropped_incompatible_calls": dropped} if dropped else {}),
        }
    return None


def _preserve_grounded_identifier_prefixes(
    arguments: dict[str, Any],
    tool: dict[str, Any],
    user_request: str,
) -> dict[str, Any]:
    if not arguments:
        return arguments
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    repaired = dict(arguments)
    for name, value in arguments.items():
        if not isinstance(value, str) or value.startswith("#"):
            continue
        spec = props.get(str(name)) if isinstance(props.get(str(name)), dict) else {}
        slot_text = f"{name} {spec.get('description') or ''}".lower()
        if not re.search(r"(?:^|_|\b)(?:id|identifier|number|code)(?:_|\b|$)", slot_text):
            continue
        prefixed = "#" + value
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(prefixed)}(?![A-Za-z0-9_])", user_request):
            repaired[str(name)] = prefixed
    return repaired


def _stateful_unique_schema_grounded_readonly_plan(
    user_request: str,
    runtime: GoalGraphRuntime,
    tools: list[dict[str, Any]],
    completed_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Advance one unambiguous read-only transition after a semantic no-action decision.

    Stateful semantic-only mode normally refuses deterministic routing. This
    bounded recovery is narrower: a candidate must be read-only, require at
    least one schema input, and have every argument grounded by the existing
    audit. Ambiguous candidate sets remain model-owned rather than guessed.
    """
    tools_by_name = {
        str(normalize_tool(tool).get("name") or ""): tool
        for tool in tools
        if isinstance(tool, dict) and str(normalize_tool(tool).get("name") or "")
    }
    candidates: list[dict[str, Any]] = []
    for tool_name, capability in runtime.registry.items():
        if (
            capability.risk != "read_only"
            or capability.kind not in {"resolve", "retrieve", "search", "rank", "decide"}
            or not capability.required_inputs
        ):
            continue
        raw_tool = tools_by_name.get(tool_name)
        if raw_tool is None:
            continue
        candidate = _slot_complete_readonly_candidate(user_request, [raw_tool])
        calls = [call for call in candidate.get("calls") or [] if isinstance(call, dict)] if candidate else []
        if candidate is None or candidate.get("tool_decision") != "call" or len(calls) != 1:
            continue
        candidate, replay_filter = _drop_successfully_replayed_calls(candidate, completed_calls)
        if replay_filter["used"] or not candidate.get("calls"):
            continue
        candidates.append(candidate)

    if len(candidates) != 1:
        return None
    candidate = copy.deepcopy(candidates[0])
    candidate["stateful_schema_recovery"] = {
        "used": True,
        "reason": "semantic no-action decision advanced by one uniquely schema-grounded read-only transition",
    }
    return candidate


def _stateful_readonly_progress_plan(
    user_request: str,
    binding_plan: dict[str, Any],
    runtime: GoalGraphRuntime,
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    if len(calls) < 2 or _stateful_request_has_tool_observation(user_request):
        return None
    terminal_names = _terminal_response_tool_names(tools)
    tool_to_capability = {capability.tool_name: capability for capability in runtime.registry.values()}

    def bucket(call: dict[str, Any]) -> int:
        name = str(call.get("tool_name") or "")
        capability = tool_to_capability.get(name)
        if capability is None:
            return 3
        if name in terminal_names or capability.kind in {"ask_user", "respond"}:
            return 2
        if capability.risk == "read_only" and capability.kind in {"resolve", "retrieve", "search", "rank", "decide"}:
            return 0
        return 1

    current_buckets = [bucket(call) for call in calls]
    if not current_buckets or current_buckets[0] == min(current_buckets):
        return None
    reordered = sorted(enumerate(calls), key=lambda item: (bucket(item[1]), item[0]))
    reordered_calls = [copy.deepcopy(call) for _, call in reordered]
    if reordered_calls == calls:
        return None
    plan = copy.deepcopy(binding_plan)
    for index, call in enumerate(reordered_calls):
        call["id"] = f"call_{index + 1}"
    plan["calls"] = reordered_calls
    plan["stateful_readonly_progress"] = {
        "used": True,
        "reason": "stateful execution has no tool observation yet, so read-only progress calls run before side effects or terminal responses",
        "previous_first_tool": str(calls[0].get("tool_name") or ""),
        "new_first_tool": str(reordered_calls[0].get("tool_name") or ""),
    }
    return plan


def _stateful_request_has_tool_observation(user_request: str) -> bool:
    return re.search(r"(?m)^tool:[^:]+:", user_request) is not None


def _terminal_response_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for raw_tool in tools or []:
        if not isinstance(raw_tool, dict):
            continue
        tool = normalize_tool(raw_tool)
        if _is_terminal_response_tool(tool):
            names.add(str(tool.get("name") or ""))
    return {name for name in names if name}


def _is_terminal_response_tool(tool: dict[str, Any]) -> bool:
    name = str(tool.get("name") or "").strip().lower()
    description = str(tool.get("description") or "").strip().lower()
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = params.get("required") if isinstance(params.get("required"), list) else []
    required = [str(item) for item in required]
    if len(required) != 1 or len(props) > 2:
        return False
    required_name = required[0].lower()
    if required_name not in {"content", "message", "response", "text", "answer"}:
        return False
    prop_text = " ".join(
        str(item.get("description") or "")
        for item in props.values()
        if isinstance(item, dict)
    ).lower()
    combined = f"{name} {description} {prop_text}"
    if name in {"respond", "final_answer", "answer", "ask_user", "clarify"}:
        return True
    return bool(
        re.search(r"\b(?:final|respond|response|clarif|customer-facing|user-facing|message to (?:the )?user)\b", combined)
        and not re.search(r"\b(?:email|sms|ticket|notification|post|comment|chat room)\b", combined)
    )


def _should_generate_call_skeleton(
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
) -> bool:
    if not tools:
        return False
    request = user_request.lower()
    calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    if len(tools) == 1:
        if not calls:
            if _single_tool_non_english_action_needs_skeleton(user_request, tools[0]):
                return True
            if _binding_plan_blocks_call_skeleton_recovery(binding_plan):
                return False
            return _allow_no_call_skeleton_recovery(user_request)
        if _binding_plan_blocks_call_skeleton_recovery(binding_plan):
            return False
        return _single_tool_repeated_request_needs_skeleton(user_request, tools[0], calls)
    if _binding_plan_blocks_call_skeleton_recovery(binding_plan):
        return False
    if not calls:
        return _allow_no_call_skeleton_recovery(user_request)
    if _has_repeated_mixed_tool_calls(calls):
        return True
    if not re.search(
        r"\b(?:also|then|after that|next|finally|lastly|another|same|both|each|respectively|as well|first|second|third)\b",
        request,
    ):
        return False
    names = [str(call.get("tool_name") or "") for call in calls]
    if len(calls) > 12:
        return True
    if len(set(names)) == 1 and len(tools) > 1:
        return True
    clauses = [clause for clause in re.split(r"[.;?]|\b(?:also|then|after that|next|finally|lastly)\b", user_request, flags=re.I) if clause.strip()]
    if len(clauses) >= 3 and len(calls) < len(clauses):
        return True
    action_markers = re.findall(
        r"\b(?:also|then|after that|next|finally|lastly)\b|\band\s+(?=(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|search|recommend|translate|analy[sz]e|display|transfer|locate|fetch|perform)\b)",
        user_request,
        flags=re.I,
    )
    if len(action_markers) >= 2 and len(calls) <= len(action_markers) + 1:
        return True
    if len(calls) > 4 and re.search(r"\b(?:same|another|after that|then|respectively|as well)\b", request):
        return True
    if re.search(r"\b(?:both|each|respectively|same .+ as well|another .+ and)\b", request) and len(calls) < 4:
        return True
    return False


def _single_tool_repeated_request_needs_skeleton(
    user_request: str,
    tool: dict[str, Any],
    calls: list[dict[str, Any]],
) -> bool:
    if len(calls) > 12:
        return True
    if _skeleton_tool_has_required_batch_array(normalize_tool(tool)):
        return _required_batch_array_plan_needs_skeleton(user_request, calls)
    request = user_request.lower()
    if _result_count_like_request(request):
        return False
    if _single_tool_command_sequence_needs_skeleton(user_request, tool, calls):
        return True
    if len(calls) <= 1 and _single_tool_repeat_signal(user_request):
        return True
    if _single_tool_result_chain_needs_skeleton(user_request, calls):
        return True
    if len(calls) < 4 and re.search(r"\b(?:respectively|each|both|same|first|second|third|fourth)\b", request):
        return True
    if len(calls) < 6 and _has_cross_product_like_single_tool_request(user_request, normalize_tool(tool)):
        return True
    if _skeleton_tool_has_any_batch_array(normalize_tool(tool)) and _calls_have_identical_tool_arguments(calls):
        return True
    return False


def _single_tool_command_sequence_needs_skeleton(
    user_request: str,
    tool: dict[str, Any],
    calls: list[dict[str, Any]],
) -> bool:
    if len(calls) != 1:
        return False
    normalized = normalize_tool(tool)
    if not _tool_is_generic_command_executor(normalized):
        return False
    return _independent_command_action_count(user_request) > len(calls)


def _single_tool_non_english_action_needs_skeleton(user_request: str, tool: dict[str, Any]) -> bool:
    if not re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", user_request):
        return False
    text = f"{tool.get('name') or ''} {tool.get('description') or ''}".lower()
    if not re.search(r"\b(?:control|execute|run|command|appliance|device|action)\b", text):
        return False
    return len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", user_request)) >= 2


def _independent_command_action_count(user_request: str) -> int:
    lowered = user_request.lower()
    if not re.search(r"\b(?:and|then|also|after that|finally|,)\b", lowered):
        return 1
    action_pattern = (
        r"\b(?:list|show|dir|make|create|write|delete|remove|copy|move|rename|"
        r"open|run|execute|launch|start|touch|mkdir|echo)\b"
    )
    actions = re.findall(action_pattern, lowered)
    if re.search(r"\b(?:file|folder|directory|drive|path|cmd|command|shell|terminal)\b", lowered):
        return len(actions) if actions else 1
    return 1


def _required_batch_array_plan_needs_skeleton(user_request: str, calls: list[dict[str, Any]]) -> bool:
    request = user_request.lower()
    if _result_count_like_request(request) or not _single_tool_repeat_signal(user_request):
        return False
    if re.search(
        r"\b(?:first|second|third|fourth|for each|each dataset|each data set|one for each|"
        r"respectively|another|both|two|three|four)\b",
        request,
    ) and len(calls) <= 1:
        return True
    if len(calls) > 1 and _calls_have_identical_tool_arguments(calls):
        return True
    return False


def _calls_have_identical_tool_arguments(calls: list[dict[str, Any]]) -> bool:
    concrete = [call for call in calls if isinstance(call, dict) and call.get("tool_name")]
    if len(concrete) <= 1:
        return False
    names = {str(call.get("tool_name") or "") for call in concrete}
    if len(names) != 1:
        return False
    keys = {
        json.dumps(call.get("arguments") or {}, sort_keys=True, default=str)
        for call in concrete
    }
    return len(keys) == 1


def _single_tool_repeat_signal(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:also|additionally|then|after that|next|finally|lastly|another|same|both|each|"
            r"respectively|as well|what if|first|second|third|fourth|fifth|two|three|four|several|multiple)\b",
            user_request,
            re.I,
        )
        or len(re.findall(r"\band\b", user_request, flags=re.I)) >= 2
    )


def _single_tool_result_chain_needs_skeleton(user_request: str, calls: list[dict[str, Any]]) -> bool:
    request = user_request.lower()
    if not re.search(
        r"\b(?:use|using|with|from|of)\s+(?:that|this|the|previous|prior)\s+results?\b"
        r"|\bresults?\s+(?:to|with|from)\b",
        request,
    ):
        return False
    parenthesized_numeric_groups = len(
        re.findall(r"\([^)]*\d+(?:\.\d+)?[^)]*,[^)]*\d+(?:\.\d+)?[^)]*\)", user_request)
    )
    named_pairs = len(re.findall(r"\bpair(?:s| of numbers)?\b", request))
    grounded_groups = max(parenthesized_numeric_groups, named_pairs)
    if grounded_groups:
        return len(calls) < min(max(grounded_groups, 2), 8)
    return len(calls) < 3


def _has_cross_product_like_single_tool_request(user_request: str, tool: dict[str, Any]) -> bool:
    scalar_slots = [
        name
        for name, spec in (tool.get("parameters", {}).get("properties") or {}).items()
        if isinstance(spec, dict) and str(spec.get("type") or "string").lower() not in {"array", "list", "object", "dict"}
    ]
    if len(scalar_slots) < 2:
        return False
    list_markers = re.findall(
        r"\b[A-Z][A-Za-z0-9_-]*\s+and\s+[A-Z][A-Za-z0-9_-]*\b|\b[A-Za-z][A-Za-z0-9_-]*\s+and\s+[A-Za-z][A-Za-z0-9_-]*\b",
        user_request,
    )
    return len(list_markers) >= 2 and bool(re.search(r"\b(?:in|for|by|with|at)\b", user_request, re.I))


def _skeleton_tool_has_required_batch_array(tool: dict[str, Any]) -> bool:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = set(params.get("required") or [])
    return any(
        name in required and isinstance(spec, dict) and str(spec.get("type") or "").lower() in {"array", "list"}
        for name, spec in props.items()
    )


def _skeleton_tool_has_any_batch_array(tool: dict[str, Any]) -> bool:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    return any(
        isinstance(spec, dict) and str(spec.get("type") or "").lower() in {"array", "list"}
        for spec in props.values()
    )


def _result_count_like_request(request: str) -> bool:
    return bool(re.search(r"\b(?:top|first|nearest|closest|best|latest|last)\s+\d+\b|\b\d+\s+(?:results|items|records)\b", request))


def _has_repeated_mixed_tool_calls(calls: list[dict[str, Any]]) -> bool:
    names = [str(call.get("tool_name") or "") for call in calls if isinstance(call, dict)]
    counts = Counter(names)
    return len(names) >= 4 and len(counts) > 1 and any(count > 1 for count in counts.values())


def _allow_no_call_skeleton_recovery(user_request: str) -> bool:
    lowered = user_request.lower()
    if re.search(r"\b(?:do not|don't|without|no need to|should not)\s+(?:call|use|invoke|run)\b", lowered):
        return False
    if re.search(r"\b(?:ignore|irrelevant|unrelated|joke|chat|hello|hi)\b", lowered) and len(user_request.split()) < 12:
        return False
    return bool(
        re.search(
            r"\b(?:find|search|look\s+up|get|retrieve|calculate|compute|predict|forecast|translate|book|order|recommend|suggest|generate|create|convert|compare|analy[sz]e|identify|locate|fetch|check|list|show|tell me|what|who|where|when|how many|how much)\b",
            lowered,
        )
    )


def _binding_plan_blocks_call_skeleton_recovery(binding_plan: dict[str, Any]) -> bool:
    calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    if calls:
        return False
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    if decision == "ask_user" and binding_plan.get("missing_inputs"):
        return True
    if decision != "no_tool":
        return False
    model_report = binding_plan.get("model_tool_binding")
    if isinstance(model_report, dict):
        diagnostics = model_report.get("diagnostics") if isinstance(model_report.get("diagnostics"), list) else []
        codes = {str(item.get("code") or "") for item in diagnostics if isinstance(item, dict)}
        if codes & {"unsupported_tool_route", "ungrounded_argument", "missing_required_argument"}:
            return True
    audits = [item for item in binding_plan.get("candidate_tool_audits") or [] if isinstance(item, dict)]
    if audits and not any(audit.get("eligible") for audit in audits):
        reasons = {str(audit.get("ineligible_reason") or "") for audit in audits}
        warnings = {
            str(warning)
            for audit in audits
            for warning in (audit.get("slot_binding_warnings") or [])
        }
        if "duplicate_required_slot_values" in warnings and _audits_have_repairable_duplicate_slots(audits):
            return False
        if reasons <= {"semantic_mismatch_or_low_score", "insufficient_required_slot_evidence"}:
            return True
        if "duplicate_required_slot_values" in warnings:
            return True
    return False


def _audits_have_repairable_duplicate_slots(audits: list[dict[str, Any]]) -> bool:
    for audit in audits:
        if str(audit.get("semantic_fit") or "") not in {"exact", "partial"}:
            continue
        if audit.get("missing_slots"):
            continue
        if "duplicate_required_slot_values" not in set(audit.get("slot_binding_warnings") or []):
            continue
        availability = audit.get("slot_availability")
        if isinstance(availability, dict) and availability:
            if not all(str(value).startswith("available") for value in availability.values()):
                continue
        planned_calls = audit.get("planned_calls") if isinstance(audit.get("planned_calls"), list) else []
        if planned_calls:
            first = planned_calls[0]
            if isinstance(first, dict) and first.get("missing_arguments"):
                continue
        return True
    return False


def _prefer_raw_binding_plan(
    user_request: str,
    binding_plan: dict[str, Any],
    raw_binding_plan: dict[str, Any] | None,
) -> bool:
    if not isinstance(raw_binding_plan, dict):
        return False
    current_calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    raw_calls = [call for call in raw_binding_plan.get("calls") or [] if isinstance(call, dict)]
    if not raw_calls:
        return False
    if any(call.get("missing_arguments") for call in raw_calls):
        return False
    current_names = [str(call.get("tool_name") or "") for call in current_calls]
    raw_names = [str(call.get("tool_name") or "") for call in raw_calls]
    if not current_names:
        return True
    if raw_names == current_names:
        return False
    if len(raw_calls) > 16:
        return False
    current_model = binding_plan.get("model_tool_binding") or {}
    if not current_model.get("used"):
        return False
    if current_model.get("accepted") and set(current_names).issubset(set(raw_names)) and len(raw_names) > len(current_names):
        if _binding_plan_semantic_missing_inputs(binding_plan):
            return True
        return False
    if set(current_names).issubset(set(raw_names)) and len(raw_names) > len(current_names):
        return True
    if len(set(raw_names)) > len(set(current_names)) and _request_allows_multiple_independent_calls(user_request):
        return True
    return False


def _prefer_raw_plan_over_semantic_terminal_veto(
    user_request: str,
    binding_plan: dict[str, Any],
    raw_binding_plan: dict[str, Any] | None,
) -> bool:
    if not isinstance(raw_binding_plan, dict):
        return False
    raw_calls = [
        call
        for call in raw_binding_plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name") and not call.get("missing_arguments")
    ]
    if not raw_calls or len(raw_calls) > 16:
        return False
    if raw_binding_plan.get("tool_decision") != "call":
        return False
    current_decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    if current_decision == "ask_user":
        missing_inputs = [str(item) for item in binding_plan.get("missing_inputs") or []]
        if _missing_identifier_inputs(missing_inputs) and not _raw_calls_bind_missing_inputs(raw_calls, missing_inputs):
            return False
        return True
    if current_decision != "no_tool":
        return False
    return _raw_recovery_allowed_for_semantic_no_tool(user_request)


def _missing_identifier_inputs(missing_inputs: list[str]) -> bool:
    return any(re.search(r"(?:^|_|\b)(?:id|identifier|number|code)(?:_|\b|$)", item.lower()) for item in missing_inputs)


def _raw_calls_bind_missing_inputs(raw_calls: list[dict[str, Any]], missing_inputs: list[str]) -> bool:
    missing = {re.sub(r"[^a-z0-9]+", "", item.lower()) for item in missing_inputs}
    for call in raw_calls:
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        arg_names = {re.sub(r"[^a-z0-9]+", "", str(name).lower()) for name in args}
        if missing & arg_names:
            return True
    return False


def _raw_recovery_allowed_for_semantic_no_tool(user_request: str) -> bool:
    return bool(
        re.search(r"(?im)^(?:earlier user|user|assistant|ai|latest prior api result):", user_request)
        or re.search(r"\b(?:prior|previous|earlier|latest)\s+(?:api\s+)?result\b", user_request, re.I)
    )


def _binding_plan_semantic_missing_inputs(binding_plan: dict[str, Any]) -> bool:
    capability = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else {}
    frame = capability.get("semantic_input_frame") if isinstance(capability, dict) else {}
    missing = frame.get("missing_inputs") if isinstance(frame, dict) else None
    return isinstance(missing, list) and any(str(item).strip() for item in missing)


def _binding_plan_semantic_requests_user_without_missing(binding_plan: dict[str, Any]) -> bool:
    capability = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else {}
    frame = capability.get("semantic_input_frame") if isinstance(capability, dict) else {}
    return _semantic_frame_requests_user(frame) and not _binding_plan_semantic_missing_inputs(binding_plan)


def _binding_plan_semantic_response_message(binding_plan: dict[str, Any]) -> str:
    capability = binding_plan.get("capability_plan") if isinstance(binding_plan, dict) else {}
    frame = capability.get("semantic_input_frame") if isinstance(capability, dict) else {}
    if not isinstance(frame, dict):
        return ""
    for key in ("response_message", "clarification_message"):
        value = frame.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:800]
    return ""


def _stateful_semantic_no_call_is_actionable(binding_plan: dict[str, Any]) -> bool:
    decision = str(binding_plan.get("tool_decision") or "").strip().lower()
    if decision not in {"ask_user", "no_tool"}:
        return False
    return _binding_plan_semantic_missing_inputs(binding_plan) or bool(
        _binding_plan_semantic_response_message(binding_plan)
    )


def _semantic_frame_requests_user(frame: Any) -> bool:
    if not isinstance(frame, dict) or _semantic_tool_bindings_present(frame):
        return False
    for key in ("tool_decision", "decision", "route_decision"):
        value = frame.get(key)
        if not isinstance(value, str):
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if normalized in {"ask_user", "respond", "clarify", "clarification", "need_more_info", "missing_inputs"}:
            return True
    return bool(frame.get("clarification_needed"))


def _semantic_tool_bindings_present(frame: dict[str, Any]) -> bool:
    return any(isinstance(item, dict) for item in frame.get("tool_bindings") or [])


def _request_allows_multiple_independent_calls(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:also|additionally|then|after that|next|finally|lastly|both|each|respectively|as well)\b|\band\s+(?=(?:find|calculate|compute|get|retrieve|compare|update|change|send|book|create|generate|convert|make|provide|search|recommend|translate|analy[sz]e|display|transfer|locate|fetch|fit)\b)",
            user_request,
            re.I,
        )
    )


def _generate_call_skeleton(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = build_call_skeleton_messages(user_request, tools, binding_plan)
    skeleton_tokens = min(max(max_new_tokens, 4200), 5200)
    raw_text = generate_text(model, tokenizer, messages, skeleton_tokens)
    parsed, parse_error = extract_call_skeleton_json_object(raw_text)
    return {
        "attempted": True,
        "used": False,
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error if parsed is None else None,
    }


def build_call_skeleton_messages(
    user_request: str,
    tools: list[dict[str, Any]],
    binding_plan: dict[str, Any],
) -> list[dict[str, str]]:
    deterministic_calls = [
        str(call.get("tool_name") or "")
        for call in binding_plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]
    payload = {
        "user_request": user_request,
        "available_tools": [_compact_tool_schema(normalize_tool(tool)) for tool in tools],
        "deterministic_tool_names": deterministic_calls,
    }
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one minified valid JSON object and nothing else. "
                'The first character of your response must be "{". '
                "You are an ordered tool-call skeleton planner. Choose only from "
                "available tool names. Do not invent tools. "
                "Do not restate the request, tools, schema, analysis, reasoning, "
                "markdown, or prose. Do not include assistantfinal, code fences, "
                "or any prefix before the JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create the ordered executable tool-call skeleton for the request. "
                "Repeat a tool once per independent entity/time/operation only when "
                "the tool schema is single-entity; do not split property/detail lists "
                "that one tool call can answer. For cross-product requests, include one "
                "call for each grounded combination. For identical repeated objects with "
                "identical arguments, include one reusable call. Use evidence_span copied "
                "from the user request for each call, preferably the full span containing "
                "the values unique to that call. You may include arguments and "
                "evidence_spans only when every value is grounded in the request. A value "
                "may be inferred only when its evidence_span directly licenses it, such as "
                "a phrase denoting zero, count, or default state; do not use external "
                "constants or infer identifiers. Unsafe or ungrounded arguments will be "
                "rejected. If unsure, keep the "
                "deterministic call count. "
                "Output fields: ordered_calls list; each call has tool_name, "
                "evidence_span, optional arguments, and optional evidence_spans; "
                "top-level confidence number. Start with {\"ordered_calls\":.\n"
                f"Input:{_compact_json(payload)}"
            ),
        },
    ]


def extract_call_skeleton_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = _json_object_candidates(text)
    fallback, fallback_error = extract_json_object(text)
    if fallback is not None:
        candidates.append(fallback)
    repaired = _repair_truncated_call_skeleton_json(text)
    if repaired is not None:
        candidates.append(repaired)
    candidates = [item for item in candidates if not _call_skeleton_candidate_is_placeholder(item)]
    if not candidates:
        return None, fallback_error or "no valid JSON object found"

    def score(obj: dict[str, Any]) -> tuple[int, int]:
        points = 0
        if isinstance(obj.get("ordered_calls"), list):
            points += 80
        if isinstance(obj.get("calls"), list) or isinstance(obj.get("tool_calls"), list):
            points += 50
        if "confidence" in obj:
            points += 5
        return points, len(json.dumps(obj, default=str))

    best = max(candidates, key=score)
    best_score, _ = score(best)
    if best_score <= 0:
        return None, fallback_error or "no call skeleton JSON object found"
    normalized = dict(best)
    if not isinstance(normalized.get("ordered_calls"), list):
        for key in ("calls", "tool_calls"):
            if isinstance(normalized.get(key), list):
                normalized["ordered_calls"] = normalized[key]
                break
    normalized.setdefault("ordered_calls", [])
    return normalized, None


def _repair_truncated_call_skeleton_json(text: str) -> dict[str, Any] | None:
    marker = '{"ordered_calls"'
    start = text.rfind(marker)
    if start < 0:
        marker = "{'ordered_calls'"
        start = text.rfind(marker)
    if start < 0:
        return None
    snippet = text[start:].strip()
    if snippet.startswith("{'"):
        return None
    repaired = _append_balanced_json_suffix(snippet)
    if repaired is None:
        return None
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_skeleton_candidate_is_placeholder(obj: dict[str, Any]) -> bool:
    raw_items = obj.get("ordered_calls")
    if not isinstance(raw_items, list) and isinstance(obj.get("calls"), list):
        raw_items = obj.get("calls")
    if not isinstance(raw_items, list) or not raw_items:
        return False
    placeholder_values = {
        "available tool name",
        "exact request span",
        "exact request span supporting this call",
        "grounded value",
    }
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or item.get("name") or item.get("tool") or "").strip().lower()
        evidence = str(item.get("evidence_span") or item.get("evidence") or "").strip().lower()
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        evidence_spans = item.get("evidence_spans") if isinstance(item.get("evidence_spans"), dict) else {}
        if tool_name in placeholder_values or evidence in placeholder_values:
            return True
        if "schema_arg" in arguments or "schema_arg" in evidence_spans:
            return True
    return False


def _append_balanced_json_suffix(snippet: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(snippet):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue
        if char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            continue
    if in_string or not stack:
        return None
    trimmed = snippet.rstrip()
    if not trimmed or trimmed[-1] in ",:":
        return None
    return trimmed + "".join(reversed(stack))


def _binding_plan_from_call_skeleton(
    user_request: str,
    tools: list[dict[str, Any]],
    parsed: Any,
    base_plan: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    normalized_tools = [normalize_tool(tool) for tool in tools or [] if isinstance(tool, dict)]
    if _binding_plan_blocks_call_skeleton_recovery(base_plan) and not (
        len(normalized_tools) == 1
        and not [call for call in base_plan.get("calls") or [] if isinstance(call, dict)]
        and _single_tool_non_english_action_needs_skeleton(user_request, normalized_tools[0])
    ):
        return None
    tools_by_name = {str(tool.get("name") or ""): tool for tool in normalized_tools}
    skeleton_items = _verified_call_skeleton_items(user_request, parsed, tools_by_name)
    if not skeleton_items:
        return None
    calls: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    full_audit_cache: dict[str, dict[str, Any]] = {}
    occurrence: Counter[str] = Counter()
    for item in skeleton_items:
        tool_name = str(item.get("tool_name") or "")
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return None
        call_index = occurrence[tool_name]
        occurrence[tool_name] += 1
        call_plan, audit = _call_plan_for_skeleton_item(
            user_request,
            tool,
            item,
            call_index,
            full_audit_cache,
        )
        if call_plan is None or call_plan.get("missing_arguments"):
            return None
        call = {
            "id": f"call_{len(calls) + 1}",
            "tool_name": tool_name,
            "arguments": call_plan.get("arguments") or {},
            "depends_on": [],
            "missing_arguments": [],
        }
        if isinstance(call_plan.get("argument_evidence"), dict):
            call["argument_evidence"] = call_plan["argument_evidence"]
        calls.append(call)
        audits.append(audit)
    calls = _dedupe_identical_verified_calls(calls)
    calls = _drop_unrequested_helper_calls(user_request, calls)
    calls = _order_independent_calls_for_benchmark(user_request, normalized_tools, calls)
    if len(calls) > 16:
        return None
    plan = copy.deepcopy(base_plan)
    plan.update(
        {
            "tool_decision": "call",
            "reason": "accepted GPT-OSS ordered call skeleton after tool-name and argument verification",
            "calls": calls,
            "missing_inputs": [],
            "candidate_tool_audits": audits,
            "call_skeleton_binding": {
                "used": True,
                "tool_names": [call["tool_name"] for call in calls],
                "confidence": parsed.get("confidence"),
            },
        }
    )
    return plan


def _verified_call_skeleton_items(
    user_request: str,
    parsed: dict[str, Any],
    tools_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_items = parsed.get("ordered_calls")
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for raw in raw_items[:16]:
        if not isinstance(raw, dict):
            continue
        tool_name = str(raw.get("tool_name") or raw.get("name") or raw.get("tool") or "").strip()
        if tool_name not in tools_by_name:
            return []
        evidence = str(raw.get("evidence_span") or raw.get("evidence") or "").strip()
        if evidence and not _evidence_in_request(evidence, user_request):
            evidence = ""
        item = {"tool_name": tool_name, "evidence_span": evidence}
        for key in ("arguments", "args", "evidence_spans", "evidence", "sources"):
            if isinstance(raw.get(key), dict):
                item[key] = raw[key]
        items.append(item)
    return items


def _call_plan_for_skeleton_item(
    user_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
    call_index: int,
    full_audit_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    tool_name = str(tool.get("name") or "")
    if tool_name not in full_audit_cache:
        full_audit_cache[tool_name] = _audit_for_request(user_request, tool)
    full_audit = full_audit_cache[tool_name]

    hinted_call = _verified_skeleton_argument_call(user_request, tool, item)
    if hinted_call is not None:
        if _skeleton_route_audit_accepts_tool(user_request, tool, item, hinted_call, full_audit):
            return hinted_call, _skeleton_argument_audit(tool, hinted_call)
        return None, full_audit

    evidence = str(item.get("evidence_span") or "").strip()
    local_request = evidence if _skeleton_evidence_can_scope_local_request(evidence) else user_request
    local_audit = _audit_for_request(local_request, tool)
    local_calls = local_audit.get("planned_calls") or []
    if (
        local_calls
        and not local_calls[0].get("missing_arguments")
        and (
            local_audit.get("eligible")
            or _local_skeleton_evidence_supports_call(evidence, tool, local_calls[0])
        )
    ):
        return local_calls[0], local_audit

    full_calls = full_audit.get("planned_calls") or []
    if not full_audit.get("eligible") or not full_calls:
        return None, full_audit
    return full_calls[min(call_index, len(full_calls) - 1)], full_audit


def _skeleton_evidence_can_scope_local_request(evidence: str) -> bool:
    if not evidence:
        return False
    if len(evidence.split()) >= 2:
        return True
    return len(re.findall(r"-?\d+(?:\.\d+)?", evidence)) >= 2


def _local_skeleton_evidence_supports_call(
    evidence: str,
    tool: dict[str, Any],
    call_plan: dict[str, Any],
) -> bool:
    if not _skeleton_evidence_can_scope_local_request(evidence):
        return False
    arguments = call_plan.get("arguments") if isinstance(call_plan.get("arguments"), dict) else {}
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    required = [str(slot) for slot in params.get("required") or []]
    if not required or any(slot not in arguments for slot in required):
        return False
    evidence_numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", evidence)]
    if not evidence_numbers:
        return False
    for slot in required:
        value = arguments.get(slot)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not any(float(value) == item for item in evidence_numbers):
            return False
    return True


def _skeleton_route_audit_accepts_tool(
    user_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
    call_plan: dict[str, Any],
    audit: dict[str, Any],
) -> bool:
    if (
        "duplicate_required_slot_values" in set(audit.get("slot_binding_warnings") or [])
        and _call_plan_has_duplicate_required_values(tool, call_plan)
    ):
        return False
    if audit.get("semantic_fit") != "rejected":
        return True
    if str(audit.get("ineligible_reason") or "") != "semantic_mismatch_or_low_score":
        return False
    return _verified_skeleton_route_has_local_support(user_request, tool, item, call_plan)


def _verified_skeleton_route_has_local_support(
    user_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
    call_plan: dict[str, Any],
) -> bool:
    evidence = str(item.get("evidence_span") or "").strip()
    if not evidence or not _evidence_in_request(evidence, user_request):
        return False
    arguments = call_plan.get("arguments") if isinstance(call_plan.get("arguments"), dict) else {}
    required = [
        str(slot)
        for slot in (((tool.get("parameters") or {}).get("required")) or [])
    ]
    if any(slot not in arguments or arguments.get(slot) in (None, "") for slot in required):
        return False
    if _tool_is_generic_command_executor(tool):
        command_slots = _command_executor_argument_names(tool)
        argument_evidence = call_plan.get("argument_evidence") if isinstance(call_plan.get("argument_evidence"), dict) else {}
        return any(slot in arguments and str(argument_evidence.get(slot) or "").strip() for slot in command_slots)
    if _has_hard_semantic_conflict(evidence, tool):
        return False
    return True


def _call_plan_has_duplicate_required_values(tool: dict[str, Any], call_plan: dict[str, Any]) -> bool:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    required = [str(item) for item in params.get("required") or []]
    arguments = call_plan.get("arguments") if isinstance(call_plan.get("arguments"), dict) else {}
    values: dict[str, list[str]] = {}
    for name in required:
        value = arguments.get(name)
        if isinstance(value, bool) or value in (None, ""):
            continue
        if isinstance(value, (str, int, float)):
            key = re.sub(r"\s+", " ", str(value).strip().lower())
            if key:
                values.setdefault(key, []).append(name)
    return any(len(set(names)) > 1 for names in values.values())


def _verified_skeleton_argument_call(
    user_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(item.get("arguments"), dict) and not isinstance(item.get("args"), dict):
        return None
    item = _skeleton_item_with_grounded_shared_evidence(user_request, tool, item)
    arguments, argument_evidence, diagnostics = _verified_model_argument_group(
        user_request,
        tool,
        item,
        "call_skeleton.ordered_calls[].arguments",
    )
    if diagnostics:
        return None
    return {
        "arguments": arguments,
        "argument_evidence": argument_evidence,
        "missing_arguments": [],
        "unbound_available_arguments": {},
        "raw_missing_arguments": [],
    }


def _skeleton_item_with_grounded_shared_evidence(
    user_request: str,
    tool: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(item.get("evidence_spans"), dict) or isinstance(item.get("evidence"), dict):
        return item
    raw_arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else None
    if raw_arguments is None and isinstance(item.get("args"), dict):
        raw_arguments = item.get("args")
    shared_evidence = str(item.get("evidence_span") or "").strip()
    if not raw_arguments or not shared_evidence or not _evidence_in_request(shared_evidence, user_request):
        return item
    evidence_spans: dict[str, str] = {}
    command_slots = _command_executor_argument_names(tool) if _tool_is_generic_command_executor(tool) else set()
    for name, value in raw_arguments.items():
        if str(name) in command_slots:
            evidence = shared_evidence
        else:
            evidence = _skeleton_shared_evidence_for_value(user_request, shared_evidence, value)
        if evidence:
            evidence_spans[str(name)] = evidence
    if not evidence_spans:
        return item
    copied = dict(item)
    copied["evidence_spans"] = evidence_spans
    return copied


def _skeleton_shared_evidence_for_value(user_request: str, evidence_span: str, value: Any) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if isinstance(value, list):
        nested = [
            _skeleton_shared_evidence_for_value(user_request, evidence_span, item)
            for item in value
        ]
        return evidence_span if nested and all(nested) else ""
    if isinstance(value, bool):
        return evidence_span if _skeleton_boolean_supported_by_evidence(evidence_span, value) else ""
    value_text = str(value).strip()
    if not value_text:
        return ""
    if _evidence_in_request(value_text, user_request):
        return value_text
    if _skeleton_iso_date_supported_by_evidence(value_text, evidence_span):
        return evidence_span
    if _skeleton_value_tokens_supported_by_evidence(value_text, evidence_span):
        return evidence_span
    return ""


def _skeleton_boolean_supported_by_evidence(evidence_span: str, value: bool) -> bool:
    lowered = evidence_span.lower()
    if value:
        return bool(re.search(r"\b(?:include|including|with|yes|true|also)\b", lowered))
    return bool(re.search(r"\b(?:without|exclude|excluding|no|false)\b", lowered))


def _skeleton_value_tokens_supported_by_evidence(value_text: str, evidence_span: str) -> bool:
    value_tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", value_text.lower())
        if len(token) > 1
        and token
        not in {
            "ca",
            "ny",
            "ma",
            "usa",
            "us",
            "uk",
            "united",
            "states",
            "china",
            "france",
            "ireland",
        }
    ]
    if not value_tokens:
        return False
    evidence_tokens = set(re.findall(r"[A-Za-z0-9]+", evidence_span.lower()))
    return all(token in evidence_tokens for token in value_tokens)


def _skeleton_iso_date_supported_by_evidence(value_text: str, evidence_span: str) -> bool:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value_text)
    if not match:
        return False
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    month_names = {
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
    for natural in re.finditer(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b",
        evidence_span,
        re.I,
    ):
        natural_month = month_names.get(natural.group(1).lower())
        if natural_month == month and int(natural.group(2)) == day and int(natural.group(3)) == year:
            return True
    compact = f"{month}/{day}/{year}"
    padded = f"{month:02d}/{day:02d}/{year}"
    dashed = f"{month:02d}-{day:02d}-{year}"
    return any(item in evidence_span for item in {compact, padded, dashed})


def _skeleton_argument_audit(tool: dict[str, Any], call_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": str(tool.get("name") or ""),
        "eligible": True,
        "semantic_fit": "exact",
        "missing_slots": [],
        "planned_calls": [call_plan],
        "source": "gptoss_call_skeleton_verified_arguments",
    }


def _dedupe_identical_verified_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        item = dict(call)
        item["id"] = f"call_{len(deduped) + 1}"
        deduped.append(item)
    return deduped


def _drop_unrequested_helper_calls(user_request: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(calls) <= 1:
        return calls
    kept = [
        call
        for call in calls
        if not _skeleton_extra_call_is_unrequested_helper(user_request, call)
    ]
    if not kept:
        return calls
    return [{**call, "id": f"call_{index + 1}"} for index, call in enumerate(kept)]


def _audit_for_request(user_request: str, tool: dict[str, Any]) -> dict[str, Any]:
    task_frame = build_task_frame(user_request)
    query_input_audit = build_query_input_audit(user_request)
    return audit_candidate_tool(user_request, tool, 1.0, task_frame, query_input_audit)


def _prefer_skeleton_binding_plan(
    user_request: str,
    binding_plan: dict[str, Any],
    skeleton_plan: dict[str, Any],
) -> bool:
    current = [
        str(call.get("tool_name") or "")
        for call in binding_plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]
    proposed = [
        str(call.get("tool_name") or "")
        for call in skeleton_plan.get("calls") or []
        if isinstance(call, dict) and call.get("tool_name")
    ]
    if not proposed:
        return False
    if not current:
        return True
    current_model = binding_plan.get("model_tool_binding") if isinstance(binding_plan.get("model_tool_binding"), dict) else {}
    if proposed == current:
        return not current_model.get("accepted") and _skeleton_plan_changes_arguments(binding_plan, skeleton_plan)
    if current_model.get("accepted") and Counter(proposed) != Counter(current):
        if _accepted_model_binding_can_expand_with_skeleton(user_request, current, proposed):
            return True
        return False
    if _skeleton_adds_only_unrequested_helper_calls(user_request, current, skeleton_plan):
        return False
    if len(current) > 12 and len(proposed) <= 12:
        return True
    if (
        len(set(current)) == 1
        and len(set(proposed)) == 1
        and set(current) == set(proposed)
        and len(proposed) < len(current)
    ):
        return False
    if len(set(proposed)) > len(set(current)):
        return True
    if Counter(proposed) == Counter(current) and proposed != current and _has_repeated_mixed_tool_calls(
        [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    ):
        return _has_explicit_order_language(user_request)
    if re.search(
        r"\b(?:also|then|after that|next|finally|lastly|another|same|both|each|respectively|as well|"
        r"first|second|third|fourth|fifth|two|three|four|several|multiple)\b",
        user_request,
        re.I,
    ):
        return 1 <= len(proposed) <= 16
    return False


def _accepted_model_binding_can_expand_with_skeleton(
    user_request: str,
    current: list[str],
    proposed: list[str],
) -> bool:
    if len(set(current + proposed)) != 1 or len(proposed) <= len(current):
        return False
    tool_name = current[0].lower() if current else ""
    if not re.search(r"\b(?:cmd|command|shell|terminal|execute|controller)\b", tool_name.replace("_", " ")):
        return False
    return _independent_command_action_count(user_request) >= len(proposed)


def _skeleton_plan_changes_arguments(binding_plan: dict[str, Any], skeleton_plan: dict[str, Any]) -> bool:
    current_calls = [call for call in binding_plan.get("calls") or [] if isinstance(call, dict)]
    proposed_calls = [call for call in skeleton_plan.get("calls") or [] if isinstance(call, dict)]
    if len(current_calls) != len(proposed_calls):
        return False
    for current, proposed in zip(current_calls, proposed_calls):
        if str(current.get("tool_name") or "") != str(proposed.get("tool_name") or ""):
            return False
        current_args = current.get("arguments") if isinstance(current.get("arguments"), dict) else {}
        proposed_args = proposed.get("arguments") if isinstance(proposed.get("arguments"), dict) else {}
        if json.dumps(current_args, sort_keys=True, default=str) != json.dumps(
            proposed_args,
            sort_keys=True,
            default=str,
        ):
            return True
    return False


def _skeleton_adds_only_unrequested_helper_calls(
    user_request: str,
    current_names: list[str],
    skeleton_plan: dict[str, Any],
) -> bool:
    current_counts = Counter(current_names)
    extra_calls = []
    for call in skeleton_plan.get("calls") or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or "")
        if current_counts.get(name, 0) > 0:
            current_counts[name] -= 1
            continue
        extra_calls.append(call)
    if not extra_calls:
        return False
    return all(_skeleton_extra_call_is_unrequested_helper(user_request, call) for call in extra_calls)


def _skeleton_extra_call_is_unrequested_helper(user_request: str, call: dict[str, Any]) -> bool:
    name = str(call.get("tool_name") or "").lower().replace(".", "_").replace("-", "_")
    if not re.search(r"(?:^|_)class_info(?:_|$)|(?:^|_)(?:info|details|metadata|describe|inspect)(?:_|$)", name):
        return False
    if re.search(
        r"\b(?:class\s+info|class\s+details|details?\s+(?:about|of|for)|information\s+(?:about|on|for)|"
        r"describe|inspect)\b",
        user_request,
        re.I,
    ):
        return False
    return True


def _has_explicit_order_language(user_request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:first|second|third|fourth|finally|lastly|after that|then|once|before|after|respectively)\b"
            r"|\b[A-Za-z_][A-Za-z0-9_.]*\s+function\b"
            r"|'[A-Za-z_][A-Za-z0-9_.]*'",
            user_request,
            re.I,
        )
    )


def _generate_semantic_frame(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    user_request: str,
    tools: list[dict[str, Any]],
    max_new_tokens: int,
    *,
    execution_history: list[dict[str, Any]] | None = None,
    require_stateful_progress: bool = False,
    stateful_feedback: str | None = None,
    stateful: bool = False,
    stateful_goal_ledger: dict[str, Any] | None = None,
    stateful_goal_ledger_required: bool = False,
    stateful_no_action_forbidden: bool = False,
) -> dict[str, Any]:
    messages = build_tool_binding_frame_messages(
        user_request,
        tools,
        execution_history=execution_history,
        require_stateful_progress=require_stateful_progress,
        stateful_feedback=stateful_feedback,
        stateful_goal_ledger=stateful_goal_ledger,
        stateful=stateful,
        stateful_goal_ledger_required=stateful_goal_ledger_required,
        stateful_no_action_forbidden=stateful_no_action_forbidden,
    )
    semantic_tokens = (
        min(max(max_new_tokens, 1400), 1800)
        if stateful
        else min(max(max_new_tokens, 4200), 5200)
    )
    raw_text = generate_text(model, tokenizer, messages, semantic_tokens)
    parsed, parse_error = extract_semantic_frame_json_object(raw_text)
    parsed, inherited_goal_ledger = _inherit_stateful_goal_ledger_if_needed(
        parsed,
        stateful_goal_ledger,
        required=stateful_goal_ledger_required,
    )
    if stateful_goal_ledger_required and isinstance(parsed, dict) and not _normalize_stateful_goal_ledger(parsed.get("goal_ledger")):
        parsed = None
        parse_error = "stateful semantic frame omitted the required goal_ledger"
    adaptive_retry: dict[str, Any] = {"attempted": False}
    if stateful and parsed is None:
        # Stateful turns usually need one compact action. When the model spends
        # its short budget on reasoning and leaves JSON unfinished, retry the
        # same semantic contract with enough room rather than treating a
        # potentially correct transition as an empty plan.
        retry_tokens = min(max(semantic_tokens * 2, 2400), 6000)
        retry_text = generate_text(model, tokenizer, messages, retry_tokens)
        retry_parsed, retry_parse_error = extract_semantic_frame_json_object(retry_text)
        retry_parsed, retry_inherited_goal_ledger = _inherit_stateful_goal_ledger_if_needed(
            retry_parsed,
            stateful_goal_ledger,
            required=stateful_goal_ledger_required,
        )
        inherited_goal_ledger = inherited_goal_ledger or retry_inherited_goal_ledger
        if (
            stateful_goal_ledger_required
            and isinstance(retry_parsed, dict)
            and not _normalize_stateful_goal_ledger(retry_parsed.get("goal_ledger"))
        ):
            retry_parsed = None
            retry_parse_error = "stateful semantic frame omitted the required goal_ledger"
        adaptive_retry = {
            "attempted": True,
            "max_new_tokens": retry_tokens,
            "initial_parse_error": parse_error,
            "raw_text": retry_text,
            "parse_error": retry_parse_error if retry_parsed is None else None,
            "used": retry_parsed is not None,
        }
        raw_text = retry_text
        parsed = retry_parsed
        parse_error = retry_parse_error
    format_recovery: dict[str, Any] = {"attempted": False}
    if stateful and parsed is None:
        # A model can sometimes state a sensible decision in prose but exhaust
        # its budget before serializing it.  Give it one short, format-only
        # chance to preserve that decision.  This recovery cannot choose a new
        # tool or invent state because it only converts the prior draft.
        recovery_messages = [
            {
                "role": "system",
                "content": (
                    "Return exactly one minified valid JSON object and nothing else. "
                    "Convert the prior semantic-binder draft into the required JSON frame. "
                    "Do not add a new action, tool, argument, entity, or value that is not "
                    "explicitly stated in the draft. The first character must be '{'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Required fields: tool_decision, canonical_request, slots_observed, "
                    "call_groups, tool_bindings, missing_inputs, optional clarification_message, "
                    + ("required goal_ledger, " if stateful_goal_ledger_required else "")
                    + "\nPrior draft:\n"
                    + str(raw_text)[-12000:]
                ),
            },
        ]
        recovery_text = generate_text(model, tokenizer, recovery_messages, 1000)
        recovery_parsed, recovery_parse_error = extract_semantic_frame_json_object(recovery_text)
        recovery_parsed, recovery_inherited_goal_ledger = _inherit_stateful_goal_ledger_if_needed(
            recovery_parsed,
            stateful_goal_ledger,
            required=stateful_goal_ledger_required,
        )
        inherited_goal_ledger = inherited_goal_ledger or recovery_inherited_goal_ledger
        if (
            stateful_goal_ledger_required
            and isinstance(recovery_parsed, dict)
            and not _normalize_stateful_goal_ledger(recovery_parsed.get("goal_ledger"))
        ):
            recovery_parsed = None
            recovery_parse_error = "stateful semantic frame omitted the required goal_ledger"
        format_recovery = {
            "attempted": True,
            "max_new_tokens": 1000,
            "raw_text": recovery_text,
            "parse_error": recovery_parse_error if recovery_parsed is None else None,
            "used": recovery_parsed is not None,
        }
        if recovery_parsed is not None:
            raw_text = recovery_text
            parsed = recovery_parsed
            parse_error = None
    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "parse_error": parse_error if parsed is None else None,
        "adaptive_retry": adaptive_retry,
        "format_recovery": format_recovery,
        "goal_ledger_recovery": {
            "used": inherited_goal_ledger,
            "reason": "preserved prior runtime continuity state when a valid semantic frame omitted it"
            if inherited_goal_ledger
            else None,
        },
    }


def _inherit_stateful_goal_ledger_if_needed(
    frame: dict[str, Any] | None,
    prior_goal_ledger: dict[str, Any] | None,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Keep runtime continuity state when a valid state transition omits an echo.

    The ledger is supplied by the runtime as context.  Once it exists, a model
    does not need to reproduce it verbatim to propose one verified transition.
    Reusing the prior ledger avoids discarding an otherwise valid, schema-bound
    action solely because the model omitted redundant state.  Initial turns
    still require the model to establish a ledger, and every action continues
    through the regular binder, compiler, and reviewer.
    """
    if not required or not isinstance(frame, dict):
        return frame, False
    if _normalize_stateful_goal_ledger(frame.get("goal_ledger")):
        return frame, False
    prior = _normalize_stateful_goal_ledger(prior_goal_ledger)
    if not prior:
        return frame, False
    inherited = copy.deepcopy(frame)
    inherited["goal_ledger"] = prior
    return inherited, True


def _semantic_frame_has_routing_evidence(frame: Any) -> bool:
    if not isinstance(frame, dict):
        return False
    slots = [item for item in frame.get("slots_observed") or [] if isinstance(item, dict)]
    groups = [item for item in frame.get("call_groups") or [] if isinstance(item, dict)]
    bindings = [item for item in frame.get("tool_bindings") or [] if isinstance(item, dict)]
    return bool(slots or groups or bindings)


def _adjudicate_semantic_enum_grounding(
    model: Any,
    tokenizer: Any,
    generate_text: GenerateTextFn,
    semantic_frame: dict[str, Any],
    binding_text: str,
    tools: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify non-literal enum normalizations without allowing a new action.

    The semantic binder can correctly infer a schema enum from conversational
    meaning even when no lexical form of that enum occurs in the transcript.
    This bounded second pass supplies provenance only: it may approve an exact
    proposed enum value and quote its supporting source, but cannot add a tool,
    argument, or value. The ordinary schema/evidence binder and candidate
    reviewer still validate the resulting call.
    """
    candidates = _semantic_enum_grounding_candidates(semantic_frame, binding_text, tools)
    info: dict[str, Any] = {
        "attempted": bool(candidates),
        "used": False,
        "candidate_count": len(candidates),
        "approved": [],
    }
    if not candidates:
        return semantic_frame, info

    payload = {
        "binding_evidence": binding_text,
        "available_tools": [_compact_tool_schema(normalize_tool(tool)) for tool in tools if isinstance(tool, dict)],
        "enum_candidates": candidates,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one minified JSON object and nothing else. You are a provenance "
                "reviewer, not an action planner. Evaluate only the supplied enum_candidates. "
                "You may approve a candidate only when its exact enum value is semantically entailed "
                "by the binding_evidence and its supplied tool schema. Each approval must exactly echo "
                "tool_name, group_index, argument_name, and value from one candidate, and quote an exact "
                "supporting evidence_span from binding_evidence. Do not create, replace, or infer any ID, "
                "credential, free-form entity, tool, argument, or non-enum value. When uncertain, omit the "
                "candidate."
            ),
        },
        {
            "role": "user",
            "content": (
                'Output only {"approved":[{"tool_name":string,"group_index":integer,'
                '"argument_name":string,"value":string,"evidence_span":string}]}. '
                f"Input:{_compact_json(payload)}"
            ),
        },
    ]
    raw_text, parsed, parse_error, format_recovery = _generate_reviewer_json(
        model,
        tokenizer,
        generate_text,
        messages,
        min(max(max_new_tokens, 700), 1000),
        "approved array with tool_name, group_index, argument_name, value, evidence_span",
    )
    info.update(
        {
            "raw_text": raw_text,
            "parse_error": parse_error,
            "format_recovery": format_recovery,
        }
    )
    if not isinstance(parsed, dict) or not isinstance(parsed.get("approved"), list):
        return semantic_frame, info

    candidate_by_key = {
        (
            item["tool_name"],
            item["binding_index"],
            item["group_index"],
            item["argument_name"],
            item["value"],
        ): item
        for item in candidates
    }
    approved: list[dict[str, Any]] = []
    for item in parsed["approved"]:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or "")
        group_index = item.get("group_index")
        argument_name = str(item.get("argument_name") or "")
        value = item.get("value")
        evidence_span = str(item.get("evidence_span") or "").strip()
        if not isinstance(group_index, int) or not isinstance(value, str) or not evidence_span:
            continue
        matches = [
            candidate
            for key, candidate in candidate_by_key.items()
            if key[0] == tool_name and key[2:] == (group_index, argument_name, value)
        ]
        if len(matches) != 1 or not _evidence_in_request(evidence_span, binding_text):
            continue
        approved.append({**matches[0], "evidence_span": evidence_span})
    if not approved:
        return semantic_frame, info

    updated = copy.deepcopy(semantic_frame)
    bindings = updated.get("tool_bindings")
    if not isinstance(bindings, list):
        return semantic_frame, info
    for item in approved:
        binding = bindings[item["binding_index"]]
        if not isinstance(binding, dict):
            continue
        groups = binding.get("argument_groups")
        if isinstance(groups, list) and item["group_index"] < len(groups) and isinstance(groups[item["group_index"]], dict):
            group = groups[item["group_index"]]
            evidence_spans = group.get("evidence_spans") if isinstance(group.get("evidence_spans"), dict) else {}
            group["evidence_spans"] = {**evidence_spans, item["argument_name"]: item["evidence_span"]}
        elif item["group_index"] == 0:
            evidence_spans = binding.get("evidence_spans") if isinstance(binding.get("evidence_spans"), dict) else {}
            binding["evidence_spans"] = {**evidence_spans, item["argument_name"]: item["evidence_span"]}
    info["used"] = True
    info["approved"] = approved
    return updated, info


def _semantic_enum_grounding_candidates(
    semantic_frame: dict[str, Any],
    binding_text: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_schemas = {
        str(normalize_tool(tool).get("name") or ""): normalize_tool(tool)
        for tool in tools
        if isinstance(tool, dict) and str(normalize_tool(tool).get("name") or "")
    }
    candidates: list[dict[str, Any]] = []
    bindings = semantic_frame.get("tool_bindings") if isinstance(semantic_frame.get("tool_bindings"), list) else []
    for binding_index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            continue
        tool_name = str(binding.get("tool_name") or binding.get("name") or binding.get("tool") or "")
        tool = tool_schemas.get(tool_name)
        if tool is None:
            continue
        params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        groups = binding.get("argument_groups")
        if not isinstance(groups, list):
            arguments = binding.get("arguments") if isinstance(binding.get("arguments"), dict) else binding.get("args")
            groups = [{"arguments": arguments}] if isinstance(arguments, dict) else []
        top_evidence = binding.get("evidence_spans") if isinstance(binding.get("evidence_spans"), dict) else {}
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            arguments = group.get("arguments") if isinstance(group.get("arguments"), dict) else group.get("args")
            if not isinstance(arguments, dict):
                continue
            group_evidence = dict(top_evidence)
            if isinstance(group.get("evidence_spans"), dict):
                group_evidence.update(group["evidence_spans"])
            for raw_name, value in arguments.items():
                argument_name = str(raw_name)
                spec = properties.get(argument_name)
                if not isinstance(spec, dict) or not isinstance(value, str):
                    continue
                enum_values = [str(item) for item in spec.get("enum") or [] if isinstance(item, (str, int, float))]
                if not enum_values or value not in enum_values:
                    continue
                if _semantic_enum_slot_is_sensitive(argument_name, spec):
                    continue
                evidence = str(group_evidence.get(argument_name) or "")
                if _evidence_in_request(evidence, binding_text) or _evidence_in_request(value, binding_text):
                    continue
                readable_value = value.replace("_", " ").replace("-", " ")
                if _evidence_in_request(readable_value, binding_text):
                    continue
                candidates.append(
                    {
                        "tool_name": tool_name,
                        "binding_index": binding_index,
                        "group_index": group_index,
                        "argument_name": argument_name,
                        "value": value,
                        "schema": {
                            "enum": enum_values,
                            "description": str(spec.get("description") or "")[:180],
                        },
                    }
                )
    return candidates[:6]


def _semantic_enum_slot_is_sensitive(argument_name: str, spec: dict[str, Any]) -> bool:
    slot_text = f"{argument_name} {spec.get('description') or ''}".lower().replace("_", " ")
    return bool(
        re.search(
            r"\b(?:account|api key|apikey|auth|credential|email|file|id|identifier|key|login|password|phone|secret|token|uri|url|username)\b",
            slot_text,
        )
    )


def build_tool_binding_frame_messages(
    user_request: str,
    tools: list[dict[str, Any]],
    *,
    execution_history: list[dict[str, Any]] | None = None,
    require_stateful_progress: bool = False,
    stateful_feedback: str | None = None,
    stateful_goal_ledger: dict[str, Any] | None = None,
    stateful: bool = False,
    stateful_goal_ledger_required: bool = False,
    stateful_no_action_forbidden: bool = False,
) -> list[dict[str, str]]:
    """Ask the model for evidence-backed semantic slots plus optional tool bindings."""
    normalized_tools = [normalize_tool(tool) for tool in tools or [] if isinstance(tool, dict)]
    payload = {
        "user_request": user_request,
        "available_tools": [_compact_tool_schema(tool) for tool in normalized_tools],
    }
    completed_calls = _successful_execution_history(execution_history)
    if completed_calls:
        payload["execution_history"] = _stateful_execution_summary(completed_calls)
        payload["verified_observation_facts"] = _stateful_verified_observation_facts(
            completed_calls,
            max_facts=48,
            per_observation_max=16,
        )
    goal_ledger = _normalize_stateful_goal_ledger(stateful_goal_ledger)
    if stateful:
        payload["stateful_goal_ledger"] = goal_ledger
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one minified valid JSON object and nothing else. "
                'The first character of your response must be "{". Do not restate '
                "the request, tools, schema, analysis, markdown, prose, comments, or "
                "explanations. Do not include analysis, assistantfinal, code fences, "
                "or any prefix before the JSON object. You are a schema-aware semantic binder, "
                "not an executor. You may propose tool bindings, but every proposed "
                "argument must include evidence copied from the user request or a recorded "
                "tool observation unless it is an allowed schema default. Do not invent IDs, "
                "tokens, entities, or values. "
                "If a value is missing, leave it out and list it in missing_inputs. Keep "
                "the JSON compact; for complex multi-call requests, prioritize concise "
                "tool_bindings over verbose slot inventories. "
                "When a clarification is needed, you may include clarification_message: a "
                "concise customer-facing question that names only schema-supported alternatives. "
                "For ask_user, include clarification_message and list the precise missing input. "
                "For no_tool, include response_message: a concise customer-facing final answer "
                "grounded in the request, policy, or recorded observations. Use ask_user, not "
                "no_tool, when explicit confirmation is required before a later action. "
                "execution_history records completed transitions. "
                "verified_observation_facts is the trusted, compact, source-attributed record derived from "
                "successful observations. Treat those facts as available values. A user declining "
                "to restate a fact is not a conflicting replacement. "
                "When a successful observation contains a field needed by a later action, bind "
                "that field instead of asking the user to repeat it, unless the user explicitly "
                "provides a conflicting replacement. "
                "Preserve semantic roles across observed records: never use a value or count from "
                "one role to fill a differently named role. When the user refers to their own "
                "details, profile, account, or record, resolve that reference to direct identity "
                "fields on the matching observed record, not to values nested in related collections. "
                "When several opaque record identifiers are available, do not select the first one "
                "arbitrarily. Use compatible retrieval schemas to resolve which candidate matches the "
                "user's stated constraints, and never mutate a candidate before an observation verifies "
                "that match. "
                "Do not repeat the identical tool name and arguments unless the user has supplied "
                "new information."
            ),
        },
        {
            "role": "user",
            "content": "".join(
                [
                    "Build a semantic frame and, when the schema supports it, propose verified "
                    "tool_bindings. Use only available tool names and schema argument names. "
                    "Separate result counts from tool-call counts.\n"
                    "Rules:\n"
                    "- tool_decision is call, ask_user, or no_tool. Use no_tool when the "
                    "available schemas are related but do not perform the requested operation.\n"
                    "- canonical_request is a clear paraphrase of the requested work.\n"
                    "- slots_observed are optional semantic facts with evidence spans from the "
                    "request. Keep them minimal when tool_bindings already contain evidence.\n"
                    "- call_groups describe independent requested operations and expected_call_count.\n"
                    "- tool_bindings are optional. Include them only when you can ground every "
                    "required argument. For repeated work, emit one argument_group per call.\n"
                    "- evidence_spans maps each argument name to the exact request or observed "
                    "tool-output span supporting that argument. It may cover a normalized value, "
                    "e.g. USD from 'dollars'.\n"
                    "- Do not use a batch/list call unless the tool schema has an array/list "
                    "argument for that entity.\n"
                    "- When multiple schemas can establish the same prerequisite and none is "
                    "currently grounded, do not arbitrarily commit to one path. Ask for a "
                    "concise, schema-supported alternative (for example, either credential A "
                    "or the fields required by credential B). If the transcript says a user "
                    "cannot provide a previously requested value, do not request that value "
                    "again; use a grounded alternative or ask for a different supported one.\n",
                    (
                        "- In a stateful environment, plan the next executable transition, not the final "
                        "operation. Requirements of a later write action are not missing inputs when a "
                        "grounded read-only, search, discovery, or lookup tool can advance the active goal "
                        "now. Before asking the user for a property, inspect the schemas for a tool that can "
                        "retrieve or resolve that property from an already grounded identifier or other "
                        "available input. Ask the user only when no available schema-valid transition can "
                        "advance the active goal with current evidence. Treat user alternatives and fallbacks "
                        "as conditional: preserve their stated priority and continue the current option until "
                        "an observation rules it out or the user explicitly replaces it. Do not skip to a "
                        "later option merely because it is also allowed. Preserve every hard user constraint "
                        "when evaluating observed options. Do not conclude that no compatible option exists "
                        "until the available retrieval tools for the user's stated fallback alternatives have "
                        "been exhausted.\n"
                        if stateful
                        else ""
                    ),
                    (
                        "- stateful_goal_ledger is semantic continuity context, not new evidence. goal_ledger "
                        "is REQUIRED in this response. Preserve "
                        "every incomplete goal until an observation verifies it completed. A later user "
                        "request adds work unless it explicitly replaces an earlier goal. Choose the next "
                        "unresolved goal or a prerequisite of it; do not ask the user to choose among "
                        "already requested goals. Preserve existing goal IDs and their order. Mark a goal "
                        "completed only when a recorded tool observation verifies its outcome. Include "
                        "goal_ledger with every active concrete user-requested outcome plus any necessary "
                        "prerequisites. Each objective must name the requested outcome or target; never "
                        "use meta-goals such as 'determine next action', 'continue task', or 'process "
                        "request'. When one request applies to all, every, or each member of an observed "
                        "collection, expand it into one concrete goal per matching entity as soon as its "
                        "identifier is available. Keep each remaining entity in the ledger; completing one "
                        "instance never completes the collection. Include goal_ledger with compact goals, status, "
                        "depends_on, and next_goal_id.\n"
                        if stateful and stateful_goal_ledger_required
                        else ""
                    ),
                    (
                        "- The runtime owns the goal ledger. Do not return goal_ledger or mark any goal "
                        "completed, cancelled, or failed. When the current user turn introduces a concrete "
                        "new obligation, you may return goal_delta with an add list. Each addition needs "
                        "goal_id, kind (identify/retrieve/mutate/communicate), objective, optional "
                        "target_expression, optional quantifier, dependencies, and evidence_ids. Goal "
                        "deltas are proposals only: they cannot change or remove an existing goal. Do not "
                        "create a meta-goal such as 'continue task'.\n"
                        if stateful and not stateful_goal_ledger_required
                        else ""
                    ),
                    (
                        "- When the latest user turn corrects or adds a desired value, you may return "
                        "requested_fact_delta with a set list. Each item needs subject "
                        "({entity_type, entity_id}), predicate, value, and evidence copied exactly from "
                        "that latest user turn. Do not emit current environment values as requested facts. "
                        "Do not use a subject identifier unless it appears in the user turn or a recorded "
                        "observation. The runtime validates evidence, identity, supersession, and request "
                        "revision; a delta is not itself state.\n"
                        if stateful and not stateful_goal_ledger_required
                        else ""
                    ),
                    (
                        "- This is a stateful progress repair. Select a different grounded next action "
                        "that can use the observed state, or ask_user with precise missing_inputs. "
                        "Do not repeat any execution_history action. Before asking the user, inspect "
                        "the full transcript for values produced by earlier tool observations and bind "
                        "those values when the schema permits. If a required value is an opaque "
                        "identifier absent from user text, but an observation supplies a related "
                        "resource identifier and an available read-only tool can resolve it, retrieve "
                        "that resource before asking the user for the opaque identifier. When several "
                        "requested operations remain, select one legal next call by dependency; do not "
                        "ask the user which requested operation to do first. Use no_tool only when no "
                        "available capability can advance the active task. If confirmation is required, "
                        "record confirmation in missing_inputs. Return at most one tool binding because "
                        "this environment executes one state transition per turn.\n"
                        if require_stateful_progress
                        else ""
                    ),
                    (
                        "- A prior state-transition reviewer rejected a no-action decision. This repair "
                        "must not use no_tool. Return a grounded call, or ask_user with at least one "
                        "precise missing_inputs item.\n"
                        if stateful_no_action_forbidden
                        else ""
                    ),
                    (
                        "- A prior candidate was rejected by a state-transition reviewer: "
                        f"{str(stateful_feedback)[:600]}. Select a different action; do not repeat it.\n"
                        if stateful_feedback
                        else ""
                    ),
                    "Output fields: tool_decision string; canonical_request string; slots_observed list; "
                    "call_groups list with expected_call_count; tool_bindings list with "
                    "tool_name, call_count, argument_groups, arguments, evidence_spans; "
                    "missing_inputs list; optional clarification_message string; optional response_message string; "
                    + (
                        "required goal_ledger object with goals ({id, objective, status, depends_on}) and next_goal_id.\n"
                        if stateful_goal_ledger_required
                        else (
                            "optional goal_delta object with add (goal_id, kind, objective, target_expression, "
                            "quantifier, dependencies, evidence_ids); optional requested_fact_delta object with "
                            "set (subject, predicate, value, evidence). Do not emit goal_ledger.\n"
                            if stateful
                            else ""
                        )
                    )
                    + f"Input:{_compact_json(payload)}",
                ]
            ),
        },
    ]


def _compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = params.get("required") if isinstance(params.get("required"), list) else []
    compact_props = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        compact = {
            "type": spec.get("type") or "string",
            "description": str(spec.get("description") or "")[:180],
        }
        if "enum" in spec:
            compact["enum"] = spec.get("enum")
        if "default" in spec:
            compact["default"] = spec.get("default")
        if isinstance(spec.get("items"), dict):
            compact["items"] = {
                key: value
                for key, value in spec["items"].items()
                if key in {"type", "enum", "description"}
            }
        compact_props[str(name)] = compact
    return {
        "name": tool.get("name") or "",
        "description": str(tool.get("description") or "")[:240],
        "required": [str(item) for item in required],
        "properties": compact_props,
    }


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def extract_goal_graph_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the best goal-graph-shaped JSON object from model text.

    The generic planner extractor returns the last JSON object in a response.
    Goal graphs contain many nested objects, so the last object is often an
    input record or ``expected_effect`` rather than the full graph.
    """
    candidates = _json_object_candidates(text)
    fallback, fallback_error = extract_json_object(text)
    if fallback is not None:
        candidates.append(fallback)
    if not candidates:
        return None, fallback_error or "no valid JSON object found"

    def score(obj: dict[str, Any]) -> tuple[int, int]:
        if isinstance(obj.get("nodes"), list):
            points = 100
            if "goal" in obj:
                points += 20
            if "clarification_needed" in obj:
                points += 5
            return points, len(json.dumps(obj, default=str))
        if {"id", "kind", "capability"} <= set(obj):
            return 60, len(json.dumps(obj, default=str))
        return 0, len(json.dumps(obj, default=str))

    best = max(candidates, key=score)
    best_score, _ = score(best)
    if isinstance(best.get("nodes"), list):
        return best, None
    if {"id", "kind", "capability"} <= set(best):
        return {
            "goal": "",
            "nodes": [best],
            "clarification_needed": False,
            "clarification_reasons": [],
        }, None
    if best_score == 0:
        return None, fallback_error or "no goal graph JSON object found"
    return best, None


def extract_semantic_frame_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidates = _json_object_candidates(text)
    fallback, fallback_error = extract_json_object(text)
    if fallback is not None:
        candidates.append(fallback)
    for repaired in _semantic_frame_repair_candidates(text):
        candidates.append(repaired)
    if not candidates:
        return None, fallback_error or "no valid JSON object found"

    def score(obj: dict[str, Any]) -> tuple[int, int]:
        points = 0
        if "tool_decision" in obj:
            points += 20
        if "canonical_request" in obj:
            points += 20
        if isinstance(obj.get("slots_observed"), list):
            points += 40
        if isinstance(obj.get("call_groups"), list):
            points += 40
        if isinstance(obj.get("tool_bindings"), list):
            points += 50
        if isinstance(obj.get("missing_inputs"), list):
            points += 10
        return points, len(json.dumps(obj, default=str))

    best = max(candidates, key=score)
    best_score, _ = score(best)
    if best_score <= 0:
        return None, fallback_error or "no semantic frame JSON object found"
    normalized = dict(best)
    normalized.setdefault("tool_decision", "")
    normalized.setdefault("canonical_request", "")
    normalized.setdefault("slots_observed", [])
    normalized.setdefault("call_groups", [])
    normalized.setdefault("tool_bindings", [])
    normalized.setdefault("missing_inputs", [])
    return normalized, None


def _semantic_frame_repair_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    starts = [match.end() for match in re.finditer(r"assistantfinal\s*", text)]
    starts.extend(match.start() for match in re.finditer(r"\{\s*\"tool_decision\"", text))
    starts.extend(match.start() for match in re.finditer(r"\{\s*\"canonical_request\"", text))
    for start in starts:
        snippet = text[start:].strip()
        if not snippet.startswith("{"):
            brace = snippet.find("{")
            if brace < 0:
                continue
            snippet = snippet[brace:]
        for repaired in _semantic_frame_repair_variants(snippet):
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
    return candidates


def _semantic_frame_repair_variants(snippet: str) -> list[str]:
    variants = [snippet]
    # GPT-OSS sometimes emits a complete binding object but forgets the closing
    # bracket for the top-level tool_bindings array before missing_inputs.
    variants.append(
        re.sub(
            r'("tool_bindings"\s*:\s*\[.*\}\s*\]\s*\})\s*,\s*("missing_inputs"\s*:)',
            r'\1],\2',
            snippet,
            count=1,
            flags=re.S,
        )
    )
    balanced = _repair_json_delimiters(snippet)
    if balanced:
        variants.append(balanced)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        key = json.dumps(obj, sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        candidates.append(obj)

    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        add_candidate(obj)
    for obj in _repaired_json_object_candidates(text):
        add_candidate(obj)
    return candidates


def _repaired_json_object_candidates(text: str) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = _repair_json_delimiters(text[start:])
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            repaired.append(obj)
    return repaired


def _repair_json_delimiters(text: str) -> str | None:
    stack: list[str] = []
    out: list[str] = []
    quote = ""
    escaped = False
    started = False
    for char in text:
        out.append(char)
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            if char == '"':
                quote = char
            continue
        if char == "{":
            started = True
            stack.append("}")
            continue
        if char == "[":
            stack.append("]")
            continue
        if char not in {"}", "]"}:
            continue
        if not stack:
            out.pop()
            continue
        if char == stack[-1]:
            stack.pop()
        elif char in stack:
            out.pop()
            while stack and stack[-1] != char:
                out.append(stack.pop())
            out.append(char)
            if stack and stack[-1] == char:
                stack.pop()
        else:
            out.pop()
            continue
        if started and not stack:
            return "".join(out)
    if quote or not started:
        return None
    while stack:
        out.append(stack.pop())
    return "".join(out)


def _compile_binder_fallback_graph(
    runtime: GoalGraphRuntime,
    user_request: str,
    tools: list[dict[str, Any]],
    *,
    allow_side_effects: bool,
) -> dict[str, Any]:
    binding_plan = build_tool_binding_plan(user_request, tools)
    graph = _graph_from_binding_plan(runtime, user_request, binding_plan)
    output = runtime.compile(graph, user_request, allow_side_effects=allow_side_effects)
    diagnostics = diagnostics_to_dicts(output.verification)
    return {
        "used": True,
        "tool_binding_plan": binding_plan,
        "graph": graph,
        "verification": output.verification,
        "verification_ok": output.verification.ok,
        "diagnostics": diagnostics,
        "compiled_call_objects": output.calls,
        "calls": compiled_calls_to_dicts(output.calls),
    }


def _graph_from_binding_plan(
    runtime: GoalGraphRuntime,
    user_request: str,
    binding_plan: dict[str, Any],
) -> dict[str, Any]:
    tool_to_capability = {capability.tool_name: capability for capability in runtime.registry.values()}
    nodes = []
    for index, call in enumerate(binding_plan.get("calls") or []):
        if not isinstance(call, dict) or call.get("missing_arguments"):
            continue
        tool_name = str(call.get("tool_name") or "")
        capability = tool_to_capability.get(tool_name)
        if capability is None:
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        argument_evidence = call.get("argument_evidence") if isinstance(call.get("argument_evidence"), dict) else {}
        inputs = {}
        for name, value in arguments.items():
            if name not in capability.inputs:
                continue
            cap_input = capability.inputs[str(name)]
            evidence = str(argument_evidence.get(name) or "")
            if _binding_value_matches_default(value, cap_input.default) and (
                not evidence or not _evidence_in_request(evidence, user_request)
            ):
                inputs[str(name)] = {
                    "value": value,
                    "source": "policy_default",
                    "evidence": "",
                    "status": "defaulted",
                }
                continue
            inputs[str(name)] = {
                "value": value,
                "source": "query",
                "evidence": str(evidence or _fallback_evidence(value, user_request)),
                "status": "resolved",
            }
        nodes.append(
            {
                "id": f"n{index + 1}",
                "kind": capability.kind,
                "capability": capability.capability,
                "description": f"Fallback graph node for {tool_name}.",
                "inputs": inputs,
                "outputs": ["result"],
                "depends_on": [],
                "must_be_unique": bool(capability.requires_unique_target),
                "risk": capability.risk,
                "authorized": False,
                "policy_evidence": [],
                "expected_effect": {},
            }
        )
    return {
        "goal": user_request,
        "nodes": nodes,
        "clarification_needed": not nodes and bool(binding_plan.get("missing_inputs")),
        "clarification_reasons": list(binding_plan.get("missing_inputs") or []),
    }


def _binding_value_matches_default(value: Any, default: Any | None) -> bool:
    if default is None:
        return False
    return str(value).strip().lower() == str(default).strip().lower()


def _fallback_evidence(value: Any, user_request: str) -> str:
    if isinstance(value, list):
        parts = [_fallback_evidence(item, user_request) for item in value]
        joined = " ".join(part for part in parts if part)
        if joined and _evidence_in_request(joined, user_request):
            return joined
        return user_request
    value_text = str(value)
    if value_text and _evidence_in_request(value_text, user_request):
        return value_text
    readable = value_text.replace("_", " ")
    if readable and _evidence_in_request(readable, user_request):
        return readable
    return user_request


def _evidence_in_request(evidence: str, user_request: str) -> bool:
    evidence_text = re.sub(r"\s+", " ", evidence.strip()).lower()
    request_text = re.sub(r"\s+", " ", user_request).lower()
    if not evidence_text:
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?", evidence_text):
        return bool(re.search(rf"(?<![\d.]){re.escape(evidence_text)}(?![\d.])", request_text))
    if re.fullmatch(r"[a-z0-9_./:-]+", evidence_text) and re.search(r"\d", evidence_text):
        return bool(
            re.search(
                rf"(?<![a-z0-9_.:-]){re.escape(evidence_text)}(?![a-z0-9_:-]|\.(?=[a-z0-9_]))",
                request_text,
            )
        )
    return evidence_text in request_text


def _serializable_binder_fallback(fallback: dict[str, Any] | None) -> dict[str, Any]:
    if not fallback:
        return {"used": False}
    return {
        "used": True,
        "verification_ok": bool(fallback.get("verification_ok")),
        "diagnostics": fallback.get("diagnostics") or [],
        "calls": fallback.get("calls") or [],
        "tool_binding_plan": fallback.get("tool_binding_plan") or {},
    }


def _build_repair_messages(
    *,
    user_request: str,
    previous_output: str,
    parse_error: str | None,
    diagnostics: list[dict[str, str]],
    attempt: int,
) -> list[dict[str, str]]:
    if parse_error:
        issue_text = f"Parse error: {parse_error}"
    else:
        issue_text = "Verification diagnostics:\n" + "\n".join(
            f"- {item.get('code')}: {item.get('message')} at {item.get('path')}"
            for item in diagnostics[:12]
        )
    return [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. Repair the previous goal graph so it "
                "satisfies the runtime contract. Preserve grounded user intent. Do not "
                "invent concrete IDs, values, entities, or tool outputs. If a required "
                "value is unavailable, mark clarification_needed true and add ask_user."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Repair attempt {attempt}.\n"
                f"User request:\n{user_request}\n\n"
                f"{issue_text}\n\n"
                f"Previous output:\n{previous_output}\n\n"
                "Return the corrected goal-graph JSON object only."
            ),
        },
    ]
