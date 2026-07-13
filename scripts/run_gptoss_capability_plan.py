#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from taskdecomp.capability_planning import (
    PASS_ORDER,
    apply_missing_input_slot_filler,
    build_evidence_chunks,
    build_missing_input_slot_filler_messages,
    build_messages_for_pass,
    build_one_shot_capability_plan_messages,
    build_rules_first_capability_plan,
    build_semantic_slot_frame_messages,
    compact_json,
    extract_json_object,
    merge_split_intent_input_audit,
    repair_capability_requirements,
    repair_capability_normalization,
    repair_intent_input_audit,
    repair_capability_ordering,
    repair_transformation_audit,
    should_run_missing_input_slot_filler,
    validate_capability_plan,
    validate_required_top_level,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_json_arg(value: str | None) -> Any:
    if not value:
        return []
    stripped = value.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(stripped)


DEFAULT_ENSEMBLE_VARIANTS = [
    "chunked:baseline",
    "chunked:lexical",
    "chunked:slot_grounded",
    "one_shot:semantic",
    "rules_first:deterministic",
]

PLANNER_LENS_INSTRUCTIONS = {
    "baseline": "",
    "deterministic": "",
    "lexical": (
        "Ensemble lens: lexical evidence. Prefer exact spans in the request: identifiers, "
        "quoted strings, numbers, labels, file paths, URLs, and phrases after words like "
        "provided, given, using, on, from, for, with. If a variable-like token such as "
        "my_data, dataset_A, nums, k, coordinates, or a budget is named in the request, "
        "treat it as an available specification input, not missing user data."
    ),
    "semantic": (
        "Ensemble lens: semantic paraphrase. Map synonyms and paraphrases of the requested "
        "work before naming capabilities. Do not require the exact words from a capability "
        "name when the user intent clearly implies that capability."
    ),
    "slot_grounded": (
        "Ensemble lens: slot availability. For each required input, explicitly ask whether "
        "the request already contains a value, constraint, identifier, or variable that can "
        "fill it. Do not mark an input missing merely because the input is abstractly named "
        "data, dataset, file, budget, coordinates, k, nums, city, or query."
    ),
    "no_call_skeptic": (
        "Ensemble lens: no-call skepticism. Distinguish requests to perform an action from "
        "mentions of related words. Avoid external actions when the user is asking a general "
        "question, discussing a tool, or giving irrelevant context."
    ),
    "cardinality": (
        "Ensemble lens: cardinality and order. Count distinct requested units of work, "
        "entities, dates, locations, tools, and repeated groups. Preserve user-stated order "
        "and distinguish one batch call from repeated single-entity calls."
    ),
}


def parse_ensemble_variants(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_ENSEMBLE_VARIANTS)
    variants = [item.strip() for item in value.split(",") if item.strip()]
    return variants or list(DEFAULT_ENSEMBLE_VARIANTS)


def split_ensemble_variant(spec: str) -> tuple[str, str]:
    if ":" in spec:
        mode, lens = spec.split(":", 1)
    else:
        mode, lens = spec, "baseline"
    mode = mode.strip() or "chunked"
    lens = lens.strip() or "baseline"
    if mode == "multi_pass":
        mode = "multi_pass"
    if mode not in {"multi_pass", "chunked", "one_shot", "rules_first"}:
        raise ValueError(f"unsupported ensemble variant mode: {mode}")
    if lens not in PLANNER_LENS_INSTRUCTIONS:
        raise ValueError(f"unsupported ensemble lens: {lens}")
    return mode, lens


def ensemble_requires_model(variants: list[str], missing_input_slot_filler: bool) -> bool:
    for variant in variants:
        mode, _ = split_ensemble_variant(variant)
        if mode != "rules_first" or missing_input_slot_filler:
            return True
    return False


def apply_planner_lens(messages: list[dict[str, str]], lens: str, pass_key: str) -> list[dict[str, str]]:
    instruction = PLANNER_LENS_INSTRUCTIONS.get(lens, "")
    if not instruction:
        return messages
    adjusted = [dict(message) for message in messages]
    lens_text = f"\n{instruction} Apply this lens only for {pass_key}; still return exactly the requested JSON schema."
    for message in adjusted:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content', '')}{lens_text}"
            return adjusted
    return [{"role": "system", "content": instruction}, *adjusted]


def load_model(model_name: str):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_path = Path(model_name)
    if (model_path / "adapter_config.json").exists():
        loader = AutoPeftModelForCausalLM
    else:
        loader = AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "auto",
        "device_map": "auto",
    }
    if "gpt-oss" in model_name.lower() or (model_path / "adapter_config.json").exists():
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    model = loader.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_text(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    # GPT-OSS otherwise emits a long reasoning channel before its final answer.
    # The runtime asks for machine-readable contracts, so reserve the generation
    # budget for that contract rather than accepting an unfinished thought trace.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def repair_json_once(
    model,
    tokenizer,
    pass_key: str,
    raw_text: str,
    parse_error: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    messages = [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. Repair syntax/schema only. Preserve the "
                "previous output's meaning and do not add new planning work."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The previous {pass_key} output was invalid JSON or missed required fields.\n"
                f"Parse/schema error: {parse_error}\n"
                f"Previous output:\n{raw_text}\n"
                "Return the corrected JSON object only."
            ),
        },
    ]
    retry_text = generate_text(model, tokenizer, messages, max_new_tokens)
    parsed, retry_error = extract_json_object(retry_text)
    if parsed is not None:
        missing = validate_required_top_level(pass_key, parsed)
        if missing:
            retry_error = f"missing top-level fields after retry: {', '.join(missing)}"
            parsed = None
    return retry_text, parsed, retry_error


def generate_json_for_pass(
    model,
    tokenizer,
    pass_key: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    raw_text = generate_text(model, tokenizer, messages, max_new_tokens)
    parsed, parse_error = extract_json_object(raw_text)
    if parsed is not None:
        missing = validate_required_top_level(pass_key, parsed)
        if missing:
            parse_error = f"missing top-level fields: {', '.join(missing)}"
            parsed = None
    retry_text = None
    if parsed is None:
        retry_text, parsed, parse_error = repair_json_once(
            model,
            tokenizer,
            pass_key,
            raw_text,
            parse_error or "unknown parse error",
            max_new_tokens,
        )
    return {
        "raw_text": raw_text,
        "retry_raw_text": retry_text,
        "parsed": parsed,
        "parse_error": parse_error if parsed is None else None,
    }


def row_to_request(row: dict[str, Any]) -> tuple[str, str, Any]:
    task = row.get("task") or row.get("request") or row.get("user_request") or row.get("prompt")
    if task is None and row.get("user_messages"):
        task = "\n\n".join(
            f"TURN {message.get('turn', index + 1)}: {message.get('content', '')}"
            for index, message in enumerate(row["user_messages"])
        )
    if task is None:
        raise ValueError(
            "Input row must contain task, request, user_request, prompt, or user_messages"
        )
    context = str(row.get("context") or "")
    attachments_metadata = row.get("attachments_metadata", row.get("attachments", []))
    return str(task), context, attachments_metadata


def run_capability_plan(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    max_new_tokens: int,
    planner_mode: str = "chunked",
    missing_input_slot_filler: bool = False,
    planner_lens: str = "baseline",
    ensemble_variants: list[str] | None = None,
) -> dict[str, Any]:
    if planner_mode == "ensemble":
        return run_ensemble_capability_plan(
            model,
            tokenizer,
            task,
            context,
            attachments_metadata,
            max_new_tokens,
            ensemble_variants or DEFAULT_ENSEMBLE_VARIANTS,
            missing_input_slot_filler,
        )
    if planner_mode == "rules_first":
        deterministic = build_rules_first_capability_plan(
            task,
            context,
            attachments_metadata,
        )
        if missing_input_slot_filler:
            deterministic = run_rules_first_missing_input_slot_filler(
                model,
                tokenizer,
                task,
                context,
                attachments_metadata,
                deterministic,
                max_new_tokens,
            )
        return {
            "task": task,
            "context": context,
            "attachments_metadata": attachments_metadata,
            "planner_mode": planner_mode,
            "planner_lens": planner_lens,
            **deterministic,
        }
    if planner_mode == "one_shot":
        return run_one_shot_capability_plan(
            model,
            tokenizer,
            task,
            context,
            attachments_metadata,
            max_new_tokens,
            planner_lens,
        )

    parsed_by_pass: dict[str, dict[str, Any]] = {}
    pass_outputs: dict[str, dict[str, Any]] = {}

    for pass_key in PASS_ORDER:
        try:
            if pass_key == "intent_input_audit" and planner_mode == "chunked":
                output = run_split_intent_input_audit(
                    model,
                    tokenizer,
                    task,
                    context,
                    attachments_metadata,
                    max_new_tokens,
                    planner_lens,
                )
                pass_outputs[pass_key] = output
                parsed_by_pass[pass_key] = output["parsed"] or {}
                continue
            messages = build_messages_for_pass(
                pass_key,
                user_request=task,
                context=context,
                attachments_metadata=attachments_metadata,
                previous=parsed_by_pass,
            )
            messages = apply_planner_lens(messages, planner_lens, pass_key)
            output = generate_json_for_pass(model, tokenizer, pass_key, messages, max_new_tokens)
            if pass_key == "intent_input_audit" and output["parsed"] is not None:
                repaired, repairs = repair_intent_input_audit(
                    task,
                    attachments_metadata,
                    output["parsed"],
                )
                output["parsed"] = repaired
                output["repairs_applied"] = repairs
            if (
                pass_key == "transformation_externality_audit"
                and output["parsed"] is not None
            ):
                repaired, repairs = repair_transformation_audit(
                    task,
                    parsed_by_pass.get("intent_input_audit"),
                    output["parsed"],
                )
                output["parsed"] = repaired
                output["repairs_applied"] = repairs
            if pass_key == "capability_requirements":
                if output["parsed"] is None:
                    output["repaired_parse_error"] = output["parse_error"]
                    output["parsed"] = {"capabilities_needed": []}
                    output["parse_error"] = None
                repaired, repairs = repair_capability_requirements(
                    parsed_by_pass.get("intent_input_audit"),
                    parsed_by_pass.get("transformation_externality_audit"),
                    output["parsed"],
                )
                output["parsed"] = repaired
                output["repairs_applied"] = repairs
            if pass_key == "capability_normalization":
                if output["parsed"] is None:
                    output["repaired_parse_error"] = output["parse_error"]
                    output["parsed"] = {"normalized_capabilities": [], "merged_capabilities": []}
                    output["parse_error"] = None
                repaired, repairs = repair_capability_normalization(
                    parsed_by_pass.get("capability_requirements"),
                    output["parsed"],
                )
                output["parsed"] = repaired
                output["repairs_applied"] = repairs
            if pass_key == "capability_ordering":
                if output["parsed"] is None:
                    output["repaired_parse_error"] = output["parse_error"]
                    output["parsed"] = {"ordered_capabilities": []}
                    output["parse_error"] = None
                repaired, repairs = repair_capability_ordering(
                    parsed_by_pass.get("capability_requirements"),
                    output["parsed"],
                )
                output["parsed"] = repaired
                output["repairs_applied"] = repairs
        except Exception as exc:
            output = {
                "raw_text": "",
                "retry_raw_text": None,
                "parsed": None,
                "parse_error": f"{type(exc).__name__}: {exc}",
            }
        pass_outputs[pass_key] = output
        parsed_by_pass[pass_key] = output["parsed"] or {}

    validation = validate_capability_plan(
        parsed_by_pass.get("intent_input_audit"),
        parsed_by_pass.get("transformation_externality_audit"),
        parsed_by_pass.get("capability_requirements"),
        parsed_by_pass.get("capability_normalization"),
        parsed_by_pass.get("capability_ordering"),
    )
    return {
        "task": task,
        "context": context,
        "attachments_metadata": attachments_metadata,
        "planner_mode": planner_mode,
        "planner_lens": planner_lens,
        "passes": pass_outputs,
        "ordered_capability_plan": parsed_by_pass.get("capability_ordering", {}),
        "validation": validation,
    }


def run_semantic_slot_frame(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    capability_plan: dict[str, Any] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one bounded GPT pass that proposes input slots/call groups for the binder."""
    if model is None or tokenizer is None:
        return {}
    intent = {}
    if isinstance(capability_plan, dict):
        passes = capability_plan.get("passes") if isinstance(capability_plan.get("passes"), dict) else {}
        intent_pass = passes.get("intent_input_audit") if isinstance(passes, dict) else {}
        if isinstance(intent_pass, dict) and isinstance(intent_pass.get("parsed"), dict):
            intent = intent_pass["parsed"]
        elif isinstance(capability_plan.get("intent_input_audit"), dict):
            intent = capability_plan["intent_input_audit"]
    messages = build_semantic_slot_frame_messages(
        task,
        context,
        attachments_metadata,
        intent,
    )
    output = generate_json_for_pass(
        model,
        tokenizer,
        "semantic_slot_frame",
        messages,
        min(max_new_tokens, 650),
    )
    if output["parsed"] is None:
        output["repaired_parse_error"] = output["parse_error"]
        output["parsed"] = {"canonical_request": task[:240], "slots_observed": [], "call_groups": [], "missing_inputs": []}
        output["parse_error"] = None
    parsed = output["parsed"] if isinstance(output["parsed"], dict) else {}
    return parsed


def run_ensemble_capability_plan(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    max_new_tokens: int,
    ensemble_variants: list[str],
    missing_input_slot_filler: bool = False,
) -> dict[str, Any]:
    variant_results = []
    for raw_spec in ensemble_variants:
        mode, lens = split_ensemble_variant(raw_spec)
        result = run_capability_plan(
            model,
            tokenizer,
            task,
            context,
            attachments_metadata,
            max_new_tokens,
            mode,
            missing_input_slot_filler if mode == "rules_first" else False,
            lens,
            [],
        )
        result["ensemble_variant"] = raw_spec
        variant_results.append(result)

    resolved = resolve_capability_ensemble(task, context, attachments_metadata, variant_results)
    return {
        "task": task,
        "context": context,
        "attachments_metadata": attachments_metadata,
        "planner_mode": "ensemble",
        "passes": resolved["passes"],
        "ordered_capability_plan": resolved["ordered_capability_plan"],
        "validation": resolved["validation"],
        "ensemble": {
            "variants_requested": ensemble_variants,
            "variant_summaries": [_summarize_variant_result(result) for result in variant_results],
            "resolver": resolved["resolver"],
        },
    }


def resolve_capability_ensemble(
    task: str,
    context: str,
    attachments_metadata: Any,
    variant_results: list[dict[str, Any]],
) -> dict[str, Any]:
    intents = [_intent_from_result(result) for result in variant_results]
    final_user_want = _choose_final_user_want(task, intents)
    available_inputs = _merge_available_inputs(intents)
    missing_inputs, missing_votes = _merge_missing_inputs(intents, available_inputs)

    intent = {
        "final_user_want": final_user_want,
        "inputs": available_inputs,
        "missing_inputs": missing_inputs,
    }
    intent, intent_repairs = repair_intent_input_audit(task, attachments_metadata, intent)

    transform = _first_parsed_section(variant_results, "transformation_externality_audit")
    transform, transform_repairs = repair_transformation_audit(task, intent, transform)
    requirements = _merge_capability_requirements(variant_results, intent, transform)
    requirements, requirement_repairs = repair_capability_requirements(intent, transform, requirements)
    normalization = _merge_capability_normalization(variant_results, requirements)
    normalization, normalization_repairs = repair_capability_normalization(requirements, normalization)
    ordering = _merge_capability_ordering(variant_results, requirements)
    ordering, ordering_repairs = repair_capability_ordering(requirements, ordering)

    validation = validate_capability_plan(intent, transform, requirements, normalization, ordering)
    passes = {
        "intent_input_audit": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": intent,
            "parse_error": None,
            "repairs_applied": intent_repairs,
        },
        "transformation_externality_audit": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": transform,
            "parse_error": None,
            "repairs_applied": transform_repairs,
        },
        "capability_requirements": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": requirements,
            "parse_error": None,
            "repairs_applied": requirement_repairs,
        },
        "capability_normalization": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": normalization,
            "parse_error": None,
            "repairs_applied": normalization_repairs,
        },
        "capability_ordering": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": ordering,
            "parse_error": None,
            "repairs_applied": ordering_repairs,
        },
    }
    return {
        "passes": passes,
        "ordered_capability_plan": ordering,
        "validation": validation,
        "resolver": {
            "strategy": "union_available_inputs_majority_missing_inputs",
            "variant_count": len(variant_results),
            "available_input_count": len(intent.get("inputs", [])),
            "missing_votes": missing_votes,
            "missing_after_rescue": intent.get("missing_inputs", []),
        },
    }


def _intent_from_result(result: dict[str, Any]) -> dict[str, Any]:
    passes = result.get("passes") if isinstance(result.get("passes"), dict) else {}
    intent_pass = passes.get("intent_input_audit") if isinstance(passes, dict) else {}
    intent = intent_pass.get("parsed") if isinstance(intent_pass, dict) else {}
    return intent if isinstance(intent, dict) else {"inputs": [], "missing_inputs": []}


def _choose_final_user_want(task: str, intents: list[dict[str, Any]]) -> str:
    for intent in intents:
        value = str(intent.get("final_user_want") or "").strip()
        if value:
            return value
    return task.strip()


def _merge_available_inputs(intents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for intent in intents:
        for item in _list(intent.get("inputs")):
            if not isinstance(item, dict) or item.get("available") is not True:
                continue
            normalized = _normalize_input_key(
                str(item.get("name") or ""),
                str(item.get("evidence") or item.get("evidence_span") or ""),
            )
            if not normalized:
                continue
            current = merged.get(normalized)
            candidate = {
                "name": str(item.get("name") or normalized),
                "needed_for": str(item.get("needed_for") or ""),
                "available": True,
                "format": str(item.get("format") or "unknown"),
                "evidence": str(item.get("evidence") or item.get("evidence_span") or "")[:240],
            }
            if current is None or _input_specificity(candidate) > _input_specificity(current):
                merged[normalized] = candidate
    return list(merged.values())


def _merge_missing_inputs(
    intents: list[dict[str, Any]],
    available_inputs: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, int]]:
    votes: Counter[str] = Counter()
    originals: dict[str, Any] = {}
    for intent in intents:
        seen_in_variant: set[str] = set()
        for item in _list(intent.get("missing_inputs")):
            name = _missing_input_name(item)
            key = _normalize_text(name)
            if not key or key in seen_in_variant:
                continue
            seen_in_variant.add(key)
            votes[key] += 1
            originals.setdefault(key, item)

    majority = max(1, len(intents) // 2 + 1)
    kept = []
    for key, count in votes.items():
        if count < majority:
            continue
        original = originals[key]
        if _available_input_rescues_missing(key, available_inputs):
            continue
        kept.append(original)
    return kept, dict(votes)


def _merge_capability_requirements(
    variant_results: list[dict[str, Any]],
    intent: dict[str, Any],
    transform: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    seen = set()
    for result in variant_results:
        parsed = _parsed_section(result, "capability_requirements")
        for cap in _list(parsed.get("capabilities_needed")):
            if not isinstance(cap, dict):
                continue
            name = str(cap.get("capability_name") or cap.get("name") or "").strip()
            if not name:
                continue
            key = _normalize_text(name)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cap)
    if candidates:
        return {"capabilities_needed": candidates}
    return repair_capability_requirements(intent, transform, {"capabilities_needed": []})[0]


def _merge_capability_normalization(
    variant_results: list[dict[str, Any]],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    parsed = _first_parsed_section(variant_results, "capability_normalization")
    if parsed:
        return parsed
    return _normalization_from_requirements(requirements)


def _merge_capability_ordering(
    variant_results: list[dict[str, Any]],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    ordered = []
    seen = set()
    for result in variant_results:
        parsed = _parsed_section(result, "capability_ordering")
        for cap in _list(parsed.get("ordered_capabilities")):
            if not isinstance(cap, dict):
                continue
            name = str(cap.get("capability_name") or cap.get("name") or "").strip()
            if not name:
                continue
            key = _normalize_text(name)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cap)
    if ordered:
        return {"ordered_capabilities": ordered}
    return repair_capability_ordering(requirements, {"ordered_capabilities": []})[0]


def _first_parsed_section(variant_results: list[dict[str, Any]], pass_key: str) -> dict[str, Any]:
    for result in variant_results:
        parsed = _parsed_section(result, pass_key)
        if parsed:
            return parsed
    return {}


def _parsed_section(result: dict[str, Any], pass_key: str) -> dict[str, Any]:
    passes = result.get("passes") if isinstance(result.get("passes"), dict) else {}
    output = passes.get(pass_key) if isinstance(passes, dict) else {}
    parsed = output.get("parsed") if isinstance(output, dict) else {}
    return parsed if isinstance(parsed, dict) else {}


def _summarize_variant_result(result: dict[str, Any]) -> dict[str, Any]:
    intent = _intent_from_result(result)
    ordering = _parsed_section(result, "capability_ordering")
    return {
        "variant": result.get("ensemble_variant"),
        "planner_mode": result.get("planner_mode"),
        "planner_lens": result.get("planner_lens"),
        "input_count": len(_list(intent.get("inputs"))),
        "missing_inputs": [_missing_input_name(item) for item in _list(intent.get("missing_inputs"))],
        "ordered_capabilities": [
            str(cap.get("capability_name") or cap.get("name") or "")
            for cap in _list(ordering.get("ordered_capabilities"))
            if isinstance(cap, dict)
        ],
    }


def _available_input_rescues_missing(missing_key: str, available_inputs: list[dict[str, Any]]) -> bool:
    missing_tokens = set(missing_key.split())
    if not missing_tokens:
        return False
    for item in available_inputs:
        text = " ".join(
            [
                str(item.get("name") or ""),
                str(item.get("needed_for") or ""),
                str(item.get("format") or ""),
                str(item.get("evidence") or ""),
            ]
        )
        key = _normalize_text(text)
        tokens = set(key.split())
        if missing_key in key or key in missing_key:
            return True
        if missing_tokens & tokens:
            return True
    return False


def _missing_input_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("input") or item.get("label") or item)
    return str(item)


def _normalize_input_key(name: str, evidence: str) -> str:
    return _normalize_text(name) or _normalize_text(evidence)


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", str(value).lower()))


def _input_specificity(item: dict[str, Any]) -> int:
    score = len(str(item.get("evidence") or ""))
    if item.get("format") not in {"", "unknown"}:
        score += 20
    return score


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def run_rules_first_missing_input_slot_filler(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    deterministic: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    passes = deterministic.get("passes", {})
    intent_output = passes.get("intent_input_audit", {})
    intent = intent_output.get("parsed") if isinstance(intent_output, dict) else {}
    task_family = deterministic.get("task_family", {})
    if not should_run_missing_input_slot_filler(task, intent, task_family):
        return deterministic
    if model is None or tokenizer is None:
        raise ValueError("rules-first missing-input slot filler requires a loaded model")

    messages = build_missing_input_slot_filler_messages(
        task,
        context,
        attachments_metadata,
        intent,
        task_family,
        deterministic.get("evidence_chunks"),
    )
    output = generate_json_for_pass(
        model,
        tokenizer,
        "missing_input_slot_filler",
        messages,
        min(max_new_tokens, 450),
    )
    output["repairs_applied"] = []
    if output["parsed"] is not None:
        filled_intent, repairs = apply_missing_input_slot_filler(
            task,
            attachments_metadata,
            intent,
            output["parsed"],
            task_family,
        )
        output["repairs_applied"] = repairs
        if repairs:
            base_repairs = []
            if isinstance(intent_output, dict):
                base_repairs = list(intent_output.get("repairs_applied") or [])
            deterministic = build_rules_first_capability_plan(
                task,
                context,
                attachments_metadata,
                intent_override=filled_intent,
                intent_extra_repairs=base_repairs + repairs,
            )
    deterministic.setdefault("passes", {})["missing_input_slot_filler"] = output
    return deterministic


def run_one_shot_capability_plan(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    max_new_tokens: int,
    planner_lens: str = "baseline",
) -> dict[str, Any]:
    messages = build_one_shot_capability_plan_messages(task, context, attachments_metadata)
    messages = apply_planner_lens(messages, planner_lens, "one_shot_capability_plan")
    one_shot = generate_json_for_pass(
        model,
        tokenizer,
        "one_shot_capability_plan",
        messages,
        max_new_tokens,
    )
    parsed = one_shot["parsed"] if isinstance(one_shot["parsed"], dict) else {}
    if one_shot["parsed"] is None:
        one_shot["repaired_parse_error"] = one_shot["parse_error"]
        one_shot["parse_error"] = None

    intent, intent_repairs = repair_intent_input_audit(
        task,
        attachments_metadata,
        _section(parsed, "intent_input_audit"),
    )
    transform, transform_repairs = repair_transformation_audit(
        task,
        intent,
        _section(parsed, "transformation_externality_audit"),
    )
    requirements, requirement_repairs = repair_capability_requirements(
        intent,
        transform,
        _section(parsed, "capability_requirements"),
    )
    normalization = _section(parsed, "capability_normalization")
    if not normalization:
        normalization = _normalization_from_requirements(requirements)
    normalization, normalization_repairs = repair_capability_normalization(
        requirements,
        normalization,
    )
    ordering, ordering_repairs = repair_capability_ordering(
        requirements,
        _section(parsed, "capability_ordering"),
    )

    pass_outputs = {
        "one_shot_capability_plan": one_shot,
        "intent_input_audit": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": intent,
            "parse_error": None,
            "repairs_applied": intent_repairs,
        },
        "transformation_externality_audit": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": transform,
            "parse_error": None,
            "repairs_applied": transform_repairs,
        },
        "capability_requirements": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": requirements,
            "parse_error": None,
            "repairs_applied": requirement_repairs,
        },
        "capability_normalization": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": normalization,
            "parse_error": None,
            "repairs_applied": normalization_repairs,
        },
        "capability_ordering": {
            "raw_text": "",
            "retry_raw_text": None,
            "parsed": ordering,
            "parse_error": None,
            "repairs_applied": ordering_repairs,
        },
    }
    validation = validate_capability_plan(
        intent,
        transform,
        requirements,
        normalization,
        ordering,
    )
    return {
        "task": task,
        "context": context,
        "attachments_metadata": attachments_metadata,
        "planner_mode": "one_shot",
        "planner_lens": planner_lens,
        "passes": pass_outputs,
        "ordered_capability_plan": ordering,
        "validation": validation,
    }


def _section(parsed: dict[str, Any], key: str) -> dict[str, Any]:
    value = parsed.get(key)
    return value if isinstance(value, dict) else {}


def _normalization_from_requirements(requirements: dict[str, Any]) -> dict[str, Any]:
    normalized = []
    for index, cap in enumerate(requirements.get("capabilities_needed") or []):
        if not isinstance(cap, dict):
            continue
        cap_id = str(cap.get("id") or f"cap_{index + 1}")
        name = str(cap.get("capability_name") or f"capability_{index + 1}")
        normalized.append(
            {
                "id": cap_id,
                "original_name": name,
                "normalized_name": _snake_case(name),
                "meaning_changed": False,
                "external_action_type": cap.get("external_action_type", "none"),
            }
        )
    return {"normalized_capabilities": normalized, "merged_capabilities": []}


def _snake_case(value: str) -> str:
    import re

    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    return "_".join(words) or "capability"


def run_split_intent_input_audit(
    model,
    tokenizer,
    task: str,
    context: str,
    attachments_metadata: Any,
    max_new_tokens: int,
    planner_lens: str = "baseline",
) -> dict[str, Any]:
    evidence_chunks = build_evidence_chunks(task, context, attachments_metadata)
    previous: dict[str, Any] = {"evidence_chunks": evidence_chunks}
    sub_outputs: dict[str, dict[str, Any]] = {}
    sub_max_new_tokens = min(max_new_tokens, 700)

    for sub_key in [
        "intent_final_user_want",
        "intent_required_inputs",
        "intent_input_availability",
    ]:
        messages = build_messages_for_pass(
            sub_key,
            user_request=task,
            context=context,
            attachments_metadata=attachments_metadata,
            previous=previous,
        )
        messages = apply_planner_lens(messages, planner_lens, sub_key)
        output = generate_json_for_pass(model, tokenizer, sub_key, messages, sub_max_new_tokens)
        if output["parsed"] is None:
            output["repaired_parse_error"] = output["parse_error"]
            output["parsed"] = _fallback_split_intent_output(sub_key, task)
            output["parse_error"] = None
        sub_outputs[sub_key] = output
        previous[sub_key] = output["parsed"] or {}

    merged = merge_split_intent_input_audit(
        previous.get("intent_final_user_want"),
        previous.get("intent_required_inputs"),
        previous.get("intent_input_availability"),
    )
    repaired, repairs = repair_intent_input_audit(task, attachments_metadata, merged)
    return {
        "raw_text": "",
        "retry_raw_text": None,
        "parsed": repaired,
        "parse_error": None,
        "sub_passes": sub_outputs,
        "evidence_chunks": evidence_chunks,
        "repairs_applied": repairs,
    }


def _fallback_split_intent_output(pass_key: str, task: str) -> dict[str, Any]:
    if pass_key == "intent_final_user_want":
        return {"final_user_want": task[:180]}
    if pass_key == "intent_required_inputs":
        return {"required_inputs": []}
    if pass_key == "intent_input_availability":
        return {"inputs": [], "missing_inputs": []}
    return {}


def write_one(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_batch(results: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(compact_json(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task")
    source.add_argument("--input", type=Path)
    parser.add_argument("--context", default="")
    parser.add_argument("--attachments-metadata", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--planner-mode",
        choices=["multi_pass", "chunked", "one_shot", "rules_first", "ensemble"],
        default="chunked",
    )
    parser.add_argument(
        "--ensemble-variants",
        default=",".join(DEFAULT_ENSEMBLE_VARIANTS),
        help=(
            "Comma-separated mode:lens variants for planner-mode=ensemble. "
            "Modes: multi_pass, chunked, one_shot, rules_first. "
            "Lenses: baseline, lexical, semantic, slot_grounded, no_call_skeptic, cardinality."
        ),
    )
    parser.add_argument(
        "--rules-first-missing-input-slot-filler",
        action="store_true",
        help=(
            "In rules_first mode, load the model for one narrow fallback call when Run 1 "
            "has no inputs or missing_inputs for a source-dependent request."
        ),
    )
    args = parser.parse_args()

    ensemble_variants = parse_ensemble_variants(args.ensemble_variants)
    if (
        args.planner_mode == "rules_first"
        and not args.rules_first_missing_input_slot_filler
    ) or (
        args.planner_mode == "ensemble"
        and not ensemble_requires_model(ensemble_variants, args.rules_first_missing_input_slot_filler)
    ):
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model(args.model)

    if args.task is not None:
        attachments_metadata = parse_json_arg(args.attachments_metadata)
        result = run_capability_plan(
            model,
            tokenizer,
            args.task,
            args.context,
            attachments_metadata,
            args.max_new_tokens,
            args.planner_mode,
            args.rules_first_missing_input_slot_filler,
            "baseline",
            ensemble_variants,
        )
        result["model"] = args.model
        result["planner_mode"] = args.planner_mode
        write_one(result, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            try:
                task, context, attachments_metadata = row_to_request(row)
                result = run_capability_plan(
                    model,
                    tokenizer,
                    task,
                    context,
                    attachments_metadata,
                    args.max_new_tokens,
                    args.planner_mode,
                    args.rules_first_missing_input_slot_filler,
                    "baseline",
                    ensemble_variants,
                )
                result.update(
                    {
                        "id": row.get("id"),
                        "row_index": row.get("row_index", index),
                        "label": row.get("label"),
                        "category": row.get("category"),
                        "model": args.model,
                        "planner_mode": args.planner_mode,
                    }
                )
            except Exception as exc:
                result = {
                    "id": row.get("id"),
                    "row_index": row.get("row_index", index),
                    "category": row.get("category"),
                    "model": args.model,
                    "planner_mode": args.planner_mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "validation": {
                        "valid": False,
                        "violations": [
                            {
                                "type": "runner_error",
                                "message": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                        "minimal_repairs": [],
                    },
                }
            handle.write(compact_json(result) + "\n")
            handle.flush()
            print(
                compact_json(
                    {
                        "id": result.get("id"),
                        "row_index": result.get("row_index"),
                        "valid": result.get("validation", {}).get("valid"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
