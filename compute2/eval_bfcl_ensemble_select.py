#!/usr/bin/env python3
"""Select BFCL tool-call predictions from multiple candidate JSONLs.

Inputs are ordinary BFCL prediction files produced by eval_bfcl_hf_adapter.py
or eval_bfcl_toolace_official.py. The output keeps the existing
`baseline.normalized_calls` lane so score_bfcl_official_ast.py can score it
unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    from bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
    from smoke_eval import dedupe_tool_calls, openai_tools
except ImportError:
    from .bfcl_compare_eval import extract_prompt, load_records, normalize_tools, record_id
    from .smoke_eval import dedupe_tool_calls, openai_tools

try:
    from taskdecomp.capability_planning import build_rules_first_capability_plan
except ImportError as exc:  # pragma: no cover - depends on compute2 PYTHONPATH
    build_rules_first_capability_plan = None
    CAPABILITY_PLANNER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    CAPABILITY_PLANNER_IMPORT_ERROR = ""


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "for",
    "from",
    "give",
    "help",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "provide",
    "show",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "want",
    "with",
    "you",
}


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def split_identifier(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.sub(r"[_\.\-/]+", " ", text)


def tokens(text: Any) -> set[str]:
    raw = split_identifier(str(text or "")).lower()
    return {tok for tok in re.findall(r"[a-z0-9]{2,}", raw) if tok not in STOPWORDS}


def schema_text(tool: dict[str, Any]) -> str:
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    parts = [tool.get("name", ""), tool.get("description", "")]
    for key, spec in props.items():
        parts.append(str(key))
        if isinstance(spec, dict):
            parts.append(str(spec.get("description") or ""))
            enum = spec.get("enum")
            if isinstance(enum, list):
                parts.extend(str(item) for item in enum)
    return " ".join(parts)


def call_relevance(candidate: dict[str, Any], tools: list[dict[str, Any]], prompt: str) -> float:
    calls = candidate.get("calls") or []
    if not calls:
        return 0.0
    prompt_tokens = tokens(prompt)
    if not prompt_tokens:
        return 0.0
    by_name = tool_map(tools)
    scores: list[float] = []
    for call in calls:
        tool = by_name.get(str(call.get("name") or ""))
        if tool is None:
            scores.append(0.0)
            continue
        tool_tokens = tokens(schema_text(tool))
        name_tokens = tokens(tool.get("name", ""))
        args = maybe_json(call.get("arguments") or {})
        arg_tokens = tokens(json.dumps(args, ensure_ascii=False, default=str))
        tool_overlap = len(prompt_tokens & tool_tokens) / max(1, min(len(prompt_tokens), len(tool_tokens)))
        name_overlap = len(prompt_tokens & name_tokens) / max(1, len(name_tokens))
        arg_overlap = len(prompt_tokens & arg_tokens) / max(1, min(len(prompt_tokens), len(arg_tokens) or 1))
        scores.append((0.55 * tool_overlap) + (0.30 * name_overlap) + (0.15 * arg_overlap))
    return min(scores) if scores else 0.0


def call_tuple(calls: list[dict[str, Any]]) -> str:
    normalized = []
    for call in calls:
        args = maybe_json(call.get("arguments") or {})
        normalized.append(
            {
                "name": str(call.get("name") or ""),
                "arguments": args if isinstance(args, dict) else {"value": args},
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)


def valid_call_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in candidates if item.get("calls") and not item.get("issues")]


def empty_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in candidates if not item.get("calls") and not item.get("issues")]


def strongest_call_consensus(candidates: list[dict[str, Any]]) -> tuple[int, float]:
    valid = valid_call_candidates(candidates)
    if not valid:
        return 0, 0.0
    counts: dict[str, int] = {}
    for item in valid:
        key = call_tuple(item.get("calls") or [])
        counts[key] = counts.get(key, 0) + 1
    strongest = max(counts.values(), default=0)
    share = strongest / max(1, len(valid))
    return strongest, share


def supported_by_consensus(candidate: dict[str, Any], candidates: list[dict[str, Any]], min_count: int = 2) -> bool:
    if not candidate.get("calls"):
        return False
    key = call_tuple(candidate.get("calls") or [])
    return sum(1 for item in valid_call_candidates(candidates) if call_tuple(item.get("calls") or []) == key) >= min_count


def best_empty_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    empty = empty_candidates(candidates)
    if not empty:
        return None
    return sorted(empty, key=lambda item: source_bias(str(item.get("source"))), reverse=True)[0]


def should_abstain(
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt: str,
    capability_hint: dict[str, Any] | None = None,
) -> bool:
    empty = empty_candidates(candidates)
    valid_calls = valid_call_candidates(candidates)
    if not empty or not valid_calls:
        return False
    if _capability_prefers_tool_call(capability_hint):
        return False
    if len(empty) < 2:
        return False
    max_rel = max(call_relevance(item, tools, prompt) for item in valid_calls)
    strongest_call_support, strongest_call_share = strongest_call_consensus(candidates)
    if strongest_call_support >= 2 and strongest_call_share >= 0.5 and max_rel >= 0.30:
        return False
    if max_rel < 0.25:
        return True
    if len(valid_calls) <= 1 and len(empty) >= 2 and max_rel < 0.55:
        return True
    if len(empty) >= len(valid_calls) and max_rel < 0.42:
        return True
    if len(empty) > len(valid_calls) and max_rel < 0.62:
        return True
    if strongest_call_support <= 1 and len(empty) >= 2 and max_rel < 0.50:
        return True
    return False



def compact_tool_context(tools: list[dict[str, Any]], limit: int = 6000) -> str:
    compact = []
    for tool in tools:
        params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        compact.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "required": params.get("required") if isinstance(params.get("required"), list) else [],
                "properties": {
                    name: {
                        "type": spec.get("type"),
                        "description": spec.get("description", ""),
                        **({"enum": spec.get("enum")} if isinstance(spec.get("enum"), list) else {}),
                    }
                    for name, spec in props.items()
                    if isinstance(spec, dict)
                },
            }
        )
    text = json.dumps({"benchmark": "bfcl", "available_tools": compact}, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def max_tool_schema_relevance(tools: list[dict[str, Any]], prompt: str) -> float:
    scores = []
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        scores.append(call_relevance({"calls": [{"name": name, "arguments": {}}]}, tools, prompt))
    return max(scores, default=0.0)


def looks_like_multiple_external_actions(prompt: str) -> bool:
    lowered = prompt.lower()
    multi_markers = (
        "for each",
        "for both",
        "for all",
        "separately",
        "respectively",
        "in parallel",
        "simultaneously",
        "each of",
        "both of",
    )
    if any(marker in lowered for marker in multi_markers):
        return True
    if re.search(r"\b(compare|calculate|compute|get|find|search|look up|retrieve)\b", lowered) and re.search(
        r"\b(and|,)\b", lowered
    ):
        return len(re.findall(r"['\"][^'\"]+['\"]", prompt)) >= 2
    return False


def build_capability_selection_hint(
    prompt: str,
    tools: list[dict[str, Any]],
    mode: str = "auto",
) -> dict[str, Any]:
    if mode == "off":
        return {"enabled": False, "decision": "neutral", "reason": "disabled"}
    if build_rules_first_capability_plan is None:
        hint = {
            "enabled": False,
            "decision": "neutral",
            "reason": "capability planner unavailable",
            "error": CAPABILITY_PLANNER_IMPORT_ERROR,
        }
        if mode == "on":
            raise RuntimeError(hint["error"])
        return hint

    try:
        plan = build_rules_first_capability_plan(prompt, context=compact_tool_context(tools))
    except Exception as exc:  # noqa: BLE001
        hint = {
            "enabled": False,
            "decision": "neutral",
            "reason": "capability planner failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if mode == "on":
            raise
        return hint

    task_route = plan.get("task_route") or plan.get("task_frame") or {}
    passes = plan.get("passes") if isinstance(plan.get("passes"), dict) else {}
    intent = (passes.get("intent_input_audit") or {}).get("parsed") or {}
    missing_inputs = intent.get("missing_inputs") or []
    route = str(task_route.get("route") or "")
    operation = str(task_route.get("operation") or "")
    source_requirement = str(task_route.get("source_requirement") or "")
    external_actions = [str(item) for item in task_route.get("external_action_types") or []]
    tool_relevance = max_tool_schema_relevance(tools, prompt)
    multi_call_expected = looks_like_multiple_external_actions(prompt)

    decision = "neutral"
    reason = "planner did not force selector direction"
    if missing_inputs:
        decision = "ask_user_expected"
        reason = "capability planner found missing user input"
    elif route == "external_tool_action" or operation == "external_tool":
        decision = "tool_expected"
        reason = "capability planner routed request to external tool action"
    elif source_requirement == "self_contained" and tool_relevance < 0.20:
        decision = "no_tool_expected"
        reason = "planner sees a self-contained request and visible tools weakly match"
    elif tool_relevance >= 0.30:
        decision = "tool_expected"
        reason = "visible tool schemas strongly match the request"

    return {
        "enabled": True,
        "influence_selection": mode == "on",
        "decision": decision,
        "reason": reason,
        "route": route,
        "operation": operation,
        "source_requirement": source_requirement,
        "external_action_types": external_actions,
        "missing_inputs": missing_inputs,
        "max_tool_schema_relevance": round(tool_relevance, 3),
        "multi_call_expected": multi_call_expected,
        "validation_valid": bool((plan.get("validation") or {}).get("valid")),
    }


def _capability_can_influence(capability_hint: dict[str, Any] | None) -> bool:
    if not capability_hint:
        return False
    return bool(capability_hint.get("enabled") and capability_hint.get("influence_selection"))


def _capability_prefers_tool_call(capability_hint: dict[str, Any] | None) -> bool:
    if not _capability_can_influence(capability_hint):
        return False
    return capability_hint.get("decision") == "tool_expected"


def _capability_prefers_no_call(capability_hint: dict[str, Any] | None) -> bool:
    if not _capability_can_influence(capability_hint):
        return False
    return capability_hint.get("decision") in {"no_tool_expected", "ask_user_expected"}


def capability_empty_gate(
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt: str,
    capability_hint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not _capability_prefers_no_call(capability_hint):
        return None
    empty = best_empty_candidate(candidates)
    if not empty:
        return None
    valid_calls = valid_call_candidates(candidates)
    if not valid_calls:
        return empty
    max_rel = max(call_relevance(item, tools, prompt) for item in valid_calls)
    strongest_call_support, strongest_call_share = strongest_call_consensus(candidates)
    reliable_call = (
        (strongest_call_support >= 2 and strongest_call_share >= 0.5 and max_rel >= 0.30)
        or max_rel >= 0.62
    )
    if reliable_call:
        return None
    if capability_hint.get("decision") == "ask_user_expected":
        return empty if strongest_call_support <= 1 and max_rel < 0.42 else None
    if max_rel >= 0.25:
        if not (strongest_call_support <= 1 and max_rel < 0.42 and len(empty_candidates(candidates)) >= len(valid_calls)):
            return None
    return empty


def apply_capability_hint_scores(
    candidates: list[dict[str, Any]],
    capability_hint: dict[str, Any] | None,
) -> None:
    if not _capability_can_influence(capability_hint):
        return
    decision = capability_hint.get("decision")
    multi_call_expected = bool(capability_hint.get("multi_call_expected"))
    for item in candidates:
        calls = item.get("calls") or []
        issues = item.get("issues") or []
        if decision == "tool_expected":
            item["score"] += 4.0 if calls and not issues else -3.0
        elif decision in {"no_tool_expected", "ask_user_expected"}:
            item["score"] += 2.0 if not calls and not issues else -1.0
        if multi_call_expected and calls and not issues:
            item["score"] += 4.0 if len(calls) > 1 else -2.0


def load_candidate_file(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("id") or row.get("question_id") or row.get("idx") or "")
            if rid:
                rows[rid] = row
    return rows


def parse_candidate_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Candidate spec must be source:path, got {spec!r}")
    source, path = spec.split(":", 1)
    return source.strip(), path.strip()


def tool_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(tool.get("name")): tool for tool in tools if tool.get("name")}


def required_args(tool: dict[str, Any]) -> set[str]:
    params = tool.get("parameters") or {}
    required = params.get("required") if isinstance(params, dict) else []
    return {str(item) for item in required or []}


def property_names(tool: dict[str, Any]) -> set[str]:
    params = tool.get("parameters") or {}
    props = params.get("properties") if isinstance(params, dict) else {}
    return {str(key) for key in (props or {}).keys()}


def call_issues(calls: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    by_name = tool_map(tools)
    for index, call in enumerate(calls):
        name = str(call.get("name") or "")
        args = maybe_json(call.get("arguments") or {})
        if not name:
            issues.append(f"call {index}: missing name")
            continue
        tool = by_name.get(name)
        if tool is None:
            issues.append(f"call {index}: unknown tool {name}")
            continue
        if not isinstance(args, dict):
            issues.append(f"call {index}: arguments not object")
            continue
        required = required_args(tool)
        missing = sorted(required - set(args.keys()))
        if missing:
            issues.append(f"call {index}: missing required {missing}")
        props = property_names(tool)
        if props:
            extra = sorted(set(args.keys()) - props)
            if extra:
                issues.append(f"call {index}: unknown args {extra}")
    return issues


def normalized_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    base = row.get("baseline") if isinstance(row.get("baseline"), dict) else row
    calls = base.get("normalized_calls") or base.get("calls") or []
    if not isinstance(calls, list):
        calls = []
    clean = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name") or call.get("function_name")
        args = maybe_json(call.get("arguments") or call.get("args") or call.get("parameters") or {})
        if name:
            clean.append({"name": str(name), "arguments": args if isinstance(args, dict) else {"value": args}})
    return dedupe_tool_calls(clean)


def quoted_schema_values(schema: dict[str, Any]) -> list[str]:
    values: list[str] = []
    enum_values = schema.get("enum") if isinstance(schema.get("enum"), list) else []
    values.extend(str(item) for item in enum_values if item is not None)
    if schema.get("const") is not None:
        values.append(str(schema["const"]))
    description = str(schema.get("description") or "")
    for single, double in re.findall(r"'([^']+)'|\"([^\"]+)\"", description):
        value = single or double
        if value:
            values.append(value)
    seen: set[str] = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def semantic_repair_calls(
    calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]]:
    """Fill optional string args when the prompt explicitly states a schema literal."""
    prompt_l = prompt.lower()
    by_name = tool_map(tools)
    repaired: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name") or "")
        args = maybe_json(call.get("arguments") or {})
        if not isinstance(args, dict):
            repaired.append(call)
            continue
        tool = by_name.get(name)
        if tool is None:
            repaired.append(call)
            continue
        props = (tool.get("parameters") or {}).get("properties") or {}
        if not isinstance(props, dict):
            repaired.append(call)
            continue
        new_args = dict(args)
        for arg_name, schema in props.items():
            if arg_name in new_args or not isinstance(schema, dict):
                continue
            if str(schema.get("type") or "").lower() != "string":
                continue
            for value in quoted_schema_values(schema):
                if re.search(rf"(?<![a-z0-9_]){re.escape(value.lower())}(?![a-z0-9_])", prompt_l):
                    new_args[str(arg_name)] = value
                    break
        repaired.append({"name": name, "arguments": new_args})
    return dedupe_tool_calls(repaired)


def source_bias(source: str) -> float:
    table = {
        "xlam": 8.0,
        "llama_xlam": 8.0,
        "toolace": 6.0,
        "gptoss_apigen": 5.0,
        "apigen": 5.0,
        "taskbench": 3.0,
        "gptoss_taskbench": 3.0,
    }
    return table.get(source, 0.0)


def static_score(candidate: dict[str, Any], tools: list[dict[str, Any]]) -> float:
    calls = candidate["calls"]
    issues = candidate["issues"]
    score = source_bias(candidate["source"])
    score += 20.0 if calls else 0.0
    score -= 50.0 * len(issues)
    score -= 2.0 * max(0, len(calls) - 1)
    return score


def relevance_adjusted_score(candidate: dict[str, Any], tools: list[dict[str, Any]], prompt: str) -> float:
    score = static_score(candidate, tools)
    if candidate.get("calls") and not candidate.get("issues"):
        relevance = call_relevance(candidate, tools, prompt)
        candidate["relevance_score"] = round(relevance, 3)
        score += 12.0 * relevance
        if relevance < 0.12:
            score -= 22.0
        elif relevance < 0.25:
            score -= 10.0
    elif not candidate.get("calls") and not candidate.get("issues"):
        candidate["relevance_score"] = 0.0
        score += 2.0
    return score


def resolve_dtype(value: str, torch: Any) -> Any:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


class Verifier:
    def __init__(self, model_name: str, precision: str, device_map: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolve_dtype(precision, torch),
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def choose(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]] | None, str]:
        compact = [
            {
                "source": item["source"],
                "calls": item["calls"],
                "schema_issues": item["issues"],
            }
            for item in candidates
        ]
        allowed_sources = [item["source"] for item in candidates]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON-only verifier for completed tool-call candidates. "
                    "Do not solve the user request. Do not call tools. Do not list tool schemas. "
                    "First decide whether any available tool is semantically appropriate for the request. "
                    "A schema-valid function call can still be wrong if the tool does not match the user intent. "
                    "If no tool is appropriate, choose a candidate whose calls list is empty. "
                    "Audit the candidates and choose exactly one source from the allowed_sources list, "
                    "or null if none is usable. Reject candidates with schema_issues unless every candidate "
                    "is invalid. If multiple valid candidates are equivalent, choose no-call over a weakly "
                    "grounded call, otherwise use this priority order: xlam, toolace, gptoss_apigen, taskbench. "
                    "Return exactly one compact JSON object and no prose, markdown, or code fence. "
                    "The first character of your answer must be { and the last character must be }. "
                    "Required JSON shape: "
                    "{\"chosen_source\":\"<allowed source or null>\",\"repair_calls\":null,\"reason\":\"short\"}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_request": prompt,
                        "allowed_sources": allowed_sources,
                        "tool_schemas_for_validation_only": openai_tools(tools),
                        "candidate_call_options": compact,
                        "answer_format": {
                            "chosen_source": allowed_sources + [None],
                            "repair_calls": None,
                            "reason": "short string inside JSON only",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        json_prefix = '{"chosen_source":'
        inputs = self.tokenizer(
            rendered_prompt + json_prefix,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        start = time.time()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = json_prefix + self.tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()
        parsed = first_json(text)
        chosen = parsed.get("chosen_source") if isinstance(parsed, dict) else None
        repair = parsed.get("repair_calls") if isinstance(parsed, dict) else None
        chosen = canonical_source(chosen, allowed_sources) or infer_chosen_source(text, allowed_sources)
        repair_calls = normalize_repair_calls(repair) if repair else None
        return chosen, repair_calls, text


def first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    return {}


def source_aliases(source: str) -> set[str]:
    base = source.strip().lower()
    aliases = {
        base,
        base.replace("_", "-"),
        base.replace("_", " "),
        base.replace("gptoss", "gpt-oss"),
    }
    if "xlam" in base:
        aliases.update({"xlam", "llama xlam", "llama-xlam", "llama_xlam", "salesforce/llama-xlam-2-8b-fc-r"})
    if "toolace" in base:
        aliases.update({"toolace", "tool ace", "team-ace/toolace-8b"})
    if "apigen" in base:
        aliases.update({"apigen", "api gen", "gpt-oss apigen", "gpt-oss-api-gen"})
    if "taskbench" in base:
        aliases.update({"taskbench", "task bench", "gpt-oss taskbench"})
    return {alias for alias in aliases if alias}


def source_alias_map(sources: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source in sources:
        for alias in source_aliases(source):
            aliases[alias] = source
    return aliases


def canonical_source(value: Any, sources: list[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("`\"'").lower()
    if not text or text == "null":
        return None
    aliases = source_alias_map(sources)
    if text in aliases:
        return aliases[text]
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    for alias, source in aliases.items():
        alias_norm = re.sub(r"[^a-z0-9]+", " ", alias).strip()
        if normalized == alias_norm:
            return source
    exact_mentions = {source for alias, source in aliases.items() if re.search(rf"\b{re.escape(alias)}\b", text)}
    return next(iter(exact_mentions)) if len(exact_mentions) == 1 else None


def infer_chosen_source(text: str, sources: list[str]) -> str | None:
    if not text.strip():
        return None
    aliases = source_alias_map(sources)
    alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    if not alias_pattern:
        return None
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    patterns = [
        rf"(?:chosen|selected)[_\s-]*source\s*[:=]\s*[\"'`]?({alias_pattern})",
        rf"(?:choose|chosen|select|selected|pick|picked|recommend|recommended)\s+(?:the\s+)?(?:candidate|source|model|option)?\s*[:\-]?\s*[\"'`]?({alias_pattern})",
        rf"(?:best|correct|preferred)\s+(?:candidate|source|model|option)\s+(?:is|:)\s*[\"'`]?({alias_pattern})",
        rf"({alias_pattern})\s+(?:is|looks|appears)\s+(?:the\s+)?(?:best|correct|preferred|strongest)\s+(?:candidate|source|model|option)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned):
            alias = match.group(1).strip().strip("`\"'")
            source = aliases.get(alias)
            if source:
                matches.append(source)
        if matches:
            unique = sorted(set(matches))
            return unique[0] if len(unique) == 1 else None
    return None


def normalize_repair_calls(value: Any) -> list[dict[str, Any]]:
    value = maybe_json(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    calls = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function_name")
        args = maybe_json(item.get("arguments") or item.get("args") or item.get("parameters") or {})
        if name:
            calls.append({"name": str(name), "arguments": args if isinstance(args, dict) else {"value": args}})
    return dedupe_tool_calls(calls)


def select_candidate(
    candidates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt: str,
    verifier: Verifier | None,
    capability_hint: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for item in candidates:
        item["score"] = relevance_adjusted_score(item, tools, prompt)
    apply_capability_hint_scores(candidates, capability_hint)
    verifier_raw = ""
    chosen_source = None
    capability_gate = capability_empty_gate(candidates, tools, prompt, capability_hint)
    empty_gate = capability_gate or (
        best_empty_candidate(candidates)
        if should_abstain(candidates, tools, prompt, capability_hint)
        else None
    )
    gate_label = "capability_no_call_gate" if capability_gate else "no_call_gate"

    def diag(fields: dict[str, Any]) -> dict[str, Any]:
        if capability_hint is not None:
            fields["capability_planner"] = capability_hint
        if capability_gate:
            fields["capability_no_call_gate"] = True
        return fields

    if verifier is not None and candidates:
        try:
            chosen_source, repair_calls, verifier_raw = verifier.choose(prompt, tools, candidates)
            if repair_calls and not call_issues(repair_calls, tools):
                repair_candidate = {"source": "verifier_repair", "calls": repair_calls, "issues": []}
                repair_relevance = call_relevance(repair_candidate, tools, prompt)
                repair_candidate["relevance_score"] = round(repair_relevance, 3)
                if empty_gate and (
                    repair_relevance < 0.62
                    or not supported_by_consensus(repair_candidate, candidates, min_count=2)
                ):
                    return [], diag({
                        "selected_source": f"{gate_label}:{empty_gate.get('source')}",
                        "verifier_raw": verifier_raw,
                        "verifier_chosen_source": chosen_source,
                        "no_call_gate": True,
                        "blocked_repair_relevance": round(repair_relevance, 3),
                    })
                return repair_calls, diag({
                    "selected_source": "verifier_repair",
                    "verifier_raw": verifier_raw,
                    "verifier_chosen_source": chosen_source,
                })
        except Exception as exc:  # noqa: BLE001
            verifier_raw = f"{type(exc).__name__}: {exc}"
    if empty_gate:
        if not chosen_source:
            return [], diag({
                "selected_source": f"{gate_label}:{empty_gate.get('source')}",
                "verifier_raw": verifier_raw,
                "verifier_chosen_source": chosen_source,
                "no_call_gate": True,
            })
        chosen_item = next((item for item in candidates if item.get("source") == chosen_source), None)
        chosen_relevance = call_relevance(chosen_item, tools, prompt) if chosen_item else 0.0
        if chosen_item and chosen_item.get("calls") and (
            chosen_relevance < 0.62
            or not supported_by_consensus(chosen_item, candidates, min_count=2)
        ):
            return [], diag({
                "selected_source": f"{gate_label}:{empty_gate.get('source')}",
                "verifier_raw": verifier_raw,
                "verifier_chosen_source": chosen_source,
                "no_call_gate": True,
                "blocked_chosen_relevance": round(chosen_relevance, 3),
            })
    if chosen_source:
        for item in candidates:
            if item["source"] == chosen_source and not item["issues"]:
                return item["calls"], diag({
                    "selected_source": item["source"],
                    "verifier_raw": verifier_raw,
                    "verifier_chosen_source": chosen_source,
                })
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    chosen = ranked[0] if ranked else {"source": "empty", "calls": []}
    return chosen.get("calls", []), diag({
        "selected_source": chosen.get("source"),
        "verifier_raw": verifier_raw,
        "verifier_chosen_source": chosen_source,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Original BFCL question JSON/JSONL")
    parser.add_argument("--output", required=True, help="Selected prediction JSONL")
    parser.add_argument("--candidate", action="append", default=[], help="source:path JSONL")
    parser.add_argument("--function-doc-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--verifier-model", default="", help="Optional HF verifier, e.g. Salesforce/Llama-xLAM-2-8b-fc-r")
    parser.add_argument("--verifier-max-new-tokens", type=int, default=256)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--capability-planner",
        choices=("off", "auto", "on"),
        default="auto",
        help=(
            "Attach the rules-first capability planner. "
            "auto records diagnostics and skips cleanly if unavailable; on raises and lets hints influence selection."
        ),
    )
    args = parser.parse_args()

    records = load_records(args.input)
    indexed_records = list(enumerate(records))
    if args.offset:
        indexed_records = indexed_records[args.offset :]
    if args.limit:
        indexed_records = indexed_records[: args.limit]
    candidate_sets = [(source, load_candidate_file(path)) for source, path in map(parse_candidate_spec, args.candidate)]
    verifier = (
        Verifier(args.verifier_model, args.precision, args.device_map, args.verifier_max_new_tokens)
        if args.verifier_model
        else None
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, record in indexed_records:
            rid = record_id(record, index)
            prompt = extract_prompt(record)
            tools = normalize_tools(record, args.function_doc_dir or None)
            candidates = []
            for source, rows in candidate_sets:
                row = rows.get(rid)
                calls = normalized_calls(row or {})
                calls = semantic_repair_calls(calls, tools, prompt)
                candidates.append(
                    {
                        "source": source,
                        "calls": calls,
                        "issues": call_issues(calls, tools),
                        "error": ((row or {}).get("baseline") or {}).get("error", ""),
                    }
                )
            capability_hint = build_capability_selection_hint(
                prompt,
                tools,
                args.capability_planner,
            )
            selected, diag = select_candidate(candidates, tools, prompt, verifier, capability_hint)
            handle.write(
                json.dumps(
                    {
                        "id": rid,
                        "baseline": {
                            "normalized_calls": selected,
                            "raw_text": "",
                            "latency_ms": 0,
                            "generated_tokens": 0,
                            "model": "ensemble",
                            "adapter": "",
                            "error": "",
                        },
                        "ensemble": {
                            **diag,
                            "candidates": candidates,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
