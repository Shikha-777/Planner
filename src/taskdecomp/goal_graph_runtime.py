from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal


NodeKind = Literal[
    "resolve",
    "retrieve",
    "search",
    "rank",
    "decide",
    "mutate",
    "communicate",
    "ask_user",
    "respond",
]
RiskLevel = Literal["read_only", "external_side_effect", "destructive_side_effect"]
InputSource = Literal[
    "query",
    "context",
    "tool_output",
    "policy_default",
    "user_confirmation",
    "node_output",
    "unknown",
]

NODE_KINDS: set[str] = {
    "resolve",
    "retrieve",
    "search",
    "rank",
    "decide",
    "mutate",
    "communicate",
    "ask_user",
    "respond",
}
READ_ONLY_KINDS = {"resolve", "retrieve", "search", "rank", "decide", "respond"}
SIDE_EFFECT_KINDS = {"mutate", "communicate"}
READ_ONLY_RISK = "read_only"
SIDE_EFFECT_RISKS = {"external_side_effect", "destructive_side_effect"}
GROUNDED_LITERAL_SOURCES = {"query", "context", "tool_output", "policy_default", "user_confirmation"}
REFERENCE_PATTERN = re.compile(r"^\$([A-Za-z][\w-]*)\.([A-Za-z_][\w-]*)$")


@dataclass(frozen=True)
class CapabilityInput:
    name: str
    type: str = "any"
    required: bool = False
    must_come_from: tuple[str, ...] = ()
    default: Any | None = None
    allowed_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Capability:
    tool_name: str
    capability: str
    kind: str
    required_inputs: dict[str, CapabilityInput] = field(default_factory=dict)
    optional_inputs: dict[str, CapabilityInput] = field(default_factory=dict)
    risk: str = READ_ONLY_RISK
    effects: tuple[dict[str, Any], ...] = ()
    requires_unique_target: bool = False
    requires_confirmation: bool = False
    supports_batch: bool = False
    raw_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, CapabilityInput]:
        return {**self.required_inputs, **self.optional_inputs}


@dataclass
class GoalInput:
    value: Any = None
    source: str = "unknown"
    evidence: str = ""
    status: str = "resolved"


@dataclass
class GoalNode:
    id: str
    kind: str
    capability: str
    description: str = ""
    inputs: dict[str, GoalInput] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    must_be_unique: bool = False
    risk: str = READ_ONLY_RISK
    authorized: bool = False
    policy_evidence: list[str] = field(default_factory=list)
    expected_effect: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalGraph:
    goal: str
    nodes: list[GoalNode] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: str = ""
    severity: str = "error"


@dataclass
class VerificationResult:
    ok: bool
    diagnostics: list[Diagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class CompiledCall:
    id: str
    graph_node_id: str
    tool_name: str
    arguments: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass
class RuntimeOutput:
    graph: GoalGraph
    calls: list[CompiledCall]
    verification: VerificationResult


class GoalGraphRuntime:
    """Small facade for the goal-graph verifier/compiler runtime."""

    def __init__(self, tools: list[dict[str, Any]]):
        self.registry = build_capability_registry(tools)

    def verify(
        self,
        graph: GoalGraph | dict[str, Any],
        user_request: str,
        *,
        context: dict[str, Any] | None = None,
        observations: dict[str, dict[str, Any]] | None = None,
        allow_side_effects: bool = False,
    ) -> VerificationResult:
        parsed = parse_goal_graph(graph) if isinstance(graph, dict) else graph
        return verify_goal_graph(
            parsed,
            self.registry,
            user_request,
            context=context,
            observations=observations,
            allow_side_effects=allow_side_effects,
        )

    def compile(
        self,
        graph: GoalGraph | dict[str, Any],
        user_request: str,
        *,
        context: dict[str, Any] | None = None,
        observations: dict[str, dict[str, Any]] | None = None,
        allow_side_effects: bool = False,
    ) -> RuntimeOutput:
        parsed = parse_goal_graph(graph) if isinstance(graph, dict) else graph
        calls, verification = compile_goal_graph(
            parsed,
            self.registry,
            user_request,
            context=context,
            observations=observations,
            allow_side_effects=allow_side_effects,
        )
        return RuntimeOutput(graph=parsed, calls=calls, verification=verification)


def build_capability_registry(tools: list[dict[str, Any]]) -> dict[str, Capability]:
    """Convert raw tool schemas into capability objects without reading the user query."""
    registry: dict[str, Capability] = {}
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        tool = _normalize_tool_schema(raw_tool)
        name = tool["name"]
        if not name:
            continue
        capability_name = str(raw_tool.get("capability") or _default_capability_name(name))
        properties = tool["properties"]
        required_names = set(tool["required"])
        required_inputs: dict[str, CapabilityInput] = {}
        optional_inputs: dict[str, CapabilityInput] = {}
        for input_name, spec in properties.items():
            cap_input = CapabilityInput(
                name=input_name,
                type=_property_type(spec),
                required=input_name in required_names,
                must_come_from=tuple(_allowed_sources_for_input(input_name, spec)),
                default=_schema_default_value(spec),
                allowed_values=tuple(_schema_allowed_values(spec)),
            )
            if input_name in required_names:
                required_inputs[input_name] = cap_input
            else:
                optional_inputs[input_name] = cap_input
        kind = str(raw_tool.get("kind") or _infer_capability_kind(name, tool["description"]))
        risk = str(raw_tool.get("risk") or _infer_risk(kind, name, tool["description"]))
        capability = Capability(
            tool_name=name,
            capability=capability_name,
            kind=kind,
            required_inputs=required_inputs,
            optional_inputs=optional_inputs,
            risk=risk,
            effects=tuple(raw_tool.get("effects") or ()),
            requires_unique_target=bool(
                raw_tool.get("requires_unique_target", risk in SIDE_EFFECT_RISKS)
            ),
            requires_confirmation=bool(
                raw_tool.get("requires_confirmation", risk in SIDE_EFFECT_RISKS)
            ),
            supports_batch=_supports_batch(properties),
            raw_schema=tool,
        )
        registry[capability.capability] = capability
    return registry


def build_goal_graph_planner_messages(
    user_request: str,
    registry: dict[str, Capability],
    *,
    policies: list[str] | None = None,
    failure_lessons: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build planner messages for an LLM that proposes a graph, not tool calls."""
    policy_text = "\n".join(f"- {item}" for item in policies or []) or "- none"
    lessons = _dedupe_strings(default_goal_graph_failure_lessons() + (failure_lessons or []))
    lesson_text = "\n".join(f"- {item}" for item in lessons) or "- none"
    capabilities = "\n".join(_capability_prompt_line(item) for item in registry.values())
    return [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. You are a semantic goal-graph planner, "
                "not a direct tool caller. Produce a grounded goal graph whose nodes use "
                "only the provided capabilities. Every concrete input must include a "
                "source and query evidence when source is query. Use node-output references "
                "like $n1.event_id for values produced by earlier nodes. Do not invent IDs, "
                "emails, order numbers, event IDs, account IDs, or tool outputs. Mark "
                "underspecified values with status 'underspecified' and add an ask_user node "
                "when clarification is needed. Side-effect nodes must include risk, "
                "must_be_unique, authorized, and expected_effect when applicable. If a "
                "capability input has an allowed_values list, use one of those values or "
                "mark the input underspecified. If a capability has a default, either omit "
                "the input or set source to policy_default; do not invent a query span for "
                "defaults. Output "
                "schema: {\"goal\":\"...\",\"nodes\":[{\"id\":\"n1\",\"kind\":\"resolve | "
                "retrieve | search | rank | decide | mutate | communicate | ask_user | "
                "respond\",\"capability\":\"capability id or empty for ask_user/respond\","
                "\"description\":\"...\",\"inputs\":{\"slot\":{\"value\":...,\"source\":"
                "\"query | context | tool_output | policy_default | user_confirmation | "
                "node_output\",\"evidence\":\"...\",\"status\":\"resolved\"}},\"outputs\":"
                "[\"...\"],\"depends_on\":[\"...\"],\"must_be_unique\":false,\"risk\":"
                "\"read_only | external_side_effect | destructive_side_effect\","
                "\"authorized\":false,\"policy_evidence\":[],\"expected_effect\":{}}],"
                "\"clarification_needed\":false,\"clarification_reasons\":[]}."
            ),
        },
        {
            "role": "user",
            "content": (
                "User request:\n"
                f"{user_request}\n\n"
                "Capabilities:\n"
                f"{capabilities or '- none'}\n\n"
                "Policies:\n"
                f"{policy_text}\n\n"
                "Failure lessons:\n"
                f"{lesson_text}"
            ),
        },
    ]


def default_goal_graph_failure_lessons() -> list[str]:
    """General lessons learned from the frozen capability/semantic-frame pipeline."""
    return [
        "Every concrete value from the query must become a graph input with evidence, or remain unresolved and trigger ask_user.",
        "Do not treat top-N/result-count phrases as repeated calls; model them as a limit/result_count input on one rank/search/retrieve node when the capability supports it.",
        "Use batch/list inputs when the capability supports arrays; use repeated nodes only for scalar single-entity capabilities.",
        "Do not infer missing IDs, emails, event IDs, order numbers, account numbers, or generated identifiers from natural language unless they are explicitly present or produced by a prior node.",
        "Schema defaults are policy/schema-grounded values, not query-grounded values.",
        "When a schema description lists allowed values such as either 'summary' or 'full', choose exactly one listed value.",
        "If a query provides semantic evidence for a required slot, the graph should bind it explicitly rather than leaving a vague missing input.",
    ]


def parse_goal_graph(data: dict[str, Any]) -> GoalGraph:
    """Normalize LLM JSON into a GoalGraph dataclass tree."""
    nodes = []
    for index, item in enumerate(_as_list(data.get("nodes"))):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("id") or f"n{index + 1}")
        nodes.append(
            GoalNode(
                id=node_id,
                kind=str(item.get("kind") or ""),
                capability=str(item.get("capability") or ""),
                description=str(item.get("description") or ""),
                inputs=_parse_goal_inputs(item.get("inputs")),
                outputs=[str(value) for value in _as_list(item.get("outputs")) if value],
                depends_on=[str(value) for value in _as_list(item.get("depends_on")) if value],
                must_be_unique=bool(item.get("must_be_unique", False)),
                risk=str(item.get("risk") or READ_ONLY_RISK),
                authorized=bool(item.get("authorized", item.get("confirmed", False))),
                policy_evidence=[str(value) for value in _as_list(item.get("policy_evidence")) if value],
                expected_effect=item.get("expected_effect") if isinstance(item.get("expected_effect"), dict) else {},
            )
        )
    return GoalGraph(
        goal=str(data.get("goal") or ""),
        nodes=nodes,
        clarification_needed=bool(data.get("clarification_needed", False)),
        clarification_reasons=[
            str(value) for value in _as_list(data.get("clarification_reasons")) if value
        ],
    )


def verify_goal_graph(
    graph: GoalGraph,
    registry: dict[str, Capability],
    user_request: str,
    *,
    context: dict[str, Any] | None = None,
    observations: dict[str, dict[str, Any]] | None = None,
    allow_side_effects: bool = False,
) -> VerificationResult:
    diagnostics: list[Diagnostic] = []
    context = context or {}
    observations = observations or {}
    node_by_id = {node.id: node for node in graph.nodes}

    _check_unique_node_ids(graph.nodes, diagnostics)
    _check_node_kinds(graph.nodes, diagnostics)
    _check_dependencies(graph.nodes, node_by_id, diagnostics)
    _check_acyclic(graph.nodes, diagnostics)

    for node_index, node in enumerate(graph.nodes):
        path = f"nodes[{node_index}]"
        if node.kind in {"ask_user", "respond"} and not node.capability:
            continue
        if _is_literal_resolve_node(node):
            _check_literal_resolve_node(
                node,
                node_by_id,
                user_request,
                context,
                observations,
                diagnostics,
                path,
            )
            continue
        capability = registry.get(node.capability)
        if capability is None:
            diagnostics.append(
                Diagnostic(
                    "unknown_capability",
                    f"Node {node.id} references unknown capability {node.capability!r}.",
                    path=f"{path}.capability",
                )
            )
            continue
        _check_capability_kind(node, capability, diagnostics, path)
        _check_node_inputs(node, capability, diagnostics, path, user_request)
        _check_input_grounding(
            node,
            capability,
            node_by_id,
            user_request,
            context,
            observations,
            diagnostics,
            path,
        )
        _check_side_effect_gate(node, capability, allow_side_effects, diagnostics, path)

    return VerificationResult(ok=not diagnostics, diagnostics=diagnostics)


def compile_goal_graph(
    graph: GoalGraph,
    registry: dict[str, Capability],
    user_request: str,
    *,
    context: dict[str, Any] | None = None,
    observations: dict[str, dict[str, Any]] | None = None,
    allow_side_effects: bool = False,
) -> tuple[list[CompiledCall], VerificationResult]:
    """Compile a verified graph into concrete tool calls."""
    verification = verify_goal_graph(
        graph,
        registry,
        user_request,
        context=context,
        observations=observations,
        allow_side_effects=allow_side_effects,
    )
    if not verification.ok:
        return [], verification

    calls: list[CompiledCall] = []
    call_id_by_node: dict[str, str] = {}
    context = context or {}
    working_observations = {key: dict(value) for key, value in (observations or {}).items()}
    for node in _topological_nodes(graph.nodes):
        if node.kind in {"ask_user", "respond", "decide"}:
            continue
        if _is_literal_resolve_node(node):
            working_observations[node.id] = _literal_resolve_outputs(
                node,
                context,
                working_observations,
            )
            continue
        capability = registry[node.capability]
        if capability.risk in SIDE_EFFECT_RISKS and not allow_side_effects:
            continue
        arguments = {
            name: _resolve_input_value(goal_input, context, working_observations)
            for name, goal_input in node.inputs.items()
            if name in capability.inputs and goal_input.status != "defaulted"
        }
        for name, cap_input in capability.inputs.items():
            if name not in arguments and cap_input.default is not None:
                arguments[name] = cap_input.default
        call_depends_on = tuple(call_id_by_node[dep] for dep in node.depends_on if dep in call_id_by_node)
        call = CompiledCall(
            id=f"call_{len(calls) + 1}",
            graph_node_id=node.id,
            tool_name=capability.tool_name,
            arguments=arguments,
            depends_on=call_depends_on,
        )
        calls.append(call)
        call_id_by_node[node.id] = call.id
    return calls, verification


def compiled_calls_to_dicts(calls: list[CompiledCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "graph_node_id": call.graph_node_id,
            "tool_name": call.tool_name,
            "arguments": call.arguments,
            "depends_on": list(call.depends_on),
        }
        for call in calls
    ]


def _normalize_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(tool.get("name") or function.get("name") or "")
    description = str(tool.get("description") or function.get("description") or "")
    parameters = tool.get("parameters") or function.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    argument_schema = tool.get("arguments") if isinstance(tool.get("arguments"), dict) else None
    if argument_schema is None and isinstance(tool.get("input_parameters"), dict):
        argument_schema = tool["input_parameters"]
    if argument_schema is not None and "properties" not in parameters:
        parameters = {"type": "object", "properties": argument_schema, "required": list(argument_schema)}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "properties": properties,
        "required": [str(item) for item in required],
    }


def _capability_prompt_line(capability: Capability) -> str:
    required = ", ".join(_capability_input_prompt(name, item) for name, item in capability.required_inputs.items()) or "none"
    optional = ", ".join(_capability_input_prompt(name, item) for name, item in capability.optional_inputs.items()) or "none"
    outputs = []
    if capability.supports_batch:
        outputs.append("supports_batch")
    if capability.requires_unique_target:
        outputs.append("requires_unique_target")
    if capability.requires_confirmation:
        outputs.append("requires_confirmation")
    flags = f"; flags={', '.join(outputs)}" if outputs else ""
    return (
        f"- {capability.capability}: tool={capability.tool_name}; kind={capability.kind}; "
        f"risk={capability.risk}; required={required}; optional={optional}{flags}"
    )


def _capability_input_prompt(name: str, cap_input: CapabilityInput) -> str:
    pieces = [name]
    if cap_input.default is not None:
        pieces.append(f"default={cap_input.default!r}")
    if cap_input.allowed_values:
        allowed = "|".join(str(value) for value in cap_input.allowed_values)
        pieces.append(f"allowed={allowed}")
    return f"{pieces[0]}({', '.join(pieces[1:])})" if len(pieces) > 1 else pieces[0]


def _default_capability_name(tool_name: str) -> str:
    return tool_name.strip()


def _infer_capability_kind(name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    if re.search(r"\b(?:send|email|notify|message|communicat)", text):
        return "communicate"
    if re.search(r"\b(?:cancel|delete|remove|update|create|book|reserve|purchase|refund|mutate)", text):
        return "mutate"
    if re.search(r"\b(?:rank|sort|top|highest|lowest|best)", text):
        return "rank"
    if re.search(r"\b(?:search|find|list|locate)", text):
        return "search"
    if re.search(r"\b(?:resolve|lookup|identify)", text):
        return "resolve"
    return "retrieve"


def _infer_risk(kind: str, name: str, description: str) -> str:
    text = f"{name} {description}".lower()
    if kind == "communicate":
        return "external_side_effect"
    if kind == "mutate":
        return "destructive_side_effect"
    if re.search(r"\b(?:send|cancel|delete|remove|update|book|reserve|purchase|refund)\b", text):
        return "destructive_side_effect"
    return READ_ONLY_RISK


def _property_type(spec: dict[str, Any]) -> str:
    if not isinstance(spec, dict):
        return "any"
    value = str(spec.get("type") or "any").lower()
    aliases = {"str": "string", "int": "integer", "float": "number", "dict": "object"}
    return aliases.get(value, value)


def _schema_default_value(spec: dict[str, Any]) -> Any | None:
    if not isinstance(spec, dict):
        return None
    if "default" in spec:
        return spec.get("default")
    description = str(spec.get("description") or "")
    match = re.search(
        r"\bdefault(?:\s+value)?\s+(?:is|=|:)?\s*['\"]?([^'\".;,)]+)",
        description,
        re.I,
    )
    if not match:
        return None
    raw = re.sub(r"^(?:to|as)\s+", "", match.group(1).strip(), flags=re.I).strip()
    typ = _property_type(spec)
    lowered = raw.lower()
    if typ == "boolean":
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        return None
    if typ in {"integer", "number"}:
        number = _number_from_text(raw)
        if number is None:
            return None
        if typ == "integer" and isinstance(number, float) and number.is_integer():
            return int(number)
        return number
    if typ == "array":
        return [] if lowered in {"empty list", "none", "null"} else None
    if typ in {"string", "any", "object"}:
        value = raw.strip("'\" ")
        return value if value else None
    return None


def _schema_allowed_values(spec: dict[str, Any]) -> list[Any]:
    if not isinstance(spec, dict):
        return []
    enum = spec.get("enum")
    if not isinstance(enum, list) and isinstance(spec.get("items"), dict):
        enum = spec["items"].get("enum")
    if isinstance(enum, list):
        return [item for item in enum if item not in (None, "")]
    return _listed_values_from_description(str(spec.get("description") or ""))


def _listed_values_from_description(description: str) -> list[str]:
    match = re.search(r"allowed values?\s*:?\s*(.+)", description, re.I)
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
        item.strip().strip("'\"")
        for item in re.split(r",|\bor\b", text)
        if item.strip().strip("'\"") and len(item.strip()) <= 40
    ]


def _quoted_strings(text: str) -> list[str]:
    values = []
    pattern = r"(?<![A-Za-z0-9])'(.+?)'(?=[\s,.;:!?)]|$)|(?<![A-Za-z0-9])\"([^\"]+)\"(?![A-Za-z0-9])"
    for left, right in re.findall(pattern, text):
        value = left or right
        if value:
            values.append(value)
    return values


def _number_from_text(text: str) -> int | float | None:
    text = text.strip().replace(",", "")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return None


def _value_in_allowed_values(value: Any, allowed_values: tuple[Any, ...]) -> bool:
    if isinstance(value, list):
        return all(_value_in_allowed_values(item, allowed_values) for item in value)
    if value in allowed_values:
        return True
    normalized = str(value).strip().lower()
    return any(str(item).strip().lower() == normalized for item in allowed_values)


def _value_matches_default(value: Any, default: Any | None) -> bool:
    if default is None:
        return False
    return str(value).strip().lower() == str(default).strip().lower()


def _is_schema_defaulted_input(goal_input: GoalInput, cap_input: CapabilityInput) -> bool:
    return (
        goal_input.source == "policy_default"
        and goal_input.status == "defaulted"
        and _value_matches_default(goal_input.value, cap_input.default)
    )


def _allowed_sources_for_input(name: str, spec: dict[str, Any]) -> list[str]:
    if not isinstance(spec, dict):
        spec = {}
    explicit = spec.get("must_come_from")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if value]
    text = f"{name} {spec.get('description') or ''}".lower()
    if _looks_like_generated_identifier(text):
        return ["node_output", "tool_output", "context", "user_confirmation", "query"]
    return ["query", "context", "tool_output", "policy_default", "user_confirmation", "node_output"]


def _supports_batch(properties: dict[str, Any]) -> bool:
    return any(_property_type(spec) == "array" for spec in properties.values())


def _parse_goal_inputs(value: Any) -> dict[str, GoalInput]:
    if not isinstance(value, dict):
        return {}
    inputs: dict[str, GoalInput] = {}
    for name, raw in value.items():
        if isinstance(raw, dict):
            inputs[str(name)] = GoalInput(
                value=raw.get("value"),
                source=str(raw.get("source") or "unknown"),
                evidence=str(raw.get("evidence") or raw.get("evidence_span") or ""),
                status=str(raw.get("status") or "resolved"),
            )
        else:
            inputs[str(name)] = GoalInput(value=raw, source="unknown", evidence="", status="resolved")
    return inputs


def _check_unique_node_ids(nodes: list[GoalNode], diagnostics: list[Diagnostic]) -> None:
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        if not node.id:
            diagnostics.append(Diagnostic("missing_node_id", "Every node must have an id.", f"nodes[{index}].id"))
        if node.id in seen:
            diagnostics.append(
                Diagnostic("duplicate_node_id", f"Duplicate node id {node.id!r}.", f"nodes[{index}].id")
            )
        seen.add(node.id)


def _check_node_kinds(nodes: list[GoalNode], diagnostics: list[Diagnostic]) -> None:
    for index, node in enumerate(nodes):
        if node.kind not in NODE_KINDS:
            diagnostics.append(
                Diagnostic("invalid_node_kind", f"Invalid node kind {node.kind!r}.", f"nodes[{index}].kind")
            )


def _check_dependencies(
    nodes: list[GoalNode],
    node_by_id: dict[str, GoalNode],
    diagnostics: list[Diagnostic],
) -> None:
    for index, node in enumerate(nodes):
        for dep in node.depends_on:
            if dep not in node_by_id:
                diagnostics.append(
                    Diagnostic(
                        "unknown_dependency",
                        f"Node {node.id} depends on unknown node {dep!r}.",
                        f"nodes[{index}].depends_on",
                    )
                )


def _check_acyclic(nodes: list[GoalNode], diagnostics: list[Diagnostic]) -> None:
    node_ids = [node.id for node in nodes]
    edges = [(dep, node.id) for node in nodes for dep in node.depends_on]
    if not _is_acyclic(node_ids, edges):
        diagnostics.append(Diagnostic("dependency_cycle", "Goal graph dependencies contain a cycle.", "nodes"))


def _check_capability_kind(
    node: GoalNode,
    capability: Capability,
    diagnostics: list[Diagnostic],
    path: str,
) -> None:
    if node.kind in {"ask_user", "respond"}:
        return
    if capability.kind != node.kind:
        diagnostics.append(
            Diagnostic(
                "kind_capability_mismatch",
                f"Node kind {node.kind!r} does not match capability kind {capability.kind!r}.",
                f"{path}.kind",
            )
        )
    if node.risk != capability.risk:
        diagnostics.append(
            Diagnostic(
                "risk_capability_mismatch",
                f"Node risk {node.risk!r} does not match capability risk {capability.risk!r}.",
                f"{path}.risk",
            )
        )


def _check_node_inputs(
    node: GoalNode,
    capability: Capability,
    diagnostics: list[Diagnostic],
    path: str,
    user_request: str,
) -> None:
    known_inputs = capability.inputs
    required_inputs = _effective_required_inputs_for_node(capability, node)
    for required_name, cap_input in required_inputs.items():
        if required_name not in node.inputs:
            if cap_input.default is not None:
                continue
            diagnostics.append(
                Diagnostic(
                    "missing_required_input",
                    f"Node {node.id} is missing required input {required_name!r}.",
                    f"{path}.inputs.{required_name}",
                )
            )
    for input_name, goal_input in node.inputs.items():
        if input_name not in known_inputs:
            diagnostics.append(
                Diagnostic(
                    "unknown_input",
                    f"Input {input_name!r} is not in capability schema.",
                    f"{path}.inputs.{input_name}",
                )
            )
        if goal_input.status in {"missing", "underspecified", "unresolved"}:
            diagnostics.append(
                Diagnostic(
                    "unresolved_input",
                    f"Input {input_name!r} is {goal_input.status}.",
                    f"{path}.inputs.{input_name}",
                )
            )
        if goal_input.value in (None, "") and goal_input.status not in {"defaulted"}:
            diagnostics.append(
                Diagnostic(
                    "empty_input",
                    f"Input {input_name!r} has no resolved value.",
                    f"{path}.inputs.{input_name}",
                )
            )
        cap_input = known_inputs.get(input_name)
        if cap_input is not None and cap_input.allowed_values and goal_input.value not in (None, ""):
            if not _value_in_allowed_values(goal_input.value, cap_input.allowed_values) and not _value_matches_default(
                goal_input.value,
                cap_input.default,
            ) and not _grounded_query_value_can_extend_enum(goal_input, user_request):
                diagnostics.append(
                    Diagnostic(
                        "value_not_allowed",
                        f"Input {input_name!r} must be one of the allowed schema values.",
                        f"{path}.inputs.{input_name}.value",
                )
            )


def _grounded_query_value_can_extend_enum(goal_input: GoalInput, user_request: str) -> bool:
    return goal_input.source == "query" and _value_grounded_in_text(goal_input.value, user_request)


def _effective_required_inputs_for_node(
    capability: Capability,
    node: GoalNode,
) -> dict[str, CapabilityInput]:
    if not capability.required_inputs:
        return capability.required_inputs
    properties = capability.raw_schema.get("properties") if isinstance(capability.raw_schema, dict) else {}
    if not isinstance(properties, dict):
        return capability.required_inputs
    if not any("only needed for" in str((properties.get(name) or {}).get("description") or "").lower() for name in capability.required_inputs):
        return capability.required_inputs
    status_input = node.inputs.get("status")
    status = str(status_input.value if status_input is not None else "").lower()
    if not status:
        return capability.required_inputs
    second_phase = bool(re.search(r"\b(?:verification|second|reset code|new password)\b", status))
    first_phase = bool(re.search(r"\b(?:forgot password|first)\b", status))
    filtered = dict(capability.required_inputs)
    for name in list(filtered):
        description = str((properties.get(name) or {}).get("description") or "").lower()
        if "only needed for the first call" in description and second_phase:
            filtered.pop(name, None)
        elif "only needed for the second call" in description and first_phase:
            filtered.pop(name, None)
    return filtered


def _check_input_grounding(
    node: GoalNode,
    capability: Capability,
    node_by_id: dict[str, GoalNode],
    user_request: str,
    context: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    diagnostics: list[Diagnostic],
    path: str,
) -> None:
    for input_name, goal_input in node.inputs.items():
        if input_name not in capability.inputs:
            continue
        cap_input = capability.inputs[input_name]
        input_path = f"{path}.inputs.{input_name}"
        if goal_input.source not in GROUNDED_LITERAL_SOURCES | {"node_output", "unknown"}:
            diagnostics.append(
                Diagnostic(
                    "invalid_input_source",
                    f"Input {input_name!r} has invalid source {goal_input.source!r}.",
                    f"{input_path}.source",
                )
            )
        if goal_input.source == "unknown":
            diagnostics.append(
                Diagnostic(
                    "ungrounded_input",
                    f"Input {input_name!r} has unknown source.",
                    f"{input_path}.source",
                )
            )
            continue
        if goal_input.source not in cap_input.must_come_from and not _is_schema_defaulted_input(goal_input, cap_input):
            diagnostics.append(
                Diagnostic(
                    "disallowed_input_source",
                    f"Input {input_name!r} cannot come from {goal_input.source!r}.",
                    f"{input_path}.source",
                )
            )
        if goal_input.source == "query":
            if not goal_input.evidence:
                diagnostics.append(
                    Diagnostic(
                        "missing_evidence",
                        f"Query-grounded input {input_name!r} needs evidence text.",
                        f"{input_path}.evidence",
                    )
                )
            elif not _span_in_text(goal_input.evidence, user_request) and not _value_grounded_in_text(
                goal_input.value,
                user_request,
            ):
                diagnostics.append(
                    Diagnostic(
                        "evidence_not_in_query",
                        f"Evidence for input {input_name!r} does not appear in the query.",
                        f"{input_path}.evidence",
                    )
                )
        elif goal_input.source == "context":
            if not _context_contains_value(context, goal_input.value):
                diagnostics.append(
                    Diagnostic(
                        "context_value_not_available",
                        f"Context does not contain value for input {input_name!r}.",
                        input_path,
                    )
                )
        elif goal_input.source == "node_output":
            _check_node_reference(node, input_name, goal_input, node_by_id, observations, diagnostics, input_path)
        elif goal_input.source == "tool_output":
            if not _observations_contain_value(observations, goal_input.value):
                diagnostics.append(
                    Diagnostic(
                        "tool_output_not_available",
                        f"Tool observations do not contain value for input {input_name!r}.",
                        input_path,
                    )
                )


def _is_literal_resolve_node(node: GoalNode) -> bool:
    return node.kind == "resolve" and node.capability in {"", "resolve", "literal.resolve"}


def _check_literal_resolve_node(
    node: GoalNode,
    node_by_id: dict[str, GoalNode],
    user_request: str,
    context: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    diagnostics: list[Diagnostic],
    path: str,
) -> None:
    if node.risk != READ_ONLY_RISK:
        diagnostics.append(
            Diagnostic(
                "risk_capability_mismatch",
                f"Literal resolve node {node.id} must be read-only.",
                f"{path}.risk",
            )
        )
    if not node.inputs:
        diagnostics.append(
            Diagnostic(
                "missing_resolve_input",
                f"Literal resolve node {node.id} must bind at least one input.",
                f"{path}.inputs",
            )
        )
    for input_name, goal_input in node.inputs.items():
        input_path = f"{path}.inputs.{input_name}"
        if goal_input.status in {"missing", "underspecified", "unresolved"}:
            diagnostics.append(
                Diagnostic(
                    "unresolved_input",
                    f"Input {input_name!r} is {goal_input.status}.",
                    input_path,
                )
            )
        if goal_input.value in (None, "") and goal_input.status not in {"defaulted"}:
            diagnostics.append(
                Diagnostic(
                    "empty_input",
                    f"Input {input_name!r} has no resolved value.",
                    input_path,
                )
            )
        if goal_input.source == "query":
            if not goal_input.evidence:
                diagnostics.append(
                    Diagnostic(
                        "missing_evidence",
                        f"Query-grounded input {input_name!r} needs evidence text.",
                        f"{input_path}.evidence",
                    )
                )
            elif not _span_in_text(goal_input.evidence, user_request) and not _value_grounded_in_text(
                goal_input.value,
                user_request,
            ):
                diagnostics.append(
                    Diagnostic(
                        "evidence_not_in_query",
                        f"Evidence for input {input_name!r} does not appear in the query.",
                        f"{input_path}.evidence",
                    )
                )
        elif goal_input.source == "context":
            if not _context_contains_value(context, goal_input.value):
                diagnostics.append(
                    Diagnostic(
                        "context_value_not_available",
                        f"Context does not contain value for input {input_name!r}.",
                        input_path,
                    )
                )
        elif goal_input.source == "node_output":
            _check_node_reference(node, input_name, goal_input, node_by_id, observations, diagnostics, input_path)
        elif goal_input.source == "tool_output":
            if not _observations_contain_value(observations, goal_input.value):
                diagnostics.append(
                    Diagnostic(
                        "tool_output_not_available",
                        f"Tool observations do not contain value for input {input_name!r}.",
                        input_path,
                    )
                )
        elif goal_input.source not in GROUNDED_LITERAL_SOURCES | {"unknown"}:
            diagnostics.append(
                Diagnostic(
                    "invalid_input_source",
                    f"Input {input_name!r} has invalid source {goal_input.source!r}.",
                    f"{input_path}.source",
                )
            )
        if goal_input.source == "unknown":
            diagnostics.append(
                Diagnostic(
                    "ungrounded_input",
                    f"Input {input_name!r} has unknown source.",
                    f"{input_path}.source",
                )
            )
    if node.outputs:
        for output_name in node.outputs:
            if output_name not in node.inputs:
                diagnostics.append(
                    Diagnostic(
                        "unknown_resolve_output",
                        f"Literal resolve node {node.id} output {output_name!r} must match a resolved input.",
                        f"{path}.outputs",
                    )
                )


def _literal_resolve_outputs(
    node: GoalNode,
    context: dict[str, Any],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_names = set(node.outputs or node.inputs.keys())
    resolved = {}
    for name, goal_input in node.inputs.items():
        if name in output_names:
            resolved[name] = _resolve_input_value(goal_input, context, observations)
    return resolved


def _check_node_reference(
    node: GoalNode,
    input_name: str,
    goal_input: GoalInput,
    node_by_id: dict[str, GoalNode],
    observations: dict[str, dict[str, Any]],
    diagnostics: list[Diagnostic],
    path: str,
) -> None:
    if not isinstance(goal_input.value, str):
        diagnostics.append(
            Diagnostic(
                "invalid_node_reference",
                f"Node-output input {input_name!r} must use a $node.output reference.",
                path,
            )
        )
        return
    match = REFERENCE_PATTERN.fullmatch(goal_input.value.strip())
    if not match:
        diagnostics.append(
            Diagnostic(
                "invalid_node_reference",
                f"Node-output input {input_name!r} must use a $node.output reference.",
                path,
            )
        )
        return
    source_node_id, output_name = match.groups()
    source_node = node_by_id.get(source_node_id)
    if source_node is None:
        diagnostics.append(
            Diagnostic(
                "unknown_reference_node",
                f"Input {input_name!r} references unknown node {source_node_id!r}.",
                path,
            )
        )
        return
    if source_node_id not in node.depends_on:
        diagnostics.append(
            Diagnostic(
                "missing_dependency_for_reference",
                f"Node {node.id} must depend on {source_node_id} to use {goal_input.value}.",
                path,
            )
        )
    if output_name not in source_node.outputs:
        diagnostics.append(
            Diagnostic(
                "unknown_reference_output",
                f"Node {source_node_id} does not declare output {output_name!r}.",
                path,
            )
        )
    if source_node_id in observations and output_name not in observations[source_node_id]:
        diagnostics.append(
            Diagnostic(
                "unresolved_reference_output",
                f"Observation for {source_node_id} does not contain {output_name!r}.",
                path,
            )
        )


def _check_side_effect_gate(
    node: GoalNode,
    capability: Capability,
    allow_side_effects: bool,
    diagnostics: list[Diagnostic],
    path: str,
) -> None:
    is_side_effect = node.kind in SIDE_EFFECT_KINDS or capability.risk in SIDE_EFFECT_RISKS
    if not is_side_effect:
        return
    if not allow_side_effects:
        diagnostics.append(
            Diagnostic(
                "side_effects_not_allowed",
                f"Side-effect node {node.id} cannot execute in read-only mode.",
                path,
            )
        )
    if capability.requires_unique_target and not node.must_be_unique:
        diagnostics.append(
            Diagnostic(
                "mutation_target_not_unique",
                f"Side-effect node {node.id} must have a unique target.",
                f"{path}.must_be_unique",
            )
        )
    if capability.requires_confirmation and not node.authorized:
        diagnostics.append(
            Diagnostic(
                "side_effect_not_authorized",
                f"Side-effect node {node.id} requires authorization.",
                f"{path}.authorized",
            )
        )
    if capability.effects and not node.expected_effect:
        diagnostics.append(
            Diagnostic(
                "missing_expected_effect",
                f"Side-effect node {node.id} must declare expected effect.",
                f"{path}.expected_effect",
            )
        )


def _resolve_input_value(
    goal_input: GoalInput,
    context: dict[str, Any],
    observations: dict[str, dict[str, Any]],
) -> Any:
    if goal_input.source == "node_output" and isinstance(goal_input.value, str):
        match = REFERENCE_PATTERN.fullmatch(goal_input.value.strip())
        if match:
            node_id, output_name = match.groups()
            if node_id in observations and output_name in observations[node_id]:
                return observations[node_id][output_name]
    if goal_input.source == "context":
        return _value_from_context(context, goal_input.value)
    return goal_input.value


def _topological_nodes(nodes: list[GoalNode]) -> list[GoalNode]:
    node_by_id = {node.id: node for node in nodes}
    indegree = {node.id: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in node.depends_on:
            if dep in node_by_id:
                indegree[node.id] += 1
                children[dep].append(node.id)
    queue = deque([node.id for node in nodes if indegree[node.id] == 0])
    ordered = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_by_id[node_id])
        for child in children[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return ordered if len(ordered) == len(nodes) else nodes


def _is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indegree = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for before, after in edges:
        if before not in indegree or after not in indegree:
            continue
        children[before].append(after)
        indegree[after] += 1
    queue = deque([node for node in nodes if indegree[node] == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def _span_in_text(span: str, text: str) -> bool:
    normalized_span = re.sub(r"\s+", " ", span.strip()).lower()
    normalized_text = re.sub(r"\s+", " ", text).lower()
    return bool(normalized_span and normalized_span in normalized_text)


def _value_grounded_in_text(value: Any, text: str) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return all(_value_grounded_in_text(item, text) for item in value)
    normalized_text = re.sub(r"\s+", " ", text).lower()
    if isinstance(value, bool):
        return str(value).lower() in normalized_text
    if isinstance(value, (int, float)):
        escaped = re.escape(str(value))
        pattern = rf"(?<![A-Za-z0-9.]){escaped}(?!\.\d)(?![A-Za-z0-9])"
        if re.search(pattern, normalized_text):
            return True
        return re.search(rf"(?<![A-Za-z0-9.]){escaped}(?=[A-Za-z%])", normalized_text) is not None
    value_text = re.sub(r"\s+", " ", str(value).strip()).lower()
    if not value_text:
        return False
    if value_text in normalized_text:
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", value_text)
    return bool(tokens) and all(re.search(rf"\b{re.escape(token)}\b", normalized_text) for token in tokens)


def _context_contains_value(context: dict[str, Any], value: Any) -> bool:
    return _contains_value(context, value)


def _observations_contain_value(observations: dict[str, dict[str, Any]], value: Any) -> bool:
    return _contains_value(observations, value)


def _contains_value(container: Any, value: Any) -> bool:
    if container == value:
        return True
    if isinstance(container, dict):
        return any(_contains_value(item, value) for item in container.values())
    if isinstance(container, (list, tuple, set)):
        return any(_contains_value(item, value) for item in container)
    return False


def _value_from_context(context: dict[str, Any], value: Any) -> Any:
    if _contains_value(context, value):
        return value
    return value


def _looks_like_generated_identifier(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:id|identifier|uuid|email|address|order number|account number|event id|case id)\b",
            text,
        )
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique
