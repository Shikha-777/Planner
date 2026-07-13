# Goal-Graph Runtime Pipeline

This is a separate pipeline idea for BFCL, tau-bench, and API-Bank. It should not be implemented by continuing to add semantic rules to the current deterministic binder.

## Core Principle

Do not make the deterministic layer understand the query.

Use:

```text
LLM = semantic planner
Runtime = execution kernel
Compiler = tool binder
Verifier = invariant checker
RAG = tool, policy, and failure memory
```

The LLM proposes a grounded goal graph. The deterministic runtime enforces universal invariants.

## Proposed Flow

```text
User query
-> RAG retrieves relevant tools, capability docs, policies, and failure lessons
-> LLM creates a goal graph
-> Runtime validates graph grounding, dependencies, completeness, and policy
-> Compiler maps graph nodes to concrete tools
-> Runtime executes safe read-only nodes
-> LLM replans with observations
-> Runtime gates mutations and communications
-> Final answer, clarification, or abstention
```

## Goal Graph Node Kinds

Use a small universal taxonomy:

```text
resolve
retrieve
search
rank
decide
mutate
communicate
ask_user
respond
```

These cover:

- BFCL: function selection, arguments, ordering, cardinality, abstention
- API-Bank: API retrieval, planning, and multi-step calls
- tau-bench: policy-aware multi-turn workflows and final state correctness

## Runtime Invariants

The deterministic layer should enforce:

```text
1. Schema validity
2. Grounding of every concrete argument
3. Dependency validity
4. Unique targets for mutations
5. No invented IDs
6. Side-effect gating
7. Policy compliance
8. Completeness of required inputs
9. Minimality of calls
10. Safe repair without new ungrounded values
```

## Capability Registry

Raw tool schemas should be compiled into capability objects:

```json
{
  "tool_name": "cancel_calendar_event",
  "capability": "calendar.cancel_event",
  "kind": "mutate",
  "required_inputs": {
    "event_id": {
      "type": "calendar_event_id",
      "must_come_from": ["calendar.search_events", "conversation_context"]
    }
  },
  "risk": "destructive_side_effect",
  "requires_unique_target": true,
  "requires_confirmation": true
}
```

## Transaction Protocol

Use progressive execution:

```text
READ phase
PLAN phase
SIMULATE phase
POLICY phase
COMMIT phase
```

Safe resolver/retrieval nodes may execute early. Mutations and communications require complete inputs, unique targets, policy permission, and side-effect authorization.

## Acceptance Contract

Execute only if:

```text
- the graph is grounded in query/context/tool observations
- every tool call implements a graph node
- every required input is resolved
- every mutation has a unique target
- every side effect is authorized
- every policy constraint is satisfied
- every call is schema-valid
- no unresolved ambiguity remains
```

## First Implementation Slice

Start with a small read-only BFCL/API-Bank slice:

```text
1. Capability registry builder from tool schemas
2. Goal graph JSON schema
3. Grounding/dependency/schema verifier
4. Compiler for retrieve/search/rank/resolve nodes
5. Local evaluator against BFCL simple_python and API-Bank retrieval-style examples
```

Only after that should mutation/transaction gates be added for tau-bench.

## Current Implementation Status

Implemented on 2026-07-05:

```text
src/taskdecomp/goal_graph_runtime.py
tests/test_goal_graph_runtime.py
```

The implemented slice includes:

- capability registry builder from raw tool schemas
- goal graph dataclasses and JSON normalization
- planner prompt builder that asks for graphs, not direct tool calls
- runtime facade for verify/compile
- invariant checks for schema validity, grounding, dependencies, cycles, required inputs, unknown inputs, unresolved inputs, and side-effect gates
- compiler from verified graph nodes to concrete tool calls
- support for read-only nodes and explicitly authorized side-effect nodes

## Frozen-Pipeline Lessons Copied Into This Framework

Copied because they agree with the goal-graph architecture:

- Grounding discipline: every concrete query value must become a graph input with evidence, or remain unresolved and trigger `ask_user`.
- Cardinality lesson: top-N/result-count phrases are usually limits on one `rank/search/retrieve` node, not repeated calls.
- Batch lesson: use list/array inputs when the capability supports batching; repeat nodes only for scalar capabilities.
- Identifier safety: IDs, emails, order numbers, event IDs, and account IDs must come from explicit query text, context, user confirmation, or prior node/tool output.
- Schema defaults: defaults are policy/schema-grounded values and can be filled by the runtime without pretending they came from the query.
- Prose enums: descriptions such as `either 'summary' or 'full'` are parsed into allowed values and verified.
- Soft-evidence lesson: the LLM planner must bind semantic evidence explicitly into graph inputs instead of leaving vague missing slots.

Deliberately not copied:

- Domain-specific English slot parsers such as currency/restaurant/recipe/team extractors.
- Semantic role aliasing such as `topic -> religion_name` inside the runtime.
- Lexical mismatch override logic.

Those belong in the LLM goal-graph planner or RAG failure lessons, not the deterministic runtime.

Validation:

```text
PYTHONPATH=src:. pytest tests/test_goal_graph_runtime.py -q
15 passed

PYTHONPATH=src:. pytest tests -q
185 passed
```

Next slice:

```text
1. Add a lightweight retrieval layer for capabilities, policies, and failure lessons.
2. Add an LLM runner that calls the planner prompt and parses graph JSON.
3. Add progressive read-only execution with observations feeding back into graph repair.
4. Add benchmark adapters for BFCL simple/retrieval-style cases before touching tau-bench mutations.
```
