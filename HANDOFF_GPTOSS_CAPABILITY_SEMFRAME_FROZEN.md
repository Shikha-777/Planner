# Frozen GPT-OSS Capability/Semantic-Frame Pipeline

Status: frozen on 2026-07-05.

This handoff marks the current GPT-OSS capability planner plus semantic-frame binder as a frozen pipeline. The next idea is a separate goal-graph runtime pipeline, not another round of deterministic binder growth.

## Frozen Snapshot

```text
/Users/deepshikhashrestha/Documents/Project I/pipeline_snapshots/gptoss_capability_semframe_2026-07-05
```

The snapshot contains:

```text
tool_binding.py
capability_planning.py
run_gptoss_capability_plan.py
eval_bfcl_gptoss_capability_ensemble.py
compute2_bfcl_gptoss_capability_ensemble.sbatch
test_tool_binding.py
MANIFEST.md
SHA256SUMS.txt
```

## Current Pipeline Summary

The current pipeline is:

```text
user request
-> GPT-OSS capability ensemble
-> GPT-OSS semantic slot-frame pass
-> Python binder consumes grounded slot facts and call groups
-> schema/slot/cardinality audit
-> concrete BFCL-style tool calls or ask/no-tool
```

The latest binder state includes:

- semantic-frame slot facts used as grounded evidence
- semantic call groups used for call-count/cardinality hints
- soft evidence promotion into call arguments
- semantic-frame canonical request used before lexical mismatch rejection
- role aliasing such as `topic -> *_name`, `person_name -> scientist`, and `nutrients -> information`
- description defaults such as `Default is 52`
- prose enum parsing such as `either 'summary' or 'full'`
- focused parsers for unit/currency conversions, restaurant filters, recipe/site slots, preferences, teams/seasons, specs, and nutrition fields

## Validation State

Remote full `simple_python` result before the last local binder repair:

```text
total: 400
tool_set: 0.9475
multiset: 0.9475
count: 0.9475
ordered: 0.9475
failures: 21
```

Local checks after the last binder repair:

```text
PYTHONPATH=src:. pytest tests/test_tool_binding.py -q
58 passed

PYTHONPATH=src:. pytest tests -q
170 passed
```

Local replay of the 21 saved remote failures:

```text
checked: 21
remaining_failures: []
```

No full post-repair Compute2 run has been launched yet.

## Resume Instructions

Only resume this frozen pipeline for:

- a final Compute2 confirmation run
- regression fixes
- packaging or reproducibility cleanup

Do not use this pipeline as the substrate for the new goal-graph runtime. The goal-graph runtime should be its own architecture with an LLM semantic planner, capability registry, deterministic invariant checker, compiler, progressive execution loop, and side-effect transaction gates.

See:

```text
/Users/deepshikhashrestha/Documents/Project I/docs/goal_graph_runtime_pipeline.md
```
