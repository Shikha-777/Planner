# Goal-Graph Runtime Repair25 Handoff

Timestamp: 2026-07-08 20:58 CDT

## Current Objective

Improve the second pipeline, the goal-graph runtime, for BFCL and API-Bank with generalizable fixes. The user explicitly does not want a pile of benchmark-only heuristics. The intended architecture is:

```text
GPT-OSS semantic understanding / call skeleton when useful
-> deterministic binder and verifier
-> goal-graph compiler/runtime
-> benchmark adapters for BFCL/API-Bank official scoring
```

The frozen/previous pipeline idea that was copied over is: do not trust the model to produce perfect final calls. Let GPT-OSS propose semantic/tool-binding evidence, then verify and compile deterministically.

## Files Changed In This Round

- `scripts/goal_graph_eval_common.py`
- `src/taskdecomp/tool_binding.py`
- `scripts/eval_goal_graph_apibank_routing.py`
- `tests/test_goal_graph_eval_common.py`
- `tests/test_tool_binding.py`
- `tests/test_apibank_tool_routing_adapter.py`

## Important Implemented Fixes

### Semantic Frame Parsing

- Added repair for malformed GPT-OSS semantic-frame JSON where `tool_bindings` is missing a closing array bracket before `missing_inputs`.
- Added regression: `test_extract_semantic_frame_repairs_missing_tool_bindings_array_bracket`.

### Model Binding Grounding

- Numeric model-binding values now require numeric evidence in the prompt or evidence span.
- Prevents cases like `specific_heat=4.184` being accepted just because the evidence says `"water"`.
- Added percent-value support when schema description says percent/percentage.
- Added duplicate numeric safeguards to stop filling multiple distinct numeric slots with the same number.

### Repeated Call Evidence

- If GPT-OSS gives one shared evidence span for a value reused across repeated argument groups, reuse that evidence safely.
- This was needed for repeated calls like AWS pricing with one shared CPU value.

### Location Normalization

- Added Shanghai aliases:
  - `shanghai -> Shanghai, China`
  - `上海 -> Shanghai, China`
- This should fix the BFCL live Shanghai official-AST failure.

### Skeleton Recovery Guard

Problem: the call-skeleton GPT pass could resurrect calls after the binder/verifier had already rejected them as unsupported/no-evidence. Example: BFCL irrelevance asked “how much gas is generated,” the verifier rejected a pressure calculator, then skeleton recovery brought it back.

Fix:
- `_should_generate_call_skeleton()` now refuses skeleton recovery when the base binding plan has strong no-call evidence.
- `_call_plan_for_skeleton_item()` now requires skeleton-proposed calls to pass route/audit eligibility.
- Skeleton no longer overrides hard semantic conflict or duplicate required-slot warnings.

Regressions:
- `test_call_skeleton_does_not_resurrect_verified_no_tool_hard_conflict`

### Duplicate Required String Slots

Problem: the binder could fill two distinct required string slots with the same quoted/entity value. Example:

```text
artwork_name = "The Scream"
museum_location = "The Scream"
```

Fix:
- String duplicate blocking now catches location-like slot reuse when another required slot is not location-like.
- Kept this narrow so legitimate domain pairs such as species/ecosystem are not blocked.

Regression:
- `test_tool_binding_rejects_duplicate_string_fill_for_distinct_required_roles`

### Result-Dependent Single-Tool Chains

Problem: BFCL parallel undercount case:

```text
HCF(45, 60), then use that result with (90, 120).
Also HCF(36, 48), then use that result with (72, 96).
```

The binder only made two calls; expected tool-name sequence has four `math.hcf` calls.

Fix:
- Skeleton pass now triggers for result-dependent single-tool chains with phrases like “use that result” / “of that result with”.
- This is intentionally general: it is not HCF-specific.

Regression:
- `test_call_skeleton_runs_for_single_tool_result_dependent_chains`

### API-Bank Auth Adapter

Problem: API-Bank adapter could override a correct compiled protected action back to `GetUserToken` because earlier dialogue said “need token,” even after the latest context said the token was obtained.

Fix:
- `_prompt_indicates_auth_step()` now returns false when latest context says the token was obtained/authentication complete and latest prior API result contains a token.

Regression:
- `test_apibank_prediction_prefers_compiled_route_after_token_obtained_message`

### API-Bank Follow-Up Recovery

Problem: GPT-OSS sometimes chooses terminal `no_tool` for “tell me more / yes please” follow-ups, but API-Bank expects the underlying API again.

Fix:
- Added prior-result schema routing in `predicted_api_tool_names()`.
- If latest user turn is a follow-up and latest/prior API result is a list/dict, match result keys against available tools’ `output_parameters`.
- This recovers cases like:
  - prior result keys `{name, description}` -> `SymptomSearch`
  - prior result keys `{name, aid}` -> `EmergencyKnowledge`

Regressions:
- `test_apibank_prediction_recovers_followup_route_from_prior_result_schema`
- `test_apibank_prediction_uses_prior_result_schema_to_disambiguate_followup_tools`

## Local Verification Already Run

From local repo:

```bash
cd "/Users/deepshikhashrestha/Documents/Project I"
PYTHONPATH=src:. python3 -m pytest \
  tests/test_goal_graph_eval_common.py \
  tests/test_tool_binding.py \
  tests/test_apibank_tool_routing_adapter.py \
  -q
```

Result:

```text
308 passed in 17.48s
```

Note: Compute2 `.venv-c2` currently does not have `pytest`, so remote pytest failed with `No module named pytest`. Local unit tests are the confirmed unit-test signal unless pytest is installed remotely.

## Compute2 Access And Paths

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu

cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss
RESULTS=/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results/goal_graph_runtime
```

Useful sync from local to Compute2:

```bash
cd "/Users/deepshikhashrestha/Documents/Project I"
rsync -av \
  scripts/goal_graph_eval_common.py \
  scripts/eval_goal_graph_apibank_routing.py \
  src/taskdecomp/tool_binding.py \
  tests/test_goal_graph_eval_common.py \
  tests/test_tool_binding.py \
  tests/test_apibank_tool_routing_adapter.py \
  dshrestha@c2-login-001.ris.wustl.edu:/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/ \
  --relative
```

## Current Compute2 Jobs

Latest observed queue:

```text
2011585  RUNNING  BFCL live_parallel + live_parallel_multiple  tag goal_graph_bfcl_live_parallel_pm_c2_repair24_generalfix_full_20260708
2011702  RUNNING  BFCL irrelevance                            tag goal_graph_bfcl_irrelevance_c2_repair25_skeleton_guard_full_20260708
2011745  RUNNING  BFCL parallel + parallel_multiple            tag goal_graph_bfcl_parallel_pm_c2_repair25_skeleton_guard_full_20260708
2011803  PENDING  API-Bank lv1/lv2                             tag goal_graph_apibank_c2_repair25_adapter_schema_full_20260708
```

Check queue:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu \
  'squeue -u dshrestha -o "%.18i %.9P %.35j %.8T %.10M %.10l %.6D %R"'
```

## Compute2 Benchmark Commands

Run BFCL static `parallel` and `parallel_multiple`:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss &&
sbatch --export=ALL,RUN_BFCL=1,RUN_APIBANK=0,BFCL_DATA_DIR=data/bfcl_v4_full,BFCL_CATEGORIES="parallel parallel_multiple",BFCL_LIMIT=0,GOAL_GRAPH_PLANNER_MODE=stepwise,TAG=goal_graph_bfcl_parallel_pm_c2_repair25_full_$(date +%Y%m%d_%H%M%S) \
  ris/compute2_goal_graph_bfcl_apibank_smoke.sbatch
'
```

Run BFCL irrelevance:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss &&
sbatch --export=ALL,RUN_BFCL=1,RUN_APIBANK=0,BFCL_DATA_DIR=data/bfcl_v4_full,BFCL_CATEGORIES="irrelevance",BFCL_LIMIT=0,GOAL_GRAPH_PLANNER_MODE=stepwise,TAG=goal_graph_bfcl_irrelevance_c2_repair25_full_$(date +%Y%m%d_%H%M%S) \
  ris/compute2_goal_graph_bfcl_apibank_smoke.sbatch
'
```

Run BFCL live parallel categories with official AST scoring:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss &&
sbatch --export=ALL,RUN_BFCL=1,RUN_APIBANK=0,BFCL_DATA_DIR=data/bfcl_v4_live_full,BFCL_CATEGORIES="live_parallel live_parallel_multiple",BFCL_LIMIT=0,BFCL_OFFICIAL_AST_SCORE=1,GOAL_GRAPH_PLANNER_MODE=stepwise,TAG=goal_graph_bfcl_live_parallel_pm_c2_repair25_full_$(date +%Y%m%d_%H%M%S) \
  ris/compute2_goal_graph_bfcl_apibank_smoke.sbatch
'
```

Run API-Bank lv1/lv2:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss &&
sbatch --export=ALL,RUN_BFCL=0,RUN_APIBANK=1,APIBANK_LEVELS="level-1-api level-2-api",APIBANK_LIMIT=0,GOAL_GRAPH_PLANNER_MODE=stepwise,TAG=goal_graph_apibank_c2_repair25_full_$(date +%Y%m%d_%H%M%S) \
  ris/compute2_goal_graph_bfcl_apibank_smoke.sbatch
'
```

Run BFCL multi-turn on Compute2:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss &&
sbatch --export=ALL,BFCL_MT_CATEGORY=multi_turn_base,BFCL_MT_LIMIT=0,BFCL_MT_PLANNER_MODE=stepwise,BFCL_MT_TAG=goal_graph_mt_base_repair25_$(date +%Y%m%d_%H%M%S) \
  ris/compute2_goal_graph_bfcl_multiturn.sbatch
'
```

Tail Compute2 logs:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu \
  'ls -lt /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2 | head -20'

ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu \
  'tail -f /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/logs/compute2/<JOBID>.goal_graph_smoke.out'
```

## Compute1 Benchmark Commands

Compute1 uses LSF/`bsub`. The repo has a dedicated launcher:

```text
ris/compute1_goal_graph_bfcl_apibank_smoke.bsub
```

Run these after logging into the Compute1 environment. The Compute1 login hostname is not documented in this repo; use the RIS Compute1 entrypoint available to the user/account.

Set project:

```bash
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss
```

BFCL static `parallel` and `parallel_multiple`:

```bash
export RUN_BFCL=1
export RUN_APIBANK=0
export BFCL_DATA_DIR=data/bfcl_v4_full
export BFCL_CATEGORIES="parallel parallel_multiple"
export BFCL_LIMIT=0
export GOAL_GRAPH_PLANNER_MODE=stepwise
export TAG=goal_graph_bfcl_parallel_pm_c1_repair25_$(date +%Y%m%d_%H%M%S)
bsub < ris/compute1_goal_graph_bfcl_apibank_smoke.bsub
```

BFCL irrelevance:

```bash
export RUN_BFCL=1
export RUN_APIBANK=0
export BFCL_DATA_DIR=data/bfcl_v4_full
export BFCL_CATEGORIES="irrelevance"
export BFCL_LIMIT=0
export GOAL_GRAPH_PLANNER_MODE=stepwise
export TAG=goal_graph_bfcl_irrelevance_c1_repair25_$(date +%Y%m%d_%H%M%S)
bsub < ris/compute1_goal_graph_bfcl_apibank_smoke.bsub
```

API-Bank lv1/lv2:

```bash
export RUN_BFCL=0
export RUN_APIBANK=1
export APIBANK_LEVELS="level-1-api level-2-api"
export APIBANK_LIMIT=0
export GOAL_GRAPH_PLANNER_MODE=stepwise
export TAG=goal_graph_apibank_c1_repair25_$(date +%Y%m%d_%H%M%S)
bsub < ris/compute1_goal_graph_bfcl_apibank_smoke.bsub
```

Small BFCL/API-Bank smoke on Compute1:

```bash
export RUN_BFCL=1
export RUN_APIBANK=1
export BFCL_DATA_DIR=data/bfcl_v4_full
export BFCL_CATEGORIES="parallel"
export BFCL_LIMIT=20
export APIBANK_LEVELS="level-1-api level-2-api"
export APIBANK_LIMIT=20
export GOAL_GRAPH_PLANNER_MODE=stepwise
export TAG=goal_graph_c1_smoke_repair25_$(date +%Y%m%d_%H%M%S)
bsub < ris/compute1_goal_graph_bfcl_apibank_smoke.bsub
```

Monitor Compute1:

```bash
bjobs -u dshrestha
tail -f /home/dshrestha/taskdecomp_status/goal_graph_stepwise.<JOBID>.out
tail -f /home/dshrestha/taskdecomp_status/goal_graph_stepwise.<JOBID>.err
```

## Scoring And Result Files

All benchmark outputs go under:

```bash
/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results/goal_graph_runtime
```

Important files by tag:

```text
<TAG>.bfcl.metrics.json
<TAG>.bfcl.review.csv
<TAG>.bfcl.plans.jsonl
<TAG>.bfcl.official_predictions.jsonl
<TAG>.bfcl.<category>.official_ast.scored.json

<TAG>.apibank.metrics.json
<TAG>.apibank.review.csv
<TAG>.apibank.plans.jsonl
```

Quick list latest results:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu \
  'ls -lt /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results/goal_graph_runtime | head -40'
```

Quick streaming tool-name score from a plans file:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu 'PLAN_FILE=/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results/goal_graph_runtime/<TAG>.bfcl.plans.jsonl python3 - <<'"'"'PY'"'"'
import json, os, pathlib, collections
path = pathlib.Path(os.environ["PLAN_FILE"])

def norm(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(str(x.get("name") or x.get("tool_name") or x.get("function") or x.get("api_name") or x))
            else:
                out.append(str(x))
        return out
    return [str(v)]

total = ordered = multiset = 0
by = collections.defaultdict(lambda: [0, 0, 0])
misses = []
for line in path.open():
    if not line.strip():
        continue
    row = json.loads(line)
    exp = norm(row.get("expected"))
    pred = norm(row.get("predicted"))
    cat = row.get("category") or row.get("level") or "unknown"
    ok = exp == pred
    ms = collections.Counter(exp) == collections.Counter(pred)
    total += 1
    ordered += ok
    multiset += ms
    by[cat][0] += 1
    by[cat][1] += ok
    by[cat][2] += ms
    if not ok and len(misses) < 12:
        misses.append((row.get("id"), cat, exp, pred))
print(path)
print(f"total={total} ordered={ordered}/{total}={ordered/total*100 if total else 0:.1f}% multiset={multiset}/{total}={multiset/total*100 if total else 0:.1f}%")
for cat, (n, o, m) in sorted(by.items()):
    print(f"  {cat}: n={n} ordered={o/n*100:.1f}% multiset={m/n*100:.1f}%")
if misses:
    print("first misses:")
    for miss in misses:
        print(" ", miss)
PY'
```

## Previous Observed Results Before Repair25 Reruns

These are stale now because repair25 jobs were restarted with new code, but useful for comparison:

- BFCL static partial before restart:
  - `parallel`: `78/79 = 98.7%`
  - miss was `parallel_77`, result-dependent HCF chain undercount.
- BFCL irrelevance stale partial:
  - around `93%`
  - common failure was skeleton resurrecting near-miss tools after verifier rejection.
- API-Bank stale partial:
  - `28/33 = 84.8%`
  - failures included follow-up no-tool, auth override, and missing-token/slot cases.
- BFCL live pre-repair official AST:
  - `live_parallel`: `12/16 = 75%`
  - `live_parallel_multiple`: `19/24 = 79.2%`
  - internal tool-name accuracy was higher; official AST caught argument normalization issues.

## What To Do Next

1. Monitor jobs `2011585`, `2011702`, `2011745`, `2011803`.
2. Score repair25 plan files as they stream.
3. If repair25 improves irrelevance/parallel/API-Bank, keep the patches; if parallel_multiple is still weak, inspect call-count/order misses first.
4. For BFCL live, compare official AST after rerun. Confirm Shanghai and shared-evidence fixes moved official AST.
5. BFCL multi-turn integration exists in `scripts/eval_goal_graph_bfcl_multiturn.py` and `ris/compute2_goal_graph_bfcl_multiturn.sbatch`, but has not been deeply optimized in this repair pass.
6. Avoid adding benchmark-name shortcuts. Prefer:
   - semantic-frame evidence,
   - schema/result-shape matching,
   - route eligibility checks,
   - grounded call-count rules,
   - skeleton only when deterministic plan is under-specified, not contradicted.
