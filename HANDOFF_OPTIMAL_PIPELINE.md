# Optimal Pipeline Handoff

This handoff captures the current compute2 setup, ensemble architecture, relevant files, latest scores, and recommended next work for the tool-calling/planning pipeline.

## Compute2 Access

Use compute2 through SSH:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu
```

Main environment:

```bash
export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
export WF=$STOR/workflow
export APIBANK=$STOR/external/DAMO-ConvAI/api-bank
export BFCL_ROOT=$WF/external/gorilla/berkeley-function-call-leaderboard
export DATA=$BFCL_ROOT/bfcl_eval/data
export BR=$WF/benchmark_results

source $PROJ/.venv-c2/bin/activate

export HOME=$PROJ/.home-c2
export HF_HOME=$PROJ/.hf_cache
export TRANSFORMERS_CACHE=$PROJ/.hf_cache/transformers
export HF_DATASETS_CACHE=$PROJ/.hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$PROJ/.hf_cache/hub
export XDG_CACHE_HOME=$PROJ/.cache-c2
export TMPDIR=$PROJ/tmp
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$WF/scripts:$APIBANK:$BFCL_ROOT:$WF/external/tau-bench:${PYTHONPATH:-}
```

Slurm:

```bash
squeue -u dshrestha -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"
sbatch path/to/job.sbatch
scancel JOBID
```

Logs:

```bash
$PROJ/logs/compute2
```

Do not hardcode or print secrets. For Gemini/TauBench user simulation, use `GEMINI_API_KEY` from the environment if it is already configured.

## Important Local Files

Local workspace:

```text
/Users/deepshikhashrestha/Documents/Project I
```

Relevant local scripts:

```text
/Users/deepshikhashrestha/Documents/Project I/compute2/eval_bfcl_ensemble_select.py
/Users/deepshikhashrestha/Documents/Project I/compute2/eval_bfcl_multiturn_current_ensemble.py
/Users/deepshikhashrestha/Documents/Project I/compute2/eval_apibank_level3_current_ensemble.py
/Users/deepshikhashrestha/Documents/Project I/compute2/eval_apibank_lv12_current_ensemble.py
/Users/deepshikhashrestha/Documents/Project I/compute2/tau_ensemble_agent.py
/Users/deepshikhashrestha/Documents/Project I/compute2/tau_run.py
/Users/deepshikhashrestha/Documents/Project I/compute2/compute2_tau_ensemble.sbatch
/Users/deepshikhashrestha/Documents/Project I/compute2/compute2_bfcl_multiturn_current_ensemble.sbatch
/Users/deepshikhashrestha/Documents/Project I/compute2/compute2_apibank_level3_current_ensemble.sbatch
/Users/deepshikhashrestha/Documents/Project I/compute2/compute2_apibank_lv12_current_ensemble.sbatch
```

Locked local snapshot:

```text
/Users/deepshikhashrestha/Documents/Project I/pipeline_snapshots/optimal_pipeline_2026-06-28
```

Do not modify the snapshot. Treat it as the current known-good archive.

## Important Remote Files

Remote scripts are under:

```text
/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts
```

Key remote scripts:

```text
$WF/scripts/eval_bfcl_ensemble_select.py
$WF/scripts/eval_bfcl_multiturn_current_ensemble.py
$WF/scripts/eval_apibank_level3_current_ensemble.py
$WF/scripts/eval_apibank_lv12_current_ensemble.py
$WF/scripts/tau_ensemble_agent.py
$WF/scripts/tau_run.py
```

TauBench agent was also copied into the upstream TauBench tree:

```text
$WF/external/tau-bench/tau_bench/agents/ensemble_agent.py
```

Locked remote snapshot:

```text
$WF/pipeline_snapshots/optimal_pipeline_2026-06-28
```

Benchmark outputs:

```text
$BR
```

## Model Roles

The intended ensemble uses specialized candidate generators plus a verifier:

```text
ToolACE-8B
Role: strong direct tool-call generator.
Model: Team-ACE/ToolACE-8B

Original xLAM
Role: strong schema/function-call generator and verifier.
Model: Salesforce/Llama-xLAM-2-8b-fc-r

GPT-OSS APIGen/xLAM LoRA
Role: diverse tool-call candidate generator.
Path: $PROJ/checkpoints/gpt-oss-20b-xlam60k-lora-2xa40-noGC1024

GPT-OSS Nemotron/BFCL LoRA
Role: main tool-call/executor candidate in some jobs.
Path: $PROJ/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024

GPT-OSS task-decomposition LoRA
Role: planner/subgoal candidate.
Path: $PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora

GPT-OSS TaskBench task-decomposition LoRA
Role: planning/recovery candidate.
Path: $PROJ/checkpoints/gpt-oss-20b-taskbench-taskdecomp-lora
```

## Pipeline Architecture

The pipeline is not a majority vote. It is a candidate-generation ensemble with deterministic selection, schema validation, optional LLM verification, and benchmark-specific adapters.

High-level flow:

```text
1. Benchmark adapter builds the prompt from visible inputs.
2. Multiple models generate candidate tool calls.
3. Parser normalizes outputs into a common call format.
4. Schema validator checks tool name, required args, illegal args, and argument object shape.
5. Selector scores candidates using source bias, schema validity, relevance, consensus, and state.
6. Verifier, usually xLAM, reviews candidate options and may choose a source or propose repair.
7. Guardrails decide whether to trust verifier output.
8. Final call/no-call is emitted in the benchmark format and scored.
```

The agent-side prompt is allowed to include the visible tools/schema. That is normal for tool calling and is not leakage. Do not give hidden labels, gold calls, future turns, scorer feedback, hidden TauBench user state, or unretrieved API-Bank Level 3 docs.

## Selector Architecture

The selector is not an LLM. It is deterministic Python logic.

For BFCL single-turn, the selector:

```text
- loads candidate JSONLs from ToolACE/xLAM/GPT-OSS/TaskBench
- normalizes each candidate to `{"name": ..., "arguments": ...}`
- validates schema issues
- computes source bias: xLAM > ToolACE > GPT-OSS APIGen > TaskBench
- adds bonus for valid calls
- penalizes schema issues heavily
- computes semantic relevance between user request, tool schema, and arguments
- detects exact concrete-call consensus across models
- detects multiple no-call/empty candidates
- applies no-call gate for irrelevance categories
- asks verifier for a source choice or repair when configured
- blocks verifier repair/chosen calls under no-call gate unless relevance and consensus are strong
```

The BFCL selector lives in:

```text
$WF/scripts/eval_bfcl_ensemble_select.py
```

Note: check the local copy before editing. A prior local view showed a duplicated `return [], {` in one branch, but remote runs completed successfully. Verify syntax on local and remote with `python -m py_compile` before launching jobs.

## Verifier Architecture

The verifier is an LLM judge, usually:

```text
Salesforce/Llama-xLAM-2-8b-fc-r
```

It receives:

```text
- user request
- visible tool schemas
- candidate calls from each source
- schema issues already detected by the selector
- allowed source names
```

It does not receive gold labels. The prompt instructs it not to solve from scratch and to return compact JSON:

```json
{"chosen_source":"xlam","repair_calls":null,"reason":"short"}
```

or, if it proposes a valid repair:

```json
{"chosen_source":null,"repair_calls":[{"name":"tool","arguments":{}}],"reason":"short"}
```

The deterministic selector still decides whether to trust the verifier. The verifier is evidence, not an unrestricted override.

## API-Bank Level 3 Selector

The Level 3 state-aware selector is currently routed by benchmark script, not by prompt classification.

That means:

```text
API-Bank Level 1/2 -> eval_apibank_lv12_current_ensemble.py
API-Bank Level 3   -> eval_apibank_level3_current_ensemble.py
BFCL single-turn   -> eval_bfcl_ensemble_select.py
BFCL multi-turn    -> eval_bfcl_multiturn_current_ensemble.py
TauBench           -> tau_ensemble_agent.py / tau_run.py
```

The Level 3 selector has general mechanisms, but it is only deployed because the Level 3 script is launched. For a more general agent, refactor this into feature-gated state-aware logic:

```text
Enable state-aware selection when the environment has prior tool results, retrieval tools,
tool-output-to-tool-input dependencies, repeated/list actions, or changing environment state.
Do not check whether the prompt says "Level 3".
```

Level 3 state-aware scoring considers:

```text
- whether ToolSearcher should be called
- whether a tool has already been retrieved
- whether a call repeats prior calls
- whether repeated/list actions are supported
- whether arguments are grounded in prior tool outputs
- whether one tool output likely provides values needed by another tool
- whether candidate values look invented
- whether IDs/dates/numbers/statuses/entities should be repaired from observed state
```

## Current Scores After Latest Regression Check

BFCL single-turn positive categories:

```text
simple_python:      before 385/400 = 96.3%, after 384/400 = 96.0%
multiple:           before 191/200 = 95.5%, after 191/200 = 95.5%
parallel:           before 182/200 = 91.0%, after 182/200 = 91.0%
parallel_multiple:  before 180/200 = 90.0%, after 180/200 = 90.0%
```

BFCL irrelevance:

```text
irrelevance:       before 207/240 = 86.3%, after 215/240 = 89.6%
live_irrelevance:  before 658/884 = 74.4%, after 677/884 = 76.6%
```

API-Bank:

```text
Level 1: 307/399 = 76.9%
Level 2: 70/135 = 51.9%
Level 3: 149/245 = 60.8%, sample 12/50 = 24.0%
```

Interpretation:

```text
- No-call gate improved BFCL irrelevance.
- Positive BFCL categories are preserved except a one-example simple_python drop.
- API-Bank Level 3 improved slightly from 147/245 to 149/245 after tightened state-aware reselect.
```

Important BFCL positive verifier-backed job:

```text
1887260
```

Outputs:

```text
$BR/bfcl_tool_ensemble_simple_python_reselect_v5_verifier_1887260.ensemble.official_ast.scored.json
$BR/bfcl_tool_ensemble_multiple_reselect_v5_verifier_1887260.ensemble.official_ast.scored.json
$BR/bfcl_tool_ensemble_parallel_reselect_v5_verifier_1887260.ensemble.official_ast.scored.json
$BR/bfcl_tool_ensemble_parallel_multiple_reselect_v5_verifier_1887260.ensemble.official_ast.scored.json
```

BFCL irrelevance verifier-backed job:

```text
1887227
```

Outputs:

```text
$BR/bfcl_tool_ensemble_irrelevance_reselect_v5_verifier_1887227.ensemble.official_ast.scored.json
$BR/bfcl_tool_ensemble_live_irrelevance_reselect_v5_verifier_1887227.ensemble.official_ast.scored.json
```

API-Bank Level 3 state-aware reselect output:

```text
$BR/apibank_level3_state_reselect_v5b_from1886612.jsonl
$BR/apibank_level3_state_reselect_v5b_from1886612.scored.json
```

## Recent Changes Already Made

BFCL selector:

```text
- concrete call consensus using canonical call tuples
- valid/empty candidate grouping
- stronger no-call abstention logic
- relevance-aware blocking of verifier repairs under no-call gate
- verifier repairs treated as candidates/evidence, not unconditional winners
```

API-Bank Level 3:

```text
- weighted retrieval scoring
- generic dependency scoring between producer/consumer tools
- prior-output value binding
- state-value repair for grounded IDs/dates/numbers/status/entity values
- list cursor/repeated-action scoring
- tighter retrieval: keyword support required; context only tie-breaker
- stronger missing-producer penalty
- avoids replacing free-form string args and sign-flipping equal-magnitude numbers
```

BFCL multi-turn:

```text
- official multi-turn runner path exists
- duplicate exact decoded-call penalty
- verifier repair is a candidate, not automatic winner
- default stop_after_multicall_success=False
- added --stop-after-multicall-success
```

TauBench:

```text
- tau_run.py wrapper with safe slugs, summary.json, CLI parser/main, empty-result handling
- compute2_tau_ensemble.sbatch uses $WF/scripts/tau_run.py
- tau_ensemble_agent.py debug writes are wrapped in try/except
- ensemble agent copied to $WF/external/tau-bench/tau_bench/agents/ensemble_agent.py
```

## Useful Score-Reading Commands

BFCL scores:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu 'python - <<PY
import json, glob, os
for p in sorted(glob.glob("/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results/bfcl_tool_ensemble_*_reselect_v5_verifier_*.ensemble.official_ast.scored.json")):
    d=json.load(open(p))
    s=(d.get("summary") or d).get("baseline", {})
    print(os.path.basename(p), s)
PY'
```

API-Bank scores:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu 'python - <<PY
import json
from pathlib import Path
BR=Path("/storage3/fs1/yvorobeychik/Active/dshrestha/workflow/benchmark_results")
for name in [
    "apibank_level1_call_full_v2_1885942.scored.json",
    "apibank_level2_retrieve_call_full_v2_1885943.scored.json",
    "apibank_level3_state_reselect_v5b_from1886612.scored.json",
]:
    d=json.load(open(BR/name))
    print(name, d.get("correct_api_calls"), "/", d.get("total_api_calls"), d.get("accuracy"), "sample", d.get("sample_correct"), "/", d.get("sample_total"), d.get("sample_accuracy"))
PY'
```

## Recommended Next Work

1. Refactor state-aware selection into benchmark-agnostic feature gates.

Current API-Bank Level 3 selector is invoked by script. Generalize it by enabling state-aware features when the environment has prior tool outputs, retrieval tools, producer/consumer dependencies, list outputs, or mutable state.

2. Investigate the one-example `simple_python` regression.

Compare before/after candidate choice for the single lost BFCL example. If the no-call/verifier logic caused it, add a narrow positive-category safeguard without weakening irrelevance.

3. Improve BFCL irrelevance further.

The verifier-backed reselect improved irrelevance but is still below the earlier selector-only ceiling in some experiments. Focus on verifier/candidate interaction: when the deterministic no-call gate is confident, the verifier should not rescue weak calls unless relevance and consensus are strong.

4. Continue BFCL multi-turn with official runner.

The proper fix is to generate and execute turn-by-turn using BFCL multi-turn infrastructure, feed state forward, and score with the official multi-turn checker. Avoid one-shot scoring for multi-turn.

5. TauBench.

Use Gemini only for user simulation if the key is present in the environment. The user side should stay objective and should not receive the agent's hidden reasoning. Improvements should target agent-side policy/state tracking and logging/scoring reliability.

6. API-Bank Level 3.

Continue improving generic dependency graphs:

```text
tool output fields -> later tool input fields
list outputs -> repeated actions
retrieved tool docs -> allowed visible tools
prior values -> repair candidate arguments
```

Avoid hardcoded API-Bank trace-order rules unless they can be reframed as general dependency/state constraints.

## Safety / Leakage Rules

Allowed:

```text
- visible tool/function schemas for the current benchmark example
- conversation history up to the current turn
- prior tool outputs that the agent actually observed
- retrieved API docs after a retrieval action
- domain policies visible to the agent
```

Not allowed:

```text
- gold labels or ground truth calls
- hidden test labels
- scorer feedback during generation
- future turns
- TauBench hidden user goal/state
- API-Bank Level 3 full API catalog when the task requires retrieval first
- benchmark-specific trace order hardcoded from labels
```

## Quick Sanity Checks Before Running Jobs

Run local syntax checks before syncing:

```bash
python -m py_compile compute2/eval_bfcl_ensemble_select.py \
  compute2/eval_bfcl_multiturn_current_ensemble.py \
  compute2/eval_apibank_level3_current_ensemble.py \
  compute2/eval_apibank_lv12_current_ensemble.py \
  compute2/tau_run.py \
  compute2/tau_ensemble_agent.py
```

After syncing to compute2, run remote syntax checks inside the venv:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
source /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/.venv-c2/bin/activate
python -m py_compile \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/eval_bfcl_ensemble_select.py \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/eval_bfcl_multiturn_current_ensemble.py \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/eval_apibank_level3_current_ensemble.py \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/eval_apibank_lv12_current_ensemble.py \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/tau_run.py \
  /storage3/fs1/yvorobeychik/Active/dshrestha/workflow/scripts/tau_ensemble_agent.py
'
```

