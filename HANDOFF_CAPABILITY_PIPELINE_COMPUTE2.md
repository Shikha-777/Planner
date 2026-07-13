# Capability Planning Pipeline + Compute2 Handoff

This handoff describes the updated planning direction for GPT-OSS. The goal is
not tool selection yet. The goal is to infer a clean capability plan from a user
request.

## Core Idea

Do not ask GPT-OSS to directly choose tools, agents, APIs, libraries, or plugins.

For now, GPT-OSS should produce abstract capability requirements:

```text
user request
-> intent/input audit
-> transformation audit
-> generated capability requirements
-> capability normalization
-> capability dependency graph
-> structural validation/repair
```

Tool matching based on capabilities is explicitly later work.

Capabilities are not a strict legal list right now. They are generated as
state transformations: what kind of work must happen to turn the available
inputs into the desired output.

Good capability examples:

```text
retrieve_current_information
extract_information_from_attached_document
revise_text_for_clarity
compute_numeric_result
analyze_provided_dataset
request_missing_input
validate_output_against_requirements
```

Bad capability examples:

```text
research_agent
coding_agent
pdfplumber
browser_tool
assistant_writer
```

Keep one tool-adjacent field only:

```text
external_action_type:
none | file_reading | file_writing | web_search | fact_checking |
calculation | code_execution | image_understanding | image_generation |
user_input | other
```

This field tells the later router what kind of outside action might be needed,
without binding the plan to a specific tool.

## Run 1: Intent + Input Audit

Ask GPT-OSS only:

```text
What final thing does the user want?
What inputs are needed?
Are those inputs already available?
What format are the inputs in?
```

Rules:

```text
Do not choose tools.
Do not choose agents.
Do not choose capabilities yet.
Use only the provided user request and attachments metadata.
Output only compact valid JSON.
```

Schema:

```json
{
  "final_user_want": "...",
  "inputs": [
    {
      "name": "...",
      "needed_for": "...",
      "available": true,
      "format": "pasted_text | attached_file | pdf | image | url | file_path | structured_data | unknown | none",
      "evidence": "..."
    }
  ],
  "missing_inputs": []
}
```

Notes:

- If the user pasted text, mark it as `pasted_text`.
- If the user attached a file but its contents are not in context, mark the file
  itself as available but its extracted contents as unavailable if needed.
- If the user asks for a generic function or template, specific runtime inputs
  may be parameters rather than missing information.

## Run 2: Transformation + Externality Audit

Give GPT-OSS:

```text
original user request
Run 1 JSON
```

Ask only:

```text
What transformations are needed to turn the available inputs into the desired output?
Does the task require current or external information?
Does it require file reading, file writing, search, calculation, code execution,
image generation, image understanding, user input, or another external action?
```

Schema:

```json
{
  "starting_state": "...",
  "desired_state": "...",
  "transformations_needed": [
    {
      "transformation": "...",
      "input_state": "...",
      "output_state": "...",
      "reason": "..."
    }
  ],
  "needs_current_or_external_info": false,
  "external_actions": [
    {
      "action_type": "file_reading | file_writing | web_search | fact_checking | calculation | code_execution | image_understanding | image_generation | user_input | none | other",
      "needed": true,
      "reason": "..."
    }
  ]
}
```

## Run 3: Generated Capability Requirements

Give GPT-OSS:

```text
original user request
Run 1 JSON
Run 2 JSON
```

Ask it to generate abstract capabilities.

Rules:

```text
Do not choose specific agents.
Do not choose concrete tools.
Do not name APIs, libraries, plugins, or software packages.
Do not use an allowed capability list.
A capability must describe a state transformation, not a worker.
Every capability must include inputs, outputs, external_action_type, and done_when.
Output only compact valid JSON.
```

Schema:

```json
{
  "capabilities_needed": [
    {
      "id": "cap_1",
      "capability_name": "...",
      "capability_description": "...",
      "input_state": "...",
      "output_state": "...",
      "requires_external_action": true,
      "external_action_type": "none | file_reading | file_writing | web_search | fact_checking | calculation | code_execution | image_understanding | image_generation | user_input | other",
      "inputs": ["..."],
      "outputs": ["..."],
      "done_when": "..."
    }
  ]
}
```

Example:

```json
{
  "id": "cap_1",
  "capability_name": "retrieve_current_pricing_information",
  "capability_description": "Find recent pricing information for the requested product or service.",
  "input_state": "product or service name is known",
  "output_state": "recent price information is available",
  "requires_external_action": true,
  "external_action_type": "web_search",
  "inputs": ["product or service name"],
  "outputs": ["recent price information"],
  "done_when": "enough current pricing information is available to answer the user"
}
```

Unsupported example:

```json
{
  "id": "cap_1",
  "capability_name": "measure_real_world_object_from_image",
  "capability_description": "Estimate real-world dimensions from an image.",
  "input_state": "image is available but no scale reference is provided",
  "output_state": "cannot reliably measure object without scale reference",
  "requires_external_action": true,
  "external_action_type": "image_understanding",
  "inputs": ["image", "scale reference"],
  "outputs": ["dimension estimate or missing-input request"],
  "done_when": "the system either has a scale reference or asks the user for one"
}
```

## Run 4: Capability Normalization

This pass keeps the ontology from becoming messy, without forcing a fixed list.

Give GPT-OSS:

```text
Run 3 JSON
```

Ask:

```text
Normalize capability names.
Merge capabilities that mean the same thing.
Use short snake_case names.
Do not map to tools.
Do not introduce new work.
```

Schema:

```json
{
  "normalized_capabilities": [
    {
      "id": "cap_1",
      "original_name": "...",
      "normalized_name": "...",
      "meaning_changed": false,
      "external_action_type": "..."
    }
  ],
  "merged_capabilities": [
    {
      "new_id": "cap_1",
      "merged_from": ["cap_1", "cap_2"],
      "normalized_name": "...",
      "reason": "..."
    }
  ]
}
```

Example:

```json
{
  "original_name": "look up recent pricing",
  "normalized_name": "retrieve_current_information",
  "external_action_type": "web_search"
}
```

## Run 5: Capability Ordering / Dependency Graph

Give GPT-OSS:

```text
Run 3 JSON
Run 4 JSON
```

Ask it to order only the capabilities already identified.

Rules:

```text
Do not choose tools.
Do not choose agents.
Do not add unrelated capabilities.
Dependencies must reference valid capability ids.
Every capability must have inputs, outputs, and done_when.
Output only compact valid JSON.
```

Schema:

```json
{
  "ordered_capabilities": [
    {
      "id": "cap_1",
      "capability_name": "...",
      "depends_on": [],
      "inputs": ["..."],
      "outputs": ["..."],
      "done_when": "..."
    }
  ]
}
```

## Run 6: Structural Validator / Minimal Repair

Prefer code for this when possible. A GPT-OSS validator can also work, but it
must be constrained not to redesign the plan.

Check:

```text
- Every capability has id, capability_name, inputs, outputs, done_when.
- Every dependency references a valid capability id.
- No dependency cycles.
- If required input is missing, include a request_missing_input-style capability.
- If pasted text was provided, do not add file extraction.
- If a PDF was attached and text is needed, include a text extraction capability.
- If current facts are needed, include an information retrieval or fact-checking capability.
- If external action is needed, external_action_type must be explicit.
```

Validator output schema:

```json
{
  "valid": false,
  "violations": [
    {
      "type": "...",
      "message": "...",
      "capability_id": "cap_1"
    }
  ],
  "minimal_repairs": [
    {
      "action": "add_capability | edit_field | remove_capability | set_dependency",
      "reason": "...",
      "patch": {}
    }
  ]
}
```

The validator should not produce a completely new plan. It should only report
violations and the smallest local repairs needed.

## Deterministic GPT-OSS Settings

Use deterministic decoding for all planning passes:

```text
do_sample=false
temperature=0 if the runner exposes it
compact JSON only
retry only the failed field if JSON validation fails
```

For GPT-OSS-20B, prefer a full H100-style GPU allocation on compute2. The
`general-short` partition can land on small MIG slices and has caused CUDA OOM
for GPT-OSS-20B.

## Current Local Project Paths

Local repo:

```text
/Users/deepshikhashrestha/Documents/Project I
```

Storage3 project:

```text
/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss
```

Useful existing scripts:

```text
scripts/run_gptoss_chat_once.py
scripts/run_gptoss_input_audit_batch.py
scripts/run_gptoss_decomp_batch.py
scripts/prepare_nemotron_messages_for_input_audit.py
scripts/prepare_trace_decomp_dataset.py
```

Useful existing Slurm files:

```text
ris/compute2_gptoss_chat_once.sbatch
ris/compute2_gptoss_nemotron_input_audit.sbatch
ris/compute2_infer_gptoss_decomp_batch.sbatch
ris/compute2_train_gptoss_trace_decomp.sbatch
```

Important existing results:

```text
results/nemotron_input_audit_gptoss_sample25.final.jsonl
results/nemotron_input_audit_gptoss_sample25.final.parsed.json
results/nemotron_input_audit_gptoss_sample25.final.md
results/gptoss_base_tasks_a_to_e_1900198.results.md
results/gptoss_tuned_tasks_a_to_e.results.md
results/gptoss_base_vs_tuned_tasks_a_to_e.compare.md
```

## Compute2 Access

Login hosts:

```text
c2-login-001.ris.wustl.edu
c2-login-002.ris.wustl.edu
```

Preferred host when reachable:

```text
c2-login-002.ris.wustl.edu
```

If SSH fails with `Permission denied` and `klist` shows an expired ticket, renew
Kerberos locally:

```bash
klist
kinit dshrestha@ACCOUNTS.AD.WUSTL.EDU
ssh c2-login-002.ris.wustl.edu
```

If `c2-login-002` times out or rejects the connection, try:

```bash
ssh c2-login-001.ris.wustl.edu
```

## Compute2 Environment Setup

Once on compute2:

```bash
export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
cd "$PROJ"

export HOME="$PROJ/.home-c2"
export HF_HOME="$PROJ/.hf_cache"
export TRANSFORMERS_CACHE="$PROJ/.hf_cache/transformers"
export HF_DATASETS_CACHE="$PROJ/.hf_cache/datasets"
export HUGGINGFACE_HUB_CACHE="$PROJ/.hf_cache/hub"
export XDG_CACHE_HOME="$PROJ/.cache-c2"
export TMPDIR="$PROJ/tmp"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/compute2 results "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

source "$PROJ/.venv-c2/bin/activate"
```

Quick GPU/PyTorch check inside a job or interactive allocation:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

## Syncing Local Changes to Storage3

From local machine:

```bash
rsync -av \
  "/Users/deepshikhashrestha/Documents/Project I/scripts/" \
  c2-login-002.ris.wustl.edu:/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/scripts/

rsync -av \
  "/Users/deepshikhashrestha/Documents/Project I/ris/" \
  c2-login-002.ris.wustl.edu:/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/ris/
```

If `c2-login-002` is not reachable, use `c2-login-001`.

To pull results back:

```bash
rsync -av \
  c2-login-002.ris.wustl.edu:/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/results/ \
  "/Users/deepshikhashrestha/Documents/Project I/results/"
```

## Running Regular GPT-OSS One-Off Prompts

The existing generic runner is:

```text
ris/compute2_gptoss_chat_once.sbatch
```

It uses:

```text
MODEL_PATH=openai/gpt-oss-20b
SYSTEM_PROMPT=...
PROMPT=...
OUT=...
MAX_NEW_TOKENS=...
```

Recommended pattern on compute2:

```bash
cd /storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss

cat > /tmp/cap_system.txt <<'EOF'
You are a capability-planning model.
Do not choose tools.
Do not choose agents.
Output only compact valid JSON.
EOF

cat > /tmp/cap_prompt.txt <<'EOF'
USER_REQUEST_START
Improve this essay: [paste user request here]
USER_REQUEST_END

Task:
Run 1: Intent + Input Audit.
Return JSON using the specified schema.
EOF

export SYSTEM_PROMPT="$(cat /tmp/cap_system.txt)"
export PROMPT="$(cat /tmp/cap_prompt.txt)"
export OUT="results/capability_run1_${USER}_${RANDOM}.json"
export MAX_NEW_TOKENS=900

sbatch --export=ALL,SYSTEM_PROMPT,PROMPT,OUT,MAX_NEW_TOKENS \
  ris/compute2_gptoss_chat_once.sbatch
```

Check job status:

```bash
squeue -u "$USER"
```

Check logs:

```bash
ls -lh logs/compute2
tail -n 80 logs/compute2/<jobid>.gptoss_chat_once.out
tail -n 80 logs/compute2/<jobid>.gptoss_chat_once.err
```

Check result:

```bash
python -m json.tool results/<result-file>.json | less
```

Note: if `squeue` says an old job id is invalid, it may simply have completed
and left the active queue. Check `results/` and `logs/compute2/`.

## Suggested Manual Multi-Run Flow

Until there is a dedicated multi-pass pipeline script, run each pass with
`compute2_gptoss_chat_once.sbatch` and feed the previous JSON into the next
prompt.

Suggested result naming:

```text
results/capability_pipeline/<task_id>.run1.intent_input.json
results/capability_pipeline/<task_id>.run2.transform_externality.json
results/capability_pipeline/<task_id>.run3.generated_capabilities.json
results/capability_pipeline/<task_id>.run4.normalized_capabilities.json
results/capability_pipeline/<task_id>.run5.dependency_graph.json
results/capability_pipeline/<task_id>.run6.validation.json
```

Suggested prompt wrapper for every pass:

```text
ORIGINAL_USER_REQUEST_START
...
ORIGINAL_USER_REQUEST_END

PREVIOUS_PASS_OUTPUT_START
...
PREVIOUS_PASS_OUTPUT_END

PASS_INSTRUCTIONS_START
...
PASS_INSTRUCTIONS_END
```

## Existing GPT-OSS Model Notes

Regular base model:

```text
openai/gpt-oss-20b
```

Trace-decomposition tuned LoRA adapter on Storage3:

```text
/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/checkpoints/gpt-oss-20b-trace-decomp-lora-v1
```

Prior observation:

```text
For the A-E decomposition prompts, base GPT-OSS gave stronger, more specific
decompositions than the trace-decomp tuned adapter. The tuned adapter parsed,
but outputs were more generic.
```

For this new capability-planning direction, start with regular base GPT-OSS
unless there is a specific reason to test the tuned adapter.

## What To Build Next

Recommended next implementation:

```text
scripts/run_gptoss_capability_pipeline.py
```

Responsibilities:

```text
1. Read JSONL tasks.
2. Run GPT-OSS pass 1 through pass 5.
3. Parse compact JSON after each pass.
4. Validate JSON schema after each pass.
5. Retry repair if parsing fails.
6. Run structural validator.
7. Write one JSONL record per task with all pass outputs.
```

Suggested output record:

```json
{
  "task_id": "...",
  "original_user_request": "...",
  "pass_1_intent_input": {},
  "pass_2_transform_externality": {},
  "pass_3_generated_capabilities": {},
  "pass_4_normalized_capabilities": {},
  "pass_5_dependency_graph": {},
  "pass_6_validation": {},
  "valid": true
}
```

The first version can use the same model loading style as:

```text
scripts/run_gptoss_input_audit_batch.py
scripts/run_gptoss_chat_once.py
```

Do not implement tool matching yet. Save that for after capability-plan quality
is good enough.
