# Capability Mapping Planner / GPT-OSS Training Handoff

Last updated: 2026-07-03 02:59 America/Chicago

## Goal

The project goal is to make a GPT-OSS-based capability planner that is useful for
tool-calling and agent benchmarks such as BFCL, API-Bank, Tau/ToolSandbox-style
customer-service tasks, and general user-request planning.

The core planner problem is not just JSON formatting. GPT-OSS was brittle at:

- input/source availability auditing
- missing-input detection
- current/external information routing
- capability decomposition and dependency order
- tool/no-tool decisions
- tool call count and order, especially repeated/parallel calls

The current approach is:

1. Use deterministic/rules-first planner logic where possible.
2. Generate a large, diverse synthetic capability-mapping SFT dataset.
3. Train GPT-OSS on that dataset so it learns the capability mapping behavior.
4. Evaluate on BFCL/API-Bank/Tau-style routing metrics after training.

## Local Workspace

Local repo:

```bash
cd "/Users/deepshikhashrestha/Documents/Project I"
```

Important local files created/updated:

```text
scripts/generate_gpt55_capability_mapping_sft.py
tests/test_gpt55_capability_mapping_sft.py
data/gpt55_capability_mapping_sft_v2/
data/gpt55_capability_mapping_sft_v3/
ris/compute2_train_gptoss_capability_mapping_full.sbatch
ris/compute2_train_gptoss_capability_mapping_lora.sbatch
scripts/submit_compute2_capability_full.sh
scripts/submit_compute2_capability_lora.sh
```

The phrase "GPT-5.5-style teacher" in earlier notes means synthetic teacher labels
based on stronger-model planner judgment. It is not actual API-distilled GPT-5.5
output.

## Dataset

Next-run dataset path:

```text
data/gpt55_capability_mapping_sft_v3/
```

The active job `1928311` is still training from the earlier `v2` dataset. Use
`v3` for the next LoRA/full run; the local submit and Slurm defaults have been
updated accordingly.

Files:

```text
all.records.jsonl
train.records.jsonl
validation.records.jsonl
train.sft.jsonl
validation.sft.jsonl
summary.json
```

Stats:

```text
total_records: 50000
train_records: 45000
validation_records: 5000
task_families: 39
operations: 69
tool_call rows: 15384
ask_user / missing-input rows: 6411
unique_requests: 49936
unique_request_rate: 99.87%
```

Main task families include:

```text
missing_input
missing_attachment_reference
ambiguous_request
pasted_text_transform
text_classification
structured_data_analysis
chart_generation
attached_document
multi_attachment_compare
media_transcription
current_fact
url_task
mixed_provided_and_current
code_edit
code_generation
repo_search
local_file_summary
image_understanding
image_edit_generation
unsupported_measurement
tool_binding_single
tool_binding_multiple
tool_binding_parallel
tool_parallel_multiple
tool_batch_capable
tool_missing_required_slot
tool_customer_service
tool_ecommerce_workflow
tool_api_crud
tool_irrelevance
tool_false_no_call_hard
tool_parameter_not_missing
tool_semantic_near_miss_hard
tool_overcall_related_hard
tool_keyword_no_call_hard
```

The new hard families target observed BFCL-style failures:

```text
false no-call on actionable requests
provided parameter names treated as missing inputs
semantic cousin tool selection
related-but-unrequested over-calls
keyword-overlap no-call gating
parallel-multiple repeated group ordering
batch vs repeated single-call contrast
```

Regenerate dataset locally:

```bash
python3 scripts/generate_gpt55_capability_mapping_sft.py \
  --count 50000 \
  --out-dir data/gpt55_capability_mapping_sft_v3 \
  --validation-fraction 0.1 \
  --seed 57
```

Validate locally:

```bash
PYTHONPATH=src:. python3 -m pytest tests/test_gpt55_capability_mapping_sft.py -q
PYTHONPATH=src:. python3 -m pytest tests -q
```

Last focused test result before handoff:

```text
148 passed
```

## Compute2 Access

Use SSH:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu
```

If SSH times out, it is usually VPN/routing. The login nodes resolve to internal
RIS IPs:

```text
c2-login-001.ris.wustl.edu -> 10.25.20.4
c2-login-002.ris.wustl.edu -> 10.25.20.5
```

Useful environment setup on Compute2:

```bash
export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
export WF=$STOR/workflow
export APIBANK=$STOR/external/DAMO-ConvAI/api-bank
export BR=$WF/benchmark_results

cd "$PROJ"
source "$PROJ/.venv-c2/bin/activate"

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
export PYTHONPATH=$WF/scripts:$APIBANK:${PYTHONPATH:-}
```

Slurm commands:

```bash
squeue -u dshrestha -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"
squeue -j JOBID -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"
sacct -j JOBID --format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocTRES%60 -P
sbatch path/to/job.sbatch
scancel JOBID
```

## Training Attempts

### Full-FSDP Attempt

A full-parameter FSDP job was prepared and submitted:

```text
job_id: 1928295
job_name: gptoss_capmap_full
script: ris/compute2_train_gptoss_capability_mapping_full.sbatch
```

It requested 4x H100 and stayed pending with reason `Resources`.
It was canceled when the user decided to run LoRA first.

Cancel command used:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu \
  'scancel 1928295 2>/dev/null || true'
```

The full-FSDP script still exists for a future stronger run:

```text
ris/compute2_train_gptoss_capability_mapping_full.sbatch
scripts/submit_compute2_capability_full.sh
```

### Current LoRA Run

Current active job:

```text
job_id: 1928311
job_name: gptoss_capmap_lora
state at handoff: RUNNING
node: c2-gpu-004
allocation: 1x H100, 160G RAM, 8 CPU
script: ris/compute2_train_gptoss_capability_mapping_lora.sbatch
```

Latest check on 2026-07-03 02:59 America/Chicago:

```text
state: RUNNING
elapsed: 2:27:58
latest eval_loss: 3.7857 at epoch ~0.14
latest observed epoch: ~0.18
```

Output adapter path:

```text
$PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2
```

It is continuing from:

```text
$PROJ/checkpoints/gpt-oss-20b-taskdecomp-lora
```

Training config:

```text
model: openai/gpt-oss-20b
tuning_method: lora
max_seq_length: 2048
epochs: 1.0
learning_rate: 1e-5
gradient_accumulation_steps: 16
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
validation rows used during training: 1000
trainable params: 7,962,624
trainable percent: 0.0381
```

Preprocessing result:

```text
train input_rows: 45000
train kept_rows: 45000
validation input_rows: 1000
validation kept_rows: 1000
avg train length: 647.86 tokens
avg train label tokens: 358.99
```

Latest observed progress:

```text
elapsed: 1:53:28
progress: around checkpoint-400 / step ~420 of 2813
epoch: around 0.14
```

Observed validation loss:

```text
epoch ~0.04: eval_loss 4.3197
epoch ~0.07: eval_loss 3.9945
epoch ~0.11: eval_loss 3.8714
```

Saved checkpoints at handoff:

```text
$PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2/checkpoint-200
$PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2/checkpoint-300
$PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2/checkpoint-400
```

Logs:

```text
$PROJ/logs/compute2/1928311.gptoss_capmap_lora.out
$PROJ/logs/compute2/1928311.gptoss_capmap_lora.err
```

Check current LoRA job:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
export PROJ=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss
squeue -j 1928311 -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R"
sacct -j 1928311 --format=JobID,JobName%30,State,Elapsed,MaxRSS,AllocTRES%60 -P 2>/dev/null | tail -20
tail -80 $PROJ/logs/compute2/1928311.gptoss_capmap_lora.out
tail -80 $PROJ/logs/compute2/1928311.gptoss_capmap_lora.err
find $PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2 -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail
'
```

Submit/re-submit LoRA job from local machine:

```bash
cd "/Users/deepshikhashrestha/Documents/Project I"
./scripts/submit_compute2_capability_lora.sh
```

The helper syncs:

```text
scripts/train_sft.py
scripts/generate_gpt55_capability_mapping_sft.py
ris/compute2_train_gptoss_capability_mapping_lora.sbatch
data/gpt55_capability_mapping_sft_v3/ if missing or if SYNC_DATA=1
```

Force data re-sync:

```bash
SYNC_DATA=1 ./scripts/submit_compute2_capability_lora.sh
```

## What To Do Next

1. Let job `1928311` finish unless validation loss or logs show a failure.
2. When complete, confirm final adapter files exist:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu '
export PROJ=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss
ls -lh $PROJ/checkpoints/gpt-oss-20b-capability-mapping-lora-v2
tail -120 $PROJ/logs/compute2/1928311.gptoss_capmap_lora.out
tail -120 $PROJ/logs/compute2/1928311.gptoss_capmap_lora.err
'
```

3. Evaluate the trained adapter against capability-planning and tool-routing
   benchmarks already in the repo:

```text
BFCL planner/tool routing
API-Bank routing
Tau/ToolSandbox-style workflow routing
gold/holdout capability planning sets
```

4. Compare against:

```text
rules-first planner
old GPT-OSS planner LoRA
current ensemble selector
synthetic teacher sample behavior
```

5. Inspect failures by category:

```text
wrong_final_intent
missed_missing_input
invented_file_reading
missed_current_info
used_tool_name
used_agent_name
vague_capability
bad_dependency
over_decomposed
under_decomposed
wrong_tool_count
wrong_tool_order
missed_no_call
missed_batch_tool
missing_required_slot_not_detected
```

## Important Caveats

- This is synthetic teacher-style data, not API-distilled model output.
- The LoRA run is lighter and faster, but less powerful than the full-FSDP run.
- The full-FSDP script is ready, but it needs a 4x H100 allocation and may sit
  pending for resources.
- The current LoRA dataset is broad and diverse, but final usefulness must be
  judged on held-out public-style benchmarks, especially BFCL/API-Bank/Tau.
