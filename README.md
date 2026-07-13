# Task Decomposition with Qwen 14B

This project builds a model pipeline for decomposing complex tasks into subtasks and temporal dependencies, while identifying requests where decomposition is not useful.

The direct benchmark is TaskLAMA from Yuan et al. 2023. Their paper frames the task as producing a directed acyclic graph over steps and notes that temporal dependency prediction remains the hardest part, so this repo separates gating, node generation, edge generation, graph repair, and evaluation.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 scripts/prepare_data.py
```

Build the trace-derived decomposition dataset from public agent/web trajectories:

```bash
PYTHONPATH=src python3 scripts/prepare_trace_decomp_dataset.py \
  --out-dir data/trace_decomp \
  --sources swe-agent openhands mind2web \
  --max-records-per-source 100
```

On compute2/Storage3, the shared output path is:
`/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/data/trace_decomp`.

Run one prediction:

```bash
python -m taskdecomp.pipeline \
  --model Qwen/Qwen3-14B \
  --task "plan and execute a small conference workshop"
```

Run the GPT-OSS capability-planning pipeline from the compute2 handoff:

```bash
PYTHONPATH=src python scripts/run_gptoss_capability_plan.py \
  --model openai/gpt-oss-20b \
  --task "Summarize the attached policy memo and flag missing evidence." \
  --attachments-metadata '[{"name":"policy memo","format":"pdf","available":true}]' \
  --output results/capability_plan_once.json
```

Batch mode expects JSONL rows with `task`, `request`, or `user_request`; optional
fields are `context`, `attachments_metadata`, `attachments`, `id`, `row_index`,
and `label`.
It runs the five GPT-OSS planning passes, then applies the deterministic
structural validator:

```bash
PYTHONPATH=src python scripts/run_gptoss_capability_plan.py \
  --model openai/gpt-oss-20b \
  --input data/manual/gptoss_base_tasks_a_to_e.jsonl \
  --output results/gptoss_capability_plan.jsonl
```

Evaluate it against the seed gold set:

```bash
PYTHONPATH=src python scripts/run_gptoss_capability_plan.py \
  --model openai/gpt-oss-20b \
  --input data/capability_planning/gold_40.jsonl \
  --output results/gptoss_capability_plan_gold40.jsonl

PYTHONPATH=src python scripts/eval_capability_plans.py \
  --gold data/capability_planning/gold_40.jsonl \
  --predictions results/gptoss_capability_plan_gold40.jsonl \
  --metrics-out results/gptoss_capability_plan_gold40.metrics.json \
  --csv-out results/gptoss_capability_plan_gold40.review.csv
```

The CSV is spreadsheet-friendly and includes `run1_ok`, `run2_ok`, `run3_ok`,
`graph_ok`, `main_failure_type`, `notes`, and a blank `human_acceptable` column
for manual plan-acceptance review.

Generate and score the mixed public-style capability-planning eval. This set
combines CLINC/MASSIVE/Super-NaturalInstructions/BFCL/API-Bank/SWE-bench-style
cases with adversarial source-span wrappers, so words inside provided source text
like `current`, `latest`, URLs, SQL, and TODOs should not accidentally trigger
web retrieval, code generation, or repo search:

```bash
PYTHONPATH=src python scripts/generate_capability_public_mix.py \
  --output data/capability_planning/public_mix_wrapped_60.jsonl

PYTHONPATH=src python scripts/run_gptoss_capability_plan.py \
  --planner-mode rules_first \
  --input data/capability_planning/public_mix_wrapped_60.jsonl \
  --output results/rules_first_capability_plan_public_mix_wrapped60.jsonl

PYTHONPATH=src python scripts/eval_capability_plans.py \
  --gold data/capability_planning/public_mix_wrapped_60.jsonl \
  --predictions results/rules_first_capability_plan_public_mix_wrapped60.jsonl \
  --metrics-out results/rules_first_capability_plan_public_mix_wrapped60.metrics.json \
  --csv-out results/rules_first_capability_plan_public_mix_wrapped60.review.csv
```

The generator does not download public datasets by itself. To include local
public-dataset exports, first convert them into this repo's capability-planning
gold JSONL schema, then pass one or more `--external-jsonl` files.

Adapt benchmark exports into the same gold schema before scoring the planner.
The adapter accepts common JSON/JSONL shapes from intent, source-task, tool/code,
repo, and web benchmarks, then labels planner properties such as input format,
external action type, required capabilities, and dependency hints:

```bash
PYTHONPATH=src python scripts/adapt_capability_benchmark.py \
  --input data/benchmarks/capability_benchmark_adapter_smoke.jsonl \
  --dataset auto \
  --output data/capability_planning/benchmark_adapter_smoke.jsonl

PYTHONPATH=src python scripts/run_gptoss_capability_plan.py \
  --planner-mode rules_first \
  --input data/capability_planning/benchmark_adapter_smoke.jsonl \
  --output results/rules_first_capability_plan_benchmark_adapter_smoke.jsonl

PYTHONPATH=src python scripts/eval_capability_plans.py \
  --gold data/capability_planning/benchmark_adapter_smoke.jsonl \
  --predictions results/rules_first_capability_plan_benchmark_adapter_smoke.jsonl \
  --metrics-out results/rules_first_capability_plan_benchmark_adapter_smoke.metrics.json \
  --csv-out results/rules_first_capability_plan_benchmark_adapter_smoke.review.csv
```

Use `--dataset clinc`, `--dataset massive`, `--dataset superni`,
`--dataset bfcl`, `--dataset api_bank`, `--dataset swe_bench`,
`--dataset repobench`, or `--dataset webarena` when the filename is not enough
to infer the benchmark family.

For repeatable benchmark suites, use a JSON config that lists benchmark slices
and lets the runner execute adapter, planner, scorer, and aggregate reporting in
one pass:

```bash
PYTHONPATH=src python scripts/prepare_capability_benchmark_slices.py

PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_smoke.json
```

The suite runner writes adapted gold rows under
`data/capability_planning/benchmark_suite*/`, per-benchmark predictions and
review CSVs under `results/capability_benchmark_suite*/`, plus aggregate metrics
and an aggregate review CSV.

Prepared public-slice configs are written by
`scripts/prepare_capability_benchmark_slices.py`:

```bash
PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_source_intent.json

PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_tool_code.json

PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_all.json
```

For a larger public-slice gate, generate a labelled suite so it does not
overwrite the small public-slice outputs:

```bash
PYTHONPATH=src python scripts/prepare_capability_benchmark_slices.py \
  --output-root data/benchmarks/public_slices_scaled \
  --config-dir configs \
  --suite-label scaled \
  --clinc-limit-per-intent 20 \
  --massive-limit-per-label 25 \
  --superni-limit-per-task 25 \
  --bfcl-limit 50 \
  --swe-limit 25 \
  --webarena-limit 25

PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_all_scaled.json
```

To create the disjoint scaled holdout v2 gate, use alternate CLINC/MASSIVE
splits plus offsets for the other public slices:

```bash
PYTHONPATH=src python scripts/prepare_capability_benchmark_slices.py \
  --output-root data/benchmarks/public_slices_scaled_holdout_v2 \
  --config-dir configs \
  --suite-label scaled_holdout_v2 \
  --clinc-split validation \
  --clinc-limit-per-intent 20 \
  --massive-split train \
  --massive-limit-per-label 25 \
  --superni-limit-per-task 25 \
  --superni-offset-per-task 25 \
  --bfcl-limit 50 \
  --bfcl-offset 50 \
  --swe-limit 25 \
  --swe-offset 25 \
  --webarena-limit 25 \
  --webarena-offset 28 \
  --local-controls-variant holdout_v2

PYTHONPATH=src python scripts/run_capability_benchmark_suite.py \
  --config configs/capability_benchmark_suite_all_scaled_holdout_v2.json
```

After a suite run, create a failure-review scaffold with owner buckets for
`adapter_mislabeled`, `planner_routing_or_capability`, and
`scorer_too_strict_or_loose`:

```bash
python scripts/review_capability_benchmark_failures.py \
  --review-csv results/capability_benchmark_benchmark_suite_all/aggregate.review.csv \
  --out-csv results/capability_benchmark_benchmark_suite_all/failure_review.csv \
  --out-json results/capability_benchmark_benchmark_suite_all/failure_review.json
```

On compute2, use the H100-targeted Slurm handoff. By default it runs the 40-case
gold eval and writes predictions, metrics, and review CSV files under `results/`:

```bash
sbatch ris/compute2_gptoss_capability_plan.sbatch
```

Evaluate predictions:

```bash
python scripts/predict_jsonl.py \
  --model Qwen/Qwen3-14B \
  --input data/processed/test.eval.jsonl \
  --output results/predictions.jsonl

python -m taskdecomp.evaluate \
  --predictions results/predictions.jsonl \
  --references data/processed/test.eval.jsonl \
  --out results/metrics.json
```

Fine-tune:

```bash
python scripts/train_sft.py \
  --model Qwen/Qwen3-14B \
  --output-dir checkpoints/qwen3-14b-taskdecomp-lora
```

Full-parameter GPT-OSS SFT on AgentLM's AgentInstruct source data:

```bash
python scripts/prepare_agentinstruct_sft.py \
  --out-dir data/agentinstruct_sft \
  --loss-policy true-only

torchrun --standalone --nproc_per_node=4 scripts/train_sft.py \
  --model openai/gpt-oss-20b \
  --train data/agentinstruct_sft/train.jsonl \
  --validation data/agentinstruct_sft/validation.jsonl \
  --output-dir checkpoints/gpt-oss-20b-agentinstruct-full-fsdp \
  --tuning-method full \
  --max-seq-length 4096 \
  --learning-rate 5e-6 \
  --num-train-epochs 3.0 \
  --gradient-accumulation-steps 8 \
  --optim paged_adamw_8bit \
  --no-load-in-4bit \
  --no-gradient-checkpointing \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap GptOssDecoderLayer \
  --fsdp-activation-checkpointing
```

Fallback model: if Qwen3's reasoning style makes JSON formatting brittle, use
`Qwen/Qwen2.5-14B-Instruct` with the same scripts.

## RIS

RIS has both Compute1 and Compute2 environments. Compute1 uses IBM LSF/`bsub`; Compute2 uses Slurm/`sbatch`. Templates for both are in `ris/`.

The Compute1 templates are prefilled for:

- `PROJECT_DIR=/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss`
- `#BSUB -G compute-yvorobeychik`
- Docker image `nvcr.io/nvidia/pytorch:25.04-py3`

Compute1:

```bash
bsub < ris/compute1_prepare.bsub
bsub < ris/compute1_train.bsub
bsub < ris/compute1_eval.bsub
```

Compute2:

```bash
sbatch ris/compute2_prepare.sbatch
sbatch ris/compute2_train.sbatch
sbatch ris/compute2_eval.sbatch
```

Compute2 full GPT-OSS AgentInstruct job:

```bash
sbatch ris/compute2_train_gptoss_agentinstruct_full.sbatch
```

API-Bank routed ensemble:

```bash
APIBANK_ENSEMBLE_LIMIT=10 \
APIBANK_ENSEMBLE_VALIDATOR_MODE=always \
APIBANK_ENSEMBLE_VALIDATOR_BASE=Team-ACE/ToolACE-8B \
sbatch compute2/compute2_apibank_ensemble.sbatch
```

The ensemble uses the task-decomposition LoRA for private planning, the
Nemotron/BFCL LoRA as the executor, deterministic schema validation as the hard
gate, and the BFCL-tuned 8B model as a validator/repair judge rather than a
general-purpose agent.

## References

- TaskLAMA paper: https://arxiv.org/abs/2308.15299
- TaskLAMA data: https://storage.googleapis.com/gresearch/tasklama/tasklama.zip
- Qwen3-14B model: https://huggingface.co/Qwen/Qwen3-14B
- Qwen2.5-14B-Instruct fallback: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
