# VAKRA Pipeline Test Status

Last updated: 2026-06-29 03:56 CDT

## Local Test

Environment:

- VAKRA container `capability_1_bi_apis` is running from `benchmark_environ`.
- Local VAKRA test data currently includes `beer_factory`.
- Local Python has no CUDA device available.
- No local GPT-OSS LoRA adapter checkpoints were found under the active workspace roots.

Command:

```bash
cd external/vakra
.venv310/bin/python ../../compute2/vakra_mcp_ensemble_runner.py \
  --vakra-root . \
  --capability-id 1 \
  --domain beer_factory \
  --agent scripted-smoke \
  --max-samples 1 \
  --max-steps 3 \
  --output-dir output/custom_vakra_pipeline_baseline
.venv310/bin/python validate_output.py output/custom_vakra_pipeline_baseline
```

Result:

- Passed VAKRA output validation.
- Output file: `external/vakra/output/custom_vakra_pipeline_baseline/beer_factory.json`
- This confirms the MCP bridge, data access, sequence serialization, and VAKRA output schema.

## Ensemble Attempt

Command:

```bash
cd external/vakra
.venv310/bin/python ../../compute2/vakra_mcp_ensemble_runner.py \
  --vakra-root . \
  --capability-id 1 \
  --domain beer_factory \
  --agent ensemble \
  --max-samples 1 \
  --max-steps 2 \
  --output-dir output/custom_vakra_ensemble_attempt
```

Result:

- Installed missing local Python packages: `peft`, `accelerate`.
- The run reached GPT-OSS adapter loading.
- It stopped because the default compute2 checkpoint path is not present locally:

```text
/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/checkpoints/gpt-oss-20b-nemotron-agentic-bfcl-lora-2xa40-noGC1024
```

The exact error was:

```text
ValueError: Can't find 'adapter_config.json'
```

## Current Blocker

True ensemble scoring needs to run where these are available:

- GPT-OSS executor LoRA
- GPT-OSS planner LoRA
- GPT-OSS recovery LoRA
- CUDA-capable GPU
- VAKRA data mounted into the VAKRA image

That is the compute2 job path prepared in:

- `compute2/compute2_container_probe.sbatch`
- `compute2/build_vakra_ensemble_image.sh`
- `compute2/compute2_vakra_ensemble.sbatch`

## Next Command on Compute2

```bash
sbatch compute2/compute2_container_probe.sbatch
```

After the probe confirms Apptainer/Singularity/Docker availability:

```bash
sbatch --export=ALL,VAKRA_IMAGE=<registry>/vakra-ensemble:amd64,DOMAIN=beer_factory,CAPABILITY_ID=1,MAX_SAMPLES=1 \
  compute2/compute2_vakra_ensemble.sbatch
```
