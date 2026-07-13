# Dataset Map

Primary benchmark:

- TaskLAMA (`https://storage.googleapis.com/gresearch/tasklama/tasklama.zip`): human-annotated structured complex task decomposition examples with assumptions, substeps, and temporal dependencies. This is the direct benchmark from Yuan et al. 2023.

High-value expansion candidates:

- MSComplexTasks: Microsoft complex task decomposition benchmark. Use it as an out-of-domain evaluation and, if licensing allows, a supplementary supervised source.
- WikiHow / WikiHowToImprove: broad procedural how-to steps. Use for weak supervision after converting ordered lists into DAG chains, but keep separate from TaskLAMA evaluation because it lacks rich dependency annotation.
- ProScript / DeScript / Script learning datasets: everyday event scripts with temporal/event structure. Useful for edge prediction pretraining, less directly aligned to user task decomposition.
- TaskBench-style agent/task automation datasets: useful for workflows with tool-like subgoals and dependency graphs.
- Math and reasoning decomposition corpora such as GSM8K rationales, StrategyQA decompositions, and decomposition-prompt datasets: useful mostly for the no-decomposition gate and domain transfer, not direct SCTD scoring.
- Atomic negative sets: single-step QA, translation, extraction, classification, and formatting tasks. These are needed because TaskLAMA contains only complex tasks, so it cannot teach the model when not to decompose.

Trace-derived decomposition dataset:

- Build script: `scripts/prepare_trace_decomp_dataset.py`
- Default output: `data/trace_decomp`
- Compute2/Storage3 output used for shared runs:
  `/storage3/fs1/yvorobeychik/Active/dshrestha/taskdecomp-gpt-oss/data/trace_decomp`
- Public sources currently supported:
  - `nebius/SWE-agent-trajectories`: coding-agent traces with pass/fail verifier signals.
  - `nebius/SWE-rebench-openhands-trajectories`: OpenHands/SWE-rebench trajectories with resolved/test signals.
  - `osunlp/Mind2Web`: human web-action demonstrations used as successful workflow decompositions.

The trace-derived files are:

- `train.records.jsonl` / `validation.records.jsonl`: canonical records with task, trace summary, verification, failure analysis, and corrected decomposition target.
- `train.sft.jsonl` / `validation.sft.jsonl`: chat-format SFT rows for training a decomposition repair/planning model.
- `annotation_queue.jsonl`: unresolved trajectories needing human or judge-model labels for `bad_decomposition`, `failure_reason`, and corrected decomposition.
- `summary.json`: counts, source list, split settings, and source errors.

Example local build:

```bash
PYTHONPATH=src python3 scripts/prepare_trace_decomp_dataset.py \
  --out-dir data/trace_decomp \
  --sources swe-agent openhands mind2web \
  --max-records-per-source 100 \
  --validation-ratio 0.05
```

Example compute2/Storage3 build:

```bash
ssh -o BatchMode=yes dshrestha@c2-login-001.ris.wustl.edu
export STOR=/storage3/fs1/yvorobeychik/Active/dshrestha
export PROJ=$STOR/taskdecomp-gpt-oss
source $PROJ/.venv-c2/bin/activate
export HF_HOME=$PROJ/.hf_cache
export HF_DATASETS_CACHE=$PROJ/.hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$PROJ/.hf_cache/hub
cd $PROJ
PYTHONPATH=src python3 scripts/prepare_trace_decomp_dataset.py \
  --out-dir $PROJ/data/trace_decomp \
  --sources swe-agent openhands mind2web \
  --max-records-per-source 1000 \
  --validation-ratio 0.05
```

Recommended split strategy:

- Train: TaskLAMA train plus atomic negatives and weak external procedural examples.
- Validation: TaskLAMA validation plus a balanced atomic-negative validation file.
- Test: keep TaskLAMA test untouched for paper comparison.
- Gate test: create a separate binary benchmark with 50% TaskLAMA complex tasks and 50% atomic tasks.
- OOD test: MSComplexTasks and WikiHow-derived workflows, reported separately.

Target metrics:

- Node quality: relaxed Hungarian F1/F2 with ROUGE-L similarity, matching TaskLAMA's spirit.
- Edge quality: edge F1 after node matching, plus pairwise dependency accuracy for comparison to the paper.
- Gate quality: decomposition/no-decomposition accuracy, precision, recall, and abstention rate if a confidence layer is added.
