# Approach

Goal: surpass the TaskLAMA paper on structured complex task decomposition while adding a decomposition gate for cases where subtasks are not useful.

The TaskLAMA paper defines Structured Complex Task Decomposition as producing a DAG of steps and temporal dependencies. Its abstract notes that LLMs produce reasonable step lists but still struggle on pairwise temporal dependencies, so the pipeline here treats node generation and edge repair as separate controllable stages.

Pipeline:

1. Gate the request as `decompose` or `no_decomposition`.
2. Generate concise verb-led subtasks in JSON.
3. Generate temporal dependencies over subtask ids.
4. Normalize the graph by removing self-loops, duplicate edges, cycles, and transitively redundant edges.
5. Score node and edge quality against TaskLAMA-style metrics.

Ways to beat the paper:

- Use `Qwen/Qwen3-14B` as the current base model. If Qwen3's reasoning style makes JSON formatting brittle, fall back to `Qwen/Qwen2.5-14B-Instruct`, whose model card emphasizes instruction following and structured outputs.
- Fine-tune with LoRA on TaskLAMA JSON outputs rather than relying only on zero-shot prompting.
- Add synthetic and curated atomic negatives so the model learns when not to decompose.
- Use graph post-processing to enforce DAG validity.
- Add a second pass for edge prediction: after generating nodes, classify each candidate ordered pair as before/not-before, then reduce the graph.
- Use self-consistency at inference: sample several decompositions, cluster near-duplicate nodes, and majority-vote pairwise edges.
- Report TaskLAMA test metrics separately from gate metrics so the comparison to the paper remains clean.

Milestones:

- M0: reproduce TaskLAMA data preparation and baseline evaluation on a small subset.
- M1: run Qwen 14B zero-shot on TaskLAMA test and record node/edge metrics.
- M2: LoRA fine-tune on TaskLAMA train and validation.
- M3: add edge-classifier second pass and self-consistency decoding.
- M4: add external procedural data and atomic negatives, then rerun TaskLAMA, gate, and OOD evals.
