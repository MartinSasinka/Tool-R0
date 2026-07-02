# Trajectory analysis (`vizualisation/`)

Research-quality **behavioral trajectory analysis** for curriculum GRPO checkpoints on NESTFUL-style tool-calling tasks.

This module studies **tool-use trajectory evolution** and **behavioral trajectory shift** across checkpoints. It does **not** claim access to internal model knowledge—only observable outputs on a fixed evaluation set.

## Purpose

Compare model-generated tool-use trajectories across:

- `base` (baseline)
- `stage1_1call` … `stage4_4call` (curriculum LoRA checkpoints)
- optional Tool-R0 baseline
- `gold` reference trajectories from NESTFUL data

## Requirements

```bash
pip install -r vizualisation/requirements-viz.txt
# optional: pip install umap-learn
```

Python 3.10+, runs on CPU (no GPU required).

## Quick start

1. Edit [`configs/trajectory_analysis_config.json`](configs/trajectory_analysis_config.json) — point `input_predictions` at your eval JSONL files.

2. Run the full pipeline:

```bash
CONFIG=vizualisation/configs/trajectory_analysis_config.json \
  bash vizualisation/run_analysis.sh
```

Or step-by-step:

```bash
python vizualisation/scripts/collect_trajectories.py --config vizualisation/configs/trajectory_analysis_config.json
python vizualisation/scripts/inspect_run.py --run_dir vizualisation/runs/qwen3_4b_lora_curriculum_run1
```

## Input formats

### Ideal JSONL (planning format)

```json
{
  "sample_id": "nestful-0001",
  "checkpoint": "stage2_2call",
  "input": "...",
  "tools": "[...]",
  "gold_output": "[...]",
  "gold_answer": "...",
  "prediction_raw": "...",
  "prediction_output": [...],
  "prediction_answer": "..."
}
```

Variants: `output` instead of `prediction_output`; `answer` / `final_answer`; raw completion only (best-effort JSON parse).

### Multiturn eval format (current curriculum eval)

From [`nestful_evaluation/run.py`](../nestful_evaluation/run.py) / [`curricullum/evaluation/`](../curricullum/evaluation/):

- `task_id` → `sample_id`
- `predicted_calls` → `prediction_output` (executed trace; may be empty)
- `predicted_final` → `prediction_answer`
- `gold_calls` joined with [`eval/data/NESTFUL-main/data_v2/nestful_data.jsonl`](../eval/data/NESTFUL-main/data_v2/nestful_data.jsonl)

**Caveat:** multiturn eval measures **executed** multi-turn traces, not single-shot JSON plan dumps from GRPO training. Empty `predicted_calls` when the model skips tools is expected and affects structural metrics.

## Rollout policy

Default raw dataset: **`all_rollouts`** — every `(sample_id, checkpoint, rollout_idx)` is kept.

Checkpoint summaries also report:

| Policy | Description |
|--------|-------------|
| `all_rollouts` | Mean over all rollout rows |
| `best_score` | Highest `score` rollout per sample (**primary report**) |
| `rollout_idx_0` | First rollout only |
| `mean_over_rollouts` | Mean of numeric metrics across rollouts |

Configure via `rollout_policy` and `summary_aggregation` in config JSON.

## Output layout

```
vizualisation/runs/<run_name>/
  config.json
  trajectories_raw.jsonl
  trajectories_canonical.jsonl
  trajectory_metrics.jsonl
  feature_matrix.csv
  feature_matrix_standardized.csv
  reducer_metadata.json
  embedding_2d.csv
  checkpoint_summary.csv
  gain_summary.csv
  distance_to_gold.csv
  shift_alignment.csv
  error_transitions.csv
  stability_metrics.csv
  centroid_distances.csv
  figures/
  reports/analysis_report.md
  reports/analysis_summary.json
```

## Interpretation guide

### Skill profiles

Component scores in `[0, 1]` aligned with GRPO reward structure: JSON format, call count, tools, labels, argument keys/values, references, dependency depth, final answer.

Answers: *“What abilities does each checkpoint improve on average?”*

### Stage gains

Delta of mean component scores between consecutive checkpoints (`stageN - stageN-1`).

Answers: *“What did this curriculum stage change?”*

### Distance to gold

L2 distance in **standardized feature space** plus canonical edit distance to gold trajectories.

Answers: *“Are model trajectories moving closer to correct trajectories?”*

### Centroid shift & alignment

Checkpoint centroids in feature space and exploratory 2D PCA/UMAP space; cosine alignment of per-sample shifts toward gold.

### Embedding maps

**Exploratory only.** 2D layout must not be used alone for quantitative claims.

## Publication caveats

- These visualizations analyze behavioral tool-use trajectories, not direct internal model knowledge.
- 2D projections are exploratory; quantitative claims use component metrics and distances in the standardized feature space.
- Observed shifts reflect model outputs on a fixed evaluation set.

## Limitations

- Train/eval format mismatch (JSON plan training vs multiturn Tool-R0 eval).
- Structural metrics are weak when models answer without tool calls (`predicted_calls=[]`).
- `stage4_4call` optional until checkpoint + eval exist.
- Error transitions paired by `rollout_idx` are approximate; sample-level transitions use `summary_aggregation`.

## Mode 2 (deferred)

Collecting new predictions from models via subprocess is **not implemented**. Use `--collect_new` to see a reminder; point config at existing JSONL from eval scripts.
