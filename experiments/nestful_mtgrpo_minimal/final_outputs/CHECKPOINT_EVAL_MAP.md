# Checkpoint → Eval → Metrics Map (NESTFUL MT-GRPO)

Audit report generated from on-disk artifacts. **No code or scoring changes.**

## Root cause of the `0.48898` vs `stage3-e3` confusion

W&B run names for **in-curriculum** `rollout_eval` follow this pattern (see `run_curriculum.sh` line 373):

```text
WANDB_RUN_NAME = eval-stage{EVAL_STAGE}-e{EPOCH}
where EVAL_STAGE = train_stage + 1
```

So **`eval-stage3-e3` means**:
- **Eval subset**: stage 3 → NESTFUL tasks with `num_calls == 3` (407 tasks in this run)
- **Model checkpoint**: **stage 2, epoch 3** (NOT stage 3 epoch 3)
- **`final_answer_pass ≈ 0.488943`** belongs to **train-s2-e3**, as in `consolidated_metrics.json`

The **actual stage 3 epoch 3** model is logged as **`eval-stage4-e3`** (evaluated on 4-call subset, 250 tasks) with **`final_answer_pass = 0.168`**.

| What you might read | W&B name | Actual model ckpt | Eval subset | final_answer_pass |
|---|---|---|---|---|
| "stage3 epoch3" (ambiguous) | `eval-stage3-e3` | **s2e3** | 3-call (407) | **0.488943** |
| "stage3 epoch3" (correct) | `eval-stage4-e3` | **s3e3** | 4-call (250) | **0.168** |

## Naming conventions

| Pattern | Meaning | Set in |
|---|---|---|
| `train-stage{N}-e{E}` | GRPO training, stage N epoch E | `run_curriculum.sh:328` |
| `eval-stage{S}-e{E}` | Rollout eval on **subset S** after training stage (N=S−1) epoch E | `run_curriculum.sh:373` |
| `final_eval_*` | Full NESTFUL benchmark (1861 tasks), Direct or ReAct | `run_curriculum.sh:598` |

## Eval subset sizes (this run)

| eval_subset_stage | Filter | Tasks in metrics.json |
|---|---|---|
| 2 | num_calls==2 | 609 |
| 3 | num_calls==3 | 407 |
| 4 | num_calls==4 | 250 (stage 3 epochs); 32 (stage 4 epoch 1 pilot cap) |
| 5 | num_calls==5 | 173 (stage 4 epoch 2) |
| full | all call counts | 1861 |

## Metric types

- **Training-time** (`strict_gold_trace_pass`, `final_answer_pass`, …): from `mode_rollout_eval`, stored in `epoch_*/eval/metrics.json`, logged to W&B as `eval/*`.
- **Subset official** (`off_*`): official NESTFUL scorer re-run on the same trajectories, same filtered subset (not full 1861).
- **Full official** (`official_*_full`): from `final_eval` on full benchmark, or re-scored predictions.

## Full mapping table

| W&B / log name | Model S | Model E | Eval subset | N tasks | final_answer | strict_pass | off_f1 | full_f1 | paradigm |
|---|---|---|---|---|---|---|---|---|---|
| eval-stage2-e1 | 1 | 1 | 2 | 609 | 0.4335 | 0.3432 | 0.7590 | — | react (rollout_eval) |
| eval-stage2-e2 | 1 | 2 | 2 | 609 | 0.4483 | 0.3612 | 0.7730 | — | react (rollout_eval) |
| eval-stage2-e3 | 1 | 3 | 2 | 609 | 0.4351 | 0.3563 | 0.7590 | — | react (rollout_eval) |
| eval-stage2-e4 | 1 | 4 | 2 | 609 | 0.4696 | 0.3777 | 0.7870 | — | react (rollout_eval) |
| eval-stage3-e1 | 2 | 1 | 3 | 407 | 0.1818 | 0.1499 | 0.6980 | — | react (rollout_eval) |
| eval-stage3-e3 | 2 | 3 | 3 | 407 | 0.4889 | 0.1130 | 0.7980 | — | react (rollout_eval) |
| eval-stage3-e4 | 2 | 4 | 3 | 407 | 0.4939 | 0.0934 | 0.7800 | — | react (rollout_eval) |
| eval-stage4-e1 | 3 | 1 | 4 | 250 | 0.4400 | 0.0120 | 0.5460 | — | react (rollout_eval) |
| eval-stage4-e2 | 3 | 2 | 4 | 250 | 0.3200 | 0.0080 | 0.3960 | — | react (rollout_eval) |
| eval-stage4-e3 | 3 | 3 | 4 | 250 | 0.1680 | 0.0 | 0.3970 | — | react (rollout_eval) |
| eval-stage5-e1 | 4 | 1 | 5 | 32 | 0.1562 | 0.0 | 0.4780 | — | react (rollout_eval) |
| eval-stage5-e2 | 4 | 2 | 5 | 173 | 0.0867 | 0.0 | 0.4730 | — | react (rollout_eval) |
| final_eval_baseline_react | — | — | full | 1861 | — | — | 0.8940 | 0.8940 | react |
| final_eval_stage4_epoch2_react | 4 | 2 | full | 1861 | — | — | 0.1530 | 0.1530 | react |
| baseline_direct | — | — | full | 1861 | — | — | 0.9210 | 0.9210 | direct |
| stage4_epoch2_direct | 4 | 2 | full | 1861 | — | — | 0.9170 | 0.9170 | direct |
| v2_baseline | — | — | full | 1861 | — | — | — | — | react(4-rollout) |
| v2_stage3_epoch1 | 3 | 1 | full | 1861 | — | — | 0.9580 | 0.9580 | react(4-rollout) |
| v2_stage5_epoch2 | 5 | 2 | full | 1861 | — | — | 0.9580 | 0.9580 | react(4-rollout) |

## Per-row detail

### `train-s1-e1 → eval subset 2-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage2-e1`
- **Checkpoint**: `outputs/curriculum/stage_1/checkpoints/adapter_epoch_1`
- **Metrics**: `outputs/curriculum/stage_1/epoch_1/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_1/epoch_1/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage2-e1 uses EVAL subset stage (num_calls==2), NOT model stage. Model is stage 1 epoch 1.

### `train-s1-e2 → eval subset 2-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage2-e2`
- **Checkpoint**: `outputs/curriculum/stage_1/checkpoints/adapter_epoch_2`
- **Metrics**: `outputs/curriculum/stage_1/epoch_2/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_1/epoch_2/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage2-e2 uses EVAL subset stage (num_calls==2), NOT model stage. Model is stage 1 epoch 2.

### `train-s1-e3 → eval subset 2-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage2-e3`
- **Checkpoint**: `outputs/curriculum/stage_1/checkpoints/adapter_epoch_3`
- **Metrics**: `outputs/curriculum/stage_1/epoch_3/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_1/epoch_3/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage2-e3 uses EVAL subset stage (num_calls==2), NOT model stage. Model is stage 1 epoch 3.

### `train-s1-e4 → eval subset 2-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage2-e4`
- **Checkpoint**: `outputs/curriculum/stage_1/checkpoints/adapter_epoch_4`
- **Metrics**: `outputs/curriculum/stage_1/epoch_4/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_1/epoch_4/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage2-e4 uses EVAL subset stage (num_calls==2), NOT model stage. Model is stage 1 epoch 4.

### `train-s2-e1 → eval subset 3-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage3-e1`
- **Checkpoint**: `outputs/curriculum/stage_2/checkpoints/adapter_epoch_1`
- **Metrics**: `outputs/curriculum/stage_2/epoch_1/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_2/epoch_1/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage3-e1 uses EVAL subset stage (num_calls==3), NOT model stage. Model is stage 2 epoch 1.

### `train-s2-e3 → eval subset 3-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage3-e3`
- **Checkpoint**: `outputs/curriculum/stage_2/checkpoints/adapter_epoch_3`
- **Metrics**: `outputs/curriculum/stage_2/epoch_3/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_2/epoch_3/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage3-e3 uses EVAL subset stage (num_calls==3), NOT model stage. Model is stage 2 epoch 3. SOURCE OF 0.488943 confusion: W&B 'eval-stage3-e3' = this row.

### `train-s2-e4 → eval subset 3-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage3-e4`
- **Checkpoint**: `outputs/curriculum/stage_2/checkpoints/adapter_epoch_4`
- **Metrics**: `outputs/curriculum/stage_2/epoch_4/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_2/epoch_4/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage3-e4 uses EVAL subset stage (num_calls==3), NOT model stage. Model is stage 2 epoch 4.

### `train-s3-e1 → eval subset 4-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage4-e1`
- **Checkpoint**: `outputs/curriculum/stage_3/checkpoints/adapter_epoch_1`
- **Metrics**: `outputs/curriculum/stage_3/epoch_1/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_3/epoch_1/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage4-e1 uses EVAL subset stage (num_calls==4), NOT model stage. Model is stage 3 epoch 1.

### `train-s3-e2 → eval subset 4-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage4-e2`
- **Checkpoint**: `outputs/curriculum/stage_3/checkpoints/adapter_epoch_2`
- **Metrics**: `outputs/curriculum/stage_3/epoch_2/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_3/epoch_2/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage4-e2 uses EVAL subset stage (num_calls==4), NOT model stage. Model is stage 3 epoch 2.

### `train-s3-e3 → eval subset 4-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage4-e3`
- **Checkpoint**: `outputs/curriculum/stage_3/checkpoints/adapter_epoch_3`
- **Metrics**: `outputs/curriculum/stage_3/epoch_3/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_3/epoch_3/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage4-e3 uses EVAL subset stage (num_calls==4), NOT model stage. Model is stage 3 epoch 3. Actual stage3-e3 model eval is W&B 'eval-stage4-e3' (0.168 final_answer).

### `train-s4-e1 → eval subset 5-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage5-e1`
- **Checkpoint**: `outputs/curriculum/stage_4/checkpoints/adapter_epoch_1`
- **Metrics**: `outputs/curriculum/stage_4/epoch_1/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_4/epoch_1/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage5-e1 uses EVAL subset stage (num_calls==5), NOT model stage. Model is stage 4 epoch 1.

### `train-s4-e2 → eval subset 5-call`

- **Source**: mt-grpo curriculum rollout_eval
- **W&B name**: `eval-stage5-e2`
- **Checkpoint**: `outputs/curriculum/stage_4/checkpoints/adapter_epoch_2`
- **Metrics**: `outputs/curriculum/stage_4/epoch_2/eval/metrics.json`
- **Trajectories**: `outputs/curriculum/stage_4/epoch_2/eval/rollout_eval_trajectories.jsonl`
- **Notes**: W&B name eval-stage5-e2 uses EVAL subset stage (num_calls==5), NOT model stage. Model is stage 4 epoch 2.

### `baseline (no LoRA) / react / full NESTFUL`

- **Source**: mt-grpo final_eval
- **W&B name**: `final_eval_baseline_react`
- **Checkpoint**: ``
- **Metrics**: `final_outputs/runs/baseline_react/metrics_official.json`
- **Trajectories**: `outputs/final_eval_baseline_react/final_eval_trajectories.jsonl`
- **Notes**: Full-benchmark eval (1861 tasks). W&B run name = output subdir when launched via RUN_FINAL_EVAL=1.

### `curriculum s4e2 / react / full NESTFUL`

- **Source**: mt-grpo final_eval
- **W&B name**: `final_eval_stage4_epoch2_react`
- **Checkpoint**: `outputs/curriculum/stage_4/checkpoints/adapter_epoch_2`
- **Metrics**: `final_outputs/runs/stage4e2_react/metrics_official.json`
- **Trajectories**: `outputs/final_eval_stage4_epoch2_react/final_eval_trajectories.jsonl`
- **Notes**: Full-benchmark eval (1861 tasks). W&B run name = output subdir when launched via RUN_FINAL_EVAL=1.

### `baseline (no LoRA) / direct / full NESTFUL`

- **Source**: mt-grpo final_eval
- **W&B name**: `baseline_direct`
- **Checkpoint**: ``
- **Metrics**: `final_outputs/runs/baseline_direct/metrics_official.json`
- **Trajectories**: `outputs/baseline_direct/direct_eval_trajectories.jsonl`
- **Predictions**: `outputs/baseline_direct/direct_predictions.jsonl`
- **Notes**: Full-benchmark eval (1861 tasks). W&B run name = output subdir when launched via RUN_FINAL_EVAL=1.

### `curriculum s4e2 / direct / full NESTFUL`

- **Source**: mt-grpo final_eval
- **W&B name**: `stage4_epoch2_direct`
- **Checkpoint**: `outputs/curriculum/stage_4/checkpoints/adapter_epoch_2`
- **Metrics**: `final_outputs/runs/stage4e2_direct/metrics_official.json`
- **Predictions**: `outputs/stage4_epoch2_direct/direct_predictions.jsonl`
- **Notes**: Full-benchmark eval (1861 tasks). W&B run name = output subdir when launched via RUN_FINAL_EVAL=1.

### `baseline / react 4-rollout / full NESTFUL`

- **Source**: legacy curriculum v2 (results_v2_20260617)
- **W&B name**: `v2_baseline`
- **Checkpoint**: ``
- **Metrics**: `final_outputs/runs/v2_baseline/metrics_official.json`
- **Notes**: Legacy run; executor_accuracy=67.18%. Uses rollout idx0 only.

### `curriculum s3e1 / react 4-rollout / full NESTFUL`

- **Source**: legacy curriculum v2 (results_v2_20260617)
- **W&B name**: `v2_stage3_epoch1`
- **Checkpoint**: ``
- **Metrics**: `final_outputs/runs/v2_stage3_epoch1/metrics_official.json`
- **Predictions**: `../../curricullum/evaluation/results_v2_20260617/curriculum_stage_3_epoch1_multiturn_predictions.jsonl`
- **Notes**: Legacy run; executor_accuracy=70.74%. Uses rollout idx0 only.

### `curriculum s5e2 / react 4-rollout / full NESTFUL`

- **Source**: legacy curriculum v2 (results_v2_20260617)
- **W&B name**: `v2_stage5_epoch2`
- **Checkpoint**: ``
- **Metrics**: `final_outputs/runs/v2_stage5_epoch2/metrics_official.json`
- **Predictions**: `../../curricullum/evaluation/results_v2_20260617/curriculum_stage_5_epoch2_multiturn_predictions.jsonl`
- **Notes**: Legacy run; executor_accuracy=70.14%. Uses rollout idx0 only.

## Gaps and duplicate paths

- **Stage 2 epoch 2** has no `epoch_2/eval/metrics.json` — training resumed at epoch 3 (see `epoch_summary.jsonl`: jumps 1 → 3 → 4). No W&B run `eval-stage3-e2` exists for this curriculum.
- **Legacy duplicate**: `outputs/curriculum/stage_4/eval/epoch_1/` (173 tasks) is NOT written by current `run_curriculum.sh` (which uses `stage_4/epoch_1/eval/`). Treat as stale/orphan unless proven otherwise.
- **`consolidated_metrics.json` / `curriculum_training.csv`** index rows by **model** `(stage, epoch)` from disk path — this is the correct key for comparing checkpoints.
- **W&B graphs** index in-curriculum evals by **eval subset** in the run name — easy to misread as model stage.

## Files audited

- `run_curriculum.sh` — W&B naming, eval subset = train_stage+1
- `run.py` — `mode_rollout_eval`, `mode_final_eval`, W&B logging
- `build_report.py` — `build_curriculum_table()` keys rows by **model** stage/epoch from disk path
- `outputs/curriculum/stage_*/epoch_*/eval/metrics.json`
- `final_outputs/consolidated_metrics.json`

Regenerate: `python final_outputs/_audit_build_map.py`

Machine-readable full table: [`checkpoint_eval_map.csv`](checkpoint_eval_map.csv)
