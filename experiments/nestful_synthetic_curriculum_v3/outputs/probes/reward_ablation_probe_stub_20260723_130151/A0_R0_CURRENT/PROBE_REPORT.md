# PROBE REPORT — A0_R0_CURRENT

Backend: **stub** — **STUB (fake numbers, pipeline self-test only)**
Dataset: `experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/data/train_subset_160.jsonl` (n_probed=16, sha256=b64d3ec24773…)
Reward: `execution_aware_v3_2_dense` | checkpoint: `base model` | executor.mode: `synthetic`
Decoding: T=1.0 top_p=0.95 seed=20260724 | 8 generations/group

## GRPO signal

| metric | value |
|---|---|
| dead_group_rate | **0.0** |
| mixed_group_rate | 1.0 |
| dead_low_rate (mean<= 0.35) | 0.0 |
| dead_high_rate (mean>=0.9, saturated) | 0.0 |
| position_artifact_rate | 0.0 |
| mean unique rewards / group | 3.5625 |
| reward entropy (bits) | 1.9613 |
| distinct reward values | 4 |

## Behavior

| metric | value |
|---|---|
| too_few_calls_rate | 0.8203 |
| avg_predicted_calls | 1.1406 |
| wrong_tool_rate | 0.6016 |
| wrong_arg_rate | 0.0 |
| parse_error_rate | 0.2188 |
| no_tool_call_rate | 0.2188 |
| invalid_reference_rate | 0.0 |

## Reward histogram

| bin | count |
|---|---|
| [0.0,0.1) | 28 |
| [0.1,0.2) | 43 |
| [0.2,0.3) | 34 |
| [0.3,0.4) | 0 |
| [0.4,0.5) | 0 |
| [0.5,0.6) | 0 |
| [0.6,0.7) | 0 |
| [0.7,0.8) | 0 |
| [0.8,0.9) | 0 |
| [0.9,1.0] | 23 |

## Verdict

**proceed_recommendation: True** (gate: dead_group_rate < 0.5 AND mean_unique_rewards_per_group >= 2)


Files: signal_positive_tasks.jsonl (16 rows), dead_low_tasks.jsonl (0 rows), motif_signal_table.csv
