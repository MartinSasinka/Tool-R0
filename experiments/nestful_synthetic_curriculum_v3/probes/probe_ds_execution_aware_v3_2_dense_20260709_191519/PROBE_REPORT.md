# PROBE REPORT — probe_ds_execution_aware_v3_2_dense_20260709_191519

Backend: **vllm**
Dataset: `experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like/filtered/v4_stage1_2call.jsonl` (n_probed=50, sha256=ad979aa72016…)
Reward: `execution_aware_v3_2_dense` | checkpoint: `base model`
Decoding: T=1.0 top_p=0.95 seed=42 | 8 generations/group

## GRPO signal

| metric | value |
|---|---|
| dead_group_rate | **0.86** |
| mixed_group_rate | 0.14 |
| dead_low_rate (mean<= 0.35) | 0.82 |
| dead_high_rate (mean>=0.9, saturated) | 0.0 |
| position_artifact_rate | 0.0 |
| mean unique rewards / group | 1.14 |
| reward entropy (bits) | 1.175 |
| distinct reward values | 10 |

## Behavior

| metric | value |
|---|---|
| too_few_calls_rate | 0.88 |
| avg_predicted_calls | 1.12 |
| wrong_tool_rate | 0.88 |
| wrong_arg_rate | 0.09 |
| parse_error_rate | 0.0 |
| no_tool_call_rate | 0.0 |
| invalid_reference_rate | 0.0 |

## Reward histogram

| bin | count |
|---|---|
| [0.0,0.1) | 0 |
| [0.1,0.2) | 0 |
| [0.2,0.3) | 352 |
| [0.3,0.4) | 0 |
| [0.4,0.5) | 15 |
| [0.5,0.6) | 29 |
| [0.6,0.7) | 0 |
| [0.7,0.8) | 0 |
| [0.8,0.9) | 0 |
| [0.9,1.0] | 4 |

## Verdict

**proceed_recommendation: False** (gate: dead_group_rate < 0.5 AND mean_unique_rewards_per_group >= 2)
GRPO training on this (stage, reward, checkpoint) combination is expected to be signal-starved. Fix the reward (densify), the init (SFT warmup), or the task mix (filtering) before spending GPU time.

Files: signal_positive_tasks.jsonl (7 rows), dead_low_tasks.jsonl (41 rows), motif_signal_table.csv
