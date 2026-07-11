# PROBE REPORT — selftest_stub_s2b

Backend: **stub** — **STUB (fake numbers, pipeline self-test only)**
Dataset: `experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1/filtered/stage2_2call_dependency.jsonl` (n_probed=6, sha256=0e7c01aaa78f…)
Reward: `execution_aware_v3_1_stepwise` | checkpoint: `base model`
Decoding: T=1.0 top_p=0.95 seed=42 | 4 generations/group

## GRPO signal

| metric | value |
|---|---|
| dead_group_rate | **0.3333** |
| mixed_group_rate | 0.6667 |
| dead_low_rate (mean<= 0.35) | 0.3333 |
| dead_high_rate (mean>=0.9, saturated) | 0.0 |
| position_artifact_rate | 0.1667 |
| mean unique rewards / group | 2.1667 |
| reward entropy (bits) | 1.5535 |
| distinct reward values | 5 |

## Behavior

| metric | value |
|---|---|
| too_few_calls_rate | 0.875 |
| avg_predicted_calls | 1.0 |
| wrong_tool_rate | 0.75 |
| wrong_arg_rate | 0.0417 |
| parse_error_rate | 0.125 |
| no_tool_call_rate | 0.125 |
| invalid_reference_rate | 0.0833 |

## Reward histogram

| bin | count |
|---|---|
| [0.0,0.1) | 3 |
| [0.1,0.2) | 2 |
| [0.2,0.3) | 2 |
| [0.3,0.4) | 16 |
| [0.4,0.5) | 0 |
| [0.5,0.6) | 1 |
| [0.6,0.7) | 0 |
| [0.7,0.8) | 0 |
| [0.8,0.9) | 0 |
| [0.9,1.0] | 0 |

## Verdict

**proceed_recommendation: True** (gate: dead_group_rate < 0.5 AND mean_unique_rewards_per_group >= 2)


Files: signal_positive_tasks.jsonl (4 rows), dead_low_tasks.jsonl (2 rows), motif_signal_table.csv
