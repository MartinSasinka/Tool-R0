# NESTFUL — consolidated results

All numbers below are produced by the **official NESTFUL scorer** (corpus macro-F1 + grounded partial/full). They were re-computed from saved predictions/trajectories so every run uses identical scoring. See `ANALYSIS.md` for methodology and caveats.

> Win Rate: **not computed on this machine (Windows)** — Win Rate needs Linux/SIGALRM; values shown are reused from prior runs where available.

## Full NESTFUL eval (1861 tasks): baseline vs curriculum, ReAct vs Direct

| run | model | paradigm | F1 Func | F1 Param | Partial | Full | Win | win src | parse_err |
|---|---|---|---|---|---|---|---|---|---|
| baseline_react | baseline (no LoRA) | react | 0.894 | 0.360 | 0.130 | 0.000 | 0.544 | recomputed | 0 |
| baseline_direct | baseline (no LoRA) | direct | 0.921 | 0.650 | 0.294 | 0.169 | 0.292 | recomputed | 0 |
| stage4e2_react | curriculum s4e2 | react | 0.153 | 0.131 | 0.050 | 0.001 | 0.325 | recomputed | 0 |
| stage4e2_direct | curriculum s4e2 | direct | 0.917 | 0.616 | 0.272 | 0.152 | 0.243 | recomputed | 0 |

## Original curriculum eval (results_v2) re-scored to NESTFUL metrics

Multi-turn ReAct rollouts on the full 1861-task NESTFUL. The original framework reported only executor-based final-answer accuracy (`executor_accuracy_pct`); the official NESTFUL metrics below are re-computed from `predicted_calls` (rollout idx 0, one prediction/task). Baseline has no stored predictions, only its executor accuracy.

| run | model | rollout | exec acc % | F1 Func | F1 Param | Partial | Full | Win |
|---|---|---|---|---|---|---|---|---|
| v2_baseline | baseline (no LoRA) | — | 67.180 | — | — | — | — | — |
| v2_stage3_epoch1 | curriculum s3e1 | idx0 | 70.740 | 0.941 | 0.447 | 0.158 | 0.000 | — |
| v2_stage5_epoch2 | curriculum s5e2 | idx0 | 70.140 | 0.951 | 0.453 | 0.163 | 0.000 | — |

## Curriculum training progression (small per-stage eval)

`strict_gold_trace_pass` / `final_answer_pass` are the training-time eval metrics; `off_*` are official re-scores of the same trajectories.

| Stage | Epoch | N | strict_pass | final_pass | zero_calls | off F1 Func | off Partial | off Full |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 609 | 0.343 | 0.433 | 0.144 | 0.759 | 0.256 | 0.000 |
| 1 | 2 | 609 | 0.361 | 0.448 | 0.117 | 0.773 | 0.258 | 0.002 |
| 1 | 3 | 609 | 0.356 | 0.435 | 0.126 | 0.759 | 0.251 | 0.002 |
| 1 | 4 | 609 | 0.378 | 0.470 | 0.138 | 0.787 | 0.262 | 0.002 |
| 2 | 1 | 407 | 0.150 | 0.182 | 0.278 | 0.698 | 0.102 | 0.002 |
| 2 | 3 | 407 | 0.113 | 0.489 | 0.273 | 0.798 | 0.100 | 0.000 |
| 2 | 4 | 407 | 0.093 | 0.494 | 0.268 | 0.780 | 0.106 | 0.000 |
| 3 | 1 | 250 | 0.012 | 0.440 | 0.228 | 0.546 | 0.032 | 0.000 |
| 3 | 2 | 250 | 0.008 | 0.320 | 0.224 | 0.396 | 0.031 | 0.000 |
| 3 | 3 | 250 | 0.000 | 0.168 | 0.236 | 0.397 | 0.038 | 0.000 |
| 4 | 1 | 32 | 0.000 | 0.156 | 0.281 | 0.478 | 0.013 | 0.000 |
| 4 | 2 | 173 | 0.000 | 0.087 | 0.225 | 0.473 | 0.043 | 0.000 |

