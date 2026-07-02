# Per-sample official Win — consistency

| run | n | mean(per_sample_win) | aggregate_win | abs_diff | status |
|---|---|---|---|---|---|
| partial_s1_e4_react | 1861 | 0.5422 | 0.543 | 0.00082 | PASS |
| partial_s4_e1_react | 1861 | 0.4498 | 0.45 | 0.00024 | PASS |
| baseline_react | 1861 | 0.5433 | 0.544 | 0.00074 | PASS |
| baseline_direct | - | - | 0.292 | - | WARNING_MISSING_TRAJECTORIES |
| minimal_s4e2_react | - | - | 0.325 | - | WARNING_MISSING_TRAJECTORIES |

## OVERALL: PASS

Consistency tolerance = 0.0012 (3-dp aggregate rounding + ~2 samples of per-sample granularity out of 1861).

Runs marked WARNING_MISSING_TRAJECTORIES kept only aggregate `metrics_official.json` (trajectories were not preserved). They CANNOT be recomputed per-sample and MUST NOT be used for per-sample overlap / correlation / failure taxonomy.
