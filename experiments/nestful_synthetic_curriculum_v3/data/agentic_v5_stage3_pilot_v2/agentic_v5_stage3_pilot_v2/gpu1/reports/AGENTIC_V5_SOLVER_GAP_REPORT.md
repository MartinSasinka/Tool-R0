# Solver-gap report (weak-fail / strong-pass filtering, real executor)

Candidates that reached the solver stage: 50
- weak/strong solved with real ``executor.mode=synthetic`` execution — a wrong predicted value never receives the gold result.

- passed the solver-gap gate (weak_fail_strong_pass): 50 (1.000)
- **finally accepted (this run): 4** (after best-of-N selection, diversity caps, the rollout probe and the judge)
- avg weak score: 0.549
- avg strong score (when run): 0.857
- avg gap (when strong ran): 0.620

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=50)

| value | count | share |
|---|---|---|
| 0.18 | 2 | 0.040 |
| 0.22 | 1 | 0.020 |
| 0.30 | 2 | 0.040 |
| 0.53 | 1 | 0.020 |
| 0.56 | 15 | 0.300 |
| 0.57 | 27 | 0.540 |
| 1.00 | 2 | 0.040 |

### strong_score histogram (when run) (n=5)

| value | count | share |
|---|---|---|
| 0.28 | 1 | 0.200 |
| 1.00 | 4 | 0.800 |

### gap histogram (when strong ran) (n=5)

| value | count | share |
|---|---|---|
| 0.07 | 1 | 0.200 |
| 0.70 | 2 | 0.400 |
| 0.82 | 2 | 0.400 |

