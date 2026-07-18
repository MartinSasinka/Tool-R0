# Solver-gap report (weak-fail / strong-pass filtering, real executor)

Candidates that reached the solver stage: 49
- weak/strong solved with real ``executor.mode=synthetic`` execution — a wrong predicted value never receives the gold result.

- passed the solver-gap gate (weak_fail_strong_pass): 49 (1.000)
- **finally accepted (this run): 4** (after best-of-N selection, diversity caps, the rollout probe and the judge)
- avg weak score: 0.553
- avg strong score (when run): 0.093
- avg gap (when strong ran): -0.159

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=49)

| value | count | share |
|---|---|---|
| 0.20 | 1 | 0.020 |
| 0.25 | 1 | 0.020 |
| 0.31 | 1 | 0.020 |
| 0.53 | 1 | 0.020 |
| 0.54 | 3 | 0.061 |
| 0.56 | 12 | 0.245 |
| 0.57 | 29 | 0.592 |
| 1.00 | 1 | 0.020 |

### strong_score histogram (when run) (n=3)

| value | count | share |
|---|---|---|
| 0.00 | 2 | 0.667 |
| 0.28 | 1 | 0.333 |

### gap histogram (when strong ran) (n=3)

| value | count | share |
|---|---|---|
| -0.25 | 1 | 0.333 |
| -0.31 | 1 | 0.333 |
| 0.08 | 1 | 0.333 |

