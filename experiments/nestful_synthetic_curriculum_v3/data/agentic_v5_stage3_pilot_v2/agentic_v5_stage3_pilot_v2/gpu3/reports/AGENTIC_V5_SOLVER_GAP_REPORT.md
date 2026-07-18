# Solver-gap report (weak-fail / strong-pass filtering, real executor)

Candidates that reached the solver stage: 50
- weak/strong solved with real ``executor.mode=synthetic`` execution — a wrong predicted value never receives the gold result.

- passed the solver-gap gate (weak_fail_strong_pass): 50 (1.000)
- **finally accepted (this run): 8** (after best-of-N selection, diversity caps, the rollout probe and the judge)
- avg weak score: 0.563
- avg strong score (when run): 0.865
- avg gap (when strong ran): 0.686

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=50)

| value | count | share |
|---|---|---|
| 0.00 | 1 | 0.020 |
| 0.20 | 1 | 0.020 |
| 0.34 | 1 | 0.020 |
| 0.52 | 1 | 0.020 |
| 0.53 | 3 | 0.060 |
| 0.54 | 4 | 0.080 |
| 0.56 | 14 | 0.280 |
| 0.57 | 22 | 0.440 |
| 1.00 | 3 | 0.060 |

### strong_score histogram (when run) (n=3)

| value | count | share |
|---|---|---|
| 0.59 | 1 | 0.333 |
| 1.00 | 2 | 0.667 |

### gap histogram (when strong ran) (n=3)

| value | count | share |
|---|---|---|
| 0.40 | 1 | 0.333 |
| 0.66 | 1 | 0.333 |
| 1.00 | 1 | 0.333 |

