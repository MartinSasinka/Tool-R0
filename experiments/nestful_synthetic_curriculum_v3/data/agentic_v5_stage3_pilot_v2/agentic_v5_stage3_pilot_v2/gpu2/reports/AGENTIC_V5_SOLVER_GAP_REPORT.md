# Solver-gap report (weak-fail / strong-pass filtering, real executor)

Candidates that reached the solver stage: 50
- weak/strong solved with real ``executor.mode=synthetic`` execution — a wrong predicted value never receives the gold result.

- passed the solver-gap gate (weak_fail_strong_pass): 50 (1.000)
- **finally accepted (this run): 4** (after best-of-N selection, diversity caps, the rollout probe and the judge)
- avg weak score: 0.547
- avg strong score (when run): 1.000
- avg gap (when strong ran): 0.818

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=50)

| value | count | share |
|---|---|---|
| 0.18 | 2 | 0.040 |
| 0.54 | 4 | 0.080 |
| 0.55 | 2 | 0.040 |
| 0.56 | 20 | 0.400 |
| 0.57 | 21 | 0.420 |
| 0.59 | 1 | 0.020 |

### strong_score histogram (when run) (n=2)

| value | count | share |
|---|---|---|
| 1.00 | 2 | 1.000 |

### gap histogram (when strong ran) (n=2)

| value | count | share |
|---|---|---|
| 0.82 | 2 | 1.000 |

