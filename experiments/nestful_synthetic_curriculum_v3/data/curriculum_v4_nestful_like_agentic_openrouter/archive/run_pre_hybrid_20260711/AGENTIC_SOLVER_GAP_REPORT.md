# Solver-gap report (weak-fail / strong-pass filtering)

Candidates that reached the solver stage: 13

- passed the solver-gap gate (weak_fail_strong_pass): 2 (0.154)
- rejected AFTER the gap gate by diversity caps: 2
- rejected AFTER the gap gate by the LLM style judge: 0
- **finally accepted (this run): 0** (= new rows in filtered/*.jsonl and the manifest — this is the only number that counts)
- strong EXACT executable wins among accepted: 0/1 (0.000)
- both_pass (too easy): 0
- both_fail (too hard): 0
- avg weak score: 0.731
- avg strong score (when run): 0.643
- avg gap (when strong ran): 0.143

Acceptance policy: weak <= 0.50, gap >= 0.25, STRONG_PASS_POLICY=`exact_win` (exact_win = strong must be a TRUE executable win / solution-equivalent, score 1.0; partial strong solutions never enter the filtered set).
Diversity caps on accepted rows: max 0.40 same weak-score bucket, max 0.40 same failure type (enforced after 25 accepted).
Strong solver runs ONLY when the weak solver failed (compute saving from the Autodata paper).

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=13)

| value | count | share |
|---|---|---|
| 0.50 | 7 | 0.538 |
| 1.00 | 6 | 0.462 |

### strong_score histogram (when run) (n=7)

| value | count | share |
|---|---|---|
| 0.50 | 5 | 0.714 |
| 1.00 | 2 | 0.286 |

### gap histogram (when strong ran) (n=7)

| value | count | share |
|---|---|---|
| 0.00 | 5 | 0.714 |
| 0.50 | 2 | 0.286 |

Near-threshold strong band [0.70, 0.80): 0 candidates

## ACCEPTED examples — diversity

### accepted weak_score histogram (n=0)

| value | count | share |
|---|---|---|

### accepted weak failure type distribution (n=0)

| key | count | share |
|---|---|---|

### weak predicted calls - gold calls (n=13)

| value | count | share |
|---|---|---|
| +0 | 11 | 0.846 |
| +1 | 1 | 0.077 |
| -1 | 1 | 0.077 |

### strong predicted calls - gold calls (when run) (n=7)

| value | count | share |
|---|---|---|
| +0 | 6 | 0.857 |
| +1 | 1 | 0.143 |

## Weak solver statuses (all solver-stage candidates)

- win: 6
- partial_prefix: 6
- correct_prefix_then_stop: 1
