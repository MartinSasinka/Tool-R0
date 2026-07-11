# Solver-gap report (weak-fail / strong-pass filtering)

Candidates that reached the solver stage: 14

- passed the solver-gap gate (weak_fail_strong_pass): 11 (0.786)
- rejected AFTER the gap gate by diversity caps: 0
- rejected AFTER the gap gate by the LLM style judge: 1
- **finally accepted (this run): 10** (= new rows in filtered/*.jsonl and the manifest — this is the only number that counts)
- strong EXACT executable wins among accepted: 10/10 (1.000)
- both_pass (too easy): 0
- both_fail (too hard): 0
- avg weak score: 0.571
- avg strong score (when run): 1.000
- avg gap (when strong ran): 0.545

Acceptance policy: weak <= 0.50, gap >= 0.25, STRONG_PASS_POLICY=`exact_win` (exact_win = strong must be a TRUE executable win / solution-equivalent, score 1.0; partial strong solutions never enter the filtered set).
Diversity caps on accepted rows: max 0.40 same weak-score bucket, max 0.40 same failure type (enforced after 25 accepted).
Strong solver runs ONLY when the weak solver failed (compute saving from the Autodata paper).

## Score histograms (all solver-stage candidates)

### weak_score histogram (n=14)

| value | count | share |
|---|---|---|
| 0.00 | 1 | 0.071 |
| 0.50 | 10 | 0.714 |
| 1.00 | 3 | 0.214 |

### strong_score histogram (when run) (n=11)

| value | count | share |
|---|---|---|
| 1.00 | 11 | 1.000 |

### gap histogram (when strong ran) (n=11)

| value | count | share |
|---|---|---|
| 0.50 | 10 | 0.909 |
| 1.00 | 1 | 0.091 |

Near-threshold strong band [0.70, 0.80): 0 candidates

## ACCEPTED examples — diversity

### accepted weak_score histogram (n=10)

| value | count | share |
|---|---|---|
| 0.00 | 1 | 0.100 |
| 0.50 | 9 | 0.900 |

### accepted weak failure type distribution (n=10)

| key | count | share |
|---|---|---|
| partial_prefix | 9 | 0.900 |
| parse_error | 1 | 0.100 |

### weak predicted calls - gold calls (n=14)

| value | count | share |
|---|---|---|
| +0 | 13 | 0.929 |
| -2 | 1 | 0.071 |

### strong predicted calls - gold calls (when run) (n=11)

| value | count | share |
|---|---|---|
| +0 | 11 | 1.000 |

### motif × weak failure type (accepted)

| motif | parse_error | partial_prefix |
|---|---|---|
| argument_binding | 1 | 2 |
| distractor_heavy | 0 | 2 |
| long_chain | 0 | 3 |
| reference_reuse | 0 | 2 |

### stage × weak failure type (accepted)

| stage | parse_error | partial_prefix |
|---|---|---|
| stage2_2call_agentic_openrouter | 1 | 9 |

## Weak solver statuses (all solver-stage candidates)

- partial_prefix: 10
- win: 3
- parse_error: 1
