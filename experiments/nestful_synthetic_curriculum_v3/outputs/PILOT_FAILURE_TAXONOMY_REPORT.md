# Pilot Failure Taxonomy Report

Baseline vs **best checkpoint (s1_e2)** on dev val subset (n=200 paired failures among non-success trajectories).

## Aggregate failure counts (non-success trajectories)

| failure_type | baseline_count | model_count | delta | interpretation |
|---|---:|---:|---:|---|
| too_few_calls | 101 | 102 | +1 | **Dominant failure** — pilot did not fix; long_chain (55+55) flat |
| no_tool_call | 20 | 16 | -4 | Slight improvement |
| executor_error | 16 | 16 | 0 | Unchanged |
| motif_inconsistent_trace | 10 | 12 | +2 | Slight worsening |
| too_many_calls | 7 | 9 | +2 | Minor increase |
| wrong_final_answer | 4 | 3 | -1 | Small improvement |

## By motif (top deltas)

| motif_type | failure_type | baseline | model | delta |
|---|---|---:|---:|---:|
| linear_dependency | too_few_calls | 28 | 30 | +2 |
| linear_dependency | no_tool_call | 12 | 7 | -5 |
| long_chain | too_few_calls | 55 | 55 | 0 |
| fan_in | too_few_calls | 16 | 15 | -1 |

## Answers

1. **Old failures removed?** Partially — no_tool_call down 4; wrong_final_answer down 1.
2. **New failures?** motif_inconsistent_trace +2, too_many_calls +2; too_few_calls essentially unchanged (+1).
3. **Still too_few_calls?** **Yes** — 51% of failed trajectories; long_chain unchanged at 55.
4. **invalid_reference?** Not a top-level category in this run (likely subsumed under trace/executor errors).
5. **wrong_final_answer with valid trace?** Rare (4→3); high NESTFUL final_answer_pass with low Win suggests partial-trace wrong answers remain common (see eval metrics).
6. **Short-trace gaming?** Training mean_reward spikes to 1.0 on synthetic while too_few_calls persists on real dev — reward not fully aligned.

See [PILOT_FAILURE_BY_MOTIF.csv](./PILOT_FAILURE_BY_MOTIF.csv).
