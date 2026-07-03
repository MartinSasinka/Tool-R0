# Pilot Motif-Level Eval Report

Best checkpoint: **stage_1 / epoch_2** vs baseline (official Win, n=200 dev subset, seed=42).

## Motif-type table

| motif_type | n | baseline_win | model_win | delta | b_fail→m_win | b_win→m_fail | net_gain | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fan_in | 27 | 0.593 | 0.667 | +0.074 | 3 | 5 | -2 | model_better (aggregate) |
| long_chain | 63 | 0.508 | 0.556 | +0.048 | 5 | 8 | -3 | neutral aggregate, net regression |
| linear_dependency | 106 | 0.585 | 0.576 | -0.009 | 8 | 7 | +1 | neutral |
| independent_calls | 4 | 0.250 | 0.250 | +0.000 | 0 | 0 | 0 | insufficient_n |

## Bucket tables (selected)

| bucket_type | bucket | n | baseline_win | model_win | delta |
|---|---|---:|---:|---:|---:|
| num_calls | 2-call | 66 | 0.470 | 0.500 | +0.030 |
| num_calls | 3-call | 44 | 0.727 | 0.682 | -0.045 |
| num_calls | 5-8 call | 47 | 0.489 | 0.575 | +0.085 |
| num_calls | 9+ call | 16 | 0.563 | 0.500 | -0.063 |
| output_type | list | 14 | 0.357 | 0.286 | -0.071 |
| output_type | string | 18 | 0.222 | 0.333 | +0.111 |

## Answers

1. **Improved motifs:** fan_in (+7.4pp aggregate), long_chain (+4.8pp aggregate), 5–8 call bucket (+8.5pp).
2. **Regressed motifs:** linear_dependency (-0.9pp); 3-call bucket (-4.5pp); list output (-7.1pp). Per-sample net regression largest on long_chain (-3) and fan_in (-2).
3. **Aligned with synthetic v3?** Partially — stage1 linear/simple training matches 2-call and linear_dependency stability; stage2 synthetic motifs (reference_reuse, object/list) did **not** run as best checkpoint and stage2 training hurt 3-call performance.
4. **Largest net regression clusters:** long_chain (8 baseline wins lost vs 5 gained), fan_in (5 lost vs 3 gained).
5. **Generation priority:** long_chain (55 too_few_calls failures unchanged), fan_in depth, list-output extraction, independent_calls (n=4, underpowered).

## Caveats

- Eval subset n=200, not full dev (1861).
- Prototype tool registry — not IBM tool-family transfer.
- See [PILOT_BUCKET_LEVEL_EVAL.csv](./PILOT_BUCKET_LEVEL_EVAL.csv) for full buckets.
