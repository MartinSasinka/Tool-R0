# Counterfactual training reward audit

**Policy:** `execution_aware_v3_2_dense` applied to saved NESTFUL trajectories
**Generated:** 2026-07-23T06:34:46.623672+00:00

## Mean R_train by official outcome

| Arm | mean R (all) | R | official win | R | official loss |
|-----|-------------:|---|-------------:|---|--------------:|
| C0 | 0.3100 | | 0.3362 | | 0.2788 |
| E1 | 0.3129 | | 0.3358 | | 0.2864 |
| E2 | 0.3147 | | 0.3358 | | 0.2906 |

## C0 win → E2 loss (n=93)

| Ordering | count | share |
|----------|------:|------:|
| R_train(E2) > R_train(C0) | 22 | 23.7% |
| R_train(E2) = R_train(C0) | 7 | 7.5% |
| R_train(E2) < R_train(C0) | 64 | 68.8% |

## Terminal class mean reward (E2)

- executable_wrong_final: **0.5378** (C0: 0.5278)
- fully_correct band: **0.0000** (C0: 0.0000)

**Note:** eval `reward_train_strict` is strict gold-trace reward, not this policy.