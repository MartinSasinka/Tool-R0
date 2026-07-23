# Pure Stage 3 Overnight — C0 / E1 / E2 Test (n=1661)

Run: `pure_stage3_2ep_20260719_221918` | Generated: 2026-07-22T22:40:18.273307+00:00

## Headline win rate

| Arm | Win | Δ vs C0 |
|-----|----:|--------:|
| C0 | 0.5443 | — |
| E1 | 0.5364 | -0.0078 |
| E2 | 0.5328 | -0.0114 |

E1 vs C0: gained 75 lost 88 net -13 p=0.3472624244875813
E2 vs C0: gained 74 lost 93 net -19 p=0.16365553128872506
E2 vs E1: gained 79 lost 85 net -6 p=0.6962153512437399

## Transitions

- **stable_win**: 772
- **stable_loss**: 642
- **lost_after_E1**: 49
- **gained_E1_lost_E2**: 44
- **gained_after_E1_lost_E2**: 41
- **gained_after_E2_only**: 40
- **lost_E1_regained_E2**: 39
- **gained_E1_kept_E2**: 34

- E1 gain → E2 loss: **41**
- E1 loss → E2 regain: **39**

## Key pattern (values↓ tool/exec↑)?
**True**

## By call bucket

| bucket | n | C0 | E1 | E2 |
|---|---:|---:|---:|---:|
| 2 | 543 | 0.4586 | 0.4549 | 0.4420 |
| 3 | 363 | 0.6006 | 0.5923 | 0.5895 |
| 4 | 223 | 0.6099 | 0.6099 | 0.5919 |
| 5 | 154 | 0.5974 | 0.5584 | 0.6169 |
| 6+ | 378 | 0.5529 | 0.5476 | 0.5397 |

## Failure rates (non-success share)

- correct keys, wrong argument values: C0 8.97% → E1 8.73% → E2 8.61%
- wrong tool: C0 9.51% → E1 9.75% → E2 10.05%
- executable trajectory ending wrong result: C0 10.17% → E1 11.32% → E2 11.68%
- no tool call: C0 6.56% → E1 6.86% → E2 6.50%
- too few calls: C0 0.78% → E1 0.96% → E2 0.84%
- parse/format error: C0 4.33% → E1 3.79% → E2 4.03%