# Round 1 — corrected comparison (true shared C0 baseline)

C0 eval: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\shared_C0_eval_500\shared_C0_eval_500\eval\C0\20260724`
A0 (R0-trained) eval: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A0_R0_CURRENT_seed20260724\reward_ablation_r1_A0_R0_CURRENT_seed20260724\eval\A0_R0_CURRENT\20260724`

| Arm | win_rate | vs C0 Δmean | gained | regressed | McNemar p | vs R0 Δmean | gained | regressed | exec_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_R0_CURRENT | 0.574 | 0.004 | 31 | 29 | 0.897278961260083 | None | None | None | 0.84 |
| A1_OUTCOME_ONLY | 0.566 | -0.004 | 27 | 29 | 0.8936946693232326 | -0.008 | 21 | 25 | 0.846 |
| A2_R3_OUTCOME_FIRST | 0.562 | -0.008 | 19 | 23 | 0.6434288435636206 | -0.012 | 20 | 26 | 0.824 |
| A3_VERIFIABLE_PROCESS | 0.530 | -0.04 | 18 | 38 | 0.011117560716106993 | -0.044 | 14 | 36 | 0.802 |
| A4_GATED_VERIFIABLE | 0.574 | 0.004 | 30 | 28 | 0.8955329031670437 | 0.0 | 21 | 21 | 0.832 |

## Gate verdicts (corrected)

- **A0_R0_CURRENT**: PASS — ok
- **A1_OUTCOME_ONLY**: PASS — ok
- **A2_R3_OUTCOME_FIRST**: PASS — ok
- **A3_VERIFIABLE_PROCESS**: FAIL — executable_rate 0.802 worse than control 0.84 by >2pp
- **A4_GATED_VERIFIABLE**: PASS — ok

## Lexicographic ranking (excl. A1 scientific control)

1. A0_R0_CURRENT
2. A4_GATED_VERIFIABLE
3. A2_R3_OUTCOME_FIRST

## Round 2 plan (if you proceed)

Arms: ['A0_R0_CURRENT', 'A4_GATED_VERIFIABLE', 'A2_R3_OUTCOME_FIRST']
Seed: 20260725