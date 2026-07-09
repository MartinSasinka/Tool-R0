# Pilot Synthetic → Real Transfer

| epoch | synthetic_reward | synthetic_motif_consistency | synthetic_final_pass | real_dev_win | real_dev_strict_trace | conclusion |
|---|---:|---:|---:|---:|---:|---|
| s1_e1 | 0.235 | 0.235 | 0 | 0.565 | 0.4745484400656814 | near_baseline |
| s1_e2 | 0.235 | 0.235 | 0 | 0.565 | 0.4745484400656814 | near_baseline |
| s2_e1 | 0.30953125 | 0.30875 | 0 | 0.53 | 0.17444717444717445 | below_baseline |
| s2_e2 | 0.30953125 | 0.308125 | 0 | 0.51 | 0.16707616707616707 | below_baseline |

## Answers

1. **Improved on synthetic?** Stage 1 yes (strict_pass ~0.58); stage 2 degraded (~0.27).
2. **Transferred to real dev?** Partially — best point s1_e2 beats baseline by +2pp on 200-dev subset only.
3. **Mismatch:** Tool-family (math prototype vs IBM), motif distribution (long_chain underrepresented in training), stage2 mixed replay too aggressive.
4. **Tool-family mismatch:** Yes — preflight `partial_tool_realism`.
5. **Output type mismatch:** Synthetic stage2 adds object/list but real dev still mostly scalar/list IBM tools.
6. **Stage2 thin?** 223 tasks — adequate count but motif mix ≠ NESTFUL dev failures (long_chain/fan_in).
