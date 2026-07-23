# Next steps roadmap (post pure Stage-3 stop)

**Updated:** 2026-07-23T06:35:03.621507+00:00

## Completed offline analyses

1. Discordant audit — `PURE_STAGE3_DISCORDANT_AUDIT.*`
2. Counterfactual R_train — `PURE_STAGE3_REWARD_COUNTERFACTUAL.*`
3. Per-turn accuracy — `PURE_STAGE3_PER_TURN_ACCURACY.*`
5. Structural similarity — `PURE_STAGE3_STRUCTURAL_SIMILARITY.*`
6. Credit decomposition — `PURE_STAGE3_CREDIT_DECOMPOSITION.*`

## Blocked / not run

4. **Synthetic held-out (200–300 new Stage-3 tasks)** — requires fresh generation with question/template/tool-combo dedup, then C0/E1/E2 eval at temp=0 and train-config rollouts.

## Decision tree (unchanged)

- Reward misalignment (E2 loss often R_train > C0 win) → **B1 outcome-first** ablation
- Held-out↑, NESTFUL↓ → new data + SFT pilot
- Held-out flat + correct reward → LR/KL ablation (deferred)

## Do not run yet

- Third epoch same recipe
- LR increase
- SFT against 60% under-calling metric
- Large Stage 4/5 dataset before held-out + reward audit