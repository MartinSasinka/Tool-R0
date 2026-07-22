# Pure Stage 3 — Reward & Credit Alignment

Generated: 2026-07-22T14:03:48.902477+00:00

**Source:** overnight train logs (`pure_stage3_2ep_20260719_221918`, 326×2 groups).
Eval paired section uses smoke test C0/E2; reward audit uses **training rollouts**, not NESTFUL eval.

## A. Reward vs outcome (train proxy)

- mean reward win rollouts: 1.0
- mean reward loss rollouts: 0.47926261903478845
- pairwise ordering (call-count proxy): {'correct': 2199, 'reversed': 95, 'tie': 0}

## B–D. Credit assignment

- R²(G₀ ~ episode_reward): **0.9821934094410196**
- corr(G₀, traj_length): 0.7007178900007827
- too_few vs full reward gap: {'n_groups': 195, 'mean_too_few_reward': 0.21713760324786316, 'mean_full_reward': 0.5722273162637359}

## Offline credit schemes (A0–A3)

- **A0_current**: dead_pos=0.249, good&neg_adv=4569, bad&pos_adv=62
- **A1_no_episode**: dead_pos=0.271, good&neg_adv=4335, bad&pos_adv=60
- **A2_local**: dead_pos=0.544, good&neg_adv=2438, bad&pos_adv=40
- **A3_local_plus_outcome**: dead_pos=0.391, good&neg_adv=3790, bad&pos_adv=40

## Checkpoint delta (overnight E1→E2 weights)

- rel move E1→E2: 0.0025506168825353537
- cosine(E1,E2): 0.9999967581322621

**Interpretation:** episode reward dominates G₀; A0 beats A1/A2/A3 on dead-position rate.
Independent IBM outcome re-score of train rollouts still recommended.
