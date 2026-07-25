# Pairwise signal similarity

**Mode: PARTIAL** — Stored training artifacts do not include parsed trajectories required for frozen registry re-scoring. Cross-arm comparison uses hash-matched rollouts and per-group logged episode_rewards with recomputed GRPO advantages.

## A0_R0_CURRENT vs A2_R3_OUTCOME_FIRST
- hash-matched rollouts: 25
- advantage cosine (hash-matched): 0.8804273645069388
- sign agreement: 0.8
- effectively equivalent (heuristic): False

## A0_R0_CURRENT vs A4_GATED_VERIFIABLE
- hash-matched rollouts: 35
- advantage cosine (hash-matched): 0.4919872570640359
- sign agreement: 0.6285714285714286
- effectively equivalent (heuristic): False

## A2_R3_OUTCOME_FIRST vs A4_GATED_VERIFIABLE
- hash-matched rollouts: 31
- advantage cosine (hash-matched): 0.803201296169162
- sign agreement: 0.6774193548387096
- effectively equivalent (heuristic): False

## A0_R0_CURRENT vs A1_OUTCOME_ONLY
- hash-matched rollouts: 29
- advantage cosine (hash-matched): 0.5208697362439445
- sign agreement: 0.6896551724137931
- effectively equivalent (heuristic): False
