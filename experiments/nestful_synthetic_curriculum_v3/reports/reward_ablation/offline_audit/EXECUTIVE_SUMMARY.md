# Executive summary

1. Hash-matched completions received IDENTICAL rewards in every arm pair (reward_pearson=1.0) — arms shared ONE reward function; see PAIRWISE_SIGNAL_SIMILARITY.md and the dispatch check in DISCOVERY.md
2. Best synthetic on-policy success: A4_GATED_VERIFIABLE
3. Proxy warning arms: []
4. Update strength: see OPTIMIZER_SIGNAL_AUDIT.md
5. A0≈A4 heuristic: False
6. Main suspicion: **REWARD_DISPATCH_BUG**
7. Next experiment: Fix reward dispatch (config policy must win over REWARD_POLICY env default), then re-run the ablation arms.
8. Do not run: Any cross-arm reward conclusion from this round