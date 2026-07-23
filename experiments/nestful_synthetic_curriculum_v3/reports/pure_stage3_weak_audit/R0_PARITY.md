# R0 reward parity report

**Gate passed:** True
**Packet reward label:** `execution_aware_v3_2_dense_recomputed_eval`

## Train log checks

- n_groups: 652
- raw_episode_match_rate: 1.0
- mean_reward_match_rate: 1.0
- max_abs_mean_error: 2.220446049250313e-16
- policy_mismatch_groups: 0

## Eval sample (vs reward_train_strict — not training R0)

- n_eval_rows_sampled: 500
- mean_abs_diff_vs_reward_train_strict: 0.26028890999999976
- max_abs_diff_vs_reward_train_strict: 0.78
- note: eval stores reward_train_strict only; this is NOT training R0

## Limitations

- Train rollout trajectories are not persisted; full logged-vs-recomputed R0 parity on every completion is impossible from artifacts alone.
- Train log scalars are internally consistent and policy matches manifest, but eval packet rewards are still recomputed on saved eval trajectories (not logged train-time scalars).