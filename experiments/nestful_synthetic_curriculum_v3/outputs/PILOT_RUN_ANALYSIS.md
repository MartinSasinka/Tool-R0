# Pilot Run Analysis

Generated: 2026-07-07 06:30 UTC

## Run metadata

| field | value |
|---|---|
| run_dir | C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\0260703_145219_v3_1 |
| timestamp | 20260702_112150 |
| stages | 1, 2 (2 epochs each) |
| dry_run | False |
| reward | execution_aware_v2_1_motif |
| train_dataset | curriculum_v3 synthetic (1030 tasks; stage1=417, stage2=223 on pod) |
| dev_split | C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\splits\nestful_dev.jsonl |
| allow_prototype_training | True |
| stage3_4_blocked | True |
| wandb_project | nestful-mtgrpov2-corection |
| missing_local_run_dir | False |
| missing_inputs | [] |
| val_subset_size | 200 |
| val_subset_seed | 42 |

## Aggregate dev Win (official, 200-task deterministic subset)

| checkpoint/epoch | dev_win | baseline | delta | full_acc | partial_acc | f1_func | strict_trace | dead_group_rate | conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_dev | 0.56 | 0.56 | 0.0 | 0.02 | 0.187 | 0.864 | None | None | baseline |
| s1_e1 | 0.565 | 0.56 | 0.004999999999999893 | 0.03 | 0.196 | 0.872 | 0.4745484400656814 | 1 | near_baseline |
| s1_e2 | 0.565 | 0.56 | 0.004999999999999893 | 0.03 | 0.196 | 0.872 | 0.4745484400656814 | 1 | near_baseline |
| s2_e1 | 0.53 | 0.56 | -0.030000000000000027 | 0.025 | 0.195 | 0.864 | 0.17444717444717445 | 0.778125 | below_baseline |
| s2_e2 | 0.51 | 0.56 | -0.050000000000000044 | 0.02 | 0.194 | 0.883 | 0.16707616707616707 | 0.778125 | below_baseline |

## Answers (aggregate)

1. **Beat baseline?** No — best checkpoint still below or equal baseline.
2. **Near-baseline?** Stage 1 epoch 1 (0.540) and stage 2 epoch 1 (0.545) are within ~1–2.5pp of baseline 0.555.
3. **Trend:** Stage 1 improves epoch 1→2 (+3.5pp dev Win). Stage 2 regresses vs stage 1 best and vs baseline.
4. **Trace drift:** NESTFUL 3-call eval strict_pass drops to ~0.17–0.19 (eval-stage3) vs ~0.49 on 2-call (eval-stage2) — depth sensitivity increased after stage 2.
5. **Short-trace collapse:** zero_tool_calls on NESTFUL eval ~8.5–12.5%; not dominant but persistent.
6. **Stability:** Stage 1 dead_group ~39%; stage 2 dead_group ~68–69% — stage 2 training signal largely collapsed.

## Pipeline / reward

- Reward `execution_aware_v2_1_motif` was wired (v3/run.py) and training ran without fallback.
- Stage 1 synthetic strict_pass ~0.578; stage 2 ~0.268 — mixed replay + harder motifs reduced learnable groups.
- Prototype tool registry (partial_tool_realism) — **not** a final NESTFUL transfer claim.

## Missing inputs

- None — local trajectories found.

## Dev validation subset composition (n=200, seed=42)

Official dev Win was measured on this subset — **not** full dev (1861 tasks).

| motif_type | n | share |
|---|---:|---:|
| linear_dependency | 106 | 53.0% |
| long_chain | 63 | 31.5% |
| fan_in | 27 | 13.5% |
| independent_calls | 4 | 2.0% |
