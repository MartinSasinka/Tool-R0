# Pilot Run Analysis

Generated: 2026-07-03 07:35 UTC

## Run metadata

| field | value |
|---|---|
| run_dir | C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\20260702_112150 |
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
| baseline_dev | 0.555 | 0.555 | 0.0 | 0.02 | 0.192 | 0.87 | None | None | baseline |
| s1_e1 | 0.54 | 0.555 | -0.015000000000000013 | 0.015 | 0.195 | 0.877 | 0.4876847290640394 | 0.3884892086330935 | below_baseline |
| s1_e2 | 0.575 | 0.555 | 0.019999999999999907 | 0.02 | 0.187 | 0.88 | 0.4860426929392447 | 0.3932853717026379 | beats_baseline |
| s2_e1 | 0.545 | 0.555 | -0.010000000000000009 | 0.015 | 0.181 | 0.869 | 0.1891891891891892 | 0.6859375 | below_baseline |
| s2_e2 | 0.53 | 0.555 | -0.025000000000000022 | 0.01 | 0.188 | 0.861 | 0.16953316953316952 | 0.6796875 | below_baseline |

## Answers (aggregate)

1. **Beat baseline?** Yes — s1_e2 dev Win=0.575 vs baseline 0.555 (Δ=+0.020)
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
