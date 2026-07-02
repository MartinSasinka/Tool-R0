# ROOT CAUSE ANALYSIS — v2 curriculum run underperformed baseline

Scope: `experiments/nestful_mtgrpo_partial/outputs/execution_v2_mixed_replay_full`
(training) and `.../outputs/final_eval_v2` (official ReAct test eval, n=1861).

## Headline (official NESTFUL ReAct, from CHECKPOINT_REEVAL_REPORT.md)

| checkpoint | Win Rate | Full Acc | Partial Acc | F1 Func | vs base |
|---|---|---|---|---|---|
| baseline   | 0.544 | 0.021 | 0.183 | 0.905 | (base) |
| stage2_e4  | 0.529 | 0.020 | 0.161 | 0.716 | -0.015 |
| stage3_e1  | 0.288 | 0.004 | 0.098 | 0.317 | -0.256 |

Monotonic degradation; `f1_func` collapses 0.905 → 0.716 → 0.317. Win/loss overlap
vs baseline: stage2_e4 net **-28** (172 gained / 200 regressed), stage3_e1 net
**-476** (131 / 607). Reward was confirmed `execution_aware_v2` (partial), not strict.

## Where degradation begins

- **stage2_e4 → stage3_e1** is the cliff (-0.24 Win, f1_func 0.72 → 0.32). Stage 3
  training accelerated trace drift / forgetting.
- Within stage 2 the slide is already present (F1 down from baseline 0.905).

## Root causes (ranked)

1. **Broken checkpoint selection — official val scorer crashed.**
   `nestful_official_score.score_items` computed Win via the official
   `calculate_win_score`, whose `if pred_ans == gold_ans` (scorer.py:127) raises
   `ValueError: truth value of an array with more than one element is ambiguous`
   when an executed answer is a numpy array. That aborted scoring for the WHOLE
   epoch, so 6/8 `val_eval` epochs logged `react_win_rate: null`. "Best" was
   effectively chosen from 2 noisy points (stage2/e2). Early stopping never fired
   because its metric was null.
   → FIXED: Win Rate is now computed per-sample (crash-isolated) and never null;
   val_eval hard-fails instead of silently continuing (see REMEDIATION_PLAN.md).

2. **Train/val on synthetic data, evaluated on real NESTFUL.**
   Training used `clean_curriculum/*`, validation used a synthetic split. The real
   NESTFUL test set (1861) was never in the optimization loop, so any synthetic
   gains did not transfer — and the reward could drift away from real-task success
   with no corrective signal.

3. **Gameable reward → trace drift.** `execution_aware_v2` with high `w_final`
   (0.55) and a high `floor_executable_final` let the model reach *an* answer via
   its own shorter trace. `strict_gold_trace_pass` fell (0.212 → 0.182 → 0.044)
   and `f1_func` collapsed while avg calls dropped (2.27 → 1.51 at stage3_e1):
   the model learned to emit fewer, non-gold calls that still sometimes execute.

4. **Weak GRPO signal + forgetting.** Easy stage-1 data yields many `dead_group`
   rollouts (all rollouts identical reward → zero advantage → no gradient). With
   low LR/KL sustained over 8 epochs, the net effect of the few live gradients
   plus KL pressure was drift/forgetting rather than improvement, worst after the
   stage-3 transition.

5. **No regression guard.** Nothing compared each epoch's Win to baseline; a
   checkpoint below baseline could still be saved as "best".

## Corroborating evidence

- `outputs/execution_v2_mixed_replay_full/**/metrics_epoch_*.json`:
  `react_win_rate: null` in 8/10 epochs (root cause #1).
- `val_win_rescore_summary.csv` (from `recompute_val_win_rates.py`): per-sample Win
  *was* stored in trajectories; rescored synthetic-val timeline shows stage1/e3 at
  0.613 while auto-selected best was stage2/e2 at 0.462 — selection ran on broken
  aggregate metrics, not true best.
- `final_eval_v2/*/final_eval_trajectories.jsonl`: falling `num_tool_calls` and
  `strict_gold_trace_pass`, stable low `parse_error_rate` (~0) — degradation is
  trace *content*, not parse failures (root cause #3).
- `TRAINING_RUN_DIAGNOSIS.md`: stage 1 dead_group_rate ~93%; stage 3 strict_trace
  proxy collapses 0.18 → 0.03 (root causes #4–5).
