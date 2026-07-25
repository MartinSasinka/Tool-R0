# Round 1 — INVALID (reward dispatch bug)

**Label:** `reward_ablation_round1_INVALID_DISPATCH`

**Status:** do not interpret as a reward ablation.

## Binding conclusion

Round 1 cannot be interpreted as a reward ablation because **every arm
trained with `execution_aware_v3_2_dense`**.

Evidence (raw artifacts, forensic audit 2026-07-25):

- every `train_log.jsonl` row: `reward_policy_resolved = execution_aware_v3_2_dense`
- console: `[v3/run.py] training reward = execution_aware_v3_2_dense` after
  `[override] reward.train_policy = 'reward_ablation_<ARM>'`
- hash-matched completions across all arm pairs: identical rewards
  (`max_abs_diff = 0.0`)

Full write-up: `reports/root_cause_forensic/IMPLEMENTATION_BUG_AUDIT.md`
and `ROOT_CAUSE_REPORT.md`.

## Forbidden claims about Round 1

Do **not** use these statements (they compare five replicates of one config):

- “A3 is a worse reward”
- “A4 is the best reward”
- “A1 lowered dead groups thanks to the outcome-only reward”

Win-rate spread 53.0–57.4 % on n=500 is **replicate noise**, not a reward effect.

## What to do next

1. Run the dispatch canary (A1 + A4, 24 tasks × 8 rollouts, no NESTFUL eval).
2. Only after the canary passes: use `A4_GATED_VERIFIABLE` as the working
   reward for the next data/transfer experiment (not because Round 1 proved
   it best — it did not).
3. Do **not** launch another 5-arm reward ablation until the canary passes.

See: `reports/reward_ablation/DISPATCH_CANARY_RUNBOOK.md`
