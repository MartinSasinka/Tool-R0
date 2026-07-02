# Audit recommendations

Prioritized, evidence-linked. Nothing here has been auto-applied to training/eval code.

## P0 — change the training target (root cause #1)
- **Switch the training reward to `execution_aware`.** It is the best-aligned target: pearson **0.876** vs Win, false-positive 0.7%, false-negative 0.0% (`reward_audit_summary.csv`, `execution_reward_correlation_summary.csv`), vs strict 0.387 / partial 0.636. Strict/partial actively penalize winning alternative paths (`reward_audit_cases.csv:correct_answer_alt_path` → strict 0.00 / exec 0.90), which is exactly what the curriculum reward did to Win (`training_process_audit_summary.csv`).
- Enable via `--override reward.train_policy=execution_aware` (already implemented + unit-tested; `execution_reward.py`).

## P0 — fix checkpoint selection (root cause #3)
- **Select and carry forward by validation ReAct Win, not `strict_gold_trace_pass`.** Today `run_curriculum.sh:762-773,955-957` carries the best strict-pass checkpoint and `advance_threshold=0.50` is never reached (max ≈0.40), so the *final* checkpoint is the last/worst stage. The best Win checkpoint is the earliest (s1_e4 = 0.543). The `best_react_win_adapter` machinery already exists in the `stabilized_curriculum` profile (`EVAL_EVERY_EPOCH=1`) — use it for final selection.

## P1 — stabilize optimization (root cause #2)
- Partial train reward falls **0.86 → 0.26** across the curriculum with ~380 GRPO updates/epoch while Win and F1-Func collapse (`training_process_audit_summary.csv`, `diagnostics.csv`). Mitigations:
  - lower LR (e.g. 0.5e-6) and raise KL (e.g. 0.04) — `stabilized_curriculum` defaults.
  - mixed curriculum replay (stages 1..N) to fight forgetting (`data.load_tasks_mixed`).
  - per-epoch validation **ReAct Win** + early stop on no-improvement (already wired).
- **Log KL, entropy, and grad-norm per step.** They are currently absent from `train_log.jsonl`, so policy-drift magnitude is unquantified (`grpo_train.py` `_wandb_log_task`).

## P1 — close train/eval gaps (root causes #4, #5)
- Train uses `SYSTEM_PROMPT` + strict parser + `max_turns=gold_n`; eval adds `_EVAL_HARDENING` + lenient parser + `gold_n+1` turns (`prompt_mismatch_audit_summary.csv`). Unify so the policy is optimized under the same conditions it is scored.
- Add NESTFUL-style tasks to the training mix or at least keep NESTFUL as the per-epoch validation set (train↔eval domains are disjoint: 0/1861 overlap).

## P2 — tighten reward edge cases (root cause #6)
- `partial_reward.py` does not zero on a parse failure (keeps prefix credit 0.35) — consider gating like strict (`reward_audit_cases.csv:parse_error`).
- Extra calls beyond gold are unpenalized (`length_penalty=0`) — consider a small penalty (`reward_audit_cases.csv:extra_calls`).
- strict/partial don't validate references/values (executor-gated). Keep `execution_aware`'s `valid_references` and the hard/soft caps.

## P2 — fill evidence gaps (limitations)
- Run the **no-tool ablation** (forbid tool calls / drop tool outputs) and **shuffled-observation ablation** on a small subset; Win must drop sharply if ReAct truly depends on execution. Not run here per the no-expensive-eval constraint.
- Restore checkpoint **sidecars** (`trainer_state.json`, `config_used.json`) on disk for lineage; none are currently present.
- Track `experiments/` in git (currently untracked working state under commit `fd222c7`).

## Pilot success gates (copy into the run plan)
- Validation ReAct Win ≥ **0.544** (baseline) by end of stage 1, and **non-decreasing** across stages.
- Train reward and validation Win move **together** (no decoupling like the partial run).
- F1 Func ≥ **0.85** throughout (no degeneration; partial run fell to 0.35).
- zero_tool_calls < 0.10; clipped < 0.05.
- Early-stop / abort if validation Win drops > 0.005 for one eval.
