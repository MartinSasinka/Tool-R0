# REMEDIATION PLAN — NESTFUL v2 pipeline

Status: **Phase 0 complete (code + offline analysis)** | **Phase 1 ready for next pod run**

Scope: fix the five root causes that made `execution_v2_mixed_replay_full` underperform
baseline (Win 0.544) on the official NESTFUL ReAct test set.

---

## Phase 0 — Truthful picture (DONE, no retrain)

| Item | Status | Artifact / change |
|---|---|---|
| Fix official val scorer crash (numpy ambiguous Win) | **DONE** | `nestful_official_score.py` — per-sample Win, empty-batch guard |
| Hard-fail val_eval when `react_win_rate` is null | **DONE** | `run.py:mode_val_eval`, `run_curriculum.sh` exit 2 |
| `_run_logged` propagates piped exit codes | **DONE** | `run_curriculum.sh` (PIPESTATUS) |
| Re-score val timeline from trajectories | **DONE** | `recompute_val_win_rates.py` → `val_win_rescore_summary.csv` |
| Full-test checkpoint report | **DONE** | `CHECKPOINT_REEVAL_REPORT.md` (baseline, stage2_e4, stage3_e1) |
| Training run diagnosis | **DONE** | `TRAINING_RUN_DIAGNOSIS.md` |
| Parallel final-eval launcher incl. `best_react_win` | **DONE** | `run_final_eval_v2_parallel.sh` |
| Unit tests for scorer robustness | **DONE** | `tests/test_nestful_official_score.py` |

**Pod action (optional, no retrain):** run missing full-test cell for auto-selected checkpoint:

```bash
CELLS="best_react_win=$CKPT_ROOT/best_react_win_adapter" \
  bash experiments/nestful_mtgrpo_partial/run_final_eval_v2_parallel.sh
```

Use `DATASET=$MINIMAL/data/splits/nestful_test.jsonl` only after Phase 1 retrain (dev used for selection).

---

## Phase 1 — Full remediation (before next training run)

### 1. Real NESTFUL dev for validation / selection

- **DONE:** `make_nestful_dev_split.py` → `data/splits/nestful_dev.jsonl` (200) + `nestful_test.jsonl` (1661)
- **DONE:** `run_curriculum.sh` defaults to `nestful_dev.jsonl` when present
- **DONE:** `run_full_v2.sh` sets `VAL_JSONL=nestful_dev.jsonl` (was `synthetic_val.jsonl`)

### 2. Regression guard

- **DONE:** `REGRESSION_GUARD=1` — measure baseline dev Win before stage 1; never crown checkpoint below baseline + margin
- **DONE:** `run_full_v2.sh` enables `REGRESSION_EARLY_ABORT=1` (patience 3)

### 3. Hardened execution_aware_v2 reward (anti trace-drift)

- **DONE:** `nestful_core/rewards.py` v2.1 defaults + `config.yaml execution_reward_v2` block
  - Lower `w_final` (0.35), higher `w_gold_trace` / `w_completeness`
  - Gated floor (`floor_gold_trace_min`), `cap_final_no_gold_trace`
  - `short_trace_penalty` for missing calls vs gold

### 4. Anti-forgetting

- **DONE:** `run_full_v2.sh`: `STABILIZED_KL=0.10`, `EPOCHS_PER_STAGE=2`
- **DONE:** `config.yaml`: `training.kl_beta=0.10`
- **DONE:** `grpo_train.py`: skip dead groups (existing), log `dead_group_rate`, `reward_train_policy` in `train_log.jsonl`

### 5. Reporting split

| Set | Path | Use |
|---|---|---|
| Train | `clean_curriculum/*` + mixed replay | GRPO only |
| Dev | `splits/nestful_dev.jsonl` | val_eval, checkpoint selection, early stop |
| Test | `splits/nestful_test.jsonl` | final_eval ONLY (never train/val) |

---

## Phase 1 launch command (pod)

```bash
cd /workspace/Tool-R0
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Ensure dev/test split exists (idempotent)
python experiments/comparison/make_nestful_dev_split.py

USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  bash experiments/nestful_mtgrpo_partial/run_full_v2.sh
```

Final reporting (after training):

```bash
DATASET=experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl \
  bash experiments/nestful_mtgrpo_partial/run_final_eval_v2_parallel.sh
```

---

## Success gates (next run)

1. **Zero** `react_win_rate=null` in any `val_eval/metrics_epoch_*.json`
2. `best_react_win_adapter` dev Win ≥ baseline dev Win (regression guard)
3. Full-test Win on `nestful_test.jsonl` **>** baseline 0.544 (headline metric)
4. `dead_group_rate` logged; stage 1 expected high, should drop in stage 2+
5. `strict_gold_trace_pass` on rollout_eval should not collapse at stage 3
