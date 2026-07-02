# NEXT TRAINING DECISION

Date: 2026-07-02  
Run analyzed: `outputs/execution_v2_mixed_replay_full` + `outputs/final_eval_v2`

---

## Does any existing checkpoint beat baseline?

**No.** Official NESTFUL ReAct test (n=1861):

| checkpoint | Win Rate | vs baseline |
|---|---|---|
| baseline | **0.544** | — |
| stage2_e4 | 0.529 | −0.015 |
| stage3_e1 | 0.288 | −0.256 |

Auto-selected `best_react_win_adapter` (stage 2 / epoch 2) was **not** full-test evaluated in
`final_eval_v2/`. Its selection metric was `react_win_rate=0.462` on **synthetic_val.jsonl**
(one of only **2 non-null** val epochs; 8/10 were null due to scorer crash).

**Pod action:** eval `best_react_win` on full test before discarding:

```bash
CELLS="best_react_win=$CKPT_ROOT/best_react_win_adapter" \
  bash experiments/nestful_mtgrpo_partial/run_final_eval_v2_parallel.sh
```

Expectation: near **stage2_e4** (same stage) — still below baseline.

---

## Should we report stage2_e4 as a result?

**Only as a near-baseline negative result** (−0.015 Win, net −28 samples vs baseline).
Do **not** report it as an improvement. F1 Func dropped 0.905 → 0.716 (trace drift).

---

## Is another training run necessary?

**Yes — but only after Phase 1 remediation is deployed** (already in repo):

| Blocker (previous run) | Fixed in repo? |
|---|---|
| val scorer crash → null Win | ✅ per-sample Win + hard-fail |
| Selection on synthetic val | ✅ `nestful_dev.jsonl` default in `run_full_v2.sh` |
| Gameable reward → trace drift | ✅ execution_aware_v2 v2.1 weights |
| Weak GRPO + forgetting | ✅ KL 0.10, fewer epochs, dead_group logging |
| No regression guard | ✅ REGRESSION_GUARD + optional early abort |

**Do NOT re-run training with the old settings** (`synthetic_val`, KL 0.05, old reward weights).

---

## What to run next

1. **Optional (5 min GPU):** full-test eval of `best_react_win_adapter` (see above).
2. **Required before retrain:** `python experiments/comparison/make_nestful_dev_split.py` on pod.
3. **Retrain:** `USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_full_v2.sh`
4. **Report on test only:** `DATASET=.../nestful_test.jsonl run_final_eval_v2_parallel.sh`

---

## Key lesson from rescored val trajectories

Per-sample Win was actually computed during val_eval (stored in trajectories), even when
`metrics_epoch_*.json` logged `react_win_rate=null`. Rescored synthetic-val timeline:

| stage | epoch | rescored Win (synthetic val) |
|---|---|---|
| 1 | 3 | **0.613** (best on synthetic) |
| 2 | 2 | 0.462 ← **auto-selected "best"** (only non-null epoch at selection time) |
| 2 | 4 | 0.613 |
| 3 | 1 | 0.527 |
| 3 | 2 | 0.237 |
| 3 | 3 | 0.027 |

Selection picked stage2/e2 because **6/8 val epochs had null aggregate Win** — not because
it was truly best. Even on synthetic val, stage1/e3 was better. On real NESTFUL test, none
transfer — confirming dev must be real NESTFUL and scorer must never return null.

---

## Decision

| Question | Answer |
|---|---|
| Report trained checkpoint as headline? | **No** |
| Report stage2_e4? | Optional footnote (−0.015 Win) |
| Run remediation training? | **Yes**, after deploying Phase 1 fixes (in repo) |
| Run old config again? | **No** |
