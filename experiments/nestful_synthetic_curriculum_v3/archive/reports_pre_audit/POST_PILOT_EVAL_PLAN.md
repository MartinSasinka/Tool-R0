# Post-Pilot Eval Plan

Date: 2026-07-02  
**Not run yet — prepare only**

## 1. Find best checkpoint

After pilot completes:

```bash
CKPT_ROOT=experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp>
ls "$CKPT_ROOT/best_react_win_adapter"
cat "$CKPT_ROOT/regression_guard_summary.json"  # if present
```

Best = highest dev ReAct Win **≥ baseline dev Win** (regression guard selection).

## 2. Dev eval (not test)

```bash
cd /workspace/Tool-R0
DATASET=experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl \
  CKPT_ROOT=experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp> \
  DRY_RUN=0 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_eval_v3.sh
```

Or partial parallel eval on dev split.

## 3. Motif-level eval (CPU)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/motif_level_eval.py \
  --split experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl \
  --out_dir experiments/nestful_synthetic_curriculum_v3/outputs/post_pilot_dev
```

Outputs:
- `motif_level_eval.csv`
- `motif_level_eval_buckets.csv`
- `MOTIF_LEVEL_EVAL_REPORT.md`

## 4. Compare model vs baseline on dev

Check:
- aggregate dev Win delta
- per-motif gains/regressions
- baseline-win tasks regressed?
- baseline-fail tasks gained?

## 5. Final test eval — only if ALL true

| gate | required |
|------|----------|
| dev Win ≥ baseline dev Win | yes |
| no severe trace drift (calls −20%, strict_trace −30%) | yes |
| motif-level gains on ≥1 failure cluster | yes |
| regression guard passed | yes |

Then:

```bash
DATASET=experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl \
  CKPT_ROOT=experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp> \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_eval_v3.sh
```

**Never run test eval if dev gates fail.**
