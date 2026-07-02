# Evaluation Plan

**GPU eval not run yet.**

## During pilot

- val_eval on `nestful_dev.jsonl` after each epoch
- monitor metrics in `STAGE1_2_PILOT_PLAN.md`

## Post-pilot

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/motif_level_eval.py \
  --split experiments/nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl
```

Full procedure: `outputs/POST_PILOT_EVAL_PLAN.md`

Final test eval only if dev gates pass.
