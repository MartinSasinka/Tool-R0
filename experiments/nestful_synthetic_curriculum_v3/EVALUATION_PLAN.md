# Evaluation Plan

**GPU eval not run yet.** Final test eval not run. NESTFUL test split not used for training.

Pre-pilot validation:
```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/final_dataset_audit_v3_1.py --use-filtered
```

## v3.1 post-pilot

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analyze_stage_transfer_v3_1.py
python experiments/nestful_synthetic_curriculum_v3/scripts/motif_level_eval.py
```

Track dev Win by num_calls bucket (1, 2, 3, 5–8) and too_few_calls rate.

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
