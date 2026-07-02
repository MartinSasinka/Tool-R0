# Training Plan

**Training started: NO**

## Pilot scope

- STAGES="1 2" only
- stage1: 417 tasks, stage2: 223 tasks
- max 2 epochs/stage, KL 0.15, regression guard ON
- reward: `execution_aware_v2_1_motif` via `v3/run.py`
- base config: `nestful_mtgrpo_partial/config.yaml`

## Preflight (automatic in run_curriculum_v3.sh)

All gates must pass; prototype requires `ALLOW_PROTOTYPE_TRAINING=1`.

## Commands

DRY RUN:
```bash
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Pilot:
```bash
ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

See `outputs/STAGE1_2_PILOT_PLAN.md` for stop criteria.
