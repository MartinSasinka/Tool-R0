# Training Plan

**Training started: NO**

## v3.1 (recommended next pilot)

- `CURRICULUM_VERSION=v3_1`, `STAGES="1 2"`
- Dataset: `outputs/curriculum_v3_1/filtered/` (800/stage, exact call counts)
- Preflight: **PASS_PILOT_READY** — pod dry-run allowed
- Reward: `execution_aware_v3_1_stepwise`
- Stage3/4 **gated** until stage1–2 dev gates pass

Local validation:
```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/final_dataset_audit_v3_1.py --use-filtered
python experiments/nestful_synthetic_curriculum_v3/scripts/run_preflight_gates.py --curriculum-version v3_1 --prototype-only
```

Pod dry-run:
```bash
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Stage1 pilot:
```bash
ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  CURRICULUM_VERSION=v3_1 STAGES="1" MAX_EPOCHS_PER_STAGE=2 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

## v3 legacy pilot scope

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
