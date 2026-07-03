# Final Pilot Readiness Report (v3.1)

**Overall audit status:** WARN
**Pilot decision:** READY_FOR_POD_DRY_RUN

## A. Dataset summary

| metric | value | status |
|---|---:|---|
| stage1 1call atomic count | 800 | WARN |
| stage1 1call atomic call-count integrity | PASS | WARN |
| stage2 2call dependency count | 800 | WARN |
| stage2 2call dependency call-count integrity | PASS | WARN |
| stage3 3call composition count | 800 | WARN |
| stage3 3call composition call-count integrity | PASS | WARN |
| stage4 4to6call persistence count | 800 | WARN |
| stage4 4to6call persistence call-count integrity | PASS | WARN |

## B. Quality gates

| gate | value | status |
|---|---:|---|
| gold replay success | 1.0 | PASS |
| process filter pass | 1.0 | PASS |
| question-trace alignment failures | 0 | PASS |
| unresolved placeholders | 0 | PASS |
| metadata leakage | 0 | PASS |
| exact duplicates | 0 | PASS |
| trace duplicate ratio (mean) | 0.0003 | PASS |
| invalid references | 0 | PASS |
| duplicate sample IDs | 0 | PASS |
| preflight status | PASS_PILOT_READY | — |

## C. Diversity

| metric | value |
|---|---:|
| unique questions | 3200 |
| unique question ratio (mean) | 1.0 |
| used tool names | 25 |
| offered tool names | 37 |
| used tool families | 6 |
| stage2+ non-scalar share | 0.3258 |
| stage3 non-scalar share | 0.1575 |
| stage4 non-scalar share | 0.2687 |

## D. Remaining caveats

- Synthetic data remains synthetic; tool-family realism is **pilot_ready**, not final NESTFUL overlap.
- NESTFUL test split was not used for generation or training.
- Final test eval has not been run.
- **Training has not started.**
- Stage3/4 training remains gated behind stage1–2 pilot success on dev.

## E. Decision

**READY_FOR_POD_DRY_RUN**

### Local build and validation

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3_1_pipeline.py
```

### Pod dry-run

```bash
cd /workspace/Tool-R0

DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

### Stage1 pilot

```bash
cd /workspace/Tool-R0

ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  CURRICULUM_VERSION=v3_1 STAGES="1" MAX_EPOCHS_PER_STAGE=2 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```
