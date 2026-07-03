# NESTFUL Synthetic Curriculum v3 / v3.1

**Training started: NO**

## Versions

| version | description | status |
|---------|-------------|--------|
| **v3** | motif-aligned prototype (max_calls staging) | prototype-only pilot completed |
| **v3.1** | prefix/motif-aware **exact call-count** curriculum | **PASS_PILOT_READY** — build complete |

v3.1 decomposes full NESTFUL failure trajectories into prefix samples per stage (1/2/3/4–6 calls). Long-chain failures are **not** placed whole into stage1/2.

## v3.1 build status

| gate | value |
|------|-------|
| Full trajectories | **888** |
| stage1 (exact 1 call) | **800** |
| stage2 (exact 2 calls) | **800** |
| stage3 (exact 3 calls) | **800** |
| stage4 (4–6 calls) | **800** |
| Final dataset audit | **WARN** (soft only) |
| Unique questions | **3200/3200 (100%)** |
| Exact duplicates | **0** |
| Gold replay | **100%** |
| Question-trace alignment | **PASS** |
| Preflight | **PASS_PILOT_READY** |
| Training started | **NO** |

Next step: pod dry-run, then stage1 pilot.

## Local build v3.1

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3_1_pipeline.py
```

## Local preflight v3.1

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/run_preflight_gates.py \
  --curriculum-version v3_1 --prototype-only
```

## Pod dry-run (v3.1)

```bash
cd /workspace/Tool-R0

DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

## Pod pilot (v3.1 stage1–2)

```bash
cd /workspace/Tool-R0

ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  CURRICULUM_VERSION=v3_1 STAGES="1 2" MAX_EPOCHS_PER_STAGE=2 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Reward default for v3.1: `execution_aware_v3_1_stepwise`. Stage3/4 gated.

## Post-pilot analysis

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/analyze_stage_transfer_v3_1.py
python experiments/nestful_synthetic_curriculum_v3/scripts/motif_level_eval.py
```

## v3 legacy run order (unchanged)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/generate_motif_synthetic_tasks.py
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3.py
python experiments/nestful_synthetic_curriculum_v3/scripts/validate_synthetic_tasks.py
python experiments/nestful_synthetic_curriculum_v3/scripts/replay_synthetic_gold_traces.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_preflight_gates.py --prototype-only
pytest experiments/nestful_synthetic_curriculum_v3/tests -q
```

See also: `outputs/CURRICULUM_V3_1_DESIGN_DECISION.md`, `outputs/curriculum_v3_1/CURRICULUM_V3_1_IMPLEMENTATION_REPORT.md`.
