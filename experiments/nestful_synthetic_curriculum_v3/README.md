# NESTFUL Synthetic Curriculum v3

**Training started: NO**

## Pilot status (2026-07-02)

| gate | status |
|------|--------|
| Tasks | **1030** (stage1=417, stage2=223) |
| Motif coverage | **80%** |
| Gold replay | **100%** |
| Tool realism | **partial_tool_realism** |
| Preflight | **PASS_PROTOTYPE_ONLY** |
| Final-ready | **NO** |

Coverage is **four-layer** — see `outputs/NESTFUL_COVERAGE_DEFINITION.md`.

## Run order (local CPU)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/generate_motif_synthetic_tasks.py
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3.py
python experiments/nestful_synthetic_curriculum_v3/scripts/validate_synthetic_tasks.py
python experiments/nestful_synthetic_curriculum_v3/scripts/replay_synthetic_gold_traces.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_distribution_audit.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_tool_family_realism.py
python experiments/nestful_synthetic_curriculum_v3/scripts/run_preflight_gates.py --prototype-only
pytest experiments/nestful_synthetic_curriculum_v3/tests -q
```

## Pod: DRY RUN first

```bash
cd /workspace/Tool-R0
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

See `outputs/POD_DRY_RUN_INSTRUCTIONS.md` and `outputs/STAGE1_2_PILOT_PLAN.md`.

## Pod: stage1–2 pilot (after DRY RUN PASS)

```bash
ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Requires `ALLOW_PROTOTYPE_TRAINING=1`. Stage 3/4 blocked.
