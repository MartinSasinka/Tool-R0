# Implementation Status — NESTFUL Synthetic Curriculum v3 / v3.1

Date: 2026-07-03  
**Training started: NO**

## v3.1 readiness

| level | status |
|-------|--------|
| v3.1 build + preflight | **PASS_PILOT_READY** |
| final dataset audit | **WARN** (stage3 non-scalar share soft) |
| v3.1 pod dry-run | **ready** |
| v3.1 stage1–2 pilot | **ready** (requires `ALLOW_PROTOTYPE_TRAINING=1`) |
| final experiment | **NO** — NESTFUL bigram overlap still low |

## v3.1 dataset

| metric | value |
|--------|-------|
| Full trajectories | **888** |
| stage1_1call_atomic | **800** (exact 1 call) |
| stage2_2call_dependency | **800** (exact 2 calls) |
| stage3_3call_composition | **800** (exact 3 calls) |
| stage4_4to6call_persistence | **800** (4–6 calls) |
| Unique questions | **3200 (ratio 1.0)** |
| Exact duplicates | **0** |
| Trace dup ratio (mean) | **0.0003** |
| Gold replay | **100%** |
| Process filter pass | **100%** |
| Question-trace alignment | **PASS** |
| Exact num_calls integrity | **PASS** |
| Tool realism | **pilot_ready** (37 offered, 25 used) |
| Preflight | **PASS_PILOT_READY** |
| Tests | **41/41** passed |

## v3 legacy (prototype pilot)

| metric | value |
|--------|-------|
| Total tasks | 1030 |
| Best checkpoint | s1_e2 (+0.020 dev Win) |
| Preflight | PASS_PROTOTYPE_ONLY |

## New v3.1 scripts

- `scripts/tool_registry_v3_1.py`
- `scripts/traj_utils_v3_1.py`
- `scripts/generate_full_motif_trajectories_v3_1.py`
- `scripts/build_prefix_curriculum_from_trajectories.py`
- `scripts/process_filter_prefix_samples.py`
- `scripts/validate_curriculum_integrity_v3_1.py`
- `scripts/replay_synthetic_gold_traces_v3_1.py`
- `scripts/run_tool_family_realism_v3_1.py`
- `scripts/build_curriculum_v3_1_pipeline.py`
- `scripts/analyze_stage_transfer_v3_1.py`
- `lib/reward_v3_1.py`
- `configs/curriculum_v3_1.yaml`, `configs/reward_v3_1_stepwise.yaml`

## Next command

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/build_curriculum_v3_1_pipeline.py
```

Then pod dry-run:

```bash
DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 CURRICULUM_VERSION=v3_1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```
