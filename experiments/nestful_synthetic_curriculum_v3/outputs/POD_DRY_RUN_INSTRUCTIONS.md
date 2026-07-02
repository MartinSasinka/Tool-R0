# Pod DRY RUN Instructions

Date: 2026-07-02  
**Training started: NO**

## Prerequisites

1. Repo cloned at `/workspace/Tool-R0`
2. `curriculum_v3/` built (1030 tasks) — commit or regenerate on pod
3. NESTFUL dev/test splits present under `experiments/nestful_mtgrpo_minimal/data/splits/`
4. GPU available (for non-DRY training later)

## DRY RUN command

```bash
cd /workspace/Tool-R0

DRY_RUN=1 ALLOW_PROTOTYPE_TRAINING=1 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

## Expected log (PASS)

```
[curriculum_v3] preflight checks ...
[validate_synthetic_tasks] 1030 tasks, 0 failures
[run_distribution_audit] status=PASS coverage=80.0% ...
[replay_synthetic_gold_traces] PASS rate=1.0000 ...
[run_tool_family_realism] status=partial_tool_realism ...
[run_preflight_gates] status=PASS_PROTOTYPE_ONLY ...
[curriculum_v3] CONFIG=.../nestful_mtgrpo_partial/config.yaml
[curriculum_v3] RUN_PY=.../nestful_synthetic_curriculum_v3/run.py
[curriculum_v3] preflight=PASS_PROTOTYPE_ONLY
[curriculum_v3] DRY_RUN=1 — skipping training invocation
```

## PASS means

- All preflight gates passed
- Stage symlinks created under `outputs/runs/<timestamp>/data_base/`
- Reward override `execution_aware_v2_1_motif` configured
- **No training started**

## FAIL means

| symptom | action |
|---------|--------|
| preflight `FAIL` | read `outputs/PREFLIGHT_GATES_REPORT.md`, fix dataset |
| gold replay < 100% | re-run `replay_synthetic_gold_traces.py`, fix generator |
| missing dev split | run `make_nestful_dev_split.py` |
| `ALLOW_PROTOTYPE_TRAINING` not set | export before command |

## Actual pilot command (after DRY RUN PASS)

```bash
cd /workspace/Tool-R0

ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Output: `experiments/nestful_synthetic_curriculum_v3/outputs/runs/<timestamp>/`
