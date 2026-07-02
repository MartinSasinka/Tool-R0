# Preflight Gates Report

Status: **PASS_PROTOTYPE_ONLY**
- prototype_only mode: True

## Gate results
- invalid_task_rate: 0.0000
- duplicate_task_ids: 0
- invalid_reference_rate: 0.0000
- gold_replay_success_rate: 1.0
- motif_coverage: 80.0%
- baseline_failure_motif_coverage: 100.0%
- tool_family_realism: partial_tool_realism
- dev/test leakage: False

## Hard failures
- (none)

## Soft warnings
- (none)

## Training policy
- `FAIL`: training must NOT start
- `PASS_PROTOTYPE_ONLY`: training allowed only with `ALLOW_PROTOTYPE_TRAINING=1`
- `PASS_FINAL_READY`: training allowed without prototype override

Recommended: **PASS_PROTOTYPE_ONLY**
