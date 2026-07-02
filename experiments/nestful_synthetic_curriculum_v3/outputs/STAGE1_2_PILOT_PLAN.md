# Stage1–2 Pilot Plan

Date: 2026-07-02  
**Training started: NO**

## Scope

- **Stages:** 1 and 2 only (`STAGES="1 2"`)
- **Train data:** 417 stage1 + 223 stage2 synthetic tasks
- **Val:** real NESTFUL dev (`nestful_dev.jsonl`)
- **Max epochs:** 2 per stage
- **KL beta:** 0.15
- **Regression guard:** ON
- **No stage 3/4**, no final test eval until dev gates pass

## Pilot goals

1. Verify training pipeline runs end-to-end on pod
2. Verify `execution_aware_v2_1_motif` reward produces non-trivial signal
3. Check `dead_group_rate` is not extreme
4. Check dev Win stable or improves vs baseline
5. Check model avoids trace collapse (avg calls, strict_trace)
6. Check motif-level dev changes (post-epoch CPU eval)

## Metrics to monitor

| metric | source |
|--------|--------|
| dev_react_win | val_eval |
| baseline_dev_react_win | regression guard baseline |
| delta_vs_baseline | computed |
| dead_group_rate | training logs |
| reward_total | GRPO logs |
| motif_trace_consistency | reward diagnostics |
| final_answer_pass | reward diagnostics |
| executable_trajectory | reward diagnostics |
| valid_references | reward diagnostics |
| avg_num_calls | val_eval |
| strict_trace_pass | val_eval |
| zero_tool_calls | failure taxonomy |
| too_few_calls | failure taxonomy |
| invalid_reference_rate | trajectories |
| clipped_completion_rate | trajectories |

## Stop criteria (immediate abort)

- dev Win < baseline dev Win − **0.005**
- avg call count drops > **20%**
- strict_trace_pass drops > **30%**
- dead_group_rate > **0.80** for **2 consecutive epochs**
- no_tool_calls / too_few_calls spike
- reward increases but dev Win decreases

## Expected interpretation

Because tool realism is `partial_tool_realism`, **do not claim final NESTFUL transfer**.

Use pilot to decide:
- motif reward works
- GRPO signal exists
- stage1–2 curriculum is stable
- whether to invest in IBM/NESTFUL-like tool generator

## Launch command

```bash
cd /workspace/Tool-R0
ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 STAGES="1 2" \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

## Local preflight (before pod)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/run_preflight_gates.py --prototype-only
```
