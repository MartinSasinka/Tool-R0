# Next Action Decision

**Recommendation: A — Continue with improved synthetic v3.1**

## Evidence

- Best checkpoint: **s1_e1** dev Win **0.565** vs baseline **0.56** (Δ **+0.005**).
- Stage 2 dead_group_rate ~**78%** — training signal collapsed in mixed stage.
- Prototype-only tool registry; no final NESTFUL transfer claim.
- Stage 1 improved real dev Win — pipeline **can work** with better data/reward.

## Risks

- +2pp on 200-task subset may not hold on full dev without per-sample verification.
- Continuing stage2-style mixed training without dataset fix may erase s1 gains.

## Exact next steps (no training yet)

1. Sync pod run dir locally for per-sample reports.
2. Implement v3.1 generator changes (long_chain + independent_calls + tool realism).
3. Re-run preflight gates.
4. Next training command (after v3.1):

```bash
ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
  STAGES="1 2" MAX_EPOCHS_PER_STAGE=2 \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Use checkpoint from s1_e2 only if regression guard confirms dev Win ≥ baseline.

## What not to claim

- Not SOTA.
- Not final NESTFUL transfer (prototype_only).
- F1 Func is diagnostic only (high ~0.87–0.88 despite low Win).
- Do not run test split until full dev gates pass.
