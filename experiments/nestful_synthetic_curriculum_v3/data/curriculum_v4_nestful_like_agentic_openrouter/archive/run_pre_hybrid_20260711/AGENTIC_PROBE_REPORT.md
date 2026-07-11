# Agentic dataset — stage probe report

STATUS: **not yet run** (requires a GPU pod; never run automatically by the builder).

Run on the pod, then paste/compare results here:

```bash
# v3.1 stage2 baseline signal
DATASET=stage2 REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 SEED=42 BACKEND=vllm \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh

# agentic v4 stage2, same reward
DATASET=experiments/nestful_synthetic_curriculum_v3/data/curriculum_v4_nestful_like_agentic_openrouter/filtered/stage2_2call_agentic_openrouter.jsonl \
  REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 SEED=42 BACKEND=vllm \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh

# agentic v4 stage2 with reward v3.2 dense
# (same command, REWARD_POLICY=execution_aware_v3_2_dense)
```

Success target (RESEARCH_FIX_PLAN): dead_group_rate lower than v3.1, mean unique rewards/group higher than v3.1. If the probe is bad: do NOT train — revise the recipe and regenerate.
