# Agentic v5 dataset — stage probe report

STATUS: **not yet run** (requires a GPU pod; never run automatically by the builder).

Run on the pod, then paste/compare results here:

```bash
DATASET=experiments/nestful_synthetic_curriculum_v3/data/curriculum_v5_agentic_synthetic/filtered/stage2_2call_agentic_openrouter.jsonl \
  REWARD_POLICY=execution_aware_v3_2_dense NUM_TASKS=50 SEED=42 BACKEND=vllm \
  bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh
```
