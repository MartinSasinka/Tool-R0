# Offline audit runbook (Windows PowerShell)

From repository root `Tool-R0`:

```powershell
cd C:\Users\Šunka\Documents\GitHub\Tool-R0

python -m pytest `
  experiments/nestful_synthetic_curriculum_v3/tests/test_reward_ablation_offline_audit.py `
  -q

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_local_offline_audit.py all `
  --runs-root experiments/nestful_synthetic_curriculum_v3/outputs/runs/_local_round1_analysis `
  --reports-dir experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/offline_audit `
  --canonical-arm A0_R0_CURRENT `
  --seed 20260724 `
  --strict
```

Outputs live under `experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/offline_audit/`.

This phase does **not** start GRPO training, model rollouts, or held-out inference unless you pass `--allow-model-inference` to `run_local_heldout_eval.py`.
