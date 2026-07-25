# Dispatch canary — RunPod runbook

**Goal:** prove the reward-dispatch fix on the real GPU stack.  
**Not a goal:** NESTFUL Win Rate, full epoch, 5-arm ablation.

Round 1 is frozen as `reward_ablation_round1_INVALID_DISPATCH`
(see `ROUND1_INVALID_DISPATCH.md`). Do not interpret Round 1 as a reward ablation.

---

## Config

| Item | Value |
|---|---|
| Arms | `A1_OUTCOME_ONLY`, `A4_GATED_VERIFIABLE` |
| Tasks | 24 Stage-3 (shared `canary_subset_24.jsonl`) |
| Rollouts | 8 / task |
| Optimizer | ~10 steps / arm (dead groups skipped) |
| Seed | `20260724` (same for both) |
| NESTFUL eval | **no** |
| Est. GPU time | ~2 GPU-h total |

---

## Krok 0 — lokálně před push/sync

```powershell
cd C:\Users\Šunka\Documents\GitHub\Tool-R0

# freeze label already in repo:
# experiments/nestful_synthetic_curriculum_v3/reports/reward_ablation/ROUND1_INVALID_DISPATCH.md

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_canary_subset_24.py

# dry-run (no GPU) — config + dispatch guard path
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py `
  --round 2 --reward-arm A1_OUTCOME_ONLY --seed 20260724 --canary --dry-run

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py `
  --round 2 --reward-arm A4_GATED_VERIFIABLE --seed 20260724 --canary --dry-run
```

Commit/push (nebo `rsync`/`git pull` na RunPod) včetně:
- `run.py` dispatch fix
- `run_reward_ablation.py` (`--canary`, guard)
- `grpo_train.py` / `vllm_dp_pool.py` traj logging
- canary skripty výše

---

## Krok 1 — RunPod setup

```bash
# SSH na pod, pak:
cd /workspace/Tool-R0   # uprav cestu podle svého checkoutu
git pull                # nebo sync větve s dispatch fixem

export WANDB_API_KEY=...          # volitelné, ale doporučené
export HF_TOKEN=...               # pokud model není cached
export USE_VLLM=1
export ROLLOUT_DP_GPUS=1,2,3      # GPU0=learner, 1-3=rollout (4×GPU pod)
export CANARY_TRAJ_LOG=1

# sanity
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.is_available())"
nvidia-smi
```

---

## Krok 2 — spustit canary (oba army sekvenčně)

**Jedním příkazem (doporučeno):**

```bash
cd /workspace/Tool-R0
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_dispatch_canary.sh
```

**Nebo arm po armu:**

```bash
cd /workspace/Tool-R0
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_canary_subset_24.py

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py \
  --round 2 \
  --reward-arm A1_OUTCOME_ONLY \
  --seed 20260724 \
  --canary \
  --run-id dispatch_canary_A1_OUTCOME_ONLY_seed20260724 \
  --wandb-project nestful-reward-ablation \
  --wandb-group dispatch_canary_$(date -u +%Y%m%d)

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py \
  --round 2 \
  --reward-arm A4_GATED_VERIFIABLE \
  --seed 20260724 \
  --canary \
  --run-id dispatch_canary_A4_GATED_VERIFIABLE_seed20260724 \
  --wandb-project nestful-reward-ablation \
  --wandb-group dispatch_canary_$(date -u +%Y%m%d)
```

Výstupy:

```
outputs/runs/dispatch_canary_A1_OUTCOME_ONLY_seed20260724/
  train/train_log.jsonl
  train/canary_rollouts.jsonl    # povinné traj dump
  SUCCESS

outputs/runs/dispatch_canary_A4_GATED_VERIFIABLE_seed20260724/
  ... totéž
```

Resume po pádu:

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_dispatch_canary.sh --resume
# nebo smazat neúspěšný běh a začít znovu:
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_dispatch_canary.sh --force-fresh
```

---

## Krok 3 — gate (musí PASS)

```bash
cd /workspace/Tool-R0
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/validate_dispatch_canary.py
# nebo:
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_dispatch_canary.sh --validate-only
```

Report: `reports/reward_ablation/dispatch_canary/CANARY_GATE.json`

### Must pass

| Check | Expected |
|---|---|
| A1 `reward_policy_resolved` | `reward_ablation_A1_OUTCOME_ONLY` (100 % train rows) |
| A4 `reward_policy_resolved` | `reward_ablation_A4_GATED_VERIFIABLE` (100 % train rows) |
| Runtime dispatch guard | nepadne |
| NaN/Inf | žádné |
| Terminal ordering | mean(success) > mean(execwrong) pokud obě třídy existují |
| Hash-matched completions | `max_abs_diff > 0` a `reward_pearson < 1` |

**Pokud oba army znovu dostanou stejné rewardy → STOP, zpět k dispatchi. Žádný další trénink.**

Ruční spot-check:

```bash
# A1 — všechny resolved politiky
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p=Path("experiments/nestful_synthetic_curriculum_v3/outputs/runs/dispatch_canary_A1_OUTCOME_ONLY_seed20260724/train/train_log.jsonl")
c=Counter()
for line in p.open():
    r=json.loads(line)
    if "reward_policy_resolved" in r and "task_id" in r:
        c[r["reward_policy_resolved"]]+=1
print(c)
PY
```

---

## Krok 4 — po PASS

Použij **`A4_GATED_VERIFIABLE`** jako pracovní reward pro další *datový* experiment.

- Ne proto, že Round 1 „dokázal“, že je nejlepší — nedokázal.
- Protože má rozumné vlastnosti (outcome-first, gated process, méně gold-trace imitation).
- `A1_OUTCOME_ONLY` nech jen jako vědeckou kontrolu.
- **Nedělej teď další 5-arm reward ablaci.**

V reportech piš jen:

> Round 1 nelze interpretovat jako reward ablaci, protože všechny arms trénovaly s `execution_aware_v3_2_dense`.

Nepoužívej: „A3 je horší“, „A4 je nejlepší“, „A1 snížil dead groups díky outcome-only“.

---

## Co canary NEřeší

- Transfer gap Stage-3 → NESTFUL
- Plný credit matched-prefix audit (to až z `canary_rollouts.jsonl` offline)
- Zvýšení Win Rate
