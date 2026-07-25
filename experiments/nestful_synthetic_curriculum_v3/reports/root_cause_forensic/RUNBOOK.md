# RUNBOOK — reprodukce forenzního auditu

Vše běží lokálně (CPU), bez GPU a bez placených služeb.
Pracovní adresář: `experiments/nestful_synthetic_curriculum_v3/`.

## 1. Forenzní analýzy a01–a08

```powershell
cd experiments\nestful_synthetic_curriculum_v3
.venv\Scripts\python.exe -X utf8 scripts\audit\root_cause_forensic\run_all.py
```

Výstupy: `reports/root_cause_forensic/analysis/a01..a08*.json` + `_run_status.json`.

Jednotlivé moduly lze pouštět samostatně, např.:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\audit\root_cause_forensic\a02_reward_dispatch.py
.venv\Scripts\python.exe -X utf8 scripts\audit\root_cause_forensic\a03_adapter_audit.py
```

Pozn.: a03 potřebuje torch (CPU) a ~2 GB RAM (trace identita, ΔW se nematerializuje).

## 2. Opravený offline audit Round-1 (regeneruje EXECUTIVE_SUMMARY s verdiktem REWARD_DISPATCH_BUG)

```powershell
.venv\Scripts\python.exe -X utf8 scripts\ablation\run_local_offline_audit.py all
```

Výstupy: `reports/reward_ablation/offline_audit/*.md|*.csv|*.json`.

## 3. Regresní testy

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest tests\test_root_cause_forensic_fixes.py -v
.venv\Scripts\python.exe -X utf8 -m pytest tests\test_reward_ablation.py tests\test_reward_ablation_offline_audit.py tests\test_reward_ablation_pipeline.py tests\test_reward_v3_2_dense.py tests\test_pure_stage3_pipeline.py -q
```

Očekávaný stav (2026-07-25): 17 passed; 74 passed + 1 skipped.

## 4. Klíčové raw artefakty pro nezávislou verifikaci

- Dispatch bug: `outputs/runs/_local_round1_analysis/reward_ablation_r1_A1_OUTCOME_ONLY_seed20260724/reward_ablation_r1_A1_OUTCOME_ONLY_seed20260724/logs/console.log`
  (hledat `training reward =`) a `…/train/train_log.jsonl` (pole `reward_policy_resolved`).
- Checkpointy: `…/train/checkpoints/adapter_epoch_1/adapter_model.safetensors` (SHA256 viz `analysis/a03_adapter_audit.json`).
- Eval: `…/eval/*/task_results.jsonl` (pole `official_win`, `_traj`).
- Referenční run: `outputs/runs/pure_stage3_2ep_20260719_221918/`.

## 5. Guard proti regresi dispatch bugu (aktivní při každém budoucím běhu)

`scripts/ablation/run_reward_ablation.py` nyní (a) nastavuje `REWARD_POLICY`
env před importem session, (b) po vytvoření session volá
`assert_dispatched_policy` — mismatch okamžitě abortuje běh.
