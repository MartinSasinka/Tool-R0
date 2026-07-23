# Weak-model audit — RUNBOOK (Windows PowerShell)

Run from **repository root** (`Tool-R0/`).

Default outputs: `experiments/nestful_synthetic_curriculum_v3/reports/pure_stage3_weak_audit/`

## Prerequisites

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."   # never commit
$env:OPENROUTER_WEAK_MODEL = "deepseek/deepseek-v3.2"
Write-Host $env:OPENROUTER_WEAK_MODEL
Write-Host ($env:OPENROUTER_API_KEY.Substring(0, 10) + "...")
```

No RunPod / local GPU required — OpenRouter HTTP only.

## 1. R0 parity gate

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py verify-r0
```

Outputs: `R0_PARITY.md`, `r0_parity_report.json`

## 2. Discovery & prepare (deterministic)

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py discover
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py prepare
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate-packets
```

## 3. Backup mock results (optional)

```powershell
$report = "experiments\nestful_synthetic_curriculum_v3\reports\pure_stage3_weak_audit"
$backup = "experiments\nestful_synthetic_curriculum_v3\reports\pure_stage3_weak_audit_mock_backup"
Copy-Item $report $backup -Recurse -Force
```

## 4. Real canary (no `--mock`)

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py canary `
  --model $env:OPENROUTER_WEAK_MODEL `
  --reasoning-effort none `
  --no-resume
```

Check `CANARY_REPORT.md` gate == PASS.

## 5. Pilot (20 tasks)

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run `
  --pass-label A --model $env:OPENROUTER_WEAK_MODEL --limit 20 --concurrency 4 `
  --temperature 0 --reasoning-effort none --no-resume

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run `
  --pass-label B --model $env:OPENROUTER_WEAK_MODEL --limit 20 --concurrency 4 `
  --temperature 0 --reasoning-effort none --no-resume

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py summarize
```

## 6. Full run (resume)

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run `
  --pass-label A --model $env:OPENROUTER_WEAK_MODEL --concurrency 4 `
  --temperature 0 --reasoning-effort none --resume

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py run `
  --pass-label B --model $env:OPENROUTER_WEAK_MODEL --concurrency 4 `
  --temperature 0 --reasoning-effort none --resume

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py summarize
```

## CLI flags (run / canary)

| Flag | Default |
|------|---------|
| `--model` | `OPENROUTER_WEAK_MODEL` |
| `--reasoning-effort` | `none` |
| `--max-output-tokens` | `350` |
| `--no-json-schema` | off (structured JSON Schema on) |
| `--concurrency` | 4 |
| `--temperature` | 0 |

Raw logs include: `requested_model`, `response_model`, `provider`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `reported_cost`.

## Reward note

Packet `reward_total` is **recomputed** on eval trajectories (`execution_aware_v3_2_dense_recomputed_eval`). See `R0_PARITY.md` — not the same as `reward_train_strict`.

## Unit tests

```powershell
python -m pytest experiments/nestful_synthetic_curriculum_v3/scripts/analysis/tests/test_weak_audit.py -q
```

## Latest real run summary

See `REAL_RUN_STATUS.md` in this directory.

## 7. Invalid retry finalization (Windows PowerShell)

After the real Pass A/B run, repair only schema-invalid rows (no full re-run).

```powershell
$env:OPENROUTER_WEAK_MODEL = "deepseek/deepseek-v3.2"
# $env:OPENROUTER_API_KEY must already be set — never commit the key

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py discover-invalid

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py retry-invalid `
  --dry-run

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py retry-invalid `
  --model $env:OPENROUTER_WEAK_MODEL `
  --concurrency 1 `
  --temperature 0 `
  --reasoning-effort none `
  --max-output-tokens 500 `
  --resume

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate-retry

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py finalize

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py summarize `
  --annotations-suffix final
```

Optional provider pinning (see `PROVIDER_AUDIT.md` for recommended ID):

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py retry-invalid `
  --provider SiliconFlow `
  --model $env:OPENROUTER_WEAK_MODEL `
  --concurrency 1 `
  --max-output-tokens 500 `
  --resume
```

Outputs:

| Step | Artifacts |
|------|-----------|
| discover-invalid | `INVALID_RETRY_DISCOVERY.md`, `invalid_retry_manifest.json`, `PROVIDER_AUDIT.md`, backup under `../pure_stage3_weak_audit_real_before_retry/` |
| retry-invalid | `retry_invalid_raw.jsonl`, `retry_invalid_validated.jsonl`, `retry_invalid_failed.jsonl` |
| finalize | `pass_*_annotations_final.jsonl`, `WEAK_AUDIT_FINAL_MANIFEST.json`, `RETRY_FINALIZATION_REPORT.md` |
| summarize --annotations-suffix final | `*_FINAL.*` agreement / clusters / high-priority |
