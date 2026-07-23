# Real weak-model audit run — status

**Generated:** 2026-07-23 (OpenRouter, Windows)

## Security note

The OpenRouter API key was pasted in chat. **Rotate/revoke it** in OpenRouter settings and use a fresh key via `$env:OPENROUTER_API_KEY` only (never commit).

## What was done

1. **R0 parity** (`verify-r0`) — see `R0_PARITY.md`
2. **Mock backup** — `reports/pure_stage3_weak_audit_mock_backup/`
3. **Real canary** (10 tasks, both passes) — **PASS** (`CANARY_REPORT.md`)
4. **Pilot** (20 tasks A/B) — 20/20 valid each
5. **Full run** (248 tasks, resume) — completed
6. **Validate + summarize** — completed

## Model & API

- **Model:** `deepseek/deepseek-v3.2` (OpenRouter)
- **Reasoning:** `effort=none`
- **Structured output:** JSON Schema (`weak_audit/schema.py`)
- **Providers observed:** AtlasCloud, StreamLake (logged per row in raw JSONL)

## R0 / reward labeling

- Train log internal consistency: **100%** (652 groups, policy `execution_aware_v3_2_dense`)
- **Trajectory-level train parity:** not possible (rollouts not persisted)
- Packet reward label: **`execution_aware_v3_2_dense_recomputed_eval`** (recomputed on eval trajectories; not logged train scalars)

## Annotation validation (real model)

| Pass | Valid | Invalid (after repair) |
|------|------:|-------------------------:|
| A | 239 | 9 |
| B | 243 | 5 |

Invalid rows: `invalid_annotations.jsonl`

Tasks with both valid passes (agreement computed): **235 / 248**

## PASS A/B agreement (annotator stability, not ground truth)

- Exact agreement: **15.7%**
- Root cause Cohen κ: **0.41**
- Root cause changed: **49.4%**
- Reward ordering changed: **30.6%**
- first_divergence_turn agreement: **90.2%**

See `ANNOTATION_AGREEMENT.md`, `annotation_agreement.csv`

## High-priority handoff

See `HIGH_PRIORITY_CASES.jsonl` / `.md` — **80 cases**

**Total OpenRouter cost (raw logs):** ~$0.25 USD (496 requests)

## Resume command (if re-validating or topping up invalid)

```powershell
$env:OPENROUTER_API_KEY = "..."   # new rotated key
$env:OPENROUTER_WEAK_MODEL = "deepseek/deepseek-v3.2"

python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py validate
python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/prepare_weak_model_audit.py summarize
```

## All output directory

`experiments/nestful_synthetic_curriculum_v3/reports/pure_stage3_weak_audit/`
