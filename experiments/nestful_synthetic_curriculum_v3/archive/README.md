# archive/ — legacy artifacts (cleanup Phase K, 2026-07-09)

Everything here was moved (never deleted) per `audits/CLEANUP_PLAN.md`. Old
results stay readable at the new locations below. Nothing in this folder is
used by active launchers; guardrails refuse these datasets unless explicitly
overridden.

## Mapping: old path → new path → reason

| Old path | New path | Reason |
|---|---|---|
| `experiments/nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic/` | `archive/legacy_dataset_B_filtered_toolr0_synthetic/` | Legacy dataset B: superseded corpus (~4% unresolved `$var$` gold answers, misleading `epoch_N_Ncall` names — see `audits/DATASET_AUDIT.md`). Guarded by `ALLOW_LEGACY_DATASET_B=1` in `run.py` / `run_curriculum.sh`. |
| `outputs/curriculum_v3/` | `archive/curriculum_v3/` | Pre-v3.1 corpus used only by the July-2 pilot runs. `run_curriculum_v3.sh` now defaults to `CURRICULUM_VERSION=v3_1`; the legacy branch requires `ALLOW_LEGACY_CURRICULUM_V3=1` and points here. |
| `scripts/build_curriculum_v3.py` | `archive/scripts_v3_legacy/build_curriculum_v3.py` | v3-only generator, superseded by `build_curriculum_v3_1_pipeline.py`. |
| `scripts/generate_motif_synthetic_tasks.py` | `archive/scripts_v3_legacy/generate_motif_synthetic_tasks.py` | v3-only generator (motif prototypes), superseded by the v3.1 pipeline. |
| `outputs/PILOT_*.{md,csv}` (14 files) | `archive/reports_pre_audit/` | July-2 pilot analyses on the old corpus; superseded by `audits/`. |
| `outputs/GENERATOR_GAP_DIAGNOSIS.md` | `archive/reports_pre_audit/` | Superseded by the audit + REMEDIATION_PLAN. |
| `outputs/NEXT_ACTION_DECISION.md`, `outputs/NEXT_DATASET_IMPROVEMENT_PLAN.md`, `outputs/NEXT_OFFLINE_ANALYSIS_SUMMARY.md` | `archive/reports_pre_audit/` | Stale planning docs, superseded by REMEDIATION_PLAN / RESEARCH_FIX_PLAN. |
| `outputs/STAGE1_2_PILOT_PLAN.md`, `outputs/POST_PILOT_EVAL_PLAN.md`, `outputs/POD_DRY_RUN_INSTRUCTIONS.md`, `outputs/TRAINING_WIRING_CHECK.md` | `archive/reports_pre_audit/` | Stale operational docs from the July-2 pilot era; current procedures live in `docs/RUNBOOK.md`. |
| `outputs/runs/20260702_112042/` | `archive/runs_pre_v3_1/20260702_112042/` | Pre-v3.1 run (data_base only, no training artifacts). Untracked by git; moved on disk. |
| `outputs/runs/20260702_112150/` | `archive/runs_pre_v3_1/20260702_112150/` | Pre-v3.1 pilot run on the old corpus. Untracked by git; moved on disk. |
| `outputs/runs/0260703_145219_v3_1/` | `archive/runs_pre_v3_1/20260703_145219_v3_1_binaryreward/` | Binary-reward-era run; folder name had a typo (missing leading `2`), fixed in the archive name. `audits/RUN_AUDIT.*` still cite the ORIGINAL id `0260703_145219_v3_1` — those are point-in-time audit records and were intentionally not rewritten. |

## Intentionally NOT moved

- `scripts/synthetic_tool_registry.py` — imported by the kept `scripts/motif_lib.py`.
- `scripts/validate_synthetic_tasks.py`, `scripts/replay_synthetic_gold_traces.py`,
  `scripts/run_tool_family_realism.py` — still invoked by the guarded legacy
  branch of `run_curriculum_v3.sh` (and an active test imports
  `validate_synthetic_tasks`).
- `outputs/curriculum_v3_1/` (canonical dataset A) — moving it to `data/` is a
  larger repoint (paths.py, configs, docs) deferred until after the next
  training round.
- Double-nested eval batch dirs under `outputs/runs/final_eval_*` — audit
  tooling references the current layout; new evals go to `outputs/evals/`
  via the batch runner, so flattening has no benefit worth the churn.
- `audits/*` — historical audit record, frozen.

## Rules

- Do not point new training/eval runs at anything in `archive/`.
- Legacy dataset B additionally requires `ALLOW_LEGACY_DATASET_B=1`;
  the legacy v3 corpus requires `ALLOW_LEGACY_CURRICULUM_V3=1`.
