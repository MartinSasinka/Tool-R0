# CLEANUP PLAN — proposal only, nothing deleted or moved yet

Date: 2026-07-09. Execute only after the team signs off; every "archive" step is a `git mv`,
never a delete.

## 1. Target structure

```
experiments/nestful_synthetic_curriculum_v3/
  configs/          # extracted YAML configs (currently implicit in env vars + partial/config.yaml)
  data/             # canonical datasets: curriculum_v3_1/filtered + manifest (moved from outputs/)
  scripts/          # launchers + generators that are still in use
  src/              # importable library code (reward_v3_1, tool_registry, traj_utils, …)
  outputs/          # run artifacts only (gitignored except small summaries)
    runs/
  audits/           # this audit (kept)
  reports/          # curated, human-written reports that are still true
  archive/          # everything legacy, moved as-is with a short README pointer
```

## 2. Classification

### Canonical (keep, move to `data/` or `src/`)

- `outputs/curriculum_v3_1/filtered/stage{1..4}_*.jsonl` + `curriculum_v3_1_manifest.json`
  → `data/curriculum_v3_1/` (these are inputs, not outputs; putting them under `outputs/`
  is why two "dataset locations" confusion arose)
- `lib/reward_v3_1.py` and whatever `lib/` modules resolve at train time → `src/`
- `scripts/run_curriculum_v3.sh`, `scripts/pilot/*.sh` (active launchers)
- `scripts/build_curriculum_v3_1_pipeline.py`, `tool_registry_v3_1.py`,
  `question_templates_v3_1.py`, `traj_utils_v3_1.py`, `uniqueness_utils_v3_1.py`,
  `motif_lib.py`, validators (`validate_curriculum_integrity_v3_1.py`,
  `validate_question_trace_alignment_v3_1.py`, `run_preflight_gates.py`,
  `final_dataset_audit_v3_1.py`)
- `scripts/sft/*` (current SFT experiment)
- NESTFUL reference data stays where it is (`nestful_mtgrpo_minimal/data/`)

### Legacy (move to `archive/`)

- **Dataset B**: `nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic/` — superseded corpus;
  before archiving, change `config.yaml` defaults (see §4)
- `outputs/curriculum_v3/` (pre-v3.1 corpus used by July-2 runs)
- v3-only scripts: `build_curriculum_v3.py`, `generate_motif_synthetic_tasks.py`,
  `synthetic_tool_registry.py`, `replay_synthetic_gold_traces.py` (non-v3_1 version),
  `run_tool_family_realism.py` (non-v3_1), `validate_synthetic_tasks.py`
- Runs `20260702_112042` (data_base only, no training), `20260702_112150`,
  `0260703_145219_v3_1` (binary-reward era; note the id typo — if kept, rename to
  `20260703_145219_v3_1_binaryreward` inside archive)

### Generated outputs (keep under `outputs/`, add `.gitignore`)

- `outputs/runs/**` heavy artifacts: `adapter_model.safetensors`, `tokenizer.json`,
  `vocab.json`, `merges.txt`, `*_trajectories.jsonl`, `*predictions*.jsonl`,
  `validation_subset.jsonl`, `data_base/*.jsonl` — none of these belong in git
  (the current git status shows a full checkpoint about to be committed)
- Keep in git: `metrics*.json`, `*_summary.json(l)`, `stage_gate_report.json`,
  `config_used.*`, `CHECKPOINT_REEVAL_REPORT.md` (small, reproducibility-critical)

### Stale reports (move to `archive/reports_pre_audit/`)

`outputs/` currently holds ~60 top-level reports from successive analysis waves. Stale or
superseded by this audit: `PILOT_*` (12 files, describe the July-2 pilot on the old corpus),
`GENERATOR_GAP_DIAGNOSIS.md`, `NEXT_ACTION_DECISION.md`, `NEXT_DATASET_IMPROVEMENT_PLAN.md`,
`NEXT_OFFLINE_ANALYSIS_SUMMARY.md`, `STAGE1_2_PILOT_PLAN.md`, `POD_DRY_RUN_INSTRUCTIONS.md`,
`POST_PILOT_EVAL_PLAN.md`, `TRAINING_WIRING_CHECK.md`, duplicated
`PREFLIGHT_GATES_REPORT.md` / `synthetic_gold_replay_*` (exist both in `outputs/` and
`outputs/curriculum_v3_1/`). Still-true design docs move to `reports/`:
`CURRICULUM_V3_1_DESIGN_DECISION.md`, `REWARD_V3_1_DESIGN.md` (currently in
`curriculum_v3_1/`), `NESTFUL_MOTIF_ANALYSIS.md`, `MOTIF_SCHEMA.md` (root).

### Duplicate / near-duplicate scripts

| keep | archive | why |
|---|---|---|
| `replay_synthetic_gold_traces_v3_1.py` | `replay_synthetic_gold_traces.py` | v3-only |
| `run_tool_family_realism_v3_1.py` | `run_tool_family_realism.py` | v3-only |
| `final_dataset_audit_v3_1.py` | — | keep |
| `run_eval_v3.sh` | review: likely superseded by final-eval flow in `run.py` | |

### Inconsistent naming to fix (rename in one commit, update references)

- Run dir `0260703_145219_v3_1` → missing leading `2`.
- Double-nested batch dirs `final_eval_*_temp0/final_eval_*_temp0/` → flatten.
- `epoch_N_Ncall.jsonl` naming reused by both corpora A-symlinks and B — after archiving B,
  rename the run-side links `stageN_*.jsonl` to match A, or keep `epoch_*` but document.
- `curriculum_toolr0_all.jsonl` is only epoch-6 — archive with a README note.
- `outputs/final_eval_*` vs `outputs/runs/final_eval_*`: the report generator expects the
  former, artifacts live in the latter — pick `outputs/evals/<batch_id>/` going forward.

## 3. Files that should never be committed (add to `.gitignore` before next commit)

```
experiments/nestful_synthetic_curriculum_v3/outputs/runs/**/checkpoints/**
experiments/nestful_synthetic_curriculum_v3/outputs/runs/**/data_base/
experiments/nestful_synthetic_curriculum_v3/outputs/runs/**/*trajectories*.jsonl
experiments/nestful_synthetic_curriculum_v3/outputs/runs/**/*predictions*.jsonl
experiments/nestful_synthetic_curriculum_v3/outputs/runs/**/validation_subset*.jsonl
experiments/nestful_synthetic_curriculum_v3/outputs/sft/**/run_*/
```

The pending git status includes `adapter_model.safetensors` and full tokenizer copies from
run `20260708_212347_v3_1` — recommend NOT committing those.

## 4. Ordered execution plan (for later)

1. Add `.gitignore` rules (no history rewrite; just stop the bleeding).
2. Change `nestful_mtgrpo_{minimal,partial}/config.yaml` `paths.*` defaults away from
   dataset B to explicit "must be overridden" placeholders (or A files) — removes the silent
   wrong-dataset footgun.
3. `git mv` dataset A to `data/curriculum_v3_1/`; update `run_curriculum_v3.sh` paths.
4. `git mv` legacy corpora + old runs + stale reports into `archive/` with a one-page
   `archive/README.md` mapping old→new locations.
5. Rename the typo run dir and flatten double-nested eval batches.
6. Only after 1–5 are merged: consider pruning heavy artifacts from git history if repo size
   matters (separate decision, destructive).
