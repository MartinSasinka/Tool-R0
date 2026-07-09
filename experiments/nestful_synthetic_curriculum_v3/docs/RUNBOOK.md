# RUNBOOK

Operational procedures. All commands run from the repo root.

## 1. Environment sanity check (always first on a new pod)

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh
```

Verifies python deps (torch/peft/bitsandbytes/vllm), CUDA, canonical dataset presence, the
IBM `executable_functions` dir (without it the official win rate cannot be computed), and
flags configs still defaulting to legacy dataset B. Informational — never modifies anything.

## 2. Evaluation batch (the only sanctioned way to compare checkpoints)

```bash
# dry-run first: prints the exact per-cell commands, runs nothing
DRY_RUN=1 \
CELLS="baseline,my_ckpt=<adapter_dir>" DATASET=nestful_test \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# tiny smoke batch (minutes; output stamped smoke, not reportable)
MAX_TASKS=5 CELLS="baseline,my_ckpt=<adapter_dir>" \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# real batch
CELLS="baseline,my_ckpt=<adapter_dir>" DATASET=nestful_test BATCH_NAME=my_eval \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh
```

Outputs land in `outputs/evals/<batch>_<UTC>_temp0p0/`: per-cell `metrics.json`,
`metrics_official.json`, `metrics_unified.json`, plus batch-level `manifest.json` and
`BATCH_REPORT.md`. Exit codes: 2 = no baseline cell, 3 = legacy dataset B, 4 = official
scorer output missing, 5 = a cell crashed.

## 3. Re-running audits

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/audit/run_all_audits.sh
```

Regenerates the machine-generated audit JSON/CSV. The hand-written `.md` analyses in
`audits/` are a frozen historical record — new analysis goes in new files.

## 4. Training / SFT

See `docs/TRAINING.md`. Reminders: probe a stage before training it (once the P1 probe
lands); Stage 1 is saturated — skip it; always set `CURRICULUM_VERSION=v3_1`; never rely on
`config.yaml` dataset defaults (legacy B).

## 5. What must never be committed

Enforced by `.gitignore` (this folder + repo root); the policy:

- model weights/adapters (`*.safetensors`, `*.bin`, `checkpoints/`), tokenizer dumps,
- per-sample dumps (`*trajectories*.jsonl`, `*predictions*.jsonl`, `validation_subset*`),
- raw logs (`*.log`, `train_log.jsonl`) and `data_base/` copies,
- SFT `run_*/` output dirs, anything > ~5 MB.

Committable: `metrics*.json`, `metrics_unified.json`, `manifest.json`, `BATCH_REPORT.md`,
`config_used.yaml`, small summary JSON/MD, gate reports. If `git status` shows a safetensors
or trajectories file, stop and fix the ignore rules instead of committing.

## 6. Archiving old artifacts (P3 — not yet executed)

Rules: **never delete** — `git mv` into `archive/` with an entry in `archive/README.md`
(old path → new path → reason). Batch the moves in one reviewed commit. Before moving
anything, repo-wide grep for the path (pod launchers and reports may reference it) and
re-dry-run every launcher afterwards. Planned moves are listed in `audits/CLEANUP_PLAN.md`
§2 (dataset B, `curriculum_v3/`, July-2/3 runs, stale pilot reports, run-id typo rename).

## 7. Claiming an improvement (checklist)

1. Same-batch baseline cell — in the SAME `BATCH_REPORT.md` table.
2. `official_nestful_win_rate`, temp0, full dataset (no `--max-tasks`).
3. Delta larger than the 95 % CI overlap suggests noise (~±2.4 pp at n=1661).
4. Paired counts: positive net with meaningful gained/regressed volume.
5. Manifest present (git commit, dataset SHA, seed, decoding).

If any item fails, the result is a diagnostic, not a claim.
