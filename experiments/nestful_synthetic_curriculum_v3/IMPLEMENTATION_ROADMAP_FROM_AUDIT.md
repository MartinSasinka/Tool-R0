# IMPLEMENTATION ROADMAP (from audit)

Date: 2026-07-09. Concrete phases; each with files, risks, tests, dry-run command, acceptance
criteria, rollback. Priorities per `REMEDIATION_PLAN.md` §5. Phase 0 is implemented in this
change-set; later phases are planned only.

Path conventions: all commands run from repo root; `V3=experiments/nestful_synthetic_curriculum_v3`,
`MIN=experiments/nestful_mtgrpo_minimal`.

---

## Phase 0 — P0 infrastructure (IMPLEMENTED NOW)

### 0.1 `.gitignore` for generated artifacts
- **Add:** `$V3/.gitignore`
- **Risks:** none (ignore rules only affect untracked files; nothing already tracked is
  hidden).
- **Tests:** `git status --short` before/after — heavy untracked artifacts disappear, code
  changes remain visible; `git check-ignore -v` spot checks.
- **Dry-run:** `git check-ignore -v $V3/outputs/runs/20260708_212347_v3_1/stage_3/checkpoints/adapter_epoch_1/adapter_model.safetensors`
- **Acceptance:** safetensors/tokenizer/trajectory/predictions files no longer listed as `??`;
  `metrics*.json`, `*summary*`, `config_used.*`, gate reports still tracked.
- **Rollback:** delete the `.gitignore` (one file).

### 0.2 Shared libs: metric schema + run manifest
- **Add:** `$V3/scripts/lib/metrics_schema.py`, `$V3/scripts/lib/run_manifest.py`,
  `$V3/scripts/lib/__init__.py`, `$V3/scripts/lib/paths.py` (single source for canonical
  dataset locations + legacy-B detection).
- **Risks:** none — pure-stdlib, additive, imported only by new scripts.
- **Tests:** `python -m compileall`; build a `metrics_unified.json` from an existing scored
  cell (Stage-3 temp0 batch) and eyeball it; unit-style `--self-test` entry points.
- **Dry-run:** `python $V3/scripts/lib/metrics_schema.py --self-test`
- **Acceptance:** unified JSON produced for an existing cell with correct primary/diagnostic
  split; manifest JSON contains git commit, dataset SHA, seed, decoding.
- **Rollback:** remove `scripts/lib/` (nothing else imports it outside new scripts).

### 0.3 Deterministic eval batch runner
- **Add:** `$V3/scripts/eval/run_eval_batch.py`, `$V3/scripts/eval/eval_batch_temp0.sh`
- **Changes:** none to `$MIN/run.py` (wrapped via subprocess).
- **Risks:** low — command construction must mirror the known-good manual invocation used for
  the 2026-07-09 Stage-3 batch (documented in `audits/STAGE3_AUDIT.md`); mitigated by
  `--dry-run` printing the exact commands for review before any GPU use.
- **Tests:** `--dry-run` on Windows (no GPU) for a baseline+2-checkpoint batch; verify
  refusal paths (no baseline → exit 2; legacy-B dataset → exit 3); report generation from
  the existing on-disk Stage-3 cells (offline mode `--report-only`).
- **Dry-run:**
  `python $V3/scripts/eval/run_eval_batch.py --cells baseline,ckpt=...adapter_epoch_1,ckpt=...adapter_epoch_2 --dataset nestful_test --dry-run`
- **Acceptance:** dry-run prints one resolved command per cell (temp0, official scorer on,
  flat batch dir); refusal paths verified; `--report-only` over existing metrics produces a
  BATCH_REPORT.md with paired counts.
- **Rollback:** remove the two files.

### 0.4 Setup + audit scripts
- **Add:** `$V3/scripts/setup/check_env.sh`, `$V3/scripts/audit/run_all_audits.sh`
- **Risks:** none — read-only checks / re-running existing audit extractors.
- **Tests:** `bash -n`; execute `check_env.sh` locally (expected to report missing GPU on
  the Windows workstation without failing the informational sections).
- **Dry-run:** `bash $V3/scripts/setup/check_env.sh`
- **Acceptance:** prints resolved dataset paths + SHAs, flags legacy-B configs, GPU/deps
  status; audit script re-produces `audits/` JSON outputs unchanged.
- **Rollback:** remove the two files.

### 0.5 Docs
- **Add:** `$V3/README.md`, `$V3/docs/{EVALUATION,DATASETS,TRAINING,REWARD,RUNBOOK}.md`
- **Risks:** none. **Acceptance:** answers all questions listed in the task spec (canonical
  vs legacy, internal-win caveat, primary metric, same-batch procedure, paired counts, probe/
  SFT/GRPO how-to, commit policy, archive procedure). **Rollback:** remove files.

---

## Phase 1a — Same-batch Stage-3 re-evaluation (P1, needs GPU pod, USER-LAUNCHED)

- **Files changed:** none (uses Phase-0 runner).
- **Risks:** pod cost (~3 cells × 1861 tasks); vLLM version drift vs July runs (manifest
  records versions, so drift is at least visible).
- **Tests before launch:** `--dry-run` on the pod; `--max-tasks 5` smoke batch first.
- **Command:** see `REMEDIATION_EXECUTION_REPORT.md` §"Next commands" (ready-made).
- **Acceptance:** one batch dir, 3 cells with official + unified metrics + manifest;
  BATCH_REPORT with CI and paired counts. This settles MASTER_AUDIT Q6/Q7 for Stage 3.
- **Rollback:** n/a (produces new outputs only).

## Phase 1b — Config default repointing (P1, NEEDS APPROVAL — shared files)

- **Change:** `experiments/nestful_mtgrpo_partial/config.yaml` train/eval paths → canonical A
  + `nestful_dev`; `$MIN/config.yaml` gets a loud warning comment (repoint only after
  confirming no other experiment depends on the B default).
- **Risks:** **medium** — these configs are shared by standalone minimal/partial runs.
- **Tests:** `run_curriculum_v3.sh` dry-run (which overrides paths anyway) + a standalone
  `$MIN/run.py --mode dataset_stats`-style smoke on the new defaults.
- **Acceptance:** no B path remains as a silent default in partial; v3 pipeline unaffected.
- **Rollback:** git revert of the one-file diff.

## Phase 1c — Stage probe (P1)

- **Add:** `$V3/scripts/probe/probe_stage.py`, `probe_stage.sh`.
- **Reuses:** `$MIN/rollout.py`, reward modules — same code paths as training, no optimizer.
- **Risks:** low; GPU needed but minutes not hours (N≈50–100 tasks × 8 gens).
- **Tests:** probe Stage 1 with the v3_1 reward — must reproduce the audited saturation
  (dead-group rate ≈ 1.0); probe Stage 2 — must land near the audited 0.65–0.88 band.
- **Dry-run:** `bash $V3/scripts/probe/probe_stage.sh --stage 2 --n 8 --dry-run`
- **Acceptance:** histogram + predicted dead-group rate JSON; matches audit within noise.
- **Rollback:** remove files.

## Phase 1d — Reward densification experiment (P1, research — see RESEARCH_FIX_PLAN E1)

- **Add:** `$V3/lib/reward_v3_2_dense.py` + registration in the reward dispatch hook +
  `configs/reward_v3_2.yaml`. `reward_v3_1.py` untouched.
- **Risks:** medium — changes training science; gated by probe before any GRPO time.
- **Tests:** unit tests on crafted trajectories (each band boundary); probe on Stage 2/3.
- **Acceptance (infra):** dispatch verification passes (`_verify_reward_dispatch`);
  **(science):** probe dead-group rate < 0.5 before a GRPO run is approved.
- **Rollback:** config switch back to `execution_aware_v3_1_stepwise`.

## Phase 1e — SFT warmup → GRPO chain (P1)

- **Add:** `$V3/scripts/training/run_sft_plus_grpo.sh` (chains existing
  `run_stage2_continuation_sft_warmup.sh` output into the GRPO launcher's checkpoint-resume
  path). No trainer changes.
- **Risks:** medium — adapter-resume path must be validated (`INIT_FROM`/`CHECKPOINT_IN`
  semantics) on a smoke run first.
- **Tests:** smoke: SFT dry-run → GRPO 1 stage × few steps × `--max-tasks` small.
- **Acceptance:** chain completes on smoke; eval only through the batch runner.
- **Rollback:** remove script.

## Phase 1f — Signal-positive filtering (P1)

- **Add:** `$V3/scripts/probe/filter_stage_by_probe.py` — writes a filtered copy of a stage
  file (tasks with probed within-group variance > 0), plus manifest of what was dropped.
- **Risks:** low — data-side only; trainer sees an ordinary JSONL.
- **Tests:** filter on probe output of Stage 2; row-count sanity; SHAs in manifest.
- **Acceptance:** filtered file + manifest; training uses it only via explicit path.
- **Rollback:** point training back at the unfiltered file.

## Phase 2 — W&B standard, training manifests, multi-GPU runner, NESTFUL-like data (P2)

- **W&B:** env-guarded block in launchers; project `nestful-curriculum-v3_1`; unified JSON as
  artifact. Rollback: unset `WANDB_API_KEY`.
- **Training manifests:** call `run_manifest.py` from `run_curriculum_v3.sh` prologue.
  Rollback: remove the call.
- **`scripts/training/run_grpo.sh`:** validates topology (`ROLLOUT_DP_GPUS`, CUDA device
  count) before launch; forwards unchanged env to the existing chain. Tests: dry-run prints
  topology; deliberately bad topology fails fast.
- **NESTFUL-like generator (research, RESEARCH_FIX_PLAN E5):** new generator + audit gate
  (0 overlap vs NESTFUL eval by question hash / trace hash / id). Rollback: don't train on it.

## Phase 3 — Archive and renames (P3, LAST)

- **Moves (git mv, per CLEANUP_PLAN §2):** dataset B → `archive/legacy_datasets/`;
  `outputs/curriculum_v3/` → archive; July-2/3 runs → `archive/runs_v3_era/`; stale pilot
  reports → `archive/reports/`; fix `0260703_145219_v3_1` run-id typo; flatten double-nested
  eval dirs. Each move logged in `archive/README.md`.
- **Risks:** medium — pod resume scripts and old report links reference old paths. Mitigation:
  grep for every path before moving; do moves in one reviewed commit; keep archive README map.
- **Tests:** all launchers `--dry-run`/`bash -n` green after moves; `run_all_audits.sh` still
  runs (audit tools take explicit paths).
- **Acceptance:** no orphan references (repo-wide grep for moved paths returns only archive
  README and historical audit docs).
- **Rollback:** `git revert` the move commit (pure renames revert cleanly).
