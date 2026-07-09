# Phase 1/2 Implementation Report — NESTFUL MT-GRPO remediation

Date: 2026-07-09. Scope: Phases A–K of the Phase-1/2 remediation task, implemented on
Windows (dev box) with dry-runs / CPU smoke tests only. **No training was launched, no
full NESTFUL eval was launched, nothing was deleted, the MT-GRPO trainer and
`lib/reward_v3_1.py` were not modified.**

---

## 1. Files changed

| File | Change |
|---|---|
| `experiments/nestful_mtgrpo_partial/config.yaml` | Repointed `train_jsonl`/`eval_jsonl` to canonical v3.1 + `nestful_dev`; `all_stages_jsonl: null`; legacy-B warning block (Phase B). |
| `experiments/nestful_mtgrpo_minimal/config.yaml` | Legacy-B warning block (Phase B); after Phase K archive, defaults repointed to canonical v3.1 files; `all_stages_jsonl: null` (unused by code). |
| `experiments/nestful_mtgrpo_minimal/run.py` | `mode_train` guard: refuses `filtered_toolr0_synthetic` unless `ALLOW_LEGACY_DATASET_B=1`; `_wandb_init` honors `WANDB_MODE=disabled` and `WANDB_TAGS` (Phases B, G). |
| `experiments/nestful_mtgrpo_minimal/run_curriculum.sh` | `DATA_BASE` legacy-B guard; auto-repair path updated to `archive/curriculum_v3` (Phases B, K). |
| `experiments/nestful_mtgrpo_minimal/vllm_dp_pool.py` | Registered `execution_aware_v3_2_dense` in reward dispatch (Phase D). |
| `experiments/nestful_synthetic_curriculum_v3/run.py` | v3.2 reward hook (`_patch_v3_2_reward`), dispatch precedence over v3.1 (Phase D). |
| `scripts/run_curriculum_v3.sh` | `STAGE<N>_FILE_OVERRIDE` support (hard-fails on legacy B); Windows-safe preflight read + `ln`→`cp` fallback; default `CURRICULUM_VERSION=v3_1`; legacy v3 branch needs `ALLOW_LEGACY_CURRICULUM_V3=1` and points to archive (Phases F–H, K). |
| `scripts/eval/run_eval_batch.py` | `preflight_official_scorer` (jsonlines/sklearn/IBM-functions before GPU jobs, exit 6); parallel cells `--parallel/--gpus/--max-parallel` (Phases A, I). |
| `scripts/eval/eval_batch_temp0.sh` | `PARALLEL` / `GPUS` / `MAX_PARALLEL` env plumbing (Phase I). |
| `scripts/setup/check_env.sh` | `jsonlines` check; legacy-B config check now ignores comment lines (Phases A, K). |
| `scripts/lib/run_manifest.py` | `--extra` JSON, GPU info (`nvidia-smi`), W&B run id in manifests (Phase G). |
| `scripts/lib/paths.py` | `LEGACY_DATASET_B_DIR` → archive location (Phase K). |
| `scripts/compare_synthetic_vs_nestful.py`, `audits/tools/dataset_audit.py`, `experiments/data/prepare_clean_training_set.py` | Legacy-B fallback paths → archive location (Phase K). |
| `README.md`, `docs/DATASETS.md`, `.gitignore` | Archive locations documented; `archive/runs_pre_v3_1/` ignored (Phase K). |

## 2. Files added

- `scripts/probe/probe_stage.py`, `scripts/probe/probe_stage.sh` — forward-only stage probe (Phase C).
- `scripts/probe/filter_stage_by_probe.py` — signal-positive filtering (Phase E).
- `lib/reward_v3_2_dense.py` + `tests/test_reward_v3_2_dense.py` — densified reward (Phase D).
- `scripts/training/run_grpo.sh` — multi-GPU GRPO wrapper w/ topology validation + manifest (Phase H).
- `scripts/training/run_sft_plus_grpo.sh` — SFT-warmup → GRPO chain (Phase F).
- `tests/test_parallel_eval_scheduler.py` — offline parallel-scheduler test (Phase I).
- `lib/nestful_like_generator.py`, `scripts/data/build_curriculum_v4_nestful_like.py`,
  `data/curriculum_v4_nestful_like/` (4×800 rows + `manifest.json`, `AUDIT_REPORT.md`,
  `DISTRIBUTION_REPORT.json`) — Phase J.
- `archive/README.md` — old→new mapping for everything moved in Phase K.

## 3. Files intentionally NOT touched

- `experiments/nestful_mtgrpo_minimal/grpo_train.py`, `rollout.py`, `executor.py` (trainer stack).
- `experiments/nestful_synthetic_curriculum_v3/lib/reward_v3_1.py` (frozen baseline).
- `audits/*` (historical record; still cite the original `0260703_145219_v3_1` run id).
- Canonical dataset A location (`outputs/curriculum_v3_1/filtered/`) — moving to `data/`
  is a larger repoint, deferred (documented in `archive/README.md`).
- NESTFUL data itself (never modified, never trained on).

## 4. Tests run (all pass)

- `python -m compileall` over `scripts/` and `lib/` (also re-run after Phase K moves).
- `tests/test_reward_v3_2_dense.py` — 9 scenarios: parse error, no tool call, wrong tool,
  one-correct-then-stop, correct-tool-wrong-args, full-executable-wrong-final, fully
  correct, invalid reference, too many calls; plus band monotonicity, within-band
  density, more distinct values than v3.1, and dispatch resolution.
- `tests/test_parallel_eval_scheduler.py` — GPU assignment, max-parallel, failure
  propagation with mock commands (no GPU).
- `scripts/lib/metrics_schema.py` self-tests.

## 5. Dry-runs executed

- Eval batch runner: refuses no-baseline batches (exit 2); refuses legacy dataset B;
  scorer preflight (jsonlines/sklearn/IBM `executable_functions`) before any GPU job
  (exit 6; on this Windows box it correctly warns official scoring is pod-only);
  sequential and `--parallel --gpus 0,1` dry-runs print per-cell commands with
  `CUDA_VISIBLE_DEVICES` assignments; outputs go to `outputs/evals/<batch_id>/<cell>/`.
- `probe_stage.py --dry-run` and CPU stub-backend runs (deterministic across repeats).
- `run_grpo.sh DRY_RUN=1` — resolved topology + exact command printed; overlapping
  learner/rollout GPU topology fails fast.
- `run_sft_plus_grpo.sh DRY_RUN=1` — SFT step + GRPO-from-adapter chaining printed.
- `run_curriculum_v3.sh DRY_RUN=1` (v3.1) — passes after the Phase K archive moves;
  `CURRICULUM_VERSION=v3` without `ALLOW_LEGACY_CURRICULUM_V3=1` aborts.
- `run_curriculum.sh` with empty `DATA_BASE` aborts on the legacy-B default.
- `check_env.sh` — reports both configs free of active legacy-B defaults.

## 6. Smoke commands (for the pod)

```bash
# environment
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh

# stage probe (v3.1 reward, Stage 2, 50 tasks) — calibration run
DATASET=stage2 REWARD_POLICY=execution_aware_v3_1_stepwise NUM_TASKS=50 \
NUM_GENERATIONS=4 TEMPERATURE=0.7 SEED=42 BACKEND=vllm \
bash experiments/nestful_synthetic_curriculum_v3/scripts/probe/probe_stage.sh

# same probe with v3.2 dense (signal comparison)
DATASET=stage2 REWARD_POLICY=execution_aware_v3_2_dense NUM_TASKS=50 ... (as above)

# 2-cell parallel smoke eval (baseline + checkpoint, 5 tasks, temp 0)
BASELINE=1 CHECKPOINTS="<adapter_dir>" SUBSET_SIZE=5 PARALLEL=1 GPUS="0 1" \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# GRPO smoke (tiny subset)
SMOKE=1 STAGES=2 GPUS="0 1 2 3" ROLLOUT_DP_GPUS="1 2 3" \
bash experiments/nestful_synthetic_curriculum_v3/scripts/training/run_grpo.sh
```

## 7. Real run commands

Documented in `docs/RUNBOOK.md` and `docs/TRAINING.md`; the SFT+GRPO chain is
`scripts/training/run_sft_plus_grpo.sh` (either `SFT_ADAPTER=<path>` to reuse an
existing adapter or `RUN_SFT=1` to warm up first). **None were run.**

## 8. Is the stage probe calibrated?

Partially. The probe reuses the training rollout (`rollout.run_episode`) and the training
reward dispatch (`vllm_dp_pool.resolve_reward_info`), so code-path parity holds by
construction, and stub-backend runs are deterministic with fixed seed. The calibration
requirement (Stage 1 v3.1 ≈ saturated dead rate; Stage 2 within the audited dead-group
range) needs a GPU model backend, which this box does not have — the exact calibration
commands are in §6 and must be run on the pod before trusting probe verdicts.

## 9. Does v3.2 reward improve probe signal?

On the CPU stub backend (same synthetic completion mix, seed 42, Stage 2 tasks):
dead_group_rate 0.333 → 0.000, mean unique rewards/group 2.17 → 3.13, reward entropy
1.55 → 2.46 bits (v3.1 → v3.2). Unit tests confirm strictly more distinct reward values
on identical trajectory sets while preserving band monotonicity. This is *mechanism*
evidence only — the stub is not a policy model; confirm with the GPU probe (§8) before
approving any GRPO run on v3.2.

## 10. Does the SFT+GRPO chain smoke test work?

Dry-run works end-to-end (adapter path plumbed into GRPO init, recorded in the
manifest as `init_adapter`). The tiny GPU smoke run was not executed here (no GPU);
`SMOKE=1` mode is implemented in both wrapper scripts.

## 11. Does W&B disabled/offline work?

`WANDB_MODE=disabled` short-circuits `_wandb_init` (no import, no network);
`offline`/`online` pass through, with `WANDB_PROJECT`/`WANDB_ENTITY`/`WANDB_GROUP`/
`WANDB_TAGS` honored. Manifests are written unconditionally and include the W&B run id
only when a run exists. No code path requires the `wandb` package.

## 12. Does parallel eval work?

The scheduler is implemented and unit-tested offline (GPU assignment via
`CUDA_VISIBLE_DEVICES`, `--max-parallel` throttling, any-cell-failure ⇒ whole batch
invalid, unified report only after all cells finish). The 2-cell GPU smoke (§6) remains
to be run on the pod.

## 13. Does v4 pass the contamination audit?

Yes. The tool library and question templates are written from scratch; only aggregate
NESTFUL statistics (call-count range, offered-tools range, naming style, `$varN.key$`
convention) informed the design — recorded in `manifest.json:extra.provenance`. Gates
enforced at build time over all 3 200 rows: overlap with NESTFUL full corpus by question
hash / trace hash / sample_id = **0**; gold replay = **1.0**; no duplicates; no metadata
or `$var` syntax leaks into questions; build refuses to write anything if NESTFUL data
is unavailable for the overlap check.

## 14. Is v4 distributionally closer to NESTFUL than v3.1?

Yes, on 4 of 5 dimensions (total-variation distance to NESTFUL; mean 0.250 vs 0.337):
call counts (0.23 vs 0.30), offered tools per task (0.64 vs 0.89), argument types
(0.03 vs 0.15), answer types (0.12 vs 0.23). Tool arity is the exception (0.22 vs 0.11 —
v4 tools average more parameters than NESTFUL's many 1-arg APIs). Full table:
`data/curriculum_v4_nestful_like/AUDIT_REPORT.md`. Per the plan, this makes v4 a
*candidate*, not an improvement claim.

## 15. Remaining risks

- **Probe calibration unverified on GPU** — until Stage 1/2 v3.1 probes reproduce the
  audited dead-group ranges, probe verdicts should not gate decisions.
- **v3.2 gains are stub-level** — a real policy may collapse to fewer distinct rewards.
- **v4 questions are longer than NESTFUL's** (60 vs 33 words) and template-generated;
  lexical style remains a transfer risk despite better structural distributions.
- **Windows dev box cannot run official scoring** — every official number must come
  from the pod through the eval batch runner.
- **Archived paths in old run artifacts** — `config_used.*` inside historical runs still
  reference pre-archive paths; they are records, not inputs, but scripts that re-read
  them should resolve paths defensively.
- Heavy artifacts from run `20260708_212347_v3_1` are still untracked in git status;
  `.gitignore` now blocks them, but do not force-add.

## 16. Recommended first real experiment

On the pod, in order (each step gates the next):
1. `check_env.sh`, then probe **Stage 2, v3.1, 50 tasks, seed 42** — confirms calibration.
2. Same probe with **v3.2 dense** — proceed only if dead_group_rate drops meaningfully
   and unique rewards/group rises (RESEARCH_FIX_PLAN E1 criteria).
3. If (2) passes: **GRPO smoke (SMOKE=1) on Stage 2 with v3.2**, from base model, then a
   short real Stage-2 run; evaluate ONLY via a same-batch temp-0 eval
   (baseline + checkpoint cells, parallel) and judge by `official_nestful_win_rate`
   with paired gains/regressions.
4. In parallel (cheap): probe v4 Stage 1 (2-call) to compare signal against v3.1 Stage 2
   before considering any training on v4.
