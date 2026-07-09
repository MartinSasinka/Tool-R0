# REMEDIATION EXECUTION REPORT

Date: 2026-07-09. Scope executed: plan documents + safe P0 infrastructure only.
P1/P2/P3 are planned (see roadmap) and intentionally **not** implemented — awaiting approval.

## 1. Audit files read

All 13: `MASTER_AUDIT_REPORT.md`, `DATASET_AUDIT.md`/`.json`, `RUN_AUDIT.md`/`.json`/`.csv`,
`FAILURE_MODE_AUDIT.md`, `REWARD_AUDIT.md`, `METRIC_AUDIT.md`, `METRIC_STANDARD_PROPOSAL.md`,
`STAGE3_AUDIT.md`, `CLEANUP_PLAN.md`, `IMPLEMENTATION_PLAN.md`.

## 2. Root causes identified (REMEDIATION_PLAN.md §1)

1. Reward-band quantization starves GRPO (65–88 % dead groups; 1.1–1.3 unique rewards/group).
2. Synthetic→NESTFUL transfer gap (toy tools, `arg_0/arg_1` schemas).
3. Evaluation protocol allowed phantom progress (internal win +6–7 pp over official; batches
   without official scores; Stage-3 batch without a baseline; mixed decoding).
4. Under-calling is the dominant behavioral failure (too_few_calls 0.41–0.56).
5. Repo hygiene erodes provenance (legacy dataset-B defaults, checkpoints about to be
   committed, confusable file naming, no manifests).

## 3. Remediation plan summary

Four documents created in this folder:

- **`REMEDIATION_PLAN.md`** — problems bucketed (eval/reporting, training/reward,
  dataset, hygiene), the P0–P3 priority table with fix/impact/risk/effort/acceptance per
  row, safest order, do-not-touch list.
- **`TARGET_ARCHITECTURE.md`** — target layout and flow (probe → train → batch-runner eval),
  component contracts (datasets, reward modules, evaluator, schema, manifests, W&B),
  archive policy. Existing MT-GRPO trainer preserved unchanged.
- **`IMPLEMENTATION_ROADMAP_FROM_AUDIT.md`** — Phase 0 (done, this change) through Phase 3,
  each with files, risks, tests, dry-run command, acceptance criteria, rollback.
- **`RESEARCH_FIX_PLAN.md`** — experiments E0–E5 (stage probe, reward densification,
  signal filtering, SFT warmup → GRPO, under-call weighting, NESTFUL-like v4 data), each
  with hypothesis / mechanism / dataset / reward / init / metrics / success criteria /
  failure interpretation. Framed strictly as experiments, no improvement claims.

## 4. P0 fixes implemented

| fix | artifact | verified by |
|---|---|---|
| stop committing heavy artifacts | `.gitignore` (this folder) | `git status` now clean of safetensors/tokenizer/trajectory files; `git check-ignore -v` spot checks |
| deterministic eval batch runner, same-batch baseline mandatory | `scripts/eval/run_eval_batch.py` + `scripts/eval/eval_batch_temp0.sh` | `--dry-run` prints correct per-cell commands (temp0, absolute paths → no double-nesting); refusal paths exercised: exit 2 (no baseline), exit 3 (legacy dataset B) |
| official scorer always verified | runner post-condition per cell (exit 4 if `metrics_official.json` missing) | code path + `--report-only` missing-file handling |
| unified metric schema (decoding block, internal win renamed) | `scripts/lib/metrics_schema.py` → `metrics_unified.json` | `--self-test` passes; built real unified metrics for the existing Stage-3 temp0 cells — s3_e1 official win 0.5438 CI [0.521, 0.566], and paired s3_e1-vs-s3_e2 counts (+95/−82) computed from per-sample official wins |
| run manifest skeleton | `scripts/lib/run_manifest.py`, wired into the eval runner | CLI test produced manifest with git commit+dirty, dataset SHA256+rows, seed, decoding, env versions |
| legacy-dataset guardrail | `scripts/lib/paths.py` (single path source) + runner check + `check_env.sh` warning | legacy-B eval attempt refused (exit 3); check_env flags both `config.yaml` defaults |
| setup + audit scripts | `scripts/setup/check_env.sh`, `scripts/audit/run_all_audits.sh` | `bash -n` clean; `check_env.sh` executed locally (correctly reports missing peft/bitsandbytes/CUDA on this workstation, finds IBM `executable_functions`, flags legacy configs) |
| batch report with paired counts | `BATCH_REPORT.md` generator (in runner) | `--report-only` over the existing Stage-3 batch produced a report stamped "no baseline — not comparable" |
| docs | `README.md` (rewritten — was stale "Training started: NO") + `docs/{EVALUATION,DATASETS,TRAINING,REWARD,RUNBOOK}.md` | cover: canonical vs legacy data, internal-win caveat, primary metric, same-batch procedure, paired-count interpretation, probe/SFT/GRPO how-to, commit policy, archive procedure |

## 5. P0 items intentionally NOT implemented (documented as risky/deferred)

- **`config.yaml` default repointing** (`nestful_mtgrpo_{minimal,partial}` still default to
  dataset B): shared files used by standalone experiments; changing them can break other
  usage. Mitigated in P0 by runner/check_env guardrails; actual repoint is Phase 1b, on
  approval.
- **Any `git mv` archival** (dataset B, `curriculum_v3`, July-2/3 runs, stale reports,
  run-id typo): breaks pod resume paths; staged as Phase 3, executed only on approval.
- **Training/SFT/probe launchers** (`scripts/training/run_grpo.sh`, `run_sft_plus_grpo.sh`,
  `scripts/probe/*`): they touch the live training path; locations fixed in
  TARGET_ARCHITECTURE, implementation is Phase 1.

## 6. Files changed / added

**Changed (2):** `README.md` (rewritten as entry point); nothing else — pre-existing
modifications to `scripts/pilot/run_stage2_continuation_sft_warmup.sh` and
`scripts/sft/train_stage2_continuation_sft.py` are from the earlier SFT task, untouched here.

**Added (17):**
`.gitignore`; `REMEDIATION_PLAN.md`; `TARGET_ARCHITECTURE.md`;
`IMPLEMENTATION_ROADMAP_FROM_AUDIT.md`; `RESEARCH_FIX_PLAN.md`;
`REMEDIATION_EXECUTION_REPORT.md`;
`scripts/lib/{__init__.py, paths.py, metrics_schema.py, run_manifest.py}`;
`scripts/eval/{run_eval_batch.py, eval_batch_temp0.sh}`;
`scripts/setup/check_env.sh`; `scripts/audit/run_all_audits.sh`;
`docs/{EVALUATION.md, DATASETS.md, TRAINING.md, REWARD.md, RUNBOOK.md}`.

**Moved to archive:** nothing (deliberately — P3).

Side effect on existing outputs: `--report-only` verification wrote `metrics_unified.json`
(2 cells) and `BATCH_REPORT.md` into the existing Stage-3 temp0 batch dir — additive,
nothing overwritten.

## 7. Tests / dry-runs executed (allowed scope only)

- `python -m compileall` on all new scripts — clean; no linter errors.
- `metrics_schema.py --self-test` — pass (schema invariants incl. "no bare win_rate key").
- Unified metrics + paired counts built from the real Stage-3 cells (values matched the
  audit: s3_e1 0.5438, s3_e2 0.5368).
- `run_manifest.py` CLI — manifest with commit/SHA/decoding verified.
- Eval runner: `--dry-run` (3-cell batch), no-baseline refusal (exit 2), legacy-dataset
  refusal (exit 3), `--report-only` with and without `--allow-no-baseline`.
- `bash -n` on all three shell scripts; `check_env.sh` executed end-to-end locally.
- No training, no SFT, no full evaluation, no GPU jobs.

## 8. Exact commands to run next (on the Linux pod)

```bash
cd /workspace/Tool-R0
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh

# 1) smoke the runner (minutes)
RUN=experiments/nestful_synthetic_curriculum_v3/outputs/runs/20260708_212347_v3_1
MAX_TASKS=5 BATCH_NAME=smoke \
CELLS="baseline,s3_e1=$RUN/stage_3/checkpoints/adapter_epoch_1" \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# 2) the real thing: same-batch baseline + s3_e1 + s3_e2, temp0, official scorer
BATCH_NAME=s3_with_baseline DATASET=nestful_test \
CELLS="baseline,s3_e1=$RUN/stage_3/checkpoints/adapter_epoch_1,s3_e2=$RUN/stage_3/checkpoints/adapter_epoch_2" \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh
```

Step 2 settles MASTER_AUDIT Q6/Q7 (did Stage 3 truly improve?) with the first-ever valid
same-batch official comparison; read `BATCH_REPORT.md` per `docs/EVALUATION.md`.

## 9. Recommended next real experiment

After the Stage-3 re-eval: **E0+E1** from `RESEARCH_FIX_PLAN.md` — build the stage probe,
calibrate it against the audited dead-group rates, then iterate reward densification
(`execution_aware_v3_2_dense`) against the probe gate (dead-group rate < 0.5) **before**
spending any GRPO pod-hours. In parallel and independently: E3 (evaluate the already-trained
Stage-2 SFT adapter through the new batch runner — its infrastructure already exists).

## 10. Remaining risks

- The runner's `run.py` invocation is dry-run-verified but not yet GPU-executed; the smoke
  batch (step 1 above) is the acceptance test. A known variable: official scorer needs the
  IBM functions dir and Linux.
- `config.yaml` defaults still point to dataset B until Phase 1b — mitigated but not fixed.
- `nestful_full` contains the dev-200 selection tasks; use `nestful_test` for headlines.
- Historical eval batches remain non-comparable forever (no baseline / no official scores);
  nothing can repair them — only the new protocol prevents recurrence.
- vLLM/version drift between the July runs and future evals is now *visible* (manifests)
  but not eliminated.
- Even with perfect evaluation, the science risks stand: reward densification may not
  transfer (RESEARCH_FIX_PLAN failure interpretations cover the branches).
