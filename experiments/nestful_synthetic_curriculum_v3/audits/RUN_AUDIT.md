# RUN AUDIT — all training and evaluation runs

Date: 2026-07-09 · Read-only audit · Companions: `RUN_AUDIT.json` (full extraction incl.
per-epoch train-log analyses), `RUN_AUDIT.csv` (one row per run/stage/epoch).
Produced by `audits/tools/run_audit.py`.

Shared config across ALL runs (from checkpoint `config_used.json`): base model
`Qwen/Qwen3-4B-Instruct-2507`, QLoRA r=16 α=32 dropout=0.05 NF4, lr=5e-7, kl_beta=0.15,
seed=42, rollout top_p=0.95, vLLM rollouts. Differences are called out per run.

## 1. Training runs

| run id | stage(s) | dataset | reward policy | n_gen | rollout T | opt steps | dead-group rate | uniq reward vals | too_few_calls | avg calls | dev win int → | dev win off → | baseline (same run) int/off | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260702_112042 | — | curriculum_v3 (old) | — | — | — | 0 | — | — | — | — | — | — | — | aborted twin of 112150 (data_base only) |
| 20260702_112150 | 1 (2 ep), 2 (2 ep) | **curriculum_v3 (old, `synthetic_v3_*`, 223 s2 rows)** | execution_aware_v2_1_motif | 4 | 0.7 | 64+64 / 51+52 | s1 0.39 / s2 0.68 | binary {0,1} only | n/a (not logged) | n/a | s1: .60→.64; s2: .60→.565 | s1: .54→.575; s2: .545→.53 | .615 / .555 | best_react_win = s1_e2 (0.575 off) |
| 0260703_145219_v3_1 | 1 (2 ep), 2 (2 ep) | curriculum_v3_1 (A) | execution_aware_v3_1_stepwise | 4 | 0.7 | **0** (s1), 89+89 (s2) | s1 **1.00**, s2 0.78 | binary {0,1} only | n/a | n/a | s1: .605; s2: .60→.565 | s1: .565; s2: .53→.51 | .615 / .56 | s1 fully dead (0 optimizer steps); reward resolved to stepwise but produced only 0/1 |
| 20260707_103035_v3_1 | 2 (1 ep) **mixed replay 0.2** | curriculum_v3_1 (A) | execution_aware_v3_1_stepwise (fractional OK) | 8 | 1.0 | 50 | 0.876 | 18 | 0.438 | 1.36 | .59 | .535 | .625 / .57 | "fixed reward" pilot; gates PASS |
| 20260707_152750_v3_1 | 2 (1 ep) **no replay** | curriculum_v3_1 (A) | execution_aware_v3_1_stepwise | 8 | 1.0 | 29 | 0.855 | 16 | 0.558 | 1.44 | .60 | .545 | .625 / .57 | no-replay ablation |
| 20260707_183801_v3_1 | 2 (1 ep) **teacher-forced prefix=1** | curriculum_v3_1 (A) | execution_aware_v3_1_stepwise | 8 | 1.0 | 32 | 0.844 | 17 | 0.555 | 1.44 | .575 | .515 | .605 / .535 | teacher-forced ablation; note its own baseline is lower (.535) |
| 20260708_212347_v3_1 | 3 (2 ep, from baseline, no replay) | curriculum_v3_1 (A) | execution_aware_v3_1_stepwise | 8 | 1.0 | 59 / 71 | 0.709 / 0.646 | 17 | 0.418 / 0.414 | 2.18 / 2.20 | .605 / .595 | .545 / .535 | .635 / .565 | stage gate FAILED (position_artifact_rate 0.35 > 0.2) |

"dev win int" = `internal_metrics_diagnostic.win_rate` in `val_eval/metrics.json` (200-task
NESTFUL dev, executed); "dev win off" = `metrics_official.json` `win_rate` (official scorer).
All dev evals were sampled at the training temperature era defaults (not temp0) unless noted.

Key facts:

1. **Every run has a same-run baseline dev eval** (`baseline_dev_eval/`), so within-run dev
   deltas are valid. **No epoch of any run beat its own baseline on dev** — internal or
   official: e.g. Stage 3 best epoch .605 int / .545 off vs baseline .635 / .565.
2. The only apparent improvement ever recorded — run 112150 stage-1 epoch-2 (.575 official vs
   .555 baseline) — is on the **old v3 dataset with the old motif reward and binary rewards**,
   within eval noise (200 tasks ⇒ ±0.035 at 1σ), and vanished by stage 2.
3. The two July-2/3 runs produced **only binary rewards {0,1} and 0 optimizer steps in the
   fully saturated stage 1** of `0260703` — those runs are learning-signal-invalid for
   analysis of the stepwise reward (they predate the reward fix).
4. Runs `20260707_*` differ from each other in exactly one knob each (replay / no-replay /
   teacher-forced), enabling a clean 3-way comparison; all three had dead-group rates
   0.84–0.88 and none moved dev win beyond noise.

## 2. Final-eval batches (full NESTFUL, 1,861 tasks — includes the 200 dev tasks)

### Batch 1: `final_eval_all_runs_20260707_215620` (sampled decoding — **no** temp override)

| cell | internal win | official win | final_answer_pass |
|---|---|---|---|
| baseline | 0.6104 | — (no metrics_official.json) | 0.5932 |
| 103035_s2_e1 (replay) | 0.6099 | — | 0.5965 |
| 152750_s2_e1 (no replay) | 0.6093 | — | 0.5975 |
| 183801_s2_e1 (teacher-forced) | 0.6158 | — | 0.6024 |

### Batch 2: `final_eval_all_runs_20260708_164607_temp0` (temperature 0.0)

| cell | internal win | official win | final_answer_pass |
|---|---|---|---|
| baseline | 0.5986 | — | 0.5814 |
| 103035_s2_e1 | 0.5938 | — | 0.5760 |
| 152750_s2_e1 | 0.6024 | — | 0.5868 |
| 183801_s2_e1 | 0.6024 | — | 0.5841 |

### Batch 3: `final_eval_stage3_e1e2_20260709_093453_temp0` (temperature 0.0, official scorer, **NO baseline cell**)

| cell | internal win | official win | full acc | partial acc |
|---|---|---|---|---|
| s3_e1 | 0.6131 | **0.5438** | 0.024 | 0.189 |
| s3_e2 | 0.6077 | **0.5368** | 0.027 | 0.191 |

Comparability verdicts:

- **Batch 1 vs Batch 2 must not be compared** (sampled vs temp0; temp0 systematically lowered
  every cell by ~0.01).
- Within Batch 2 (valid same-batch comparison at temp0): best checkpoint beats baseline by
  **+0.38 pp internal** (0.6024 vs 0.5986) ≈ 7 tasks out of 1,861 — inside noise
  (±1.1 pp at 1σ for p≈0.6, n=1861). **No demonstrated improvement.**
- **Batch 3 has no baseline cell** ⇒ Stage-3 official numbers are unrankable. Batch 2's
  baseline is internal-metric-only and from a different batch/date; per policy do not compare.
- Batches 1–2 lack `metrics_official.json` entirely — the official scorer was only added to
  the final-eval path before Batch 3. Any historical "official win" claims for the Stage-2
  checkpoints do not exist for full NESTFUL.
- The `CHECKPOINT_REEVAL_REPORT.md` in Batches 1–2 says "No scored cells found" — the report
  generator looked under `outputs/final_eval_*` while metrics were written under
  `outputs/runs/final_eval_*` (path bug). The Batch-3 report worked but the paired win/loss
  table shows `shared=0` against a baseline that isn't there.

## 3. Checkpoint inventory

| checkpoint | path (under `outputs/runs/`) | status |
|---|---|---|
| s1/s2 adapters of 112150 + best_react_win | `20260702_112150/...` | legacy (old dataset+reward) |
| s1/s2 adapters of 0260703 + best_react_win | `0260703_145219_v3_1/...` | invalid (binary reward era; note the run-id typo missing leading "2") |
| s2_e1 of 103035 / 152750 / 183801 | `20260707_*/stage_2/checkpoints/adapter_epoch_1` | valid pilots, no dev improvement |
| s3_e1, s3_e2 | `20260708_212347_v3_1/stage_3/checkpoints/adapter_epoch_{1,2}` | trained OK; stage gate failed on position-artifact rate; not ranked vs baseline officially |

## 4. Hygiene issues

- Run id `0260703_145219_v3_1` is missing the leading `2` (timestamp typo) — sorts wrongly.
- Batches 2 and 3 have a doubled directory nesting (`<name>/<name>/…`).
- `curriculum_summary.jsonl` is empty for `20260707_*` and `20260708_*` runs (stage-gate flow
  bypasses it); the July runs' `FIXED_REWARD_PILOT_REPORT.md` shows "model n/a, dev Win n/a"
  because it reads the empty summary.
- Batch 1–2 evals were run with `paths.eval_jsonl` pointing at full NESTFUL, while per-epoch
  `eval/` (rollout_eval) used `data.eval_stage`-filtered slices of full NESTFUL (3-call: 407
  tasks, 4-call: 250 tasks) — three different eval populations coexist under one run tree.
