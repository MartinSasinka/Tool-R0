# NESTFUL Synthetic Curriculum v3.1

MT-GRPO (and SFT) experiments on a synthetic nested tool-use curriculum, evaluated on the
NESTFUL benchmark. This README is the entry point; the docs answer the recurring questions,
the audit records the full July-2026 state analysis, and the remediation plan defines what
gets fixed in which order.

**Status (2026-07-09):** pilots ran on stages 1–3; no checkpoint has beaten a same-batch
baseline on official NESTFUL win yet (see `audits/MASTER_AUDIT_REPORT.md`). P0 evaluation
infrastructure is in place; next step is the same-batch Stage-3 re-evaluation
(`REMEDIATION_EXECUTION_REPORT.md` §Next commands).

## The three rules

1. **Canonical training data** is `outputs/curriculum_v3_1/filtered/stage{1..4}_*.jsonl`
   (800 rows each). **Legacy dataset B** (now archived at
   `archive/legacy_dataset_B_filtered_toolr0_synthetic/`) must never be used; new
   tooling refuses it. Details: `docs/DATASETS.md`, `archive/README.md`.
2. **The only headline metric** is `official_nestful_win_rate` at temperature 0, from a batch
   that contains a baseline cell. The internal win rate inflates it by ~6–7 pp and is
   diagnostic-only. Details: `docs/EVALUATION.md`.
3. **Every eval goes through the batch runner**
   (`scripts/eval/run_eval_batch.py`) — it enforces rule 2, records provenance
   (git commit, dataset SHA, seed, decoding), and emits `BATCH_REPORT.md` with paired
   gained/regressed counts.

## Quick start

```bash
# sanity-check environment, datasets, official-scorer prerequisites
bash experiments/nestful_synthetic_curriculum_v3/scripts/setup/check_env.sh

# run an eval batch (baseline REQUIRED; temp0; official scorer verified per cell)
CELLS="baseline,s3_e1=experiments/nestful_synthetic_curriculum_v3/outputs/runs/20260708_212347_v3_1/stage_3/checkpoints/adapter_epoch_1" \
DATASET=nestful_test \
bash experiments/nestful_synthetic_curriculum_v3/scripts/eval/eval_batch_temp0.sh

# re-run the audit extractors
bash experiments/nestful_synthetic_curriculum_v3/scripts/audit/run_all_audits.sh
```

Training and SFT how-tos: `docs/TRAINING.md`. Full operational procedures (smoke tests,
archive policy, what not to commit): `docs/RUNBOOK.md`.

## Map of this folder

| Path | What it is |
|---|---|
| `docs/` | EVALUATION, DATASETS, TRAINING, REWARD, RUNBOOK |
| `audits/` | frozen July-2026 audit (13 reports) + `tools/` extractors |
| `REMEDIATION_PLAN.md` | root causes, P0–P3 priority table |
| `TARGET_ARCHITECTURE.md` | target layout and component contracts |
| `IMPLEMENTATION_ROADMAP_FROM_AUDIT.md` | phased implementation with risks/tests/rollback |
| `RESEARCH_FIX_PLAN.md` | reward/probe/SFT+GRPO/data experiments (hypotheses, not claims) |
| `REMEDIATION_EXECUTION_REPORT.md` | what P0 implemented, what to run next |
| `scripts/setup/`, `scripts/audit/`, `scripts/eval/`, `scripts/lib/` | P0 tooling (this change) |
| `scripts/sft/`, `scripts/pilot/` | Stage-2 SFT view/training/eval (existing) |
| `scripts/run_curriculum_v3.sh` | GRPO curriculum launcher (existing, pod) |
| `lib/` | reward modules (`reward_v3_1.py` frozen baseline) |
| `outputs/curriculum_v3_1/` | canonical dataset + manifest |
| `outputs/runs/`, `outputs/sft/` | training outputs (heavy artifacts gitignored) |
| `outputs/evals/` | eval batches produced by the batch runner |

## Historical build docs

The v3.1 corpus build (888 source trajectories → 4×800 prefix-decomposed stages, 100 % gold
replay) is documented in `outputs/CURRICULUM_V3_1_DESIGN_DECISION.md` and
`outputs/curriculum_v3_1/CURRICULUM_V3_1_IMPLEMENTATION_REPORT.md`. The v3 (non-.1) corpus
and its scripts are legacy.
