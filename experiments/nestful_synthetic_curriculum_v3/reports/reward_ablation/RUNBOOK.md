# Reward Ablation — Runbook

See `ABLATION_PLAN.md` for the full design and `ABLATION_PARITY.md` for the
exact frozen hyperparameters. Commands below assume repo root
`Tool-R0/`. PowerShell examples use `;` as the statement separator (not
`&&`); Linux/RunPod examples use bash.

## 1. Discovery (already done; kept for reproducibility)

```powershell
# Confirm the 326-task Stage-3 source dataset + hash
python -c "import hashlib; h=hashlib.sha256(); f=open('experiments/nestful_synthetic_curriculum_v3/data/training_ready_v5/filtered/stage3_train_ready.jsonl','rb'); [h.update(c) for c in iter(lambda: f.read(1<<20), b'')]; print(h.hexdigest())"
```

## 2. Prepare + freeze the 160-task train subset

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_train_subset_160.py
```

Writes `reports/reward_ablation/data/train_subset_160.jsonl` +
`train_subset_manifest.json` + `TRAIN_SUBSET_REPORT.md`. Re-running must
reproduce an identical SHA-256 (enforced by
`test_train_subset_manifest_hash_matches_file`).

## 3. Prepare + freeze the 500-task NESTFUL eval subset

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/prepare_nestful_diagnostic_500.py
```

Writes `reports/reward_ablation/data/nestful_diagnostic_500_ids.json` +
`nestful_diagnostic_500_manifest.json` + `NESTFUL_DIAGNOSTIC_SUBSET.md`.

## 4. Freeze reward specs (unit tests + probe + invariants)

```powershell
# CPU pipeline self-test (this environment) — structural validation only:
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/freeze_reward_specs.py --backend stub
```

```bash
# REAL calibration — RunPod, GPU required, before trusting Round 1 GPU time:
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/freeze_reward_specs.py --backend vllm
```

Writes `reports/reward_ablation/FROZEN_REWARD_SPECS.json`. Do not edit
terminal bands/epsilon/process weights/gates after this without bumping
`ablation_version`.

## 5. Unit tests

```powershell
python -m pytest experiments/nestful_synthetic_curriculum_v3/tests/test_reward_ablation.py experiments/nestful_synthetic_curriculum_v3/tests/test_reward_ablation_pipeline.py -q
```

## 6. Smoke test — every arm (RunPod, before committing full Round 1 GPU time)

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh --smoke
# or one arm:
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh --arm A2_R3_OUTCOME_FIRST --smoke
```

Gate (spec §9): no crash, no NaN/Inf, optimizer step ran, reward components
logged, terminal inversions=0 (A1–A4), checkpoint round-trips, real
synthetic executor, eval parity held, W&B run created. Only proceed to
Round 1 once every arm's smoke test passes.

## 7. Round 1 — RunPod (sequential, all 5 arms)

```bash
export WANDB_API_KEY=...        # never logged/printed by the launcher
export HF_TOKEN=...
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh
```

One arm only:

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh \
  --arm A2_R3_OUTCOME_FIRST --seed 20260724
```

Direct CLI (equivalent, without the launcher's environment checks):

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py \
  --round 1 --reward-arm A2_R3_OUTCOME_FIRST --seed 20260724 \
  --wandb-project nestful-reward-ablation \
  --wandb-group reward_ablation_round1_$(date -u +%Y%m%d)
```

Dry-run first (no GPU needed, works anywhere, validates config/hash resolution):

```powershell
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation.py --round 1 --reward-arm A2_R3_OUTCOME_FIRST --seed 20260724 --dry-run
```

## 8. Resume one arm

```bash
bash experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_reward_ablation_round1.sh \
  --arm A3_VERIFIABLE_PROCESS --seed 20260724 --resume
```

`run_reward_ablation.py` refuses `--resume` if the target run directory's
`run_manifest.json` recorded a different `reward_arm` or `seed` — it will
never silently overwrite another arm's checkpoint.

## 9. Evaluation

Evaluation runs automatically as part of `run_reward_ablation.py` (shared
C0 eval once, then the arm's final checkpoint) — see
`outputs/runs/<experiment_id>/eval/<arm>/<seed>/`.

## 10. Summarize (per arm, then cross-arm)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/summarize_reward_ablation.py arm \
  --arm A2_R3_OUTCOME_FIRST \
  --eval-dir outputs/runs/reward_ablation_r1_A2_R3_OUTCOME_FIRST_seed20260724/eval/A2_R3_OUTCOME_FIRST/20260724 \
  --c0-dir outputs/runs/reward_ablation_r1_A0_R0_CURRENT_seed20260724/eval/A0_R0_CURRENT/20260724 \
  --r0-dir outputs/runs/reward_ablation_r1_A0_R0_CURRENT_seed20260724/eval/A0_R0_CURRENT/20260724

python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/summarize_reward_ablation.py round-summary \
  --round 1 \
  --arm-dir A0_R0_CURRENT=outputs/runs/.../eval/A0_R0_CURRENT/20260724 \
  --arm-dir A1_OUTCOME_ONLY=outputs/runs/.../eval/A1_OUTCOME_ONLY/20260724 \
  --arm-dir A2_R3_OUTCOME_FIRST=outputs/runs/.../eval/A2_R3_OUTCOME_FIRST/20260724 \
  --arm-dir A3_VERIFIABLE_PROCESS=outputs/runs/.../eval/A3_VERIFIABLE_PROCESS/20260724 \
  --arm-dir A4_GATED_VERIFIABLE=outputs/runs/.../eval/A4_GATED_VERIFIABLE/20260724
```

Writes `reports/reward_ablation/round1/ROUND1_SUMMARY.{json,md}`.

## 11. Select top arms + prepare Round 2 (not auto-launched)

```bash
python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/select_reward_arms.py \
  --round 1 \
  --summary reports/reward_ablation/round1/ROUND1_SUMMARY.json \
  --training-diagnostics reports/reward_ablation/round1/training_diagnostics.json
```

Writes `ROUND1_DECISION.{json,md}` and `ROUND2_PLAN.json` (exact Round 2
commands; Round 2 itself must be launched manually, per spec §2).

## 12. W&B

- Project: `nestful-reward-ablation`
- Group: `reward_ablation_round1_<timestamp>` (Round 2:
  `reward_ablation_round2_<timestamp>`)
- Run names: `<ARM_ID>_seed<SEED>` (e.g. `A2_R3_OUTCOME_FIRST_seed20260724`)
- Tags: `reward_ablation`, `pure_stage3`, `round1`/`round2`, reward ID, seed,
  dataset hash, C0 hash.

## 13. Expected scale (Round 1)

- 5 arms x 160 tasks x 8 rollouts x 1 epoch = 5 training sessions, each
  ~160 rollout groups of 8 (1280 rollouts) per epoch.
- 6 evaluations total: 1 shared C0-on-500 + 5 arm-final-on-500 (500 tasks
  each, temperature=0, 1 rollout/task).
- 5 checkpoints published (`outputs/runs/<experiment_id>/checkpoints/FINAL/`),
  one per arm, never overwritten across arms (`assert_resume_compatible`).
- Optional midpoint eval on n=100 mini-subset per arm (not wired into the
  default CLI path; add manually if desired before spending full training time).

## 14. On error

- Every arm writes `outputs/runs/<experiment_id>/logs/console.log` (full
  stdout/stderr via `tee`) and `ablation_run_state.json` (resume-safe step
  markers: `preflight`, `train`, `eval_C0`, `eval_arm`).
- `run_reward_ablation_round1.sh` defaults to `STOP_ON_FAILURE=1` (stops the
  whole Round 1 sequence on the first failed arm); set `STOP_ON_FAILURE=0`
  to keep going and collect a full failure summary.
- Never delete another arm's `checkpoints/FINAL/` — `--resume` is guarded
  against reward-arm/seed mismatches, but a fresh `--run-id` into an
  existing directory without `--resume` is rejected outright.
- If `FROZEN_REWARD_SPECS.json` is missing or was frozen with
  `--backend stub`, the launcher aborts before spending GPU time (see §4).
