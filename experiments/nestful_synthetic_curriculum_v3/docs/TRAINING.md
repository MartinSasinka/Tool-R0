# TRAINING

The MT-GRPO trainer lives in `experiments/nestful_mtgrpo_minimal/` (`grpo_train.py`,
`rollout.py`, `run.py`) and is deliberately left unchanged by the remediation — this folder
wraps it. Base model: `Qwen/Qwen3-4B-Instruct-2507`, QLoRA r=16 α=32 4-bit NF4.

## GRPO curriculum run (pod, 4 GPU)

```bash
cd /workspace/Tool-R0

ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 \
ROLLOUT_DP_GPUS="1,2,3" DP_LEARNER_GPU=0 \
CURRICULUM_VERSION=v3_1 STAGES="2" MAX_EPOCHS_PER_STAGE=2 \
REWARD_POLICY=execution_aware_v3_1_stepwise \
bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh
```

Key env vars: `STAGES` (space-separated stage list), `MAX_EPOCHS_PER_STAGE`,
`CHECKPOINT_IN` (resume from an adapter), `DRY_RUN=1` (resolve and print, no training),
`WANDB_API_KEY`/`WANDB_PROJECT` for logging. The launcher copies canonical stage files into
`OUTPUT_ROOT/data_base/epoch_N_*.jsonl` — those are per-run copies of dataset A, not
dataset B (see docs/DATASETS.md).

Before training a new stage or reward: run the stage probe first (P1, planned —
`scripts/probe/probe_stage.sh`). Stage 1 is saturated under the v3.1 reward (dead-group
rate ≈ 1.0, zero optimizer steps in the July-3 run); do not spend GPU time on it.

## SFT warmup (Stage-2 continuation)

```bash
# 1) build the SFT view of the canonical stage-2 file (derived, not a new dataset)
python experiments/nestful_synthetic_curriculum_v3/scripts/sft/build_stage2_sft_dataset.py

# 2) train (QLoRA; loss masked to target tokens)
SFT_EPOCHS=1 SFT_LR=1e-5 SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=16 \
bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/run_stage2_continuation_sft_warmup.sh

# 3) evaluate (continuation-conditioned + free ReAct + NESTFUL)
bash experiments/nestful_synthetic_curriculum_v3/scripts/pilot/eval_stage2_sft_warmup.sh
```

The SFT→GRPO chain (`scripts/training/run_sft_plus_grpo.sh`) is planned P1
(RESEARCH_FIX_PLAN E3): SFT adapter → GRPO resume via `CHECKPOINT_IN`.

## Evaluating a trained checkpoint

Always via the batch runner — never by calling `run.py --mode final_eval` directly, because
the runner is what guarantees a same-batch baseline, temp0, official scorer, manifest and
report. See `docs/EVALUATION.md`.

Checkpoint selection during training uses the dev-200 `val_eval`; dev numbers are selection
signals only, never headline.

## Known training-signal issues (audited)

- 65–88 % dead groups under the v3.1 reward at 8 generations, T=1.0 — reward band
  quantization, not decoding (FAILURE_MODE_AUDIT §1). Reward densification is experiment E1.
- Under-calling dominates failures (too_few_calls 0.41–0.56 on eval).
- Stage 3 gate fails on position-artifact rate (0.35–0.41 > 0.2): a third of "alive" groups
  get variance from turn position, not completion differences.
- `strict_gold_trace_pass` in `train_log.jsonl` is mean episode reward, not the eval metric.

## Provenance requirements for new runs

Record with every run (P2 wires this into the launcher; until then, manually): git commit,
dataset SHAs (`scripts/lib/run_manifest.py` does both), seed, reward policy name, decoding
settings, and the exact launch command. Never train on NESTFUL dev/test.
