# Reward Ablation — Parity Report

Generated from `configs/reward_ablation/round1_base.yaml` +
`configs/reward_ablation/arms/*.yaml`. This is the single source of truth
for what is held identical across A0–A4; regenerate this file by hand
whenever those YAMLs change (the values below are copied verbatim from
them, not recomputed by a script, so they must stay in sync).

**The only key that legitimately differs between arms is `reward.train_policy`.**
Everything else below is copy-pasted from `round1_base.yaml` and applies
identically to every one of A0_R0_CURRENT, A1_OUTCOME_ONLY,
A2_R3_OUTCOME_FIRST, A3_VERIFIABLE_PROCESS, A4_GATED_VERIFIABLE.

## Model / checkpoint

| Key | Value |
|---|---|
| base_model | `Qwen/Qwen3-4B-Instruct-2507` |
| base_model_revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| start_from | `base_model_C0` (identical C0 checkpoint for every arm) |

## Data

| Key | Value |
|---|---|
| train_dataset | `reports/reward_ablation/data/train_subset_160.jsonl` |
| train_dataset_sha256 (LF-normalized) | `7df704bff35c8f8fd0ffb2b50e3c7c4c1e8d7f9a0f3e0c02a43327ef820dd596` |
| eval_dataset_ids | `reports/reward_ablation/data/nestful_diagnostic_500_ids.json` |
| eval_dataset_source | `experiments/nestful_mtgrpo_minimal/data/splits/nestful_test.jsonl` |
| executor_mode | `synthetic` |
| no_stage2 | `true` |
| no_gold_replay | `true` |
| mixed_replay | `false` |

## Training

| Key | Value | Note |
|---|---|---|
| epochs | `1` | ROUND 1 DELIBERATE CHANGE vs production (2 epochs) |
| num_generations | `8` | rollouts per task, identical every arm |
| learning_rate | `3.0e-7` | |
| kl_beta | `0.15` | |
| gamma | `1.0` | |
| lambda_episode | `1.0` | |
| mt_grpo_mode | `turn_level_minimal` | |
| max_grad_norm | `1.0` | |
| per_device_train_batch_size | `1` | |
| gradient_accumulation_steps | `4` | |
| save_every_epoch | `true` | |
| checkpoint_interval | `every_epoch` | |

## Generation (rollout decoding)

| Key | Value |
|---|---|
| temperature | `1.0` |
| top_p | `0.95` |
| max_new_tokens_train | `2048` |
| max_prompt_tokens | `4096` |

## Eval decoding (fixed for C0 and every arm's checkpoint)

| Key | Value |
|---|---|
| temperature | `0.0` |
| top_p | `1.0` |
| num_eval_rollouts | `1` |
| eval_paradigm | `react` |
| midpoint_checkpoint_minisubset | `100` (optional) |

## Precision / quantization

| Key | Value |
|---|---|
| bf16 | `true` |
| load_in_4bit | `true` |
| bnb_4bit_quant_type | `nf4` |
| bnb_4bit_compute_dtype | `bfloat16` |
| bnb_4bit_use_double_quant | `true` |
| finetuning_method | `qlora` |
| lora_r | `16` |
| lora_alpha | `32` |
| lora_dropout | `0.05` |

## Hardware / GPU topology

| Key | Value |
|---|---|
| gpu_topology | GPU0=learner, GPU1-3=rollout_workers (vLLM DP) |
| eval_gpu_topology | 4xTP eval AFTER learner/optimizer released (no concurrent TP4 eval while learner resident) |
| rollout_data_parallel_gpus | `[1, 2, 3]` |

## Seeds (Round 1)

| Key | Value | Note |
|---|---|---|
| SEED | `20260724` | ROUND 1 DELIBERATE CHANGE vs production SEED=42 |
| DATA_SEED | `20260724` | |
| ROLLOUT_SEED | `20260724` | |

Round 2 re-uses the identical 160 train tasks / 500 eval tasks / all other
hyperparameters with `SEED=DATA_SEED=ROLLOUT_SEED=20260725` (see
`ROUND2_PLAN.json` once produced).

## Reproducibility anchors

| Key | Value |
|---|---|
| registry_version | `5.0.2` |
| registry_hash | `f945b18ccdc260b1960e5fbb20e4d76312628af42fef0a65d4977af83dd6dc0d` |

## The single experimental variable

| Arm | `reward.train_policy` |
|---|---|
| A0_R0_CURRENT | `execution_aware_v3_2_dense` |
| A1_OUTCOME_ONLY | `reward_ablation_A1_OUTCOME_ONLY` |
| A2_R3_OUTCOME_FIRST | `reward_ablation_A2_R3_OUTCOME_FIRST` |
| A3_VERIFIABLE_PROCESS | `reward_ablation_A3_VERIFIABLE_PROCESS` |
| A4_GATED_VERIFIABLE | `reward_ablation_A4_GATED_VERIFIABLE` |

`tests/test_reward_ablation_pipeline.py::test_effective_config_only_differs_by_reward_train_policy`
enforces this programmatically: it loads all 5 effective configs (base +
arm overlay) and asserts they are byte-identical after stripping the
`reward`/`wandb`/`reward_id`/`description` keys.

## Trainer / rollout / executor / credit-assignment code reused verbatim

- `scripts/training/two_phase_train_session.py::TwoPhaseTrainSession` — same
  class the production `pure_stage3_two_epoch` orchestrator uses (`load_learner`,
  `start_rollout_workers`, `train_phase`, `sync_rollout_policy`, `close`).
- `scripts/training/two_phase_utils.py` — atomic checkpoint publish, repro
  manifest, GPU-eval prep, dataset audits.
- `scripts/training/preflight_training_datasets.py` — same synthetic-executor
  replay gate.
- `scripts/eval/final_eval_v5.py` — same eval CLI (forces
  temperature=0/top_p=1/react, same prompt/parser/executor/scorer).
- `vllm_dp_pool.py::resolve_reward_info` — same reward-dispatch mechanism
  every existing reward policy in this repo goes through; the ablation only
  adds one more `elif` branch (`reward_ablation_<ARM_ID>`).

None of these files were modified in a way that changes behavior for any
non-ablation caller (`resolve_reward_info` gained a new branch only reached
when `reward.train_policy` starts with `reward_ablation_a`).
