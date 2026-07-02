# Curriculum GRPO Training

Minimal Tool-R0-style **LoRA + GRPO** training on synthetic NESTFUL curriculum JSONL produced by [`../run_generate_curriculum.sh`](../run_generate_curriculum.sh).

This module trains **sequentially** (default **3 stages**, 1→2→3 calls):

```text
base Qwen3-4B
  -> stage 1 (1-call JSONL)  -> adapter checkpoint
  -> stage 2 (2-call JSONL)  -> continues from stage 1 adapter
  -> stage 3 (3-call JSONL)  -> final adapter checkpoint (default)
  -> stage 4 (4-call JSONL)  -> optional (MAX_STAGES=4)
```

Checkpoints are **LoRA adapters only** (no merged full model by default).

## Relation to Tool-R0

| Tool-R0 | This module |
|---------|-------------|
| [`step3_solver.py`](../../step3_solver.py) + [`run_step3.sh`](../../run_step3.sh) | [`train_grpo_stage.py`](train_grpo_stage.py) + [`run_train_curriculum.sh`](run_train_curriculum.sh) |
| [`grpo_processing.py`](../../grpo_processing.py) | Reused (tokenizer, PEFT workarounds) |
| [`rewards_solver.py`](../../rewards_solver.py) (tag format) | [`rewards_nestful.py`](rewards_nestful.py) (JSON `output` + `answer`) |
| [`configs/deepseed_zero2_offload.yaml`](../../configs/deepseed_zero2_offload.yaml) | Default DeepSpeed config |

## Why LoRA, not QLoRA

Curriculum training targets **1× A100 40GB** first with conservative GRPO settings (`num_generations=2`, `batch=1`, `grad_accum=8`, `lora_r=16`). Full bf16 LoRA avoids quantization + PEFT edge cases; fits the same stack as Tool-R0 step3.

## Prerequisites

1. Generated curriculum files:

```text
curricullum/data/filtered/epoch_1_1call.jsonl   # required
curricullum/data/filtered/epoch_2_2call.jsonl   # required
curricullum/data/filtered/epoch_3_3call.jsonl   # required
curricullum/data/filtered/epoch_4_4call.jsonl   # optional (MAX_STAGES=4 only)
```

2. Dependencies from repo root:

```bash
pip install -r requirements.txt
wandb login
```

3. Inspect data before training:

```bash
python curricullum/train/inspect_training_data.py \
  --path curricullum/data/filtered/epoch_1_1call.jsonl
```

## Model output format

The model should emit **plain JSON** (no markdown):

```json
{
  "output": [{"name": "add", "label": "$var_1", "arguments": {"arg_0": 1, "arg_1": 2}}],
  "answer": "3"
}
```

Gold fields in the dataset remain `output` + `gold_answer`. The reward function also accepts `answer`, `final_answer`, or `gold_answer` in completions.

Prompts **do not** include gold trajectories.

## Config

[`configs/qwen3_4b_lora_grpo.yaml`](configs/qwen3_4b_lora_grpo.yaml):

- LoRA: r=16, alpha=32, 7 target modules
- GRPO: `num_generations=2`, `max_prompt_length=4096`, stage-specific `max_completion_length`
- Training: **`num_train_epochs=1.0`**, `max_steps: null` (~50 optimizer steps per stage with ~400 samples, batch=1, grad_accum=8)

Stage completion caps:

| Stage | max_completion_length |
|-------|----------------------|
| 1 | 1536 |
| 2 | 2048 |
| 3 | 3072 |
| 4 (optional) | 3072 |

## Single-GPU run (default, 3 stages)

```bash
export NUM_PROCESSES=1
bash curricullum/train/run_train_curriculum.sh
```

Include stage 4 later (when enough 4-call data exists):

```bash
export MAX_STAGES=4
bash curricullum/train/run_train_curriculum.sh
```

Smoke test (2 optimizer steps per stage):

```bash
export MAX_STEPS=2
export OVERWRITE=1
bash curricullum/train/run_train_curriculum.sh
```

## Multi-GPU run

```bash
export NUM_PROCESSES=3
export TOOL_R0_DEEPSPEED_CONFIG=./configs/deepseed_zero2_offload.yaml
bash curricullum/train/run_train_curriculum.sh
```

Or one stage manually:

```bash
accelerate launch \
  --config_file ./configs/deepseed_zero2_offload.yaml \
  --num_processes 3 \
  curricullum/train/train_grpo_stage.py \
  --config curricullum/train/configs/qwen3_4b_lora_grpo.yaml \
  --stage stage_1 \
  --output_dir curricullum/checkpoints/qwen3_4b_lora_grpo/stage1_1call \
  --previous_adapter none \
  --wandb_run_name qwen3-4b-lora-stage1-1call
```

## Environment overrides

| Variable | Default |
|----------|---------|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` |
| `WANDB_PROJECT` | `nestful-curriculum-grpo` |
| `WANDB_RUN_GROUP` | `qwen3-4b-lora-curriculum` |
| `NUM_GENERATIONS` | `2` |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` |
| `GRADIENT_ACCUMULATION_STEPS` | `8` |
| `LEARNING_RATE` | `5e-6` |
| `LORA_R` / `LORA_ALPHA` | `16` / `32` |
| `MAX_STAGES` | `3` (set `4` to train 4-call stage) |
| `MAX_STEPS` | empty (use full epoch) |
| `OVERWRITE` | `0` |
| `EVAL_CMD` | optional post-stage hook |

## Checkpoints

```text
curricullum/checkpoints/qwen3_4b_lora_grpo/
  stage1_1call/   adapter + training_summary.json
  stage2_2call/
  stage3_3call/     final adapter (default 3-stage run)
  stage4_4call/     optional 4-call stage
```

Each stage loads the **previous adapter as trainable** and saves a **new** adapter directory (never overwrites prior stages unless `OVERWRITE=1`).

Optional merge for serving:

```bash
python curricullum/train/train_grpo_stage.py ... --merge_adapter
```

## Resume / overwrite

If `adapter_config.json` exists in the output dir, training exits unless `--overwrite` or `OVERWRITE=1`.

To resume mid-curriculum, run a single stage with `--previous_adapter` pointing at the last completed stage.

## W&B metrics

Per stage (one run per stage, shared group):

- `train/reward`, `reward/format`, `reward/call_count`, … (8 components)
- `metrics/invalid_json_rate`, `metrics/truncated_rate`, `metrics/avg_completion_length`
- `curriculum/stage`, `curriculum/num_calls`, `curriculum/max_completion_length`, `curriculum/estimated_optimizer_steps`

Component logging is batch-aggregated on the main process only (not per-sample).

## Optional evaluation hook

```bash
export EVAL_CMD='echo eval $STAGE_NAME $ADAPTER_PATH'
bash curricullum/train/run_train_curriculum.sh
```

## Prompt budget

Rows exceeding `max_prompt_length` are **skipped** (not silently truncated). `inspect_training_data.py` reports over-budget counts.

## Limitations

- No full NESTFUL benchmark eval in this folder (use `EVAL_CMD` hook).
- Output format differs from legacy Tool-R0 solver tags (`<tool_call_answer>`).
- Requires filtered JSONL from the curriculum generator; training fails early if files are missing.
