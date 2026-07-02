# Curriculum NESTFUL evaluation

Multi-turn NESTFUL evaluation for the curriculum training run: **baseline** plus every checkpoint under `curricullum/checkpoints/qwen3_4b_lora_grpo/`.

Uses the same driver as [`nestful_evaluation/run.py`](../../nestful_evaluation/run.py) (Tool-R0 system prompt, vLLM, IBM executable helpers). Outputs are compatible with [`eval_viewer.html`](../../eval_viewer.html).

## Requirements

- Linux + CUDA (vLLM does not run on Windows)
- Dependencies from repo root + `nestful_evaluation/requirements.txt`

## Google Colab

Notebook: [`curriculum_eval_colab.ipynb`](curriculum_eval_colab.ipynb)

Upload to **`MyDrive/Tool-R0/`**:

| Path on Drive | Required |
|---------------|----------|
| `grpo_processing.py` | yes |
| `nestful_evaluation/run.py` | yes |
| `curricullum/evaluation/run_eval.py` | yes |
| `curricullum/checkpoints/qwen3_4b_lora_grpo/stage{1,2,3}_*/adapter_model.safetensors` | yes (per stage) |
| same folders: `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | yes |
| `eval_viewer.html` | optional (view at home) |

Not needed on Drive: base model (HF Hub), IBM `nestful_repo` (git clone in Colab), NESTFUL dataset (HF).

Outputs land on Drive in `curricullum/evaluation/results/`; merged LoRA in `curricullum/evaluation/prepared/`.

### Colab: `torchao` merge error

If stage checkpoints fail with `incompatible version of torchao (0.10.0, need >=0.16.0)`:

```python
!pip install -U "torchao>=0.16.0"
!rm -rf /content/drive/MyDrive/Tool-R0/curricullum/evaluation/prepared/curriculum_stage*
```

Re-run eval with `SKIP_EXISTING=1` and `ONLY="stage1_1call stage2_2call stage3_3call"` (baseline already done).
Update `grpo_processing.py` on Drive from repo — it auto-upgrades torchao before merge.

## Quick start

```bash
# Pilot (50 tasks, 2 rollouts) — baseline + stage1/2/3
MAX_TASKS=50 NUM_ROLLOUTS=2 bash curricullum/evaluation/run_all_eval.sh

# Full benchmark (1861 tasks × 8 rollouts) on one GPU
bash curricullum/evaluation/run_all_eval.sh

# Two GPUs
TENSOR_PARALLEL_SIZE=2 bash curricullum/evaluation/run_all_eval.sh
```

## Profiles

| key | output profile slug | model |
|-----|---------------------|-------|
| `baseline` | `curriculum_baseline` | `Qwen/Qwen3-4B-Instruct-2507` |
| `stage1_1call` | `curriculum_stage1_1call` | LoRA merged for vLLM |
| `stage2_2call` | `curriculum_stage2_2call` | LoRA merged for vLLM |
| `stage3_3call` | `curriculum_stage3_3call` | LoRA merged for vLLM |

Checkpoints are auto-discovered from `--ckpt-root`. LoRA adapters are merged into `curricullum/evaluation/prepared/` (training checkpoints are not modified).

## Output files

Written to `curricullum/evaluation/results/`:

- `<profile>_multiturn_predictions.jsonl`
- `<profile>_multiturn_summary.json`
- `<profile>_multiturn_progress.json` (resume helper from nestful driver)

## View results

1. Open `eval_viewer.html` in a browser (local file or simple HTTP server).
2. Drag and drop one or more `*_multiturn_predictions.jsonl` files (and optional matching `*_multiturn_summary.json`).
3. Compare baseline vs `curriculum_stage*` in the multiturn mode.

## Useful env overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | HF base for baseline + LoRA merge |
| `CKPT_ROOT` | `curricullum/checkpoints/qwen3_4b_lora_grpo` | Curriculum checkpoint root |
| `OUTPUT_DIR` | `curricullum/evaluation/results` | JSONL + summary output |
| `NUM_ROLLOUTS` | `8` | Rollouts per task |
| `MAX_TASKS` | (all 1861) | Pilot limit |
| `MAX_STEPS` | `10` | Max model turns per rollout |
| `SKIP_EXISTING` | `0` | Set `1` to skip profiles with summary already present |
| `ONLY` | (all) | Space-separated subset, e.g. `ONLY="baseline stage3_3call"` |

## Python CLI

```bash
python curricullum/evaluation/run_eval.py --help

# Single checkpoint
python curricullum/evaluation/run_eval.py --only stage2_2call --max-tasks 20 --num-rollouts 1

# Resume after partial run
python curricullum/evaluation/run_eval.py --skip-existing
```

All flags from `nestful_evaluation/run.py` can be passed through `run_eval.py` (`--temperature`, `--tensor-parallel-size`, etc.).
