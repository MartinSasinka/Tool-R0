# NESTFUL MT-GRPO Minimal

Online multi-turn tool-use RL on NESTFUL tasks. Self-contained folder — copy with `data/` and run.

Training uses strict gold-trace reward only. `solution_equivalent_pass` is eval-only.

## Quick start

```bash
cd nestful_mtgrpo_minimal

conda create -n mtgrpo python=3.10 -y && conda activate mtgrpo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."
export WANDB_PROJECT="nestful-mtgrpo"

python -m pytest tests/ -q

CUDA_VISIBLE_DEVICES=0 python run.py --mode smoke --config config.yaml
```

Needs 1× GPU (A100 40 GB recommended), `data/` in the folder, HuggingFace access for the base model.

## `run_curriculum.sh` — example

Run from inside this folder. The script trains stage N, evaluates on stage N+1, saves checkpoints under `outputs/curriculum/`.

**Pilot** (quick check, stage 3, 16 train / 32 eval tasks):

```bash
cd nestful_mtgrpo_minimal

CUDA_VISIBLE_DEVICES=0 \
USE_VLLM=1 \
PROFILE=pilot \
STAGES="3" \
bash run_curriculum.sh
```

**Full curriculum** (stages 1–4, all tasks, up to 4 epochs per stage):

```bash
cd nestful_mtgrpo_minimal

CUDA_VISIBLE_DEVICES=0,1,2,4 \
USE_VLLM=1 \
PROFILE=curriculum \
STAGES="1 2 3 4" \
MAX_EPOCHS_PER_STAGE=4 \
RUN_FINAL_EVAL=1 \
bash run_curriculum.sh
```

**Resume** from a saved adapter (e.g. continue from stage 3):

```bash
CUDA_VISIBLE_DEVICES=0 \
USE_VLLM=1 \
PROFILE=curriculum \
STAGES="3 4" \
CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_2 \
bash run_curriculum.sh
```

Dry-run (print commands only, no GPU):

```bash
DRY_RUN=1 PROFILE=pilot STAGES="3" bash run_curriculum.sh
```

## Evaluation metrics — what to report vs what to debug with

There are two metric sources. Full audit: [`docs/AUDIT.md`](docs/AUDIT.md).

- **`official_*` (CANONICAL — use in paper/benchmark tables).** Produced by the
  real NESTFUL scorer (`data/NESTFUL-main/src`) via `nestful_official_score.py`
  and written to **`metrics_official.json`**. Report exactly these:
  `official_f1_func`, `official_f1_param`, `official_partial`, `official_full`,
  `official_win`.
- **`internal_*` (DIAGNOSTIC ONLY — never report as paper numbers).** Produced by
  `metrics.py`, a fixed replica of the official semantics, written to
  **`metrics.json`** under `internal_metrics_diagnostic`. Use these to
  cross-check and debug, alongside `mismatch_reason`.

**Hard rule:**

- Paper tables → `official_f1_func`, `official_f1_param`, `official_partial`,
  `official_full`, `official_win` (from `metrics_official.json`).
- Debug/analysis → `internal_*` + `mismatch` / `mismatch_reason` in the
  per-sample trajectory files.

Per-sample diagnostics (in `*_trajectories.jsonl`): `official_partial_match`,
`official_full_match`, `official_win`, `pred_answer`, `parse_valid`,
`executable`, `execution_error`, plus `internal_*` and a `mismatch` flag. When
`mismatch` is true, **trust `official_*`** — the flag only exists so we can find
where the internal replica diverges.

Interpretation notes:

- **Win Rate** comes exclusively from the official scorer, which re-executes the
  predicted calls. It requires `signal.SIGALRM` (Linux/RunPod) and is skipped on
  Windows. **Win Rate can exceed Full Match**: an alternative valid trajectory
  can reach the gold answer without matching the gold steps.
- **F1 Func / F1 Param are corpus-level macro metrics** (not per-sample). The
  official macro-F1 has a *different interpretation* than a per-sample or micro
  score — over a large (~900) function vocabulary it tends to read high because
  rare, distinctive functions are easy to match. It is the correct official
  definition; we simply also surface supplementary diagnostics
  (`internal_f1_*`, corpus macro replica) next to it for transparency.
- **ReAct vs Direct**: our default ReAct (multi-turn) numbers are **not**
  comparable to the paper's Direct (Table 1) row. Use `data.eval_paradigm=direct`
  for the Direct paradigm.

## Main commands

```bash
python run.py --mode smoke        --config config.yaml
python run.py --mode train        --config config.yaml
python run.py --mode rollout_eval --config config.yaml --checkpoint outputs/checkpoints/adapter_epoch_1
python run.py --mode final_eval   --config config.yaml --checkpoint outputs/checkpoints/adapter_epoch_1
```

Evaluation / scoring commands:

```bash
# Unit tests (Win Rate tests auto-skip on Windows; run on Linux for full coverage)
python -m pytest tests -q

# NOTE: --override takes ONE key=value; repeat the flag for each override.

# Small-subset eval (50 tasks), baseline (no adapter)
python run.py --mode final_eval --override data.max_eval_tasks=50 --override model.lora_adapter=null
# ...same, Direct paradigm
python run.py --mode final_eval --override data.max_eval_tasks=50 --override model.lora_adapter=null --override data.eval_paradigm=direct

# Full eval (1861 tasks); Win Rate requires Linux/RunPod
ONLY_FINAL_EVAL=1 USE_VLLM=1 CHECKPOINT_IN=<adapter> bash run_curriculum.sh
python run.py --mode final_eval --checkpoint <adapter>

# Direct eval via vLLM (fast); repeat --override per key
python run.py --mode final_eval --checkpoint <adapter> \
  --override data.eval_paradigm=direct \
  --override hardware.use_vllm=true \
  --override hardware.vllm_gpu_memory_utilization=0.85

# Re-score an existing trajectories file with the official scorer
python nestful_official_score.py --trajectories <run>/final_eval_trajectories.jsonl --no-win-rate

# Re-score old curriculum predictions offline (ablation)
python curricullum/evaluation/rescore_official.py --no-win-rate
```

Edit paths and model in `config.yaml`. Details: `IMPLEMENTATION_CHECK.md` and
`docs/AUDIT.md`.
