# RunPod resume — NESTFUL MT-GRPO (4× RTX 4090 / 24 GB)

Step-by-step to resume a curriculum run on a **fresh** RunPod from the last
finished checkpoint `stage_2/checkpoints/adapter_epoch_1` (continue **stage 2
from epoch 2** onward, then stages 3 and 4 normally from epoch 1).

> **Secrets:** never write real tokens into files or git. Use the placeholders
> below and export them only into the live shell (or a gitignored `.env`).

---

## 0. Pod / GPU assumptions

- 4× RTX 4090 (24 GB each), ~31 GB RAM, 8 vCPU.
- Training learner runs on **1 GPU**; evaluation runs **vLLM tensor-parallel on
  all 4 GPUs** (TP=4 auto). Token budgets in `config.yaml` are already tuned for
  24 GB cards, and tool-observation truncation + prompt-overflow guards prevent
  context-window crashes.

---

## 1. Download the project archive from Google Drive

```bash
cd /workspace
pip install -q gdown

# Replace <FILE_ID> with your Drive file id (the long id in the share URL).
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O nestful_mtgrpo_minimal.tar
```

## 2. Extract

```bash
cd /workspace
tar -xf nestful_mtgrpo_minimal.tar      # add -z if it is a .tar.gz
# You should now have /workspace/nestful_mtgrpo_minimal
ls /workspace/nestful_mtgrpo_minimal
```

**If shell scripts fail with `$'\r': command not found` or `set: pipefail` invalid option**
(the tar was packed on Windows), fix line endings once:

```bash
# fix all .sh scripts under both experiment folders
find /workspace/nestful_mtgrpo_minimal /workspace/nestful_mtgrpo_partial -name '*.sh' \
  -exec sed -i 's/\r$//' {} +
```

## 3. Install dependencies (check + install)

`install_deps.sh` keeps the pod's existing CUDA `torch`, adds everything else
plus a vLLM build matching it, and verifies the whole stack at the end.

```bash
cd /workspace/nestful_mtgrpo_minimal
bash install_deps.sh
```

If model download fails with `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER=1`, either
install the fast-download helper or disable it:

```bash
pip install hf_transfer          # preferred on RunPod
# OR: unset HF_HUB_ENABLE_HF_TRANSFER
```

For W&B logging (recommended during training):

```bash
pip install wandb
export WANDB_API_KEY="<your_wandb_key>"
export WANDB_PROJECT="nestful-mtgrpo-partial"   # or nestful-mtgrpo for strict
```

> The runner also runs a fast dependency **preflight** at startup: if anything is
> missing it auto-runs `install_deps.sh` (`AUTO_INSTALL_DEPS=1`, the default).
> Set `CHECK_DEPS=0` to skip the check, or `AUTO_INSTALL_DEPS=0` to only warn.

## 4. Set secrets as environment variables (placeholders only)

Option A — export directly into the shell:

```bash
export WANDB_API_KEY="<your_wandb_key>"
export HF_TOKEN="<your_hf_token>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_PROJECT="nestful-mtgrpo"
```

Option B — use a gitignored `.env` (template provided as `.env.example`):

```bash
cp .env.example .env          # then edit .env and fill the <...> placeholders
set -a; source .env; set +a   # export everything from .env
```

## 5. Verify the checkpoint exists

```bash
cd /workspace/nestful_mtgrpo_minimal
ls -lah outputs/curriculum/stage_2/checkpoints/adapter_epoch_1
```

You should see an adapter dir (e.g. `adapter_config.json`,
`adapter_model.safetensors`). If it is missing, the resume will hard-fail with a
clear error instead of silently restarting from the base model.

## 6. Start the run inside `tmux`

```bash
tmux new -s nestful
# (inside the tmux session)

export WANDB_API_KEY="<your_wandb_key>"
export HF_TOKEN="<your_hf_token>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_PROJECT="nestful-mtgrpo"

cd /workspace/nestful_mtgrpo_minimal

CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_VLLM=1 \
PROFILE=curriculum \
STAGES="2 3 4" \
START_EPOCH=2 \
CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_1 \
MAX_EPOCHS_PER_STAGE=4 \
RUN_FINAL_EVAL=1 \
bash run_curriculum.sh
```

What this does:
- `START_EPOCH=2` applies **only to the first stage in `STAGES`** (stage 2), so
  stage 2 runs epochs 2→4; stages 3 and 4 then run normally from epoch 1.
- `CHECKPOINT_IN=...adapter_epoch_1` seeds the resume (used directly if the
  previous-epoch adapter dir is absent on this fresh pod).
- `RUN_FINAL_EVAL=1` runs the full-NESTFUL final eval (ReAct + Direct, TP=4)
  after training.

Detach/reattach tmux (the run keeps going after you close your laptop):

```bash
# detach:           press  Ctrl-b  then  d
tmux attach -t nestful     # reattach later
```

---

## 6b. (Optional) Faster training — data-parallel rollouts across GPUs

By default the HF QLoRA learner **and** the vLLM rollout engine share **one** GPU
(GPU 0); the other GPUs sit idle during the train phase and are only used (TP=4)
during evaluation. Rollout generation (`NUM_GENERATIONS` episodes per task) is the
dominant cost and is independent of the learner, so it can be parallelised.

Set `ROLLOUT_DP_GPUS` to run **one vLLM engine per GPU** for rollouts while the
learner stays on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_VLLM=1 \
ROLLOUT_DP_GPUS="1,2,3" \      # rollout workers on GPUs 1,2,3 (learner on GPU 0)
PROFILE=curriculum \
STAGES="2 3 4" \
START_EPOCH=2 \
CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_1 \
MAX_EPOCHS_PER_STAGE=4 \
RUN_FINAL_EVAL=1 \
bash run_curriculum.sh
```

Notes:
- Expected speedup of the train phase is roughly **proportional to the number of
  rollout GPUs** (e.g. ~3× with 3 workers), since each worker owns its GPU
  (`tensor_parallel_size=1`).
- The updated adapter is hot-synced to every worker after each epoch
  (look for `dp_pool_adapter_synced` in `train.log`).
- This is **opt-in and safe**: if the pool fails to start it logs a warning and
  falls back to the single-engine path, so a run never silently dies.
- Works identically for the **partial** experiment — the workers pick the partial
  reward from `reward.train_policy`, so partial-credit training is preserved.
- Evaluation phases are unchanged (they already use all GPUs at TP=4).
- Verify utilisation with `nvidia-smi` during the train phase: you should now see
  a `VLLM::EngineCore` process on **each** of GPUs 1–3 plus the learner on GPU 0.

---

## Sanity checks

Run these before/while launching (none of them print secrets):

```bash
# 1) GPUs visible and healthy
nvidia-smi

# 2) Unit tests pass (no GPU needed)
python -m pytest tests -q

# 3) The resume checkpoint exists
ls -lah outputs/curriculum/stage_2/checkpoints/adapter_epoch_1

# 4) Env vars are SET without revealing their values
for v in WANDB_API_KEY HF_TOKEN WANDB_PROJECT; do
  if [ -n "${!v}" ]; then echo "$v: set (${#v} chars name; value hidden)"; else echo "$v: MISSING"; fi
done

# 4b) Confirm W&B auth works (prints username, not the key)
python -c "import wandb; print('wandb user:', wandb.Api().viewer.username)" 2>/dev/null \
  || echo "wandb not logged in / offline — training still runs, logging may be disabled"
```

---

## Quick reference — relevant env vars

| Var | Meaning | Example |
|-----|---------|---------|
| `PROFILE` | `pilot` (small) or `curriculum` (full) | `curriculum` |
| `STAGES` | space-separated stages to run | `"2 3 4"` |
| `START_EPOCH` | epoch to start the FIRST stage at (resume) | `2` |
| `CHECKPOINT_IN` | adapter dir to resume from | `outputs/curriculum/stage_2/checkpoints/adapter_epoch_1` |
| `MAX_EPOCHS_PER_STAGE` | hard cap on epochs per stage | `4` |
| `USE_VLLM` | `1` to use vLLM (recommended) | `1` |
| `RUN_FINAL_EVAL` | `1` to run full-NESTFUL final eval after training | `1` |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0,1,2,3` |
| `ROLLOUT_DP_GPUS` | data-parallel rollout worker GPUs (empty = off) | `"1,2,3"` |
| `DP_LEARNER_GPU` | GPU for the HF learner when DP is on (default: first train GPU) | `0` |
| `VLLM_GPU_UTIL_DP` | vLLM memory fraction per rollout worker | `0.85` |
| `CHECK_DEPS` / `AUTO_INSTALL_DEPS` | dep preflight + auto-install (default `1`/`1`) | `1` |

> Final-eval-only on a checkpoint (no training): see `run_all.sh`
> (`MODE=final CHECKPOINT_IN=... bash run_all.sh`).
