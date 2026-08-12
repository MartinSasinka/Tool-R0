#!/usr/bin/env bash
# Final temp-0 NESTFUL eval (v5-compatible env interface) for nestful_mtgrpo_minimal.
#
# Full NESTFUL set, temperature=0.0, top_p=1.0, one rollout, ReAct, IBM executor,
# official scorer. Decoding knobs are always forced via --override.
#
# Example (continue350 @ last adapter, 4× GPU vLLM TP):
#   cd /workspace/Tool-R0/experiments/nestful_mtgrpo_minimal
#   CKPT=outputs/qwen3_p43_profile1000_dynamic_online_continue350_enrich30/checkpoints
#   OUT=outputs/evals/p43_nestful_t0_step350
#   export WANDB_MODE=disabled
#   export USE_VLLM=1 EVAL_TP=4 VLLM_GPU_UTIL=0.85
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   LABEL=final CHECKPOINT=$CKPT/adapter_epoch_XX OUT_DIR=$OUT/step350 \
#     bash scripts/final_eval.sh
#
# Baseline (no adapter):
#   LABEL=baseline OUT_DIR=$OUT/baseline bash scripts/final_eval.sh
#
# Env:
#   LABEL        required: baseline | best | final | any tag
#   CHECKPOINT   adapter dir (required unless LABEL=baseline)
#   OUT_DIR      required output directory
#   CONFIG       yaml (default: continue350 enrich30)
#   EVAL_SET     optional nestful jsonl override
#   MAX_TASKS    0 = full set (default); >0 caps for smoke
#   USE_VLLM=1   EVAL_TP=4  VLLM_GPU_UTIL=0.85
#   PYTHON       default python3
set -euo pipefail

MINIMAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
CFG="${CONFIG:-$MINIMAL/configs/qwen3_p43_profile1000_dynamic_online_continue350_enrich30.yaml}"

LABEL="${LABEL:?set LABEL (e.g. baseline / best / final)}"
OUT_DIR="${OUT_DIR:?set OUT_DIR for the eval outputs}"
MAX_TASKS="${MAX_TASKS:-0}"
USE_VLLM="${USE_VLLM:-1}"
EVAL_TP="${EVAL_TP:-4}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

if [[ "$LABEL" == "baseline" && -n "${CHECKPOINT:-}" ]]; then
  echo "[final_eval] ERROR: LABEL=baseline conflicts with CHECKPOINT=$CHECKPOINT" >&2
  exit 1
fi
if [[ "$LABEL" != "baseline" && -z "${CHECKPOINT:-}" ]]; then
  echo "[final_eval] ERROR: LABEL=$LABEL requires CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "[final_eval] ERROR: missing config $CFG" >&2
  exit 1
fi
if [[ -n "${CHECKPOINT:-}" && ! -f "$CHECKPOINT/adapter_config.json" ]]; then
  echo "[final_eval] ERROR: not a LoRA adapter: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "──────────────────────────────────────────────────────────────"
echo "[final_eval] temp-0 NESTFUL — label='$LABEL'"
echo "[final_eval] CONFIG=$CFG"
echo "[final_eval] CHECKPOINT=${CHECKPOINT:-<base model>}"
echo "[final_eval] OUT_DIR=$OUT_DIR"
echo "[final_eval] USE_VLLM=$USE_VLLM EVAL_TP=$EVAL_TP VLLM_GPU_UTIL=$VLLM_GPU_UTIL"
echo "[final_eval] MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288} MAX_TASKS=$MAX_TASKS EVAL_SET=${EVAL_SET:-<config default>}"
echo "──────────────────────────────────────────────────────────────"

# Context: train yamls often set generation.max_model_length=6144 while
# max_new_tokens_eval=2560 → prompt budget only 3584. Prior v5 evals used
# nestful_mtgrpo_partial token_budget up to vllm_max_model_length=12288.
# Force that here so NESTFUL long prompts are not skipped as overflow.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

ARGS=(
  "$MINIMAL/run.py" --mode final_eval
  --config "$CFG"
  --override "experiment.output_dir=$OUT_DIR"
  --override generation.temperature=0.0
  --override generation.top_p=1.0
  --override data.num_eval_rollouts=1
  --override data.eval_paradigm=react
  --override "generation.max_model_length=$MAX_MODEL_LEN"
)

# Train yamls often cap max_eval_tasks (e.g. 64) — force full set when MAX_TASKS=0.
if [[ "$MAX_TASKS" == "0" ]]; then
  ARGS+=(--override data.max_eval_tasks=null)
else
  ARGS+=(--override "data.max_eval_tasks=$MAX_TASKS")
fi

if [[ -n "${EVAL_SET:-}" ]]; then
  if [[ ! -f "$EVAL_SET" ]]; then
    echo "[final_eval] ERROR: EVAL_SET not found: $EVAL_SET" >&2
    exit 1
  fi
  EVAL_SET_ABS="$(cd "$(dirname "$EVAL_SET")" && pwd)/$(basename "$EVAL_SET")"
  ARGS+=(--override "paths.full_nestful_jsonl=$EVAL_SET_ABS")
else
  EVAL_SET_ABS=""
fi

if [[ "$USE_VLLM" == "1" ]]; then
  ARGS+=(--override hardware.use_vllm=true)
  ARGS+=(--override "hardware.vllm_tensor_parallel_size=$EVAL_TP")
  ARGS+=(--override "hardware.vllm_gpu_memory_utilization=$VLLM_GPU_UTIL")
fi

CKPT_ABS=""
if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT_ABS="$(cd "$CHECKPOINT" && pwd)"
  ARGS+=(--checkpoint "$CKPT_ABS")
else
  ARGS+=(--override model.lora_adapter=null)
fi

CMD_JSON="$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' -- "${ARGS[@]}")"
export LABEL OUT_DIR MAX_TASKS CKPT_ABS EVAL_SET_ABS CMD_JSON
"$PY" - <<'PY'
import json, os
from datetime import datetime, timezone
manifest = {
    "label": os.environ["LABEL"],
    "checkpoint": os.environ.get("CKPT_ABS") or None,
    "eval_set": os.environ.get("EVAL_SET_ABS") or "config default",
    "decoding": {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_rollouts": 1,
        "paradigm": "react",
    },
    "max_tasks": None if os.environ.get("MAX_TASKS", "0") == "0"
                 else int(os.environ["MAX_TASKS"]),
    "command": json.loads(os.environ["CMD_JSON"]),
    "created_at": datetime.now(timezone.utc).isoformat(),
}
out = os.path.join(os.environ["OUT_DIR"], "eval_manifest.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)
print("[final_eval] wrote", out)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
PY

cd "$MINIMAL"
unset SYNTHETIC_TOOLS_DIR || true
"$PY" "${ARGS[@]}"

echo "──────────────────────────────────────────────────────────────"
echo "[final_eval] done: $OUT_DIR/metrics_official.json"
echo "──────────────────────────────────────────────────────────────"
