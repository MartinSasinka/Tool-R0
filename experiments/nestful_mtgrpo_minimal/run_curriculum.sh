#!/usr/bin/env bash
# run_curriculum.sh — Shell orchestrator for experiments/nestful_mtgrpo_minimal/
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Two profiles:                                                          │
# │    PROFILE=pilot      (default) — small safety run, 1 epoch/stage      │
# │    PROFILE=curriculum           — full overnight, multi-epoch/stage     │
# │                                                                         │
# │  Key env vars (all have defaults):                                      │
# │    STAGES            space-separated stage list,  e.g. "1 2 3 4"       │
# │    MAX_EPOCHS_PER_STAGE  hard cap; always advances  (default 4)         │
# │    EPOCHS            starting / pilot epoch count  (default 1/3)       │
# │    ADVANCE_THRESHOLD strict_gold_trace_pass to advance early (0.50)    │
# │    PLATEAU_PATIENCE  epochs w/o improvement → advance anyway  (2)      │
# │    GATE_MODE         warn | stop  (warn = never abort overnight run)    │
# │    USE_VLLM=1        enable vLLM for generation (opt-in)                │
# │    USE_FLASH_ATTENTION=1  enable FA2 on HF model                        │
# │    VLLM_GPU_UTIL_TRAIN   GPU fraction for vLLM during train  (0.45)    │
# │    VLLM_GPU_UTIL_EVAL    GPU fraction for vLLM during eval   (0.85)    │
# │    MAX_TRAIN_TASKS   null = all tasks; set number to cap     (null)     │
# │    MAX_EVAL_TASKS    null = all tasks; set number to cap     (null)     │
# │    STOP_ON_FAIL      1 = stop on gate fail (overridden by GATE_MODE)   │
# │    DRY_RUN=1         print commands without executing                   │
# │    RUN_FINAL_EVAL=1  run final_eval on baseline + last checkpoint      │
# │    ONLY_FINAL_EVAL=1 skip training; final_eval only (uses CHECKPOINT_IN)│
# │    FINAL_EVAL_PARADIGM  react|direct|both  (Table 2 / Table 1) (both)  │
# │    FINAL_EVAL_BASELINE  1 = also eval base model (no adapter)    (1)   │
# │    FINAL_EVAL_NUM_ICL   in-context examples for direct           (1)   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Examples:
#   # Dry-run to inspect commands
#   DRY_RUN=1 USE_VLLM=1 PROFILE=curriculum STAGES="1 2 3 4" bash run_curriculum.sh
#
#   # Full overnight curriculum, all tasks, warn-mode gate
#   CUDA_VISIBLE_DEVICES=2 USE_VLLM=1 PROFILE=curriculum STAGES="1 2 3 4" \
#     bash experiments/nestful_mtgrpo_minimal/run_curriculum.sh
#
#   # Pilot: 16 tasks, 1 epoch
#   CUDA_VISIBLE_DEVICES=2 PROFILE=pilot STAGES="3" bash run_curriculum.sh

set -euo pipefail

# Resolve artifact root — works whether this folder is inside Tool-R0 or a standalone repo.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ══════════════════════════════════════════════════════════════════════════════
#  Python interpreter
# ══════════════════════════════════════════════════════════════════════════════
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="python3"

# Entry point. Override with RUN_PY to reuse this curriculum loop from a sibling
# experiment (e.g. nestful_mtgrpo_partial sets RUN_PY=.../nestful_mtgrpo_partial/run.py).
# Default is unchanged, so existing strict runs behave identically.
RUN_PY="${RUN_PY:-$ROOT/run.py}"

# ══════════════════════════════════════════════════════════════════════════════
#  Profile defaults — user env vars always win
# ══════════════════════════════════════════════════════════════════════════════
PROFILE="${PROFILE:-pilot}"

case "$PROFILE" in
  pilot)
    # Small, fast, safe — verify nothing is broken before a real run.
    STAGES="${STAGES:-3}"
    MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-16}"
    MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-32}"
    EPOCHS="${EPOCHS:-1}"
    MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-1}"
    NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
    ;;
  curriculum)
    # Full overnight staged curriculum: all tasks, multi-epoch with smart gate.
    STAGES="${STAGES:-1 2 3 4}"
    MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-}"         # empty = null = all tasks
    MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-}"           # empty = null = all tasks
    EPOCHS="${EPOCHS:-3}"
    MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-4}"
    NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
    ;;
  stabilized_curriculum)
    # Stabilized run (reward UNCHANGED): clean data + mixed replay + lower LR +
    # higher KL + per-epoch validation ReAct Win + early stopping. See
    # docs/STABILIZED_CURRICULUM_PLAN.md.
    STAGES="${STAGES:-1 2 3 4}"
    MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-}"
    MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-}"
    EPOCHS="${EPOCHS:-3}"
    MAX_EPOCHS_PER_STAGE="${MAX_EPOCHS_PER_STAGE:-4}"
    NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
    # Clean/repaired curriculum data (produced by experiments/data/prepare_clean_training_set.py).
    DATA_BASE="${DATA_BASE:-$ROOT/data/clean_curriculum}"
    # Stabilized knobs (user env always wins via :- defaults below).
    CURRICULUM_MIXED_REPLAY="${CURRICULUM_MIXED_REPLAY:-1}"
    EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-1}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-react_win_rate}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-1}"
    EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.005}"
    # Reward UNCHANGED — only optimisation stability: LR 0.5x, KL 2x of config defaults.
    STABILIZED_LR="${STABILIZED_LR:-0.5e-6}"
    STABILIZED_KL="${STABILIZED_KL:-0.04}"
    ;;
  *)
    echo "[run_curriculum] ERROR: Unknown PROFILE='$PROFILE'. Use: pilot | curriculum | stabilized_curriculum" >&2
    exit 1
    ;;
esac

# ── Common settings (profile-independent, all overridable) ───────────────────
CONFIG="${CONFIG:-$ROOT/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/curriculum}"
CHECKPOINT_IN="${CHECKPOINT_IN:-}"
# START_EPOCH: skip to a specific epoch within the FIRST stage in STAGES (for resume).
# e.g. START_EPOCH=3 with STAGES="2 3 4" resumes at stage 2 epoch 3.
# The script automatically picks up adapter_epoch_$((START_EPOCH-1)) as the checkpoint.
START_EPOCH="${START_EPOCH:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-0}"
# ONLY_FINAL_EVAL=1 — skip all training and just run final_eval (baseline + the
# checkpoint given via CHECKPOINT_IN/FINAL_CHECKPOINT). Forces RUN_FINAL_EVAL=1.
ONLY_FINAL_EVAL="${ONLY_FINAL_EVAL:-0}"
if [ "$ONLY_FINAL_EVAL" = "1" ]; then
    STAGES=""
    RUN_FINAL_EVAL=1
fi
DATA_BASE="${DATA_BASE:-$ROOT/data/filtered_toolr0_synthetic}"

# Normalize run paths to absolute (resume / symlink safety).
if [ -d "$OUTPUT_ROOT" ]; then
    OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
elif [ -n "$OUTPUT_ROOT" ]; then
    OUTPUT_ROOT="$(cd "$(dirname "$OUTPUT_ROOT")" 2>/dev/null && pwd)/$(basename "$OUTPUT_ROOT")"
fi
if [ -d "$DATA_BASE" ]; then
    DATA_BASE="$(cd "$DATA_BASE" && pwd)"
fi

# Auto-repair v3 mixed-replay symlinks when resuming via run_curriculum.sh directly.
_v3_curr="$ROOT/../nestful_synthetic_curriculum_v3/outputs/curriculum_v3"
if [ -d "$_v3_curr" ] && [ -n "$DATA_BASE" ]; then
    _need_repair=0
    for _n in 1 2 3 4; do
        _f="$DATA_BASE/epoch_${_n}_${_n}call.jsonl"
        if [ ! -f "$_f" ]; then _need_repair=1; break; fi
    done
    if [ "$_need_repair" = "1" ]; then
        echo "[data] repairing DATA_BASE symlinks from $_v3_curr ..."
        mkdir -p "$DATA_BASE"
        ln -sf "$_v3_curr/stage1_linear_simple.jsonl" "$DATA_BASE/epoch_1_1call.jsonl"
        ln -sf "$_v3_curr/stage2_reference_reuse.jsonl" "$DATA_BASE/epoch_2_2call.jsonl"
        ln -sf "$_v3_curr/stage3_structural_motifs.jsonl" "$DATA_BASE/epoch_3_3call.jsonl"
        ln -sf "$_v3_curr/stage4_nestful_like_mixed.jsonl" "$DATA_BASE/epoch_4_4call.jsonl"
        DATA_BASE="$(cd "$DATA_BASE" && pwd)"
    fi
fi

# ── Stabilized-curriculum knobs (safe defaults; only active when enabled) ──────
#   INIT_FROM                 baseline | checkpoint  (checkpoint requires CHECKPOINT_IN)
#   CURRICULUM_MIXED_REPLAY   1 = stage N trains on a weighted mix of stages 1..N
#   CURRICULUM_REPLAY_WEIGHTS CSV weights, e.g. "1.0,1.0,1.0,1.0" (uniform if empty)
#   EVAL_EVERY_EPOCH          1 = run val_eval (ReAct Win) after each epoch
#   EARLY_STOP_METRIC         metric for early stopping (react_win_rate)
#   EARLY_STOP_PATIENCE       #evals without >= min_delta improvement before stopping
#   EARLY_STOP_MIN_DELTA      minimum ReAct Win improvement to reset patience
#   VAL_SUBSET_SIZE           0 = full NESTFUL; >0 = deterministic subset (saved ids)
#   STABILIZED_LR / STABILIZED_KL  optional training.learning_rate / kl_beta overrides
INIT_FROM="${INIT_FROM:-baseline}"
CURRICULUM_MIXED_REPLAY="${CURRICULUM_MIXED_REPLAY:-0}"
CURRICULUM_REPLAY_WEIGHTS="${CURRICULUM_REPLAY_WEIGHTS:-}"
EVAL_EVERY_EPOCH="${EVAL_EVERY_EPOCH:-0}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-react_win_rate}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-1}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.005}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-0}"
STABILIZED_LR="${STABILIZED_LR:-}"
STABILIZED_KL="${STABILIZED_KL:-}"
# Best-by-validation-ReAct-Win adapter (copied across stages; final selection).
BEST_REACT_WIN_ADAPTER="${BEST_REACT_WIN_ADAPTER:-$OUTPUT_ROOT/best_react_win_adapter}"

# ── Regression guard (never crown a checkpoint worse than the base model) ─────
#   REGRESSION_GUARD=1        enable the guard (default on)
#   REGRESSION_BASELINE_WIN   dev ReAct Win of the base model; auto-measured on
#                             the dev split before stage 1 if left empty
#   REGRESSION_MARGIN         require Win >= baseline + margin to save as best
#   REGRESSION_EARLY_ABORT    1 = abort the whole run if a full stage stays below
#                             baseline (default 0 = keep training, just don't crown)
REGRESSION_GUARD="${REGRESSION_GUARD:-1}"
REGRESSION_BASELINE_WIN="${REGRESSION_BASELINE_WIN:-}"
REGRESSION_MARGIN="${REGRESSION_MARGIN:-0.0}"
REGRESSION_EARLY_ABORT="${REGRESSION_EARLY_ABORT:-0}"
# Disabling the regression guard requires an explicit double opt-in (audit Bug 5).
ALLOW_NO_REGRESSION_GUARD="${ALLOW_NO_REGRESSION_GUARD:-0}"
# Reward dispatch: NEVER silently fall back to the strict reward (audit Bug 1).
ALLOW_STRICT_REWARD_FALLBACK="${ALLOW_STRICT_REWARD_FALLBACK:-0}"
export ALLOW_STRICT_REWARD_FALLBACK
# STAGE_GATES=1 — evaluate hard stage-advancement gates (check_stage_gates.py)
# after each stage; a failing stage STOPS the run (exit 4).
STAGE_GATES="${STAGE_GATES:-0}"

if [ "$REGRESSION_GUARD" != "1" ]; then
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "  ⚠⚠⚠  REGRESSION_GUARD IS DISABLED  ⚠⚠⚠" >&2
    echo "  Checkpoints WORSE than the baseline model can be crowned best." >&2
    echo "  This is how the previous pilot overwrote its best adapter with" >&2
    echo "  a regressed stage-2 checkpoint." >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    if [ "$ALLOW_NO_REGRESSION_GUARD" != "1" ]; then
        echo "[run_curriculum] ABORT: set ALLOW_NO_REGRESSION_GUARD=1 to run without" >&2
        echo "  the regression guard (NOT recommended), or set REGRESSION_GUARD=1." >&2
        exit 1
    fi
    echo "[run_curriculum] continuing WITHOUT regression guard (ALLOW_NO_REGRESSION_GUARD=1)" >&2
fi

# Gate behaviour:
#   GATE_MODE=warn  — log failures, ALWAYS continue (safe for overnight runs)
#   GATE_MODE=stop  — hard stop on gate failure (for careful interactive runs)
GATE_MODE="${GATE_MODE:-warn}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"   # default 0 so warn mode is natural
# Gate thresholds
ADVANCE_THRESHOLD="${ADVANCE_THRESHOLD:-0.50}"   # strict_gold_trace_pass rate → advance early
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-2}"        # epochs without improvement → advance
MAX_CLIPPED_COMPLETION_RATE="${MAX_CLIPPED_COMPLETION_RATE:-0.25}"

# vLLM settings
USE_VLLM="${USE_VLLM:-0}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-0}"
VLLM_GPU_UTIL_TRAIN="${VLLM_GPU_UTIL_TRAIN:-0.45}"
VLLM_GPU_UTIL_EVAL="${VLLM_GPU_UTIL_EVAL:-0.85}"

# ── GPU split (4×24GB strategy) ───────────────────────────────────────────────
# The learner (HF QLoRA) trains on ONE GPU; evaluation runs vLLM tensor-parallel
# across all available GPUs (eval does not load the HF model, so TP=4 is clean).
# Each phase is a separate `python run.py` process so it can be pinned to a
# different CUDA_VISIBLE_DEVICES and tensor_parallel_size.
#   TRAIN_GPUS     GPUs visible during train  (default: first detected GPU)
#   EVAL_GPUS      GPUs visible during eval   (default: all detected GPUs)
#   VLLM_TP_TRAIN  tensor-parallel size, train (default: 1)
#   VLLM_TP_EVAL   tensor-parallel size, eval  (default: #EVAL_GPUS; auto-clamped
#                  to a valid divisor of the model KV heads inside vllm_generate)
_detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES"
        return
    fi
    local n
    n="$("$PYTHON" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
    [ -z "$n" ] && n=1
    "$PYTHON" -c "print(','.join(str(i) for i in range($n)))"
}
_count_csv() { echo "$1" | awk -F, '{print NF}'; }

ALL_GPUS="$(_detect_gpus)"
TRAIN_GPUS="${TRAIN_GPUS:-$(echo "$ALL_GPUS" | cut -d, -f1)}"
EVAL_GPUS="${EVAL_GPUS:-$ALL_GPUS}"
VLLM_TP_TRAIN="${VLLM_TP_TRAIN:-1}"
VLLM_TP_EVAL="${VLLM_TP_EVAL:-$(_count_csv "$EVAL_GPUS")}"

# ── Data-parallel rollouts (opt-in, TRAIN-phase multi-GPU speedup) ────────────
#   ROLLOUT_DP_GPUS   CSV of GPU ids that run vLLM rollout workers, e.g. "1,2,3".
#                     Empty = OFF (single in-process engine, unchanged). When set,
#                     the HF learner is pinned to DP_LEARNER_GPU and the rollouts
#                     run one vLLM engine per listed GPU.
#   DP_LEARNER_GPU    GPU id for the HF QLoRA learner (default: first of TRAIN_GPUS).
#   VLLM_GPU_UTIL_DP  memory fraction per rollout worker (default: 0.85).
ROLLOUT_DP_GPUS="${ROLLOUT_DP_GPUS:-}"
DP_LEARNER_GPU="${DP_LEARNER_GPU:-$(echo "$TRAIN_GPUS" | cut -d, -f1)}"
VLLM_GPU_UTIL_DP="${VLLM_GPU_UTIL_DP:-0.85}"

# Phase-scoped CUDA_VISIBLE_DEVICES, set by _build_vllm_overrides before each run.
_PHASE_GPUS=""

# W&B settings (all optional — W&B is disabled unless WANDB_PROJECT is set)
# WANDB_API_KEY is expected to be already exported in the environment.
# WANDB_PROJECT is required to enable logging; e.g. export WANDB_PROJECT=nestful-mtgrpo
if [ -n "${WANDB_PROJECT:-}" ] && [ -z "${WANDB_RUN_GROUP:-}" ]; then
    # Auto-generate a group ID for this curriculum run so all stages are grouped together.
    WANDB_RUN_GROUP="curriculum-$("$PYTHON" -c 'import datetime; print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))')"
    export WANDB_RUN_GROUP
    echo "  [wandb] WANDB_RUN_GROUP = $WANDB_RUN_GROUP"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build vLLM --override args
# ══════════════════════════════════════════════════════════════════════════════
VLLM_OVERRIDES=()
_build_vllm_overrides() {
    local mode="$1"
    VLLM_OVERRIDES=()
    # Pin the GPU set for this phase (applies to both HF and vLLM paths).
    local tp
    if [ "$mode" = "train" ]; then
        # With data-parallel rollouts the train process must SEE the learner GPU
        # (pinned to visible index 0) AND every rollout-worker GPU.
        if [ -n "$ROLLOUT_DP_GPUS" ]; then
            _PHASE_GPUS="$DP_LEARNER_GPU,$ROLLOUT_DP_GPUS"
        else
            _PHASE_GPUS="$TRAIN_GPUS"
        fi
        tp="$VLLM_TP_TRAIN"
    else
        _PHASE_GPUS="$EVAL_GPUS"
        tp="$VLLM_TP_EVAL"
    fi
    if [ "$USE_VLLM" != "1" ]; then
        return
    fi
    local util="$VLLM_GPU_UTIL_EVAL"
    if [ "$mode" = "train" ]; then
        util="$VLLM_GPU_UTIL_TRAIN"
    fi
    VLLM_OVERRIDES=(
        --override "hardware.use_vllm=true"
        --override "hardware.vllm_gpu_memory_utilization=$util"
        --override "hardware.vllm_tensor_parallel_size=$tp"
    )
    # TRAIN-only: enable the data-parallel rollout pool (one engine per worker GPU).
    if [ "$mode" = "train" ] && [ -n "$ROLLOUT_DP_GPUS" ]; then
        VLLM_OVERRIDES+=(
            --override "hardware.rollout_data_parallel_gpus=$ROLLOUT_DP_GPUS"
            --override "hardware.vllm_gpu_memory_utilization_dp=$VLLM_GPU_UTIL_DP"
        )
    fi
    if [ "$USE_FLASH_ATTENTION" = "1" ]; then
        VLLM_OVERRIDES+=(--override "hardware.use_flash_attention=true")
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: run command with tee logging (or DRY_RUN print)
# ══════════════════════════════════════════════════════════════════════════════
_run_logged() {
    local logfile="$1"; shift
    local prefix=()
    if [ -n "${_PHASE_GPUS:-}" ]; then
        prefix=(env "CUDA_VISIBLE_DEVICES=$_PHASE_GPUS")
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] ${prefix[*]} $* 2>&1 | tee $logfile"
        return 0
    fi
    mkdir -p "$(dirname "$logfile")"
    # pipefail is on: propagate the *command* exit code, not tee's (PIPESTATUS[0]).
    "${prefix[@]}" "$@" 2>&1 | tee "$logfile"
    return "${PIPESTATUS[0]}"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: read single key from JSON
# ══════════════════════════════════════════════════════════════════════════════
_json_get() {
    "$PYTHON" - "$1" "$2" "$3" <<'PYEOF'
import json, sys
file, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(file) as f:
        d = json.load(f)
    v = d.get(key)
    if v is None: print(default)
    elif isinstance(v, bool): print(str(v).lower())
    else: print(v)
except Exception:
    print(default)
PYEOF
}

# Return 0 when $1 is a directory containing adapter_config.json.
_adapter_valid() {
    [ -n "${1:-}" ] && [ -f "$1/adapter_config.json" ]
}

# Resolve a checkpoint path: prefer $1 if valid, else newest adapter_epoch_* under $2.
_resolve_checkpoint() {
    local preferred="$1" ckpt_root="$2"
    if _adapter_valid "$preferred"; then
        echo "$preferred"
        return 0
    fi
    local best="" best_n=0
    local d n
    for d in "$ckpt_root"/adapter_epoch_*; do
        [ -d "$d" ] || continue
        _adapter_valid "$d" || continue
        n="${d##*adapter_epoch_}"
        if [ "$n" -gt "$best_n" ] 2>/dev/null; then
            best_n="$n"
            best="$d"
        fi
    done
    if [ -n "$best" ]; then
        echo "$best"
        return 0
    fi
    return 1
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: float comparison a <= b
# ══════════════════════════════════════════════════════════════════════════════
_float_le() {
    "$PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$1" "$2"
}

_float_ge() {
    "$PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$1" "$2"
}

# _improved <new> <best> <min_delta> → exit 0 if (new - best) >= min_delta.
_improved() {
    "$PYTHON" -c "import sys; a,b,d=map(float,sys.argv[1:4]); sys.exit(0 if (a-b)>=d else 1)" "$1" "$2" "$3"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build data override args (handles empty MAX_TRAIN/EVAL_TASKS = null)
# ══════════════════════════════════════════════════════════════════════════════
_data_overrides_train() {
    # Append to $@; empty MAX_TRAIN_TASKS → pass null (all tasks)
    if [ -n "$MAX_TRAIN_TASKS" ]; then
        echo "--override data.max_train_tasks=$MAX_TRAIN_TASKS"
    else
        echo "--override data.max_train_tasks=null"
    fi
}
_data_overrides_eval() {
    if [ -n "$MAX_EVAL_TASKS" ]; then
        echo "--override data.max_eval_tasks=$MAX_EVAL_TASKS"
    else
        echo "--override data.max_eval_tasks=null"
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: dependency preflight (fast import check + optional auto-install)
# ══════════════════════════════════════════════════════════════════════════════
#   CHECK_DEPS=1         run the import check (default 1; set 0 to skip entirely)
#   AUTO_INSTALL_DEPS=1  if the check fails, run install_deps.sh automatically (default 1)
# The check is a no-op cost when everything is already installed.
_fix_crlf() {
    # Archives packed on Windows often carry CRLF; Linux bash breaks on \r in paths.
    local f="$1"
    if [ -f "$f" ] && grep -q $'\r' "$f" 2>/dev/null; then
        echo "[preflight] fixing CRLF line endings in $f ..." >&2
        sed -i 's/\r$//' "$f"
    fi
}

_preflight_deps() {
    [ "${CHECK_DEPS:-1}" = "1" ] || return 0
    if "$PYTHON" - <<'PYEOF'
import importlib, sys
missing = []
for m in ("torch", "transformers", "peft", "trl", "accelerate",
          "bitsandbytes", "yaml", "sklearn", "jsonlines"):
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
# vLLM is only required when USE_VLLM=1.
import os
if os.environ.get("USE_VLLM") == "1":
    try:
        importlib.import_module("vllm")
    except Exception:
        missing.append("vllm")
# W&B logging when the user enabled it via env.
if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_PROJECT"):
    try:
        importlib.import_module("wandb")
    except Exception:
        missing.append("wandb")
# RunPod images often set HF_HUB_ENABLE_HF_TRANSFER=1 without installing the pkg.
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").lower() in ("1", "true", "yes"):
    try:
        importlib.import_module("hf_transfer")
    except Exception:
        missing.append("hf_transfer")
if missing:
    print("MISSING: " + ",".join(missing), file=sys.stderr)
    sys.exit(1)
print("[preflight] all required dependencies present")
PYEOF
    then
        return 0
    fi
    echo "[preflight] missing dependencies detected." >&2
    if [ "${AUTO_INSTALL_DEPS:-1}" = "1" ] && [ -f "$ROOT/install_deps.sh" ]; then
        _fix_crlf "$ROOT/install_deps.sh"
        echo "[preflight] running install_deps.sh (AUTO_INSTALL_DEPS=1) ..." >&2
        PYTHON="$PYTHON" bash "$ROOT/install_deps.sh"
    else
        echo "[preflight] run 'bash install_deps.sh' first, or set AUTO_INSTALL_DEPS=1." >&2
        exit 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Print banner
# ══════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════"
echo "  nestful_mtgrpo_minimal — curriculum runner"
echo "────────────────────────────────────────────────────────────────"
printf "  %-26s %s\n" "PROFILE"                "$PROFILE"
printf "  %-26s %s\n" "STAGES"                 "$STAGES"
printf "  %-26s %s\n" "MAX_TRAIN_TASKS"        "${MAX_TRAIN_TASKS:-null (all)}"
printf "  %-26s %s\n" "MAX_EVAL_TASKS"         "${MAX_EVAL_TASKS:-null (all)}"
printf "  %-26s %s\n" "EPOCHS (start)"         "$EPOCHS"
printf "  %-26s %s\n" "MAX_EPOCHS_PER_STAGE"   "$MAX_EPOCHS_PER_STAGE"
printf "  %-26s %s\n" "NUM_GENERATIONS"        "$NUM_GENERATIONS"
printf "  %-26s %s\n" "ADVANCE_THRESHOLD"      "$ADVANCE_THRESHOLD"
printf "  %-26s %s\n" "PLATEAU_PATIENCE"       "$PLATEAU_PATIENCE"
printf "  %-26s %s\n" "GATE_MODE"              "$GATE_MODE"
printf "  %-26s %s\n" "USE_VLLM"               "$USE_VLLM"
printf "  %-26s %s\n" "USE_FLASH_ATTENTION"    "$USE_FLASH_ATTENTION"
printf "  %-26s %s\n" "ALL_GPUS"               "$ALL_GPUS"
printf "  %-26s %s\n" "TRAIN_GPUS / TP"        "$TRAIN_GPUS  (tp=$VLLM_TP_TRAIN)"
printf "  %-26s %s\n" "EVAL_GPUS / TP"         "$EVAL_GPUS  (tp=$VLLM_TP_EVAL)"
if [ -n "$ROLLOUT_DP_GPUS" ]; then
printf "  %-26s %s\n" "ROLLOUT_DP (train)"     "learner=$DP_LEARNER_GPU  workers=[$ROLLOUT_DP_GPUS]  util=$VLLM_GPU_UTIL_DP"
else
printf "  %-26s %s\n" "ROLLOUT_DP (train)"     "off (single in-process engine)"
fi
printf "  %-26s %s\n" "CONFIG"                 "$CONFIG"
printf "  %-26s %s\n" "OUTPUT_ROOT"            "$OUTPUT_ROOT"
printf "  %-26s %s\n" "CHECKPOINT_IN"          "${CHECKPOINT_IN:-<none>}"
printf "  %-26s %s\n" "INIT_FROM"              "$INIT_FROM"
printf "  %-26s %s\n" "DATA_BASE"              "$DATA_BASE"
printf "  %-26s %s\n" "MIXED_REPLAY"           "$CURRICULUM_MIXED_REPLAY  (weights=${CURRICULUM_REPLAY_WEIGHTS:-uniform})"
printf "  %-26s %s\n" "EVAL_EVERY_EPOCH"       "$EVAL_EVERY_EPOCH  (metric=$EARLY_STOP_METRIC, patience=$EARLY_STOP_PATIENCE, min_delta=$EARLY_STOP_MIN_DELTA)"
printf "  %-26s %s\n" "VAL_SUBSET_SIZE"        "${VAL_SUBSET_SIZE:-0 (full NESTFUL)}"
printf "  %-26s %s\n" "STABILIZED_LR / KL"     "${STABILIZED_LR:-<config>} / ${STABILIZED_KL:-<config>}"
printf "  %-26s %s\n" "START_EPOCH"            "$START_EPOCH"
printf "  %-26s %s\n" "DRY_RUN"                "$DRY_RUN"
printf "  %-26s %s\n" "RUN_FINAL_EVAL"         "$RUN_FINAL_EVAL"
echo "════════════════════════════════════════════════════════════════"

# Verify (and optionally install) Python dependencies before doing any work, so
# a missing package fails fast / self-heals instead of crashing mid-stage.
if [ "$DRY_RUN" != "1" ]; then
    _preflight_deps
fi

mkdir -p "$OUTPUT_ROOT"
SUMMARY_FILE="$OUTPUT_ROOT/curriculum_summary.jsonl"
# Don't wipe an existing training summary when only running final_eval.
if [ "$ONLY_FINAL_EVAL" != "1" ]; then
    : > "$SUMMARY_FILE"
fi

# ── Initialization source: baseline (base model) | checkpoint (CHECKPOINT_IN) ──
case "$INIT_FROM" in
  baseline)
    if [ -n "$CHECKPOINT_IN" ]; then
        echo "[init] INIT_FROM=baseline but CHECKPOINT_IN='$CHECKPOINT_IN' is set; ignoring it."
    fi
    CHECKPOINT_IN=""
    echo "[init] INIT_FROM=baseline — training stage 1 starts from the BASE model (no adapter)."
    ;;
  checkpoint)
    if [ -z "$CHECKPOINT_IN" ]; then
        echo "[init] ERROR: INIT_FROM=checkpoint requires CHECKPOINT_IN=<adapter dir>." >&2
        echo "  e.g. CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_1" >&2
        exit 1
    fi
    if [ "$DRY_RUN" != "1" ] && [ ! -f "$CHECKPOINT_IN/adapter_config.json" ]; then
        echo "[init] ERROR: CHECKPOINT_IN is not a valid LoRA adapter dir: $CHECKPOINT_IN" >&2
        echo "  expected to find: $CHECKPOINT_IN/adapter_config.json" >&2
        exit 1
    fi
    echo "[init] INIT_FROM=checkpoint — seeding training from: $CHECKPOINT_IN"
    ;;
  *)
    echo "[init] ERROR: unknown INIT_FROM='$INIT_FROM' (use: baseline | checkpoint)" >&2
    exit 1
    ;;
esac

CURRENT_CHECKPOINT="$CHECKPOINT_IN"
# Seed FINAL_CHECKPOINT from env so final_eval can run standalone (STAGES="" with
# RUN_FINAL_EVAL=1) against an existing checkpoint. The training loop overwrites
# it with the freshly trained adapter when stages actually run.
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-$CHECKPOINT_IN}"
_IS_FIRST_STAGE=1   # used to apply START_EPOCH only to the first stage

# ── Global stabilized training overrides (LR / KL); reward is NOT touched ──────
STABILIZED_TRAIN_OVERRIDES=()
if [ -n "$STABILIZED_LR" ]; then
    STABILIZED_TRAIN_OVERRIDES+=(--override "training.learning_rate=$STABILIZED_LR")
fi
if [ -n "$STABILIZED_KL" ]; then
    STABILIZED_TRAIN_OVERRIDES+=(--override "training.kl_beta=$STABILIZED_KL")
fi
# Extra per-train overrides injected by wrappers (e.g. run_pilot_v2.sh sets the
# v2 reward policy + turn budget). Space-separated "--override k=v" pairs.
EXTRA_TRAIN_OVERRIDES=()
if [ -n "${EXTRA_TRAIN_OVERRIDES_STR:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_TRAIN_OVERRIDES=(${EXTRA_TRAIN_OVERRIDES_STR})
fi
# Track the best validation ReAct Win across the WHOLE run (all stages/epochs).
# PERSISTED across shell invocations (audit Bug 5: a resumed stage reset this to
# -1.0, letting a worse checkpoint overwrite best_react_win_adapter).
GLOBAL_BEST_FILE="$OUTPUT_ROOT/global_best_react_win.json"
GLOBAL_BEST_REACT_WIN="-1.0"
if [ -f "$GLOBAL_BEST_FILE" ]; then
    _PERSISTED_BEST=$(_json_get "$GLOBAL_BEST_FILE" "react_win_rate" "")
    if [ -n "$_PERSISTED_BEST" ]; then
        GLOBAL_BEST_REACT_WIN="$_PERSISTED_BEST"
        echo "[global-best] loaded persisted global best ReAct Win = $GLOBAL_BEST_REACT_WIN"
        echo "  (from $GLOBAL_BEST_FILE — carried across stages/invocations)"
    fi
    # Reuse the persisted baseline when none is set (keeps the guard floor stable).
    if [ -z "$REGRESSION_BASELINE_WIN" ]; then
        _PERSISTED_BASE=$(_json_get "$GLOBAL_BEST_FILE" "baseline_win" "")
        if [ -n "$_PERSISTED_BASE" ] && [ "$_PERSISTED_BASE" != "None" ]; then
            REGRESSION_BASELINE_WIN="$_PERSISTED_BASE"
            echo "[global-best] loaded persisted baseline dev Win = $REGRESSION_BASELINE_WIN"
        fi
    fi
fi

# Regression-guard bookkeeping.
REGRESSION_ABORT_PATIENCE="${REGRESSION_ABORT_PATIENCE:-3}"
BELOW_BASELINE_STREAK=0

# ── Establish baseline dev ReAct Win once (floor for the regression guard) ────
if [ "$REGRESSION_GUARD" = "1" ] && [ -z "$REGRESSION_BASELINE_WIN" ] \
   && [ "$EVAL_EVERY_EPOCH" = "1" ] && [ "$DRY_RUN" != "1" ]; then
    _GUARD_VAL="${VAL_JSONL:-}"
    if [ -z "$_GUARD_VAL" ]; then
        if [ -f "$ROOT/data/splits/nestful_dev.jsonl" ]; then
            _GUARD_VAL="$ROOT/data/splits/nestful_dev.jsonl"
        else
            _GUARD_VAL="$ROOT/data/splits/synthetic_val.jsonl"
        fi
    fi
    _GUARD_OUT="$OUTPUT_ROOT/baseline_dev_eval"
    _GUARD_METRICS="$_GUARD_OUT/metrics_epoch_0.json"
    if [ -f "$_GUARD_METRICS" ]; then
        REGRESSION_BASELINE_WIN=$(_json_get "$_GUARD_METRICS" "react_win_rate" "")
        if [ -n "$REGRESSION_BASELINE_WIN" ]; then
            echo "[regression-guard] reusing cached baseline dev ReAct Win = $REGRESSION_BASELINE_WIN"
            echo "  (from $_GUARD_METRICS — delete dir to re-measure)"
        fi
    fi
    if [ -z "$REGRESSION_BASELINE_WIN" ]; then
        echo "[regression-guard] measuring baseline (no-adapter) dev ReAct Win on $_GUARD_VAL ..."
        _build_vllm_overrides eval
        mkdir -p "$_GUARD_OUT"
        set +e
        _run_logged "$_GUARD_OUT/val_eval.log" \
            "$PYTHON" "$RUN_PY" \
            --mode val_eval \
            --config "$CONFIG" \
            --override "experiment.output_dir=$_GUARD_OUT" \
            --override "validation.subset_size=$VAL_SUBSET_SIZE" \
            --override "validation.require_win_rate=false" \
            --override "validation.stage=0" \
            --override "validation.epoch=0" \
            --override "paths.full_nestful_jsonl=$_GUARD_VAL" \
            --override "model.lora_adapter=null" \
            "${VLLM_OVERRIDES[@]}"
        _GUARD_RC=$?
        set -e
        if [ -f "$_GUARD_METRICS" ]; then
            REGRESSION_BASELINE_WIN=$(_json_get "$_GUARD_METRICS" "react_win_rate" "")
        fi
        if [ "$_GUARD_RC" -ne 0 ] && [ -n "$REGRESSION_BASELINE_WIN" ]; then
            echo "[regression-guard] WARNING: baseline val_eval exit=$_GUARD_RC but metrics present — continuing"
        fi
    fi
    if [ -n "$REGRESSION_BASELINE_WIN" ]; then
        echo "[regression-guard] baseline dev ReAct Win = $REGRESSION_BASELINE_WIN "\
"(checkpoints must reach >= baseline + $REGRESSION_MARGIN to be crowned best)"
    else
        echo "[regression-guard] ERROR: could not measure baseline dev Win." >&2
        if [ "$ALLOW_NO_REGRESSION_GUARD" = "1" ]; then
            echo "[regression-guard] WARNING: continuing WITHOUT the guard "\
"(ALLOW_NO_REGRESSION_GUARD=1)." >&2
            REGRESSION_GUARD=0
        else
            echo "[regression-guard] ABORT: refusing to train without a baseline floor." >&2
            echo "  Fix the baseline val_eval (see $_GUARD_OUT/val_eval.log), set" >&2
            echo "  REGRESSION_BASELINE_WIN=<win> explicitly, or set ALLOW_NO_REGRESSION_GUARD=1." >&2
            exit 1
        fi
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Per-stage loop
# ══════════════════════════════════════════════════════════════════════════════
# Best dev Win of the PREVIOUS stage (used by the hard stage gates).
PREV_STAGE_DEV_WIN="${PREV_STAGE_DEV_WIN:-}"
# Optional deterministic eval temperature (e.g. EVAL_TEMPERATURE=0.0).
EVAL_TEMP_OVERRIDE=()
if [ -n "${EVAL_TEMPERATURE:-}" ]; then
    EVAL_TEMP_OVERRIDE=(--override "generation.temperature=$EVAL_TEMPERATURE")
fi
for N in $STAGES; do
    EVAL_STAGE=$((N + 1))
    STAGE_OUT="$OUTPUT_ROOT/stage_${N}"
    TRAIN_JSONL="$DATA_BASE/epoch_${N}_${N}call.jsonl"
    # Per-epoch gate eval: prefer NESTFUL benchmark (aligned with final target);
    # fall back to synthetic if NESTFUL data is not present.
    _NESTFUL_JSONL="$ROOT/data/NESTFUL-main/data_v2/nestful_data.jsonl"
    if [ -f "$_NESTFUL_JSONL" ]; then
        EVAL_JSONL="$_NESTFUL_JSONL"
        EVAL_SOURCE="nestful"
    else
        EVAL_JSONL="$DATA_BASE/epoch_${EVAL_STAGE}_${EVAL_STAGE}call.jsonl"
        EVAL_SOURCE="synthetic"
    fi

    echo ""
    echo "╔══ Stage ${N} ════════════════════════════════════════════════════╗"
    printf "  train_stage = %s  |  eval_stage = %s\n" "$N" "$EVAL_STAGE"
    printf "  train data  = %s\n" "$TRAIN_JSONL"
    printf "  eval data   = %s  [%s]\n" "$EVAL_JSONL" "$EVAL_SOURCE"
    printf "  output      = %s\n" "$STAGE_OUT"
    echo "╚══════════════════════════════════════════════════════════════╝"

    mkdir -p "$STAGE_OUT/checkpoints" "$STAGE_OUT/eval"

    # Sanity check train data
    if [ "$DRY_RUN" != "1" ] && [ ! -f "$TRAIN_JSONL" ]; then
        echo "[stage $N] ERROR: train data not found: $TRAIN_JSONL" >&2
        echo "  For v3 resume run: bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh" >&2
        echo "  (creates DATA_BASE symlinks under OUTPUT_ROOT/data_base)" >&2
        exit 1
    fi

    # ── Mixed curriculum replay overrides for THIS stage (stages 1..N) ───────
    MIXED_OVERRIDES=()
    if [ "$CURRICULUM_MIXED_REPLAY" = "1" ]; then
        _MIX_FILES=""
        for _s in $(seq 1 "$N"); do
            _f="$DATA_BASE/epoch_${_s}_${_s}call.jsonl"
            if [ "$DRY_RUN" != "1" ] && [ ! -f "$_f" ]; then
                echo "[stage $N] ERROR: mixed-replay stage file not found: $_f" >&2
                echo "  Did you run experiments/data/prepare_clean_training_set.py and set DATA_BASE?" >&2
                exit 1
            fi
            if [ -z "$_MIX_FILES" ]; then _MIX_FILES="$_f"; else _MIX_FILES="$_MIX_FILES,$_f"; fi
        done
        MIXED_OVERRIDES+=(--override "data.mixed_replay=true"
                          --override "data.mixed_stage_files=$_MIX_FILES")
        # Per-stage replay weights (aligned to stages 1..N). A global
        # CURRICULUM_REPLAY_WEIGHTS overrides everything; otherwise a per-stage
        # env CURRICULUM_REPLAY_WEIGHTS_S<N>; otherwise the v2 default schedule
        # (heavier on the current stage, light replay of earlier stages to fight
        # forgetting / curriculum drift):
        #   s1: 1.0          s2: .35,.65
        #   s3: .20,.30,.50  s4: .15,.20,.25,.40
        _STAGE_WEIGHTS=""
        if [ -n "$CURRICULUM_REPLAY_WEIGHTS" ]; then
            _STAGE_WEIGHTS="$CURRICULUM_REPLAY_WEIGHTS"
        else
            _PSW_VAR="CURRICULUM_REPLAY_WEIGHTS_S${N}"
            _PSW_VAL="${!_PSW_VAR:-}"
            if [ -n "$_PSW_VAL" ]; then
                _STAGE_WEIGHTS="$_PSW_VAL"
            else
                case "$N" in
                    1) _STAGE_WEIGHTS="1.0" ;;
                    2) _STAGE_WEIGHTS="0.35,0.65" ;;
                    3) _STAGE_WEIGHTS="0.20,0.30,0.50" ;;
                    4) _STAGE_WEIGHTS="0.15,0.20,0.25,0.40" ;;
                    *) _STAGE_WEIGHTS="" ;;  # >4: fall back to uniform
                esac
            fi
        fi
        # A SCALAR weight for a multi-file mix means REPLAY RATIO (audit Bug 6):
        # e.g. 0.20 => previous stages total 20%, current stage 80%. The old code
        # padded it to [0.2, 0.2] => an unintended 50/50 mix. Lists with one
        # weight per file keep explicit-weights semantics.
        if [ -n "$_STAGE_WEIGHTS" ]; then
            if [ "$N" -gt 1 ] && ! echo "$_STAGE_WEIGHTS" | grep -q ','; then
                MIXED_OVERRIDES+=(--override "data.replay_ratio=$_STAGE_WEIGHTS")
                echo "[stage $N] replay_ratio=$_STAGE_WEIGHTS → previous stages ${_STAGE_WEIGHTS} total, current stage gets the rest"
            else
                MIXED_OVERRIDES+=(--override "data.replay_weights=$_STAGE_WEIGHTS")
            fi
        fi
        echo "[stage $N] mixed replay ON — stages 1..$N: $_MIX_FILES (weights=${_STAGE_WEIGHTS:-uniform})"
    fi

    # ── Multi-epoch loop per stage ──────────────────────────────────────────
    BEST_STRICT_PASS="0.0"
    BEST_EPOCH_CKPT=""
    PREV_STRICT_PASS="0.0"
    PLATEAU_COUNT=0
    ADVANCE_REASON="max_epochs"
    STAGE_FALLBACK_USED="false"
    STAGE_CLIPPED_RATE="0.0"
    # Early-stopping state (validation ReAct Win), reset per stage.
    STAGE_BEST_REACT_WIN="-1.0"
    EARLY_STOP_COUNT=0
    EARLY_STOPPED="false"

    # Apply START_EPOCH only to the first stage (resume support).
    _EPOCH_START=1
    # _RESUME_BOUNDARY marks the FIRST executed epoch of a resumed run. On that
    # epoch the previous-epoch adapter dir may not exist in this output tree (e.g.
    # a fresh pod that only has the single checkpoint passed via CHECKPOINT_IN),
    # so we fall back to CHECKPOINT_IN and HARD-FAIL rather than silently training
    # from the base model.
    _RESUME_BOUNDARY=0
    if [ "$_IS_FIRST_STAGE" = "1" ] && [ "${START_EPOCH:-1}" -gt 1 ]; then
        _EPOCH_START="$START_EPOCH"
        _RESUME_BOUNDARY=1
        echo "  [resume] starting stage $N from epoch $_EPOCH_START (START_EPOCH=$START_EPOCH)"
    fi
    _IS_FIRST_STAGE=0

    for EPOCH in $(seq "$_EPOCH_START" "$MAX_EPOCHS_PER_STAGE"); do
        EPOCH_OUT="$STAGE_OUT/epoch_${EPOCH}"
        CKPT_DIR="$STAGE_OUT/checkpoints/adapter_epoch_${EPOCH}"
        mkdir -p "$EPOCH_OUT"

        # Build checkpoint arg (either from previous epoch or inherited from prior stage)
        CKPT_ARG=""
        if [ "$EPOCH" -gt 1 ]; then
            PREV_CKPT="$STAGE_OUT/checkpoints/adapter_epoch_$((EPOCH-1))"
            if [ -d "$PREV_CKPT" ]; then
                CKPT_ARG="--checkpoint $PREV_CKPT"
            elif [ "$_RESUME_BOUNDARY" = "1" ] && [ -n "$CURRENT_CHECKPOINT" ]; then
                # Resume boundary: previous-epoch adapter not in this tree → use
                # the explicitly provided CHECKPOINT_IN to seed the resume.
                CKPT_ARG="--checkpoint $CURRENT_CHECKPOINT"
                echo "  [resume] seeding from CHECKPOINT_IN: $CURRENT_CHECKPOINT"
            fi
            # On the resume boundary a checkpoint is mandatory: refuse to silently
            # restart from the base model (that would throw away prior training).
            if [ "$_RESUME_BOUNDARY" = "1" ] && [ "$DRY_RUN" != "1" ] && [ -z "$CKPT_ARG" ]; then
                echo "[stage $N] ERROR: resume requested (START_EPOCH=$_EPOCH_START) but no checkpoint found." >&2
                echo "  Looked for previous-epoch adapter: $PREV_CKPT" >&2
                echo "  and CHECKPOINT_IN='$CURRENT_CHECKPOINT' (empty or missing)." >&2
                echo "  Fix: set CHECKPOINT_IN to a valid adapter dir, e.g." >&2
                echo "       CHECKPOINT_IN=outputs/curriculum/stage_2/checkpoints/adapter_epoch_1" >&2
                exit 1
            fi
        elif [ -n "$CURRENT_CHECKPOINT" ]; then
            _INHERITED_CKPT="$(_resolve_checkpoint "$CURRENT_CHECKPOINT" "$STAGE_OUT/checkpoints" || true)"
            if [ -z "$_INHERITED_CKPT" ]; then
                echo "[stage $N] ERROR: inherited checkpoint missing or invalid: $CURRENT_CHECKPOINT" >&2
                echo "  Fix: set CHECKPOINT_IN to an existing adapter dir under $STAGE_OUT/checkpoints" >&2
                exit 1
            fi
            if [ "$_INHERITED_CKPT" != "$CURRENT_CHECKPOINT" ]; then
                echo "  [ckpt] inherited path missing; using $_INHERITED_CKPT instead of $CURRENT_CHECKPOINT"
                CURRENT_CHECKPOINT="$_INHERITED_CKPT"
            fi
            CKPT_ARG="--checkpoint $CURRENT_CHECKPOINT"
        fi
        # Only the first executed epoch is a resume boundary; clear it afterwards.
        _RESUME_BOUNDARY=0

        GENERIC_CKPT="$STAGE_OUT/checkpoints/adapter_epoch_1"
        _SNAP_RESTORE=""
        # Each curriculum epoch runs with training.epochs=1, so grpo_train always writes
        # adapter_epoch_1. When resuming from adapter_epoch_{E-1} and that path IS the
        # generic slot (notably E=2 after E=1), snapshot it before train overwrites it.
        if [ "$EPOCH" -gt 1 ] && [ "$DRY_RUN" != "1" ]; then
            _PREV_E=$((EPOCH - 1))
            _PREV_DIR="$STAGE_OUT/checkpoints/adapter_epoch_${_PREV_E}"
            if [ -d "$_PREV_DIR" ] && [ "$_PREV_DIR" = "$GENERIC_CKPT" ]; then
                _SNAP="$STAGE_OUT/checkpoints/.snap_epoch_${_PREV_E}"
                rm -rf "$_SNAP"
                cp -a "$_PREV_DIR" "$_SNAP"
                _SNAP_RESTORE="$_PREV_E"
                echo "  [ckpt] preserved adapter_epoch_${_PREV_E} before train (trainer overwrites generic slot)"
            fi
        fi

        echo ""
        echo "  ── Stage $N / Epoch $EPOCH / $MAX_EPOCHS_PER_STAGE ──────────────────────────────"

        # ── TRAIN ────────────────────────────────────────────────────────────
        echo "[stage $N / epoch $EPOCH] train ..."
        _build_vllm_overrides train
        TRAIN_DATA_OVERRIDE="$(_data_overrides_train)"
        export WANDB_RUN_NAME="train-stage${N}-e${EPOCH}"
        # Stage id visible to stage-aware rewards in the trainer AND in the DP
        # rollout workers (audit Bug 7).
        export TRAIN_STAGE="$N"
        # shellcheck disable=SC2086
        _run_logged "$EPOCH_OUT/train.log" \
            "$PYTHON" "$RUN_PY" \
            --mode train \
            --config "$CONFIG" \
            --override "paths.train_jsonl=$TRAIN_JSONL" \
            --override "data.train_stage=$N" \
            $TRAIN_DATA_OVERRIDE \
            --override "training.epochs=1" \
            --override "generation.num_generations=$NUM_GENERATIONS" \
            --override "experiment.output_dir=$EPOCH_OUT" \
            --override "model.output_adapter_dir=$STAGE_OUT/checkpoints" \
            "${MIXED_OVERRIDES[@]}" \
            "${STABILIZED_TRAIN_OVERRIDES[@]}" \
            "${EXTRA_TRAIN_OVERRIDES[@]}" \
            "${VLLM_OVERRIDES[@]}" \
            $CKPT_ARG

        # Rename generic adapter_epoch_1 → adapter_epoch_N (train always writes epoch_1)
        if [ "$DRY_RUN" != "1" ] && [ -d "$GENERIC_CKPT" ] && [ ! -d "$CKPT_DIR" ]; then
            mv "$GENERIC_CKPT" "$CKPT_DIR"
        fi
        if [ -n "$_SNAP_RESTORE" ] && [ -d "$STAGE_OUT/checkpoints/.snap_epoch_${_SNAP_RESTORE}" ]; then
            rm -rf "$STAGE_OUT/checkpoints/adapter_epoch_${_SNAP_RESTORE}"
            mv "$STAGE_OUT/checkpoints/.snap_epoch_${_SNAP_RESTORE}" \
               "$STAGE_OUT/checkpoints/adapter_epoch_${_SNAP_RESTORE}"
            echo "  [ckpt] restored adapter_epoch_${_SNAP_RESTORE} after rename"
        fi
        if [ "$DRY_RUN" = "1" ]; then
            echo "[DRY_RUN] would checkpoint: $CKPT_DIR"
        fi

        # Read fallback flag from this epoch's train_summary.json
        TRAIN_SUMMARY="$EPOCH_OUT/train_summary.json"
        EPOCH_FALLBACK="false"
        if [ "$DRY_RUN" != "1" ] && [ -f "$TRAIN_SUMMARY" ]; then
            EPOCH_FALLBACK=$(_json_get "$TRAIN_SUMMARY" "fallback_used" "false")
            if [ "$EPOCH_FALLBACK" = "true" ]; then STAGE_FALLBACK_USED="true"; fi
        fi

        # ── EVAL ─────────────────────────────────────────────────────────────
        EVAL_SKIPPED=0
        EPOCH_STRICT_PASS="0.0"
        if [ "$DRY_RUN" != "1" ] && [ ! -f "$EVAL_JSONL" ]; then
            echo "[stage $N / epoch $EPOCH] WARNING: eval data '$EVAL_JSONL' not found — skipping eval."
            EVAL_SKIPPED=1
        else
            EVAL_CKPT_ARG=""
            if [ "$DRY_RUN" != "1" ] && [ -d "$CKPT_DIR" ]; then
                EVAL_CKPT_ARG="--checkpoint $CKPT_DIR"
            fi
            echo "[stage $N / epoch $EPOCH] rollout_eval on stage $EVAL_STAGE ..."
            export WANDB_RUN_NAME="eval-stage${EVAL_STAGE}-e${EPOCH}"
            _build_vllm_overrides eval
            EVAL_DATA_OVERRIDE="$(_data_overrides_eval)"
            # shellcheck disable=SC2086
            _run_logged "$EPOCH_OUT/eval.log" \
                "$PYTHON" "$RUN_PY" \
                --mode rollout_eval \
                --config "$CONFIG" \
                --override "paths.eval_jsonl=$EVAL_JSONL" \
                --override "data.eval_stage=$EVAL_STAGE" \
                $EVAL_DATA_OVERRIDE \
                --override "experiment.output_dir=$EPOCH_OUT/eval" \
                "${VLLM_OVERRIDES[@]}" \
                "${EVAL_TEMP_OVERRIDE[@]}" \
                $EVAL_CKPT_ARG

            EVAL_METRICS="$EPOCH_OUT/eval/metrics.json"
            if [ "$DRY_RUN" != "1" ] && [ -f "$EVAL_METRICS" ]; then
                EPOCH_STRICT_PASS=$(_json_get "$EVAL_METRICS" "strict_gold_trace_pass" "0.0")
                STAGE_CLIPPED_RATE=$(_json_get "$EVAL_METRICS" "clipped_completion_rate" "0.0")
            fi
        fi

        echo "[stage $N / epoch $EPOCH] strict_gold_trace_pass=$EPOCH_STRICT_PASS  fallback=$EPOCH_FALLBACK"

        # ── VALIDATION ReAct Win + early stopping (stabilized profile) ───────
        # Reward UNCHANGED: this only EVALUATES the freshly trained adapter on the
        # validation set (official ReAct Win) and uses it to (a) keep the global
        # best_react_win_adapter and (b) early-stop when Win stops improving.
        EPOCH_REACT_WIN="none"
        if [ "$EVAL_EVERY_EPOCH" = "1" ] && [ "$DRY_RUN" != "1" ] && [ -d "$CKPT_DIR" ]; then
            echo "[stage $N / epoch $EPOCH] val_eval (ReAct Win) ..."
            export WANDB_RUN_NAME="valeval-stage${N}-e${EPOCH}"
            _build_vllm_overrides eval
            VAL_OUT="$EPOCH_OUT/val_eval"
            mkdir -p "$VAL_OUT"
            # shellcheck disable=SC2086
            # v2 stabilized: validate/select on the REAL held-out NESTFUL dev
            # subset (nestful_dev.jsonl), which is DISJOINT from the reporting
            # test set (nestful_test.jsonl). This makes checkpoint selection track
            # real-task ReAct Win instead of synthetic proxy Win (ROOT_CAUSE #2).
            # Build it once with:
            #   python experiments/comparison/make_nestful_dev_split.py
            # Fallbacks: legacy synthetic val if the dev split is absent; set
            # VAL_JSONL="" to use the full NESTFUL validation path from config.
            _NESTFUL_DEV="$ROOT/data/splits/nestful_dev.jsonl"
            if [ -n "${VAL_JSONL:-}" ]; then
                :  # explicit override respected
            elif [ -f "$_NESTFUL_DEV" ]; then
                VAL_JSONL="$_NESTFUL_DEV"
            else
                VAL_JSONL="$ROOT/data/splits/synthetic_val.jsonl"
                echo "[stage $N / epoch $EPOCH] WARNING: NESTFUL dev split not found "\
"($_NESTFUL_DEV) — falling back to SYNTHETIC val. Run make_nestful_dev_split.py "\
"for real-NESTFUL selection."
            fi
            VAL_PATH_OVERRIDE=()
            if [ -n "$VAL_JSONL" ] && [ -f "$VAL_JSONL" ]; then
                VAL_PATH_OVERRIDE=(--override "paths.full_nestful_jsonl=$VAL_JSONL")
                case "$VAL_JSONL" in
                    *nestful_dev.jsonl) echo "[stage $N / epoch $EPOCH] validation set = $VAL_JSONL (REAL NESTFUL dev, held-out)";;
                    *) echo "[stage $N / epoch $EPOCH] validation set = $VAL_JSONL";;
                esac
            fi
            # shellcheck disable=SC2086
            if ! _run_logged "$EPOCH_OUT/val_eval.log" \
                "$PYTHON" "$RUN_PY" \
                --mode val_eval \
                --config "$CONFIG" \
                --override "experiment.output_dir=$VAL_OUT" \
                --override "validation.subset_size=$VAL_SUBSET_SIZE" \
                --override "validation.subset_ids_path=$OUTPUT_ROOT/validation_subset_ids.json" \
                --override "validation.subset_jsonl=$OUTPUT_ROOT/validation_subset.jsonl" \
                --override "validation.stage=$N" \
                --override "validation.epoch=$EPOCH" \
                --override "validation.require_win_rate=true" \
                "${VAL_PATH_OVERRIDE[@]}" \
                "${VLLM_OVERRIDES[@]}" \
                "${EVAL_TEMP_OVERRIDE[@]}" \
                --checkpoint "$CKPT_DIR"; then
                echo "[stage $N / epoch $EPOCH] FATAL: val_eval failed (react_win_rate null or "\
"official scorer error). Aborting run — checkpoint selection must not proceed on "\
"an invalid signal. Fix nestful_official_score / val set and re-run." >&2
                exit 2
            fi

            VAL_METRICS="$VAL_OUT/metrics_epoch_${EPOCH}.json"
            if [ -f "$VAL_METRICS" ]; then
                EPOCH_REACT_WIN=$(_json_get "$VAL_METRICS" "react_win_rate" "none")
            fi
            echo "[stage $N / epoch $EPOCH] react_win_rate=$EPOCH_REACT_WIN"

            if [ "$EPOCH_REACT_WIN" != "none" ] && [ -n "$EPOCH_REACT_WIN" ]; then
                # ── Regression guard: never crown a checkpoint below baseline ──
                _GUARD_OK=1
                if [ "$REGRESSION_GUARD" = "1" ] && [ -n "$REGRESSION_BASELINE_WIN" ]; then
                    if _improved "$EPOCH_REACT_WIN" "$REGRESSION_BASELINE_WIN" "$REGRESSION_MARGIN" 2>/dev/null; then
                        _GUARD_OK=1
                        BELOW_BASELINE_STREAK=0
                    else
                        _GUARD_OK=0
                        BELOW_BASELINE_STREAK=$((BELOW_BASELINE_STREAK + 1))
                        echo "[stage $N / epoch $EPOCH] regression-guard: Win $EPOCH_REACT_WIN "\
"< baseline $REGRESSION_BASELINE_WIN (+$REGRESSION_MARGIN) — NOT saved as best "\
"(below-baseline streak=$BELOW_BASELINE_STREAK)"
                        if [ "$REGRESSION_EARLY_ABORT" = "1" ] \
                           && [ "$BELOW_BASELINE_STREAK" -ge "$REGRESSION_ABORT_PATIENCE" ]; then
                            echo "[regression-guard] ABORT: $BELOW_BASELINE_STREAK consecutive epochs "\
"below baseline (patience=$REGRESSION_ABORT_PATIENCE). Stopping run so no compute is "\
"wasted on a regressing model. Set REGRESSION_EARLY_ABORT=0 to disable."
                            exit 3
                        fi
                    fi
                fi

                # Global best adapter (selection target = validation ReAct Win).
                # Crowning is additionally gated by checkpoint ELIGIBILITY
                # (audit Bug 4): a 0-step / all-dead-groups / reward-fallback
                # checkpoint must NEVER become best_react_win_adapter.
                if [ "$_GUARD_OK" = "1" ] && _improved "$EPOCH_REACT_WIN" "$GLOBAL_BEST_REACT_WIN" "0" 2>/dev/null; then
                    _ELIG_OUT="$EPOCH_OUT/checkpoint_eligibility.json"
                    _ELIG_OK=0
                    if "$PYTHON" "$ROOT/checkpoint_eligibility.py" \
                        --train-summary "$TRAIN_SUMMARY" \
                        --react-win "$EPOCH_REACT_WIN" \
                        --global-best "$GLOBAL_BEST_REACT_WIN" \
                        --baseline-win "${REGRESSION_BASELINE_WIN:-}" \
                        --regression-guard "$REGRESSION_GUARD" \
                        --regression-margin "$REGRESSION_MARGIN" \
                        --out "$_ELIG_OUT"; then
                        _ELIG_OK=1
                    fi
                    if [ "$_ELIG_OK" = "1" ]; then
                        GLOBAL_BEST_REACT_WIN="$EPOCH_REACT_WIN"
                        rm -rf "$BEST_REACT_WIN_ADAPTER"
                        mkdir -p "$BEST_REACT_WIN_ADAPTER"
                        cp -r "$CKPT_DIR/." "$BEST_REACT_WIN_ADAPTER/"
                        _BW_STAGE="$N" _BW_EPOCH="$EPOCH" _BW_WIN="$EPOCH_REACT_WIN" \
                        _BW_SRC="$CKPT_DIR" _BW_OUT="$BEST_REACT_WIN_ADAPTER/best_meta.json" \
                        _BW_ELIG="$_ELIG_OUT" _BW_BASE="${REGRESSION_BASELINE_WIN:-}" \
                        _BW_GLOBAL_FILE="$GLOBAL_BEST_FILE" \
                        "$PYTHON" - <<'PYEOF'
import json, os
e = os.environ
elig = {}
try:
    with open(e["_BW_ELIG"]) as f:
        elig = json.load(f)
except Exception:
    pass
meta = {
    "stage": int(e["_BW_STAGE"]),
    "epoch": int(e["_BW_EPOCH"]),
    "react_win_rate": float(e["_BW_WIN"]),
    "source_checkpoint": e["_BW_SRC"],
    "selection_metric": "react_win_rate",
    "baseline_win": float(e["_BW_BASE"]) if e.get("_BW_BASE") else None,
    # Eligibility evidence (audit Bug 4): steps / contributing_turns /
    # dead_group_rate / reward policy / resolved fn / eligible_for_best.
    **{k: elig.get(k) for k in (
        "steps", "contributing_turns", "dead_group_rate", "reward_policy",
        "resolved_reward_fn", "reward_fallback_used", "trained",
        "eligible_for_best", "reason", "regression_guard")},
}
with open(e["_BW_OUT"], "w") as f:
    json.dump(meta, f, indent=2)
# Persist the global best across stages AND shell invocations (audit Bug 5).
with open(e["_BW_GLOBAL_FILE"], "w") as f:
    json.dump({
        "react_win_rate": float(e["_BW_WIN"]),
        "stage": int(e["_BW_STAGE"]),
        "epoch": int(e["_BW_EPOCH"]),
        "source_checkpoint": e["_BW_SRC"],
        "baseline_win": float(e["_BW_BASE"]) if e.get("_BW_BASE") else None,
    }, f, indent=2)
PYEOF
                        echo "[stage $N / epoch $EPOCH] NEW BEST react_win_rate=$GLOBAL_BEST_REACT_WIN -> $BEST_REACT_WIN_ADAPTER"
                        echo "  (persisted to $GLOBAL_BEST_FILE)"
                    else
                        echo "[stage $N / epoch $EPOCH] checkpoint NOT crowned: $(cat "$_ELIG_OUT" 2>/dev/null | "$PYTHON" -c 'import json,sys;print(json.load(sys.stdin).get("reason","ineligible"))' 2>/dev/null || echo ineligible)"
                    fi
                fi

                # Early stopping: reset patience only on >= min_delta improvement.
                if _improved "$EPOCH_REACT_WIN" "$STAGE_BEST_REACT_WIN" "$EARLY_STOP_MIN_DELTA" 2>/dev/null; then
                    STAGE_BEST_REACT_WIN="$EPOCH_REACT_WIN"
                    EARLY_STOP_COUNT=0
                else
                    EARLY_STOP_COUNT=$((EARLY_STOP_COUNT + 1))
                    echo "[stage $N / epoch $EPOCH] no ReAct Win improvement (>= $EARLY_STOP_MIN_DELTA): "\
"early_stop_count=$EARLY_STOP_COUNT / patience=$EARLY_STOP_PATIENCE"
                fi
            fi
        fi

        # Track best checkpoint for this stage
        if [ "$DRY_RUN" != "1" ]; then
            if [ "$BEST_EPOCH_CKPT" = "" ]; then
                BEST_EPOCH_CKPT="$CKPT_DIR"
                BEST_STRICT_PASS="$EPOCH_STRICT_PASS"
            elif _float_ge "$EPOCH_STRICT_PASS" "$BEST_STRICT_PASS"; then
                BEST_STRICT_PASS="$EPOCH_STRICT_PASS"
                BEST_EPOCH_CKPT="$CKPT_DIR"
            fi
        else
            BEST_EPOCH_CKPT="$CKPT_DIR"
        fi

        # Write per-epoch summary row to stage's epoch_summary.jsonl
        _E_STAGE="$N" _E_EPOCH="$EPOCH" _E_STRICT="$EPOCH_STRICT_PASS" \
        _E_FALLBACK="$EPOCH_FALLBACK" _E_CLIPPED="$STAGE_CLIPPED_RATE" \
        _E_CKPT="$CKPT_DIR" _E_OUT="$STAGE_OUT/epoch_summary.jsonl" \
        _E_REACT_WIN="$EPOCH_REACT_WIN" _E_ESCOUNT="$EARLY_STOP_COUNT" \
        "$PYTHON" - <<'PYEOF'
import json, os
e = os.environ
_rw = e.get("_E_REACT_WIN", "none")
row = {
    "stage": int(e.get("_E_STAGE", "0")),
    "epoch": int(e.get("_E_EPOCH", "0")),
    "strict_gold_trace_pass": float(e.get("_E_STRICT", "0") or "0"),
    "react_win_rate": (float(_rw) if _rw not in ("none", "", None) else None),
    "early_stop_count": int(e.get("_E_ESCOUNT", "0") or "0"),
    "fallback_used": e.get("_E_FALLBACK", "false") == "true",
    "clipped_completion_rate": float(e.get("_E_CLIPPED", "0") or "0"),
    "checkpoint": e.get("_E_CKPT", ""),
}
with open(e.get("_E_OUT", "epoch_summary.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF

        # ── Advancement gate ───────────────────────────────────────────────
        # Logic mirrors curricullum/train/run_curriculum_training.py:
        #   advance if: strict_pass >= threshold  OR  plateau  OR  max_epochs
        if [ "$DRY_RUN" != "1" ] && [ "$EVAL_SKIPPED" = "0" ]; then
            # 1. Threshold reached → advance early
            if _float_ge "$EPOCH_STRICT_PASS" "$ADVANCE_THRESHOLD" 2>/dev/null; then
                ADVANCE_REASON="threshold_reached (${EPOCH_STRICT_PASS} >= ${ADVANCE_THRESHOLD})"
                echo "[stage $N / epoch $EPOCH] ADVANCE: $ADVANCE_REASON"
                break
            fi

            # 2. Plateau detection
            if _float_ge "$EPOCH_STRICT_PASS" "$PREV_STRICT_PASS" 2>/dev/null; then
                PLATEAU_COUNT=0
            else
                PLATEAU_COUNT=$((PLATEAU_COUNT + 1))
            fi
            PREV_STRICT_PASS="$EPOCH_STRICT_PASS"

            if [ "$PLATEAU_COUNT" -ge "$PLATEAU_PATIENCE" ] && [ "$EPOCH" -lt "$MAX_EPOCHS_PER_STAGE" ]; then
                ADVANCE_REASON="plateau (no improvement for ${PLATEAU_PATIENCE} epochs)"
                echo "[stage $N / epoch $EPOCH] ADVANCE: $ADVANCE_REASON"
                break
            fi
        fi

        # 2b. Early stopping on validation ReAct Win (stabilized profile).
        if [ "$EVAL_EVERY_EPOCH" = "1" ] && [ "$DRY_RUN" != "1" ] \
           && [ "$EARLY_STOP_COUNT" -ge "$EARLY_STOP_PATIENCE" ] \
           && [ "$EPOCH" -lt "$MAX_EPOCHS_PER_STAGE" ]; then
            EARLY_STOPPED="true"
            ADVANCE_REASON="early_stop ($EARLY_STOP_METRIC no >= ${EARLY_STOP_MIN_DELTA} gain for ${EARLY_STOP_PATIENCE} evals; best=${STAGE_BEST_REACT_WIN})"
            echo "[stage $N / epoch $EPOCH] EARLY STOP: $ADVANCE_REASON"
            break
        fi

        # 3. Reached max epochs
        if [ "$EPOCH" -ge "$MAX_EPOCHS_PER_STAGE" ]; then
            ADVANCE_REASON="max_epochs_per_stage (${MAX_EPOCHS_PER_STAGE})"
        fi

    done   # end epoch loop

    echo ""
    echo "[stage $N] best strict_pass=$BEST_STRICT_PASS  reason=$ADVANCE_REASON  ckpt=$BEST_EPOCH_CKPT"

    # ── Stage gate ─────────────────────────────────────────────────────────
    GATE_PASS="true"
    GATE_REASON="ok"
    GATE_WARNING=""

    if [ "$DRY_RUN" != "1" ]; then
        if [ "$STAGE_FALLBACK_USED" = "true" ]; then
            GATE_PASS="false"
            GATE_REASON="fallback_used=true in at least one epoch (episode-level fallback)"
        fi
        if ! _float_le "$STAGE_CLIPPED_RATE" "$MAX_CLIPPED_COMPLETION_RATE" 2>/dev/null; then
            GATE_PASS="false"
            GATE_REASON="${GATE_REASON}; clipped_completion_rate=${STAGE_CLIPPED_RATE} > ${MAX_CLIPPED_COMPLETION_RATE}"
        fi
    else
        GATE_REASON="dry_run"
    fi

    if [ "$GATE_PASS" = "false" ]; then
        echo "[stage $N] ⚠ GATE FAILED: $GATE_REASON"
        if [ "$GATE_MODE" = "stop" ] || [ "$STOP_ON_FAIL" = "1" ]; then
            echo "[stage $N] GATE_MODE=stop — aborting."
            echo "  Set GATE_MODE=warn or STOP_ON_FAIL=0 to continue overnight."
            exit 1
        else
            echo "[stage $N] GATE_MODE=warn — logging failure and continuing to next stage."
        fi
    else
        echo "[stage $N] gate_pass=true | $GATE_REASON"
    fi

    # ── Hard stage-advancement gates (STAGE_GATES=1; post-audit pilot) ──────
    # Evaluates steps / dead-group rate / reward dispatch / fractional rewards /
    # dev Win vs baseline / position-artifact rate for THIS stage. A failing
    # stage STOPS the pilot (exit 4) — no advancing on a broken learning signal.
    if [ "$STAGE_GATES" = "1" ] && [ "$DRY_RUN" != "1" ]; then
        echo "[stage $N] evaluating hard stage-advancement gates ..."
        _SG_DEV_WIN="$STAGE_BEST_REACT_WIN"
        if [ "$_SG_DEV_WIN" = "-1.0" ]; then _SG_DEV_WIN=""; fi
        set +e
        "$PYTHON" "$ROOT/check_stage_gates.py" \
            --stage "$N" \
            --stage-out "$STAGE_OUT" \
            --dev-win "${_SG_DEV_WIN:-}" \
            --baseline-win "${REGRESSION_BASELINE_WIN:-}" \
            --prev-stage-win "${PREV_STAGE_DEV_WIN:-}" \
            --out "$STAGE_OUT/stage_gate_report.json"
        _SG_RC=$?
        set -e
        if [ "$_SG_RC" -ne 0 ]; then
            echo "[stage $N] ✗ STAGE GATES FAILED — stopping the pilot (no advancement)." >&2
            echo "  Report: $STAGE_OUT/stage_gate_report.json" >&2
            echo "  Do NOT run the next stage until the failure is understood." >&2
            exit 4
        fi
        echo "[stage $N] ✓ stage gates passed — advancement to the next stage allowed."
    fi
    # Track this stage's best dev Win for the next stage's gate comparison.
    if [ "$STAGE_BEST_REACT_WIN" != "-1.0" ]; then
        PREV_STAGE_DEV_WIN="$STAGE_BEST_REACT_WIN"
    fi

    # ── Stage manifest ─────────────────────────────────────────────────────
    _PROFILE="$PROFILE" _N="$N" _EVAL_STAGE="$EVAL_STAGE" \
    _BEST_CKPT="$BEST_EPOCH_CKPT" _BEST_PASS="$BEST_STRICT_PASS" \
    _CKPT_IN="$CURRENT_CHECKPOINT" _STAGE_OUT="$STAGE_OUT" \
    _TRAIN_JSONL="$TRAIN_JSONL" _EVAL_JSONL="$EVAL_JSONL" \
    _MTT="${MAX_TRAIN_TASKS:-null}" _MET="${MAX_EVAL_TASKS:-null}" \
    _MAX_EPOCHS="$MAX_EPOCHS_PER_STAGE" _NGEN="$NUM_GENERATIONS" \
    _ADVANCE="$ADVANCE_REASON" \
    _GATE_PASS="$GATE_PASS" _GATE_REASON="$GATE_REASON" \
    _FALLBACK="$STAGE_FALLBACK_USED" _CLIPPED="$STAGE_CLIPPED_RATE" \
    _USE_VLLM="$USE_VLLM" _USE_FA="$USE_FLASH_ATTENTION" \
    _BEST_REACT_WIN="$STAGE_BEST_REACT_WIN" _EARLY_STOPPED="$EARLY_STOPPED" \
    _GLOBAL_BEST_REACT_WIN="$GLOBAL_BEST_REACT_WIN" _BEST_REACT_ADAPTER="$BEST_REACT_WIN_ADAPTER" \
    _INIT_FROM="$INIT_FROM" _MIXED_REPLAY="$CURRICULUM_MIXED_REPLAY" \
    "$PYTHON" - <<'PYEOF'
import json, os
e = os.environ
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x < 0 else x   # -1.0 sentinel = not evaluated
m = {
    "profile":              e.get("_PROFILE"),
    "stage":                int(e.get("_N", "0")),
    "eval_stage":           int(e.get("_EVAL_STAGE", "0")),
    "init_from":            e.get("_INIT_FROM"),
    "mixed_replay":         e.get("_MIXED_REPLAY", "0") == "1",
    "checkpoint_in":        e.get("_CKPT_IN") or None,
    "best_checkpoint":      e.get("_BEST_CKPT") or None,
    "best_strict_pass":     float(e.get("_BEST_PASS", "0") or "0"),
    "best_react_win_stage": _f(e.get("_BEST_REACT_WIN")),
    "early_stopped":        e.get("_EARLY_STOPPED", "false") == "true",
    "global_best_react_win": _f(e.get("_GLOBAL_BEST_REACT_WIN")),
    "best_react_win_adapter": e.get("_BEST_REACT_ADAPTER") or None,
    "train_output_dir":     e.get("_STAGE_OUT"),
    "train_jsonl":          e.get("_TRAIN_JSONL"),
    "eval_jsonl":           e.get("_EVAL_JSONL"),
    "max_train_tasks":      e.get("_MTT") if e.get("_MTT") != "null" else None,
    "max_eval_tasks":       e.get("_MET") if e.get("_MET") != "null" else None,
    "max_epochs_per_stage": int(e.get("_MAX_EPOCHS", "1")),
    "num_generations":      int(e.get("_NGEN", "4")),
    "advance_reason":       e.get("_ADVANCE", ""),
    "gate_pass":            e.get("_GATE_PASS", "false") == "true",
    "gate_reason":          e.get("_GATE_REASON", ""),
    "fallback_used_any":    e.get("_FALLBACK", "false") == "true",
    "best_clipped_rate":    float(e.get("_CLIPPED", "0") or "0"),
    "use_vllm":             e.get("_USE_VLLM", "0") == "1",
    "use_flash_attention":  e.get("_USE_FA", "0") == "1",
}
out = (e.get("_STAGE_OUT") or ".") + "/stage_manifest.json"
with open(out, "w") as f:
    json.dump(m, f, indent=2)
print(f"  stage_manifest -> {out}")
PYEOF

    # ── Curriculum summary row ─────────────────────────────────────────────
    _N="$N" _EVAL_STAGE="$EVAL_STAGE" _BEST_CKPT="$BEST_EPOCH_CKPT" \
    _BEST_PASS="$BEST_STRICT_PASS" _FALLBACK="$STAGE_FALLBACK_USED" \
    _CLIPPED="$STAGE_CLIPPED_RATE" _GATE="$GATE_PASS" \
    _ADVANCE="$ADVANCE_REASON" _SUMMARY="$SUMMARY_FILE" \
    "$PYTHON" - <<'PYEOF'
import json, os
e = os.environ
row = {
    "stage":                    int(e.get("_N", "0")),
    "eval_stage":               int(e.get("_EVAL_STAGE", "0")),
    "best_checkpoint":          e.get("_BEST_CKPT") or None,
    "best_strict_gold_trace_pass": float(e.get("_BEST_PASS", "0") or "0"),
    "fallback_used_any":        e.get("_FALLBACK", "false") == "true",
    "best_clipped_rate":        float(e.get("_CLIPPED", "0") or "0"),
    "gate_pass":                e.get("_GATE", "false") == "true",
    "advance_reason":           e.get("_ADVANCE", ""),
}
with open(e.get("_SUMMARY", "curriculum_summary.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF

    # Carry best checkpoint forward to next stage (must exist on disk)
    if [ -n "$BEST_EPOCH_CKPT" ]; then
        _NEXT_CKPT="$(_resolve_checkpoint "$BEST_EPOCH_CKPT" "$STAGE_OUT/checkpoints" || true)"
        if [ -z "$_NEXT_CKPT" ]; then
            echo "[stage $N] ERROR: best checkpoint path missing: $BEST_EPOCH_CKPT" >&2
            echo "  Contents of $STAGE_OUT/checkpoints:" >&2
            ls -la "$STAGE_OUT/checkpoints" >&2 || true
            exit 1
        fi
        if [ "$_NEXT_CKPT" != "$BEST_EPOCH_CKPT" ]; then
            echo "[stage $N] WARNING: best ckpt $BEST_EPOCH_CKPT missing; using $_NEXT_CKPT for next stage"
            BEST_EPOCH_CKPT="$_NEXT_CKPT"
        fi
        CURRENT_CHECKPOINT="$BEST_EPOCH_CKPT"
    fi
    FINAL_CHECKPOINT="$CURRENT_CHECKPOINT"

done   # end stage loop

# ── Report best-by-validation-ReAct-Win adapter (stabilized profile) ──────────
if [ "$EVAL_EVERY_EPOCH" = "1" ]; then
    if [ -d "$BEST_REACT_WIN_ADAPTER" ]; then
        echo ""
        echo "[curriculum] BEST validation ReAct Win = $GLOBAL_BEST_REACT_WIN"
        echo "[curriculum] best_react_win_adapter   = $BEST_REACT_WIN_ADAPTER"
        echo "  (use this adapter for final_eval / deployment; selected by validation ReAct Win)"
    else
        echo "[curriculum] WARNING: EVAL_EVERY_EPOCH=1 but no best_react_win_adapter was produced"
        echo "  (val_eval may have failed to compute react_win_rate — check val_eval.log)."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Optional final_eval on last best checkpoint
# ══════════════════════════════════════════════════════════════════════════════
if [ "$RUN_FINAL_EVAL" = "1" ]; then
    # Final eval runs the FULL NESTFUL benchmark and writes, per run dir:
    #   metrics.json           — our internal ReAct diagnostics
    #   metrics_official.json  — paper-comparable scores from the REAL NESTFUL
    #                            scorer (F1 Func/Param, Part./Full Acc, Win Rate)
    # Win Rate is computed automatically on Linux when ibm_functions_dir exists.
    #
    # Toggles:
    #   FINAL_EVAL_PARADIGM    react | direct | both   (default: both)
    #     react  = our multi-turn rollout       (paper Table 2)
    #     direct = single-shot full sequence+ICL (paper Table 1)
    #   FINAL_EVAL_BASELINE    1 = also eval the base model (no adapter)  (default: 1)
    #   FINAL_EVAL_NUM_ICL     in-context examples for direct paradigm    (default: 1)
    FINAL_EVAL_PARADIGM="${FINAL_EVAL_PARADIGM:-both}"
    FINAL_EVAL_BASELINE="${FINAL_EVAL_BASELINE:-1}"
    FINAL_EVAL_NUM_ICL="${FINAL_EVAL_NUM_ICL:-1}"

    case "$FINAL_EVAL_PARADIGM" in
        both) _PARADIGMS="react direct" ;;
        *)    _PARADIGMS="$FINAL_EVAL_PARADIGM" ;;
    esac

    _build_vllm_overrides eval

    # _final_eval_run <label> <subdir> <paradigm> [extra run.py args...]
    _final_eval_run() {
        local label="$1"; local subdir="$2"; local paradigm="$3"; shift 3
        export WANDB_RUN_NAME="$subdir"
        echo ""
        echo "╔══ Final eval [$label / $paradigm] ═════════════════════════════╗"
        printf "  output = %s\n" "$OUTPUT_ROOT/$subdir"
        echo "╚════════════════════════════════════════════════════════════════╝"
        _run_logged "$OUTPUT_ROOT/$subdir.log" \
            "$PYTHON" "$RUN_PY" \
            --mode final_eval \
            --config "$CONFIG" \
            "$@" \
            --override "experiment.output_dir=$OUTPUT_ROOT/$subdir" \
            --override "data.eval_paradigm=$paradigm" \
            --override "data.num_icl_examples=$FINAL_EVAL_NUM_ICL" \
            "${VLLM_OVERRIDES[@]}"
    }

    for _P in $_PARADIGMS; do
        # Baseline (base model, no LoRA adapter)
        if [ "$FINAL_EVAL_BASELINE" = "1" ]; then
            _final_eval_run "baseline" "final_eval_baseline_${_P}" "$_P" \
                --override "model.lora_adapter=null"
        fi
        # Last trained checkpoint
        if [ -z "$FINAL_CHECKPOINT" ]; then
            echo "[run_curriculum] WARNING: RUN_FINAL_EVAL=1 but no checkpoint found — skipping checkpoint eval ($_P)."
        else
            _final_eval_run "checkpoint" "final_eval_ckpt_${_P}" "$_P" \
                --checkpoint "$FINAL_CHECKPOINT"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  curriculum run complete"
printf "  summary  : %s\n" "$SUMMARY_FILE"
printf "  output   : %s\n" "$OUTPUT_ROOT"
printf "  final ckpt: %s\n" "${FINAL_CHECKPOINT:-<none>}"
echo "════════════════════════════════════════════════════════════════"

# Print per-stage summary table from curriculum_summary.jsonl
echo ""
"$PYTHON" - "$SUMMARY_FILE" <<'PYEOF'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except Exception:
    sys.exit(0)
if not rows:
    sys.exit(0)
hdr = f"{'Stage':<7} {'Eval':<6} {'BestPass':<10} {'Epochs':<7} {'Advance reason':<35} {'Gate'}"
print(hdr)
print("─" * len(hdr))
for r in rows:
    print(
        f"  {r.get('stage','?'):<5} {r.get('eval_stage','?'):<6}"
        f" {r.get('best_strict_gold_trace_pass',0.0):<10.3f}"
        f" {'?':<7}"
        f" {str(r.get('advance_reason',''))[:33]:<35}"
        f" {'✓' if r.get('gate_pass') else '✗'}"
    )
PYEOF
