"""VRAM / context length estimator for Tool-R0 GRPO training.

This script answers a single question before you launch a GRPO run:

    "Given my GPU(s), my (LoRA / full FT) config, and my batch / prompt sizes —
     what `max_completion_length` will fit, and what is the VRAM breakdown?"

It loads the model **config only** via `transformers.AutoConfig` (no weights),
derives every architectural quantity it needs (hidden_size, layers, attn heads,
head_dim, intermediate_size, vocab_size, ...), and then computes:

    1. Total parameter count from the architecture (no model download).
    2. Trainable parameter count
        - full FT  : all params
        - LoRA     : sum over target modules of  r * (d_in + d_out) * num_layers
        - QLoRA    : LoRA trainable + base in n-bit quantization
    3. Per-GPU static VRAM:
        - base weights (sharded if ZeRO-3 / quantized for QLoRA)
        - gradients   (sharded if ZeRO-2/3, only over trainable params)
        - optimizer states (sharded if ZeRO-2/3, zeroed if CPU offload)
        - runtime overhead + safety reserve
    4. Activation VRAM via the scaling model:
            M_act ~= k_act * B_dev * G * S * num_layers * hidden_size * bytes
        where  S = prompt_length + completion_length.
    5. Theoretical mode uses a default `k_act` (with / without grad ckpt).
       Calibrated mode reverse-solves `k_act` from a known peak VRAM measured
       on a real run, then re-applies it to the new config.

Usage
-----

Theoretical (Tool-R0 default config):

    python Usage_calculator/estimate_context_vram.py \\
        --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \\
        --gpu_memory_gb 40 --num_gpus 3 --zero_stage 2 \\
        --train_mode lora --lora_r 32 \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 16 \\
        --num_generations 4 \\
        --prompt_length 1024 --max_completion_length 4096 \\
        --gradient_checkpointing true

Calibrated (after one real run):

    python Usage_calculator/estimate_context_vram.py \\
        ... (same model + train_mode flags) ... \\
        --calibrate_peak_gb 32.5 \\
        --calibrate_prompt_length 1024 \\
        --calibrate_completion_length 3072 \\
        --calibrate_per_device_batch_size 1 \\
        --calibrate_num_generations 2

The script prints a structured report and exits 0 on success, 2 if the chosen
config does not fit within the GPU budget.

Notes on accounting conventions
-------------------------------
* AdamW optimizer states default to **8 bytes / trainable param** (fp32 m + v).
  This matches the Tool-R0 stack: `--dtype bfloat16` in GRPOConfig means params
  are stored bf16-only with no fp32 master copy. Override via
  `--optimizer_state_bytes_per_param` if you use a different optimizer.
* Gradients are stored in the training precision (bf16 = 2 bytes / param).
* ZeRO-2 shards gradients + optimizer states across `num_gpus`. Base weights
  stay replicated.
* ZeRO-3 shards base weights too (no benefit for LoRA — base is frozen).
* `gradient_accumulation_steps` does **not** scale activation VRAM (each
  micro-step's activations are freed before the next), so this estimator
  intentionally ignores it for VRAM but reports it for the effective batch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from typing import Iterable, Optional


PRECISION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

QUANT_BYTES_PER_PARAM = {2: 0.25, 3: 0.375, 4: 0.5, 8: 1.0}
QLORA_QUANT_OVERHEAD_BYTES = 0.4

DEFAULT_K_ACT_GRAD_CKPT = 2.0
DEFAULT_K_ACT_NO_CKPT = 18.0

QWEN_LIKE_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclasses.dataclass
class ModelDims:
    name: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean flag {value!r}")


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def _dims_from_config_dict(model_name: str, cfg: dict) -> ModelDims:
    """Project a raw HF `config.json` dict (or AutoConfig.to_dict()) to ModelDims.

    Supports multimodal wrappers via the standard `text_config` sub-dict.
    """
    text_cfg = cfg.get("text_config") or cfg

    hidden_size = text_cfg.get("hidden_size")
    num_layers = text_cfg.get("num_hidden_layers")
    num_heads = text_cfg.get("num_attention_heads")
    if hidden_size is None or num_layers is None or num_heads is None:
        raise SystemExit(
            f"Config {model_name!r} does not expose the expected transformer "
            "dims (hidden_size / num_hidden_layers / num_attention_heads)."
        )

    num_kv_heads = text_cfg.get("num_key_value_heads") or num_heads
    head_dim = text_cfg.get("head_dim") or (hidden_size // num_heads)
    intermediate_size = text_cfg.get("intermediate_size") or (4 * hidden_size)
    max_pos = text_cfg.get("max_position_embeddings") or 0
    vocab_size = text_cfg.get("vocab_size") or 0
    tie = bool(text_cfg.get("tie_word_embeddings", False))

    return ModelDims(
        name=model_name,
        hidden_size=int(hidden_size),
        num_hidden_layers=int(num_layers),
        intermediate_size=int(intermediate_size),
        num_attention_heads=int(num_heads),
        num_key_value_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        max_position_embeddings=int(max_pos),
        vocab_size=int(vocab_size),
        tie_word_embeddings=tie,
    )


def _load_raw_config_json(model_name_or_path: str) -> dict:
    """Fallback path: fetch raw `config.json` without instantiating any model class.

    Works even when the local `transformers` is older than the model architecture
    (for example Qwen3 on a transformers <4.51 install). Tries the local path
    first, then `huggingface_hub.hf_hub_download` for hub IDs.
    """
    import os
    local_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(local_path):
        with open(local_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Could not locate config.json for {model_name_or_path!r} locally and "
            "huggingface_hub is not installed for the fallback download."
        ) from exc

    cached = hf_hub_download(repo_id=model_name_or_path, filename="config.json")
    with open(cached, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_model_dims(model_name_or_path: str) -> ModelDims:
    """Read the HF config for `model_name_or_path` and project to ModelDims.

    Multimodal configs (e.g. Qwen3.5-VL) wrap the language model in
    `text_config`; we recurse into it so the same script works for VLMs too.

    Loading strategy:
      1. Try `transformers.AutoConfig.from_pretrained(..., trust_remote_code=True)`.
      2. If that fails because the local `transformers` doesn't recognize the
         `model_type` (typical when running this estimator on a slightly older
         env than the training one), fall back to reading raw `config.json`.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "transformers is required to introspect model configs. "
            "Install it with `pip install transformers`."
        ) from exc

    try:
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg_dict = cfg.to_dict()
    except (ValueError, KeyError) as err:
        print(
            f"[warn] AutoConfig could not parse {model_name_or_path!r} "
            f"({type(err).__name__}: {err}); falling back to raw config.json.",
            file=sys.stderr,
        )
        cfg_dict = _load_raw_config_json(model_name_or_path)

    return _dims_from_config_dict(model_name_or_path, cfg_dict)


def estimate_total_params(dims: ModelDims) -> int:
    """Closed-form param count for a Llama/Qwen3-style decoder.

    Counts: token embedding, per-layer (4 attn projections + 3 MLP projections
    + 2 RMSNorms), final RMSNorm, and lm_head if not tied. This is exact for
    Qwen3 / Llama-style architectures; minor terms (rotary buffers, biases on
    attention if `attention_bias=True`) are negligible for VRAM accounting.
    """
    H = dims.hidden_size
    Hi = dims.intermediate_size
    L = dims.num_hidden_layers
    H_q = dims.num_attention_heads * dims.head_dim
    H_kv = dims.num_key_value_heads * dims.head_dim
    V = dims.vocab_size

    embed = V * H
    per_layer = (
        H * H_q          # q_proj
        + H * H_kv       # k_proj
        + H * H_kv       # v_proj
        + H_q * H        # o_proj
        + 3 * H * Hi     # gate_proj + up_proj + down_proj
        + 2 * H          # input_layernorm + post_attention_layernorm
    )
    total = embed + L * per_layer + H  # final RMSNorm
    if not dims.tie_word_embeddings:
        total += V * H
    return int(total)


def lora_param_breakdown(
    dims: ModelDims, r: int, target_modules: Iterable[str]
) -> dict:
    """Per-target LoRA param count using the actual layer shapes.

    Rule per LoRA pair: ``r * (d_in + d_out)`` adapter params per layer (A is
    `d_in x r`, B is `r x d_out`, total `r * (d_in + d_out)`).
    """
    H = dims.hidden_size
    Hi = dims.intermediate_size
    H_q = dims.num_attention_heads * dims.head_dim
    H_kv = dims.num_key_value_heads * dims.head_dim
    L = dims.num_hidden_layers

    module_dims = {
        "q_proj":    (H,   H_q),
        "k_proj":    (H,   H_kv),
        "v_proj":    (H,   H_kv),
        "o_proj":    (H_q, H),
        "gate_proj": (H,   Hi),
        "up_proj":   (H,   Hi),
        "down_proj": (Hi,  H),
    }

    breakdown: dict[str, dict] = {}
    total = 0
    for name in target_modules:
        if name not in module_dims:
            print(
                f"[warn] target_module {name!r} not in known Qwen-like map; "
                f"ignored. Known: {sorted(module_dims)}",
                file=sys.stderr,
            )
            continue
        d_in, d_out = module_dims[name]
        per_layer = r * (d_in + d_out)
        layer_total = per_layer * L
        breakdown[name] = {
            "d_in": d_in,
            "d_out": d_out,
            "per_layer": per_layer,
            "all_layers": layer_total,
        }
        total += layer_total

    return {"total": total, "per_module": breakdown}


@dataclasses.dataclass
class StaticVram:
    base_weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    runtime_overhead_gb: float
    reserve_gb: float
    base_bytes_per_param: float
    base_replicated: bool

    @property
    def total_gb(self) -> float:
        return (
            self.base_weights_gb
            + self.gradients_gb
            + self.optimizer_states_gb
            + self.runtime_overhead_gb
            + self.reserve_gb
        )


def static_vram_per_gpu(
    args: argparse.Namespace,
    dims: ModelDims,
    total_params: int,
    trainable_params: int,
) -> StaticVram:
    """Per-GPU static VRAM breakdown given precision, ZeRO stage, offload."""
    weight_bytes = PRECISION_BYTES[args.precision]

    if args.train_mode == "qlora":
        base_bytes_per_param = (
            QUANT_BYTES_PER_PARAM[args.quant_bits] + QLORA_QUANT_OVERHEAD_BYTES
        )
    else:
        base_bytes_per_param = float(weight_bytes)

    base_total_bytes = total_params * base_bytes_per_param
    base_replicated = True
    if args.zero_stage == 3 and args.train_mode == "full":
        base_total_bytes /= max(args.num_gpus, 1)
        base_replicated = False

    grad_bytes = trainable_params * weight_bytes
    if args.zero_stage in (2, 3):
        grad_bytes /= max(args.num_gpus, 1)

    optim_bytes = trainable_params * args.optimizer_state_bytes_per_param
    if args.zero_stage in (2, 3):
        optim_bytes /= max(args.num_gpus, 1)
    if args.cpu_offload_optimizer:
        optim_bytes = 0.0

    return StaticVram(
        base_weights_gb=_gb(base_total_bytes),
        gradients_gb=_gb(grad_bytes),
        optimizer_states_gb=_gb(optim_bytes),
        runtime_overhead_gb=float(args.runtime_overhead_gb),
        reserve_gb=float(args.reserve_gb),
        base_bytes_per_param=base_bytes_per_param,
        base_replicated=base_replicated,
    )


def activation_vram_gb(
    dims: ModelDims,
    *,
    k_act: float,
    per_device_batch_size: int,
    num_generations: int,
    seq_len: int,
    bytes_per_activation: float,
) -> float:
    """Scaling-law estimate of activation VRAM per GPU."""
    return (
        k_act
        * per_device_batch_size
        * num_generations
        * seq_len
        * dims.num_hidden_layers
        * dims.hidden_size
        * bytes_per_activation
        / (1024 ** 3)
    )


def calibrate_k_act_from_peak(
    *,
    dims: ModelDims,
    static: StaticVram,
    bytes_per_activation: float,
    peak_gb: float,
    per_device_batch_size: int,
    num_generations: int,
    prompt_length: int,
    completion_length: int,
) -> tuple[float, float]:
    """Inverse-solve k_act from a known peak VRAM measurement.

    Returns ``(k_act, memory_per_token_bytes)`` where memory_per_token bundles
    ``k_act * num_layers * hidden_size * bytes`` so it can be cited directly.
    """
    activations_gb = peak_gb - static.total_gb
    if activations_gb <= 0:
        raise ValueError(
            f"Calibration peak {peak_gb:.2f} GB is at/below the static budget "
            f"({static.total_gb:.2f} GB). Re-check inputs (peak measurement, "
            "reserve_gb, runtime_overhead_gb)."
        )
    seq_len = prompt_length + completion_length
    denominator = (
        per_device_batch_size
        * num_generations
        * seq_len
        * dims.num_hidden_layers
        * dims.hidden_size
        * bytes_per_activation
    )
    if denominator <= 0:
        raise ValueError("Calibration token budget is zero — invalid inputs.")
    k_act = activations_gb * (1024 ** 3) / denominator
    mem_per_token = (
        k_act
        * dims.num_hidden_layers
        * dims.hidden_size
        * bytes_per_activation
    )
    return k_act, mem_per_token


def max_completion_for_budget(
    dims: ModelDims,
    *,
    k_act: float,
    free_gb: float,
    per_device_batch_size: int,
    num_generations: int,
    prompt_length: int,
    bytes_per_activation: float,
) -> int:
    """Largest `max_completion_length` that fits in `free_gb` of activation budget."""
    if free_gb <= 0:
        return 0
    bytes_per_token = (
        k_act
        * per_device_batch_size
        * num_generations
        * dims.num_hidden_layers
        * dims.hidden_size
        * bytes_per_activation
    )
    if bytes_per_token <= 0:
        return 0
    total_tokens = (free_gb * (1024 ** 3)) / bytes_per_token
    completion = int(math.floor(total_tokens - prompt_length))
    return max(completion, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model_name_or_path",
        required=True,
        help="HF model id or local path. Loaded via AutoConfig (no weights downloaded).",
    )

    p.add_argument("--gpu_memory_gb", type=float, default=40.0,
                   help="Per-GPU VRAM budget in GB.")
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Number of training GPUs (i.e. accelerate num_processes).")
    p.add_argument("--zero_stage", type=int, default=2, choices=(0, 2, 3),
                   help="DeepSpeed ZeRO stage (0 = none / DDP).")
    p.add_argument("--cpu_offload_optimizer", type=_str2bool, default=False,
                   help="If true, optimizer states are offloaded to CPU (zeroed from GPU budget).")
    p.add_argument("--precision", default="bf16", choices=tuple(PRECISION_BYTES),
                   help="Mixed-precision dtype for weights/grads/activations.")

    p.add_argument("--train_mode", default="lora", choices=("full", "lora", "qlora"),
                   help="Training regime.")
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (informational).")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (informational).")
    p.add_argument(
        "--lora_target_modules",
        type=_csv_list,
        default=list(QWEN_LIKE_TARGETS),
        help="Comma-separated LoRA target module names (Qwen-like 7 projections by default).",
    )
    p.add_argument("--quant_bits", type=int, default=4, choices=tuple(QUANT_BYTES_PER_PARAM),
                   help="QLoRA base-model quantization width (only used with --train_mode qlora).")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_generations", type=int, default=2,
                   help="GRPO rollouts per prompt (G in the activation model).")
    p.add_argument("--prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=3072)
    p.add_argument("--gradient_checkpointing", type=_str2bool, default=True)

    p.add_argument("--optimizer", default="adamw",
                   help="Free-form optimizer label (informational).")
    p.add_argument("--optimizer_state_bytes_per_param", type=float, default=8.0,
                   help="Per-trainable-param optimizer footprint. AdamW + bf16 = 8 (m+v in fp32).")
    p.add_argument("--runtime_overhead_gb", type=float, default=3.0,
                   help="Allocator fragmentation, NCCL, kernel buffers, comms staging.")
    p.add_argument("--reserve_gb", type=float, default=4.0,
                   help="Hard safety reserve subtracted before the activation budget.")

    p.add_argument("--k_act", type=float, default=None,
                   help="Override the activation scaling coefficient (else default by grad ckpt).")
    p.add_argument("--bytes_per_activation", type=float, default=None,
                   help="Override bytes per activation (default = precision bytes).")

    p.add_argument("--calibrate_peak_gb", type=float, default=None,
                   help="Known peak per-GPU VRAM from a real run, in GB. Triggers calibrated mode.")
    p.add_argument("--calibrate_prompt_length", type=int, default=None)
    p.add_argument("--calibrate_completion_length", type=int, default=None)
    p.add_argument("--calibrate_per_device_batch_size", type=int, default=None)
    p.add_argument("--calibrate_num_generations", type=int, default=None)

    p.add_argument("--json", action="store_true",
                   help="Emit a machine-readable JSON report instead of the text view.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero (2) if the chosen config does not fit in the GPU budget.",
    )

    return p.parse_args()


def _format_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def _print_text_report(
    args: argparse.Namespace,
    dims: ModelDims,
    total_params: int,
    trainable_params: int,
    lora_info: Optional[dict],
    static: StaticVram,
    k_act: float,
    bytes_per_activation: float,
    activation_gb: float,
    free_for_act_gb: float,
    fits: bool,
    suggested_completion: int,
    calibration: Optional[dict],
) -> None:
    line = "-" * 72
    print(line)
    print(f"Tool-R0 VRAM / context estimator")
    print(line)
    print(f"Model:              {dims.name}")
    print(
        f"  hidden={dims.hidden_size}  layers={dims.num_hidden_layers}  "
        f"heads={dims.num_attention_heads}  kv_heads={dims.num_key_value_heads}  "
        f"head_dim={dims.head_dim}"
    )
    print(
        f"  intermediate={dims.intermediate_size}  vocab={dims.vocab_size}  "
        f"max_pos={dims.max_position_embeddings}  tied_embed={dims.tie_word_embeddings}"
    )
    print(f"  total params (closed-form): {_format_params(total_params)}  ({total_params:,})")
    print()

    print(
        f"Training: mode={args.train_mode}  precision={args.precision}  "
        f"zero={args.zero_stage}  num_gpus={args.num_gpus}  "
        f"offload_optim={args.cpu_offload_optimizer}"
    )
    eff_batch = args.per_device_train_batch_size * args.num_generations * args.num_gpus * args.gradient_accumulation_steps
    print(
        f"  per_device_batch={args.per_device_train_batch_size}  "
        f"grad_accum={args.gradient_accumulation_steps}  "
        f"num_generations={args.num_generations}  "
        f"effective_optimizer_batch={eff_batch}"
    )
    print(
        f"  prompt_len={args.prompt_length}  max_completion={args.max_completion_length}  "
        f"S=prompt+completion={args.prompt_length + args.max_completion_length}"
    )
    print(
        f"  grad_ckpt={args.gradient_checkpointing}  "
        f"opt_state_bytes/param={args.optimizer_state_bytes_per_param}  "
        f"k_act={k_act:.3f}  bytes_per_activation={bytes_per_activation}"
    )
    print()

    print(f"Trainable params:   {_format_params(trainable_params)}  ({trainable_params:,})")
    if lora_info is not None:
        print(f"  LoRA r={args.lora_r}  alpha={args.lora_alpha}  dropout={args.lora_dropout}")
        for name, info in lora_info["per_module"].items():
            print(
                f"    {name:>9s}: d_in={info['d_in']:>5d}  d_out={info['d_out']:>5d}  "
                f"per_layer={info['per_layer']:>10,}  total={info['all_layers']:>12,}"
            )
        print(f"  LoRA target params total: {_format_params(lora_info['total'])}")
    print()

    print(f"Per-GPU static VRAM (GB):")
    print(
        f"  base_weights        {static.base_weights_gb:8.2f}   "
        f"({'replicated' if static.base_replicated else 'sharded ZeRO-3'}, "
        f"{static.base_bytes_per_param:.2f} B/param)"
    )
    print(f"  gradients           {static.gradients_gb:8.2f}   (sharded if ZeRO>=2)")
    if args.cpu_offload_optimizer:
        print(f"  optimizer_states    {static.optimizer_states_gb:8.2f}   (CPU offloaded)")
    else:
        print(
            f"  optimizer_states    {static.optimizer_states_gb:8.2f}   "
            f"(sharded if ZeRO>=2, {args.optimizer_state_bytes_per_param} B/param)"
        )
    print(f"  runtime_overhead    {static.runtime_overhead_gb:8.2f}")
    print(f"  reserve             {static.reserve_gb:8.2f}")
    print(f"  -----------------------------")
    print(f"  static total        {static.total_gb:8.2f}")
    print()

    print(f"Per-GPU dynamic VRAM (GB):")
    print(f"  activations         {activation_gb:8.2f}   "
          f"(B={args.per_device_train_batch_size} G={args.num_generations} "
          f"S={args.prompt_length + args.max_completion_length})")
    print()

    used = static.total_gb + activation_gb
    print(f"Per-GPU budget: target {args.gpu_memory_gb:.2f} GB")
    print(f"  used (static + activations)       {used:8.2f}")
    print(f"  free for activations after static {free_for_act_gb:8.2f}")
    print(f"  largest fitting max_completion    {suggested_completion}")
    print()

    if calibration is not None:
        print("Calibration (mode B):")
        print(f"  peak_gb               {calibration['peak_gb']:.2f}")
        print(
            f"  baseline B={calibration['per_device_batch_size']} "
            f"G={calibration['num_generations']} "
            f"S={calibration['prompt_length'] + calibration['completion_length']}"
        )
        print(f"  k_act (solved)        {calibration['k_act']:.4f}")
        print(f"  memory/token (B)      {calibration['memory_per_token']:.2f}")
        print()

    if fits:
        print("VERDICT: chosen config FITS in the GPU budget.")
    else:
        print("VERDICT: chosen config DOES NOT fit. Try one of:")
        print(f"  - reduce max_completion_length to {suggested_completion}")
        print(f"  - reduce num_generations or per_device_train_batch_size")
        print(f"  - enable --cpu_offload_optimizer (frees {static.optimizer_states_gb:.2f} GB)")
        if args.train_mode == "full" and args.zero_stage != 3:
            print(f"  - set --zero_stage 3 (shards base weights)")
        if not args.gradient_checkpointing:
            print(f"  - enable --gradient_checkpointing true (large k_act savings)")
    print(line)


def main() -> int:
    args = parse_args()

    if args.bytes_per_activation is None:
        args.bytes_per_activation = float(PRECISION_BYTES[args.precision])
    bytes_per_activation = float(args.bytes_per_activation)

    if args.k_act is None:
        args.k_act = (
            DEFAULT_K_ACT_GRAD_CKPT
            if args.gradient_checkpointing
            else DEFAULT_K_ACT_NO_CKPT
        )

    dims = get_model_dims(args.model_name_or_path)
    total_params = estimate_total_params(dims)

    lora_info: Optional[dict] = None
    if args.train_mode in ("lora", "qlora"):
        lora_info = lora_param_breakdown(dims, args.lora_r, args.lora_target_modules)
        trainable_params = lora_info["total"]
    else:
        trainable_params = total_params

    static = static_vram_per_gpu(args, dims, total_params, trainable_params)

    calibration: Optional[dict] = None
    k_act = float(args.k_act)
    if args.calibrate_peak_gb is not None:
        cal_prompt = args.calibrate_prompt_length or args.prompt_length
        cal_completion = (
            args.calibrate_completion_length or args.max_completion_length
        )
        cal_b = args.calibrate_per_device_batch_size or args.per_device_train_batch_size
        cal_g = args.calibrate_num_generations or args.num_generations
        k_act, mem_per_token = calibrate_k_act_from_peak(
            dims=dims,
            static=static,
            bytes_per_activation=bytes_per_activation,
            peak_gb=args.calibrate_peak_gb,
            per_device_batch_size=cal_b,
            num_generations=cal_g,
            prompt_length=cal_prompt,
            completion_length=cal_completion,
        )
        calibration = {
            "peak_gb": args.calibrate_peak_gb,
            "prompt_length": cal_prompt,
            "completion_length": cal_completion,
            "per_device_batch_size": cal_b,
            "num_generations": cal_g,
            "k_act": k_act,
            "memory_per_token": mem_per_token,
        }

    seq_len = args.prompt_length + args.max_completion_length
    activation_gb = activation_vram_gb(
        dims,
        k_act=k_act,
        per_device_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        seq_len=seq_len,
        bytes_per_activation=bytes_per_activation,
    )

    free_for_act_gb = max(args.gpu_memory_gb - static.total_gb, 0.0)
    fits = (static.total_gb + activation_gb) <= args.gpu_memory_gb

    suggested_completion = max_completion_for_budget(
        dims,
        k_act=k_act,
        free_gb=free_for_act_gb,
        per_device_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        prompt_length=args.prompt_length,
        bytes_per_activation=bytes_per_activation,
    )

    if args.json:
        report = {
            "model": dataclasses.asdict(dims),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora": lora_info,
            "static_vram_gb": dataclasses.asdict(static),
            "activation_vram_gb": activation_gb,
            "k_act_used": k_act,
            "bytes_per_activation": bytes_per_activation,
            "free_for_activations_gb": free_for_act_gb,
            "max_completion_length_fit": suggested_completion,
            "fits_budget": fits,
            "calibration": calibration,
            "args": vars(args),
        }
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(
            args=args,
            dims=dims,
            total_params=total_params,
            trainable_params=trainable_params,
            lora_info=lora_info,
            static=static,
            k_act=k_act,
            bytes_per_activation=bytes_per_activation,
            activation_gb=activation_gb,
            free_for_act_gb=free_for_act_gb,
            fits=fits,
            suggested_completion=suggested_completion,
            calibration=calibration,
        )

    if args.strict and not fits:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
