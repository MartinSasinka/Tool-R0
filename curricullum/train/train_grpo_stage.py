#!/usr/bin/env python3
"""
train_grpo_stage.py

Train one curriculum stage with LoRA + GRPO on NESTFUL JSONL data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import yaml  # imported before torch: needed to choose the CUDA allocator config below

# Match CUDA device ids to `nvidia-smi` ordering so CUDA_VISIBLE_DEVICES picks the
# intended physical GPUs (default FASTEST_FIRST can map an id onto the display GPU).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _peek_vllm_flags(argv) -> Tuple[bool, bool]:
    """Return ``(use_vllm, sleep_mode)`` read from the ``--config`` YAML.

    Must run BEFORE ``import torch``: torch caches PYTORCH_CUDA_ALLOC_CONF on first
    parse (during the trl/deepspeed import chain), and vLLM sleep mode (CuMemAllocator
    / torch.cuda.MemPool) is incompatible with expandable_segments (vllm#14189,
    pytorch#147851). We also decide the vLLM attention backend up front.
    """
    cfg_path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            break
        if a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
            break
    if not cfg_path or not os.path.isfile(cfg_path):
        return False, False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        grpo = cfg.get("grpo", {}) or {}
        return bool(grpo.get("use_vllm", False)), bool(grpo.get("vllm_enable_sleep_mode", False))
    except Exception:
        return False, False


_USE_VLLM, _VLLM_SLEEP = _peek_vllm_flags(sys.argv)

# vLLM colocate generation: force its bundled FlashAttention (FA2, supported on A100
# SM 8.0) instead of letting vLLM enumerate/import FlashInfer. This env's flashinfer
# (0.5.3) and flashinfer-cubin (0.6.6) are mismatched and crash on import; FLASH_ATTN
# avoids that path, and DISABLE_VERSION_CHECK keeps any stray flashinfer import (e.g.
# the sampler) from hard-failing. Both are setdefault so the env can still override.
if _USE_VLLM:
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

# Choose the CUDA allocator config BEFORE torch initializes (it caches the value):
# - vLLM sleep mode: expandable_segments MUST be off (CuMemAllocator/MemPool conflict).
# - otherwise: expandable_segments reduces "reserved but unallocated" fragmentation OOMs.
if _VLLM_SLEEP:
    for _var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        _val = os.environ.get(_var, "")
        if "expandable_segments" not in _val:
            continue
        _kept = ",".join(p for p in _val.split(",") if p.strip() and "expandable_segments" not in p)
        if _kept:
            os.environ[_var] = _kept
        else:
            os.environ.pop(_var, None)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            "[train] vLLM sleep mode: expandable_segments disabled before torch init",
            file=sys.stderr,
        )
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

# DeepSpeed JIT-compiles its CPU/Fused Adam ops and caches the .so under
# ~/.cache/torch_extensions/py310_cu128/, a path that does NOT encode the torch
# version. After a torch up/downgrade the cached .so links against the old libtorch
# ABI and fails to load ("undefined symbol: ..._incref_pyobject..."). Key the cache by
# torch build so any torch change forces a clean recompile against the current torch.
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    os.path.join(os.path.expanduser("~/.cache/torch_extensions"), f"toolr0_{torch.__version__}"),
)

from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from grpo_processing import (  # noqa: E402
    _maybe_merge_lora_adapter,
    apply_grpo_peft_workarounds,
    load_grpo_tokenizer,
)
from prepare_dataset import (  # noqa: E402
    filter_by_tokenizer_length,
    load_nestful_jsonl,
    records_to_dataset,
)
from rewards_nestful import DEFAULT_WEIGHTS, build_curriculum_reward_func, get_batch_stats_snapshot  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore


def is_main_process() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_main_process():
        print(f"[train] {msg}", file=sys.stderr)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adapter_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def resolve_attn_implementation(requested: str) -> str:
    if requested != "flash_attention_2":
        return requested
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        log(
            "=" * 70 + "\n"
            "  WARNING: flash_attention_2 requested but NOT installed.\n"
            "  Falling back to 'sdpa', which MATERIALIZES the full attention\n"
            "  mask [batch, heads, q, kv]. For long prompts (>~3k tokens) this\n"
            "  can allocate tens of GiB and OOM intermittently during generation.\n"
            "  Fix: pip install flash-attn --no-build-isolation\n"
            "  Or lower grpo.max_prompt_length in the config.\n"
            + "=" * 70
        )
        return "sdpa"


_VLLM_KEYS = (
    "vllm_mode",
    "vllm_tensor_parallel_size",
    "vllm_gpu_memory_utilization",
    "vllm_enable_sleep_mode",
    "vllm_max_model_length",
    "vllm_model_impl",
)

def _vllm_available() -> bool:
    try:
        from trl.import_utils import is_vllm_available  # type: ignore

        return bool(is_vllm_available())
    except Exception:
        try:
            import vllm  # noqa: F401

            return True
        except Exception:
            return False


def resolve_vllm(grpo_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[int]]:
    """Resolve vLLM colocate settings for GRPO generation.

    Returns ``(vllm_kwargs, fallback_grad_accum)``:
    - If ``use_vllm`` is requested AND vLLM is importable, ``vllm_kwargs`` carries
      ``use_vllm=True`` plus the ``vllm_*`` knobs, and ``fallback_grad_accum`` is None.
    - If vLLM is unavailable (or not requested), ``vllm_kwargs`` is empty. When it was
      *requested* but unavailable we also return a reduced ``gradient_accumulation_steps``
      so the HF ``model.generate()`` fallback halves the generation batch and does not OOM.
    """
    if not bool(grpo_cfg.get("use_vllm", False)):
        return {}, None

    if not _vllm_available():
        log(
            "=" * 70 + "\n"
            "  WARNING: use_vllm=true but vLLM is not importable.\n"
            "  Falling back to HF model.generate() and HALVING the generation\n"
            "  batch (gradient_accumulation_steps -> 2) to avoid the prefill OOM.\n"
            "  Fix: pip install \"vllm==0.12.0\" in this env (TRL supports <=0.12.0).\n"
            + "=" * 70
        )
        return {}, 2

    vllm_kwargs: Dict[str, Any] = {"use_vllm": True}
    for key in _VLLM_KEYS:
        if key in grpo_cfg and grpo_cfg[key] is not None:
            vllm_kwargs[key] = grpo_cfg[key]
    log(
        "vLLM colocate enabled for GRPO generation: "
        + ", ".join(f"{k}={v}" for k, v in vllm_kwargs.items())
    )
    return vllm_kwargs, None


def estimate_optimizer_steps(
    dataset_size: int,
    per_device_batch: int,
    grad_accum: int,
    num_train_epochs: float,
    max_steps: Optional[int],
) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    denom = max(1, per_device_batch * grad_accum * world_size)
    steps_per_epoch = max(1, math.ceil(dataset_size / denom))
    if max_steps is not None and max_steps > 0:
        return max_steps
    return max(1, math.ceil(steps_per_epoch * num_train_epochs))


def build_grpo_config(grpo_kwargs: Dict[str, Any]) -> Tuple[GRPOConfig, Dict[str, Any]]:
    kwargs = dict(grpo_kwargs)
    dropped: Dict[str, Any] = {}
    if int(kwargs.get("save_steps", 1)) == 0:
        kwargs["save_strategy"] = "no"
        kwargs.pop("save_steps", None)

    optional_keys = [
        "max_prompt_length",
        "beta",
        "temperature",
        "top_p",
        "log_completions",
        "loss_type",
    ]
    while True:
        try:
            return GRPOConfig(**kwargs), dropped
        except TypeError as e:
            removed = False
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m and m.group(1) in kwargs:
                key = m.group(1)
                dropped[key] = kwargs.pop(key)
                log(f"GRPOConfig: dropped unsupported {key} ({e})")
                removed = True
            if not removed:
                for key in optional_keys:
                    if key in kwargs:
                        dropped[key] = kwargs.pop(key)
                        log(f"GRPOConfig: dropped unsupported {key}")
                        removed = True
                        break
            if not removed:
                raise


def build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    lora = cfg.get("lora", {})
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GRPO train one curriculum stage")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["stage_1", "stage_2", "stage_3", "stage_4", "stage_5", "stage_6"])
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--previous_adapter", default="none")
    ap.add_argument("--wandb_project", default="nestful-curriculum-grpo")
    ap.add_argument("--wandb_run_name", required=True)
    ap.add_argument("--wandb_run_group", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge_adapter", action="store_true")
    ap.add_argument(
        "--replay_jsonl",
        default=None,
        help="Optional JSONL of replay rows (tool_r0 format) appended to training data",
    )
    ap.add_argument("--num_generations", type=int, default=None)
    ap.add_argument("--per_device_train_batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--lora_r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="Override; <=0 uses num_train_epochs")
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument(
        "--training_format",
        default=os.environ.get("TRAINING_FORMAT", "json"),
        choices=["json", "tool_r0"],
        help="json=legacy JSON plan; tool_r0=eval-aligned Tool-R0 tags + IBM exec reward",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    stage_cfg = cfg["stages"][args.stage]
    stage_name = stage_cfg["name"]
    data_path = args.data_path or stage_cfg["data_path"]
    max_completion_length = int(stage_cfg["max_completion_length"])
    num_calls = int(stage_cfg.get("num_calls", args.stage.split("_")[1]))

    model_name = args.model_name or cfg["model"]["name"]
    grpo_cfg = cfg.get("grpo", {})
    train_cfg = cfg.get("training", {})

    num_generations = args.num_generations or int(grpo_cfg.get("num_generations", 2))
    per_device_batch = args.per_device_train_batch_size or int(grpo_cfg.get("per_device_train_batch_size", 1))
    grad_accum = args.gradient_accumulation_steps or int(grpo_cfg.get("gradient_accumulation_steps", 8))
    learning_rate = args.learning_rate or float(grpo_cfg.get("learning_rate", 5e-6))
    # Per-stage override wins over the global grpo default. Lowering this for
    # long stages bounds the sdpa attention-mask size (fallback when flash-attn
    # is unavailable), at the cost of dropping over-budget prompts.
    max_prompt_length = int(stage_cfg.get("max_prompt_length", grpo_cfg.get("max_prompt_length", 4096)))
    num_train_epochs = args.num_train_epochs
    if num_train_epochs is None:
        num_train_epochs = float(train_cfg.get("num_train_epochs", 1.0))

    max_steps_raw = args.max_steps
    if max_steps_raw is None:
        max_steps_raw = train_cfg.get("max_steps")
    if max_steps_raw is None:
        env_ms = os.environ.get("MAX_STEPS", "").strip()
        if env_ms:
            max_steps_raw = env_ms
    max_steps: Optional[int] = None
    if max_steps_raw is not None and str(max_steps_raw).lower() not in ("null", "none", ""):
        try:
            if int(max_steps_raw) > 0:
                max_steps = int(max_steps_raw)
        except (TypeError, ValueError):
            pass

    lora_cfg = cfg.get("lora", {})
    if args.lora_r is not None:
        lora_cfg = {**lora_cfg, "r": args.lora_r}
    if args.lora_alpha is not None:
        lora_cfg = {**lora_cfg, "alpha": args.lora_alpha}
    cfg = {**cfg, "lora": lora_cfg}

    os.makedirs(args.output_dir, exist_ok=True)

    if adapter_exists(args.output_dir) and not args.overwrite:
        log(f"checkpoint already exists at {args.output_dir}; pass --overwrite to replace")
        sys.exit(1)

    if not os.path.isfile(data_path):
        log(f"data file missing: {data_path}")
        sys.exit(1)

    tokenizer = load_grpo_tokenizer(model_name)
    training_format = args.training_format
    if training_format == "tool_r0":
        from prepare_dataset_toolr0 import (  # noqa: E402
            load_toolr0_jsonl,
            records_to_toolr0_dataset,
        )
        from rewards_toolr0_exec import (  # noqa: E402
            DEFAULT_WEIGHTS as TOOLR0_WEIGHTS,
            build_toolr0_reward_func,
        )

        records, load_stats = load_toolr0_jsonl(
            data_path,
            default_num_calls=num_calls,
            max_prompt_tokens=max_prompt_length,
            skip_over_budget=True,
        )
        if args.replay_jsonl and os.path.isfile(args.replay_jsonl):
            replay_records, replay_stats = load_toolr0_jsonl(
                args.replay_jsonl,
                default_num_calls=None,
                max_prompt_tokens=max_prompt_length,
                skip_over_budget=True,
            )
            log(f"replay: appending {len(replay_records)} rows from {args.replay_jsonl}")
            records = records + replay_records
        # Filter the COMBINED set (main + replay) by true tokenizer length. Replay rows
        # must be filtered too: load_toolr0_jsonl does NOT enforce max_prompt_tokens, and
        # an unfiltered multi-turn prefix with a large baked tool result can blow past
        # vllm_max_model_length and hard-crash generation (vLLM "decoder prompt longer
        # than maximum model length" ValueError; TRL no longer truncates — issue #4358).
        records, skipped_tok = filter_by_tokenizer_length(records, tokenizer, max_prompt_length)
        train_dataset = records_to_toolr0_dataset(records)
        reward_weights = cfg.get("reward_weights") or TOOLR0_WEIGHTS
        reward_fn = build_toolr0_reward_func(reward_weights)
    else:
        records, load_stats = load_nestful_jsonl(
            data_path,
            default_num_calls=num_calls,
            max_prompt_tokens=max_prompt_length,
            skip_over_budget=True,
        )
        records, skipped_tok = filter_by_tokenizer_length(records, tokenizer, max_prompt_length)
        train_dataset = records_to_dataset(records)
        reward_weights = cfg.get("reward_weights") or DEFAULT_WEIGHTS
        reward_fn = build_curriculum_reward_func(reward_weights)

    est_steps = estimate_optimizer_steps(
        len(train_dataset), per_device_batch, grad_accum, num_train_epochs, max_steps
    )
    log(
        f"stage={stage_name} samples={len(train_dataset)} "
        f"num_train_epochs={num_train_epochs} est_optimizer_steps={est_steps}"
    )

    wandb_run = None
    wandb_run_id = None
    if is_main_process() and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_run_group or os.environ.get("WANDB_RUN_GROUP"),
            config={
                "stage": args.stage,
                "stage_name": stage_name,
                "model_name": model_name,
                "data_path": data_path,
                "previous_adapter": args.previous_adapter,
                "num_calls": num_calls,
                "max_completion_length": max_completion_length,
                "max_prompt_length": max_prompt_length,
                "num_generations": num_generations,
                "num_train_epochs": num_train_epochs,
                "estimated_optimizer_steps": est_steps,
                "lora": lora_cfg,
                "grpo": grpo_cfg,
            },
        )
        wandb_run_id = wandb_run.id
        wandb.log(
            {
                "curriculum/stage": args.stage,
                "curriculum/num_calls": num_calls,
                "curriculum/max_completion_length": max_completion_length,
                "curriculum/num_generations": num_generations,
                "curriculum/num_train_epochs": num_train_epochs,
                "curriculum/estimated_optimizer_steps": est_steps,
                "curriculum/train_samples": len(train_dataset),
            }
        )
        log(f"wandb run started: {args.wandb_run_name}")

    attn_impl = resolve_attn_implementation(cfg["model"].get("attn_implementation", "sdpa"))
    dtype = torch.bfloat16

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        logging_steps=int(grpo_cfg.get("logging_steps", 1)),
        save_steps=int(grpo_cfg.get("save_steps", 0)),
        report_to="wandb" if wandb is not None else "none",
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
        loss_type=str(grpo_cfg.get("loss_type", "grpo")),
        log_completions=bool(grpo_cfg.get("log_completions", True)),
        bf16=True,
        num_train_epochs=num_train_epochs,
    )
    if grpo_cfg.get("num_completions_to_print") is not None:
        grpo_kwargs["num_completions_to_print"] = int(grpo_cfg["num_completions_to_print"])
    if max_steps is not None:
        grpo_kwargs["max_steps"] = max_steps

    for key in ("beta", "temperature", "top_p"):
        if key in grpo_cfg and grpo_cfg[key] is not None:
            grpo_kwargs[key] = grpo_cfg[key]

    # vLLM colocate generation (replaces HF model.generate() for rollouts). If
    # requested but vLLM is missing, fall back to HF generate with a smaller
    # generation batch so the prefill does not OOM at the 40 GB edge.
    vllm_kwargs, fallback_grad_accum = resolve_vllm(grpo_cfg)
    if vllm_kwargs:
        grpo_kwargs.update(vllm_kwargs)
    elif fallback_grad_accum is not None and fallback_grad_accum < grad_accum:
        log(
            f"vLLM fallback: gradient_accumulation_steps {grad_accum} -> {fallback_grad_accum}"
        )
        grad_accum = fallback_grad_accum
        grpo_kwargs["gradient_accumulation_steps"] = grad_accum

    try:
        training_args, dropped_grpo_kwargs = build_grpo_config(grpo_kwargs)
    except TypeError as e:
        log(f"GRPOConfig failed: {e}")
        raise
    if dropped_grpo_kwargs:
        log(
            "GRPOConfig: prompt/completion limits still enforced in prepare_dataset; "
            f"dropped TRL kwargs={list(dropped_grpo_kwargs.keys())}"
        )
    apply_grpo_peft_workarounds(training_args)

    previous = (args.previous_adapter or "none").strip()
    use_previous = previous.lower() not in ("none", "")

    peft_config = None
    model = None

    model_init_kwargs = dict(
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if use_previous:
        if not adapter_exists(previous):
            log(f"previous_adapter not found: {previous}")
            sys.exit(1)
        log(f"loading base={model_name} + adapter={previous}")
        base = AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs)
        model = PeftModel.from_pretrained(base, previous, is_trainable=True)
    else:
        log(f"loading base={model_name} with fresh LoRA r={lora_cfg.get('r')}")
        peft_config = build_lora_config(cfg)
        training_args.model_init_kwargs = model_init_kwargs

    trainer_kwargs = dict(
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
    )
    if model is not None:
        trainer_kwargs["model"] = model
        trainer_kwargs["peft_config"] = None
    else:
        trainer_kwargs["model"] = model_name
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    t0 = datetime.now(timezone.utc)
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    log(f"saved LoRA adapter -> {args.output_dir}")

    if args.merge_adapter:
        if not adapter_exists(args.output_dir):
            log("--merge_adapter requested but no adapter saved in output_dir")
            sys.exit(1)
        log("merging adapter into base weights (--merge_adapter); backup in _lora_adapter")
        merged = _maybe_merge_lora_adapter(args.output_dir, model_name)
        if not merged:
            log("--merge_adapter failed: no adapter found or merge dependency missing")
            sys.exit(1)

    t1 = datetime.now(timezone.utc)
    final_reward = None
    if hasattr(train_result, "metrics") and train_result.metrics:
        final_reward = train_result.metrics.get("train_reward")

    summary = {
        "model_name": model_name,
        "previous_adapter": previous if use_previous else "none",
        "output_dir": args.output_dir,
        "data_path": data_path,
        "stage": args.stage,
        "stage_name": stage_name,
        "num_training_samples": len(train_dataset),
        "max_completion_length": max_completion_length,
        "max_prompt_length": max_prompt_length,
        "dropped_grpo_kwargs": dropped_grpo_kwargs,
        "num_generations": num_generations,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        "estimated_optimizer_steps": est_steps,
        "lora_config": lora_cfg,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_run_id": wandb_run_id,
        "wandb_run_group": args.wandb_run_group or os.environ.get("WANDB_RUN_GROUP"),
        "load_stats": load_stats,
        "skipped_tokenizer_budget": skipped_tok,
        "started_at": t0.isoformat(),
        "finished_at": t1.isoformat(),
        "final_train_reward": final_reward,
        "merge_adapter": args.merge_adapter,
        "reward_stats": get_batch_stats_snapshot(),
    }
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"wrote {summary_path}")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
