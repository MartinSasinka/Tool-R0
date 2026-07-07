#!/usr/bin/env python3
"""NESTFUL MT-GRPO Minimal — standalone entry point.

Modes:
    python run.py --mode smoke       --config config.yaml
    python run.py --mode rollout_eval --config config.yaml [--checkpoint PATH]
    python run.py --mode train        --config config.yaml
    python run.py --mode final_eval   --config config.yaml --checkpoint PATH

This folder is a self-contained experimental artifact. It imports nothing from
curricullum/ or nestful_evaluation/. The only external dependencies are the
listed Python packages, the dataset path, and (optionally) the IBM/NESTFUL
executable-functions directory used by the full executor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make sibling modules importable whether run as `python run.py` or `-m`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import yaml  # noqa: E402

from data import load_tasks, load_tasks_mixed  # noqa: E402
from executor import IBMFunctionRegistry, detect_ibm_functions_dir  # noqa: E402
from rollout import run_episode  # noqa: E402
from reward import strict_gold_trace_reward, compute_gold_observations  # noqa: E402
from metrics import (  # noqa: E402
    compute_nestful_official_metrics,
    compute_paper_metrics,
    aggregate_final_eval,
)


# =====================================================================
#  W&B helpers — opt-in via WANDB_PROJECT env var
# =====================================================================

def _wandb_init(mode: str, config: dict, stage: int | None = None,
                epoch: int | None = None, checkpoint: str | None = None):
    """Initialise a W&B run if WANDB_PROJECT is set, otherwise return None.

    Env vars consumed (all optional):
        WANDB_PROJECT   – required to enable; e.g. "nestful-mtgrpo"
        WANDB_RUN_NAME  – human-readable run name
        WANDB_RUN_GROUP – group tag for grouping curriculum stages
        WANDB_ENTITY    – W&B team/entity (defaults to personal account)
        WANDB_API_KEY   – W&B API key (usually pre-exported in the shell)
    """
    project = os.environ.get("WANDB_PROJECT", "")
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed — skipping. pip install wandb", flush=True)
        return None
    try:
        run_name = os.environ.get("WANDB_RUN_NAME") or (
            f"{mode}-stage{stage or '?'}" + (f"-e{epoch}" if epoch is not None else "")
        )
        wcfg = {
            "mode": mode,
            "stage": stage,
            "epoch": epoch,
            "model": config.get("model", {}).get("base_model", "?"),
            "train_stage": config.get("data", {}).get("train_stage"),
            "eval_stage": config.get("data", {}).get("eval_stage"),
            "num_generations": config.get("generation", {}).get("num_generations"),
            "lora_r": config.get("finetuning", {}).get("lora_r"),
            "kl_beta": config.get("training", {}).get("kl_beta"),
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "use_vllm": config.get("hardware", {}).get("use_vllm", False),
            "checkpoint": checkpoint,
        }
        run = wandb.init(
            project=project,
            name=run_name,
            group=os.environ.get("WANDB_RUN_GROUP") or None,
            entity=os.environ.get("WANDB_ENTITY") or None,
            config=wcfg,
            reinit=True,
        )
        print(f"[wandb] run started: {run.url}", flush=True)
        return run
    except Exception as exc:
        print(f"[wandb] init failed (non-fatal): {exc}", flush=True)
        return None


def _wandb_log_eval(wandb_run, metrics: dict, prefix: str = "eval") -> None:
    """Log eval metrics dict to W&B as a single step + summary."""
    if wandb_run is None:
        return
    try:
        import wandb
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()
                   if isinstance(v, (int, float, bool))}
        wandb_run.log(payload)
        for k, v in payload.items():
            wandb_run.summary[k] = v
    except Exception:
        pass


def _wandb_finish(wandb_run) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception:
        pass


# =====================================================================
#  Setup helpers
# =====================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def print_versions() -> None:
    print("=" * 60)
    print("Environment")
    print("-" * 60)
    mods = ["torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes"]
    for m in mods:
        try:
            mod = __import__(m)
            print(f"  {m:14s} {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"  {m:14s} <not installed>")
    try:
        import torch
        print(f"  cuda available {torch.cuda.is_available()}")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"  gpu count      {n}")
        for i in range(n):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    except Exception:
        print("  torch/cuda     <unavailable>")
    print("=" * 60)


def _resolve_path(path: str) -> str:
    """Resolve a config path relative to this artifact folder (_HERE).

    Accepts both short paths (``data/...``) and legacy Tool-R0 paths
    (``experiments/nestful_mtgrpo_minimal/data/...``).
    """
    if not path or os.path.isabs(path):
        return path
    p = path.replace("\\", "/")
    legacy = "experiments/nestful_mtgrpo_minimal/"
    if p.startswith(legacy):
        p = p[len(legacy):]
    return os.path.normpath(os.path.join(_HERE, p))


def _normalize_config_paths(config: dict) -> None:
    """Resolve all filesystem paths in config against the artifact root."""
    paths = config.get("paths", {})
    for key in list(paths.keys()):
        val = paths.get(key)
        if isinstance(val, str) and val:
            paths[key] = _resolve_path(val)
    exp = config.get("experiment", {})
    if isinstance(exp.get("output_dir"), str):
        exp["output_dir"] = _resolve_path(exp["output_dir"])
    mcfg = config.get("model", {})
    for key in ("output_adapter_dir", "lora_adapter"):
        val = mcfg.get(key)
        if isinstance(val, str) and val:
            mcfg[key] = _resolve_path(val)


def _ensure_local_data(config: dict) -> None:
    """If configured data paths don't exist, try to copy from known fallback locations.

    This makes the folder self-contained after a fresh clone: the first run
    populates data/ from wherever the original files live in the repo.
    Missing files are reported as warnings; execution continues (executor will
    fall back to gold_replay mode if IBM functions are absent).
    """
    import shutil

    repo_root = os.getcwd()

    # ── nestful_data.jsonl ──────────────────────────────────────────────────
    nestful_dst = config.get("paths", {}).get("full_nestful_jsonl", "")
    if nestful_dst and not os.path.isfile(nestful_dst):
        candidates = [
            os.path.join(_HERE, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"),
            os.path.join(repo_root, "eval", "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"),
            os.path.join(repo_root, "nestful_repo", "data_v2", "nestful_data.jsonl"),
        ]
        for src in candidates:
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(nestful_dst), exist_ok=True)
                shutil.copy2(src, nestful_dst)
                print(f"[data] copied nestful_data.jsonl: {src} -> {nestful_dst}")
                break
        else:
            print(f"[data] WARNING: full_nestful_jsonl not found at '{nestful_dst}' "
                  "and no fallback source located. final_eval will be skipped.")

    # ── IBM executable functions ────────────────────────────────────────────
    ibm_dst = config.get("paths", {}).get("ibm_functions_dir", "")
    if ibm_dst and not os.path.isdir(ibm_dst):
        candidates = [
            os.path.join(_HERE, "data", "NESTFUL-main", "data_v2", "executable_functions"),
            os.path.join(repo_root, "eval", "data", "NESTFUL-main", "data_v2", "executable_functions"),
            os.path.join(repo_root, "nestful_repo", "data_v2", "executable_functions"),
        ]
        for src in candidates:
            if os.path.isdir(src):
                os.makedirs(os.path.dirname(ibm_dst), exist_ok=True)
                shutil.copytree(src, ibm_dst)
                print(f"[data] copied IBM functions: {src} -> {ibm_dst}")
                break
        else:
            print(f"[data] WARNING: ibm_functions_dir not found at '{ibm_dst}' "
                  "and no fallback source located. executor will use gold_replay mode.")


def load_tokenizer_only(config: dict):
    """Load only the tokenizer (no model weights). Used by vLLM eval modes where
    the HF model is not needed — vLLM handles generation internally."""
    from transformers import AutoTokenizer
    base = config["model"]["base_model"]
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[model] tokenizer loaded: {base}")
    return tok


def build_registry(config: dict):
    paths = config.get("paths", {})
    funcs_dir = detect_ibm_functions_dir(
        explicit=paths.get("ibm_functions_dir"),
        repo_root=_HERE,
    )
    if funcs_dir is None:
        print("[executor] IBM functions dir NOT found -> gold_replay mode "
              "(win_rate / solution_equivalent_pass will be LIMITED).")
        return None
    reg = IBMFunctionRegistry(funcs_dir)
    print(f"[executor] IBM functions dir: {funcs_dir} (available={reg.available})")
    return reg


def _parse_gpu_list(value) -> list:
    """Parse hardware.rollout_data_parallel_gpus into a list of int GPU ids.

    Accepts a list ([1,2,3]), a CSV string ("1,2,3"), a single int, or None/empty
    (→ []). Used to opt into data-parallel rollouts; empty means the feature is off.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, int):
        items = [value]
    else:
        items = [p for p in str(value).replace(";", ",").split(",")]
    out = []
    for it in items:
        s = str(it).strip()
        if s == "":
            continue
        try:
            out.append(int(s))
        except ValueError:
            continue
    return out


def _truthy(value) -> bool:
    """Interpret a config/override value as a boolean (handles '1'/'true'/'yes')."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_str_list(value) -> list:
    """Parse a list or CSV string into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace(";", ",").split(",")
    return [str(it).strip() for it in items if str(it).strip()]


def _parse_float_list(value) -> list:
    """Parse a list or CSV string into a list of floats (skips bad entries)."""
    out = []
    for it in _parse_str_list(value):
        try:
            out.append(float(it))
        except ValueError:
            continue
    return out


def load_model_and_tokenizer(config: dict, checkpoint: str | None, for_training: bool):
    """Load base model (+ optional LoRA adapter) for eval, or a fresh PEFT model
    for training. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = config.get("model", {})
    hw = config.get("hardware", {})
    ft = config.get("finetuning", {})
    base = mcfg["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method = ft.get("method", "qlora")
    dtype = torch.bfloat16 if hw.get("bf16", True) else torch.float16
    # hardware.load_in_4bit (CLI --override) takes precedence over finetuning.load_in_4bit
    # so eval can disable 4-bit on large GPUs without editing config.yaml.
    if "load_in_4bit" in hw:
        load_4bit_pref = bool(hw["load_in_4bit"])
    else:
        load_4bit_pref = bool(ft.get("load_in_4bit", True))
    use_4bit = load_4bit_pref and method == "qlora"
    if use_4bit:
        print("[model] loading in 4-bit (qlora)", flush=True)
    else:
        print(f"[model] loading in {dtype} (no 4-bit quant)", flush=True)

    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=ft.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(ft.get("bnb_4bit_use_double_quant", True)),
        )

    # Flash Attention 2 — opt-in (hardware.use_flash_attention: true).
    # vLLM uses FA internally regardless; this flag only affects the HF model
    # (log-prob computation + gradient steps in QLoRA training).
    # Requires: pip install flash-attn
    attn_impl = None
    if hw.get("use_flash_attention", False):
        try:
            import flash_attn  # noqa: F401 — probe only
            attn_impl = "flash_attention_2"
            print("[model] flash_attention_2 enabled", flush=True)
        except ImportError:
            print("[model] WARNING: use_flash_attention=true but flash-attn not found "
                  "— falling back to default attention. Install with: pip install flash-attn",
                  flush=True)

    # device_map defaults to "auto". When data-parallel rollouts are enabled the
    # caller pins the HF learner to a single GPU (visible index 0) via
    # hardware.hf_device_map so it does not collide with the rollout workers that
    # own the other GPUs.
    device_map = hw.get("hf_device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True,
        **({"attn_implementation": attn_impl} if attn_impl else {}),
    )

    adapter = checkpoint or mcfg.get("lora_adapter")

    # Validate a local adapter path early so we fail with a clear message instead
    # of PEFT's cryptic "Repo id must be in the form ..." HFValidationError that
    # appears when a non-existent local path is reinterpreted as a HF hub repo id.
    if adapter and (os.path.sep in str(adapter) or os.path.altsep and os.path.altsep in str(adapter)):
        if not os.path.isdir(adapter):
            raise FileNotFoundError(
                f"[model] adapter checkpoint directory does not exist: {adapter}\n"
                f"        Pass an existing --checkpoint path (a dir containing "
                f"adapter_config.json), or omit --checkpoint to start from the base model."
            )
        if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
            raise FileNotFoundError(
                f"[model] '{adapter}' exists but has no adapter_config.json — "
                f"it is not a valid LoRA adapter checkpoint.\n"
                f"        Contents: {sorted(os.listdir(adapter))[:10]}"
            )

    if for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if use_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=hw.get("gradient_checkpointing", True)
            )
        elif hw.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        tgt = ft.get("target_modules", "auto")
        if tgt == "auto":
            tgt = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(
            r=int(ft.get("lora_r", 16)),
            lora_alpha=int(ft.get("lora_alpha", 32)),
            lora_dropout=float(ft.get("lora_dropout", 0.05)),
            target_modules=tgt,
            task_type="CAUSAL_LM",
        )
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            print(f"[model] loaded adapter: {adapter}")
        model.eval()

    return model, tokenizer


# =====================================================================
#  Output helpers
# =====================================================================

def _outputs_dir(config: dict) -> str:
    d = config.get("experiment", {}).get("output_dir", "outputs")
    os.makedirs(d, exist_ok=True)
    return d


class _SafeJsonEncoder(json.JSONEncoder):
    """Extend the default encoder to handle types that IBM tool functions can return
    (e.g. set, complex, numpy scalars) so trajectory JSONL files are always writable."""

    def default(self, obj):
        import datetime
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, bytearray):
            return bytes(obj).decode("utf-8", errors="replace")
        if isinstance(obj, set):
            return sorted(obj, key=lambda x: (str(type(x).__name__), str(x)))
        if isinstance(obj, frozenset):
            return sorted(obj, key=lambda x: (str(type(x).__name__), str(x)))
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        if isinstance(obj, datetime.timedelta):
            return obj.total_seconds()
        try:
            import numpy as np  # optional dep
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Last-resort fallback: never let an exotic IBM-tool return type abort the
        # whole JSONL write (datetime, Decimal, custom objects, ...). Stringify it.
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _write_jsonl(path: str, rows) -> None:
    n_failed = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            try:
                line = json.dumps(r, cls=_SafeJsonEncoder, ensure_ascii=False)
            except Exception as exc:
                # A single unserializable row must never abort the whole dump.
                n_failed += 1
                line = json.dumps(
                    {"_serialization_error": str(exc), "_row_index": i},
                    ensure_ascii=False,
                )
            fh.write(line + "\n")
    if n_failed:
        print(f"[write_jsonl] WARNING: {n_failed} row(s) could not be serialized "
              f"and were written as error placeholders in {path}", flush=True)


# =====================================================================
#  Modes
# =====================================================================

def _load_inference_backend(
    config: dict, checkpoint: str | None, *, mode: str = "eval"
):
    """Return (model, tokenizer, generate_fn). model is None when vLLM generates."""
    use_vllm = bool(config.get("hardware", {}).get("use_vllm", False))
    if use_vllm:
        try:
            tokenizer = load_tokenizer_only(config)
            from vllm_generate import build_vllm_generator
            vgen = build_vllm_generator(
                config, tokenizer, adapter_path=checkpoint, mode=mode,
            )
            print("[inference] using vLLM", flush=True)
            return None, tokenizer, vgen.generate_fn
        except ImportError as exc:
            print(
                "[inference] WARNING: vLLM requested but unavailable — "
                "falling back to HuggingFace generate.",
                flush=True,
            )
            print(f"  {exc}", flush=True)
    model, tokenizer = load_model_and_tokenizer(config, checkpoint, for_training=False)
    return model, tokenizer, None


def mode_smoke(config: dict) -> int:
    out_dir = _outputs_dir(config)
    registry = build_registry(config)
    paths = config["paths"]
    data_cfg = config.get("data", {})
    tasks = load_tasks(
        paths["train_jsonl"],
        stage=data_cfg.get("train_stage"),
        max_tasks=3,
        seed=config.get("experiment", {}).get("seed", 42),
    )
    print(f"[smoke] loaded {len(tasks)} tasks")

    model, tokenizer, generate_fn = _load_inference_backend(config, None, mode="eval")

    try:
        from tqdm import tqdm
        task_iter = tqdm(tasks, desc="[smoke]", unit="task")
    except ImportError:
        task_iter = tasks

    rows, rewards = [], []
    for task in task_iter:
        traj = run_episode(
            model, tokenizer, task, config,
            registry=registry, mode="smoke", generate_fn=generate_fn,
        )
        gold_obs = compute_gold_observations(task, registry)
        rr = strict_gold_trace_reward(traj, task, gold_obs)
        rewards.append(rr.reward)
        rows.append({**traj.to_dict(), "reward_train_strict": rr.reward,
                     "diagnostics": rr.diagnostics})

    _write_jsonl(os.path.join(out_dir, "smoke_trajectories.jsonl"), rows)
    print(f"[smoke] mean strict reward = {sum(rewards)/max(1,len(rewards)):.3f}")
    print(f"[smoke] executor_mode = {rows[0]['executor_mode'] if rows else 'n/a'}")
    print(f"[smoke] wrote {os.path.join(out_dir, 'smoke_trajectories.jsonl')}")
    return 0


def mode_rollout_eval(config: dict, checkpoint: str | None) -> int:
    out_dir = _outputs_dir(config)
    registry = build_registry(config)
    paths = config["paths"]
    data_cfg = config.get("data", {})
    wandb_run = _wandb_init("rollout_eval", config,
                            stage=data_cfg.get("eval_stage"), checkpoint=checkpoint)
    tasks = load_tasks(
        paths["eval_jsonl"],
        stage=data_cfg.get("eval_stage"),
        max_tasks=data_cfg.get("max_eval_tasks"),
        seed=config.get("experiment", {}).get("seed", 42),
    )
    print(f"[rollout_eval] loaded {len(tasks)} tasks")

    model, tokenizer, generate_fn = _load_inference_backend(
        config, checkpoint, mode="eval",
    )

    try:
        from tqdm import tqdm
        task_iter = tqdm(tasks, desc="[rollout_eval]", unit="task")
    except ImportError:
        task_iter = tasks

    rows = []
    agg = {"strict_gold_trace_pass": [], "final_answer_pass": [],
           "zero_tool_calls": [], "clipped": [],
           # Diagnostic for continuation-training experiments (e.g. teacher-forced
           # Stage2b): did the model emit FEWER calls than the gold trace under
           # ordinary (non-forced) generation? Cheap to compute; helps measure
           # whether a training intervention actually fixed "stops too early"
           # without needing a separate analysis pass.
           "too_few_calls": [], "predicted_calls": []}
    for task in task_iter:
        traj = run_episode(
            model, tokenizer, task, config,
            registry=registry, mode="eval", generate_fn=generate_fn,
        )
        gold_obs = compute_gold_observations(task, registry)
        rr = strict_gold_trace_reward(traj, task, gold_obs)
        agg["strict_gold_trace_pass"].append(rr.reward)
        agg["final_answer_pass"].append(1.0 if rr.diagnostics["final_answer_pass"] else 0.0)
        agg["zero_tool_calls"].append(1.0 if traj.zero_tool_calls else 0.0)
        agg["clipped"].append(1.0 if traj.clipped_any else 0.0)
        agg["too_few_calls"].append(
            1.0 if traj.num_tool_calls < traj.gold_num_turns else 0.0)
        agg["predicted_calls"].append(float(traj.num_tool_calls))
        rows.append({**traj.to_dict(), "reward_train_strict": rr.reward,
                     "diagnostics": rr.diagnostics})

    _write_jsonl(os.path.join(out_dir, "rollout_eval_trajectories.jsonl"), rows)
    metrics = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    metrics["too_few_calls_rate"] = metrics.pop("too_few_calls", 0.0)
    metrics["avg_predicted_calls"] = metrics.pop("predicted_calls", 0.0)
    metrics["num_tasks"] = len(tasks)
    exec_mode = rows[0]["executor_mode"] if rows else "n/a"
    reportable = exec_mode == "full"
    metrics["executor_mode"] = exec_mode
    metrics["solution_equivalent_reportable"] = reportable
    metrics["win_rate_reportable"] = reportable
    if not reportable:
        metrics["warning"] = (
            "Alternative-path metrics are limited because non-gold calls cannot "
            "be genuinely executed."
        )
    metrics["clipped_completion_rate"] = metrics.pop("clipped", 0.0)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[rollout_eval] metrics: {json.dumps(metrics, indent=2)}")
    _wandb_log_eval(wandb_run, metrics, prefix="eval")
    _wandb_finish(wandb_run)
    return 0


def _run_direct_final_eval(
    config, tasks, model, tokenizer, generate_fn,
    out_dir, eval_path, paths, data_cfg, wandb_run,
) -> int:
    """NESTFUL Direct-prompting (single-shot) eval, scored with the official scorer."""
    from direct_eval import run_direct_eval
    from metrics import compute_nestful_official_metrics
    from nestful_official_score import (
        build_item, load_raw_dataset, score_items, score_items_per_sample,
    )

    gen_cfg = config.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens_direct", gen_cfg.get("max_new_tokens_eval", 1024)))
    num_icl = int(data_cfg.get("num_icl_examples", 1))

    gen_fn = generate_fn
    if gen_fn is None:
        from rollout import generate_once
        temperature = float(gen_cfg.get("temperature", 0.7))
        top_p = float(gen_cfg.get("top_p", 0.95))
        max_prompt_tokens = int(gen_cfg.get("max_prompt_tokens", 3072))

        def gen_fn(messages, max_new_tokens):  # noqa: ANN001
            return generate_once(
                model, tokenizer, messages, max_new_tokens,
                temperature=temperature, top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
            )

    print(f"[final_eval] paradigm=direct  num_icl={num_icl}  max_new_tokens={max_new}")
    result = run_direct_eval(tasks, gen_fn, num_icl=num_icl, max_new_tokens=max_new)
    predicted = result["predicted"]

    # Persist raw predictions IMMEDIATELY, before any scoring. Generation can take
    # an hour+; a scoring/dependency error must never throw that work away.
    _write_jsonl(
        os.path.join(out_dir, "direct_predictions.jsonl"),
        [{"sample_id": t["task_id"], "predicted_calls": predicted.get(t["task_id"], []),
          "raw_text": result["raw_text"].get(t["task_id"], "")} for t in tasks],
    )

    raw_rows = load_raw_dataset(eval_path)
    scored_tids = [t["task_id"] for t in tasks if t["task_id"] in raw_rows]
    gold_by_tid = {t["task_id"]: t.get("gold_calls", []) for t in tasks}
    fdir = paths.get("ibm_functions_dir")
    want_win = os.name != "nt" and bool(fdir) and os.path.isdir(fdir)

    # CANONICAL official scoring — wrapped so scoring never aborts after generation.
    off_by_tid: dict = {}
    items: list = []
    try:
        items = [build_item(predicted.get(tid, []), raw_rows[tid]) for tid in scored_tids]
        metrics = score_items(items, executable_func_dir=fdir, win_rate=want_win)
        metrics["paradigm"] = "direct"
        metrics["num_icl_examples"] = num_icl
        with open(os.path.join(out_dir, "metrics_official.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print("[final_eval] OFFICIAL NESTFUL metrics (direct, CANONICAL):")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        _wandb_log_eval(wandb_run, {f"official_{k}": v for k, v in metrics.items()
                                    if isinstance(v, (int, float, bool))}, prefix="final_eval")
        per_item = score_items_per_sample(items, executable_func_dir=fdir, win_rate=want_win)
        off_by_tid = dict(zip(scored_tids, per_item))
    except Exception as exc:  # noqa: BLE001 - eval must finish even if scorer fails
        print(f"[final_eval] WARNING: official scoring failed: {exc!r}")
        print("[final_eval] predictions saved in direct_predictions.jsonl; re-score with:")
        print(f"  python nestful_official_score.py --direct-predictions "
              f"{os.path.join(out_dir, 'direct_predictions.jsonl')} "
              f"--out {os.path.join(out_dir, 'metrics_official.json')}")

    traj_out = []
    for t in tasks:
        tid = t["task_id"]
        pred = predicted.get(tid, [])
        internal = compute_nestful_official_metrics(pred, gold_by_tid.get(tid, []))
        row = {
            "sample_id": tid,
            "predicted_calls": pred,
            "raw_text": result["raw_text"].get(tid, ""),
            "internal_partial_match": internal["partial_sequence_accuracy"],
            "internal_full_match": internal["full_sequence_accuracy"],
            "internal_diagnostic_only": {
                "internal_f1_func": internal["f1_func"],
                "internal_f1_param": internal["f1_param"],
            },
        }
        off = off_by_tid.get(tid)
        if off is not None:
            for k in ("official_partial_match", "official_full_match", "official_win",
                      "pred_answer", "parse_valid", "executable", "execution_error"):
                row[k] = off.get(k)
            mm, reasons = _compute_mismatch(internal, off)
            row["mismatch"] = mm
            row["mismatch_reason"] = reasons
        traj_out.append(row)

    _write_jsonl(os.path.join(out_dir, "direct_eval_trajectories.jsonl"), traj_out)
    return 0


def _compute_mismatch(internal: dict, official: dict, tol: float = 1e-6):
    """Compare the INTERNAL replica against the CANONICAL official per-sample
    diagnostics on partial/full/win. Returns (mismatch: bool, reasons: list[str]).
    When they disagree, trust official_*; the flag exists for debugging only."""
    reasons = []
    pairs = [
        ("partial", internal.get("partial_sequence_accuracy"), official.get("official_partial_match")),
        ("full", internal.get("full_sequence_accuracy"), official.get("official_full_match")),
        ("win", internal.get("win_rate"), official.get("official_win")),
    ]
    for name, iv, ov in pairs:
        if iv is None or ov is None:
            continue  # official win is None on Windows / when win_rate disabled
        try:
            if abs(float(iv) - float(ov)) > tol:
                reasons.append(f"{name}: internal={float(iv):.4g} official={float(ov):.4g}")
        except (TypeError, ValueError):
            continue
    return (len(reasons) > 0), reasons


def mode_final_eval(config: dict, checkpoint: str | None) -> int:
    out_dir = _outputs_dir(config)
    registry = build_registry(config)
    paths = config["paths"]
    data_cfg = config.get("data", {})
    wandb_run = _wandb_init("final_eval", config, checkpoint=checkpoint)
    eval_path = paths.get("full_nestful_jsonl") or paths["eval_jsonl"]
    tasks = load_tasks(
        eval_path,
        stage=None,  # full benchmark, all call counts
        max_tasks=data_cfg.get("max_eval_tasks"),
        seed=config.get("experiment", {}).get("seed", 42),
    )
    print(f"[final_eval] loaded {len(tasks)} tasks from {eval_path}")

    model, tokenizer, generate_fn = _load_inference_backend(
        config, checkpoint, mode="eval",
    )

    # Paradigm switch: NESTFUL Table 1 = "direct" (single-shot full sequence),
    # Table 2 = "react" (our multi-turn rollout, the default).
    if data_cfg.get("eval_paradigm", "react") == "direct":
        return _run_direct_final_eval(
            config, tasks, model, tokenizer, generate_fn,
            out_dir, eval_path, paths, data_cfg, wandb_run,
        )

    try:
        from tqdm import tqdm
        task_iter = tqdm(tasks, desc="[final_eval]", unit="task")
    except ImportError:
        task_iter = tasks

    # num_eval_rollouts > 1: run multiple stochastic rollouts per task and
    # aggregate. Binary metrics (full_acc, win_rate, strict, final_answer) use
    # MAX across rollouts (pass@N style). Continuous metrics (f1, partial_acc)
    # use MEAN. Set via --override data.num_eval_rollouts=4 or in config.
    num_eval_rollouts = int(data_cfg.get("num_eval_rollouts", 1))
    if num_eval_rollouts > 1:
        print(f"[final_eval] num_eval_rollouts={num_eval_rollouts}  "
              f"(binary=pass@N, continuous=mean)")

    # Internal numeric metric keys (the 5 internal_* aggregates). The compute
    # function also returns underscore-prefixed grounded lists for corpus macro-F1;
    # those are handled separately (never averaged as scalars).
    _INTERNAL_NUMERIC = (
        "f1_func", "f1_param", "partial_sequence_accuracy",
        "full_sequence_accuracy", "win_rate",
    )

    per_sample, traj_rows = [], []
    official_pred: dict = {}
    executor_mode = "gold_replay"
    n_skipped = 0
    # Incremental safety dump: ReAct generation can take an hour+; append each
    # sample's predicted calls as we go so a late crash never throws work away.
    # Same schema as direct_predictions.jsonl, so re-score later with:
    #   nestful_official_score.py --direct-predictions <this file> --out metrics.json
    partial_path = os.path.join(out_dir, "final_eval_predictions.partial.jsonl")
    partial_fh = open(partial_path, "w", encoding="utf-8")
    for task in task_iter:
        try:
            gold_obs = compute_gold_observations(task, registry)
            rollout_internals, rollout_papers, rollout_trajs, rollout_rrs = [], [], [], []

            for _ in range(num_eval_rollouts):
                traj = run_episode(
                    model, tokenizer, task, config,
                    registry=registry, mode="eval", generate_fn=generate_fn,
                )
                executor_mode = traj.executor_mode
                rr = strict_gold_trace_reward(traj, task, gold_obs)
                internal = compute_nestful_official_metrics(
                    traj.predicted_calls, task["gold_calls"], traj, task
                )
                paper = compute_paper_metrics(traj, task, rr, internal)
                rollout_internals.append(internal)
                rollout_papers.append(paper)
                rollout_trajs.append(traj)
                rollout_rrs.append(rr)

            # Aggregate rollouts: binary -> max (pass@N), continuous -> mean.
            def _agg_internal(key):
                vals = [o[key] for o in rollout_internals]
                if key in ("full_sequence_accuracy", "win_rate"):
                    return max(vals)
                return sum(vals) / len(vals)

            def _agg_paper(key):
                vals = [p[key] for p in rollout_papers]
                if isinstance(vals[0], bool):
                    return any(vals)
                return max(vals) if all(isinstance(v, (int, float)) for v in vals) else vals[0]

            internal = {k: _agg_internal(k) for k in _INTERNAL_NUMERIC}
            paper = {k: _agg_paper(k) for k in rollout_papers[0]}
            rr = rollout_rrs[0]   # first rollout's rr for compat; reward overridden below
            traj = rollout_trajs[0]
            # Grounded lists (from the first rollout) feed the corpus macro-F1.
            r0 = rollout_internals[0]
            internal_lists = {
                "gold_func": r0.get("_gold_func_names", []),
                "pred_func": r0.get("_pred_func_names", []),
                "gold_slots": r0.get("_gold_param_slots", []),
                "pred_slots": r0.get("_pred_param_slots", []),
            }

            sample = {
                "sample_id": task["task_id"],
                "num_gold_calls": task["num_calls"],
                "num_eval_rollouts": num_eval_rollouts,
                **{f"internal_{k}": internal[k] for k in _INTERNAL_NUMERIC},
                **{k: paper[k] for k in (
                    "strict_gold_trace_pass", "solution_equivalent_pass",
                    "strict_fail_but_solution_equivalent_pass",
                    "correct_answer_but_unsupported_trace", "final_answer_pass",
                    "alternative_valid_solution_pass",
                )},
                "internal": internal,
                "_internal_lists": internal_lists,
                "paper": paper,
            }
            per_sample.append(sample)
            traj_rows.append({**traj.to_dict(), "internal": internal, "paper": paper,
                              "reward_train_strict": rr.reward})
            official_pred[task["task_id"]] = list(traj.predicted_calls)
            try:
                partial_fh.write(json.dumps(
                    {"sample_id": task["task_id"],
                     "num_gold_calls": task.get("num_calls"),
                     "predicted_calls": list(traj.predicted_calls)},
                    ensure_ascii=False, cls=_SafeJsonEncoder) + "\n")
                partial_fh.flush()
            except Exception:  # noqa: BLE001 - safety dump must never abort the run
                pass
        except Exception as exc:  # noqa: BLE001 - one bad sample must never abort the run
            n_skipped += 1
            print(f"[final_eval] WARNING: skipped sample "
                  f"{task.get('task_id', '?')}: {exc!r}")
            continue

    try:
        partial_fh.close()
    except Exception:  # noqa: BLE001
        pass

    if n_skipped:
        print(f"[final_eval] WARNING: {n_skipped} sample(s) skipped due to errors.")

    report = aggregate_final_eval(per_sample, executor_mode)
    report["num_skipped"] = n_skipped
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[final_eval] executor_mode = {executor_mode}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # ---- Official NESTFUL scoring (CANONICAL paper metrics) -------------------
    # metrics.json above is the INTERNAL diagnostic replica; metrics_official.json
    # runs the real NESTFUL scorer (grounding + sklearn macro-F1 + Win re-exec).
    # Per-sample official_* diagnostics are merged into the trajectory rows so the
    # mismatch flag can be computed. Never let a scorer hiccup abort the eval.
    off_by_tid: dict = {}
    try:
        from nestful_official_score import (
            build_item, load_raw_dataset, score_items, score_items_per_sample,
        )
        raw_rows = load_raw_dataset(eval_path)
        scored_tids = [t["task_id"] for t in tasks
                       if t["task_id"] in raw_rows and t["task_id"] in official_pred]
        items = [build_item(official_pred[tid], raw_rows[tid]) for tid in scored_tids]
        fdir = paths.get("ibm_functions_dir")
        want_win = os.name != "nt" and bool(fdir) and os.path.isdir(fdir)
        # Per-sample FIRST (isolates each sample's win/exec in its own try/except),
        # then reuse it for the aggregate so Win Rate is crash-proof AND the IBM
        # functions are executed only once (not twice).
        per_item = score_items_per_sample(items, executable_func_dir=fdir, win_rate=want_win)
        off_by_tid = dict(zip(scored_tids, per_item))
        official_metrics = score_items(
            items, executable_func_dir=fdir, win_rate=want_win, per_sample=per_item
        )
        official_metrics["paradigm"] = data_cfg.get("eval_paradigm", "react")
        with open(os.path.join(out_dir, "metrics_official.json"), "w", encoding="utf-8") as fh:
            json.dump(official_metrics, fh, indent=2, ensure_ascii=False)
        print("[final_eval] OFFICIAL NESTFUL metrics (CANONICAL):")
        print(json.dumps(official_metrics, indent=2, ensure_ascii=False))
        _wandb_log_eval(wandb_run, {f"official_{k}": v for k, v in official_metrics.items()
                                    if isinstance(v, (int, float, bool))}, prefix="final_eval")
    except Exception as exc:  # noqa: BLE001 - eval must finish even if scorer fails
        print(f"[final_eval] WARNING: official scoring failed: {exc!r}")

    # Merge official_* + mismatch flag into per-sample trajectory rows.
    for s, tr in zip(per_sample, traj_rows):
        tr["internal_diagnostic_only"] = {
            "internal_f1_func": s["internal"]["f1_func"],
            "internal_f1_param": s["internal"]["f1_param"],
        }
        off = off_by_tid.get(s["sample_id"])
        if off is not None:
            for k in ("official_partial_match", "official_full_match", "official_win",
                      "pred_answer", "parse_valid", "executable", "execution_error"):
                tr[k] = off.get(k)
            mm, reasons = _compute_mismatch(s["internal"], off)
            tr["mismatch"] = mm
            tr["mismatch_reason"] = reasons

    _write_jsonl(os.path.join(out_dir, "final_eval_trajectories.jsonl"),
                 [{k: v for k, v in s.items()
                   if k not in ("internal", "paper", "_internal_lists")} | {"_traj": tr}
                  for s, tr in zip(per_sample, traj_rows)])
    _wandb_log_eval(wandb_run, {k: v for k, v in report.items()
                                if isinstance(v, (int, float, bool))}, prefix="final_eval")
    _wandb_finish(wandb_run)
    return 0


def _build_validation_subset(
    full_path: str, subset_size: int, ids_path: str, subset_jsonl: str, seed: int
) -> str:
    """Create (or reuse) a deterministic validation subset; return subset JSONL path.

    Writes validation_subset_ids.json so the SAME subset is used for the baseline
    and every checkpoint. If both files already exist, they are reused unchanged.
    Selection is stable (sample_ids sorted, then a seeded shuffle) and therefore
    independent of input row order.
    """
    if os.path.isfile(ids_path) and os.path.isfile(subset_jsonl):
        print(f"[val_eval] reusing existing validation subset: {subset_jsonl}", flush=True)
        return subset_jsonl

    rows: list = []
    with open(full_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("sample_id") or r.get("task_id") or r.get("id"))
            rows.append((sid, line))
    rows.sort(key=lambda x: x[0])
    import random as _random
    idxs = list(range(len(rows)))
    _random.Random(seed).shuffle(idxs)
    keep = sorted(idxs[: max(0, subset_size)])
    chosen = [rows[i] for i in keep]

    os.makedirs(os.path.dirname(subset_jsonl) or ".", exist_ok=True)
    with open(subset_jsonl, "w", encoding="utf-8") as fh:
        for _sid, line in chosen:
            fh.write(line + "\n")
    os.makedirs(os.path.dirname(ids_path) or ".", exist_ok=True)
    with open(ids_path, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": seed,
            "subset_size": subset_size,
            "num_full": len(rows),
            "sample_ids": [sid for sid, _ in chosen],
        }, fh, indent=2)
    print(f"[val_eval] wrote validation subset ({len(chosen)} tasks): {subset_jsonl}", flush=True)
    return subset_jsonl


def mode_val_eval(config: dict, checkpoint: str | None) -> int:
    """Per-epoch VALIDATION ReAct Win (official scorer) → metrics_epoch_<E>.json.

    Used by run_curriculum.sh after each training epoch to drive early stopping on
    validation ReAct Win Rate. Full NESTFUL by default; set validation.subset_size>0
    for a deterministic subset (validation_subset_ids.json, identical for the
    baseline and all checkpoints). Reward is NOT involved.
    """
    val = config.get("validation", {}) or {}
    paths = config["paths"]
    seed = config.get("experiment", {}).get("seed", 42)
    subset_size = int(val.get("subset_size", 0) or 0)
    full_path = paths.get("full_nestful_jsonl") or paths["eval_jsonl"]
    out_dir = _outputs_dir(config)

    if subset_size > 0:
        ids_path = val.get("subset_ids_path") or os.path.join(out_dir, "validation_subset_ids.json")
        subset_jsonl = val.get("subset_jsonl") or os.path.join(out_dir, "validation_subset.jsonl")
        sp = _build_validation_subset(full_path, subset_size, ids_path, subset_jsonl, seed)
        config["paths"]["full_nestful_jsonl"] = sp
        print(f"[val_eval] deterministic validation subset (n={subset_size}): {sp}", flush=True)
    else:
        print(f"[val_eval] using FULL validation set: {full_path}", flush=True)

    # Validation is always ReAct (the early-stopping target metric is ReAct Win).
    config.setdefault("data", {})["eval_paradigm"] = "react"

    rc = mode_final_eval(config, checkpoint)

    # Surface react_win_rate from the official metrics as metrics_epoch_<E>.json.
    win = None
    mo = os.path.join(out_dir, "metrics_official.json")
    if os.path.isfile(mo):
        try:
            with open(mo, encoding="utf-8") as fh:
                win = json.load(fh).get("win_rate")
        except (json.JSONDecodeError, OSError):
            pass
    epoch = val.get("epoch")
    stage = val.get("stage")
    metrics = {
        "react_win_rate": win,
        "metric": val.get("metric", "react_win_rate"),
        "epoch": epoch,
        "stage": stage,
        "checkpoint": checkpoint,
        "subset_size": subset_size,
        "eval_path": config["paths"].get("full_nestful_jsonl"),
    }
    ep_name = f"metrics_epoch_{epoch}.json" if epoch is not None else "metrics_epoch.json"
    with open(os.path.join(out_dir, ep_name), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"[val_eval] react_win_rate={win} -> {ep_name}", flush=True)

    from nestful_official_score import _reset_sigalrm
    _reset_sigalrm()

    # Hard-fail guard: a null validation Win means the official scorer could not
    # produce a number, so checkpoint selection / early stopping would silently
    # run on an invalid signal (this is exactly what broke the previous run: 6/8
    # epochs logged react_win_rate=null and the "best" adapter was chosen from 2
    # noisy points). Refuse to continue unless validation.require_win_rate=false.
    require_win = _truthy(val.get("require_win_rate", True))
    if win is None and require_win:
        print(
            "[val_eval] ERROR: react_win_rate is null (official Win Rate could not "
            "be computed). Aborting so checkpoint selection never runs on an "
            "invalid signal. Set validation.require_win_rate=false to override.",
            flush=True,
        )
        return 2
    return rc


def mode_train(config: dict, checkpoint: str | None = None) -> int:
    out_dir = _outputs_dir(config)
    registry = build_registry(config)
    paths = config["paths"]
    data_cfg = config.get("data", {})
    stage = data_cfg.get("train_stage")
    seed = config.get("experiment", {}).get("seed", 42)

    # ── Mixed curriculum replay ──────────────────────────────────────────────
    # When data.mixed_replay is on, stage N trains on a weighted mix of the
    # per-stage files for stages 1..N (data.mixed_stage_files) instead of a
    # single stage file filtered by num_calls. Reward is unaffected.
    if _truthy(data_cfg.get("mixed_replay")):
        stage_files = _parse_str_list(data_cfg.get("mixed_stage_files"))
        if not stage_files:
            raise ValueError(
                "[train] mixed_replay=True but data.mixed_stage_files is empty. "
                "Pass --override data.mixed_stage_files=f1.jsonl,f2.jsonl,..."
            )
        weights = _parse_float_list(data_cfg.get("replay_weights"))
        replay_ratio = data_cfg.get("replay_ratio")
        if replay_ratio is not None:
            replay_ratio = float(replay_ratio)
        mix = load_tasks_mixed(
            stage_files,
            weights=(weights or None) if replay_ratio is None else None,
            replay_ratio=replay_ratio,
            max_tasks=data_cfg.get("max_train_tasks"),
            seed=seed,
        )
        tasks = mix["tasks"]
        print(f"[train] MIXED REPLAY enabled (stage {stage}): {len(tasks)} tasks "
              f"from {len(stage_files)} stage files"
              + (f" (replay_ratio={replay_ratio})" if replay_ratio is not None else ""),
              flush=True)
        for ps, eff in zip(mix["per_stage"], mix.get("effective_mix", [])):
            print(f"[train]   stage {ps['stage_index']}: sampled {ps['sampled']} / "
                  f"available {ps['available']} (intended={ps['weight']} "
                  f"effective={eff:.4f}) :: {ps['file']}",
                  flush=True)
        print(f"[train]   normalized sampling weights: {mix['weights']}", flush=True)
        _bs = int(config.get("training", {}).get("per_device_train_batch_size", 1))
        _ga = int(config.get("training", {}).get("gradient_accumulation_steps", 1))
        _eff = max(1, _bs * _ga)
        print(f"[train]   batches/epoch ~ {len(tasks) // _eff} "
              f"(effective batch size {_eff})", flush=True)
    else:
        tasks = load_tasks(
            paths["train_jsonl"],
            stage=stage,
            max_tasks=data_cfg.get("max_train_tasks"),
            seed=seed,
        )
    print(f"[train] EXPERIMENTAL episode-level GRPO on {len(tasks)} tasks")
    print("[train] This is a minimal pilot trainer; see README 'Known limitations'.")

    wandb_run = _wandb_init("train", config, stage=stage)

    hw = config.get("hardware", {})
    use_vllm = bool(hw.get("use_vllm", False))

    # ── Optional data-parallel rollouts ──────────────────────────────────────
    # When hardware.rollout_data_parallel_gpus lists GPUs, rollouts run in worker
    # processes (one vLLM engine per GPU) while the HF learner is pinned to visible
    # index 0. This requires the train process to SEE all those GPUs. Falls back to
    # the single-engine path on any setup error so a run never silently dies.
    dp_gpus = _parse_gpu_list(hw.get("rollout_data_parallel_gpus"))
    use_pool = use_vllm and bool(dp_gpus)
    if use_pool:
        # Pin the HF learner to a single GPU so it doesn't fight the rollout
        # workers for the shared devices.
        config.setdefault("hardware", {})["hf_device_map"] = {"": 0}

    # HF model always needed for log-prob computation and gradient steps.
    model, tokenizer = load_model_and_tokenizer(config, checkpoint, for_training=True)

    # Optional vLLM engine(s) for fast rollout generation.
    # IMPORTANT: pass the same checkpoint so rollouts come from the same policy as
    # the HF model — avoids off-policy gradient computation.
    vllm_gen = None
    rollout_pool = None
    if use_pool:
        try:
            from vllm_dp_pool import DataParallelRolloutPool
            rollout_pool = DataParallelRolloutPool(config, dp_gpus, adapter_path=checkpoint)
            print(f"[train] data-parallel rollouts on GPUs {dp_gpus}; "
                  f"HF learner pinned to visible GPU 0", flush=True)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            print(f"[train] WARNING: data-parallel pool init failed ({exc}); "
                  f"falling back to single in-process vLLM engine", flush=True)
            rollout_pool = None
    if rollout_pool is None and use_vllm:
        from vllm_generate import build_vllm_generator
        vllm_gen = build_vllm_generator(config, tokenizer, adapter_path=checkpoint, mode="train")
        print("[train] vLLM generator ready for rollout generation", flush=True)

    # Record the init source so grpo_train can log it into trainer_state.json.
    config.setdefault("_runtime", {})["init_checkpoint"] = checkpoint

    from grpo_train import train
    log_path = os.path.join(out_dir, "train_log.jsonl")
    try:
        summary = train(config, model, tokenizer, registry, tasks, log_path,
                        vllm_gen=vllm_gen, rollout_pool=rollout_pool, wandb_run=wandb_run)
    finally:
        if rollout_pool is not None:
            try:
                rollout_pool.close()
            except Exception:
                pass
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[train] done: {json.dumps(summary)}")
    print(f"[train] log: {log_path}")
    _wandb_log_eval(wandb_run, {k: v for k, v in summary.items()
                                if isinstance(v, (int, float, bool))}, prefix="train_summary")
    _wandb_finish(wandb_run)
    return 0


def _apply_overrides(config: dict, overrides: list) -> None:
    """Apply dot-notation key=value pairs to the config dict in-place.

    Type coercion: null → None, true/false → bool, integers → int,
    floats (with '.') → float, everything else → str. Used by the shell
    curriculum runner to avoid per-stage config files.
    """
    for raw in overrides:
        if "=" not in raw:
            print(f"[override] WARNING: skip '{raw}' — no '=' found", flush=True)
            continue
        key, _, raw_val = raw.partition("=")
        keys = key.strip().split(".")
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d.get(k), dict):
                d[k] = {}
            d = d[k]
        fk = keys[-1].strip()
        if raw_val.lower() == "null":
            val: Any = None
        elif raw_val.lower() == "true":
            val = True
        elif raw_val.lower() == "false":
            val = False
        else:
            try:
                val = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = raw_val
        d[fk] = val
        print(f"[override] {key} = {val!r}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="NESTFUL MT-GRPO Minimal")
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "rollout_eval", "train", "final_eval", "val_eval"])
    ap.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA adapter or model path for eval modes")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value using dot notation. "
                         "Example: --override data.train_stage=4 "
                         "--override data.max_train_tasks=32")
    args = ap.parse_args()

    config = load_config(args.config)
    _apply_overrides(config, args.override)
    _normalize_config_paths(config)
    # Load tool-observation truncation limits so a runaway tool output can never
    # blow the prompt past the context window (see prompt.py / config.generation).
    from prompt import set_observation_limits
    set_observation_limits(config)
    print_versions()
    _ensure_local_data(config)

    if args.mode == "smoke":
        return mode_smoke(config)
    if args.mode == "rollout_eval":
        return mode_rollout_eval(config, args.checkpoint)
    if args.mode == "final_eval":
        return mode_final_eval(config, args.checkpoint)
    if args.mode == "val_eval":
        return mode_val_eval(config, args.checkpoint)
    if args.mode == "train":
        return mode_train(config, args.checkpoint)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
