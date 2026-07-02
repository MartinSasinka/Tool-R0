"""vLLM generation backend — opt-in, activated by hardware.use_vllm: true.

Two use cases
─────────────
1. Eval modes (smoke, rollout_eval, final_eval)
   • A single LLM instance handles all generation.
   • The HF model is NOT loaded — only the tokenizer is needed.
   • generate_fn() is passed directly to run_episode() as a drop-in for
     rollout.generate_once().

2. Train rollout generation (grpo_train)
   • vLLM handles fast rollout generation (no gradients needed).
   • The HF model (QLoRA, trainable) handles log-prob computation + GRPO gradients.
   • After each training epoch the adapter is saved and vLLM switches to the new
     adapter via LoRARequest — no vLLM restart required.

Memory budget on 1× A100 40GB with Qwen3-4B
─────────────────────────────────────────────
  Eval only        vLLM bf16 ~8–10 GB     → safe, lots of headroom
  Train + eval     vLLM bf16 ~8–10 GB
                 + QLoRA 4-bit ~3–5 GB
                 + activations/grad ~3 GB  → ~15–18 GB total, safe on 40 GB

Config knobs (hardware section)
────────────────────────────────
  use_vllm: true
  vllm_gpu_memory_utilization: 0.45    # train: ~18 GB, leave room for HF model
  vllm_gpu_memory_utilization: 0.85    # eval only: ~34 GB
  vllm_tensor_parallel_size: 1
  vllm_enforce_eager: false            # set true to debug CUDA graph issues
  vllm_weight_sync: after_epoch        # when to sync adapter to vLLM in train

This file imports nothing from curricullum/ or nestful_evaluation/.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  VLLMGenerator
# ─────────────────────────────────────────────────────────────────────────────

def _log_gpu_memory(prefix: str) -> None:
    """Print free/total VRAM per visible GPU (best-effort, never raises)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        parts = []
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            parts.append(f"cuda:{i} {free_b/1e9:.1f}/{total_b/1e9:.1f} GB free")
        print(f"{prefix} " + " | ".join(parts), flush=True)
    except Exception:
        pass


def _log_disk_space(prefix: str) -> None:
    """Print free disk space for /tmp and the cwd filesystem (best-effort)."""
    try:
        import shutil
        parts = []
        for path in ("/tmp", os.getcwd()):
            try:
                u = shutil.disk_usage(path)
                parts.append(f"{path} {u.free/1e9:.1f}/{u.total/1e9:.1f} GB free")
            except Exception:
                pass
        if parts:
            print(f"{prefix} " + " | ".join(parts), flush=True)
    except Exception:
        pass


def _import_vllm_llm():
    """Import vLLM LLM with a clear message on torch ABI mismatch."""
    import torch
    try:
        from vllm import LLM
        return LLM
    except ImportError as exc:
        msg = str(exc)
        if "_C" in msg or "undefined symbol" in msg or "c10" in msg:
            raise ImportError(
                "vLLM failed to load its CUDA extension — almost always a torch/vLLM "
                "version mismatch.\n"
                f"  torch installed: {torch.__version__}\n"
                "  vLLM 0.12.x requires torch==2.9.x; torch 2.11 needs vLLM >= 0.20.x.\n"
                "Fix (keep torch 2.11):\n"
                "  pip uninstall -y vllm\n"
                "  pip install vllm==0.23.0 \\\n"
                "    --extra-index-url https://wheels.vllm.ai/0.23.0/cu128 \\\n"
                "    --extra-index-url https://download.pytorch.org/whl/cu128\n"
                "Or run without vLLM: omit --override hardware.use_vllm=true\n"
                f"Original error: {exc}"
            ) from exc
        if "libcudart.so.13" in msg:
            raise ImportError(
                "vLLM wheel is built for CUDA 13 but your torch/system is CUDA 12.x.\n"
                f"  torch installed: {torch.__version__}\n"
                "Plain `pip install vllm` pulls CUDA-13 wheels from PyPI.\n"
                "Fix — reinstall from cu128 index (match your torch+cu128):\n"
                "  pip uninstall -y vllm\n"
                "  pip install vllm==0.23.0 \\\n"
                "    --extra-index-url https://wheels.vllm.ai/0.23.0/cu128 \\\n"
                "    --extra-index-url https://download.pytorch.org/whl/cu128\n"
                "Or run without vLLM: omit --override hardware.use_vllm=true\n"
                f"Original error: {exc}"
            ) from exc
        raise


class VLLMGenerator:
    """Thin wrapper around vllm.LLM providing a generate_fn interface.

    The generate_fn interface (compatible with rollout.run_episode):

        fn(messages: List[Dict[str, str]], max_new_tokens: int) -> Dict[str, Any]

    Return dict keys:
        text              str   — decoded completion
        prompt_tokens     int   — number of prompt tokens
        completion_tokens int   — number of generated tokens
        clipped           bool  — True if max_new_tokens was reached
        prompt_overflow   bool  — always False (vLLM handles truncation internally)
    """

    def __init__(
        self,
        model: str,
        tokenizer,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        top_p: float = 0.95,
        adapter_path: Optional[str] = None,
        max_lora_rank: int = 64,
        enforce_eager: bool = False,
        enable_lora: bool = False,
    ) -> None:
        # vLLM v1 creates a ZMQ IPC socket under VLLM_RPC_BASE_PATH. Unix domain socket
        # paths are limited to 107 chars; a long project path like
        # /mnt/raid/.../nestful_mtgrpo_minimal plus a UUID overflows this and raises
        # ZMQError. Force a short base dir. We must NOT rely on tempfile.gettempdir()
        # because TMPDIR may be exported to the (long) project cwd on some clusters.
        # Both VLLM_RPC_BASE_PATH (vLLM honours it) and TMPDIR are set, before vLLM
        # is imported/initialised, so any code path that builds the socket stays short.
        _candidates = ["/tmp", "/run/user/%d" % os.getuid() if hasattr(os, "getuid") else "/tmp"]
        _short = next((d for d in _candidates if os.path.isdir(d) and len(d) <= 40), "/tmp")
        _rpc = os.environ.get("VLLM_RPC_BASE_PATH")
        if not _rpc or len(_rpc) > 64:
            os.environ["VLLM_RPC_BASE_PATH"] = _short
            print(f"[vllm] VLLM_RPC_BASE_PATH set to {_short} (short IPC socket path)",
                  flush=True)
        # Keep TMPDIR short too, in case the installed vLLM version derives the IPC
        # base from tempfile.gettempdir() rather than VLLM_RPC_BASE_PATH.
        _tmpdir = os.environ.get("TMPDIR", "")
        if not _tmpdir or len(_tmpdir) > 64:
            os.environ["TMPDIR"] = _short

        LLM = _import_vllm_llm()

        self._model_name = model
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_p = top_p
        self._max_model_len = int(max_model_len)
        self._adapter_path = adapter_path
        # The vLLM engine must be built with enable_lora=True if we will EVER serve
        # a LoRA adapter — including adapters synced later via sync_adapter() during
        # training. Otherwise generate() with a lora_request fails on an engine that
        # was not configured for LoRA. So enable it if an adapter is set now OR if the
        # caller explicitly requests it (train mode, where adapters are synced later).
        self._enable_lora = (adapter_path is not None) or enable_lora
        # vLLM caches LoRA adapters by integer id. Bump the id on every sync so the
        # engine reloads the freshly-trained weights instead of serving stale ones.
        self._lora_id = 1

        kwargs: Dict[str, Any] = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        if self._enable_lora:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = max_lora_rank

        print(
            f"[vllm] initialising LLM: {model}"
            f"  tp={tensor_parallel_size}"
            f"  gpu_util={gpu_memory_utilization}"
            f"  max_len={max_model_len}"
            f"  lora={'yes (' + str(adapter_path) + ')' if self._enable_lora else 'no'}",
            flush=True,
        )
        _log_gpu_memory("[vllm] GPU before init:")
        _log_disk_space("[vllm] disk before init:")
        try:
            self._llm = LLM(**kwargs)
        except Exception as exc:
            # Engine core init failures have three common root causes:
            #   (a) DISK FULL — vLLM JIT-compiles Triton/inductor kernels via gcc and
            #       writes temp files; "No space left on device" kills the compile.
            #   (b) GPU OOM — another process / zombie vLLM worker holds VRAM.
            #   (c) CUDA graph capture issue.
            _log_gpu_memory("[vllm] GPU at failure:")
            _log_disk_space("[vllm] disk at failure:")
            _msg = str(exc).lower()
            _disk_full = ("no space left" in _msg or "errno 28" in _msg
                          or "returned non-zero exit status" in _msg)
            hint = (
                "vLLM engine core failed to initialise.\n"
            )
            if _disk_full:
                hint += (
                    "  ROOT CAUSE: DISK FULL (No space left on device). vLLM compiles\n"
                    "  LoRA/Triton kernels via gcc and writes temp + cache files; a full\n"
                    "  filesystem makes the compile fail.\n"
                    "    → Check free space:   df -h /tmp /mnt/raid\n"
                    "    → Clear vLLM/torch compile cache:\n"
                    "        rm -rf ~/.cache/vllm /mnt/raid/data/*/cache/vllm\n"
                    "        rm -rf ~/.triton/cache ~/.cache/torch\n"
                    "    → Clear /tmp leftovers:  rm -rf /tmp/tmp* /tmp/cc*\n"
                    "    → Remove old run logs/trajectories under outputs/ if huge.\n"
                )
            else:
                hint += (
                    f"  Likely GPU OOM: requested gpu_memory_utilization="
                    f"{gpu_memory_utilization}.\n"
                    f"    → nvidia-smi ; pkill -f VLLM::EngineCore ; pkill -f 'run.py'\n"
                    f"    → Lower util: --override hardware.vllm_gpu_memory_utilization=0.45\n"
                    f"    → Or CUDA graph issue: "
                    f"--override hardware.vllm_enforce_eager=true\n"
                )
            hint += f"Original error: {type(exc).__name__}: {exc}"
            raise RuntimeError(hint) from exc
        print("[vllm] LLM ready", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_fn(self, messages: List[Dict[str, str]], max_new_tokens: int) -> Dict[str, Any]:
        """Drop-in for rollout.generate_once().  Formats messages with the
        tokenizer's chat template, generates via vLLM, returns a compatible dict.
        """
        from vllm import SamplingParams

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # ── Pre-check: never even call the engine with an overlong prompt ──────
        # The engine raises a hard ValueError ("decoder prompt ... longer than the
        # maximum model length") that historically crashed the whole run. We must
        # leave room for `max_new_tokens` of generation, so the prompt budget is
        # (max_model_len - max_new_tokens). On overflow we return a graceful
        # overflow dict instead of generating — the episode is then ended/skipped.
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        est_prompt_len = len(prompt_ids)
        prompt_budget = self._max_model_len - int(max_new_tokens)
        if prompt_budget > 0 and est_prompt_len > prompt_budget:
            print(
                f"[vllm] prompt_overflow (pre-check): {est_prompt_len} tokens "
                f"> budget {prompt_budget} (max_model_len {self._max_model_len} "
                f"- max_new_tokens {max_new_tokens}) — skipping generation",
                flush=True,
            )
            return {
                "text": "",
                "prompt_tokens": est_prompt_len,
                "completion_tokens": 0,
                "clipped": False,
                "prompt_overflow": True,
            }

        params = SamplingParams(
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        lora_req = self._make_lora_request()

        try:
            outputs = self._llm.generate([prompt], sampling_params=params, lora_request=lora_req)
        except Exception as exc:
            # vLLM signals an overflow in several different ways depending on
            # version: VLLMValidationError, or a plain ValueError whose message
            # mentions the context/model length. Match all of them so a long
            # multi-turn history never crashes the run — the episode is skipped
            # from GRPO updates instead, matching rollout.generate_once().
            exc_name = type(exc).__name__
            exc_msg = str(exc)
            low = exc_msg.lower()
            is_overflow = (
                "VLLMValidationError" in exc_name
                or "input_tokens" in exc_msg
                or "context length" in low
                or "maximum context" in low
                or "maximum model length" in low          # ValueError wording (this crash)
                or "longer than the maximum" in low        # decoder-prompt wording
                or "max_model_len" in low
            )
            if is_overflow:
                print(
                    f"[vllm] prompt_overflow (engine): est {est_prompt_len} tokens "
                    f"> max_model_len {self._max_model_len} "
                    f"— skipping episode ({exc_name})",
                    flush=True,
                )
                return {
                    "text": "",
                    "prompt_tokens": est_prompt_len,
                    "completion_tokens": 0,
                    "clipped": False,
                    "prompt_overflow": True,
                }
            raise

        out = outputs[0].outputs[0]
        completion_len = len(out.token_ids)
        clipped = completion_len >= max_new_tokens

        # Prompt token count from the request object (vLLM tracks this).
        prompt_len = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0

        return {
            "text": out.text,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "clipped": clipped,
            "prompt_overflow": False,
        }

    def sync_adapter(self, adapter_path: Optional[str]) -> None:
        """Update the LoRA adapter path used in subsequent generate_fn() calls.

        Called by grpo_train.train() after saving a checkpoint at the end of
        each epoch.  No vLLM restart is required — LoRARequest is hot-swapped
        per generation call (vLLM caches adapter weights internally).
        """
        old = self._adapter_path
        self._adapter_path = adapter_path
        if adapter_path is not None:
            self._enable_lora = True
            # New id forces vLLM to reload the adapter (it caches by int id).
            self._lora_id += 1
        print(
            f"[vllm] adapter sync: {old} -> {adapter_path} (lora_id={self._lora_id})",
            flush=True,
        )

    def tokenize_for_logprob(
        self, messages: List[Dict[str, str]], completion_text: str
    ):
        """Re-tokenize a (prompt, completion) pair as 1-D LongTensors for
        _sequence_logprob() in grpo_train.  Needed because vLLM generate_fn
        does not return token IDs directly.
        """
        import torch
        _p = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_ids = (_p.input_ids if hasattr(_p, "input_ids") else _p)[0]
        comp_ids = self._tokenizer.encode(
            completion_text,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        return prompt_ids, comp_ids

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_lora_request(self):
        if not self._adapter_path:
            return None
        from vllm.lora.request import LoRARequest
        # Use a unique (name, id) per synced adapter so vLLM reloads weights.
        return LoRARequest(f"adapter_{self._lora_id}", self._lora_id, self._adapter_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def _visible_gpu_count() -> int:
    """Number of GPUs visible to this process (honours CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        items = [x for x in cvd.split(",") if x.strip() != ""]
        return len(items)
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 1


def _resolve_tensor_parallel_size(requested: int) -> int:
    """Clamp tensor_parallel_size to a value vLLM can actually start.

    Constraints:
      * 1 <= tp <= visible GPU count
      * tp must evenly divide the model's KV heads (vLLM requirement). For
        GQA models like Qwen3-4B (8 KV heads) valid TP ∈ {1,2,4,8}. We pick the
        largest valid value that is <= requested and <= visible GPUs.
    """
    requested = max(1, int(requested))
    n_visible = max(1, _visible_gpu_count())
    cap = min(requested, n_visible)
    # Valid divisors of a typical GQA KV-head count (8). Using powers of two is
    # safe across the common 4B/7B Qwen configs; if the true head count allows
    # more, a power-of-two choice is still always valid.
    valid = [v for v in (8, 4, 2, 1) if v <= cap]
    chosen = valid[0] if valid else 1
    if chosen != requested:
        print(
            f"[vllm] tensor_parallel_size {requested} -> {chosen} "
            f"(visible GPUs={n_visible}; must divide KV heads)",
            flush=True,
        )
    return chosen


def build_vllm_generator(
    config: dict,
    tokenizer,
    *,
    adapter_path: Optional[str] = None,
    mode: str = "eval",
) -> VLLMGenerator:
    """Read hardware + generation config and return a ready VLLMGenerator.

    mode: "eval"           → higher gpu_memory_utilization (eval-only, no HF model)
          "train"          → lower gpu_memory_utilization (leaves room for QLoRA HF model)
          "rollout_worker" → data-parallel rollout worker: owns its GPU alone (no HF
                             model shares it) so it may use a high memory fraction,
                             but it MUST be LoRA-enabled up front to receive per-epoch
                             sync_adapter() updates, and TP is forced to 1 (one GPU).
    """
    hw = config.get("hardware", {})
    gen = config.get("generation", {})
    mcfg = config.get("model", {})
    ft = config.get("finetuning", {})

    # Max model length: use the largest vllm_max_model_length across all stages.
    stage_defaults = (config.get("token_budget", {}) or {}).get("stage_defaults", {}) or {}
    lens = [int(v.get("vllm_max_model_length", 0)) for v in stage_defaults.values() if v]
    fallback_len = int(gen.get("max_model_length", 8192))
    max_model_len = max(lens + [fallback_len])

    # GPU memory utilization: train mode leaves headroom for QLoRA HF model;
    # a data-parallel rollout worker owns its GPU alone and may use a high fraction.
    if mode == "train":
        default_util = 0.45
    elif mode == "rollout_worker":
        default_util = float(hw.get("vllm_gpu_memory_utilization_dp", 0.85))
    else:
        default_util = 0.85
    # rollout_worker has its own util key so it is not capped by the train-shared one.
    if mode == "rollout_worker":
        gpu_util = float(hw.get("vllm_gpu_memory_utilization_dp", default_util))
    else:
        gpu_util = float(hw.get("vllm_gpu_memory_utilization", default_util))

    adapter_path = adapter_path or mcfg.get("lora_adapter") or None

    # max_lora_rank: use 4× the configured lora_r as a safety margin so that
    # different run configs with larger ranks don't require a vLLM restart.
    lora_r = int(ft.get("lora_r", 16))
    max_lora_rank = max(lora_r * 4, 64)

    # In train mode the trainer saves a new adapter each epoch and calls
    # sync_adapter(); the engine must be LoRA-enabled up front even if no adapter
    # exists yet (e.g. stage 1 epoch 1 starting from the base model). The same
    # holds for data-parallel rollout workers, which receive sync_adapter() too.
    enable_lora = mode in ("train", "rollout_worker")

    # Tensor-parallel size must (a) not exceed the visible GPU count and (b) evenly
    # divide the model's KV/attention heads — vLLM hard-crashes otherwise (e.g.
    # TP=3 on Qwen3-4B whose 8 KV heads are not divisible by 3). Clamp to the
    # nearest valid lower value with a warning instead of crashing.
    # A data-parallel rollout worker is pinned to a single GPU, so TP is always 1.
    tp_requested = 1 if mode == "rollout_worker" else int(hw.get("vllm_tensor_parallel_size", 1))
    tp = _resolve_tensor_parallel_size(tp_requested)

    return VLLMGenerator(
        model=mcfg["base_model"],
        tokenizer=tokenizer,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        dtype="bfloat16" if hw.get("bf16", True) else "float16",
        temperature=float(gen.get("temperature", 0.7)),
        top_p=float(gen.get("top_p", 0.95)),
        adapter_path=adapter_path,
        max_lora_rank=max_lora_rank,
        enforce_eager=bool(hw.get("vllm_enforce_eager", False)),
        enable_lora=enable_lora,
    )
