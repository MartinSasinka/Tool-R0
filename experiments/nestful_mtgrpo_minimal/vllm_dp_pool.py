"""Data-parallel rollout pool for vLLM-accelerated MT-GRPO training.

Motivation
----------
During training the HF QLoRA learner lives on GPU 0. Rollout *generation* (the
dominant cost: num_generations episodes per task, each multi-turn) is fully
INDEPENDENT of the HF model when vLLM is used — it needs only the tokenizer, the
executor and a vLLM engine. So we can run rollouts in parallel on the OTHER
GPUs, each worker owning a single vLLM engine (tensor_parallel_size=1), and feed
the results back to the learner on GPU 0.

Design (one worker per GPU, whole-episode in worker)
----------------------------------------------------
* Each worker process pins itself to ONE GPU (CUDA_VISIBLE_DEVICES set BEFORE
  importing torch/vllm), builds a single vLLM engine, and runs ENTIRE episodes
  (generate + tool-execute loop) — never touching the HF model.
* The worker also computes the training reward (strict OR partial, selected from
  ``config['reward']['train_policy']``) so that raw tool observations — which can
  be arbitrary, non-picklable Python objects — NEVER cross the process boundary.
* The worker returns a small, fully-picklable :class:`RolloutResult`: per-turn
  token-id lists (for the parent's log-prob pass), the episode reward, the
  per-turn reward sequence, and a few diagnostic scalars. The parent re-wraps the
  token-id lists as tensors and runs the existing GRPO update on GPU 0.

Reward policy across processes
------------------------------
The partial experiment selects its reward by monkeypatching
``grpo_train.episode_turn_reward_seq`` in the PARENT. Spawned workers would not
see that. Instead the worker picks the reward function explicitly from
``config['reward']['train_policy']`` (``partial_gold_trace`` → partial_reward,
else strict reward) and loads partial weights from ``config['partial_reward']``.
The parent's snapshot of ``sys.path`` is forwarded so ``partial_reward`` (which
lives in the sibling folder) is importable in the worker.

Opt-in
------
This whole machinery is OFF by default. It is only constructed when the caller
passes a non-empty GPU list (driven by ``hardware.rollout_data_parallel_gpus`` /
the ``ROLLOUT_DP_GPUS`` env var). The single-engine path is unchanged.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Serializable result (worker -> parent). Plain Python types only.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Everything the trainer needs from one episode, fully picklable.

    ``turn_token_ids`` is a list of (prompt_ids, completion_ids) Python int lists;
    the parent re-wraps them as 1-D LongTensors for the log-prob computation.
    """
    turn_token_ids: List[Tuple[List[int], List[int]]] = field(default_factory=list)
    episode_reward: float = 0.0
    r_seq: List[float] = field(default_factory=list)
    clipped_any: bool = False
    prompt_overflow: bool = False
    zero_tool_calls: bool = False
    num_tool_calls: int = 0
    stop_reason: Optional[str] = None
    first_error_turn: Optional[int] = None
    error: Optional[str] = None  # set if the episode crashed in the worker
    # Scalar-only reward diagnostics (sanitized in the worker) for group logging.
    reward_diag: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Reward-policy resolution (runs in the worker, no monkeypatch dependency)
# ─────────────────────────────────────────────────────────────────────────────

_STRICT_POLICY_ALIASES = ("strict", "strict_gold_trace", "strict_gold_trace_legacy")


def _ensure_v3_experiment_on_path() -> None:
    """Make experiments/nestful_synthetic_curriculum_v3 importable (lib.*)."""
    v3_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "nestful_synthetic_curriculum_v3",
    )
    # APPEND, don't insert(0): the v3 dir also ships its own run.py etc. and
    # must not shadow same-named modules of the minimal experiment.
    if os.path.isdir(v3_dir) and v3_dir not in sys.path:
        sys.path.append(v3_dir)


def resolve_reward_info(config: Dict[str, Any]) -> Tuple[Callable, Dict[str, Any]]:
    """Resolve config['reward']['train_policy'] to a reward function.

    Returns (fn, info) where info records exactly what was resolved:
        configured_policy / resolved_policy / reward_fn_module /
        reward_fn_name / fallback_used

    HARD-FAILS (ValueError) on an unknown policy unless the environment
    explicitly allows the strict fallback via ALLOW_STRICT_REWARD_FALLBACK=1.
    This replaces the previous SILENT fallback that invalidated the v3/v3.1
    pilots (audit Bug 1).
    """
    configured = str((config.get("reward", {}) or {}).get("train_policy", "strict"))
    policy = configured.lower()
    fallback_used = False

    if policy in ("partial_gold_trace", "partial"):
        import partial_reward
        partial_reward.set_weights_from_config(config)
        fn = partial_reward.episode_turn_reward_seq
    elif policy in ("execution_aware_v2", "execution_v2"):
        import execution_reward_v2
        execution_reward_v2.set_weights_from_config(config)
        fn = execution_reward_v2.episode_turn_reward_seq
    elif policy in ("partial_gold_trace_v2", "partial_v2"):
        import partial_reward_v2
        partial_reward_v2.set_weights_from_config(config)
        fn = partial_reward_v2.episode_turn_reward_seq
    elif policy in ("execution_aware", "execution"):
        import execution_reward
        execution_reward.set_weights_from_config(config)
        fn = execution_reward.episode_turn_reward_seq
    elif policy in ("execution_aware_v2_1_motif", "v2_1_motif", "motif"):
        _ensure_v3_experiment_on_path()
        from lib import reward_motif
        fn = reward_motif.episode_turn_reward_seq
    elif policy in ("execution_aware_v3_1_stepwise", "v3_1_stepwise", "stepwise"):
        _ensure_v3_experiment_on_path()
        from lib import reward_v3_1
        fn = reward_v3_1.episode_turn_reward_seq
    elif policy in _STRICT_POLICY_ALIASES:
        from reward import episode_turn_reward_seq as strict_seq
        fn = strict_seq
    else:
        if os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") == "1":
            print(f"[reward_dispatch] WARNING: unknown reward policy '{configured}' — "
                  f"falling back to STRICT gold-trace reward because "
                  f"ALLOW_STRICT_REWARD_FALLBACK=1", flush=True)
            from reward import episode_turn_reward_seq as strict_seq
            fn = strict_seq
            fallback_used = True
        else:
            raise ValueError(
                f"[reward_dispatch] Unknown reward policy '{configured}'. "
                f"Known: partial_gold_trace, execution_aware_v2, partial_gold_trace_v2, "
                f"execution_aware, execution_aware_v2_1_motif, "
                f"execution_aware_v3_1_stepwise, strict. "
                f"Refusing to silently fall back to the strict binary reward "
                f"(set ALLOW_STRICT_REWARD_FALLBACK=1 to override — NOT recommended)."
            )

    resolved_policy = getattr(fn, "reward_policy", None) or (
        "strict" if fallback_used or policy in _STRICT_POLICY_ALIASES else configured)
    info = {
        "configured_policy": configured,
        "resolved_policy": resolved_policy,
        "reward_fn_module": getattr(fn, "__module__", "?"),
        "reward_fn_name": getattr(fn, "__name__", "?"),
        "fallback_used": fallback_used,
    }
    return fn, info


def _resolve_reward_fn(config: Dict[str, Any]) -> Callable:
    """Return the episode_turn_reward_seq matching config['reward']['train_policy']."""
    fn, _ = resolve_reward_info(config)
    return fn


def _encode_for_logprob(tokenizer, messages, completion_text: str) -> Tuple[List[int], List[int]]:
    """Re-tokenise (messages, completion) as plain int lists.

    Mirrors grpo_train._retokenize_for_logprob token IDs exactly (chat-template
    tokenisation is deterministic), but returns lists so the result is cheap and
    safe to ship across the process boundary.
    """
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    # Some templates return a nested [[...]] when batched; flatten one level.
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    prompt_ids = [int(x) for x in enc]
    comp_ids = [int(x) for x in tokenizer.encode(completion_text, add_special_tokens=False)]
    return prompt_ids, comp_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Per-episode rollout (worker side). Testable in-process via generate_fn inject.
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_collect(
    *,
    tokenizer,
    task: Dict[str, Any],
    config: Dict[str, Any],
    registry,
    generate_fn: Callable[[List[Dict[str, str]], int], Dict[str, Any]],
    reward_fn: Callable,
    gold_obs=None,
) -> RolloutResult:
    """Run ONE training episode with vLLM-style generation and collect a
    fully-picklable RolloutResult (token-id lists + reward + diagnostics).

    Mirrors the vLLM branch of grpo_train._rollout_episode_for_train, but the
    reward is computed here (worker side) and observations stay local.
    """
    from rollout import Trajectory, Turn, get_stage_token_budget
    from prompt import build_messages, format_tool_response
    from parser import parse_tool_call
    from executor import ToolExecutor
    from reward import compute_gold_observations, strict_gold_trace_reward

    gen = config.get("generation", {})
    exec_cfg = config.get("executor", {})
    gold_n = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    budget = get_stage_token_budget(config, gold_n, "train")
    max_new_tokens = budget["max_new_tokens"]

    if gold_obs is None:
        gold_obs = compute_gold_observations(task, registry)

    executor = ToolExecutor(
        task, registry=registry, mode=exec_cfg.get("mode", "auto"),
        ibm_call_timeout=float(exec_cfg.get("ibm_call_timeout", 30.0)),
    )
    traj = Trajectory(task["task_id"], gold_n, gold_n, executor_mode=executor.mode)
    turn_token_ids: List[Tuple[List[int], List[int]]] = []
    history: List[Dict[str, str]] = []

    # v2: train turn budget = gold_n + max_extra_turns_train (cap +4); default 0
    # reproduces the legacy gold_n-turn loop exactly.
    _extra = int(config.get("train", {}).get("max_extra_turns_train", 0))
    _train_max_turns = max(1, min(gold_n + _extra, gold_n + 4))
    for _turn_idx in range(_train_max_turns):
        messages = build_messages(task, history)
        g = generate_fn(messages, max_new_tokens)
        if g.get("prompt_overflow"):
            traj.prompt_overflow = True
            traj.clipped_any = True
            traj.stop_reason = "prompt_overflow"
            break
        text = g["text"]
        clipped = bool(g.get("clipped", False))
        p_ids, c_ids = _encode_for_logprob(tokenizer, messages, text)
        turn = Turn(_turn_idx, text, prompt_tokens=len(p_ids),
                    completion_tokens=len(c_ids), clipped_completion=clipped)
        turn_token_ids.append((p_ids, c_ids))
        history.append({"role": "assistant", "content": text})

        if clipped:
            traj.clipped_any = True
            turn.fail_reason = "clipped_completion"
            traj.turns.append(turn)
            traj.stop_reason = "clipped"
            break

        pr = parse_tool_call(text)
        if pr.is_terminal:
            turn.is_terminal = True
            traj.turns.append(turn)
            traj.stop_reason = "terminal"
            break
        if not pr.ok:
            turn.fail_reason = f"parse:{pr.reason}"
            traj.turns.append(turn)
            traj.stop_reason = "parse_fail"
            break

        call = pr.call
        turn.parsed_call = call
        res = executor.execute(call)
        turn.observation = res.observation
        if res.error is not None:
            turn.fail_reason = f"exec:{res.error}"
            traj.turns.append(turn)
            traj.stop_reason = "executor_error"
            break
        traj.final_observation = res.observation
        traj.turns.append(turn)
        history.append({"role": "user", "content": format_tool_response(call, res.observation)})

    if traj.stop_reason is None:
        traj.stop_reason = "max_turns"

    # Reward (policy-selected) computed HERE so observations never leave the worker.
    rinfo = reward_fn(traj, task, gold_obs)
    # strict first_error_turn for logging parity with the single-engine path.
    strict_diag = strict_gold_trace_reward(traj, task, gold_obs).diagnostics

    return RolloutResult(
        turn_token_ids=turn_token_ids,
        episode_reward=float(rinfo["episode_reward"]),
        r_seq=[float(x) for x in rinfo["r_seq"]],
        clipped_any=bool(traj.clipped_any),
        prompt_overflow=bool(traj.prompt_overflow),
        zero_tool_calls=bool(traj.zero_tool_calls),
        num_tool_calls=int(traj.num_tool_calls),
        stop_reason=traj.stop_reason,
        first_error_turn=strict_diag.get("first_error_turn"),
        reward_diag=_sanitize_diag(rinfo.get("diagnostics") or {}),
    )


def _sanitize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only cheap picklable scalars (and short float lists) for transport."""
    out: Dict[str, Any] = {}
    for k, v in diag.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, list) and len(v) <= 16 and all(
                isinstance(x, (bool, int, float)) for x in v):
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process main loop
# ─────────────────────────────────────────────────────────────────────────────

def _worker_main(worker_id: int, gpu: int, config: Dict[str, Any],
                 adapter_path: Optional[str], extra_sys_path: List[str],
                 in_q, out_q) -> None:
    """Worker entry point. Pins to ONE GPU, builds a vLLM engine, serves rollouts.

    Protocol (messages on in_q):
        ("rollout", (req_id, task))   -> out_q.put((req_id, RolloutResult))
        ("sync",    adapter_path)     -> out_q.put((("__ack__", worker_id), "sync"))
        ("ping",    None)             -> out_q.put((("__ack__", worker_id), "ready"))
        ("stop",    None)             -> exits
    """
    # MUST happen before importing torch / vllm so the worker sees only its GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for p in reversed(extra_sys_path):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    try:
        # Resolve the reward FIRST so a dispatch failure aborts before any
        # engine is built and before any rollout happens (audit Bug 1).
        reward_fn, reward_info = resolve_reward_info(config)
        print(
            f"[dp_worker {worker_id}] reward.train_policy={reward_info['configured_policy']} "
            f"resolved_reward_fn={reward_info['reward_fn_module']}."
            f"{reward_info['reward_fn_name']} "
            f"resolved_policy={reward_info['resolved_policy']} "
            f"fallback_used={str(reward_info['fallback_used']).lower()}",
            flush=True,
        )

        from transformers import AutoTokenizer
        from vllm_generate import build_vllm_generator

        base_model = config["model"]["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # No HF model shares this GPU -> the engine may use a high memory fraction.
        vgen = build_vllm_generator(config, tokenizer,
                                    adapter_path=adapter_path, mode="rollout_worker")
        out_q.put((("__ack__", worker_id), "ready"))
    except Exception as exc:  # noqa: BLE001 — report init failure, do not hang parent
        import traceback
        out_q.put((("__ack__", worker_id), f"init_error: {exc}\n{traceback.format_exc()}"))
        return

    while True:
        cmd, payload = in_q.get()
        if cmd == "stop":
            break
        if cmd == "ping":
            out_q.put((("__ack__", worker_id), "ready"))
            continue
        if cmd == "sync":
            try:
                vgen.sync_adapter(payload)
            except Exception as exc:  # noqa: BLE001
                out_q.put((("__ack__", worker_id), f"sync_error: {exc}"))
                continue
            out_q.put((("__ack__", worker_id), "sync"))
            continue
        if cmd == "rollout":
            req_id, task = payload
            try:
                res = run_episode_collect(
                    tokenizer=tokenizer, task=task, config=config,
                    registry=_worker_registry(config), generate_fn=vgen.generate_fn,
                    reward_fn=reward_fn, gold_obs=None,
                )
            except Exception as exc:  # noqa: BLE001 — never kill the worker on one task
                import traceback
                res = RolloutResult(error=f"{exc}\n{traceback.format_exc()}")
            out_q.put((req_id, res))
            continue
        # Unknown command — ignore.

    # graceful engine teardown best-effort
    try:
        del vgen
    except Exception:
        pass


_WORKER_REGISTRY_CACHE = {"reg": None, "built": False}


def _worker_registry(config: Dict[str, Any]):
    """Build (once per worker) the executor registry from config paths.

    Replicates ``run.build_registry`` WITHOUT importing run.py — the partial
    experiment's run.py does not define build_registry, and importing either
    run.py inside a worker would needlessly re-run its heavy module body.
    """
    if not _WORKER_REGISTRY_CACHE["built"]:
        try:
            import executor as _ex
            from executor import IBMFunctionRegistry, detect_ibm_functions_dir
            paths = config.get("paths", {}) or {}
            funcs_dir = detect_ibm_functions_dir(
                explicit=paths.get("ibm_functions_dir"),
                repo_root=os.path.dirname(os.path.abspath(_ex.__file__)),
            )
            _WORKER_REGISTRY_CACHE["reg"] = (
                IBMFunctionRegistry(funcs_dir) if funcs_dir else None
            )
        except Exception:
            _WORKER_REGISTRY_CACHE["reg"] = None
        _WORKER_REGISTRY_CACHE["built"] = True
    return _WORKER_REGISTRY_CACHE["reg"]


# ─────────────────────────────────────────────────────────────────────────────
#  Parent-side pool
# ─────────────────────────────────────────────────────────────────────────────

class DataParallelRolloutPool:
    """Manages N worker processes (one vLLM engine per GPU) for parallel rollouts."""

    def __init__(self, config: Dict[str, Any], gpus: List[int],
                 adapter_path: Optional[str] = None, *, start_timeout: float = 1800.0):
        import multiprocessing as mp

        self.gpus = list(gpus)
        if not self.gpus:
            raise ValueError("DataParallelRolloutPool requires a non-empty GPU list")

        # Parent-side reward-dispatch assertion: resolve with the EXACT same
        # resolver the workers use, and abort BEFORE any rollout when the
        # configured policy cannot be honoured (audit Bug 1). resolve_reward_info
        # raises on unknown policies unless ALLOW_STRICT_REWARD_FALLBACK=1.
        _fn, self.reward_info = resolve_reward_info(config)
        configured = self.reward_info["configured_policy"]
        resolved = self.reward_info["resolved_policy"]
        print(f"[dp_pool] parent reward check: configured={configured} "
              f"resolved={resolved} "
              f"fn={self.reward_info['reward_fn_module']}."
              f"{self.reward_info['reward_fn_name']} "
              f"fallback_used={str(self.reward_info['fallback_used']).lower()}",
              flush=True)
        if self.reward_info["fallback_used"] and \
                os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") != "1":
            raise RuntimeError(
                f"[dp_pool] reward fallback engaged for policy '{configured}' but "
                f"ALLOW_STRICT_REWARD_FALLBACK != 1 — aborting before any rollout.")
        _is_strict = (self.reward_info["reward_fn_module"] == "reward")
        _strict_requested = configured.lower() in _STRICT_POLICY_ALIASES
        if _is_strict and not _strict_requested and not self.reward_info["fallback_used"]:
            raise RuntimeError(
                f"[dp_pool] configured reward '{configured}' resolved to the STRICT "
                f"gold-trace reward without an explicit request — aborting.")
        self._ctx = mp.get_context("spawn")
        self._in_qs = []
        self._out_q = self._ctx.Queue()
        self._procs = []
        extra_sys_path = list(sys.path)

        for wid, gpu in enumerate(self.gpus):
            in_q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_worker_main,
                args=(wid, gpu, config, adapter_path, extra_sys_path, in_q, self._out_q),
                # MUST be non-daemonic: vLLM v1 spawns its own EngineCore subprocess
                # inside each worker, and Python forbids a daemonic process from
                # having children (AssertionError). close() handles teardown.
                daemon=False,
            )
            p.start()
            self._in_qs.append(in_q)
            self._procs.append(p)

        # Wait for every worker to report readiness (or surface an init error).
        self._await_acks(len(self.gpus), timeout=start_timeout, what="startup")
        print(f"[dp_pool] {len(self.gpus)} rollout workers ready on GPUs {self.gpus}",
              flush=True)

    # ── public API ──────────────────────────────────────────────────────────

    def rollout_many(self, tasks: List[Dict[str, Any]]) -> List[RolloutResult]:
        """Run one episode per task across the workers; results in input order."""
        n = len(tasks)
        for i, task in enumerate(tasks):
            wid = i % len(self._in_qs)
            self._in_qs[wid].put(("rollout", (i, task)))
        results: Dict[int, RolloutResult] = {}
        while len(results) < n:
            req_id, res = self._out_q.get()
            if isinstance(req_id, tuple):  # stray ack — ignore
                continue
            results[req_id] = res
        return [results[i] for i in range(n)]

    def sync_adapter(self, adapter_path: Optional[str], timeout: float = 600.0) -> None:
        for in_q in self._in_qs:
            in_q.put(("sync", adapter_path))
        self._await_acks(len(self._in_qs), timeout=timeout, what="sync")

    def close(self) -> None:
        # Ask workers to exit cleanly (lets vLLM tear down its EngineCore subprocess).
        for in_q in self._in_qs:
            try:
                in_q.put(("stop", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=15)
            if p.is_alive():  # last resort
                try:
                    p.kill()
                except Exception:
                    pass

    # ── internals ─────────────────────────────────────────────────────────────

    def _await_acks(self, count: int, *, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        got = 0
        while got < count:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"[dp_pool] timed out waiting for {what} acks "
                                   f"({got}/{count})")
            try:
                tag, status = self._out_q.get(timeout=min(remaining, 30))
            except Exception:
                continue
            if isinstance(tag, tuple) and tag[0] == "__ack__":
                if isinstance(status, str) and status.startswith(("init_error", "sync_error")):
                    raise RuntimeError(f"[dp_pool] worker {tag[1]} {what} failed: {status}")
                got += 1
            # Non-ack messages during startup/sync are unexpected; ignore.
