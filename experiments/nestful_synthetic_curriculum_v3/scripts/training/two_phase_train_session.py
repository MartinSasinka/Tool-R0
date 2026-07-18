"""In-process two-phase GRPO training session (single Python interpreter).

Loads the HF learner once, runs Phase 1 and Phase 2 with a *shared* AdamW
optimizer and monotonic ``global_step``. Rollout workers stay up between
phases; the pool is torn down only when the training session ends (before
deferred C1/C2 eval with TP=4 on all GPUs).
"""
from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PARTIAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_partial"))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))

for p in reversed((_V3, _PARTIAL, _MINIMAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

_v3_run_path = os.path.join(_V3, "run.py")
_spec = importlib.util.spec_from_file_location("v3_run_module", _v3_run_path)
assert _spec and _spec.loader
v3_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3_run)
v3_run._hook_select_train_reward()
os.environ.setdefault("REWARD_POLICY", "execution_aware_v3_2_dense")

import grpo_train  # noqa: E402
from data import load_tasks  # noqa: E402

import importlib.util as _ilu

_minimal_run_path = os.path.join(_MINIMAL, "run.py")
_mspec = _ilu.spec_from_file_location("minimal_run", _minimal_run_path)
assert _mspec and _mspec.loader
minimal_run = _ilu.module_from_spec(_mspec)
_mspec.loader.exec_module(minimal_run)

_apply_overrides = minimal_run._apply_overrides
_parse_gpu_list = minimal_run._parse_gpu_list
_wandb_finish = minimal_run._wandb_finish
_wandb_init = minimal_run._wandb_init
_wandb_log_eval = minimal_run._wandb_log_eval
build_registry = minimal_run.build_registry
load_config = minimal_run.load_config
load_model_and_tokenizer = minimal_run.load_model_and_tokenizer

from scripts.training.two_phase_utils import (  # noqa: E402
    adapter_dir_hash,
    json_safe_summary,
    task_seed,
    verify_epoch_coverage,
    wait_for_gpu_memory,
)


class TwoPhaseTrainSession:
    """Single-process trainer with optional continuous optimizer state."""

    def __init__(
        self,
        config_path: str,
        overrides: List[str],
        *,
        seed: int = 42,
        data_seed: int = 42,
        rollout_seed: int = 42,
    ) -> None:
        self.seed = int(seed)
        self.data_seed = int(data_seed)
        self.rollout_seed = int(rollout_seed)
        random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass

        self.config = load_config(config_path)
        _apply_overrides(self.config, overrides)
        self.config.setdefault("experiment", {})["seed"] = self.seed
        v3_run._partial._select_train_reward(self.config)

        self.registry = build_registry(self.config)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.global_step = 0
        self.task_prev_mean: Dict[str, float] = {}
        self.task_best_mean: Dict[str, float] = {}
        self.task_prev_rollout_rewards: Dict[str, List[float]] = {}
        self.rollout_pool = None
        self.vllm_gen = None
        self._last_pool_pids: List[int] = []
        self._dp_gpus: List[int] = []

    def load_learner(self, checkpoint: Optional[str] = None) -> None:
        if self.model is not None:
            return
        # Pin learner to GPU 0 *before* from_pretrained — start_rollout_workers()
        # runs later but the model must not use device_map=auto across all GPUs.
        if self._use_pool():
            self.config.setdefault("hardware", {})["hf_device_map"] = {"": 0}
            print("[session] hf_device_map pinned to GPU 0 (rollout pool on other GPUs)",
                  flush=True)
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config, checkpoint, for_training=True)

    def _use_pool(self) -> bool:
        hw = self.config.get("hardware", {})
        dp = _parse_gpu_list(hw.get("rollout_data_parallel_gpus"))
        return bool(hw.get("use_vllm", False)) and bool(dp)

    def start_rollout_workers(self, adapter_path: Optional[str] = None) -> List[int]:
        """Start vLLM DP rollout pool on GPUs 1..N (learner stays on GPU 0)."""
        self.shutdown_rollout_workers()
        hw = self.config.get("hardware", {})
        self._dp_gpus = _parse_gpu_list(hw.get("rollout_data_parallel_gpus"))
        if not self._use_pool():
            if hw.get("use_vllm", False):
                from vllm_generate import build_vllm_generator
                self.vllm_gen = build_vllm_generator(
                    self.config, self.tokenizer, adapter_path=adapter_path, mode="train")
            return []
        self.config.setdefault("hardware", {})["hf_device_map"] = {"": 0}
        from vllm_dp_pool import DataParallelRolloutPool
        self.rollout_pool = DataParallelRolloutPool(
            self.config, self._dp_gpus, adapter_path=adapter_path)
        self._last_pool_pids = list(self.rollout_pool.worker_pids)
        print(f"[session] rollout pool PIDs={self._last_pool_pids}", flush=True)
        return self._last_pool_pids

    def shutdown_rollout_workers(self) -> List[int]:
        pids = list(self._last_pool_pids)
        if self.rollout_pool is not None:
            pids = list(self.rollout_pool.worker_pids)
            try:
                self.rollout_pool.close()
            finally:
                self.rollout_pool = None
        self.vllm_gen = None
        self._last_pool_pids = []
        wait_for_gpu_memory(self._dp_gpus or None)
        return pids

    def sync_rollout_policy(self, adapter_path: str, *, label: str) -> str:
        """Push canonical checkpoint weights to rollout workers and log parity."""
        adapter_path = os.path.abspath(adapter_path)
        adapter_hash = adapter_dir_hash(adapter_path)
        print(f"[session] learner checkpoint/version: {label}", flush=True)
        print(f"[session] adapter_hash={adapter_hash}", flush=True)
        if self.rollout_pool is not None:
            self.rollout_pool.sync_adapter(adapter_path)
            print(f"[session] rollout worker weights version: {label}", flush=True)
            print(
                f"[session] all workers acknowledged current adapter hash "
                f"{adapter_hash[:16]}...",
                flush=True,
            )
        elif self.vllm_gen is not None:
            self.vllm_gen.sync_adapter(adapter_path)
            print(f"[session] vLLM generator synced to {label}", flush=True)
        else:
            print("[session] WARNING: no rollout backend to sync", flush=True)
        return adapter_hash

    def train_phase(
        self,
        *,
        dataset_path: str,
        train_out: str,
        phase_name: str,
        max_train_tasks: int = 0,
        expected_rows: Optional[int] = None,
        wandb_run_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """One exact epoch on ``dataset_path``; returns (adapter_dir, summary)."""
        if self.model is None:
            raise RuntimeError("call load_learner() first")

        os.makedirs(train_out, exist_ok=True)
        ckpt_dir = os.path.join(train_out, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.config["model"]["output_adapter_dir"] = ckpt_dir
        self.config["experiment"]["output_dir"] = train_out

        data_seed = task_seed(data_seed=self.data_seed, task_idx=0, phase=phase_name)
        tasks = load_tasks(
            dataset_path,
            stage=None,
            max_tasks=max_train_tasks or None,
            seed=data_seed,
        )
        if expected_rows and max_train_tasks == 0:
            if len(tasks) != expected_rows:
                raise SystemExit(
                    f"[session] {phase_name}: loaded {len(tasks)} tasks, "
                    f"expected {expected_rows}")

        wandb_run = _wandb_init("train", self.config, checkpoint=None)
        if wandb_run is not None:
            try:
                import wandb
                wandb.config.update({
                    "phase": phase_name,
                    "dataset": os.path.abspath(dataset_path),
                    "n_tasks": len(tasks),
                    "continuous_training": self.global_step > 0,
                    "global_step_start": self.global_step,
                }, allow_val_change=True)
            except Exception:
                pass
            os.environ["WANDB_RUN_NAME"] = wandb_run_name

        log_path = os.path.join(train_out, "train_log.jsonl")
        summary = grpo_train.train(
            self.config,
            self.model,
            self.tokenizer,
            self.registry,
            tasks,
            log_path,
            vllm_gen=self.vllm_gen,
            rollout_pool=self.rollout_pool,
            wandb_run=wandb_run,
            optimizer=self.optimizer,
            global_step_start=self.global_step,
            log_append=self.global_step > 0,
            task_prev_mean=self.task_prev_mean,
            task_best_mean=self.task_best_mean,
            task_prev_rollout_rewards=self.task_prev_rollout_rewards,
            phase_name=phase_name,
        )

        self.optimizer = summary.pop("_optimizer", self.optimizer)
        self.global_step = int(summary.get("steps", self.global_step))

        cov = verify_epoch_coverage(
            dataset_path, log_path, expected_rows=expected_rows if not max_train_tasks else None)
        if not cov["ok"] and max_train_tasks == 0:
            raise SystemExit(f"[session] {phase_name} epoch coverage failed: {cov}")

        adapter = os.path.join(ckpt_dir, "adapter_epoch_1")
        if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
            raise SystemExit(f"[session] no adapter saved at {adapter}")

        safe = json_safe_summary(summary)
        safe["epoch_coverage"] = cov
        with open(os.path.join(train_out, "train_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(safe, fh, indent=2, ensure_ascii=False)

        _wandb_log_eval(wandb_run, {k: v for k, v in safe.items()
                                    if isinstance(v, (int, float, bool))},
                        prefix=f"train_summary/{phase_name}")
        _wandb_finish(wandb_run)
        return adapter, safe

    def close(self) -> None:
        self.shutdown_rollout_workers()
