"""Independent per-rollout sampling seeds for MT-GRPO groups.

Version: ``p43_independent_rollout_sampling_v1``

Root cause fixed: DP vLLM workers previously shared LLM seed=0 and never set
``SamplingParams.seed``, so round-robin pairs (0,1)/(3,4)/(6,7) on synced workers
produced identical completions.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence, Union

ROLLOUT_SAMPLING_VERSION = "p43_independent_rollout_sampling_v1"
_MAX_SEED = 2**31 - 1


def stable_hash_u32(*parts: Any) -> int:
    """Stable cross-process hash → uint32 (not Python's salted ``hash()``)."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def derive_rollout_seed(
    *,
    base_seed: int,
    global_step: int,
    task_id: str,
    rollout_index: int,
    epoch: int = 0,
) -> int:
    """Unique seed for one logical rollout; independent of GPU placement."""
    h = stable_hash_u32(
        "p43_rollout_seed_v1",
        int(base_seed),
        int(global_step),
        int(epoch),
        str(task_id),
        int(rollout_index),
    )
    # Avoid 0 (some backends treat 0 as "unseeded").
    return (h % _MAX_SEED) or 1


def derive_turn_seed(rollout_seed: int, turn_idx: int) -> int:
    """Per-generate-call seed inside a multi-turn episode."""
    h = stable_hash_u32("p43_turn_seed_v1", int(rollout_seed), int(turn_idx))
    return (h % _MAX_SEED) or 1


def stamp_rollout_tasks(
    task: Dict[str, Any],
    *,
    num_generations: int,
    base_seed: int,
    global_step: int,
    epoch: int = 0,
) -> list:
    """Return ``num_generations`` shallow task copies with unique rollout meta."""
    tid = str(task.get("task_id") or task.get("sample_id") or "")
    out = []
    for i in range(int(num_generations)):
        t = dict(task)
        seed = derive_rollout_seed(
            base_seed=base_seed,
            global_step=global_step,
            task_id=tid,
            rollout_index=i,
            epoch=epoch,
        )
        t["_rollout_index"] = i
        t["_rollout_seed"] = seed
        t["_rollout_sampling_version"] = ROLLOUT_SAMPLING_VERSION
        t["_rollout_base_seed"] = int(base_seed)
        t["_rollout_global_step"] = int(global_step)
        t["_rollout_epoch"] = int(epoch)
        out.append(t)
    return out


def rollout_seeds_from_tasks(tasks: Sequence[Dict[str, Any]]) -> list:
    return [int(t.get("_rollout_seed") or 0) for t in tasks]


def sampling_source_hash() -> str:
    import inspect
    return hashlib.sha256(inspect.getsource(inspect.getmodule(derive_rollout_seed)).encode(
        "utf-8")).hexdigest()[:16]
