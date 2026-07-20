"""Shared helpers for the two-phase v5 GRPO orchestrator."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

_V3 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
_DEVTEST_MANIFEST = os.path.join(_MINIMAL, "data", "splits", "nestful_devtest_manifest.json")

REQUIRED_REGISTRY_VERSION = "5.0.2"
REQUIRED_REWARD = "execution_aware_v3_2_dense"
REQUIRED_EXECUTOR = "synthetic"


def rollout_seed(*, base: int, task_idx: int, rollout_idx: int) -> int:
    """Deterministic per-task/per-rollout seed derivation."""
    h = hashlib.sha256(f"{base}:{task_idx}:{rollout_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def task_seed(*, data_seed: int, task_idx: int, phase: str) -> int:
    h = hashlib.sha256(f"{data_seed}:{phase}:{task_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def count_jsonl_rows(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def dataset_sample_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("sample_id") or row.get("task_id") or row.get("id")
            if sid is not None:
                ids.append(str(sid))
    return ids


def audit_dataset_ids(path: str) -> Dict[str, Any]:
    ids = dataset_sample_ids(path)
    counts = Counter(ids)
    dups = {k: v for k, v in counts.items() if v > 1}
    return {
        "path": os.path.abspath(path),
        "rows": len(ids),
        "unique_sample_ids": len(counts),
        "duplicate_sample_ids": dups,
        "ok": not dups and len(ids) == len(counts),
    }


def verify_dev_test_disjoint(dev_path: str) -> Dict[str, Any]:
    """Ensure dev task IDs are fixed and disjoint from the held-out test split."""
    if not os.path.isfile(_DEVTEST_MANIFEST):
        raise SystemExit(f"[two-phase] missing dev/test manifest: {_DEVTEST_MANIFEST}")
    with open(_DEVTEST_MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    dev_ids: Set[str] = set(man.get("dev_sample_ids") or [])
    if not dev_ids:
        raise SystemExit("[two-phase] devtest manifest has no dev_sample_ids")

    dev_path = os.path.abspath(dev_path)
    seen: Set[str] = set()
    with open(dev_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or row.get("id"))
            seen.add(sid)

    test_path = man.get("test_path")
    if test_path and not os.path.isabs(test_path):
        test_path = os.path.join(_MINIMAL, "data", "splits", "nestful_test.jsonl")
    test_ids: Set[str] = set()
    if test_path and os.path.isfile(test_path):
        with open(test_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                test_ids.add(str(row.get("sample_id") or row.get("task_id") or row.get("id")))

    overlap = dev_ids & test_ids
    extra_in_dev_file = seen - dev_ids
    missing_from_dev_file = dev_ids - seen
    ok = (not overlap and not extra_in_dev_file and not missing_from_dev_file
          and len(seen) == len(dev_ids))
    return {
        "dev_manifest_ids": len(dev_ids),
        "dev_file_ids": len(seen),
        "test_ids": len(test_ids),
        "overlap_with_test": sorted(overlap)[:10],
        "n_overlap": len(overlap),
        "extra_in_dev_file": len(extra_in_dev_file),
        "missing_from_dev_file": len(missing_from_dev_file),
        "ok": ok,
    }


def verify_epoch_coverage(
    dataset_path: str,
    train_log_path: str,
    *,
    expected_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """One epoch = each dataset row seen exactly once in train_log task_id records."""
    expected = set(dataset_sample_ids(dataset_path))
    if expected_rows is not None and len(expected) != expected_rows:
        raise SystemExit(
            f"[two-phase] dataset row count {len(expected)} != expected {expected_rows}")

    seen: List[str] = []
    if os.path.isfile(train_log_path):
        with open(train_log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "task_id" in rec and "mean_reward" in rec:
                    seen.append(str(rec["task_id"]))

    seen_counts = Counter(seen)
    dup = {k: v for k, v in seen_counts.items() if v > 1}
    missing = expected - set(seen_counts)
    extra = set(seen_counts) - expected
    return {
        "expected_unique": len(expected),
        "logged_unique": len(seen_counts),
        "duplicate_task_ids": dup,
        "missing_task_ids": sorted(missing)[:20],
        "n_missing": len(missing),
        "n_extra": len(extra),
        "ok": (not dup and not missing and not extra
               and len(seen_counts) == len(expected)),
    }


def collect_repro_manifest(*, git: dict, registry_version: str, registry_hash: str,
                           datasets: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Environment + model provenance for run_manifest / W&B."""
    env: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import importlib.metadata as md
        for pkg in ("torch", "transformers", "trl", "peft", "vllm"):
            try:
                env[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                env[pkg] = None
    except ImportError:
        pass
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            env["nvidia_driver"] = r.stdout.strip().splitlines()[0]
    except OSError:
        pass

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    model_revision = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        model_revision = getattr(info, "sha", None)
    except Exception:
        pass

    return {
        "seeds": {
            "SEED": int(os.environ.get("SEED", "42")),
            "DATA_SEED": int(os.environ.get("DATA_SEED", "42")),
            "ROLLOUT_SEED": int(os.environ.get("ROLLOUT_SEED", "42")),
        },
        "git": git,
        "registry_version": registry_version,
        "registry_hash": registry_hash,
        "model": {"id": model_id, "revision": model_revision},
        "environment": env,
        "datasets": [
            {"path": os.path.abspath(p), "sha256": h, "rows": count_jsonl_rows(p)}
            for p, h in datasets
        ],
        "training_mode": "continuous_in_process_two_phase",
    }


def assert_canonical_training(*, executor_mode: str, reward_policy: str,
                              registry_version: str) -> None:
    if executor_mode != REQUIRED_EXECUTOR:
        raise SystemExit(
            f"[two-phase] ABORT: executor.mode must be {REQUIRED_EXECUTOR!r}, "
            f"got {executor_mode!r} (gold_replay disabled)")
    if reward_policy != REQUIRED_REWARD:
        raise SystemExit(
            f"[two-phase] ABORT: reward must be {REQUIRED_REWARD!r}, "
            f"got {reward_policy!r}")
    if registry_version != REQUIRED_REGISTRY_VERSION:
        raise SystemExit(
            f"[two-phase] ABORT: registry must be {REQUIRED_REGISTRY_VERSION!r}, "
            f"got {registry_version!r}")


def wait_for_gpu_memory(
    gpus: Optional[List[int]] = None,
    *,
    timeout_s: float = 180.0,
    min_free_mib: int = 1024,
) -> None:
    """Poll GPU free memory until rollout workers have released VRAM."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            import torch
            if not torch.cuda.is_available():
                return
            ids = gpus if gpus is not None else list(range(torch.cuda.device_count()))
            ok = True
            for gid in ids:
                free, _total = torch.cuda.mem_get_info(gid)
                if free < min_free_mib * (1024 ** 2):
                    ok = False
                    break
            if ok:
                return
        except Exception:
            return
        time.sleep(2.0)
    print("[two-phase] WARNING: GPU memory wait timed out; proceeding to eval anyway",
          flush=True)


def prep_gpus_for_eval(*, min_free_mib: int = 15 * 1024) -> None:
    """Best-effort cleanup before EVAL_TP subprocess (zombie EngineCore / CUDA cache)."""
    import subprocess
    for pattern in ("VLLM::EngineCore", "VLLM::Worker"):
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"[eval-prep] cuda:{i} {free/1e9:.1f}/{total/1e9:.1f} GB free",
                      flush=True)
    except Exception as exc:
        print(f"[eval-prep] WARNING: cuda cleanup skipped: {exc}", flush=True)
    wait_for_gpu_memory(None, timeout_s=90.0, min_free_mib=min_free_mib)


def json_safe_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in summary.items() if not str(k).startswith("_")}


def adapter_dir_hash(adapter_dir: str) -> str:
    """SHA256 over sorted adapter artifact file names + contents."""
    h = hashlib.sha256()
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(adapter_dir)
    for fn in sorted(os.listdir(adapter_dir)):
        if fn == "checkpoint_manifest.json":
            continue
        if fn.endswith((".safetensors", ".bin", ".json", ".pt")):
            fp = os.path.join(adapter_dir, fn)
            if os.path.isfile(fp):
                h.update(fn.encode())
                with open(fp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
    return h.hexdigest()


def verify_adapter_dir(adapter_dir: str) -> Dict[str, Any]:
    """Ensure a LoRA adapter directory is complete and readable."""
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg):
        return {"ok": False, "error": "missing adapter_config.json"}
    has_weights = any(
        os.path.isfile(os.path.join(adapter_dir, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_weights:
        return {"ok": False, "error": "missing adapter_model.safetensors or .bin"}
    files = sorted(os.listdir(adapter_dir))
    return {
        "ok": True,
        "files": files,
        "adapter_hash": adapter_dir_hash(adapter_dir),
    }


def atomic_publish_checkpoint(
    src_adapter: str,
    dest_dir: str,
    *,
    label: str,
) -> Dict[str, Any]:
    """Publish adapter atomically: ``dest.tmp`` → verify → manifest → rename.

    State files must mark a phase complete only *after* this returns.
    """
    src = os.path.abspath(src_adapter)
    dest = os.path.abspath(dest_dir)
    tmp = dest + ".tmp"
    parent = os.path.dirname(dest) or "."
    os.makedirs(parent, exist_ok=True)

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)

    verified = verify_adapter_dir(tmp)
    if not verified["ok"]:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"[checkpoint {label}] verify failed: {verified}")

    manifest = {
        "label": label,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "source": src,
        "adapter_hash": verified["adapter_hash"],
        "files": verified["files"],
    }
    with open(os.path.join(tmp, "checkpoint_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    manifest["path"] = dest
    return manifest


def discard_incomplete_checkpoint(dest_dir: str) -> None:
    """Remove a partial checkpoint directory and its ``.tmp`` staging copy."""
    dest = os.path.abspath(dest_dir)
    for path in (dest, dest + ".tmp"):
        if os.path.isdir(path):
            shutil.rmtree(path)
