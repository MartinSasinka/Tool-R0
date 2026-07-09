"""Run manifest: provenance record written next to every eval batch (and, in P2,
every training run).

Captures everything needed to reproduce a number: git commit + dirty flag, the
invoking command, config hash, dataset paths + SHA256s, seed, decoding settings,
host and library versions. Pure stdlib except optional torch/vllm version probes.

Usage as a library:
    from run_manifest import build_manifest, write_manifest
Usage as a CLI (writes manifest for ad-hoc contexts):
    python run_manifest.py --out manifest.json --dataset path.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:  # both import styles: package (scripts.lib) and same-dir script
    from .paths import REPO_ROOT, dataset_info  # type: ignore
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import REPO_ROOT, dataset_info  # type: ignore

MANIFEST_VERSION = 1


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_state() -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _version_of(module: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(module)
    except Exception:
        return None


def _gpu_info() -> Dict[str, Any]:
    """Best-effort GPU inventory via nvidia-smi (no torch import needed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            return {"count": len(gpus), "gpus": gpus,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"count": 0, "gpus": [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def config_hash(config_path: str) -> Optional[str]:
    if not config_path or not os.path.isfile(config_path):
        return None
    with open(config_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_manifest(
    *,
    kind: str,                              # "eval_batch" | "train_run" | ...
    command: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,   # dataset file paths, hashed here
    seed: Optional[int] = None,
    decoding: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "command": command or sys.argv,
        "config": {
            "path": os.path.relpath(config_path, REPO_ROOT).replace("\\", "/")
                    if config_path else None,
            "sha256": config_hash(config_path) if config_path else None,
            "overrides": overrides or [],
        },
        "datasets": [dataset_info(p) for p in (datasets or []) if os.path.isfile(p)],
        "seed": seed,
        "decoding": decoding or {},
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": _version_of("torch"),
            "vllm": _version_of("vllm"),
            "transformers": _version_of("transformers"),
            "peft": _version_of("peft"),
            "gpu": _gpu_info(),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID") or None,
            "wandb_mode": os.environ.get("WANDB_MODE") or None,
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(manifest: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a run manifest JSON.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="adhoc")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--decoding", default=None, help="JSON string, e.g. '{\"temperature\":0.0}'")
    ap.add_argument("--extra", default=None,
                    help="JSON string merged into manifest['extra'] "
                         "(e.g. reward/topology/init adapter for training wrappers)")
    args = ap.parse_args()

    decoding = json.loads(args.decoding) if args.decoding else None
    extra = json.loads(args.extra) if args.extra else None
    m = build_manifest(kind=args.kind, config_path=args.config, datasets=args.dataset,
                       seed=args.seed, decoding=decoding, extra=extra)
    path = write_manifest(m, args.out)
    print(f"[run_manifest] wrote {path} (commit={m['git']['commit']}, dirty={m['git']['dirty']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
