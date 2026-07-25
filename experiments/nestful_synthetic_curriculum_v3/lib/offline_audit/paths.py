from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.offline_audit import ARMS, DEFAULT_SEED

V3_ROOT = Path(__file__).resolve().parents[2]
REPO_TRAIN_SUBSET = V3_ROOT / "reports" / "reward_ablation" / "data" / "train_subset_160.jsonl"
STAGE3_SOURCE = V3_ROOT / "data" / "training_ready_v5" / "filtered" / "stage3_train_ready.jsonl"
EVAL_IDS = V3_ROOT / "reports" / "reward_ablation" / "data" / "nestful_diagnostic_500_ids.json"


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(runs_root: Path, arm: str, seed: str = DEFAULT_SEED) -> Path:
    outer = runs_root / f"reward_ablation_r1_{arm}_seed{seed}"
    inner = outer / f"reward_ablation_r1_{arm}_seed{seed}"
    if inner.is_dir():
        return inner
    return outer


def c0_eval_dir(runs_root: Path, seed: str = DEFAULT_SEED) -> Path:
    base = runs_root / "shared_C0_eval_500" / "shared_C0_eval_500" / "eval" / "C0" / seed
    return base


def eval_dir(runs_root: Path, arm: str, seed: str = DEFAULT_SEED) -> Path:
    return run_dir(runs_root, arm, seed) / "eval" / arm / seed


def train_log_path(runs_root: Path, arm: str, seed: str = DEFAULT_SEED) -> Path:
    return run_dir(runs_root, arm, seed) / "train" / "train_log.jsonl"


def train_summary_path(runs_root: Path, arm: str, seed: str = DEFAULT_SEED) -> Path:
    return run_dir(runs_root, arm, seed) / "train" / "train_summary.json"


def final_adapter_path(runs_root: Path, arm: str, seed: str = DEFAULT_SEED) -> Path:
    return run_dir(runs_root, arm, seed) / "checkpoints" / "FINAL" / "adapter_model.safetensors"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_groups(path: Path) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not path.is_file():
        return groups
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("episode_rewards") and row.get("turn_rewards"):
                groups.append(row)
    return groups


def resolve_local_path(manifest_path: str) -> Path:
    """Map RunPod /workspace paths to local V3-relative paths."""
    marker = "experiments/nestful_synthetic_curriculum_v3/"
    if marker in manifest_path.replace("\\", "/"):
        tail = manifest_path.replace("\\", "/").split(marker, 1)[1]
        return V3_ROOT / tail.replace("/", "\\") if "\\" in str(V3_ROOT) else V3_ROOT / tail
    p = Path(manifest_path)
    if p.is_file():
        return p
    return V3_ROOT / Path(manifest_path).name
