"""Shared helpers for the root-cause forensic audit (CPU-only, read-only on runs)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[2]
REPO = V3.parents[1]
MINIMAL = V3.parent / "nestful_mtgrpo_minimal"

R1_ROOT = V3 / "outputs" / "runs" / "_local_round1_analysis"
PURE_S3 = V3 / "outputs" / "runs" / "pure_stage3_2ep_20260719_221918"
REPORTS = V3 / "reports" / "root_cause_forensic"
ANALYSIS = REPORTS / "analysis"

SEED = "20260724"
ARMS = [
    "A0_R0_CURRENT",
    "A1_OUTCOME_ONLY",
    "A2_R3_OUTCOME_FIRST",
    "A3_VERIFIABLE_PROCESS",
    "A4_GATED_VERIFIABLE",
]

# Intended terminal scalars (band midpoints) per lib/reward_ablation_registry.py.
# A1 <- OUTCOME_BANDS_R1 mids; A2/A3/A4 <- OUTCOME_BANDS_R3 mids.
INTENDED_TERMINAL_SCALARS = {
    "A1_OUTCOME_ONLY": {
        "official_success": 0.96,
        "executable_wrong_result": 0.53,
        "executable_partial": 0.34,
        "execution_failure": 0.15,
        "parse_or_no_call": 0.02,
    },
    "A2_R3_OUTCOME_FIRST": {
        "official_success": 0.97,
        "executable_wrong_result": 0.20,
        "executable_partial": 0.1575,
        "execution_failure": 0.115,
        "parse_or_no_call": 0.02,
    },
}
INTENDED_TERMINAL_SCALARS["A3_VERIFIABLE_PROCESS"] = INTENDED_TERMINAL_SCALARS["A2_R3_OUTCOME_FIRST"]
INTENDED_TERMINAL_SCALARS["A4_GATED_VERIFIABLE"] = INTENDED_TERMINAL_SCALARS["A2_R3_OUTCOME_FIRST"]
INTENDED_EPSILON = {"A1_OUTCOME_ONLY": 0.0, "A2_R3_OUTCOME_FIRST": 0.02,
                    "A3_VERIFIABLE_PROCESS": 0.02, "A4_GATED_VERIFIABLE": 0.02}


def run_dir(arm: str) -> Path:
    outer = R1_ROOT / f"reward_ablation_r1_{arm}_seed{SEED}"
    inner = outer / f"reward_ablation_r1_{arm}_seed{SEED}"
    return inner if inner.is_dir() else outer


def eval_dir(arm: str) -> Path:
    return run_dir(arm) / "eval" / arm / SEED


def c0_eval_dir() -> Path:
    return R1_ROOT / "shared_C0_eval_500" / "shared_C0_eval_500" / "eval" / "C0" / SEED


def train_log_path(arm: str) -> Path:
    return run_dir(arm) / "train" / "train_log.jsonl"


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_train_log(arm: str):
    """Returns (header_row_or_None, group_rows)."""
    rows = load_jsonl(train_log_path(arm))
    header = None
    groups = []
    for r in rows:
        if "reward_dispatch" in r and "episode_rewards" not in r:
            header = r
        elif r.get("episode_rewards") is not None and r.get("turn_rewards") is not None:
            groups.append(r)
    return header, groups


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, payload: Any) -> Path:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = ANALYSIS / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p


def eval_ids_500() -> List[str]:
    p = V3 / "reports" / "reward_ablation" / "data" / "nestful_diagnostic_500_ids.json"
    return list(load_json(p)["task_ids"])


# ── trainer math (self-contained copies; parity asserted in tests) ─────────
# _turn_returns copied from experiments/nestful_mtgrpo_minimal/grpo_train.py:1221
def turn_returns(r_seq: List[float], episode_reward: float,
                 gamma: float = 1.0, lambda_episode: float = 1.0) -> List[float]:
    T = len(r_seq) - 1
    returns: List[float] = []
    for t in range(len(r_seq)):
        disc = 0.0
        for k in range(t, len(r_seq)):
            disc += (gamma ** (k - t)) * r_seq[k]
        disc += lambda_episode * (gamma ** (T - t + 1)) * episode_reward
        returns.append(disc)
    return returns


def import_group_stats():
    import sys
    if str(MINIMAL) not in sys.path:
        sys.path.insert(0, str(MINIMAL))
    from group_stats import compute_group_stats  # noqa: E402
    return compute_group_stats


def group_advantages(turn_rewards: List[List[float]], episode_rewards: List[float]):
    compute_group_stats = import_group_stats()
    ep_returns = [turn_returns([float(x) for x in seq], float(R))
                  for seq, R in zip(turn_rewards, episode_rewards)]
    return ep_returns, compute_group_stats(ep_returns, [float(x) for x in episode_rewards])
