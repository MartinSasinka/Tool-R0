from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS, SYNTHETIC_SUCCESS_REWARD
from lib.offline_audit.on_policy import _group_metrics
from lib.offline_audit.paths import load_train_groups, train_log_path
from lib.offline_audit.stats_util import linear_slope, quartile_indices


def training_progress(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_arm": {}}
    csv_rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        groups = load_train_groups(train_log_path(runs_root, arm, seed))
        groups = sorted(groups, key=lambda g: (g.get("task_idx") or 0, g.get("task_id") or ""))
        n = len(groups)
        slices = quartile_indices(n)
        qstats = []
        for qi, (lo, hi) in enumerate(slices):
            chunk = groups[lo:hi]
            if not chunk:
                continue
            gm = [_group_metrics(g) for g in chunk]
            mean_r = sum(x["mean_reward"] for x in gm) / len(gm)
            term = sum(x["n_success_rollouts"] for x in gm) / sum(x["n_rollouts"] for x in gm)
            dead = sum(1 for x in gm if x["dead_group"]) / len(gm)
            mixed = sum(1 for x in gm if x["mixed_group"]) / len(gm)
            spread = sum(x["reward_range"] for x in gm) / len(gm)
            kl = sum(float(g.get("kl") or 0) for g in chunk) / len(chunk)
            clip = sum(float(g.get("clipped_rate") or 0) for g in chunk) / len(chunk)
            qstats.append(
                {
                    "quartile": qi + 1,
                    "mean_reward": mean_r,
                    "terminal_success": term,
                    "dead_rate": dead,
                    "mixed_rate": mixed,
                    "reward_spread": spread,
                    "kl_mean": kl,
                    "clipping_mean": clip,
                }
            )
            csv_rows.append({"arm": arm, **qstats[-1]})
        xs = list(range(len(qstats)))
        slope_r = linear_slope(xs, [q["mean_reward"] for q in qstats]) if qstats else None
        slope_t = linear_slope(xs, [q["terminal_success"] for q in qstats]) if qstats else None
        slope_kl = linear_slope(xs, [q["kl_mean"] for q in qstats]) if qstats else None
        reward_proxy_warning = False
        if slope_r and slope_r > 0.01 and slope_t is not None and slope_t <= 0:
            reward_proxy_warning = True
        out["per_arm"][arm] = {
            "quartiles": qstats,
            "slope_reward": slope_r,
            "slope_terminal_success": slope_t,
            "slope_kl": slope_kl,
            "reward_proxy_warning": reward_proxy_warning,
        }

    (reports_dir / "training_progress_by_arm.csv").write_text("", encoding="utf-8")  # overwritten below
    if csv_rows:
        with open(reports_dir / "training_progress_by_arm.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    (reports_dir / "training_progress.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# Training progress (task_idx quartiles)", ""]
    for arm, d in out["per_arm"].items():
        lines.append(f"## {arm}")
        lines.append(f"- reward_proxy_warning: **{d['reward_proxy_warning']}**")
        lines.append(f"- slope_reward: {d.get('slope_reward')}")
        lines.append(f"- slope_terminal_success: {d.get('slope_terminal_success')}")
        lines.append("")
    (reports_dir / "TRAINING_PROGRESS.md").write_text("\n".join(lines), encoding="utf-8")
    return out
