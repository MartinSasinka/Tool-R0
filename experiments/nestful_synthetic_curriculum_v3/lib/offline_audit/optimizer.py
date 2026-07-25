from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from lib.offline_audit import ARMS
from lib.offline_audit.on_policy import _group_metrics
from lib.offline_audit.paths import load_json, load_train_groups, train_log_path, train_summary_path


def optimizer_audit(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    rows = []
    by_arm: Dict[str, Any] = {}
    for arm in ARMS:
        summary = load_json(train_summary_path(runs_root, arm, seed))
        groups = load_train_groups(train_log_path(runs_root, arm, seed))
        gm = [_group_metrics(g) for g in groups]
        kls = [float(g.get("kl") or 0) for g in groups]
        clips = [float(g.get("clipped_rate") or 0) for g in groups]
        updates = [bool(g.get("update")) for g in groups]
        eff_groups = sum(1 for x in gm if x["nonzero_advantage_group"])
        dead_rate = summary.get("dead_group_rate")
        flags = []
        if eff_groups / len(groups) < 0.5 if groups else False:
            flags.append("SPARSE_EFFECTIVE_GROUPS")
        if kls and sum(kls) / len(kls) < 1e-4:
            flags.append("KL_SUPPRESSION")
        if clips and sum(clips) / len(clips) > 0.5:
            flags.append("EXCESSIVE_CLIPPING")
        if dead_rate and dead_rate > 0.25 and eff_groups / len(groups) < 0.75:
            flags.append("WEAK_UPDATE_SIGNAL")
        if not flags:
            flags.append("NO_OBVIOUS_OPTIMIZER_BOTTLENECK")
        entry = {
            "arm": arm,
            "optimizer_steps": summary.get("steps"),
            "dead_group_rate": dead_rate,
            "mean_kl": sum(kls) / len(kls) if kls else None,
            "max_kl": max(kls) if kls else None,
            "mean_clipped_rate": sum(clips) / len(clips) if clips else None,
            "fraction_groups_with_update_true": sum(updates) / len(updates) if updates else None,
            "effective_signal_groups": eff_groups,
            "effective_signal_group_rate": eff_groups / len(groups) if groups else None,
            "nan_or_inf": summary.get("nan_or_inf_detected"),
            "flags": "|".join(flags),
        }
        by_arm[arm] = entry
        rows.append(entry)

    with open(reports_dir / "optimizer_signal_by_arm.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Optimizer signal audit", ""]
    for arm, e in by_arm.items():
        n_grp = len(load_train_groups(train_log_path(runs_root, arm, seed)))
        lines.append(f"## {arm}")
        lines.append(f"- steps: {e['optimizer_steps']}")
        lines.append(f"- mean KL: {e['mean_kl']}")
        lines.append(f"- effective signal groups: {e['effective_signal_groups']}/{n_grp}")
        lines.append(f"- flags: `{e['flags']}`")
        lines.append("")
    (reports_dir / "OPTIMIZER_SIGNAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return by_arm
